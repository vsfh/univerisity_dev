import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
import yaml
from transformers import AutoImageProcessor, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset import ShiftedSatelliteDroneDataset
from model import CACHE_DIR, SIGLIP2_MODEL_NAME
from unified_siglip_supp import Config, apply_config_overrides, build_encoder, effective_model_name


# --- Configuration ---
DEFAULT_CONFIG = "configs/unified_siglip_supp/end_num/model_end1.yaml"
DEFAULT_CHECKPOINT = "/data/feihong/ckpt/model_1/last.pth"
DEFAULT_LIMIT = 128
DEFAULT_BATCH_SIZE = 32


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        payload = yaml.safe_load(f) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return payload


def load_checkpoint(model: torch.nn.Module, checkpoint_path: str) -> None:
    state = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if not isinstance(state, dict):
        raise ValueError(f"Unexpected checkpoint format: {checkpoint_path}")
    state = {str(k).replace("module.", ""): v for k, v in state.items()}
    state = {
        key: value
        for key, value in state.items()
        if not key.startswith("text_projector.")
    }
    model.load_state_dict(state, strict=True)


def collect_unique_texts(dataset: ShiftedSatelliteDroneDataset, limit: int) -> List[Dict[str, Any]]:
    seen = set()
    rows: List[Dict[str, Any]] = []
    for sample in dataset.samples:
        sat_id = int(sample["satellite_id"])
        if sat_id in seen:
            continue
        seen.add(sat_id)
        rows.append(
            {
                "satellite_id": sat_id,
                "text": str(sample["text"]),
            }
        )
        if len(rows) >= limit:
            break
    if not rows:
        raise ValueError("No unique texts collected from dataset.")
    return rows


def tokenize_texts(
    tokenizer: AutoTokenizer,
    texts: List[str],
    max_length: int,
    batch_size: int,
    device: torch.device,
):
    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start : start + batch_size]
        encoded = tokenizer(
            batch_texts,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        attention_mask = encoded.get("attention_mask")
        if attention_mask is None:
            pad_token_id = getattr(tokenizer, "pad_token_id", None)
            if pad_token_id is None:
                attention_mask = torch.ones_like(encoded["input_ids"], dtype=torch.long)
            else:
                attention_mask = (encoded["input_ids"] != int(pad_token_id)).long()
        yield (
            encoded["input_ids"].to(device),
            attention_mask.to(device),
        )


def masked_mean_hidden(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.to(device=last_hidden_state.device, dtype=last_hidden_state.dtype)
    mask = mask.unsqueeze(-1)
    summed = (last_hidden_state * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp_min(1.0)
    return summed / denom


def encode_text_representations(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    texts: List[str],
    max_length: int,
    batch_size: int,
    device: torch.device,
    include_projected: bool,
) -> Dict[str, torch.Tensor]:
    model.eval()
    outputs: Dict[str, List[torch.Tensor]] = {
        "raw_pooler": [],
        "hidden_mean": [],
    }
    has_text_projector = hasattr(model, "text_projector")
    if include_projected and has_text_projector:
        outputs["projected_pooler"] = []

    with torch.no_grad():
        for input_ids, attention_mask in tokenize_texts(
            tokenizer,
            texts,
            max_length=max_length,
            batch_size=batch_size,
            device=device,
        ):
            text_outputs = model.text_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            raw_pooler = F.normalize(text_outputs.pooler_output.float(), p=2, dim=1)
            hidden_mean = F.normalize(
                masked_mean_hidden(text_outputs.last_hidden_state.float(), attention_mask),
                p=2,
                dim=1,
            )
            outputs["raw_pooler"].append(raw_pooler.cpu())
            outputs["hidden_mean"].append(hidden_mean.cpu())

            if include_projected and has_text_projector:
                projected = model.text_projector(text_outputs.pooler_output)
                projected = F.normalize(projected.float(), p=2, dim=1)
                outputs["projected_pooler"].append(projected.cpu())

    return {key: torch.cat(value, dim=0) for key, value in outputs.items()}


def cosine_summary(features: torch.Tensor) -> Dict[str, float]:
    features = F.normalize(features.float(), p=2, dim=1)
    sim = features @ features.T
    n = sim.shape[0]
    if n < 2:
        raise ValueError("Need at least two features to compute off-diagonal cosine.")
    mask = ~torch.eye(n, dtype=torch.bool)
    values = sim[mask]
    quantiles = torch.quantile(values, torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9, 0.95]))
    return {
        "count": float(n),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "min": float(values.min()),
        "p10": float(quantiles[0]),
        "p25": float(quantiles[1]),
        "p50": float(quantiles[2]),
        "p75": float(quantiles[3]),
        "p90": float(quantiles[4]),
        "p95": float(quantiles[5]),
        "max": float(values.max()),
    }


def print_summary(title: str, summary: Dict[str, float]) -> None:
    print(f"\n{title}")
    for key in ["count", "mean", "std", "min", "p10", "p25", "p50", "p75", "p90", "p95", "max"]:
        print(f"  {key}: {summary[key]:.6f}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure raw SigLIP text hidden-state cosine without text_projector."
    )
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--compare-pretrained", action="store_true")
    parser.add_argument("--output", type=str, default="")
    args = parser.parse_args()

    payload = load_yaml(args.config)
    apply_config_overrides(payload.get("config", {}) or {})
    model_name = effective_model_name()
    if Config.ENCODER_TYPE in {"sig", "lora"} and model_name != SIGLIP2_MODEL_NAME:
        raise ValueError(f"Expected SigLIP2 model for {Config.ENCODER_TYPE}, got {model_name}.")

    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=CACHE_DIR, local_files_only=True)
    processor = AutoImageProcessor.from_pretrained(model_name, cache_dir=CACHE_DIR, local_files_only=True)
    processor_sat = AutoImageProcessor.from_pretrained(
        model_name,
        cache_dir=CACHE_DIR,
        size=Config.UNIV_SAT_SIZE,
        local_files_only=True,
    )
    dataset = ShiftedSatelliteDroneDataset(
        processor=processor,
        processor_sat=processor_sat,
        tokenizer=tokenizer,
        split=args.split,
    )
    rows = collect_unique_texts(dataset, limit=args.limit)
    texts = [row["text"] for row in rows]
    token_lengths = []
    for text in texts:
        encoded = tokenizer(text, truncation=False, add_special_tokens=True)
        token_lengths.append(len(encoded["input_ids"]))

    device = torch.device(args.device)
    model = build_encoder(
        use_ap=bool(payload.get("use_ap", True)),
        usesg=Config.OPTIMIZE_OBJECTIVE != "bbox_only",
    )
    max_positions = int(getattr(model.text_model.config, "max_position_embeddings", args.max_length))
    if args.max_length > max_positions:
        raise ValueError(
            f"--max-length={args.max_length} exceeds text encoder "
            f"max_position_embeddings={max_positions}."
        )
    load_checkpoint(model, args.checkpoint)
    model.to(device)

    result: Dict[str, Any] = {
        "config": args.config,
        "checkpoint": args.checkpoint,
        "model_name": model_name,
        "split": args.split,
        "unique_text_count": len(texts),
        "max_length": args.max_length,
        "token_length": {
            "mean": float(torch.tensor(token_lengths, dtype=torch.float32).mean()),
            "max": int(max(token_lengths)),
            "truncated_count": int(sum(length > args.max_length for length in token_lengths)),
        },
        "summaries": {},
    }

    checkpoint_features = encode_text_representations(
        model=model,
        tokenizer=tokenizer,
        texts=texts,
        max_length=args.max_length,
        batch_size=args.batch_size,
        device=device,
        include_projected=True,
    )
    for name, features in checkpoint_features.items():
        result["summaries"][f"checkpoint_{name}"] = cosine_summary(features)

    if args.compare_pretrained:
        pretrained = build_encoder(
            use_ap=bool(payload.get("use_ap", True)),
            usesg=Config.OPTIMIZE_OBJECTIVE != "bbox_only",
        )
        pretrained.to(device)
        pretrained_features = encode_text_representations(
            model=pretrained,
            tokenizer=tokenizer,
            texts=texts,
            max_length=args.max_length,
            batch_size=args.batch_size,
            device=device,
            include_projected=False,
        )
        for name, features in pretrained_features.items():
            result["summaries"][f"pretrained_{name}"] = cosine_summary(features)

    print(
        f"Collected {len(texts)} unique texts from split={args.split}; "
        f"token_length_mean={result['token_length']['mean']:.2f}, "
        f"token_length_max={result['token_length']['max']}, "
        f"truncated={result['token_length']['truncated_count']}/{len(texts)}."
    )
    print("Example texts:")
    for row in rows[:3]:
        print(f"  {row['satellite_id']}: {row['text'][:220]}")

    for name, summary in result["summaries"].items():
        print_summary(name, summary)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, sort_keys=True, ensure_ascii=False)
        print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
