import argparse
import csv
import hashlib
import json
import os
import random
import shutil
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoTokenizer


# --- Configuration ---
REPO_ROOT = Path(__file__).resolve().parents[1]
AUTO_PROMPT_DIR = Path(__file__).resolve().parent
DEFAULT_PROMPT_FILE = AUTO_PROMPT_DIR / "current_prompt.md"
DEFAULT_BEST_PROMPT_FILE = AUTO_PROMPT_DIR / "best_prompt.md"
DEFAULT_RESULT_CSV = AUTO_PROMPT_DIR / "result.csv"
DEFAULT_HISTORY_JSONL = AUTO_PROMPT_DIR / "prompt_history.jsonl"
DEFAULT_DESCRIPTION_DIR = AUTO_PROMPT_DIR / "generated_descriptions"
DEFAULT_QUERY_RECORD_DIR = AUTO_PROMPT_DIR / "query_records"

DEFAULT_MODEL_NAME = "google/siglip2-base-patch16-224"
DEFAULT_CACHE_DIR = "/data/feihong/hf_cache"
DEFAULT_CHECKPOINT = "/data/feihong/ckpt/model_test_geo_input_ids/last.pth"
DEFAULT_INCLUDE_FILE = "/data/feihong/ckpt/include2.json"
DEFAULT_TEST_SPLIT_FILE = "/data/feihong/ckpt/test_2.txt"
DEFAULT_TEST_SATELLITE_ROOT = "/data/feihong/img_test_2"
DEFAULT_DRONE_IMAGE_ROOT = "/data/feihong/drone_img"
DEFAULT_DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
DEFAULT_SAT_SIZE = (432, 768)
DEFAULT_SUBSET_HEIGHTS = [150, 200, 250, 300]
DEFAULT_SUBSET_ANGLES = [0, 45, 90, 135, 180, 225, 270, 315]
RESULT_FIELDS = [
    "timestamp",
    "iteration",
    "accepted",
    "score_metric",
    "score",
    "previous_best_score",
    "prompt_sha1",
    "num_cases",
    "num_queries",
    "top1_hits",
    "top5_hits",
    "top10_hits",
    "recall@1",
    "recall@5",
    "recall@10",
    "candidate_size",
    "text_score_weight",
    "text_rerank_topk",
    "checkpoint",
    "descriptions_path",
    "query_records_path",
    "prompt",
]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset import ShiftedSatelliteDroneDataset
from model import Encoder_heat, Encoder_test
from tools import generate_qwen_6_4 as qwen_gen


class ExternalDescriptionDataset(ShiftedSatelliteDroneDataset):
    def __init__(self, descriptions_by_sample: Dict[str, Dict[str, str]], *args, **kwargs):
        self.external_descriptions_by_sample = descriptions_by_sample
        super().__init__(*args, **kwargs)

    def _load_height_descriptions(self, drone_dir: Path) -> Optional[Dict[int, str]]:
        sample_id = _normalize_sample_id(drone_dir.name)
        raw_descriptions = self.external_descriptions_by_sample.get(sample_id)
        if not raw_descriptions:
            return None

        descriptions: Dict[int, str] = {}
        for raw_height, raw_text in raw_descriptions.items():
            height = _safe_int(str(raw_height))
            if height is None or not isinstance(raw_text, str):
                continue
            text = raw_text.strip()
            if text:
                descriptions[int(height)] = text
        return descriptions or None


def _safe_int(text: str) -> Optional[int]:
    try:
        return int(str(text).strip())
    except ValueError:
        return None


def _normalize_sample_id(value: object) -> str:
    parsed = _safe_int(str(value))
    if parsed is None:
        return str(value).strip()
    return f"{parsed:04d}"


def _normalize_label(label: str) -> str:
    return str(label).strip().split(".")[0]


def _path_label(path: str) -> str:
    return _normalize_label(Path(str(path)).stem)


def _load_include_map(include_file: Optional[str]) -> Dict[str, Set[str]]:
    include_map: Dict[str, Set[str]] = {}
    if not include_file:
        return include_map
    if not os.path.exists(include_file):
        print(f"Warning: include file not found: {include_file}. Using exact labels only.")
        return include_map

    with open(include_file, "r", encoding="utf-8") as f:
        include_raw = json.load(f)

    for key, values in include_raw.items():
        norm_key = _normalize_label(key)
        if isinstance(values, list):
            include_map[norm_key] = {_normalize_label(value) for value in values}
        else:
            include_map[norm_key] = {_normalize_label(values)}
    return include_map


def _sha1_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _read_prompt(path: Path) -> str:
    prompt = path.read_text(encoding="utf-8").strip()
    if not prompt:
        raise ValueError(f"Prompt file is empty: {path}")
    return prompt


def _iter_split_case_ids(split_file: str) -> Iterable[str]:
    path = Path(split_file)
    if not path.exists():
        raise FileNotFoundError(f"Test split file not found: {path}")

    seen: Set[str] = set()
    with path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            token = raw_line.strip().split(",")[0].strip()
            if not token:
                continue
            sample_id = _normalize_sample_id(Path(token).stem)
            if sample_id in seen:
                continue
            seen.add(sample_id)
            yield sample_id


def _select_valid_case_ids(
    split_file: str,
    max_cases: int,
    drone_image_root: str,
    test_satellite_root: str,
) -> List[str]:
    valid_ids: List[str] = []
    drone_root = Path(drone_image_root)
    sat_root = Path(test_satellite_root)

    for sample_id in _iter_split_case_ids(split_file):
        sat_candidates = [
            sat_root / f"{sample_id}.png",
            sat_root / f"{sample_id}.jpg",
            sat_root / f"{sample_id}.jpeg",
            sat_root / f"{sample_id}.webp",
        ]
        if not any(path.exists() for path in sat_candidates):
            continue
        if not (drone_root / sample_id).is_dir():
            continue
        valid_ids.append(sample_id)
        if len(valid_ids) >= int(max_cases):
            break

    if not valid_ids:
        raise ValueError("No selected test cases have both satellite and drone images.")
    return valid_ids


def _load_qwen_model(args: argparse.Namespace) -> Tuple[Any, Any]:
    selected_gpu_id = None if int(args.qwen_gpu_id) == -1 else int(args.qwen_gpu_id)
    load_model_name = args.qwen_model_name
    pre_download = not args.skip_pre_download

    if args.prefer_local_snapshot or args.local_files_only:
        local_snapshot = qwen_gen.resolve_local_snapshot(args.qwen_model_name, args.qwen_cache_dir)
        if local_snapshot is not None:
            load_model_name = str(local_snapshot)
            pre_download = False
            print(f"Using local Qwen snapshot: {load_model_name}")
        elif args.local_files_only:
            raise FileNotFoundError(
                f"No complete local snapshot found for {args.qwen_model_name} "
                f"under {args.qwen_cache_dir}."
            )

    return qwen_gen.load_qwen36(
        model_name=load_model_name,
        cache_dir=args.qwen_cache_dir,
        gpu_id=selected_gpu_id,
        dtype=args.qwen_dtype,
        use_flash_attention=args.flash_attn,
        hf_token=args.hf_token,
        hf_endpoint=args.hf_endpoint,
        pre_download=pre_download,
        download_retries=args.download_retries,
        download_max_workers=args.download_max_workers,
    )


def generate_descriptions(
    args: argparse.Namespace,
    prompt: str,
    case_ids: Sequence[str],
    descriptions_path: Path,
) -> Dict[str, Dict[str, str]]:
    if descriptions_path.exists() and not args.overwrite_descriptions:
        with descriptions_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        return {
            _normalize_sample_id(sample_id): descriptions
            for sample_id, descriptions in payload.get("descriptions_by_sample", {}).items()
        }

    model, processor = _load_qwen_model(args)
    qwen_args = SimpleNamespace(
        prompt=prompt,
        model_name=args.qwen_model_name,
        image_width=args.image_width,
        image_height=args.image_height,
        image_field=args.image_field,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        enable_thinking=args.enable_thinking,
    )

    descriptions_by_sample: Dict[str, Dict[str, str]] = {}
    for idx, sample_id in enumerate(case_ids, start=1):
        image_dir = str((Path(args.drone_image_root) / sample_id).resolve())
        print(f"[Qwen {idx}/{len(case_ids)}] {image_dir}")
        result = qwen_gen.generate_descriptions_for_one_dir(
            resolved_image_dir=image_dir,
            model=model,
            processor=processor,
            args=qwen_args,
        )
        descriptions_by_sample[sample_id] = {
            str(height): str(text).strip()
            for height, text in result["descriptions"].items()
        }

    descriptions_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": {
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "prompt_sha1": _sha1_text(prompt),
            "prompt": prompt,
            "case_ids": list(case_ids),
            "qwen_model_name": args.qwen_model_name,
        },
        "descriptions_by_sample": descriptions_by_sample,
    }
    with descriptions_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return descriptions_by_sample


def load_descriptions(path: Path) -> Dict[str, Dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    descriptions_by_sample = payload.get("descriptions_by_sample")
    if not isinstance(descriptions_by_sample, dict):
        raise ValueError(f"Missing descriptions_by_sample in: {path}")
    return {
        _normalize_sample_id(sample_id): {
            str(height): str(text)
            for height, text in descriptions.items()
        }
        for sample_id, descriptions in descriptions_by_sample.items()
        if isinstance(descriptions, dict)
    }


def load_checkpoint(model: torch.nn.Module, checkpoint_path: str) -> None:
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    state = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    elif isinstance(state, dict) and "model" in state:
        state = state["model"]
    if isinstance(state, dict):
        state = {str(key).replace("module.", ""): value for key, value in state.items()}
        state = {
            key: value
            for key, value in state.items()
            if not key.startswith("text_projector.")
        }

    model.load_state_dict(state, strict=True)


def build_geo_features(batch: Dict[str, Any], device: torch.device) -> torch.Tensor:
    angles = batch["angle"].to(device, non_blocking=True).float()
    angles_rad = torch.deg2rad(angles)
    heights = batch["height"].to(device, non_blocking=True).float()
    return torch.cat(
        [
            torch.cos(angles_rad)[..., None],
            torch.sin(angles_rad)[..., None],
            heights[..., None] / 300.0,
        ],
        dim=1,
    )


def create_loader(
    args: argparse.Namespace,
    descriptions_by_sample: Dict[str, Dict[str, str]],
) -> DataLoader:
    sat_h, sat_w = int(args.sat_size[0]), int(args.sat_size[1])
    processor = AutoImageProcessor.from_pretrained(args.model_name, cache_dir=args.cache_dir)
    processor_sat = AutoImageProcessor.from_pretrained(
        args.model_name,
        cache_dir=args.cache_dir,
        size={"height": sat_h, "width": sat_w},
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, cache_dir=args.cache_dir)
    dataset = ExternalDescriptionDataset(
        descriptions_by_sample=descriptions_by_sample,
        processor=processor,
        processor_sat=processor_sat,
        tokenizer=tokenizer,
        split="test",
        test_satellite_root=args.test_satellite_root,
        drone_image_root=args.drone_image_root,
        bbox_file=args.bbox_file,
        test_split_file=args.test_split_file,
        sat_target_size=(sat_h, sat_w),
        test_crop_ratio=args.test_crop_ratio,
        subset_heights=args.subset_heights,
        subset_angles=args.subset_angles,
    )
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
        prefetch_factor=4 if args.num_workers > 0 else None,
    )


def score_queries(
    query_feats: torch.Tensor,
    query_text_feats: Optional[torch.Tensor],
    query_labels: List[str],
    query_drone_paths: List[str],
    query_satellite_paths: List[str],
    query_heights: List[int],
    query_angles: List[int],
    gallery_feats: torch.Tensor,
    gallery_labels: List[str],
    gallery_satellite_paths: List[str],
    include_map: Dict[str, Set[str]],
    device: torch.device,
    candidate_size: Optional[int],
    text_score_weight: float,
    text_rerank_topk: int,
    seed: int,
) -> Dict[str, Any]:
    label_to_gallery_index = {label: idx for idx, label in enumerate(gallery_labels)}
    norm_gallery_labels = [_normalize_label(label) for label in gallery_labels]
    all_indices = list(range(len(gallery_labels)))
    query_records: List[Dict[str, Any]] = []
    top1 = 0
    top5 = 0
    top10 = 0
    valid_queries = 0

    if candidate_size is not None and int(candidate_size) <= 0:
        candidate_size = None

    gallery_feats = gallery_feats.to(device)
    for q_idx, gt_label in enumerate(query_labels):
        gt_gallery_index = label_to_gallery_index.get(gt_label)
        if gt_gallery_index is None:
            continue

        if candidate_size is None or int(candidate_size) >= len(all_indices):
            candidate_indices = all_indices
        else:
            negative_pool = [idx for idx in all_indices if idx != gt_gallery_index]
            rng = random.Random(int(seed) + int(q_idx))
            sampled_negatives = rng.sample(negative_pool, int(candidate_size) - 1)
            candidate_indices = sampled_negatives + [gt_gallery_index]

        candidate_gallery = gallery_feats[candidate_indices]
        image_scores = torch.einsum(
            "d,knd->kn",
            query_feats[q_idx].to(device),
            candidate_gallery,
        ).max(dim=1)[0]
        score_vec = image_scores

        if query_text_feats is not None and float(text_score_weight) > 0.0:
            text_scores = torch.einsum(
                "d,knd->kn",
                query_text_feats[q_idx].to(device),
                candidate_gallery,
            ).max(dim=1)[0]
            if text_rerank_topk <= 0 or text_rerank_topk >= int(image_scores.numel()):
                score_vec = image_scores + float(text_score_weight) * text_scores
            else:
                score_vec = image_scores.clone()
                topk_indices = image_scores.topk(int(text_rerank_topk), dim=0).indices
                score_vec[topk_indices] = (
                    image_scores[topk_indices]
                    + float(text_score_weight) * text_scores[topk_indices]
                )

        k_max = min(10, int(score_vec.shape[0]))
        topk_local_indices = torch.topk(score_vec, k=k_max, dim=0).indices.tolist()
        top1_local_idx = int(topk_local_indices[0])
        top1_global_idx = int(candidate_indices[top1_local_idx])

        gt_norm = _normalize_label(gt_label)
        positive_label_set = set(include_map.get(gt_norm, set()))
        positive_label_set.add(gt_norm)
        positive_local_indices = [
            local_idx
            for local_idx, global_idx in enumerate(candidate_indices)
            if norm_gallery_labels[global_idx] in positive_label_set
        ]
        top1_correct = any(idx in topk_local_indices[:1] for idx in positive_local_indices)
        top5_correct = any(idx in topk_local_indices[: min(5, k_max)] for idx in positive_local_indices)
        top10_correct = any(idx in topk_local_indices[: min(10, k_max)] for idx in positive_local_indices)

        valid_queries += 1
        top1 += int(top1_correct)
        top5 += int(top5_correct)
        top10 += int(top10_correct)
        query_records.append(
            {
                "query_index": int(q_idx),
                "height": int(query_heights[q_idx]),
                "angle": int(query_angles[q_idx]),
                "gt_label": gt_norm,
                "pred_label": _normalize_label(gallery_labels[top1_global_idx]),
                "top1_correct": bool(top1_correct),
                "top5_correct": bool(top5_correct),
                "top10_correct": bool(top10_correct),
                "drone_path": query_drone_paths[q_idx],
                "gt_satellite_path": query_satellite_paths[q_idx],
                "pred_satellite_path": gallery_satellite_paths[top1_global_idx],
            }
        )

    if valid_queries == 0:
        raise RuntimeError("No valid queries matched gallery labels.")

    return {
        "num_queries": int(valid_queries),
        "top1_hits": int(top1),
        "top5_hits": int(top5),
        "top10_hits": int(top10),
        "recall@1": float(top1 / valid_queries),
        "recall@5": float(top5 / valid_queries),
        "recall@10": float(top10 / valid_queries),
        "query_records": query_records,
    }


def evaluate_retrieval(
    args: argparse.Namespace,
    descriptions_by_sample: Dict[str, Dict[str, str]],
) -> Dict[str, Any]:
    device = torch.device(args.device)
    encoder_cls = Encoder_test if args.encoder_type == "test" else Encoder_heat
    model = encoder_cls(
        model_name=args.model_name,
        proj_dim=768,
        usesg=True,
        useap=args.use_ap,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    ).to(device)
    load_checkpoint(model, args.checkpoint)
    model.eval()

    loader = create_loader(args, descriptions_by_sample)
    include_map = _load_include_map(args.include_file)

    query_feats: List[torch.Tensor] = []
    query_text_feats: List[torch.Tensor] = []
    query_labels: List[str] = []
    query_drone_paths: List[str] = []
    query_satellite_paths: List[str] = []
    query_heights: List[int] = []
    query_angles: List[int] = []
    gallery_feat_dict: Dict[str, torch.Tensor] = {}
    gallery_path_dict: Dict[str, str] = {}

    with torch.inference_mode():
        for batch in tqdm(loader, desc=f"Evaluate Encoder_{args.encoder_type}"):
            query_imgs = batch["target_pixel_values"].to(device, non_blocking=True)
            search_imgs = batch["search_pixel_values"].to(device, non_blocking=True)
            input_ids = batch["input_ids"].to(device, non_blocking=True) if args.use_text else None
            attention_mask = (
                batch["attention_mask"].to(device, non_blocking=True)
                if args.use_text and "attention_mask" in batch
                else None
            )
            geo = build_geo_features(batch, device) if args.use_angle else None
            outputs = model(
                query_imgs,
                search_imgs,
                input_ids=input_ids,
                angle=geo,
                attention_mask=attention_mask,
            )
            if len(outputs) != 7:
                raise ValueError(f"Expected 7 model outputs, got {len(outputs)}.")
            _, _, text_pooler, anchor_pooler, grid_feats, _, _ = outputs

            query_feats.append(F.normalize(anchor_pooler, p=2, dim=1).cpu())
            if text_pooler is not None:
                query_text_feats.append(F.normalize(text_pooler, p=2, dim=1).cpu())

            gallery_grid_batch = F.normalize(grid_feats, p=2, dim=2).cpu()
            batch_labels = [_path_label(path) for path in batch["satellite_path"]]
            query_labels.extend(batch_labels)
            query_drone_paths.extend([str(path) for path in batch["drone_path"]])
            query_satellite_paths.extend([str(path) for path in batch["satellite_path"]])
            query_heights.extend([int(value) for value in batch["height"].tolist()])
            query_angles.extend([int(value) for value in batch["angle"].tolist()])

            for idx, label in enumerate(batch_labels):
                if label not in gallery_feat_dict:
                    gallery_feat_dict[label] = gallery_grid_batch[idx]
                    gallery_path_dict[label] = str(batch["satellite_path"][idx])

    if not query_feats or not gallery_feat_dict:
        raise RuntimeError("No retrieval features were extracted.")

    gallery_labels = sorted(gallery_feat_dict.keys())
    query_text_tensor = torch.cat(query_text_feats, dim=0) if query_text_feats else None
    metrics = score_queries(
        query_feats=torch.cat(query_feats, dim=0),
        query_text_feats=query_text_tensor,
        query_labels=query_labels,
        query_drone_paths=query_drone_paths,
        query_satellite_paths=query_satellite_paths,
        query_heights=query_heights,
        query_angles=query_angles,
        gallery_feats=torch.stack([gallery_feat_dict[label] for label in gallery_labels], dim=0),
        gallery_labels=gallery_labels,
        gallery_satellite_paths=[gallery_path_dict.get(label, "") for label in gallery_labels],
        include_map=include_map,
        device=device,
        candidate_size=args.candidate_size,
        text_score_weight=args.text_score_weight,
        text_rerank_topk=args.text_rerank_topk,
        seed=args.seed,
    )
    metrics["num_cases"] = int(len(gallery_labels))
    return metrics


def _existing_scores(result_csv: Path, metric_name: str) -> List[float]:
    if not result_csv.exists():
        return []
    scores: List[float] = []
    with result_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("score_metric") != metric_name:
                continue
            try:
                scores.append(float(row.get("score", "")))
            except ValueError:
                continue
    return scores


def append_result(
    args: argparse.Namespace,
    prompt: str,
    prompt_sha1: str,
    metrics: Dict[str, Any],
    descriptions_path: Path,
    query_records_path: Path,
) -> Dict[str, Any]:
    score = float(metrics[args.score_metric])
    previous_scores = _existing_scores(Path(args.result_csv), args.score_metric)
    previous_best = max(previous_scores) if previous_scores else None
    accepted = previous_best is None or score > previous_best

    row = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "iteration": int(args.iteration),
        "accepted": bool(accepted),
        "score_metric": args.score_metric,
        "score": score,
        "previous_best_score": "" if previous_best is None else previous_best,
        "prompt_sha1": prompt_sha1,
        "num_cases": int(metrics["num_cases"]),
        "num_queries": int(metrics["num_queries"]),
        "top1_hits": int(metrics["top1_hits"]),
        "top5_hits": int(metrics["top5_hits"]),
        "top10_hits": int(metrics["top10_hits"]),
        "recall@1": float(metrics["recall@1"]),
        "recall@5": float(metrics["recall@5"]),
        "recall@10": float(metrics["recall@10"]),
        "candidate_size": "" if args.candidate_size is None else int(args.candidate_size),
        "text_score_weight": float(args.text_score_weight),
        "text_rerank_topk": int(args.text_rerank_topk),
        "checkpoint": args.checkpoint,
        "descriptions_path": str(descriptions_path),
        "query_records_path": str(query_records_path),
        "prompt": prompt,
    }

    result_csv = Path(args.result_csv)
    result_csv.parent.mkdir(parents=True, exist_ok=True)
    write_header = not result_csv.exists()
    with result_csv.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    history_path = Path(args.history_jsonl)
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with history_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

    archive_dir = AUTO_PROMPT_DIR / "prompt_archive"
    archive_dir.mkdir(parents=True, exist_ok=True)
    archive_path = archive_dir / f"iter_{int(args.iteration):03d}_{prompt_sha1[:10]}.md"
    archive_path.write_text(prompt + "\n", encoding="utf-8")

    if accepted:
        shutil.copyfile(args.prompt_file, args.best_prompt_file)
    elif args.restore_best_on_reject and Path(args.best_prompt_file).exists():
        shutil.copyfile(args.best_prompt_file, args.prompt_file)

    row["accepted"] = accepted
    row["archive_path"] = str(archive_path)
    return row


def write_query_records(path: Path, records: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one auto-prompt retrieval research iteration on the first test cases."
    )
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--prompt-file", type=Path, default=DEFAULT_PROMPT_FILE)
    parser.add_argument("--best-prompt-file", type=Path, default=DEFAULT_BEST_PROMPT_FILE)
    parser.add_argument("--result-csv", type=Path, default=DEFAULT_RESULT_CSV)
    parser.add_argument("--history-jsonl", type=Path, default=DEFAULT_HISTORY_JSONL)
    parser.add_argument("--description-dir", type=Path, default=DEFAULT_DESCRIPTION_DIR)
    parser.add_argument("--query-record-dir", type=Path, default=DEFAULT_QUERY_RECORD_DIR)
    parser.add_argument("--score-metric", choices=["recall@1", "recall@5", "recall@10"], default="recall@1")
    parser.add_argument("--restore-best-on-reject", action="store_true")

    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--encoder-type", choices=["test", "heat"], default="test")
    parser.add_argument("--model-name", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--cache-dir", type=str, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--sat-size", type=int, nargs=2, default=list(DEFAULT_SAT_SIZE), metavar=("HEIGHT", "WIDTH"))
    parser.add_argument("--test-crop-ratio", type=float, default=1.0)
    parser.add_argument("--subset-heights", type=int, nargs="*", default=DEFAULT_SUBSET_HEIGHTS)
    parser.add_argument("--subset-angles", type=int, nargs="*", default=DEFAULT_SUBSET_ANGLES)
    parser.add_argument("--candidate-size", type=int, default=50)
    parser.add_argument("--include-file", type=str, default=DEFAULT_INCLUDE_FILE)
    parser.add_argument("--bbox-file", type=str, default="/data/feihong/ckpt/bbox_test_2.json")
    parser.add_argument("--test-split-file", type=str, default=DEFAULT_TEST_SPLIT_FILE)
    parser.add_argument("--test-satellite-root", type=str, default=DEFAULT_TEST_SATELLITE_ROOT)
    parser.add_argument("--drone-image-root", type=str, default=DEFAULT_DRONE_IMAGE_ROOT)
    parser.add_argument("--max-cases", type=int, default=50)
    parser.add_argument("--use-text", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-angle", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-ap", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--text-score-weight", type=float, default=0.3)
    parser.add_argument("--text-rerank-topk", type=int, default=50)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=43)

    parser.add_argument("--skip-generate", action="store_true")
    parser.add_argument("--overwrite-descriptions", action="store_true")
    parser.add_argument("--qwen-model-name", type=str, default=qwen_gen.MODEL_NAME)
    parser.add_argument("--qwen-cache-dir", type=str, default=qwen_gen.CACHE_DIR)
    parser.add_argument("--qwen-gpu-id", type=int, default=qwen_gen.DEFAULT_GPU_ID)
    parser.add_argument("--qwen-dtype", type=str, default="auto", choices=["auto", "bf16", "fp16", "fp32"])
    parser.add_argument("--flash-attn", action="store_true")
    parser.add_argument("--hf-token", type=str, default=None)
    parser.add_argument("--hf-endpoint", type=str, default=qwen_gen.DEFAULT_HF_ENDPOINT)
    parser.add_argument("--skip-pre-download", action="store_true")
    parser.add_argument("--prefer-local-snapshot", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--download-retries", type=int, default=5)
    parser.add_argument("--download-max-workers", type=int, default=8)
    parser.add_argument("--image-width", type=int, default=qwen_gen.IMAGE_SIZE[0])
    parser.add_argument("--image-height", type=int, default=qwen_gen.IMAGE_SIZE[1])
    parser.add_argument("--image-field", choices=["pil", "url", "image"], default="image")
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=64)
    parser.add_argument("--enable-thinking", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))

    prompt = _read_prompt(Path(args.prompt_file))
    prompt_sha1 = _sha1_text(prompt)
    case_ids = _select_valid_case_ids(
        split_file=args.test_split_file,
        max_cases=args.max_cases,
        drone_image_root=args.drone_image_root,
        test_satellite_root=args.test_satellite_root,
    )
    descriptions_path = Path(args.description_dir) / f"iter_{int(args.iteration):03d}_{prompt_sha1[:10]}.json"

    if args.skip_generate:
        descriptions_by_sample = load_descriptions(descriptions_path)
    else:
        descriptions_by_sample = generate_descriptions(
            args=args,
            prompt=prompt,
            case_ids=case_ids,
            descriptions_path=descriptions_path,
        )

    metrics = evaluate_retrieval(args, descriptions_by_sample)
    query_records_path = Path(args.query_record_dir) / f"iter_{int(args.iteration):03d}_{prompt_sha1[:10]}.jsonl"
    write_query_records(query_records_path, metrics.pop("query_records"))

    row = append_result(
        args=args,
        prompt=prompt,
        prompt_sha1=prompt_sha1,
        metrics=metrics,
        descriptions_path=descriptions_path,
        query_records_path=query_records_path,
    )

    print(
        "Iteration {iteration} | accepted={accepted} | "
        "R@1={r1:.4f} R@5={r5:.4f} R@10={r10:.4f} | "
        "result_csv={csv_path}".format(
            iteration=args.iteration,
            accepted=row["accepted"],
            r1=metrics["recall@1"],
            r5=metrics["recall@5"],
            r10=metrics["recall@10"],
            csv_path=args.result_csv,
        )
    )


if __name__ == "__main__":
    main()
