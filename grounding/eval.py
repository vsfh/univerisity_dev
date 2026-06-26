import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bbox.yolo_utils import get_tensor_anchors
from dataset import ShiftedSatelliteDroneDataset
from grounding.config import load_config
from grounding.losses import compute_iou_metrics
from grounding.processors import build_grounding_image_processors
from grounding.registry import build_model_and_adapter


def _device_from_config(cfg: Dict[str, Any]) -> torch.device:
    requested = str(cfg["train"]["device"])
    if requested.startswith("cuda") and torch.cuda.is_available():
        return torch.device(requested)
    return torch.device("cpu")


def _build_loader(cfg: Dict[str, Any]) -> DataLoader:
    model_name = cfg["model"]["model_name"]
    cache_dir = cfg["model"]["cache_dir"]
    processor, processor_sat = build_grounding_image_processors(cfg)
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    dataset = ShiftedSatelliteDroneDataset(
        processor=processor,
        processor_sat=processor_sat,
        tokenizer=tokenizer,
        split="test",
    )
    num_workers = int(cfg["data"]["num_workers"])
    return DataLoader(
        dataset,
        batch_size=int(cfg["eval"]["batch_size"]),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=num_workers > 0,
    )


def _checkpoint_path(cfg: Dict[str, Any]) -> str:
    checkpoint = cfg["eval"]["checkpoint"]
    if os.path.isabs(str(checkpoint)):
        return str(checkpoint)
    return os.path.join(cfg["save_dir"], str(checkpoint))


def _load_checkpoint(model: torch.nn.Module, path: str) -> None:
    state = torch.load(path, map_location="cpu")
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state, strict=True)


def _average_metrics(total: Dict[str, float], count: int) -> Dict[str, float]:
    return {key: float(value / max(count, 1)) for key, value in total.items()}


def _print_result(result: Dict[str, Any]) -> None:
    print(json.dumps(result, indent=2, sort_keys=True))


def evaluate(cfg: Dict[str, Any], dry_run: bool = False, max_batches: int = 0) -> Dict[str, Any]:
    checkpoint = _checkpoint_path(cfg)
    if dry_run:
        result = {
            "status": "dry_run",
            "config": cfg["config_path"],
            "checkpoint": checkpoint,
            "output_dir": cfg["eval"]["output_dir"],
        }
        _print_result(result)
        return result

    device = _device_from_config(cfg)
    model, adapter = build_model_and_adapter(cfg)
    model.to(device)
    anchors_full = get_tensor_anchors(str(device))

    _load_checkpoint(model, checkpoint)
    loader = _build_loader(cfg)
    model.eval()
    total = {
        "mean_iou": 0.0,
        "iou_at_0_5": 0.0,
        "iou_at_0_25": 0.0,
        "mean_center_distance": 0.0,
    }
    count = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(loader, desc=f"Eval {cfg['exp_name']}")):
            target_bbox = batch["bbox"].to(device)
            output = adapter.forward(batch, device)
            pred_bbox = adapter.decode(output, batch, anchors_full)
            target_bbox = target_bbox.to(pred_bbox.device)
            metrics = compute_iou_metrics(pred_bbox, target_bbox)
            batch_n = int(target_bbox.shape[0])
            for key, value in metrics.items():
                total[key] += float(value) * batch_n
            count += batch_n
            if max_batches > 0 and batch_idx + 1 >= max_batches:
                break

    metrics = _average_metrics(total, count)
    result = {
        "status": "ok",
        "config": cfg["config_path"],
        "checkpoint": checkpoint,
        "num_samples": count,
        **metrics,
    }
    output_dir = Path(cfg["eval"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, sort_keys=True)
    _print_result(result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate grounding-only models from YAML.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if args.device:
        cfg["train"]["device"] = args.device
    evaluate(cfg, dry_run=args.dry_run, max_batches=args.max_batches)


if __name__ == "__main__":
    main()
