import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bbox.yolo_utils import get_tensor_anchors
from dataset import ShiftedSatelliteDroneDataset
from grounding.config import load_config
from grounding.losses import compute_grounding_loss
from grounding.registry import build_model_and_adapter


def _device_from_config(cfg: Dict[str, Any]) -> torch.device:
    requested = str(cfg["train"]["device"])
    if requested.startswith("cuda") and torch.cuda.is_available():
        return torch.device(requested)
    return torch.device("cpu")


def _build_loader(cfg: Dict[str, Any], split: str) -> DataLoader:
    model_name = cfg["model"]["model_name"]
    cache_dir = cfg["model"]["cache_dir"]
    processor = AutoImageProcessor.from_pretrained(model_name, cache_dir=cache_dir)
    processor_sat = AutoImageProcessor.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        size=cfg["data"]["sat_size"],
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    dataset = ShiftedSatelliteDroneDataset(
        processor=processor,
        processor_sat=processor_sat,
        tokenizer=tokenizer,
        split=split,
    )
    num_workers = int(cfg["data"]["num_workers"])
    return DataLoader(
        dataset,
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=(split == "train"),
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=num_workers > 0,
    )


def _save_checkpoint(model: torch.nn.Module, save_dir: str, name: str) -> str:
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, name)
    torch.save(model.state_dict(), path)
    return path


def train(cfg: Dict[str, Any], dry_run: bool = False, max_steps: int = 0) -> Dict[str, Any]:
    if dry_run:
        return {"status": "dry_run", "save_dir": cfg["save_dir"], "config_path": cfg["config_path"]}

    device = _device_from_config(cfg)
    model, adapter = build_model_and_adapter(cfg)
    model.to(device)
    anchors_full = get_tensor_anchors(str(device))
    loader = _build_loader(cfg, "train")
    optimizer = AdamW(
        model.parameters(),
        lr=float(cfg["train"]["lr"]),
        weight_decay=float(cfg["train"]["weight_decay"]),
    )
    amp_enabled = bool(cfg["train"]["amp"]) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    grad_accumulation_steps = max(1, int(cfg["train"]["grad_accumulation_steps"]))
    grad_clip_norm = float(cfg["train"]["grad_clip_norm"])

    global_step = 0
    for epoch in range(int(cfg["train"]["epochs"])):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        progress = tqdm(loader, desc=f"Epoch {epoch + 1}/{cfg['train']['epochs']}")
        for batch_idx, batch in enumerate(progress):
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                output = adapter.forward(batch, device)
                losses = compute_grounding_loss(output, batch, anchors_full, cfg)
                loss_to_backward = losses.total / grad_accumulation_steps

            scaler.scale(loss_to_backward).backward()
            should_step = (batch_idx + 1) % grad_accumulation_steps == 0
            should_stop = max_steps > 0 and global_step + 1 >= max_steps
            if should_step or should_stop or batch_idx + 1 == len(loader):
                if grad_clip_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            global_step += 1
            progress.set_postfix(
                {
                    "loss": f"{losses.total.item():.4f}",
                    "bbox": f"{losses.bbox.item():.4f}",
                    "heatmap": f"{losses.heatmap.item():.4f}",
                }
            )
            if should_stop:
                checkpoint = _save_checkpoint(model, cfg["save_dir"], "last.pth")
                return {"status": "max_steps", "checkpoint": checkpoint, "steps": global_step}

        _save_checkpoint(model, cfg["save_dir"], "last.pth")

    checkpoint = _save_checkpoint(model, cfg["save_dir"], "last.pth")
    return {"status": "ok", "checkpoint": checkpoint, "steps": global_step}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train grounding-only models from YAML.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-steps", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    result = train(cfg, dry_run=args.dry_run, max_steps=args.max_steps)
    print(result)


if __name__ == "__main__":
    main()
