import argparse
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
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


class _DistributedInfo(dict):
    def __getattr__(self, key: str) -> Any:
        return self[key]


def _distributed_info() -> _DistributedInfo:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    return _DistributedInfo({
        "enabled": world_size > 1,
        "world_size": world_size,
        "rank": rank,
        "local_rank": local_rank,
    })


def _init_distributed() -> _DistributedInfo:
    info = _distributed_info()
    if not info["enabled"]:
        return info
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    if torch.cuda.is_available():
        torch.cuda.set_device(int(info["local_rank"]))
    dist.init_process_group(backend=backend)
    return info


def _cleanup_distributed(distributed: _DistributedInfo) -> None:
    if distributed["enabled"] and dist.is_initialized():
        dist.destroy_process_group()


def _is_rank_zero(distributed: _DistributedInfo) -> bool:
    return int(distributed["rank"]) == 0


def _device_from_config(cfg: Dict[str, Any]) -> torch.device:
    requested = str(cfg["train"]["device"])
    if requested.startswith("cuda") and torch.cuda.is_available():
        return torch.device(requested)
    return torch.device("cpu")


def _build_loader(
    cfg: Dict[str, Any],
    split: str,
    distributed: _DistributedInfo,
) -> tuple[DataLoader, DistributedSampler | None]:
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
    sampler = None
    shuffle = split == "train"
    if distributed["enabled"]:
        sampler = DistributedSampler(
            dataset,
            num_replicas=int(distributed["world_size"]),
            rank=int(distributed["rank"]),
            shuffle=shuffle,
        )
        shuffle = False

    global_batch_size = int(cfg["train"]["batch_size"])
    batch_size = global_batch_size
    if distributed["enabled"]:
        batch_size = max(1, math.ceil(global_batch_size / distributed.world_size))

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=num_workers > 0,
    ), sampler


def _save_checkpoint(model: torch.nn.Module, save_dir: str, name: str) -> str:
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, name)
    if isinstance(model, DistributedDataParallel):
        model = model.module
    torch.save(model.state_dict(), path)
    return path


def train(cfg: Dict[str, Any], dry_run: bool = False, max_steps: int = 0) -> Dict[str, Any]:
    if dry_run:
        return {"status": "dry_run", "save_dir": cfg["save_dir"], "config_path": cfg["config_path"]}

    device = _device_from_config(cfg)
    distributed = _init_distributed()
    if distributed["enabled"] and device.type == "cuda":
        device = torch.device(f"cuda:{int(distributed['local_rank'])}")
    model, adapter = build_model_and_adapter(cfg)
    model.to(device)
    if distributed["enabled"]:
        device_ids = [int(distributed["local_rank"])] if device.type == "cuda" else None
        model = DistributedDataParallel(
            model,
            device_ids=device_ids,
            find_unused_parameters=True,
        )
        adapter.model = model
    anchors_full = get_tensor_anchors(str(device))
    loader, sampler = _build_loader(cfg, "train", distributed)
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
    try:
        for epoch in range(int(cfg["train"]["epochs"])):
            if sampler is not None:
                sampler.set_epoch(epoch)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            progress = tqdm(
                loader,
                desc=f"Epoch {epoch + 1}/{cfg['train']['epochs']}",
                disable=not _is_rank_zero(distributed),
            )
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
                if _is_rank_zero(distributed):
                    progress.set_postfix(
                        {
                            "loss": f"{losses.total.item():.4f}",
                            "bbox": f"{losses.bbox.item():.4f}",
                            "heatmap": f"{losses.heatmap.item():.4f}",
                        }
                    )
                if should_stop:
                    if _is_rank_zero(distributed):
                        checkpoint = _save_checkpoint(model, cfg["save_dir"], "last.pth")
                        return {"status": "max_steps", "checkpoint": checkpoint, "steps": global_step}
                    return {"status": "max_steps", "steps": global_step}

            if _is_rank_zero(distributed):
                _save_checkpoint(model, cfg["save_dir"], "last.pth")

        if _is_rank_zero(distributed):
            checkpoint = _save_checkpoint(model, cfg["save_dir"], "last.pth")
            return {"status": "ok", "checkpoint": checkpoint, "steps": global_step}
        return {"status": "ok", "steps": global_step}
    finally:
        _cleanup_distributed(distributed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train grounding-only models from YAML.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if args.device:
        cfg["train"]["device"] = args.device
    result = train(cfg, dry_run=args.dry_run, max_steps=args.max_steps)
    print(result)


if __name__ == "__main__":
    main()
