import argparse
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
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
from grounding.letterbox import black_fill_from_processor, letterbox_satellite_batch
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


def _normalize_for_writer(images: torch.Tensor) -> torch.Tensor:
    images = images.detach().float().cpu()
    images = images[:, :3]
    image_min = images.amin(dim=(1, 2, 3), keepdim=True)
    image_max = images.amax(dim=(1, 2, 3), keepdim=True)
    return ((images - image_min) / (image_max - image_min).clamp_min(1e-6)).clamp(0.0, 1.0)


def _draw_bbox(image: torch.Tensor, bbox: torch.Tensor, color: tuple[float, float, float]) -> torch.Tensor:
    _, height, width = image.shape
    x1, y1, x2, y2 = bbox.detach().float().cpu().tolist()
    x1 = max(0, min(width - 1, int(round(x1))))
    x2 = max(0, min(width - 1, int(round(x2))))
    y1 = max(0, min(height - 1, int(round(y1))))
    y2 = max(0, min(height - 1, int(round(y2))))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    color_tensor = image.new_tensor(color).view(3, 1, 1)
    thickness = max(2, min(height, width) // 160)
    image[:, y1 : min(y1 + thickness, height), x1 : x2 + 1] = color_tensor
    image[:, max(y2 - thickness + 1, 0) : y2 + 1, x1 : x2 + 1] = color_tensor
    image[:, y1 : y2 + 1, x1 : min(x1 + thickness, width)] = color_tensor
    image[:, y1 : y2 + 1, max(x2 - thickness + 1, 0) : x2 + 1] = color_tensor
    return image


def _prediction_visualizations(
    batch: Dict[str, Any],
    pred_bbox: torch.Tensor,
    max_cases: int = 4,
) -> torch.Tensor:
    query = _normalize_for_writer(batch["target_pixel_values"][:max_cases])
    search = _normalize_for_writer(batch["search_pixel_values"][:max_cases])
    target_bbox = batch["bbox"][:max_cases].detach().float().cpu()
    pred_bbox = pred_bbox[:max_cases].detach().float().cpu()

    case_count = min(query.shape[0], search.shape[0], target_bbox.shape[0], pred_bbox.shape[0])
    query = query[:case_count]
    search = search[:case_count]
    target_bbox = target_bbox[:case_count]
    pred_bbox = pred_bbox[:case_count]

    sat_h, sat_w = int(search.shape[-2]), int(search.shape[-1])
    query_w = max(1, int(round(query.shape[-1] * sat_h / max(float(query.shape[-2]), 1.0))))
    query = F.interpolate(query, size=(sat_h, query_w), mode="bilinear", align_corners=False)

    images = []
    for idx in range(case_count):
        sat = search[idx].clone()
        sat = _draw_bbox(sat, target_bbox[idx], (0.0, 1.0, 0.0))
        sat = _draw_bbox(sat, pred_bbox[idx], (1.0, 0.0, 0.0))
        canvas = torch.zeros((3, sat_h, query_w + sat_w), dtype=sat.dtype)
        canvas[:, :, :query_w] = query[idx]
        canvas[:, :, query_w:] = sat
        images.append(canvas)
    if not images:
        return torch.empty((0, 3, sat_h, query_w + sat_w))
    return torch.stack(images, dim=0)


def train(cfg: Dict[str, Any], dry_run: bool = False, max_steps: int = 0) -> Dict[str, Any]:
    if dry_run:
        os.makedirs(cfg["save_dir"], exist_ok=True)
        return {"status": "dry_run", "save_dir": cfg["save_dir"], "config_path": cfg["config_path"]}

    device = _device_from_config(cfg)
    distributed = _init_distributed()
    if _is_rank_zero(distributed):
        os.makedirs(cfg["save_dir"], exist_ok=True)
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
    letterbox_fill = black_fill_from_processor(loader.dataset.processor_sat)
    optimizer = AdamW(
        model.parameters(),
        lr=float(cfg["train"]["lr"]),
        weight_decay=float(cfg["train"]["weight_decay"]),
    )
    amp_enabled = bool(cfg["train"]["amp"]) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    grad_accumulation_steps = max(1, int(cfg["train"]["grad_accumulation_steps"]))
    grad_clip_norm = float(cfg["train"]["grad_clip_norm"])
    writer = None
    if _is_rank_zero(distributed):
        writer = SummaryWriter(os.path.join("runs", "grounding", str(cfg["exp_name"])))

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
            epoch_totals = {
                "total": 0.0,
                "bbox": 0.0,
                "geo": 0.0,
                "cls": 0.0,
                "heatmap": 0.0,
            }
            epoch_count = 0
            for batch_idx, batch in enumerate(progress):
                batch, _ = letterbox_satellite_batch(batch, letterbox_fill)
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
                    batch_n = int(batch["bbox"].shape[0])
                    epoch_count += batch_n
                    epoch_totals["total"] += float(losses.total.detach().item()) * batch_n
                    epoch_totals["bbox"] += float(losses.bbox.detach().item()) * batch_n
                    epoch_totals["geo"] += float(losses.geo.detach().item()) * batch_n
                    epoch_totals["cls"] += float(losses.cls.detach().item()) * batch_n
                    epoch_totals["heatmap"] += float(losses.heatmap.detach().item()) * batch_n
                    progress.set_postfix(
                        {
                            "loss": f"{losses.total.item():.4f}",
                            "bbox": f"{losses.bbox.item():.4f}",
                            "heatmap": f"{losses.heatmap.item():.4f}",
                        }
                    )
                    if writer is not None:
                        writer.add_scalar("Loss/train_step", losses.total.item(), global_step)
                        writer.add_scalar("Loss/bbox_step", losses.bbox.item(), global_step)
                        writer.add_scalar("Loss/geo_step", losses.geo.item(), global_step)
                        writer.add_scalar("Loss/cls_step", losses.cls.item(), global_step)
                        writer.add_scalar("Loss/heatmap_step", losses.heatmap.item(), global_step)
                        if batch_idx == 0:
                            with torch.no_grad():
                                pred_bbox = adapter.decode(output, batch, anchors_full)
                            images = _prediction_visualizations(batch, pred_bbox)
                            if images.numel() > 0:
                                writer.add_images("Train/predictions", images, epoch)
                if should_stop:
                    if _is_rank_zero(distributed):
                        checkpoint = _save_checkpoint(model, cfg["save_dir"], "last.pth")
                        return {"status": "max_steps", "checkpoint": checkpoint, "steps": global_step}
                    return {"status": "max_steps", "steps": global_step}

            if _is_rank_zero(distributed):
                if writer is not None and epoch_count > 0:
                    writer.add_scalar("Loss/train_epoch", epoch_totals["total"] / epoch_count, epoch)
                    writer.add_scalar("Loss/bbox_epoch", epoch_totals["bbox"] / epoch_count, epoch)
                    writer.add_scalar("Loss/geo_epoch", epoch_totals["geo"] / epoch_count, epoch)
                    writer.add_scalar("Loss/cls_epoch", epoch_totals["cls"] / epoch_count, epoch)
                    writer.add_scalar("Loss/heatmap_epoch", epoch_totals["heatmap"] / epoch_count, epoch)
                _save_checkpoint(model, cfg["save_dir"], "last.pth")

        if _is_rank_zero(distributed):
            checkpoint = _save_checkpoint(model, cfg["save_dir"], "last.pth")
            return {"status": "ok", "checkpoint": checkpoint, "steps": global_step}
        return {"status": "ok", "steps": global_step}
    finally:
        if writer is not None:
            writer.close()
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
