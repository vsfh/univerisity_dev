import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset import ShiftedSatelliteDroneDataset
from retrieval.config import load_config
from retrieval.losses import compute_retrieval_loss
from retrieval.registry import build_model_and_adapter, build_processors_and_tokenizer


def _device_from_config(cfg: Dict[str, Any]) -> torch.device:
    requested = str(cfg["train"]["device"])
    if requested.startswith("cuda") and torch.cuda.is_available():
        return torch.device(requested)
    return torch.device("cpu")


def _configure_precision(cfg: Dict[str, Any], device: torch.device) -> None:
    if device.type != "cuda":
        return
    enabled = bool(cfg["train"].get("enable_tf32", True))
    torch.backends.cuda.matmul.allow_tf32 = enabled
    torch.backends.cudnn.allow_tf32 = enabled


def _checkpoint_path(cfg: Dict[str, Any], name: str) -> str:
    return str(name) if os.path.isabs(str(name)) else os.path.join(str(cfg["save_dir"]), str(name))


def _build_loader(cfg: Dict[str, Any], split: str) -> DataLoader:
    (processor, processor_sat), tokenizer = build_processors_and_tokenizer(cfg)
    data_cfg = cfg["data"]
    sat_size = data_cfg["sat_size"]
    dataset = ShiftedSatelliteDroneDataset(
        processor=processor,
        processor_sat=processor_sat,
        tokenizer=tokenizer,
        split=split,
        sat_target_size=(int(sat_size["height"]), int(sat_size["width"])),
        test_crop_ratio=float(data_cfg["test_crop_ratio"]),
        subset_heights=data_cfg.get("subset_heights"),
        subset_angles=data_cfg.get("subset_angles"),
    )
    num_workers = int(data_cfg["num_workers"])
    return DataLoader(
        dataset,
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=split == "train",
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=bool(cfg["train"].get("drop_last", split == "train")),
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,
    )


def _save_checkpoint(model: torch.nn.Module, save_dir: str, name: str) -> str:
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, name)
    torch.save(model.state_dict(), path)
    return path


def train(cfg: Dict[str, Any], dry_run: bool = False, max_steps: int = 0) -> Dict[str, Any]:
    if dry_run:
        os.makedirs(cfg["save_dir"], exist_ok=True)
        return {"status": "dry_run", "save_dir": cfg["save_dir"], "config_path": cfg["config_path"]}

    device = _device_from_config(cfg)
    _configure_precision(cfg, device)
    os.makedirs(cfg["save_dir"], exist_ok=True)
    model, adapter = build_model_and_adapter(cfg)
    model.to(device)
    loader = _build_loader(cfg, "train")
    optimizer = AdamW(model.parameters(), lr=float(cfg["train"]["lr"]), weight_decay=float(cfg["train"]["weight_decay"]))
    amp_enabled = bool(cfg["train"]["amp"]) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    grad_accumulation_steps = max(1, int(cfg["train"]["grad_accumulation_steps"]))
    grad_clip_norm = float(cfg["train"]["grad_clip_norm"])
    writer = SummaryWriter(os.path.join("runs", "retrieval", str(cfg["exp_name"])))
    best_loss = float("inf")
    global_step = 0

    for epoch in range(int(cfg["train"]["epochs"])):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        count = 0
        progress = tqdm(loader, desc=f"Epoch {epoch + 1}/{cfg['train']['epochs']}")
        for batch_idx, batch in enumerate(progress):
            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                payload = adapter.forward(batch, device)
                losses = compute_retrieval_loss(
                    payload.query_feats,
                    payload.candidate_feats,
                    batch,
                    temperature=float(cfg["loss"]["temperature"]),
                    requested_granularity=str(cfg["loss"]["granularity"]),
                    text_feats=payload.text_feats,
                    use_text_loss=bool(cfg["loss"]["use_text_loss"]),
                )
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

            loss_value = float(losses.total.detach().cpu())
            total_loss += loss_value
            count += 1
            writer.add_scalar("Loss/train_batch", loss_value, global_step)
            progress.set_postfix({"loss": f"{total_loss / max(count, 1):.4f}", "mode": losses.granularity})
            if should_stop:
                break
        avg_loss = total_loss / max(count, 1)
        writer.add_scalar("Loss/train_epoch", avg_loss, epoch)
        if avg_loss < best_loss:
            best_loss = avg_loss
            _save_checkpoint(model, cfg["save_dir"], "best.pth")
        if max_steps > 0 and global_step >= max_steps:
            break

    last_path = _save_checkpoint(model, cfg["save_dir"], "last.pth")
    writer.close()
    return {"status": "ok", "save_dir": cfg["save_dir"], "checkpoint": last_path, "best_loss": best_loss}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train retrieval-only models from YAML.")
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
    print(train(cfg, dry_run=args.dry_run, max_steps=args.max_steps))


if __name__ == "__main__":
    main()
