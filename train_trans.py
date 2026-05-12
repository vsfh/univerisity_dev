#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TransGeo-style retrieval + grounding training.

The retrieval backbone follows the TransGeo idea of separate query/reference
distilled ViT branches. A lightweight patch-level decoder is added so the same
model can predict a satellite heatmap and bounding box for grounding metrics.
"""

import argparse
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bbox.yolo_utils import bbox_iou
from dataset import ShiftedSatelliteDroneDataset
from grounding.utils.utils import AverageMeter
from train_uni import (
    DRONE_SIZE,
    HEATMAP_SIGMA,
    SAT_SIZE,
    DummyTokenizer,
    TransformProcessorWrapper,
    bbox_l1_loss,
    decode_bbox,
    spatial_ce_loss,
    symmetric_info_nce,
)


# --- Configuration ---
BATCH_SIZE = 16
NUM_EPOCHS = 8
LEARNING_RATE = 4.5e-4
WEIGHT_DECAY = 0.04
PRINT_FREQ = 50
PROJECTION_DIM = 768
DETAIL_DIM = 384
EMBED_DIM = 384
DEPTH = 12
NUM_HEADS = 6
PATCH_SIZE = 16
RETRIEVAL_LOSS_WEIGHT = 1.0
LOCALIZATION_LOSS_WEIGHT = 1.0
BBOX_LOSS_WEIGHT = 5.0
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


class PatchEmbed(nn.Module):
    def __init__(
        self,
        image_size: Tuple[int, int],
        patch_size: int = PATCH_SIZE,
        embed_dim: int = EMBED_DIM,
    ):
        super().__init__()
        image_w, image_h = int(image_size[0]), int(image_size[1])
        self.grid_size = (image_h // patch_size, image_w // patch_size)
        self.proj = nn.Conv2d(
            3,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x):
        x = self.proj(x)
        grid_hw = (int(x.shape[-2]), int(x.shape[-1]))
        x = x.flatten(2).transpose(1, 2)
        return x, grid_hw


class DistilledViTBranch(nn.Module):
    """DeiT-small distilled style branch used by TransGeo."""

    def __init__(
        self,
        image_size: Tuple[int, int],
        embed_dim: int = EMBED_DIM,
        depth: int = DEPTH,
        num_heads: int = NUM_HEADS,
        proj_dim: int = PROJECTION_DIM,
        patch_size: int = PATCH_SIZE,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.patch_embed = PatchEmbed(
            image_size=image_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
        )
        base_h, base_w = self.patch_embed.grid_size
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, base_h * base_w + 2, embed_dim))
        self.pos_drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=embed_dim,
                    nhead=num_heads,
                    dim_feedforward=int(embed_dim * mlp_ratio),
                    dropout=dropout,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, proj_dim)
        self.head_dist = nn.Linear(embed_dim, proj_dim)
        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.dist_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def _pos_embed_for_grid(self, grid_hw: Tuple[int, int]) -> torch.Tensor:
        grid_h, grid_w = int(grid_hw[0]), int(grid_hw[1])
        cls_pos = self.pos_embed[:, :2]
        patch_pos = self.pos_embed[:, 2:]
        base_h, base_w = self.patch_embed.grid_size
        if (grid_h, grid_w) == (base_h, base_w):
            return self.pos_embed
        patch_pos = patch_pos.reshape(1, base_h, base_w, -1).permute(0, 3, 1, 2)
        patch_pos = F.interpolate(
            patch_pos,
            size=(grid_h, grid_w),
            mode="bicubic",
            align_corners=False,
        )
        patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, grid_h * grid_w, -1)
        return torch.cat([cls_pos, patch_pos], dim=1)

    def forward_features(self, x):
        patch_tokens, grid_hw = self.patch_embed(x)
        batch_size = int(patch_tokens.shape[0])
        cls_token = self.cls_token.expand(batch_size, -1, -1)
        dist_token = self.dist_token.expand(batch_size, -1, -1)
        tokens = torch.cat([cls_token, dist_token, patch_tokens], dim=1)
        tokens = self.pos_drop(tokens + self._pos_embed_for_grid(grid_hw))
        for block in self.blocks:
            tokens = block(tokens)
        tokens = self.norm(tokens)
        return tokens, grid_hw

    def forward(self, x):
        tokens, _ = self.forward_features(x)
        cls_out = self.head(tokens[:, 0])
        dist_out = self.head_dist(tokens[:, 1])
        return F.normalize(0.5 * (cls_out + dist_out), p=2, dim=1)


class TransGeoGrounding(nn.Module):
    def __init__(
        self,
        sat_size: Tuple[int, int] = SAT_SIZE,
        drone_size: Tuple[int, int] = DRONE_SIZE,
        proj_dim: int = PROJECTION_DIM,
        detail_dim: int = DETAIL_DIM,
        embed_dim: int = EMBED_DIM,
        depth: int = DEPTH,
        num_heads: int = NUM_HEADS,
        patch_size: int = PATCH_SIZE,
    ):
        super().__init__()
        self.query_net = DistilledViTBranch(
            image_size=drone_size,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            proj_dim=proj_dim,
            patch_size=patch_size,
        )
        self.reference_net = DistilledViTBranch(
            image_size=sat_size,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            proj_dim=proj_dim,
            patch_size=patch_size,
        )
        self.query_detail = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, detail_dim),
        )
        self.reference_detail = nn.Sequential(
            nn.Conv2d(embed_dim, detail_dim, kernel_size=1),
            nn.BatchNorm2d(detail_dim),
            nn.GELU(),
            nn.Conv2d(detail_dim, detail_dim, kernel_size=3, padding=1),
        )
        self.bbox_head = nn.Sequential(
            nn.Conv2d(detail_dim * 2, detail_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(detail_dim),
            nn.GELU(),
            nn.Conv2d(detail_dim, 4, kernel_size=1),
        )
        self.logit_scale = nn.Parameter(torch.tensor(np.log(1 / 0.07), dtype=torch.float32))
        self.temperature = nn.Parameter(torch.tensor(0.07, dtype=torch.float32))

    def _reference_feature_map(self, tokens, grid_hw: Tuple[int, int]):
        patch_tokens = tokens[:, 2:]
        grid_h, grid_w = int(grid_hw[0]), int(grid_hw[1])
        return patch_tokens.transpose(1, 2).reshape(tokens.shape[0], -1, grid_h, grid_w)

    def forward(self, query_imgs, aerial_imgs):
        query_tokens, _ = self.query_net.forward_features(query_imgs)
        aerial_tokens, aerial_grid_hw = self.reference_net.forward_features(aerial_imgs)

        query_global = F.normalize(
            0.5
            * (
                self.query_net.head(query_tokens[:, 0])
                + self.query_net.head_dist(query_tokens[:, 1])
            ),
            p=2,
            dim=1,
        )
        aerial_global = F.normalize(
            0.5
            * (
                self.reference_net.head(aerial_tokens[:, 0])
                + self.reference_net.head_dist(aerial_tokens[:, 1])
            ),
            p=2,
            dim=1,
        )

        query_patch_mean = query_tokens[:, 2:].mean(dim=1)
        ground_detail = F.normalize(self.query_detail(query_patch_mean), p=2, dim=1)
        aerial_map = self._reference_feature_map(aerial_tokens, aerial_grid_hw)
        aerial_detail = F.normalize(self.reference_detail(aerial_map), p=2, dim=1)

        temp = self.temperature.clamp(min=0.01, max=1.0)
        heatmap_logits = torch.einsum("bd,bdhw->bhw", ground_detail, aerial_detail) / temp
        query_map = ground_detail[:, :, None, None].expand_as(aerial_detail)
        bbox_raw = self.bbox_head(torch.cat([aerial_detail, query_map], dim=1))

        scale = self.logit_scale.exp().clamp(max=100.0)
        retrieval_logits = scale * query_global @ aerial_global.t()

        return {
            "query_global": query_global,
            "aerial_global": aerial_global,
            "ground_detail": ground_detail,
            "aerial_detail": aerial_detail,
            "heatmap_logits": heatmap_logits.unsqueeze(1),
            "bbox_raw": bbox_raw,
            "retrieval_logits": retrieval_logits,
        }

    def bbox_forward(self, query_imgs, aerial_imgs):
        outputs = self.forward(query_imgs, aerial_imgs)
        return decode_bbox(
            outputs["heatmap_logits"],
            outputs["bbox_raw"],
            image_wh=(aerial_imgs.shape[-1], aerial_imgs.shape[-2]),
        )


def compute_losses(outputs: Dict[str, torch.Tensor], gt_bbox, image_wh, args):
    retrieval_loss = symmetric_info_nce(
        outputs["retrieval_logits"],
        label_smoothing=args.label_smoothing,
    )
    localization_loss = spatial_ce_loss(
        outputs["heatmap_logits"],
        gt_bbox,
        image_wh,
        sigma=args.heatmap_sigma,
    )
    bbox_loss = bbox_l1_loss(outputs["bbox_raw"], gt_bbox, image_wh)
    total = (
        args.retrieval_loss_weight * retrieval_loss
        + args.localization_loss_weight * localization_loss
        + args.bbox_loss_weight * bbox_loss
    )
    return total, {
        "retrieval": retrieval_loss,
        "localization": localization_loss,
        "bbox": bbox_loss,
    }


def train_epoch(loader, model, optimizer, epoch, writer, args):
    model.train()
    meters = {
        "loss": AverageMeter(),
        "retrieval": AverageMeter(),
        "localization": AverageMeter(),
        "bbox": AverageMeter(),
    }
    batch_time = AverageMeter()
    end = time.time()

    for batch_idx, batch in enumerate(loader):
        global_step = epoch * len(loader) + batch_idx
        query_imgs = batch["target_pixel_values"].to(DEVICE, non_blocking=True)
        aerial_imgs = batch["search_pixel_values"].to(DEVICE, non_blocking=True)
        gt_bbox = batch["bbox"].to(DEVICE, non_blocking=True)
        image_wh = (aerial_imgs.shape[-1], aerial_imgs.shape[-2])

        outputs = model(query_imgs, aerial_imgs)
        loss, loss_items = compute_losses(outputs, gt_bbox, image_wh, args)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if args.grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        bs = query_imgs.shape[0]
        meters["loss"].update(loss.item(), bs)
        for name, value in loss_items.items():
            meters[name].update(value.item(), bs)
        batch_time.update(time.time() - end)
        end = time.time()

        if writer is not None:
            writer.add_scalar("TrainStep/loss", loss.item(), global_step)
            for name, value in loss_items.items():
                writer.add_scalar(f"TrainStep/{name}", value.item(), global_step)
            writer.add_scalar("lr/learning_rate", optimizer.param_groups[0]["lr"], global_step)

        if batch_idx % args.print_freq == 0:
            print(
                f"Epoch: [{epoch}][{batch_idx}/{len(loader)}]\t"
                f"Time: {batch_time.val:.3f}\t"
                f"Loss: {meters['loss'].val:.4f} ({meters['loss'].avg:.4f})\t"
                f"Ret: {meters['retrieval'].val:.4f}\t"
                f"Loc: {meters['localization'].val:.4f}\t"
                f"Bbox: {meters['bbox'].val:.4f}"
            )

    return {name: meter.avg for name, meter in meters.items()}


def validate(loader, model):
    model.eval()
    iou_values: List[float] = []
    center_distances: List[float] = []
    top1_hits = 0
    total = 0

    with torch.inference_mode():
        for batch in tqdm(loader, desc="Validating"):
            query_imgs = batch["target_pixel_values"].to(DEVICE, non_blocking=True)
            aerial_imgs = batch["search_pixel_values"].to(DEVICE, non_blocking=True)
            gt_bbox = batch["bbox"].to(DEVICE, non_blocking=True)
            outputs = model(query_imgs, aerial_imgs)
            pred_bbox = decode_bbox(
                outputs["heatmap_logits"],
                outputs["bbox_raw"],
                image_wh=(aerial_imgs.shape[-1], aerial_imgs.shape[-2]),
            )
            labels = torch.arange(outputs["retrieval_logits"].shape[0], device=DEVICE)
            top1_hits += int((outputs["retrieval_logits"].argmax(dim=1) == labels).sum().item())
            total += int(labels.numel())

            for idx in range(pred_bbox.shape[0]):
                pred_xyxy = pred_bbox[idx].float()
                gt_xyxy = gt_bbox[idx].float()
                iou = float(
                    bbox_iou(
                        pred_xyxy.unsqueeze(0),
                        gt_xyxy.unsqueeze(0),
                        x1y1x2y2=True,
                    ).item()
                )
                pred_cx = 0.5 * (pred_xyxy[0] + pred_xyxy[2])
                pred_cy = 0.5 * (pred_xyxy[1] + pred_xyxy[3])
                gt_cx = 0.5 * (gt_xyxy[0] + gt_xyxy[2])
                gt_cy = 0.5 * (gt_xyxy[1] + gt_xyxy[3])
                center_dist = float(
                    torch.sqrt((pred_cx - gt_cx) ** 2 + (pred_cy - gt_cy) ** 2).item()
                )
                iou_values.append(iou)
                center_distances.append(center_dist)

    if not iou_values:
        return {"mean_iou": 0.0, "accu50": 0.0, "center_distance": 0.0, "retrieval_top1": 0.0}

    iou_arr = np.array(iou_values, dtype=np.float32)
    center_arr = np.array(center_distances, dtype=np.float32)
    return {
        "mean_iou": float(iou_arr.mean()),
        "accu50": float((iou_arr > 0.5).mean()),
        "center_distance": float(center_arr.mean()),
        "retrieval_top1": float(top1_hits / max(total, 1)),
    }


def build_dataloaders(args):
    processor = TransformProcessorWrapper(DRONE_SIZE)
    processor_sat = TransformProcessorWrapper(args.sat_size)
    tokenizer = DummyTokenizer()

    train_dataset = ShiftedSatelliteDroneDataset(
        processor=processor,
        processor_sat=processor_sat,
        tokenizer=tokenizer,
        split="train",
    )
    val_dataset = ShiftedSatelliteDroneDataset(
        processor=processor,
        processor_sat=processor_sat,
        tokenizer=tokenizer,
        split="test",
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=4 if args.num_workers > 0 else None,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=4 if args.num_workers > 0 else None,
    )
    return train_loader, val_loader, len(train_dataset), len(val_dataset)


def adjust_learning_rate(args, optimizer, epoch):
    if epoch < args.warmup_epochs:
        lr = args.lr * float(epoch + 1) / max(float(args.warmup_epochs), 1.0)
    else:
        progress = (epoch - args.warmup_epochs) / max(float(args.max_epoch - args.warmup_epochs), 1.0)
        lr = args.min_lr + 0.5 * (args.lr - args.min_lr) * (1.0 + np.cos(np.pi * progress))
    print(("lr", lr))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def main(args):
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    os.makedirs(args.checkpoint, exist_ok=True)
    writer = SummaryWriter(f"runs/{args.savename}")

    print("Creating datasets from shared data source...")
    train_loader, val_loader, train_count, val_count = build_dataloaders(args)
    print(f"Found {train_count} training samples, {val_count} validation samples")

    print("Creating TransGeoGrounding model...")
    model = TransGeoGrounding(
        sat_size=args.sat_size,
        drone_size=DRONE_SIZE,
        proj_dim=args.proj_dim,
        detail_dim=args.detail_dim,
        embed_dim=args.embed_dim,
        depth=args.depth,
        num_heads=args.num_heads,
        patch_size=args.patch_size,
    ).to(DEVICE)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_iou = -1.0

    print(f"Starting training for {args.max_epoch} epochs...")
    for epoch in range(args.max_epoch):
        adjust_learning_rate(args, optimizer, epoch)
        train_metrics = train_epoch(train_loader, model, optimizer, epoch, writer, args)
        val_metrics = validate(val_loader, model)

        for name, value in train_metrics.items():
            writer.add_scalar(f"Train/{name}", value, epoch)
        for name, value in val_metrics.items():
            writer.add_scalar(f"Val/{name}", value, epoch)

        print(
            f"Epoch {epoch + 1}/{args.max_epoch}:\t"
            f"Train Loss: {train_metrics['loss']:.4f}\t"
            f"Ret: {train_metrics['retrieval']:.4f}\t"
            f"Loc: {train_metrics['localization']:.4f}\t"
            f"Bbox: {train_metrics['bbox']:.4f}\t"
            f"Val mIoU: {val_metrics['mean_iou']:.4f}\t"
            f"Val R@1: {val_metrics['retrieval_top1']:.4f}"
        )

        torch.save(model.state_dict(), os.path.join(args.checkpoint, "last.pth"))
        if val_metrics["mean_iou"] > best_iou:
            best_iou = val_metrics["mean_iou"]
            torch.save(model.state_dict(), os.path.join(args.checkpoint, "best_iou.pth"))

    print("\nTraining complete. Saved checkpoint to last.pth")
    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TransGeo-style training on ShiftedSatelliteDroneDataset")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--max-epoch", type=int, default=NUM_EPOCHS)
    parser.add_argument("--warmup-epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument(
        "--sat-size",
        type=int,
        nargs=2,
        metavar=("WIDTH", "HEIGHT"),
        default=list(SAT_SIZE),
        help="satellite image size as WIDTH HEIGHT",
    )
    parser.add_argument("--savename", type=str, default="trans_geo")
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--print-freq", type=int, default=PRINT_FREQ)
    parser.add_argument("--proj-dim", type=int, default=PROJECTION_DIM)
    parser.add_argument("--detail-dim", type=int, default=DETAIL_DIM)
    parser.add_argument("--embed-dim", type=int, default=EMBED_DIM)
    parser.add_argument("--depth", type=int, default=DEPTH)
    parser.add_argument("--num-heads", type=int, default=NUM_HEADS)
    parser.add_argument("--patch-size", type=int, default=PATCH_SIZE)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--heatmap-sigma", type=float, default=HEATMAP_SIGMA)
    parser.add_argument("--retrieval-loss-weight", type=float, default=RETRIEVAL_LOSS_WEIGHT)
    parser.add_argument("--localization-loss-weight", type=float, default=LOCALIZATION_LOSS_WEIGHT)
    parser.add_argument("--bbox-loss-weight", type=float, default=BBOX_LOSS_WEIGHT)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="/media/data1/feihong/ckpt/trans_geo",
        help="Path to save model checkpoints",
    )
    args = parser.parse_args()
    args.sat_size = (int(args.sat_size[0]), int(args.sat_size[1]))
    main(args)
