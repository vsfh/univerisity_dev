#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@file train_uni.py
@desc UnifyGeo-style joint retrieval + localization training on ShiftedSatelliteDroneDataset.

The implementation adapts the ideas from arXiv:2505.07622:
- pseudo-Siamese ground/aerial branches;
- multi-granularity representations with semantic/global and fine/local heads;
- self-attention + GeM aggregation for retrieval descriptors;
- detailed matching heatmap for metric localization;
- batch-wise re-ranking supervision from detailed matching scores.
"""

import argparse
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
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


# --- Configuration ---
SAT_SIZE = (768, 432)  # (width, height)
DRONE_SIZE = (256, 256)  # (width, height)
BATCH_SIZE = 16
NUM_EPOCHS = 8
LEARNING_RATE = 4.5e-4
WEIGHT_DECAY = 0.04
PRINT_FREQ = 50
PROJECTION_DIM = 768
HEATMAP_SIGMA = 1.5
RETRIEVAL_LOSS_WEIGHT = 1.0
LOCALIZATION_LOSS_WEIGHT = 1.0
BBOX_LOSS_WEIGHT = 5.0
RERANK_LOSS_WEIGHT = 1.0
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


class TransformProcessorWrapper:
    def __init__(self, image_size: Tuple[int, int]):
        self.size = {"height": int(image_size[1]), "width": int(image_size[0])}

    def __call__(self, images, return_tensors="pt"):
        target_h = int(self.size["height"])
        target_w = int(self.size["width"])
        if images.size != (target_w, target_h):
            images = images.resize((target_w, target_h), Image.Resampling.BILINEAR)
        image_np = np.array(images, dtype=np.float32) / 255.0
        pixel_values = torch.from_numpy(image_np).permute(2, 0, 1)
        return {"pixel_values": pixel_values.unsqueeze(0)}


class DummyTokenizer:
    def __call__(
        self,
        text,
        padding="max_length",
        truncation=True,
        max_length=64,
        return_tensors="pt",
    ):
        del text, padding, truncation, return_tensors
        input_ids = torch.zeros((1, max_length), dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        return {"input_ids": input_ids, "attention_mask": attention_mask}


class LayerNorm2d(nn.Module):
    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(channels, eps=eps)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        return x.permute(0, 3, 1, 2).contiguous()


class ConvNeXtBlock(nn.Module):
    def __init__(self, channels: int, layer_scale_init: float = 1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(channels, channels, kernel_size=7, padding=3, groups=channels)
        self.norm = nn.LayerNorm(channels)
        self.pwconv1 = nn.Linear(channels, channels * 4)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(channels * 4, channels)
        self.gamma = nn.Parameter(layer_scale_init * torch.ones(channels))

    def forward(self, x):
        shortcut = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = self.gamma * x
        x = x.permute(0, 3, 1, 2).contiguous()
        return shortcut + x


class Downsample(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, stride: int):
        super().__init__(
            LayerNorm2d(in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=stride, stride=stride),
        )


class BranchEncoder(nn.Module):
    """Shared branch encoder plus separate semantic/fine extractor heads."""

    def __init__(self, dims: Tuple[int, int, int] = (96, 192, 384)):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, dims[0], kernel_size=4, stride=4),
            LayerNorm2d(dims[0]),
        )
        self.stage1 = nn.Sequential(ConvNeXtBlock(dims[0]), ConvNeXtBlock(dims[0]))
        self.down1 = Downsample(dims[0], dims[1], stride=2)
        self.stage2 = nn.Sequential(ConvNeXtBlock(dims[1]), ConvNeXtBlock(dims[1]))
        self.down2 = Downsample(dims[1], dims[2], stride=2)
        self.stage3 = nn.Sequential(ConvNeXtBlock(dims[2]), ConvNeXtBlock(dims[2]))

        self.fine_head = nn.Sequential(ConvNeXtBlock(dims[2]), ConvNeXtBlock(dims[2]))
        self.semantic_head = nn.Sequential(ConvNeXtBlock(dims[2]), ConvNeXtBlock(dims[2]))
        self.out_dim = dims[2]

    def forward(self, x):
        x1 = self.stage1(self.stem(x))
        x2 = self.stage2(self.down1(x1))
        x3 = self.stage3(self.down2(x2))
        fine = self.fine_head(x3)
        semantic = self.semantic_head(x3)
        return {
            "multi_scale": (x1, x2, fine),
            "fine": fine,
            "semantic": semantic,
        }


class GeM(nn.Module):
    def __init__(self, p: float = 3.0, eps: float = 1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.tensor(float(p)))
        self.eps = float(eps)

    def forward(self, x):
        p = self.p.clamp(min=1.0, max=8.0)
        x = x.clamp(min=self.eps).pow(p)
        x = F.adaptive_avg_pool2d(x, 1).pow(1.0 / p)
        return x.flatten(1)


class SelfAttentionAggregator(nn.Module):
    """Single-head attention aggregator followed by GeM, as in UnifyGeo."""

    def __init__(self, in_channels: int, proj_dim: int):
        super().__init__()
        hidden = max(in_channels // 2, 64)
        self.norm = nn.LayerNorm(in_channels)
        self.q = nn.Linear(in_channels, hidden)
        self.k = nn.Linear(in_channels, hidden)
        self.v = nn.Linear(in_channels, hidden)
        self.restore = nn.Linear(hidden, in_channels)
        self.gem = GeM()
        self.project = nn.Linear(in_channels, proj_dim)

    def forward(self, feat):
        B, C, H, W = feat.shape
        tokens = feat.flatten(2).transpose(1, 2)
        tokens_norm = self.norm(tokens)
        q = self.q(tokens_norm)
        k = self.k(tokens_norm)
        v = self.v(tokens_norm)
        attn = torch.matmul(q, k.transpose(1, 2)) / max(q.shape[-1] ** 0.5, 1.0)
        tokens = tokens + self.restore(torch.matmul(attn.softmax(dim=-1), v))
        feat = tokens.transpose(1, 2).reshape(B, C, H, W)
        return F.normalize(self.project(self.gem(feat)), p=2, dim=1)


class GroundDetailProjector(nn.Module):
    """Column-aware detailed descriptor projector for the ground branch."""

    def __init__(self, in_channels: int, detail_dim: int):
        super().__init__()
        self.reduce = nn.Sequential(
            nn.Conv2d(in_channels, detail_dim, kernel_size=1),
            nn.BatchNorm2d(detail_dim),
            nn.GELU(),
        )
        self.column_fc = nn.Sequential(
            nn.Conv1d(detail_dim, detail_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(detail_dim, detail_dim, kernel_size=1),
        )

    def forward(self, feat):
        feat = self.reduce(feat)
        # Aggregate vertically and then summarize azimuth/column information.
        columns = feat.mean(dim=2)
        columns = self.column_fc(columns)
        return F.normalize(columns.mean(dim=-1), p=2, dim=1)


class AerialDetailProjector(nn.Module):
    def __init__(self, in_channels: int, detail_dim: int):
        super().__init__()
        self.project = nn.Sequential(
            nn.Conv2d(in_channels, detail_dim, kernel_size=1),
            nn.BatchNorm2d(detail_dim),
            nn.GELU(),
            nn.Conv2d(detail_dim, detail_dim, kernel_size=3, padding=1),
        )

    def forward(self, feat):
        return F.normalize(self.project(feat), p=2, dim=1)


class LocalizationDecoder(nn.Module):
    def __init__(self, detail_dim: int):
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor(0.07))
        self.bbox_head = nn.Sequential(
            nn.Conv2d(detail_dim * 2, detail_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(detail_dim),
            nn.GELU(),
            nn.Conv2d(detail_dim, 4, kernel_size=1),
        )

    def forward(self, ground_detail, aerial_detail):
        temp = self.temperature.clamp(min=0.01, max=1.0)
        heatmap_logits = torch.einsum("bd,bdhw->bhw", ground_detail, aerial_detail) / temp
        query_map = ground_detail[:, :, None, None].expand_as(aerial_detail)
        bbox_raw = self.bbox_head(torch.cat([aerial_detail, query_map], dim=1))
        return heatmap_logits.unsqueeze(1), bbox_raw

    def batch_rerank_logits(self, ground_detail, aerial_detail, retrieval_logits):
        temp = self.temperature.clamp(min=0.01, max=1.0)
        detail_scores = torch.einsum("bd,kdhw->bkhw", ground_detail, aerial_detail) / temp
        detail_scores = detail_scores.flatten(2).max(dim=-1)[0]
        return retrieval_logits + detail_scores


class UnifyGeoLite(nn.Module):
    def __init__(
        self,
        proj_dim: int = PROJECTION_DIM,
        detail_dim: int = 384,
        dims: Tuple[int, int, int] = (96, 192, 384),
    ):
        super().__init__()
        self.ground_encoder = BranchEncoder(dims=dims)
        self.aerial_encoder = BranchEncoder(dims=dims)
        self.ground_aggregator = SelfAttentionAggregator(dims[-1], proj_dim)
        self.aerial_aggregator = SelfAttentionAggregator(dims[-1], proj_dim)
        self.ground_detail = GroundDetailProjector(dims[-1], detail_dim)
        self.aerial_detail = AerialDetailProjector(dims[-1], detail_dim)
        self.decoder = LocalizationDecoder(detail_dim)
        self.logit_scale = nn.Parameter(torch.tensor(np.log(1 / 0.07), dtype=torch.float32))

    def forward(self, query_imgs, aerial_imgs):
        ground = self.ground_encoder(query_imgs)
        aerial = self.aerial_encoder(aerial_imgs)

        query_global = self.ground_aggregator(ground["semantic"])
        aerial_global = self.aerial_aggregator(aerial["semantic"])
        ground_detail = self.ground_detail(ground["fine"])
        aerial_detail = self.aerial_detail(aerial["fine"])

        heatmap_logits, bbox_raw = self.decoder(ground_detail, aerial_detail)
        scale = self.logit_scale.exp().clamp(max=100.0)
        retrieval_logits = scale * query_global @ aerial_global.t()
        rerank_logits = self.decoder.batch_rerank_logits(
            ground_detail,
            aerial_detail,
            retrieval_logits,
        )

        return {
            "query_global": query_global,
            "aerial_global": aerial_global,
            "ground_detail": ground_detail,
            "aerial_detail": aerial_detail,
            "heatmap_logits": heatmap_logits,
            "bbox_raw": bbox_raw,
            "retrieval_logits": retrieval_logits,
            "rerank_logits": rerank_logits,
        }

    def bbox_forward(self, query_imgs, aerial_imgs):
        outputs = self.forward(query_imgs, aerial_imgs)
        return decode_bbox(
            outputs["heatmap_logits"],
            outputs["bbox_raw"],
            image_wh=(aerial_imgs.shape[-1], aerial_imgs.shape[-2]),
        )


def symmetric_info_nce(logits, label_smoothing: float = 0.1):
    labels = torch.arange(logits.shape[0], device=logits.device)
    loss_q2r = F.cross_entropy(logits, labels, label_smoothing=label_smoothing)
    loss_r2q = F.cross_entropy(logits.t(), labels, label_smoothing=label_smoothing)
    return 0.5 * (loss_q2r + loss_r2q)


def build_heatmap_target(gt_bbox, feature_hw: Tuple[int, int], image_wh: Tuple[int, int], sigma: float):
    B = gt_bbox.shape[0]
    feat_h, feat_w = int(feature_hw[0]), int(feature_hw[1])
    image_w, image_h = float(image_wh[0]), float(image_wh[1])
    center_x = ((gt_bbox[:, 0] + gt_bbox[:, 2]) * 0.5 / image_w) * feat_w
    center_y = ((gt_bbox[:, 1] + gt_bbox[:, 3]) * 0.5 / image_h) * feat_h
    grid_y = torch.arange(feat_h, device=gt_bbox.device, dtype=gt_bbox.dtype).view(1, feat_h, 1)
    grid_x = torch.arange(feat_w, device=gt_bbox.device, dtype=gt_bbox.dtype).view(1, 1, feat_w)
    heatmap = torch.exp(
        -0.5
        * (
            (grid_x - center_x.view(B, 1, 1)).square()
            + (grid_y - center_y.view(B, 1, 1)).square()
        )
        / max(float(sigma) ** 2, 1e-6)
    )
    return heatmap.unsqueeze(1).clamp_(0.0, 1.0)


def spatial_ce_loss(heatmap_logits, gt_bbox, image_wh: Tuple[int, int], sigma: float):
    B, _, H, W = heatmap_logits.shape
    target = build_heatmap_target(gt_bbox, (H, W), image_wh, sigma=sigma)
    target = target.flatten(1)
    target = target / target.sum(dim=1, keepdim=True).clamp_min(1e-6)
    log_probs = F.log_softmax(heatmap_logits.flatten(1), dim=1)
    return -(target * log_probs).sum(dim=1).mean()


def build_bbox_target(gt_bbox, feature_hw: Tuple[int, int], image_wh: Tuple[int, int]):
    feat_h, feat_w = int(feature_hw[0]), int(feature_hw[1])
    image_w, image_h = float(image_wh[0]), float(image_wh[1])
    cx = ((gt_bbox[:, 0] + gt_bbox[:, 2]) * 0.5 / image_w) * feat_w
    cy = ((gt_bbox[:, 1] + gt_bbox[:, 3]) * 0.5 / image_h) * feat_h
    gi = cx.floor().long().clamp_(0, feat_w - 1)
    gj = cy.floor().long().clamp_(0, feat_h - 1)
    frac_x = cx - gi.to(dtype=gt_bbox.dtype)
    frac_y = cy - gj.to(dtype=gt_bbox.dtype)
    box_w = (gt_bbox[:, 2] - gt_bbox[:, 0]).clamp_min(1.0) / image_w
    box_h = (gt_bbox[:, 3] - gt_bbox[:, 1]).clamp_min(1.0) / image_h
    target = torch.stack([frac_x, frac_y, box_w, box_h], dim=1)
    return target, gi, gj


def bbox_l1_loss(bbox_raw, gt_bbox, image_wh: Tuple[int, int]):
    B, _, H, W = bbox_raw.shape
    target, gi, gj = build_bbox_target(gt_bbox, (H, W), image_wh)
    batch_idx = torch.arange(B, device=bbox_raw.device)
    selected = bbox_raw[batch_idx, :, gj, gi]
    selected = torch.cat([selected[:, :2].sigmoid(), selected[:, 2:4].sigmoid()], dim=1)
    return F.l1_loss(selected, target)


def decode_bbox(heatmap_logits, bbox_raw, image_wh: Tuple[int, int]):
    B, _, H, W = heatmap_logits.shape
    image_w, image_h = float(image_wh[0]), float(image_wh[1])
    max_idx = heatmap_logits.flatten(1).argmax(dim=1)
    gj = max_idx // W
    gi = max_idx % W
    batch_idx = torch.arange(B, device=heatmap_logits.device)
    selected = bbox_raw[batch_idx, :, gj, gi]
    offsets = selected[:, :2].sigmoid()
    sizes = selected[:, 2:4].sigmoid()
    cx = (gi.to(selected.dtype) + offsets[:, 0]) / float(W) * image_w
    cy = (gj.to(selected.dtype) + offsets[:, 1]) / float(H) * image_h
    bw = sizes[:, 0] * image_w
    bh = sizes[:, 1] * image_h
    x1 = (cx - 0.5 * bw).clamp(0.0, image_w)
    y1 = (cy - 0.5 * bh).clamp(0.0, image_h)
    x2 = (cx + 0.5 * bw).clamp(0.0, image_w)
    y2 = (cy + 0.5 * bh).clamp(0.0, image_h)
    return torch.stack([x1, y1, x2, y2], dim=1)


def compute_losses(outputs: Dict[str, torch.Tensor], gt_bbox, image_wh, args):
    labels = torch.arange(outputs["retrieval_logits"].shape[0], device=gt_bbox.device)
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
    rerank_loss = F.cross_entropy(outputs["rerank_logits"], labels)
    total = (
        args.retrieval_loss_weight * retrieval_loss
        + args.localization_loss_weight * localization_loss
        + args.bbox_loss_weight * bbox_loss
        + args.rerank_loss_weight * rerank_loss
    )
    return total, {
        "retrieval": retrieval_loss,
        "localization": localization_loss,
        "bbox": bbox_loss,
        "rerank": rerank_loss,
    }


def train_epoch(loader, model, optimizer, epoch, args):
    model.train()
    meters = {
        "loss": AverageMeter(),
        "retrieval": AverageMeter(),
        "localization": AverageMeter(),
        "bbox": AverageMeter(),
        "rerank": AverageMeter(),
    }
    batch_time = AverageMeter()
    end = time.time()

    for batch_idx, batch in enumerate(loader):
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

        if batch_idx % args.print_freq == 0:
            print(
                f"Epoch: [{epoch}][{batch_idx}/{len(loader)}]\t"
                f"Time: {batch_time.val:.3f}\t"
                f"Loss: {meters['loss'].val:.4f} ({meters['loss'].avg:.4f})\t"
                f"Ret: {meters['retrieval'].val:.4f}\t"
                f"Loc: {meters['localization'].val:.4f}\t"
                f"Bbox: {meters['bbox'].val:.4f}\t"
                f"Rerank: {meters['rerank'].val:.4f}"
            )

    return {name: meter.avg for name, meter in meters.items()}


def validate(loader, model):
    model.eval()
    iou_values: List[float] = []
    center_distances: List[float] = []
    top1_hits = 0
    total = 0

    for batch in tqdm(loader, desc="Validating"):
        query_imgs = batch["target_pixel_values"].to(DEVICE, non_blocking=True)
        aerial_imgs = batch["search_pixel_values"].to(DEVICE, non_blocking=True)
        gt_bbox = batch["bbox"].to(DEVICE, non_blocking=True)
        with torch.no_grad():
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
            center_dist = float(torch.sqrt((pred_cx - gt_cx) ** 2 + (pred_cy - gt_cy) ** 2).item())
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

    print("Creating UnifyGeoLite model...")
    model = UnifyGeoLite(
        proj_dim=args.proj_dim,
        detail_dim=args.detail_dim,
        dims=(args.base_dim, args.base_dim * 2, args.base_dim * 4),
    ).to(DEVICE)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_iou = -1.0

    print(f"Starting training for {args.max_epoch} epochs...")
    for epoch in range(args.max_epoch):
        adjust_learning_rate(args, optimizer, epoch)
        train_metrics = train_epoch(train_loader, model, optimizer, epoch, args)
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
            f"Rerank: {train_metrics['rerank']:.4f}\t"
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
    parser = argparse.ArgumentParser(description="UnifyGeo-style training on ShiftedSatelliteDroneDataset")
    parser.add_argument("--num-workers", type=int, default=8, help="num workers for data loading")
    parser.add_argument("--max-epoch", type=int, default=NUM_EPOCHS, help="number of epochs")
    parser.add_argument("--warmup-epochs", type=int, default=1, help="warmup epochs")
    parser.add_argument("--lr", type=float, default=LEARNING_RATE, help="learning rate")
    parser.add_argument("--min-lr", type=float, default=1e-6, help="minimum cosine LR")
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY, help="AdamW weight decay")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="batch size")
    parser.add_argument(
        "--sat-size",
        type=int,
        nargs=2,
        metavar=("WIDTH", "HEIGHT"),
        default=list(SAT_SIZE),
        help="satellite image size as WIDTH HEIGHT",
    )
    parser.add_argument("--savename", type=str, default="unify_geo", help="TensorBoard run name")
    parser.add_argument("--seed", type=int, default=2024, help="random seed")
    parser.add_argument("--print-freq", type=int, default=PRINT_FREQ, help="print frequency")
    parser.add_argument("--base-dim", type=int, default=96, help="base ConvNeXt channel dimension")
    parser.add_argument("--proj-dim", type=int, default=PROJECTION_DIM, help="retrieval descriptor dimension")
    parser.add_argument("--detail-dim", type=int, default=384, help="detailed matching feature dimension")
    parser.add_argument("--label-smoothing", type=float, default=0.1, help="InfoNCE label smoothing")
    parser.add_argument("--heatmap-sigma", type=float, default=HEATMAP_SIGMA, help="Gaussian target sigma in feature cells")
    parser.add_argument("--retrieval-loss-weight", type=float, default=RETRIEVAL_LOSS_WEIGHT)
    parser.add_argument("--localization-loss-weight", type=float, default=LOCALIZATION_LOSS_WEIGHT)
    parser.add_argument("--bbox-loss-weight", type=float, default=BBOX_LOSS_WEIGHT)
    parser.add_argument("--rerank-loss-weight", type=float, default=RERANK_LOSS_WEIGHT)
    parser.add_argument("--grad-clip", type=float, default=1.0, help="gradient clipping norm, <=0 disables")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="/data/feihong/ckpt/unify_geo",
        help="Path to save model checkpoints",
    )
    args = parser.parse_args()
    args.sat_size = (int(args.sat_size[0]), int(args.sat_size[1]))
    main(args)
