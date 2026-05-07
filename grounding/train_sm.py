#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@file train_sm.py
@desc Train an SMGeo-style anchor-free cross-view grounding model.

This script keeps the grounding data path used by the other scripts in this
folder, while adapting the core ideas from SMGeo:
- multi-input Swin-style query/satellite encoder;
- grid-level sparse Mixture-of-Experts blocks;
- anchor-free detection head predicting a center heatmap and bbox parameters.
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
from PIL import Image
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from dataset import ShiftedSatelliteDroneDataset
from grounding.utils.utils import AverageMeter
from bbox.yolo_utils import bbox_iou


# --- Configuration ---
IMG_SIZE = (768, 432)  # (width, height)
DRONE_SIZE = (256, 256)  # (width, height)
BATCH_SIZE = 16
NUM_EPOCHS = 8
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
PRINT_FREQ = 50
HEATMAP_SIGMA = 1.5
BBOX_LOSS_WEIGHT = 5.0
MOE_ENTROPY_WEIGHT = 0.01
DEVICE = "cuda:1" if torch.cuda.is_available() else "cpu"


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
        return {"input_ids": torch.zeros((1, max_length), dtype=torch.long)}


class PatchEmbed(nn.Module):
    def __init__(self, in_channels=3, embed_dim=64, patch_size=8):
        super().__init__()
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        B, C, H, W = x.shape
        tokens = x.flatten(2).transpose(1, 2)
        tokens = self.norm(tokens)
        return tokens.transpose(1, 2).reshape(B, C, H, W)


def window_partition(x, window_size: int):
    B, H, W, C = x.shape
    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    if pad_h or pad_w:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    Hp, Wp = H + pad_h, W + pad_w
    windows = x.view(
        B,
        Hp // window_size,
        window_size,
        Wp // window_size,
        window_size,
        C,
    )
    windows = windows.permute(0, 1, 3, 2, 4, 5).reshape(-1, window_size * window_size, C)
    return windows, (Hp, Wp), (pad_h, pad_w)


def window_reverse(windows, window_size: int, hw_padded: Tuple[int, int], hw_original: Tuple[int, int], batch_size: int):
    Hp, Wp = hw_padded
    H, W = hw_original
    C = windows.shape[-1]
    x = windows.view(
        batch_size,
        Hp // window_size,
        Wp // window_size,
        window_size,
        window_size,
        C,
    )
    x = x.permute(0, 1, 3, 2, 4, 5).reshape(batch_size, Hp, Wp, C)
    return x[:, :H, :W, :].contiguous()


class SparseMoEFFN(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, num_experts: int = 4, top_k: int = 2):
        super().__init__()
        self.num_experts = int(num_experts)
        self.top_k = max(1, min(int(top_k), self.num_experts))
        self.router = nn.Linear(dim, self.num_experts)
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(dim, hidden_dim),
                    nn.GELU(),
                    nn.Linear(hidden_dim, dim),
                )
                for _ in range(self.num_experts)
            ]
        )

    def forward(self, x):
        router_logits = self.router(x)
        router_probs = F.softmax(router_logits, dim=-1)
        topk_probs, topk_idx = router_probs.topk(self.top_k, dim=-1)
        sparse_probs = torch.zeros_like(router_probs)
        sparse_probs.scatter_(dim=-1, index=topk_idx, src=topk_probs)
        sparse_probs = sparse_probs / sparse_probs.sum(dim=-1, keepdim=True).clamp_min(1e-6)

        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)
        out = torch.einsum("ne,ned->nd", sparse_probs, expert_outputs)
        entropy = -(router_probs * router_probs.clamp_min(1e-6).log()).sum(dim=-1).mean()
        return out, entropy


class SwinMoEBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: int = 8,
        mlp_ratio: int = 4,
        use_moe: bool = False,
        num_experts: int = 4,
        top_k: int = 2,
    ):
        super().__init__()
        self.window_size = int(window_size)
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.use_moe = bool(use_moe)
        hidden_dim = dim * mlp_ratio
        if self.use_moe:
            self.ffn = SparseMoEFFN(dim, hidden_dim, num_experts=num_experts, top_k=top_k)
        else:
            self.ffn = nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, dim),
            )

    def forward(self, x):
        B, C, H, W = x.shape
        x_nhwc = x.permute(0, 2, 3, 1).contiguous()
        windows, hw_padded, _ = window_partition(x_nhwc, self.window_size)
        attn_windows = self.norm1(windows)
        attn_windows, _ = self.attn(attn_windows, attn_windows, attn_windows, need_weights=False)
        x_nhwc = window_reverse(attn_windows, self.window_size, hw_padded, (H, W), B) + x_nhwc

        tokens = x_nhwc.reshape(B * H * W, C)
        tokens_norm = self.norm2(tokens)
        entropy = x.new_tensor(0.0)
        if self.use_moe:
            ffn_out, entropy = self.ffn(tokens_norm)
        else:
            ffn_out = self.ffn(tokens_norm)
        tokens = tokens + ffn_out
        return tokens.view(B, H, W, C).permute(0, 3, 1, 2).contiguous(), entropy


class PatchMerging(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.proj = nn.Conv2d(in_dim, out_dim, kernel_size=2, stride=2)
        self.norm = nn.BatchNorm2d(out_dim)

    def forward(self, x):
        return self.norm(self.proj(x))


class SwinMoEStage(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        window_size: int,
        moe_blocks: Tuple[int, ...],
        num_experts: int,
        top_k: int,
    ):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                SwinMoEBlock(
                    dim=dim,
                    num_heads=num_heads,
                    window_size=window_size,
                    use_moe=idx in set(moe_blocks),
                    num_experts=num_experts,
                    top_k=top_k,
                )
                for idx in range(depth)
            ]
        )

    def forward(self, x):
        entropies = []
        for block in self.blocks:
            x, entropy = block(x)
            entropies.append(entropy)
        if entropies:
            return x, torch.stack(entropies).mean()
        return x, x.new_tensor(0.0)


class SwinMoEBackbone(nn.Module):
    """Multi-input Swin-MoE backbone returning query vector and satellite feature map."""

    def __init__(
        self,
        embed_dim: int = 64,
        patch_size: int = 8,
        window_size: int = 8,
        depths: Tuple[int, ...] = (1, 1, 2),
        num_heads: Tuple[int, ...] = (2, 4, 8),
        num_experts: int = 4,
        top_k: int = 2,
    ):
        super().__init__()
        self.query_patch = PatchEmbed(3, embed_dim, patch_size=patch_size)
        self.sat_patch = PatchEmbed(3, embed_dim, patch_size=patch_size)
        self.stages = nn.ModuleList()
        self.downsamples = nn.ModuleList()

        dim = embed_dim
        for stage_idx, depth in enumerate(depths):
            moe_blocks = (depth - 1,) if stage_idx >= 1 else ()
            self.stages.append(
                SwinMoEStage(
                    dim=dim,
                    depth=depth,
                    num_heads=num_heads[stage_idx],
                    window_size=window_size,
                    moe_blocks=moe_blocks,
                    num_experts=num_experts,
                    top_k=top_k,
                )
            )
            if stage_idx < len(depths) - 1:
                self.downsamples.append(PatchMerging(dim, dim * 2))
                dim *= 2
            else:
                self.downsamples.append(nn.Identity())
        self.out_dim = dim

    def _forward_one(self, x):
        entropies = []
        for idx, stage in enumerate(self.stages):
            x, entropy = stage(x)
            entropies.append(entropy)
            if idx < len(self.stages) - 1:
                x = self.downsamples[idx](x)
        return x, torch.stack(entropies).mean()

    def forward(self, query_imgs, sat_imgs):
        query_feat = self.query_patch(query_imgs)
        sat_feat = self.sat_patch(sat_imgs)
        query_feat, query_entropy = self._forward_one(query_feat)
        sat_feat, sat_entropy = self._forward_one(sat_feat)
        query_vec = query_feat.flatten(2).mean(dim=-1)
        return query_vec, sat_feat, 0.5 * (query_entropy + sat_entropy)


class CrossViewConditioning(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.query_to_scale_shift = nn.Sequential(
            nn.LayerNorm(channels),
            nn.Linear(channels, channels * 2),
        )
        nn.init.zeros_(self.query_to_scale_shift[-1].weight)
        nn.init.zeros_(self.query_to_scale_shift[-1].bias)

    def forward(self, query_vec, sat_feat):
        gamma, beta = self.query_to_scale_shift(query_vec).chunk(2, dim=-1)
        gamma = gamma.view(-1, sat_feat.shape[1], 1, 1)
        beta = beta.view(-1, sat_feat.shape[1], 1, 1)
        return sat_feat * (1.0 + gamma) + beta


class AnchorFreeHead(nn.Module):
    """SMGeo-style head: center heatmap logits plus bbox offset/size parameters."""

    def __init__(self, in_channels: int, feat_channels: int = 256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, feat_channels, 3, padding=1),
            nn.BatchNorm2d(feat_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_channels, feat_channels, 3, padding=1),
            nn.BatchNorm2d(feat_channels),
            nn.ReLU(inplace=True),
        )
        self.head_heatmap = nn.Conv2d(feat_channels, 1, 1)
        self.head_bbox = nn.Conv2d(feat_channels, 4, 1)
        nn.init.constant_(self.head_heatmap.bias, -2.0)

    def forward(self, x):
        feat = self.conv(x)
        return self.head_heatmap(feat), self.head_bbox(feat)


class SMGeoLite(nn.Module):
    def __init__(
        self,
        embed_dim: int = 64,
        patch_size: int = 8,
        window_size: int = 8,
        num_experts: int = 4,
        top_k: int = 2,
    ):
        super().__init__()
        self.backbone = SwinMoEBackbone(
            embed_dim=embed_dim,
            patch_size=patch_size,
            window_size=window_size,
            num_experts=num_experts,
            top_k=top_k,
        )
        self.condition = CrossViewConditioning(self.backbone.out_dim)
        self.head = AnchorFreeHead(self.backbone.out_dim)

    def forward(self, query_imgs, sat_imgs):
        query_vec, sat_feat, moe_entropy = self.backbone(query_imgs, sat_imgs)
        sat_feat = self.condition(query_vec, sat_feat)
        heatmap_logits, bbox_raw = self.head(sat_feat)
        return heatmap_logits, bbox_raw, moe_entropy

    def bbox_forward(self, query_imgs, sat_imgs):
        heatmap_logits, bbox_raw, _ = self.forward(query_imgs, sat_imgs)
        return decode_anchor_free(heatmap_logits, bbox_raw, (sat_imgs.shape[-1], sat_imgs.shape[-2]))


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


def anchor_free_loss(
    heatmap_logits,
    bbox_raw,
    gt_bbox,
    image_wh: Tuple[int, int],
    heatmap_sigma: float = HEATMAP_SIGMA,
):
    B, _, feat_h, feat_w = heatmap_logits.shape
    heatmap_target = build_heatmap_target(
        gt_bbox,
        feature_hw=(feat_h, feat_w),
        image_wh=image_wh,
        sigma=heatmap_sigma,
    )
    bbox_target, gi, gj = build_bbox_target(gt_bbox, (feat_h, feat_w), image_wh)
    batch_idx = torch.arange(B, device=gt_bbox.device)

    # Keep the heatmap positive cell from being averaged away by the full map.
    # The gaussian target still shapes nearby cells, while the exact center
    # receives an explicit positive-objectness loss.
    center_target = torch.zeros_like(heatmap_logits)
    center_target[batch_idx, 0, gj, gi] = 1.0
    objectness_bce = F.binary_cross_entropy_with_logits(
        heatmap_logits,
        center_target,
        reduction="none",
    )
    pos_mask = center_target.bool()
    neg_mask = ~pos_mask
    pos_heatmap_loss = objectness_bce[pos_mask].mean()
    neg_heatmap_loss = objectness_bce[neg_mask].mean()

    pred_prob = torch.sigmoid(heatmap_logits)
    gaussian_bce = F.binary_cross_entropy_with_logits(
        heatmap_logits,
        heatmap_target,
        reduction="none",
    )
    gaussian_weight = torch.where(
        heatmap_target > 0.5,
        (1.0 - pred_prob).pow(2.0),
        pred_prob.pow(2.0) * (1.0 - heatmap_target).pow(4.0),
    )
    gaussian_heatmap_loss = (gaussian_bce * gaussian_weight).sum() / max(float(B), 1.0)
    heatmap_loss = pos_heatmap_loss + 0.25 * neg_heatmap_loss + 0.1 * gaussian_heatmap_loss

    selected_bbox = bbox_raw[batch_idx, :, gj, gi]
    selected_bbox = torch.cat(
        [
            selected_bbox[:, :2].sigmoid(),
            selected_bbox[:, 2:4].sigmoid(),
        ],
        dim=1,
    )
    bbox_loss = F.l1_loss(selected_bbox, bbox_target)
    return heatmap_loss, bbox_loss


def decode_anchor_free(heatmap_logits, bbox_raw, image_wh: Tuple[int, int]):
    B, _, feat_h, feat_w = heatmap_logits.shape
    image_w, image_h = float(image_wh[0]), float(image_wh[1])
    heatmap_flat = heatmap_logits.flatten(1)
    max_idx = heatmap_flat.argmax(dim=1)
    gj = max_idx // feat_w
    gi = max_idx % feat_w
    batch_idx = torch.arange(B, device=heatmap_logits.device)
    selected_bbox = bbox_raw[batch_idx, :, gj, gi]

    offsets = selected_bbox[:, :2].sigmoid()
    sizes = selected_bbox[:, 2:4].sigmoid()
    cx = (gi.to(selected_bbox.dtype) + offsets[:, 0]) / float(feat_w) * image_w
    cy = (gj.to(selected_bbox.dtype) + offsets[:, 1]) / float(feat_h) * image_h
    bw = sizes[:, 0] * image_w
    bh = sizes[:, 1] * image_h

    x1 = (cx - 0.5 * bw).clamp(0.0, image_w)
    y1 = (cy - 0.5 * bh).clamp(0.0, image_h)
    x2 = (cx + 0.5 * bw).clamp(0.0, image_w)
    y2 = (cy + 0.5 * bh).clamp(0.0, image_h)
    return torch.stack([x1, y1, x2, y2], dim=1)


def adjust_learning_rate(args, optimizer, epoch):
    lr = args.lr * (0.1 ** (epoch // 10))
    print(("lr", lr))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def train_epoch(loader, model, optimizer, epoch, args):
    model.train()
    batch_time = AverageMeter()
    losses = AverageMeter()
    heat_losses = AverageMeter()
    bbox_losses = AverageMeter()
    moe_losses = AverageMeter()

    end = time.time()
    for batch_idx, batch in enumerate(loader):
        query_imgs = batch["target_pixel_values"].to(DEVICE, non_blocking=True)
        sat_imgs = batch["search_pixel_values"].to(DEVICE, non_blocking=True)
        gt_bbox = batch["bbox"].to(DEVICE, non_blocking=True)

        heatmap_logits, bbox_raw, moe_entropy = model(query_imgs, sat_imgs)
        image_wh = (sat_imgs.shape[-1], sat_imgs.shape[-2])
        heatmap_loss, bbox_loss = anchor_free_loss(
            heatmap_logits,
            bbox_raw,
            gt_bbox,
            image_wh,
            heatmap_sigma=args.heatmap_sigma,
        )
        moe_loss = -moe_entropy
        loss = heatmap_loss + args.bbox_loss_weight * bbox_loss + args.moe_entropy_weight * moe_loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        losses.update(loss.item(), query_imgs.shape[0])
        heat_losses.update(heatmap_loss.item(), query_imgs.shape[0])
        bbox_losses.update(bbox_loss.item(), query_imgs.shape[0])
        moe_losses.update(float(moe_entropy.detach().item()), query_imgs.shape[0])
        batch_time.update(time.time() - end)
        end = time.time()

        if batch_idx % args.print_freq == 0:
            print(
                f"Epoch: [{epoch}][{batch_idx}/{len(loader)}]\t"
                f"Time: {batch_time.val:.3f}\t"
                f"Loss: {losses.val:.4f} ({losses.avg:.4f})\t"
                f"Heat: {heat_losses.val:.4f} ({heat_losses.avg:.4f})\t"
                f"Bbox: {bbox_losses.val:.4f} ({bbox_losses.avg:.4f})\t"
                f"MoE-H: {moe_losses.val:.4f} ({moe_losses.avg:.4f})"
            )

    return losses.avg, heat_losses.avg, bbox_losses.avg, moe_losses.avg


def validate(loader, model):
    model.eval()
    iou_values: List[float] = []
    center_distances: List[float] = []

    for batch in tqdm(loader, desc="Validating"):
        query_imgs = batch["target_pixel_values"].to(DEVICE, non_blocking=True)
        sat_imgs = batch["search_pixel_values"].to(DEVICE, non_blocking=True)
        gt_bbox = batch["bbox"].to(DEVICE, non_blocking=True)

        with torch.no_grad():
            heatmap_logits, bbox_raw, _ = model(query_imgs, sat_imgs)
            pred_bbox_xyxy = decode_anchor_free(
                heatmap_logits,
                bbox_raw,
                image_wh=(sat_imgs.shape[-1], sat_imgs.shape[-2]),
            )

        for idx in range(pred_bbox_xyxy.shape[0]):
            pred_xyxy = pred_bbox_xyxy[idx].float()
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
        return 0.0, 0.0, 0.0

    iou_arr = np.array(iou_values, dtype=np.float32)
    center_arr = np.array(center_distances, dtype=np.float32)
    return (
        float(iou_arr.mean()),
        float((iou_arr > 0.5).mean()),
        float(center_arr.mean()),
    )


def build_loaders(args):
    processor = TransformProcessorWrapper(DRONE_SIZE)
    processor_sat = TransformProcessorWrapper(args.img_size)
    tokenizer = DummyTokenizer()

    train_dataset = ShiftedSatelliteDroneDataset(
        processor=processor,
        processor_sat=processor_sat,
        tokenizer=tokenizer,
        split="train",
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        drop_last=False,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=4 if args.num_workers > 0 else None,
    )
    return train_loader, len(train_dataset)


def main(args):
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    os.makedirs(args.checkpoint, exist_ok=True)
    writer = SummaryWriter(f"runs/{args.savename}")

    print("Creating datasets from shared data source...")
    train_loader, train_count = build_loaders(args)
    print(f"Found {train_count} training samples")

    print("Creating SMGeoLite model...")
    model = SMGeoLite(
        embed_dim=args.embed_dim,
        patch_size=args.patch_size,
        window_size=args.window_size,
        num_experts=args.num_experts,
        top_k=args.top_k,
    ).to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)

    print(f"Starting training for {args.max_epoch} epochs...")
    for epoch in range(args.max_epoch):
        adjust_learning_rate(args, optimizer, epoch)
        train_loss, train_heat, train_bbox, moe_entropy = train_epoch(
            train_loader,
            model,
            optimizer,
            epoch,
            args,
        )

        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/train_heatmap", train_heat, epoch)
        writer.add_scalar("Loss/train_bbox", train_bbox, epoch)
        writer.add_scalar("MoE/entropy", moe_entropy, epoch)

        print(
            f"Epoch {epoch + 1}/{args.max_epoch}:\t"
            f"Train Loss: {train_loss:.4f} (Heat: {train_heat:.4f}, Bbox: {train_bbox:.4f})\t"
            f"MoE-H: {moe_entropy:.4f}"
        )

        torch.save(model.state_dict(), os.path.join(args.checkpoint, "last.pth"))

    print("\nTraining complete. Saved checkpoint to last.pth")
    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SMGeo-style training on ShiftedSatelliteDroneDataset")
    parser.add_argument("--num-workers", type=int, default=8, help="num workers for data loading")
    parser.add_argument("--max-epoch", type=int, default=NUM_EPOCHS, help="number of epochs")
    parser.add_argument("--lr", type=float, default=LEARNING_RATE, help="learning rate")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="batch size")
    parser.add_argument(
        "--img-size",
        type=int,
        nargs=2,
        metavar=("WIDTH", "HEIGHT"),
        default=list(IMG_SIZE),
        help="satellite image size as WIDTH HEIGHT",
    )
    parser.add_argument("--savename", type=str, default="ground_sm", help="TensorBoard run name")
    parser.add_argument("--seed", type=int, default=2024, help="random seed")
    parser.add_argument("--print-freq", type=int, default=PRINT_FREQ, help="print frequency")
    parser.add_argument("--embed-dim", type=int, default=64, help="Swin patch embedding dimension")
    parser.add_argument("--patch-size", type=int, default=8, help="patch size for query and satellite inputs")
    parser.add_argument("--window-size", type=int, default=8, help="local attention window size")
    parser.add_argument("--num-experts", type=int, default=4, help="number of sparse MoE experts")
    parser.add_argument("--top-k", type=int, default=2, help="number of active experts per token")
    parser.add_argument("--heatmap-sigma", type=float, default=HEATMAP_SIGMA, help="Gaussian target sigma in feature cells")
    parser.add_argument("--bbox-loss-weight", type=float, default=BBOX_LOSS_WEIGHT, help="bbox loss weight")
    parser.add_argument("--moe-entropy-weight", type=float, default=MOE_ENTROPY_WEIGHT, help="weight for encouraging high expert entropy")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="/data/feihong/ckpt/ground_sm",
        help="Path to save model checkpoints",
    )
    args = parser.parse_args()
    args.img_size = (int(args.img_size[0]), int(args.img_size[1]))

    main(args)
