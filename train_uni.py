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
import timm
from PIL import Image
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset
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
NUM_EPOCHS = 40
LEARNING_RATE = 4.5e-4
BACKBONE_LEARNING_RATE = 1e-5
WEIGHT_DECAY = 0.04
PRINT_FREQ = 50
PROJECTION_DIM = 768
HEATMAP_SIGMA = 1.5
RETRIEVAL_LOSS_WEIGHT = 1.0
LOCALIZATION_LOSS_WEIGHT = 1.0
BBOX_LOSS_WEIGHT = 5.0
RERANK_LOSS_WEIGHT = 1.0
RETRIEVAL_ONLY_EPOCHS = 0
GROUNDING_RAMP_EPOCHS = 1
RERANK_START_EPOCH = 1
MEMORY_QUEUE_SIZE = 0
BACKBONE_NAME = "swin_small_patch4_window7_224"
PRETRAINED_CHECKPOINT = "/media/data1/feihong/ckpt/pretrained/smgeo/swin_s_imagenet1k_v1.pth"
VAL_FRACTION = 0.25
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


class TransformProcessorWrapper:
    def __init__(self, image_size: Tuple[int, int]):
        self.size = {"height": int(image_size[1]), "width": int(image_size[0])}
        self.mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)

    def __call__(self, images, return_tensors="pt"):
        target_h = int(self.size["height"])
        target_w = int(self.size["width"])
        if images.size != (target_w, target_h):
            images = images.resize((target_w, target_h), Image.Resampling.BILINEAR)
        image_np = np.array(images, dtype=np.float32) / 255.0
        pixel_values = torch.from_numpy(image_np).permute(2, 0, 1)
        pixel_values = (pixel_values - self.mean) / self.std
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
        self.map_refine = nn.Sequential(
            nn.Conv2d(detail_dim, detail_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(detail_dim),
            nn.GELU(),
            nn.Conv2d(detail_dim, detail_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(detail_dim),
            nn.GELU(),
        )
        self.column_fc = nn.Sequential(
            nn.Conv1d(detail_dim, detail_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(detail_dim, detail_dim, kernel_size=1),
        )

    def forward(self, feat):
        feat = self.map_refine(self.reduce(feat))
        # Aggregate vertically and then summarize azimuth/column information.
        columns = feat.mean(dim=2)
        columns = self.column_fc(columns)
        detail_vector = F.normalize(columns.mean(dim=-1), p=2, dim=1)
        detail_map = F.normalize(feat, p=2, dim=1)
        return detail_vector, detail_map


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
        self.heat_kernel_size = 3
        self.bbox_head = nn.Sequential(
            nn.Conv2d(detail_dim * 2 + 1, detail_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(detail_dim),
            nn.GELU(),
            nn.Conv2d(detail_dim, detail_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(detail_dim),
            nn.GELU(),
            nn.Conv2d(detail_dim, 4, kernel_size=1),
        )

    def _dynamic_heatmap(self, ground_detail_map, aerial_detail):
        if ground_detail_map.ndim != 4:
            raise ValueError(f"Expected ground_detail_map shape (B, C, H, W), got {tuple(ground_detail_map.shape)}.")
        if aerial_detail.ndim != 4:
            raise ValueError(f"Expected aerial_detail shape (B, C, H, W), got {tuple(aerial_detail.shape)}.")
        if ground_detail_map.shape[0] != aerial_detail.shape[0] or ground_detail_map.shape[1] != aerial_detail.shape[1]:
            raise ValueError(
                f"Ground/aerial detail mismatch: ground={tuple(ground_detail_map.shape)}, "
                f"aerial={tuple(aerial_detail.shape)}."
            )

        batch_size, channels, height, width = aerial_detail.shape
        kernel = F.adaptive_avg_pool2d(
            ground_detail_map,
            (self.heat_kernel_size, self.heat_kernel_size),
        )
        kernel = kernel - kernel.mean(dim=(2, 3), keepdim=True)
        kernel = F.normalize(kernel.flatten(1), p=2, dim=1).view_as(kernel)
        aerial_detail = F.normalize(aerial_detail.contiguous(), p=2, dim=1)
        conv_input = aerial_detail.reshape(1, batch_size * channels, height, width)
        conv_kernel = kernel.reshape(batch_size, channels, self.heat_kernel_size, self.heat_kernel_size)
        heatmap = F.conv2d(
            conv_input,
            conv_kernel,
            padding=self.heat_kernel_size // 2,
            groups=batch_size,
        )
        return heatmap.view(batch_size, 1, height, width)

    def forward(self, ground_detail, ground_detail_map, aerial_detail):
        temp = self.temperature.clamp(min=0.01, max=1.0)
        heatmap_logits = self._dynamic_heatmap(ground_detail_map, aerial_detail) / temp
        heat_gate = F.softmax(heatmap_logits.flatten(1), dim=1).view_as(heatmap_logits)
        heat_gate = heat_gate * float(heat_gate.shape[-2] * heat_gate.shape[-1])
        query_map = ground_detail[:, :, None, None].expand_as(aerial_detail)
        bbox_raw = self.bbox_head(torch.cat([aerial_detail, query_map, heat_gate], dim=1))
        return heatmap_logits, bbox_raw

    def batch_rerank_logits(self, ground_detail, aerial_detail, retrieval_logits):
        temp = self.temperature.clamp(min=0.01, max=1.0)
        detail_scores = torch.einsum("bd,kdhw->bkhw", ground_detail, aerial_detail) / temp
        detail_scores = detail_scores.flatten(2).max(dim=-1)[0]
        return retrieval_logits + detail_scores


def _map_torchvision_swin_key(key: str) -> Optional[str]:
    parts = key.split(".")
    if parts[:2] == ["features", "0"]:
        if parts[2] == "0":
            return "patch_embed.proj." + ".".join(parts[3:])
        if parts[2] == "2":
            return "patch_embed.norm." + ".".join(parts[3:])
    if parts[0] == "features" and parts[1] in {"1", "3", "5", "7"}:
        layer_map = {"1": "layers_0", "3": "layers_1", "5": "layers_2", "7": "layers_3"}
        rest = parts[3:]
        if rest[:2] == ["mlp", "0"]:
            rest = ["mlp", "fc1"] + rest[2:]
        elif rest[:2] == ["mlp", "3"]:
            rest = ["mlp", "fc2"] + rest[2:]
        return f"{layer_map[parts[1]]}.blocks.{parts[2]}." + ".".join(rest)
    if parts[0] == "features" and parts[1] in {"2", "4", "6"}:
        layer_map = {"2": "layers_1", "4": "layers_2", "6": "layers_3"}
        if parts[2] == "reduction":
            return f"{layer_map[parts[1]]}.downsample.reduction." + ".".join(parts[3:])
        if parts[2] == "norm":
            return f"{layer_map[parts[1]]}.downsample.norm." + ".".join(parts[3:])
    return None


def load_feature_backbone_checkpoint(backbone: nn.Module, checkpoint_path: str, backbone_name: str) -> Dict[str, int]:
    if not checkpoint_path:
        return {"loaded": 0, "skipped": 0, "shape_mismatch": 0, "missing_after_load": 0}
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Backbone checkpoint not found: {checkpoint_path}")

    state = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    elif isinstance(state, dict) and "model" in state:
        state = state["model"]
    if not isinstance(state, dict):
        raise TypeError(f"Expected checkpoint dict, got {type(state)}.")

    model_state = backbone.state_dict()
    candidates: Dict[str, torch.Tensor] = {}
    skipped = 0
    shape_mismatch = 0
    for key, value in state.items():
        clean_key = str(key).removeprefix("module.").removeprefix("backbone.")
        mapped_key = clean_key
        if mapped_key not in model_state and "swin" in backbone_name:
            mapped_key = _map_torchvision_swin_key(clean_key) or clean_key
        if mapped_key not in model_state:
            skipped += 1
            continue
        if tuple(model_state[mapped_key].shape) != tuple(value.shape):
            shape_mismatch += 1
            continue
        candidates[mapped_key] = value

    result = backbone.load_state_dict(candidates, strict=False)
    return {
        "loaded": len(candidates),
        "skipped": skipped,
        "shape_mismatch": shape_mismatch,
        "missing_after_load": len(result.missing_keys),
    }


class TimmHierarchicalBranch(nn.Module):
    """ConvNeXt feature pyramid branch for coarse-to-fine cross-view matching."""

    def __init__(
        self,
        backbone_name: str = "convnext_tiny",
        pretrained: bool = False,
        pretrained_checkpoint: Optional[str] = None,
        detail_dim: int = 384,
        image_size: Tuple[int, int] = DRONE_SIZE,
    ):
        super().__init__()
        image_h = int(image_size[1])
        image_w = int(image_size[0])
        create_kwargs = {
            "pretrained": pretrained and not pretrained_checkpoint,
            "features_only": True,
            "out_indices": (1, 2, 3),
        }
        if "swin" in backbone_name:
            create_kwargs["img_size"] = (image_h, image_w)
        self.backbone = timm.create_model(backbone_name, **create_kwargs)
        if pretrained_checkpoint:
            load_info = load_feature_backbone_checkpoint(self.backbone, pretrained_checkpoint, backbone_name)
            print(f"Loaded {backbone_name} backbone checkpoint: {load_info}")
        channels = self.backbone.feature_info.channels()
        if len(channels) != 3:
            raise ValueError(f"Expected three feature levels from {backbone_name}, got {channels}.")

        fine_channels, mid_channels, deep_channels = [int(v) for v in channels]
        self.feature_channels = (fine_channels, mid_channels, deep_channels)
        self.out_dim = int(detail_dim)
        self.fine_proj = nn.Sequential(
            nn.Conv2d(fine_channels, detail_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(detail_dim),
            nn.GELU(),
        )
        self.mid_proj = nn.Sequential(
            nn.Conv2d(mid_channels, detail_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(detail_dim),
            nn.GELU(),
        )
        self.deep_proj = nn.Sequential(
            nn.Conv2d(deep_channels, detail_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(detail_dim),
            nn.GELU(),
        )
        self.fpn = nn.Sequential(
            nn.Conv2d(detail_dim * 3, detail_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(detail_dim),
            nn.GELU(),
            nn.Conv2d(detail_dim, detail_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(detail_dim),
            nn.GELU(),
        )
        self.semantic_refine = nn.Sequential(
            nn.Conv2d(detail_dim, detail_dim, kernel_size=3, padding=1, groups=detail_dim, bias=False),
            nn.Conv2d(detail_dim, detail_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(detail_dim),
            nn.GELU(),
        )

    def _to_nchw(self, feat: torch.Tensor, expected_channels: int) -> torch.Tensor:
        if feat.ndim != 4:
            raise ValueError(f"Expected 4D feature map, got {tuple(feat.shape)}.")
        if feat.shape[1] == expected_channels:
            return feat.contiguous()
        if feat.shape[-1] == expected_channels:
            return feat.permute(0, 3, 1, 2).contiguous()
        raise ValueError(
            f"Cannot infer feature layout for shape {tuple(feat.shape)} "
            f"with expected channels={expected_channels}."
        )

    def forward(self, x):
        fine_raw, mid_raw, deep_raw = self.backbone(x)
        fine_raw = self._to_nchw(fine_raw, self.feature_channels[0])
        mid_raw = self._to_nchw(mid_raw, self.feature_channels[1])
        deep_raw = self._to_nchw(deep_raw, self.feature_channels[2])
        fine = self.fine_proj(fine_raw)
        mid = F.interpolate(
            self.mid_proj(mid_raw),
            size=fine.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        deep = self.deep_proj(deep_raw)
        deep_up = F.interpolate(
            deep,
            size=fine.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        fused = self.fpn(torch.cat([fine, mid, deep_up], dim=1))
        semantic = self.semantic_refine(deep)
        return {
            "multi_scale": (fine, mid, semantic),
            "fine": fused,
            "semantic": semantic,
        }


class UnifyGeoLite(nn.Module):
    def __init__(
        self,
        proj_dim: int = PROJECTION_DIM,
        detail_dim: int = 384,
        dims: Tuple[int, int, int] = (96, 192, 384),
        backbone_name: str = BACKBONE_NAME,
        pretrained_backbone: bool = False,
        pretrained_checkpoint: Optional[str] = PRETRAINED_CHECKPOINT,
    ):
        super().__init__()
        del dims
        self.ground_encoder = TimmHierarchicalBranch(
            backbone_name=backbone_name,
            pretrained=pretrained_backbone,
            pretrained_checkpoint=pretrained_checkpoint,
            detail_dim=detail_dim,
            image_size=DRONE_SIZE,
        )
        self.aerial_encoder = TimmHierarchicalBranch(
            backbone_name=backbone_name,
            pretrained=pretrained_backbone,
            pretrained_checkpoint=pretrained_checkpoint,
            detail_dim=detail_dim,
            image_size=SAT_SIZE,
        )
        self.ground_aggregator = SelfAttentionAggregator(detail_dim, proj_dim)
        self.aerial_aggregator = SelfAttentionAggregator(detail_dim, proj_dim)
        self.ground_detail = GroundDetailProjector(detail_dim, detail_dim)
        self.aerial_detail = AerialDetailProjector(detail_dim, detail_dim)
        self.decoder = LocalizationDecoder(detail_dim)
        self.logit_scale = nn.Parameter(torch.tensor(np.log(1 / 0.07), dtype=torch.float32))

    def forward(self, query_imgs, aerial_imgs):
        ground = self.ground_encoder(query_imgs)
        aerial = self.aerial_encoder(aerial_imgs)

        query_global = self.ground_aggregator(ground["semantic"])
        aerial_global = self.aerial_aggregator(aerial["semantic"])
        ground_detail, ground_detail_map = self.ground_detail(ground["fine"])
        aerial_detail = self.aerial_detail(aerial["fine"])

        heatmap_logits, bbox_raw = self.decoder(ground_detail, ground_detail_map, aerial_detail)
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
            "logit_scale": scale,
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


class RetrievalMemoryQueue:
    """Detached cross-batch negatives for the global retrieval objective."""

    def __init__(self, capacity: int):
        self.capacity = max(0, int(capacity))
        self.query_feats: Optional[torch.Tensor] = None
        self.aerial_feats: Optional[torch.Tensor] = None

    def __len__(self) -> int:
        if self.query_feats is None:
            return 0
        return int(self.query_feats.shape[0])

    def enqueue(self, query_feats: torch.Tensor, aerial_feats: torch.Tensor) -> None:
        if self.capacity <= 0:
            return
        query_feats = F.normalize(query_feats.detach(), p=2, dim=1).cpu()
        aerial_feats = F.normalize(aerial_feats.detach(), p=2, dim=1).cpu()
        if self.query_feats is None:
            self.query_feats = query_feats[-self.capacity :]
            self.aerial_feats = aerial_feats[-self.capacity :]
            return
        self.query_feats = torch.cat([self.query_feats, query_feats], dim=0)[-self.capacity :]
        self.aerial_feats = torch.cat([self.aerial_feats, aerial_feats], dim=0)[-self.capacity :]

    def get(self, device: torch.device) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if self.query_feats is None or self.aerial_feats is None:
            return None, None
        return (
            self.query_feats.to(device=device, non_blocking=True),
            self.aerial_feats.to(device=device, non_blocking=True),
        )


def retrieval_loss_with_memory(
    query_feats: torch.Tensor,
    aerial_feats: torch.Tensor,
    logit_scale: torch.Tensor,
    memory_queue: Optional[RetrievalMemoryQueue],
    label_smoothing: float,
) -> torch.Tensor:
    query_feats = F.normalize(query_feats, p=2, dim=1)
    aerial_feats = F.normalize(aerial_feats, p=2, dim=1)
    logits = logit_scale * query_feats @ aerial_feats.t()
    labels = torch.arange(query_feats.shape[0], device=query_feats.device)
    if memory_queue is None or len(memory_queue) == 0:
        return 0.5 * (
            F.cross_entropy(logits, labels, label_smoothing=label_smoothing)
            + F.cross_entropy(logits.t(), labels, label_smoothing=label_smoothing)
        )

    memory_query, memory_aerial = memory_queue.get(query_feats.device)
    if memory_query is None or memory_aerial is None:
        return symmetric_info_nce(logits, label_smoothing=label_smoothing)

    query_logits = torch.cat([logits, logit_scale * query_feats @ memory_aerial.t()], dim=1)
    aerial_logits = torch.cat([logits.t(), logit_scale * aerial_feats @ memory_query.t()], dim=1)
    return 0.5 * (
        F.cross_entropy(query_logits, labels, label_smoothing=label_smoothing)
        + F.cross_entropy(aerial_logits, labels, label_smoothing=label_smoothing)
    )


def scheduled_loss_weights(epoch: int, args) -> Dict[str, float]:
    retrieval_weight = float(args.retrieval_loss_weight)
    if epoch < int(args.retrieval_only_epochs):
        grounding_scale = 0.0
    else:
        ramp_epoch = epoch - int(args.retrieval_only_epochs) + 1
        grounding_scale = min(1.0, ramp_epoch / max(float(args.grounding_ramp_epochs), 1.0))

    rerank_scale = 0.0
    if epoch >= int(args.rerank_start_epoch):
        rerank_epoch = epoch - int(args.rerank_start_epoch) + 1
        rerank_scale = min(1.0, rerank_epoch / max(float(args.grounding_ramp_epochs), 1.0))

    return {
        "retrieval": retrieval_weight,
        "localization": float(args.localization_loss_weight) * grounding_scale,
        "bbox": float(args.bbox_loss_weight) * grounding_scale,
        "rerank": float(args.rerank_loss_weight) * rerank_scale,
    }


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


def compute_losses(
    outputs: Dict[str, torch.Tensor],
    gt_bbox,
    image_wh,
    args,
    epoch: int,
    memory_queue: Optional[RetrievalMemoryQueue] = None,
):
    labels = torch.arange(outputs["retrieval_logits"].shape[0], device=gt_bbox.device)
    weights = scheduled_loss_weights(epoch, args)
    retrieval_loss = retrieval_loss_with_memory(
        outputs["query_global"],
        outputs["aerial_global"],
        outputs["logit_scale"],
        memory_queue,
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
        weights["retrieval"] * retrieval_loss
        + weights["localization"] * localization_loss
        + weights["bbox"] * bbox_loss
        + weights["rerank"] * rerank_loss
    )
    return total, {
        "retrieval": retrieval_loss,
        "localization": localization_loss,
        "bbox": bbox_loss,
        "rerank": rerank_loss,
    }, weights


def train_epoch(loader, model, optimizer, epoch, args, memory_queue: Optional[RetrievalMemoryQueue] = None):
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
        loss, loss_items, loss_weights = compute_losses(
            outputs,
            gt_bbox,
            image_wh,
            args,
            epoch=epoch,
            memory_queue=memory_queue,
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if args.grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        if memory_queue is not None:
            memory_queue.enqueue(outputs["query_global"], outputs["aerial_global"])

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
                f"Rerank: {meters['rerank'].val:.4f}\t"
                f"W(loc/bbox/rerank): {loss_weights['localization']:.3f}/"
                f"{loss_weights['bbox']:.3f}/{loss_weights['rerank']:.3f}"
            )

    metrics = {name: meter.avg for name, meter in meters.items()}
    metrics.update({f"weight_{name}": float(value) for name, value in scheduled_loss_weights(epoch, args).items()})
    metrics["memory_queue_size"] = float(len(memory_queue) if memory_queue is not None else 0)
    return metrics


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
    raw_val_count = len(val_dataset)
    val_fraction = float(args.val_fraction)
    if val_fraction <= 0.0 or val_fraction > 1.0:
        raise ValueError(f"Expected --val-fraction in (0, 1], got {val_fraction}.")
    if val_fraction < 1.0:
        step = max(int(round(1.0 / val_fraction)), 1)
        val_indices = list(range(0, raw_val_count, step))
        val_dataset = Subset(val_dataset, val_indices)

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
    for param_group in optimizer.param_groups:
        base_lr = float(param_group.get("base_lr", args.lr))
        min_lr = float(args.min_lr) * base_lr / max(float(args.lr), 1e-12)
        if epoch < args.warmup_epochs:
            lr = base_lr * float(epoch + 1) / max(float(args.warmup_epochs), 1.0)
        else:
            progress = (epoch - args.warmup_epochs) / max(float(args.max_epoch - args.warmup_epochs), 1.0)
            lr = min_lr + 0.5 * (base_lr - min_lr) * (1.0 + np.cos(np.pi * progress))
        param_group["lr"] = lr
    print(("lr", [param_group["lr"] for param_group in optimizer.param_groups]))


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
        backbone_name=args.backbone_name,
        pretrained_backbone=args.timm_pretrained,
        pretrained_checkpoint=args.pretrained_checkpoint,
    ).to(DEVICE)

    backbone_params = []
    head_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("ground_encoder.backbone.") or name.startswith("aerial_encoder.backbone."):
            backbone_params.append(param)
        else:
            head_params.append(param)
    optimizer = AdamW(
        [
            {"params": backbone_params, "lr": args.backbone_lr, "base_lr": args.backbone_lr},
            {"params": head_params, "lr": args.lr, "base_lr": args.lr},
        ],
        weight_decay=args.weight_decay,
    )
    print(
        f"Optimizer param groups: backbone lr={args.backbone_lr:.2e}, "
        f"heads lr={args.lr:.2e}"
    )
    memory_queue = (
        RetrievalMemoryQueue(args.memory_queue_size)
        if int(args.memory_queue_size) > 0
        else None
    )
    best_iou = -1.0

    print(f"Starting training for {args.max_epoch} epochs...")
    for epoch in range(args.max_epoch):
        adjust_learning_rate(args, optimizer, epoch)
        train_metrics = train_epoch(train_loader, model, optimizer, epoch, args, memory_queue=memory_queue)
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
            f"W(loc/bbox/rerank): {train_metrics['weight_localization']:.3f}/"
            f"{train_metrics['weight_bbox']:.3f}/{train_metrics['weight_rerank']:.3f}\t"
            f"Queue: {int(train_metrics['memory_queue_size'])}\t"
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
    parser.add_argument("--backbone-lr", type=float, default=BACKBONE_LEARNING_RATE, help="backbone learning rate")
    parser.add_argument("--min-lr", type=float, default=1e-6, help="minimum cosine LR")
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY, help="AdamW weight decay")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="batch size")
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=VAL_FRACTION,
        help="fraction of the validation split evaluated each epoch",
    )
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
    parser.add_argument("--base-dim", type=int, default=96, help="deprecated; kept for test_unify CLI compatibility")
    parser.add_argument("--backbone-name", type=str, default=BACKBONE_NAME, help="timm feature backbone for UnifyGeo")
    parser.add_argument(
        "--pretrained-checkpoint",
        type=str,
        default=PRETRAINED_CHECKPOINT,
        help="local backbone checkpoint; defaults to the cached Swin-S ImageNet weights",
    )
    parser.add_argument(
        "--no-pretrained-checkpoint",
        dest="pretrained_checkpoint",
        action="store_const",
        const=None,
        help="disable local backbone checkpoint loading",
    )
    parser.add_argument(
        "--timm-pretrained",
        action="store_true",
        help="let timm load/download pretrained weights when no local checkpoint is used",
    )
    parser.add_argument("--proj-dim", type=int, default=PROJECTION_DIM, help="retrieval descriptor dimension")
    parser.add_argument("--detail-dim", type=int, default=384, help="detailed matching feature dimension")
    parser.add_argument("--label-smoothing", type=float, default=0.1, help="InfoNCE label smoothing")
    parser.add_argument("--heatmap-sigma", type=float, default=HEATMAP_SIGMA, help="Gaussian target sigma in feature cells")
    parser.add_argument("--retrieval-loss-weight", type=float, default=RETRIEVAL_LOSS_WEIGHT)
    parser.add_argument("--localization-loss-weight", type=float, default=LOCALIZATION_LOSS_WEIGHT)
    parser.add_argument("--bbox-loss-weight", type=float, default=BBOX_LOSS_WEIGHT)
    parser.add_argument("--rerank-loss-weight", type=float, default=RERANK_LOSS_WEIGHT)
    parser.add_argument(
        "--retrieval-only-epochs",
        type=int,
        default=RETRIEVAL_ONLY_EPOCHS,
        help="epochs that optimize only the global retrieval loss",
    )
    parser.add_argument(
        "--grounding-ramp-epochs",
        type=int,
        default=GROUNDING_RAMP_EPOCHS,
        help="epochs used to ramp localization and bbox losses to their configured weights",
    )
    parser.add_argument(
        "--rerank-start-epoch",
        type=int,
        default=RERANK_START_EPOCH,
        help="epoch index where detailed rerank supervision starts",
    )
    parser.add_argument(
        "--memory-queue-size",
        type=int,
        default=MEMORY_QUEUE_SIZE,
        help="detached cross-batch retrieval negatives; <=0 disables the queue",
    )
    parser.add_argument("--grad-clip", type=float, default=1.0, help="gradient clipping norm, <=0 disables")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="/media/data1/feihong/ckpt/unify_geo",
        help="Path to save model checkpoints",
    )
    args = parser.parse_args()
    args.sat_size = (int(args.sat_size[0]), int(args.sat_size[1]))
    main(args)
