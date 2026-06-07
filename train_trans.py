#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GeoFormer/TransGeo-style retrieval + grounding training.

The backbone is rebuilt around the available GeoFormer checkpoint: ResNet101
visual features are projected through the original visual embedding layout and
encoded by the pretrained T5 encoder. Task-specific retrieval, heatmap, and
bbox heads adapt it to the shared drone/satellite dataset.
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
import torchvision.models as torchvision_models
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from transformers import T5Config, T5EncoderModel
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
NUM_EPOCHS = 40
LEARNING_RATE = 4.5e-4
BACKBONE_LEARNING_RATE = 1e-5
WEIGHT_DECAY = 0.04
PRINT_FREQ = 50
PROJECTION_DIM = 768
DETAIL_DIM = 384
EMBED_DIM = 768
DEPTH = 12
NUM_HEADS = 12
PATCH_SIZE = 32
RETRIEVAL_LOSS_WEIGHT = 1.0
LOCALIZATION_LOSS_WEIGHT = 1.0
BBOX_LOSS_WEIGHT = 5.0
PRETRAINED_CHECKPOINT = "/media/data1/feihong/ckpt/geoformer.pth"
VAL_FRACTION = 0.25
DEVICE = "cuda:1" if torch.cuda.is_available() else "cpu"


class GeoFormerVisualEmbedding(nn.Module):
    """Visual token embedding layout matching geoformer.pth."""

    def __init__(self, feat_dim: int = 2048, hidden_dim: int = EMBED_DIM, max_order: int = 32200):
        super().__init__()
        self.feat_embedding = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.absolute_vis_pos_embedding = nn.Sequential(
            nn.Linear(5, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.obj_order_embedding = nn.Embedding(max_order, hidden_dim)
        self.img_order_embedding = nn.Embedding(2, hidden_dim)

    def forward(self, feat_map: torch.Tensor, image_order_id: int) -> Tuple[torch.Tensor, Tuple[int, int]]:
        batch_size, channels, grid_h, grid_w = feat_map.shape
        feat_tokens = feat_map.flatten(2).transpose(1, 2)

        y0 = torch.arange(grid_h, device=feat_map.device, dtype=feat_map.dtype) / float(grid_h)
        x0 = torch.arange(grid_w, device=feat_map.device, dtype=feat_map.dtype) / float(grid_w)
        y1 = (torch.arange(grid_h, device=feat_map.device, dtype=feat_map.dtype) + 1.0) / float(grid_h)
        x1 = (torch.arange(grid_w, device=feat_map.device, dtype=feat_map.dtype) + 1.0) / float(grid_w)
        yy0, xx0 = torch.meshgrid(y0, x0, indexing="ij")
        yy1, xx1 = torch.meshgrid(y1, x1, indexing="ij")
        area = (yy1 - yy0) * (xx1 - xx0)
        pos = torch.stack([xx0, yy0, xx1, yy1, area], dim=-1).view(1, grid_h * grid_w, 5)
        pos = pos.expand(batch_size, -1, -1)

        order_ids = torch.arange(grid_h * grid_w, device=feat_map.device).clamp_max(
            self.obj_order_embedding.num_embeddings - 1
        )
        order_emb = self.obj_order_embedding(order_ids).unsqueeze(0)
        image_ids = torch.full(
            (batch_size, grid_h * grid_w),
            int(image_order_id),
            device=feat_map.device,
            dtype=torch.long,
        )
        image_emb = self.img_order_embedding(image_ids)
        tokens = self.feat_embedding(feat_tokens) + self.absolute_vis_pos_embedding(pos) + order_emb + image_emb
        return tokens, (grid_h, grid_w)


class GeoFormerImageEncoder(nn.Module):
    """ResNet101 + T5 encoder subset that can load geoformer.pth."""

    def __init__(self, hidden_dim: int = EMBED_DIM, num_layers: int = DEPTH, num_heads: int = NUM_HEADS):
        super().__init__()
        if hidden_dim != 768 or num_layers != 12 or num_heads != 12:
            raise ValueError("geoformer.pth requires hidden_dim=768, num_layers=12, num_heads=12.")
        resnet = torchvision_models.resnet101(weights=None)
        self.resnet = nn.Sequential(*list(resnet.children())[:-2])
        config = T5Config(
            vocab_size=32200,
            d_model=hidden_dim,
            d_kv=hidden_dim // num_heads,
            d_ff=hidden_dim * 4,
            num_layers=num_layers,
            num_decoder_layers=num_layers,
            num_heads=num_heads,
            relative_attention_num_buckets=32,
            dropout_rate=0.0,
            feed_forward_proj="relu",
            pad_token_id=0,
            eos_token_id=1,
            decoder_start_token_id=0,
        )
        t5_encoder = T5EncoderModel(config)
        self.shared = t5_encoder.shared
        self.encoder = t5_encoder.encoder
        self.encoder.visual_embedding = GeoFormerVisualEmbedding(hidden_dim=hidden_dim)

    def forward(self, images: torch.Tensor, image_order_id: int) -> Tuple[torch.Tensor, Tuple[int, int], torch.Tensor]:
        x = self.resnet[0](images)
        x = self.resnet[1](x)
        x = self.resnet[2](x)
        x = self.resnet[3](x)
        x = self.resnet[4](x)
        x = self.resnet[5](x)
        fine_map = self.resnet[6](x)
        feat_map = self.resnet[7](fine_map)
        tokens, grid_hw = self.encoder.visual_embedding(feat_map, image_order_id=image_order_id)
        encoded = self.encoder(inputs_embeds=tokens, return_dict=True).last_hidden_state
        return encoded, grid_hw, fine_map


def load_geoformer_pretrained(model: nn.Module, checkpoint_path: str) -> Dict[str, int]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Expected GeoFormer state_dict dict, got {type(checkpoint)}.")

    mapped_state: Dict[str, torch.Tensor] = {}
    skipped_decoder = 0
    for key, value in checkpoint.items():
        clean_key = key.removeprefix("module.")
        if clean_key.startswith("decoder.") or clean_key.startswith("lm_head."):
            skipped_decoder += 1
            continue
        if clean_key.startswith("resnet."):
            mapped_state[f"geoformer.{clean_key}"] = value
        elif clean_key.startswith("shared.") or clean_key.startswith("encoder."):
            mapped_state[f"geoformer.{clean_key}"] = value

    model_state = model.state_dict()
    compatible_state = {}
    skipped_shape = 0
    skipped_missing = 0
    for key, value in mapped_state.items():
        if key not in model_state:
            skipped_missing += 1
            continue
        if tuple(model_state[key].shape) != tuple(value.shape):
            skipped_shape += 1
            continue
        compatible_state[key] = value
    result = model.load_state_dict(compatible_state, strict=False)
    return {
        "loaded": len(compatible_state),
        "skipped_decoder": skipped_decoder,
        "skipped_missing": skipped_missing,
        "skipped_shape": skipped_shape,
        "model_missing": len(result.missing_keys),
        "unexpected": len(result.unexpected_keys),
    }


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
        del sat_size, drone_size, patch_size
        self.geoformer = GeoFormerImageEncoder(
            hidden_dim=embed_dim,
            num_layers=depth,
            num_heads=num_heads,
        )
        self.query_global = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, proj_dim),
        )
        self.reference_global = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, proj_dim),
        )
        self.query_detail = nn.Sequential(
            nn.Conv2d(1024, detail_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(detail_dim),
            nn.GELU(),
            nn.Conv2d(detail_dim, detail_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(detail_dim),
            nn.GELU(),
        )
        self.reference_detail = nn.Sequential(
            nn.Conv2d(1024, detail_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(detail_dim),
            nn.GELU(),
            nn.Conv2d(detail_dim, detail_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(detail_dim),
            nn.GELU(),
        )
        self.bbox_head = nn.Sequential(
            nn.Conv2d(detail_dim * 2 + 1, detail_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(detail_dim),
            nn.GELU(),
            nn.Conv2d(detail_dim, detail_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(detail_dim),
            nn.GELU(),
            nn.Conv2d(detail_dim, 4, kernel_size=1),
        )
        self.logit_scale = nn.Parameter(torch.tensor(np.log(1 / 0.07), dtype=torch.float32))
        self.temperature = nn.Parameter(torch.tensor(0.07, dtype=torch.float32))
        self.heat_kernel_size = 3

    def _reference_feature_map(self, tokens, grid_hw: Tuple[int, int]):
        grid_h, grid_w = int(grid_hw[0]), int(grid_hw[1])
        return tokens.transpose(1, 2).reshape(tokens.shape[0], -1, grid_h, grid_w)

    def _dynamic_heatmap(self, query_detail_map: torch.Tensor, aerial_detail: torch.Tensor) -> torch.Tensor:
        if query_detail_map.ndim != 4:
            raise ValueError(f"Expected query_detail_map shape (B, C, H, W), got {tuple(query_detail_map.shape)}.")
        if aerial_detail.ndim != 4:
            raise ValueError(f"Expected aerial_detail shape (B, C, H, W), got {tuple(aerial_detail.shape)}.")
        if query_detail_map.shape[0] != aerial_detail.shape[0] or query_detail_map.shape[1] != aerial_detail.shape[1]:
            raise ValueError(
                f"Query/aerial detail mismatch: query={tuple(query_detail_map.shape)}, "
                f"aerial={tuple(aerial_detail.shape)}."
            )

        batch_size, channels, height, width = aerial_detail.shape
        kernel = F.adaptive_avg_pool2d(
            query_detail_map,
            (self.heat_kernel_size, self.heat_kernel_size),
        )
        kernel = kernel - kernel.mean(dim=(2, 3), keepdim=True)
        kernel = F.normalize(kernel.flatten(1), p=2, dim=1).view_as(kernel)
        sat_features = F.normalize(aerial_detail.contiguous(), p=2, dim=1)
        conv_input = sat_features.reshape(1, batch_size * channels, height, width)
        conv_kernel = kernel.reshape(batch_size, channels, self.heat_kernel_size, self.heat_kernel_size)
        heatmap = F.conv2d(
            conv_input,
            conv_kernel,
            padding=self.heat_kernel_size // 2,
            groups=batch_size,
        )
        return heatmap.view(batch_size, 1, height, width)

    def forward(self, query_imgs, aerial_imgs):
        query_tokens, _, query_fine_map = self.geoformer(query_imgs, image_order_id=0)
        aerial_tokens, _, aerial_fine_map = self.geoformer(aerial_imgs, image_order_id=1)

        query_global = F.normalize(self.query_global(query_tokens.mean(dim=1)), p=2, dim=1)
        aerial_global = F.normalize(self.reference_global(aerial_tokens.mean(dim=1)), p=2, dim=1)

        query_detail_map = F.normalize(self.query_detail(query_fine_map), p=2, dim=1)
        aerial_detail = F.normalize(self.reference_detail(aerial_fine_map), p=2, dim=1)
        ground_detail = F.normalize(F.adaptive_avg_pool2d(query_detail_map, 1).flatten(1), p=2, dim=1)

        temp = self.temperature.clamp(min=0.01, max=1.0)
        heatmap_logits = self._dynamic_heatmap(query_detail_map, aerial_detail) / temp
        heat_gate = F.softmax(heatmap_logits.flatten(1), dim=1).view_as(heatmap_logits)
        heat_gate = heat_gate * float(heat_gate.shape[-2] * heat_gate.shape[-1])
        query_map = ground_detail[:, :, None, None].expand_as(aerial_detail)
        bbox_raw = self.bbox_head(torch.cat([aerial_detail, query_map, heat_gate], dim=1))

        scale = self.logit_scale.exp().clamp(max=100.0)
        retrieval_logits = scale * query_global @ aerial_global.t()

        return {
            "query_global": query_global,
            "aerial_global": aerial_global,
            "ground_detail": ground_detail,
            "aerial_detail": aerial_detail,
            "heatmap_logits": heatmap_logits,
            "bbox_raw": bbox_raw,
            "logit_scale": scale,
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
    if args.pretrained_checkpoint:
        if not os.path.exists(args.pretrained_checkpoint):
            raise FileNotFoundError(f"GeoFormer checkpoint not found: {args.pretrained_checkpoint}")
        load_info = load_geoformer_pretrained(model, args.pretrained_checkpoint)
        print(f"Loaded GeoFormer checkpoint: {load_info}")
    if args.freeze_geoformer:
        for param in model.geoformer.parameters():
            param.requires_grad = False
        print("Frozen GeoFormer backbone; training retrieval/detail/bbox heads only.")

    if args.freeze_geoformer:
        trainable_params = [param for param in model.parameters() if param.requires_grad]
        optimizer = AdamW(
            [{"params": trainable_params, "lr": args.lr, "base_lr": args.lr}],
            weight_decay=args.weight_decay,
        )
    else:
        backbone_params = [param for param in model.geoformer.parameters() if param.requires_grad]
        head_params = [
            param
            for name, param in model.named_parameters()
            if param.requires_grad and not name.startswith("geoformer.")
        ]
        optimizer = AdamW(
            [
                {"params": backbone_params, "lr": args.backbone_lr, "base_lr": args.backbone_lr},
                {"params": head_params, "lr": args.lr, "base_lr": args.lr},
            ],
            weight_decay=args.weight_decay,
        )
        print(
            f"Optimizer param groups: geoformer lr={args.backbone_lr:.2e}, "
            f"heads lr={args.lr:.2e}"
        )
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
    parser.add_argument("--backbone-lr", type=float, default=BACKBONE_LEARNING_RATE)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
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
        "--pretrained-checkpoint",
        type=str,
        default=PRETRAINED_CHECKPOINT,
        help="GeoFormer checkpoint used to initialize ResNet101 + T5 visual encoder",
    )
    parser.add_argument(
        "--no-pretrained-checkpoint",
        dest="pretrained_checkpoint",
        action="store_const",
        const=None,
        help="Disable GeoFormer checkpoint initialization",
    )
    parser.add_argument(
        "--freeze-geoformer",
        action="store_true",
        help="Freeze the loaded GeoFormer backbone and train only task heads",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="/media/data1/feihong/ckpt/trans_geo",
        help="Path to save model checkpoints",
    )
    args = parser.parse_args()
    args.sat_size = (int(args.sat_size[0]), int(args.sat_size[1]))
    main(args)
