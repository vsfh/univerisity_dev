#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@file train_siglip.py
@desc Training SiglipLite on GroundSmgeoDataset.
"""

import time
import random
import json
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, Subset
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
from PIL import Image, ImageDraw
from transformers import AutoImageProcessor
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import argparse

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from bbox.yolo_utils import bbox_iou, build_target, eval_iou_acc, yolo_loss
from grounding.legacy.model.loss import adjust_learning_rate
from grounding.legacy.utils.utils import AverageMeter
from model_ground import Encoder_ground as SiglipLite
from dataset import ShiftedSatelliteDroneDataset

MODEL_NAME = "google/siglip2-base-patch16-224"
CACHE_DIR = "/media/data1/feihong/remote/hf_cache"
IMG_SIZE = (768, 432)  # (width, height)
BATCH_SIZE = 32
GRAD_ACCUMULATION_STEPS = 2
GRAD_CLIP_NORM = 1.0
USE_AMP = True
USE_HEATMAP_LOSS = True
HEATMAP_LOSS_WEIGHT = 0.2
HEATMAP_CONFIDENCE_WEIGHT = 0.5
HEATMAP_LOSS_TYPE = ("mse", "cross_entropy")
HEATMAP_SIGMA = 1.5
HEATMAP_RADIUS = 4.5
HEATMAP_BBOX_CENTER_EDGE_VALUE = 0.2
HEATMAP_BBOX_CENTER_LOG_SCALE = 9.0
ANCHORS = "37,41, 78,84, 96,215, 129,129, 194,82, 198,179, 246,280, 395,342, 550,573"

NUM_EPOCHS = 20
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 1e-4
PRINT_FREQ = 50

UNIV_IMAGE_FOLDER = "/media/data1/feihong/remote/image_1024"
UNIV_BBOX_FILE = "/media/data1/feihong/remote/univerisity_dev/runs/test.json"
UNIV_TRAIN_FILE = "/media/data1/feihong/remote/ckpt/train.txt"
UNIV_TEST_FILE = "/media/data1/feihong/remote/ckpt/test.txt"
UNIV_CROP_SIZE = (IMG_SIZE[1], IMG_SIZE[0])
UNIV_DRONE_SIZE = (256, 256)
# Keep satellite tensors in HxW format for the shared dataset pipeline.
UNIV_SAT_SIZE = (IMG_SIZE[1], IMG_SIZE[0])
DEVICE = "cuda:1" if torch.cuda.is_available() else "cpu"

CVOGL_TRANSFORM = None


class TransformProcessorWrapper:
    def __init__(self):
        self.to_tensor = torch.nn.Identity()
        self.size = {"height": UNIV_SAT_SIZE[0], "width": UNIV_SAT_SIZE[1]}

    def __call__(self, images, return_tensors="pt"):
        image_np = np.array(images, dtype=np.float32) / 255.0
        pixel_values = torch.from_numpy(image_np).permute(2, 0, 1)
        return {"pixel_values": pixel_values.unsqueeze(0)}


class SiglipProcessorWrapper:
    """SigLIP2 image processor wrapper matching unified_siglip_supp.py."""

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        cache_dir: str = CACHE_DIR,
        size: Optional[Dict[str, int]] = None,
    ):
        kwargs = {"cache_dir": cache_dir}
        if size is not None:
            kwargs["size"] = size
        self.processor = AutoImageProcessor.from_pretrained(model_name, **kwargs)
        self.size = self.processor.size

    def __call__(self, images, return_tensors="pt"):
        return self.processor(images=images, return_tensors=return_tensors)


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


def build_geo_features(batch: Dict, device: torch.device) -> torch.Tensor:
    angles = batch["angle"].to(device, non_blocking=True).float()
    angles_rad = torch.deg2rad(angles)
    heights = batch["height"].to(device, non_blocking=True).float()
    return torch.cat(
        [
            torch.cos(angles_rad)[..., None],
            torch.sin(angles_rad)[..., None],
            heights[..., None] / 300.0,
        ],
        dim=1,
    )


def format_satellite_img_bbox(
    image,
    bbox,
    mode="train",
    target_size=UNIV_SAT_SIZE,
):
    """Crop satellite image around bbox and resize."""
    x1, y1, x2, y2 = bbox
    width, height = image.size

    min_crop = max(y2 - y1, x2 - x1) * 3
    crop_size = random.uniform(min_crop, max(min_crop, height))

    if mode == "test":
        crop_size = int(0.8 * height)
    min_left = (width / 2) - crop_size
    max_left = width / 2
    min_top = (height / 2) - crop_size
    max_top = height / 2

    min_left = max(0, min_left)
    min_top = max(0, min_top)
    max_left = min(width - crop_size, max_left)
    max_top = min(height - crop_size, max_top)

    if max_left < min_left:
        max_left = min_left
    if max_top < min_top:
        max_top = min_top

    min_left = max(min_left, x2 - crop_size)
    min_top = max(min_top, y2 - crop_size)
    max_left = min(max_left, x1)
    max_top = min(max_top, y1)

    left = random.uniform(min_left, max_left)
    top = random.uniform(min_top, max_top)
    if mode == "test":
        left = (min_left + max_left) / 2
        top = (min_top + max_top) / 2

    right = left + crop_size
    bottom = top + crop_size

    image = image.crop((left, top, right, bottom))
    crop_w, crop_h = image.size

    target_h, target_w = int(target_size[0]), int(target_size[1])
    image = image.resize((target_w, target_h), Image.Resampling.BILINEAR)

    scale_x = target_w / max(float(crop_w), 1.0)
    scale_y = target_h / max(float(crop_h), 1.0)
    new_x1 = (x1 - left) * scale_x
    new_y1 = (y1 - top) * scale_y
    new_x2 = (x2 - left) * scale_x
    new_y2 = (y2 - top) * scale_y
    return image, [new_x1, new_y1, new_x2, new_y2]


def visualize(
    image_tensor: torch.Tensor,
    bbox_tensor: torch.Tensor,
    output_path: str = "a.png",
    color: Tuple[int, int, int] = (0, 255, 0),
    width: int = 3,
) -> None:
    """Draw bbox on (3, H, W) tensor image and save to file."""
    image = image_tensor
    image = torch.clamp(image, 0, 1)

    image = image.permute(1, 2, 0).cpu().numpy()
    H, W = image.shape[:2]

    x1, y1, x2, y2 = bbox_tensor[0].cpu().tolist()

    x1 = int(np.clip(x1, 0, W))
    y1 = int(np.clip(y1, 0, H))
    x2 = int(np.clip(x2, 0, W))
    y2 = int(np.clip(y2, 0, H))

    image_uint8 = (image * 255).astype(np.uint8)
    pil_image = Image.fromarray(image_uint8)
    draw = ImageDraw.Draw(pil_image)
    draw.rectangle([x1, y1, x2, y2], outline=color, width=width)

    os.makedirs(
        os.path.dirname(output_path) if os.path.dirname(output_path) else ".",
        exist_ok=True,
    )
    pil_image.save(output_path)


def build_heatmap_target(
    target_bbox: torch.Tensor,
    heatmap_hw: Tuple[int, int],
    image_wh: Tuple[int, int],
) -> torch.Tensor:
    """Build bbox-center heatmap targets matching unified_siglip_supp defaults."""
    grid_h, grid_w = int(heatmap_hw[0]), int(heatmap_hw[1])
    image_w, image_h = float(image_wh[0]), float(image_wh[1])
    device = target_bbox.device
    dtype = target_bbox.dtype

    x1 = torch.minimum(target_bbox[:, 0], target_bbox[:, 2]).clamp(0.0, image_w)
    y1 = torch.minimum(target_bbox[:, 1], target_bbox[:, 3]).clamp(0.0, image_h)
    x2 = torch.maximum(target_bbox[:, 0], target_bbox[:, 2]).clamp(0.0, image_w)
    y2 = torch.maximum(target_bbox[:, 1], target_bbox[:, 3]).clamp(0.0, image_h)

    center_x = (x1 + x2) * 0.5 / max(image_w, 1.0) * grid_w
    center_y = (y1 + y2) * 0.5 / max(image_h, 1.0) * grid_h
    center_x = center_x.clamp(0.0, grid_w - 1e-6)
    center_y = center_y.clamp(0.0, grid_h - 1e-6)

    ys = torch.arange(grid_h, device=device, dtype=dtype).view(1, grid_h, 1)
    xs = torch.arange(grid_w, device=device, dtype=dtype).view(1, 1, grid_w)
    x_img = (xs + 0.5) / max(float(grid_w), 1.0) * image_w
    y_img = (ys + 0.5) / max(float(grid_h), 1.0) * image_h

    inside_x = (x_img >= x1.view(-1, 1, 1)) & (x_img <= x2.view(-1, 1, 1))
    inside_y = (y_img >= y1.view(-1, 1, 1)) & (y_img <= y2.view(-1, 1, 1))
    inside = inside_x & inside_y

    center_img_x = (x1 + x2).view(-1, 1, 1) * 0.5
    center_img_y = (y1 + y2).view(-1, 1, 1) * 0.5
    half_w = ((x2 - x1) * 0.5).view(-1, 1, 1).clamp_min(1e-6)
    half_h = ((y2 - y1) * 0.5).view(-1, 1, 1).clamp_min(1e-6)

    norm_radius = torch.maximum(
        (x_img - center_img_x).abs() / half_w,
        (y_img - center_img_y).abs() / half_h,
    ).clamp(0.0, 1.0)
    log_scale = max(float(HEATMAP_BBOX_CENTER_LOG_SCALE), 1e-6)
    decay = torch.log1p(norm_radius * log_scale) / np.log1p(log_scale)
    target = 1.0 - (1.0 - float(HEATMAP_BBOX_CENTER_EDGE_VALUE)) * decay
    target = target * inside.to(dtype=dtype)

    center_x_idx = torch.floor(center_x).long().clamp(0, grid_w - 1)
    center_y_idx = torch.floor(center_y).long().clamp(0, grid_h - 1)
    batch_idx = torch.arange(target.shape[0], device=device)
    target[batch_idx, center_y_idx, center_x_idx] = 1.0
    return target.unsqueeze(1).clamp(0.0, 1.0)


def heatmap_loss_fn(
    heatmap_logits: torch.Tensor,
    target_bbox: torch.Tensor,
    image_wh: Tuple[int, int],
) -> torch.Tensor:
    heatmap_target = build_heatmap_target(
        target_bbox=target_bbox,
        heatmap_hw=heatmap_logits.shape[-2:],
        image_wh=image_wh,
    )
    pred = heatmap_logits.float().flatten(1)
    target = heatmap_target.float().flatten(1)
    target_prob = target / target.sum(dim=1, keepdim=True).clamp_min(1e-6)
    pred_prob = pred / pred.sum(dim=1, keepdim=True).clamp_min(1e-6)

    loss = heatmap_logits.new_zeros(())
    if "mse" in HEATMAP_LOSS_TYPE:
        loss = loss + F.mse_loss(
            pred_prob,
            target_prob,
            reduction="mean",
        ) * pred_prob.shape[1]
    if "cross_entropy" in HEATMAP_LOSS_TYPE:
        loss = loss + -(
            target_prob * pred_prob.clamp_min(1e-8).log()
        ).sum(dim=1).mean()
    return loss


def add_heatmap_to_confidence(
    pred_anchor: torch.Tensor,
    heatmap_logits: Optional[torch.Tensor],
    confidence_weight: float = HEATMAP_CONFIDENCE_WEIGHT,
) -> torch.Tensor:
    if heatmap_logits is None or confidence_weight <= 0.0:
        return pred_anchor

    if heatmap_logits.shape[-2:] != pred_anchor.shape[-2:]:
        heatmap_logits = F.interpolate(
            heatmap_logits,
            size=pred_anchor.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

    heatmap_logits = heatmap_logits.detach().to(dtype=pred_anchor.dtype)
    heat_confidence = float(confidence_weight) * heatmap_logits.unsqueeze(1)
    return torch.cat(
        [
            pred_anchor[:, :, :4, :, :],
            pred_anchor[:, :, 4:5, :, :] + heat_confidence,
        ],
        dim=2,
    )


def train_epoch(
    loader,
    model,
    optimizer,
    scaler,
    epoch,
    anchors_full,
    img_size,
    use_amp=USE_AMP,
    grad_accumulation_steps=GRAD_ACCUMULATION_STEPS,
    grad_clip_norm=GRAD_CLIP_NORM,
    use_heatmap_loss=USE_HEATMAP_LOSS,
    heatmap_loss_weight=HEATMAP_LOSS_WEIGHT,
    heatmap_confidence_weight=HEATMAP_CONFIDENCE_WEIGHT,
    print_freq=PRINT_FREQ,
):
    """Train for one epoch."""
    model.train()

    batch_time = AverageMeter()
    losses = AverageMeter()
    geo_losses = AverageMeter()
    cls_losses = AverageMeter()
    heatmap_losses = AverageMeter()

    amp_enabled = (
        bool(use_amp) and torch.cuda.is_available() and str(DEVICE).startswith("cuda")
    )
    grad_accumulation_steps = max(1, int(grad_accumulation_steps))
    optimizer.zero_grad(set_to_none=True)

    end = time.time()
    for batch_idx, batch in enumerate(loader):
        query_imgs = batch["target_pixel_values"].to(DEVICE, non_blocking=True)
        rs_imgs = batch["search_pixel_values"].to(DEVICE, non_blocking=True)
        ori_gt_bbox = batch["bbox"].to(DEVICE)
        geo = build_geo_features(batch, torch.device(DEVICE))

        with torch.amp.autocast("cuda", enabled=amp_enabled):
            output = model(query_imgs, rs_imgs, angle=geo)
            pred_anchor = output[0]
            heatmap_logits = output[6] if len(output) > 6 else None
            pred_anchor = pred_anchor.view(
                pred_anchor.shape[0], 9, 5, pred_anchor.shape[2], pred_anchor.shape[3]
            )
            pred_anchor = add_heatmap_to_confidence(
                pred_anchor,
                heatmap_logits,
                confidence_weight=heatmap_confidence_weight,
            )

            image_wh = (rs_imgs.shape[-1], rs_imgs.shape[-2])
            grid_wh = (pred_anchor.shape[4], pred_anchor.shape[3])

            new_gt_bbox, best_anchor_gi_gj = build_target(
                ori_gt_bbox, anchors_full, image_wh, grid_wh
            )

            loss_geo, loss_cls = yolo_loss(
                pred_anchor, new_gt_bbox, anchors_full, best_anchor_gi_gj, image_wh
            )
            heatmap_loss = pred_anchor.new_zeros(())
            if use_heatmap_loss and heatmap_logits is not None:
                heatmap_loss = heatmap_loss_fn(
                    heatmap_logits,
                    ori_gt_bbox,
                    image_wh=image_wh,
                )
            loss = loss_geo + loss_cls + heatmap_loss_weight * heatmap_loss

        loss_to_backward = loss / grad_accumulation_steps
        scaler.scale(loss_to_backward).backward()
        should_step = (
            (batch_idx + 1) % grad_accumulation_steps == 0
            or (batch_idx + 1) == len(loader)
        )
        if should_step:
            if grad_clip_norm is not None and grad_clip_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        losses.update(loss.item(), query_imgs.shape[0])
        geo_losses.update(loss_geo.item(), query_imgs.shape[0])
        cls_losses.update(loss_cls.item(), query_imgs.shape[0])
        heatmap_losses.update(heatmap_loss.item(), query_imgs.shape[0])

        batch_time.update(time.time() - end)
        end = time.time()

        if batch_idx % print_freq == 0:
            print(
                f"Epoch: [{epoch}][{batch_idx}/{len(loader)}]\t"
                f"Time: {batch_time.val:.3f}\t"
                f"Loss: {losses.val:.4f} ({losses.avg:.4f})\t"
                f"Geo: {geo_losses.val:.4f} ({geo_losses.avg:.4f})\t"
                f"Cls: {cls_losses.val:.4f} ({cls_losses.avg:.4f})\t"
                f"Heatmap: {heatmap_losses.val:.4f} ({heatmap_losses.avg:.4f})"
            )

    return losses.avg, geo_losses.avg, cls_losses.avg, heatmap_losses.avg


def validate(loader, model, anchors_full, img_size):
    """Validate model with eval_ground-style metrics."""
    model.eval()

    iou_values: List[float] = []
    center_distances: List[float] = []

    for batch in tqdm(loader, desc="Validating"):
        query_imgs = batch["target_pixel_values"].to(DEVICE, non_blocking=True)
        rs_imgs = batch["search_pixel_values"].to(DEVICE, non_blocking=True)
        ori_gt_bbox = batch["bbox"].to(DEVICE)
        geo = build_geo_features(batch, torch.device(DEVICE))

        with torch.no_grad():
            pred_anchor, _ = model(query_imgs, rs_imgs, geo=geo)

        pred_anchor = pred_anchor.view(
            pred_anchor.shape[0], 9, 5, pred_anchor.shape[2], pred_anchor.shape[3]
        )

        image_wh = (rs_imgs.shape[-1], rs_imgs.shape[-2])
        grid_wh = (pred_anchor.shape[4], pred_anchor.shape[3])

        _, best_anchor_gi_gj = build_target(
            ori_gt_bbox, anchors_full, image_wh, grid_wh
        )

        _, _, _, _, pred_bbox_xyxy, target_bbox_xyxy = eval_iou_acc(
            pred_anchor,
            ori_gt_bbox,
            anchors_full,
            best_anchor_gi_gj[:, 1],
            best_anchor_gi_gj[:, 2],
            image_wh,
            iou_threshold_list=[0.5, 0.25],
        )

        for idx in range(pred_bbox_xyxy.shape[0]):
            pred_xyxy = pred_bbox_xyxy[idx].float()
            gt_xyxy = target_bbox_xyxy[idx].float()

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
        return 0.0, 0.0, 0.0

    iou_arr = np.array(iou_values, dtype=np.float32)
    center_arr = np.array(center_distances, dtype=np.float32)
    mean_iou = float(iou_arr.mean())
    ratio_iou_gt_0_5 = float((iou_arr > 0.5).mean())
    mean_center_distance = float(center_arr.mean())

    return mean_iou, ratio_iou_gt_0_5, mean_center_distance


def main(args):
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    exp_name = args.savename if args.savename else "ground_siglip"
    writer = SummaryWriter(f"runs/{exp_name}")

    print("Creating datasets from shared data source...")
    processor = SiglipProcessorWrapper(MODEL_NAME, cache_dir=CACHE_DIR)
    processor_sat = SiglipProcessorWrapper(
        MODEL_NAME,
        cache_dir=CACHE_DIR,
        size={"height": UNIV_SAT_SIZE[0], "width": UNIV_SAT_SIZE[1]},
    )
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

    print(f"Found {len(train_dataset)} training samples, {len(val_dataset)} validation samples")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        drop_last=False,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
    )

    print("Creating model...")
    model = SiglipLite(usesg=True).to(DEVICE)

    optimizer = AdamW(model.parameters(), lr=args.lr)
    amp_enabled = (
        bool(args.use_amp) and torch.cuda.is_available() and DEVICE.startswith("cuda")
    )
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    anchors_full = np.array([float(x) for x in ANCHORS.split(",")])
    anchors_full = anchors_full.reshape(-1, 2)[::-1].copy()
    anchors_full = torch.tensor(anchors_full, dtype=torch.float32).to(DEVICE)

    print(f"Starting training for {args.max_epoch} epochs...")
    for epoch in range(args.max_epoch):
        adjust_learning_rate(args, optimizer, epoch)

        train_loss, train_geo, train_cls, train_heatmap = train_epoch(
            train_loader,
            model,
            optimizer,
            scaler,
            epoch,
            anchors_full,
            args.img_size,
            args.use_amp,
            args.grad_accumulation_steps,
            args.grad_clip_norm,
            args.use_heatmap_loss,
            args.heatmap_loss_weight,
            args.heatmap_confidence_weight,
            args.print_freq,
        )

        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/train_geo", train_geo, epoch)
        writer.add_scalar("Loss/train_cls", train_cls, epoch)
        writer.add_scalar("Loss/train_heatmap", train_heatmap, epoch)

        print(
            f"Epoch {epoch + 1}/{args.max_epoch}:\t"
            f"Train Loss: {train_loss:.4f} "
            f"(Geo: {train_geo:.4f}, Cls: {train_cls:.4f}, "
            f"Heatmap: {train_heatmap:.4f})"
        )
        torch.save(model.state_dict(), f"{args.checkpoint}/last.pth")

    print("\nTraining complete. Saved checkpoint to last.pth")
    writer.close()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="DetGeoLite Training on GroundSmgeoDataset"
    )
    parser.add_argument(
        "--num-workers", type=int, default=8, help="num workers for data loading"
    )
    parser.add_argument(
        "--max-epoch", type=int, default=NUM_EPOCHS, help="num workers for data loading"
    )
    parser.add_argument("--lr", type=float, default=LEARNING_RATE, help="learning rate")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="batch size")
    parser.add_argument(
        "--grad-accumulation-steps",
        type=int,
        default=GRAD_ACCUMULATION_STEPS,
        help="Number of mini-batches to accumulate before optimizer step",
    )
    parser.add_argument(
        "--grad-clip-norm",
        type=float,
        default=GRAD_CLIP_NORM,
        help="Max gradient norm before optimizer step. Use <= 0 to disable.",
    )
    parser.add_argument(
        "--img-size",
        type=int,
        nargs=2,
        metavar=("WIDTH", "HEIGHT"),
        default=list(IMG_SIZE),
        help="image size as WIDTH HEIGHT",
    )
    parser.add_argument(
        "--savename", type=str, default="ground_siglip", help="Name head for saved model"
    )
    parser.add_argument("--seed", type=int, default=2024, help="random seed")
    parser.add_argument(
        "--print-freq", type=int, default=PRINT_FREQ, help="print frequency"
    )
    parser.add_argument(
        "--no-amp",
        dest="use_amp",
        action="store_false",
        help="Disable AMP autocast and GradScaler during training",
    )
    parser.set_defaults(use_amp=USE_AMP)
    parser.add_argument(
        "--no-heatmap-loss",
        dest="use_heatmap_loss",
        action="store_false",
        help="Disable heatmap loss during training",
    )
    parser.set_defaults(use_heatmap_loss=USE_HEATMAP_LOSS)
    parser.add_argument(
        "--heatmap-loss-weight",
        type=float,
        default=HEATMAP_LOSS_WEIGHT,
        help="Weight for heatmap loss",
    )
    parser.add_argument(
        "--heatmap-confidence-weight",
        type=float,
        default=HEATMAP_CONFIDENCE_WEIGHT,
        help="Weight added from heatmap to anchor confidence",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="/media/data1/feihong/remote/ckpt/ground_siglip",
        help="Path to save model checkpoints",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Only run evaluation on test split",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default=None,
        help="Explicit checkpoint path for evaluation",
    )
    parser.add_argument(
        "--subset-height",
        type=int,
        default=None,
        help="Subset test data by exact height",
    )
    parser.add_argument(
        "--subset-angle",
        type=int,
        default=None,
        help="Subset test data by exact angle",
    )
    args = parser.parse_args()
    args.img_size = (int(args.img_size[0]), int(args.img_size[1]))

    os.makedirs(args.checkpoint, exist_ok=True)

    if args.eval_only:
        eval(args)
    else:
        main(args)
