#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@file train_sample.py
@desc Training original Sample4Geo-style retrieval with an auxiliary grounding head.
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
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import argparse

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from ground_cvos import SampleGeoGrounding, load_sample4geo_backbone
from grounding.legacy.model.loss import adjust_learning_rate
from grounding.legacy.utils.utils import AverageMeter

from dataset import ShiftedSatelliteDroneDataset

IMG_SIZE = (768, 432)  # (width, height)
BATCH_SIZE = 8
NUM_EPOCHS = 20
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
PRINT_FREQ = 50
GRAD_CLIP_NORM = 100.0
UNIV_IMAGE_FOLDER = "/media/data1/feihong/image_1024"
UNIV_BBOX_FILE = "/media/data1/feihong/univerisity_dev/runs/test.json"
UNIV_TRAIN_FILE = "/media/data1/feihong/ckpt/train.txt"
UNIV_TEST_FILE = "/media/data1/feihong/ckpt/test.txt"
# UNIV_CROP_SIZE = (640, 640)
UNIV_DRONE_SIZE = (256, 256)
UNIV_SAT_SIZE = IMG_SIZE
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

CVOGL_TRANSFORM = None

class TransformProcessorWrapper:
    def __init__(self, image_size=IMG_SIZE):
        self.to_tensor = torch.nn.Identity()
        self.size = {"height": int(image_size[1]), "width": int(image_size[0])}
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

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
        return {"input_ids": torch.zeros((1, max_length), dtype=torch.long)}


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
    ratio = image.size[0] / target_size[0]
    image = image.resize(target_size)

    new_x1 = (x1 - left) / ratio
    new_y1 = (y1 - top) / ratio
    new_x2 = (x2 - left) / ratio
    new_y2 = (y2 - top) / ratio
    return image, [new_x1, new_y1, new_x2, new_y2]


def resize_drone_image(image, target_size=UNIV_DRONE_SIZE):
    """Resize drone image to target size."""
    return image.resize(target_size, Image.Resampling.BILINEAR)


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


def symmetric_info_nce_loss(query_feats, reference_feats, temperature: float = 0.07, logit_scale=None):
    if query_feats.ndim != 2 or reference_feats.ndim != 2:
        raise ValueError(
            f"Expected retrieval features with shape (B, C), got {tuple(query_feats.shape)} and {tuple(reference_feats.shape)}."
        )
    if query_feats.shape != reference_feats.shape:
        raise ValueError(
            f"Expected paired retrieval features with the same shape, got {tuple(query_feats.shape)} and {tuple(reference_feats.shape)}."
        )
    query_feats = F.normalize(query_feats, p=2, dim=1)
    reference_feats = F.normalize(reference_feats, p=2, dim=1)
    if logit_scale is None:
        logit_scale = 1.0 / float(temperature)
    logits = torch.matmul(query_feats, reference_feats.T) * logit_scale
    labels = torch.arange(query_feats.shape[0], device=query_feats.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))


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
    return heatmap.unsqueeze(1).clamp(0.0, 1.0)


def build_bbox_target(gt_bbox, feature_hw: Tuple[int, int], image_wh: Tuple[int, int]):
    feat_h, feat_w = int(feature_hw[0]), int(feature_hw[1])
    image_w, image_h = float(image_wh[0]), float(image_wh[1])
    cx = ((gt_bbox[:, 0] + gt_bbox[:, 2]) * 0.5 / image_w) * feat_w
    cy = ((gt_bbox[:, 1] + gt_bbox[:, 3]) * 0.5 / image_h) * feat_h
    gi = cx.floor().long().clamp(0, feat_w - 1)
    gj = cy.floor().long().clamp(0, feat_h - 1)
    frac_x = cx - gi.to(dtype=gt_bbox.dtype)
    frac_y = cy - gj.to(dtype=gt_bbox.dtype)
    box_w = (gt_bbox[:, 2] - gt_bbox[:, 0]).clamp_min(1.0) / image_w
    box_h = (gt_bbox[:, 3] - gt_bbox[:, 1]).clamp_min(1.0) / image_h
    return torch.stack([frac_x, frac_y, box_w, box_h], dim=1), gi, gj


def box_iou_diagonal(pred_bbox, target_bbox):
    pred_x1 = torch.maximum(pred_bbox[:, 0], target_bbox[:, 0])
    pred_y1 = torch.maximum(pred_bbox[:, 1], target_bbox[:, 1])
    pred_x2 = torch.minimum(pred_bbox[:, 2], target_bbox[:, 2])
    pred_y2 = torch.minimum(pred_bbox[:, 3], target_bbox[:, 3])
    inter = (pred_x2 - pred_x1).clamp_min(0.0) * (pred_y2 - pred_y1).clamp_min(0.0)
    pred_area = (pred_bbox[:, 2] - pred_bbox[:, 0]).clamp_min(0.0) * (
        pred_bbox[:, 3] - pred_bbox[:, 1]
    ).clamp_min(0.0)
    target_area = (target_bbox[:, 2] - target_bbox[:, 0]).clamp_min(0.0) * (
        target_bbox[:, 3] - target_bbox[:, 1]
    ).clamp_min(0.0)
    return inter / (pred_area + target_area - inter).clamp_min(1e-6)


def normalized_cxcywh_to_xyxy(box, image_wh: Tuple[int, int]):
    image_w, image_h = float(image_wh[0]), float(image_wh[1])
    cx = box[:, 0] * image_w
    cy = box[:, 1] * image_h
    bw = box[:, 2] * image_w
    bh = box[:, 3] * image_h
    x1 = (cx - bw * 0.5).clamp(0.0, image_w)
    y1 = (cy - bh * 0.5).clamp(0.0, image_h)
    x2 = (cx + bw * 0.5).clamp(0.0, image_w)
    y2 = (cy + bh * 0.5).clamp(0.0, image_h)
    return torch.stack([x1, y1, x2, y2], dim=1)


def anchor_free_bbox_loss(
    heatmap_logits,
    bbox_raw,
    gt_bbox,
    image_wh: Tuple[int, int],
    sigma: float,
    focal_alpha: float = 2.0,
    focal_beta: float = 4.0,
):
    B, _, feat_h, feat_w = heatmap_logits.shape
    heatmap_target = build_heatmap_target(gt_bbox, (feat_h, feat_w), image_wh, sigma)
    bbox_target, gi, gj = build_bbox_target(gt_bbox, (feat_h, feat_w), image_wh)
    batch_idx = torch.arange(B, device=gt_bbox.device)

    heat_prob = torch.sigmoid(heatmap_logits)
    heatmap_target = heatmap_target.clone()
    heatmap_target[batch_idx, 0, gj, gi] = 1.0
    pos_mask = heatmap_target.eq(1.0)
    neg_mask = heatmap_target.lt(1.0)
    pos_loss = -torch.log(heat_prob.clamp_min(1e-6)) * (1.0 - heat_prob).pow(focal_alpha)
    neg_loss = (
        -torch.log((1.0 - heat_prob).clamp_min(1e-6))
        * heat_prob.pow(focal_alpha)
        * (1.0 - heatmap_target).pow(focal_beta)
    )
    heat_loss = (pos_loss[pos_mask].sum() + neg_loss[neg_mask].sum()) / pos_mask.sum().clamp_min(1)

    selected_bbox = bbox_raw[batch_idx, :, gj, gi]
    pred_offset = torch.sigmoid(selected_bbox[:, :2])
    pred_wh = torch.sigmoid(selected_bbox[:, 2:]).clamp_min(1e-4)
    pred_cx = (gi.to(dtype=gt_bbox.dtype) + pred_offset[:, 0]) / float(feat_w)
    pred_cy = (gj.to(dtype=gt_bbox.dtype) + pred_offset[:, 1]) / float(feat_h)
    pred_norm = torch.stack([pred_cx, pred_cy, pred_wh[:, 0], pred_wh[:, 1]], dim=1)
    target_norm = torch.stack(
        [
            (gi.to(dtype=gt_bbox.dtype) + bbox_target[:, 0]) / float(feat_w),
            (gj.to(dtype=gt_bbox.dtype) + bbox_target[:, 1]) / float(feat_h),
            bbox_target[:, 2],
            bbox_target[:, 3],
        ],
        dim=1,
    )
    l1_loss = F.l1_loss(pred_norm, target_norm)
    pred_xyxy = normalized_cxcywh_to_xyxy(pred_norm, image_wh)
    target_xyxy = normalized_cxcywh_to_xyxy(target_norm, image_wh)
    iou_loss = 1.0 - box_iou_diagonal(pred_xyxy, target_xyxy).mean()
    box_loss = l1_loss + iou_loss
    return heat_loss + box_loss, heat_loss, box_loss


def decode_anchor_free_bbox(heatmap_logits, bbox_raw, image_wh: Tuple[int, int]):
    B, _, feat_h, feat_w = heatmap_logits.shape
    heat_prob = torch.sigmoid(heatmap_logits[:, 0])
    flat_idx = heat_prob.flatten(1).argmax(dim=1)
    gj = torch.div(flat_idx, feat_w, rounding_mode="floor")
    gi = flat_idx % feat_w
    batch_idx = torch.arange(B, device=heatmap_logits.device)
    selected_bbox = bbox_raw[batch_idx, :, gj, gi]
    offset = torch.sigmoid(selected_bbox[:, :2])
    wh = torch.sigmoid(selected_bbox[:, 2:]).clamp_min(1e-4)
    cx = (gi.to(dtype=bbox_raw.dtype) + offset[:, 0]) / float(feat_w)
    cy = (gj.to(dtype=bbox_raw.dtype) + offset[:, 1]) / float(feat_h)
    pred_norm = torch.stack([cx, cy, wh[:, 0], wh[:, 1]], dim=1)
    return normalized_cxcywh_to_xyxy(pred_norm, image_wh)


def train_epoch(
    loader,
    model,
    optimizer,
    epoch,
    args,
    print_freq=PRINT_FREQ,
):
    """Train for one epoch."""
    model.train()

    batch_time = AverageMeter()
    losses = AverageMeter()
    ret_losses = AverageMeter()
    bbox_losses = AverageMeter()
    heat_losses = AverageMeter()
    box_losses = AverageMeter()

    end = time.time()
    for batch_idx, batch in enumerate(loader):
        query_imgs = batch["target_pixel_values"].to(DEVICE, non_blocking=True)
        rs_imgs = batch["search_pixel_values"].to(DEVICE, non_blocking=True)
        ori_gt_bbox = batch["bbox"].to(DEVICE)

        outputs = model(query_imgs, rs_imgs)

        image_wh = (rs_imgs.shape[-1], rs_imgs.shape[-2])
        retrieval_loss = symmetric_info_nce_loss(
            outputs["query_feats"],
            outputs["reference_feats"],
            temperature=args.temperature,
            logit_scale=outputs["logit_scale"],
        )
        bbox_loss, heat_loss, box_loss = anchor_free_bbox_loss(
            outputs["heatmap_logits"],
            outputs["bbox_raw"],
            ori_gt_bbox,
            image_wh,
            sigma=args.heatmap_sigma,
        )
        loss = retrieval_loss + args.bbox_loss_weight * bbox_loss

        optimizer.zero_grad()
        loss.backward()
        if args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
        optimizer.step()

        losses.update(loss.item(), query_imgs.shape[0])
        ret_losses.update(retrieval_loss.item(), query_imgs.shape[0])
        bbox_losses.update(bbox_loss.item(), query_imgs.shape[0])
        heat_losses.update(heat_loss.item(), query_imgs.shape[0])
        box_losses.update(box_loss.item(), query_imgs.shape[0])

        batch_time.update(time.time() - end)
        end = time.time()

        if batch_idx % print_freq == 0:
            print(
                f"Epoch: [{epoch}][{batch_idx}/{len(loader)}]\t"
                f"Time: {batch_time.val:.3f}\t"
                f"Loss: {losses.val:.4f} ({losses.avg:.4f})\t"
                f"Ret: {ret_losses.val:.4f} ({ret_losses.avg:.4f})\t"
                f"Bbox: {bbox_losses.val:.4f} ({bbox_losses.avg:.4f})\t"
                f"Heat: {heat_losses.val:.4f} ({heat_losses.avg:.4f})\t"
                f"Box: {box_losses.val:.4f} ({box_losses.avg:.4f})"
            )

    return losses.avg, ret_losses.avg, bbox_losses.avg, heat_losses.avg, box_losses.avg


def validate(loader, model):
    """Validate model."""
    model.eval()

    accu50_meter = AverageMeter()
    accu25_meter = AverageMeter()
    iou_meter = AverageMeter()

    for batch in tqdm(loader, desc="Validating"):
        query_imgs = batch["target_pixel_values"].to(DEVICE, non_blocking=True)
        rs_imgs = batch["search_pixel_values"].to(DEVICE, non_blocking=True)
        ori_gt_bbox = batch["bbox"].to(DEVICE)

        with torch.no_grad():
            outputs = model(query_imgs, rs_imgs)

        image_wh = (rs_imgs.shape[-1], rs_imgs.shape[-2])
        pred_bbox = decode_anchor_free_bbox(outputs["heatmap_logits"], outputs["bbox_raw"], image_wh)
        iou = box_iou_diagonal(pred_bbox, ori_gt_bbox)
        accu50_meter.update((iou >= 0.5).float().mean().item(), query_imgs.shape[0])
        accu25_meter.update((iou >= 0.25).float().mean().item(), query_imgs.shape[0])
        iou_meter.update(iou.mean().item(), query_imgs.shape[0])

    return accu50_meter.avg, accu25_meter.avg, iou_meter.avg


def main(args):
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    exp_name = args.savename if args.savename else "ground_sample_exp"
    writer = SummaryWriter(f"runs/{exp_name}")

    print("Creating datasets from shared data source...")
    processor = TransformProcessorWrapper(UNIV_DRONE_SIZE)
    processor_sat = TransformProcessorWrapper(args.img_size)
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
        split="train",
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
    model = SampleGeoGrounding(emb_size=1024, pretrained=args.timm_pretrained).to(DEVICE)
    if args.pretrained_checkpoint:
        if not os.path.exists(args.pretrained_checkpoint):
            raise FileNotFoundError(f"Sample4Geo pretrained checkpoint not found: {args.pretrained_checkpoint}")
        load_info = load_sample4geo_backbone(model, args.pretrained_checkpoint)
        print(f"Loaded Sample4Geo pretrained backbone: {load_info}")

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    print(f"Starting training for {args.max_epoch} epochs...")
    for epoch in range(args.max_epoch):
        adjust_learning_rate(args, optimizer, epoch)

        train_loss, train_ret, train_bbox, train_heat, train_box = train_epoch(
            train_loader,
            model,
            optimizer,
            epoch,
            args,
            args.print_freq,
        )

        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/train_retrieval", train_ret, epoch)
        writer.add_scalar("Loss/train_bbox", train_bbox, epoch)
        writer.add_scalar("Loss/train_heatmap", train_heat, epoch)
        writer.add_scalar("Loss/train_box", train_box, epoch)
        val_accu50, val_accu25, val_iou = validate(val_loader, model)
        writer.add_scalar("ValTrain/mean_iou", val_iou, epoch)
        writer.add_scalar("ValTrain/accu50", val_accu50, epoch)
        writer.add_scalar("ValTrain/accu25", val_accu25, epoch)

        print(
            f"Epoch {epoch + 1}/{args.max_epoch}:\t"
            f"Train Loss: {train_loss:.4f} (Ret: {train_ret:.4f}, Bbox: {train_bbox:.4f}, "
            f"Heat: {train_heat:.4f}, Box: {train_box:.4f})\t"
            f"ValTrain IoU: {val_iou:.4f}, Accu50: {val_accu50:.4f}, Accu25: {val_accu25:.4f}"
        )
        torch.save(model.state_dict(), f"{args.checkpoint}/last.pth")

    print("\nTraining complete. Saved checkpoint to last.pth")
    writer.close()


def eval(args):
    print("Evaluating checkpoint on test split...")
    processor = TransformProcessorWrapper(UNIV_DRONE_SIZE)
    processor_sat = TransformProcessorWrapper(args.img_size)
    tokenizer = DummyTokenizer()
    test_dataset = ShiftedSatelliteDroneDataset(
        processor=processor,
        processor_sat=processor_sat,
        tokenizer=tokenizer,
        split="test",
    )

    if args.subset_height is not None or args.subset_angle is not None:
        subset_indices = []
        for idx, sample in enumerate(test_dataset.samples):
            if args.subset_height is not None and int(sample["height"]) != args.subset_height:
                continue
            if args.subset_angle is not None and int(sample["angle"]) != args.subset_angle:
                continue
            subset_indices.append(idx)
        if not subset_indices:
            print(
                f"No samples for subset filters: height={args.subset_height}, angle={args.subset_angle}"
            )
            return
        test_dataset = Subset(test_dataset, subset_indices)
        print(
            f"Subset test samples: {len(subset_indices)} (height={args.subset_height}, angle={args.subset_angle})"
        )
    else:
        print(f"Full test samples: {len(test_dataset)}")

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
    )

    model = SampleGeoGrounding(emb_size=1024, pretrained=args.timm_pretrained).to(DEVICE)
    checkpoint_path = args.checkpoint_path or f"{args.checkpoint}/last.pth"
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"), strict=False)

    accu50, accu25, val_iou = validate(test_loader, model)
    print(
        f"Eval Accu50: {accu50:.4f}, Accu25: {accu25:.4f}, IoU: {val_iou:.4f}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sample4Geo-style retrieval training with an auxiliary grounding head"
    )
    parser.add_argument("--gpu", default="1", help="GPU id (e.g., '1' or '1,2')")
    parser.add_argument(
        "--num-workers", type=int, default=8, help="num workers for data loading"
    )
    parser.add_argument(
        "--max-epoch", type=int, default=NUM_EPOCHS, help="training epoch"
    )
    parser.add_argument("--lr", type=float, default=LEARNING_RATE, help="learning rate")
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY, help="AdamW weight decay")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="batch size")
    parser.add_argument(
        "--img-size",
        type=int,
        nargs=2,
        metavar=("WIDTH", "HEIGHT"),
        default=list(IMG_SIZE),
        help="image size as WIDTH HEIGHT",
    )
    parser.add_argument(
        "--savename",
        type=str,
        default="ground_sample",
        help="Name head for saved model",
    )
    parser.add_argument("--seed", type=int, default=2024, help="random seed")
    parser.add_argument(
        "--print-freq", type=int, default=PRINT_FREQ, help="print frequency"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.07,
        help="InfoNCE temperature for Sample4Geo-style retrieval",
    )
    parser.add_argument(
        "--bbox-loss-weight",
        type=float,
        default=1.0,
        help="Weight for the auxiliary grounding bbox loss",
    )
    parser.add_argument(
        "--heatmap-sigma",
        type=float,
        default=1.5,
        help="Gaussian target sigma in feature cells for the grounding heatmap",
    )
    parser.add_argument(
        "--grad-clip-norm",
        type=float,
        default=GRAD_CLIP_NORM,
        help="Gradient clipping norm; set <=0 to disable",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="/media/data1/feihong/ckpt/ground_sample",
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
    parser.add_argument(
        "--pretrained-checkpoint",
        type=str,
        default="/media/data1/feihong/ckpt/Sample4Geo.pth",
        help="Official Sample4Geo checkpoint used to initialize query/reference ConvNeXt backbones during training",
    )
    parser.add_argument(
        "--no-pretrained-checkpoint",
        dest="pretrained_checkpoint",
        action="store_const",
        const=None,
        help="Disable Sample4Geo checkpoint initialization",
    )
    parser.add_argument(
        "--timm-pretrained",
        action="store_true",
        help="Also ask timm to load its ImageNet ConvNeXt weights before any Sample4Geo checkpoint",
    )
    args = parser.parse_args()
    args.img_size = (int(args.img_size[0]), int(args.img_size[1]))

    os.makedirs(args.checkpoint, exist_ok=True)

    if args.eval_only:
        eval(args)
    else:
        main(args)
