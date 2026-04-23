#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@file train_wild.py
@desc Training TROGeoLite on GroundSmgeoDataset with heading prediction.
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
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from PIL import Image, ImageDraw
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import argparse

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from ground_cvos import TROGeoLite
from bbox.yolo_utils import yolo_loss, build_target
from grounding.model.loss import adjust_learning_rate
from utils.utils import AverageMeter, eval_iou_acc

from dataset import ShiftedSatelliteDroneDataset

IMG_SIZE = (768, 432)  # (width, height)
BATCH_SIZE = 8
ANCHORS = "37,41, 78,84, 96,215, 129,129, 194,82, 198,179, 246,280, 395,342, 550,573"

NUM_EPOCHS = 8
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
PRINT_FREQ = 50
HEADING_LOSS_WEIGHT = 8.0

UNIV_IMAGE_FOLDER = "/data/feihong/image_1024"
HEADING_FOLDER = "/data/feihong/range_250"
UNIV_BBOX_FILE = "/data/feihong/univerisity_dev/runs/test.json"
UNIV_TRAIN_FILE = "/data/feihong/ckpt/train.txt"
UNIV_TEST_FILE = "/data/feihong/ckpt/test.txt"
UNIV_CROP_SIZE = (640, 640)
UNIV_DRONE_SIZE = (256, 256)
UNIV_SAT_SIZE = IMG_SIZE
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CVOGL_TRANSFORM = None

HEADING_TO_TARGET = {
    0: [0.0, 0.0],
    90: [0.0, 1.0],
    180: [1.0, 0.0],
    270: [1.0, 1.0],
}


class TransformProcessorWrapper:
    def __init__(self):
        self.to_tensor = torch.nn.Identity()
        self.size = {"height": IMG_SIZE[1], "width": IMG_SIZE[0]}

    def __call__(self, images, return_tensors="pt"):
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


def angle_to_heading_target(angle_tensor: torch.Tensor) -> torch.Tensor:
    # Quantize arbitrary angles to nearest heading target used by this model.
    angles = torch.remainder(angle_tensor.float(), 360.0)
    heading_classes = torch.round(angles / 90.0).remainder(4).long()
    mapping = torch.tensor(
        [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]],
        device=angle_tensor.device,
        dtype=torch.float32,
    )
    return mapping[heading_classes]


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


def train_epoch(
    loader,
    model,
    optimizer,
    epoch,
    anchors_full,
    img_size,
    heading_loss_weight,
    print_freq=PRINT_FREQ,
):
    """Train for one epoch."""
    model.train()

    batch_time = AverageMeter()
    losses = AverageMeter()
    geo_losses = AverageMeter()
    cls_losses = AverageMeter()
    heading_losses = AverageMeter()

    end = time.time()
    for batch_idx, batch in enumerate(loader):
        query_imgs = batch["target_pixel_values"].to(DEVICE, non_blocking=True)
        rs_imgs = batch["search_pixel_values"].to(DEVICE, non_blocking=True)
        ori_gt_bbox = batch["bbox"].to(DEVICE)
        heading_target = angle_to_heading_target(batch["angle"].to(DEVICE))

        pred_anchor, _, pred_heading = model(query_imgs, rs_imgs)

        pred_anchor = pred_anchor.view(
            pred_anchor.shape[0], 9, 5, pred_anchor.shape[2], pred_anchor.shape[3]
        )

        image_wh = (rs_imgs.shape[-1], rs_imgs.shape[-2])
        grid_wh = (pred_anchor.shape[4], pred_anchor.shape[3])

        new_gt_bbox, best_anchor_gi_gj = build_target(
            ori_gt_bbox, anchors_full, image_wh, grid_wh
        )

        loss_geo, loss_cls = yolo_loss(
            pred_anchor, new_gt_bbox, anchors_full, best_anchor_gi_gj, image_wh
        )
        bbox_loss = loss_geo + loss_cls
        heading_loss = F.mse_loss(pred_heading, heading_target)
        loss = bbox_loss + heading_loss * heading_loss_weight

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.update(loss.item(), query_imgs.shape[0])
        geo_losses.update(loss_geo.item(), query_imgs.shape[0])
        cls_losses.update(loss_cls.item(), query_imgs.shape[0])
        heading_losses.update(heading_loss.item(), query_imgs.shape[0])

        batch_time.update(time.time() - end)
        end = time.time()

        if batch_idx % print_freq == 0:
            print(
                f"Epoch: [{epoch}][{batch_idx}/{len(loader)}]\t"
                f"Time: {batch_time.val:.3f}\t"
                f"Loss: {losses.val:.4f} ({losses.avg:.4f})\t"
                f"Geo: {geo_losses.val:.4f} ({geo_losses.avg:.4f})\t"
                f"Cls: {cls_losses.val:.4f} ({cls_losses.avg:.4f})\t"
                f"Heading: {heading_losses.val:.4f} ({heading_losses.avg:.4f})"
            )

    return losses.avg, geo_losses.avg, cls_losses.avg, heading_losses.avg


def validate(loader, model, anchors_full, img_size):
    """Validate model."""
    model.eval()

    accu50_meter = AverageMeter()
    accu25_meter = AverageMeter()
    iou_meter = AverageMeter()
    heading_meter = AverageMeter()

    for batch in tqdm(loader, desc="Validating"):
        query_imgs = batch["target_pixel_values"].to(DEVICE, non_blocking=True)
        rs_imgs = batch["search_pixel_values"].to(DEVICE, non_blocking=True)
        ori_gt_bbox = batch["bbox"].to(DEVICE)
        heading_target = angle_to_heading_target(batch["angle"].to(DEVICE))

        with torch.no_grad():
            pred_anchor, _, pred_heading = model(query_imgs, rs_imgs)

        pred_anchor = pred_anchor.view(
            pred_anchor.shape[0], 9, 5, pred_anchor.shape[2], pred_anchor.shape[3]
        )

        image_wh = (rs_imgs.shape[-1], rs_imgs.shape[-2])
        grid_wh = (pred_anchor.shape[4], pred_anchor.shape[3])

        _, best_anchor_gi_gj = build_target(
            ori_gt_bbox, anchors_full, image_wh, grid_wh
        )

        accu_list, accu_center, iou, _, _, _ = eval_iou_acc(
            pred_anchor,
            ori_gt_bbox,
            anchors_full,
            best_anchor_gi_gj[:, 1],
            best_anchor_gi_gj[:, 2],
            image_wh,
            iou_threshold_list=[0.5, 0.25],
        )

        heading_loss = F.mse_loss(pred_heading, heading_target)

        accu50_meter.update(accu_list[0].item(), query_imgs.shape[0])
        accu25_meter.update(accu_list[1].item(), query_imgs.shape[0])
        iou_meter.update(iou.item(), query_imgs.shape[0])
        heading_meter.update(heading_loss.item(), query_imgs.shape[0])

    return accu50_meter.avg, accu25_meter.avg, iou_meter.avg, heading_meter.avg


def main(args):
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    exp_name = args.savename if args.savename else "ground_smgeo_exp"
    writer = SummaryWriter(f"runs/{exp_name}")

    print("Creating datasets from shared data source...")
    processor = TransformProcessorWrapper()
    tokenizer = DummyTokenizer()
    train_dataset = ShiftedSatelliteDroneDataset(
        processor=processor,
        processor_sat=processor,
        tokenizer=tokenizer,
        split="train",
    )
    val_dataset = ShiftedSatelliteDroneDataset(
        processor=processor,
        processor_sat=processor,
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
    model = TROGeoLite(emb_size=768).to(DEVICE)

    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)

    anchors_full = np.array([float(x) for x in ANCHORS.split(",")])
    anchors_full = anchors_full.reshape(-1, 2)[::-1].copy()
    anchors_full = torch.tensor(anchors_full, dtype=torch.float32).to(DEVICE)

    heading_loss_weight = args.heading_loss_weight
    print(f"Starting training for {args.max_epoch} epochs...")
    for epoch in range(args.max_epoch):
        adjust_learning_rate(args, optimizer, epoch)

        train_loss, train_geo, train_cls, train_heading = train_epoch(
            train_loader,
            model,
            optimizer,
            epoch,
            anchors_full,
            args.img_size,
            heading_loss_weight,
            args.print_freq,
        )

        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/train_geo", train_geo, epoch)
        writer.add_scalar("Loss/train_cls", train_cls, epoch)
        writer.add_scalar("Loss/train_heading", train_heading, epoch)

        print(
            f"Epoch {epoch + 1}/{args.max_epoch}:\t"
            f"Train Loss: {train_loss:.4f} (Geo: {train_geo:.4f}, Cls: {train_cls:.4f}, Heading: {train_heading:.4f})"
        )
        torch.save(model.state_dict(), f"{args.checkpoint}/last.pth")

    print("\nTraining complete. Saved checkpoint to last.pth")
    writer.close()


def eval(args):
    print("Evaluating checkpoint on test split...")
    processor = TransformProcessorWrapper()
    tokenizer = DummyTokenizer()
    test_dataset = ShiftedSatelliteDroneDataset(
        processor=processor,
        processor_sat=processor,
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

    model = TROGeoLite(emb_size=768).to(DEVICE)
    checkpoint_path = args.checkpoint_path or f"{args.checkpoint}/last.pth"
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"), strict=False)

    anchors_full = np.array([float(x) for x in ANCHORS.split(",")])
    anchors_full = anchors_full.reshape(-1, 2)[::-1].copy()
    anchors_full = torch.tensor(anchors_full, dtype=torch.float32).to(DEVICE)

    accu50, accu25, val_iou, val_heading = validate(
        test_loader, model, anchors_full, args.img_size
    )
    print(
        f"Eval Accu50: {accu50:.4f}, Accu25: {accu25:.4f}, IoU: {val_iou:.4f}, Heading: {val_heading:.4f}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="TROGeoLite Training on GroundSmgeoDataset with Heading"
    )
    parser.add_argument("--gpu", default="1", help="GPU id (e.g., '1' or '1,2')")
    parser.add_argument(
        "--num-workers", type=int, default=8, help="num workers for data loading"
    )
    parser.add_argument(
        "--max-epoch", type=int, default=NUM_EPOCHS, help="training epoch"
    )
    parser.add_argument("--lr", type=float, default=LEARNING_RATE, help="learning rate")
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
        "--heading-loss-weight",
        type=float,
        default=HEADING_LOSS_WEIGHT,
        help="heading loss weight",
    )
    parser.add_argument(
        "--savename", type=str, default="ground_smgeo", help="Name head for saved model"
    )
    parser.add_argument("--seed", type=int, default=2024, help="random seed")
    parser.add_argument(
        "--print-freq", type=int, default=PRINT_FREQ, help="print frequency"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="/data/feihong/ckpt/ground_cvos",
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
