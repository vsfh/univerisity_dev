#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@file train_lpn.py
@desc Training LPNGeoLite on GroundSmgeoDataset with heading prediction.
"""

import time
import random
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from PIL import Image, ImageDraw
import os
from typing import Dict, List, Tuple, Optional
import argparse
from ground_cvos import LPNGeoLite
from model.loss import yolo_loss, build_target, adjust_learning_rate
from utils.utils import AverageMeter, eval_iou_acc

IMG_SIZE = 640
BATCH_SIZE = 4
ANCHORS = "37,41, 78,84, 96,215, 129,129, 194,82, 198,179, 246,280, 395,342, 550,573"

NUM_EPOCHS = 25
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
UNIV_SAT_SIZE = (640, 640)
DEVICE = "cuda:2" if torch.cuda.is_available() else "cpu"

CVOGL_TRANSFORM = None

HEADING_TO_TARGET = {
    0: [0.0, 0.0],
    90: [0.0, 1.0],
    180: [1.0, 0.0],
    270: [1.0, 1.0],
}


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


class HeadingGeoDataset(Dataset):
    """GroundSmgeoDataset with heading prediction."""

    def __init__(self, image_pairs, bbox_dict, mode="train", transform=None):
        self.image_paths = image_pairs
        self.bbox_dict = bbox_dict
        self.mode = mode
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        query_path, search_path = self.image_paths[idx]

        if self.mode == "train":
            heading = random.choice([0, 90, 180, 270])
            query_path = query_path.replace("heading0", f"heading{heading}")
        else:
            heading = 0

        name = query_path.split("/")[-2]

        try:
            query_image = Image.open(query_path).convert("RGB")
            search_image = Image.open(search_path).convert("RGB")
        except FileNotFoundError:
            return self.__getitem__((idx + 1) % len(self))

        query_image = resize_drone_image(query_image)

        base_name = name
        bbox_key = base_name
        bbox = self.bbox_dict.get(bbox_key, None)

        if bbox is None:
            bbox = (1536, 656, 2268, 1374)

        search_image, normalized_bbox = format_satellite_img_bbox(
            search_image, bbox, mode=self.mode, target_size=UNIV_CROP_SIZE
        )

        query_img_np = np.array(query_image)
        rs_img_np = np.array(search_image)

        queryimg = (
            torch.tensor(query_img_np, dtype=torch.float32).permute(2, 0, 1) / 255.0
        )
        rsimg = torch.tensor(rs_img_np, dtype=torch.float32).permute(2, 0, 1) / 255.0

        heading_target = torch.tensor(HEADING_TO_TARGET[heading], dtype=torch.float32)

        if self.transform:
            from torchvision.transforms import Compose, ToTensor, Normalize

            transform = Compose(
                [
                    ToTensor(),
                ]
            )
            queryimg = transform(query_img_np)
            rsimg = transform(rs_img_np)

        return {
            "query_imgs": queryimg,
            "rs_imgs": rsimg,
            "ori_gt_bbox": torch.tensor(normalized_bbox, dtype=torch.float32),
            "heading": heading_target,
            "query_img_np": query_img_np,
            "rs_img_np": rs_img_np,
            "query_name": name,
            "rs_name": f"{name}.png",
        }


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
        query_imgs = batch["query_imgs"].to(DEVICE)
        rs_imgs = batch["rs_imgs"].to(DEVICE)
        ori_gt_bbox = batch["ori_gt_bbox"].to(DEVICE)
        heading_target = batch["heading"].to(DEVICE)

        pred_anchor, _, pred_heading = model(query_imgs, rs_imgs)

        pred_anchor = pred_anchor.view(
            pred_anchor.shape[0], 9, 5, pred_anchor.shape[2], pred_anchor.shape[3]
        )

        new_gt_bbox, best_anchor_gi_gj = build_target(
            ori_gt_bbox, anchors_full, img_size, pred_anchor.shape[3]
        )

        loss_geo, loss_cls = yolo_loss(
            pred_anchor, new_gt_bbox, anchors_full, best_anchor_gi_gj, img_size
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
        query_imgs = batch["query_imgs"].to(DEVICE)
        rs_imgs = batch["rs_imgs"].to(DEVICE)
        ori_gt_bbox = batch["ori_gt_bbox"].to(DEVICE)
        heading_target = batch["heading"].to(DEVICE)

        with torch.no_grad():
            pred_anchor, _, pred_heading = model(query_imgs, rs_imgs)

        pred_anchor = pred_anchor.view(
            pred_anchor.shape[0], 9, 5, pred_anchor.shape[2], pred_anchor.shape[3]
        )

        _, best_anchor_gi_gj = build_target(
            ori_gt_bbox, anchors_full, img_size, pred_anchor.shape[3]
        )

        accu_list, accu_center, iou, _, _, _ = eval_iou_acc(
            pred_anchor,
            ori_gt_bbox,
            anchors_full,
            best_anchor_gi_gj[:, 1],
            best_anchor_gi_gj[:, 2],
            img_size,
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

    exp_name = args.savename if args.savename else "ground_lpn_exp"
    writer = SummaryWriter(f"runs/{exp_name}")

    print("Loading bbox annotations...")
    test_bbox_file = "/data/feihong/univerisity_dev/runs/test.json"
    train_bbox_file = "/data/feihong/univerisity_dev/runs/train.json"

    bbox_dict = {}
    for f in [test_bbox_file, train_bbox_file]:
        with open(f, "r") as file:
            data = json.load(file)
            bbox_dict.update(data)

    print(f"Loaded {len(bbox_dict)} bbox annotations from train.json and test.json")

    print("Loading image pairs...")
    train_pairs = []
    with open(UNIV_TRAIN_FILE, "r") as f:
        for line in f:
            query_path = line.strip()
            name = query_path.split("/")[-2]
            heading_path = f"{HEADING_FOLDER}/{name}_range250_heading0.png"
            search_path = f"{UNIV_IMAGE_FOLDER}/{name}.png"
            if os.path.exists(heading_path) and os.path.exists(search_path):
                train_pairs.append((heading_path, search_path))

    val_pairs = []
    with open(UNIV_TEST_FILE, "r") as f:
        for line in f:
            query_path = line.strip()
            name = query_path.split("/")[-2]
            heading_path = f"{HEADING_FOLDER}/{name}_range250_heading0.png"
            search_path = f"{UNIV_IMAGE_FOLDER}/{name}.png"
            if os.path.exists(heading_path) and os.path.exists(search_path):
                val_pairs.append((heading_path, search_path))

    print(f"Found {len(train_pairs)} training pairs, {len(val_pairs)} validation pairs")

    print("Creating datasets...")
    train_dataset = HeadingGeoDataset(
        image_pairs=train_pairs,
        bbox_dict=bbox_dict,
        mode="train",
        transform=CVOGL_TRANSFORM,
    )
    val_dataset = HeadingGeoDataset(
        image_pairs=val_pairs,
        bbox_dict=bbox_dict,
        mode="test",
        transform=CVOGL_TRANSFORM,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        drop_last=False,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
        num_workers=args.num_workers,
    )

    print("Creating model...")
    model = LPNGeoLite(emb_size=2048).to(DEVICE)

    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)

    anchors_full = np.array([float(x) for x in ANCHORS.split(",")])
    anchors_full = anchors_full.reshape(-1, 2)[::-1].copy()
    anchors_full = torch.tensor(anchors_full, dtype=torch.float32).to(DEVICE)

    best_iou = -float("Inf")
    update = 0
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

        accu50, accu25, val_iou, val_heading = validate(
            val_loader, model, anchors_full, args.img_size
        )

        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/train_geo", train_geo, epoch)
        writer.add_scalar("Loss/train_cls", train_cls, epoch)
        writer.add_scalar("Loss/train_heading", train_heading, epoch)
        writer.add_scalar("Metrics/val_accu50", accu50, epoch)
        writer.add_scalar("Metrics/val_accu25", accu25, epoch)
        writer.add_scalar("Metrics/val_iou", val_iou, epoch)
        writer.add_scalar("Metrics/val_heading", val_heading, epoch)

        print(
            f"Epoch {epoch + 1}/{args.max_epoch}:\t"
            f"Train Loss: {train_loss:.4f} (Geo: {train_geo:.4f}, Cls: {train_cls:.4f}, Heading: {train_heading:.4f})\t"
            f"Val Accu50: {accu50:.4f}\t"
            f"Val Accu25: {accu25:.4f}\t"
            f"Val IoU: {val_iou:.4f}\t"
            f"Val Heading: {val_heading:.4f}"
        )

        is_best = val_iou > best_iou
        if val_iou > best_iou:
            best_iou = val_iou
            torch.save(model.state_dict(), f"{args.checkpoint}/best.pth")
            update = 0
        else:
            update += 1
        if update > 5:
            print("No improvement for 6 epochs, stopping early.")

    print(f"\nTraining complete. Best Val IoU: {best_iou:.4f}")
    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="LPNGeoLite Training on GroundSmgeoDataset with Heading"
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
    parser.add_argument("--img-size", type=int, default=IMG_SIZE, help="image size")
    parser.add_argument(
        "--heading-loss-weight",
        type=float,
        default=HEADING_LOSS_WEIGHT,
        help="heading loss weight",
    )
    parser.add_argument(
        "--savename", type=str, default="ground_lpn", help="Name head for saved model"
    )
    parser.add_argument("--seed", type=int, default=2024, help="random seed")
    parser.add_argument(
        "--print-freq", type=int, default=PRINT_FREQ, help="print frequency"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="/data/feihong/ckpt/ground_lpn",
        help="Path to save model checkpoints",
    )
    args = parser.parse_args()

    os.makedirs(args.checkpoint, exist_ok=True)

    main(args)
