#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@file train_ocg.py
@desc Train an OCGNet-style cross-view grounding model on ShiftedSatelliteDroneDataset.

The implementation follows the main ideas from OCGNet:
- query image is encoded together with a click/prompt heatmap;
- prompt knowledge modulates query features;
- query and satellite/reference features are fused with cross attention;
- a YOLO-style anchor head predicts the target bbox on the satellite image.
"""

import argparse
import os
import random
import sys
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import models
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from bbox.yolo_utils import bbox_iou, build_target, eval_iou_acc, yolo_loss
from dataset import ShiftedSatelliteDroneDataset
from grounding.utils.utils import AverageMeter


# --- Configuration ---
IMG_SIZE = (768, 432)  # (width, height)
DRONE_SIZE = (256, 256)  # (width, height)
BATCH_SIZE = 16
NUM_EPOCHS = 8
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
PRINT_FREQ = 50
CLICK_SIGMA = 0.16
DEVICE = "cuda:1" if torch.cuda.is_available() else "cpu"
ANCHORS = "37,41, 78,84, 96,215, 129,129, 194,82, 198,179, 246,280, 395,342, 550,573"


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


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


def _resnet18_backbone(pretrained: bool = False):
    weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    return models.resnet18(weights=weights)


class QueryEncoder(nn.Module):
    """ResNet18 query encoder with an extra click heatmap input channel."""

    def __init__(self, pretrained: bool = False):
        super().__init__()
        resnet = _resnet18_backbone(pretrained=pretrained)
        self.conv1 = nn.Conv2d(
            4,
            64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        with torch.no_grad():
            self.conv1.weight[:, :3].copy_(resnet.conv1.weight)
            self.conv1.weight[:, 3:4].copy_(resnet.conv1.weight.mean(dim=1, keepdim=True))

        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

    def forward(self, query_imgs, click_maps):
        x = torch.cat([query_imgs, click_maps], dim=1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


class ReferenceEncoder(nn.Module):
    """ResNet18 satellite/reference encoder."""

    def __init__(self, pretrained: bool = False):
        super().__init__()
        resnet = _resnet18_backbone(pretrained=pretrained)
        self.stem = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

    def forward(self, reference_imgs):
        x = self.stem(reference_imgs)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


class GaussianKnowledgeTransfer(nn.Module):
    """Convert the click prompt into a channel-wise spatial gate for query features."""

    def __init__(self, channels: int):
        super().__init__()
        self.gate = nn.Sequential(
            ConvBNReLU(1, channels // 4, kernel_size=3, padding=1),
            nn.Conv2d(channels // 4, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, query_feats, click_maps):
        click_maps = F.interpolate(
            click_maps,
            size=query_feats.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        gate = self.gate(click_maps)
        return query_feats * (1.0 + gate)


class CrossViewAttention(nn.Module):
    def __init__(self, channels: int, num_heads: int = 8):
        super().__init__()
        self.query_norm = nn.LayerNorm(channels)
        self.ref_norm = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=num_heads,
            batch_first=True,
        )
        self.ffn = nn.Sequential(
            nn.LayerNorm(channels),
            nn.Linear(channels, channels * 4),
            nn.GELU(),
            nn.Linear(channels * 4, channels),
        )

    def forward(self, query_feats, reference_feats):
        B, C, Hq, Wq = query_feats.shape
        query_tokens = query_feats.flatten(2).transpose(1, 2)
        ref_tokens = reference_feats.flatten(2).transpose(1, 2)

        attended, _ = self.attn(
            self.query_norm(query_tokens),
            self.ref_norm(ref_tokens),
            self.ref_norm(ref_tokens),
            need_weights=False,
        )
        query_tokens = query_tokens + attended
        query_tokens = query_tokens + self.ffn(query_tokens)
        return query_tokens.transpose(1, 2).reshape(B, C, Hq, Wq)


class CrossViewFusion(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.query_proj = nn.Conv2d(channels, channels, kernel_size=1)
        self.reference_proj = nn.Conv2d(channels, channels, kernel_size=1)
        self.fuse = nn.Sequential(
            ConvBNReLU(channels * 2, channels, kernel_size=3, padding=1),
            ConvBNReLU(channels, channels // 2, kernel_size=3, padding=1),
        )

    def forward(self, query_feats, reference_feats):
        query_context = self.query_proj(query_feats)
        query_context = F.adaptive_avg_pool2d(query_context, output_size=1)
        query_context = query_context.expand(-1, -1, reference_feats.shape[-2], reference_feats.shape[-1])
        reference_feats = self.reference_proj(reference_feats)
        return self.fuse(torch.cat([reference_feats, query_context], dim=1))


class OCGNetLite(nn.Module):
    """OCGNet-style network adapted to this repository's YOLO bbox head contract."""

    def __init__(
        self,
        channels: int = 512,
        num_heads: int = 8,
        pretrained_backbone: bool = False,
    ):
        super().__init__()
        self.query_encoder = QueryEncoder(pretrained=pretrained_backbone)
        self.reference_encoder = ReferenceEncoder(pretrained=pretrained_backbone)
        self.gkt = GaussianKnowledgeTransfer(channels)
        self.cross_attention = CrossViewAttention(channels, num_heads=num_heads)
        self.fusion = CrossViewFusion(channels)
        self.bbox_head = nn.Sequential(
            ConvBNReLU(channels // 2, channels // 2, kernel_size=3, padding=1),
            nn.Conv2d(channels // 2, 9 * 5, kernel_size=1),
        )

    def forward(self, query_imgs, reference_imgs, click_maps=None):
        if click_maps is None:
            click_maps = build_center_click_maps(query_imgs)

        query_feats = self.query_encoder(query_imgs, click_maps)
        query_feats = self.gkt(query_feats, click_maps)
        reference_feats = self.reference_encoder(reference_imgs)
        query_feats = self.cross_attention(query_feats, reference_feats)
        fused_feats = self.fusion(query_feats, reference_feats)
        pred_anchor = self.bbox_head(fused_feats)
        return pred_anchor, click_maps

    def bbox_forward(self, query_imgs, reference_imgs):
        return self.forward(query_imgs, reference_imgs)[0]


def build_center_click_maps(query_imgs: torch.Tensor, sigma: float = CLICK_SIGMA) -> torch.Tensor:
    B, _, H, W = query_imgs.shape
    y_coords = torch.linspace(-1.0, 1.0, H, device=query_imgs.device, dtype=query_imgs.dtype)
    x_coords = torch.linspace(-1.0, 1.0, W, device=query_imgs.device, dtype=query_imgs.dtype)
    grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing="ij")
    heatmap = torch.exp(-0.5 * (grid_x.square() + grid_y.square()) / max(sigma * sigma, 1e-6))
    return heatmap.view(1, 1, H, W).expand(B, -1, -1, -1).contiguous()


def parse_anchors(device: str) -> torch.Tensor:
    anchors_full = np.array([float(x) for x in ANCHORS.split(",")], dtype=np.float32)
    anchors_full = anchors_full.reshape(-1, 2)[::-1].copy()
    return torch.tensor(anchors_full, dtype=torch.float32, device=device)


def adjust_learning_rate(args, optimizer, epoch):
    lr = args.lr * (0.1 ** (epoch // 10))
    print(("lr", lr))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def train_epoch(
    loader,
    model,
    optimizer,
    epoch,
    anchors_full,
    print_freq=PRINT_FREQ,
):
    model.train()

    batch_time = AverageMeter()
    losses = AverageMeter()
    geo_losses = AverageMeter()
    cls_losses = AverageMeter()

    end = time.time()
    for batch_idx, batch in enumerate(loader):
        query_imgs = batch["target_pixel_values"].to(DEVICE, non_blocking=True)
        reference_imgs = batch["search_pixel_values"].to(DEVICE, non_blocking=True)
        gt_bbox = batch["bbox"].to(DEVICE, non_blocking=True)

        pred_anchor, _ = model(query_imgs, reference_imgs)
        pred_anchor = pred_anchor.view(
            pred_anchor.shape[0],
            9,
            5,
            pred_anchor.shape[2],
            pred_anchor.shape[3],
        )

        image_wh = (reference_imgs.shape[-1], reference_imgs.shape[-2])
        grid_wh = (pred_anchor.shape[4], pred_anchor.shape[3])
        new_gt_bbox, best_anchor_gi_gj = build_target(
            gt_bbox,
            anchors_full,
            image_wh,
            grid_wh,
        )

        loss_geo, loss_cls = yolo_loss(
            pred_anchor,
            new_gt_bbox,
            anchors_full,
            best_anchor_gi_gj,
            image_wh,
        )
        loss = loss_geo + loss_cls

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        losses.update(loss.item(), query_imgs.shape[0])
        geo_losses.update(loss_geo.item(), query_imgs.shape[0])
        cls_losses.update(loss_cls.item(), query_imgs.shape[0])
        batch_time.update(time.time() - end)
        end = time.time()

        if batch_idx % print_freq == 0:
            print(
                f"Epoch: [{epoch}][{batch_idx}/{len(loader)}]\t"
                f"Time: {batch_time.val:.3f}\t"
                f"Loss: {losses.val:.4f} ({losses.avg:.4f})\t"
                f"Geo: {geo_losses.val:.4f} ({geo_losses.avg:.4f})\t"
                f"Cls: {cls_losses.val:.4f} ({cls_losses.avg:.4f})"
            )

    return losses.avg, geo_losses.avg, cls_losses.avg


def validate(loader, model, anchors_full):
    model.eval()
    iou_values: List[float] = []
    center_distances: List[float] = []

    for batch in tqdm(loader, desc="Validating"):
        query_imgs = batch["target_pixel_values"].to(DEVICE, non_blocking=True)
        reference_imgs = batch["search_pixel_values"].to(DEVICE, non_blocking=True)
        gt_bbox = batch["bbox"].to(DEVICE, non_blocking=True)

        with torch.no_grad():
            pred_anchor, _ = model(query_imgs, reference_imgs)

        pred_anchor = pred_anchor.view(
            pred_anchor.shape[0],
            9,
            5,
            pred_anchor.shape[2],
            pred_anchor.shape[3],
        )

        image_wh = (reference_imgs.shape[-1], reference_imgs.shape[-2])
        grid_wh = (pred_anchor.shape[4], pred_anchor.shape[3])
        _, best_anchor_gi_gj = build_target(gt_bbox, anchors_full, image_wh, grid_wh)

        _, _, _, _, pred_bbox_xyxy, target_bbox_xyxy = eval_iou_acc(
            pred_anchor,
            gt_bbox,
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

    print("Creating OCGNetLite model...")
    model = OCGNetLite(
        channels=512,
        num_heads=args.num_heads,
        pretrained_backbone=args.pretrained_backbone,
    ).to(DEVICE)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)
    anchors_full = parse_anchors(DEVICE)

    print(f"Starting training for {args.max_epoch} epochs...")
    for epoch in range(args.max_epoch):
        adjust_learning_rate(args, optimizer, epoch)
        train_loss, train_geo, train_cls = train_epoch(
            train_loader,
            model,
            optimizer,
            epoch,
            anchors_full,
            args.print_freq,
        )

        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/train_geo", train_geo, epoch)
        writer.add_scalar("Loss/train_cls", train_cls, epoch)

        print(
            f"Epoch {epoch + 1}/{args.max_epoch}:\t"
            f"Train Loss: {train_loss:.4f} (Geo: {train_geo:.4f}, Cls: {train_cls:.4f})"
        )

        torch.save(model.state_dict(), os.path.join(args.checkpoint, "last.pth"))

    print("\nTraining complete. Saved checkpoint to last.pth")
    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OCGNet-style training on ShiftedSatelliteDroneDataset")
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
    parser.add_argument("--savename", type=str, default="ground_ocg", help="TensorBoard run name")
    parser.add_argument("--seed", type=int, default=2024, help="random seed")
    parser.add_argument("--print-freq", type=int, default=PRINT_FREQ, help="print frequency")
    parser.add_argument("--num-heads", type=int, default=8, help="cross-attention heads")
    parser.add_argument(
        "--pretrained-backbone",
        action="store_true",
        help="Use torchvision ImageNet ResNet18 weights for query/reference backbones",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="/media/data1/feihong/ckpt/ground_ocg",
        help="Path to save model checkpoints",
    )
    args = parser.parse_args()
    args.img_size = (int(args.img_size[0]), int(args.img_size[1]))

    main(args)
