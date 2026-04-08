#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@file ground_cvos.py
@desc CVOS bbox evaluation using TROGeoLite model on CVOGL dataset.
      Simplified TROGeo without click point input.
      Includes training, evaluation, and visualization.
"""

import time
import random
import json
import torch
import torch.nn as nn
import cv2
import numpy as np
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torchvision.transforms import Compose, ToTensor, Normalize
from torch.utils.tensorboard import SummaryWriter
from PIL import Image, ImageDraw
from tqdm import tqdm
import os
from typing import Dict, List, Tuple, Optional
import argparse
from einops import rearrange
import timm
import torchvision.models as torchvision_models
from torch.autograd import Variable
from model.TROGeo import SwinTransformer
from model.attention import SpatialTransformer
from model.loss import yolo_loss, build_target, adjust_learning_rate
from utils.utils import AverageMeter, eval_iou_acc
from utils.checkpoint import save_checkpoint
from model.darknet import *
import sys

sys.path.insert(0, "/data/feihong/univerisity_dev")
from out_model import ResidualBlock

DATA_ROOT = "/data/feihong/CVOGL"
DATA_NAME = "CVOGL_DroneAerial"
IMG_SIZE = 1024
BATCH_SIZE = 4
ANCHORS = "37,41, 78,84, 96,215, 129,129, 194,82, 198,179, 246,280, 395,342, 550,573"

NUM_EPOCHS = 25
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
PRINT_FREQ = 50
SAVE_DIR = "/data/feihong/ckpt/ground_cvos"

CVOGL_TRANSFORM = Compose(
    [
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
    )


class MyResnet(nn.Module):
    def __init__(self):
        super(MyResnet, self).__init__()
        self.base_model = torchvision_models.resnet18(pretrained=True)
        self.base_model.avgpool = nn.Sequential()
        self.base_model.fc = nn.Sequential()

    def forward(self, x):
        x = self.base_model.conv1(x)
        x = self.base_model.bn1(x)
        x = self.base_model.relu(x)
        x = self.base_model.maxpool(x)
        x = self.base_model.layer1(x)
        x = self.base_model.layer2(x)
        x = self.base_model.layer3(x)
        x = self.base_model.layer4(x)
        return x


class CrossViewFusionModule(nn.Module):
    def __init__(self):
        super(CrossViewFusionModule, self).__init__()

    def forward(self, global_query, value):
        global_query = F.normalize(global_query, p=2, dim=-1)
        value = F.normalize(value, p=2, dim=1)

        B, D, W, H = value.shape
        new_value = value.permute(0, 2, 3, 1).view(B, W * H, D)
        score = torch.bmm(global_query.view(B, 1, D), new_value.transpose(1, 2))
        score = score.view(B, W * H)

        with torch.no_grad():
            score_np = score.clone().detach().cpu().numpy()
            max_score, min_score = score_np.max(axis=1), score_np.min(axis=1)

        device = value.device
        attn = Variable(torch.zeros(B, H * W, device=device))
        for ii in range(B):
            attn[ii, :] = (score[ii] - min_score[ii]) / (
                max_score[ii] - min_score[ii] + 1e-8
            )

        attn = attn.view(B, 1, W, H)
        context = attn * value
        return context, attn


class TROGeoLite(nn.Module):
    """Simplified TROGeo without click point input for direct bbox regression."""

    def __init__(self, emb_size=768):
        super(TROGeoLite, self).__init__()

        base_model = SwinTransformer()
        self.query_model = base_model
        self.reference_model = base_model

        self.cross_attention = SpatialTransformer(
            in_channels=emb_size, n_heads=12, d_head=64, depth=1, context_dim=emb_size
        )

        self.fcn_out = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=emb_size,
                out_channels=emb_size // 2,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(emb_size // 2, 9 * 5, kernel_size=1),
        )

        self.coodrs_out = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=emb_size,
                out_channels=emb_size // 2,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(emb_size // 2, 1, kernel_size=1),
        )

        self.bbox_heading = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=emb_size,
                out_channels=emb_size // 2,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.ReLU(inplace=True),
            ResidualBlock(emb_size // 2),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(emb_size // 2, 2),
        )

    def forward(self, query_imgs, reference_imgs):
        query_fvisu = self.query_model(query_imgs)
        reference_fvisu = self.reference_model(reference_imgs)

        context = rearrange(query_fvisu, "b c h w -> b (h w) c").contiguous()
        fused_features = self.cross_attention(x=reference_fvisu, context=context)

        outbox = self.fcn_out(fused_features)
        coodrs = self.coodrs_out(fused_features)
        pred_heading = torch.sigmoid(self.bbox_heading(fused_features))

        return outbox, coodrs, pred_heading

    def bbox_forward(self, query_imgs, reference_imgs):
        query_fvisu = self.query_model(query_imgs)
        reference_fvisu = self.reference_model(reference_imgs)

        context = rearrange(query_fvisu, "b c h w -> b (h w) c").contiguous()
        fused_features = self.cross_attention(x=reference_fvisu, context=context)

        outbox = self.fcn_out(fused_features)
        coodrs = self.coodrs_out(fused_features)
        pred_heading = torch.sigmoid(self.bbox_heading(fused_features))

        return outbox, coodrs, pred_heading


class SampleGeoLite(nn.Module):
    """Simplified TROGeo without click point input for direct bbox regression."""

    def __init__(self, emb_size=1024):
        super(SampleGeoLite, self).__init__()

        model_name = "convnext_base.fb_in22k_ft_in1k_384"
        base_model = timm.create_model(model_name, pretrained=True, num_classes=0)
        self.query_model = base_model
        self.reference_model = base_model

        self.cross_attention = SpatialTransformer(
            in_channels=emb_size, n_heads=12, d_head=64, depth=1, context_dim=emb_size
        )

        self.fcn_out = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=emb_size,
                out_channels=emb_size // 2,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(emb_size // 2, 9 * 5, kernel_size=1),
        )

        self.coodrs_out = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=emb_size,
                out_channels=emb_size // 2,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(emb_size // 2, 1, kernel_size=1),
        )

        self.bbox_heading = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=emb_size,
                out_channels=emb_size // 2,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.ReLU(inplace=True),
            ResidualBlock(emb_size // 2),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(emb_size // 2, 2),
        )

    def forward(self, query_imgs, reference_imgs):
        query_fvisu = self.query_model.forward_features(query_imgs)
        reference_fvisu = self.reference_model.forward_features(reference_imgs)

        context = rearrange(query_fvisu, "b c h w -> b (h w) c").contiguous()
        fused_features = self.cross_attention(x=reference_fvisu, context=context)

        outbox = self.fcn_out(fused_features)
        coodrs = self.coodrs_out(fused_features)
        pred_heading = torch.sigmoid(self.bbox_heading(fused_features))

        return outbox, coodrs, pred_heading

    def bbox_forward(self, query_imgs, reference_imgs):
        query_fvisu = self.query_model.forward_features(query_imgs)
        reference_fvisu = self.reference_model.forward_features(reference_imgs)

        context = rearrange(query_fvisu, "b c h w -> b (h w) c").contiguous()
        fused_features = self.cross_attention(x=reference_fvisu, context=context)

        outbox = self.fcn_out(fused_features)
        coodrs = self.coodrs_out(fused_features)
        pred_heading = torch.sigmoid(self.bbox_heading(fused_features))

        return outbox, coodrs, pred_heading


class LPNGeoLite(nn.Module):
    """Simplified TROGeo without click point input for direct bbox regression."""

    def __init__(self, emb_size=2048):
        super(LPNGeoLite, self).__init__()

        model_name = "resnet50"
        base_model = timm.create_model(
            model_name, pretrained=True, features_only=True, out_indices=[4]
        )
        self.query_model = base_model
        self.reference_model = base_model

        self.cross_attention = SpatialTransformer(
            in_channels=emb_size, n_heads=12, d_head=64, depth=1, context_dim=emb_size
        )

        self.fcn_out = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=emb_size,
                out_channels=emb_size // 2,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(emb_size // 2, 9 * 5, kernel_size=1),
        )

        self.coodrs_out = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=emb_size,
                out_channels=emb_size // 2,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(emb_size // 2, 1, kernel_size=1),
        )

        self.bbox_heading = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=emb_size,
                out_channels=emb_size // 2,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.ReLU(inplace=True),
            ResidualBlock(emb_size // 2),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(emb_size // 2, 2),
        )

    def forward(self, query_imgs, reference_imgs):
        query_fvisu = self.query_model(query_imgs)[0]
        reference_fvisu = self.reference_model(reference_imgs)[0]

        context = rearrange(query_fvisu, "b c h w -> b (h w) c").contiguous()
        fused_features = self.cross_attention(x=reference_fvisu, context=context)

        outbox = self.fcn_out(fused_features)
        coodrs = self.coodrs_out(fused_features)
        pred_heading = torch.sigmoid(self.bbox_heading(fused_features))

        return outbox, coodrs, pred_heading

    def bbox_forward(self, query_imgs, reference_imgs):
        query_fvisu = self.query_model(query_imgs)[0]
        reference_fvisu = self.reference_model(reference_imgs)[0]

        context = rearrange(query_fvisu, "b c h w -> b (h w) c").contiguous()
        fused_features = self.cross_attention(x=reference_fvisu, context=context)

        outbox = self.fcn_out(fused_features)
        coodrs = self.coodrs_out(fused_features)
        pred_heading = torch.sigmoid(self.bbox_heading(fused_features))

        return outbox, coodrs, pred_heading


class DetGeoLite(nn.Module):
    """DetGeo based on original DetGeo architecture with ResNet-18 and Darknet."""

    def __init__(
        self,
        emb_size=512,
        config_path="/data/feihong/univerisity_dev/runs/yolov3_rs.cfg",
        weights_path="/data/feihong/univerisity_dev/runs/yolov3.weights",
        use_instnorm=False,
    ):
        super(DetGeoLite, self).__init__()

        self.query_resnet = MyResnet()

        self.reference_darknet = Darknet(config_path=config_path, img_size=640)
        if os.path.exists(weights_path):
            self.reference_darknet.load_weights(weights_path)

        self.combine_clickptns_conv = ConvBatchNormReLU(
            4, 3, 1, 1, 0, 1, leaky=True, instance=use_instnorm
        )
        self.crossview_fusionmodule = CrossViewFusionModule()

        self.query_visudim = 512
        self.reference_visudim = 512

        self.query_mapping_visu = ConvBatchNormReLU(
            self.query_visudim, emb_size, 1, 1, 0, 1, leaky=True, instance=use_instnorm
        )
        self.reference_mapping_visu = ConvBatchNormReLU(
            self.reference_visudim,
            emb_size,
            1,
            1,
            0,
            1,
            leaky=True,
            instance=use_instnorm,
        )

        self.fcn_out = nn.Sequential(
            ConvBatchNormReLU(
                emb_size, emb_size // 2, 1, 1, 0, 1, leaky=True, instance=use_instnorm
            ),
            nn.Conv2d(emb_size // 2, 9 * 5, kernel_size=1),
        )

        self.coodrs_out = nn.Sequential(
            ConvBatchNormReLU(
                emb_size, emb_size // 2, 1, 1, 0, 1, leaky=True, instance=use_instnorm
            ),
            nn.Conv2d(emb_size // 2, 1, kernel_size=1),
        )

    def forward(self, query_imgs, reference_imgs, click_pts=None):
        if click_pts is not None:
            click_pts = click_pts.unsqueeze(1)
            query_imgs = self.combine_clickptns_conv(
                torch.cat((query_imgs, click_pts), dim=1)
            )

        query_fvisu = self.query_resnet(query_imgs)

        reference_raw_fvisu = self.reference_darknet(reference_imgs)
        reference_fvisu = reference_raw_fvisu[1]

        query_fvisu = self.query_mapping_visu(query_fvisu)
        reference_fvisu = self.reference_mapping_visu(reference_fvisu)

        B, D, Hquery, Wquery = query_fvisu.shape
        B, D, Hreference, Wreference = reference_fvisu.shape

        query_gvisu = torch.mean(
            query_fvisu.view(B, D, Hquery * Wquery), dim=2, keepdims=False
        ).view(B, D)
        fused_features, attn_score = self.crossview_fusionmodule(
            query_gvisu, reference_fvisu
        )
        attn_score = attn_score.squeeze(1)

        outbox = self.fcn_out(fused_features)
        coodrs = self.coodrs_out(fused_features)

        return outbox, coodrs

    def bbox_forward(self, query_imgs, reference_imgs, click_pts=None):
        if click_pts is not None:
            click_pts = click_pts.unsqueeze(1)
            query_imgs = self.combine_clickptns_conv(
                torch.cat((query_imgs, click_pts), dim=1)
            )

        query_fvisu = self.query_resnet(query_imgs)

        reference_raw_fvisu = self.reference_darknet(reference_imgs)
        reference_fvisu = reference_raw_fvisu[1]

        query_fvisu = self.query_mapping_visu(query_fvisu)
        reference_fvisu = self.reference_mapping_visu(reference_fvisu)

        B, D, Hquery, Wquery = query_fvisu.shape
        B, D, Hreference, Wreference = reference_fvisu.shape

        query_gvisu = torch.mean(
            query_fvisu.view(B, D, Hquery * Wquery), dim=2, keepdims=False
        ).view(B, D)
        fused_features, attn_score = self.crossview_fusionmodule(
            query_gvisu, reference_fvisu
        )
        attn_score = attn_score.squeeze(1)

        outbox = self.fcn_out(fused_features)
        coodrs = self.coodrs_out(fused_features)

        return outbox, coodrs


def visualize_bbox_comparison(
    query_img: np.ndarray,
    rs_img: np.ndarray,
    pred_bbox: List[float],
    gt_bbox: List[float],
    output_path: str,
    iou: float = None,
    query_name: str = None,
    rs_name: str = None,
) -> None:
    """Draw predicted and ground truth bboxes on satellite image.

    Args:
        query_img: Query/drone image (H, W, C)
        rs_img: Reference/satellite image (H, W, C)
        pred_bbox: Predicted bbox [x1, y1, x2, y2] in pixel coords
        gt_bbox: Ground truth bbox [x1, y1, x2, y2] in pixel coords
        output_path: Path to save visualization
        iou: Optional IoU score to display
        query_name: Optional query image name
        rs_name: Optional reference image name
    """
    H, W = rs_img.shape[:2]

    rs_pil = Image.fromarray(rs_img.astype(np.uint8))
    draw = ImageDraw.Draw(rs_pil)

    x1_gt, y1_gt, x2_gt, y2_gt = gt_bbox
    x1_pred, y1_pred, x2_pred, y2_pred = pred_bbox

    x1_gt = int(np.clip(x1_gt, 0, W))
    y1_gt = int(np.clip(y1_gt, 0, H))
    x2_gt = int(np.clip(x2_gt, 0, W))
    y2_gt = int(np.clip(y2_gt, 0, H))

    x1_pred = int(np.clip(x1_pred, 0, W))
    y1_pred = int(np.clip(y1_pred, 0, H))
    x2_pred = int(np.clip(x2_pred, 0, W))
    y2_pred = int(np.clip(y2_pred, 0, H))

    draw.rectangle([x1_gt, y1_gt, x2_gt, y2_gt], outline="green", width=3)

    draw.rectangle([x1_pred, y1_pred, x2_pred, y2_pred], outline="red", width=3)

    label_text = []
    if query_name:
        label_text.append(f"Query: {query_name}")
    if rs_name:
        label_text.append(f"Ref: {rs_name}")
    if iou is not None:
        label_text.append(f"IoU: {iou:.4f}")

    if label_text:
        text = " | ".join(label_text)
        draw.rectangle([5, 5, 350, 25], fill=(0, 0, 0, 180))
        draw.text((10, 10), text, fill="white")

    legend_x = W - 120
    draw.rectangle([legend_x, 10, legend_x + 100, 30], fill=(0, 0, 0, 180))
    draw.text((legend_x, 12), "GT: Green", fill="green")
    draw.rectangle([legend_x, 35, legend_x + 100, 55], fill=(0, 0, 0, 180))
    draw.text((legend_x, 37), "Pred: Red", fill="red")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    rs_pil.save(output_path)


def visualize(
    image_tensor: torch.Tensor,
    bbox_tensor: torch.Tensor,
    output_path: str = "a.png",
    color: Tuple[int, int, int] = (0, 255, 0),
    width: int = 3,
) -> None:
    """Draw bbox on (3, H, W) tensor image and save to file.

    Args:
        image_tensor: Image tensor of shape (3, H, W)
        bbox_tensor: Bbox tensor of shape (1, 4) with [x1, y1, x2, y2]
        output_path: Path to save the visualization
        color: RGB color tuple for the bbox
        width: Line width for the bbox rectangle
    """
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    image = image_tensor * std + mean
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


class CVOGLDataset(Dataset):
    """CVOGL dataset for TROGeoLite bbox evaluation (no click heatmap)."""

    def __init__(
        self,
        data_root: str,
        data_name: str = "CVOGL_DroneAerial",
        split_name: str = "test",
        transform=None,
        img_size: int = 1024,
    ):
        self.data_root = data_root
        self.data_name = data_name
        self.split_name = split_name
        self.transform = transform
        self.img_size = img_size

        data_dir = os.path.join(data_root, data_name)
        data_path = os.path.join(data_dir, f"{data_name}_{split_name}.pth")
        self.data_list = torch.load(data_path)

        self.queryimg_dir = os.path.join(data_dir, "query")
        self.rsimg_dir = os.path.join(data_dir, "satellite")

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        _, queryimg_name, rsimg_name, _, _, bbox, _, cls_name = self.data_list[idx]

        queryimg = cv2.imread(os.path.join(self.queryimg_dir, queryimg_name))
        queryimg = cv2.cvtColor(queryimg, cv2.COLOR_BGR2RGB)

        rsimg = cv2.imread(os.path.join(self.rsimg_dir, rsimg_name))
        rsimg = cv2.cvtColor(rsimg, cv2.COLOR_BGR2RGB)

        query_img_orig = queryimg.copy()
        rs_img_orig = rsimg.copy()

        if self.transform:
            queryimg = self.transform(queryimg)
            rsimg = self.transform(rsimg)

        ori_gt_bbox = np.array(bbox, dtype=np.float32)

        return {
            "query_imgs": queryimg,
            "rs_imgs": rsimg,
            "ori_gt_bbox": torch.tensor(ori_gt_bbox, dtype=torch.float32),
            "query_img_np": query_img_orig,
            "rs_img_np": rs_img_orig,
            "query_name": queryimg_name,
            "rs_name": rsimg_name,
        }


def eval_CVOGL(
    checkpoint_path: str,
    data_root: str = DATA_ROOT,
    data_name: str = DATA_NAME,
    split_name: str = "test",
    batch_size: int = BATCH_SIZE,
    img_size: int = IMG_SIZE,
    device: str = "cuda:0",
    vis_dir: str = None,
) -> Dict[str, float]:
    """Evaluate TROGeoLite on CVOGL dataset with full metrics.

    Args:
        checkpoint_path: Path to model checkpoint
        data_root: CVOGL dataset root
        data_name: Dataset name
        split_name: train/val/test
        batch_size: Batch size
        img_size: Input image size
        device: Device to use
        vis_dir: Directory to save visualizations (optional)

    Returns:
        Dict with: accu50, accu25, mean_iou, accu_center, num_samples
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"Loading checkpoint from {checkpoint_path}")
    model = TROGeoLite(emb_size=768)
    model = torch.nn.DataParallel(model).cuda()
    checkpoint = torch.load(f"{checkpoint_path}/best.pth", map_location="cpu")
    model.load_state_dict(checkpoint)
    model.eval()

    dataset = CVOGLDataset(
        data_root, data_name, split_name, transform=CVOGL_TRANSFORM, img_size=img_size
    )
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=4
    )

    anchors_full = np.array([float(x) for x in ANCHORS.split(",")])
    anchors_full = anchors_full.reshape(-1, 2)[::-1].copy()
    anchors_full = torch.tensor(anchors_full, dtype=torch.float32).cuda()

    all_ious = []
    all_accu50 = []
    all_accu25 = []
    all_accu_center = []
    all_results = []

    if vis_dir:
        os.makedirs(vis_dir, exist_ok=True)

    print(f"Running evaluation on {data_name}/{split_name}...")
    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
        query_imgs = batch["query_imgs"].cuda()
        rs_imgs = batch["rs_imgs"].cuda()
        ori_gt_bbox = batch["ori_gt_bbox"].cuda()

        with torch.no_grad():
            pred_anchor, _ = model(query_imgs, rs_imgs)

        pred_anchor = pred_anchor.view(
            pred_anchor.shape[0], 9, 5, pred_anchor.shape[2], pred_anchor.shape[3]
        )

        _, best_anchor_gi_gj = build_target(
            ori_gt_bbox, anchors_full, img_size, pred_anchor.shape[3]
        )

        accu_list, accu_center, iou, _, pred_bbox, target_bbox = eval_iou_acc(
            pred_anchor,
            ori_gt_bbox,
            anchors_full,
            best_anchor_gi_gj[:, 1],
            best_anchor_gi_gj[:, 2],
            img_size,
            iou_threshold_list=[0.5, 0.25],
        )

        for i in range(pred_bbox.shape[0]):
            all_ious.append(iou.item() if isinstance(iou, torch.Tensor) else iou)
            all_accu50.append(
                accu_list[0].item()
                if isinstance(accu_list[0], torch.Tensor)
                else accu_list[0]
            )
            all_accu25.append(
                accu_list[1].item()
                if isinstance(accu_list[1], torch.Tensor)
                else accu_list[1]
            )
            all_accu_center.append(
                accu_center.item()
                if isinstance(accu_center, torch.Tensor)
                else accu_center
            )

            result = {
                "query_name": batch["query_name"][i],
                "rs_name": batch["rs_name"][i],
                "pred_bbox": pred_bbox[i].cpu().tolist(),
                "gt_bbox": target_bbox[i].cpu().tolist(),
            }
            all_results.append(result)

            if vis_dir:
                vis_path = os.path.join(vis_dir, f"vis_{batch_idx:04d}_{i:04d}.jpg")
                sample_iou = compute_sample_iou(
                    pred_bbox[i].cpu().numpy(), target_bbox[i].cpu().numpy()
                )
                visualize_bbox_comparison(
                    batch["query_img_np"][i],
                    batch["rs_img_np"][i],
                    pred_bbox[i].cpu().tolist(),
                    target_bbox[i].cpu().tolist(),
                    vis_path,
                    iou=sample_iou,
                    query_name=batch["query_name"][i],
                    rs_name=batch["rs_name"][i],
                )

    count = len(all_ious)
    mean_iou = float(np.mean(all_ious)) if all_ious else 0.0
    accu50 = float(np.mean(all_accu50)) if all_accu50 else 0.0
    accu25 = float(np.mean(all_accu25)) if all_accu25 else 0.0
    accu_center = float(np.mean(all_accu_center)) if all_accu_center else 0.0

    print(f"\n=== Evaluation Results ({data_name}/{split_name}) ===")
    print(f"Accu50: {accu50:.4f}")
    print(f"Accu25: {accu25:.4f}")
    print(f"Mean IoU: {mean_iou:.4f}")
    print(f"Accu Center: {accu_center:.4f}")
    print(f"Samples: {count}")

    return {
        "accu50": accu50,
        "accu25": accu25,
        "mean_iou": mean_iou,
        "accu_center": accu_center,
        "num_samples": count,
        "results": all_results,
    }


def compute_sample_iou(pred_bbox: np.ndarray, gt_bbox: np.ndarray) -> float:
    """Compute IoU for a single sample.

    Args:
        pred_bbox: Predicted bbox [x1, y1, x2, y2]
        gt_bbox: Ground truth bbox [x1, y1, x2, y2]

    Returns:
        IoU score
    """
    x1_p, y1_p, x2_p, y2_p = pred_bbox
    x1_g, y1_g, x2_g, y2_g = gt_bbox

    inter_x1 = max(x1_p, x1_g)
    inter_y1 = max(y1_p, y1_g)
    inter_x2 = min(x2_p, x2_g)
    inter_y2 = min(y2_p, y2_g)

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)

    pred_area = (x2_p - x1_p) * (y2_p - y1_p)
    gt_area = (x2_g - x1_g) * (y2_g - y1_g)

    union_area = pred_area + gt_area - inter_area

    if union_area == 0:
        return 0.0

    return inter_area / union_area


def train_epoch(
    loader, model, optimizer, epoch, anchors_full, img_size, print_freq=PRINT_FREQ
):
    """Train for one epoch.

    Args:
        loader: DataLoader
        model: TROGeoLite model
        optimizer: Optimizer
        epoch: Current epoch
        anchors_full: Anchors tensor
        img_size: Image size
        print_freq: Print frequency

    Returns:
        tuple: (avg_loss, avg_geo_loss, avg_cls_loss)
    """
    model.train()

    batch_time = AverageMeter()
    losses = AverageMeter()
    geo_losses = AverageMeter()
    cls_losses = AverageMeter()

    end = time.time()
    for batch_idx, batch in enumerate(loader):
        query_imgs = batch["query_imgs"].cuda()
        rs_imgs = batch["rs_imgs"].cuda()
        ori_gt_bbox = batch["ori_gt_bbox"].cuda()

        pred_anchor, _ = model(query_imgs, rs_imgs)

        pred_anchor = pred_anchor.view(
            pred_anchor.shape[0], 9, 5, pred_anchor.shape[2], pred_anchor.shape[3]
        )

        new_gt_bbox, best_anchor_gi_gj = build_target(
            ori_gt_bbox, anchors_full, img_size, pred_anchor.shape[3]
        )

        loss_geo, loss_cls = yolo_loss(
            pred_anchor, new_gt_bbox, anchors_full, best_anchor_gi_gj, img_size
        )
        loss = loss_geo + loss_cls

        optimizer.zero_grad()
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


def validate(loader, model, anchors_full, img_size):
    """Validate model.

    Args:
        loader: DataLoader
        model: TROGeoLite model
        anchors_full: Anchors tensor
        img_size: Image size

    Returns:
        tuple: (accu50, accu25, mean_iou)
    """
    model.eval()

    accu50_meter = AverageMeter()
    accu25_meter = AverageMeter()
    iou_meter = AverageMeter()

    for batch in tqdm(loader, desc="Validating"):
        query_imgs = batch["query_imgs"].cuda()
        rs_imgs = batch["rs_imgs"].cuda()
        ori_gt_bbox = batch["ori_gt_bbox"].cuda()

        with torch.no_grad():
            pred_anchor, _ = model(query_imgs, rs_imgs)

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

        accu50_meter.update(accu_list[0].item(), query_imgs.shape[0])
        accu25_meter.update(accu_list[1].item(), query_imgs.shape[0])
        iou_meter.update(iou.item(), query_imgs.shape[0])

    return accu50_meter.avg, accu25_meter.avg, iou_meter.avg


def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    exp_name = args.savename if args.savename else "default_exp"
    writer = SummaryWriter(f"runs/{exp_name}")

    print(f"Creating dataset...")
    train_dataset = CVOGLDataset(
        args.data_root,
        args.data_name,
        "train",
        transform=CVOGL_TRANSFORM,
        img_size=args.img_size,
    )
    val_dataset = CVOGLDataset(
        args.data_root,
        args.data_name,
        "val",
        transform=CVOGL_TRANSFORM,
        img_size=args.img_size,
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

    print(f"Creating model...")
    model = TROGeoLite(emb_size=768)
    model = torch.nn.DataParallel(model).cuda()

    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)

    anchors_full = np.array([float(x) for x in ANCHORS.split(",")])
    anchors_full = anchors_full.reshape(-1, 2)[::-1].copy()
    anchors_full = torch.tensor(anchors_full, dtype=torch.float32).cuda()

    best_iou = -float("Inf")
    update = 0
    print(f"Starting training for {args.max_epoch} epochs...")
    for epoch in range(args.max_epoch):
        adjust_learning_rate(args, optimizer, epoch)

        train_loss, train_geo, train_cls = train_epoch(
            train_loader,
            model,
            optimizer,
            epoch,
            anchors_full,
            args.img_size,
            args.print_freq,
        )

        accu50, accu25, val_iou = validate(
            val_loader, model, anchors_full, args.img_size
        )

        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/train_geo", train_geo, epoch)
        writer.add_scalar("Loss/train_cls", train_cls, epoch)
        writer.add_scalar("Metrics/val_accu50", accu50, epoch)
        writer.add_scalar("Metrics/val_accu25", accu25, epoch)
        writer.add_scalar("Metrics/val_iou", val_iou, epoch)

        print(
            f"Epoch {epoch + 1}/{args.max_epoch}:\t"
            f"Train Loss: {train_loss:.4f} (Geo: {train_geo:.4f}, Cls: {train_cls:.4f})\t"
            f"Val Accu50: {accu50:.4f}\t"
            f"Val Accu25: {accu25:.4f}\t"
            f"Val IoU: {val_iou:.4f}"
        )

        is_best = val_iou > best_iou
        if val_iou > best_iou:
            best_iou = val_iou

            torch.save(model.state_dict(), f"{args.checkpoint}/best.pth")
            updata = 0
        else:
            update += 1
        if update > 5:
            print("no update")
            break

    print(f"\nTraining complete. Best Val IoU: {best_iou:.4f}")
    writer.close()


# --- GroundSmgeoDataset Configuration ---
UNIV_IMAGE_FOLDER = "/data/feihong/image_1024"
UNIV_BBOX_FILE = "/data/feihong/univerisity_dev/runs/bbox_isaac.json"
UNIV_TRAIN_FILE = "/data/feihong/ckpt/train.txt"
UNIV_TEST_FILE = "/data/feihong/ckpt/test.txt"
UNIV_CROP_SIZE = (1024, 1024)
UNIV_DRONE_SIZE = (1024, 1024)


def format_satellite_img_bbox(
    image,
    bbox,
    mode="train",
    target_size=UNIV_DRONE_SIZE,
):
    """Crop satellite image around bbox and resize."""
    x1, y1, x2, y2 = bbox
    width, height = image.size
    crop_size = height // 5 * 4

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


class GroundSmgeoDataset(Dataset):
    """GroundSmgeoDataset adapted for TROGeoLite evaluation."""

    def __init__(self, image_pairs, bbox_dict, mode="test", transform=None):
        self.image_paths = image_pairs
        self.bbox_dict = bbox_dict
        self.mode = mode
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        query_path, search_path = self.image_paths[idx]

        if self.mode == "train":
            choice = []
            for number in ["01", "21", "31", "41", "51"]:
                new_query_path = query_path.replace("01", number)
                if not os.path.exists(new_query_path):
                    continue
                choice.append(new_query_path)
            if choice:
                query_path = random.sample(choice, 1)[0]

        name = query_path.split("/")[-2]

        try:
            query_image = Image.open(query_path).convert("RGB")
            search_image = Image.open(search_path).convert("RGB")
        except FileNotFoundError:
            return self.__getitem__((idx + 1) % len(self))

        query_image = resize_drone_image(query_image)

        base_name = name
        bbox_key = f"{base_name}.png"
        # bbox = self.bbox_dict.get(bbox_key, None)
        # if bbox is None:
        #     bbox = self.bbox_dict.get(bbox_key.replace(".png", ".jpeg"), None)

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

        if self.transform:
            from torchvision.transforms import Compose, ToTensor, Normalize

            transform = Compose(
                [
                    ToTensor(),
                    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )
            queryimg = transform(query_img_np)
            rsimg = transform(rs_img_np)

        return {
            "query_imgs": queryimg,
            "rs_imgs": rsimg,
            "ori_gt_bbox": torch.tensor(normalized_bbox, dtype=torch.float32),
            "query_img_np": query_img_np,
            "rs_img_np": rs_img_np,
            "query_name": name,
            "rs_name": f"{name}.png",
        }


def eval_univ(
    checkpoint_path: str,
    split_name: str = "test",
    batch_size: int = BATCH_SIZE,
    device: str = "cuda:0",
) -> Dict[str, float]:
    """Evaluate TROGeoLite on GroundSmgeoDataset.

    Args:
        checkpoint_path: Path to model checkpoint
        split_name: 'train', 'val', or 'test'
        batch_size: Batch size
        device: Device to use

    Returns:
        Dict with: accu50, accu25, mean_iou, accu_center, num_samples
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"Loading checkpoint from {checkpoint_path}")
    model = TROGeoLite(emb_size=768)
    model = torch.nn.DataParallel(model).cuda()
    checkpoint = torch.load(f"{checkpoint_path}/best.pth", map_location="cpu")
    model.load_state_dict(checkpoint)
    model.eval()

    print("Loading bbox annotations...")
    bbox_dict = json.load(open(UNIV_BBOX_FILE, "r"))

    image_pairs = []
    test_file = UNIV_TEST_FILE if split_name == "test" else UNIV_TRAIN_FILE
    with open(test_file, "r") as f:
        for line in f:
            query_path = line.strip()
            name = query_path.split("/")[-2]
            search_path = f"{UNIV_IMAGE_FOLDER}/{name}.png"
            if os.path.exists(search_path):
                image_pairs.append((query_path, search_path))

    print(f"Found {len(image_pairs)} valid pairs for {split_name}.")

    dataset = GroundSmgeoDataset(
        image_pairs=image_pairs,
        bbox_dict=bbox_dict,
        mode="test",
    )
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=4
    )

    anchors_full = np.array([float(x) for x in ANCHORS.split(",")])
    anchors_full = anchors_full.reshape(-1, 2)[::-1].copy()
    anchors_full = torch.tensor(anchors_full, dtype=torch.float32).cuda()

    all_ious = []
    all_center_dists = []
    all_results = []

    print(f"Running evaluation on GroundSmgeoDataset/{split_name}...")
    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
        query_imgs = batch["query_imgs"].cuda()
        rs_imgs = batch["rs_imgs"].cuda()
        ori_gt_bbox = batch["ori_gt_bbox"].cuda()

        with torch.no_grad():
            pred_anchor, _ = model(query_imgs, rs_imgs)

        pred_anchor = pred_anchor.view(
            pred_anchor.shape[0], 9, 5, pred_anchor.shape[2], pred_anchor.shape[3]
        )

        _, best_anchor_gi_gj = build_target(
            ori_gt_bbox, anchors_full, IMG_SIZE, pred_anchor.shape[3]
        )

        accu_list, accu_center, iou, _, pred_bbox, target_bbox = eval_iou_acc(
            pred_anchor,
            ori_gt_bbox,
            anchors_full,
            best_anchor_gi_gj[:, 1],
            best_anchor_gi_gj[:, 2],
            IMG_SIZE,
            iou_threshold_list=[0.5, 0.25],
        )

        for i in range(pred_bbox.shape[0]):
            pred = pred_bbox[i].cpu().tolist()
            gt = target_bbox[i].cpu().tolist()

            iou_val = compute_sample_iou(np.array(pred), np.array(gt))
            all_ious.append(iou_val)

            pred_center = [(pred[0] + pred[2]) / 2, (pred[1] + pred[3]) / 2]
            gt_center = [(gt[0] + gt[2]) / 2, (gt[1] + gt[3]) / 2]
            center_dist = np.sqrt(
                (pred_center[0] - gt_center[0]) ** 2
                + (pred_center[1] - gt_center[1]) ** 2
            )
            all_center_dists.append(center_dist)

            all_results.append(
                {
                    "query_name": batch["query_name"][i],
                    "pred_bbox": pred,
                    "gt_bbox": gt,
                    "iou": iou_val,
                }
            )

    count = len(all_ious)
    accu50 = sum(1 for iou in all_ious if iou >= 0.5) / count if count > 0 else 0
    accu25 = sum(1 for iou in all_ious if iou >= 0.25) / count if count > 0 else 0
    mean_iou = sum(all_ious) / count if count > 0 else 0
    accu_center = (
        sum(1 for d in all_center_dists if d < 0.05) / count if count > 0 else 0
    )

    print(f"\n=== Evaluation Results (GroundSmgeoDataset/{split_name}) ===")
    print(f"Accu50: {accu50:.4f}")
    print(f"Accu25: {accu25:.4f}")
    print(f"Mean IoU: {mean_iou:.4f}")
    print(f"Accu Center: {accu_center:.4f}")
    print(f"Samples: {count}")

    output_path = os.path.join(
        os.path.dirname(checkpoint_path), f"eval_univ_{split_name}_results.json"
    )
    with open(output_path, "w") as f:
        json.dump(
            {
                "accu50": accu50,
                "accu25": accu25,
                "mean_iou": mean_iou,
                "accu_center": accu_center,
                "num_samples": count,
                "results": all_results,
            },
            f,
            indent=2,
        )
    print(f"Results saved to {output_path}")

    return {
        "accu50": accu50,
        "accu25": accu25,
        "mean_iou": mean_iou,
        "accu_center": accu_center,
        "num_samples": count,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TROGeoLite Training and Evaluation")
    parser.add_argument("--train", action="store_true", help="Train mode")
    parser.add_argument("--eval", action="store_true", help="Evaluation mode (CVOGL)")
    parser.add_argument(
        "--eval-univ", action="store_true", help="Evaluation mode (GroundSmgeoDataset)"
    )
    parser.add_argument("--gpu", default="0", help="GPU id")
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
        "--data-root", type=str, default=DATA_ROOT, help="CVOGL dataset root"
    )
    parser.add_argument(
        "--data-name", type=str, default=DATA_NAME, help="CVOGL_DroneAerial/CVOGL_SVI"
    )
    parser.add_argument(
        "--savename", type=str, default="default", help="Name head for saved model"
    )
    parser.add_argument("--seed", type=int, default=2024, help="random seed")
    parser.add_argument(
        "--print-freq", type=int, default=PRINT_FREQ, help="print frequency"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Dataset split (for evaluation)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="/data/feihong/ckpt/ground_cvos",
        help="Path to model checkpoint (for evaluation)",
    )
    parser.add_argument(
        "--vis-dir",
        type=str,
        default=None,
        help="Directory to save visualizations (for evaluation)",
    )
    args = parser.parse_args()

    os.makedirs(args.checkpoint, exist_ok=True)

    # if args.train:
    # main(args)
    # elif args.eval:
    #     if args.checkpoint is None:
    #         print("Error: --checkpoint is required for evaluation mode")
    #         exit(1)
    eval_CVOGL(
        args.checkpoint,
        args.data_root,
        args.data_name,
        args.split,
        args.batch_size,
        vis_dir=args.vis_dir,
    )
    # eval_univ(
    #     args.checkpoint,
    #     args.split,
    #     args.batch_size,

    # )
    # else:
    #     print("Please specify --train or --eval mode")
    #     print("Examples:")
    #     print("  Train: python ground_cvos.py --train --data-root /data/feihong/CVOGL")
    #     print(
    #         "  Eval:  python ground_cvos.py --eval --checkpoint saved_models/default_model_best.pth.tar"
    #     )
