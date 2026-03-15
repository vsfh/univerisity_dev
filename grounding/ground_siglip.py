#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@file ground_siglip.py
@desc Training SigLIP with heading prediction.
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
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import os
from typing import Dict, List, Tuple, Optional
import argparse
from einops import rearrange

from yolo_utils import (
    get_tensor_anchors,
    build_target,
    yolo_loss,
    eval_iou_acc,
    SpatialTransformer,
)


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, x):
        return x + self.conv(x)


class AverageMeter:
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


MODEL_NAME = "google/siglip-base-patch16-224"
CACHE_DIR = "/data/feihong/hf_cache"
IMAGE_FOLDER = "/data/feihong/image_1024"
HEADING_FOLDER = "/data/feihong/range_250"
TRAIN_BBOX_FILE = "/data/feihong/univerisity_dev/runs/train.json"
TEST_BBOX_FILE = "/data/feihong/univerisity_dev/runs/test.json"
TEXT_FILE = "/data/feihong/drone_text_single_long.json"

UNIV_SAT_SIZE = (640, 640)
UNIV_DRONE_SIZE = (224, 224)

NUM_EPOCHS = 25
BATCH_SIZE = 12
LEARNING_RATE = 1e-4
HEADING_LOSS_WEIGHT = 10.0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PROJECTION_DIM = 768
FEATURE_DIM = 768

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

    min_crop = max(y2 - y1, x2 - x1)
    crop_size = random.uniform(min_crop, max(min_crop, height))

    if mode == "test":
        crop_size = height*0.5

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
    """Dataset with heading prediction."""

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

        bbox_key = name
        bbox = self.bbox_dict.get(bbox_key, None)

        if bbox is None:
            bbox = (1536, 656, 2268, 1374)

        search_image, normalized_bbox = format_satellite_img_bbox(
            search_image, bbox, mode=self.mode, target_size=UNIV_SAT_SIZE
        )

        query_img_np = np.array(query_image)
        rs_img_np = np.array(search_image)

        queryimg = (
            torch.tensor(query_img_np, dtype=torch.float32).permute(2, 0, 1) / 255.0
        )
        rsimg = torch.tensor(rs_img_np, dtype=torch.float32).permute(2, 0, 1) / 255.0

        heading_target = torch.tensor(HEADING_TO_TARGET[heading], dtype=torch.float32)

        return {
            "query_imgs": queryimg,
            "rs_imgs": rsimg,
            "ori_gt_bbox": torch.tensor(normalized_bbox, dtype=torch.float32),
            "heading": heading_target,
            "query_name": name,
            "rs_name": f"{name}.png",
        }


class GeoDataset(Dataset):
    def __init__(
        self,
        image_pairs,
        processor,
        processor_sat,
        tokenizer,
        img_to_text_dict,
        bbox_dict=None,
        mode="train",
    ):
        self.image_paths = image_pairs
        self.processor = processor
        self.processor_sat = processor_sat
        self.tokenizer = tokenizer
        self.img_to_text_dict = img_to_text_dict
        self.bbox_dict = bbox_dict
        self.mode = mode

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        query_path, search_path = self.image_paths[idx]

        heading = 0

        name = query_path.split("/")[-2]
        choice = []
        if self.mode == "train":
            for number in [
                "01.",
                "02.",
                "03.",
                "04.",
                "05.",
            ]:
                new_query_path = query_path.replace("01.", number)
                if not os.path.exists(new_query_path):
                    continue
                choice.append(new_query_path)

            if choice:
                query_path = random.sample(choice, 1)[0]
        text_name = name + "_01"
        text_description = self.img_to_text_dict.get(text_name, "")

        for noun in [
            "**",
            "\n",
            "noun",
            "phrases",
            "Phrase",
            "Noun",
            "Summary",
            "Environment",
            "32 tokens",
            "Description",
            "()",
        ]:
            text_description = text_description.replace(noun, "")

        try:
            query_image = Image.open(query_path).convert("RGB")
            search_image = Image.open(search_path).convert("RGB")
        except FileNotFoundError as e:
            print(f"Error loading image: {e}. Skipping this item.")
            return self.__getitem__((idx + 1) % len(self))
        # score = random.random()
        # if score > 0.5:
        #     crop_length = int(150 * (score - 0.5))
        #     query_image = query_image.crop(
        #         (crop_length, crop_length, 1080 - crop_length, 1080 - crop_length)
        #     )

        query_image = resize_drone_image(query_image)

        bbox_key = name
        bbox = None
        if self.bbox_dict:
            bbox = self.bbox_dict.get(bbox_key, None)
        if bbox is None:
            bbox = (1536, 656, 2268, 1374)

        augmented_crop_image, normalized_bbox = format_satellite_img_bbox(
            search_image, bbox, mode=self.mode, target_size=UNIV_SAT_SIZE
        )

        center_x = (normalized_bbox[0] + normalized_bbox[2]) / 2 / UNIV_SAT_SIZE[0]
        center_y = (normalized_bbox[1] + normalized_bbox[3]) / 2 / UNIV_SAT_SIZE[0]

        col_idx = min(int(center_x * 3), 2)
        row_idx = min(int(center_y * 3), 2)
        index = row_idx * 3 + col_idx

        heading_target = torch.tensor(HEADING_TO_TARGET[heading], dtype=torch.float32)

        text_inputs = self.tokenizer(
            [text_description],
            padding="max_length",
            truncation=True,
            max_length=64,
            return_tensors="pt",
        )
        query_inputs = self.processor(images=query_image, return_tensors="pt")
        search_inputs = self.processor_sat(
            images=augmented_crop_image, return_tensors="pt"
        )

        return {
            "target_pixel_values": query_inputs["pixel_values"][0],
            "search_pixel_values": search_inputs["pixel_values"][0],
            "input_ids": text_inputs["input_ids"][0],
            "index": index,
            "bbox": torch.tensor(normalized_bbox, dtype=torch.float32),
            "heading": heading_target,
        }


class Encoder(nn.Module):
    def __init__(self, model_name=MODEL_NAME, proj_dim=768):
        super().__init__()

        try:
            from transformers import AutoModel, AutoImageProcessor, AutoTokenizer

            self.model = AutoModel.from_pretrained(model_name, cache_dir=CACHE_DIR)
            self.vision_model = self.model.vision_model
            self.text_model = self.model.text_model

            self.feature_dim = self.vision_model.config.hidden_size
            self.text_feature_dim = self.text_model.config.hidden_size
        except Exception as e:
            print(f"Error loading SIGLIP model: {e}")
            raise

        self.pool_sat = nn.AdaptiveAvgPool2d((20, 20))
        self.pool_dro = nn.AdaptiveAvgPool2d((8, 8))
        self.pool_info_sat = nn.AdaptiveAvgPool2d((3, 3))

        self.text_projector = nn.Sequential(
            nn.Linear(self.text_feature_dim, self.text_feature_dim * 2),
            nn.ReLU(),
            nn.Linear(self.text_feature_dim * 2, proj_dim),
        )

        self.bbox_transformer = SpatialTransformer(
            in_channels=self.feature_dim,
            n_heads=8,
            d_head=64,
            depth=1,
            context_dim=self.feature_dim,
        )

        self.bbox_fcn_out = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=self.feature_dim,
                out_channels=self.feature_dim // 2,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.ReLU(inplace=True),
            ResidualBlock(self.feature_dim // 2),
            nn.Conv2d(self.feature_dim // 2, 9 * 5, kernel_size=1),
        )

        self.bbox_adapter = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.GELU(),
            nn.Linear(self.feature_dim, self.feature_dim),
        )

        self.bbox_heading = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=self.feature_dim,
                out_channels=self.feature_dim // 2,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.ReLU(inplace=True),
            ResidualBlock(self.feature_dim // 2),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(self.feature_dim // 2, 2),
        )

    def sat_forward(self, pixel_values):
        embedding = self.vision_model.embeddings.patch_embedding(pixel_values)
        embedding = self.pool_sat(embedding).flatten(2).transpose(1, 2)
        sat_feat = self.vision_model.encoder(embedding).last_hidden_state
        return sat_feat

    def query_forward(self, pixel_values):
        pooled_features = self.vision_model(pixel_values).pooler_output
        return pooled_features

    def text_forward(self, input_ids, attention_mask=None):
        text_outputs = self.text_model(
            input_ids=input_ids, attention_mask=attention_mask
        )
        pooler_output = text_outputs.pooler_output
        proj_feature = self.text_projector(pooler_output)
        return proj_feature

    def forward(self, anchor_pixel_values, search_pixel_values, input_ids=None):
        B = anchor_pixel_values.shape[0]

        anchor_output = self.vision_model(anchor_pixel_values)
        anchor_feats = anchor_output.last_hidden_state
        anchor_pooler = anchor_output.pooler_output

        sat_output = self.vision_model(
            search_pixel_values, interpolate_pos_encoding=True
        )
        sat_feats = sat_output.last_hidden_state
        N = sat_feats.shape[1]
        H = W = int(N**0.5)
        sat_features_2d = sat_feats.permute(0, 2, 1).reshape(B, self.feature_dim, H, W)

        anchor_context = (
            self.bbox_adapter(anchor_feats.detach()) + anchor_feats.detach()
        )
        fused_features = self.bbox_transformer(
            x=sat_features_2d, context=anchor_context
        )

        pred_anchor = self.bbox_fcn_out(fused_features)
        pred_heading = torch.sigmoid(self.bbox_heading(fused_features))

        sat_feature_2d_pool = (
            self.pool_info_sat(sat_features_2d)
            .reshape(B, self.feature_dim, -1)
            .permute(0, 2, 1)
        )

        text_feats = None
        if input_ids is not None:
            text_outputs = self.text_model(input_ids=input_ids)
            pooler_output = text_outputs.pooler_output
            text_feats = self.text_projector(pooler_output)

        return pred_anchor, pred_heading, text_feats, anchor_pooler, sat_feature_2d_pool

    def bbox_forward(self, anchor_pixel_values, search_pixel_values):
        return self.forward(anchor_pixel_values, search_pixel_values, None)


def train_epoch(
    loader,
    model,
    optimizer,
    epoch,
    anchors_full,
    img_size,
    heading_loss_weight,
    print_freq=50,
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
        query_imgs = batch["target_pixel_values"].to(DEVICE)
        rs_imgs = batch["search_pixel_values"].to(DEVICE)
        ori_gt_bbox = batch["bbox"].to(DEVICE)
        heading_target = batch["heading"].to(DEVICE)

        pred_anchor, pred_heading, _, _, _ = model(query_imgs, rs_imgs)

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
        query_imgs = batch["target_pixel_values"].to(DEVICE)
        rs_imgs = batch["search_pixel_values"].to(DEVICE)
        ori_gt_bbox = batch["bbox"].to(DEVICE)
        heading_target = batch["heading"].to(DEVICE)

        with torch.no_grad():
            pred_anchor, pred_heading, _, _, _ = model(query_imgs, rs_imgs)

        pred_anchor = pred_anchor.view(
            pred_anchor.shape[0], 9, 5, pred_anchor.shape[2], pred_anchor.shape[3]
        )

        _, best_anchor_gi_gj = build_target(
            ori_gt_bbox, anchors_full, img_size, pred_anchor.shape[3]
        )
        for i in range(pred_anchor.shape[0]):
            accu_list, accu_center, iou, _, _, _ = eval_iou_acc(
                pred_anchor[i : i + 1],
                ori_gt_bbox[i : i + 1],
                anchors_full,
                best_anchor_gi_gj[:, 1],
                best_anchor_gi_gj[:, 2],
                img_size,
                iou_threshold_list=[0.5, 0.25],
            )

            heading_loss = F.mse_loss(pred_heading[i], heading_target[i])

            accu50_meter.update(1.0 if iou.item() >= 0.5 else 0.0, 1)
            accu25_meter.update(1.0 if iou.item() >= 0.25 else 0.0, 1)
            iou_meter.update(iou.item(), 1)
            # iou_meter.update(iou.item(), query_imgs.shape[0])
            heading_meter.update(heading_loss.item(), query_imgs.shape[0])

    return accu50_meter.avg, accu25_meter.avg, iou_meter.avg, heading_meter.avg


def main(save_path):
    not_update = 0
    exp_name = save_path.split("/")[-1] if save_path else "default_exp"
    writer = SummaryWriter(f"runs/{exp_name}")

    print("Loading bbox annotations...")
    bbox_dict = {}
    for f in [TEST_BBOX_FILE, TRAIN_BBOX_FILE]:
        with open(f, "r") as file:
            data = json.load(file)
            bbox_dict.update(data)
    print(f"Loaded {len(bbox_dict)} bbox annotations")

    print("Loading processors and tokenizer...")
    from transformers import AutoImageProcessor, AutoTokenizer

    processor = AutoImageProcessor.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)
    processor_sat = AutoImageProcessor.from_pretrained(
        MODEL_NAME, cache_dir=CACHE_DIR, size={"height": 640, "width": 640}
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    img_to_text_dict = json.load(open(TEXT_FILE, "r"))

    print("Loading image pairs...")
    train_pairs = []
    with open("/data/feihong/ckpt/train.txt", "r") as f:
        for line in f:
            heading_path = line.strip()
            name = heading_path.split("/")[-2]
            # heading_path = f"{HEADING_FOLDER}/{name}_range250_heading0.png"
            search_path = f"{IMAGE_FOLDER}/{name}.png"
            if os.path.exists(heading_path) and os.path.exists(search_path):
                train_pairs.append((heading_path, search_path))

    val_pairs = []
    with open("/data/feihong/ckpt/test.txt", "r") as f:
        for line in f:
            heading_path = line.strip()
            name = heading_path.split("/")[-2]
            # heading_path = f"{HEADING_FOLDER}/{name}_range250_heading0.png"
            search_path = f"{IMAGE_FOLDER}/{name}.png"
            if os.path.exists(heading_path) and os.path.exists(search_path):
                val_pairs.append((heading_path, search_path))

    print(f"Found {len(train_pairs)} training pairs, {len(val_pairs)} validation pairs")

    print("Creating model...")
    model = Encoder(model_name=MODEL_NAME, proj_dim=PROJECTION_DIM).to(DEVICE)
    model.train()

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

    anchors_full = get_tensor_anchors(DEVICE)

    print("Creating datasets...")
    train_dataset = GeoDataset(
        image_pairs=train_pairs,
        processor=processor,
        processor_sat=processor_sat,
        tokenizer=tokenizer,
        img_to_text_dict=img_to_text_dict,
        bbox_dict=bbox_dict,
        mode="train",
    )
    val_dataset = GeoDataset(
        image_pairs=val_pairs,
        processor=processor,
        processor_sat=processor_sat,
        tokenizer=tokenizer,
        img_to_text_dict=img_to_text_dict,
        bbox_dict=bbox_dict,
        mode="test",
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        pin_memory=True,
        drop_last=False,
        num_workers=4,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
        num_workers=4,
    )

    print(f"Starting training on {DEVICE} for {NUM_EPOCHS} epochs...")
    max_iou = 0

    for epoch in range(NUM_EPOCHS):
        accu50, accu25, val_iou, val_heading = validate(
            val_loader, model, anchors_full, UNIV_SAT_SIZE[0]
        )

        train_loss, train_geo, train_cls, train_heading = train_epoch(
            train_loader,
            model,
            optimizer,
            epoch,
            anchors_full,
            UNIV_SAT_SIZE[0],
            HEADING_LOSS_WEIGHT,
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
            f"Epoch {epoch + 1}/{NUM_EPOCHS}:\t"
            f"Train Loss: {train_loss:.4f} (Geo: {train_geo:.4f}, Cls: {train_cls:.4f}, Heading: {train_heading:.4f})\t"
            f"Val Accu50: {accu50:.4f}\t"
            f"Val Accu25: {accu25:.4f}\t"
            f"Val IoU: {val_iou:.4f}\t"
            f"Val Heading: {val_heading:.4f}"
        )

        if val_iou > max_iou:
            os.makedirs(save_path, exist_ok=True)
            torch.save(model.state_dict(), f"{save_path}/best.pth")
            max_iou = val_iou
            not_update = 0
        else:
            not_update += 1

        if not_update > 5:
            print(f"Validation loss not improving for {not_update} epochs.")
            torch.save(model.state_dict(), f"{save_path}/last.pth")

            break

    print(f"Training complete. Best Val IoU: {max_iou:.4f}")
    writer.close()


def eval_ground():
    from transformers import AutoImageProcessor, AutoTokenizer

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading bbox annotations...")
    bbox_dict = {}
    for f in [TEST_BBOX_FILE, TRAIN_BBOX_FILE]:
        with open(f, "r") as file:
            data = json.load(file)
            bbox_dict.update(data)
    print(f"Loaded {len(bbox_dict)} bbox annotations")

    print("Loading processors and tokenizer...")
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)
    processor_sat = AutoImageProcessor.from_pretrained(
        MODEL_NAME, cache_dir=CACHE_DIR, size={"height": 640, "width": 640}
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    img_to_text_dict = json.load(open(TEXT_FILE, "r"))

    print("Loading test image pairs...")
    test_pairs = []
    with open("/data/feihong/ckpt/test.txt", "r") as f:
        for line in f:
            heading_path = line.strip()
            name = heading_path.split("/")[-2]
            search_path = f"{IMAGE_FOLDER}/{name}.png"
            if os.path.exists(heading_path) and os.path.exists(search_path):
                test_pairs.append((heading_path, search_path))
    print(f"Found {len(test_pairs)} test pairs")

    print("Creating model...")
    model = Encoder(model_name=MODEL_NAME, proj_dim=PROJECTION_DIM).to(DEVICE)
    model_path = f"/data/feihong/ckpt/ground_siglip/last.pth"

    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        print(f"Loaded checkpoint: {model_path}")
    else:
        print(f"Model path {model_path} does not exist")
        return

    anchors_full = get_tensor_anchors(DEVICE)

    print("Creating test dataset...")
    test_dataset = GeoDataset(
        image_pairs=test_pairs,
        processor=processor,
        processor_sat=processor_sat,
        tokenizer=tokenizer,
        img_to_text_dict=img_to_text_dict,
        bbox_dict=bbox_dict,
        mode="test",
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
        num_workers=4,
    )

    model.eval()

    accu50_meter = AverageMeter()
    accu25_meter = AverageMeter()
    iou_meter = AverageMeter()
    center_dist_meter = AverageMeter()

    print("Running evaluation...")
    for batch in tqdm(test_loader, desc="Evaluating"):
        query_imgs = batch["target_pixel_values"].to(DEVICE)
        rs_imgs = batch["search_pixel_values"].to(DEVICE)
        ori_gt_bbox = batch["bbox"].to(DEVICE)

        with torch.no_grad():
            pred_anchor, pred_heading, _, _, _ = model(query_imgs, rs_imgs)

        pred_anchor = pred_anchor.view(
            pred_anchor.shape[0], 9, 5, pred_anchor.shape[2], pred_anchor.shape[3]
        )

        _, best_anchor_gi_gj = build_target(
            ori_gt_bbox, anchors_full, UNIV_SAT_SIZE[0], pred_anchor.shape[3]
        )

        for i in range(pred_anchor.shape[0]):
            accu_list, accu_center, iou, _, pred_bbox, target_bbox = eval_iou_acc(
                pred_anchor[i : i + 1],
                ori_gt_bbox[i : i + 1],
                anchors_full,
                best_anchor_gi_gj[:, 1],
                best_anchor_gi_gj[:, 2],
                UNIV_SAT_SIZE[0],
                iou_threshold_list=[0.5, 0.25],
            )

            pred_center_x = (pred_bbox[0, 0] + pred_bbox[0, 2]) / 2
            pred_center_y = (pred_bbox[0, 1] + pred_bbox[0, 3]) / 2
            target_center_x = (target_bbox[0, 0] + target_bbox[0, 2]) / 2
            target_center_y = (target_bbox[0, 1] + target_bbox[0, 3]) / 2
            center_dist = torch.sqrt(
                (pred_center_x - target_center_x) ** 2
                + (pred_center_y - target_center_y) ** 2
            )

            accu50_meter.update(1.0 if iou.item() >= 0.5 else 0.0, 1)
            accu25_meter.update(1.0 if iou.item() >= 0.25 else 0.0, 1)
            iou_meter.update(iou.item(), 1)
            center_dist_meter.update(center_dist.item(), 1)

    print("\n" + "=" * 50)
    print("Evaluation Results:")
    print("=" * 50)
    print(f"IoU:          {iou_meter.avg:.4f}")
    print(f"Center Dist:  {center_dist_meter.avg:.4f}")
    print(f"Acc@25:       {accu25_meter.avg:.4f}")
    print(f"Acc@50:       {accu50_meter.avg:.4f}")
    print("=" * 50)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", action="store_true", help="Run evaluation only")
    args = parser.parse_args()

    if args.eval:
        eval_ground()
    else:
        exp_name = "ground_siglip"
        save_dir = f"/data/feihong/ckpt/{exp_name}"

        if os.path.exists(save_dir):
            print(f"Experiment directory '{save_dir}' already exists.")
            print("Re-using directory. Logs and checkpoints might be overwritten.")
        else:
            os.makedirs(save_dir, exist_ok=True)
            print(f"Created experiment directory: {save_dir}")

        main(save_dir)
