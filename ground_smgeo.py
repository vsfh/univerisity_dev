#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@file ground_smgeo.py
@desc Ground-based bbox regression using SMGeo network.
      Supervises bbox loss only (no text description).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageDraw
from tqdm import tqdm
import os
from glob import glob
import numpy as np
import random
import cv2
from torch.utils.tensorboard import SummaryWriter
import json
from typing import Union, Tuple, List, Dict, Optional
from torchvision.transforms import Compose, ToTensor, Normalize

from model.swin_moe_geo import SwinTransformer_MoE_MultiInput, CrossViewFusionModule
from model.xyexy import AnchorFreeHead

# --- Configuration ---
DRONE_VIEW_FOLDER = "/data/feihong/drone_view/drone"
IMAGE_FOLDER = "/data/feihong/image_1024"
BBOX_FILE = "/data/feihong/univerisity_dev/runs/bbox_isaac.json"
LOG_PATH = "/data/feihong/ckpt/ground_smgeo"

SAT_ORIG_SIZE = (3840, 2160)
CROP_SIZE = (512, 512)  # Must be divisible by 32 for Swin Transformer compatibility
DRONE_SIZE = (512, 512)

NUM_EPOCHS = 40
BATCH_SIZE = 8
LEARNING_RATE = 1e-4
DEVICE = "cuda:1" if torch.cuda.is_available() else "cpu"

CVOGL_TRANSFORM = Compose(
    [
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# SMGeo model config
EMBED_DIM = 96
PATCH_SIZE = 4
WINDOW_SIZE = 8
DEPTHS = (2, 2, 6, 2)
NUM_HEADS = (3, 6, 12, 24)
FFN_RATIO = 4
NUM_EXPERTS = 6
TOP_K = 2
MOE_BLOCK_INDICES = None


def format_satellite_img_bbox(
    image: Image.Image,
    bbox: List[int],
    mode: str = "train",
    target_size: Tuple[int, int] = DRONE_SIZE,
) -> Tuple[Image.Image, List[float]]:
    x1, y1, x2, y2 = bbox
    width, height = image.size
    crop_size = height // 5 * 4  # 1296

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
    image = image.resize(target_size)

    new_x1 = (x1 - left) / crop_size
    new_y1 = (y1 - top) / crop_size
    new_x2 = (x2 - left) / crop_size
    new_y2 = (y2 - top) / crop_size
    return image, [new_x1, new_y1, new_x2, new_y2]


def pad_to_swin_size(
    image: Image.Image, target_size: Tuple[int, int] = (512, 512)
) -> Tuple[Image.Image, Tuple[int, int, int, int]]:
    """
    Pad image to ensure Swin Transformer compatibility (dimensions divisible by 32).

    Args:
        image: Input image
        target_size: Target size (width, height)

    Returns:
        Tuple of (padded image, padding (left, top, right, bottom))
    """
    width, height = image.size
    target_width, target_height = target_size

    padded_width = target_width
    padded_height = target_height

    pad_left = (padded_width - width) // 2
    pad_top = (padded_height - height) // 2
    pad_right = padded_width - width - pad_left
    pad_bottom = padded_height - height - pad_top

    padding = (pad_left, pad_top, pad_right, pad_bottom)

    padded_image = Image.new(image.mode, (padded_width, padded_height), (0, 0, 0))
    padded_image.paste(image, (pad_left, pad_top))

    return padded_image, padding


def resize_drone_image(
    image: Image.Image, target_size: Tuple[int, int] = (512, 512)
) -> Image.Image:
    """
    Resize drone image to target size.

    Args:
        image: Input drone image
        target_size: Target (width, height)

    Returns:
        Resized image
    """
    return image.resize(target_size, Image.Resampling.BILINEAR)


def compute_iou(pred_bbox: List[float], gt_bbox: List[float]) -> float:
    """
    Compute Intersection over Union (IoU) between predicted and ground truth bboxes.

    Args:
        pred_bbox: Predicted bounding box [x1, y1, x2, y2] normalized [0, 1]
        gt_bbox: Ground truth bounding box [x1, y1, x2, y2] normalized [0, 1]

    Returns:
        IoU score between 0 and 1
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


def visualize(
    image: Union[torch.Tensor, Image.Image],
    bbox: Union[torch.Tensor, List[float]],
) -> None:
    """
    Draw a red rectangle on the image based on the normalized bbox and save to 'a.png'.

    Args:
        image: Input image as tensor or PIL Image
        bbox: Normalized bounding box [x1, y1, x2, y2]
    """
    if isinstance(image, torch.Tensor):
        # Convert tensor to PIL (assume RGB, shape (C, H, W))
        image_np = (image.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        image = Image.fromarray(image_np)

    # Convert bbox to list if tensor
    if isinstance(bbox, torch.Tensor):
        bbox = bbox.cpu().tolist()

    # Scale bbox to image size
    width, height = image.size
    x1, y1, x2, y2 = bbox
    x1_pix = int(x1 * width)
    y1_pix = int(y1 * height)
    x2_pix = int(x2 * width)
    y2_pix = int(y2 * height)

    # Draw rectangle
    draw = ImageDraw.Draw(image)
    draw.rectangle([x1_pix, y1_pix, x2_pix, y2_pix], outline="red", width=2)

    # Save
    image.save("a.png")


class GroundSmgeoDataset(Dataset):
    def __init__(
        self,
        image_pairs,
        bbox_dict,
        mode="train",
    ):
        self.image_paths = image_pairs
        self.bbox_dict = bbox_dict
        self.mode = mode

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
        except FileNotFoundError as e:
            print(f"Error loading image: {e}. Skipping this item.")
            return self.__getitem__((idx + 1) % len(self))

        query_image = resize_drone_image(query_image, DRONE_SIZE)

        base_name = name
        bbox_key = f"{base_name}.png"
        bbox = self.bbox_dict.get(bbox_key, None)
        if bbox is None:
            bbox = self.bbox_dict.get(bbox_key.replace(".png", ".jpeg"), None)

        bbox = (1536, 656, 2268, 1374)

        search_image, normalized_bbox = format_satellite_img_bbox(
            search_image, bbox, mode=self.mode, target_size=CROP_SIZE
        )

        return {
            "drone_img": torch.tensor(
                np.array(query_image), dtype=torch.float32
            ).permute(2, 0, 1)
            / 255.0,
            "sat_img": torch.tensor(
                np.array(search_image), dtype=torch.float32
            ).permute(2, 0, 1)
            / 255.0,
            "bbox": torch.tensor(normalized_bbox, dtype=torch.float32),
            "name": name,
        }


class CVOGLDataset(Dataset):
    """CVOGL dataset for drone-to-satellite bbox regression without segmentation labels."""

    def __init__(
        self,
        data_root: str,
        data_name: str = "CVOGL_DroneAerial",
        split_name: str = "test",
        img_size: int = 1024,
        transform: Optional[callable] = None,
    ):
        """
        Args:
            data_root: Path to CVOGL dataset root
            data_name: Dataset name (default: CVOGL_DroneAerial)
            split_name: 'train', 'val', or 'test'
            img_size: Input image size
            transform: Optional transform to apply
        """
        self.data_root = data_root
        self.data_name = data_name
        self.split_name = split_name
        self.img_size = img_size
        self.transform = transform

        data_dir = os.path.join(data_root, data_name)
        data_path = os.path.join(data_dir, f"{data_name}_{split_name}.pth")
        self.data_list = torch.load(data_path)

        self.queryimg_dir = os.path.join(data_dir, "query")
        self.rsimg_dir = os.path.join(data_dir, "satellite")

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        _, queryimg_name, rsimg_name, _, click_xy, bbox, _, _ = self.data_list[idx]

        queryimg = cv2.imread(os.path.join(self.queryimg_dir, queryimg_name))
        queryimg = cv2.cvtColor(queryimg, cv2.COLOR_BGR2RGB)

        rsimg = cv2.imread(os.path.join(self.rsimg_dir, rsimg_name))
        rsimg = cv2.cvtColor(rsimg, cv2.COLOR_BGR2RGB)

        if self.transform:
            queryimg = self.transform(queryimg)
            rsimg = self.transform(rsimg)

        bbox = np.array(bbox, dtype=np.float32) / self.img_size

        return {
            "query_img": queryimg,
            "sat_img": rsimg,
            "bbox": torch.tensor(bbox, dtype=torch.float32),
        }


class GroundSmgeoEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.backbone = SwinTransformer_MoE_MultiInput(
            in_channels=4,
            embed_dim=EMBED_DIM,
            patch_size=PATCH_SIZE,
            window_size=WINDOW_SIZE,
            depths=DEPTHS,
            num_heads=NUM_HEADS,
            ffn_ratio=FFN_RATIO,
            num_experts=NUM_EXPERTS,
            top_k=TOP_K,
            moe_block_indices=MOE_BLOCK_INDICES,
            datasets=("query", "sat"),
        )

        out_channels = self.backbone.out_dim  # Final channel dimension (768)
        fusion_channels = out_channels * 2  # 1536

        self.head = AnchorFreeHead(
            in_channels=fusion_channels, feat_channels=256, num_classes=1
        )

    def forward(self, drone_img, sat_img):
        drone_img = drone_img[:, :3, :, :]  # Take RGB
        drone_img_4ch = torch.cat([drone_img, drone_img[:, :1, :, :]], dim=1)

        query_vec, sat_feat, _ = self.backbone(
            [drone_img_4ch, sat_img], datasets=("query", "sat")
        )

        B, C, H, W = sat_feat.shape
        query_vec_expanded = query_vec.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)

        fused = torch.cat([sat_feat, query_vec_expanded], dim=1)

        heatmap, bbox_map = self.head(fused)

        return heatmap, bbox_map

    def get_bbox_from_prediction(self, heatmap, bbox_map):
        B, _, H, W = heatmap.shape
        device = heatmap.device
        pred_bboxes = []

        for b in range(B):
            hmap = heatmap[b, 0]
            bm = bbox_map[b]

            peak_y, peak_x = torch.where(hmap == hmap.max())
            if len(peak_y) == 0:
                pred_bboxes.append(torch.tensor([0.5, 0.5, 0.5, 0.5], device=device))
                continue

            peak_y = peak_y[0]
            peak_x = peak_x[0]

            bx1 = torch.sigmoid(bm[0, peak_y, peak_x])
            by1 = torch.sigmoid(bm[1, peak_y, peak_x])
            bx2 = torch.sigmoid(bm[2, peak_y, peak_x])
            by2 = torch.sigmoid(bm[3, peak_y, peak_x])

            pred_bboxes.append(torch.stack([bx1, by1, bx2, by2]))

        return torch.stack(pred_bboxes)


def compute_bbox_loss(
    pred_bbox: torch.Tensor, target_bbox: torch.Tensor
) -> torch.Tensor:
    """
    Compute bbox loss: L1 loss + IoU loss.

    Args:
        pred_bbox: Predicted bounding box [B, 4] normalized [0, 1]
        target_bbox: Target bounding box [B, 4] normalized [0, 1]

    Returns:
        Combined loss value
    """
    l1_loss = F.l1_loss(pred_bbox, target_bbox)

    pred_x1 = pred_bbox[:, 0]
    pred_y1 = pred_bbox[:, 1]
    pred_x2 = pred_bbox[:, 2]
    pred_y2 = pred_bbox[:, 3]

    target_x1 = target_bbox[:, 0]
    target_y1 = target_bbox[:, 1]
    target_x2 = target_bbox[:, 2]
    target_y2 = target_bbox[:, 3]

    inter_x1 = torch.max(pred_x1, target_x1)
    inter_y1 = torch.max(pred_y1, target_y1)
    inter_x2 = torch.min(pred_x2, target_x2)
    inter_y2 = torch.min(pred_y2, target_y2)

    inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(
        inter_y2 - inter_y1, min=0
    )

    pred_area = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)
    target_area = (target_x2 - target_x1) * (target_y2 - target_y1)

    union_area = pred_area + target_area - inter_area + 1e-6

    iou = inter_area / union_area
    iou_loss = -torch.log(iou + 1e-6)

    total_loss = 0.5 * l1_loss + 0.5 * iou_loss.mean()

    return total_loss


def main(save_path):
    exp_name = save_path.split("/")[-1] if save_path else "default_exp"
    writer = SummaryWriter(f"runs/{exp_name}")

    print("Gathering image pairs...")
    image_pairs = []
    for query_path in tqdm(glob(f"{DRONE_VIEW_FOLDER}/*/image-01.jpeg")):
        name = query_path.split("/")[-2]
        search_path = f"{IMAGE_FOLDER}/{name}.png"
        if os.path.exists(search_path):
            image_pairs.append((query_path, search_path))
    print(f"Found {len(image_pairs)} valid pairs.")

    print("Loading bbox dictionary...")
    bbox_dict = json.load(open(BBOX_FILE, "r"))

    print("Building SMGeo model...")
    model = GroundSmgeoEncoder().to(DEVICE)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    print("Setting up dataset and dataloader...")
    train_image_pairs = []
    with open("/data/feihong/ckpt/train.txt", "r") as f:
        for line in f:
            query_path = line.strip()
            name = query_path.split("/")[-2]
            search_path = f"{IMAGE_FOLDER}/{name}.png"
            if os.path.exists(search_path):
                train_image_pairs.append((query_path, search_path))

    test_image_pairs = []
    with open("/data/feihong/ckpt/test.txt", "r") as f:
        for line in f:
            query_path = line.strip()
            name = query_path.split("/")[-2]
            search_path = f"{IMAGE_FOLDER}/{name}.png"
            if os.path.exists(search_path):
                test_image_pairs.append((query_path, search_path))

    train_dataset = GroundSmgeoDataset(
        image_pairs=train_image_pairs,
        bbox_dict=bbox_dict,
    )
    train_dataloader = DataLoader(
        train_dataset, shuffle=True, batch_size=BATCH_SIZE, num_workers=4
    )

    test_dataset = GroundSmgeoDataset(
        image_pairs=test_image_pairs,
        bbox_dict=bbox_dict,
        mode="test",
    )
    test_dataloader = DataLoader(
        test_dataset, shuffle=False, batch_size=BATCH_SIZE, num_workers=4
    )

    print(f"Starting training on {DEVICE} for {NUM_EPOCHS} epochs...")
    best_iou = 0
    not_update = 0
    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0
        total_bbox_loss = 0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS}")

        for i, batch in enumerate(progress_bar):
            drone_img = batch["drone_img"].to(DEVICE)
            sat_img = batch["sat_img"].to(DEVICE)
            target_bbox = batch["bbox"].to(DEVICE)

            heatmap, bbox_map = model(drone_img, sat_img)
            pred_bbox = model.get_bbox_from_prediction(heatmap, bbox_map)

            bbox_loss = compute_bbox_loss(pred_bbox, target_bbox)

            loss = bbox_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_bbox_loss += bbox_loss.item()

            progress_bar.set_postfix(
                {
                    "loss": f"{loss.item():.4f}",
                    "bbox": f"{bbox_loss.item():.4f}",
                }
            )
            writer.add_scalar(
                "Loss/train_batch", loss.item(), epoch * len(train_dataloader) + i
            )

        avg_loss = total_loss / len(train_dataloader)
        avg_bbox_loss = total_bbox_loss / len(train_dataloader)

        model.eval()
        val_metrics = validation(model, test_dataloader, epoch)
        writer.add_scalar("Loss/train_epoch", avg_loss, epoch)
        writer.add_scalar("Loss/val_bbox_epoch", val_metrics["bbox_loss"], epoch)
        writer.add_scalar("Metrics/mean_iou", val_metrics["mean_iou"], epoch)
        writer.add_scalar("Metrics/iou@0.5", val_metrics["iou_at_0.5"], epoch)
        writer.add_scalar("Metrics/iou@0.25", val_metrics["iou_at_0.25"], epoch)

        print(
            f"Epoch {epoch + 1} finished. "
            f"Avg train loss: {avg_loss:.4f} (bbox: {avg_bbox_loss:.4f}). "
            f"Val mean IoU: {val_metrics['mean_iou']:.4f}, IoU@0.5: {val_metrics['iou_at_0.5']:.4f}, IoU@0.25: {val_metrics['iou_at_0.25']:.4f}"
        )

        if val_metrics["mean_iou"] > best_iou:
            os.makedirs(save_path, exist_ok=True)
            torch.save(model.state_dict(), f"{save_path}/best.pth")
            best_iou = val_metrics["mean_iou"]
            not_update = 0
        else:
            not_update += 1
        if not_update > 5:
            print("Validation loss not improving. Stopping early.")
            break
    print(f"Training complete. Best IoU: {best_iou:.4f}")
    writer.close()


def validation(model, loader, epoch):
    progress_bar = tqdm(loader, desc=f"Valid {epoch + 1}/{NUM_EPOCHS}")
    total_bbox_loss = 0
    all_ious = []

    for batch in progress_bar:
        drone_img = batch["drone_img"].to(DEVICE)
        sat_img = batch["sat_img"].to(DEVICE)
        target_bbox = batch["bbox"].to(DEVICE)

        with torch.no_grad():
            heatmap, bbox_map = model(drone_img, sat_img)
            pred_bbox = model.get_bbox_from_prediction(heatmap, bbox_map)

            bbox_loss = compute_bbox_loss(pred_bbox, target_bbox)

            total_bbox_loss += bbox_loss.item()

            for j in range(pred_bbox.shape[0]):
                pred = pred_bbox[j].cpu().tolist()
                gt = target_bbox[j].cpu().tolist()
                iou = compute_iou(pred, gt)
                all_ious.append(iou)

    avg_bbox_loss = total_bbox_loss / len(loader)

    mean_iou = np.mean(all_ious) if all_ious else 0.0
    iou_at_05 = (
        sum(1 for iou in all_ious if iou >= 0.5) / len(all_ious) if all_ious else 0.0
    )
    iou_at_025 = (
        sum(1 for iou in all_ious if iou >= 0.25) / len(all_ious) if all_ious else 0.0
    )

    return {
        "bbox_loss": avg_bbox_loss,
        "mean_iou": mean_iou,
        "iou_at_0.5": iou_at_05,
        "iou_at_0.25": iou_at_025,
    }


def eval(checkpoint_path: str) -> Dict[str, float]:
    """Evaluate GroundSmgeoEncoder on GroundSmgeoDataset with full metrics.

    Returns:
        Dict with: accu50, accu25, mean_iou, accu_center, num_samples
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"Loading checkpoint from {checkpoint_path}")
    model = GroundSmgeoEncoder().to(DEVICE)
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    model.eval()

    print("Loading bbox annotations...")
    bbox_dict = json.load(open(BBOX_FILE, "r"))

    test_image_pairs = []
    with open("/data/feihong/ckpt/test.txt", "r") as f:
        for line in f:
            query_path = line.strip()
            name = query_path.split("/")[-2]
            search_path = f"{IMAGE_FOLDER}/{name}.png"
            if os.path.exists(search_path):
                test_image_pairs.append((query_path, search_path))

    test_dataset = GroundSmgeoDataset(
        image_pairs=test_image_pairs,
        bbox_dict=bbox_dict,
        mode="test",
    )
    test_dataloader = DataLoader(
        test_dataset, shuffle=False, batch_size=BATCH_SIZE, num_workers=4
    )

    print("Running evaluation...")
    all_ious = []
    all_center_dists = []
    results = []

    for batch in tqdm(test_dataloader, desc="Evaluating"):
        drone_img = batch["drone_img"].to(DEVICE)
        sat_img = batch["sat_img"].to(DEVICE)
        target_bbox = batch["bbox"].cpu().tolist()
        names = batch["name"]

        with torch.no_grad():
            heatmap, bbox_map = model(drone_img, sat_img)
            pred_bbox = model.get_bbox_from_prediction(heatmap, bbox_map)
            pred_bbox_cpu = pred_bbox.cpu().tolist()

        for j in range(len(names)):
            pred = pred_bbox_cpu[j]
            gt = target_bbox[j]
            iou = compute_iou(pred, gt)
            all_ious.append(iou)

            pred_center = [(pred[0] + pred[2]) / 2, (pred[1] + pred[3]) / 2]
            gt_center = [(gt[0] + gt[2]) / 2, (gt[1] + gt[3]) / 2]
            center_dist = np.sqrt(
                (pred_center[0] - gt_center[0]) ** 2
                + (pred_center[1] - gt_center[1]) ** 2
            )
            all_center_dists.append(center_dist)

            results.append(
                {
                    "name": names[j],
                    "pred_bbox": pred,
                    "gt_bbox": gt,
                    "iou": iou,
                }
            )

    count = len(all_ious)
    accu50 = sum(1 for iou in all_ious if iou >= 0.5) / count if count > 0 else 0
    accu25 = sum(1 for iou in all_ious if iou >= 0.25) / count if count > 0 else 0
    mean_iou = sum(all_ious) / count if count > 0 else 0
    accu_center = (
        sum(1 for d in all_center_dists if d < 0.05) / count if count > 0 else 0
    )

    print(f"\n=== Evaluation Results (GroundSmgeoDataset) ===")
    print(f"Accu50: {accu50:.4f}")
    print(f"Accu25: {accu25:.4f}")
    print(f"Mean IoU: {mean_iou:.4f}")
    print(f"Accu Center: {accu_center:.4f}")
    print(f"Samples: {count}")

    output_path = os.path.join(os.path.dirname(checkpoint_path), "eval_results.json")
    with open(output_path, "w") as f:
        json.dump(
            {
                "accu50": accu50,
                "accu25": accu25,
                "mean_iou": mean_iou,
                "accu_center": accu_center,
                "num_samples": count,
                "results": results,
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


def eval_CVOGL(
    checkpoint_path: str,
    data_root: str = "/data/feihong/CVOGL",
    data_name: str = "CVOGL_DroneAerial",
    split_name: str = "test",
    batch_size: int = 8,
    device: str = DEVICE,
) -> Dict[str, float]:
    """Evaluate GroundSmgeoEncoder on CVOGL dataset with full metrics.

    Returns:
        Dict with: accu50, accu25, mean_iou, accu_center, num_samples
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"Loading checkpoint from {checkpoint_path}")
    model = GroundSmgeoEncoder().to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    dataset = CVOGLDataset(data_root, data_name, split_name, transform=CVOGL_TRANSFORM)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=4
    )

    print(f"Running evaluation on {data_name}/{split_name}...")
    all_ious = []
    all_center_dists = []
    results = []

    for batch in tqdm(dataloader, desc="Evaluating"):
        query_img = batch["query_img"].to(device)
        sat_img = batch["sat_img"].to(device)
        gt_bbox = batch["bbox"].to(device)

        with torch.no_grad():
            heatmap, bbox_map = model(query_img, sat_img)
            pred_bbox = model.get_bbox_from_prediction(heatmap, bbox_map)

        for i in range(pred_bbox.shape[0]):
            pred = pred_bbox[i].cpu().numpy()
            gt = gt_bbox[i].cpu().numpy()

            iou = compute_iou(pred, gt)
            all_ious.append(iou)

            pred_center = [(pred[0] + pred[2]) / 2, (pred[1] + pred[3]) / 2]
            gt_center = [(gt[0] + gt[2]) / 2, (gt[1] + gt[3]) / 2]
            center_dist = np.sqrt(
                (pred_center[0] - gt_center[0]) ** 2
                + (pred_center[1] - gt_center[1]) ** 2
            )
            all_center_dists.append(center_dist)

    count = len(all_ious)
    accu50 = sum(1 for iou in all_ious if iou >= 0.5) / count if count > 0 else 0
    accu25 = sum(1 for iou in all_ious if iou >= 0.25) / count if count > 0 else 0
    mean_iou = sum(all_ious) / count if count > 0 else 0
    accu_center = (
        sum(1 for d in all_center_dists if d < 0.05) / count if count > 0 else 0
    )

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
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="GroundSmgeo Evaluation")
    parser.add_argument(
        "--checkpoint", type=str, default="/data/feihong/ckpt/ground_smgeo/best.pth", help="Path to model checkpoint"
    )
    parser.add_argument(
        "--cvogl",
        action="store_true",
        help="Evaluate on CVOGL dataset instead of GroundSmgeoDataset",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="/data/feihong/CVOGL",
        help="CVOGL dataset root path (for --cvogl)",
    )
    parser.add_argument(
        "--data-name",
        type=str,
        default="CVOGL_DroneAerial",
        help="Dataset name (for --cvogl)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Dataset split (for --cvogl)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for evaluation (for --cvogl)",
    )
    parser.add_argument(
        "--device", type=str, default=None, help="Device to use (cuda or cpu)"
    )
    args = parser.parse_args()

    device = args.device if args.device else DEVICE

    # if args.cvogl:
    eval_CVOGL(
        args.checkpoint,
        args.data_root,
        args.data_name,
        args.split,
        args.batch_size,
        device,
    )
    # else:
    #     eval(args.checkpoint)
