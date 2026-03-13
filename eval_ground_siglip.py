#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@file eval_ground_siglip.py
@desc Evaluation script for SigLIP heading model from unified_siglip_heading.py.
     Evaluates bbox prediction and heading prediction on ground-to-satellite matching.
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
import argparse
import json
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import matplotlib.pyplot as plt

from transformers import AutoImageProcessor, AutoModel, AutoTokenizer
from model import Encoder_heading, Encoder_abla
from bbox.yolo_utils import get_tensor_anchors, build_target, eval_iou_acc, bbox_iou

HEADING_TO_TARGET = {
    0: [0.0, 0.0],
    90: [0.0, 1.0],
    180: [1.0, 0.0],
    270: [1.0, 1.0],
}

TARGET_TO_HEADING = {
    (0.0, 0.0): 0,
    (0.0, 1.0): 90,
    (1.0, 0.0): 180,
    (1.0, 1.0): 270,
}

HEADING_FOLDER = "/data/feihong/range_250"
IMAGE_FOLDER = "/data/feihong/image_1024"

UNIV_SAT_SIZE = (640, 640)
UNIV_DRONE_SIZE = (256, 256)
SAT_ORIG_SIZE = (3840, 2160)

MODEL_NAME = "google/siglip-base-patch16-224"
CACHE_DIR = "/data/feihong/hf_cache"
PROJECTION_DIM = 768


def pred_to_heading(pred):
    """Convert predicted heading vector to heading angle."""
    if hasattr(pred, "detach"):
        pred = pred.detach().cpu().numpy()
    min_dist = float("inf")
    best_heading = 0
    for (t0, t1), heading in TARGET_TO_HEADING.items():
        dist = (pred[0] - t0) ** 2 + (pred[1] - t1) ** 2
        if dist < min_dist:
            min_dist = dist
            best_heading = heading
    return best_heading


def format_satellite_img_bbox(
    image,
    bbox,
    mode="test",
    ratio=0.8,
    target_size=UNIV_SAT_SIZE,
):
    """Crop satellite image around bbox and resize."""
    x1, y1, x2, y2 = bbox
    width, height = image.size

    crop_size = int(ratio * height)

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


@dataclass
class EvalConfig:
    model_name: str
    checkpoint_path: str
    device: str = "cuda:1" if torch.cuda.is_available() else "cpu"
    batch_size: int = 1
    num_workers: int = 4
    output_dir: str = "./eval_results"
    visualize: bool = True
    save_images: bool = False
    vis_limit: int = 50


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


def compute_iou(pred_bbox: List[float], gt_bbox: List[float]) -> float:
    """Compute Intersection over Union (IoU)."""
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


def tensor_to_pil(tensor: torch.Tensor, preprocessor) -> Image.Image:
    """Convert CHW tensor (0-1 range) to PIL Image."""
    mean = preprocessor.image_mean
    std = preprocessor.image_std
    
    # Reshape mean and std to (3, 1, 1) so they broadcast correctly over (C, H, W)
    mean_tensor = torch.tensor(mean).view(3, 1, 1)
    std_tensor = torch.tensor(std).view(3, 1, 1)
    
    # 3. Un-normalize the tensor
    unnormalized_tensor = (tensor.cpu() * std_tensor) + mean_tensor
    tensor = unnormalized_tensor.clamp_(0, 1)
    array = (tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return Image.fromarray(array)


def draw_bbox_pixel(
    image: Image.Image,
    pred_bbox: List[float],
    gt_bbox: List[float],
    img_width: int,
    img_height: int,
    pred_color: str = "red",
    gt_color: str = "green",
) -> Image.Image:
    """Draw bboxes using pixel coordinates (unnormalized)."""
    draw = ImageDraw.Draw(image)

    x1_p, y1_p, x2_p, y2_p = pred_bbox
    x1_g, y1_g, x2_g, y2_g = gt_bbox

    x1_pix = int(x1_p)
    y1_pix = int(y1_p)
    x2_pix = int(x2_p)
    y2_pix = int(y2_p)

    x1_g_pix = int(x1_g)
    y1_g_pix = int(y1_g)
    x2_g_pix = int(x2_g)
    y2_g_pix = int(y2_g)

    draw.rectangle([x1_pix, y1_pix, x2_pix, y2_pix], outline=pred_color, width=2)
    draw.rectangle([x1_g_pix, y1_g_pix, x2_g_pix, y2_g_pix], outline=gt_color, width=2)

    return image


def visualize_results(
    query_tensor: torch.Tensor,
    search_tensor: torch.Tensor,
    pred_bbox: torch.Tensor,
    gt_bbox: torch.Tensor,
    name: str,
    iou: float,
    pred_heading: int,
    gt_heading: int,
    output_path: str,
    preprocessor
) -> None:
    """Visualize query, search with bboxes, and save."""
    query_image = tensor_to_pil(query_tensor, preprocessor)
    search_image = tensor_to_pil(search_tensor, preprocessor)

    _, search_h, search_w = search_tensor.shape

    pred_bbox_list = pred_bbox.cpu().tolist()
    gt_bbox_list = gt_bbox.cpu().tolist()

    fig, axes = plt.subplots(1, 1, figsize=(15, 5))

    # axes[0].imshow(query_image)
    # axes[0].set_title(f"Query Image: {name}\nGT Heading: {gt_heading}")
    # axes[0].axis("off")

    search_draw = search_image.copy()
    search_draw = draw_bbox_pixel(
        search_draw,
        pred_bbox_list,
        gt_bbox_list,
        search_w,
        search_h,
        pred_color="red",
        gt_color="green",
    )
    axes.imshow(search_draw)
    # axes[1].set_title(f"Search Image (IoU: {iou:.3f})\nPred Heading: {pred_heading}")
    axes.axis("off")

    # overlay = Image.blend(
    #     search_image.convert("RGBA"),
    #     Image.new("RGBA", search_image.size, (0, 255, 0, 100)),
    #     0.3,
    # )
    # overlay = draw_bbox_pixel(
    #     overlay,
    #     pred_bbox_list,
    #     gt_bbox_list,
    #     search_w,
    #     search_h,
    #     pred_color="red",
    #     gt_color="green",
    # )
    # axes[2].imshow(overlay)
    # axes[2].set_title("Overlay")
    # axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=700, bbox_inches="tight")
    plt.close()


def get_test_pairs():
    """Get test image pairs from test.txt with heading=0."""
    test_pairs = []
    with open("/data/feihong/ckpt/test.txt", "r") as f:
        for line in f:
            query_path = line.strip()
            name = query_path.split("/")[-2]
            search_path = f"{IMAGE_FOLDER}/{name}.png"
            heading_path = f"{HEADING_FOLDER}/{name}_range250_heading0.png"
            heading = 0
            if os.path.exists(heading_path):
                heading = 0
            if os.path.exists(search_path):
                test_pairs.append((query_path, search_path, name, heading))
    return test_pairs


def load_success_filenames():
    """Load filenames that are considered successful from the success file."""
    success_set = set()
    success_file = "/data/feihong/univerisity_dev/eval_success_unified.txt"
    if os.path.exists(success_file):
        with open(success_file, "r") as f:
            for line in f:
                name = line.strip()
                if name:
                    success_set.add(name)
    return success_set


def get_gt_heading(query_path: str, name: str) -> int:
    """Extract heading from query path or heading file."""
    heading_path = f"{HEADING_FOLDER}/{name}_range250_heading0.png"
    if os.path.exists(heading_path):
        for h in [0, 90, 180, 270]:
            hp = f"{HEADING_FOLDER}/{name}_range250_heading{h}.png"
            if os.path.exists(hp) and hp == query_path:
                return h
    return 0


def heading_to_target(heading: int) -> List[float]:
    """Convert heading angle to target vector."""
    return HEADING_TO_TARGET.get(heading, [0.0, 0.0])


def get_gt_bbox(name: str, test_bbox_dict: Dict) -> List[float]:
    """Get ground truth bbox for a sample."""
    bbox = test_bbox_dict.get(name, None)
    if bbox is None:
        bbox = test_bbox_dict.get(f"{name}.png", None)
    if bbox is None:
        return [0.25, 0.25, 0.75, 0.75]
    x1, y1, x2, y2 = bbox
    return [x1 / 3840, y1 / 2160, x2 / 3840, y2 / 2160]


class EvalDataset(Dataset):
    """Evaluation dataset for SigLIP heading model."""

    def __init__(
        self,
        eval_list,
        processor,
        processor_sat,
        test_bbox_dict,
        heading_folder=HEADING_FOLDER,
    ):
        self.eval_list = eval_list
        self.processor = processor
        self.processor_sat = processor_sat
        self.test_bbox_dict = test_bbox_dict
        self.heading_folder = heading_folder

    def __len__(self):
        return len(self.eval_list)

    def __getitem__(self, idx):
        query_path = self.eval_list[idx]
        name = query_path.split("/")[-2]

        search_path = f"{IMAGE_FOLDER}/{name}.png"
        if not os.path.exists(search_path):
            return None

        try:
            query_image = Image.open(query_path).convert("RGB")
            search_image = Image.open(search_path).convert("RGB")
        except Exception:
            return None

        gt_bbox = get_gt_bbox(name, self.test_bbox_dict)

        bbox_pixel = [
            gt_bbox[0] * SAT_ORIG_SIZE[0],
            gt_bbox[1] * SAT_ORIG_SIZE[1],
            gt_bbox[2] * SAT_ORIG_SIZE[0],
            gt_bbox[3] * SAT_ORIG_SIZE[1],
        ]
        bbox_adjusted = [
            (bbox_pixel[0] - 840) * (640 / 2160),
            bbox_pixel[1] * (640 / 2160),
            (bbox_pixel[2] - 840) * (640 / 2160),
            bbox_pixel[3] * (640 / 2160),
        ]

        query_image = query_image.crop((420, 0, 1500, 1080))
        query_image = resize_drone_image(query_image, UNIV_DRONE_SIZE)

        search_image = search_image.crop((840, 0, 3000, 2160))
        search_image = search_image.resize(UNIV_SAT_SIZE)

        query_pixels = self.processor(images=query_image, return_tensors="pt")[
            "pixel_values"
        ][0]
        search_pixels = self.processor_sat(images=search_image, return_tensors="pt")[
            "pixel_values"
        ][0]

        heading = 0
        heading_target = torch.tensor(heading_to_target(heading), dtype=torch.float32)

        return {
            "name": name,
            "query_pixels": query_pixels,
            "search_pixels": search_pixels,
            "bbox_adjusted": torch.tensor(bbox_adjusted, dtype=torch.float32),
            "heading_target": heading_target,
            "heading": heading,
            "gt_bbox": gt_bbox,
        }


def collate_fn(batch):
    """Collate function to filter None items and batch tensors."""
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    names = [item["name"] for item in batch]
    query_pixels = torch.stack([item["query_pixels"] for item in batch])
    search_pixels = torch.stack([item["search_pixels"] for item in batch])
    bbox_adjusted = torch.stack([item["bbox_adjusted"] for item in batch])
    heading_target = torch.stack([item["heading_target"] for item in batch])
    headings = [item["heading"] for item in batch]
    gt_bboxes = [item["gt_bbox"] for item in batch]

    return {
        "name": names,
        "query_pixels": query_pixels,
        "search_pixels": search_pixels,
        "bbox_adjusted": bbox_adjusted,
        "heading_target": heading_target,
        "heading": headings,
        "gt_bbox": gt_bboxes,
    }


def preprocess_images_siglip(query_image, search_image, bbox, processor, processor_sat):
    """Preprocess images using AutoImageProcessor like unified_siglip_abla.py."""
    query_image = query_image.crop((420, 0, 1500, 1080))
    query_image = resize_drone_image(query_image, UNIV_DRONE_SIZE)

    search_image = search_image.crop((840, 0, 3000, 2160))
    search_image = search_image.resize(UNIV_SAT_SIZE)

    query_inputs = processor(images=query_image, return_tensors="pt")
    search_inputs = processor_sat(images=search_image, return_tensors="pt")

    return query_inputs["pixel_values"][0], search_inputs["pixel_values"][0], bbox


def load_model(checkpoint_path: str, device: str):
    """Load Encoder_heading model from checkpoint."""
    # checkpoint_path = "/data/feihong/ckpt/unified_siglip_abla_ap/best_iou_41.pth"
    # checkpoint_path = "/data/feihong/ckpt/unified_siglip_heading/best_iou_44.pth"
    model = Encoder_heading(model_name=MODEL_NAME, proj_dim=PROJECTION_DIM)
    if os.path.exists(checkpoint_path):
        model.load_state_dict(
            torch.load(checkpoint_path, map_location="cpu"), strict=True
        )
        print(f"Loaded checkpoint: {checkpoint_path}")
    else:
        print(f"Checkpoint not found: {checkpoint_path}")
    model = model.to(device)
    model.eval()
    return model


def evaluate_model(
    checkpoint_path: str,
    config: EvalConfig,
) -> Dict:
    """Evaluate the SigLIP heading model."""
    print(f"\n{'=' * 60}")
    print(f"Evaluating SigLIP Heading Model")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"{'=' * 60}")

    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found: {checkpoint_path}")
        return {}

    test_bbox_dict = json.load(
        open("/data/feihong/univerisity_dev/runs/test.json", "r")
    )
    test_pairs = get_test_pairs()
    print(f"Found {len(test_pairs)} test samples")

    model = load_model(checkpoint_path, config.device)
    anchors_full = get_tensor_anchors(config.device)

    os.makedirs(config.output_dir, exist_ok=True)
    vis_dir = os.path.join(config.output_dir, "visualizations", "siglip_heading")
    os.makedirs(vis_dir, exist_ok=True)

    accu50_meter = AverageMeter()
    accu25_meter = AverageMeter()
    iou_meter = AverageMeter()
    center_dist_meter = AverageMeter()
    heading_mse_meter = AverageMeter()
    heading_acc_meter = AverageMeter()
    unified_iou_meter = AverageMeter()

    results = []
    vis_count = 0

    success_filenames = load_success_filenames()

    processor = AutoImageProcessor.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)
    processor_sat = AutoImageProcessor.from_pretrained(
        MODEL_NAME, cache_dir=CACHE_DIR, size={"height": 640, "width": 640}
    )

    eval_list = [pair[0] for pair in test_pairs]
    dataset = EvalDataset(eval_list, processor, processor_sat, test_bbox_dict)

    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    print(f"Starting batched evaluation with {len(loader)} batches...")

    with torch.inference_mode():
        for batch in tqdm(loader, desc="Evaluating SigLIP Heading"):
            if batch is None:
                continue

            names = batch["name"]
            query_pixels = batch["query_pixels"].to(config.device)
            search_pixels = batch["search_pixels"].to(config.device)
            gt_bbox_tensor = batch["bbox_adjusted"].to(config.device)
            heading_target = batch["heading_target"].to(config.device)
            gt_headings = batch["heading"]
            gt_bboxes = batch["gt_bbox"]

            pred_anchor, pred_heading, _, _, _, _ = model(
                query_pixels, search_pixels, None
            )

            batch_size = query_pixels.shape[0]
            for i in range(batch_size):
                name = names[i]
                gt_heading = gt_headings[i]
                gt_bbox = gt_bboxes[i]

                pred_anchor_i = pred_anchor[i].unsqueeze(0)
                gt_bbox_i = gt_bbox_tensor[i].unsqueeze(0)

                pred_anchor_i = pred_anchor_i.view(
                    1, 9, 5, pred_anchor_i.shape[2], pred_anchor_i.shape[3]
                )

                _, best_anchor_gi_gj = build_target(
                    gt_bbox_i, anchors_full, 640, pred_anchor_i.shape[3]
                )
                accu_list, accu_center, iou_val, _, pred_bbox, target_bbox = (
                    eval_iou_acc(
                        pred_anchor_i,
                        gt_bbox_i,
                        anchors_full,
                        best_anchor_gi_gj[:, 1],
                        best_anchor_gi_gj[:, 2],
                        640,
                        iou_threshold_list=[0.5, 0.25],
                    )
                )

                pred_heading_i = pred_heading[i].cpu().numpy()
                pred_heading_val = pred_to_heading(pred_heading_i)
                heading_target_i = heading_target[i].unsqueeze(0)
                heading_mse = F.mse_loss(
                    pred_heading[i].unsqueeze(0), heading_target_i
                ).item()
                heading_correct = 1.0 if pred_heading_val == gt_heading else 0.0

                pred = pred_bbox[0].cpu().tolist()
                gt = target_bbox[0].cpu().tolist()

                pred_center = [(pred[0] + pred[2]) / 2, (pred[1] + pred[3]) / 2]
                gt_center = [(gt[0] + gt[2]) / 2, (gt[1] + gt[3]) / 2]
                center_dist = np.sqrt(
                    (pred_center[0] - gt_center[0]) ** 2
                    + (pred_center[1] - gt_center[1]) ** 2
                )

                unified_iou = iou_val if name in success_filenames else 0.0
                accu50_meter.update(1.0 if iou_val >= 0.5 else 0.0, 1)
                accu25_meter.update(1.0 if iou_val >= 0.25 else 0.0, 1)
                iou_meter.update(iou_val, 1)
                center_dist_meter.update(center_dist, 1)
                heading_mse_meter.update(heading_mse, 1)
                heading_acc_meter.update(heading_correct, 1)
                unified_iou_meter.update(unified_iou, 1)

                results.append(
                    {
                        "name": name,
                        "pred_bbox": pred_bbox[0].cpu().tolist(),
                        "gt_bbox": gt_bbox,
                        "iou": iou_val,
                        "center_dist": center_dist,
                        "heading": gt_heading,
                        "pred_heading": pred_heading_val,
                        "heading_mse": heading_mse,
                        "unified_iou": unified_iou,
                    }
                )

                if config.visualize and vis_count < config.vis_limit:
                    vis_path = os.path.join(vis_dir, f"{name}_iou_{iou_val:.3f}.png")
                    visualize_results(
                        query_pixels[i].cpu(),
                        search_pixels[i].cpu(),
                        pred_bbox[0],
                        target_bbox[0],
                        name,
                        iou_val,
                        pred_heading_val,
                        gt_heading,
                        vis_path,
                        processor
                    )
                    vis_count += 1

    metrics = {
        "model_type": "siglip_heading",
        "checkpoint": checkpoint_path,
        "accu50": float(accu50_meter.avg),
        "accu25": float(accu25_meter.avg),
        "mean_iou": float(iou_meter.avg),
        "mean_center_dist": float(center_dist_meter.avg),
        "mean_heading_mse": float(heading_mse_meter.avg),
        "mean_heading_acc": float(heading_acc_meter.avg),
        "mean_unified_iou": float(unified_iou_meter.avg),
        "num_samples": len(results),
    }

    print(f"\n=== SigLIP Heading Evaluation Results ===")
    print(f"Accu50: {metrics['accu50']:.4f}")
    print(f"Accu25: {metrics['accu25']:.4f}")
    print(f"Mean IoU: {metrics['mean_iou']:.4f}")
    print(f"Mean Unified IoU: {metrics['mean_unified_iou']:.4f}")
    print(f"Mean Center Dist: {metrics['mean_center_dist']:.4f}")
    print(f"Mean Heading MSE: {metrics['mean_heading_mse']:.4f}")
    print(f"Mean Heading Acc: {metrics['mean_heading_acc']:.4f}")
    print(f"Samples: {metrics['num_samples']}")

    output_path = os.path.join(config.output_dir, "eval_siglip_heading.json")
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Results saved to {output_path}")

    return metrics


def main():
    parser = argparse.ArgumentParser(description="SigLIP Heading Model Evaluation")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="/data/feihong/ckpt/supp_0.9/last.pth",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/data/feihong/univerisity_dev/eval_results",
        help="Output directory for results",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:2" if torch.cuda.is_available() else "cpu",
        help="Device to use",
    )
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--visualize", action="store_true", help="Visualize results")
    parser.add_argument(
        "--vis_limit", type=int, default=400, help="Max visualizations to save"
    )
    args = parser.parse_args()

    config = EvalConfig(
        model_name="siglip_heading",
        checkpoint_path=args.checkpoint,
        device=args.device,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
        visualize=True,
        vis_limit=args.vis_limit,
    )

    evaluate_model(args.checkpoint, config)


if __name__ == "__main__":
    main()
