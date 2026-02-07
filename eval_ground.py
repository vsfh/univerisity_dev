#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@file eval_ground.py
@desc Unified evaluation and visualization script for all ground-based bbox models.
       Supports: SigLIP, EVA-CLIP, LPN, SMGeo, TROGeoLite
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
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from transformers import AutoImageProcessor, AutoModel, AutoTokenizer
import open_clip
from bbox.yolo_utils import get_tensor_anchors, build_target, eval_iou_acc, bbox_iou
from ground_siglip import Encoder as SigLIPModel
from grounding_cvos.ground_cvos import LPNGeoLite,DetGeoLite,SampleGeoLite,TROGeoLite 
# UNIV_SAT_SIZE = (640, 640)
# UNIV_DRONE_SIZE = (256, 256)

# smgeo
UNIV_SAT_SIZE = (640, 640)
UNIV_DRONE_SIZE = (256, 256)

SAT_ORIG_SIZE = (3840, 2160)


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
    visualize: bool = False
    save_images: bool = False
    vis_limit: int = 50
    preprocess_mode: str = "siglip"


MODEL_REGISTRY = {
    "siglip": {
        "module": "__main__",
        "class": "SigLIPModel",
    },
    "det": {
        "module": "__main__",
        "class": "DetGeoLite",
    },
    "lpn": {
        "module": "__main__",
        "class": "LPNGeoLite",
    },
    "smgeo": {
        "module": "__main__",
        "class": "SampleGeoLite",
    },
    "trogeolite": {
        "module": "__main__",
        "class": "TROGeoLite",
    },
}


def load_model_by_string(model_type: str, checkpoint_path: str, device: str):
    """Load model class dynamically from string and instantiate."""
    import importlib

    if model_type not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model type: {model_type}. Choose from: {list(MODEL_REGISTRY.keys())}"
        )

    registry = MODEL_REGISTRY[model_type]
    module_name = registry["module"]
    class_name = registry["class"]

    if module_name == "__main__":
        model_class = globals()[class_name]
    else:
        module = importlib.import_module(module_name)
        model_class = getattr(module, class_name)
    model = model_class()
    if 'tro' in class_name.lower():
        new_dict = {k.replace('module.',''):v for k,v in torch.load(checkpoint_path, map_location="cpu").items()}
        model.load_state_dict(new_dict)
    else:
        model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    model = model.to(device)
    return model


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


def draw_bbox_on_image(
    image: Image.Image,
    pred_bbox: List[float],
    gt_bbox: List[float],
    pred_color: str = "red",
    gt_color: str = "green",
    pred_label: str = "Pred",
    gt_label: str = "GT",
) -> Image.Image:
    """Draw predicted and ground truth bboxes on image."""
    draw = ImageDraw.Draw(image)
    width, height = image.size

    x1_p, y1_p, x2_p, y2_p = pred_bbox
    x1_g, y1_g, x2_g, y2_g = gt_bbox

    x1_pix = int(x1_p * width)
    y1_pix = int(y1_p * height)
    x2_pix = int(x2_p * width)
    y2_pix = int(y2_p * height)

    x1_g_pix = int(x1_g * width)
    y1_g_pix = int(y1_g * height)
    x2_g_pix = int(x2_g * width)
    y2_g_pix = int(y2_g * height)

    draw.rectangle([x1_pix, y1_pix, x2_pix, y2_pix], outline=pred_color, width=3)
    draw.rectangle([x1_g_pix, y1_g_pix, x2_g_pix, y2_g_pix], outline=gt_color, width=3)

    return image


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Convert CHW tensor (0-1 range) to PIL Image."""
    tensor = tensor.cpu().clamp_(0, 1)
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

    draw.rectangle([x1_pix, y1_pix, x2_pix, y2_pix], outline=pred_color, width=4)
    draw.rectangle([x1_g_pix, y1_g_pix, x2_g_pix, y2_g_pix], outline=gt_color, width=4)

    return image


def visualize_results(
    query_tensor: torch.Tensor,
    search_tensor: torch.Tensor,
    pred_bbox: torch.Tensor,
    gt_bbox: torch.Tensor,
    name: str,
    iou: float,
    output_path: str,
) -> None:
    """Visualize query, search with bboxes, and save. Accepts tensors and unnormalized bboxes."""
    query_image = tensor_to_pil(query_tensor)
    search_image = tensor_to_pil(search_tensor)

    _, search_h, search_w = search_tensor.shape

    pred_bbox_list = pred_bbox.cpu().tolist()
    gt_bbox_list = gt_bbox.cpu().tolist()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(query_image)
    axes[0].set_title(f"Query Image: {name}")
    axes[0].axis("off")

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
    axes[1].imshow(search_draw)
    axes[1].set_title(f"Search Image (IoU: {iou:.3f})")
    axes[1].axis("off")

    overlay = Image.blend(
        search_image.convert("RGBA"),
        Image.new("RGBA", search_image.size, (0, 255, 0, 100)),
        0.3,
    )
    overlay = draw_bbox_pixel(
        overlay,
        pred_bbox_list,
        gt_bbox_list,
        search_w,
        search_h,
        pred_color="red",
        gt_color="green",
    )
    axes[2].imshow(overlay)
    axes[2].set_title("Overlay")
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def get_test_pairs():
    """Get test image pairs from train.txt/test.txt."""
    test_pairs = []
    with open("/data/feihong/ckpt/test.txt", "r") as f:
        for line in f:
            query_path = line.strip()
            name = query_path.split("/")[-2]
            search_path = f"/data/feihong/image_1024/{name}.png"
            if os.path.exists(search_path):
                test_pairs.append((query_path, search_path, name))
    return test_pairs


def load_model(model_type: str, checkpoint_path: str, device: str):
    """Load model based on type using dynamic import."""
    return load_model_by_string(model_type, checkpoint_path, device)


def get_gt_bbox(name: str, test_bbox_dict: Dict) -> List[float]:
    """Get ground truth bbox for a sample."""
    bbox = test_bbox_dict.get(name, None)
    if bbox is None:
        bbox = test_bbox_dict.get(f"{name}.png", None)
    if bbox is None:
        return [0.25, 0.25, 0.75, 0.75]
    x1, y1, x2, y2 = bbox
    return [x1 / 3840, y1 / 2160, x2 / 3840, y2 / 2160]


def preprocess_images_siglip(query_image, search_image, bbox, processor, processor_sat):
    """Preprocess images like ground_siglip.py: use AutoImageProcessor."""
    query_image = resize_drone_image(query_image, UNIV_DRONE_SIZE)
    search_image, pre_bbox = format_satellite_img_bbox(
        search_image, bbox, mode="test", target_size=UNIV_SAT_SIZE
    )
    query_inputs = processor(images=query_image, return_tensors="pt")
    search_inputs = processor_sat(images=search_image, return_tensors="pt")
    return query_inputs["pixel_values"][0], search_inputs["pixel_values"][0], pre_bbox


def preprocess_images_cvos(
    query_image, search_image, bbox, normalize_mean=None, normalize_std=None
):
    """Preprocess images like grounding_cvos/train_wild.py: resize+crop to tensor.

    Args:
        query_image: PIL Image for query (drone)
        search_image: PIL Image for search (satellite)
        bbox: Ground truth bbox coordinates
        normalize_mean: Optional mean for normalization (ImageNet default: [0.485, 0.456, 0.406])
        normalize_std: Optional std for normalization (ImageNet default: [0.229, 0.224, 0.225])
    """
    query_image = resize_drone_image(query_image, UNIV_DRONE_SIZE)
    search_image, pre_bbox = format_satellite_img_bbox(
        search_image, bbox, mode="test", target_size=UNIV_SAT_SIZE
    )
    query_tensor = (
        torch.tensor(np.array(query_image), dtype=torch.float32).permute(2, 0, 1)
        / 255.0
    )
    search_tensor = (
        torch.tensor(np.array(search_image), dtype=torch.float32).permute(2, 0, 1)
        / 255.0
    )

    if normalize_mean is not None and normalize_std is not None:
        if isinstance(normalize_mean, (int, float)):
            normalize_mean = [normalize_mean] * 3
            normalize_std = [normalize_std] * 3
        mean = torch.tensor(normalize_mean, dtype=torch.float32).view(3, 1, 1)
        std = torch.tensor(normalize_std, dtype=torch.float32).view(3, 1, 1)
        query_tensor = (query_tensor - mean) / std
        search_tensor = (search_tensor - mean) / std

    return query_tensor, search_tensor, pre_bbox


def evaluate_model(
    model_type: str,
    checkpoint_path: str,
    config: EvalConfig,
) -> Dict:
    """Evaluate a single model."""
    print(f"\n{'=' * 60}")
    print(f"Evaluating model: {model_type}")
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

    model = load_model(model_type, checkpoint_path, config.device)

    os.makedirs(config.output_dir, exist_ok=True)
    vis_dir = os.path.join(config.output_dir, "visualizations", model_type)
    os.makedirs(vis_dir, exist_ok=True)

    accu50_meter = AverageMeter()
    accu25_meter = AverageMeter()
    iou_meter = AverageMeter()
    center_dist_meter = AverageMeter()

    results = []
    vis_count = 0

    if config.preprocess_mode == "siglip":
        processor = AutoImageProcessor.from_pretrained(
            "google/siglip-base-patch16-224", cache_dir="/data/feihong/hf_cache"
        )
        processor_sat = AutoImageProcessor.from_pretrained(
            "google/siglip-base-patch16-224",
            cache_dir="/data/feihong/hf_cache",
            size={"height": 640, "width": 640},
        )

    for query_path, search_path, name in tqdm(
        test_pairs, desc=f"Evaluating {model_type}"
    ):
        query_image = Image.open(query_path).convert("RGB")
        search_image = Image.open(search_path).convert("RGB")

        gt_bbox = get_gt_bbox(name, test_bbox_dict)
        bbox_pixel = [
            gt_bbox[0] * SAT_ORIG_SIZE[0],
            gt_bbox[1] * SAT_ORIG_SIZE[1],
            gt_bbox[2] * SAT_ORIG_SIZE[0],
            gt_bbox[3] * SAT_ORIG_SIZE[1],
        ]

        if config.preprocess_mode == "siglip":
            query_tensor, search_tensor, pre_bbox = preprocess_images_siglip(
                query_image, search_image, bbox_pixel, processor, processor_sat
            )
        else:
            query_tensor, search_tensor, pre_bbox = preprocess_images_cvos(
                query_image, search_image, bbox_pixel,
                # normalize_mean=[0.485, 0.456, 0.406],
                # normalize_std=[0.229, 0.224, 0.225],
            )

        query_tensor = query_tensor.unsqueeze(0).to(config.device)
        search_tensor = search_tensor.unsqueeze(0).to(config.device)
        gt_bbox_tensor = torch.tensor(
            [pre_bbox], dtype=torch.float32, device=config.device
        )
        with torch.no_grad():
            output = model.bbox_forward(query_tensor, search_tensor)

        if isinstance(output, torch.Tensor):
            pred_bbox = search_tensor.shape[-1] * output
            target_bbox = gt_bbox_tensor
            iou_val = float(bbox_iou(pred_bbox, gt_bbox_tensor))
        else:
            pred_anchor = output[0]

            anchors_full = get_tensor_anchors(config.device)
            pred_anchor = pred_anchor.view(
                1, 9, 5, pred_anchor.shape[2], pred_anchor.shape[3]
            )

            _, best_anchor_gi_gj = build_target(
                gt_bbox_tensor, anchors_full, 640, pred_anchor.shape[3]
            )
            accu_list, accu_center, iou_val, _, pred_bbox, target_bbox = eval_iou_acc(
                pred_anchor,
                gt_bbox_tensor,
                anchors_full,
                best_anchor_gi_gj[:, 1],
                best_anchor_gi_gj[:, 2],
                640,
                iou_threshold_list=[0.5, 0.25],
            )

        pred = pred_bbox[0].cpu().tolist()
        gt = target_bbox[0].cpu().tolist()

        pred_center = [(pred[0] + pred[2]) / 2, (pred[1] + pred[3]) / 2]
        gt_center = [(gt[0] + gt[2]) / 2, (gt[1] + gt[3]) / 2]
        center_dist = np.sqrt(
            (pred_center[0] - gt_center[0]) ** 2 + (pred_center[1] - gt_center[1]) ** 2
        )
        accu50_meter.update(1.0 if iou_val >= 0.5 else 0.0, 1)
        accu25_meter.update(1.0 if iou_val >= 0.25 else 0.0, 1)
        iou_meter.update(iou_val, 1)
        center_dist_meter.update(center_dist, 1)

        results.append(
            {
                "name": name,
                "pred_bbox": pred_bbox,
                "gt_bbox": gt_bbox,
                "iou": iou_val,
                "center_dist": center_dist,
            }
        )

        if config.visualize and vis_count < config.vis_limit:
            vis_path = os.path.join(vis_dir, f"{name}_iou_{iou_val:.3f}.png")
            visualize_results(
                query_tensor.squeeze(0),
                search_tensor.squeeze(0),
                pred_bbox[0],
                target_bbox[0],
                name,
                iou_val,
                vis_path,
            )
            vis_count += 1

    metrics = {
        "model_type": model_type,
        "checkpoint": checkpoint_path,
        "accu50": float(accu50_meter.avg),
        "accu25": float(accu25_meter.avg),
        "mean_iou": float(iou_meter.avg),
        "mean_center_dist": float(center_dist_meter.avg),
        "num_samples": len(results),
    }

    print(f"\n=== {model_type.upper()} Evaluation Results ===")
    print(f"Accu50: {metrics['accu50']:.4f}")
    print(f"Accu25: {metrics['accu25']:.4f}")
    print(f"Mean IoU: {metrics['mean_iou']:.4f}")
    print(f"Mean Center Dist: {metrics['mean_center_dist']:.4f}")
    print(f"Samples: {metrics['num_samples']}")

    output_path = os.path.join(config.output_dir, f"eval_{model_type}.json")
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Results saved to {output_path}")

    return metrics


def compare_all_models(checkpoints: Dict[str, str], config: EvalConfig):
    """Evaluate all models and compare."""
    all_metrics = []

    for model_type, checkpoint_path in checkpoints.items():
        if not os.path.exists(checkpoint_path):
            print(f"Checkpoint not found: {checkpoint_path}, skipping...")
            continue
        metrics = evaluate_model(model_type, checkpoint_path, config)
        if metrics:
            all_metrics.append(metrics)

    print("\n" + "=" * 80)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 80)
    print(
        f"{'Model':<15} {'Accu50':<10} {'Accu25':<10} {'Mean IoU':<10} {'Center Dist':<12} {'Samples':<10}"
    )
    print("-" * 80)

    for m in all_metrics:
        print(
            f"{m['model_type']:<15} {m['accu50']:<10.4f} {m['accu25']:<10.4f} "
            f"{m['mean_iou']:<10.4f} {m['mean_center_dist']:<12.4f} {m['num_samples']:<10}"
        )

    best_iou_model = max(all_metrics, key=lambda x: x["mean_iou"])
    print(
        f"\nBest Model (by IoU): {best_iou_model['model_type']} ({best_iou_model['mean_iou']:.4f})"
    )

    summary_path = os.path.join(config.output_dir, "comparison_summary.json")
    with open(summary_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"Comparison saved to {summary_path}")


def main():
    parser = argparse.ArgumentParser(description="Unified Ground Model Evaluation")
    parser.add_argument(
        "--model",
        type=str,
        default="trogeolite",
        choices=["siglip", "det", "lpn", "smgeo", "trogeolite", "all"],
        help="Model to evaluate",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="/data/feihong/ckpt/ground_cvos/best.pth",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./eval_results",
        help="Output directory for results",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:1" if torch.cuda.is_available() else "cpu",
        help="Device to use",
    )
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument(
        "--preprocess_mode",
        type=str,
        default="cvos",
        choices=["siglip", "cvos"],
        help="Preprocessing mode: siglip (use AutoImageProcessor) or cvos (resize+crop to tensor)",
    )
    parser.add_argument("--visualize", action="store_true", help="Visualize results")
    parser.add_argument(
        "--vis_limit", type=int, default=50, help="Max visualizations to save"
    )
    args = parser.parse_args()

    config = EvalConfig(
        model_name=args.model,
        checkpoint_path=args.checkpoint,
        device=args.device,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
        visualize=True,
        vis_limit=args.vis_limit,
        preprocess_mode=args.preprocess_mode,
    )

    default_checkpoints = {
        "siglip": "/data/feihong/ckpt/ground_siglip/best.pth",
        "det": "/data/feihong/ckpt/ground_det/best.pth",
        "lpn": "/data/feihong/ckpt/ground_lpn/best.pth",
        "smgeo": "/data/feihong/ckpt/ground_sample/best.pth",
        "trogeolite": "/data/feihong/ckpt/ground_cvos/best.pth",
    }

    if args.model == "all":
        compare_all_models(default_checkpoints, config)
    else:
        checkpoint = args.checkpoint or default_checkpoints.get(args.model)
        if not checkpoint:
            print(f"No checkpoint specified for {args.model}")
            return
        evaluate_model(args.model, checkpoint, config)


if __name__ == "__main__":
    main()
