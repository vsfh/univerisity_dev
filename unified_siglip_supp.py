# --- Configuration ---
import argparse
import os
import json
import random
import time
from contextlib import nullcontext
from glob import glob
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
from torch.backends import cudnn

import numpy as np
from PIL import Image
from tqdm import tqdm
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from transformers import AutoImageProcessor, AutoTokenizer
import yaml

from model import Encoder_heat, Encoder_test
from bbox.yolo_utils import (
    bbox_iou,
    get_tensor_anchors,
    build_target,
    yolo_loss,
    eval_iou_acc,
)
from dataset import ShiftedSatelliteDroneDataset
from hf_cache_utils import from_pretrained_prefer_local

cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


# --- Configuration ---
class Config:
    CLEARML_ENABLED = False
    CLEARML_PROJECT = "unified_siglip"
    CLEARML_TASK_NAME = None
    MODEL_NAME = "google/siglip2-base-patch16-224"
    CACHE_DIR = "/media/data1/feihong/hf_cache"
    DRONE_VIEW_FOLDER = "/media/data1/feihong/drone_view"
    IMAGE_FOLDER = "/media/data1/feihong/image_1024"
    HEADING_FOLDER = "/media/data1/feihong/range_250"
    TEXT_FILE = "/media/data1/feihong/ckpt/drone_text_single_long.json"
    TEXT_JSON_NAME = "udes_siglip2_gemma4.json"
    TRAIN_BBOX_FILE = "/media/data1/feihong/univerisity_dev/runs/train.json"
    TEST_BBOX_FILE = "/media/data1/feihong/univerisity_dev/runs/test.json"
    SATELLITE_FOLDER = "/media/data1/feihong/asian_univ"

    SAT_ORIG_SIZE = (3840, 2160)
    UNIV_SAT_SIZE = {"height": 432, "width": 768}
    DRONE_SIZE = (256, 256)

    NUM_EPOCHS = 16
    BATCH_SIZE = 16
    GRAD_ACCUMULATION_STEPS = 2
    LEARNING_RATE = 5e-5
    GRAD_CLIP_NORM = 1.0
    LR_MIN = 1e-10
    COSINE_EPOCHS = NUM_EPOCHS
    BBOX_LOSS_WEIGHT = 0.5
    HEADING_LOSS_WEIGHT = 0.01
    PROJECTION_DIM = 768

    NUM_WORKERS_TRAIN = 8
    NUM_WORKERS_VAL = 4
    NUM_WORKERS_EVAL = 8
    PIN_MEMORY = True
    PERSISTENT_WORKERS = False
    PREFETCH_FACTOR = 4
    DROP_LAST_TRAIN = True

    USE_AMP = True
    ENABLE_TF32 = True
    USE_ANGLE_INPUT = True
    USE_TEXT_INPUT = False
    ENCODER_TYPE = "heat"  # heat | test
    LORA_RANK = 8
    LORA_ALPHA = 16.0
    LORA_DROPOUT = 0.05
    OPTIMIZE_OBJECTIVE = "combined"  # combined | img_text_only | bbox_only
    USE_HEATMAP_LOSS = True
    HEATMAP_LOSS_WEIGHT = 0.2
    HEATMAP_LOSS_TYPE = [ "focal"]  # mse | cross_entropy | weighted_bce | focal
    HEATMAP_FOCAL_GAMMA = 2.0
    HEATMAP_FOCAL_ALPHA = 0.25
    HEATMAP_FOCAL_BETA = 0.75
    HEATMAP_TARGET_MODE = "bbox_center"  # center | bbox_aware | bbox_center
    HEATMAP_SIGMA = 1.5
    HEATMAP_RADIUS = 4.5
    HEATMAP_BBOX_INSIDE_VALUE = 0.5
    HEATMAP_BBOX_OUTSIDE_VALUE = 0.2
    HEATMAP_BBOX_CENTER_EDGE_VALUE = 0.2
    HEATMAP_BBOX_CENTER_LOG_SCALE = 9.0
    HEATMAP_POS_WEIGHT = 8.0
    HEATMAP_CONFIDENCE_WEIGHT = 0.5
    HEATMAP_VIS_MAX_SAMPLES = 10
    USE_REFINE_LOSS = False
    REFINE_LOSS_WEIGHT = 0.25
    REFINE_SCORE_LOSS_WEIGHT = 0.2
    USE_TEXT_GROUNDING_PATH = False
    USE_TEXT_ANCHOR_LOSS = False
    TEXT_ANCHOR_LOSS_WEIGHT = 0.5
    TEXT_POOLER_ALIGN_START_WEIGHT = 0.05
    TEXT_POOLER_ALIGN_END_WEIGHT = 0.005
    TEXT_POOLER_ALIGN_DECAY_EPOCHS = 10
    TEXT_POOLER_ALIGN_STOP_EPOCH = 10


def config_to_dict() -> Dict[str, Any]:
    return {
        key: getattr(Config, key)
        for key in dir(Config)
        if key.isupper() and not key.startswith("__")
    }


def apply_config_overrides(overrides: Optional[Dict[str, Any]]) -> None:
    if not overrides:
        return

    unknown_keys = [key for key in overrides if not hasattr(Config, key)]
    if unknown_keys:
        raise KeyError(
            "Unknown Config override(s): "
            + ", ".join(sorted(str(key) for key in unknown_keys))
        )

    for key, value in overrides.items():
        current_value = getattr(Config, key)
        if isinstance(current_value, tuple) and isinstance(value, list):
            value = tuple(value)
        setattr(Config, key, value)

    if "NUM_EPOCHS" in overrides and "COSINE_EPOCHS" not in overrides:
        Config.COSINE_EPOCHS = Config.NUM_EPOCHS


def load_experiment_from_yaml(
    yaml_path: Optional[str],
    experiment_name: Optional[str],
) -> Dict[str, Any]:
    if not yaml_path:
        return {}

    with open(yaml_path, "r", encoding="utf-8") as f:
        payload = yaml.safe_load(f) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"YAML root must be a mapping: {yaml_path}")

    defaults = payload.get("defaults", {}) or {}
    if not isinstance(defaults, dict):
        raise ValueError("YAML 'defaults' must be a mapping when provided.")

    experiments = payload.get("experiments")
    if experiments is None:
        selected = payload
    else:
        if not isinstance(experiments, list):
            raise ValueError("YAML 'experiments' must be a list.")
        if not experiments:
            raise ValueError("YAML 'experiments' is empty.")

        if experiment_name is None:
            if len(experiments) != 1:
                raise ValueError(
                    "YAML contains multiple experiments; pass --experiment NAME."
                )
            selected = experiments[0]
        else:
            matches = [
                item
                for item in experiments
                if str(item.get("name", item.get("exp_name", ""))) == experiment_name
            ]
            if not matches:
                raise ValueError(
                    f"Experiment '{experiment_name}' not found in {yaml_path}."
                )
            selected = matches[0]

    if not isinstance(selected, dict):
        raise ValueError("Selected experiment must be a mapping.")

    merged = dict(defaults)
    merged.update(selected)

    default_config = defaults.get("config", {}) or {}
    selected_config = selected.get("config", {}) or {}
    if not isinstance(default_config, dict) or not isinstance(selected_config, dict):
        raise ValueError("'config' entries in YAML must be mappings.")
    merged["config"] = {**default_config, **selected_config}
    return merged


# --- Utility Functions ---
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


def _to_clearml_serializable(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(k): _to_clearml_serializable(v)
            for k, v in value.items()
        }
    if isinstance(value, (list, tuple, set)):
        return [_to_clearml_serializable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def build_dataloader_kwargs(num_workers: int, drop_last: bool = False) -> Dict:
    kwargs = {
        "num_workers": num_workers,
        "pin_memory": Config.PIN_MEMORY,
        "drop_last": drop_last,
        "persistent_workers": Config.PERSISTENT_WORKERS and num_workers > 0,
    }
    if num_workers > 0:
        kwargs["prefetch_factor"] = Config.PREFETCH_FACTOR
    return kwargs


def build_optimizer(model: nn.Module) -> AdamW:
    trainable_params = [param for param in model.parameters() if param.requires_grad]
    if not trainable_params:
        raise ValueError("No trainable parameters found for optimizer.")
    return AdamW(trainable_params, lr=Config.LEARNING_RATE)


def text_pooler_align_weight(epoch: int) -> float:
    start_weight = float(Config.TEXT_POOLER_ALIGN_START_WEIGHT)
    end_weight = float(Config.TEXT_POOLER_ALIGN_END_WEIGHT)
    if start_weight <= 0 and end_weight <= 0:
        return 0.0

    epoch_number = int(epoch) + 1
    stop_epoch = max(1, int(Config.TEXT_POOLER_ALIGN_STOP_EPOCH))
    if epoch_number > stop_epoch:
        return 0.0
    decay_epochs = max(1, min(int(Config.TEXT_POOLER_ALIGN_DECAY_EPOCHS), stop_epoch))
    if epoch_number >= decay_epochs:
        return end_weight

    progress = float(epoch_number - 1) / float(decay_epochs - 1)
    return start_weight + (end_weight - start_weight) * progress


def text_pooler_align_detach_image(epoch: int) -> bool:
    return False


def effective_model_name() -> str:
    return Config.MODEL_NAME


def build_geo_features(batch: Dict, device: torch.device) -> Optional[torch.Tensor]:
    if not Config.USE_ANGLE_INPUT:
        return None

    angles = batch["angle"].to(device, non_blocking=True).float()
    angles_rad = torch.deg2rad(angles)
    height = batch["height"].to(device, non_blocking=True).float()
    return torch.cat(
        [
            torch.cos(angles_rad)[..., None],
            torch.sin(angles_rad)[..., None],
            height[..., None] / 300.0,
        ],
        dim=1,
    )


def unpack_model_outputs(outputs: Tuple) -> Tuple[
    torch.Tensor,
    Any,
    Optional[torch.Tensor],
    torch.Tensor,
    torch.Tensor,
    Any,
    Optional[torch.Tensor],
]:
    if len(outputs) != 7:
        raise ValueError(f"Expected 7 model outputs, got {len(outputs)}.")
    return outputs


def build_heatmap_target(
    target_bbox: torch.Tensor,
    heatmap_hw: Tuple[int, int],
    image_wh: Tuple[int, int],
    sigma: float,
    radius: Optional[float] = None,
) -> torch.Tensor:
    grid_h, grid_w = int(heatmap_hw[0]), int(heatmap_hw[1])
    image_w, image_h = float(image_wh[0]), float(image_wh[1])
    device = target_bbox.device
    dtype = target_bbox.dtype

    x1 = torch.minimum(target_bbox[:, 0], target_bbox[:, 2]).clamp(0.0, image_w)
    y1 = torch.minimum(target_bbox[:, 1], target_bbox[:, 3]).clamp(0.0, image_h)
    x2 = torch.maximum(target_bbox[:, 0], target_bbox[:, 2]).clamp(0.0, image_w)
    y2 = torch.maximum(target_bbox[:, 1], target_bbox[:, 3]).clamp(0.0, image_h)

    center_x = (target_bbox[:, 0] + target_bbox[:, 2]) * 0.5 / max(image_w, 1.0) * grid_w
    center_y = (target_bbox[:, 1] + target_bbox[:, 3]) * 0.5 / max(image_h, 1.0) * grid_h
    center_x = center_x.clamp(0.0, grid_w - 1e-6)
    center_y = center_y.clamp(0.0, grid_h - 1e-6)

    ys = torch.arange(grid_h, device=device, dtype=dtype).view(1, grid_h, 1)
    xs = torch.arange(grid_w, device=device, dtype=dtype).view(1, 1, grid_w)
    dx2 = (xs - center_x.view(-1, 1, 1)) ** 2
    dy2 = (ys - center_y.view(-1, 1, 1)) ** 2
    dist2 = dx2 + dy2
    sigma = max(float(sigma), 1e-6)
    center_target = torch.exp(-dist2 / (2.0 * sigma**2))
    if radius is not None and float(radius) > 0.0:
        center_target = center_target * (dist2 <= float(radius) ** 2).to(dtype=dtype)

    if Config.HEATMAP_TARGET_MODE == "center":
        target = center_target
    elif Config.HEATMAP_TARGET_MODE == "bbox_aware":
        x_img = (xs + 0.5) / max(float(grid_w), 1.0) * image_w
        y_img = (ys + 0.5) / max(float(grid_h), 1.0) * image_h

        inside_x = (x_img >= x1.view(-1, 1, 1)) & (x_img <= x2.view(-1, 1, 1))
        inside_y = (y_img >= y1.view(-1, 1, 1)) & (y_img <= y2.view(-1, 1, 1))
        inside = (inside_x & inside_y).to(dtype=dtype)

        dx_out = torch.maximum(
            torch.maximum(x1.view(-1, 1, 1) - x_img, x_img - x2.view(-1, 1, 1)),
            torch.zeros((), device=device, dtype=dtype),
        )
        dy_out = torch.maximum(
            torch.maximum(y1.view(-1, 1, 1) - y_img, y_img - y2.view(-1, 1, 1)),
            torch.zeros((), device=device, dtype=dtype),
        )
        dx_out = dx_out / max(image_w, 1.0) * grid_w
        dy_out = dy_out / max(image_h, 1.0) * grid_h
        outside_dist2 = dx_out**2 + dy_out**2
        outside_band = torch.exp(-outside_dist2 / (2.0 * sigma**2)) * (1.0 - inside)
        if radius is not None and float(radius) > 0.0:
            outside_band = outside_band * (
                outside_dist2 <= float(radius) ** 2
            ).to(dtype=dtype)

        inside_target = inside * float(Config.HEATMAP_BBOX_INSIDE_VALUE)
        outside_target = outside_band * float(Config.HEATMAP_BBOX_OUTSIDE_VALUE)
        target = torch.maximum(center_target, torch.maximum(inside_target, outside_target))
    elif Config.HEATMAP_TARGET_MODE == "bbox_center":
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
        log_scale = max(float(Config.HEATMAP_BBOX_CENTER_LOG_SCALE), 1e-6)
        decay = torch.log1p(norm_radius * log_scale) / np.log1p(log_scale)
        edge_value = float(Config.HEATMAP_BBOX_CENTER_EDGE_VALUE)
        target = 1.0 - (1.0 - edge_value) * decay
        target = target * inside.to(dtype=dtype)

        center_x_idx = torch.floor(center_x).long().clamp(0, grid_w - 1)
        center_y_idx = torch.floor(center_y).long().clamp(0, grid_h - 1)
        batch_idx = torch.arange(target.shape[0], device=device)
        target[batch_idx, center_y_idx, center_x_idx] = 1.0
    else:
        raise ValueError(
            f"Invalid HEATMAP_TARGET_MODE={Config.HEATMAP_TARGET_MODE}. "
            "Choose 'center', 'bbox_aware', or 'bbox_center'."
        )
    return target.unsqueeze(1).clamp(0.0, 1.0)


def parse_heatmap_loss_types(loss_type: Any) -> List[str]:
    if isinstance(loss_type, str):
        loss_types = loss_type.replace(",", "+").split("+")
    elif isinstance(loss_type, (list, tuple, set)):
        loss_types = []
        for item in loss_type:
            loss_types.extend(str(item).replace(",", "+").split("+"))
    else:
        raise TypeError(
            f"Invalid HEATMAP_LOSS_TYPE={loss_type!r}. Use a string or a list."
        )

    normalized = []
    aliases = {
        "ce": "cross_entropy",
        "spatial_ce": "cross_entropy",
        "spatial_cross_entropy": "cross_entropy",
        "smgeo": "focal",
        "focal_loss": "focal",
    }
    for item in loss_types:
        item = str(item).strip().lower()
        if not item:
            continue
        normalized.append(aliases.get(item, item))

    if not normalized:
        raise ValueError("HEATMAP_LOSS_TYPE must contain at least one loss type.")
    return normalized


def heatmap_loss_fn(
    heatmap_logits: torch.Tensor,
    target_bbox: torch.Tensor,
    return_target: bool = False,
):
    heatmap_target = build_heatmap_target(
        target_bbox=target_bbox,
        heatmap_hw=heatmap_logits.shape[-2:],
        image_wh=(Config.UNIV_SAT_SIZE["width"], Config.UNIV_SAT_SIZE["height"]),
        sigma=Config.HEATMAP_SIGMA,
        radius=Config.HEATMAP_RADIUS,
    )
    target_prob = heatmap_logits.float().flatten(1).clamp_min(1e-6)
    pred_prob = heatmap_target.float().flatten(1)
    # target_prob = target / target.sum(dim=1, keepdim=True).clamp_min(1e-6)
    # pred_prob = pred / pred.sum(dim=1, keepdim=True).clamp_min(1e-6)

    loss = heatmap_logits.new_zeros(())
    for loss_type in parse_heatmap_loss_types(Config.HEATMAP_LOSS_TYPE):
        if loss_type == "mse":
            loss = loss + F.mse_loss(
                pred_prob,
                target_prob,
                reduction="mean",
            ) * pred_prob.shape[1]
            continue

        if loss_type == "cross_entropy":
            loss = loss + -(
                target_prob * pred_prob.clamp_min(1e-8).log()
            ).sum(dim=1).mean()
            continue

        if loss_type == "weighted_bce":
            bce_loss = F.binary_cross_entropy(
                heatmap_logits.float().clamp(1e-6, 1.0 - 1e-6),
                heatmap_target.float(),
                reduction="none",
            )
            weight = 1.0 + heatmap_target.float() * (
                Config.HEATMAP_POS_WEIGHT - 1.0
            )
            loss = loss + (bce_loss * weight).mean()
            continue

        if loss_type == "focal":
            pred_sigmoid = torch.sigmoid(heatmap_logits.float().flatten(1)).clamp(1e-4, 1.0 - 1e-4)
            soft_target = heatmap_target.float().flatten(1)
            gamma = float(Config.HEATMAP_FOCAL_GAMMA)

            pos_loss = (
                -torch.log(pred_sigmoid)
                * (1.0 - pred_sigmoid).pow(gamma)
                * soft_target
            ).sum(1) / soft_target.sum(1).clamp_min(1e-6)
            neg_loss = (
                -torch.log(1.0 - pred_sigmoid)
                * pred_sigmoid.pow(gamma)
                * (1.0 - soft_target)
            ).sum(1) / (1.0 - soft_target).sum(1).clamp_min(1e-6)
            loss = loss + (pos_loss * 15 + neg_loss * 45).mean()
            continue

        raise ValueError(
            f"Invalid HEATMAP_LOSS_TYPE={Config.HEATMAP_LOSS_TYPE}. "
            "Choose from 'mse', 'cross_entropy', 'spatial_ce', 'weighted_bce', or 'focal'."
        )

    if return_target:
        return loss, heatmap_target
    return loss


def refine_bbox_loss_fn(
    refine_outputs: Any,
    target_bbox: torch.Tensor,
    image_wh: Tuple[int, int],
) -> torch.Tensor:
    if not isinstance(refine_outputs, dict):
        return target_bbox.new_zeros(())
    if "candidates" not in refine_outputs or "logits" not in refine_outputs:
        raise ValueError("refine_outputs must contain 'candidates' and 'logits'.")

    candidates = refine_outputs["candidates"]
    logits = refine_outputs["logits"]
    if candidates.ndim != 3 or candidates.shape[-1] != 4:
        raise ValueError(f"Expected refine candidates shape (B, K, 4), got {tuple(candidates.shape)}.")
    if logits.ndim != 2 or logits.shape != candidates.shape[:2]:
        raise ValueError(f"Expected refine logits shape {tuple(candidates.shape[:2])}, got {tuple(logits.shape)}.")
    if target_bbox.ndim != 2 or target_bbox.shape != (candidates.shape[0], 4):
        raise ValueError(f"Expected target bbox shape {(candidates.shape[0], 4)}, got {tuple(target_bbox.shape)}.")

    batch_size, topk, _ = candidates.shape
    image_w, image_h = image_wh
    norm = candidates.new_tensor([float(image_w), float(image_h), float(image_w), float(image_h)])
    candidates_norm = candidates / norm
    target_norm = target_bbox.to(device=candidates.device, dtype=candidates.dtype) / norm
    target_expand = target_norm.unsqueeze(1).expand(-1, topk, -1)

    l1 = F.smooth_l1_loss(candidates_norm, target_expand, reduction="none").mean(dim=-1)
    flat_candidates = candidates.reshape(batch_size * topk, 4)
    flat_target = target_bbox.to(device=candidates.device, dtype=candidates.dtype).unsqueeze(1).expand(-1, topk, -1)
    iou = bbox_iou(flat_candidates, flat_target.reshape(batch_size * topk, 4), x1y1x2y2=True)
    candidate_loss = l1 + (1.0 - iou.view(batch_size, topk))

    best_idx = candidate_loss.detach().argmin(dim=1)
    batch_idx = torch.arange(batch_size, device=candidates.device)
    selected_loss = candidate_loss[batch_idx, best_idx].mean()
    score_loss = F.cross_entropy(logits.float(), best_idx)
    return selected_loss + float(Config.REFINE_SCORE_LOSS_WEIGHT) * score_loss


def add_heatmap_to_confidence(
    pred_anchor: torch.Tensor,
    heatmap_logits: Optional[torch.Tensor],
) -> torch.Tensor:
    if (
        heatmap_logits is None
        or Config.ENCODER_TYPE not in {"heat", "test"}
        or Config.HEATMAP_CONFIDENCE_WEIGHT <= 0.0
    ):
        return pred_anchor

    if heatmap_logits.shape[-2:] != pred_anchor.shape[-2:]:
        heatmap_logits = F.interpolate(
            heatmap_logits,
            size=pred_anchor.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

    heatmap_logits = torch.sigmoid(heatmap_logits.detach()).to(dtype=pred_anchor.dtype)
    heat_confidence = Config.HEATMAP_CONFIDENCE_WEIGHT * heatmap_logits.unsqueeze(1)
    return torch.cat(
        [
            pred_anchor[:, :, :4, :, :],
            pred_anchor[:, :, 4:5, :, :] + heat_confidence,
        ],
        dim=2,
    )


def _normalize_heatmap_for_writer(heatmap: torch.Tensor) -> torch.Tensor:
    heatmap = heatmap.detach().float()
    heatmap_min = heatmap.amin(dim=(-2, -1), keepdim=True)
    heatmap_max = heatmap.amax(dim=(-2, -1), keepdim=True)
    return (heatmap - heatmap_min) / (heatmap_max - heatmap_min).clamp_min(1e-6)


def _colorize_heatmap_for_writer(heatmap: torch.Tensor) -> torch.Tensor:
    heatmap = heatmap.clamp(0.0, 1.0)
    red = heatmap
    green = torch.zeros_like(heatmap)
    blue = 1.0 - heatmap
    return torch.cat([red, green, blue], dim=1)


def log_heatmap_images(
    writer: Optional[SummaryWriter],
    heatmap_logits: Optional[torch.Tensor],
    target_bbox: Optional[torch.Tensor],
    epoch: int,
    heatmap_target: Optional[torch.Tensor] = None,
    tag_prefix: str = "Heatmap",
) -> None:
    if writer is None or heatmap_logits is None:
        return

    if heatmap_target is None and target_bbox is None:
        return

    sample_count = heatmap_logits.shape[0]
    if heatmap_target is not None:
        sample_count = min(sample_count, heatmap_target.shape[0])
    max_samples = min(Config.HEATMAP_VIS_MAX_SAMPLES, sample_count)
    if max_samples <= 0:
        return

    pred_heatmap = heatmap_logits[:max_samples]
    if heatmap_target is None:
        target_bbox = target_bbox[:max_samples]
        target_heatmap = build_heatmap_target(
            target_bbox=target_bbox,
            heatmap_hw=pred_heatmap.shape[-2:],
            image_wh=(Config.UNIV_SAT_SIZE["width"], Config.UNIV_SAT_SIZE["height"]),
            sigma=Config.HEATMAP_SIGMA,
            radius=Config.HEATMAP_RADIUS,
        )
    else:
        target_heatmap = heatmap_target[:max_samples]

    pred_vis = _colorize_heatmap_for_writer(
        _normalize_heatmap_for_writer(pred_heatmap)
    ).cpu()
    target_vis = _colorize_heatmap_for_writer(
        target_heatmap.detach().float()
    ).cpu()
    target_range_vis = _colorize_heatmap_for_writer(
        _normalize_heatmap_for_writer(target_heatmap)
    ).cpu()
    side_by_side = torch.cat([pred_vis, target_vis], dim=-1)

    writer.add_images(f"{tag_prefix}/pred", pred_vis, epoch)
    writer.add_images(f"{tag_prefix}/target", target_vis, epoch)
    writer.add_images(f"{tag_prefix}/target_range_vis", target_range_vis, epoch)
    writer.add_images(f"{tag_prefix}/pred_target", side_by_side, epoch)


def build_encoder(use_ap: bool, usesg: bool = True) -> nn.Module:
    if Config.ENCODER_TYPE == "heat":
        return Encoder_heat(
            model_name=Config.MODEL_NAME,
            proj_dim=Config.PROJECTION_DIM,
            usesg=usesg,
            useap=use_ap,
            use_heatmap=Config.USE_HEATMAP_LOSS,
            lora_rank=Config.LORA_RANK,
            lora_alpha=Config.LORA_ALPHA,
            lora_dropout=Config.LORA_DROPOUT,
        )
    if Config.ENCODER_TYPE == "test":
        return Encoder_test(
            model_name=Config.MODEL_NAME,
            proj_dim=Config.PROJECTION_DIM,
            usesg=usesg,
            useap=use_ap,
            use_heatmap=Config.USE_HEATMAP_LOSS,
            lora_rank=Config.LORA_RANK,
            lora_alpha=Config.LORA_ALPHA,
            lora_dropout=Config.LORA_DROPOUT,
            use_text_grounding_path=Config.USE_TEXT_GROUNDING_PATH,
        )
    raise ValueError(
        f"Invalid ENCODER_TYPE={Config.ENCODER_TYPE}. Choose 'heat' or 'test'."
    )


def visualize_batch(
    batch: Dict, save_dir: str = "runs/visualizations", max_samples: int = 6, sat_size: Tuple[int, int] = (Config.UNIV_SAT_SIZE["width"], Config.UNIV_SAT_SIZE["height"])
) -> None:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    os.makedirs(save_dir, exist_ok=True)

    target_pixels = batch["target_pixel_values"]
    search_pixels = batch["search_pixel_values"]
    bboxes = batch.get("bbox", None)
    names = batch.get("name", [f"sample_{i}" for i in range(target_pixels.shape[0])])

    for i in range(min(target_pixels.shape[0], max_samples)):
        target_img = target_pixels[i].cpu().numpy()
        target_img = np.transpose(target_img, (1, 2, 0))
        target_img = (target_img * 0.5 + 0.5).clip(0, 1)

        search_img = search_pixels[i].cpu().numpy()
        search_img = np.transpose(search_img, (1, 2, 0))
        search_img = (search_img * 0.5 + 0.5).clip(0, 1)

        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        fig.suptitle(f"Query: {names[i]}", fontsize=14, fontweight="bold")

        axes[0].imshow(target_img)
        axes[0].set_title("Target (Drone View)", fontsize=12)
        axes[0].axis("off")

        axes[1].imshow(search_img)
        axes[1].set_title("Search (Satellite View)", fontsize=12)

        if bboxes is not None:
            bbox = bboxes[i].detach().cpu().float().tolist()
            x1, y1, x2, y2 = bbox

            h, w = search_img.shape[0], search_img.shape[1]
            sx = w / float(sat_size[0])
            sy = h / float(sat_size[1])

            x1, x2 = x1 * sx, x2 * sx
            y1, y2 = y1 * sy, y2 * sy

            x1 = max(0.0, min(x1, w - 1))
            y1 = max(0.0, min(y1, h - 1))
            x2 = max(0.0, min(x2, w - 1))
            y2 = max(0.0, min(y2, h - 1))

            rect = patches.Rectangle(
                (x1, y1),
                max(1.0, x2 - x1),
                max(1.0, y2 - y1),
                linewidth=2,
                edgecolor="lime",
                facecolor="none",
            )
            axes[1].add_patch(rect)

        axes[1].axis("off")

        plt.tight_layout()
        plt.savefig(f"{save_dir}/{names[i]}_combined.png", dpi=150, bbox_inches="tight")
        plt.close()

    print(
        f"Saved {min(target_pixels.shape[0], max_samples)} visualizations to {save_dir}"
    )



# --- Loss Functions ---
def info_nce_loss(
    query_feats: torch.Tensor,
    candidate_feats: torch.Tensor,
    positive_indices: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    query_feats = F.normalize(query_feats, p=2, dim=1)
    candidate_feats = F.normalize(candidate_feats, p=2, dim=1)
    sim_matrix = torch.matmul(query_feats, candidate_feats.T) / temperature
    return F.cross_entropy(sim_matrix, positive_indices)


# --- Validation ---
def validate(
    loader: DataLoader,
    model: nn.Module,
    accelerator: Accelerator,
    anchors_full: torch.Tensor,
    img_size: Tuple[int, int],
    useap: bool = True,
) -> Tuple[float, float, float, float]:
    model.eval()

    accu50_sum = torch.tensor(0.0, device=accelerator.device)
    accu25_sum = torch.tensor(0.0, device=accelerator.device)
    iou_sum = torch.tensor(0.0, device=accelerator.device)
    info_nce_sum = torch.tensor(0.0, device=accelerator.device)
    sample_count = torch.tensor(0.0, device=accelerator.device)

    amp_enabled = Config.USE_AMP and accelerator.device.type == "cuda"

    for batch in tqdm(loader, desc="Validating", disable=not accelerator.is_main_process):
        query_imgs = batch["target_pixel_values"].to(
            accelerator.device, non_blocking=True
        )
        rs_imgs = batch["search_pixel_values"].to(
            accelerator.device, non_blocking=True
        )
        ori_gt_bbox = batch["bbox"].to(accelerator.device, non_blocking=True)
        use_global_text = Config.USE_TEXT_INPUT and Config.USE_TEXT_GROUNDING_PATH
        input_ids = (
            batch["input_ids"].to(accelerator.device, non_blocking=True)
            if use_global_text
            else None
        )
        attention_mask = (
            batch["attention_mask"].to(accelerator.device, non_blocking=True)
            if use_global_text and "attention_mask" in batch
            else None
        )
        local_indices = batch["index"].to(accelerator.device, non_blocking=True)
        geo = build_geo_features(batch, accelerator.device)

        with torch.no_grad(), torch.amp.autocast("cuda", enabled=amp_enabled):
            outputs = model(
                query_imgs,
                rs_imgs,
                input_ids,
                geo,
                attention_mask=attention_mask,
            )
            (
                pred_anchor,
                _,
                text_feats,
                anchor_feats,
                grid_feats,
                fused_feats,
                heatmap_logits,
            ) = unpack_model_outputs(outputs)

        B = pred_anchor.shape[0]
        pred_anchor = pred_anchor.view(
            B, 9, 5, pred_anchor.shape[2], pred_anchor.shape[3]
        )
        pred_anchor = add_heatmap_to_confidence(pred_anchor, heatmap_logits)

        _, best_anchor_gi_gj = build_target(
            ori_gt_bbox,
            anchors_full,
            (Config.UNIV_SAT_SIZE["width"], Config.UNIV_SAT_SIZE["height"]),
            (pred_anchor.shape[4], pred_anchor.shape[3]),
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
        if isinstance(fused_feats, dict) and "bbox" in fused_feats:
            refined_iou = bbox_iou(
                fused_feats["bbox"].to(device=ori_gt_bbox.device, dtype=ori_gt_bbox.dtype),
                ori_gt_bbox,
                x1y1x2y2=True,
            )
            accu_list = [
                (refined_iou > 0.5).float().mean(),
                (refined_iou > 0.25).float().mean(),
            ]
            iou = refined_iou.mean()

        candidate_feats = grid_feats.reshape(-1, Config.PROJECTION_DIM)
        positive_indices = torch.zeros(B, B * 9, device=accelerator.device)
        
        batch_offsets = torch.arange(B, device=accelerator.device) * 9
        row_indices_broad = torch.arange(B, device=accelerator.device).unsqueeze(1)
        col_offsets = torch.arange(9, device=accelerator.device).unsqueeze(0)

        same_image_cols = batch_offsets.unsqueeze(1) + col_offsets
        positive_indices[row_indices_broad, same_image_cols] = 0.5

        global_positive_indices = local_indices + batch_offsets
        row_indices_flat = torch.arange(B, device=accelerator.device)
        positive_indices[row_indices_flat, global_positive_indices] = 0.95

        if useap:
            if isinstance(fused_feats, torch.Tensor):
                image_retrieval_loss = 0.7 * info_nce_loss(
                    fused_feats, candidate_feats, positive_indices
                )
                image_retrieval_loss += 0.3 * info_nce_loss(
                    anchor_feats, candidate_feats, positive_indices
                )
            else:
                image_retrieval_loss = info_nce_loss(
                    anchor_feats, candidate_feats, positive_indices
                )
        else:
            scorer_input = fused_feats if isinstance(fused_feats, torch.Tensor) else anchor_feats
            scores = model.scorer(scorer_input, candidate_feats)
            image_retrieval_loss = F.cross_entropy(scores, positive_indices)

        batch_n = torch.tensor(float(query_imgs.shape[0]), device=accelerator.device)
        accu50_sum += accu_list[0].detach() * batch_n
        accu25_sum += accu_list[1].detach() * batch_n
        iou_sum += iou.detach() * batch_n
        info_nce_sum += image_retrieval_loss.detach() * batch_n
        sample_count += batch_n

    reduced = accelerator.reduce(
        torch.stack([accu50_sum, accu25_sum, iou_sum, info_nce_sum, sample_count]),
        reduction="sum",
    )
    denom = reduced[4].clamp(min=1.0)

    return (
        (reduced[0] / denom).item(),
        (reduced[1] / denom).item(),
        (reduced[2] / denom).item(),
        (reduced[3] / denom).item(),
    )


# --- Training ---
def load_data_splits() -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]], set, set]:
    train_image_pairs = []
    test_image_pairs = []
    train_ids = set()
    test_ids = set()

    with open("/media/data1/feihong/ckpt/train.txt", "r") as f:
        for line in f:
            query_path = line.strip()
            name = query_path.split("/")[-2]
            search_path = f"{Config.IMAGE_FOLDER}/{name}.png"
            train_image_pairs.append((query_path, search_path))
            train_ids.add(name)

    with open("/media/data1/feihong/ckpt/test.txt", "r") as f:
        for line in f:
            query_path = line.strip()
            name = query_path.split("/")[-2]
            search_path = f"{Config.IMAGE_FOLDER}/{name}.png"
            test_image_pairs.append((query_path, search_path))
            test_ids.add(name)

    return train_image_pairs, test_image_pairs, train_ids, test_ids


def train(save_path: str, end_num: float, use_ap: bool = True) -> None:
    valid_objectives = {"combined", "img_text_only", "bbox_only"}
    if Config.OPTIMIZE_OBJECTIVE not in valid_objectives:
        raise ValueError(
            f"Invalid OPTIMIZE_OBJECTIVE={Config.OPTIMIZE_OBJECTIVE}. "
            f"Choose from {sorted(valid_objectives)}."
        )
    if Config.USE_TEXT_ANCHOR_LOSS and not Config.USE_TEXT_GROUNDING_PATH:
        raise ValueError("USE_TEXT_ANCHOR_LOSS requires USE_TEXT_GROUNDING_PATH=True.")

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        mixed_precision="fp16" if Config.USE_AMP else "no",
        gradient_accumulation_steps=Config.GRAD_ACCUMULATION_STEPS,
        step_scheduler_with_optimizer=False,
        kwargs_handlers=[ddp_kwargs],
    )

    if accelerator.is_main_process:
        torch.backends.cuda.matmul.allow_tf32 = Config.ENABLE_TF32
        torch.backends.cudnn.allow_tf32 = Config.ENABLE_TF32

    exp_name = save_path.split("/")[-1] if save_path else "default_exp"
    writer = SummaryWriter(f"runs/{exp_name}") if accelerator.is_main_process else None

    clearml_task = None
    clearml_logger = None
    if Config.CLEARML_ENABLED and accelerator.is_main_process:
        from clearml import Task

        task_name = Config.CLEARML_TASK_NAME or exp_name
        clearml_task = Task.init(
            project_name=Config.CLEARML_PROJECT,
            task_name=task_name,
        )

        raw_config_dict = {
            key: getattr(Config, key)
            for key in dir(Config)
            if key.isupper() and not key.startswith("__")
        }
        config_dict = _to_clearml_serializable(raw_config_dict)
        clearml_task.connect(config_dict, name="config")

        clearml_logger = clearml_task.get_logger()
        clearml_logger.report_text(f"Experiment: {exp_name}")
        clearml_logger.report_text(
            f"Config: batch_size={Config.BATCH_SIZE}, lr={Config.LEARNING_RATE}, epochs={Config.NUM_EPOCHS}"
        )
        clearml_logger.report_text(
            "Full Config:\n" + json.dumps(config_dict, indent=2, sort_keys=True, ensure_ascii=False)
        )

    accelerator.print("Loading models and processor...")
    model_name = effective_model_name()
    processor = from_pretrained_prefer_local(
        AutoImageProcessor,
        model_name,
        Config.CACHE_DIR,
    )
    processor_sat = from_pretrained_prefer_local(
        AutoImageProcessor,
        model_name,
        Config.CACHE_DIR,
        size=Config.UNIV_SAT_SIZE,
    )
    tokenizer = from_pretrained_prefer_local(AutoTokenizer, model_name, Config.CACHE_DIR)

    model = build_encoder(use_ap, usesg=Config.OPTIMIZE_OBJECTIVE != "bbox_only")
    # model = Encoder_dino()

    anchors_full = get_tensor_anchors(accelerator.device)

    accelerator.print("Setting up dataset and dataloader...")
    train_dataset = ShiftedSatelliteDroneDataset(
        processor=processor,
        processor_sat=processor_sat,
        tokenizer=tokenizer,
        split="train",
    )
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=Config.BATCH_SIZE,
        **build_dataloader_kwargs(
            Config.NUM_WORKERS_TRAIN, drop_last=Config.DROP_LAST_TRAIN
        ),
    )

    test_dataset = ShiftedSatelliteDroneDataset(
        processor=processor,
        processor_sat=processor_sat,
        tokenizer=tokenizer,
        split="test",
    )
    test_dataloader = DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=Config.BATCH_SIZE,
        **build_dataloader_kwargs(Config.NUM_WORKERS_VAL),
    )
    accelerator.print(f"Train: {len(train_dataset)}, Test: {len(test_dataset)}")

    trainable_params = [param for param in model.parameters() if param.requires_grad]
    if not trainable_params:
        raise ValueError("No trainable parameters found for optimizer.")
    optimizer = AdamW(trainable_params, lr=Config.LEARNING_RATE)
    trainable_param_count = sum(param.numel() for param in trainable_params)
    accelerator.print(
        f"Optimizer: lr={Config.LEARNING_RATE:.2e}, "
        f"trainable params={trainable_param_count:,}"
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=Config.COSINE_EPOCHS
    )

    model, optimizer, scheduler, train_dataloader, test_dataloader = (
        accelerator.prepare(
            model, optimizer, scheduler, train_dataloader, test_dataloader
        )
    )

    accelerator.print(f"Starting training for {Config.NUM_EPOCHS} epochs...")
    soft_label = 0.01
    max_iou = 0
    min_info = 1000

    for epoch in range(Config.NUM_EPOCHS):
        train_dataset.text_cycle_step = epoch
        model.train()
        total_loss = 0
        total_bbox_loss = 0
        total_heatmap_loss = 0
        total_text_anchor_loss = 0
        total_text_pooler_align_loss = 0
        last_heatmap_logits = None
        last_target_bbox = None
        last_heatmap_target = None
        progress_bar = tqdm(
            train_dataloader,
            desc=f"Epoch {epoch + 1}/{Config.NUM_EPOCHS}",
            disable=not accelerator.is_main_process,
        )

        for i, batch in enumerate(progress_bar):
            with accelerator.accumulate(model):
                target_pixel_values = batch["target_pixel_values"].to(
                    accelerator.device, non_blocking=True
                )
                search_pixel_values = batch["search_pixel_values"].to(
                    accelerator.device, non_blocking=True
                )
                current_text_pooler_align_weight = text_pooler_align_weight(epoch)
                use_global_text = Config.USE_TEXT_INPUT and (
                    Config.USE_TEXT_GROUNDING_PATH or current_text_pooler_align_weight > 0
                )
                input_ids = (
                    batch["input_ids"].to(accelerator.device, non_blocking=True)
                    if use_global_text
                    else None
                )
                attention_mask = (
                    batch["attention_mask"].to(accelerator.device, non_blocking=True)
                    if use_global_text and "attention_mask" in batch
                    else None
                )
                local_indices = batch["index"].to(
                    accelerator.device, non_blocking=True
                )
                target_bbox = batch["bbox"].to(accelerator.device, non_blocking=True)
                geo = build_geo_features(batch, accelerator.device)

                with accelerator.autocast():
                    outputs = model(
                        target_pixel_values,
                        search_pixel_values,
                        input_ids,
                        geo,
                        attention_mask=attention_mask,
                    )
                    (
                        pred_anchor,
                        _,
                        text_feats,
                        anchor_feats,
                        grid_feats,
                        fused_feats,
                        heatmap_logits,
                    ) = unpack_model_outputs(outputs)

                    B = pred_anchor.shape[0]
                    pred_anchor = pred_anchor.view(
                        B, 9, 5, pred_anchor.shape[2], pred_anchor.shape[3]
                    )

                    pred_anchor = add_heatmap_to_confidence(pred_anchor, heatmap_logits)

                    new_gt_bbox, best_anchor_gi_gj = build_target(
                        target_bbox,
                        anchors_full,
                        (Config.UNIV_SAT_SIZE["width"], Config.UNIV_SAT_SIZE["height"]),
                        (pred_anchor.shape[4], pred_anchor.shape[3]),
                    )

                    loss_geo, loss_cls = yolo_loss(
                        pred_anchor,
                        new_gt_bbox,
                        anchors_full,
                        best_anchor_gi_gj,
                        (Config.UNIV_SAT_SIZE["width"], Config.UNIV_SAT_SIZE["height"]),
                    )
                    bbox_loss = loss_geo + loss_cls
                    if (
                        Config.USE_HEATMAP_LOSS
                        and Config.ENCODER_TYPE in {"heat", "test"}
                        and heatmap_logits is not None
                    ):
                        heatmap_loss, heatmap_target = heatmap_loss_fn(
                            heatmap_logits,
                            target_bbox,
                            return_target=True,
                        )
                        bbox_loss = bbox_loss + Config.HEATMAP_LOSS_WEIGHT * heatmap_loss
                    else:
                        heatmap_loss = bbox_loss.new_zeros(())
                        heatmap_target = None

                    text_anchor_loss = bbox_loss.new_zeros(())
                    text_pooler_align_loss = bbox_loss.new_zeros(())
                    if heatmap_logits is not None:
                        last_heatmap_logits = heatmap_logits.detach()
                        last_target_bbox = target_bbox.detach()
                        last_heatmap_target = (
                            heatmap_target.detach()
                            if heatmap_target is not None
                            else None
                        )

                    candidate_feats = grid_feats.reshape(-1, Config.PROJECTION_DIM)
                    positive_indices = torch.zeros(B, B * 9, device=accelerator.device)

                    batch_offsets = torch.arange(B, device=accelerator.device) * 9
                    row_indices_broad = torch.arange(
                        B, device=accelerator.device
                    ).unsqueeze(1)
                    col_offsets = torch.arange(9, device=accelerator.device).unsqueeze(0)

                    same_image_cols = batch_offsets.unsqueeze(1) + col_offsets
                    positive_indices[row_indices_broad, same_image_cols] = soft_label

                    global_positive_indices = local_indices + batch_offsets
                    row_indices_flat = torch.arange(B, device=accelerator.device)
                    positive_indices[row_indices_flat, global_positive_indices] = 0.92


                    image_retrieval_loss = info_nce_loss(
                        anchor_feats, candidate_feats, positive_indices
                    )
                    if (
                        Config.ENCODER_TYPE == "test"
                        and text_feats is not None
                        and current_text_pooler_align_weight > 0
                    ):
                        pair_labels = torch.arange(B, device=accelerator.device)
                        detach_image_for_text_align = text_pooler_align_detach_image(epoch)
                        align_anchor_feats = (
                            anchor_feats.detach()
                            if detach_image_for_text_align
                            else anchor_feats
                        )
                        text_pooler_align_loss = info_nce_loss(
                            text_feats.detach(),
                            align_anchor_feats,
                            pair_labels,
                        )
                        # text_satellite_retrieval_loss = info_nce_loss(
                        #     text_feats.detach(),
                        #     candidate_feats,
                        #     positive_indices,
                        # )
                        image_retrieval_loss = (
                            image_retrieval_loss
                            + current_text_pooler_align_weight * text_pooler_align_loss
                            # + current_text_pooler_align_weight * text_satellite_retrieval_loss
                        )

                    # scheduled_bbox_weight, scheduled_retrieval_weight = get_loss_weights(
                    #     epoch, Config.COSINE_EPOCHS, end_num
                    # )
                    if Config.OPTIMIZE_OBJECTIVE == "combined":
                        bbox_weight = end_num
                        retrieval_weight = 1 - end_num
                        loss = retrieval_weight * image_retrieval_loss + bbox_weight * bbox_loss
                    elif Config.OPTIMIZE_OBJECTIVE == "img_text_only":
                        bbox_weight = 0.0
                        retrieval_weight = 1.0
                        loss = image_retrieval_loss
                    else:  # bbox_only
                        bbox_weight = 1.0
                        retrieval_weight = 0.0
                        loss = retrieval_weight * image_retrieval_loss + bbox_weight * bbox_loss

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    if Config.GRAD_CLIP_NORM is not None and Config.GRAD_CLIP_NORM > 0:
                        accelerator.clip_grad_norm_(trainable_params, Config.GRAD_CLIP_NORM)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

            total_loss += loss.item()
            total_bbox_loss += bbox_loss.item()
            total_heatmap_loss += heatmap_loss.item()
            total_text_anchor_loss += text_anchor_loss.item()
            total_text_pooler_align_loss += text_pooler_align_loss.item()
            # total_text_satellite_retrieval_loss += text_satellite_retrieval_loss.item()
            global_step = epoch * len(train_dataloader) + i
            progress_bar.set_postfix(
                {
                    "bbox_loss": f"{bbox_loss.item():.4f}",
                    "heatmap_loss": f"{heatmap_loss.item():.4f}",
                    "text_anchor": f"{text_anchor_loss.item():.4f}",
                    "text_pooler": f"{text_pooler_align_loss.item():.4f}",
                    # "text_sat": f"{text_satellite_retrieval_loss.item():.4f}",
                    "text_pooler_w": f"{current_text_pooler_align_weight:.4f}",
                    "retrieval_loss": f"{image_retrieval_loss.item():.4f}",
                    "retrieval_weight": f"{retrieval_weight:.4f}",
                    "bbox_weight": f"{bbox_weight:.4f}",
                }
            )

            if writer is not None:
                writer.add_scalar("Loss/train_step", loss.item(), global_step)
                writer.add_scalar("Loss/bbox_step", bbox_loss.item(), global_step)
                writer.add_scalar("Loss/bbox_geo_step", loss_geo.item(), global_step)
                writer.add_scalar("Loss/bbox_cls_step", loss_cls.item(), global_step)
                writer.add_scalar("Loss/heatmap_step", heatmap_loss.item(), global_step)
                writer.add_scalar("Loss/text_anchor_step", text_anchor_loss.item(), global_step)
                writer.add_scalar("Loss/text_pooler_align_step", text_pooler_align_loss.item(), global_step)
                # writer.add_scalar("Loss/text_satellite_retrieval_step", text_satellite_retrieval_loss.item(), global_step)
                writer.add_scalar("Weight/text_pooler_align", current_text_pooler_align_weight, global_step)
                writer.add_scalar("Loss/retrieval_step", image_retrieval_loss.item(), global_step)
                writer.add_scalar("Weight/retrieval", retrieval_weight, global_step)
                writer.add_scalar("Weight/bbox", bbox_weight, global_step)
                writer.add_scalar(
                    "Loss/image_retrieval_step",
                    image_retrieval_loss.item(),
                    global_step,
                )
                writer.add_scalar("Weight/retrieval", retrieval_weight, global_step)
                writer.add_scalar("Weight/bbox", bbox_weight, global_step)

            if clearml_logger is not None:
                clearml_logger.report_scalar("train", "loss", loss.item(), global_step)
                clearml_logger.report_scalar(
                    "train", "bbox_loss", bbox_loss.item(), global_step
                )
                clearml_logger.report_scalar(
                    "train", "heatmap_loss", heatmap_loss.item(), global_step
                )
                clearml_logger.report_scalar(
                    "train", "text_anchor_loss", text_anchor_loss.item(), global_step
                )
                clearml_logger.report_scalar(
                    "train", "text_pooler_align_loss", text_pooler_align_loss.item(), global_step
                )
                # clearml_logger.report_scalar(
                #     "train",
                #     "text_satellite_retrieval_loss",
                #     text_satellite_retrieval_loss.item(),
                #     global_step,
                # )
                clearml_logger.report_scalar(
                    "train", "image_retrieval_loss", image_retrieval_loss.item(), global_step
                )
                clearml_logger.report_scalar(
                    "train",
                    "image_retrieval_loss",
                    image_retrieval_loss.item(),
                    global_step,
                )

        scheduler.step()

        if writer is not None:
            writer.add_scalar("Loss/train_epoch", total_loss / len(train_dataloader), epoch)
            writer.add_scalar(
                "Loss/bbox_epoch",
                total_bbox_loss / len(train_dataloader),
                epoch,
            )
            writer.add_scalar(
                "Loss/heatmap_epoch",
                total_heatmap_loss / len(train_dataloader),
                epoch,
            )
            writer.add_scalar(
                "Loss/text_anchor_epoch",
                total_text_anchor_loss / len(train_dataloader),
                epoch,
            )
            writer.add_scalar(
                "Loss/text_pooler_align_epoch",
                total_text_pooler_align_loss / len(train_dataloader),
                epoch,
            )
            # writer.add_scalar(
            #     "Loss/text_satellite_retrieval_epoch",
            #     total_text_satellite_retrieval_loss / len(train_dataloader),
            #     epoch,
            # )
            writer.add_scalar("lr/learning_rate", optimizer.param_groups[0]["lr"], epoch)
            for group_idx, group in enumerate(optimizer.param_groups):
                group_name = str(group.get("name", f"group_{group_idx}"))
                writer.add_scalar(f"lr/{group_name}", group["lr"], epoch)
            log_heatmap_images(
                writer,
                last_heatmap_logits,
                last_target_bbox,
                epoch,
                heatmap_target=last_heatmap_target,
            )
        if epoch % 2 == 0:
            os.makedirs(save_path, exist_ok=True)
            unwrapped_model = accelerator.unwrap_model(model)
            save_filename = f"{save_path}/best_iou.pth"

            accelerator.save(unwrapped_model.state_dict(), save_filename)

            # accelerator.print(f"Running validation at epoch {epoch + 1}...")
            # accu50, accu25, iou, info_nce = validate(
            #     test_dataloader,
            #     model,
            #     accelerator,
            #     anchors_full,
            #     (Config.UNIV_SAT_SIZE[0], Config.UNIV_SAT_SIZE[1]),
            #     useap=use_ap,
            # )
            # if accelerator.is_main_process:
            #     accelerator.print(
            #         f"Val Epoch {epoch + 1}: Accu@50={accu50:.4f}, Accu@25={accu25:.4f}, IoU={iou:.4f}, "
            #         f"InfoNCE={info_nce:.4f}"
            #     )

            #     if clearml_logger is not None:
            #         clearml_logger.report_scalar("validation", "accu50", accu50, epoch)
            #         clearml_logger.report_scalar("validation", "accu25", accu25, epoch)
            #         clearml_logger.report_scalar("validation", "iou", iou, epoch)
            #         clearml_logger.report_scalar("validation", "info_nce", info_nce, epoch)

            #     if iou > max_iou:
            #         max_iou = iou
            #         os.makedirs(save_path, exist_ok=True)
            #         unwrapped_model = accelerator.unwrap_model(model)
            #         save_filename = f"{save_path}/best_iou.pth"
            #         accelerator.save(unwrapped_model.state_dict(), save_filename)
            #         accelerator.print(f"Saved best IoU checkpoint: {save_filename}")

        if clearml_logger is not None:
            current_lr = optimizer.param_groups[0]["lr"]
            clearml_logger.report_scalar("lr", "learning_rate", current_lr, epoch)

    accelerator.print("Training complete.")
    if accelerator.is_main_process:
        os.makedirs(save_path, exist_ok=True)
        unwrapped_model = accelerator.unwrap_model(model)
        save_filename = f"{save_path}/last.pth"
        accelerator.save(unwrapped_model.state_dict(), save_filename)
        print(f"Saved checkpoint: {save_filename}")
        if clearml_task is not None:
            clearml_task.close()
    if writer is not None:
        writer.close()



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train unified SigLIP grounding/retrieval model."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="YAML file with defaults/config and optional experiments list.",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default=None,
        help="Experiment name to select from YAML experiments list.",
    )
    parser.add_argument(
        "--exp-name",
        type=str,
        default=None,
        help="Experiment name used for checkpoint/log directory.",
    )
    parser.add_argument(
        "--save-root",
        type=str,
        default="/media/data1/feihong/ckpt",
        help="Root directory for checkpoints when --save-dir is not set.",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default=None,
        help="Full checkpoint directory. Overrides --save-root/--exp-name.",
    )
    parser.add_argument(
        "--end-num",
        type=float,
        default=None,
        help="Final/static bbox loss weight used by train().",
    )
    parser.add_argument(
        "--use-ap",
        dest="use_ap",
        action="store_true",
        default=None,
        help="Use attention-pooling retrieval branch.",
    )
    parser.add_argument(
        "--no-use-ap",
        dest="use_ap",
        action="store_false",
        help="Disable attention-pooling retrieval branch.",
    )
    return parser.parse_args()


def resolve_run_settings(args: argparse.Namespace) -> Tuple[str, str, float, bool]:
    yaml_exp = load_experiment_from_yaml(args.config, args.experiment)
    apply_config_overrides(yaml_exp.get("config", {}))

    end_num = (
        float(args.end_num)
        if args.end_num is not None
        else float(yaml_exp.get("end_num", 0.5))
    )
    exp_name = (
        args.exp_name
        or yaml_exp.get("exp_name")
        or yaml_exp.get("name")
        or f"{end_num}_mix_model_heat_2"
    )
    exp_name = str(exp_name)

    use_ap = bool(yaml_exp.get("use_ap", True))
    if args.use_ap is not None:
        use_ap = bool(args.use_ap)

    save_root = str(yaml_exp.get("save_root", args.save_root))
    save_dir = args.save_dir or yaml_exp.get("save_dir")
    if save_dir is None:
        save_dir = os.path.join(save_root, exp_name)
    save_dir = str(save_dir)

    return exp_name, save_dir, end_num, use_ap


def write_effective_config(save_dir: str, exp_name: str, end_num: float, use_ap: bool) -> None:
    payload = {
        "exp_name": exp_name,
        "save_dir": save_dir,
        "end_num": float(end_num),
        "use_ap": bool(use_ap),
        "config": _to_clearml_serializable(config_to_dict()),
    }
    out_path = os.path.join(save_dir, "effective_config.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True, ensure_ascii=False)
    print(f"Saved effective config: {out_path}")


# --- Main ---
if __name__ == "__main__":
    args = parse_args()
    exp_name, save_dir, end_num, use_ap = resolve_run_settings(args)

    if os.path.exists(save_dir):
        print(f"Experiment directory '{save_dir}' already exists.")
        print("Re-using directory. Logs and checkpoints might be overwritten.")
    else:
        os.makedirs(save_dir, exist_ok=True)
        print(f"Created experiment directory: {save_dir}")

    write_effective_config(save_dir, exp_name, end_num, use_ap)
    print(
        f"Starting experiment '{exp_name}' with end_num={end_num}, "
        f"use_ap={use_ap}, save_dir={save_dir}"
    )
    train(save_dir, end_num, use_ap=use_ap)
