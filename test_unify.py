import argparse
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoTokenizer

from bbox.yolo_utils import bbox_iou, build_target, eval_iou_acc
from dataset import DEFAULT_SUBSET_ANGLES, DEFAULT_SUBSET_HEIGHTS, ShiftedSatelliteDroneDataset
from model import Encoder_heat, Encoder_test
from train_uni import (
    BACKBONE_NAME as UNIFY_BACKBONE_NAME,
    DRONE_SIZE,
    PROJECTION_DIM,
    DummyTokenizer,
    TransformProcessorWrapper,
    UnifyGeoLite,
    decode_bbox,
)
from train_trans import (
    DETAIL_DIM as TRANS_DETAIL_DIM,
    EMBED_DIM as TRANS_EMBED_DIM,
    NUM_HEADS as TRANS_NUM_HEADS,
    PATCH_SIZE as TRANS_PATCH_SIZE,
    TransGeoGrounding,
)


# --- Configuration ---
MODEL_NAME = "google/siglip2-base-patch16-224"
CACHE_DIR = "/media/data1/feihong/hf_cache"
DEFAULT_DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
DEFAULT_OUTPUT_DIR = "/media/data1/feihong/univerisity_dev/eval_results/test_unify"
DEFAULT_INCLUDE_FILE = "/media/data1/feihong/ckpt/include2.json"
DEFAULT_ENCODER_HEAT_CONFIG_DIR = "/media/data1/feihong/univerisity_dev/configs/unified_siglip_supp"
DEFAULT_ENCODER_HEAT_CHECKPOINT = "/media/data1/feihong/ckpt/model_full/last.pth"
DEFAULT_UNIFY_CHECKPOINT = "/media/data1/feihong/ckpt/unify_geo/last.pth"
DEFAULT_TRANS_CHECKPOINT = "/media/data1/feihong/ckpt/trans_geo/last.pth"
ANCHORS = "37,41, 78,84, 96,215, 129,129, 194,82, 198,179, 246,280, 395,342, 550,573"


@dataclass
class FeatureBundle:
    query_feats: torch.Tensor
    query_labels: List[str]
    query_drone_paths: List[str]
    query_satellite_paths: List[str]
    query_heights: List[int]
    query_angles: List[int]
    iou_values: List[float]
    center_distances: List[float]
    gallery_labels: List[str]
    gallery_satellite_paths: List[str]
    gallery_feats: torch.Tensor
    query_text_feats: Optional[torch.Tensor] = None
    query_detail_feats: Optional[torch.Tensor] = None
    gallery_detail_feats: Optional[torch.Tensor] = None
    unify_logit_scale: Optional[float] = None
    unify_temperature: Optional[float] = None


def _normalize_label(label: str) -> str:
    return str(label).strip().split(".")[0]


def _path_label(path: str) -> str:
    return _normalize_label(Path(str(path)).stem)


def _drone_image_name(path: str) -> str:
    return Path(str(path)).name


def _load_include_map(include_file: Optional[str]) -> Dict[str, Set[str]]:
    include_map: Dict[str, Set[str]] = {}
    if not include_file:
        return include_map
    if not os.path.exists(include_file):
        print(f"Warning: include file not found: {include_file}. Using exact labels only.")
        return include_map

    with open(include_file, "r", encoding="utf-8") as f:
        include_raw = json.load(f)

    for key, values in include_raw.items():
        norm_key = _normalize_label(key)
        if isinstance(values, list):
            include_map[norm_key] = {_normalize_label(v) for v in values}
        else:
            include_map[norm_key] = {_normalize_label(values)}
    return include_map


def parse_anchors(device: torch.device) -> torch.Tensor:
    anchors = np.array([float(x) for x in ANCHORS.split(",")], dtype=np.float32)
    anchors = anchors.reshape(-1, 2)[::-1].copy()
    return torch.tensor(anchors, dtype=torch.float32, device=device)


def add_heatmap_to_confidence(
    pred_anchor: torch.Tensor,
    heatmap_logits: Optional[torch.Tensor],
    confidence_weight: float,
) -> torch.Tensor:
    if heatmap_logits is None or confidence_weight <= 0.0:
        return pred_anchor

    if heatmap_logits.shape[-2:] != pred_anchor.shape[-2:]:
        heatmap_logits = F.interpolate(
            heatmap_logits,
            size=pred_anchor.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

    heatmap_confidence = torch.sigmoid(heatmap_logits.detach()).to(dtype=pred_anchor.dtype)
    heat_confidence = float(confidence_weight) * heatmap_confidence.unsqueeze(1)
    return torch.cat(
        [
            pred_anchor[:, :, :4, :, :],
            pred_anchor[:, :, 4:5, :, :] + heat_confidence,
        ],
        dim=2,
    )


def build_geo_features(batch: Dict[str, Any], device: torch.device) -> torch.Tensor:
    angles = batch["angle"].to(device, non_blocking=True).float()
    angles_rad = torch.deg2rad(angles)
    heights = batch["height"].to(device, non_blocking=True).float()
    return torch.cat(
        [
            torch.cos(angles_rad)[..., None],
            torch.sin(angles_rad)[..., None],
            heights[..., None] / 300.0,
        ],
        dim=1,
    )


def center_distance(pred_xyxy: torch.Tensor, gt_xyxy: torch.Tensor) -> float:
    pred_cx = 0.5 * (pred_xyxy[0] + pred_xyxy[2])
    pred_cy = 0.5 * (pred_xyxy[1] + pred_xyxy[3])
    gt_cx = 0.5 * (gt_xyxy[0] + gt_xyxy[2])
    gt_cy = 0.5 * (gt_xyxy[1] + gt_xyxy[3])
    return float(torch.sqrt((pred_cx - gt_cx) ** 2 + (pred_cy - gt_cy) ** 2).item())


def load_checkpoint(model: torch.nn.Module, checkpoint_path: str) -> None:
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    state = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    elif isinstance(state, dict) and "model" in state:
        state = state["model"]
    if isinstance(state, dict):
        state = {str(k).replace("module.", ""): v for k, v in state.items()}
        state = {
            key: value
            for key, value in state.items()
            if not key.startswith("text_projector.")
        }

    model.load_state_dict(state, strict=True)


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        payload = yaml.safe_load(f) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return payload


def bool_from_config(config: Dict[str, Any], key: str, default: bool) -> bool:
    value = config.get(key, default)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def resolve_encoder_heat_checkpoint(config_path: str, payload: Dict[str, Any], checkpoint_name: str) -> str:
    config_file = Path(config_path)
    exp_name = str(payload.get("exp_name") or payload.get("name") or config_file.stem)
    save_root = str(payload.get("save_root", "/media/data1/feihong/ckpt"))
    save_dir = payload.get("save_dir")

    candidates: List[Path] = []
    if save_dir:
        candidates.append(Path(str(save_dir)))
    candidates.append(Path(save_root) / exp_name)
    candidates.append(Path(save_root) / config_file.stem)

    # The current checkpoint directory uses model_full, while the YAML exp_name is 0.5_heat.
    if config_file.stem == "full_model":
        candidates.append(Path(save_root) / "model_full")

    for directory in candidates:
        checkpoint_path = directory / checkpoint_name
        if checkpoint_path.exists():
            return str(checkpoint_path)

    for directory in candidates:
        checkpoint_path = directory / "last.pth"
        if checkpoint_path.exists():
            print(f"Warning: {directory / checkpoint_name} not found. Falling back to {checkpoint_path}.")
            return str(checkpoint_path)

    checked = ", ".join(str(path / checkpoint_name) for path in candidates)
    raise FileNotFoundError(f"No checkpoint found for {config_path}. Checked: {checked}")


def discover_encoder_heat_runs(config_dir: str, checkpoint_name: str) -> List[Dict[str, Any]]:
    config_paths = sorted(Path(config_dir).glob("*.yaml"))
    if not config_paths:
        raise FileNotFoundError(f"No YAML configs found in: {config_dir}")

    runs: List[Dict[str, Any]] = []
    for config_path in config_paths:
        payload = load_yaml(str(config_path))
        config = payload.get("config", {}) or {}
        if not isinstance(config, dict):
            raise ValueError(f"'config' must be a mapping in {config_path}")
        if str(config.get("ENCODER_TYPE", "heat")).lower() != "heat":
            continue

        checkpoint_path = resolve_encoder_heat_checkpoint(
            config_path=str(config_path),
            payload=payload,
            checkpoint_name=checkpoint_name,
        )
        runs.append(
            {
                "config_path": str(config_path),
                "config_name": config_path.stem,
                "exp_name": str(payload.get("exp_name") or payload.get("name") or config_path.stem),
                "checkpoint_path": checkpoint_path,
                "use_text": bool_from_config(config, "USE_TEXT_INPUT", True),
                "use_angle": bool_from_config(config, "USE_ANGLE_INPUT", True),
                "use_ap": bool(payload.get("use_ap", True)),
                "use_heatmap": bool_from_config(config, "USE_HEATMAP_LOSS", True),
                "heatmap_confidence_weight": float(
                    config.get("HEATMAP_CONFIDENCE_WEIGHT", 0.5)
                ),
                "raw_config": config,
            }
        )

    if not runs:
        raise RuntimeError(f"No encoder_heat configs found in: {config_dir}")
    return runs


def create_encoder_loader(
    batch_size: int,
    num_workers: int,
    sat_size: Tuple[int, int],
    test_crop_ratio: float,
    subset_heights: Optional[Sequence[int]],
    subset_angles: Optional[Sequence[int]],
    model_name: str,
) -> DataLoader:
    processor = AutoImageProcessor.from_pretrained(model_name, cache_dir=CACHE_DIR)
    processor_sat = AutoImageProcessor.from_pretrained(
        model_name,
        cache_dir=CACHE_DIR,
        size={"height": int(sat_size[0]), "width": int(sat_size[1])},
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=CACHE_DIR)
    dataset = ShiftedSatelliteDroneDataset(
        processor=processor,
        processor_sat=processor_sat,
        tokenizer=tokenizer,
        split="test",
        sat_target_size=sat_size,
        test_crop_ratio=test_crop_ratio,
        subset_heights=subset_heights,
        subset_angles=subset_angles,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,
    )


def create_encoder_heat_loader(
    batch_size: int,
    num_workers: int,
    sat_size: Tuple[int, int],
    test_crop_ratio: float,
    subset_heights: Optional[Sequence[int]],
    subset_angles: Optional[Sequence[int]],
) -> DataLoader:
    return create_encoder_loader(
        batch_size=batch_size,
        num_workers=num_workers,
        sat_size=sat_size,
        test_crop_ratio=test_crop_ratio,
        subset_heights=subset_heights,
        subset_angles=subset_angles,
        model_name=MODEL_NAME,
    )


def create_unify_loader(
    batch_size: int,
    num_workers: int,
    sat_size: Tuple[int, int],
    test_crop_ratio: float,
    subset_heights: Optional[Sequence[int]],
    subset_angles: Optional[Sequence[int]],
) -> DataLoader:
    sat_hw = (int(sat_size[0]), int(sat_size[1]))
    processor = TransformProcessorWrapper(DRONE_SIZE)
    processor_sat = TransformProcessorWrapper((sat_hw[1], sat_hw[0]))
    dataset = ShiftedSatelliteDroneDataset(
        processor=processor,
        processor_sat=processor_sat,
        tokenizer=DummyTokenizer(),
        split="test",
        sat_target_size=sat_hw,
        test_crop_ratio=test_crop_ratio,
        subset_heights=subset_heights,
        subset_angles=subset_angles,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,
    )


def create_trans_loader(
    batch_size: int,
    num_workers: int,
    sat_size: Tuple[int, int],
    test_crop_ratio: float,
    subset_heights: Optional[Sequence[int]],
    subset_angles: Optional[Sequence[int]],
) -> DataLoader:
    return create_unify_loader(
        batch_size=batch_size,
        num_workers=num_workers,
        sat_size=sat_size,
        test_crop_ratio=test_crop_ratio,
        subset_heights=subset_heights,
        subset_angles=subset_angles,
    )


def extract_encoder_heat_features(
    checkpoint_path: str,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    sat_size: Tuple[int, int],
    test_crop_ratio: float,
    subset_heights: Optional[Sequence[int]],
    subset_angles: Optional[Sequence[int]],
    heatmap_confidence_weight: float,
    use_text: bool = True,
    use_angle: bool = True,
    use_ap: bool = True,
    use_heatmap: bool = True,
    encoder_cls: type = Encoder_heat,
    desc: str = "encoder_heat",
    lora_rank: int = 8,
    lora_alpha: float = 16.0,
    lora_dropout: float = 0.05,
) -> FeatureBundle:
    model = encoder_cls(
        model_name=MODEL_NAME,
        proj_dim=768,
        usesg=True,
        useap=use_ap,
        use_heatmap=use_heatmap,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
    ).to(device)
    load_checkpoint(model, checkpoint_path)
    model.eval()

    loader = create_encoder_loader(
        batch_size=batch_size,
        num_workers=num_workers,
        sat_size=sat_size,
        test_crop_ratio=test_crop_ratio,
        subset_heights=subset_heights,
        subset_angles=subset_angles,
        model_name=MODEL_NAME,
    )
    anchors_full = parse_anchors(device)

    query_feats: List[torch.Tensor] = []
    query_labels: List[str] = []
    query_drone_paths: List[str] = []
    query_satellite_paths: List[str] = []
    query_heights: List[int] = []
    query_angles: List[int] = []
    query_text_feats: List[torch.Tensor] = []
    iou_values: List[float] = []
    center_distances: List[float] = []
    gallery_feat_dict: Dict[str, torch.Tensor] = {}
    gallery_path_dict: Dict[str, str] = {}

    with torch.inference_mode():
        for batch in tqdm(loader, desc=f"Extract/eval [{desc}]"):
            query_imgs = batch["target_pixel_values"].to(device, non_blocking=True)
            search_imgs = batch["search_pixel_values"].to(device, non_blocking=True)
            gt_bbox = batch["bbox"].to(device, non_blocking=True)
            input_ids = (
                batch["input_ids"].to(device, non_blocking=True)
                if use_text
                else None
            )
            attention_mask = (
                batch["attention_mask"].to(device, non_blocking=True)
                if use_text and "attention_mask" in batch
                else None
            )
            geo = build_geo_features(batch, device) if use_angle else None

            outputs = model(
                query_imgs,
                search_imgs,
                input_ids=input_ids,
                angle=geo,
                attention_mask=attention_mask,
            )
            if len(outputs) != 7:
                raise ValueError(f"Expected 7 model outputs, got {len(outputs)}.")
            (
                pred_anchor,
                _,
                text_feats,
                anchor_pooler,
                grid_feats,
                refine_outputs,
                heatmap_logits,
            ) = outputs

            query_batch = F.normalize(anchor_pooler, p=2, dim=1)
            gallery_grid_batch = F.normalize(grid_feats, p=2, dim=2)
            query_feats.append(query_batch.cpu())
            if text_feats is not None:
                query_text_feats.append(F.normalize(text_feats, p=2, dim=1).cpu())

            batch_labels = [_path_label(path) for path in batch["satellite_path"]]
            query_labels.extend(batch_labels)
            query_drone_paths.extend([str(p) for p in batch["drone_path"]])
            query_satellite_paths.extend([str(p) for p in batch["satellite_path"]])
            query_heights.extend([int(v) for v in batch["height"].tolist()])
            query_angles.extend([int(v) for v in batch["angle"].tolist()])

            for idx, label in enumerate(batch_labels):
                if label not in gallery_feat_dict:
                    gallery_feat_dict[label] = gallery_grid_batch[idx].detach().cpu()
                    gallery_path_dict[label] = str(batch["satellite_path"][idx])

            bsz = int(pred_anchor.shape[0])
            pred_anchor = pred_anchor.view(
                bsz,
                9,
                5,
                pred_anchor.shape[-2],
                pred_anchor.shape[-1],
            )
            pred_anchor = add_heatmap_to_confidence(
                pred_anchor,
                heatmap_logits if use_heatmap else None,
                confidence_weight=heatmap_confidence_weight if use_heatmap else 0.0,
            )
            image_wh = (int(sat_size[1]), int(sat_size[0]))
            grid_wh = (int(pred_anchor.shape[-1]), int(pred_anchor.shape[-2]))
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
            if isinstance(refine_outputs, dict) and "bbox" in refine_outputs:
                pred_bbox_xyxy = refine_outputs["bbox"].to(device=gt_bbox.device, dtype=gt_bbox.dtype)
                target_bbox_xyxy = gt_bbox

            ious = bbox_iou(pred_bbox_xyxy, target_bbox_xyxy, x1y1x2y2=True)
            for idx in range(pred_bbox_xyxy.shape[0]):
                iou_values.append(float(ious[idx].item()))
                center_distances.append(
                    center_distance(pred_bbox_xyxy[idx].float(), target_bbox_xyxy[idx].float())
                )

    if not query_feats or not gallery_feat_dict:
        raise RuntimeError(f"No {desc} features were extracted.")

    gallery_labels = sorted(gallery_feat_dict.keys())
    return FeatureBundle(
        query_feats=torch.cat(query_feats, dim=0),
        query_labels=query_labels,
        query_drone_paths=query_drone_paths,
        query_satellite_paths=query_satellite_paths,
        query_heights=query_heights,
        query_angles=query_angles,
        iou_values=iou_values,
        center_distances=center_distances,
        gallery_labels=gallery_labels,
        gallery_satellite_paths=[gallery_path_dict.get(label, "") for label in gallery_labels],
        gallery_feats=torch.stack([gallery_feat_dict[label] for label in gallery_labels], dim=0),
        query_text_feats=torch.cat(query_text_feats, dim=0) if query_text_feats else None,
    )


def extract_unify_features(
    checkpoint_path: str,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    sat_size: Tuple[int, int],
    test_crop_ratio: float,
    subset_heights: Optional[Sequence[int]],
    subset_angles: Optional[Sequence[int]],
    base_dim: int,
    proj_dim: int,
    detail_dim: int,
    backbone_name: str,
) -> FeatureBundle:
    model = UnifyGeoLite(
        proj_dim=proj_dim,
        detail_dim=detail_dim,
        dims=(base_dim, base_dim * 2, base_dim * 4),
        backbone_name=backbone_name,
        pretrained_backbone=False,
        pretrained_checkpoint=None,
    ).to(device)
    load_checkpoint(model, checkpoint_path)
    model.eval()

    loader = create_unify_loader(
        batch_size=batch_size,
        num_workers=num_workers,
        sat_size=sat_size,
        test_crop_ratio=test_crop_ratio,
        subset_heights=subset_heights,
        subset_angles=subset_angles,
    )

    query_feats: List[torch.Tensor] = []
    query_detail_feats: List[torch.Tensor] = []
    query_labels: List[str] = []
    query_drone_paths: List[str] = []
    query_satellite_paths: List[str] = []
    query_heights: List[int] = []
    query_angles: List[int] = []
    iou_values: List[float] = []
    center_distances: List[float] = []
    gallery_feat_dict: Dict[str, torch.Tensor] = {}
    gallery_detail_dict: Dict[str, torch.Tensor] = {}
    gallery_path_dict: Dict[str, str] = {}

    with torch.inference_mode():
        for batch in tqdm(loader, desc="Extract/eval [unify_geo]"):
            query_imgs = batch["target_pixel_values"].to(device, non_blocking=True)
            search_imgs = batch["search_pixel_values"].to(device, non_blocking=True)
            gt_bbox = batch["bbox"].to(device, non_blocking=True)
            outputs = model(query_imgs, search_imgs)

            query_feats.append(outputs["query_global"].detach().cpu())
            query_detail_feats.append(outputs["ground_detail"].detach().cpu())

            batch_labels = [_path_label(path) for path in batch["satellite_path"]]
            query_labels.extend(batch_labels)
            query_drone_paths.extend([str(p) for p in batch["drone_path"]])
            query_satellite_paths.extend([str(p) for p in batch["satellite_path"]])
            query_heights.extend([int(v) for v in batch["height"].tolist()])
            query_angles.extend([int(v) for v in batch["angle"].tolist()])

            aerial_global = outputs["aerial_global"].detach().cpu()
            aerial_detail = outputs["aerial_detail"].detach().cpu()
            for idx, label in enumerate(batch_labels):
                if label not in gallery_feat_dict:
                    gallery_feat_dict[label] = aerial_global[idx]
                    gallery_detail_dict[label] = aerial_detail[idx]
                    gallery_path_dict[label] = str(batch["satellite_path"][idx])

            pred_bbox = decode_bbox(
                outputs["heatmap_logits"],
                outputs["bbox_raw"],
                image_wh=(search_imgs.shape[-1], search_imgs.shape[-2]),
            )
            ious = bbox_iou(pred_bbox, gt_bbox, x1y1x2y2=True)
            for idx in range(pred_bbox.shape[0]):
                iou_values.append(float(ious[idx].item()))
                center_distances.append(center_distance(pred_bbox[idx].float(), gt_bbox[idx].float()))

    if not query_feats or not gallery_feat_dict:
        raise RuntimeError("No unify_geo features were extracted.")

    gallery_labels = sorted(gallery_feat_dict.keys())
    temperature = float(model.decoder.temperature.detach().clamp(min=0.01, max=1.0).cpu().item())
    logit_scale = float(model.logit_scale.detach().exp().clamp(max=100.0).cpu().item())

    return FeatureBundle(
        query_feats=torch.cat(query_feats, dim=0),
        query_labels=query_labels,
        query_drone_paths=query_drone_paths,
        query_satellite_paths=query_satellite_paths,
        query_heights=query_heights,
        query_angles=query_angles,
        iou_values=iou_values,
        center_distances=center_distances,
        gallery_labels=gallery_labels,
        gallery_satellite_paths=[gallery_path_dict.get(label, "") for label in gallery_labels],
        gallery_feats=torch.stack([gallery_feat_dict[label] for label in gallery_labels], dim=0),
        query_detail_feats=torch.cat(query_detail_feats, dim=0),
        gallery_detail_feats=torch.stack([gallery_detail_dict[label] for label in gallery_labels], dim=0),
        unify_logit_scale=logit_scale,
        unify_temperature=temperature,
    )


def extract_trans_features(
    checkpoint_path: str,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    sat_size: Tuple[int, int],
    test_crop_ratio: float,
    subset_heights: Optional[Sequence[int]],
    subset_angles: Optional[Sequence[int]],
    proj_dim: int,
    detail_dim: int,
    embed_dim: int,
    depth: int,
    num_heads: int,
    patch_size: int,
) -> FeatureBundle:
    sat_hw = (int(sat_size[1]), int(sat_size[0]))
    model = TransGeoGrounding(
        sat_size=sat_hw,
        drone_size=DRONE_SIZE,
        proj_dim=proj_dim,
        detail_dim=detail_dim,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        patch_size=patch_size,
    ).to(device)
    load_checkpoint(model, checkpoint_path)
    model.eval()

    loader = create_trans_loader(
        batch_size=batch_size,
        num_workers=num_workers,
        sat_size=sat_size,
        test_crop_ratio=test_crop_ratio,
        subset_heights=subset_heights,
        subset_angles=subset_angles,
    )

    query_feats: List[torch.Tensor] = []
    query_detail_feats: List[torch.Tensor] = []
    query_labels: List[str] = []
    query_drone_paths: List[str] = []
    query_satellite_paths: List[str] = []
    query_heights: List[int] = []
    query_angles: List[int] = []
    iou_values: List[float] = []
    center_distances: List[float] = []
    gallery_feat_dict: Dict[str, torch.Tensor] = {}
    gallery_detail_dict: Dict[str, torch.Tensor] = {}
    gallery_path_dict: Dict[str, str] = {}

    with torch.inference_mode():
        for batch in tqdm(loader, desc="Extract/eval [trans_geo]"):
            query_imgs = batch["target_pixel_values"].to(device, non_blocking=True)
            search_imgs = batch["search_pixel_values"].to(device, non_blocking=True)
            gt_bbox = batch["bbox"].to(device, non_blocking=True)
            outputs = model(query_imgs, search_imgs)

            query_feats.append(outputs["query_global"].detach().cpu())
            query_detail_feats.append(outputs["ground_detail"].detach().cpu())

            batch_labels = [_path_label(path) for path in batch["satellite_path"]]
            query_labels.extend(batch_labels)
            query_drone_paths.extend([str(p) for p in batch["drone_path"]])
            query_satellite_paths.extend([str(p) for p in batch["satellite_path"]])
            query_heights.extend([int(v) for v in batch["height"].tolist()])
            query_angles.extend([int(v) for v in batch["angle"].tolist()])

            aerial_global = outputs["aerial_global"].detach().cpu()
            aerial_detail = outputs["aerial_detail"].detach().cpu()
            for idx, label in enumerate(batch_labels):
                if label not in gallery_feat_dict:
                    gallery_feat_dict[label] = aerial_global[idx]
                    gallery_detail_dict[label] = aerial_detail[idx]
                    gallery_path_dict[label] = str(batch["satellite_path"][idx])

            pred_bbox = decode_bbox(
                outputs["heatmap_logits"],
                outputs["bbox_raw"],
                image_wh=(search_imgs.shape[-1], search_imgs.shape[-2]),
            )
            ious = bbox_iou(pred_bbox, gt_bbox, x1y1x2y2=True)
            for idx in range(pred_bbox.shape[0]):
                iou_values.append(float(ious[idx].item()))
                center_distances.append(center_distance(pred_bbox[idx].float(), gt_bbox[idx].float()))

    if not query_feats or not gallery_feat_dict:
        raise RuntimeError("No trans_geo features were extracted.")

    gallery_labels = sorted(gallery_feat_dict.keys())
    temperature = float(model.temperature.detach().clamp(min=0.01, max=1.0).cpu().item())
    logit_scale = float(model.logit_scale.detach().exp().clamp(max=100.0).cpu().item())

    return FeatureBundle(
        query_feats=torch.cat(query_feats, dim=0),
        query_labels=query_labels,
        query_drone_paths=query_drone_paths,
        query_satellite_paths=query_satellite_paths,
        query_heights=query_heights,
        query_angles=query_angles,
        iou_values=iou_values,
        center_distances=center_distances,
        gallery_labels=gallery_labels,
        gallery_satellite_paths=[gallery_path_dict.get(label, "") for label in gallery_labels],
        gallery_feats=torch.stack([gallery_feat_dict[label] for label in gallery_labels], dim=0),
        query_detail_feats=torch.cat(query_detail_feats, dim=0),
        gallery_detail_feats=torch.stack([gallery_detail_dict[label] for label in gallery_labels], dim=0),
        unify_logit_scale=logit_scale,
        unify_temperature=temperature,
    )


def score_grid_encoder_query(
    query_feat: torch.Tensor,
    gallery_feats: torch.Tensor,
    candidate_indices: List[int],
    device: torch.device,
    text_feat: Optional[torch.Tensor] = None,
    text_score_weight: float = 0.0,
    text_rerank_topk: int = 50,
) -> torch.Tensor:
    candidate_gallery = gallery_feats[candidate_indices].to(device)
    query_feat = query_feat.to(device)
    image_scores = torch.einsum("d,knd->kn", query_feat, candidate_gallery).max(dim=1)[0]
    if text_feat is None or text_score_weight <= 0.0:
        return image_scores

    text_feat = text_feat.to(device)
    text_scores = torch.einsum("d,knd->kn", text_feat, candidate_gallery).max(dim=1)[0]
    if text_rerank_topk <= 0 or text_rerank_topk >= image_scores.numel():
        return image_scores + float(text_score_weight) * text_scores

    reranked_scores = image_scores.clone()
    topk_indices = image_scores.topk(int(text_rerank_topk), dim=0).indices
    reranked_scores[topk_indices] = (
        image_scores[topk_indices]
        + float(text_score_weight) * text_scores[topk_indices]
    )
    return reranked_scores


def score_encoder_heat_query(
    query_feat: torch.Tensor,
    gallery_feats: torch.Tensor,
    candidate_indices: List[int],
    device: torch.device,
    text_feat: Optional[torch.Tensor] = None,
    text_score_weight: float = 0.0,
    text_rerank_topk: int = 50,
) -> torch.Tensor:
    return score_grid_encoder_query(
        query_feat=query_feat,
        gallery_feats=gallery_feats,
        candidate_indices=candidate_indices,
        device=device,
        text_feat=text_feat,
        text_score_weight=text_score_weight,
        text_rerank_topk=text_rerank_topk,
    )


def score_unify_query(
    bundle: FeatureBundle,
    query_index: int,
    candidate_indices: List[int],
    device: torch.device,
    score_mode: str,
) -> torch.Tensor:
    query_feat = bundle.query_feats[query_index].to(device)
    gallery_feats = bundle.gallery_feats[candidate_indices].to(device)
    scale = float(bundle.unify_logit_scale or 1.0)
    scores = scale * torch.einsum("d,kd->k", query_feat, gallery_feats)

    if score_mode == "global":
        return scores

    if bundle.query_detail_feats is None or bundle.gallery_detail_feats is None:
        raise ValueError("Unify rerank scoring requires detail features.")

    query_detail = bundle.query_detail_feats[query_index].to(device)
    gallery_detail = bundle.gallery_detail_feats[candidate_indices].to(device)
    temp = max(float(bundle.unify_temperature or 0.07), 1e-6)
    detail_scores = torch.einsum("d,kdhw->khw", query_detail, gallery_detail) / temp
    return scores + detail_scores.flatten(1).max(dim=1)[0]


def score_retrieval_and_uiou(
    model_type: str,
    bundle: FeatureBundle,
    include_map: Dict[str, Set[str]],
    device: torch.device,
    candidate_size: Optional[int],
    unify_score_mode: str,
    sampling_seed: int,
    encoder_heat_text_score_weight: float = 0.0,
    encoder_heat_text_rerank_topk: int = 50,
) -> Dict[str, Any]:
    label_to_gallery_index = {label: idx for idx, label in enumerate(bundle.gallery_labels)}
    norm_gallery_labels = [_normalize_label(label) for label in bundle.gallery_labels]
    all_indices = list(range(len(bundle.gallery_labels)))

    valid_queries = 0
    top1 = 0
    top5 = 0
    top10 = 0
    top1_success_drone_names: List[str] = []
    query_records: List[Dict[str, Any]] = []

    if candidate_size is not None and candidate_size <= 0:
        candidate_size = None

    for q_idx, gt_label in enumerate(bundle.query_labels):
        gt_gallery_index = label_to_gallery_index.get(gt_label)
        if gt_gallery_index is None:
            continue

        gt_norm = _normalize_label(gt_label)
        positive_label_set = set(include_map.get(gt_norm, set()))
        positive_label_set.add(gt_norm)

        if candidate_size is None or candidate_size >= len(all_indices):
            candidate_indices = all_indices
        else:
            negative_pool = [idx for idx in all_indices if idx != gt_gallery_index]
            rng = random.Random(int(sampling_seed) + int(q_idx))
            sampled_negatives = rng.sample(negative_pool, int(candidate_size) - 1)
            candidate_indices = sampled_negatives + [gt_gallery_index]

        if model_type in {"encoder_heat", "encoder_test"}:
            text_feat = (
                bundle.query_text_feats[q_idx]
                if bundle.query_text_feats is not None
                else None
            )
            score_vec = score_encoder_heat_query(
                bundle.query_feats[q_idx],
                bundle.gallery_feats,
                candidate_indices,
                device,
                text_feat=text_feat,
                text_score_weight=encoder_heat_text_score_weight,
                text_rerank_topk=encoder_heat_text_rerank_topk,
            )
        else:
            score_vec = score_unify_query(
                bundle=bundle,
                query_index=q_idx,
                candidate_indices=candidate_indices,
                device=device,
                score_mode=unify_score_mode,
            )

        k_max = min(10, int(score_vec.shape[0]))
        topk_local_indices = torch.topk(score_vec, k=k_max, dim=0).indices.tolist()
        top1_local_idx = int(topk_local_indices[0])
        top1_global_idx = int(candidate_indices[top1_local_idx])
        positive_local_indices = [
            local_idx
            for local_idx, global_idx in enumerate(candidate_indices)
            if norm_gallery_labels[global_idx] in positive_label_set
        ]

        top1_correct = any(idx in topk_local_indices[:1] for idx in positive_local_indices)
        top5_correct = any(idx in topk_local_indices[: min(5, k_max)] for idx in positive_local_indices)
        top10_correct = any(idx in topk_local_indices[: min(10, k_max)] for idx in positive_local_indices)

        valid_queries += 1
        top1 += int(top1_correct)
        top5 += int(top5_correct)
        top10 += int(top10_correct)
        if top1_correct:
            top1_success_drone_names.append(_drone_image_name(bundle.query_drone_paths[q_idx]))

        iou = float(bundle.iou_values[q_idx])
        u_iou = iou if top1_correct else 0.0
        query_records.append(
            {
                "query_index": int(q_idx),
                "height": int(bundle.query_heights[q_idx]),
                "angle": int(bundle.query_angles[q_idx]),
                "gt_label": gt_norm,
                "pred_label": _normalize_label(bundle.gallery_labels[top1_global_idx]),
                "top1_correct": bool(top1_correct),
                "top5_correct": bool(top5_correct),
                "top10_correct": bool(top10_correct),
                "iou": iou,
                "uIoU": float(u_iou),
                "center_distance": float(bundle.center_distances[q_idx]),
                "drone_path": bundle.query_drone_paths[q_idx],
                "gt_satellite_path": bundle.query_satellite_paths[q_idx],
                "pred_satellite_path": bundle.gallery_satellite_paths[top1_global_idx],
            }
        )

    if valid_queries == 0:
        raise RuntimeError("No valid queries matched gallery labels.")

    return {
        "num_queries": int(valid_queries),
        "top1_hits": int(top1),
        "top5_hits": int(top5),
        "top10_hits": int(top10),
        "recall@1": float(top1 / valid_queries),
        "recall@5": float(top5 / valid_queries),
        "recall@10": float(top10 / valid_queries),
        "top1_errors": int(valid_queries - top1),
        "top1_success_drone_names": top1_success_drone_names,
        "query_records": query_records,
    }


def summarize_records(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not records:
        return {
            "num_samples": 0,
            "top1_hits": 0,
            "top5_hits": 0,
            "top10_hits": 0,
            "recall@1": 0.0,
            "recall@5": 0.0,
            "recall@10": 0.0,
            "mean_iou": 0.0,
            "ratio_iou_gt_0_5": 0.0,
            "ratio_iou_gt_0_25": 0.0,
            "uIoU": 0.0,
            "ratio_uIoU_gt_0_25": 0.0,
            "ratio_uIoU_gt_25": 0.0,
            "mean_center_distance": 0.0,
        }

    iou_arr = np.array([float(item["iou"]) for item in records], dtype=np.float32)
    uiou_arr = np.array([float(item["uIoU"]) for item in records], dtype=np.float32)
    center_arr = np.array([float(item["center_distance"]) for item in records], dtype=np.float32)
    n = int(len(records))
    top1_hits = int(sum(bool(item["top1_correct"]) for item in records))
    top5_hits = int(sum(bool(item["top5_correct"]) for item in records))
    top10_hits = int(sum(bool(item["top10_correct"]) for item in records))
    ratio_uiou_gt_25 = float((uiou_arr > 0.25).mean())
    return {
        "num_samples": n,
        "top1_hits": top1_hits,
        "top5_hits": top5_hits,
        "top10_hits": top10_hits,
        "recall@1": float(top1_hits / n),
        "recall@5": float(top5_hits / n),
        "recall@10": float(top10_hits / n),
        "mean_iou": float(iou_arr.mean()),
        "ratio_iou_gt_0_5": float((iou_arr > 0.5).mean()),
        "ratio_iou_gt_0_25": float((iou_arr > 0.25).mean()),
        "uIoU": float(uiou_arr.mean()),
        "ratio_uIoU_gt_0_25": ratio_uiou_gt_25,
        "ratio_uIoU_gt_25": ratio_uiou_gt_25,
        "mean_center_distance": float(center_arr.mean()),
    }


def group_summaries(
    records: Sequence[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    subset_groups: Dict[Tuple[int, int], List[Dict[str, Any]]] = {}
    height_groups: Dict[int, List[Dict[str, Any]]] = {}
    angle_groups: Dict[int, List[Dict[str, Any]]] = {}
    for item in records:
        height = int(item["height"])
        angle = int(item["angle"])
        subset_groups.setdefault((height, angle), []).append(item)
        height_groups.setdefault(height, []).append(item)
        angle_groups.setdefault(angle, []).append(item)

    per_subset = [
        {"height": height, "angle": angle, **summarize_records(items)}
        for (height, angle), items in sorted(subset_groups.items())
    ]
    per_height = [
        {"height": height, **summarize_records(items)}
        for height, items in sorted(height_groups.items())
    ]
    per_angle = [
        {"angle": angle, **summarize_records(items)}
        for angle, items in sorted(angle_groups.items())
    ]
    return per_subset, per_height, per_angle


def evaluate_model(model_type: str, checkpoint_path: str, args: argparse.Namespace) -> Dict[str, Any]:
    device = torch.device(args.device)
    sat_size = (int(args.sat_size[0]), int(args.sat_size[1]))
    subset_heights = args.subset_heights if args.subset_heights else None
    subset_angles = args.subset_angles if args.subset_angles else None

    if model_type in {"encoder_heat", "encoder_test"}:
        encoder_cls = Encoder_test if model_type == "encoder_test" else Encoder_heat
        encoder_eval_use_text = False
        bundle = extract_encoder_heat_features(
            checkpoint_path=checkpoint_path,
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            sat_size=sat_size,
            test_crop_ratio=args.test_crop_ratio,
            subset_heights=subset_heights,
            subset_angles=subset_angles,
            heatmap_confidence_weight=args.heatmap_confidence_weight,
            use_text=encoder_eval_use_text,
            use_angle=args.encoder_heat_use_angle,
            use_ap=args.encoder_heat_use_ap,
            use_heatmap=args.encoder_heat_use_heatmap,
            encoder_cls=encoder_cls,
            desc=model_type,
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
        )
    elif model_type == "unify_geo":
        bundle = extract_unify_features(
            checkpoint_path=checkpoint_path,
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            sat_size=sat_size,
            test_crop_ratio=args.test_crop_ratio,
            subset_heights=subset_heights,
            subset_angles=subset_angles,
            base_dim=args.base_dim,
            proj_dim=args.proj_dim,
            detail_dim=args.detail_dim,
            backbone_name=args.unify_backbone_name,
        )
    elif model_type == "trans_geo":
        bundle = extract_trans_features(
            checkpoint_path=checkpoint_path,
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            sat_size=sat_size,
            test_crop_ratio=args.test_crop_ratio,
            subset_heights=subset_heights,
            subset_angles=subset_angles,
            proj_dim=args.trans_proj_dim,
            detail_dim=args.trans_detail_dim,
            embed_dim=args.trans_embed_dim,
            depth=args.trans_depth,
            num_heads=args.trans_num_heads,
            patch_size=args.trans_patch_size,
        )
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    include_map = _load_include_map(args.include_file)
    scored = score_retrieval_and_uiou(
        model_type=model_type,
        bundle=bundle,
        include_map=include_map,
        device=device,
        candidate_size=args.candidate_size,
        unify_score_mode=args.unify_score_mode,
        sampling_seed=args.seed,
        encoder_heat_text_score_weight=0.0,
        encoder_heat_text_rerank_topk=0,
    )
    records = scored.pop("query_records")
    per_subset, per_height, per_angle = group_summaries(records)
    overall = summarize_records(records)

    return {
        "model_type": model_type,
        "checkpoint": checkpoint_path,
        "sat_size": {"height": int(sat_size[0]), "width": int(sat_size[1])},
        "candidate_size": args.candidate_size,
        "test_crop_ratio": float(args.test_crop_ratio),
        "seed": int(args.seed),
        "encoder_use_text": False if model_type in {"encoder_heat", "encoder_test"} else None,
        "encoder_use_angle": bool(args.encoder_heat_use_angle) if model_type in {"encoder_heat", "encoder_test"} else None,
        "encoder_heat_use_ap": bool(args.encoder_heat_use_ap) if model_type in {"encoder_heat", "encoder_test"} else None,
        "encoder_heat_use_heatmap": bool(args.encoder_heat_use_heatmap) if model_type in {"encoder_heat", "encoder_test"} else None,
        "heatmap_confidence_weight": float(args.heatmap_confidence_weight) if model_type in {"encoder_heat", "encoder_test"} else None,
        "encoder_heat_text_score_weight": 0.0 if model_type in {"encoder_heat", "encoder_test"} else None,
        "encoder_heat_text_rerank_topk": 0 if model_type in {"encoder_heat", "encoder_test"} else None,
        "lora": (
            {
                "rank": int(args.lora_rank),
                "alpha": float(args.lora_alpha),
                "dropout": float(args.lora_dropout),
            }
            if model_type in {"encoder_heat", "encoder_test"}
            else None
        ),
        "include_file": args.include_file,
        "subset_heights": subset_heights or DEFAULT_SUBSET_HEIGHTS,
        "subset_angles": subset_angles or DEFAULT_SUBSET_ANGLES,
        "num_gallery": int(len(bundle.gallery_labels)),
        "num_samples": int(overall["num_samples"]),
        "overall": overall,
        "retrieval": {key: value for key, value in scored.items() if key != "top1_success_drone_names"},
        "top1_success_drone_names": scored["top1_success_drone_names"],
        "per_subset": per_subset,
        "per_height": per_height,
        "per_angle": per_angle,
        "query_records": records if args.save_query_records else [],
    }


def checkpoint_for_model(model_type: str, args: argparse.Namespace) -> str:
    if args.checkpoint:
        if len(args.model_types) != 1:
            raise ValueError("--checkpoint can only be used with exactly one --model-types value.")
        return args.checkpoint
    if model_type in {"encoder_heat", "encoder_test"}:
        return args.encoder_heat_checkpoint
    if model_type == "unify_geo":
        return args.unify_checkpoint
    if model_type == "trans_geo":
        return args.trans_checkpoint
    raise ValueError(f"Unsupported model_type: {model_type}")


def save_metrics(metrics: Dict[str, Any], output_dir: str, name_suffix: Optional[str] = None) -> str:
    os.makedirs(output_dir, exist_ok=True)
    checkpoint_name = Path(str(metrics["checkpoint"])).resolve().parent.name
    model_type = str(metrics["model_type"])
    suffix = name_suffix or checkpoint_name
    out_file = os.path.join(output_dir, f"test_unify_{model_type}_{suffix}.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    success_txt = os.path.join(
        output_dir,
        f"test_unify_{model_type}_{suffix}_top1_success_drone_names.txt",
    )
    with open(success_txt, "w", encoding="utf-8") as f:
        for name in metrics.get("top1_success_drone_names", []):
            f.write(f"{name}\n")
    return out_file


def print_summary(metrics: Dict[str, Any], out_file: str) -> None:
    overall = metrics["overall"]
    print("\n=== test_unify Summary ===")
    print(f"Model: {metrics['model_type']}")
    print(f"Checkpoint: {metrics['checkpoint']}")
    print(f"Samples: {overall['num_samples']} | Gallery: {metrics['num_gallery']}")
    print(
        f"R@1={overall['recall@1']:.4f} "
        f"R@5={overall['recall@5']:.4f} "
        f"R@10={overall['recall@10']:.4f}"
    )
    print(
        f"mIoU={overall['mean_iou']:.4f} "
        f"IoU>0.5={overall['ratio_iou_gt_0_5']:.4f} "
        f"uIoU={overall['uIoU']:.4f} "
        f"uIoU>0.25={overall['ratio_uIoU_gt_0_25']:.4f}"
    )
    print(f"Mean center distance: {overall['mean_center_distance']:.4f}px")
    print(f"Saved: {out_file}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate retrieval, grounding, and uIoU in one feature-forward pass."
    )
    parser.add_argument(
        "--model-types",
        nargs="+",
        default=["encoder_heat"],
        choices=["encoder_heat", "encoder_test", "unify_geo", "trans_geo"],
    )
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--encoder-heat-checkpoint", type=str, default=DEFAULT_ENCODER_HEAT_CHECKPOINT)
    parser.add_argument("--encoder-heat-config-dir", type=str, default=DEFAULT_ENCODER_HEAT_CONFIG_DIR)
    parser.add_argument("--eval-encoder-heat-configs", action="store_true")
    parser.add_argument("--encoder-heat-checkpoint-name", type=str, default="last.pth")
    parser.add_argument("--encoder-heat-use-text", action="store_true", default=True)
    parser.add_argument("--no-encoder-heat-use-text", dest="encoder_heat_use_text", action="store_false")
    parser.add_argument("--encoder-heat-use-angle", action="store_true", default=True)
    parser.add_argument("--no-encoder-heat-use-angle", dest="encoder_heat_use_angle", action="store_false")
    parser.add_argument("--encoder-heat-use-ap", action="store_true", default=True)
    parser.add_argument("--no-encoder-heat-use-ap", dest="encoder_heat_use_ap", action="store_false")
    parser.add_argument("--encoder-heat-use-heatmap", action="store_true", default=True)
    parser.add_argument("--no-encoder-heat-use-heatmap", dest="encoder_heat_use_heatmap", action="store_false")
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--unify-checkpoint", type=str, default=DEFAULT_UNIFY_CHECKPOINT)
    parser.add_argument("--trans-checkpoint", type=str, default=DEFAULT_TRANS_CHECKPOINT)
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument(
        "--sat-size",
        type=int,
        nargs=2,
        metavar=("HEIGHT", "WIDTH"),
        default=[432, 768],
        help="Satellite image size as HEIGHT WIDTH.",
    )
    parser.add_argument("--test-crop-ratio", type=float, default=1.0)
    parser.add_argument("--subset-heights", type=int, nargs="*", default=None)
    parser.add_argument("--subset-angles", type=int, nargs="*", default=None)
    parser.add_argument("--candidate-size", type=int, default=100, help="Use <=0 for full-gallery retrieval.")
    parser.add_argument("--include-file", type=str, default=DEFAULT_INCLUDE_FILE)
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--heatmap-confidence-weight", type=float, default=0.5)
    parser.add_argument(
        "--encoder-heat-text-score-weight",
        type=float,
        default=0.0,
        help="Deprecated: text score is disabled during test.",
    )
    parser.add_argument(
        "--encoder-heat-text-rerank-topk",
        type=int,
        default=50,
        help="Deprecated: text score is disabled during test.",
    )
    parser.add_argument(
        "--output-suffix",
        type=str,
        default=None,
        help="Optional suffix for the saved metrics filename. Defaults to checkpoint directory name.",
    )
    parser.add_argument("--unify-score-mode", choices=["rerank", "global"], default="rerank")
    parser.add_argument("--base-dim", type=int, default=96)
    parser.add_argument("--unify-backbone-name", type=str, default=UNIFY_BACKBONE_NAME)
    parser.add_argument("--proj-dim", type=int, default=PROJECTION_DIM)
    parser.add_argument("--detail-dim", type=int, default=384)
    parser.add_argument("--trans-proj-dim", type=int, default=PROJECTION_DIM)
    parser.add_argument("--trans-detail-dim", type=int, default=TRANS_DETAIL_DIM)
    parser.add_argument("--trans-embed-dim", type=int, default=TRANS_EMBED_DIM)
    parser.add_argument("--trans-depth", type=int, default=12)
    parser.add_argument("--trans-num-heads", type=int, default=TRANS_NUM_HEADS)
    parser.add_argument("--trans-patch-size", type=int, default=TRANS_PATCH_SIZE)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--save-query-records", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.candidate_size is not None and args.candidate_size <= 0:
        args.candidate_size = None
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    all_results: List[Dict[str, Any]] = []
    if args.eval_encoder_heat_configs:
        runs = discover_encoder_heat_runs(
            config_dir=args.encoder_heat_config_dir,
            checkpoint_name=args.encoder_heat_checkpoint_name,
        )
        for run in runs:
            print(
                f"\nEvaluating encoder_heat config={run['config_name']} "
                f"checkpoint={run['checkpoint_path']}"
            )
            args.encoder_heat_use_text = False
            args.encoder_heat_use_angle = bool(run["use_angle"])
            args.encoder_heat_use_ap = bool(run["use_ap"])
            args.encoder_heat_use_heatmap = bool(run["use_heatmap"])
            args.heatmap_confidence_weight = float(run["heatmap_confidence_weight"])
            metrics = evaluate_model("encoder_heat", str(run["checkpoint_path"]), args)
            metrics["config_path"] = run["config_path"]
            metrics["config_name"] = run["config_name"]
            metrics["exp_name"] = run["exp_name"]
            metrics["encoder_heat_settings"] = {
                "use_text": False,
                "train_config_use_text": bool(run["use_text"]),
                "use_angle": bool(run["use_angle"]),
                "use_ap": bool(run["use_ap"]),
                "use_heatmap": bool(run["use_heatmap"]),
                "heatmap_confidence_weight": float(run["heatmap_confidence_weight"]),
                "text_score_weight": 0.0,
            }
            out_file = save_metrics(metrics, args.output_dir, name_suffix=str(run["config_name"]))
            print_summary(metrics, out_file)
            all_results.append(metrics)

        combined_file = os.path.join(args.output_dir, "test_unify_encoder_heat_configs_summary.json")
        os.makedirs(args.output_dir, exist_ok=True)
        with open(combined_file, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2)
        print(f"Combined encoder_heat config summary saved: {combined_file}")
        return

    for model_type in args.model_types:
        checkpoint_path = checkpoint_for_model(model_type, args)
        metrics = evaluate_model(model_type, checkpoint_path, args)
        out_file = save_metrics(metrics, args.output_dir, name_suffix=args.output_suffix)
        print_summary(metrics, out_file)
        all_results.append(metrics)

    if len(all_results) > 1:
        combined_file = os.path.join(args.output_dir, "test_unify_summary.json")
        with open(combined_file, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2)
        print(f"Combined summary saved: {combined_file}")


if __name__ == "__main__":
    main()
