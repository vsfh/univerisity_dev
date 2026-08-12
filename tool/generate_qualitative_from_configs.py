#!/usr/bin/env python3
"""Run two configured models on four cases and generate the qualitative figure.

This is the inference-backed companion to ``generate_qualitative_figure.py``.
It uses the repository's test dataset, retrieval scoring, candidate sampling,
grounding decoder, and checkpoint loader. The full model supplies retrieval,
oracle grounding, and heatmaps; the ablation model supplies w/o-heading boxes.
An actual per-case YAML is saved before the figure is rendered.
"""

from __future__ import annotations

import argparse
import gc
import os
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Subset, default_collate
from tqdm import tqdm

from bbox.yolo_utils import bbox_iou, build_target, eval_iou_acc
from test_unify import (
    DEFAULT_INCLUDE_FILE,
    ENCODER_CLASSES,
    MODEL_NAME,
    _load_include_map,
    _normalize_label,
    _path_label,
    add_heatmap_to_confidence,
    bool_from_config,
    build_geo_features,
    center_distance,
    create_encoder_loader,
    load_checkpoint,
    load_yaml,
    parse_anchors,
    resolve_encoder_heat_checkpoint,
    score_encoder_heat_query,
)

from generate_qualitative_figure import (
    Selection,
    load_samples,
    make_figure,
    report_selection,
    save_selection,
    select_samples,
)


# --- Configuration ---
DEFAULT_FULL_CONFIG = (
    ROOT / "configs/unified_siglip_supp/single_config/baseline.yaml"
)
DEFAULT_ABLATION_CONFIG = (
    ROOT / "configs/unified_siglip_supp/single_config/baseline_wo_heading.yaml"
)
DEFAULT_OUTPUT = ROOT / "figures/qualitative.pdf"
DEFAULT_SAT_SIZE = (432, 768)


@dataclass(frozen=True)
class RunSpec:
    role: str
    config_path: Path
    checkpoint_path: Path
    model_name: str
    encoder_type: str
    use_text: bool
    use_angle: bool
    use_ap: bool
    use_heatmap: bool
    heatmap_confidence_weight: float
    lora_rank: int
    lora_alpha: float
    lora_dropout: float
    text_score_weight: float
    text_rerank_topk: int


@dataclass
class GroundingOutput:
    pred_bboxes: List[List[float]]
    gt_bboxes: List[List[float]]
    ious: List[float]
    cdes: List[float]
    query_feats: torch.Tensor
    text_feats: Optional[torch.Tensor]
    heatmaps: List[Optional[np.ndarray]]


@dataclass(frozen=True)
class AutoCandidate:
    dataset_index: int
    satellite_path: str
    top1_correct: bool
    oracle_iou: float
    heading_gain: float
    altitude: int
    heading: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run full and w/o-heading configs on four UAV cases, then generate "
            "a paper-ready 4x4 qualitative figure."
        )
    )
    parser.add_argument(
        "--full-config",
        type=Path,
        default=DEFAULT_FULL_CONFIG,
        help=f"Full-model YAML. Default: {DEFAULT_FULL_CONFIG}",
    )
    parser.add_argument(
        "--ablation-config",
        type=Path,
        default=DEFAULT_ABLATION_CONFIG,
        help=f"w/o-heading YAML. Default: {DEFAULT_ABLATION_CONFIG}",
    )
    parser.add_argument("--full-checkpoint", type=Path, default=None)
    parser.add_argument("--ablation-checkpoint", type=Path, default=None)
    parser.add_argument("--checkpoint-name", default="last.pth")
    case_group = parser.add_mutually_exclusive_group(required=True)
    case_group.add_argument(
        "--cases",
        nargs=4,
        metavar=("CASE1", "CASE2", "CASE3", "CASE4"),
        help=(
            "Exactly four UAV selectors. Use an absolute drone path, "
            "SAT_ID/HEIGHT_ANGLE.png, or SAT_ID:HEIGHT:ANGLE."
        ),
    )
    case_group.add_argument(
        "--auto-select",
        action="store_true",
        help=(
            "Scan the test split and automatically find four unique satellite "
            "cases satisfying the strict IoU and heading-gain constraints."
        ),
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--png-output", type=Path, default=None)
    parser.add_argument("--records-output", type=Path, default=None)
    parser.add_argument("--selected-yaml", type=Path, default=None)
    parser.add_argument("--asset-dir", type=Path, default=None)
    parser.add_argument(
        "--device",
        default="cuda:0" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--gallery-batch-size", type=int, default=24)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--sat-size",
        type=int,
        nargs=2,
        metavar=("HEIGHT", "WIDTH"),
        default=list(DEFAULT_SAT_SIZE),
    )
    parser.add_argument("--test-crop-ratio", type=float, default=1.0)
    parser.add_argument(
        "--candidate-size",
        type=int,
        default=100,
        help="Candidate set size matching test_unify.py; use <=0 for full gallery.",
    )
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--include-file", default=DEFAULT_INCLUDE_FILE)
    parser.add_argument("--success-iou", type=float, default=0.5)
    parser.add_argument("--failure-iou", type=float, default=0.25)
    parser.add_argument("--heading-gain", type=float, default=0.15)
    parser.add_argument(
        "--min-oracle-iou",
        type=float,
        default=0.60,
        help="Strict lower IoU bound used by --auto-select. Default: 0.60",
    )
    parser.add_argument(
        "--min-heading-gain",
        type=float,
        default=0.20,
        help="Minimum full minus w/o-heading IoU used by --auto-select. Default: 0.20",
    )
    parser.add_argument(
        "--scan-batch-size",
        type=int,
        default=24,
        help="Inference batch size while searching with --auto-select.",
    )
    parser.add_argument(
        "--max-scan-samples",
        type=int,
        default=20000,
        help="Maximum test queries inspected by --auto-select.",
    )
    parser.add_argument("--width", type=float, default=7.1)
    parser.add_argument("--height", type=float, default=4.65)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--heatmap-cmap", choices=("turbo", "magma"), default="turbo")
    parser.add_argument("--heatmap-alpha", type=float, default=0.42)
    args = parser.parse_args()
    if (
        args.batch_size <= 0
        or args.gallery_batch_size <= 0
        or args.scan_batch_size <= 0
    ):
        parser.error("batch sizes must be positive")
    if args.num_workers < 0:
        parser.error("--num-workers cannot be negative")
    if not 0.0 < args.test_crop_ratio <= 1.0:
        parser.error("--test-crop-ratio must be in (0, 1]")
    if not 0.0 <= args.min_oracle_iou <= 1.0:
        parser.error("--min-oracle-iou must be in [0, 1]")
    if not 0.0 <= args.min_heading_gain <= 1.0:
        parser.error("--min-heading-gain must be in [0, 1]")
    if args.max_scan_samples <= 0:
        parser.error("--max-scan-samples must be positive")
    return args


def load_run_spec(
    role: str,
    config_path: Path,
    explicit_checkpoint: Optional[Path],
    checkpoint_name: str,
) -> RunSpec:
    config_path = config_path.expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"{role} config not found: {config_path}")
    payload = load_yaml(str(config_path))
    config = payload.get("config", {}) or {}
    if not isinstance(config, Mapping):
        raise ValueError(f"{role} config.config must be a mapping: {config_path}")
    encoder_type = str(config.get("ENCODER_TYPE", "heat")).lower()
    if encoder_type not in {"heat", "test"}:
        raise ValueError(
            f"{role} ENCODER_TYPE={encoder_type!r} is unsupported by this tool; "
            "use heat or test"
        )
    checkpoint_path = (
        explicit_checkpoint.expanduser().resolve()
        if explicit_checkpoint is not None
        else Path(
            resolve_encoder_heat_checkpoint(
                str(config_path), payload, checkpoint_name
            )
        ).resolve()
    )
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"{role} checkpoint not found: {checkpoint_path}")
    return RunSpec(
        role=role,
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        model_name=str(config.get("MODEL_NAME", MODEL_NAME)),
        encoder_type=encoder_type,
        use_text=bool_from_config(config, "USE_TEXT_INPUT", True),
        use_angle=bool_from_config(config, "USE_ANGLE_INPUT", True),
        use_ap=bool(payload.get("use_ap", True)),
        use_heatmap=bool_from_config(config, "USE_HEATMAP_LOSS", True),
        heatmap_confidence_weight=float(
            config.get("HEATMAP_CONFIDENCE_WEIGHT", 0.5)
        ),
        lora_rank=int(config.get("LORA_RANK", 8)),
        lora_alpha=float(config.get("LORA_ALPHA", 16.0)),
        lora_dropout=float(config.get("LORA_DROPOUT", 0.05)),
        text_score_weight=float(config.get("ENCODER_HEAT_TEXT_SCORE_WEIGHT", 0.0)),
        text_rerank_topk=int(config.get("ENCODER_HEAT_TEXT_RERANK_TOPK", 0)),
    )


def create_model(spec: RunSpec, device: torch.device) -> torch.nn.Module:
    encoder_cls = ENCODER_CLASSES[
        "encoder_heat" if spec.encoder_type == "heat" else "encoder_test"
    ]
    model = encoder_cls(
        model_name=spec.model_name,
        proj_dim=768,
        usesg=True,
        useap=spec.use_ap,
        use_heatmap=spec.use_heatmap,
        lora_rank=spec.lora_rank,
        lora_alpha=spec.lora_alpha,
        lora_dropout=spec.lora_dropout,
    ).to(device)
    load_checkpoint(model, str(spec.checkpoint_path))
    model.eval()
    print(
        f"[model] {spec.role}: {spec.config_path.name} -> "
        f"{spec.checkpoint_path} (text={spec.use_text}, angle={spec.use_angle})"
    )
    return model


def create_dataset(
    model_name: str,
    sat_size: Tuple[int, int],
    test_crop_ratio: float,
) -> Any:
    loader = create_encoder_loader(
        batch_size=1,
        num_workers=0,
        sat_size=sat_size,
        test_crop_ratio=test_crop_ratio,
        subset_heights=None,
        subset_angles=None,
        model_name=model_name,
    )
    return loader.dataset


def canonical_selector(selector: str) -> str:
    selector = selector.strip()
    colon_match = re.fullmatch(r"(\d+):(\d+):(\d+)", selector)
    if colon_match:
        satellite_id, height, angle = colon_match.groups()
        return f"{int(satellite_id):04d}/{int(height)}_{int(angle)}.png"
    return selector.replace("\\", "/")


def resolve_case_indices(dataset: Any, selectors: Sequence[str]) -> List[int]:
    path_to_index = {
        os.path.abspath(str(sample["drone_path"])): index
        for index, sample in enumerate(dataset.samples)
    }
    resolved: List[int] = []
    for raw_selector in selectors:
        selector = canonical_selector(raw_selector)
        expanded = os.path.abspath(os.path.expanduser(selector))
        if expanded in path_to_index:
            matches = [path_to_index[expanded]]
        else:
            normalized_suffix = selector.lstrip("./")
            matches = [
                index
                for path, index in path_to_index.items()
                if path.replace("\\", "/").endswith("/" + normalized_suffix)
            ]
        if not matches:
            raise ValueError(
                f"Case {raw_selector!r} was not found in the test dataset. "
                "Use SAT_ID/HEIGHT_ANGLE.png, e.g. 0000/250_45.png."
            )
        if len(matches) > 1:
            examples = [
                str(dataset.samples[index]["drone_path"]) for index in matches[:4]
            ]
            raise ValueError(
                f"Case {raw_selector!r} is ambiguous ({len(matches)} matches): "
                + ", ".join(examples)
            )
        resolved.append(matches[0])
    if len(set(resolved)) != 4:
        raise ValueError("The four --cases must resolve to four distinct samples")
    for row, index in enumerate(resolved, start=1):
        print(f"[case] input {row}: {dataset.samples[index]['drone_path']}")
    return resolved


def prepare_case_batch(
    dataset: Any,
    indices: Sequence[int],
) -> Tuple[Dict[str, Any], List[Image.Image]]:
    dataset.return_search_image = True
    raw_items: List[Dict[str, Any]] = []
    oracle_images: List[Image.Image] = []
    for index in indices:
        item = dataset[index]
        item.pop("query_image", None)
        oracle_image = item.pop("search_image")
        oracle_images.append(oracle_image.copy())
        raw_items.append(item)
    dataset.return_search_image = False
    return default_collate(raw_items), oracle_images


def model_inputs(
    batch: Mapping[str, Any],
    spec: RunSpec,
    device: torch.device,
) -> Dict[str, Any]:
    return {
        "anchor_pixel_values": batch["target_pixel_values"].to(
            device, non_blocking=True
        ),
        "search_pixel_values": batch["search_pixel_values"].to(
            device, non_blocking=True
        ),
        "input_ids": (
            batch["input_ids"].to(device, non_blocking=True)
            if spec.use_text
            else None
        ),
        "attention_mask": (
            batch["attention_mask"].to(device, non_blocking=True)
            if spec.use_text
            else None
        ),
        "angle": build_geo_features(batch, device) if spec.use_angle else None,
    }


def decode_grounding(
    pred_anchor: torch.Tensor,
    refine_outputs: Any,
    heatmap_logits: Optional[torch.Tensor],
    gt_bbox: torch.Tensor,
    anchors: torch.Tensor,
    sat_size: Tuple[int, int],
    confidence_weight: float,
    use_heatmap: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    batch_size = int(pred_anchor.shape[0])
    pred_anchor = pred_anchor.view(
        batch_size,
        9,
        5,
        pred_anchor.shape[-2],
        pred_anchor.shape[-1],
    )
    pred_anchor = add_heatmap_to_confidence(
        pred_anchor,
        heatmap_logits if use_heatmap else None,
        confidence_weight=confidence_weight if use_heatmap else 0.0,
    )
    image_wh = (int(sat_size[1]), int(sat_size[0]))
    grid_wh = (int(pred_anchor.shape[-1]), int(pred_anchor.shape[-2]))
    _, best_anchor_gi_gj = build_target(gt_bbox, anchors, image_wh, grid_wh)
    _, _, _, _, pred_bbox, target_bbox = eval_iou_acc(
        pred_anchor,
        gt_bbox,
        anchors,
        best_anchor_gi_gj[:, 1],
        best_anchor_gi_gj[:, 2],
        image_wh,
        iou_threshold_list=[0.5, 0.25],
    )
    if isinstance(refine_outputs, Mapping) and "bbox" in refine_outputs:
        pred_bbox = refine_outputs["bbox"].to(
            device=gt_bbox.device, dtype=gt_bbox.dtype
        )
        target_bbox = gt_bbox
    return pred_bbox, target_bbox


def run_grounding(
    model: torch.nn.Module,
    spec: RunSpec,
    batch: Mapping[str, Any],
    device: torch.device,
    sat_size: Tuple[int, int],
) -> GroundingOutput:
    anchors = parse_anchors(device)
    gt_bbox = batch["bbox"].to(device, non_blocking=True)
    with torch.inference_mode():
        outputs = model(**model_inputs(batch, spec, device))
        if len(outputs) != 7:
            raise ValueError(
                f"{spec.role} model returned {len(outputs)} outputs; expected 7"
            )
        (
            pred_anchor,
            _,
            text_feats,
            query_feats,
            _,
            refine_outputs,
            heatmap_logits,
        ) = outputs
        pred_bbox, target_bbox = decode_grounding(
            pred_anchor=pred_anchor,
            refine_outputs=refine_outputs,
            heatmap_logits=heatmap_logits,
            gt_bbox=gt_bbox,
            anchors=anchors,
            sat_size=sat_size,
            confidence_weight=spec.heatmap_confidence_weight,
            use_heatmap=spec.use_heatmap,
        )
        ious = bbox_iou(pred_bbox, target_bbox, x1y1x2y2=True)

    heatmaps: List[Optional[np.ndarray]] = [None] * int(pred_bbox.shape[0])
    if heatmap_logits is not None:
        heatmaps = [
            heatmap_logits[index].detach().float().cpu().numpy().squeeze()
            for index in range(int(heatmap_logits.shape[0]))
        ]
    return GroundingOutput(
        pred_bboxes=pred_bbox.detach().float().cpu().tolist(),
        gt_bboxes=target_bbox.detach().float().cpu().tolist(),
        ious=[float(value) for value in ious.detach().float().cpu().tolist()],
        cdes=[
            center_distance(pred_bbox[index].float(), target_bbox[index].float())
            for index in range(int(pred_bbox.shape[0]))
        ],
        query_feats=F.normalize(query_feats.detach().float(), p=2, dim=1).cpu(),
        text_feats=(
            F.normalize(text_feats.detach().float(), p=2, dim=1).cpu()
            if text_feats is not None
            else None
        ),
        heatmaps=heatmaps,
    )


def unique_gallery_indices(dataset: Any) -> List[int]:
    label_to_index: Dict[str, int] = {}
    for index, sample in enumerate(dataset.samples):
        label = _path_label(str(sample["satellite_path"]))
        label_to_index.setdefault(label, index)
    return [label_to_index[label] for label in sorted(label_to_index)]


def extract_gallery(
    model: torch.nn.Module,
    dataset: Any,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> Tuple[List[str], List[str], torch.Tensor]:
    indices = unique_gallery_indices(dataset)
    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,
    )
    labels: List[str] = []
    paths: List[str] = []
    features: List[torch.Tensor] = []
    with torch.inference_mode():
        for batch in tqdm(loader, desc="Full-model gallery extraction"):
            search_images = batch["search_pixel_values"].to(
                device, non_blocking=True
            )
            satellite_output = model._vision_forward(
                search_images,
                interpolate_pos_encoding=True,
            )
            grid_features = model.attnPooling(
                satellite_output.last_hidden_state, 9
            )
            features.append(F.normalize(grid_features, p=2, dim=2).cpu())
            batch_paths = [str(path) for path in batch["satellite_path"]]
            paths.extend(batch_paths)
            labels.extend([_path_label(path) for path in batch_paths])
    if not features:
        raise RuntimeError("No gallery features were extracted")
    return labels, paths, torch.cat(features, dim=0)


def score_retrieval(
    full_output: GroundingOutput,
    dataset: Any,
    case_indices: Sequence[int],
    gallery_labels: Sequence[str],
    gallery_paths: Sequence[str],
    gallery_feats: torch.Tensor,
    include_map: Mapping[str, Sequence[str]],
    device: torch.device,
    candidate_size: int,
    seed: int,
    text_score_weight: float,
    text_rerank_topk: int,
) -> List[Dict[str, Any]]:
    label_to_index = {label: index for index, label in enumerate(gallery_labels)}
    normalized_gallery = [_normalize_label(label) for label in gallery_labels]
    all_indices = list(range(len(gallery_labels)))
    records: List[Dict[str, Any]] = []
    for local_index, dataset_index in enumerate(case_indices):
        sample = dataset.samples[dataset_index]
        gt_label = _path_label(str(sample["satellite_path"]))
        gt_gallery_index = label_to_index.get(gt_label)
        if gt_gallery_index is None:
            raise RuntimeError(f"Ground-truth tile absent from gallery: {gt_label}")
        if candidate_size <= 0 or candidate_size >= len(all_indices):
            candidate_indices = all_indices
        else:
            negative_pool = [
                index for index in all_indices if index != gt_gallery_index
            ]
            rng = random.Random(int(seed) + int(dataset_index))
            candidate_indices = (
                rng.sample(negative_pool, int(candidate_size) - 1)
                + [gt_gallery_index]
            )
        text_feat = (
            full_output.text_feats[local_index]
            if full_output.text_feats is not None
            else None
        )
        scores = score_encoder_heat_query(
            query_feat=full_output.query_feats[local_index],
            gallery_feats=gallery_feats,
            candidate_indices=candidate_indices,
            device=device,
            text_feat=text_feat,
            text_score_weight=text_score_weight,
            text_rerank_topk=text_rerank_topk,
        )
        top1_local = int(torch.argmax(scores).item())
        top1_global = int(candidate_indices[top1_local])
        positive_labels = set(include_map.get(_normalize_label(gt_label), set()))
        positive_labels.add(_normalize_label(gt_label))
        top1_correct = normalized_gallery[top1_global] in positive_labels
        records.append(
            {
                "top1_correct": bool(top1_correct),
                "retrieval_score": float(scores[top1_local].item()),
                "pred_satellite_path": str(gallery_paths[top1_global]),
                "pred_label": normalized_gallery[top1_global],
                "gt_label": _normalize_label(gt_label),
            }
        )
    return records


def enough_auto_candidates(candidates: Sequence[AutoCandidate]) -> bool:
    correct_satellites = {
        candidate.satellite_path
        for candidate in candidates
        if candidate.top1_correct
    }
    incorrect_satellites = {
        candidate.satellite_path
        for candidate in candidates
        if not candidate.top1_correct
    }
    all_satellites = {
        candidate.satellite_path
        for candidate in candidates
    }
    return (
        len(correct_satellites) >= 4
        and len(incorrect_satellites) >= 2
        and len(all_satellites) >= 8
    )


def scan_strict_candidates(
    full_model: torch.nn.Module,
    ablation_model: torch.nn.Module,
    full_spec: RunSpec,
    ablation_spec: RunSpec,
    dataset: Any,
    gallery_labels: Sequence[str],
    gallery_paths: Sequence[str],
    gallery_feats: torch.Tensor,
    include_map: Mapping[str, Sequence[str]],
    device: torch.device,
    sat_size: Tuple[int, int],
    candidate_size: int,
    seed: int,
    batch_size: int,
    num_workers: int,
    max_scan_samples: int,
    min_oracle_iou: float,
    min_heading_gain: float,
) -> List[AutoCandidate]:
    scan_count = min(int(max_scan_samples), len(dataset))
    loader = DataLoader(
        Subset(dataset, list(range(scan_count))),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,
    )
    candidates: List[AutoCandidate] = []
    processed = 0
    progress = tqdm(loader, desc="Strict case search")
    for batch in progress:
        current_size = int(batch["bbox"].shape[0])
        indices = list(range(processed, processed + current_size))
        processed += current_size
        full_output = run_grounding(
            full_model, full_spec, batch, device, sat_size
        )
        ablation_output = run_grounding(
            ablation_model, ablation_spec, batch, device, sat_size
        )
        retrieval = score_retrieval(
            full_output=full_output,
            dataset=dataset,
            case_indices=indices,
            gallery_labels=gallery_labels,
            gallery_paths=gallery_paths,
            gallery_feats=gallery_feats,
            include_map=include_map,
            device=device,
            candidate_size=candidate_size,
            seed=seed,
            text_score_weight=full_spec.text_score_weight,
            text_rerank_topk=full_spec.text_rerank_topk,
        )
        for local_index, dataset_index in enumerate(indices):
            oracle_iou = float(full_output.ious[local_index])
            heading_gain = oracle_iou - float(
                ablation_output.ious[local_index]
            )
            if not (
                oracle_iou > float(min_oracle_iou)
                and heading_gain >= float(min_heading_gain)
            ):
                continue
            sample = dataset.samples[dataset_index]
            candidates.append(
                AutoCandidate(
                    dataset_index=dataset_index,
                    satellite_path=str(sample["satellite_path"]),
                    top1_correct=bool(
                        retrieval[local_index]["top1_correct"]
                    ),
                    oracle_iou=oracle_iou,
                    heading_gain=heading_gain,
                    altitude=int(sample["height"]),
                    heading=int(sample["angle"]),
                )
            )
        unique_satellites = len(
            {candidate.satellite_path for candidate in candidates}
        )
        progress.set_postfix(
            {
                "qualified": len(candidates),
                "unique_sat": unique_satellites,
            }
        )
        if enough_auto_candidates(candidates):
            break
    print(
        f"[auto-select] scanned {processed} queries; found {len(candidates)} "
        "qualified queries across "
        f"{len({candidate.satellite_path for candidate in candidates})} "
        "unique satellites"
    )
    return candidates


def choose_strict_four(
    candidates: Sequence[AutoCandidate],
) -> List[AutoCandidate]:
    if not candidates:
        raise RuntimeError("Strict case search found no qualified samples")
    chosen: Dict[int, AutoCandidate] = {}
    used_satellites: set[str] = set()
    used_indices: set[int] = set()

    def choose(
        row: int,
        pool: Sequence[AutoCandidate],
        key: Any,
        description: str,
    ) -> None:
        available = [
            candidate
            for candidate in pool
            if candidate.satellite_path not in used_satellites
            and candidate.dataset_index not in used_indices
        ]
        if not available:
            raise RuntimeError(
                f"Strict search cannot fill Row {row} ({description}) with "
                "a new satellite tile; increase --max-scan-samples"
            )
        candidate = min(available, key=key)
        chosen[row] = candidate
        used_satellites.add(candidate.satellite_path)
        used_indices.add(candidate.dataset_index)

    # The latest >0.60 rule replaces the old Row-2 grounding-failure rule.
    choose(
        3,
        [candidate for candidate in candidates if not candidate.top1_correct],
        key=lambda candidate: (
            -candidate.oracle_iou,
            -candidate.heading_gain,
            candidate.dataset_index,
        ),
        description="retrieval incorrect with strong oracle grounding",
    )
    choose(
        2,
        [candidate for candidate in candidates if candidate.top1_correct],
        key=lambda candidate: (
            candidate.oracle_iou,
            -candidate.heading_gain,
            candidate.dataset_index,
        ),
        description="hardest correct retrieval above the IoU floor",
    )
    choose(
        4,
        candidates,
        key=lambda candidate: (
            -candidate.heading_gain,
            -candidate.altitude,
            -min(candidate.heading % 90, 90 - (candidate.heading % 90)),
            candidate.dataset_index,
        ),
        description="largest heading-conditioning gain",
    )
    choose(
        1,
        [candidate for candidate in candidates if candidate.top1_correct],
        key=lambda candidate: (
            -candidate.oracle_iou,
            -candidate.heading_gain,
            candidate.dataset_index,
        ),
        description="strong end-to-end success",
    )
    ordered = [chosen[row] for row in range(1, 5)]
    print("[auto-select] selected strict four:")
    for row, candidate in enumerate(ordered, start=1):
        print(
            f"  Row {row}: {dataset_case_name(candidate)} | "
            f"top1={'correct' if candidate.top1_correct else 'incorrect'}, "
            f"IoU={candidate.oracle_iou:.3f}, "
            f"delta_IoU={candidate.heading_gain:+.3f}"
        )
    return ordered


def dataset_case_name(candidate: AutoCandidate) -> str:
    satellite_id = Path(candidate.satellite_path).stem
    return (
        f"{satellite_id}:{candidate.altitude}:{candidate.heading}"
    )


def release_model(model: torch.nn.Module) -> None:
    model.to("cpu")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def safe_stem(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
    return value.strip("._") or "case"


def default_sidecar(output: Path, suffix: str) -> Path:
    return output.with_name(output.stem + suffix)


def write_records(
    path: Path,
    asset_dir: Path,
    full_spec: RunSpec,
    ablation_spec: RunSpec,
    dataset: Any,
    case_indices: Sequence[int],
    oracle_images: Sequence[Image.Image],
    full_output: GroundingOutput,
    ablation_output: GroundingOutput,
    retrieval_records: Sequence[Mapping[str, Any]],
    sat_size: Tuple[int, int],
    candidate_size: int,
    seed: int,
) -> None:
    path = path.expanduser().resolve()
    asset_dir = asset_dir.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    asset_dir.mkdir(parents=True, exist_ok=True)
    samples: List[Dict[str, Any]] = []
    for row, dataset_index in enumerate(case_indices, start=1):
        dataset_sample = dataset.samples[dataset_index]
        case_name = safe_stem(
            f"{Path(str(dataset_sample['drone_path'])).parent.name}_"
            f"{Path(str(dataset_sample['drone_path'])).stem}"
        )
        oracle_path = asset_dir / f"{row}_{case_name}_oracle.png"
        oracle_images[row - 1].save(oracle_path)
        heatmap_path: Optional[Path] = None
        heatmap = full_output.heatmaps[row - 1]
        if heatmap is not None:
            heatmap_path = asset_dir / f"{row}_{case_name}_heatmap.npy"
            np.save(heatmap_path, heatmap)
        retrieval = retrieval_records[row - 1]
        samples.append(
            {
                "id": case_name,
                "query": {
                    "path": str(Path(str(dataset_sample["drone_path"])).resolve()),
                    "altitude": int(dataset_sample["height"]),
                    "heading": int(dataset_sample["angle"]),
                },
                "retrieval": {
                    "path": retrieval["pred_satellite_path"],
                    "correct": bool(retrieval["top1_correct"]),
                    "score": float(retrieval["retrieval_score"]),
                    "pred_label": retrieval["pred_label"],
                    "gt_label": retrieval["gt_label"],
                },
                "oracle": {
                    "path": str(oracle_path),
                    "source_path": str(
                        Path(str(dataset_sample["satellite_path"])).resolve()
                    ),
                    "gt_bbox": full_output.gt_bboxes[row - 1],
                    "pred_bbox": full_output.pred_bboxes[row - 1],
                    "iou": full_output.ious[row - 1],
                    "cde": full_output.cdes[row - 1],
                },
                "diagnostic": {
                    "wo_heading_bbox": ablation_output.pred_bboxes[row - 1],
                    "wo_heading_iou": ablation_output.ious[row - 1],
                    "heatmap_path": str(heatmap_path) if heatmap_path else None,
                },
            }
        )
    payload = {
        "metadata": {
            "generated_by": str(Path(__file__).resolve()),
            "full_config": str(full_spec.config_path),
            "full_checkpoint": str(full_spec.checkpoint_path),
            "ablation_config": str(ablation_spec.config_path),
            "ablation_checkpoint": str(ablation_spec.checkpoint_path),
            "sat_size": {"height": sat_size[0], "width": sat_size[1]},
            "candidate_size": candidate_size,
            "seed": seed,
        },
        "defaults": {
            "bbox_format": "xyxy",
            "normalized": False,
            "cde_unit": "px",
        },
        "samples": samples,
    }
    with path.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(payload, stream, sort_keys=False, allow_unicode=True)
    print(f"[output] saved actual per-case records: {path}")


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            f"CUDA device {args.device!r} was requested but CUDA is unavailable"
        )
    full_spec = load_run_spec(
        "full",
        args.full_config,
        args.full_checkpoint,
        args.checkpoint_name,
    )
    ablation_spec = load_run_spec(
        "w/o-heading",
        args.ablation_config,
        args.ablation_checkpoint,
        args.checkpoint_name,
    )
    if full_spec.model_name != ablation_spec.model_name:
        raise ValueError(
            "The two configs use different MODEL_NAME values; their input "
            "preprocessing is not directly comparable"
        )
    if not full_spec.use_angle:
        print(
            "[warning] full config has USE_ANGLE_INPUT=false; it cannot demonstrate "
            "heading conditioning",
            file=sys.stderr,
        )
    if ablation_spec.use_angle:
        print(
            "[warning] ablation config has USE_ANGLE_INPUT=true; it is not a "
            "w/o-heading comparison",
            file=sys.stderr,
        )

    sat_size = (int(args.sat_size[0]), int(args.sat_size[1]))
    dataset = create_dataset(
        model_name=full_spec.model_name,
        sat_size=sat_size,
        test_crop_ratio=args.test_crop_ratio,
    )
    full_model = create_model(full_spec, device)
    gallery_labels, gallery_paths, gallery_feats = extract_gallery(
        full_model,
        dataset,
        device,
        batch_size=args.gallery_batch_size,
        num_workers=args.num_workers,
    )
    include_map = _load_include_map(args.include_file)

    if args.auto_select:
        ablation_model = create_model(ablation_spec, device)
        candidates = scan_strict_candidates(
            full_model=full_model,
            ablation_model=ablation_model,
            full_spec=full_spec,
            ablation_spec=ablation_spec,
            dataset=dataset,
            gallery_labels=gallery_labels,
            gallery_paths=gallery_paths,
            gallery_feats=gallery_feats,
            include_map=include_map,
            device=device,
            sat_size=sat_size,
            candidate_size=args.candidate_size,
            seed=args.seed,
            batch_size=args.scan_batch_size,
            num_workers=args.num_workers,
            max_scan_samples=args.max_scan_samples,
            min_oracle_iou=args.min_oracle_iou,
            min_heading_gain=args.min_heading_gain,
        )
        strict_four = choose_strict_four(candidates)
        case_indices = [
            candidate.dataset_index for candidate in strict_four
        ]
        case_batch, oracle_images = prepare_case_batch(
            dataset, case_indices
        )
        full_output = run_grounding(
            full_model, full_spec, case_batch, device, sat_size
        )
        ablation_output = run_grounding(
            ablation_model, ablation_spec, case_batch, device, sat_size
        )
    else:
        if args.cases is None:
            raise ValueError("--cases is required unless --auto-select is used")
        case_indices = resolve_case_indices(dataset, args.cases)
        case_batch, oracle_images = prepare_case_batch(
            dataset, case_indices
        )
        full_output = run_grounding(
            full_model, full_spec, case_batch, device, sat_size
        )
        ablation_model = create_model(ablation_spec, device)
        ablation_output = run_grounding(
            ablation_model, ablation_spec, case_batch, device, sat_size
        )

    retrieval_records = score_retrieval(
        full_output=full_output,
        dataset=dataset,
        case_indices=case_indices,
        gallery_labels=gallery_labels,
        gallery_paths=gallery_paths,
        gallery_feats=gallery_feats,
        include_map=include_map,
        device=device,
        candidate_size=args.candidate_size,
        seed=args.seed,
        text_score_weight=full_spec.text_score_weight,
        text_rerank_topk=full_spec.text_rerank_topk,
    )
    release_model(full_model)
    del full_model
    release_model(ablation_model)
    del ablation_model
    del gallery_feats

    output = args.output.expanduser().resolve()
    records_output = (
        args.records_output.expanduser().resolve()
        if args.records_output is not None
        else default_sidecar(output, "_cases.yaml")
    )
    asset_dir = (
        args.asset_dir.expanduser().resolve()
        if args.asset_dir is not None
        else output.parent / (output.stem + "_assets")
    )
    selected_yaml = (
        args.selected_yaml.expanduser().resolve()
        if args.selected_yaml is not None
        else default_sidecar(output, "_selection.yaml")
    )
    write_records(
        path=records_output,
        asset_dir=asset_dir,
        full_spec=full_spec,
        ablation_spec=ablation_spec,
        dataset=dataset,
        case_indices=case_indices,
        oracle_images=oracle_images,
        full_output=full_output,
        ablation_output=ablation_output,
        retrieval_records=retrieval_records,
        sat_size=sat_size,
        candidate_size=args.candidate_size,
        seed=args.seed,
    )

    samples = load_samples(records_output)
    if args.auto_select:
        source_satellites = {
            str(dataset.samples[index]["satellite_path"])
            for index in case_indices
        }
        if len(source_satellites) != 4:
            raise RuntimeError(
                "Auto-selected records do not contain four distinct oracle tiles"
            )
        selections = []
        for row, sample in enumerate(samples, start=1):
            oracle_iou, _, heading_gain = sample.metrics()
            if not (
                oracle_iou > args.min_oracle_iou
                and heading_gain is not None
                and heading_gain >= args.min_heading_gain
            ):
                raise RuntimeError(
                    f"Row {row} failed post-inference constraints: "
                    f"IoU={oracle_iou:.3f}, delta_IoU={heading_gain}"
                )
            explanation = (
                f"strict auto-selection; IoU={oracle_iou:.3f}, "
                f"delta_IoU={heading_gain:+.3f}, unique satellite"
            )
            selections.append(Selection(row, sample, True, explanation))
        print("[selection] all four rows pass the strict final checks:")
        for selection in selections:
            sample = selection.sample
            oracle_iou, cde, heading_gain = sample.metrics()
            print(
                f"  Row {selection.row}: {sample.sample_id}; "
                f"top1={'correct' if sample.top1_correct else 'incorrect'}, "
                f"IoU={oracle_iou:.3f}, CDE={cde:.2f}, "
                f"delta_IoU={heading_gain:+.3f}"
            )
    else:
        selections = select_samples(
            samples,
            success_iou=args.success_iou,
            failure_iou=args.failure_iou,
            heading_gain_threshold=args.heading_gain,
        )
        report_selection(
            selections,
            success_iou=args.success_iou,
            failure_iou=args.failure_iou,
            heading_gain=args.heading_gain,
        )
    make_figure(
        selections,
        output=output,
        png_output=args.png_output,
        width=args.width,
        height=args.height,
        dpi=args.dpi,
        heatmap_cmap=args.heatmap_cmap,
        heatmap_alpha=args.heatmap_alpha,
    )
    save_selection(selected_yaml, selections)


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, RuntimeError, ValueError, yaml.YAMLError) as exc:
        print(f"[error] {exc}", file=sys.stderr)
        raise SystemExit(2) from None
