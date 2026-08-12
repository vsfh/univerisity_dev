#!/usr/bin/env python3
"""Evaluate a grounding checkpoint on another run's retrieval successes.

The legacy ``*_top1_success_drone_names.txt`` files only contain basenames such
as ``250_90.png``.  Those names are repeated under every satellite directory,
so they are not sufficient to identify samples by themselves.  This script
therefore uses, in order:

1. a validated full-path manifest cached by an earlier invocation;
2. full ``query_records`` from the retrieval result JSON, when available;
3. the existing ``test_unify.py`` retrieval implementation to reconstruct the
   exact successful sample paths.

No dataset implementation is modified. Grounding is evaluated on every query
through the existing loader factory. The reconstructed retrieval-success paths
act only as a mask: failed retrieval queries receive uIoU=0 while their normal
IoU still contributes to all-query mIoU, matching ``test_unify.py`` semantics.
"""

import argparse
import gc
import hashlib
import json
import os
import random
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from bbox.yolo_utils import bbox_iou, build_target, eval_iou_acc
from model import Encoder_heat, Encoder_test
from test_unify import (
    DEFAULT_INCLUDE_FILE,
    FeatureBundle,
    MODEL_NAME,
    _load_include_map,
    _normalize_label,
    add_heatmap_to_confidence,
    build_geo_features,
    center_distance,
    create_encoder_loader,
    extract_encoder_heat_features,
    load_checkpoint,
    load_yaml,
    parse_anchors,
    resolve_encoder_heat_checkpoint,
)


# --- Configuration ---
DEFAULT_SUCCESS_FILE = (
    "eval_results/test_unify/"
    "test_unify_encoder_test_baseline_sat_top1_success_drone_names.txt"
)
DEFAULT_RETRIEVAL_CONFIG = (
    "configs/unified_siglip_supp/single_config/baseline_sat.yaml"
)
DEFAULT_GROUNDING_CONFIG = (
    "configs/unified_siglip_supp/single_config/baseline_grounding_full.yaml"
)
DEFAULT_OUTPUT = (
    "eval_results/test_unify/"
    "baseline_grounding_full_on_baseline_sat_top1_success.json"
)
DEFAULT_SAT_SIZE = (432, 768)  # height, width
DEFAULT_HEIGHTS = (150, 200, 250, 300)
DEFAULT_ANGLES = (0, 45, 90, 135, 180, 225, 270, 315)


def _as_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_success_entries(path: str) -> List[str]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Success file not found: {path}")
    with open(path, "r", encoding="utf-8") as file:
        entries = [line.strip() for line in file if line.strip()]
    if not entries:
        raise ValueError(f"Success file is empty: {path}")
    return entries


def _infer_retrieval_json(success_file: str) -> str:
    suffix = "_top1_success_drone_names.txt"
    if success_file.endswith(suffix):
        return success_file[: -len(suffix)] + ".json"
    return str(Path(success_file).with_suffix(".json"))


def _infer_manifest_path(success_file: str) -> str:
    suffix = "_top1_success_drone_names.txt"
    if success_file.endswith(suffix):
        return success_file[: -len(suffix)] + "_top1_success_manifest.json"
    path = Path(success_file)
    return str(path.with_name(path.stem + "_manifest.json"))


def _experiment_config(config_path: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    payload = load_yaml(config_path)
    config = payload.get("config", {}) or {}
    if not isinstance(config, dict):
        raise ValueError(f"'config' must be a mapping: {config_path}")
    return payload, config


def _resolve_checkpoint(
    config_path: str,
    explicit_checkpoint: Optional[str],
) -> str:
    if explicit_checkpoint:
        checkpoint = explicit_checkpoint
    else:
        payload, _ = _experiment_config(config_path)
        checkpoint = resolve_encoder_heat_checkpoint(
            config_path=config_path,
            payload=payload,
            checkpoint_name="last.pth",
        )
    if not os.path.exists(checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    return os.path.abspath(checkpoint)


def _validate_success_names(expected: Sequence[str], paths: Sequence[str]) -> None:
    actual = [Path(path).name for path in paths]
    if actual == list(expected):
        return

    expected_counts = Counter(expected)
    actual_counts = Counter(actual)
    missing = expected_counts - actual_counts
    extra = actual_counts - expected_counts
    details = []
    if missing:
        details.append(f"missing={dict(missing.most_common(5))}")
    if extra:
        details.append(f"extra={dict(extra.most_common(5))}")
    if not details:
        details.append("the same basenames were found in a different order")
    raise RuntimeError(
        "Reconstructed successes do not match the success txt: " + "; ".join(details)
    )


def _load_paths_from_result_json(
    result_json: str,
    expected_names: Sequence[str],
) -> Optional[Tuple[List[str], int]]:
    if not result_json or not os.path.exists(result_json):
        return None
    with open(result_json, "r", encoding="utf-8") as file:
        payload = json.load(file)
    records = payload.get("query_records")
    if not isinstance(records, list) or not records:
        return None
    paths = [
        str(record["drone_path"])
        for record in records
        if bool(record.get("top1_correct")) and record.get("drone_path")
    ]
    _validate_success_names(expected_names, paths)
    total = int(payload.get("num_samples") or len(records))
    return paths, total


def _load_paths_from_manifest(
    manifest_path: str,
    success_sha256: str,
    retrieval_checkpoint: str,
    candidate_size: int,
    seed: int,
    expected_names: Sequence[str],
) -> Optional[Tuple[List[str], int]]:
    if not manifest_path or not os.path.exists(manifest_path):
        return None
    with open(manifest_path, "r", encoding="utf-8") as file:
        payload = json.load(file)
    expected_metadata = {
        "success_file_sha256": success_sha256,
        "retrieval_checkpoint": os.path.abspath(retrieval_checkpoint),
        "candidate_size": int(candidate_size),
        "seed": int(seed),
    }
    if any(payload.get(key) != value for key, value in expected_metadata.items()):
        return None
    paths = payload.get("success_drone_paths")
    if not isinstance(paths, list) or not paths:
        return None
    paths = [str(path) for path in paths]
    _validate_success_names(expected_names, paths)
    total = int(payload.get("num_total_samples", 0))
    if total <= 0:
        return None
    return paths, total


def _save_manifest(
    manifest_path: str,
    success_file: str,
    success_sha256: str,
    retrieval_config: str,
    retrieval_checkpoint: str,
    include_file: Optional[str],
    candidate_size: int,
    seed: int,
    test_crop_ratio: float,
    success_paths: Sequence[str],
    num_total_samples: int,
) -> None:
    payload = {
        "format_version": 1,
        "source_success_file": os.path.abspath(success_file),
        "success_file_sha256": success_sha256,
        "retrieval_config": os.path.abspath(retrieval_config),
        "retrieval_checkpoint": os.path.abspath(retrieval_checkpoint),
        "include_file": os.path.abspath(include_file) if include_file else None,
        "candidate_size": int(candidate_size),
        "seed": int(seed),
        "test_crop_ratio": float(test_crop_ratio),
        "num_total_samples": int(num_total_samples),
        "num_success_samples": int(len(success_paths)),
        "success_drone_paths": list(success_paths),
    }
    Path(manifest_path).parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)


def _score_top1_success_paths_fast(
    bundle: FeatureBundle,
    include_map: Dict[str, set],
    device: torch.device,
    candidate_size: int,
    seed: int,
    score_batch_size: int,
) -> List[str]:
    """Batch the image-only score used by ``score_retrieval_and_uiou``.

    Candidate construction and positive-label handling intentionally mirror the
    existing scorer.  Only top-1 is computed because that is all the manifest
    needs; batching avoids one small GPU launch and transfer per query.
    """
    label_to_gallery_index = {
        label: index for index, label in enumerate(bundle.gallery_labels)
    }
    normalized_gallery_labels = [
        _normalize_label(label) for label in bundle.gallery_labels
    ]
    num_gallery = len(bundle.gallery_labels)
    effective_candidate_size = min(int(candidate_size), num_gallery)
    all_indices = list(range(num_gallery))
    negative_pools: Dict[int, List[int]] = {}
    gallery_features = bundle.gallery_feats.to(device)
    success_paths: List[str] = []

    for start in tqdm(
        range(0, len(bundle.query_labels), score_batch_size),
        desc="Score retrieval candidates (batched)",
    ):
        end = min(start + score_batch_size, len(bundle.query_labels))
        valid_query_indices: List[int] = []
        candidate_rows: List[List[int]] = []
        positive_sets: List[set] = []

        for query_index in range(start, end):
            gt_label = bundle.query_labels[query_index]
            gt_gallery_index = label_to_gallery_index.get(gt_label)
            if gt_gallery_index is None:
                continue
            gt_normalized = _normalize_label(gt_label)
            positives = set(include_map.get(gt_normalized, set()))
            positives.add(gt_normalized)

            if effective_candidate_size >= num_gallery:
                candidates = all_indices
            else:
                negative_pool = negative_pools.get(gt_gallery_index)
                if negative_pool is None:
                    negative_pool = [
                        index for index in all_indices if index != gt_gallery_index
                    ]
                    negative_pools[gt_gallery_index] = negative_pool
                rng = random.Random(int(seed) + int(query_index))
                candidates = rng.sample(
                    negative_pool, effective_candidate_size - 1
                ) + [gt_gallery_index]

            valid_query_indices.append(query_index)
            candidate_rows.append(candidates)
            positive_sets.append(positives)

        if not valid_query_indices:
            continue
        candidate_tensor = torch.tensor(
            candidate_rows, dtype=torch.long, device=device
        )
        query_features = bundle.query_feats[valid_query_indices].to(device)
        candidate_features = gallery_features[candidate_tensor]
        scores = torch.einsum(
            "bd,bknd->bkn", query_features, candidate_features
        ).amax(dim=-1)
        top1_local = scores.argmax(dim=1)
        top1_global = candidate_tensor[
            torch.arange(len(valid_query_indices), device=device), top1_local
        ].cpu().tolist()

        for row, query_index in enumerate(valid_query_indices):
            predicted_label = normalized_gallery_labels[int(top1_global[row])]
            if predicted_label in positive_sets[row]:
                success_paths.append(bundle.query_drone_paths[query_index])

    return success_paths


def _reconstruct_success_paths(
    args: argparse.Namespace,
    expected_names: Sequence[str],
    retrieval_checkpoint: str,
    manifest_path: str,
) -> Tuple[List[str], int, str]:
    success_sha256 = _sha256(args.success_file)

    if not args.force_reconstruct:
        cached = _load_paths_from_manifest(
            manifest_path=manifest_path,
            success_sha256=success_sha256,
            retrieval_checkpoint=retrieval_checkpoint,
            candidate_size=args.candidate_size,
            seed=args.seed,
            expected_names=expected_names,
        )
        if cached is not None:
            return cached[0], cached[1], "manifest"

        result_records = _load_paths_from_result_json(
            result_json=args.retrieval_result_json,
            expected_names=expected_names,
        )
        if result_records is not None:
            paths, total = result_records
            _save_manifest(
                manifest_path,
                args.success_file,
                success_sha256,
                args.retrieval_config,
                retrieval_checkpoint,
                args.include_file,
                args.candidate_size,
                args.seed,
                args.test_crop_ratio,
                paths,
                total,
            )
            return paths, total, "retrieval_result_json"

    payload, config = _experiment_config(args.retrieval_config)
    encoder_type = str(config.get("ENCODER_TYPE", "test")).lower()
    encoder_classes = {"heat": Encoder_heat, "test": Encoder_test}
    if encoder_type not in encoder_classes:
        raise ValueError(
            f"Retrieval reconstruction supports ENCODER_TYPE heat/test, got {encoder_type!r}."
        )

    print(
        "The success txt has basename-only entries and no usable full-path records. "
        "Re-running retrieval once to reconstruct exact successes."
    )
    bundle = extract_encoder_heat_features(
        checkpoint_path=retrieval_checkpoint,
        device=torch.device(args.device),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        sat_size=tuple(args.sat_size),
        test_crop_ratio=args.test_crop_ratio,
        subset_heights=args.subset_heights,
        subset_angles=args.subset_angles,
        heatmap_confidence_weight=float(config.get("HEATMAP_CONFIDENCE_WEIGHT", 0.5)),
        use_text=False,
        use_angle=_as_bool(config.get("USE_ANGLE_INPUT"), True),
        use_ap=_as_bool(payload.get("use_ap"), True),
        use_heatmap=_as_bool(config.get("USE_HEATMAP_LOSS"), True),
        encoder_cls=encoder_classes[encoder_type],
        desc="reconstruct_retrieval_successes",
        lora_rank=int(config.get("LORA_RANK", 8)),
        lora_alpha=float(config.get("LORA_ALPHA", 16.0)),
        lora_dropout=float(config.get("LORA_DROPOUT", 0.05)),
    )
    success_paths = _score_top1_success_paths_fast(
        bundle=bundle,
        include_map=_load_include_map(args.include_file),
        device=torch.device(args.device),
        candidate_size=args.candidate_size,
        seed=args.seed,
        score_batch_size=args.retrieval_score_batch_size,
    )
    _validate_success_names(expected_names, success_paths)
    num_total_samples = int(len(bundle.query_labels))
    _save_manifest(
        manifest_path,
        args.success_file,
        success_sha256,
        args.retrieval_config,
        retrieval_checkpoint,
        args.include_file,
        args.candidate_size,
        args.seed,
        args.test_crop_ratio,
        success_paths,
        num_total_samples,
    )
    del bundle
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return success_paths, num_total_samples, "reconstructed_retrieval"


def _full_loader(args: argparse.Namespace) -> DataLoader:
    return create_encoder_loader(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        sat_size=tuple(args.sat_size),
        test_crop_ratio=args.test_crop_ratio,
        subset_heights=args.subset_heights,
        subset_angles=args.subset_angles,
        model_name=MODEL_NAME,
    )


def _build_grounding_model(
    config: Dict[str, Any],
    use_ap: bool,
    device: torch.device,
) -> torch.nn.Module:
    encoder_type = str(config.get("ENCODER_TYPE", "test")).lower()
    encoder_classes = {"heat": Encoder_heat, "test": Encoder_test}
    if encoder_type not in encoder_classes:
        raise ValueError(
            f"Grounding evaluation supports ENCODER_TYPE heat/test, got {encoder_type!r}."
        )
    kwargs = {
        "model_name": str(config.get("MODEL_NAME", MODEL_NAME)),
        "proj_dim": int(config.get("PROJECTION_DIM", 768)),
        "usesg": True,
        "useap": bool(use_ap),
        "use_heatmap": _as_bool(config.get("USE_HEATMAP_LOSS"), True),
        "lora_rank": int(config.get("LORA_RANK", 8)),
        "lora_alpha": float(config.get("LORA_ALPHA", 16.0)),
        "lora_dropout": float(config.get("LORA_DROPOUT", 0.05)),
    }
    if encoder_type == "test":
        kwargs["use_text_grounding_path"] = _as_bool(
            config.get("USE_TEXT_GROUNDING_PATH"), False
        )
    return encoder_classes[encoder_type](**kwargs).to(device)


def _evaluate_grounding(
    args: argparse.Namespace,
    grounding_checkpoint: str,
    success_paths: Sequence[str],
    num_total_samples: int,
) -> List[Dict[str, Any]]:
    payload, config = _experiment_config(args.grounding_config)
    use_ap = _as_bool(payload.get("use_ap"), True)
    use_angle = _as_bool(config.get("USE_ANGLE_INPUT"), True)
    use_text = _as_bool(config.get("USE_TEXT_INPUT"), False)
    use_heatmap = _as_bool(config.get("USE_HEATMAP_LOSS"), True)
    confidence_weight = float(config.get("HEATMAP_CONFIDENCE_WEIGHT", 0.5))

    device = torch.device(args.device)
    model = _build_grounding_model(config, use_ap=use_ap, device=device)
    load_checkpoint(model, grounding_checkpoint)
    model.eval()
    loader = _full_loader(args)
    if len(loader.dataset) != num_total_samples:
        raise RuntimeError(
            f"Grounding test dataset has {len(loader.dataset)} samples, but retrieval "
            f"was evaluated on {num_total_samples}. Check subset/crop arguments."
        )
    success_keys = {
        os.path.abspath(os.path.normpath(str(path))) for path in success_paths
    }
    if len(success_keys) != len(success_paths):
        raise RuntimeError("The reconstructed success manifest contains duplicate paths.")
    seen_success_keys = set()
    anchors = parse_anchors(device)
    image_wh = (int(args.sat_size[1]), int(args.sat_size[0]))
    records: List[Dict[str, Any]] = []

    with torch.inference_mode():
        for batch in tqdm(loader, desc="Grounding on all retrieval queries"):
            query_images = batch["target_pixel_values"].to(device, non_blocking=True)
            search_images = batch["search_pixel_values"].to(device, non_blocking=True)
            gt_bbox = batch["bbox"].to(device, non_blocking=True)
            input_ids = (
                batch["input_ids"].to(device, non_blocking=True) if use_text else None
            )
            attention_mask = (
                batch["attention_mask"].to(device, non_blocking=True)
                if use_text and "attention_mask" in batch
                else None
            )
            geo = build_geo_features(batch, device) if use_angle else None
            outputs = model(
                query_images,
                search_images,
                input_ids=input_ids,
                angle=geo,
                attention_mask=attention_mask,
            )
            if len(outputs) != 7:
                raise ValueError(f"Expected 7 model outputs, got {len(outputs)}.")
            pred_anchor, _, _, _, _, refine_outputs, heatmap_logits = outputs

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
            if isinstance(refine_outputs, dict) and "bbox" in refine_outputs:
                pred_bbox = refine_outputs["bbox"].to(
                    device=gt_bbox.device, dtype=gt_bbox.dtype
                )
                target_bbox = gt_bbox

            ious = bbox_iou(pred_bbox, target_bbox, x1y1x2y2=True)
            if not torch.isfinite(ious).all():
                raise FloatingPointError("Grounding checkpoint produced a non-finite IoU.")
            for index in range(batch_size):
                iou = float(ious[index].item())
                drone_path = str(batch["drone_path"][index])
                path_key = os.path.abspath(os.path.normpath(drone_path))
                top1_correct = path_key in success_keys
                if top1_correct:
                    seen_success_keys.add(path_key)
                records.append(
                    {
                        "height": int(batch["height"][index].item()),
                        "angle": int(batch["angle"][index].item()),
                        "top1_correct": bool(top1_correct),
                        "iou": iou,
                        "uIoU": iou if top1_correct else 0.0,
                        "center_distance": center_distance(
                            pred_bbox[index].float(), target_bbox[index].float()
                        ),
                        "drone_path": drone_path,
                        "satellite_path": str(batch["satellite_path"][index]),
                    }
                )
    if len(records) != num_total_samples:
        raise RuntimeError(
            f"Evaluated {len(records)} grounding samples, expected {num_total_samples}."
        )
    missing_successes = success_keys - seen_success_keys
    if missing_successes:
        first_missing = next(iter(missing_successes))
        raise RuntimeError(
            f"{len(missing_successes)} retrieval-success paths were not present during "
            f"grounding evaluation; first missing path: {first_missing}"
        )
    return records


def _load_retrieval_result(
    result_json: str,
    num_total_samples: int,
    num_successes: int,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    if not result_json or not os.path.exists(result_json):
        raise FileNotFoundError(
            "The baseline retrieval result JSON is required to preserve its exact "
            f"R@1/R@5/R@10 metrics: {result_json}"
        )
    with open(result_json, "r", encoding="utf-8") as file:
        payload = json.load(file)
    checks = {
        "candidate_size": int(args.candidate_size),
        "seed": int(args.seed),
        "test_crop_ratio": float(args.test_crop_ratio),
    }
    for key, expected in checks.items():
        actual = payload.get(key)
        if actual is not None and actual != expected:
            raise RuntimeError(
                f"Retrieval result {key}={actual!r}, but evaluation requested {expected!r}."
            )
    retrieval = payload.get("retrieval")
    if not isinstance(retrieval, dict):
        raise ValueError(f"Missing retrieval metrics in: {result_json}")
    if int(retrieval.get("num_queries", -1)) != num_total_samples:
        raise RuntimeError("Retrieval result query count does not match the success manifest.")
    if int(retrieval.get("top1_hits", -1)) != num_successes:
        raise RuntimeError("Retrieval result top1_hits does not match the success manifest.")
    return payload


def _conditional_grounding_metrics(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not records:
        return {
            "num_samples": 0,
            "mean_iou": 0.0,
            "ratio_iou_gt_0_25": 0.0,
            "ratio_iou_gt_0_5": 0.0,
            "mean_center_distance": 0.0,
        }
    ious = np.asarray([float(record["iou"]) for record in records], dtype=np.float64)
    distances = np.asarray(
        [float(record["center_distance"]) for record in records], dtype=np.float64
    )
    return {
        "num_samples": int(len(records)),
        "mean_iou": float(ious.mean()),
        "ratio_iou_gt_0_25": float((ious > 0.25).mean()),
        "ratio_iou_gt_0_5": float((ious > 0.5).mean()),
        "mean_center_distance": float(distances.mean()),
    }


def _joint_metrics(
    records: Sequence[Dict[str, Any]],
    retrieval_summary: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if not records:
        raise ValueError("Cannot summarize an empty record list.")
    ious = np.asarray([float(record["iou"]) for record in records], dtype=np.float64)
    uious = np.asarray([float(record["uIoU"]) for record in records], dtype=np.float64)
    distances = np.asarray(
        [float(record["center_distance"]) for record in records], dtype=np.float64
    )
    num_samples = len(records)
    top1_hits = sum(bool(record["top1_correct"]) for record in records)
    metrics: Dict[str, Any] = {
        "num_samples": int(num_samples),
        "top1_hits": int(top1_hits),
        "recall@1": float(top1_hits / num_samples),
        "mean_iou": float(ious.mean()),
        "ratio_iou_gt_0_5": float((ious > 0.5).mean()),
        "ratio_iou_gt_0_25": float((ious > 0.25).mean()),
        "uIoU": float(uious.mean()),
        "ratio_uIoU_gt_0_25": float((uious > 0.25).mean()),
        "ratio_uIoU_gt_25": float((uious > 0.25).mean()),
        "mean_center_distance": float(distances.mean()),
    }
    if retrieval_summary is not None:
        expected_top1 = retrieval_summary.get("top1_hits")
        if expected_top1 is not None and int(expected_top1) != top1_hits:
            raise RuntimeError(
                f"Retrieval top1 mismatch: records={top1_hits}, summary={expected_top1}."
            )
        for key in ("top5_hits", "top10_hits", "recall@5", "recall@10"):
            if key in retrieval_summary:
                metrics[key] = retrieval_summary[key]
    return metrics


def _group_joint_metrics(
    records: Sequence[Dict[str, Any]],
    retrieval_payload: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    subset_groups: Dict[Tuple[int, int], List[Dict[str, Any]]] = {}
    height_groups: Dict[int, List[Dict[str, Any]]] = {}
    angle_groups: Dict[int, List[Dict[str, Any]]] = {}
    for record in records:
        height = int(record["height"])
        angle = int(record["angle"])
        subset_groups.setdefault((height, angle), []).append(record)
        height_groups.setdefault(height, []).append(record)
        angle_groups.setdefault(angle, []).append(record)

    subset_retrieval = {
        (int(item["height"]), int(item["angle"])): item
        for item in retrieval_payload.get("per_subset", [])
    }
    height_retrieval = {
        int(item["height"]): item for item in retrieval_payload.get("per_height", [])
    }
    angle_retrieval = {
        int(item["angle"]): item for item in retrieval_payload.get("per_angle", [])
    }
    per_subset = [
        {
            "height": height,
            "angle": angle,
            **_joint_metrics(items, subset_retrieval.get((height, angle))),
        }
        for (height, angle), items in sorted(subset_groups.items())
    ]
    per_height = [
        {"height": height, **_joint_metrics(items, height_retrieval.get(height))}
        for height, items in sorted(height_groups.items())
    ]
    per_angle = [
        {"angle": angle, **_joint_metrics(items, angle_retrieval.get(angle))}
        for angle, items in sorted(angle_groups.items())
    ]
    return per_subset, per_height, per_angle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate baseline_grounding_full on the exact top-1 successes from "
            "a baseline_sat retrieval run."
        )
    )
    parser.add_argument("--success-file", default=DEFAULT_SUCCESS_FILE)
    parser.add_argument("--retrieval-result-json", default=None)
    parser.add_argument("--success-manifest", default=None)
    parser.add_argument("--retrieval-config", default=DEFAULT_RETRIEVAL_CONFIG)
    parser.add_argument("--retrieval-checkpoint", default=None)
    parser.add_argument("--grounding-config", default=DEFAULT_GROUNDING_CONFIG)
    parser.add_argument("--grounding-checkpoint", default=None)
    parser.add_argument("--include-file", default=DEFAULT_INCLUDE_FILE)
    parser.add_argument("--candidate-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--test-crop-ratio", type=float, default=1.0)
    parser.add_argument("--sat-size", type=int, nargs=2, default=DEFAULT_SAT_SIZE)
    parser.add_argument("--subset-heights", type=int, nargs="+", default=list(DEFAULT_HEIGHTS))
    parser.add_argument("--subset-angles", type=int, nargs="+", default=list(DEFAULT_ANGLES))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--retrieval-score-batch-size",
        type=int,
        default=64,
        help="Batch size for the exact candidate-100 reconstruction scorer.",
    )
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument(
        "--device",
        default="cuda:0" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--force-reconstruct",
        action="store_true",
        help="Ignore a compatible cached full-path manifest and rerun retrieval.",
    )
    parser.add_argument(
        "--save-records",
        action="store_true",
        help="Store all per-sample grounding records in the output JSON.",
    )
    args = parser.parse_args()
    if args.candidate_size <= 0:
        raise ValueError("--candidate-size must be positive.")
    if args.retrieval_score_batch_size <= 0:
        raise ValueError("--retrieval-score-batch-size must be positive.")
    if not 0.0 <= args.test_crop_ratio <= 1.0:
        raise ValueError("--test-crop-ratio must be between 0 and 1.")
    if args.retrieval_result_json is None:
        args.retrieval_result_json = _infer_retrieval_json(args.success_file)
    if args.success_manifest is None:
        args.success_manifest = _infer_manifest_path(args.success_file)
    return args


def main() -> None:
    args = parse_args()
    success_names = _read_success_entries(args.success_file)
    retrieval_checkpoint = _resolve_checkpoint(
        args.retrieval_config, args.retrieval_checkpoint
    )
    grounding_checkpoint = _resolve_checkpoint(
        args.grounding_config, args.grounding_checkpoint
    )

    success_paths, num_total_samples, success_source = _reconstruct_success_paths(
        args=args,
        expected_names=success_names,
        retrieval_checkpoint=retrieval_checkpoint,
        manifest_path=args.success_manifest,
    )
    print(
        f"Resolved {len(success_paths)} exact successes from {success_source}; "
        f"total retrieval queries: {num_total_samples}."
    )
    records = _evaluate_grounding(
        args=args,
        grounding_checkpoint=grounding_checkpoint,
        success_paths=success_paths,
        num_total_samples=num_total_samples,
    )

    retrieval_payload = _load_retrieval_result(
        result_json=args.retrieval_result_json,
        num_total_samples=num_total_samples,
        num_successes=len(success_paths),
        args=args,
    )
    retrieval_metrics = dict(retrieval_payload["retrieval"])
    success_records = [record for record in records if record["top1_correct"]]
    failure_records = [record for record in records if not record["top1_correct"]]
    overall = _joint_metrics(records, retrieval_metrics)
    grounding_metrics = _conditional_grounding_metrics(records)
    success_grounding_metrics = _conditional_grounding_metrics(success_records)
    failure_grounding_metrics = _conditional_grounding_metrics(failure_records)
    per_subset, per_height, per_angle = _group_joint_metrics(
        records, retrieval_payload
    )
    output = {
        "success_file": os.path.abspath(args.success_file),
        "success_path_source": success_source,
        "success_manifest": os.path.abspath(args.success_manifest),
        "retrieval_config": os.path.abspath(args.retrieval_config),
        "retrieval_checkpoint": retrieval_checkpoint,
        "grounding_config": os.path.abspath(args.grounding_config),
        "grounding_checkpoint": grounding_checkpoint,
        "candidate_size": int(args.candidate_size),
        "test_crop_ratio": float(args.test_crop_ratio),
        "seed": int(args.seed),
        "sat_size": {"height": int(args.sat_size[0]), "width": int(args.sat_size[1])},
        "subset_heights": list(args.subset_heights),
        "subset_angles": list(args.subset_angles),
        "num_total_samples": int(num_total_samples),
        "num_retrieval_successes": int(len(success_records)),
        "metric_definition": {
            "mean_iou": "mean grounding IoU over all retrieval queries",
            "uIoU": (
                "mean over all retrieval queries of grounding IoU when top1 "
                "retrieval is correct, otherwise 0"
            ),
            "ratio_uIoU_gt_0_25": (
                "fraction of all retrieval queries with correct top1 retrieval "
                "and grounding IoU > 0.25"
            ),
        },
        "retrieval": retrieval_metrics,
        "overall": overall,
        "grounding_all_queries": grounding_metrics,
        "grounding_on_retrieval_successes": success_grounding_metrics,
        "grounding_on_retrieval_failures": failure_grounding_metrics,
        "per_subset": per_subset,
        "per_height": per_height,
        "per_angle": per_angle,
        "records": records if args.save_records else [],
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as file:
        json.dump(output, file, indent=2, ensure_ascii=False)

    print("\n=== Retrieval-success grounding evaluation ===")
    print(
        "Retrieval: "
        f"R@1={retrieval_metrics['recall@1']:.4f} "
        f"R@5={retrieval_metrics['recall@5']:.4f} "
        f"R@10={retrieval_metrics['recall@10']:.4f} "
        f"({len(success_records)}/{num_total_samples})"
    )
    print(
        "All-query grounding: "
        f"mIoU={overall['mean_iou']:.4f} "
        f"IoU>0.25={overall['ratio_iou_gt_0_25']:.4f} "
        f"IoU>0.5={overall['ratio_iou_gt_0_5']:.4f} "
        f"CDE={overall['mean_center_distance']:.4f}px"
    )
    print(
        "Joint metrics (retrieval failures are zero): "
        f"uIoU={overall['uIoU']:.4f} "
        f"ratio_uIoU_gt_0_25={overall['ratio_uIoU_gt_0_25']:.4f}"
    )
    print(
        "Grounding diagnostic on retrieval successes only: "
        f"conditional_mIoU={success_grounding_metrics['mean_iou']:.4f}"
    )
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
