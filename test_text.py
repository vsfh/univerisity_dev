import argparse
import json
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoProcessor

from bbox.yolo_utils import bbox_iou, build_target, eval_iou_acc, get_tensor_anchors
from text import (
    Config,
    TextQwenVLRetrievalGrounding,
    TextVLMSatelliteDataset,
    _dtype_from_name,
    add_heatmap_to_confidence,
    build_dataloader_kwargs,
    build_geo_features,
    text_vlm_collate_fn,
)


# --- Configuration ---
DEFAULT_CHECKPOINT = "/data/feihong/ckpt/text_qwen3vl/last.pth"
DEFAULT_OUTPUT_DIR = "/data/feihong/univerisity_dev/eval_results/test_unify"
DEFAULT_INCLUDE_FILE = "/data/feihong/ckpt/include1.json"


def _normalize_label(label: str) -> str:
    return str(label).strip().split(".")[0]


def _path_label(path: str) -> str:
    return _normalize_label(Path(str(path)).stem)


def _drone_image_name(path: str) -> str:
    return Path(str(path)).name


def center_distance(pred_xyxy: torch.Tensor, gt_xyxy: torch.Tensor) -> float:
    pred_cx = 0.5 * (pred_xyxy[0] + pred_xyxy[2])
    pred_cy = 0.5 * (pred_xyxy[1] + pred_xyxy[3])
    gt_cx = 0.5 * (gt_xyxy[0] + gt_xyxy[2])
    gt_cy = 0.5 * (gt_xyxy[1] + gt_xyxy[3])
    return float(torch.sqrt((pred_cx - gt_cx) ** 2 + (pred_cy - gt_cy) ** 2).item())


def load_include_map(include_file: Optional[str]) -> Dict[str, Set[str]]:
    include_map: Dict[str, Set[str]] = {}
    if not include_file or not os.path.exists(include_file):
        return include_map
    with open(include_file, "r", encoding="utf-8") as f:
        raw = json.load(f)
    for key, values in raw.items():
        norm_key = _normalize_label(key)
        if isinstance(values, list):
            include_map[norm_key] = {_normalize_label(v) for v in values}
        else:
            include_map[norm_key] = {_normalize_label(values)}
    return include_map


def load_text_checkpoint(model: torch.nn.Module, checkpoint_path: str) -> Dict[str, Any]:
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    payload = torch.load(checkpoint_path, map_location="cpu")
    state = payload.get("model", payload) if isinstance(payload, dict) else payload
    if not isinstance(state, dict):
        raise TypeError(f"Checkpoint state must be a dict, got {type(state)}.")
    state = {str(k).replace("module.", ""): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    unexpected = list(unexpected)
    if unexpected:
        raise RuntimeError(f"Unexpected checkpoint keys: {unexpected[:20]}")
    trainable_keys = {name for name, param in model.named_parameters() if param.requires_grad}
    missing_trainable = [name for name in missing if name in trainable_keys]
    if missing_trainable:
        raise RuntimeError(f"Missing trainable checkpoint keys: {missing_trainable[:20]}")
    return payload if isinstance(payload, dict) else {}


def create_loader(
    processor: Any,
    tokenizer: Any,
    batch_size: int,
    num_workers: int,
    test_crop_ratio: float,
    subset_heights: Optional[Sequence[int]],
    subset_angles: Optional[Sequence[int]],
) -> DataLoader:
    Config.TEST_CROP_RATIO = float(test_crop_ratio)
    if subset_heights is not None:
        Config.SUBSET_HEIGHTS = [int(v) for v in subset_heights]
    if subset_angles is not None:
        Config.SUBSET_ANGLES = [int(v) for v in subset_angles]

    dataset = TextVLMSatelliteDataset(
        processor=processor,
        tokenizer=tokenizer,
        split="test",
    )
    return DataLoader(
        dataset,
        shuffle=False,
        batch_size=batch_size,
        **build_dataloader_kwargs(num_workers, drop_last=False),
    )


@torch.inference_mode()
def extract_records(
    model: TextQwenVLRetrievalGrounding,
    loader: DataLoader,
    device: torch.device,
    heatmap_confidence_weight: float,
) -> Dict[str, Any]:
    anchors_full = get_tensor_anchors(device)
    model.eval()

    query_feats: List[torch.Tensor] = []
    query_labels: List[str] = []
    query_drone_paths: List[str] = []
    query_satellite_paths: List[str] = []
    query_heights: List[int] = []
    query_angles: List[int] = []
    iou_values: List[float] = []
    center_distances: List[float] = []
    gallery_feat_dict: Dict[str, torch.Tensor] = {}
    gallery_path_dict: Dict[str, str] = {}
    amp_enabled = Config.USE_AMP and device.type == "cuda"

    for batch in tqdm(loader, desc="Extract/eval [text_qwen3vl]"):
        search_pixel_values = batch["search_pixel_values"].to(device, non_blocking=True)
        image_grid_thw = batch["image_grid_thw"].to(device, non_blocking=True)
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        target_bbox = batch["bbox"].to(device, non_blocking=True)
        geo = build_geo_features(batch, device)

        with torch.amp.autocast("cuda", enabled=amp_enabled):
            outputs = model(search_pixel_values, image_grid_thw, input_ids, attention_mask, geo)

        pred_anchor, _, _, text_feats, grid_feats, _, heatmap_logits = outputs
        query_feats.append(F.normalize(text_feats, p=2, dim=1).cpu())
        gallery_grid_batch = F.normalize(grid_feats, p=2, dim=2)

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

        batch_size = int(pred_anchor.shape[0])
        pred_anchor = pred_anchor.view(batch_size, 9, 5, pred_anchor.shape[-2], pred_anchor.shape[-1])
        old_weight = Config.HEATMAP_CONFIDENCE_WEIGHT
        Config.HEATMAP_CONFIDENCE_WEIGHT = float(heatmap_confidence_weight)
        try:
            pred_anchor = add_heatmap_to_confidence(pred_anchor, heatmap_logits)
        finally:
            Config.HEATMAP_CONFIDENCE_WEIGHT = old_weight

        image_wh = (int(Config.UNIV_SAT_SIZE["width"]), int(Config.UNIV_SAT_SIZE["height"]))
        grid_wh = (int(pred_anchor.shape[-1]), int(pred_anchor.shape[-2]))
        _, best_anchor_gi_gj = build_target(target_bbox, anchors_full, image_wh, grid_wh)
        _, _, _, _, pred_bbox_xyxy, target_bbox_xyxy = eval_iou_acc(
            pred_anchor,
            target_bbox,
            anchors_full,
            best_anchor_gi_gj[:, 1],
            best_anchor_gi_gj[:, 2],
            image_wh,
            iou_threshold_list=[0.5, 0.25],
        )
        ious = bbox_iou(pred_bbox_xyxy, target_bbox_xyxy, x1y1x2y2=True)
        for idx in range(pred_bbox_xyxy.shape[0]):
            iou_values.append(float(ious[idx].item()))
            center_distances.append(center_distance(pred_bbox_xyxy[idx].float(), target_bbox_xyxy[idx].float()))

    if not query_feats or not gallery_feat_dict:
        raise RuntimeError("No text_qwen3vl features were extracted.")

    gallery_labels = sorted(gallery_feat_dict.keys())
    return {
        "query_feats": torch.cat(query_feats, dim=0),
        "query_labels": query_labels,
        "query_drone_paths": query_drone_paths,
        "query_satellite_paths": query_satellite_paths,
        "query_heights": query_heights,
        "query_angles": query_angles,
        "iou_values": iou_values,
        "center_distances": center_distances,
        "gallery_labels": gallery_labels,
        "gallery_satellite_paths": [gallery_path_dict[label] for label in gallery_labels],
        "gallery_feats": torch.stack([gallery_feat_dict[label] for label in gallery_labels], dim=0),
    }


def score_records(
    bundle: Dict[str, Any],
    include_map: Dict[str, Set[str]],
    device: torch.device,
    candidate_size: Optional[int],
    seed: int,
) -> List[Dict[str, Any]]:
    gallery_labels = bundle["gallery_labels"]
    norm_gallery_labels = [_normalize_label(label) for label in gallery_labels]
    label_to_gallery_index = {label: idx for idx, label in enumerate(gallery_labels)}
    all_indices = list(range(len(gallery_labels)))
    records: List[Dict[str, Any]] = []

    for q_idx, gt_label in enumerate(bundle["query_labels"]):
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
            rng = random.Random(int(seed) + int(q_idx))
            candidate_indices = rng.sample(negative_pool, int(candidate_size) - 1) + [gt_gallery_index]

        query_feat = bundle["query_feats"][q_idx].to(device)
        gallery_feats = bundle["gallery_feats"][candidate_indices].to(device)
        score_vec = torch.einsum("d,knd->kn", query_feat, gallery_feats).max(dim=1)[0]

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
        iou = float(bundle["iou_values"][q_idx])
        records.append(
            {
                "query_index": int(q_idx),
                "height": int(bundle["query_heights"][q_idx]),
                "angle": int(bundle["query_angles"][q_idx]),
                "gt_label": gt_norm,
                "pred_label": _normalize_label(gallery_labels[top1_global_idx]),
                "top1_correct": bool(top1_correct),
                "top5_correct": bool(top5_correct),
                "top10_correct": bool(top10_correct),
                "iou": iou,
                "uIoU": float(iou if top1_correct else 0.0),
                "center_distance": float(bundle["center_distances"][q_idx]),
                "drone_path": bundle["query_drone_paths"][q_idx],
                "gt_satellite_path": bundle["query_satellite_paths"][q_idx],
                "pred_satellite_path": bundle["gallery_satellite_paths"][top1_global_idx],
            }
        )

    if not records:
        raise RuntimeError("No valid text_qwen3vl queries matched gallery labels.")
    return records


def summarize_records(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    iou_arr = np.array([float(item["iou"]) for item in records], dtype=np.float32)
    uiou_arr = np.array([float(item["uIoU"]) for item in records], dtype=np.float32)
    center_arr = np.array([float(item["center_distance"]) for item in records], dtype=np.float32)
    n = int(len(records))
    top1_hits = int(sum(bool(item["top1_correct"]) for item in records))
    top5_hits = int(sum(bool(item["top5_correct"]) for item in records))
    top10_hits = int(sum(bool(item["top10_correct"]) for item in records))
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
        "ratio_uIoU_gt_0_25": float((uiou_arr > 0.25).mean()),
        "ratio_uIoU_gt_25": float((uiou_arr > 0.25).mean()),
        "mean_center_distance": float(center_arr.mean()),
    }


def group_summaries(records: Sequence[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
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


def save_metrics(metrics: Dict[str, Any], output_dir: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    checkpoint_name = Path(str(metrics["checkpoint"])).resolve().parent.name
    out_file = os.path.join(output_dir, f"test_text_qwen3vl_{checkpoint_name}.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    return out_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate text.py Qwen3-VL retrieval and grounding.")
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--include-file", type=str, default=DEFAULT_INCLUDE_FILE)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--candidate-size", type=int, default=100)
    parser.add_argument("--test-crop-ratio", type=float, default=1.0)
    parser.add_argument("--subset-heights", type=int, nargs="*", default=None)
    parser.add_argument("--subset-angles", type=int, nargs="*", default=None)
    parser.add_argument("--heatmap-confidence-weight", type=float, default=0.5)
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

    device = torch.device(args.device)
    processor = AutoProcessor.from_pretrained(Config.MODEL_NAME, cache_dir=Config.CACHE_DIR)
    tokenizer = processor.tokenizer
    if getattr(tokenizer, "pad_token_id", None) is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = TextQwenVLRetrievalGrounding(
        model_name=Config.MODEL_NAME,
        cache_dir=Config.CACHE_DIR,
        dtype=_dtype_from_name(Config.DTYPE),
        head_dim=Config.HEAD_DIM,
        retrieval_dim=Config.RETRIEVAL_DIM,
        heat_softmax_temperature=Config.HEAT_SOFTMAX_TEMPERATURE,
    ).to(device)
    checkpoint_payload = load_text_checkpoint(model, args.checkpoint)

    loader = create_loader(
        processor=processor,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        test_crop_ratio=args.test_crop_ratio,
        subset_heights=args.subset_heights,
        subset_angles=args.subset_angles,
    )
    bundle = extract_records(
        model=model,
        loader=loader,
        device=device,
        heatmap_confidence_weight=args.heatmap_confidence_weight,
    )
    records = score_records(
        bundle=bundle,
        include_map=load_include_map(args.include_file),
        device=device,
        candidate_size=args.candidate_size,
        seed=args.seed,
    )
    per_subset, per_height, per_angle = group_summaries(records)
    overall = summarize_records(records)
    metrics = {
        "model_type": "text_qwen3vl",
        "checkpoint": args.checkpoint,
        "checkpoint_epoch": checkpoint_payload.get("epoch") if isinstance(checkpoint_payload, dict) else None,
        "candidate_size": args.candidate_size,
        "include_file": args.include_file,
        "num_gallery": int(len(bundle["gallery_labels"])),
        "num_samples": int(overall["num_samples"]),
        "overall": overall,
        "retrieval": {
            "num_queries": int(overall["num_samples"]),
            "recall@1": overall["recall@1"],
            "recall@5": overall["recall@5"],
            "recall@10": overall["recall@10"],
        },
        "top1_success_drone_names": [
            _drone_image_name(item["drone_path"]) for item in records if item["top1_correct"]
        ],
        "per_subset": per_subset,
        "per_height": per_height,
        "per_angle": per_angle,
        "query_records": records if args.save_query_records else [],
    }
    out_file = save_metrics(metrics, args.output_dir)
    print("\n=== test_text Summary ===")
    print(f"Checkpoint: {args.checkpoint}")
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
    print(f"Saved: {out_file}")


if __name__ == "__main__":
    main()
