import argparse
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from bbox.yolo_utils import bbox_iou
from dataset import ShiftedSatelliteDroneDataset
from grounding.config import load_config
from grounding.processors import build_grounding_image_processors
from grounding.registry import build_model_and_adapter
from grounding.training_records import load_v2_resume_checkpoint
from test_unify import (
    DEFAULT_INCLUDE_FILE,
    FeatureBundle,
    _load_include_map,
    _path_label,
    center_distance,
    group_summaries,
    parse_anchors,
    print_summary,
    save_metrics,
    score_retrieval_and_uiou,
    summarize_records,
)


GROUND_MODEL_TYPES = (
    "det",
    "lpn",
    "sample4geo",
    "trogeolite",
    "ocg",
    "smgeo",
)
DEFAULT_CONFIG_DIR = Path("configs/grounding")
DEFAULT_OUTPUT_DIR = "eval_results/test_unify_ground"
DEFAULT_DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
DEFAULT_SUBSET_HEIGHTS = [150, 200, 250, 300]
DEFAULT_SUBSET_ANGLES = [0, 45, 90, 135, 180, 225, 270, 315]


def _config_path(model_type: str, config_dir: Path) -> Path:
    path = Path(config_dir) / f"{model_type}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Grounding config not found: {path}")
    return path


def _checkpoint_for_model(
    model_type: str,
    cfg: Dict[str, Any],
    explicit_checkpoint: Optional[str],
    selected_count: int,
) -> str:
    if explicit_checkpoint:
        if selected_count != 1:
            raise ValueError("--checkpoint requires exactly one --model-types value.")
        return str(explicit_checkpoint)
    return str(Path(cfg["save_dir"]) / str(cfg["eval"].get("checkpoint", "last.pth")))


def _create_ground_loader(
    cfg: Dict[str, Any],
    batch_size: int,
    num_workers: int,
    sat_size: Tuple[int, int],
    test_crop_ratio: float,
    subset_heights: Optional[Sequence[int]],
    subset_angles: Optional[Sequence[int]],
) -> DataLoader:
    cfg = dict(cfg)
    cfg["data"] = dict(cfg["data"])
    cfg["data"]["sat_size"] = {"height": int(sat_size[0]), "width": int(sat_size[1])}
    processor, processor_sat = build_grounding_image_processors(cfg)
    tokenizer = AutoTokenizer.from_pretrained(
        cfg["model"]["model_name"],
        cache_dir=cfg["model"]["cache_dir"],
        local_files_only=True,
    )
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


def extract_ground_features(
    model_type: str,
    config_path: str,
    checkpoint_path: str,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    sat_size: Tuple[int, int],
    test_crop_ratio: float,
    subset_heights: Optional[Sequence[int]],
    subset_angles: Optional[Sequence[int]],
) -> FeatureBundle:
    cfg = load_config(config_path)
    cfg["model"]["type"] = model_type
    cfg["data"]["sat_size"] = {"height": int(sat_size[0]), "width": int(sat_size[1])}
    model, adapter = build_model_and_adapter(cfg)
    payload = load_v2_resume_checkpoint(Path(checkpoint_path))
    model.load_state_dict(payload["model"], strict=True)
    model.to(device).eval()
    adapter.model = model

    loader = _create_ground_loader(
        cfg=cfg,
        batch_size=batch_size,
        num_workers=num_workers,
        sat_size=sat_size,
        test_crop_ratio=test_crop_ratio,
        subset_heights=subset_heights,
        subset_angles=subset_angles,
    )
    anchors_full = parse_anchors(device)

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

    with torch.inference_mode():
        for batch in tqdm(loader, desc=f"Extract/eval [{model_type}]"):
            output = adapter.forward(batch, device)
            if output.query_embedding is None or output.search_local_features is None:
                raise ValueError(f"{model_type} did not return query/local features.")

            query_batch = F.normalize(output.query_embedding, p=2, dim=1)
            local = F.normalize(output.search_local_features, p=2, dim=1)
            local = local.permute(0, 2, 3, 1).reshape(local.shape[0], -1, local.shape[1])
            query_feats.append(query_batch.cpu())

            batch_labels = [_path_label(path) for path in batch["satellite_path"]]
            query_labels.extend(batch_labels)
            query_drone_paths.extend([str(path) for path in batch["drone_path"]])
            query_satellite_paths.extend([str(path) for path in batch["satellite_path"]])
            query_heights.extend([int(value) for value in batch["height"].tolist()])
            query_angles.extend([int(value) for value in batch["angle"].tolist()])
            for index, label in enumerate(batch_labels):
                if label not in gallery_feat_dict:
                    gallery_feat_dict[label] = local[index].cpu()
                    gallery_path_dict[label] = str(batch["satellite_path"][index])

            pred_bbox = adapter.decode(output, batch, anchors_full)
            target_bbox = batch["bbox"].to(device=pred_bbox.device, dtype=pred_bbox.dtype)
            ious = bbox_iou(pred_bbox, target_bbox, x1y1x2y2=True)
            for index in range(pred_bbox.shape[0]):
                iou_values.append(float(ious[index].item()))
                center_distances.append(center_distance(pred_bbox[index], target_bbox[index]))

    if not query_feats or not gallery_feat_dict:
        raise RuntimeError(f"No features extracted for {model_type}.")
    gallery_labels = sorted(gallery_feat_dict)
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
        gallery_satellite_paths=[gallery_path_dict[label] for label in gallery_labels],
        gallery_feats=torch.stack([gallery_feat_dict[label] for label in gallery_labels], dim=0),
    )


def build_ground_metrics(
    model_type: str,
    checkpoint_path: str,
    bundle: FeatureBundle,
    include_map: Dict[str, Set[str]],
    device: torch.device,
    candidate_size: Optional[int],
    seed: int,
    test_crop_ratio: float,
    subset_heights: Optional[Sequence[int]],
    subset_angles: Optional[Sequence[int]],
    include_query_records: bool,
) -> Dict[str, Any]:
    scored = score_retrieval_and_uiou(
        model_type="encoder_heat",
        bundle=bundle,
        include_map=include_map,
        device=device,
        candidate_size=candidate_size,
        unify_score_mode="rerank",
        sampling_seed=seed,
    )
    records = scored["query_records"]
    overall = summarize_records(records)
    per_subset, per_height, per_angle = group_summaries(records)
    return {
        "model_type": model_type,
        "checkpoint": checkpoint_path,
        "candidate_size": candidate_size,
        "test_crop_ratio": float(test_crop_ratio),
        "seed": int(seed),
        "subset_heights": list(subset_heights or DEFAULT_SUBSET_HEIGHTS),
        "subset_angles": list(subset_angles or DEFAULT_SUBSET_ANGLES),
        "num_gallery": len(bundle.gallery_labels),
        "num_samples": int(overall["num_samples"]),
        "overall": overall,
        "retrieval": {
            key: value
            for key, value in scored.items()
            if key not in {"top1_success_drone_names", "query_records"}
        },
        "top1_success_drone_names": scored["top1_success_drone_names"],
        "per_subset": per_subset,
        "per_height": per_height,
        "per_angle": per_angle,
        "query_records": records if include_query_records else [],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate trained grounding models with the test_unify protocol."
    )
    parser.add_argument("--model-types", nargs="+", choices=GROUND_MODEL_TYPES, default=["det"])
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--config-dir", type=Path, default=DEFAULT_CONFIG_DIR)
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--sat-size", type=int, nargs=2, default=[432, 768], metavar=("HEIGHT", "WIDTH"))
    parser.add_argument("--test-crop-ratio", type=float, default=1.0)
    parser.add_argument("--subset-heights", type=int, nargs="*", default=None)
    parser.add_argument("--subset-angles", type=int, nargs="*", default=None)
    parser.add_argument("--candidate-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--include-file", type=str, default=DEFAULT_INCLUDE_FILE)
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-suffix", type=str, default=None)
    parser.add_argument("--save-query-records", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    candidate_size = None if args.candidate_size <= 0 else int(args.candidate_size)
    device = torch.device(args.device if not args.device.startswith("cuda") or torch.cuda.is_available() else "cpu")
    include_map = _load_include_map(args.include_file)

    for model_type in args.model_types:
        config_path = _config_path(model_type, args.config_dir)
        cfg = load_config(str(config_path))
        checkpoint_path = _checkpoint_for_model(
            model_type,
            cfg,
            args.checkpoint,
            len(args.model_types),
        )
        bundle = extract_ground_features(
            model_type=model_type,
            config_path=str(config_path),
            checkpoint_path=checkpoint_path,
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            sat_size=(int(args.sat_size[0]), int(args.sat_size[1])),
            test_crop_ratio=args.test_crop_ratio,
            subset_heights=args.subset_heights,
            subset_angles=args.subset_angles,
        )
        metrics = build_ground_metrics(
            model_type=model_type,
            checkpoint_path=checkpoint_path,
            bundle=bundle,
            include_map=include_map,
            device=device,
            candidate_size=candidate_size,
            seed=args.seed,
            test_crop_ratio=args.test_crop_ratio,
            subset_heights=args.subset_heights,
            subset_angles=args.subset_angles,
            include_query_records=args.save_query_records,
        )
        out_file = save_metrics(metrics, args.output_dir, name_suffix=args.output_suffix)
        print_summary(metrics, out_file)


if __name__ == "__main__":
    main()
