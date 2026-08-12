import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoTokenizer

from bbox.yolo_utils import bbox_iou, build_target, eval_iou_acc
from dataset import DEFAULT_SUBSET_ANGLES, DEFAULT_SUBSET_HEIGHTS, ShiftedSatelliteDroneDataset
from hf_cache_utils import from_pretrained_prefer_local
from model import Encoder_ada, Encoder_test
from model_abla import model_bi, model_pre


# --- Configuration ---
MODEL_NAME = "google/siglip2-base-patch16-224"
CACHE_DIR = "/media/data1/feihong/hf_cache"
INCLUDE_FILE = "/media/data1/feihong/ckpt/include2.json"
ANCHORS = "37,41, 78,84, 96,215, 129,129, 194,82, 198,179, 246,280, 395,342, 550,573"


def normalize_label(label):
    return str(label).strip().split(".")[0]


def load_include_map(path):
    with open(path, "r", encoding="utf-8") as f:
        raw_map = json.load(f)
    return {
        normalize_label(key): {normalize_label(value) for value in values}
        for key, values in raw_map.items()
    }


def parse_anchors(device):
    anchors = np.array([float(value) for value in ANCHORS.split(",")], dtype=np.float32)
    return torch.tensor(anchors.reshape(-1, 2)[::-1].copy(), device=device)


def build_geo_features(batch, device):
    angles = batch["angle"].to(device).float()
    heights = batch["height"].to(device).float()
    radians = torch.deg2rad(angles)
    return torch.stack(
        [torch.cos(radians), torch.sin(radians), heights / 300.0],
        dim=1,
    )


def add_heatmap_to_confidence(pred_anchor, heatmap_logits, weight):
    if heatmap_logits.shape[-2:] != pred_anchor.shape[-2:]:
        heatmap_logits = F.interpolate(
            heatmap_logits,
            size=pred_anchor.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
    heatmap_confidence = weight * torch.sigmoid(heatmap_logits.detach()).unsqueeze(1)
    return torch.cat(
        [pred_anchor[:, :, :4], pred_anchor[:, :, 4:5] + heatmap_confidence],
        dim=2,
    )


def center_distance(pred_bbox, target_bbox):
    pred_center = 0.5 * (pred_bbox[:2] + pred_bbox[2:])
    target_center = 0.5 * (target_bbox[:2] + target_bbox[2:])
    return float(torch.linalg.vector_norm(pred_center - target_center).item())


def create_loader(args):
    processor = from_pretrained_prefer_local(
        AutoImageProcessor,
        args.model_name,
        CACHE_DIR,
    )
    processor_sat = from_pretrained_prefer_local(
        AutoImageProcessor,
        args.model_name,
        CACHE_DIR,
        size={"height": args.sat_size[0], "width": args.sat_size[1]},
    )
    tokenizer = from_pretrained_prefer_local(
        AutoTokenizer,
        args.model_name,
        CACHE_DIR,
    )
    dataset = ShiftedSatelliteDroneDataset(
        processor=processor,
        processor_sat=processor_sat,
        tokenizer=tokenizer,
        split="test",
        sat_target_size=tuple(args.sat_size),
        test_crop_ratio=args.test_crop_ratio,
        subset_heights=args.subset_heights,
        subset_angles=args.subset_angles,
    )
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
        prefetch_factor=4 if args.num_workers > 0 else None,
    )


def load_model(args, device):
    kwargs = {
        "model_name": args.model_name,
        "proj_dim": 768,
        "usesg": True,
        "useap": args.use_ap,
        "use_heatmap": args.use_heatmap,
        "lora_rank": 8,
        "lora_alpha": 16.0,
        "lora_dropout": 0.05,
    }
    if args.encoder_type == "model_pre":
        model = model_pre(
            ckpt_path=args.pretrained_checkpoint,
            **kwargs,
        )
    elif args.encoder_type == "model_bi":
        model = model_bi(**kwargs)
    elif args.encoder_type == "ada":
        model = Encoder_ada(**kwargs)
    else:
        model = Encoder_test(**kwargs)
    model = model.to(device)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def extract_features(args, device):
    model = load_model(args, device)
    loader = create_loader(args)
    anchors = parse_anchors(device)

    query_features = []
    query_labels = []
    query_data = []
    gallery_features = {}
    gallery_paths = {}

    with torch.inference_mode():
        for batch in tqdm(loader, desc=f"Extract/test [{args.encoder_type}]"):
            query_images = batch["target_pixel_values"].to(device)
            satellite_images = batch["search_pixel_values"].to(device)
            target_bbox = batch["bbox"].to(device)
            geo = build_geo_features(batch, device) if args.use_angle else None

            outputs = model(query_images, satellite_images, angle=geo)
            pred_anchor, _, _, anchor_features, grid_features, refine_outputs, heatmap = outputs
            query_features.append(F.normalize(anchor_features, p=2, dim=1).cpu())
            grid_features = F.normalize(grid_features, p=2, dim=2)

            labels = [normalize_label(Path(path).stem) for path in batch["satellite_path"]]
            query_labels.extend(labels)
            for index, label in enumerate(labels):
                if label not in gallery_features:
                    gallery_features[label] = grid_features[index].cpu()
                    gallery_paths[label] = batch["satellite_path"][index]

            batch_size = pred_anchor.shape[0]
            pred_anchor = pred_anchor.view(
                batch_size,
                9,
                5,
                pred_anchor.shape[-2],
                pred_anchor.shape[-1],
            )
            if args.use_heatmap:
                pred_anchor = add_heatmap_to_confidence(
                    pred_anchor,
                    heatmap,
                    args.heatmap_confidence_weight,
                )
            image_wh = (args.sat_size[1], args.sat_size[0])
            grid_wh = (pred_anchor.shape[-1], pred_anchor.shape[-2])
            _, best_anchor = build_target(target_bbox, anchors, image_wh, grid_wh)
            _, _, _, _, pred_bbox, gt_bbox = eval_iou_acc(
                pred_anchor,
                target_bbox,
                anchors,
                best_anchor[:, 1],
                best_anchor[:, 2],
                image_wh,
                iou_threshold_list=[0.5, 0.25],
            )
            if "bbox" in refine_outputs:
                pred_bbox = refine_outputs["bbox"].to(target_bbox)
                gt_bbox = target_bbox
            ious = bbox_iou(pred_bbox, gt_bbox, x1y1x2y2=True)

            for index in range(batch_size):
                query_data.append(
                    {
                        "height": int(batch["height"][index]),
                        "angle": int(batch["angle"][index]),
                        "iou": float(ious[index]),
                        "center_distance": center_distance(pred_bbox[index], gt_bbox[index]),
                        "drone_path": batch["drone_path"][index],
                        "satellite_path": batch["satellite_path"][index],
                    }
                )

    gallery_labels = sorted(gallery_features)
    return {
        "query_features": torch.cat(query_features),
        "query_labels": query_labels,
        "query_data": query_data,
        "gallery_labels": gallery_labels,
        "gallery_paths": [gallery_paths[label] for label in gallery_labels],
        "gallery_features": torch.stack([gallery_features[label] for label in gallery_labels]),
    }


def score_queries(features, args, device):
    include_map = load_include_map(args.include_file)
    gallery_labels = features["gallery_labels"]
    gallery_lookup = {label: index for index, label in enumerate(gallery_labels)}
    all_indices = list(range(len(gallery_labels)))
    records = []

    for query_index, gt_label in enumerate(features["query_labels"]):
        gt_index = gallery_lookup[gt_label]
        if args.candidate_size >= len(all_indices):
            candidate_indices = all_indices
        else:
            negatives = [index for index in all_indices if index != gt_index]
            rng = random.Random(args.seed + query_index)
            candidate_indices = rng.sample(negatives, args.candidate_size - 1) + [gt_index]

        gallery = features["gallery_features"][candidate_indices].to(device)
        query = features["query_features"][query_index].to(device)
        scores = torch.einsum("d,knd->kn", query, gallery).max(dim=1).values
        top_indices = scores.topk(min(10, scores.numel())).indices.tolist()

        positive_labels = set(include_map.get(gt_label, set()))
        positive_labels.add(gt_label)
        positive_indices = {
            local_index
            for local_index, global_index in enumerate(candidate_indices)
            if gallery_labels[global_index] in positive_labels
        }
        top1_correct = top_indices[0] in positive_indices
        top5_correct = any(index in positive_indices for index in top_indices[:5])
        top10_correct = any(index in positive_indices for index in top_indices[:10])
        pred_global_index = candidate_indices[top_indices[0]]
        data = features["query_data"][query_index]
        records.append(
            {
                **data,
                "gt_label": gt_label,
                "pred_label": gallery_labels[pred_global_index],
                "pred_satellite_path": features["gallery_paths"][pred_global_index],
                "top1_correct": top1_correct,
                "top5_correct": top5_correct,
                "top10_correct": top10_correct,
                "uIoU": data["iou"] if top1_correct else 0.0,
            }
        )
    return records


def summarize(records):
    count = len(records)
    ious = np.array([record["iou"] for record in records], dtype=np.float32)
    unified_ious = np.array([record["uIoU"] for record in records], dtype=np.float32)
    distances = np.array([record["center_distance"] for record in records], dtype=np.float32)
    successful_distances = np.array(
        [record["center_distance"] for record in records if record["top1_correct"]],
        dtype=np.float32,
    )
    top1 = sum(record["top1_correct"] for record in records)
    top5 = sum(record["top5_correct"] for record in records)
    top10 = sum(record["top10_correct"] for record in records)
    return {
        "num_samples": count,
        "top1_hits": top1,
        "top5_hits": top5,
        "top10_hits": top10,
        "recall@1": top1 / count,
        "recall@5": top5 / count,
        "recall@10": top10 / count,
        "mean_iou": float(ious.mean()),
        "ratio_iou_gt_0_5": float((ious > 0.5).mean()),
        "ratio_iou_gt_0_25": float((ious > 0.25).mean()),
        "uIoU": float(unified_ious.mean()),
        "ratio_uIoU_gt_0_25": float((unified_ious > 0.25).mean()),
        "mean_center_distance": float(distances.mean()),
        "uCDE": float(successful_distances.mean()) if len(successful_distances) else None,
    }


def group_summaries(records, key_names):
    groups = {}
    for record in records:
        key = tuple(record[name] for name in key_names)
        groups.setdefault(key, []).append(record)
    return [
        {**dict(zip(key_names, key)), **summarize(group)}
        for key, group in sorted(groups.items())
    ]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default="eval_results")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--sat-size", type=int, nargs=2, default=[432, 768])
    parser.add_argument("--test-crop-ratio", type=float, default=1.0)
    parser.add_argument("--candidate-size", type=int, default=100)
    parser.add_argument("--subset-heights", type=int, nargs="*", default=DEFAULT_SUBSET_HEIGHTS)
    parser.add_argument("--subset-angles", type=int, nargs="*", default=DEFAULT_SUBSET_ANGLES)
    parser.add_argument("--include-file", default=INCLUDE_FILE)
    parser.add_argument("--heatmap-confidence-weight", type=float, default=0.5)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=43)
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        payload = yaml.safe_load(f)
    config = payload["config"]
    args.exp_name = payload["exp_name"]
    args.encoder_type = config["ENCODER_TYPE"]
    args.model_name = config["MODEL_NAME"]
    args.use_ap = payload["use_ap"]
    args.use_angle = config["USE_ANGLE_INPUT"]
    args.use_heatmap = config["USE_HEATMAP_LOSS"]
    args.heatmap_confidence_weight = config["HEATMAP_CONFIDENCE_WEIGHT"]
    args.pretrained_checkpoint = config.get("PRETRAINED_CHECKPOINT")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    features = extract_features(args, device)
    records = score_queries(features, args, device)
    metrics = {
        "model_type": args.encoder_type,
        "checkpoint": args.checkpoint,
        "sat_size": {"height": args.sat_size[0], "width": args.sat_size[1]},
        "candidate_size": args.candidate_size,
        "test_crop_ratio": args.test_crop_ratio,
        "num_gallery": len(features["gallery_labels"]),
        "overall": summarize(records),
        "per_subset": group_summaries(records, ["height", "angle"]),
        "per_height": group_summaries(records, ["height"]),
        "per_angle": group_summaries(records, ["angle"]),
        "query_records": records,
    }
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, f"{args.exp_name}.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    overall = metrics["overall"]
    print(
        f"R@1={overall['recall@1']:.4f} R@5={overall['recall@5']:.4f} "
        f"R@10={overall['recall@10']:.4f} mIoU={overall['mean_iou']:.4f} "
        f"uIoU={overall['uIoU']:.4f}"
    )
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
