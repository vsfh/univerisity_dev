import importlib.util
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from ground_cvos import DetGeoLite, LPNGeoLite, SampleGeoLite, TROGeoLite
from ground_siglip import Encoder as SigLIPModel
from bbox.yolo_utils import build_target
from utils.utils import bbox_iou, eval_iou_acc


# --- Configuration ---
MODEL_NAME_SIGLIP = "google/siglip-base-patch16-224"
CACHE_DIR = "/data/feihong/hf_cache"
DEFAULT_SAT_SIZE = (432, 768)  # (H, W)
DEFAULT_DRONE_SIZE = (256, 256)  # (H, W)
DEFAULT_DEVICE = "cuda:1" if torch.cuda.is_available() else "cpu"
DEFAULT_SUBSET_HEIGHTS = [150, 200, 250, 300]
DEFAULT_SUBSET_ANGLES = [0, 90, 180, 270]
ANCHORS = "37,41, 78,84, 96,215, 129,129, 194,82, 198,179, 246,280, 395,342, 550,573"

EVAL_CONFIG = {
    "device": DEFAULT_DEVICE,
    "batch_size": 2,
    "num_workers": 8,
    "sat_size": [432, 768],
    "test_crop_ratio": 0.5,
    "subset_heights": [300],
    "subset_angles": [0],
    "output_dir": "/data/feihong/univerisity_dev/eval_results",
    "visualize_output_dir": "",
    "models": {
        "det": {
            "run": True,
            "checkpoint": "/data/feihong/ckpt/ground_det/last.pth",
        },
        "siglip": {"run": False, "checkpoint": None},
        "encoder_abla": {"run": True, "checkpoint": '/data/feihong/ckpt/supp_0.1/best_iou.pth'},
        "lpn": {"run": False, "checkpoint": None},
        "smgeo": {"run": False, "checkpoint": None},
        "trogeolite": {"run": False, "checkpoint": None},
    },
}


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_encoder_abla_class():
    """Load root model.py explicitly to avoid import ambiguity."""
    repo_root_str = str(REPO_ROOT)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)
    model_path = REPO_ROOT / "model.py"
    spec = importlib.util.spec_from_file_location("root_model_module", str(model_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load model module from: {model_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Encoder_abla


EncoderAbla = _load_encoder_abla_class()


def _load_shared_dataset_class():
    """Load root dataset.py explicitly to avoid importing grounding/dataset.py by name."""
    dataset_path = Path(__file__).resolve().parents[1] / "dataset.py"
    spec = importlib.util.spec_from_file_location("shared_dataset_module", str(dataset_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load dataset module from: {dataset_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.ShiftedSatelliteDroneDataset


ShiftedSatelliteDroneDataset = _load_shared_dataset_class()



class TransformProcessorWrapper:
    """Tensor wrapper matching training behavior for cvos-based models."""

    def __init__(self, image_size: Tuple[int, int]):
        self.size = {"height": int(image_size[0]), "width": int(image_size[1])}

    def __call__(self, images, return_tensors="pt"):
        target_h = int(self.size["height"])
        target_w = int(self.size["width"])
        if images.size != (target_w, target_h):
            images = images.resize((target_w, target_h))
        image_np = np.array(images, dtype=np.float32) / 255.0
        pixel_values = torch.from_numpy(image_np).permute(2, 0, 1)
        return {"pixel_values": pixel_values.unsqueeze(0)}


class DummyTokenizer:
    def __call__(
        self,
        text,
        padding="max_length",
        truncation=True,
        max_length=64,
        return_tensors="pt",
    ):
        del text, padding, truncation, return_tensors
        return {"input_ids": torch.zeros((1, max_length), dtype=torch.long)}


@dataclass
class EvalConfig:
    model_type: str
    checkpoint_path: str
    device: str = DEFAULT_DEVICE
    batch_size: int = 1
    num_workers: int = 4
    sat_size: Tuple[int, int] = DEFAULT_SAT_SIZE
    test_crop_ratio: float = 1.0
    subset_heights: Optional[List[int]] = None
    subset_angles: Optional[List[int]] = None
    visualize_output_dir: str = ""
    output_dir: str = "/data/feihong/univerisity_dev/eval_results"


def _canonical_model_type(model_type: str) -> str:
    name = model_type.lower()
    if name in {"abla", "encoder_abla"}:
        return "encoder_abla"
    return name


def _normalize_subset(values: Optional[Sequence[int]], defaults: Sequence[int]) -> Set[int]:
    if values is None:
        return {int(v) for v in defaults}
    if len(values) == 0:
        return {int(v) for v in defaults}
    return {int(v) for v in values}


def parse_anchors(device: str) -> torch.Tensor:
    anchors = np.array([float(x) for x in ANCHORS.split(",")], dtype=np.float32)
    anchors = anchors.reshape(-1, 2)[::-1].copy()
    return torch.tensor(anchors, dtype=torch.float32, device=device)


def build_model(model_type: str) -> torch.nn.Module:
    model_type = _canonical_model_type(model_type)
    if model_type == "siglip":
        return SigLIPModel(model_name=MODEL_NAME_SIGLIP, proj_dim=768)
    if model_type == "encoder_abla":
        return EncoderAbla(model_name=MODEL_NAME_SIGLIP, proj_dim=768, usesg=True, useap=True)
    if model_type == "det":
        return DetGeoLite(emb_size=512)
    if model_type == "lpn":
        return LPNGeoLite()
    if model_type == "smgeo":
        return SampleGeoLite()
    if model_type == "trogeolite":
        return TROGeoLite()
    raise ValueError(f"Unknown model type: {model_type}")


def load_checkpoint(model: torch.nn.Module, checkpoint_path: str) -> None:
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    state = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    if isinstance(state, dict):
        state = {k.replace("module.", ""): v for k, v in state.items()}

    model.load_state_dict(state, strict=False)


def create_test_loader(config: EvalConfig) -> DataLoader:
    model_type = _canonical_model_type(config.model_type)
    # Use SigLIP processor only for siglip model; others use simple tensor wrapper.
    if model_type in {"siglip", "encoder_abla"}:
        processor = AutoImageProcessor.from_pretrained(MODEL_NAME_SIGLIP, cache_dir=CACHE_DIR)
        processor_sat = AutoImageProcessor.from_pretrained(
            MODEL_NAME_SIGLIP,
            cache_dir=CACHE_DIR,
            size={"height": int(config.sat_size[0]), "width": int(config.sat_size[1])},
        )
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_SIGLIP)
    else:
        processor = TransformProcessorWrapper(DEFAULT_DRONE_SIZE)
        processor_sat = TransformProcessorWrapper(config.sat_size)
        tokenizer = DummyTokenizer()

    test_dataset = ShiftedSatelliteDroneDataset(
        processor=processor,
        processor_sat=processor_sat,
        tokenizer=tokenizer,
        split="train",
        sat_target_size=config.sat_size,
        test_crop_ratio=config.test_crop_ratio,
        subset_heights=config.subset_heights,
        subset_angles=config.subset_angles,
    )

    use_cuda = config.device.startswith("cuda")
    loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=use_cuda,
        persistent_workers=config.num_workers > 0,
        prefetch_factor=4 if config.num_workers > 0 else None,
    )
    return loader


def _to_xyxy_pixels(pred_bbox: torch.Tensor, image_h: int, image_w: int) -> torch.Tensor:
    """Normalize-to-pixel conversion if needed, then clamp."""
    pred = pred_bbox.clone().float()
    if pred.numel() != 4:
        raise ValueError(f"Expected bbox with 4 values, got shape {tuple(pred.shape)}")

    # Heuristic: if values look normalized, scale x by W and y by H.
    if pred.abs().max().item() <= 2.0:
        pred[[0, 2]] = pred[[0, 2]] * float(image_w)
        pred[[1, 3]] = pred[[1, 3]] * float(image_h)

    pred[0] = pred[0].clamp(0, image_w)
    pred[2] = pred[2].clamp(0, image_w)
    pred[1] = pred[1].clamp(0, image_h)
    pred[3] = pred[3].clamp(0, image_h)
    return pred


def _center_distance(pred_xyxy: torch.Tensor, gt_xyxy: torch.Tensor) -> float:
    pred_cx = 0.5 * (pred_xyxy[0] + pred_xyxy[2])
    pred_cy = 0.5 * (pred_xyxy[1] + pred_xyxy[3])
    gt_cx = 0.5 * (gt_xyxy[0] + gt_xyxy[2])
    gt_cy = 0.5 * (gt_xyxy[1] + gt_xyxy[3])
    return float(torch.sqrt((pred_cx - gt_cx) ** 2 + (pred_cy - gt_cy) ** 2).item())


def _tensor_chw_to_pil(image_tensor: torch.Tensor) -> Image.Image:
    """Convert CHW tensor to RGB PIL image with robust range handling."""
    image = image_tensor.detach().float().cpu()
    if image.ndim != 3:
        raise ValueError(f"Expected CHW tensor, got shape: {tuple(image.shape)}")

    image = image.permute(1, 2, 0).numpy()
    img_min = float(image.min())
    img_max = float(image.max())

    if img_max <= 1.0 and img_min >= 0.0:
        scaled = image
    elif img_max > img_min:
        scaled = (image - img_min) / (img_max - img_min)
    else:
        scaled = np.zeros_like(image)

    scaled = np.clip(scaled * 255.0, 0.0, 255.0).astype(np.uint8)
    return Image.fromarray(scaled)


def visualize_batch(
    batch: Dict[str, Any],
    output_dir: str = "runs/visualizations/eval_batch",
) -> List[str]:
    """Save one image per sample: left drone, right satellite with target bbox overlay."""
    query_batch = batch["target_pixel_values"]
    satellite_batch = batch["search_pixel_values"]
    bbox_batch = batch["bbox"]

    num_samples = int(min(query_batch.shape[0], satellite_batch.shape[0], bbox_batch.shape[0]))
    if num_samples <= 0:
        raise ValueError("No samples available for visualization.")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    saved_files: List[str] = []
    for idx in range(num_samples):
        query_img = _tensor_chw_to_pil(query_batch[idx]).convert("RGB")
        sat_img = _tensor_chw_to_pil(satellite_batch[idx]).convert("RGB")

        draw = ImageDraw.Draw(sat_img)
        x1, y1, x2, y2 = [float(v) for v in bbox_batch[idx].detach().float().cpu().tolist()]
        x1 = max(0.0, min(float(sat_img.width - 1), x1))
        y1 = max(0.0, min(float(sat_img.height - 1), y1))
        x2 = max(0.0, min(float(sat_img.width - 1), x2))
        y2 = max(0.0, min(float(sat_img.height - 1), y2))
        draw.rectangle((x1, y1, x2, y2), outline=(0, 255, 0), width=3)

        target_h = 320
        q_w = max(1, int(round(query_img.width * target_h / max(1, query_img.height))))
        s_w = max(1, int(round(sat_img.width * target_h / max(1, sat_img.height))))
        query_img = query_img.resize((q_w, target_h), Image.Resampling.BILINEAR)
        sat_img = sat_img.resize((s_w, target_h), Image.Resampling.BILINEAR)

        pair = Image.new("RGB", (q_w + s_w + 12, target_h), color=(15, 15, 15))
        pair.paste(query_img, (0, 0))
        pair.paste(sat_img, (q_w + 12, 0))

        save_path = out_dir / f"sample_{idx:03d}.png"
        pair.save(save_path)
        saved_files.append(str(save_path))

    return saved_files


def evaluate(config: EvalConfig) -> Dict[str, Any]:
    model_type = _canonical_model_type(config.model_type)
    subset_heights = sorted(
        _normalize_subset(config.subset_heights, DEFAULT_SUBSET_HEIGHTS)
    )
    subset_angles = sorted(
        _normalize_subset(config.subset_angles, DEFAULT_SUBSET_ANGLES)
    )

    config.subset_heights = subset_heights
    config.subset_angles = subset_angles

    print(
        f"Subset filter: heights={subset_heights}, angles={subset_angles}"
    )

    model = build_model(model_type).to(config.device)
    load_checkpoint(model, config.checkpoint_path)
    model.eval()

    loader = create_test_loader(config)
    anchors_full = parse_anchors(config.device)

    iou_values: List[float] = []
    center_distances: List[float] = []
    vis_saved = False

    for batch in tqdm(loader, desc=f"Evaluating {model_type}"):
        if config.visualize_output_dir and not vis_saved:
            saved_files = visualize_batch(batch, output_dir=config.visualize_output_dir)
            print(f"Saved {len(saved_files)} batch visualizations to: {config.visualize_output_dir}")
            vis_saved = True

        query_imgs = batch["target_pixel_values"].to(config.device, non_blocking=True)
        search_imgs = batch["search_pixel_values"].to(config.device, non_blocking=True)
        gt_bbox = batch["bbox"].to(config.device)

        image_h = int(search_imgs.shape[-2])
        image_w = int(search_imgs.shape[-1])

        with torch.no_grad():
            if hasattr(model, "bbox_forward"):
                output = model.bbox_forward(query_imgs, search_imgs)
            elif model_type == "encoder_abla":
                output = model(query_imgs, search_imgs)
            else:
                raise AttributeError(
                    f"Model '{model_type}' has no bbox_forward and no fallback path."
                )

        if isinstance(output, torch.Tensor):
            # Direct bbox regression path.
            pred_bbox_batch = output
            if pred_bbox_batch.ndim == 1:
                pred_bbox_batch = pred_bbox_batch.unsqueeze(0)

            for b in range(pred_bbox_batch.shape[0]):
                pred_xyxy = _to_xyxy_pixels(pred_bbox_batch[b], image_h, image_w)
                gt_xyxy = gt_bbox[b].float()
                iou = float(bbox_iou(pred_xyxy.unsqueeze(0), gt_xyxy.unsqueeze(0), x1y1x2y2=True).item())
                dist = _center_distance(pred_xyxy, gt_xyxy)
                iou_values.append(iou)
                center_distances.append(dist)
        else:
            # Anchor-map path: first output is [B, 9*5, Hf, Wf].
            pred_anchor = output[0]
            pred_anchor = pred_anchor.view(
                pred_anchor.shape[0], 9, 5, pred_anchor.shape[2], pred_anchor.shape[3]
            )

            image_hw = (image_h, image_w)
            grid_hw = (pred_anchor.shape[3], pred_anchor.shape[4])
            _, best_anchor_gi_gj = build_target(gt_bbox, anchors_full, image_hw, grid_hw)

            _, _, _, _, pred_bbox_xyxy, target_bbox_xyxy = eval_iou_acc(
                pred_anchor,
                gt_bbox,
                anchors_full,
                best_anchor_gi_gj[:, 1],
                best_anchor_gi_gj[:, 2],
                image_hw,
                iou_threshold_list=[0.5],
            )

            for b in range(pred_bbox_xyxy.shape[0]):
                pred_xyxy = pred_bbox_xyxy[b].float()
                gt_xyxy = target_bbox_xyxy[b].float()
                iou = float(bbox_iou(pred_xyxy.unsqueeze(0), gt_xyxy.unsqueeze(0), x1y1x2y2=True).item())
                dist = _center_distance(pred_xyxy, gt_xyxy)
                iou_values.append(iou)
                center_distances.append(dist)

    if not iou_values:
        raise RuntimeError("No evaluation samples were processed.")

    iou_arr = np.array(iou_values, dtype=np.float32)
    center_arr = np.array(center_distances, dtype=np.float32)

    metrics = {
        "model_type": model_type,
        "checkpoint": config.checkpoint_path,
        "sat_size": {"height": int(config.sat_size[0]), "width": int(config.sat_size[1])},
        "num_samples": int(len(iou_values)),
        "mean_iou": float(iou_arr.mean()),
        "ratio_iou_gt_0_5": float((iou_arr > 0.5).mean()),
        "mean_center_distance": float(center_arr.mean()),
    }

    os.makedirs(config.output_dir, exist_ok=True)
    out_file = os.path.join(config.output_dir, f"eval_{model_type}_test_split.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("\n=== Evaluation Summary ===")
    print(f"Model: {metrics['model_type']}")
    print(f"Samples: {metrics['num_samples']}")
    print(f"Mean IoU: {metrics['mean_iou']:.4f}")
    print(f"IoU > 0.5 Ratio: {metrics['ratio_iou_gt_0_5']:.4f}")
    print(f"Mean Center Distance (px): {metrics['mean_center_distance']:.4f}")
    print(f"Saved: {out_file}")

    return metrics


def main():
    model_cfg = EVAL_CONFIG["models"]
    selected_models = [
        name for name, cfg in model_cfg.items() if cfg.get("run", False)
    ]

    if not selected_models:
        raise ValueError(
            "No model selected. Set EVAL_CONFIG['models'][name]['run'] = True."
        )

    sat_size_cfg = EVAL_CONFIG.get("sat_size", list(DEFAULT_SAT_SIZE))
    if not isinstance(sat_size_cfg, (list, tuple)) or len(sat_size_cfg) != 2:
        raise ValueError("EVAL_CONFIG['sat_size'] must be [height, width].")

    for model_type in selected_models:
        checkpoint = model_cfg[model_type].get("checkpoint")
        if not checkpoint:
            raise ValueError(
                f"Missing checkpoint for model '{model_type}'. "
                "Set EVAL_CONFIG['models'][model]['checkpoint']."
            )

        cfg = EvalConfig(
            model_type=model_type,
            checkpoint_path=str(checkpoint),
            device=str(EVAL_CONFIG.get("device", DEFAULT_DEVICE)),
            batch_size=int(EVAL_CONFIG.get("batch_size", 2)),
            num_workers=int(EVAL_CONFIG.get("num_workers", 8)),
            sat_size=(int(sat_size_cfg[0]), int(sat_size_cfg[1])),
            test_crop_ratio=float(EVAL_CONFIG.get("test_crop_ratio", 0.5)),
            subset_heights=EVAL_CONFIG.get("subset_heights"),
            subset_angles=EVAL_CONFIG.get("subset_angles"),
            visualize_output_dir=str(EVAL_CONFIG.get("visualize_output_dir", "")),
            output_dir=str(EVAL_CONFIG.get("output_dir", "/data/feihong/univerisity_dev/eval_results")),
        )
        evaluate(cfg)


if __name__ == "__main__":
    main()
