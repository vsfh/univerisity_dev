import argparse
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

REPO_ROOT = Path(__file__).resolve().parents[2]
GROUNDING_ROOT = Path(__file__).resolve().parent
if str(GROUNDING_ROOT) not in sys.path:
    sys.path.insert(0, str(GROUNDING_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from ground_cvos import DetGeoLite, LPNGeoLite, SampleGeoLite, TROGeoLite
from model_ground import Encoder_ground as SigLIPModel
from train_ocg import OCGNetLite
from train_sm import SMGeoLite
from bbox.yolo_utils import bbox_iou, build_target, eval_iou_acc


# --- Configuration ---
MODEL_NAME_SIGLIP = "google/siglip2-base-patch16-224"
CACHE_DIR = "/media/data1/feihong/remote/hf_cache"
DEFAULT_SAT_SIZE = (432, 768)  # (H, W)
DEFAULT_DRONE_SIZE = (256, 256)  # (H, W)
DEFAULT_DEVICE = "cuda:1" if torch.cuda.is_available() else "cpu"
ANCHORS = "37,41, 78,84, 96,215, 129,129, 194,82, 198,179, 246,280, 395,342, 550,573"
BATCH_SIZE = 8
DEFAULT_GROUNDING_RUNS = {
    "det": "/media/data1/feihong/remote/ckpt/ground_det/last.pth",
    "lpn": "/media/data1/feihong/remote/ckpt/ground_lpn/last.pth",
    "sample4geo": "/media/data1/feihong/remote/ckpt/ground_sample/last.pth",
    "smgeo": "/media/data1/feihong/remote/ckpt/ground_sm/last.pth",
    "ocg": "/media/data1/feihong/remote/ckpt/ground_ocg/last.pth",
    "trogeolite": "/media/data1/feihong/remote/ckpt/ground_cvos/last.pth",
    "encoder_test": "/media/data1/feihong/remote/ckpt/model_test_geo_input_ids/last.pth",
}


def _resolve_local_hf_snapshot(model_name: str, cache_dir: str) -> str:
    """Use an existing Hugging Face snapshot path when available to avoid downloads."""
    repo_cache_name = f"models--{model_name.replace('/', '--')}"
    repo_cache_dir = Path(cache_dir) / repo_cache_name
    refs_main = repo_cache_dir / "refs" / "main"
    snapshots_dir = repo_cache_dir / "snapshots"

    if refs_main.exists():
        revision = refs_main.read_text(encoding="utf-8").strip()
        snapshot_dir = snapshots_dir / revision
        if snapshot_dir.is_dir():
            return str(snapshot_dir)

    if snapshots_dir.is_dir():
        snapshots = sorted(path for path in snapshots_dir.iterdir() if path.is_dir())
        if snapshots:
            return str(snapshots[-1])

    return model_name


EVAL_CONFIG = {
    "device": DEFAULT_DEVICE,
    "batch_size": BATCH_SIZE,
    "num_workers": 8,
    "sat_size": [432, 768],
    "test_crop_ratio": 1.0,
    "subset_heights": [150, 200, 250, 300],
    "subset_angles": [0, 45, 90, 135, 180, 225, 270, 315],
    # "subset_heights": [250, 300],
    # "subset_angles": [0, 45, 90, 135],
    "output_dir": "/media/data1/feihong/remote/univerisity_dev/eval_results",
    "visualize_output_dir": "/media/data1/feihong/remote/univerisity_dev/eval_results/visualizations",
    "visualize_num_cases": 10,
    "heatmap_confidence_weight": 0.5,
    "models": {
        "det": {"run": False, "checkpoint": "/media/data1/feihong/remote/ckpt/ground_det/last.pth",},
        "siglip": {"run": True, "checkpoint": "/media/data1/feihong/remote/ckpt/ground_siglip/last.pth"},
        # "encoder_abla": {"run": True, "checkpoint": '/media/data1/feihong/remote/ckpt/0.1_full_model/best_iou.pth'},
        # "encoder_heat": {"run": False, "checkpoint": "/media/data1/feihong/remote/ckpt/0.5_mix_model_heat/last.pth"},
        "lpn": {"run": False, "checkpoint": "/media/data1/feihong/remote/ckpt/ground_lpn/last.pth"},
        "sample4geo": {"run": False, "checkpoint": "/media/data1/feihong/remote/ckpt/ground_sample/last.pth"},
        "smgeo": {"run": False, "checkpoint": "/media/data1/feihong/remote/ckpt/ground_sm/last.pth"},
        "ocg": {"run": False, "checkpoint": "/media/data1/feihong/remote/ckpt/ground_ocg/last.pth"},
        "trogeolite": {"run": False, "checkpoint": "/media/data1/feihong/remote/ckpt/ground_cvos/last.pth"},
        "encoder_test": {"run": False, "checkpoint": "/media/data1/feihong/remote/ckpt/model_test_geo_input_ids/last.pth"},
    },
}


REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_encoder_classes() -> Tuple[Optional[type], type, type]:
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
    return module.Encoder_text_angle, module.Encoder_heat


# EncoderAbla, EncoderHeat = _load_encoder_classes()


def _load_shared_dataset_class():
    """Load root dataset.py explicitly to avoid importing grounding/dataset.py by name."""
    dataset_path = Path(__file__).resolve().parents[2] / "dataset.py"
    spec = importlib.util.spec_from_file_location("shared_dataset_module", str(dataset_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load dataset module from: {dataset_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.ShiftedSatelliteDroneDataset


ShiftedSatelliteDroneDataset = _load_shared_dataset_class()



class TransformProcessorWrapper:
    """Tensor wrapper matching training behavior for cvos-based models."""

    def __init__(self, image_size: Tuple[int, int], normalize: bool = True):
        self.size = {"height": int(image_size[0]), "width": int(image_size[1])}
        self.normalize = bool(normalize)
        self.mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)

    def __call__(self, images, return_tensors="pt"):
        target_h = int(self.size["height"])
        target_w = int(self.size["width"])
        if images.size != (target_w, target_h):
            images = images.resize((target_w, target_h), Image.Resampling.BILINEAR)
        image_np = np.array(images, dtype=np.float32) / 255.0
        pixel_values = torch.from_numpy(image_np).permute(2, 0, 1)
        if self.normalize:
            pixel_values = (pixel_values - self.mean) / self.std
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
    device: str
    batch_size: int
    num_workers: int
    sat_size: Tuple[int, int]
    test_crop_ratio: float
    subset_heights: Optional[List[int]]
    subset_angles: Optional[List[int]]
    visualize_output_dir: str
    output_dir: str
    heatmap_confidence_weight: float = 0.0
    drone_name_filter_path: Optional[str] = None
    visualize_num_cases: int = 10


def _canonical_model_type(model_type: str) -> str:
    name = model_type.lower()
    if name in {"abla", "encoder_abla", "encoder_text_angle"}:
        return "encoder_abla"
    if name in {"heat", "encoder_heat"}:
        return "encoder_heat"
    if name in {"test", "encoder_test"}:
        return "encoder_test"
    if name in {"sample", "samplegeo", "sample4geo"}:
        return "sample4geo"
    if name in {"sm", "smgeo", "smgeolite"}:
        return "smgeo"
    if name in {"wild", "trogeo", "trogeolite"}:
        return "trogeolite"
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


def add_heatmap_to_confidence(
    pred_anchor: torch.Tensor,
    heatmap_logits: Optional[torch.Tensor],
    confidence_weight: float,
) -> torch.Tensor:
    if heatmap_logits is None or confidence_weight <= 0.0:
        return pred_anchor

    if heatmap_logits.shape[-2:] != pred_anchor.shape[-2:]:
        heatmap_logits = torch.nn.functional.interpolate(
            heatmap_logits,
            size=pred_anchor.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

    heatmap_logits = heatmap_logits.detach().to(dtype=pred_anchor.dtype)
    heat_confidence = float(confidence_weight) * heatmap_logits.unsqueeze(1)
    return torch.cat(
        [
            pred_anchor[:, :, :4, :, :],
            pred_anchor[:, :, 4:5, :, :] + heat_confidence,
        ],
        dim=2,
    )


def load_checkpoint_state(checkpoint_path: str) -> Dict[str, torch.Tensor]:
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    state = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    if isinstance(state, dict):
        return {k.replace("module.", ""): v for k, v in state.items()}

    raise TypeError(f"Unsupported checkpoint format: {type(state)!r}")


def _infer_smgeo_kwargs(state: Optional[Dict[str, torch.Tensor]]) -> Dict[str, int]:
    if not state:
        return {}

    kwargs: Dict[str, int] = {}
    patch_weight = state.get("backbone.query_patch.proj.weight")
    if isinstance(patch_weight, torch.Tensor) and patch_weight.ndim == 4:
        kwargs["embed_dim"] = int(patch_weight.shape[0])
        kwargs["patch_size"] = int(patch_weight.shape[-1])

    expert_counts = {
        int(tensor.shape[0])
        for key, tensor in state.items()
        if key.endswith("ffn.router.weight")
        and isinstance(tensor, torch.Tensor)
        and tensor.ndim == 2
    }
    if len(expert_counts) == 1:
        kwargs["num_experts"] = expert_counts.pop()

    return kwargs


def build_model(
    model_type: str,
    checkpoint_state: Optional[Dict[str, torch.Tensor]] = None,
) -> torch.nn.Module:
    model_type = _canonical_model_type(model_type)
    if model_type == "siglip":
        return SigLIPModel()
    if model_type == "encoder_abla":
        EncoderAbla, _, _ = _load_encoder_classes()
        if EncoderAbla is None:
            raise AttributeError("model.py does not export Encoder_text_angle required by encoder_abla.")
        return EncoderAbla(
            model_name=_resolve_local_hf_snapshot(MODEL_NAME_SIGLIP, CACHE_DIR),
            proj_dim=768,
            usesg=True,
            useap=True,
        )
    if model_type == "encoder_heat":
        _, EncoderHeat, _ = _load_encoder_classes()
        return EncoderHeat(
            model_name=_resolve_local_hf_snapshot(MODEL_NAME_SIGLIP2, CACHE_DIR),
            proj_dim=768,
            usesg=True,
            useap=True,
        )
    if model_type == "encoder_test":
        _, _, EncoderTest = _load_encoder_classes()
        return EncoderTest(
            model_name=_resolve_local_hf_snapshot(MODEL_NAME_SIGLIP2, CACHE_DIR),
            proj_dim=768,
            usesg=True,
            useap=True,
        )
    if model_type == "det":
        return DetGeoLite(emb_size=512)
    if model_type == "lpn":
        return LPNGeoLite()
    if model_type == "sample4geo":
        return SampleGeoGrounding(emb_size=1024, pretrained=False)
    if model_type == "smgeo":
        kwargs = _infer_smgeo_kwargs(checkpoint_state)
        if kwargs:
            print(f"Inferred SMGeoLite checkpoint config: {kwargs}")
        return SMGeoLite(**kwargs)
    if model_type == "ocg":
        return OCGNetLite(pretrained_backbone=False)
    if model_type == "trogeolite":
        return TROGeoLite()
    raise ValueError(f"Unknown model type: {model_type}")


def load_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: str,
    checkpoint_state: Optional[Dict[str, torch.Tensor]] = None,
) -> None:
    state = checkpoint_state or load_checkpoint_state(checkpoint_path)
    model_state = model.state_dict()
    compatible_state: Dict[str, torch.Tensor] = {}
    skipped_shapes: List[Tuple[str, Tuple[int, ...], Tuple[int, ...]]] = []

    for key, value in state.items():
        if (
            key in model_state
            and isinstance(value, torch.Tensor)
            and model_state[key].shape != value.shape
        ):
            skipped_shapes.append((key, tuple(value.shape), tuple(model_state[key].shape)))
            continue
        compatible_state[key] = value

    model.load_state_dict(compatible_state, strict=False)
    if skipped_shapes:
        print(f"Skipped {len(skipped_shapes)} checkpoint tensors with incompatible shapes:")
        for key, checkpoint_shape, model_shape in skipped_shapes[:10]:
            print(f"  {key}: checkpoint={checkpoint_shape}, model={model_shape}")
        if len(skipped_shapes) > 10:
            print(f"  ... {len(skipped_shapes) - 10} more")


def _load_drone_name_filter(txt_path: Optional[str]) -> Optional[Set[str]]:
    if not txt_path:
        return None
    if not os.path.exists(txt_path):
        raise FileNotFoundError(f"Drone-name filter txt not found: {txt_path}")

    allowed: Set[str] = set()
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            name = line.strip()
            if not name:
                continue
            path_name = Path(name).name
            allowed.add(name)
            allowed.add(path_name)
            allowed.add(Path(path_name).stem)
    return allowed


def _filter_dataset_by_drone_names(dataset: Any, allowed_names: Optional[Set[str]]) -> None:
    if not allowed_names:
        return

    filtered_samples = []
    for sample in dataset.samples:
        drone_name = Path(str(sample.get("drone_path", ""))).name
        drone_stem = Path(drone_name).stem
        if drone_name in allowed_names or drone_stem in allowed_names:
            filtered_samples.append(sample)

    if not filtered_samples:
        raise ValueError("No grounding samples matched the provided drone-name filter.")

    original_count = len(dataset.samples)
    dataset.samples = filtered_samples
    print(
        f"Filtered grounding dataset by drone names: "
        f"{len(dataset.samples)} / {original_count} samples kept."
    )


def create_test_loader(config: EvalConfig) -> DataLoader:
    model_type = _canonical_model_type(config.model_type)
    # Use SigLIP processor only for siglip model; others use simple tensor wrapper.
    if model_type in {"encoder_abla", "encoder_heat"}:
        processor = AutoImageProcessor.from_pretrained(MODEL_NAME_SIGLIP, cache_dir=CACHE_DIR)
        processor_sat = AutoImageProcessor.from_pretrained(
            hf_model_path,
            cache_dir=CACHE_DIR,
            size={"height": int(config.sat_size[0]), "width": int(config.sat_size[1])},
        )
        tokenizer = AutoTokenizer.from_pretrained(hf_model_path, cache_dir=CACHE_DIR)
    else:
        processor = TransformProcessorWrapper(DEFAULT_DRONE_SIZE)
        processor_sat = TransformProcessorWrapper(config.sat_size)
        tokenizer = DummyTokenizer()

    test_dataset = ShiftedSatelliteDroneDataset(
        processor=processor,
        processor_sat=processor_sat,
        tokenizer=tokenizer,
        split="test",
        sat_target_size=config.sat_size,
        test_crop_ratio=config.test_crop_ratio,
        subset_heights=config.subset_heights,
        subset_angles=config.subset_angles,
    )
    _filter_dataset_by_drone_names(
        test_dataset,
        _load_drone_name_filter(config.drone_name_filter_path),
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


def _decode_anchor_free_bbox(
    heatmap_logits: torch.Tensor,
    bbox_raw: torch.Tensor,
    image_wh: Tuple[int, int],
) -> torch.Tensor:
    if heatmap_logits.ndim != 4 or heatmap_logits.shape[1] != 1:
        raise ValueError(f"Expected heatmap logits shape (B, 1, H, W), got {tuple(heatmap_logits.shape)}.")
    if bbox_raw.ndim != 4 or bbox_raw.shape[1] != 4:
        raise ValueError(f"Expected bbox raw shape (B, 4, H, W), got {tuple(bbox_raw.shape)}.")

    batch_size, _, feat_h, feat_w = heatmap_logits.shape
    image_w, image_h = float(image_wh[0]), float(image_wh[1])
    heat_prob = torch.sigmoid(heatmap_logits[:, 0])
    flat_idx = heat_prob.flatten(1).argmax(dim=1)
    gj = torch.div(flat_idx, feat_w, rounding_mode="floor")
    gi = flat_idx % feat_w
    batch_idx = torch.arange(batch_size, device=heatmap_logits.device)

    selected_bbox = bbox_raw[batch_idx, :, gj, gi]
    offset = torch.sigmoid(selected_bbox[:, :2])
    wh = torch.sigmoid(selected_bbox[:, 2:]).clamp_min(1e-4)
    cx = (gi.to(dtype=bbox_raw.dtype) + offset[:, 0]) / float(feat_w) * image_w
    cy = (gj.to(dtype=bbox_raw.dtype) + offset[:, 1]) / float(feat_h) * image_h
    bw = wh[:, 0] * image_w
    bh = wh[:, 1] * image_h

    x1 = (cx - 0.5 * bw).clamp(0.0, image_w)
    y1 = (cy - 0.5 * bh).clamp(0.0, image_h)
    x2 = (cx + 0.5 * bw).clamp(0.0, image_w)
    y2 = (cy + 0.5 * bh).clamp(0.0, image_h)
    return torch.stack([x1, y1, x2, y2], dim=1)


def _is_direct_bbox_output(output: Any) -> bool:
    return isinstance(output, torch.Tensor) and output.ndim <= 2 and output.shape[-1] == 4


def _center_distance(pred_xyxy: torch.Tensor, gt_xyxy: torch.Tensor) -> float:
    pred_cx = 0.5 * (pred_xyxy[0] + pred_xyxy[2])
    pred_cy = 0.5 * (pred_xyxy[1] + pred_xyxy[3])
    gt_cx = 0.5 * (gt_xyxy[0] + gt_xyxy[2])
    gt_cy = 0.5 * (gt_xyxy[1] + gt_xyxy[3])
    return float(torch.sqrt((pred_cx - gt_cx) ** 2 + (pred_cy - gt_cy) ** 2).item())


def _summarize_detection_metrics(iou_values: List[float], center_distances: List[float]) -> Dict[str, float]:
    if not iou_values or not center_distances:
        raise ValueError("Expected non-empty metric lists.")

    iou_arr = np.array(iou_values, dtype=np.float32)
    center_arr = np.array(center_distances, dtype=np.float32)
    return {
        "num_samples": int(len(iou_values)),
        "mean_iou": float(iou_arr.mean()),
        "ratio_iou_gt_0_5": float((iou_arr > 0.5).mean()),
        "mean_center_distance": float(center_arr.mean()),
    }


def _tensor_chw_to_pil(image_tensor: torch.Tensor) -> Image.Image:
    """Convert CHW tensor to RGB PIL image with robust range handling."""
    image = image_tensor.detach().float().cpu()
    if image.ndim != 3:
        raise ValueError(f"Expected CHW tensor, got shape: {tuple(image.shape)}")

    if image.shape[0] == 3 and (image.min().item() < -0.1 or image.max().item() > 1.1):
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=image.dtype).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=image.dtype).view(3, 1, 1)
        image = image * std + mean

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


def visualize_batch_with_prediction(
    batch: Dict[str, Any],
    pred_bbox_batch: torch.Tensor,
    output_dir: str = "runs/visualizations/eval_batch_pred",
) -> List[str]:
    """Save one image per sample with query, GT bbox, and predicted bbox."""
    query_batch = batch["target_pixel_values"]
    satellite_batch = batch["search_pixel_values"]
    gt_bbox_batch = batch["bbox"]

    num_samples = int(
        min(query_batch.shape[0], satellite_batch.shape[0], gt_bbox_batch.shape[0], pred_bbox_batch.shape[0])
    )
    if num_samples <= 0:
        raise ValueError("No samples available for visualization.")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    saved_files: List[str] = []
    for idx in range(num_samples):
        query_img = _tensor_chw_to_pil(query_batch[idx]).convert("RGB")
        sat_img = _tensor_chw_to_pil(satellite_batch[idx]).convert("RGB")

        gt_xyxy = [float(v) for v in gt_bbox_batch[idx].detach().float().cpu().tolist()]
        pred_xyxy = [float(v) for v in pred_bbox_batch[idx].detach().float().cpu().tolist()]

        gt_xyxy[0] = max(0.0, min(float(sat_img.width - 1), gt_xyxy[0]))
        gt_xyxy[1] = max(0.0, min(float(sat_img.height - 1), gt_xyxy[1]))
        gt_xyxy[2] = max(0.0, min(float(sat_img.width - 1), gt_xyxy[2]))
        gt_xyxy[3] = max(0.0, min(float(sat_img.height - 1), gt_xyxy[3]))

        pred_xyxy[0] = max(0.0, min(float(sat_img.width - 1), pred_xyxy[0]))
        pred_xyxy[1] = max(0.0, min(float(sat_img.height - 1), pred_xyxy[1]))
        pred_xyxy[2] = max(0.0, min(float(sat_img.width - 1), pred_xyxy[2]))
        pred_xyxy[3] = max(0.0, min(float(sat_img.height - 1), pred_xyxy[3]))

        draw = ImageDraw.Draw(sat_img)
        draw.rectangle(tuple(gt_xyxy), outline=(0, 255, 0), width=3)
        draw.rectangle(tuple(pred_xyxy), outline=(255, 0, 0), width=3)

        target_h = 320
        q_w = max(1, int(round(query_img.width * target_h / max(1, query_img.height))))
        s_w = max(1, int(round(sat_img.width * target_h / max(1, sat_img.height))))
        query_img = query_img.resize((q_w, target_h), Image.Resampling.BILINEAR)
        sat_img = sat_img.resize((s_w, target_h), Image.Resampling.BILINEAR)

        pair = Image.new("RGB", (q_w + s_w + 12, target_h), color=(15, 15, 15))
        pair.paste(query_img, (0, 0))
        pair.paste(sat_img, (q_w + 12, 0))

        draw = ImageDraw.Draw(pair)
        draw.text((12, 8), "GT: green | Pred: red", fill=(240, 240, 240))

        save_path = out_dir / f"sample_{idx:03d}.png"
        pair.save(save_path)
        saved_files.append(str(save_path))

    return saved_files


def _clamp_bbox_list(
    bbox: Sequence[float],
    image_w: int,
    image_h: int,
) -> List[float]:
    x1, y1, x2, y2 = [float(v) for v in bbox]
    x1 = max(0.0, min(float(image_w - 1), x1))
    y1 = max(0.0, min(float(image_h - 1), y1))
    x2 = max(0.0, min(float(image_w - 1), x2))
    y2 = max(0.0, min(float(image_h - 1), y2))
    if x2 <= x1:
        x2 = min(float(image_w - 1), x1 + 1.0)
    if y2 <= y1:
        y2 = min(float(image_h - 1), y1 + 1.0)
    return [x1, y1, x2, y2]


def _resize_to_height_keep_aspect(image: Image.Image, target_h: int) -> Image.Image:
    if image.height == target_h:
        return image
    target_w = max(1, int(round(image.width * target_h / max(1, image.height))))
    return image.resize((target_w, target_h), Image.Resampling.BILINEAR)


def _path_stem(value: Any) -> str:
    return Path(str(value)).stem


def _safe_filename(text: str) -> str:
    keep = []
    for char in str(text):
        if char.isalnum() or char in {"-", "_", "."}:
            keep.append(char)
        else:
            keep.append("_")
    return "".join(keep).strip("_") or "sample"


def save_composite_prediction_visualizations(
    batch: Dict[str, Any],
    pred_bbox_batch: torch.Tensor,
    output_dir: str,
    model_tag: str,
    start_index: int,
    max_cases: int,
) -> List[str]:
    """Save per-case panels: drone, satellite with GT/Pred boxes, and GT crop."""
    if not output_dir or max_cases <= 0 or start_index >= max_cases:
        return []

    query_batch = batch["target_pixel_values"]
    satellite_batch = batch["search_pixel_values"]
    gt_bbox_batch = batch["bbox"]

    num_samples = int(
        min(query_batch.shape[0], satellite_batch.shape[0], gt_bbox_batch.shape[0], pred_bbox_batch.shape[0])
    )
    remaining = max(0, int(max_cases) - int(start_index))
    num_samples = min(num_samples, remaining)
    if num_samples <= 0:
        return []

    out_dir = Path(output_dir) / _safe_filename(model_tag)
    out_dir.mkdir(parents=True, exist_ok=True)

    saved_files: List[str] = []
    panel_h = 360
    padding = 12
    label_h = 26

    for idx in range(num_samples):
        query_img = _tensor_chw_to_pil(query_batch[idx]).convert("RGB")
        sat_img = _tensor_chw_to_pil(satellite_batch[idx]).convert("RGB")

        gt_xyxy = _clamp_bbox_list(
            gt_bbox_batch[idx].detach().float().cpu().tolist(),
            sat_img.width,
            sat_img.height,
        )
        pred_xyxy = _clamp_bbox_list(
            pred_bbox_batch[idx].detach().float().cpu().tolist(),
            sat_img.width,
            sat_img.height,
        )

        crop_box = tuple(int(round(v)) for v in gt_xyxy)
        gt_crop = sat_img.crop(crop_box)
        if gt_crop.width <= 1 or gt_crop.height <= 1:
            gt_crop = Image.new("RGB", (panel_h, panel_h), color=(20, 20, 20))

        sat_draw = sat_img.copy()
        draw = ImageDraw.Draw(sat_draw)
        draw.rectangle(tuple(gt_xyxy), outline=(0, 255, 0), width=4)
        draw.rectangle(tuple(pred_xyxy), outline=(255, 0, 0), width=4)

        query_panel = _resize_to_height_keep_aspect(query_img, panel_h)
        sat_panel = _resize_to_height_keep_aspect(sat_draw, panel_h)
        crop_panel = _resize_to_height_keep_aspect(gt_crop, panel_h)

        panels = [
            ("drone query", query_panel),
            ("satellite GT green / Pred red", sat_panel),
            ("GT bbox crop", crop_panel),
        ]
        canvas_w = sum(panel.width for _, panel in panels) + padding * (len(panels) + 1)
        canvas_h = panel_h + label_h + padding * 2
        canvas = Image.new("RGB", (canvas_w, canvas_h), color=(18, 18, 18))
        draw_canvas = ImageDraw.Draw(canvas)

        x = padding
        for label, panel in panels:
            draw_canvas.text((x, padding), label, fill=(235, 235, 235))
            canvas.paste(panel, (x, padding + label_h))
            x += panel.width + padding

        sample_idx = start_index + idx
        sat_id = _path_stem(batch["satellite_path"][idx])
        drone_id = _path_stem(batch["drone_path"][idx])
        height = int(batch["height"][idx].item()) if hasattr(batch["height"][idx], "item") else int(batch["height"][idx])
        angle = int(batch["angle"][idx].item()) if hasattr(batch["angle"][idx], "item") else int(batch["angle"][idx])
        save_name = (
            f"{sample_idx:03d}_{_safe_filename(model_tag)}_sat{_safe_filename(sat_id)}_"
            f"drone{_safe_filename(drone_id)}_h{height}_a{angle}.png"
        )
        save_path = out_dir / save_name
        canvas.save(save_path)
        saved_files.append(str(save_path))

    return saved_files


def evaluate(config: EvalConfig) -> Dict[str, Any]:
    model_type = _canonical_model_type(config.model_type)

    checkpoint_state = load_checkpoint_state(config.checkpoint_path)
    model = build_model(model_type, checkpoint_state=checkpoint_state).to(config.device)
    load_checkpoint(model, config.checkpoint_path, checkpoint_state=checkpoint_state)
    model.eval()

    loader = create_test_loader(config)
    anchors_full = parse_anchors(config.device)

    iou_values: List[float] = []
    center_distances: List[float] = []
    per_subset_values: Dict[Tuple[int, int], Dict[str, List[float]]] = {}
    per_angle_values: Dict[int, Dict[str, List[float]]] = {}
    visualized_count = 0
    visualize_num_cases = max(0, int(config.visualize_num_cases))
    checkpoint_name = Path(str(config.checkpoint_path)).resolve().parent.name
    model_tag = f"{model_type}_{checkpoint_name}"
    checkpoint_for_flags = str(config.checkpoint_path)
    for batch in tqdm(loader, desc=f"Evaluating {model_type}"):
        query_imgs = batch["target_pixel_values"].to(config.device, non_blocking=True)
        search_imgs = batch["search_pixel_values"].to(config.device, non_blocking=True)
        gt_bbox = batch["bbox"].to(config.device)
        input_ids = batch["input_ids"].to(config.device, non_blocking=True) if "wo_text" not in checkpoint_for_flags else None
        if "wo_angle" not in checkpoint_for_flags:
            angles = batch["angle"].to(config.device, non_blocking=True).float()
            angles_rad = torch.deg2rad(angles)
            heights = batch["height"].to(config.device, non_blocking=True).float()
            geo = torch.cat(
                [
                    torch.cos(angles_rad)[..., None],
                    torch.sin(angles_rad)[..., None],
                    heights[..., None] / 300.0,
                ],
                dim=1,
            )
        else:
            geo = None
        batch_heights = batch["height"]
        batch_angles = batch["angle"]

        image_h = int(search_imgs.shape[-2])
        image_w = int(search_imgs.shape[-1])

        with torch.no_grad():
            if model_type == "sample4geo":
                sample_outputs = model(query_imgs, search_imgs)
                output = _decode_anchor_free_bbox(
                    sample_outputs["heatmap_logits"],
                    sample_outputs["bbox_raw"],
                    image_wh=(image_w, image_h),
                )
            elif hasattr(model, "bbox_forward"):
                try:
                    output = model.bbox_forward(query_imgs, search_imgs, angle=geo)
                except TypeError:
                    output = model.bbox_forward(query_imgs, search_imgs)
            elif model_type in {"encoder_abla", "encoder_heat", "encoder_test"}:
                attention_mask = None
                if model_type == "encoder_test" and input_ids is not None and "attention_mask" in batch:
                    attention_mask = batch["attention_mask"].to(config.device, non_blocking=True)
                output = model(
                    query_imgs,
                    search_imgs,
                    input_ids=input_ids,
                    angle=geo,
                    attention_mask=attention_mask,
                )
            else:
                raise AttributeError(
                    f"Model '{model_type}' has no bbox_forward and no fallback path."
                )

        if _is_direct_bbox_output(output):
            # Direct bbox regression path.
            pred_bbox_batch = output
            if pred_bbox_batch.ndim == 1:
                pred_bbox_batch = pred_bbox_batch.unsqueeze(0)

            if config.visualize_output_dir and visualized_count < visualize_num_cases:
                pred_vis = [
                    _to_xyxy_pixels(pred_bbox_batch[b], image_h, image_w).detach().cpu()
                    for b in range(pred_bbox_batch.shape[0])
                ]
                saved_files = save_composite_prediction_visualizations(
                    batch,
                    torch.stack(pred_vis, dim=0),
                    model_tag=model_tag,
                    output_dir=config.visualize_output_dir,
                    start_index=visualized_count,
                    max_cases=visualize_num_cases,
                )
                visualized_count += len(saved_files)
                if saved_files:
                    print(
                        f"Saved {len(saved_files)} {model_type} prediction visualizations "
                        f"to: {Path(config.visualize_output_dir) / _safe_filename(model_tag)}"
                    )

            for b in range(pred_bbox_batch.shape[0]):
                pred_xyxy = _to_xyxy_pixels(pred_bbox_batch[b], image_h, image_w)
                gt_xyxy = gt_bbox[b].float()
                iou = float(bbox_iou(pred_xyxy.unsqueeze(0), gt_xyxy.unsqueeze(0), x1y1x2y2=True).item())
                dist = _center_distance(pred_xyxy, gt_xyxy)

                height = int(batch_heights[b].item())
                angle = int(batch_angles[b].item())
                subset_bucket = per_subset_values.setdefault((height, angle), {"iou": [], "center_distance": []})
                subset_bucket["iou"].append(iou)
                subset_bucket["center_distance"].append(dist)
                angle_bucket = per_angle_values.setdefault(angle, {"iou": [], "center_distance": []})
                angle_bucket["iou"].append(iou)
                angle_bucket["center_distance"].append(dist)

                iou_values.append(iou)
                center_distances.append(dist)
        else:
            # Anchor-map path: first output is [B, 9*5, Hf, Wf].
            pred_anchor = output if isinstance(output, torch.Tensor) else output[0]
            heatmap_logits = output[6] if model_type in {"encoder_heat", "encoder_test"} and len(output) > 6 else None
            pred_anchor = pred_anchor.view(
                pred_anchor.shape[0], 9, 5, pred_anchor.shape[2], pred_anchor.shape[3]
            )
            pred_anchor = add_heatmap_to_confidence(
                pred_anchor,
                heatmap_logits,
                config.heatmap_confidence_weight,
            )

            image_wh = (image_w, image_h)
            grid_wh = (pred_anchor.shape[4], pred_anchor.shape[3])
            _, best_anchor_gi_gj = build_target(gt_bbox, anchors_full, image_wh, grid_wh)

            _, _, _, _, pred_bbox_xyxy, target_bbox_xyxy = eval_iou_acc(
                pred_anchor,
                gt_bbox,
                anchors_full,
                best_anchor_gi_gj[:, 1],
                best_anchor_gi_gj[:, 2],
                image_wh,
                iou_threshold_list=[0.5],
            )

            if config.visualize_output_dir and visualized_count < visualize_num_cases:
                saved_files = save_composite_prediction_visualizations(
                    batch,
                    pred_bbox_xyxy.detach().cpu(),
                    model_tag=model_tag,
                    output_dir=config.visualize_output_dir,
                    start_index=visualized_count,
                    max_cases=visualize_num_cases,
                )
                visualized_count += len(saved_files)
                if saved_files:
                    print(
                        f"Saved {len(saved_files)} {model_type} prediction visualizations "
                        f"to: {Path(config.visualize_output_dir) / _safe_filename(model_tag)}"
                    )

            for b in range(pred_bbox_xyxy.shape[0]):
                pred_xyxy = pred_bbox_xyxy[b].float()
                gt_xyxy = target_bbox_xyxy[b].float()
                iou = float(bbox_iou(pred_xyxy.unsqueeze(0), gt_xyxy.unsqueeze(0), x1y1x2y2=True).item())
                dist = _center_distance(pred_xyxy, gt_xyxy)

                height = int(batch_heights[b].item())
                angle = int(batch_angles[b].item())
                subset_bucket = per_subset_values.setdefault((height, angle), {"iou": [], "center_distance": []})
                subset_bucket["iou"].append(iou)
                subset_bucket["center_distance"].append(dist)
                angle_bucket = per_angle_values.setdefault(angle, {"iou": [], "center_distance": []})
                angle_bucket["iou"].append(iou)
                angle_bucket["center_distance"].append(dist)

                iou_values.append(iou)
                center_distances.append(dist)

    if not iou_values:
        raise RuntimeError("No evaluation samples were processed.")

    overall_metrics = _summarize_detection_metrics(iou_values, center_distances)

    per_subset_metrics: List[Dict[str, Any]] = []
    for (height, angle) in sorted(per_subset_values.keys()):
        subset_stats = _summarize_detection_metrics(
            per_subset_values[(height, angle)]["iou"],
            per_subset_values[(height, angle)]["center_distance"],
        )
        per_subset_metrics.append(
            {
                "height": int(height),
                "angle": int(angle),
                **subset_stats,
            }
        )

    per_angle_metrics: List[Dict[str, Any]] = []
    for angle in sorted(per_angle_values.keys()):
        angle_stats = _summarize_detection_metrics(
            per_angle_values[angle]["iou"],
            per_angle_values[angle]["center_distance"],
        )
        per_angle_metrics.append(
            {
                "angle": int(angle),
                **angle_stats,
            }
        )

    metrics = {
        "model_type": model_type,
        "checkpoint": config.checkpoint_path,
        "sat_size": {"height": int(config.sat_size[0]), "width": int(config.sat_size[1])},
        "num_samples": int(overall_metrics["num_samples"]),
        "mean_iou": float(overall_metrics["mean_iou"]),
        "ratio_iou_gt_0_5": float(overall_metrics["ratio_iou_gt_0_5"]),
        "mean_center_distance": float(overall_metrics["mean_center_distance"]),
        "overall": overall_metrics,
        "per_subset": per_subset_metrics,
        "per_angle": per_angle_metrics,
    }

    os.makedirs(config.output_dir, exist_ok=True)
    name = Path(config.checkpoint_path).parent.name
    out_file = os.path.join(config.output_dir, f"eval_{name}_test_split.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("\n=== Evaluation Summary ===")
    print(f"Model: {metrics['model_type']}")
    print(f"Samples: {metrics['num_samples']}")
    print(f"Mean IoU: {metrics['mean_iou']:.4f}")
    print(f"IoU > 0.5 Ratio: {metrics['ratio_iou_gt_0_5']:.4f}")
    print(f"Mean Center Distance (px): {metrics['mean_center_distance']:.4f}")
    if per_subset_metrics:
        print("Per-subset metrics (height, angle):")
        for item in per_subset_metrics:
            print(
                f"  h={item['height']:>3d}, a={item['angle']:>3d}, n={item['num_samples']:>4d}, "
                f"mean_iou={item['mean_iou']:.4f}, iou>0.5={item['ratio_iou_gt_0_5']:.4f}, "
                f"mean_center={item['mean_center_distance']:.4f}"
            )
    if per_angle_metrics:
        print("Per-angle metrics:")
        for item in per_angle_metrics:
            print(
                f"  a={item['angle']:>3d}, n={item['num_samples']:>4d}, "
                f"mean_iou={item['mean_iou']:.4f}, iou>0.5={item['ratio_iou_gt_0_5']:.4f}, "
                f"mean_center={item['mean_center_distance']:.4f}"
            )
    print(f"Saved: {out_file}")

    return metrics


def eval_encoder_text_angle_checkpoints(
    device: Optional[str] = None,
    batch_size: int = BATCH_SIZE,
    num_workers: int = 8,
    sat_size: Tuple[int, int] = DEFAULT_SAT_SIZE,
    test_crop_ratio: float = 1.0,
    subset_heights: Optional[Sequence[int]] = None,
    subset_angles: Optional[Sequence[int]] = None,
    visualize_output_dir: str = "",
    output_dir: str = "/media/data1/feihong/remote/univerisity_dev/eval_results",
    output_json_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Evaluate the same Encoder_text_angle model across multiple checkpoints."""
    checkpoint_paths = [
        "/media/data1/feihong/remote/ckpt/0.5_mix_model_heat_2/last.pth",
		# "/media/data1/feihong/remote/ckpt/0.5_mix_model_wo_angle/last.pth",
		# "/media/data1/feihong/remote/ckpt/0.5_mix_model_wo_text/last.pth",
		# "/media/data1/feihong/remote/ckpt/0.9999_mix_model_bbox_only/last.pth",
    ]
    if not checkpoint_paths:
        raise ValueError("checkpoint_paths must not be empty.")

    if device is None:
        device = DEFAULT_DEVICE

    subset_height_list = sorted(_normalize_subset(subset_heights, EVAL_CONFIG.get("subset_heights", None)))
    subset_angle_list = sorted(_normalize_subset(subset_angles, EVAL_CONFIG.get("subset_angles", None)))

    results: List[Dict[str, Any]] = []
    for checkpoint_path in checkpoint_paths:
        checkpoint_name = Path(str(checkpoint_path)).resolve().parent.name
        per_checkpoint_output_dir = os.path.join(output_dir, f"encoder_text_angle_{checkpoint_name}")

        cfg = EvalConfig(
            model_type="encoder_heat",
            checkpoint_path=str(checkpoint_path),
            device=str(device),
            batch_size=int(batch_size),
            num_workers=int(num_workers),
            sat_size=(int(sat_size[0]), int(sat_size[1])),
            test_crop_ratio=float(test_crop_ratio),
            subset_heights=[int(v) for v in subset_height_list],
            subset_angles=[int(v) for v in subset_angle_list],
            visualize_output_dir=str(visualize_output_dir),
            output_dir=str(per_checkpoint_output_dir),
            heatmap_confidence_weight=float(EVAL_CONFIG.get("heatmap_confidence_weight", 0.5)),
            visualize_num_cases=int(EVAL_CONFIG.get("visualize_num_cases", 10)),
        )
        metrics = evaluate(cfg)
        results.append(
            {
                "model_type": "encoder_heat",
                "checkpoint": str(checkpoint_path),
                "metrics": metrics,
            }
        )

    if output_json_path:
        output_json_dir = os.path.dirname(output_json_path)
        if output_json_dir:
            os.makedirs(output_json_dir, exist_ok=True)

        payload = {
            "model_type": "encoder_heat",
            "subset_heights": [int(v) for v in subset_height_list],
            "subset_angles": [int(v) for v in subset_angle_list],
            "num_checkpoints": len(results),
            "results": results,
        }

        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"Saved Encoder_heat checkpoint sweep to: {output_json_path}")

    return results


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
            test_crop_ratio=float(EVAL_CONFIG.get("test_crop_ratio", 1.0)),
            subset_heights=EVAL_CONFIG.get("subset_heights"),
            subset_angles=EVAL_CONFIG.get("subset_angles"),
            visualize_output_dir=str(EVAL_CONFIG.get("visualize_output_dir", "")),
            output_dir=str(EVAL_CONFIG.get("output_dir", "/media/data1/feihong/remote/univerisity_dev/eval_results")),
            heatmap_confidence_weight=float(EVAL_CONFIG.get("heatmap_confidence_weight", 0.5)),
            visualize_num_cases=int(EVAL_CONFIG.get("visualize_num_cases", 10)),
        )
        evaluate(cfg)


def main_with_drone_name_filter(drone_name_filter_path: str):
    """Run grounding evaluation like main(), but only on drone images listed in txt."""
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
            test_crop_ratio=float(EVAL_CONFIG.get("test_crop_ratio", 1.0)),
            subset_heights=EVAL_CONFIG.get("subset_heights"),
            subset_angles=EVAL_CONFIG.get("subset_angles"),
            visualize_output_dir=str(EVAL_CONFIG.get("visualize_output_dir", "")),
            output_dir=str(EVAL_CONFIG.get("output_dir", "/media/data1/feihong/remote/univerisity_dev/eval_results")),
            heatmap_confidence_weight=float(EVAL_CONFIG.get("heatmap_confidence_weight", 0.5)),
            drone_name_filter_path=str(drone_name_filter_path),
            visualize_num_cases=int(EVAL_CONFIG.get("visualize_num_cases", 10)),
        )
        evaluate(cfg)


def _parse_optional_int_list(values: Optional[List[int]]) -> Optional[List[int]]:
    if values is None:
        return None
    if len(values) == 0:
        return None
    return [int(value) for value in values]


def _build_eval_config(
    model_type: str,
    checkpoint_path: str,
    args: argparse.Namespace,
) -> EvalConfig:
    return EvalConfig(
        model_type=model_type,
        checkpoint_path=checkpoint_path,
        device=str(args.device),
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        sat_size=(int(args.sat_size[0]), int(args.sat_size[1])),
        test_crop_ratio=float(args.test_crop_ratio),
        subset_heights=_parse_optional_int_list(args.subset_heights),
        subset_angles=_parse_optional_int_list(args.subset_angles),
        visualize_output_dir=str(args.visualize_output_dir or ""),
        output_dir=str(args.output_dir),
        heatmap_confidence_weight=float(args.heatmap_confidence_weight),
        drone_name_filter_path=args.drone_name_filter,
        visualize_num_cases=int(args.visualize_num_cases),
    )


def _run_cli(args: argparse.Namespace) -> None:
    selected_models = [_canonical_model_type(model) for model in args.models]
    if args.all_grounding:
        selected_models = list(DEFAULT_GROUNDING_RUNS.keys())

    if not selected_models:
        raise ValueError("No models selected. Use --all-grounding or --models ...")
    if args.checkpoint is not None and len(selected_models) != 1:
        raise ValueError("--checkpoint can only be used with exactly one --models entry.")

    for model_type in selected_models:
        checkpoint = args.checkpoint
        if checkpoint is None:
            checkpoint = DEFAULT_GROUNDING_RUNS.get(model_type)
        if not checkpoint:
            raise ValueError(f"No default checkpoint is registered for model '{model_type}'.")
        if not os.path.exists(checkpoint):
            print(f"Skipping {model_type}: checkpoint not found: {checkpoint}")
            continue

        cfg = _build_eval_config(
            model_type=model_type,
            checkpoint_path=str(checkpoint),
            args=args,
        )
        evaluate(cfg)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate grounding train_* checkpoints.")
    parser.add_argument(
        "--all-grounding",
        action="store_true",
        help="Evaluate all trained grounding models with default checkpoint paths.",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=[],
        choices=[
            "det",
            "lpn",
            "sample4geo",
            "sample",
            "smgeo",
            "sm",
            "ocg",
            "trogeolite",
            "wild",
            "encoder_test",
            "test",
            "encoder_heat",
            "heat",
        ],
        help="Model types to evaluate. Ignored when --all-grounding is set.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Explicit checkpoint path for a single model run.",
    )
    parser.add_argument("--device", type=str, default=str(EVAL_CONFIG.get("device", DEFAULT_DEVICE)))
    parser.add_argument("--batch-size", type=int, default=int(EVAL_CONFIG.get("batch_size", BATCH_SIZE)))
    parser.add_argument("--num-workers", type=int, default=int(EVAL_CONFIG.get("num_workers", 8)))
    parser.add_argument(
        "--sat-size",
        type=int,
        nargs=2,
        metavar=("HEIGHT", "WIDTH"),
        default=list(EVAL_CONFIG.get("sat_size", list(DEFAULT_SAT_SIZE))),
        help="Satellite image size as HEIGHT WIDTH.",
    )
    parser.add_argument("--test-crop-ratio", type=float, default=float(EVAL_CONFIG.get("test_crop_ratio", 1.0)))
    parser.add_argument("--subset-heights", type=int, nargs="*", default=EVAL_CONFIG.get("subset_heights"))
    parser.add_argument("--subset-angles", type=int, nargs="*", default=EVAL_CONFIG.get("subset_angles"))
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(EVAL_CONFIG.get("output_dir", "/media/data1/feihong/remote/univerisity_dev/eval_results")),
    )
    parser.add_argument("--visualize-output-dir", type=str, default=str(EVAL_CONFIG.get("visualize_output_dir", "")))
    parser.add_argument(
        "--visualize-num-cases",
        type=int,
        default=int(EVAL_CONFIG.get("visualize_num_cases", 10)),
        help="Number of composite prediction visualizations to save per evaluated model.",
    )
    parser.add_argument(
        "--heatmap-confidence-weight",
        type=float,
        default=float(EVAL_CONFIG.get("heatmap_confidence_weight", 0.5)),
    )
    parser.add_argument("--drone-name-filter", type=str, default=None)
    parser.add_argument(
        "--use-config",
        action="store_true",
        help="Use the legacy EVAL_CONFIG run flags instead of CLI model selection.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    # eval_encoder_text_angle_checkpoints()
    cli_args = parse_args()
    if cli_args.use_config or (not cli_args.all_grounding and not cli_args.models and not cli_args.checkpoint):
        main()
    else:
        _run_cli(cli_args)
    # main_with_drone_name_filter(
    #     "/media/data1/feihong/remote/univerisity_dev/eval_results/retrieval_xxx/eval_retrieval_xxx_top1_success_drone_names.txt"
    # )
