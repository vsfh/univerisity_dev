import argparse
import json
import os
import random
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from PIL import Image
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from bbox.yolo_utils import build_target, eval_iou_acc, get_tensor_anchors, yolo_loss
from dataset import (
    BBOX_FILE,
    DEFAULT_SAT_TARGET_SIZE,
    DEFAULT_SUBSET_ANGLES,
    DEFAULT_SUBSET_HEIGHTS,
    DEFAULT_TEST_CROP_RATIO,
    DRONE_IMAGE_ROOT,
    TEST_SATELLITE_ROOT,
    TRAIN_HEIGHT_TO_BOX_SIZE,
    TRAIN_MAX_SATELLITE_ID,
    TRAIN_SATELLITE_ROOT,
    TRAIN_SPLIT_FILE,
    VAL_SPLIT_COUNT,
    VAL_SPLIT_FILE,
    TEST_SPLIT_FILE,
    ShiftedSatelliteDroneDataset,
    _augment_satellite_image_with_bbox_640,
    _build_center_bbox,
    _sanitize_bbox_xyxy,
)
from model import AngleConditionedFiLM, ResidualBlock
from bbox.yolo_utils import SpatialTransformer


# --- Configuration ---
class Config:
    MODEL_NAME = "Qwen/Qwen3-VL-4B-Instruct"
    CACHE_DIR = "/media/data1/feihong/remote/hf_cache"
    SAVE_ROOT = "/media/data1/feihong/remote/ckpt"
    EXP_NAME = "text_qwen3vl"

    TRAIN_SATELLITE_ROOT = TRAIN_SATELLITE_ROOT
    TEST_SATELLITE_ROOT = TEST_SATELLITE_ROOT
    DRONE_IMAGE_ROOT = DRONE_IMAGE_ROOT
    BBOX_FILE = BBOX_FILE
    TRAIN_MAX_SATELLITE_ID = TRAIN_MAX_SATELLITE_ID
    TRAIN_SPLIT_FILE = TRAIN_SPLIT_FILE
    VAL_SPLIT_FILE = VAL_SPLIT_FILE
    TEST_SPLIT_FILE = TEST_SPLIT_FILE
    VAL_SPLIT_COUNT = VAL_SPLIT_COUNT

    UNIV_SAT_SIZE = {"height": DEFAULT_SAT_TARGET_SIZE[0], "width": DEFAULT_SAT_TARGET_SIZE[1]}
    TEST_CROP_RATIO = DEFAULT_TEST_CROP_RATIO
    SUBSET_HEIGHTS = DEFAULT_SUBSET_HEIGHTS
    SUBSET_ANGLES = DEFAULT_SUBSET_ANGLES
    MAX_TEXT_LENGTH = 128

    NUM_EPOCHS = 20
    BATCH_SIZE = 15
    GRAD_ACCUMULATION_STEPS = 8
    NUM_WORKERS_TRAIN = 4
    NUM_WORKERS_VAL = 2
    PIN_MEMORY = True
    PERSISTENT_WORKERS = True
    PREFETCH_FACTOR = 2
    DROP_LAST_TRAIN = True

    LEARNING_RATE = 2e-4
    WEIGHT_DECAY = 0.01
    GRAD_CLIP_NORM = 1.0
    USE_AMP = True
    ENABLE_TF32 = True
    DTYPE = "bfloat16"

    HEAD_DIM = 768
    RETRIEVAL_DIM = 768
    USE_RETRIEVAL_PROJECTOR = False
    HEAT_SOFTMAX_TEMPERATURE = 1.0
    HEATMAP_CONFIDENCE_WEIGHT = 0.5
    BBOX_LOSS_WEIGHT = 0.1
    RETRIEVAL_LOSS_WEIGHT = 0.9
    HEATMAP_LOSS_WEIGHT = 0.2
    USE_HEATMAP_LOSS = True
    HEATMAP_LOSS_TYPE = ["mse", "cross_entropy"]
    HEATMAP_TARGET_MODE = "bbox_center"
    HEATMAP_SIGMA = 1.5
    HEATMAP_RADIUS = 4.5
    HEATMAP_BBOX_CENTER_EDGE_VALUE = 0.2
    HEATMAP_BBOX_CENTER_LOG_SCALE = 9.0
    HEATMAP_POS_WEIGHT = 8.0

    USE_GEO_INPUT = True
    USE_LORA = True
    LORA_RANK = 8
    LORA_ALPHA = 16.0
    LORA_DROPOUT = 0.05
    BACKBONE_GRADIENT_CHECKPOINTING = True
    VALIDATE_EVERY_EPOCH = False


class _SampleOnlyProcessor:
    def __init__(self, size: Tuple[int, int]):
        self.size = {"height": int(size[0]), "width": int(size[1])}

    def __call__(self, *args, **kwargs):
        raise RuntimeError("_SampleOnlyProcessor is only used to build ShiftedSatelliteDroneDataset.samples.")


class _SampleOnlyTokenizer:
    pad_token_id = 0

    def __call__(self, *args, **kwargs):
        raise RuntimeError("_SampleOnlyTokenizer is only used to build ShiftedSatelliteDroneDataset.samples.")


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
        raise KeyError("Unknown Config override(s): " + ", ".join(sorted(unknown_keys)))
    for key, value in overrides.items():
        current_value = getattr(Config, key)
        if isinstance(current_value, tuple) and isinstance(value, list):
            value = tuple(value)
        setattr(Config, key, value)


def load_yaml_overrides(yaml_path: Optional[str], experiment_name: Optional[str]) -> Dict[str, Any]:
    if not yaml_path:
        return {}
    with open(yaml_path, "r", encoding="utf-8") as f:
        payload = yaml.safe_load(f) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"YAML root must be a mapping: {yaml_path}")

    defaults = payload.get("defaults", {}) or {}
    experiments = payload.get("experiments")
    if experiments is None:
        selected = payload
    else:
        if not isinstance(experiments, list) or not experiments:
            raise ValueError("YAML 'experiments' must be a non-empty list.")
        if experiment_name is None:
            if len(experiments) != 1:
                raise ValueError("YAML contains multiple experiments; pass --experiment NAME.")
            selected = experiments[0]
        else:
            matches = [
                item
                for item in experiments
                if str(item.get("name", item.get("exp_name", ""))) == experiment_name
            ]
            if not matches:
                raise ValueError(f"Experiment '{experiment_name}' not found in {yaml_path}.")
            selected = matches[0]

    merged = dict(defaults)
    merged.update(selected)
    overrides = merged.get("config", merged)
    if not isinstance(overrides, dict):
        raise ValueError("YAML config overrides must be a mapping.")
    return overrides


def _dtype_from_name(name: str) -> torch.dtype:
    normalized = str(name).lower()
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp16", "float16", "half"}:
        return torch.float16
    if normalized in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


class TextVLMSatelliteDataset(Dataset):
    def __init__(
        self,
        processor: Any,
        tokenizer: Any,
        split: str,
    ):
        self.processor = processor
        self.tokenizer = tokenizer
        self.split = split
        self.sat_target_size = (
            int(Config.UNIV_SAT_SIZE["height"]),
            int(Config.UNIV_SAT_SIZE["width"]),
        )

        sample_builder = ShiftedSatelliteDroneDataset(
            processor=_SampleOnlyProcessor(self.sat_target_size),
            processor_sat=_SampleOnlyProcessor(self.sat_target_size),
            tokenizer=_SampleOnlyTokenizer(),
            split=split,
            train_satellite_root=Config.TRAIN_SATELLITE_ROOT,
            test_satellite_root=Config.TEST_SATELLITE_ROOT,
            drone_image_root=Config.DRONE_IMAGE_ROOT,
            bbox_file=Config.BBOX_FILE,
            train_max_satellite_id=Config.TRAIN_MAX_SATELLITE_ID,
            train_split_file=Config.TRAIN_SPLIT_FILE,
            val_split_file=Config.VAL_SPLIT_FILE,
            test_split_file=Config.TEST_SPLIT_FILE,
            val_split_count=Config.VAL_SPLIT_COUNT,
            max_text_length=Config.MAX_TEXT_LENGTH,
            sat_target_size=self.sat_target_size,
            test_crop_ratio=Config.TEST_CROP_RATIO,
            subset_heights=Config.SUBSET_HEIGHTS,
            subset_angles=Config.SUBSET_ANGLES,
        )
        self.samples = sample_builder.samples

    def __len__(self) -> int:
        return len(self.samples)

    def _tokenize_text(self, text: str) -> Tuple[torch.Tensor, torch.Tensor]:
        inputs = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=int(Config.MAX_TEXT_LENGTH),
            return_tensors="pt",
        )
        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask")
        if attention_mask is None:
            pad_token_id = getattr(self.tokenizer, "pad_token_id", None)
            if pad_token_id is None:
                attention_mask = torch.ones_like(input_ids, dtype=torch.long)
            else:
                attention_mask = (input_ids != int(pad_token_id)).long()
        return input_ids[0].long(), attention_mask[0].long()

    def _process_satellite(self, image: Image.Image) -> Tuple[torch.Tensor, torch.Tensor]:
        image_inputs = self.processor.image_processor(images=image, return_tensors="pt")
        pixel_values = image_inputs["pixel_values"]
        image_grid_thw = image_inputs["image_grid_thw"]

        if pixel_values.ndim == 3 and pixel_values.shape[0] == 1:
            pixel_values = pixel_values[0]
        if image_grid_thw.ndim == 2 and image_grid_thw.shape[0] == 1:
            image_grid_thw = image_grid_thw[0]
        if pixel_values.ndim != 2:
            raise ValueError(f"Expected Qwen image pixel_values shape (N, D), got {tuple(pixel_values.shape)}.")
        if image_grid_thw.shape != (3,):
            raise ValueError(f"Expected image_grid_thw shape (3,), got {tuple(image_grid_thw.shape)}.")
        return pixel_values, image_grid_thw.long()

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        try:
            search_image = Image.open(sample["satellite_path"]).convert("RGB")
        except FileNotFoundError:
            return self.__getitem__((idx + 1) % len(self.samples))

        orig_w, orig_h = search_image.size
        if self.split == "train":
            bbox_size = float(sample["bbox_size"])
            base_bbox = _build_center_bbox(
                image_width=orig_w,
                image_height=orig_h,
                bbox_width=bbox_size,
                bbox_height=bbox_size,
            )
        else:
            base_bbox = [float(v) for v in sample["bbox"]]

        augmented_search_image, augmented_bbox = _augment_satellite_image_with_bbox_640(
            image=search_image,
            bbox=base_bbox,
            mode=self.split,
            target_size=self.sat_target_size,
            test_crop_ratio=Config.TEST_CROP_RATIO,
        )

        target_h, target_w = self.sat_target_size
        resized_bbox = _sanitize_bbox_xyxy(
            augmented_bbox,
            image_width=float(target_w),
            image_height=float(target_h),
        )

        center_x = (resized_bbox[0] + resized_bbox[2]) / 2.0 / max(float(target_w), 1.0)
        center_y = (resized_bbox[1] + resized_bbox[3]) / 2.0 / max(float(target_h), 1.0)
        col_idx = min(int(center_x * 3), 2)
        row_idx = min(int(center_y * 3), 2)
        index = row_idx * 3 + col_idx

        input_ids, attention_mask = self._tokenize_text(sample["text"])
        search_pixel_values, image_grid_thw = self._process_satellite(augmented_search_image)

        return {
            "search_pixel_values": search_pixel_values,
            "image_grid_thw": image_grid_thw,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "index": torch.tensor(index, dtype=torch.long),
            "bbox": torch.tensor(resized_bbox, dtype=torch.float32),
            "height": torch.tensor(sample["height"], dtype=torch.long),
            "angle": torch.tensor(sample["angle"], dtype=torch.long),
            "drone_path": sample["drone_path"],
            "satellite_path": sample["satellite_path"],
            "text": sample["text"],
        }


def text_vlm_collate_fn(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    pixel_shapes = {tuple(item["search_pixel_values"].shape) for item in items}
    if len(pixel_shapes) != 1:
        raise ValueError(f"Qwen image pixel_values must have equal shape in a batch, got {sorted(pixel_shapes)}.")
    return {
        "search_pixel_values": torch.cat([item["search_pixel_values"] for item in items], dim=0),
        "image_grid_thw": torch.stack([item["image_grid_thw"] for item in items], dim=0),
        "input_ids": torch.stack([item["input_ids"] for item in items], dim=0),
        "attention_mask": torch.stack([item["attention_mask"] for item in items], dim=0),
        "index": torch.stack([item["index"] for item in items], dim=0),
        "bbox": torch.stack([item["bbox"] for item in items], dim=0),
        "height": torch.stack([item["height"] for item in items], dim=0),
        "angle": torch.stack([item["angle"] for item in items], dim=0),
        "drone_path": [item["drone_path"] for item in items],
        "satellite_path": [item["satellite_path"] for item in items],
        "text": [item["text"] for item in items],
    }


class LoRALinear(nn.Module):
    def __init__(
        self,
        base: nn.Linear,
        rank: int,
        alpha: float,
        dropout: float,
    ):
        super().__init__()
        if rank <= 0:
            raise ValueError(f"LoRA rank must be positive, got {rank}.")

        self.base = base
        for param in self.base.parameters():
            param.requires_grad = False

        self.lora_a = nn.Linear(base.in_features, rank, bias=False)
        self.lora_b = nn.Linear(rank, base.out_features, bias=False)
        self.dropout = nn.Dropout(float(dropout))
        self.scaling = float(alpha) / float(rank)

        nn.init.kaiming_uniform_(self.lora_a.weight, a=5**0.5)
        nn.init.zeros_(self.lora_b.weight)

    @property
    def weight(self) -> torch.Tensor:
        return self.base.weight

    @property
    def bias(self) -> Optional[torch.Tensor]:
        return self.base.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.lora_b(self.lora_a(self.dropout(x))) * self.scaling


class TextQwenVLRetrievalGrounding(nn.Module):
    def __init__(
        self,
        model_name: str = Config.MODEL_NAME,
        cache_dir: str = Config.CACHE_DIR,
        dtype: torch.dtype = torch.bfloat16,
        head_dim: int = Config.HEAD_DIM,
        retrieval_dim: int = Config.RETRIEVAL_DIM,
        heat_softmax_temperature: float = Config.HEAT_SOFTMAX_TEMPERATURE,
    ):
        super().__init__()
        self.vlm = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            dtype=dtype,
        )
        self.vlm.config.output_hidden_states = False
        for param in self.vlm.parameters():
            param.requires_grad = False
        self.lora_layer_count = 0
        if Config.USE_LORA:
            self.lora_layer_count = self._inject_lora(
                self.vlm.model,
                rank=int(Config.LORA_RANK),
                alpha=float(Config.LORA_ALPHA),
                dropout=float(Config.LORA_DROPOUT),
            )
            if self.lora_layer_count <= 0:
                raise ValueError("No Qwen3-VL backbone Linear layers were replaced by LoRA.")
            self.vlm.config.use_cache = False
            self.vlm.model.config.use_cache = False
            if Config.BACKBONE_GRADIENT_CHECKPOINTING:
                self.vlm.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False}
                )

        self.hidden_dim = int(self.vlm.config.text_config.hidden_size)
        self.head_dim = int(head_dim)
        self.retrieval_dim = int(retrieval_dim)
        self.heat_softmax_temperature = float(heat_softmax_temperature)

        if Config.USE_RETRIEVAL_PROJECTOR:
            self.text_retrieval_proj = nn.Sequential(
                nn.LayerNorm(self.hidden_dim),
                nn.Linear(self.hidden_dim, self.retrieval_dim),
            )
            self.image_retrieval_proj = nn.Sequential(
                nn.LayerNorm(self.hidden_dim),
                nn.Linear(self.hidden_dim, self.retrieval_dim),
            )
        else:
            self.text_retrieval_proj = None
            self.image_retrieval_proj = None
        self.text_ground_proj = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.head_dim),
            nn.GELU(),
            nn.Linear(self.head_dim, self.head_dim),
        )
        self.sat_ground_proj = nn.Conv2d(self.hidden_dim, self.head_dim, kernel_size=1)
        self.angle_film = AngleConditionedFiLM(angle_dim=3, feature_dim=self.head_dim)
        self.heat_fuse = nn.Conv2d(self.head_dim + 1, self.head_dim, kernel_size=1)
        self.bbox_pos_proj = nn.Conv2d(2, self.head_dim, kernel_size=1)
        self.bbox_transformer = SpatialTransformer(
            in_channels=self.head_dim,
            n_heads=8,
            d_head=64,
            depth=1,
            context_dim=self.head_dim,
        )
        self.bbox_fcn_out = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=self.head_dim,
                out_channels=self.head_dim // 2,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.ReLU(inplace=True),
            ResidualBlock(self.head_dim // 2),
            nn.Conv2d(self.head_dim // 2, 9 * 5, kernel_size=1),
        )

        nn.init.zeros_(self.heat_fuse.weight)
        nn.init.zeros_(self.heat_fuse.bias)
        with torch.no_grad():
            channels_to_copy = min(self.head_dim, self.heat_fuse.weight.shape[1] - 1)
            for idx in range(channels_to_copy):
                self.heat_fuse.weight[idx, idx, 0, 0] = 1.0
        nn.init.zeros_(self.bbox_pos_proj.weight)
        nn.init.zeros_(self.bbox_pos_proj.bias)

    def _inject_lora(
        self,
        module: nn.Module,
        rank: int,
        alpha: float,
        dropout: float,
    ) -> int:
        replaced = 0
        for child_name, child in list(module.named_children()):
            if isinstance(child, LoRALinear):
                continue
            if isinstance(child, nn.Linear):
                setattr(
                    module,
                    child_name,
                    LoRALinear(
                        base=child,
                        rank=rank,
                        alpha=alpha,
                        dropout=dropout,
                    ),
                )
                replaced += 1
                continue
            replaced += self._inject_lora(
                child,
                rank=rank,
                alpha=alpha,
                dropout=dropout,
            )
        return replaced

    def _pool_text(self, hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if hidden_state.ndim != 3:
            raise ValueError(f"Expected text hidden_state shape (B, L, C), got {tuple(hidden_state.shape)}.")
        mask = attention_mask.to(device=hidden_state.device, dtype=hidden_state.dtype).unsqueeze(-1)
        pooled = (hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        return pooled

    def _encode_text(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.vlm.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
        hidden_state = outputs.last_hidden_state
        return self._pool_text(hidden_state, attention_mask)

    def _encode_satellite(
        self,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
        batch_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if image_grid_thw.ndim != 2 or image_grid_thw.shape != (batch_size, 3):
            raise ValueError(f"Expected image_grid_thw shape {(batch_size, 3)}, got {tuple(image_grid_thw.shape)}.")

        image_outputs = self.vlm.model.get_image_features(
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            return_dict=True,
        )
        token_groups = list(image_outputs.pooler_output)
        if len(token_groups) != batch_size:
            raise ValueError(f"Expected {batch_size} image token groups, got {len(token_groups)}.")
        token_shapes = {tuple(tokens.shape) for tokens in token_groups}
        if len(token_shapes) != 1:
            raise ValueError(f"All images in a batch must have equal token shape, got {sorted(token_shapes)}.")

        sat_tokens = torch.stack(token_groups, dim=0)
        merge_size = int(self.vlm.model.visual.spatial_merge_size)
        grid_h = image_grid_thw[:, 1] // merge_size
        grid_w = image_grid_thw[:, 2] // merge_size
        if not torch.all(grid_h == grid_h[0]) or not torch.all(grid_w == grid_w[0]):
            raise ValueError("All images in a batch must share the same Qwen visual grid size.")
        grid_h_int = int(grid_h[0].item())
        grid_w_int = int(grid_w[0].item())
        expected_tokens = grid_h_int * grid_w_int
        if sat_tokens.shape[1] != expected_tokens:
            raise ValueError(
                f"Qwen image token count mismatch: tokens={sat_tokens.shape[1]}, grid={grid_h_int}x{grid_w_int}."
            )

        sat_features_2d = sat_tokens.transpose(1, 2).reshape(
            batch_size,
            self.hidden_dim,
            grid_h_int,
            grid_w_int,
        ).contiguous()
        return sat_tokens, sat_features_2d

    def _build_position_embedding(self, feature_map: torch.Tensor) -> torch.Tensor:
        batch_size, _, height, width = feature_map.shape
        y_coords = torch.linspace(-1.0, 1.0, height, device=feature_map.device, dtype=feature_map.dtype)
        x_coords = torch.linspace(-1.0, 1.0, width, device=feature_map.device, dtype=feature_map.dtype)
        grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing="ij")
        coords = torch.stack([grid_x, grid_y], dim=0).unsqueeze(0).expand(batch_size, -1, -1, -1)
        return self.bbox_pos_proj(coords)

    def _spatial_softmax(self, heatmap: torch.Tensor) -> torch.Tensor:
        if heatmap.ndim != 4:
            raise ValueError(f"Expected heatmap shape (B, C, H, W), got {tuple(heatmap.shape)}.")
        batch_size, channels, height, width = heatmap.shape
        temperature = max(self.heat_softmax_temperature, 1e-6)
        heatmap = heatmap.view(batch_size, channels, -1) / temperature
        heatmap = F.softmax(heatmap, dim=-1)
        return heatmap.view(batch_size, channels, height, width)

    def _text_heatmap(self, text_ground: torch.Tensor, sat_ground_2d: torch.Tensor) -> torch.Tensor:
        text_kernel = F.normalize(text_ground, p=2, dim=1)
        sat_ground = F.normalize(sat_ground_2d, p=2, dim=1)
        return torch.einsum("bc,bchw->bhw", text_kernel, sat_ground).unsqueeze(1)

    def forward(
        self,
        search_pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        geo: Optional[torch.Tensor] = None,
    ):
        batch_size = input_ids.shape[0]

        text_pooled = self._encode_text(input_ids=input_ids, attention_mask=attention_mask)
        _, sat_features_2d = self._encode_satellite(
            pixel_values=search_pixel_values,
            image_grid_thw=image_grid_thw,
            batch_size=batch_size,
        )

        if Config.USE_RETRIEVAL_PROJECTOR:
            if self.text_retrieval_proj is None or self.image_retrieval_proj is None:
                raise ValueError("Retrieval projector is enabled but projector modules are missing.")
            text_feats = self.text_retrieval_proj(text_pooled.float())
            sat_retrieval_2d = self.image_retrieval_proj(
                sat_features_2d.permute(0, 2, 3, 1).contiguous().float()
            ).permute(0, 3, 1, 2).contiguous()
        else:
            text_feats = text_pooled.float()
            sat_retrieval_2d = sat_features_2d.float().contiguous()

        text_feats = F.normalize(text_feats, p=2, dim=1)
        sat_feature_2d_pool = F.adaptive_avg_pool2d(sat_retrieval_2d, (3, 3)).flatten(2).transpose(1, 2)
        sat_feature_2d_pool = F.normalize(sat_feature_2d_pool, p=2, dim=2)

        text_ground = self.text_ground_proj(text_pooled.float())
        if geo is not None:
            text_ground = self.angle_film(text_ground, geo)

        sat_ground_2d = self.sat_ground_proj(sat_features_2d.float().contiguous())
        heatmap_logits = self._text_heatmap(text_ground, sat_ground_2d)
        heatmap = self._spatial_softmax(heatmap_logits)

        heat_gate = heatmap * heatmap.shape[-2] * heatmap.shape[-1] / 2
        guided_sat_features = self.heat_fuse(torch.cat([sat_ground_2d, heat_gate], dim=1))
        guided_sat_features = guided_sat_features + self._build_position_embedding(guided_sat_features)

        fused_features = self.bbox_transformer(
            x=guided_sat_features,
            context=text_ground.unsqueeze(1),
        )
        pred_anchor = self.bbox_fcn_out(fused_features)

        heatmap_out = heatmap
        if heatmap_out.shape[-2:] != pred_anchor.shape[-2:]:
            heatmap_logits = F.interpolate(
                heatmap_logits,
                size=pred_anchor.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            heatmap_out = self._spatial_softmax(heatmap_logits)

        return (
            pred_anchor,
            None,
            text_feats,
            text_feats,
            sat_feature_2d_pool,
            None,
            heatmap_out,
        )


def build_geo_features(batch: Dict[str, torch.Tensor], device: torch.device) -> Optional[torch.Tensor]:
    if not Config.USE_GEO_INPUT:
        return None
    angles_rad = torch.deg2rad(batch["angle"].to(device, non_blocking=True).float())
    height = batch["height"].to(device, non_blocking=True).float()
    return torch.cat(
        [
            torch.cos(angles_rad)[..., None],
            torch.sin(angles_rad)[..., None],
            height[..., None] / 300.0,
        ],
        dim=1,
    )


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


def build_heatmap_target(
    target_bbox: torch.Tensor,
    heatmap_hw: Tuple[int, int],
    image_wh: Tuple[int, int],
    sigma: float,
    radius: Optional[float],
) -> torch.Tensor:
    grid_h, grid_w = int(heatmap_hw[0]), int(heatmap_hw[1])
    image_w, image_h = float(image_wh[0]), float(image_wh[1])
    device = target_bbox.device
    dtype = target_bbox.dtype

    x1 = torch.minimum(target_bbox[:, 0], target_bbox[:, 2]).clamp(0.0, image_w)
    y1 = torch.minimum(target_bbox[:, 1], target_bbox[:, 3]).clamp(0.0, image_h)
    x2 = torch.maximum(target_bbox[:, 0], target_bbox[:, 2]).clamp(0.0, image_w)
    y2 = torch.maximum(target_bbox[:, 1], target_bbox[:, 3]).clamp(0.0, image_h)

    center_x = (x1 + x2) * 0.5 / max(image_w, 1.0) * grid_w
    center_y = (y1 + y2) * 0.5 / max(image_h, 1.0) * grid_h
    center_x = center_x.clamp(0.0, grid_w - 1e-6)
    center_y = center_y.clamp(0.0, grid_h - 1e-6)

    ys = torch.arange(grid_h, device=device, dtype=dtype).view(1, grid_h, 1)
    xs = torch.arange(grid_w, device=device, dtype=dtype).view(1, 1, grid_w)
    dx2 = (xs - center_x.view(-1, 1, 1)) ** 2
    dy2 = (ys - center_y.view(-1, 1, 1)) ** 2
    dist2 = dx2 + dy2
    center_target = torch.exp(-dist2 / (2.0 * max(float(sigma), 1e-6) ** 2))
    if radius is not None and float(radius) > 0.0:
        center_target = center_target * (dist2 <= float(radius) ** 2).to(dtype=dtype)

    if Config.HEATMAP_TARGET_MODE == "center":
        target = center_target
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
        raise ValueError(f"Invalid HEATMAP_TARGET_MODE={Config.HEATMAP_TARGET_MODE}.")
    return target.unsqueeze(1).clamp(0.0, 1.0)


def heatmap_loss_fn(heatmap_logits: torch.Tensor, target_bbox: torch.Tensor) -> torch.Tensor:
    heatmap_target = build_heatmap_target(
        target_bbox=target_bbox,
        heatmap_hw=heatmap_logits.shape[-2:],
        image_wh=(Config.UNIV_SAT_SIZE["width"], Config.UNIV_SAT_SIZE["height"]),
        sigma=Config.HEATMAP_SIGMA,
        radius=Config.HEATMAP_RADIUS,
    )
    pred = heatmap_logits.float().flatten(1)
    target = heatmap_target.float().flatten(1)
    target_prob = target / target.sum(dim=1, keepdim=True).clamp_min(1e-6)
    pred_prob = pred / pred.sum(dim=1, keepdim=True).clamp_min(1e-6)

    loss = heatmap_logits.new_zeros(())
    for loss_type in Config.HEATMAP_LOSS_TYPE:
        if loss_type == "mse":
            loss = loss + F.mse_loss(pred_prob, target_prob, reduction="mean") * pred_prob.shape[1]
        elif loss_type in {"cross_entropy", "ce", "spatial_ce"}:
            loss = loss + -(target_prob * pred_prob.clamp_min(1e-8).log()).sum(dim=1).mean()
        elif loss_type == "weighted_bce":
            bce_loss = F.binary_cross_entropy(
                heatmap_logits.float().clamp(1e-6, 1.0 - 1e-6),
                heatmap_target.float(),
                reduction="none",
            )
            weight = 1.0 + heatmap_target.float() * (Config.HEATMAP_POS_WEIGHT - 1.0)
            loss = loss + (bce_loss * weight).mean()
        else:
            raise ValueError(f"Invalid heatmap loss type: {loss_type}")
    return loss


def add_heatmap_to_confidence(pred_anchor: torch.Tensor, heatmap_logits: Optional[torch.Tensor]) -> torch.Tensor:
    if heatmap_logits is None or Config.HEATMAP_CONFIDENCE_WEIGHT <= 0.0:
        return pred_anchor
    if heatmap_logits.shape[-2:] != pred_anchor.shape[-2:]:
        heatmap_logits = F.interpolate(
            heatmap_logits,
            size=pred_anchor.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
    heat_confidence = Config.HEATMAP_CONFIDENCE_WEIGHT * heatmap_logits.unsqueeze(1)
    pred_anchor = pred_anchor.clone()
    pred_anchor[:, :, 4:5, :, :] = pred_anchor[:, :, 4:5, :, :] + heat_confidence
    return pred_anchor


def build_positive_indices(batch_size: int, local_indices: torch.Tensor, device: torch.device) -> torch.Tensor:
    positive_indices = torch.zeros(batch_size, batch_size * 9, device=device)
    batch_offsets = torch.arange(batch_size, device=device) * 9
    row_indices_broad = torch.arange(batch_size, device=device).unsqueeze(1)
    col_offsets = torch.arange(9, device=device).unsqueeze(0)
    same_image_cols = batch_offsets.unsqueeze(1) + col_offsets
    positive_indices[row_indices_broad, same_image_cols] = 0.01
    global_positive_indices = local_indices + batch_offsets
    row_indices_flat = torch.arange(batch_size, device=device)
    positive_indices[row_indices_flat, global_positive_indices] = 0.92
    return positive_indices


def build_dataloader_kwargs(num_workers: int, drop_last: bool = False) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {
        "num_workers": int(num_workers),
        "pin_memory": bool(Config.PIN_MEMORY),
        "drop_last": bool(drop_last),
        "collate_fn": text_vlm_collate_fn,
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = bool(Config.PERSISTENT_WORKERS)
        kwargs["prefetch_factor"] = int(Config.PREFETCH_FACTOR)
    return kwargs


def compute_losses(
    outputs: Tuple,
    batch: Dict[str, Any],
    anchors_full: torch.Tensor,
    device: torch.device,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    pred_anchor, _, _, text_feats, grid_feats, _, heatmap_logits = outputs
    batch_size = pred_anchor.shape[0]
    pred_anchor = pred_anchor.view(batch_size, 9, 5, pred_anchor.shape[2], pred_anchor.shape[3])
    pred_anchor = add_heatmap_to_confidence(pred_anchor, heatmap_logits)

    target_bbox = batch["bbox"].to(device, non_blocking=True)
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
    heatmap_loss = bbox_loss.new_zeros(())
    if Config.USE_HEATMAP_LOSS and heatmap_logits is not None:
        heatmap_loss = heatmap_loss_fn(heatmap_logits, target_bbox)
        bbox_loss = bbox_loss + Config.HEATMAP_LOSS_WEIGHT * heatmap_loss

    local_indices = batch["index"].to(device, non_blocking=True)
    candidate_feats = grid_feats.reshape(-1, grid_feats.shape[-1])
    positive_indices = build_positive_indices(batch_size, local_indices, device)
    retrieval_loss = info_nce_loss(text_feats, candidate_feats, positive_indices)

    total_loss = Config.RETRIEVAL_LOSS_WEIGHT * retrieval_loss + Config.BBOX_LOSS_WEIGHT * bbox_loss
    metrics = {
        "loss": total_loss.detach(),
        "bbox_loss": bbox_loss.detach(),
        "bbox_geo_loss": loss_geo.detach(),
        "bbox_cls_loss": loss_cls.detach(),
        "heatmap_loss": heatmap_loss.detach(),
        "retrieval_loss": retrieval_loss.detach(),
    }
    return total_loss, metrics


@torch.no_grad()
def validate(
    loader: DataLoader,
    model: nn.Module,
    accelerator: Accelerator,
    anchors_full: torch.Tensor,
) -> Tuple[float, float, float, float]:
    model.eval()
    accu50_sum = torch.tensor(0.0, device=accelerator.device)
    accu25_sum = torch.tensor(0.0, device=accelerator.device)
    iou_sum = torch.tensor(0.0, device=accelerator.device)
    retrieval_sum = torch.tensor(0.0, device=accelerator.device)
    sample_count = torch.tensor(0.0, device=accelerator.device)
    amp_enabled = Config.USE_AMP and accelerator.device.type == "cuda"

    for batch in tqdm(loader, desc="Validating", disable=not accelerator.is_main_process):
        search_pixel_values = batch["search_pixel_values"].to(accelerator.device, non_blocking=True)
        image_grid_thw = batch["image_grid_thw"].to(accelerator.device, non_blocking=True)
        input_ids = batch["input_ids"].to(accelerator.device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(accelerator.device, non_blocking=True)
        geo = build_geo_features(batch, accelerator.device)
        target_bbox = batch["bbox"].to(accelerator.device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=amp_enabled):
            outputs = model(search_pixel_values, image_grid_thw, input_ids, attention_mask, geo)
            _, metrics = compute_losses(outputs, batch, anchors_full, accelerator.device)
            pred_anchor, _, _, _, grid_feats, _, heatmap_logits = outputs

        batch_size = pred_anchor.shape[0]
        pred_anchor = pred_anchor.view(batch_size, 9, 5, pred_anchor.shape[2], pred_anchor.shape[3])
        pred_anchor = add_heatmap_to_confidence(pred_anchor, heatmap_logits)
        _, best_anchor_gi_gj = build_target(
            target_bbox,
            anchors_full,
            (Config.UNIV_SAT_SIZE["width"], Config.UNIV_SAT_SIZE["height"]),
            (pred_anchor.shape[4], pred_anchor.shape[3]),
        )
        accu_list, _, iou, _, _, _ = eval_iou_acc(
            pred_anchor,
            target_bbox,
            anchors_full,
            best_anchor_gi_gj[:, 1],
            best_anchor_gi_gj[:, 2],
            (Config.UNIV_SAT_SIZE["width"], Config.UNIV_SAT_SIZE["height"]),
            iou_threshold_list=[0.5, 0.25],
        )
        n = torch.tensor(float(batch_size), device=accelerator.device)
        accu50_sum += accu_list[0].detach() * n
        accu25_sum += accu_list[1].detach() * n
        iou_sum += iou.detach() * n
        retrieval_sum += metrics["retrieval_loss"] * n
        sample_count += n

    reduced = accelerator.reduce(
        torch.stack([accu50_sum, accu25_sum, iou_sum, retrieval_sum, sample_count]),
        reduction="sum",
    )
    denom = reduced[4].clamp_min(1.0)
    return (
        (reduced[0] / denom).item(),
        (reduced[1] / denom).item(),
        (reduced[2] / denom).item(),
        (reduced[3] / denom).item(),
    )


def train(save_dir: str) -> None:
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        mixed_precision="no",
        gradient_accumulation_steps=Config.GRAD_ACCUMULATION_STEPS,
        step_scheduler_with_optimizer=False,
        kwargs_handlers=[ddp_kwargs],
    )

    if accelerator.is_main_process:
        torch.backends.cuda.matmul.allow_tf32 = Config.ENABLE_TF32
        torch.backends.cudnn.allow_tf32 = Config.ENABLE_TF32
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, "effective_config.json"), "w", encoding="utf-8") as f:
            json.dump(config_to_dict(), f, indent=2, ensure_ascii=False, sort_keys=True)

    exp_name = Path(save_dir).name
    writer = SummaryWriter(f"runs/{exp_name}") if accelerator.is_main_process else None
    scaler = torch.amp.GradScaler("cuda", enabled=Config.USE_AMP and accelerator.device.type == "cuda")
    amp_enabled = Config.USE_AMP and accelerator.device.type == "cuda"

    accelerator.print("Loading Qwen3-VL processor and frozen backbone...")
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
    )
    accelerator.print(
        f"Backbone LoRA: enabled={Config.USE_LORA}, "
        f"layers={model.lora_layer_count}, "
        f"rank={Config.LORA_RANK}, alpha={Config.LORA_ALPHA}, "
        f"retrieval_projector={Config.USE_RETRIEVAL_PROJECTOR}"
    )
    anchors_full = get_tensor_anchors(accelerator.device)

    accelerator.print("Setting up text-query / satellite-gallery dataset...")
    train_dataset = TextVLMSatelliteDataset(processor=processor, tokenizer=tokenizer, split="train")
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=Config.BATCH_SIZE,
        **build_dataloader_kwargs(Config.NUM_WORKERS_TRAIN, drop_last=Config.DROP_LAST_TRAIN),
    )

    test_dataset = TextVLMSatelliteDataset(processor=processor, tokenizer=tokenizer, split="test")
    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=Config.BATCH_SIZE,
        **build_dataloader_kwargs(Config.NUM_WORKERS_VAL),
    )
    accelerator.print(f"Train: {len(train_dataset)}, Test: {len(test_dataset)}")

    trainable_params = [param for param in model.parameters() if param.requires_grad]
    if not trainable_params:
        raise ValueError("No trainable parameters found.")
    optimizer = AdamW(trainable_params, lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, Config.NUM_EPOCHS))
    trainable_count = sum(param.numel() for param in trainable_params)
    accelerator.print(
        f"Optimizer: lr={Config.LEARNING_RATE:.2e}, trainable params={trainable_count:,}"
    )

    model, optimizer, scheduler, train_loader, test_loader = accelerator.prepare(
        model,
        optimizer,
        scheduler,
        train_loader,
        test_loader,
    )
    trainable_params = [param for param in model.parameters() if param.requires_grad]

    global_step = 0
    for epoch in range(Config.NUM_EPOCHS):
        model.train()
        epoch_metrics = {
            "loss": 0.0,
            "bbox_loss": 0.0,
            "heatmap_loss": 0.0,
            "retrieval_loss": 0.0,
        }
        progress_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{Config.NUM_EPOCHS}",
            disable=not accelerator.is_main_process,
        )
        for batch in progress_bar:
            with accelerator.accumulate(model):
                search_pixel_values = batch["search_pixel_values"].to(accelerator.device, non_blocking=True)
                image_grid_thw = batch["image_grid_thw"].to(accelerator.device, non_blocking=True)
                input_ids = batch["input_ids"].to(accelerator.device, non_blocking=True)
                attention_mask = batch["attention_mask"].to(accelerator.device, non_blocking=True)
                geo = build_geo_features(batch, accelerator.device)

                with torch.amp.autocast("cuda", enabled=amp_enabled):
                    outputs = model(search_pixel_values, image_grid_thw, input_ids, attention_mask, geo)
                    loss, metrics = compute_losses(outputs, batch, anchors_full, accelerator.device)

                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                if Config.GRAD_CLIP_NORM is not None and Config.GRAD_CLIP_NORM > 0:
                    scaler.unscale_(optimizer)
                    accelerator.clip_grad_norm_(trainable_params, Config.GRAD_CLIP_NORM)
                scaler.step(optimizer)
                scaler.update()

            global_step += 1
            for key in epoch_metrics:
                epoch_metrics[key] += float(metrics[key].item())
            progress_bar.set_postfix(
                {
                    "loss": f"{metrics['loss'].item():.4f}",
                    "bbox": f"{metrics['bbox_loss'].item():.4f}",
                    "heat": f"{metrics['heatmap_loss'].item():.4f}",
                    "ret": f"{metrics['retrieval_loss'].item():.4f}",
                }
            )
            if writer is not None:
                writer.add_scalar("Loss/train_step", metrics["loss"].item(), global_step)
                writer.add_scalar("Loss/bbox_step", metrics["bbox_loss"].item(), global_step)
                writer.add_scalar("Loss/heatmap_step", metrics["heatmap_loss"].item(), global_step)
                writer.add_scalar("Loss/retrieval_step", metrics["retrieval_loss"].item(), global_step)

        scheduler.step()
        denom = max(1, len(train_loader))
        if writer is not None:
            for key, value in epoch_metrics.items():
                writer.add_scalar(f"Loss/{key}_epoch", value / denom, epoch)
            writer.add_scalar("LR", optimizer.param_groups[0]["lr"], epoch)

        if Config.VALIDATE_EVERY_EPOCH:
            accu50, accu25, iou, retrieval = validate(test_loader, model, accelerator, anchors_full)
            accelerator.print(
                f"Validation epoch {epoch + 1}: Acc@0.50={accu50:.4f}, "
                f"Acc@0.25={accu25:.4f}, IoU={iou:.4f}, RetrievalLoss={retrieval:.4f}"
            )
            if writer is not None:
                writer.add_scalar("Val/accu50", accu50, epoch)
                writer.add_scalar("Val/accu25", accu25, epoch)
                writer.add_scalar("Val/iou", iou, epoch)
                writer.add_scalar("Val/retrieval_loss", retrieval, epoch)

        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            unwrapped = accelerator.unwrap_model(model)
            trainable_names = {
                name
                for name, param in unwrapped.named_parameters()
                if param.requires_grad
            }
            model_state = {
                name: tensor.detach().cpu()
                for name, tensor in unwrapped.state_dict().items()
                if name in trainable_names
            }
            checkpoint = {
                "model": model_state,
                "checkpoint_type": "trainable_only",
                "config": config_to_dict(),
                "epoch": epoch + 1,
            }
            torch.save(checkpoint, os.path.join(save_dir, "last.pth"))
            accelerator.print(f"Saved checkpoint: {os.path.join(save_dir, 'last.pth')}")

    if writer is not None:
        writer.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train text-query / satellite-gallery retrieval and grounding with Qwen3-VL features."
    )
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--experiment", type=str, default=None)
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    args = parse_args()
    overrides = load_yaml_overrides(args.config, args.experiment)
    apply_config_overrides(overrides)
    set_seed(args.seed)

    save_dir = args.save_dir
    if save_dir is None:
        save_dir = os.path.join(Config.SAVE_ROOT, Config.EXP_NAME)

    train(save_dir)
