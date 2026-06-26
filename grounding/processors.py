from typing import Any, Dict, Tuple

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor


SIGLIP_PROCESSOR_MODEL_TYPES = {"siglip2_heat", "siglip2_test", "siglip_ground"}


def _size_to_hw(size: Any) -> Tuple[int, int]:
    if isinstance(size, dict):
        return int(size["height"]), int(size["width"])
    if isinstance(size, (list, tuple)) and len(size) == 2:
        return int(size[0]), int(size[1])
    edge = int(size)
    return edge, edge


class ImageNetProcessor:
    def __init__(self, size: Any):
        height, width = _size_to_hw(size)
        self.size = {"height": height, "width": width}
        self.image_mean = [0.485, 0.456, 0.406]
        self.image_std = [0.229, 0.224, 0.225]

    def __call__(self, images: Image.Image, return_tensors: str = "pt") -> Dict[str, torch.Tensor]:
        del return_tensors
        target_h = int(self.size["height"])
        target_w = int(self.size["width"])
        if images.mode != "RGB":
            images = images.convert("RGB")
        if images.size != (target_w, target_h):
            images = images.resize((target_w, target_h), Image.Resampling.BILINEAR)

        image_np = np.asarray(images, dtype=np.float32) / 255.0
        pixel_values = torch.from_numpy(image_np).permute(2, 0, 1)
        mean = pixel_values.new_tensor(self.image_mean).view(3, 1, 1)
        std = pixel_values.new_tensor(self.image_std).view(3, 1, 1)
        pixel_values = (pixel_values - mean) / std
        return {"pixel_values": pixel_values.unsqueeze(0)}


def build_grounding_image_processors(cfg: Dict[str, Any]):
    model_type = str(cfg["model"]["type"])
    processor_style = str(
        cfg.get("data", {}).get(
            "processor_style",
            "siglip" if model_type in SIGLIP_PROCESSOR_MODEL_TYPES else "imagenet",
        )
    ).lower()

    if processor_style == "siglip":
        model_name = cfg["model"]["model_name"]
        cache_dir = cfg["model"]["cache_dir"]
        processor = AutoImageProcessor.from_pretrained(model_name, cache_dir=cache_dir)
        processor_sat = AutoImageProcessor.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            size=cfg["data"]["sat_size"],
        )
        return processor, processor_sat

    if processor_style in {"imagenet", "timm", "torchvision", "swin"}:
        processor = ImageNetProcessor(cfg["data"]["drone_size"])
        processor_sat = ImageNetProcessor(cfg["data"]["sat_size"])
        return processor, processor_sat

    raise ValueError(
        f"Invalid data.processor_style={processor_style!r}. "
        "Use 'siglip' or 'imagenet'."
    )
