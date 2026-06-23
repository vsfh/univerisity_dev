from typing import Any, Dict, Tuple

import torch


def black_fill_from_processor(processor: Any, channels: int = 3) -> torch.Tensor:
    mean = list(getattr(processor, "image_mean", [0.0] * channels))
    std = list(getattr(processor, "image_std", [1.0] * channels))
    if len(mean) < channels:
        mean.extend([0.0] * (channels - len(mean)))
    if len(std) < channels:
        std.extend([1.0] * (channels - len(std)))
    values = [float(-mean[idx] / max(float(std[idx]), 1e-6)) for idx in range(channels)]
    return torch.tensor(values, dtype=torch.float32)


def letterbox_satellite_batch(
    batch: Dict[str, Any],
    fill_value: torch.Tensor,
    target_size: int = 768,
) -> Tuple[Dict[str, Any], Dict[str, int]]:
    search = batch["search_pixel_values"]
    if search.ndim != 4:
        raise ValueError(f"Expected search_pixel_values shape (B, C, H, W), got {tuple(search.shape)}.")
    _, channels, height, width = search.shape
    target_size = int(target_size)
    if height > target_size or width > target_size:
        raise ValueError(f"Cannot letterbox image {height}x{width} into {target_size}x{target_size}.")

    pad_top = (target_size - height) // 2
    pad_left = (target_size - width) // 2
    fill = fill_value.to(device=search.device, dtype=search.dtype)
    if fill.numel() < channels:
        fill = torch.cat([fill, fill.new_zeros(channels - fill.numel())])
    padded = fill[:channels].view(1, channels, 1, 1).expand(search.shape[0], channels, target_size, target_size).clone()
    padded[:, :, pad_top : pad_top + height, pad_left : pad_left + width] = search

    out = dict(batch)
    out["search_pixel_values"] = padded
    bbox = batch["bbox"].clone()
    offset = bbox.new_tensor([pad_left, pad_top, pad_left, pad_top])
    out["bbox"] = bbox + offset
    return out, {
        "pad_left": int(pad_left),
        "pad_top": int(pad_top),
        "orig_width": int(width),
        "orig_height": int(height),
        "target_size": int(target_size),
    }


def crop_letterbox_bbox(pred_bbox: torch.Tensor, meta: Dict[str, int]) -> torch.Tensor:
    out = pred_bbox.clone()
    out[:, [0, 2]] -= float(meta["pad_left"])
    out[:, [1, 3]] -= float(meta["pad_top"])
    out[:, [0, 2]] = out[:, [0, 2]].clamp(0.0, float(meta["orig_width"]))
    out[:, [1, 3]] = out[:, [1, 3]].clamp(0.0, float(meta["orig_height"]))
    return out
