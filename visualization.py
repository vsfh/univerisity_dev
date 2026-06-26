import argparse
import textwrap
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from dataset import initialize_shifted_dataset_like_unified_siglip_supp
from unified_siglip_supp import (
    Config,
    apply_config_overrides,
    build_heatmap_target,
    load_experiment_from_yaml,
)


DEFAULT_CONFIG = "configs/unified_siglip_supp/single_config/model_test_geo_input_ids.yaml"
TARGET_HEIGHT = 200
TARGET_ANGLE = 0
PANEL_HEIGHT = 1024


def parse_satellite_id(raw_id: str) -> int:
    try:
        return int(Path(str(raw_id)).stem)
    except ValueError as exc:
        raise ValueError(f"Invalid image id: {raw_id}") from exc


def load_training_config(config_path: str) -> Dict[str, Any]:
    experiment = load_experiment_from_yaml(config_path, None)
    apply_config_overrides(experiment.get("config", {}))
    return experiment


def build_dataset(split: str):
    sat_size = (
        int(Config.UNIV_SAT_SIZE["height"]),
        int(Config.UNIV_SAT_SIZE["width"]),
    )
    return initialize_shifted_dataset_like_unified_siglip_supp(
        split=split,
        model_name=Config.MODEL_NAME,
        cache_dir=Config.CACHE_DIR,
        sat_target_size=sat_size,
    )


def find_sample(dataset, satellite_id: int) -> Optional[int]:
    for idx, sample in enumerate(dataset.samples):
        if (
            int(sample["satellite_id"]) == satellite_id
            and int(sample["height"]) == TARGET_HEIGHT
            and int(sample["angle"]) == TARGET_ANGLE
        ):
            return idx
    return None


def find_any_sample(dataset, satellite_id: int) -> Optional[Dict[str, Any]]:
    for sample in dataset.samples:
        if int(sample["satellite_id"]) == satellite_id:
            return sample
    return None


def tensor_image_to_pil(tensor: torch.Tensor, processor: Any) -> Image.Image:
    image = tensor.detach().cpu().float()
    if image.ndim != 3:
        raise ValueError(f"Expected image tensor shape (C, H, W), got {tuple(image.shape)}.")

    image = image.permute(1, 2, 0).numpy()
    do_normalize = bool(getattr(processor, "do_normalize", True))
    if do_normalize:
        mean = np.asarray(getattr(processor, "image_mean", [0.5, 0.5, 0.5]), dtype=np.float32)
        std = np.asarray(getattr(processor, "image_std", [0.5, 0.5, 0.5]), dtype=np.float32)
        image = image * std.reshape(1, 1, 3) + mean.reshape(1, 1, 3)
    elif image.max() > 1.0:
        image = image / 255.0

    image = np.clip(image, 0.0, 1.0)
    return Image.fromarray((image * 255.0).round().astype(np.uint8), mode="RGB")


def resize_to_height(image: Image.Image, target_height: int = PANEL_HEIGHT) -> Image.Image:
    width = max(1, int(round(image.width * target_height / max(1, image.height))))
    return image.resize((width, target_height), Image.Resampling.BILINEAR)


def draw_bbox(image: Image.Image, bbox: Sequence[float]) -> Image.Image:
    canvas = image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)
    x1, y1, x2, y2 = [float(v) for v in bbox]
    line_width = max(3, int(round(min(canvas.size) / 120)))
    draw.rectangle((x1, y1, x2, y2), outline=(255, 40, 40), width=line_width)
    return canvas


def heatmap_to_pil(heatmap: torch.Tensor) -> Image.Image:
    heatmap_np = heatmap.detach().cpu().float().squeeze().numpy()
    heatmap_np = heatmap_np - float(heatmap_np.min())
    denom = float(heatmap_np.max())
    if denom > 1e-6:
        heatmap_np = heatmap_np / denom

    red = heatmap_np
    green = 1.0 - np.abs(heatmap_np - 0.5) * 2.0
    blue = 1.0 - heatmap_np
    rgb = np.stack([red, np.clip(green, 0.0, 1.0), blue], axis=-1)
    return Image.fromarray((rgb * 255.0).round().astype(np.uint8), mode="RGB")


def compute_target_heatmap(bbox: torch.Tensor) -> torch.Tensor:
    sat_h = int(Config.UNIV_SAT_SIZE["height"])
    sat_w = int(Config.UNIV_SAT_SIZE["width"])
    heatmap_hw = (max(1, sat_h // 8), max(1, sat_w // 8))
    return build_heatmap_target(
        target_bbox=bbox.detach().cpu().float().view(1, 4),
        heatmap_hw=heatmap_hw,
        image_wh=(sat_w, sat_h),
        sigma=Config.HEATMAP_SIGMA,
        radius=Config.HEATMAP_RADIUS,
    )[0]


def get_font(size: int) -> ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def make_info_panel(
    satellite_id: int,
    split: str,
    sample: Dict[str, Any],
    output_bbox: Sequence[float],
    panel_height: int = PANEL_HEIGHT,
    panel_width: int = 720,
) -> Image.Image:
    panel = Image.new("RGB", (panel_width, panel_height), color=(248, 248, 244))
    draw = ImageDraw.Draw(panel)
    title_font = get_font(42)
    body_font = get_font(26)
    small_font = get_font(20)

    y = 36
    draw.text((32, y), f"Image {satellite_id:04d}", fill=(20, 20, 20), font=title_font)
    y += 70
    lines = [
        f"split: {split}",
        f"height: {int(sample['height'])}",
        f"angle: {int(sample['angle'])}",
        f"target heatmap: {int(Config.UNIV_SAT_SIZE['height']) // 8} x {int(Config.UNIV_SAT_SIZE['width']) // 8}",
        "bbox: "
        + ", ".join(f"{float(v):.1f}" for v in output_bbox),
    ]
    for line in lines:
        draw.text((32, y), line, fill=(45, 45, 45), font=body_font)
        y += 42

    y += 24
    draw.text((32, y), "height description", fill=(20, 20, 20), font=body_font)
    y += 44
    text = str(sample.get("text", "")).strip()
    for line in textwrap.wrap(text, width=42):
        draw.text((32, y), line, fill=(45, 45, 45), font=small_font)
        y += 30

    y = min(y + 40, panel_height - 170)
    path_lines = [
        f"drone: {Path(sample['drone_path']).name}",
        f"satellite: {Path(sample['satellite_path']).name}",
    ]
    for line in path_lines:
        for wrapped in textwrap.wrap(line, width=54):
            draw.text((32, y), wrapped, fill=(95, 95, 95), font=small_font)
            y += 28

    return panel


def compose_visualization(
    dataset,
    sample_index: int,
    split: str,
    satellite_id: int,
    output_path: Path,
) -> Path:
    sample = dataset.samples[sample_index]
    item = dataset[sample_index]

    drone_panel = resize_to_height(Image.open(sample["drone_path"]).convert("RGB"))
    satellite_panel = tensor_image_to_pil(
        item["search_pixel_values"],
        dataset.processor_sat,
    )
    bbox = item["bbox"].detach().cpu().float().tolist()
    satellite_panel = resize_to_height(draw_bbox(satellite_panel, bbox))
    heatmap_panel = resize_to_height(heatmap_to_pil(compute_target_heatmap(item["bbox"])))
    info_panel = make_info_panel(satellite_id, split, sample, bbox)

    panels = [drone_panel, heatmap_panel, satellite_panel, info_panel]
    padding = 18
    width = sum(panel.width for panel in panels) + padding * (len(panels) + 1)
    height = PANEL_HEIGHT + padding * 2
    canvas = Image.new("RGB", (width, height), color=(236, 236, 232))

    x = padding
    for panel in panels:
        canvas.paste(panel, (x, padding))
        x += panel.width + padding

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize one train/test dataset sample with drone, target heatmap, satellite bbox, and description."
    )
    parser.add_argument("image_id", help="Satellite image id, for example 2156 or 02156.")
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG,
        help="Unified SigLIP config used by train_group.sh.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output image path. Defaults to runs/visualizations/<id>_<split>_h200_a0.png.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    satellite_id = parse_satellite_id(args.image_id)
    load_training_config(args.config)

    train_dataset = build_dataset("train")
    test_dataset = build_dataset("test")
    datasets = [("train", train_dataset), ("test", test_dataset)]

    found_split = None
    found_dataset = None
    found_index = None
    for split, dataset in datasets:
        sample_index = find_sample(dataset, satellite_id)
        if sample_index is not None:
            found_split = split
            found_dataset = dataset
            found_index = sample_index
            break

    if found_dataset is None or found_index is None or found_split is None:
        available = []
        for split, dataset in datasets:
            sample = find_any_sample(dataset, satellite_id)
            if sample is not None:
                available.append(
                    f"{split}: found id, example height={sample['height']} angle={sample['angle']}"
                )
        detail = "; ".join(available) if available else "id not found in train or test"
        raise SystemExit(
            f"Could not find image {satellite_id:04d} with height={TARGET_HEIGHT}, "
            f"angle={TARGET_ANGLE}. {detail}."
        )

    output_path = (
        Path(args.output)
        if args.output
        else Path("runs/visualizations")
        / f"{satellite_id:04d}_{found_split}_h{TARGET_HEIGHT}_a{TARGET_ANGLE}.png"
    )
    saved_path = compose_visualization(
        dataset=found_dataset,
        sample_index=found_index,
        split=found_split,
        satellite_id=satellite_id,
        output_path=output_path,
    )
    print(f"Found image {satellite_id:04d} in {found_split} dataset.")
    print(f"Saved visualization: {saved_path}")


if __name__ == "__main__":
    main()
