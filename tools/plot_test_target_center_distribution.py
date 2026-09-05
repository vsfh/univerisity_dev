#!/usr/bin/env python3
"""Plot test target-center density inside a normalized satellite tile."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-test-target-centers")

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Rectangle


# --- Configuration ---
DEFAULT_BBOX_FILE = Path("/media/data1/feihong/ckpt/bbox_test_2.json")
DEFAULT_TEST_SPLIT_FILE = Path("/media/data1/feihong/ckpt/test_2.txt")
DEFAULT_SATELLITE_ROOT = Path("/media/data1/feihong/img_test_2")
DEFAULT_DRONE_ROOT = Path("/media/data1/feihong/drone_img")
DEFAULT_TEXT_FILENAME = "qwen_6_28_description.json"
DEFAULT_OUTPUT = Path(__file__).resolve().with_name("test_target_center_distribution.png")

IMAGE_WIDTH_PX = 3840
IMAGE_HEIGHT_PX = 2160
TILE_WIDTH_M = 1183.0
TILE_HEIGHT_M = 660.0
CANONICAL_BBOX_HEIGHT_M = 150
VALID_HEIGHTS = (150, 200, 250, 300)
VALID_ANGLES = (0, 45, 90, 135, 180, 225, 270, 315)

BACKGROUND = "#FCFBF8"
INK = "#26323B"
MUTED = "#68747E"
GRID = "#D8DEE1"
BLUE = "#4E7E9D"
BLUE_DARK = "#204A68"
ORANGE = "#E8752E"

HEATMAP_CMAP = LinearSegmentedColormap.from_list(
    "target_density_blue",
    ("#EDF3F5", "#BDD3DD", "#78A8BB", "#3D7796", "#173F5B"),
)


@dataclass(frozen=True)
class TargetCenter:
    location_id: str
    center_x_px: float
    center_y_px: float
    east_m: float
    north_m: float
    displacement_m: float
    bbox: tuple[float, float, float, float]
    satellite_path: str
    drone_dir: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bbox-file", type=Path, default=DEFAULT_BBOX_FILE)
    parser.add_argument("--test-split-file", type=Path, default=DEFAULT_TEST_SPLIT_FILE)
    parser.add_argument("--satellite-root", type=Path, default=DEFAULT_SATELLITE_ROOT)
    parser.add_argument("--drone-root", type=Path, default=DEFAULT_DRONE_ROOT)
    parser.add_argument("--text-filename", default=DEFAULT_TEXT_FILENAME)
    parser.add_argument("--image-width-px", type=int, default=IMAGE_WIDTH_PX)
    parser.add_argument("--image-height-px", type=int, default=IMAGE_HEIGHT_PX)
    parser.add_argument("--tile-width-m", type=float, default=TILE_WIDTH_M)
    parser.add_argument("--tile-height-m", type=float, default=TILE_HEIGHT_M)
    parser.add_argument("--canonical-bbox-height", type=int, default=CANONICAL_BBOX_HEIGHT_M)
    parser.add_argument("--gridsize-x", type=int, default=54)
    parser.add_argument("--gridsize-y", type=int, default=30)
    parser.add_argument("--marginal-bins", type=int, default=42)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--pdf-output", type=Path, default=None)
    parser.add_argument("--csv-output", type=Path, default=None)
    parser.add_argument("--summary-output", type=Path, default=None)
    parser.add_argument("--notes-output", type=Path, default=None)
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def parse_split_ids(path: Path) -> set[int]:
    if not path.is_file():
        raise FileNotFoundError(f"Test split file does not exist: {path}")
    identifiers: set[int] = set()
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        first_field = raw_line.strip().split(",")[0].strip()
        if not first_field:
            continue
        candidates = (first_field, Path(first_field).name, Path(first_field).stem)
        for candidate in candidates:
            try:
                identifiers.add(int(candidate))
                break
            except ValueError:
                continue
    return identifiers


def list_satellite_paths(root: Path) -> dict[int, Path]:
    if not root.is_dir():
        raise NotADirectoryError(f"Satellite root does not exist: {root}")
    paths: dict[int, Path] = {}
    for pattern in ("*.png", "*.jpg", "*.jpeg", "*.webp"):
        for path in root.glob(pattern):
            try:
                paths[int(path.stem)] = path
            except ValueError:
                continue
    return paths


def load_height_texts(path: Path) -> dict[int, list[str]]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    segments = payload.get("description_segments")
    if not isinstance(segments, dict):
        return {}

    output: dict[int, list[str]] = {}
    for raw_height, raw_items in segments.items():
        try:
            height = int(raw_height)
        except (TypeError, ValueError):
            continue
        if not isinstance(raw_items, list):
            continue
        texts = [
            item["text"].strip()
            for item in raw_items
            if isinstance(item, dict)
            and isinstance(item.get("text"), str)
            and item["text"].strip()
        ]
        if texts:
            output[height] = texts
    return output


def parse_drone_view(path: Path) -> tuple[int, int] | None:
    parts = path.stem.split("_")
    if len(parts) != 2:
        return None
    try:
        return int(parts[0]), int(parts[1])
    except ValueError:
        return None


def has_valid_test_query(
    drone_dir: Path,
    height_texts: dict[int, list[str]],
    bbox_by_height: dict[str, Any],
) -> bool:
    for drone_path in drone_dir.glob("*.png"):
        parsed = parse_drone_view(drone_path)
        if parsed is None:
            continue
        height, angle = parsed
        bbox = bbox_by_height.get(str(height))
        if (
            height in VALID_HEIGHTS
            and angle in VALID_ANGLES
            and height in height_texts
            and isinstance(bbox, list)
            and len(bbox) == 4
        ):
            return True
    return False


def validate_bbox(raw_bbox: Any, location_id: str, height: int) -> tuple[float, float, float, float]:
    if not isinstance(raw_bbox, list) or len(raw_bbox) != 4:
        raise ValueError(f"Invalid bbox for ID {location_id}, height {height}: {raw_bbox!r}")
    bbox = tuple(float(value) for value in raw_bbox)
    if not all(math.isfinite(value) for value in bbox):
        raise ValueError(f"Non-finite bbox for ID {location_id}, height {height}: {bbox}")
    x1, y1, x2, y2 = bbox
    if not (x1 < x2 and y1 < y2):
        raise ValueError(f"Non-positive bbox for ID {location_id}, height {height}: {bbox}")
    return bbox


def build_target_centers(args: argparse.Namespace) -> tuple[list[TargetCenter], dict[str, Any]]:
    bbox_payload = json.loads(args.bbox_file.read_text(encoding="utf-8"))
    if not isinstance(bbox_payload, dict):
        raise TypeError(f"Expected a JSON object in {args.bbox_file}.")

    split_ids = parse_split_ids(args.test_split_file)
    satellite_paths = list_satellite_paths(args.satellite_root)
    candidate_ids = sorted(split_ids.intersection(satellite_paths))
    targets: list[TargetCenter] = []
    skipped = {"missing_drone_directory": [], "missing_valid_text_or_view": [], "missing_bbox": []}

    for numeric_id in candidate_ids:
        location_id = f"{numeric_id:04d}"
        drone_dir = args.drone_root / location_id
        if not drone_dir.is_dir():
            skipped["missing_drone_directory"].append(location_id)
            continue
        bbox_by_height = bbox_payload.get(location_id)
        if not isinstance(bbox_by_height, dict):
            skipped["missing_bbox"].append(location_id)
            continue
        height_texts = load_height_texts(drone_dir / args.text_filename)
        if not has_valid_test_query(drone_dir, height_texts, bbox_by_height):
            skipped["missing_valid_text_or_view"].append(location_id)
            continue

        bbox = validate_bbox(
            bbox_by_height.get(str(args.canonical_bbox_height)),
            location_id,
            args.canonical_bbox_height,
        )
        x1, y1, x2, y2 = bbox
        center_x_px = (x1 + x2) / 2
        center_y_px = (y1 + y2) / 2
        east_m = (center_x_px - args.image_width_px / 2) * args.tile_width_m / args.image_width_px
        north_m = (args.image_height_px / 2 - center_y_px) * args.tile_height_m / args.image_height_px
        displacement_m = math.hypot(east_m, north_m)
        targets.append(
            TargetCenter(
                location_id=location_id,
                center_x_px=center_x_px,
                center_y_px=center_y_px,
                east_m=east_m,
                north_m=north_m,
                displacement_m=displacement_m,
                bbox=bbox,
                satellite_path=str(satellite_paths[numeric_id]),
                drone_dir=str(drone_dir),
            )
        )

    audit = {
        "bbox_json_ids": len(bbox_payload),
        "test_split_ids": len(split_ids),
        "satellite_image_ids": len(satellite_paths),
        "candidate_ids_in_split_and_satellite_root": len(candidate_ids),
        "included_unique_test_regions": len(targets),
        "skipped": skipped,
    }
    return targets, audit


def target_arrays(targets: Iterable[TargetCenter]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    materialized = list(targets)
    east = np.asarray([item.east_m for item in materialized], dtype=np.float64)
    north = np.asarray([item.north_m for item in materialized], dtype=np.float64)
    displacement = np.asarray([item.displacement_m for item in materialized], dtype=np.float64)
    return east, north, displacement


def create_figure(
    targets: list[TargetCenter],
    args: argparse.Namespace,
) -> tuple[plt.Figure, dict[str, float]]:
    east, north, displacement = target_arrays(targets)
    median_displacement = float(np.median(displacement))
    p90_displacement = float(np.quantile(displacement, 0.90))
    half_width = args.tile_width_m / 2
    half_height = args.tile_height_m / 2

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    figure = plt.figure(figsize=(12.0, 7.25), facecolor=BACKGROUND)
    grid = figure.add_gridspec(
        2, 2,
        width_ratios=(6.0, 1.25),
        height_ratios=(1.15, 4.8),
        left=0.085, right=0.92, bottom=0.12, top=0.83,
        hspace=0.04, wspace=0.04,
    )
    top_axis = figure.add_subplot(grid[0, 0])
    main_axis = figure.add_subplot(grid[1, 0], sharex=top_axis)
    right_axis = figure.add_subplot(grid[1, 1], sharey=main_axis)
    info_axis = figure.add_subplot(grid[0, 1])

    for axis in (top_axis, main_axis, right_axis, info_axis):
        axis.set_facecolor(BACKGROUND)

    extent = (-half_width, half_width, -half_height, half_height)
    hexbin = main_axis.hexbin(
        east,
        north,
        gridsize=(args.gridsize_x, args.gridsize_y),
        extent=extent,
        mincnt=1,
        cmap=HEATMAP_CMAP,
        norm=LogNorm(vmin=1),
        linewidths=0.16,
        edgecolors="#F7FAFA",
        zorder=2,
    )
    main_axis.add_patch(
        Rectangle(
            (-half_width, -half_height),
            args.tile_width_m,
            args.tile_height_m,
            fill=False,
            edgecolor=INK,
            linewidth=1.7,
            zorder=8,
        )
    )
    main_axis.axvline(0, color=INK, linewidth=1.05, linestyle=(0, (4, 3)), alpha=0.78, zorder=7)
    main_axis.axhline(0, color=INK, linewidth=1.05, linestyle=(0, (4, 3)), alpha=0.78, zorder=7)
    main_axis.add_patch(
        Circle((0, 0), median_displacement, fill=False, edgecolor=ORANGE, linewidth=1.7, zorder=6)
    )
    main_axis.add_patch(
        Circle(
            (0, 0), p90_displacement, fill=False, edgecolor=ORANGE,
            linewidth=1.7, linestyle=(0, (5, 3)), zorder=6,
        )
    )
    main_axis.scatter([0], [0], s=28, marker="+", color=INK, linewidths=1.2, zorder=9)
    main_axis.set_xlim(-half_width, half_width)
    main_axis.set_ylim(-half_height, half_height)
    main_axis.set_aspect("equal", adjustable="box")
    main_axis.set_xlabel("East–west displacement from tile center (m)", fontsize=10)
    main_axis.set_ylabel("North–south displacement from tile center (m)", fontsize=10)
    main_axis.grid(color=GRID, linewidth=0.5, alpha=0.40, zorder=0)
    main_axis.tick_params(labelsize=8.5)

    legend_handles = [
        Line2D([0], [0], color=INK, linestyle=(0, (4, 3)), linewidth=1.2, label="Tile center crosshair"),
        Line2D([0], [0], color=ORANGE, linewidth=1.7, label=f"Median radial displacement: {median_displacement:.0f} m"),
        Line2D([0], [0], color=ORANGE, linestyle=(0, (5, 3)), linewidth=1.7, label=f"P90 radial displacement: {p90_displacement:.0f} m"),
    ]
    main_axis.legend(
        handles=legend_handles,
        loc="lower left",
        frameon=True,
        framealpha=0.92,
        facecolor="white",
        edgecolor=GRID,
        fontsize=8.2,
    )

    bins_x = np.linspace(-half_width, half_width, args.marginal_bins + 1)
    bins_y = np.linspace(-half_height, half_height, args.marginal_bins + 1)
    top_axis.hist(east, bins=bins_x, color=BLUE, edgecolor="white", linewidth=0.35, alpha=0.92)
    top_axis.axvline(0, color=INK, linewidth=1.0, linestyle=(0, (4, 3)), alpha=0.72)
    top_axis.set_ylabel("Count", fontsize=8)
    top_axis.tick_params(axis="x", labelbottom=False, bottom=False)
    top_axis.tick_params(axis="y", labelsize=7, length=2.5)
    top_axis.spines[["top", "right", "bottom"]].set_visible(False)
    top_axis.grid(axis="y", color=GRID, linewidth=0.5, linestyle="--", alpha=0.65)

    right_axis.hist(
        north,
        bins=bins_y,
        orientation="horizontal",
        color=BLUE,
        edgecolor="white",
        linewidth=0.35,
        alpha=0.92,
    )
    right_axis.axhline(0, color=INK, linewidth=1.0, linestyle=(0, (4, 3)), alpha=0.72)
    right_axis.set_xlabel("Count", fontsize=8)
    right_axis.tick_params(axis="y", labelleft=False, left=False)
    right_axis.tick_params(axis="x", labelsize=7, length=2.5)
    right_axis.spines[["top", "right", "left"]].set_visible(False)
    right_axis.grid(axis="x", color=GRID, linewidth=0.5, linestyle="--", alpha=0.65)

    info_axis.axis("off")
    info_axis.text(
        0.02,
        0.95,
        "Radial displacement",
        ha="left",
        va="top",
        fontsize=9.5,
        fontweight="bold",
        color=INK,
    )
    info_axis.text(
        0.02,
        0.70,
        f"Median   {median_displacement:.0f} m\nP90       {p90_displacement:.0f} m",
        ha="left",
        va="top",
        fontsize=9.2,
        color=ORANGE,
        linespacing=1.55,
    )
    colorbar = figure.colorbar(hexbin, ax=info_axis, orientation="horizontal", fraction=0.20, pad=0.05)
    heatmap_max = max(1, int(np.max(hexbin.get_array())))
    colorbar_ticks = sorted({
        max(1, int(round(value)))
        for value in np.geomspace(1, heatmap_max, min(4, heatmap_max))
    })
    colorbar.set_ticks(colorbar_ticks)
    colorbar.set_ticklabels([str(value) for value in colorbar_ticks])
    colorbar.minorticks_off()
    colorbar.set_label("Centers per hexagon (log scale)", fontsize=7.5)
    colorbar.ax.tick_params(labelsize=6.5, length=2)
    colorbar.outline.set_linewidth(0.6)

    figure.suptitle(
        "Test target-center distribution within the satellite tile",
        x=0.085,
        y=0.965,
        ha="left",
        fontsize=17,
        fontweight="bold",
        color=INK,
    )
    figure.text(
        0.085,
        0.915,
        f"{len(targets):,} unique test regions · 3,840 × 2,160 px mapped to "
        f"{args.tile_width_m:g} × {args.tile_height_m:g} m · north is up",
        ha="left",
        fontsize=9.4,
        color=MUTED,
    )
    figure.text(
        0.085,
        0.875,
        f"Center source: unclipped {args.canonical_bbox_height} m bbox in {args.bbox_file}",
        ha="left",
        fontsize=8.2,
        color=MUTED,
    )

    statistics = {
        "median_displacement_m": median_displacement,
        "p90_displacement_m": p90_displacement,
        "mean_displacement_m": float(displacement.mean()),
        "mean_east_m": float(east.mean()),
        "median_east_m": float(np.median(east)),
        "mean_north_m": float(north.mean()),
        "median_north_m": float(np.median(north)),
        "max_displacement_m": float(displacement.max()),
    }
    return figure, statistics


def write_csv(path: Path, targets: list[TargetCenter]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow((
            "location_id", "center_x_px", "center_y_px", "east_m", "north_m",
            "radial_displacement_m", "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2",
            "satellite_path", "drone_dir",
        ))
        for item in targets:
            writer.writerow((
                item.location_id,
                item.center_x_px,
                item.center_y_px,
                item.east_m,
                item.north_m,
                item.displacement_m,
                *item.bbox,
                item.satellite_path,
                item.drone_dir,
            ))


def write_notes(
    path: Path,
    args: argparse.Namespace,
    statistics: dict[str, float],
    audit: dict[str, Any],
) -> None:
    skipped = audit["skipped"]
    path.write_text(
        "测试集 target center 空间分布——图表说明\n"
        "=======================================\n\n"
        f"数据源：{args.bbox_file}\n"
        f"测试 region：{audit['included_unique_test_regions']:,} 个。原始 JSON/测试划分有 "
        f"{audit['bbox_json_ids']:,}/{audit['test_split_ids']:,} 个 ID；按 dataset.py 的有效样本条件，"
        f"排除缺 drone 目录 {len(skipped['missing_drone_directory'])} 个、缺有效文本或视图 "
        f"{len(skipped['missing_valid_text_or_view'])} 个、缺 bbox {len(skipped['missing_bbox'])} 个。\n\n"
        f"坐标转换：原始 satellite 图像 {args.image_width_px}×{args.image_height_px} px 线性映射到 "
        f"{args.tile_width_m:g}×{args.tile_height_m:g} m。图像中心映射为 (0, 0)，向东和向北为正。\n"
        f"每个 region 只统计一次，中心取 {args.canonical_bbox_height} m bbox 的几何中心。该层 bbox "
        "均未被图像边界裁切；更大高度的部分 bbox 会被裁切，其几何中心不能可靠代表原始 target center。\n\n"
        f"径向 displacement：median={statistics['median_displacement_m']:.2f} m，"
        f"P90={statistics['p90_displacement_m']:.2f} m，mean={statistics['mean_displacement_m']:.2f} m，"
        f"max={statistics['max_displacement_m']:.2f} m。\n"
        f"方向中心：east mean={statistics['mean_east_m']:.2f} m、median={statistics['median_east_m']:.2f} m；"
        f"north mean={statistics['mean_north_m']:.2f} m、median={statistics['median_north_m']:.2f} m。\n\n"
        "主图为二维 hexbin，颜色使用对数计数；上方和右侧分别为东西、南北方向的边缘直方图。"
        "中心十字虚线表示 tile 几何中心，橙色实线/虚线圆分别表示 median/P90 径向 displacement。\n",
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    if args.image_width_px <= 0 or args.image_height_px <= 0:
        raise ValueError("Image dimensions must be positive.")
    if args.tile_width_m <= 0 or args.tile_height_m <= 0:
        raise ValueError("Tile dimensions must be positive.")

    targets, audit = build_target_centers(args)
    if not targets:
        raise RuntimeError("No valid test target centers were found.")
    figure, statistics = create_figure(targets, args)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=args.dpi, facecolor=BACKGROUND)
    pdf_output = args.pdf_output or args.output.with_suffix(".pdf")
    pdf_output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(pdf_output, facecolor=BACKGROUND)
    plt.close(figure)

    csv_output = args.csv_output or args.output.with_suffix(".csv")
    summary_output = args.summary_output or args.output.with_suffix(".json")
    notes_output = args.notes_output or args.output.with_name(f"{args.output.stem}_notes.txt")
    write_csv(csv_output, targets)
    summary = {
        "source": {
            "bbox_file": str(args.bbox_file),
            "test_split_file": str(args.test_split_file),
            "satellite_root": str(args.satellite_root),
            "drone_root": str(args.drone_root),
        },
        "coordinate_system": {
            "source_image_pixels": [args.image_width_px, args.image_height_px],
            "normalized_tile_meters": [args.tile_width_m, args.tile_height_m],
            "origin": "tile center",
            "positive_x": "east",
            "positive_y": "north",
            "canonical_bbox_height_m": args.canonical_bbox_height,
        },
        "audit": audit,
        "statistics": statistics,
    }
    summary_output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_notes(notes_output, args, statistics, audit)

    print(
        f"Saved {args.output}, {pdf_output}, {csv_output}, {summary_output}, and {notes_output}\n"
        f"n={len(targets):,}; median displacement={statistics['median_displacement_m']:.2f} m; "
        f"P90 displacement={statistics['p90_displacement_m']:.2f} m"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
