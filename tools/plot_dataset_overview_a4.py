#!/usr/bin/env python3
"""Build an A4-landscape overview of geography and dataset characteristics."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-dataset-overview")

import matplotlib.pyplot as plt

from plot_drone_vision_embedding import (
    DEFAULT_ANGLE,
    DEFAULT_CACHE_DIR,
    DEFAULT_FEATURE_CACHE,
    DEFAULT_HEIGHT,
    DEFAULT_IMAGE_ROOT,
    DEFAULT_MODEL,
    DEFAULT_TSNE_CACHE,
    compute_tsne as compute_cached_tsne,
    load_image_records,
    load_or_encode_features,
    resolve_device,
    sampled_cosine_distances,
)
from plot_global_kml_coverage import (
    DEFAULT_WORLD_GEOJSON,
    draw_world,
    equal_earth,
    load_locations,
    map_boundary_path,
)
from plot_kml_coordinate_histograms import (
    ZONE_BOUNDS,
    ZONE_CODES,
    coordinate_array,
    neighbor_counts_within_radius,
    zone_indices,
)
from plot_local_retrieval_ambiguity import (
    DEFAULT_FOOTPRINT_SIDE_KM,
    DEFAULT_LOCATION_JSON,
    DEFAULT_RADIUS_KM,
    choose_query_index,
    draw_local_example,
    draw_neighbor_ccdf,
    load_metadata,
)
# --- Configuration ---
DEFAULT_TRAIN_DIR = Path("/media/data1/feihong/train_kml_2048")
DEFAULT_TEST_DIR = Path("/media/data1/feihong/kml_test_2")
DEFAULT_OUTPUT = Path(__file__).resolve().with_name("dataset_overview_a4.png")

# Bright pastel colors retain category identity after A4-scale reduction.
CARTOON = ("#FF986A", "#FFD166", "#69D2A4", "#62B6E7", "#B49CE5", "#F582AE")
BLUE_SHADES = ("#3976B7", "#66A1D2", "#8BC5DA", "#B9DDE0", "#E0EEE7")
INK = "#2F3440"
MUTED = "#66717E"
GRID = "#D9D3C5"
BACKGROUND = "#FFF8E8"
HAND_FONT = "Purisa"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dir", type=Path, default=DEFAULT_TRAIN_DIR)
    parser.add_argument("--test-dir", type=Path, default=DEFAULT_TEST_DIR)
    parser.add_argument("--world-geojson", type=Path, default=DEFAULT_WORLD_GEOJSON)
    parser.add_argument("--location-json", type=Path, default=DEFAULT_LOCATION_JSON)
    parser.add_argument("--ambiguity-radius-km", type=float, default=DEFAULT_RADIUS_KM)
    parser.add_argument("--footprint-side-km", type=float, default=DEFAULT_FOOTPRINT_SIDE_KM)
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--image-height", type=int, default=DEFAULT_HEIGHT)
    parser.add_argument("--image-angle", type=int, default=DEFAULT_ANGLE)
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--clusters", type=int, default=6)
    parser.add_argument("--pair-samples", type=int, default=100_000)
    parser.add_argument("--vision-feature-cache", type=Path, default=DEFAULT_FEATURE_CACHE)
    parser.add_argument("--tsne-cache", type=Path, default=DEFAULT_TSNE_CACHE)
    parser.add_argument("--recompute-vision-features", action="store_true")
    parser.add_argument("--recompute-tsne", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--pdf-output", type=Path, default=None)
    parser.add_argument("--notes-output", type=Path, default=None)
    parser.add_argument("--summary-output", type=Path, default=None)
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def panel_title(axis: plt.Axes, marker: str, title: str, size: float = 9.5) -> None:
    axis.set_title(
        f"({marker})  {title}", loc="left", fontsize=size,
        fontfamily=HAND_FONT, fontweight="bold", pad=5, color=INK,
    )


def style_small_axis(axis: plt.Axes) -> None:
    axis.set_facecolor(BACKGROUND)
    axis.grid(axis="y", color=GRID, linewidth=0.55, alpha=0.72, linestyle="--")
    axis.tick_params(labelsize=6.5, length=2.8, pad=2)
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["left", "bottom"]].set_linewidth(1.0)
    axis.set_axisbelow(True)


def draw_map(axis: plt.Axes, coordinates: np.ndarray, zones: np.ndarray, geojson: Path) -> None:
    boundary = draw_world(axis, geojson)
    for zone, (code, color) in enumerate(zip(ZONE_CODES, CARTOON[:5])):
        selected = zones == zone
        if not selected.any():
            continue
        x, y = equal_earth(coordinates[selected, 0], coordinates[selected, 1])
        points = axis.scatter(
            x, y, s=9.0, color=color, edgecolors=INK, linewidths=0.20,
            alpha=0.82, label=code, zorder=8, rasterized=True,
        )
        points.set_clip_path(boundary)
    vertices = map_boundary_path().vertices
    axis.set_xlim(vertices[:, 0].min() - 0.04, vertices[:, 0].max() + 0.04)
    axis.set_ylim(vertices[:, 1].min() - 0.03, vertices[:, 1].max() + 0.03)
    axis.set_aspect("equal", adjustable="box")
    axis.axis("off")
    panel_title(axis, "a", "Global location coverage", 11.0)
    axis.legend(
        loc="lower center", bbox_to_anchor=(0.5, -0.055), ncol=5,
        frameon=False, prop={"family": HAND_FONT, "size": 7.0},
        handletextpad=0.25, columnspacing=0.8,
    )


def draw_climate_hist(axis: plt.Axes, zones: np.ndarray) -> None:
    style_small_axis(axis)
    counts = np.bincount(zones, minlength=len(ZONE_CODES))
    shares = counts / len(zones) * 100
    axis.bar(ZONE_CODES, shares, color=CARTOON[:5], width=0.72, edgecolor=INK, linewidth=0.7)
    axis.set_ylim(0, max(shares) * 1.16)
    panel_title(axis, "b", "Climate zones", 8.6)
    axis.set_ylabel("Locations (%)", fontsize=7.0, labelpad=2)
    axis.set_xlabel("Zone", fontsize=7.0, labelpad=2)
    for index, share in enumerate(shares):
        if share >= 1:
            axis.text(index, share + 1.1, f"{share:.0f}%", ha="center", fontsize=6.4, color=INK)


def compute_vision_tsne(
    records: list[dict[str, str]],
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    device = resolve_device(args.device)
    features, signature = load_or_encode_features(
        records,
        args.model_name,
        args.cache_dir,
        device,
        args.batch_size,
        args.image_height,
        args.image_angle,
        args.vision_feature_cache,
        args.recompute_vision_features,
    )
    coordinates, labels = compute_cached_tsne(
        features,
        records,
        signature,
        args.model_name,
        args.image_height,
        args.image_angle,
        args.clusters,
        args.tsne_cache,
        args.recompute_tsne,
    )
    return coordinates, labels, features


def draw_tsne(axis: plt.Axes, coordinates: np.ndarray, labels: np.ndarray) -> None:
    axis.set_facecolor(BACKGROUND)
    for cluster in range(int(labels.max()) + 1):
        selected = labels == cluster
        axis.scatter(
            coordinates[selected, 0], coordinates[selected, 1],
            s=7.5, color=CARTOON[cluster % len(CARTOON)], alpha=0.75,
            edgecolors=INK, linewidths=0.10, label=f"C{cluster + 1}", rasterized=True,
        )
    panel_title(axis, "e", "SigLIP2 vision t-SNE", 11.0)
    axis.set_xticks([])
    axis.set_yticks([])
    axis.spines[["top", "right", "left", "bottom"]].set_visible(False)
    axis.legend(
        loc="lower center", bbox_to_anchor=(0.5, -0.06), ncol=6,
        frameon=False, prop={"family": HAND_FONT, "size": 7.0},
        handletextpad=0.2, columnspacing=0.7,
    )


def draw_vision_distance_distribution(
    axis: plt.Axes,
    distances: np.ndarray,
) -> None:
    style_small_axis(axis)
    axis.hist(
        distances,
        bins=36,
        color=CARTOON[3],
        edgecolor=INK,
        linewidth=0.45,
        alpha=0.88,
    )
    median = float(np.median(distances))
    axis.axvline(median, color="#EF6A5B", linewidth=1.7)
    panel_title(axis, "f", "Vision feature distance", 11.0)
    axis.set_xlabel("Cosine distance", fontsize=7.2, labelpad=2)
    axis.set_ylabel("Sampled image pairs", fontsize=7.2, labelpad=2)
    axis.text(
        0.96,
        0.92,
        f"median = {median:.3f}",
        transform=axis.transAxes,
        ha="right",
        va="top",
        fontsize=7.2,
        color="#C94F43",
    )


def create_figure(
    coordinates: np.ndarray,
    locations: list,
    zones: np.ndarray,
    neighbor_counts: np.ndarray,
    ambiguity_query_index: int,
    location_metadata: dict[str, dict[str, str]],
    tsne_coordinates: np.ndarray,
    tsne_labels: np.ndarray,
    vision_distances: np.ndarray,
    world_geojson: Path,
    ambiguity_radius_km: float,
    footprint_side_km: float,
) -> plt.Figure:
    plt.rcParams.update({
        "font.family": HAND_FONT,
        "font.size": 7,
        "axes.linewidth": 1.0,
        "path.sketch": None,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    figure = plt.figure(figsize=(11.69, 8.27), facecolor=BACKGROUND)

    # Deliberately asymmetric, poster-like placement.
    map_axis = figure.add_axes((0.040, 0.500, 0.465, 0.440))
    draw_map(map_axis, coordinates, zones, world_geojson)

    climate_axis = figure.add_axes((0.040, 0.230, 0.145, 0.205))
    local_axis = figure.add_axes((0.200, 0.230, 0.145, 0.205))
    ambiguity_axis = figure.add_axes((0.360, 0.230, 0.145, 0.205))
    for axis in (climate_axis, local_axis, ambiguity_axis):
        axis.set_box_aspect(1)
    draw_climate_hist(climate_axis, zones)
    draw_local_example(
        local_axis, coordinates, locations, neighbor_counts,
        ambiguity_query_index, location_metadata, ambiguity_radius_km,
        1.18, footprint_side_km, 7, marker="c", title_size=8.2, compact=True,
        title="Dense-city example",
    )
    draw_neighbor_ccdf(
        ambiguity_axis, neighbor_counts, ambiguity_radius_km,
        marker="d", title_size=8.2, compact=True,
        title="Neighbors within 1 km",
    )

    tsne_axis = figure.add_axes((0.550, 0.545, 0.420, 0.390))
    draw_tsne(tsne_axis, tsne_coordinates, tsne_labels)

    draw_vision_distance_distribution(
        figure.add_axes((0.590, 0.095, 0.385, 0.365)),
        vision_distances,
    )
    return figure


def write_notes(
    path: Path,
    num_locations: int,
    num_vision_images: int,
    vision_model_name: str,
    vision_feature_dimension: int,
    image_height: int,
    image_angle: int,
    vision_distances: np.ndarray,
    neighbor_counts: np.ndarray,
    ambiguity_radius_km: float,
    ambiguity_location_id: int,
    ambiguity_place: str,
) -> None:
    path.write_text(
        "A4 数据集总览——面板说明\n"
        "========================\n\n"
        f"(a) Global location coverage：合并训练与测试后的 {num_locations:,} 条 KML 地点记录。颜色 Z1–Z5 表示热带、亚热带、温带、冷温带和极地纬度带。\n"
        "(b) Climate zones：各纬度气候代理带的地点占比；它不是实测 Köppen 气候标签。\n"
        f"(c) Local candidate overlap：真实密集样例 {ambiguity_location_id:04d}（{ambiguity_place}）；"
        f"半径 {ambiguity_radius_km:g} km 内有 {neighbor_counts.max()} 个其他 candidate region。橙色为 query，"
        "灰蓝色为其他候选；方形 tile outline 仅为重叠关系示意，不是地理配准边界。\n"
        f"(d) Local ambiguity distribution：每个地点半径 {ambiguity_radius_km:g} km 内其他 candidate-region center 数量的 CCDF。"
        f"中位数 {np.median(neighbor_counts):.0f}、P75={np.quantile(neighbor_counts, 0.75):.0f}、"
        f"P90={np.quantile(neighbor_counts, 0.90):.0f}、均值={neighbor_counts.mean():.2f}；"
        f"{(neighbor_counts == 0).mean():.1%} 的地点没有邻居。\n"
        f"(e) SigLIP2 vision t-SNE：每个地点固定取 {image_height} 米、{image_angle}° 的无人机图像，"
        f"用 {vision_model_name} 得到 {vision_feature_dimension} 维 L2 归一化 pooled vision feature；"
        f"共 {num_vision_images:,} 张图像（相对 KML 地点缺 {num_locations - num_vision_images:,} 张固定视角图），"
        "颜色 C1–C6 是无监督视觉分组。\n"
        f"(f) Vision feature distance：随机抽取 {len(vision_distances):,} 对固定视角图像，展示其"
        f" SigLIP2 特征余弦距离分布；中位数为 {np.median(vision_distances):.3f}。\n\n"
        "植被/气候解释边界：纬度只提供宏观气候代理，实际植被还取决于降水、海拔、土壤、洋流和土地利用。\n",
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()

    train, train_errors = load_locations(args.train_dir, "train", 1652, strict=False)
    test, test_errors = load_locations(args.test_dir, "test", 1652, strict=False)
    locations = train + test
    coordinates = coordinate_array(locations)
    zones = zone_indices(coordinates[:, 1])
    neighbor_counts = neighbor_counts_within_radius(coordinates, args.ambiguity_radius_km)
    ambiguity_query_index = choose_query_index(locations, neighbor_counts, None)
    location_metadata = load_metadata(args.location_json)

    image_records = load_image_records(
        args.image_root,
        args.image_height,
        args.image_angle,
    )
    tsne_coordinates, tsne_labels, vision_features = compute_vision_tsne(
        image_records,
        args,
    )
    vision_distances = sampled_cosine_distances(vision_features, args.pair_samples)

    figure = create_figure(
        coordinates, locations, zones, neighbor_counts,
        ambiguity_query_index, location_metadata,
        tsne_coordinates, tsne_labels,
        vision_distances, args.world_geojson,
        args.ambiguity_radius_km, args.footprint_side_km,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=args.dpi, facecolor=BACKGROUND)
    if args.pdf_output:
        args.pdf_output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(args.pdf_output, facecolor=BACKGROUND)
    plt.close(figure)

    notes_output = args.notes_output or args.output.with_name(f"{args.output.stem}_notes.txt")
    ambiguity_location = locations[ambiguity_query_index]
    ambiguity_detail = location_metadata.get(f"{ambiguity_location.location_id:04d}", {})
    ambiguity_place = ", ".join(
        value for value in (ambiguity_detail.get("city"), ambiguity_detail.get("country")) if value
    ) or "unknown place"
    write_notes(
        notes_output,
        len(locations),
        len(image_records),
        args.model_name,
        int(vision_features.shape[1]),
        args.image_height,
        args.image_angle,
        vision_distances,
        neighbor_counts,
        args.ambiguity_radius_km,
        ambiguity_location.location_id,
        ambiguity_place,
    )
    summary_output = args.summary_output or args.output.with_suffix(".json")
    summary_output.write_text(
        json.dumps(
            {
                "page": "A4 landscape",
                "num_kml_records": len(locations),
                "vision_tsne": {
                    "num_images": len(image_records),
                    "missing_fixed_view_images": len(locations) - len(image_records),
                    "height_m": args.image_height,
                    "angle_deg": args.image_angle,
                    "model_name": args.model_name,
                    "feature_dimension": int(vision_features.shape[1]),
                    "feature_cache": str(args.vision_feature_cache),
                    "tsne_cache": str(args.tsne_cache),
                    "cluster_sizes": {
                        str(index + 1): int((tsne_labels == index).sum())
                        for index in range(args.clusters)
                    },
                    "pairwise_cosine_distance": {
                        "num_sampled_pairs": len(vision_distances),
                        "mean": float(vision_distances.mean()),
                        "median": float(np.median(vision_distances)),
                        "p10": float(np.quantile(vision_distances, 0.10)),
                        "p90": float(np.quantile(vision_distances, 0.90)),
                    },
                },
                "local_candidate_ambiguity": {
                    "radius_km": args.ambiguity_radius_km,
                    "mean": float(neighbor_counts.mean()),
                    "median": float(np.median(neighbor_counts)),
                    "p75": float(np.quantile(neighbor_counts, 0.75)),
                    "p90": float(np.quantile(neighbor_counts, 0.90)),
                    "zero_neighbor_share": float((neighbor_counts == 0).mean()),
                    "max": int(neighbor_counts.max()),
                    "example_location_id": f"{ambiguity_location.location_id:04d}",
                    "example_place": ambiguity_detail,
                    "schematic_footprint_side_km": args.footprint_side_km,
                },
                "kml_parse_errors": train_errors + test_errors,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved {args.output}, {notes_output}, and {summary_output}")
    if args.pdf_output:
        print(f"Saved {args.pdf_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
