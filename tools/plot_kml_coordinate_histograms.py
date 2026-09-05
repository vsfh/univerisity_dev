#!/usr/bin/env python3
"""Summarize all KML locations with geographic-zone and distance views."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Sequence

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-kml-histograms")

import matplotlib.pyplot as plt

from plot_global_kml_coverage import (
    DEFAULT_WORLD_GEOJSON,
    Location,
    draw_world,
    equal_earth,
    load_locations,
    map_boundary_path,
)


# --- Configuration ---
DEFAULT_TRAIN_DIR = Path("/media/data1/feihong/train_kml_2048")
DEFAULT_TEST_DIR = Path("/media/data1/feihong/kml_test_2")
DEFAULT_OUTPUT = Path(__file__).resolve().with_name("kml_coordinate_histograms.png")
EARTH_RADIUS_KM = 6371.0088

# Latitude-based geographic proxies, not observed Köppen climate classes.
ZONE_BOUNDS = np.asarray((0.0, 23.436, 35.0, 55.0, 66.563, 90.0))
ZONE_CODES = ("Z1", "Z2", "Z3", "Z4", "Z5")
ZONE_LABELS = ("Z1\nTropical", "Z2\nSubtropical", "Z3\nTemperate", "Z4\nCool", "Z5\nPolar")
ZONE_COLORS = ("#C7863C", "#C6A43A", "#5D9B69", "#568BA7", "#7A739E")

# Mean great-circle distance from each record to all other records.
MEAN_DISTANCE_BOUNDS = np.asarray((0.0, 5000.0, 6000.0, 7000.0, 10000.0, np.inf))
MEAN_DISTANCE_CODES = ("D1", "D2", "D3", "D4", "D5")
MEAN_DISTANCE_LABELS = ("D1\n<5k", "D2\n5–6k", "D3\n6–7k", "D4\n7–10k", "D5\n≥10k")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dir", type=Path, default=DEFAULT_TRAIN_DIR)
    parser.add_argument("--test-dir", type=Path, default=DEFAULT_TEST_DIR)
    parser.add_argument("--inherited-count", type=int, default=1652)
    parser.add_argument("--world-geojson", type=Path, default=DEFAULT_WORLD_GEOJSON)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--pdf-output", type=Path, default=None)
    parser.add_argument("--csv-output", type=Path, default=None)
    parser.add_argument("--summary-output", type=Path, default=None)
    parser.add_argument("--notes-output", type=Path, default=None)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def coordinate_array(locations: Sequence[Location]) -> np.ndarray:
    return np.asarray([(item.longitude, item.latitude) for item in locations], dtype=np.float64)


def minimum_haversine_distances(coordinates: np.ndarray, chunk_size: int = 256) -> np.ndarray:
    radians = np.deg2rad(coordinates)
    reference_lon = radians[:, 0][None, :]
    reference_lat = radians[:, 1][None, :]
    output = np.empty(len(coordinates), dtype=np.float64)

    for start in range(0, len(coordinates), chunk_size):
        stop = min(start + chunk_size, len(coordinates))
        lon = radians[start:stop, 0][:, None]
        lat = radians[start:stop, 1][:, None]
        a = (
            np.sin((reference_lat - lat) / 2) ** 2
            + np.cos(lat) * np.cos(reference_lat) * np.sin((reference_lon - lon) / 2) ** 2
        )
        distances = 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
        rows = np.arange(stop - start)
        distances[rows, start + rows] = np.inf
        output[start:stop] = distances.min(axis=1)
    return output


def neighbor_counts_within_radius(
    coordinates: np.ndarray,
    radius_km: float = 1.0,
    chunk_size: int = 256,
) -> np.ndarray:
    """Count other coordinate records within ``radius_km`` of each record."""
    if radius_km <= 0:
        raise ValueError(f"radius_km must be positive, got {radius_km}.")

    radians = np.deg2rad(coordinates)
    reference_lon = radians[:, 0][None, :]
    reference_lat = radians[:, 1][None, :]
    output = np.empty(len(coordinates), dtype=np.int32)

    for start in range(0, len(coordinates), chunk_size):
        stop = min(start + chunk_size, len(coordinates))
        lon = radians[start:stop, 0][:, None]
        lat = radians[start:stop, 1][:, None]
        a = (
            np.sin((reference_lat - lat) / 2) ** 2
            + np.cos(lat) * np.cos(reference_lat) * np.sin((reference_lon - lon) / 2) ** 2
        )
        distances = 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
        # Every row includes itself at distance zero, so remove exactly one.
        output[start:stop] = np.count_nonzero(distances <= radius_km, axis=1) - 1
    return output


def mean_haversine_distances(coordinates: np.ndarray, chunk_size: int = 256) -> np.ndarray:
    """Mean great-circle distance from each point to every other point."""
    radians = np.deg2rad(coordinates)
    reference_lon = radians[:, 0][None, :]
    reference_lat = radians[:, 1][None, :]
    output = np.empty(len(coordinates), dtype=np.float64)

    for start in range(0, len(coordinates), chunk_size):
        stop = min(start + chunk_size, len(coordinates))
        lon = radians[start:stop, 0][:, None]
        lat = radians[start:stop, 1][:, None]
        a = (
            np.sin((reference_lat - lat) / 2) ** 2
            + np.cos(lat) * np.cos(reference_lat) * np.sin((reference_lon - lon) / 2) ** 2
        )
        distances = 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
        output[start:stop] = distances.sum(axis=1) / (len(coordinates) - 1)
    return output


def zone_indices(latitudes: np.ndarray) -> np.ndarray:
    return np.digitize(np.abs(latitudes), ZONE_BOUNDS[1:-1], right=False)


def mean_distance_indices(distances: np.ndarray) -> np.ndarray:
    return np.digitize(distances, MEAN_DISTANCE_BOUNDS[1:-1], right=False)


def style_axis(axis: plt.Axes) -> None:
    axis.set_facecolor("#FCFCFA")
    axis.grid(axis="y", color="#D9DEE3", linewidth=0.55, alpha=0.72)
    axis.spines[["top", "right"]].set_visible(False)


def draw_location_map(
    axis: plt.Axes,
    coordinates: np.ndarray,
    zones: np.ndarray,
    world_geojson: Path,
) -> None:
    boundary = draw_world(axis, world_geojson)
    for zone, (code, color) in enumerate(zip(ZONE_CODES, ZONE_COLORS)):
        selected = zones == zone
        if not selected.any():
            continue
        x, y = equal_earth(coordinates[selected, 0], coordinates[selected, 1])
        points = axis.scatter(
            x, y, s=8.5, color=color, edgecolors="white", linewidths=0.15,
            alpha=0.72, label=code, zorder=8, rasterized=True,
        )
        points.set_clip_path(boundary)
    vertices = map_boundary_path().vertices
    axis.set_xlim(vertices[:, 0].min() - 0.04, vertices[:, 0].max() + 0.04)
    axis.set_ylim(vertices[:, 1].min() - 0.03, vertices[:, 1].max() + 0.03)
    axis.set_aspect("equal", adjustable="box")
    axis.axis("off")
    axis.set_title("(G1)  Global location coverage", loc="left", fontsize=12, fontweight="bold")
    axis.legend(loc="lower center", bbox_to_anchor=(0.5, -0.04), ncol=5, frameon=False, fontsize=8)


def create_figure(
    coordinates: np.ndarray,
    zones: np.ndarray,
    nearest: np.ndarray,
    mean_distances: np.ndarray,
    world_geojson: Path,
) -> plt.Figure:
    figure, axes = plt.subplots(2, 2, figsize=(13.2, 8.7))
    figure.patch.set_facecolor("#FCFCFA")
    map_axis, zone_axis, distance_axis, density_axis = axes.flat
    draw_location_map(map_axis, coordinates, zones, world_geojson)
    for axis in (zone_axis, distance_axis, density_axis):
        style_axis(axis)

    zone_counts = np.bincount(zones, minlength=len(ZONE_CODES))
    zone_shares = zone_counts / len(zones) * 100
    zone_axis.bar(ZONE_LABELS, zone_shares, color=ZONE_COLORS, width=0.72)
    zone_axis.set_title("(G2)  Latitude-based climate zones", loc="left", fontsize=12, fontweight="bold")
    zone_axis.set_ylabel("Locations (%)")
    for index, share in enumerate(zone_shares):
        if share > 0:
            zone_axis.text(index, share + 1.0, f"{share:.1f}%", ha="center", va="bottom", fontsize=8)

    positive = nearest[nearest > 0]
    plot_floor = max(float(positive.min()), 0.01)
    bins = np.geomspace(plot_floor, float(nearest.max()) * 1.01, 34)
    distance_axis.hist(
        np.maximum(nearest, plot_floor), bins=bins,
        color="#6F91B7", edgecolor="white", linewidth=0.45,
    )
    distance_axis.axvline(np.median(nearest), color="#C85C3C", linewidth=1.7)
    distance_axis.set_xscale("log")
    distance_axis.set_title("(G3)  Nearest-location distance", loc="left", fontsize=12, fontweight="bold")
    distance_axis.set_xlabel("Nearest distance (km)")
    distance_axis.set_ylabel("Locations")
    distance_axis.text(
        0.98, 0.94, f"Median {np.median(nearest):.2f} km",
        transform=distance_axis.transAxes, ha="right", va="top", fontsize=8.5, color="#A64832",
    )

    density_counts = np.bincount(mean_distance_indices(mean_distances), minlength=len(MEAN_DISTANCE_CODES))
    density_axis.bar(
        MEAN_DISTANCE_LABELS,
        density_counts / len(mean_distances) * 100,
        color=("#355F8A", "#547DA1", "#7E9DB5", "#AAB8C0", "#D0D3D2"),
        width=0.72,
    )
    density_axis.set_title("(G4)  Mean distance to all locations", loc="left", fontsize=12, fontweight="bold")
    density_axis.set_xlabel("Class / mean distance (thousand km)")
    density_axis.set_ylabel("Locations (%)")

    figure.suptitle(
        "Location characteristics",
        x=0.055, y=0.982, ha="left", fontsize=17, fontweight="bold", color="#20262E",
    )
    figure.text(
        0.055, 0.944, f"All {len(coordinates):,} KML coordinates · geographic zones and inter-location distances",
        ha="left", fontsize=9.2, color="#65717C",
    )
    figure.subplots_adjust(left=0.075, right=0.985, top=0.88, bottom=0.075, hspace=0.34, wspace=0.22)
    return figure


def write_notes(
    path: Path,
    locations: Sequence[Location],
    zones: np.ndarray,
    nearest: np.ndarray,
    mean_distances: np.ndarray,
) -> None:
    zone_counts = np.bincount(zones, minlength=len(ZONE_CODES))
    mean_classes = mean_distance_indices(mean_distances)
    density_counts = np.bincount(mean_classes, minlength=len(MEAN_DISTANCE_CODES))
    zone_details = (
        ("Z1", "热带纬度带", "0–23.436°", "全年太阳辐射较强；在水分充足地区可能出现热带植被"),
        ("Z2", "亚热带纬度带", "23.436–35°", "根据水分条件，可能出现暖温性植被、灌丛、草地或农田"),
        ("Z3", "温带纬度带", "35–55°", "温带森林、草地及季节性明显的城市景观较常见"),
        ("Z4", "冷温带纬度带", "55–66.563°", "生长季缩短，寒温带针叶林或近北方林环境的可能性增加"),
        ("Z5", "极地纬度带", "66.563–90°", "苔原或极地环境的可能性增加"),
    )
    density_details = (
        ("D1", "< 5,000 km"), ("D2", "5,000–6,000 km"), ("D3", "6,000–7,000 km"),
        ("D4", "7,000–10,000 km"), ("D5", "≥ 10,000 km"),
    )
    lines = [
        "地点特征——标识说明",
        "====================",
        "",
        f"数据范围：合并全部 {len(locations):,} 条有效 KML 坐标记录，不区分训练集和测试集。",
        "",
        "G1：所有地点的全球空间分布。点的颜色表示由绝对纬度推导的 Z1–Z5 地理带。",
        "G2：Z1–Z5 在全部地点中的占比。",
    ]
    for index, (code, name, span, interpretation) in enumerate(zone_details):
        share = zone_counts[index] / len(locations) * 100
        lines.append(f"  {code}：{name}（{span}）；n={zone_counts[index]:,}，占 {share:.1f}%。{interpretation}。")
    lines.extend(
        [
            "",
            "G3：每条坐标到最近另一条坐标的大圆距离；红色竖线表示中位数。",
            f"  中位数={np.median(nearest):.3f} km；P10={np.quantile(nearest, 0.10):.3f} km；P90={np.quantile(nearest, 0.90):.3f} km。",
            "G4：先计算每个地点到其余全部地点的平均大圆距离，再划分为 D1–D5 五档。",
            f"  平均距离的总体中位数={np.median(mean_distances):.1f} km；范围={mean_distances.min():.1f}–{mean_distances.max():.1f} km。",
        ]
    )
    for index, (code, span) in enumerate(density_details):
        share = density_counts[index] / len(locations) * 100
        lines.append(f"  {code}：{span}；n={density_counts[index]:,}，占 {share:.1f}%。")
    lines.extend(
        [
            "",
            "解释边界：",
            "- Z1–Z5 是由纬度得到的地理代理分组，不是实测的 Köppen 气候分类。",
            "- 实际气候和植被还受海拔、降水、洋流、土壤及土地利用影响。",
            "- D1–D5 表示每个地点相对于整个全球地点集合的平均距离，而不是局部采样密度或最近邻距离。",
            "- 当前没有接入植被、土地覆盖或城市数据库，因此图中没有把这些代理指标当作真实环境标签。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    train, train_errors = load_locations(args.train_dir, "train", args.inherited_count, args.strict)
    test, test_errors = load_locations(args.test_dir, "test", args.inherited_count, args.strict)
    locations = train + test
    if not locations:
        raise RuntimeError("No valid KML coordinates were found.")

    coordinates = coordinate_array(locations)
    nearest = minimum_haversine_distances(coordinates)
    zones = zone_indices(coordinates[:, 1])
    mean_distances = mean_haversine_distances(coordinates)
    mean_classes = mean_distance_indices(mean_distances)

    figure = create_figure(coordinates, zones, nearest, mean_distances, args.world_geojson)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=args.dpi, bbox_inches="tight", pad_inches=0.12)
    if args.pdf_output:
        args.pdf_output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(args.pdf_output, bbox_inches="tight", pad_inches=0.12)
    plt.close(figure)

    csv_output = args.csv_output or args.output.with_suffix(".csv")
    csv_output.parent.mkdir(parents=True, exist_ok=True)
    with csv_output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("location_id", "longitude", "latitude", "zone", "nearest_location_km", "mean_distance_to_all_km", "mean_distance_class", "kml_path"))
        for item, zone, nearest_distance, mean_distance, distance_class in zip(locations, zones, nearest, mean_distances, mean_classes):
            writer.writerow((item.location_id, item.longitude, item.latitude, ZONE_CODES[zone], nearest_distance, mean_distance, MEAN_DISTANCE_CODES[distance_class], item.kml_path))

    zone_counts = np.bincount(zones, minlength=len(ZONE_CODES))
    density_counts = np.bincount(mean_classes, minlength=len(MEAN_DISTANCE_CODES))
    summary = {
        "input_directories": [str(args.train_dir), str(args.test_dir)],
        "num_coordinate_records": len(locations),
        "num_unique_coordinates_7dp": len({tuple(row) for row in np.round(coordinates, 7)}),
        "zone_counts": {code: int(zone_counts[i]) for i, code in enumerate(ZONE_CODES)},
        "zone_shares": {code: float(zone_counts[i] / len(locations)) for i, code in enumerate(ZONE_CODES)},
        "nearest_location_km": {
            "median": float(np.median(nearest)),
            "p10": float(np.quantile(nearest, 0.10)),
            "p90": float(np.quantile(nearest, 0.90)),
        },
        "mean_distance_to_all_km": {
            "min": float(mean_distances.min()),
            "median": float(np.median(mean_distances)),
            "max": float(mean_distances.max()),
        },
        "mean_distance_class_counts": {code: int(density_counts[i]) for i, code in enumerate(MEAN_DISTANCE_CODES)},
        "parse_errors": train_errors + test_errors,
    }
    summary_output = args.summary_output or args.output.with_suffix(".json")
    summary_output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    notes_output = args.notes_output or args.output.with_name(f"{args.output.stem}_notes.txt")
    write_notes(notes_output, locations, zones, nearest, mean_distances)

    print(
        f"Saved {args.output}, {csv_output}, {summary_output}, and {notes_output} "
        f"({len(locations):,} pooled locations; {len(train_errors) + len(test_errors)} parse errors)."
    )
    if args.pdf_output:
        print(f"Saved {args.pdf_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
