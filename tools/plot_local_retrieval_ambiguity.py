#!/usr/bin/env python3
"""Visualize local candidate-region ambiguity from KML center coordinates."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Any, Sequence

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-local-retrieval-ambiguity")

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle

from plot_global_kml_coverage import Location, load_locations
from plot_kml_coordinate_histograms import (
    EARTH_RADIUS_KM,
    coordinate_array,
    neighbor_counts_within_radius,
)


# --- Configuration ---
DEFAULT_TRAIN_DIR = Path("/media/data1/feihong/train_kml_2048")
DEFAULT_TEST_DIR = Path("/media/data1/feihong/kml_test_2")
DEFAULT_LOCATION_JSON = Path(__file__).resolve().parents[1] / "tool/kml_locations.json"
DEFAULT_OUTPUT = Path(__file__).resolve().with_name("local_retrieval_ambiguity.png")
DEFAULT_RADIUS_KM = 1.0
DEFAULT_FOOTPRINT_SIDE_KM = 0.65

BACKGROUND = "#FBF8F0"
INK = "#26313A"
MUTED = "#65727D"
CANDIDATE = "#6687A0"
CANDIDATE_LIGHT = "#AEBFCB"
QUERY = "#ED7D31"
GRID = "#D6DDD9"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dir", type=Path, default=DEFAULT_TRAIN_DIR)
    parser.add_argument("--test-dir", type=Path, default=DEFAULT_TEST_DIR)
    parser.add_argument("--location-json", type=Path, default=DEFAULT_LOCATION_JSON)
    parser.add_argument("--radius-km", type=float, default=DEFAULT_RADIUS_KM)
    parser.add_argument(
        "--query-id",
        type=int,
        default=None,
        help="Location ID for the local example; default selects the densest record.",
    )
    parser.add_argument("--local-extent-km", type=float, default=1.18)
    parser.add_argument(
        "--footprint-side-km",
        type=float,
        default=DEFAULT_FOOTPRINT_SIDE_KM,
        help="Side length of explicitly schematic tile footprints.",
    )
    parser.add_argument("--num-footprints", type=int, default=7)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--pdf-output", type=Path, default=None)
    parser.add_argument("--csv-output", type=Path, default=None)
    parser.add_argument("--summary-output", type=Path, default=None)
    parser.add_argument("--notes-output", type=Path, default=None)
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def load_metadata(path: Path) -> dict[str, dict[str, str]]:
    if not path.is_file():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {str(key): value for key, value in raw.items() if isinstance(value, dict)}


def haversine_distances_from(coordinates: np.ndarray, index: int) -> np.ndarray:
    radians = np.deg2rad(coordinates)
    lon = radians[:, 0]
    lat = radians[:, 1]
    lon0, lat0 = radians[index]
    a = (
        np.sin((lat - lat0) / 2) ** 2
        + np.cos(lat0) * np.cos(lat) * np.sin((lon - lon0) / 2) ** 2
    )
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def local_xy_km(coordinates: np.ndarray, query_index: int) -> tuple[np.ndarray, np.ndarray]:
    """Project nearby lon/lat to local east/north offsets around the query."""
    lon0, lat0 = np.deg2rad(coordinates[query_index])
    lon = np.deg2rad(coordinates[:, 0])
    lat = np.deg2rad(coordinates[:, 1])
    x = EARTH_RADIUS_KM * math.cos(lat0) * (lon - lon0)
    y = EARTH_RADIUS_KM * (lat - lat0)
    return x, y


def choose_query_index(
    locations: Sequence[Location],
    counts: np.ndarray,
    query_id: int | None,
) -> int:
    if query_id is None:
        return int(np.argmax(counts))
    matches = [index for index, item in enumerate(locations) if item.location_id == query_id]
    if not matches:
        raise ValueError(f"query ID {query_id:04d} is not present in the loaded KML records.")
    return matches[0]


def choose_footprint_indices(
    x: np.ndarray,
    y: np.ndarray,
    distances: np.ndarray,
    query_index: int,
    radius_km: float,
    num_footprints: int,
) -> list[int]:
    """Choose nearby centers spread across azimuth sectors for legible outlines."""
    if num_footprints <= 1:
        return [query_index]
    eligible = np.flatnonzero((distances > 0) & (distances <= radius_km))
    if not len(eligible):
        return [query_index]

    angles = np.mod(np.arctan2(y[eligible], x[eligible]), 2 * np.pi)
    selected = [query_index]
    for sector in range(num_footprints - 1):
        low = 2 * np.pi * sector / (num_footprints - 1)
        high = 2 * np.pi * (sector + 1) / (num_footprints - 1)
        candidates = eligible[(angles >= low) & (angles < high)]
        if len(candidates):
            target_distance = min(radius_km * 0.48, float(np.median(distances[candidates])))
            chosen = candidates[np.argmin(np.abs(distances[candidates] - target_distance))]
            selected.append(int(chosen))
    return selected


def location_label(location: Location, metadata: dict[str, dict[str, str]]) -> str:
    detail = metadata.get(f"{location.location_id:04d}", {})
    place = ", ".join(value for value in (detail.get("city"), detail.get("country")) if value)
    return place or f"{location.latitude:.3f}°, {location.longitude:.3f}°"


def draw_local_example(
    axis: plt.Axes,
    coordinates: np.ndarray,
    locations: Sequence[Location],
    counts: np.ndarray,
    query_index: int,
    metadata: dict[str, dict[str, str]],
    radius_km: float,
    local_extent_km: float,
    footprint_side_km: float,
    num_footprints: int,
    marker: str = "a",
    title_size: float = 12.0,
    compact: bool = False,
    title: str = "Local candidate overlap",
) -> None:
    x, y = local_xy_km(coordinates, query_index)
    distances = haversine_distances_from(coordinates, query_index)
    local = distances <= local_extent_km * math.sqrt(2)
    neighbors = (distances > 0) & (distances <= radius_km)

    axis.set_facecolor(BACKGROUND)
    axis.scatter(
        x[local & ~neighbors], y[local & ~neighbors], s=15, color=CANDIDATE_LIGHT,
        alpha=0.38, edgecolors="none", zorder=2, rasterized=True,
    )
    axis.scatter(
        x[neighbors], y[neighbors], s=15 if compact else 25, color=CANDIDATE, alpha=0.82,
        edgecolors="white", linewidths=0.35, zorder=4, rasterized=True,
    )

    footprint_indices = choose_footprint_indices(
        x, y, distances, query_index, radius_km, num_footprints,
    )
    half = footprint_side_km / 2
    for index in footprint_indices:
        is_query = index == query_index
        axis.add_patch(
            Rectangle(
                (x[index] - half, y[index] - half),
                footprint_side_km,
                footprint_side_km,
                facecolor=QUERY if is_query else CANDIDATE,
                edgecolor=QUERY if is_query else CANDIDATE,
                linewidth=1.5 if is_query else 0.9,
                alpha=0.10 if is_query else 0.075,
                zorder=3,
            )
        )

    axis.add_patch(
        Circle(
            (0, 0), radius_km, fill=False, edgecolor=QUERY,
            linewidth=1.8, linestyle=(0, (5, 3)), zorder=5,
        )
    )
    axis.scatter(
        [0], [0], s=62 if compact else 105, color=QUERY, marker="o", edgecolors="white",
        linewidths=1.0, zorder=7, label=f"Query {locations[query_index].location_id:04d}",
    )
    if not compact:
        axis.annotate(
            f"1 km radius\n{counts[query_index]} other regions",
            xy=(radius_km / math.sqrt(2), radius_km / math.sqrt(2)),
            xytext=(0.97, 0.96), textcoords="axes fraction", ha="right", va="top",
            fontsize=8.5, color=QUERY,
            arrowprops={"arrowstyle": "-", "color": QUERY, "lw": 1.0},
        )
    detail = metadata.get(f"{locations[query_index].location_id:04d}", {})
    place = detail.get("city") or location_label(locations[query_index], metadata)
    axis.text(
        0.03, 0.04,
        (
            f"{place}\n{counts[query_index]} within {radius_km:g} km"
            if compact
            else f"{location_label(locations[query_index], metadata)}\n"
            f"outlined tiles: schematic {footprint_side_km:.2f} × {footprint_side_km:.2f} km"
        ),
        transform=axis.transAxes, ha="left", va="bottom", fontsize=6.0 if compact else 8.0, color=MUTED,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": BACKGROUND, "edgecolor": "none", "alpha": 0.9},
        zorder=8,
    )
    axis.set_xlim(-local_extent_km, local_extent_km)
    axis.set_ylim(-local_extent_km, local_extent_km)
    axis.set_aspect("equal", adjustable="box")
    axis.grid(color=GRID, linewidth=0.65, linestyle="--", alpha=0.75)
    axis.set_xlabel("E–W offset (km)" if compact else "East–west offset (km)")
    axis.set_ylabel("N–S offset (km)" if compact else "North–south offset (km)")
    if compact:
        axis.tick_params(labelsize=6.0, length=2.5, pad=1.5)
        axis.xaxis.label.set_size(6.2)
        axis.yaxis.label.set_size(6.2)
    axis.spines[["top", "right"]].set_visible(False)
    axis.set_title(
        f"({marker})  {title}",
        loc="left", fontsize=title_size, fontweight="bold", color=INK, pad=8,
    )


def draw_neighbor_ccdf(
    axis: plt.Axes,
    counts: np.ndarray,
    radius_km: float,
    marker: str = "b",
    title_size: float = 12.0,
    compact: bool = False,
    title: str = "Local ambiguity distribution (CCDF)",
) -> None:
    support = np.arange(0, int(counts.max()) + 1)
    ccdf = np.asarray([(counts >= value).mean() * 100 for value in support])
    median = float(np.median(counts))
    p75 = float(np.quantile(counts, 0.75))
    p90 = float(np.quantile(counts, 0.90))
    zero_share = float((counts == 0).mean() * 100)

    axis.set_facecolor(BACKGROUND)
    axis.fill_between(support, ccdf, step="post", color=CANDIDATE, alpha=0.18)
    axis.step(support, ccdf, where="post", color=CANDIDATE, linewidth=2.1)
    quantile_marks = ((median, "median", 5), (p75, "P75", -9), (p90, "P90", 5))
    for value, label, y_offset in quantile_marks:
        y_value = float((counts >= value).mean() * 100)
        axis.scatter([value], [y_value], s=22 if compact else 34, color=QUERY, edgecolors="white", linewidths=0.6, zorder=5)
        if not compact or label != "P75":
            axis.annotate(
                f"{label}={value:.0f}" if compact else f"{label} = {value:.0f}",
                xy=(value, y_value), xytext=(3 if compact else 6, y_offset),
                textcoords="offset points", fontsize=5.8 if compact else 8.0, color=QUERY,
                va="bottom" if y_offset >= 0 else "top",
            )
    axis.text(
        0.97, 0.96,
        (
            f"mean {counts.mean():.1f}\nzero {zero_share:.1f}%"
            if compact
            else f"mean  {counts.mean():.1f}\n"
            f"no neighbors  {zero_share:.1f}%\n"
            f"≥1 neighbor  {100 - zero_share:.1f}%"
        ),
        transform=axis.transAxes, ha="right", va="top", fontsize=6.0 if compact else 9.0, color=INK,
        linespacing=1.45,
        bbox={"boxstyle": "round,pad=0.45", "facecolor": "white", "edgecolor": GRID, "alpha": 0.92},
    )
    axis.set_xlim(0, max(int(counts.max()), 1))
    axis.set_ylim(0, 102)
    axis.set_xlabel(f"Neighbors within {radius_km:g} km" if compact else f"Other candidate regions within {radius_km:g} km")
    axis.set_ylabel("Locations above count (%)" if compact else "Locations with at least this many (%)")
    if compact:
        axis.tick_params(labelsize=6.0, length=2.5, pad=1.5)
        axis.xaxis.label.set_size(6.2)
        axis.yaxis.label.set_size(6.2)
    axis.grid(axis="both", color=GRID, linewidth=0.65, linestyle="--", alpha=0.75)
    axis.spines[["top", "right"]].set_visible(False)
    axis.set_title(
        f"({marker})  {title}",
        loc="left", fontsize=title_size, fontweight="bold", color=INK, pad=8,
    )


def create_figure(
    coordinates: np.ndarray,
    locations: Sequence[Location],
    counts: np.ndarray,
    query_index: int,
    metadata: dict[str, dict[str, str]],
    args: argparse.Namespace,
) -> plt.Figure:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    figure, axes = plt.subplots(1, 2, figsize=(12.4, 5.5), facecolor=BACKGROUND)
    draw_local_example(
        axes[0], coordinates, locations, counts, query_index, metadata,
        args.radius_km, args.local_extent_km, args.footprint_side_km,
        args.num_footprints,
    )
    draw_neighbor_ccdf(axes[1], counts, args.radius_km)
    figure.suptitle(
        "Candidate-region density reveals retrieval ambiguity",
        x=0.06, y=0.985, ha="left", fontsize=17, fontweight="bold", color=INK,
    )
    figure.text(
        0.06, 0.925,
        f"{len(coordinates):,} KML centers · ambiguity = number of other candidate regions within {args.radius_km:g} km",
        ha="left", fontsize=9.5, color=MUTED,
    )
    figure.subplots_adjust(left=0.075, right=0.985, top=0.84, bottom=0.14, wspace=0.25)
    return figure


def write_csv(
    path: Path,
    locations: Sequence[Location],
    counts: np.ndarray,
    metadata: dict[str, dict[str, str]],
    radius_km: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow((
            "location_id", "split", "longitude", "latitude", "continent", "country", "city",
            "neighbor_count_within_radius", "radius_km",
        ))
        for item, count in zip(locations, counts):
            detail = metadata.get(f"{item.location_id:04d}", {})
            writer.writerow((
                f"{item.location_id:04d}", item.split, item.longitude, item.latitude,
                detail.get("continent", ""), detail.get("country", ""), detail.get("city", ""),
                int(count), radius_km,
            ))


def summary_dict(
    locations: Sequence[Location],
    counts: np.ndarray,
    query_index: int,
    metadata: dict[str, dict[str, str]],
    args: argparse.Namespace,
    parse_errors: list[str],
) -> dict[str, Any]:
    query = locations[query_index]
    return {
        "metric": "number of other KML candidate-region centers within the radius",
        "radius_km": args.radius_km,
        "num_locations": len(locations),
        "statistics": {
            "mean": float(counts.mean()),
            "median": float(np.median(counts)),
            "p75": float(np.quantile(counts, 0.75)),
            "p90": float(np.quantile(counts, 0.90)),
            "min": int(counts.min()),
            "max": int(counts.max()),
            "zero_neighbor_count": int((counts == 0).sum()),
            "zero_neighbor_share": float((counts == 0).mean()),
        },
        "local_example": {
            "location_id": f"{query.location_id:04d}",
            "split": query.split,
            "longitude": query.longitude,
            "latitude": query.latitude,
            "place": metadata.get(f"{query.location_id:04d}", {}),
            "neighbors_within_radius": int(counts[query_index]),
            "schematic_footprint_side_km": args.footprint_side_km,
        },
        "footprint_note": "Outlined footprints are schematic; KML LookAt range is camera distance, not a georeferenced ground-tile boundary.",
        "kml_parse_errors": parse_errors,
    }


def write_notes(path: Path, summary: dict[str, Any]) -> None:
    stats = summary["statistics"]
    example = summary["local_example"]
    place = example.get("place", {})
    path.write_text(
        "局部候选区域歧义——图表说明\n"
        "============================\n\n"
        f"指标：对每个地点，统计半径 {summary['radius_km']:g} km 内的其他 KML candidate-region center 数量；自身不计入。\n"
        f"样本：{summary['num_locations']:,} 个地点。\n\n"
        f"统计：中位数 {stats['median']:.0f}，P75={stats['p75']:.0f}，P90={stats['p90']:.0f}，"
        f"均值={stats['mean']:.2f}；{stats['zero_neighbor_share']:.1%} 的地点没有 1 km 内邻居。\n\n"
        f"左图：真实密集地点 {example['location_id']}（{place.get('city', '')}, {place.get('country', '')}），"
        f"其 1 km 内有 {example['neighbors_within_radius']} 个其他 candidate region。橙色为 query，灰蓝色为其他候选。\n"
        f"tile outline 使用 {example['schematic_footprint_side_km']:.2f} km 的示意边长，仅表达候选区域可能重叠；"
        "它不是影像的地理配准边界。KML LookAt 的 range 表示相机到观察点的距离，不能直接解释为地面 footprint 边长。\n\n"
        "右图：邻居数量的互补累积分布（CCDF）；纵轴表示至少具有横轴所示邻居数的地点比例。\n",
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    train, train_errors = load_locations(args.train_dir, "train", 1652, strict=False)
    test, test_errors = load_locations(args.test_dir, "test", 1652, strict=False)
    locations = train + test
    if not locations:
        raise RuntimeError("No valid KML locations were loaded.")

    coordinates = coordinate_array(locations)
    counts = neighbor_counts_within_radius(coordinates, args.radius_km)
    query_index = choose_query_index(locations, counts, args.query_id)
    metadata = load_metadata(args.location_json)

    figure = create_figure(coordinates, locations, counts, query_index, metadata, args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=args.dpi, facecolor=BACKGROUND)
    pdf_output = args.pdf_output or args.output.with_suffix(".pdf")
    pdf_output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(pdf_output, facecolor=BACKGROUND)
    plt.close(figure)

    csv_output = args.csv_output or args.output.with_suffix(".csv")
    summary_output = args.summary_output or args.output.with_suffix(".json")
    notes_output = args.notes_output or args.output.with_name(f"{args.output.stem}_notes.txt")
    write_csv(csv_output, locations, counts, metadata, args.radius_km)
    summary = summary_dict(
        locations, counts, query_index, metadata, args, train_errors + test_errors,
    )
    summary_output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_notes(notes_output, summary)

    print(
        f"Saved {args.output}, {pdf_output}, {csv_output}, {summary_output}, and {notes_output}\n"
        f"n={len(counts):,}; mean={counts.mean():.3f}; median={np.median(counts):.0f}; "
        f"P75={np.quantile(counts, 0.75):.0f}; P90={np.quantile(counts, 0.90):.0f}; "
        f"zero={(counts == 0).mean():.1%}; query={locations[query_index].location_id:04d} "
        f"({counts[query_index]} neighbors)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
