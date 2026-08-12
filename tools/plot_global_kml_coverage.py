#!/usr/bin/env python3
"""Render a publication-style global coverage map from train/test KML files.

Visual encoding follows the supplied geographic-coverage reference:

* marker shape encodes source: circle = inherited, square = added;
* marker color encodes split: blue = train, green = test.

The first ``--inherited-count`` integer IDs are treated as inherited.  With the
repository defaults this means IDs 0000--1651 are inherited and IDs >=1652 are
added.  Only the Python standard library, NumPy and Matplotlib are required.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Sequence, Tuple

import numpy as np

# Matplotlib otherwise tries to write below ~/.config, which is not always
# writable on training servers.
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-global-kml-map")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, PathPatch
from matplotlib.path import Path as MplPath


# --- Configuration ---
DEFAULT_TRAIN_DIR = Path("/media/data1/feihong/train_kml_2048")
DEFAULT_TEST_DIR = Path("/media/data1/feihong/kml_test_2")
DEFAULT_OUTPUT = Path(__file__).resolve().with_name("global_geographic_coverage.png")
DEFAULT_WORLD_GEOJSON = (
    Path(__file__).resolve().parent
    / "assets"
    / "ne_110m_admin_0_countries.geojson"
)
DEFAULT_INHERITED_COUNT = 1652

# Color-blind-friendly, restrained blue/green pair inspired by the reference.
TRAIN_COLOR = "#315DA8"
TEST_COLOR = "#3A9B55"
TEXT_COLOR = "#20262E"
MUTED_TEXT = "#66717E"
LAND_COLOR = "#F6F4EF"
OCEAN_COLOR = "#EEF6FA"
BORDER_COLOR = "#B7C0C8"
GRID_COLOR = "#C7D8E3"
PANEL_COLOR = "#F7F9FB"

# Equal Earth projection constants (EPSG:8857 formulation).
EE_A1 = 1.340264
EE_A2 = -0.081106
EE_A3 = 0.000893
EE_A4 = 0.003796


@dataclass(frozen=True)
class Location:
    location_id: int
    longitude: float
    latitude: float
    split: str
    source: str
    kml_path: str


def local_name(tag: str) -> str:
    """Return an XML tag without its namespace."""
    return tag.rsplit("}", 1)[-1]


def parse_location_id(path: Path) -> int:
    match = re.match(r"^(\d+)", path.stem)
    if match is None:
        raise ValueError(f"KML filename does not start with a numeric ID: {path.name}")
    return int(match.group(1))


def parse_kml_coordinate(path: Path) -> Tuple[float, float]:
    """Read the first Point coordinate, falling back to LookAt lon/lat."""
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError as exc:
        raise ValueError(f"Invalid KML XML in {path}: {exc}") from exc

    # Prefer Point/coordinates because that is the actual placemark position.
    for point in (element for element in root.iter() if local_name(element.tag) == "Point"):
        for child in point.iter():
            if local_name(child.tag) != "coordinates" or not child.text:
                continue
            first_coordinate = child.text.strip().split()[0]
            parts = first_coordinate.split(",")
            if len(parts) < 2:
                continue
            longitude, latitude = float(parts[0]), float(parts[1])
            return validate_coordinate(longitude, latitude, path)

    longitude = None
    latitude = None
    for element in root.iter():
        name = local_name(element.tag)
        if name == "longitude" and element.text and longitude is None:
            longitude = float(element.text.strip())
        elif name == "latitude" and element.text and latitude is None:
            latitude = float(element.text.strip())
        if longitude is not None and latitude is not None:
            return validate_coordinate(longitude, latitude, path)

    raise ValueError(f"No Point coordinates or LookAt longitude/latitude found: {path}")


def validate_coordinate(longitude: float, latitude: float, path: Path) -> Tuple[float, float]:
    if not (math.isfinite(longitude) and math.isfinite(latitude)):
        raise ValueError(f"Non-finite coordinate in {path}: {longitude}, {latitude}")
    if not (-180.0 <= longitude <= 180.0 and -90.0 <= latitude <= 90.0):
        raise ValueError(f"Out-of-range coordinate in {path}: {longitude}, {latitude}")
    return longitude, latitude


def load_locations(
    directory: Path,
    split: str,
    inherited_count: int,
    strict: bool = False,
) -> Tuple[List[Location], List[str]]:
    if not directory.is_dir():
        raise NotADirectoryError(f"KML directory does not exist: {directory}")

    locations: List[Location] = []
    errors: List[str] = []
    paths = sorted(directory.glob("*.kml"), key=lambda path: path.name)
    for path in paths:
        try:
            location_id = parse_location_id(path)
            longitude, latitude = parse_kml_coordinate(path)
            source = "inherited" if location_id < inherited_count else "added"
            locations.append(
                Location(
                    location_id=location_id,
                    longitude=longitude,
                    latitude=latitude,
                    split=split,
                    source=source,
                    kml_path=str(path),
                )
            )
        except (OSError, ValueError) as exc:
            errors.append(str(exc))

    if strict and errors:
        raise RuntimeError("Failed to parse KML files:\n" + "\n".join(errors[:20]))
    return locations, errors


def equal_earth(
    longitude: Sequence[float] | np.ndarray,
    latitude: Sequence[float] | np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Project longitude/latitude degrees to unitless Equal Earth coordinates."""
    lon = np.deg2rad(np.asarray(longitude, dtype=np.float64))
    lat = np.deg2rad(np.asarray(latitude, dtype=np.float64))
    theta = np.arcsin((math.sqrt(3.0) / 2.0) * np.sin(lat))
    theta2 = theta * theta
    theta6 = theta2 * theta2 * theta2
    denominator = 3.0 * (
        9.0 * EE_A4 * theta6 * theta2
        + 7.0 * EE_A3 * theta6
        + 3.0 * EE_A2 * theta2
        + EE_A1
    )
    x = 2.0 * math.sqrt(3.0) * lon * np.cos(theta) / denominator
    y = theta * (EE_A1 + EE_A2 * theta2 + EE_A3 * theta6 + EE_A4 * theta6 * theta2)
    return x, y


def map_boundary_path() -> MplPath:
    """Construct the curved outer boundary of the Equal Earth world map."""
    lat_side = np.linspace(-89.999, 89.999, 361)
    lon_top = np.linspace(180.0, -180.0, 721)
    lat_other_side = np.linspace(89.999, -89.999, 361)
    lon_bottom = np.linspace(-180.0, 180.0, 721)

    lon = np.concatenate(
        [
            np.full_like(lat_side, 180.0),
            lon_top,
            np.full_like(lat_other_side, -180.0),
            lon_bottom,
        ]
    )
    lat = np.concatenate(
        [
            lat_side,
            np.full_like(lon_top, 89.999),
            lat_other_side,
            np.full_like(lon_bottom, -89.999),
        ]
    )
    x, y = equal_earth(lon, lat)
    vertices = np.column_stack([x, y])
    codes = np.full(len(vertices), MplPath.LINETO, dtype=np.uint8)
    codes[0] = MplPath.MOVETO
    vertices = np.vstack([vertices, vertices[0]])
    codes = np.append(codes, MplPath.CLOSEPOLY)
    return MplPath(vertices, codes)


def iter_polygon_rings(geometry: Dict) -> Iterator[List[List[float]]]:
    geometry_type = geometry.get("type")
    coordinates = geometry.get("coordinates", [])
    if geometry_type == "Polygon":
        yield from coordinates
    elif geometry_type == "MultiPolygon":
        for polygon in coordinates:
            yield from polygon


def projected_ring_path(ring: Sequence[Sequence[float]]) -> MplPath | None:
    if len(ring) < 3:
        return None
    coordinates = np.asarray(ring, dtype=np.float64)
    x, y = equal_earth(coordinates[:, 0], coordinates[:, 1])
    vertices = np.column_stack([x, y])
    finite = np.isfinite(vertices).all(axis=1)
    vertices = vertices[finite]
    if len(vertices) < 3:
        return None
    if not np.allclose(vertices[0], vertices[-1]):
        vertices = np.vstack([vertices, vertices[0]])
    codes = np.full(len(vertices), MplPath.LINETO, dtype=np.uint8)
    codes[0] = MplPath.MOVETO
    codes[-1] = MplPath.CLOSEPOLY
    return MplPath(vertices, codes)


def draw_world(ax, geojson_path: Path) -> PathPatch:
    if not geojson_path.is_file():
        raise FileNotFoundError(
            f"World GeoJSON not found: {geojson_path}. Expected the bundled "
            "Natural Earth 110m countries file."
        )
    with open(geojson_path, "r", encoding="utf-8") as handle:
        world = json.load(handle)

    boundary = PathPatch(
        map_boundary_path(),
        facecolor=OCEAN_COLOR,
        edgecolor="#9EACB7",
        linewidth=0.9,
        zorder=0,
    )
    ax.add_patch(boundary)

    # Subtle geographic graticule, clipped to the world boundary.
    for latitude in [-60, -30, 0, 30, 60]:
        longitude = np.linspace(-180, 180, 721)
        x, y = equal_earth(longitude, np.full_like(longitude, latitude))
        line = ax.plot(x, y, color=GRID_COLOR, linewidth=0.45, alpha=0.75, zorder=1)[0]
        line.set_clip_path(boundary)
    for longitude in [-120, -60, 0, 60, 120]:
        latitude = np.linspace(-89.9, 89.9, 361)
        x, y = equal_earth(np.full_like(latitude, longitude), latitude)
        line = ax.plot(x, y, color=GRID_COLOR, linewidth=0.45, alpha=0.75, zorder=1)[0]
        line.set_clip_path(boundary)

    for feature in world.get("features", []):
        geometry = feature.get("geometry") or {}
        rings = list(iter_polygon_rings(geometry))
        if not rings:
            continue
        # Natural Earth exterior rings are first in each polygon. Drawing all
        # rings as land keeps this dependency-free and preserves small islands;
        # inland lakes are overdrawn later by the light border color only.
        for ring in rings:
            path = projected_ring_path(ring)
            if path is None:
                continue
            patch = PathPatch(
                path,
                facecolor=LAND_COLOR,
                edgecolor=BORDER_COLOR,
                linewidth=0.32,
                joinstyle="round",
                zorder=2,
            )
            patch.set_clip_path(boundary)
            ax.add_patch(patch)

    # Redraw the outline above countries for a clean publication boundary.
    outline = PathPatch(
        map_boundary_path(),
        facecolor="none",
        edgecolor="#96A5B0",
        linewidth=0.9,
        zorder=6,
    )
    ax.add_patch(outline)
    return boundary


def draw_locations(ax, locations: Sequence[Location], boundary: PathPatch) -> None:
    draw_density_halos(ax, locations, boundary)
    layer_order = [
        ("added", "test", "s", TEST_COLOR),
        ("inherited", "test", "o", TEST_COLOR),
        ("added", "train", "s", TRAIN_COLOR),
        ("inherited", "train", "o", TRAIN_COLOR),
    ]
    for source, split, marker, color in layer_order:
        selected = [item for item in locations if item.source == source and item.split == split]
        if not selected:
            continue
        x, y = equal_earth(
            [item.longitude for item in selected],
            [item.latitude for item in selected],
        )

        # Thin white halo keeps overlapping markers legible over borders.
        halo = ax.scatter(
            x,
            y,
            s=22.0,
            marker=marker,
            facecolors="none",
            edgecolors="white",
            linewidths=1.15,
            alpha=0.72,
            zorder=7,
        )
        points = ax.scatter(
            x,
            y,
            s=16.0,
            marker=marker,
            facecolors="none",
            edgecolors=color,
            linewidths=0.82,
            alpha=0.82,
            zorder=8,
        )
        halo.set_clip_path(boundary)
        points.set_clip_path(boundary)


def draw_density_halos(
    ax,
    locations: Sequence[Location],
    boundary: PathPatch,
    grid_size_degrees: float = 0.35,
) -> None:
    """Show dense city/campus groups without moving individual locations.

    At global scale thousands of distinct coordinates collapse to a few screen
    pixels.  The translucent, split-colored halos aggregate a fine 0.35-degree
    grid only for visual density; every source/split glyph is still plotted at
    its exact KML coordinate above the halo.
    """
    for split, color in (("train", TRAIN_COLOR), ("test", TEST_COLOR)):
        cells: Dict[Tuple[int, int], List[Location]] = defaultdict(list)
        for item in locations:
            if item.split != split:
                continue
            cell = (
                math.floor((item.longitude + 180.0) / grid_size_degrees),
                math.floor((item.latitude + 90.0) / grid_size_degrees),
            )
            cells[cell].append(item)

        cluster_lon: List[float] = []
        cluster_lat: List[float] = []
        cluster_count: List[int] = []
        for members in cells.values():
            cluster_lon.append(float(np.mean([item.longitude for item in members])))
            cluster_lat.append(float(np.mean([item.latitude for item in members])))
            cluster_count.append(len(members))

        x, y = equal_earth(cluster_lon, cluster_lat)
        counts = np.asarray(cluster_count, dtype=np.float64)
        sizes = 7.0 + 2.25 * np.power(counts, 0.80)
        halos = ax.scatter(
            x,
            y,
            s=sizes,
            marker="o",
            facecolors=color,
            edgecolors=color,
            linewidths=0.45,
            alpha=0.11,
            zorder=6.5,
        )
        halos.set_clip_path(boundary)


def add_north_arrow(ax) -> None:
    ax.text(
        0.035,
        0.87,
        "N",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="bold",
        color=TEXT_COLOR,
        zorder=20,
    )
    ax.annotate(
        "",
        xy=(0.035, 0.865),
        xytext=(0.035, 0.795),
        xycoords=ax.transAxes,
        arrowprops={
            "arrowstyle": "simple",
            "facecolor": TEXT_COLOR,
            "edgecolor": TEXT_COLOR,
            "linewidth": 0.4,
            "mutation_scale": 16,
        },
        zorder=20,
    )


def add_side_panel(ax, counts: Counter, total: int) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    legend_box = FancyBboxPatch(
        (0.04, 0.47),
        0.92,
        0.48,
        boxstyle="round,pad=0.022,rounding_size=0.018",
        facecolor="white",
        edgecolor="#AAB2BB",
        linewidth=0.9,
    )
    ax.add_patch(legend_box)
    ax.text(0.14, 0.89, "Source (shape)", fontsize=10.5, fontweight="bold", color=TEXT_COLOR)

    source_handles = [
        Line2D([], [], marker="o", linestyle="none", markersize=7.5, markerfacecolor="white", markeredgecolor="#4E5863", markeredgewidth=1.2),
        Line2D([], [], marker="s", linestyle="none", markersize=7.2, markerfacecolor="white", markeredgecolor="#4E5863", markeredgewidth=1.2),
    ]
    source_legend = ax.legend(
        source_handles,
        ["Inherited", "Added"],
        loc="upper left",
        bbox_to_anchor=(0.10, 0.855),
        frameon=False,
        borderaxespad=0,
        handletextpad=0.75,
        labelspacing=0.85,
        fontsize=9.5,
    )
    ax.add_artist(source_legend)
    ax.plot([0.12, 0.88], [0.705, 0.705], color="#CDD3D9", linewidth=0.8, linestyle=(0, (1.2, 1.2)))
    ax.text(0.14, 0.66, "Split (color)", fontsize=10.5, fontweight="bold", color=TEXT_COLOR)

    split_handles = [
        Line2D([], [], marker="s", linestyle="none", markersize=7.5, markerfacecolor="white", markeredgecolor=TRAIN_COLOR, markeredgewidth=1.35),
        Line2D([], [], marker="s", linestyle="none", markersize=7.5, markerfacecolor="white", markeredgecolor=TEST_COLOR, markeredgewidth=1.35),
    ]
    ax.legend(
        split_handles,
        ["Train", "Test"],
        loc="upper left",
        bbox_to_anchor=(0.10, 0.625),
        frameon=False,
        borderaxespad=0,
        handletextpad=0.75,
        labelspacing=0.85,
        fontsize=9.5,
    )

    stats_box = FancyBboxPatch(
        (0.04, 0.08),
        0.92,
        0.32,
        boxstyle="round,pad=0.022,rounding_size=0.018",
        facecolor=PANEL_COLOR,
        edgecolor="#AAB2BB",
        linewidth=0.9,
    )
    ax.add_patch(stats_box)

    rows = [
        (f"{total:,}", "locations"),
        (f"{counts['inherited']:,}", "inherited"),
        (f"{counts['added']:,}", "added"),
        (f"{counts['train']:,} / {counts['test']:,}", "train / test"),
    ]
    y_positions = [0.335, 0.27, 0.205, 0.14]
    for (value, label), y in zip(rows, y_positions):
        ax.text(0.12, y, value, fontsize=10.5, fontweight="bold", color=TRAIN_COLOR, ha="left", va="center")
        ax.text(0.51, y, label, fontsize=9.2, color=TEXT_COLOR, ha="left", va="center")


def create_figure(
    locations: Sequence[Location],
    world_geojson: Path,
    title: str,
) -> plt.Figure:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.titlecolor": TEXT_COLOR,
            "text.color": TEXT_COLOR,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )
    figure = plt.figure(figsize=(15.5, 8.2), constrained_layout=False)
    grid = figure.add_gridspec(
        nrows=1,
        ncols=2,
        width_ratios=[5.6, 1.35],
        left=0.025,
        right=0.985,
        top=0.92,
        bottom=0.075,
        wspace=0.025,
    )
    map_ax = figure.add_subplot(grid[0, 0])
    panel_ax = figure.add_subplot(grid[0, 1])

    boundary = draw_world(map_ax, world_geojson)
    draw_locations(map_ax, locations, boundary)
    add_north_arrow(map_ax)

    boundary_vertices = map_boundary_path().vertices
    x_min, y_min = boundary_vertices.min(axis=0)
    x_max, y_max = boundary_vertices.max(axis=0)
    map_ax.set_xlim(x_min - 0.05, x_max + 0.05)
    map_ax.set_ylim(y_min - 0.04, y_max + 0.04)
    map_ax.set_aspect("equal", adjustable="box")
    map_ax.axis("off")
    map_ax.text(
        0.0,
        1.035,
        title,
        transform=map_ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=16,
        fontweight="bold",
        color=TEXT_COLOR,
    )
    map_ax.text(
        0.001,
        1.005,
        "Source is encoded by shape, split by color; translucent halos show local density.",
        transform=map_ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9.2,
        color=MUTED_TEXT,
    )

    counts = Counter(item.source for item in locations)
    counts.update(item.split for item in locations)
    add_side_panel(panel_ax, counts, len(locations))

    figure.text(
        0.03,
        0.028,
        "Equal Earth projection  •  Density halos: 0.35° cells  •  Boundaries: Natural Earth 1:110m (public domain)",
        ha="left",
        va="center",
        fontsize=7.8,
        color="#7A858F",
    )
    return figure


def build_summary(
    locations: Sequence[Location],
    train_dir: Path,
    test_dir: Path,
    inherited_count: int,
    parse_errors: Sequence[str],
) -> Dict:
    id_counts = Counter(item.location_id for item in locations)
    duplicate_ids = sorted(location_id for location_id, count in id_counts.items() if count > 1)
    category_counts = Counter((item.split, item.source) for item in locations)
    return {
        "train_dir": str(train_dir),
        "test_dir": str(test_dir),
        "inherited_id_rule": f"0 <= id < {inherited_count}",
        "num_locations": len(locations),
        "num_train": sum(item.split == "train" for item in locations),
        "num_test": sum(item.split == "test" for item in locations),
        "num_inherited": sum(item.source == "inherited" for item in locations),
        "num_added": sum(item.source == "added" for item in locations),
        "category_counts": {
            f"{split}_{source}": category_counts[(split, source)]
            for split in ("train", "test")
            for source in ("inherited", "added")
        },
        "longitude_range": [
            min(item.longitude for item in locations),
            max(item.longitude for item in locations),
        ],
        "latitude_range": [
            min(item.latitude for item in locations),
            max(item.latitude for item in locations),
        ],
        "duplicate_ids": duplicate_ids,
        "num_parse_errors": len(parse_errors),
        "parse_errors": list(parse_errors),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a publication-style global train/test coverage map from KML files."
    )
    parser.add_argument("--train-dir", type=Path, default=DEFAULT_TRAIN_DIR)
    parser.add_argument("--test-dir", type=Path, default=DEFAULT_TEST_DIR)
    parser.add_argument(
        "--inherited-count",
        type=int,
        default=DEFAULT_INHERITED_COUNT,
        help="IDs 0 through inherited-count-1 are inherited; later IDs are added.",
    )
    parser.add_argument("--world-geojson", type=Path, default=DEFAULT_WORLD_GEOJSON)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--pdf-output",
        type=Path,
        default=None,
        help="Optional vector PDF output generated from the same figure.",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=None,
        help="Defaults to OUTPUT with a .json suffix.",
    )
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--title", default="(a) Global geographic coverage")
    parser.add_argument("--strict", action="store_true", help="Fail on any malformed KML file.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.inherited_count <= 0:
        raise ValueError("--inherited-count must be positive.")
    if args.dpi <= 0:
        raise ValueError("--dpi must be positive.")

    train_locations, train_errors = load_locations(
        args.train_dir,
        split="train",
        inherited_count=args.inherited_count,
        strict=args.strict,
    )
    test_locations, test_errors = load_locations(
        args.test_dir,
        split="test",
        inherited_count=args.inherited_count,
        strict=args.strict,
    )
    locations = train_locations + test_locations
    parse_errors = train_errors + test_errors
    if not locations:
        raise RuntimeError("No valid locations were loaded from the KML directories.")

    summary = build_summary(
        locations,
        train_dir=args.train_dir,
        test_dir=args.test_dir,
        inherited_count=args.inherited_count,
        parse_errors=parse_errors,
    )
    if summary["duplicate_ids"]:
        message = f"Duplicate location IDs across inputs: {summary['duplicate_ids'][:20]}"
        if args.strict:
            raise RuntimeError(message)
        print(f"Warning: {message}", file=sys.stderr)

    figure = create_figure(locations, args.world_geojson, args.title)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=args.dpi, bbox_inches="tight", pad_inches=0.10)
    if args.pdf_output is not None:
        args.pdf_output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(args.pdf_output, bbox_inches="tight", pad_inches=0.10)
    plt.close(figure)

    summary_output = args.summary_output or args.output.with_suffix(".json")
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_output, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    print(
        f"Saved {args.output} with {summary['num_locations']:,} locations "
        f"({summary['num_train']:,} train, {summary['num_test']:,} test; "
        f"{summary['num_inherited']:,} inherited, {summary['num_added']:,} added)."
    )
    if args.pdf_output is not None:
        print(f"Saved {args.pdf_output}")
    print(f"Saved {summary_output}")
    if parse_errors:
        print(f"Skipped {len(parse_errors)} malformed KML files; see summary JSON.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
