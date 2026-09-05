#!/usr/bin/env python3
"""Package the data behind two repository plots into one Excel workbook.

The workbook is intentionally built from the CSV/JSON snapshots written at the
same time as the PNG figures.  This preserves the exact plotted population even
if the upstream KML, bbox, image, or text directories change later.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-replot-data")

import matplotlib.pyplot as plt


# --- Configuration ---
TOOLS_DIR = Path(__file__).resolve().parent
DEFAULT_LOCAL_CSV = TOOLS_DIR / "local_retrieval_ambiguity.csv"
DEFAULT_LOCAL_JSON = TOOLS_DIR / "local_retrieval_ambiguity.json"
DEFAULT_LOCAL_PNG = TOOLS_DIR / "local_retrieval_ambiguity.png"
DEFAULT_TARGET_CSV = TOOLS_DIR / "test_target_center_distribution.csv"
DEFAULT_TARGET_JSON = TOOLS_DIR / "test_target_center_distribution.json"
DEFAULT_TARGET_PNG = TOOLS_DIR / "test_target_center_distribution.png"
DEFAULT_OUTPUT = TOOLS_DIR / "replot_data_bundle.xlsx"

EARTH_RADIUS_KM = 6371.0088
LOCAL_EXTENT_KM = 1.18
LOCAL_NUM_FOOTPRINTS = 7
TARGET_GRIDSIZE = (54, 30)
TARGET_MARGINAL_BINS = 42

HEADER_COLOR = 0x244A64
SUBHEADER_COLOR = 0xDCE8EE
ACCENT_COLOR = 0xE8752E
LIGHT_COLOR = 0xF4F7F8
WHITE = 0xFFFFFF
TEXT_COLOR = 0x26323B


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-csv", type=Path, default=DEFAULT_LOCAL_CSV)
    parser.add_argument("--local-json", type=Path, default=DEFAULT_LOCAL_JSON)
    parser.add_argument("--local-png", type=Path, default=DEFAULT_LOCAL_PNG)
    parser.add_argument("--target-csv", type=Path, default=DEFAULT_TARGET_CSV)
    parser.add_argument("--target-json", type=Path, default=DEFAULT_TARGET_JSON)
    parser.add_argument("--target-png", type=Path, default=DEFAULT_TARGET_PNG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def read_csv(path: Path) -> tuple[list[str], list[list[Any]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        rows = [list(row) for row in reader]
    return header, rows


def numeric_rows(header: Sequence[str], rows: Sequence[Sequence[Any]]) -> list[list[Any]]:
    text_columns = {
        "location_id", "split", "continent", "country", "city",
        "satellite_path", "drone_dir",
    }
    converted: list[list[Any]] = []
    for row in rows:
        output_row: list[Any] = []
        for key, value in zip(header, row):
            if key in text_columns or value == "":
                output_row.append(value)
            else:
                output_row.append(float(value))
        converted.append(output_row)
    return converted


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def haversine_distances(coordinates: np.ndarray, index: int) -> np.ndarray:
    radians = np.deg2rad(coordinates)
    lon = radians[:, 0]
    lat = radians[:, 1]
    lon0, lat0 = radians[index]
    a = (
        np.sin((lat - lat0) / 2) ** 2
        + np.cos(lat0) * np.cos(lat) * np.sin((lon - lon0) / 2) ** 2
    )
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def local_xy(coordinates: np.ndarray, query_index: int) -> tuple[np.ndarray, np.ndarray]:
    lon0, lat0 = np.deg2rad(coordinates[query_index])
    lon = np.deg2rad(coordinates[:, 0])
    lat = np.deg2rad(coordinates[:, 1])
    x = EARTH_RADIUS_KM * math.cos(lat0) * (lon - lon0)
    y = EARTH_RADIUS_KM * (lat - lat0)
    return x, y


def choose_footprint_indices(
    x: np.ndarray,
    y: np.ndarray,
    distances: np.ndarray,
    query_index: int,
    radius_km: float,
    num_footprints: int,
) -> list[int]:
    eligible = np.flatnonzero((distances > 0) & (distances <= radius_km))
    if num_footprints <= 1 or not len(eligible):
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


def build_local_example(
    header: Sequence[str],
    rows: Sequence[Sequence[Any]],
    summary: dict[str, Any],
) -> tuple[list[str], list[list[Any]]]:
    columns = {name: index for index, name in enumerate(header)}
    query_id = summary["local_example"]["location_id"]
    query_index = next(
        index for index, row in enumerate(rows) if str(row[columns["location_id"]]) == query_id
    )
    coordinates = np.asarray(
        [[row[columns["longitude"]], row[columns["latitude"]]] for row in rows],
        dtype=np.float64,
    )
    radius_km = float(summary["radius_km"])
    distances = haversine_distances(coordinates, query_index)
    x, y = local_xy(coordinates, query_index)
    footprint_indices = set(
        choose_footprint_indices(
            x, y, distances, query_index, radius_km, LOCAL_NUM_FOOTPRINTS,
        )
    )
    local_mask = distances <= LOCAL_EXTENT_KM * math.sqrt(2)
    output_header = [
        "source_row", "location_id", "split", "longitude", "latitude", "continent",
        "country", "city", "east_offset_km", "north_offset_km",
        "haversine_distance_km", "is_query", "counted_within_radius",
        "plotted_as_neighbor", "is_in_local_plot_extent",
        "selected_schematic_footprint",
    ]
    output_rows: list[list[Any]] = []
    for index, row in enumerate(rows):
        if not local_mask[index]:
            continue
        output_rows.append([
            index + 2,
            row[columns["location_id"]],
            row[columns["split"]],
            float(row[columns["longitude"]]),
            float(row[columns["latitude"]]),
            row[columns["continent"]],
            row[columns["country"]],
            row[columns["city"]],
            float(x[index]),
            float(y[index]),
            float(distances[index]),
            index == query_index,
            index != query_index and distances[index] <= radius_km,
            distances[index] > 0 and distances[index] <= radius_km,
            True,
            index in footprint_indices,
        ])
    return output_header, output_rows


def build_ccdf(
    header: Sequence[str], rows: Sequence[Sequence[Any]], radius_km: float,
) -> tuple[list[str], list[list[Any]]]:
    count_index = header.index("neighbor_count_within_radius")
    counts = np.asarray([row[count_index] for row in rows], dtype=np.int32)
    output = []
    for threshold in range(int(counts.max()) + 1):
        locations = int(np.count_nonzero(counts >= threshold))
        output.append([threshold, locations, locations / len(counts), locations / len(counts) * 100, radius_km])
    return (
        ["neighbor_threshold", "locations_at_or_above", "ccdf_fraction", "ccdf_percent", "radius_km"],
        output,
    )


def build_target_marginals(
    header: Sequence[str], rows: Sequence[Sequence[Any]], tile_width: float, tile_height: float,
) -> tuple[list[str], list[list[Any]]]:
    east = np.asarray([row[header.index("east_m")] for row in rows], dtype=np.float64)
    north = np.asarray([row[header.index("north_m")] for row in rows], dtype=np.float64)
    output: list[list[Any]] = []
    for axis_name, values, half_extent in (
        ("east_m", east, tile_width / 2),
        ("north_m", north, tile_height / 2),
    ):
        edges = np.linspace(-half_extent, half_extent, TARGET_MARGINAL_BINS + 1)
        counts, _ = np.histogram(values, bins=edges)
        for index, count in enumerate(counts):
            output.append([
                axis_name, index + 1, float(edges[index]), float(edges[index + 1]),
                float((edges[index] + edges[index + 1]) / 2), int(count),
            ])
    return ["axis", "bin_number", "bin_left", "bin_right", "bin_center", "count"], output


def build_target_hexbin(
    header: Sequence[str], rows: Sequence[Sequence[Any]], tile_width: float, tile_height: float,
) -> tuple[list[str], list[list[Any]]]:
    east = np.asarray([row[header.index("east_m")] for row in rows], dtype=np.float64)
    north = np.asarray([row[header.index("north_m")] for row in rows], dtype=np.float64)
    figure, axis = plt.subplots()
    collection = axis.hexbin(
        east,
        north,
        gridsize=TARGET_GRIDSIZE,
        extent=(-tile_width / 2, tile_width / 2, -tile_height / 2, tile_height / 2),
        mincnt=1,
    )
    offsets = collection.get_offsets()
    counts = collection.get_array()
    plt.close(figure)
    output = [
        [index + 1, float(point[0]), float(point[1]), int(count)]
        for index, (point, count) in enumerate(zip(offsets, counts))
    ]
    return ["hexagon_number", "center_east_m", "center_north_m", "count"], output


def flatten_json(payload: Any, prefix: str = "") -> Iterable[tuple[str, Any]]:
    if isinstance(payload, dict):
        for key, value in payload.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            yield from flatten_json(value, child)
    elif isinstance(payload, list):
        if all(not isinstance(item, (dict, list)) for item in payload):
            yield prefix, ", ".join(str(item) for item in payload)
        else:
            for index, value in enumerate(payload):
                yield from flatten_json(value, f"{prefix}[{index}]")
    else:
        yield prefix, payload


def build_exclusions(target_summary: dict[str, Any]) -> tuple[list[str], list[list[Any]]]:
    rows: list[list[Any]] = []
    for reason, identifiers in target_summary["audit"]["skipped"].items():
        for location_id in identifiers:
            rows.append([location_id, reason])
    return ["location_id", "exclusion_reason"], rows


def make_property(uno_module: Any, name: str, value: Any) -> Any:
    prop = uno_module.createUnoStruct("com.sun.star.beans.PropertyValue")
    prop.Name = name
    prop.Value = value
    return prop


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def connect_to_calc(profile: Path) -> tuple[Any, Any, subprocess.Popen[bytes]]:
    for path in ("/usr/lib/python3/dist-packages", "/usr/lib/libreoffice/program"):
        if path not in sys.path:
            sys.path.append(path)
    import uno

    port = free_port()
    command = [
        "soffice",
        "--headless",
        "--invisible",
        "--norestore",
        "--nodefault",
        "--nolockcheck",
        f"-env:UserInstallation={uno.systemPathToFileUrl(str(profile))}",
        f"--accept=socket,host=127.0.0.1,port={port};urp;StarOffice.ComponentContext",
    ]
    process = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    local_context = uno.getComponentContext()
    resolver = local_context.ServiceManager.createInstanceWithContext(
        "com.sun.star.bridge.UnoUrlResolver", local_context,
    )
    remote_context = None
    for _ in range(80):
        if process.poll() is not None:
            raise RuntimeError("LibreOffice exited before accepting a connection.")
        try:
            remote_context = resolver.resolve(
                f"uno:socket,host=127.0.0.1,port={port};urp;StarOffice.ComponentContext"
            )
            break
        except Exception:
            time.sleep(0.1)
    if remote_context is None:
        process.terminate()
        raise RuntimeError("Timed out while connecting to headless LibreOffice.")
    service_manager = remote_context.ServiceManager
    desktop = service_manager.createInstanceWithContext(
        "com.sun.star.frame.Desktop", remote_context,
    )
    return uno, desktop, process


def write_table(sheet: Any, header: Sequence[Any], rows: Sequence[Sequence[Any]]) -> None:
    def uno_cell(value: Any) -> Any:
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, bool):
            return "TRUE" if value else "FALSE"
        if value is None:
            return ""
        return value

    all_rows = [
        [uno_cell(value) for value in header],
        *[[uno_cell(value) for value in row] for row in rows],
    ]
    width = len(header)
    chunk_size = 500
    for start in range(0, len(all_rows), chunk_size):
        chunk = all_rows[start:start + chunk_size]
        cell_range = sheet.getCellRangeByPosition(0, start, width - 1, start + len(chunk) - 1)
        cell_range.setDataArray(tuple(tuple(row) for row in chunk))
    header_range = sheet.getCellRangeByPosition(0, 0, width - 1, 0)
    header_range.CellBackColor = HEADER_COLOR
    header_range.CharColor = WHITE
    header_range.CharWeight = 150.0
    header_range.CharHeight = 10.5
    sheet.getCellRangeByPosition(0, 0, width - 1, len(all_rows) - 1).CharColor = TEXT_COLOR
    sheet.getCellRangeByPosition(0, 0, width - 1, len(all_rows) - 1).VertJustify = 2
    sheet.getCellRangeByPosition(0, 0, width - 1, len(all_rows) - 1).HoriJustify = 0
    for index, name in enumerate(header):
        column = sheet.Columns.getByIndex(index)
        if any(token in str(name) for token in ("path", "description", "formula", "notes", "value")):
            column.Width = 9000
        elif str(name) in {"country", "city", "continent", "exclusion_reason"}:
            column.Width = 4200
        else:
            column.Width = 3200
    sheet.Rows.getByIndex(0).Height = 720
    sheet.getCellRangeByPosition(0, 0, width - 1, 0).IsCellBackgroundTransparent = False
    sheet.getCellRangeByPosition(0, 0, width - 1, len(all_rows) - 1).CellBackColor = WHITE
    header_range.CellBackColor = HEADER_COLOR


def add_sheet(document: Any, name: str, header: Sequence[Any], rows: Sequence[Sequence[Any]]) -> Any:
    sheets = document.getSheets()
    if sheets.hasByName(name):
        sheet = sheets.getByName(name)
    else:
        sheets.insertNewByName(name, sheets.getCount())
        sheet = sheets.getByName(name)
    write_table(sheet, header, rows)
    return sheet


def build_readme_rows(
    local_summary: dict[str, Any],
    target_summary: dict[str, Any],
) -> list[list[Any]]:
    return [
        ["用途", "用于在本地以不同格式重绘 local_retrieval_ambiguity.png 和 test_target_center_distribution.png。"],
        ["快照原则", "点级数据来自与 PNG 同时生成的 CSV/JSON 快照；未重新读取上游目录。"],
        ["工作表：amb_locations", "局部歧义图的 3,530 个 KML center 及每个点 1 km 内邻居数。"],
        ["工作表：amb_ccdf", "右图的完整 CCDF：threshold 与至少达到该邻居数的地点比例。"],
        ["工作表：amb_local_example", "左图 Toronto 示例的局部坐标、距离和绘制标记。"],
        ["工作表：target_centers", "target-center 图的 2,804 个测试 region 点级坐标与 bbox。"],
        ["工作表：target_marginals", "上方/右侧边缘直方图的 42 个等宽 bins。"],
        ["工作表：target_hexbin", "主图 54×30 hexbin 的非空六边形中心与计数。"],
        ["工作表：target_exclusions", "25 个未进入 target-center 图的测试 ID 及排除原因。"],
        ["工作表：summaries", "两张图生成时写出的 JSON 参数、口径、审计和统计。"],
        ["工作表：data_dictionary", "字段、单位和含义。"],
        ["工作表：QA", "行数、聚合守恒和关键统计的自动校验。"],
        ["局部歧义口径", f"其他 KML center 在 {local_summary['radius_km']:g} km 内的数量；自身不计。"],
        ["局部示例", f"ID {local_summary['local_example']['location_id']}，"
                       f"{local_summary['local_example']['place'].get('city', '')}, "
                       f"{local_summary['local_example']['place'].get('country', '')}。"],
        ["target 坐标系", "3840×2160 px 线性映射为 1183×660 m；原点为 tile 中心，东/北为正。"],
        ["target 中心来源", f"每个 region 的 {target_summary['coordinate_system']['canonical_bbox_height_m']} m bbox 几何中心。"],
        ["重绘建议", "优先使用点级数据；需要完全复刻原图时可直接使用派生 CCDF、marginal 和 hexbin 工作表。"],
    ]


def build_dictionary_rows() -> list[list[Any]]:
    definitions = {
        "amb_locations": [
            ("location_id", "文本", "零填充地点 ID"),
            ("split", "类别", "train 或 test"),
            ("longitude / latitude", "度", "KML center，经度/纬度"),
            ("continent / country / city", "文本", "反向地理编码元数据"),
            ("neighbor_count_within_radius", "个", "半径 radius_km 内其他 center 数量"),
            ("radius_km", "km", "邻居半径，原图为 1 km"),
        ],
        "amb_ccdf": [
            ("neighbor_threshold", "个", "CCDF 横轴阈值"),
            ("locations_at_or_above", "个", "邻居数大于等于阈值的地点数"),
            ("ccdf_fraction / ccdf_percent", "比例 / %", "CCDF 纵轴"),
        ],
        "amb_local_example": [
            ("east_offset_km / north_offset_km", "km", "相对 query 的局部近似东西/南北偏移"),
            ("haversine_distance_km", "km", "相对 query 的大圆距离"),
            ("counted_within_radius", "布尔", "是否计入 query 的 1 km 邻居数"),
            ("plotted_as_neighbor", "布尔", "是否按原脚本的灰蓝邻居点绘制；精确重合点不在此层"),
            ("selected_schematic_footprint", "布尔", "是否绘制示意 tile outline"),
        ],
        "target_centers": [
            ("center_x_px / center_y_px", "px", "150 m bbox 的几何中心（图像坐标）"),
            ("east_m", "m", "相对 tile 中心向东偏移"),
            ("north_m", "m", "相对 tile 中心向北偏移"),
            ("radial_displacement_m", "m", "sqrt(east_m^2 + north_m^2)"),
            ("bbox_x1 / bbox_y1 / bbox_x2 / bbox_y2", "px", "150 m bbox 边界"),
            ("satellite_path / drone_dir", "路径", "生成快照时的本地数据路径，仅用于溯源"),
        ],
        "target_marginals": [
            ("axis", "类别", "east_m 或 north_m"),
            ("bin_left / bin_right / bin_center", "m", "42-bin 等宽直方图边界/中心"),
            ("count", "个", "bin 内 target center 数"),
        ],
        "target_hexbin": [
            ("center_east_m / center_north_m", "m", "Matplotlib hexbin 非空六边形中心"),
            ("count", "个", "六边形内 target center 数；原图颜色使用 log scale"),
        ],
    }
    return [[sheet, field, unit, description] for sheet, items in definitions.items() for field, unit, description in items]


def build_workbook(args: argparse.Namespace) -> dict[str, Any]:
    required = [
        args.local_csv, args.local_json, args.local_png,
        args.target_csv, args.target_json, args.target_png,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing required plot snapshots: " + ", ".join(missing))

    local_header, local_raw_rows = read_csv(args.local_csv)
    target_header, target_raw_rows = read_csv(args.target_csv)
    local_rows = numeric_rows(local_header, local_raw_rows)
    target_rows = numeric_rows(target_header, target_raw_rows)
    local_summary = json.loads(args.local_json.read_text(encoding="utf-8"))
    target_summary = json.loads(args.target_json.read_text(encoding="utf-8"))

    ccdf_header, ccdf_rows = build_ccdf(
        local_header, local_rows, float(local_summary["radius_km"]),
    )
    example_header, example_rows = build_local_example(local_header, local_rows, local_summary)
    target_marginal_header, target_marginal_rows = build_target_marginals(
        target_header,
        target_rows,
        float(target_summary["coordinate_system"]["normalized_tile_meters"][0]),
        float(target_summary["coordinate_system"]["normalized_tile_meters"][1]),
    )
    target_hexbin_header, target_hexbin_rows = build_target_hexbin(
        target_header,
        target_rows,
        float(target_summary["coordinate_system"]["normalized_tile_meters"][0]),
        float(target_summary["coordinate_system"]["normalized_tile_meters"][1]),
    )
    exclusion_header, exclusion_rows = build_exclusions(target_summary)

    summaries_rows = [
        ["local_retrieval_ambiguity", key, value]
        for key, value in flatten_json(local_summary)
    ] + [
        ["test_target_center_distribution", key, value]
        for key, value in flatten_json(target_summary)
    ]
    source_rows = []
    for path in required:
        stat = path.stat()
        source_rows.append([
            path.name,
            str(path),
            stat.st_size,
            time.strftime("%Y-%m-%d %H:%M:%S %z", time.localtime(stat.st_mtime)),
            sha256(path),
        ])

    local_count_index = local_header.index("neighbor_count_within_radius")
    local_counts = np.asarray([row[local_count_index] for row in local_rows], dtype=np.float64)
    target_displacement_index = target_header.index("radial_displacement_m")
    target_displacement = np.asarray(
        [row[target_displacement_index] for row in target_rows], dtype=np.float64,
    )
    east_hist_total = sum(row[-1] for row in target_marginal_rows if row[0] == "east_m")
    north_hist_total = sum(row[-1] for row in target_marginal_rows if row[0] == "north_m")
    hexbin_total = sum(row[-1] for row in target_hexbin_rows)
    qa_rows = [
        ["local row count", len(local_rows), local_summary["num_locations"], len(local_rows) == local_summary["num_locations"]],
        ["local mean neighbors", float(local_counts.mean()), local_summary["statistics"]["mean"], np.isclose(local_counts.mean(), local_summary["statistics"]["mean"])],
        ["local median neighbors", float(np.median(local_counts)), local_summary["statistics"]["median"], np.isclose(np.median(local_counts), local_summary["statistics"]["median"])],
        ["local P90 neighbors", float(np.quantile(local_counts, 0.90)), local_summary["statistics"]["p90"], np.isclose(np.quantile(local_counts, 0.90), local_summary["statistics"]["p90"])],
        ["local max neighbors", float(local_counts.max()), local_summary["statistics"]["max"], local_counts.max() == local_summary["statistics"]["max"]],
        ["target row count", len(target_rows), target_summary["audit"]["included_unique_test_regions"], len(target_rows) == target_summary["audit"]["included_unique_test_regions"]],
        ["target median displacement", float(np.median(target_displacement)), target_summary["statistics"]["median_displacement_m"], np.isclose(np.median(target_displacement), target_summary["statistics"]["median_displacement_m"])],
        ["target P90 displacement", float(np.quantile(target_displacement, 0.90)), target_summary["statistics"]["p90_displacement_m"], np.isclose(np.quantile(target_displacement, 0.90), target_summary["statistics"]["p90_displacement_m"])],
        ["east histogram total", east_hist_total, len(target_rows), east_hist_total == len(target_rows)],
        ["north histogram total", north_hist_total, len(target_rows), north_hist_total == len(target_rows)],
        ["hexbin total", hexbin_total, len(target_rows), hexbin_total == len(target_rows)],
        ["target exclusions", len(exclusion_rows), 25, len(exclusion_rows) == 25],
    ]
    if not all(bool(row[-1]) for row in qa_rows):
        failed = [row[0] for row in qa_rows if not bool(row[-1])]
        raise AssertionError("QA checks failed: " + ", ".join(failed))

    profile = Path(tempfile.mkdtemp(prefix="lo-replot-bundle-"))
    document = None
    process = None
    try:
        uno, desktop, process = connect_to_calc(profile)
        hidden = make_property(uno, "Hidden", True)
        document = desktop.loadComponentFromURL("private:factory/scalc", "_blank", 0, (hidden,))
        default_sheet = document.getSheets().getByIndex(0)
        default_sheet.Name = "README"
        write_table(default_sheet, ["项目", "说明"], build_readme_rows(local_summary, target_summary))
        add_sheet(document, "summaries", ["figure", "key", "value"], summaries_rows)
        add_sheet(document, "source_snapshots", ["file", "path", "bytes", "modified_time", "sha256"], source_rows)
        add_sheet(document, "amb_locations", local_header, local_rows)
        add_sheet(document, "amb_ccdf", ccdf_header, ccdf_rows)
        add_sheet(document, "amb_local_example", example_header, example_rows)
        add_sheet(document, "target_centers", target_header, target_rows)
        add_sheet(document, "target_marginals", target_marginal_header, target_marginal_rows)
        add_sheet(document, "target_hexbin", target_hexbin_header, target_hexbin_rows)
        add_sheet(document, "target_exclusions", exclusion_header, exclusion_rows)
        add_sheet(document, "data_dictionary", ["sheet", "field", "unit", "description"], build_dictionary_rows())
        add_sheet(document, "QA", ["check", "actual", "expected", "passed"], qa_rows)

        args.output.parent.mkdir(parents=True, exist_ok=True)
        output_url = uno.systemPathToFileUrl(str(args.output.resolve()))
        store_properties = (
            make_property(uno, "FilterName", "Calc MS Excel 2007 XML"),
            make_property(uno, "Overwrite", True),
        )
        document.storeAsURL(output_url, store_properties)
        document.close(True)
        document = None

        read_only = make_property(uno, "ReadOnly", True)
        reopened = desktop.loadComponentFromURL(output_url, "_blank", 0, (hidden, read_only))
        expected_sheets = (
            "README", "summaries", "source_snapshots", "amb_locations", "amb_ccdf",
            "amb_local_example", "target_centers", "target_marginals", "target_hexbin",
            "target_exclusions", "data_dictionary", "QA",
        )
        actual_sheets = tuple(reopened.getSheets().getElementNames())
        if actual_sheets != expected_sheets:
            reopened.close(True)
            raise AssertionError(f"LibreOffice round-trip sheet mismatch: {actual_sheets}")
        for sheet_name, expected_rows in (
            ("amb_locations", len(local_rows) + 1),
            ("target_centers", len(target_rows) + 1),
            ("QA", len(qa_rows) + 1),
        ):
            cursor = reopened.getSheets().getByName(sheet_name).createCursor()
            cursor.gotoEndOfUsedArea(True)
            actual_rows = cursor.getRangeAddress().EndRow + 1
            if actual_rows != expected_rows:
                reopened.close(True)
                raise AssertionError(
                    f"LibreOffice round-trip row mismatch for {sheet_name}: "
                    f"{actual_rows} != {expected_rows}"
                )
        reopened.close(True)
    finally:
        if document is not None:
            try:
                document.close(True)
            except Exception:
                pass
        if process is not None:
            process.terminate()
            try:
                process.wait(timeout=8)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=3)
        shutil.rmtree(profile, ignore_errors=True)

    return {
        "output": str(args.output),
        "sheets": 12,
        "local_locations": len(local_rows),
        "local_ccdf_rows": len(ccdf_rows),
        "local_example_rows": len(example_rows),
        "target_centers": len(target_rows),
        "target_marginal_rows": len(target_marginal_rows),
        "target_hexagons": len(target_hexbin_rows),
        "target_exclusions": len(exclusion_rows),
        "qa_checks": len(qa_rows),
        "libreoffice_roundtrip_verified": True,
    }


def main() -> int:
    result = build_workbook(parse_args())
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
