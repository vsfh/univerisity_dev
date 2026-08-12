#!/usr/bin/env python3
"""Generate a paper-ready 4x4 qualitative retrieval/grounding figure.

The input is a YAML file with a top-level ``samples`` list. ``results``,
``records``, and ``query_records`` are accepted as aliases. Both the nested
schema below and the flat field names emitted by this repository's evaluation
scripts are supported::

    samples:
      - id: example-001
        query:
          path: images/query.png
          altitude: 250
          heading: 45
        retrieval:
          path: images/top1.png
          correct: true
          score: 0.873
        oracle:
          path: images/paired_satellite.png
          gt_bbox: [210, 95, 315, 180]
          pred_bbox: [202, 90, 322, 187]
          iou: 0.71
          cde: 8.4
          bbox_format: xyxy
          normalized: false
        diagnostic:
          wo_heading_bbox: [170, 60, 292, 155]
          wo_heading_iou: 0.32
          heatmap_path: null

Paths are resolved relative to the YAML file. Boxes may be ``xyxy`` or
``xywh`` and may use pixel or normalized coordinates. A top-level ``defaults``
mapping can set ``bbox_format``, ``normalized``, and ``cde_unit``.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import yaml
from PIL import Image, ImageOps

# Matplotlib otherwise tries to write under ~/.config, which is often read-only
# on evaluation servers.
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-qualitative-figure")

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib import patheffects
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle


# --- Configuration ---
FIGURE_WIDTH_IN = 7.1
FIGURE_HEIGHT_IN = 4.65
PANEL_WIDTH = 960
PANEL_HEIGHT = 540
DEFAULT_DPI = 300

GT_COLOR = "#356B4F"
FULL_COLOR = "#B96532"
RETRIEVAL_SUCCESS_COLOR = "#287C72"
RETRIEVAL_FAILURE_COLOR = "#9F423C"
WO_HEADING_COLOR = "#FFFFFF"
TEXT_COLOR = "#222222"
PADDING_COLOR = (247, 247, 247)

ROW_DESCRIPTIONS = (
    "retrieval correct + oracle IoU >= {success_iou:.2f}",
    "retrieval correct + oracle IoU < {failure_iou:.2f}",
    "retrieval incorrect + oracle IoU >= {success_iou:.2f}",
    "heading gain >= {heading_gain:.2f}",
)


@dataclass(frozen=True)
class BoxSpec:
    values: Tuple[float, float, float, float]
    bbox_format: str = "xyxy"
    normalized: Optional[bool] = None

    def to_xyxy(self, width: int, height: int) -> Tuple[float, float, float, float]:
        values = list(self.values)
        normalized = self.normalized
        if normalized is None:
            normalized = all(-1e-6 <= value <= 1.0 + 1e-6 for value in values)
        if normalized:
            values[0] *= width
            values[2] *= width
            values[1] *= height
            values[3] *= height
        if self.bbox_format == "xywh":
            values[2] += values[0]
            values[3] += values[1]
        x1, x2 = sorted((values[0], values[2]))
        y1, y2 = sorted((values[1], values[3]))
        return (
            min(max(x1, 0.0), float(width)),
            min(max(y1, 0.0), float(height)),
            min(max(x2, 0.0), float(width)),
            min(max(y2, 0.0), float(height)),
        )


@dataclass
class Sample:
    sample_id: str
    query_path: Path
    retrieval_path: Path
    oracle_path: Path
    altitude: Optional[float]
    heading: Optional[float]
    top1_correct: bool
    retrieval_score: Optional[float]
    gt_bbox: BoxSpec
    pred_bbox: BoxSpec
    oracle_iou: Optional[float]
    cde: Optional[float]
    cde_unit: str
    wo_heading_bbox: Optional[BoxSpec]
    wo_heading_iou: Optional[float]
    heatmap_path: Optional[Path]
    heatmap_values: Optional[np.ndarray]

    def metrics(self) -> Tuple[float, float, Optional[float]]:
        image = open_rgb(self.oracle_path)
        width, height = image.size
        gt = self.gt_bbox.to_xyxy(width, height)
        pred = self.pred_bbox.to_xyxy(width, height)
        oracle_iou = (
            float(self.oracle_iou)
            if self.oracle_iou is not None
            else compute_iou(pred, gt)
        )
        cde = float(self.cde) if self.cde is not None else compute_cde(pred, gt)
        heading_gain: Optional[float] = None
        if self.wo_heading_iou is not None:
            heading_gain = oracle_iou - float(self.wo_heading_iou)
        elif self.wo_heading_bbox is not None:
            wo_bbox = self.wo_heading_bbox.to_xyxy(width, height)
            heading_gain = oracle_iou - compute_iou(wo_bbox, gt)
        return oracle_iou, cde, heading_gain


@dataclass(frozen=True)
class Selection:
    row: int
    sample: Sample
    exact: bool
    explanation: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate an ICLR-width 4x4 qualitative figure from experiment YAML."
    )
    parser.add_argument(
        "results",
        type=Path,
        nargs="?",
        help="Experiment result YAML.",
    )
    parser.add_argument(
        "--write-template",
        type=Path,
        default=None,
        metavar="PATH",
        help="Write an editable four-row result YAML template and exit.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("qualitative_figure.pdf"),
        help="Main output (.pdf, .png, or .svg). Default: qualitative_figure.pdf",
    )
    parser.add_argument(
        "--png-output",
        type=Path,
        default=None,
        help="Optional additional high-resolution PNG output.",
    )
    parser.add_argument("--width", type=float, default=FIGURE_WIDTH_IN)
    parser.add_argument("--height", type=float, default=FIGURE_HEIGHT_IN)
    parser.add_argument("--dpi", type=int, default=DEFAULT_DPI)
    parser.add_argument("--success-iou", type=float, default=0.5)
    parser.add_argument("--failure-iou", type=float, default=0.25)
    parser.add_argument("--heading-gain", type=float, default=0.15)
    parser.add_argument(
        "--heatmap-cmap",
        choices=("turbo", "magma"),
        default="turbo",
    )
    parser.add_argument("--heatmap-alpha", type=float, default=0.42)
    parser.add_argument(
        "--selected-yaml",
        type=Path,
        default=None,
        help="Optionally save the selected row IDs and selection status.",
    )
    args = parser.parse_args()
    if args.write_template is None and args.results is None:
        parser.error("results is required unless --write-template is used")
    if args.write_template is not None and args.results is not None:
        parser.error("do not pass results together with --write-template")
    return args


def write_template(path: Path) -> None:
    def template_sample(
        sample_id: str,
        row_target: str,
        top1_correct: bool,
        oracle_iou: float,
        *,
        altitude: int,
        heading: int,
        wo_heading_iou: Optional[float] = None,
    ) -> Dict[str, Any]:
        diagnostic: Dict[str, Any] = {
            "wo_heading_bbox": None,
            "wo_heading_iou": wo_heading_iou,
            "heatmap_path": None,
        }
        if wo_heading_iou is not None:
            diagnostic["wo_heading_bbox"] = [160, 80, 310, 220]
        return {
            "id": sample_id,
            "row_target": row_target,
            "query": {
                "path": "REPLACE_WITH_DRONE_QUERY_PATH",
                "altitude": altitude,
                "heading": heading,
            },
            "retrieval": {
                "path": "REPLACE_WITH_TOP1_TILE_PATH",
                "correct": top1_correct,
                "score": None,
            },
            "oracle": {
                "path": "REPLACE_WITH_PAIRED_GT_TILE_PATH",
                "gt_bbox": [200, 100, 340, 230],
                "pred_bbox": [210, 105, 350, 235],
                "iou": oracle_iou,
                "cde": None,
            },
            "diagnostic": diagnostic,
        }

    payload = {
        "_instructions": (
            "Replace every REPLACE_WITH_* path and all placeholder boxes/metrics "
            "with per-sample evaluation outputs. Do not use the example numbers "
            "as experimental results."
        ),
        "defaults": {
            "bbox_format": "xyxy",
            "normalized": False,
            "cde_unit": "px",
        },
        "samples": [
            template_sample(
                "row1-end-to-end-success",
                "retrieval correct, oracle IoU >= 0.5",
                True,
                0.70,
                altitude=200,
                heading=0,
            ),
            template_sample(
                "row2-grounding-failure",
                "retrieval correct, oracle IoU < 0.25",
                True,
                0.15,
                altitude=250,
                heading=90,
            ),
            template_sample(
                "row3-retrieval-bottleneck",
                "retrieval incorrect, oracle IoU >= 0.5",
                False,
                0.65,
                altitude=250,
                heading=180,
            ),
            template_sample(
                "row4-heading-benefit",
                "full IoU - w/o-heading IoU >= 0.15",
                True,
                0.62,
                altitude=300,
                heading=45,
                wo_heading_iou=0.30,
            ),
        ],
    }
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(payload, stream, sort_keys=False, allow_unicode=True)
    print(f"[template] saved editable result YAML: {path}")
    print(
        "[template] replace the placeholder paths, boxes, and metrics before "
        "generating a paper figure."
    )


def first(mapping: Mapping[str, Any], paths: Iterable[str], default: Any = None) -> Any:
    for dotted_path in paths:
        value: Any = mapping
        found = True
        for part in dotted_path.split("."):
            if not isinstance(value, Mapping) or part not in value:
                found = False
                break
            value = value[part]
        if found and value is not None:
            return value
    return default


def as_optional_float(value: Any, field: str, sample_id: str) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{sample_id}: {field} must be numeric, got {value!r}") from exc
    if not math.isfinite(result):
        raise ValueError(f"{sample_id}: {field} must be finite, got {value!r}")
    return result


def as_bool(value: Any, field: str, sample_id: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "yes", "correct", "1"}:
            return True
        if normalized in {"false", "no", "incorrect", "0"}:
            return False
    raise ValueError(f"{sample_id}: {field} must be boolean, got {value!r}")


def resolve_path(value: Any, base_dir: Path, field: str, sample_id: str) -> Path:
    if not isinstance(value, (str, os.PathLike)) or not str(value).strip():
        raise ValueError(f"{sample_id}: missing required image field {field}")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def optional_path(value: Any, base_dir: Path) -> Optional[Path]:
    if value is None or not str(value).strip():
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def parse_box(
    value: Any,
    field: str,
    sample_id: str,
    bbox_format: str,
    normalized: Optional[bool],
) -> Optional[BoxSpec]:
    if value is None:
        return None
    local_format = bbox_format
    local_normalized = normalized
    if isinstance(value, Mapping):
        local_format = str(value.get("format", value.get("bbox_format", bbox_format))).lower()
        local_normalized = value.get("normalized", normalized)
        if "values" in value:
            value = value["values"]
        elif all(key in value for key in ("x1", "y1", "x2", "y2")):
            value = [value["x1"], value["y1"], value["x2"], value["y2"]]
            local_format = "xyxy"
        elif all(key in value for key in ("x", "y", "w", "h")):
            value = [value["x"], value["y"], value["w"], value["h"]]
            local_format = "xywh"
    if local_format not in {"xyxy", "xywh"}:
        raise ValueError(f"{sample_id}: {field} has unsupported format {local_format!r}")
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 4:
        raise ValueError(f"{sample_id}: {field} must contain four coordinates")
    try:
        coordinates = tuple(float(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{sample_id}: {field} contains a non-numeric value") from exc
    if not all(math.isfinite(item) for item in coordinates):
        raise ValueError(f"{sample_id}: {field} contains a non-finite value")
    if local_normalized is not None:
        local_normalized = as_bool(local_normalized, f"{field}.normalized", sample_id)
    return BoxSpec(coordinates, local_format, local_normalized)


def load_heatmap_values(value: Any, path: Optional[Path], sample_id: str) -> Optional[np.ndarray]:
    if value is not None:
        array = np.asarray(value, dtype=np.float32)
    elif path is not None and path.suffix.lower() in {".npy", ".npz"}:
        if path.suffix.lower() == ".npy":
            array = np.load(path)
        else:
            archive = np.load(path)
            if not archive.files:
                raise ValueError(f"{sample_id}: heatmap archive is empty: {path}")
            array = archive[archive.files[0]]
    else:
        return None
    array = np.asarray(array, dtype=np.float32).squeeze()
    if array.ndim != 2:
        raise ValueError(f"{sample_id}: heatmap must be 2-D after squeeze, got {array.shape}")
    return array


def sample_from_mapping(
    item: Mapping[str, Any],
    index: int,
    base_dir: Path,
    defaults: Mapping[str, Any],
) -> Sample:
    sample_id = str(first(item, ("id", "sample_id", "name", "gt_label"), f"sample-{index:05d}"))
    bbox_format = str(
        first(item, ("oracle.bbox_format", "bbox_format"), defaults.get("bbox_format", "xyxy"))
    ).lower()
    normalized = first(
        item,
        ("oracle.normalized", "bbox_normalized", "normalized"),
        defaults.get("normalized"),
    )
    if normalized is not None:
        normalized = as_bool(normalized, "normalized", sample_id)

    query_path = resolve_path(
        first(item, ("query.path", "query.image", "drone_path", "query_path")),
        base_dir,
        "query.path / drone_path",
        sample_id,
    )
    retrieval_path = resolve_path(
        first(
            item,
            (
                "retrieval.path",
                "retrieval.image",
                "pred_satellite_path",
                "retrieved_path",
                "top1_path",
            ),
        ),
        base_dir,
        "retrieval.path / pred_satellite_path",
        sample_id,
    )
    oracle_path = resolve_path(
        first(
            item,
            (
                "oracle.path",
                "oracle.image",
                "gt_satellite_path",
                "oracle_tile_path",
                "satellite_path",
            ),
        ),
        base_dir,
        "oracle.path / gt_satellite_path",
        sample_id,
    )

    gt_bbox = parse_box(
        first(item, ("oracle.gt_bbox", "gt_bbox", "target_bbox", "bbox")),
        "gt_bbox",
        sample_id,
        bbox_format,
        normalized,
    )
    pred_bbox = parse_box(
        first(item, ("oracle.pred_bbox", "pred_bbox", "full_bbox", "prediction_bbox")),
        "pred_bbox",
        sample_id,
        bbox_format,
        normalized,
    )
    if gt_bbox is None or pred_bbox is None:
        raise ValueError(f"{sample_id}: both gt_bbox and pred_bbox are required")

    wo_heading_bbox = parse_box(
        first(
            item,
            (
                "diagnostic.wo_heading_bbox",
                "ablation.wo_heading_bbox",
                "wo_heading_bbox",
                "no_heading_bbox",
            ),
        ),
        "wo_heading_bbox",
        sample_id,
        bbox_format,
        normalized,
    )
    heatmap_path = optional_path(
        first(item, ("diagnostic.heatmap_path", "heatmap_path")), base_dir
    )
    heatmap_values = load_heatmap_values(
        first(item, ("diagnostic.heatmap", "heatmap")),
        heatmap_path,
        sample_id,
    )

    return Sample(
        sample_id=sample_id,
        query_path=query_path,
        retrieval_path=retrieval_path,
        oracle_path=oracle_path,
        altitude=as_optional_float(
            first(item, ("query.altitude", "altitude", "height")),
            "altitude",
            sample_id,
        ),
        heading=as_optional_float(
            first(item, ("query.heading", "heading", "angle")),
            "heading",
            sample_id,
        ),
        top1_correct=as_bool(
            first(item, ("retrieval.correct", "top1_correct", "retrieval_correct")),
            "top1_correct",
            sample_id,
        ),
        retrieval_score=as_optional_float(
            first(item, ("retrieval.score", "retrieval_score", "top1_score")),
            "retrieval_score",
            sample_id,
        ),
        gt_bbox=gt_bbox,
        pred_bbox=pred_bbox,
        oracle_iou=as_optional_float(
            first(item, ("oracle.iou", "oracle_iou", "iou")),
            "oracle_iou",
            sample_id,
        ),
        cde=as_optional_float(
            first(item, ("oracle.cde", "cde", "center_distance")),
            "cde",
            sample_id,
        ),
        cde_unit=str(
            first(item, ("oracle.cde_unit", "cde_unit"), defaults.get("cde_unit", "px"))
        ),
        wo_heading_bbox=wo_heading_bbox,
        wo_heading_iou=as_optional_float(
            first(
                item,
                (
                    "diagnostic.wo_heading_iou",
                    "ablation.wo_heading_iou",
                    "wo_heading_iou",
                    "no_heading_iou",
                ),
            ),
            "wo_heading_iou",
            sample_id,
        ),
        heatmap_path=heatmap_path,
        heatmap_values=heatmap_values,
    )


def load_samples(yaml_path: Path) -> List[Sample]:
    yaml_path = yaml_path.expanduser().resolve()
    if not yaml_path.exists():
        raise FileNotFoundError(f"Result YAML not found: {yaml_path}")
    with yaml_path.open("r", encoding="utf-8") as stream:
        payload = yaml.safe_load(stream)
    defaults: Mapping[str, Any] = {}
    if isinstance(payload, list):
        records = payload
    elif isinstance(payload, Mapping):
        defaults = payload.get("defaults", {}) or {}
        records = first(payload, ("samples", "results", "records", "query_records"))
    else:
        records = None
    if not isinstance(records, list) or not records:
        raise ValueError(
            "YAML must contain a non-empty samples/results/records/query_records list"
        )
    if not isinstance(defaults, Mapping):
        raise ValueError("defaults must be a mapping")

    samples: List[Sample] = []
    errors: List[str] = []
    for index, item in enumerate(records):
        if not isinstance(item, Mapping):
            errors.append(f"entry {index}: expected a mapping")
            continue
        try:
            sample = sample_from_mapping(item, index, yaml_path.parent, defaults)
            missing_paths = [
                str(path)
                for path in (sample.query_path, sample.retrieval_path, sample.oracle_path)
                if not path.is_file()
            ]
            if sample.heatmap_path is not None and not sample.heatmap_path.is_file():
                missing_paths.append(str(sample.heatmap_path))
            if missing_paths:
                raise FileNotFoundError("missing file(s): " + ", ".join(missing_paths))
            samples.append(sample)
        except (OSError, ValueError) as exc:
            errors.append(f"entry {index}: {exc}")
    if errors:
        print(f"[input] skipped {len(errors)} invalid sample(s):", file=sys.stderr)
        for error in errors[:20]:
            print(f"  - {error}", file=sys.stderr)
        if len(errors) > 20:
            print(f"  - ... and {len(errors) - 20} more", file=sys.stderr)
    if len(samples) < 4:
        raise ValueError(
            f"Need at least four valid samples, found {len(samples)}. "
            "Each sample requires three images plus gt_bbox and pred_bbox."
        )
    return samples


def compute_iou(
    box_a: Sequence[float], box_b: Sequence[float]
) -> float:
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union = area_a + area_b - intersection
    return intersection / union if union > 0.0 else 0.0


def compute_cde(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    center_a = ((box_a[0] + box_a[2]) / 2.0, (box_a[1] + box_a[3]) / 2.0)
    center_b = ((box_b[0] + box_b[2]) / 2.0, (box_b[1] + box_b[3]) / 2.0)
    return math.hypot(center_a[0] - center_b[0], center_a[1] - center_b[1])


def heading_difficulty(heading: Optional[float]) -> float:
    if heading is None:
        return 0.0
    wrapped = float(heading) % 90.0
    return min(wrapped, 90.0 - wrapped) / 45.0


def select_samples(
    samples: Sequence[Sample],
    success_iou: float,
    failure_iou: float,
    heading_gain_threshold: float,
) -> List[Selection]:
    enriched = [(sample, *sample.metrics()) for sample in samples]
    used: set[str] = set()
    selected: List[Selection] = []

    def choose(
        row: int,
        exact_predicate: Any,
        exact_key: Any,
        fallback_key: Any,
        exact_text: Any,
        fallback_text: Any,
    ) -> None:
        available = [entry for entry in enriched if entry[0].sample_id not in used]
        exact = [entry for entry in available if exact_predicate(entry)]
        is_exact = bool(exact)
        pool = exact if exact else available
        chosen = min(pool, key=exact_key if exact else fallback_key)
        sample = chosen[0]
        used.add(sample.sample_id)
        text = exact_text(chosen) if is_exact else fallback_text(chosen)
        selected.append(Selection(row, sample, is_exact, text))

    # Reserve the rarer diagnostic roles before selecting the comparatively
    # common end-to-end success row. This avoids consuming the only strong
    # heading-gain sample for Row 1.
    choose(
        2,
        lambda entry: entry[0].top1_correct and entry[1] < failure_iou,
        lambda entry: (entry[1], -entry[2], entry[0].sample_id),
        lambda entry: (
            0 if entry[0].top1_correct else 1,
            max(0.0, entry[1] - failure_iou),
            entry[1],
            entry[0].sample_id,
        ),
        lambda entry: f"exact grounding failure; oracle IoU={entry[1]:.3f}",
        lambda entry: (
            f"closest unused sample; top1={entry[0].top1_correct}, "
            f"oracle IoU={entry[1]:.3f}"
        ),
    )
    choose(
        3,
        lambda entry: (not entry[0].top1_correct) and entry[1] >= success_iou,
        lambda entry: (-entry[1], entry[2], entry[0].sample_id),
        lambda entry: (
            0 if not entry[0].top1_correct else 1,
            max(0.0, success_iou - entry[1]),
            -entry[1],
            entry[0].sample_id,
        ),
        lambda entry: f"exact retrieval bottleneck; oracle IoU={entry[1]:.3f}",
        lambda entry: (
            f"closest unused sample; top1={entry[0].top1_correct}, "
            f"oracle IoU={entry[1]:.3f}"
        ),
    )
    choose(
        4,
        lambda entry: entry[3] is not None and entry[3] >= heading_gain_threshold,
        lambda entry: (
            -float(entry[3]),
            -(entry[0].altitude or 0.0),
            -heading_difficulty(entry[0].heading),
            entry[0].sample_id,
        ),
        lambda entry: (
            0 if entry[3] is not None else 1,
            -float(entry[3] or 0.0),
            -(entry[0].altitude or 0.0),
            -heading_difficulty(entry[0].heading),
            entry[0].sample_id,
        ),
        lambda entry: (
            f"exact heading benefit; IoU gain={entry[3]:+.3f}, "
            f"altitude={format_number(entry[0].altitude)}"
        ),
        lambda entry: (
            "closest unused heading diagnostic; "
            + (
                f"IoU gain={entry[3]:+.3f}"
                if entry[3] is not None
                else "no paired w/o-heading IoU"
            )
            + f", altitude={format_number(entry[0].altitude)}"
        ),
    )
    choose(
        1,
        lambda entry: entry[0].top1_correct and entry[1] >= success_iou,
        lambda entry: (-entry[1], entry[2], entry[0].sample_id),
        lambda entry: (
            0 if entry[0].top1_correct else 1,
            max(0.0, success_iou - entry[1]),
            -entry[1],
            entry[0].sample_id,
        ),
        lambda entry: f"exact success; high oracle IoU={entry[1]:.3f}",
        lambda entry: (
            f"closest unused sample; top1={entry[0].top1_correct}, "
            f"oracle IoU={entry[1]:.3f}"
        ),
    )
    return sorted(selected, key=lambda item: item.row)


def format_number(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    if abs(value - round(value)) < 1e-6:
        return str(int(round(value)))
    return f"{value:.1f}"


def configure_fonts() -> str:
    available = {font.name for font in font_manager.fontManager.ttflist}
    selected = "Times New Roman" if "Times New Roman" in available else "DejaVu Serif"
    mpl.rcParams.update(
        {
            "font.family": selected,
            "font.size": 7.5,
            "axes.titlesize": 8.5,
            "axes.titleweight": "semibold",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )
    return selected


def open_rgb(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


def contain_image(
    image: Image.Image,
    canvas_size: Tuple[int, int] = (PANEL_WIDTH, PANEL_HEIGHT),
) -> Tuple[np.ndarray, float, float, float]:
    canvas_width, canvas_height = canvas_size
    width, height = image.size
    scale = min(canvas_width / width, canvas_height / height)
    scaled_size = (
        max(1, int(round(width * scale))),
        max(1, int(round(height * scale))),
    )
    resized = image.resize(scaled_size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", canvas_size, PADDING_COLOR)
    offset_x = (canvas_width - scaled_size[0]) / 2.0
    offset_y = (canvas_height - scaled_size[1]) / 2.0
    canvas.paste(resized, (int(round(offset_x)), int(round(offset_y))))
    return np.asarray(canvas), scale, offset_x, offset_y


def transform_box(
    box: Sequence[float], scale: float, offset_x: float, offset_y: float
) -> Tuple[float, float, float, float]:
    return (
        box[0] * scale + offset_x,
        box[1] * scale + offset_y,
        box[2] * scale + offset_x,
        box[3] * scale + offset_y,
    )


def add_box(
    ax: plt.Axes,
    box: Sequence[float],
    color: str,
    linestyle: str = "-",
    linewidth: float = 1.10,
) -> None:
    x1, y1, x2, y2 = box
    ax.add_patch(
        Rectangle(
            (x1, y1),
            max(0.0, x2 - x1),
            max(0.0, y2 - y1),
            fill=False,
            edgecolor=color,
            linewidth=linewidth,
            linestyle=linestyle,
            joinstyle="round",
            clip_on=True,
            zorder=5,
        )
    )


def badge(
    ax: plt.Axes,
    text: str,
    x: float,
    y: float,
    *,
    color: str = TEXT_COLOR,
    align: str = "left",
    vertical: str = "top",
    fontsize: float = 7.0,
    alpha: float = 0.88,
) -> None:
    ax.text(
        x,
        y,
        text,
        transform=ax.transAxes,
        ha=align,
        va=vertical,
        fontsize=fontsize,
        color=color,
        zorder=10,
        clip_on=True,
        bbox={
            "boxstyle": "round,pad=0.18",
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": alpha,
        },
    )


def show_base(ax: plt.Axes, image: Image.Image) -> Tuple[float, float, float]:
    canvas, scale, offset_x, offset_y = contain_image(image)
    ax.imshow(canvas, interpolation="none")
    ax.set_xlim(0, PANEL_WIDTH)
    ax.set_ylim(PANEL_HEIGHT, 0)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    return scale, offset_x, offset_y


def load_heatmap(sample: Sample) -> Optional[np.ndarray]:
    if sample.heatmap_values is not None:
        return sample.heatmap_values
    if sample.heatmap_path is None:
        return None
    with Image.open(sample.heatmap_path) as image:
        return np.asarray(ImageOps.exif_transpose(image).convert("L"), dtype=np.float32)


def normalized_heatmap(heatmap: np.ndarray) -> np.ndarray:
    finite = np.isfinite(heatmap)
    if not finite.any():
        return np.zeros_like(heatmap, dtype=np.float32)
    low, high = np.percentile(heatmap[finite], (2.0, 98.0))
    if high <= low:
        return np.zeros_like(heatmap, dtype=np.float32)
    result = (heatmap.astype(np.float32) - low) / (high - low)
    return np.clip(result, 0.0, 1.0)


def heatmap_canvas(
    heatmap: np.ndarray,
    source_size: Tuple[int, int],
    scale: float,
    offset_x: float,
    offset_y: float,
) -> np.ma.MaskedArray:
    source_width, source_height = source_size
    heatmap_image = Image.fromarray(
        np.uint8(normalized_heatmap(heatmap) * 255.0), mode="L"
    ).resize((source_width, source_height), Image.Resampling.BILINEAR)
    scaled_size = (
        max(1, int(round(source_width * scale))),
        max(1, int(round(source_height * scale))),
    )
    resized = np.asarray(
        heatmap_image.resize(scaled_size, Image.Resampling.BILINEAR),
        dtype=np.float32,
    ) / 255.0
    canvas = np.full((PANEL_HEIGHT, PANEL_WIDTH), np.nan, dtype=np.float32)
    x = int(round(offset_x))
    y = int(round(offset_y))
    end_x = min(PANEL_WIDTH, x + resized.shape[1])
    end_y = min(PANEL_HEIGHT, y + resized.shape[0])
    canvas[y:end_y, x:end_x] = resized[: end_y - y, : end_x - x]
    return np.ma.masked_invalid(canvas)


def draw_query(ax: plt.Axes, sample: Sample) -> None:
    image = open_rgb(sample.query_path)
    show_base(ax, image)
    altitude = (
        f"{format_number(sample.altitude)} m"
        if sample.altitude is not None
        else "alt. n/a"
    )
    heading = (
        f"{format_number(sample.heading)}°"
        if sample.heading is not None
        else "heading n/a"
    )
    badge(ax, f"{altitude} · {heading}", 0.025, 0.965)


def draw_retrieval(ax: plt.Axes, sample: Sample) -> None:
    image = open_rgb(sample.retrieval_path)
    show_base(ax, image)
    color = RETRIEVAL_SUCCESS_COLOR if sample.top1_correct else RETRIEVAL_FAILURE_COLOR
    status = "Top-1 correct" if sample.top1_correct else "Top-1 incorrect"
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(color)
        spine.set_linewidth(2.1)
    badge(ax, status, 0.025, 0.965, color=color, fontsize=7.2)
    if sample.retrieval_score is not None:
        badge(
            ax,
            f"score {sample.retrieval_score:.3f}",
            0.975,
            0.965,
            align="right",
            fontsize=6.5,
        )


def oracle_geometry(
    sample: Sample, image: Image.Image
) -> Tuple[Tuple[float, ...], Tuple[float, ...], Optional[Tuple[float, ...]]]:
    width, height = image.size
    gt = sample.gt_bbox.to_xyxy(width, height)
    pred = sample.pred_bbox.to_xyxy(width, height)
    wo = (
        sample.wo_heading_bbox.to_xyxy(width, height)
        if sample.wo_heading_bbox is not None
        else None
    )
    return gt, pred, wo


def draw_oracle(ax: plt.Axes, sample: Sample) -> None:
    image = open_rgb(sample.oracle_path)
    gt, pred, _ = oracle_geometry(sample, image)
    scale, offset_x, offset_y = show_base(ax, image)
    add_box(ax, transform_box(gt, scale, offset_x, offset_y), GT_COLOR)
    add_box(ax, transform_box(pred, scale, offset_x, offset_y), FULL_COLOR)
    iou, cde, _ = sample.metrics()
    unit = f" {sample.cde_unit}" if sample.cde_unit else ""
    badge(ax, f"IoU {iou:.2f} · CDE {cde:.1f}{unit}", 0.025, 0.965)
    if not sample.top1_correct:
        badge(
            ax,
            "Oracle diagnostic only",
            0.025,
            0.035,
            vertical="bottom",
            color=RETRIEVAL_FAILURE_COLOR,
            fontsize=6.7,
        )


def crop_for_zoom(
    image: Image.Image,
    boxes: Sequence[Sequence[float]],
) -> Tuple[Image.Image, List[Tuple[float, float, float, float]]]:
    width, height = image.size
    x1 = min(box[0] for box in boxes)
    y1 = min(box[1] for box in boxes)
    x2 = max(box[2] for box in boxes)
    y2 = max(box[3] for box in boxes)
    box_width = max(x2 - x1, width * 0.28)
    box_height = max(y2 - y1, height * 0.28)
    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0
    crop_width = min(width, box_width * 1.45)
    crop_height = min(height, box_height * 1.45)
    left = min(max(center_x - crop_width / 2.0, 0.0), width - crop_width)
    top = min(max(center_y - crop_height / 2.0, 0.0), height - crop_height)
    right = left + crop_width
    bottom = top + crop_height
    crop_bounds = (
        int(math.floor(left)),
        int(math.floor(top)),
        int(math.ceil(right)),
        int(math.ceil(bottom)),
    )
    cropped = image.crop(crop_bounds)
    shifted = [
        (
            box[0] - crop_bounds[0],
            box[1] - crop_bounds[1],
            box[2] - crop_bounds[0],
            box[3] - crop_bounds[1],
        )
        for box in boxes
    ]
    return cropped, shifted


def draw_diagnostic(
    ax: plt.Axes,
    sample: Sample,
    heatmap_cmap: str,
    heatmap_alpha: float,
) -> str:
    image = open_rgb(sample.oracle_path)
    gt, pred, wo = oracle_geometry(sample, image)
    if wo is not None:
        scale, offset_x, offset_y = show_base(ax, image)
        add_box(ax, transform_box(gt, scale, offset_x, offset_y), GT_COLOR)
        add_box(ax, transform_box(pred, scale, offset_x, offset_y), FULL_COLOR)
        add_box(
            ax,
            transform_box(wo, scale, offset_x, offset_y),
            WO_HEADING_COLOR,
            linestyle=(0, (4, 2)),
            linewidth=1.0,
        )
        _, _, gain = sample.metrics()
        label = "Heading ablation"
        if gain is not None:
            label += f" · ΔIoU {gain:+.2f}"
        badge(ax, label, 0.025, 0.965)
        return "ablation"

    heatmap = load_heatmap(sample)
    if heatmap is not None:
        scale, offset_x, offset_y = show_base(ax, image)
        overlay = heatmap_canvas(heatmap, image.size, scale, offset_x, offset_y)
        ax.imshow(
            overlay,
            cmap=heatmap_cmap,
            alpha=min(max(heatmap_alpha, 0.0), 1.0),
            vmin=0.0,
            vmax=1.0,
            interpolation="bilinear",
            zorder=3,
        )
        add_box(ax, transform_box(gt, scale, offset_x, offset_y), GT_COLOR)
        add_box(ax, transform_box(pred, scale, offset_x, offset_y), FULL_COLOR)
        badge(ax, "Grounding heatmap", 0.025, 0.965)
        return "heatmap"

    cropped, shifted = crop_for_zoom(image, (gt, pred))
    scale, offset_x, offset_y = show_base(ax, cropped)
    add_box(ax, transform_box(shifted[0], scale, offset_x, offset_y), GT_COLOR)
    add_box(ax, transform_box(shifted[1], scale, offset_x, offset_y), FULL_COLOR)
    badge(ax, "Local oracle zoom", 0.025, 0.965)
    return "zoom"


def make_figure(
    selections: Sequence[Selection],
    output: Path,
    png_output: Optional[Path],
    width: float,
    height: float,
    dpi: int,
    heatmap_cmap: str,
    heatmap_alpha: float,
) -> None:
    font_name = configure_fonts()
    print(f"[style] font: {font_name}")
    fig, axes = plt.subplots(
        4,
        4,
        figsize=(width, height),
        gridspec_kw={"wspace": 0.035, "hspace": 0.075},
    )
    titles = (
        r"(a) Drone query $q$",
        "(b) Top-1 retrieval",
        "(c) Oracle-tile grounding",
        "(d) Grounding diagnostic",
    )
    for column, title in enumerate(titles):
        axes[0, column].set_title(title, pad=4.0)

    diagnostic_modes: set[str] = set()
    for row, selection in enumerate(selections):
        sample = selection.sample
        draw_query(axes[row, 0], sample)
        draw_retrieval(axes[row, 1], sample)
        draw_oracle(axes[row, 2], sample)
        diagnostic_modes.add(
            draw_diagnostic(
                axes[row, 3],
                sample,
                heatmap_cmap=heatmap_cmap,
                heatmap_alpha=heatmap_alpha,
            )
        )

    legend_handles: List[Any] = [
        Line2D([0], [0], color=GT_COLOR, linewidth=1.8, label="Ground truth"),
        Line2D([0], [0], color=FULL_COLOR, linewidth=1.8, label="Full prediction"),
    ]
    if "ablation" in diagnostic_modes:
        wo_heading_handle = Line2D(
            [0],
            [0],
            color=WO_HEADING_COLOR,
            linewidth=1.8,
            linestyle=(0, (4, 2)),
            label="w/o heading",
        )
        wo_heading_handle.set_path_effects(
            [
                patheffects.Stroke(linewidth=2.8, foreground="#9A9A9A"),
                patheffects.Normal(),
            ]
        )
        legend_handles.append(wo_heading_handle)
    if "heatmap" in diagnostic_modes:
        legend_handles.append(
            Patch(facecolor=mpl.colormaps[heatmap_cmap](0.78), alpha=0.6, label="Heatmap")
        )
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.008),
        ncol=len(legend_handles),
        frameon=False,
        fontsize=7.2,
        handlelength=2.5,
        columnspacing=1.6,
    )
    fig.subplots_adjust(left=0.012, right=0.995, top=0.955, bottom=0.065)

    targets = [output]
    if png_output is not None and png_output.resolve() != output.resolve():
        targets.append(png_output)
    for target in targets:
        target = target.expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(target, dpi=dpi, facecolor="white")
        print(f"[output] saved: {target}")
    plt.close(fig)


def save_selection(path: Path, selections: Sequence[Selection]) -> None:
    payload = {
        "selected_rows": [
            {
                "row": selection.row,
                "sample_id": selection.sample.sample_id,
                "exact_match": selection.exact,
                "selection_logic": selection.explanation,
            }
            for selection in selections
        ]
    }
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(payload, stream, sort_keys=False, allow_unicode=True)
    print(f"[output] saved selection manifest: {path}")


def report_selection(
    selections: Sequence[Selection],
    success_iou: float,
    failure_iou: float,
    heading_gain: float,
) -> None:
    descriptions = [
        text.format(
            success_iou=success_iou,
            failure_iou=failure_iou,
            heading_gain=heading_gain,
        )
        for text in ROW_DESCRIPTIONS
    ]
    print("[selection] four qualitative rows:")
    for selection, target in zip(selections, descriptions):
        sample = selection.sample
        iou, cde, gain = sample.metrics()
        status = "exact" if selection.exact else "fallback"
        print(
            f"  Row {selection.row} [{status}] {sample.sample_id}: "
            f"target=({target}); top1={'correct' if sample.top1_correct else 'incorrect'}, "
            f"IoU={iou:.3f}, CDE={cde:.2f}, "
            f"heading_gain={gain if gain is not None else 'n/a'}"
        )
        print(f"           logic: {selection.explanation}")
    fallback_rows = [str(item.row) for item in selections if not item.exact]
    if fallback_rows:
        print(
            "[selection] thresholds were not met by unused samples for row(s) "
            + ", ".join(fallback_rows)
            + "; closest-condition fallbacks are explicitly reported above."
        )


def main() -> None:
    args = parse_args()
    if args.write_template is not None:
        write_template(args.write_template)
        return
    if args.results is None:
        raise ValueError("Internal error: missing result YAML argument")
    if not 6.8 <= args.width <= 7.4:
        print(
            f"[style] warning: width {args.width:.2f} in is outside the recommended "
            "ICLR full-width range (about 7.0-7.2 in).",
            file=sys.stderr,
        )
    samples = load_samples(args.results)
    print(f"[input] loaded {len(samples)} valid samples from {args.results.resolve()}")
    selections = select_samples(
        samples,
        success_iou=args.success_iou,
        failure_iou=args.failure_iou,
        heading_gain_threshold=args.heading_gain,
    )
    report_selection(
        selections,
        success_iou=args.success_iou,
        failure_iou=args.failure_iou,
        heading_gain=args.heading_gain,
    )
    make_figure(
        selections,
        output=args.output,
        png_output=args.png_output,
        width=args.width,
        height=args.height,
        dpi=args.dpi,
        heatmap_cmap=args.heatmap_cmap,
        heatmap_alpha=args.heatmap_alpha,
    )
    if args.selected_yaml is not None:
        save_selection(args.selected_yaml, selections)


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as exc:
        print(f"[error] {exc}", file=sys.stderr)
        print(
            "[hint] The path must point to a real per-sample result YAML; "
            "experiments/results.yaml is not included in this repository.",
            file=sys.stderr,
        )
        print(
            "[hint] Create an editable schema template with:\n"
            "       python tool/generate_qualitative_figure.py "
            "--write-template experiments/results.yaml",
            file=sys.stderr,
        )
        raise SystemExit(2) from None
    except (ValueError, yaml.YAMLError) as exc:
        print(f"[error] {exc}", file=sys.stderr)
        raise SystemExit(2) from None
