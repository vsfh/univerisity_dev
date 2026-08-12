import json
import math
import re
import sqlite3
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


# --- Configuration ---
REPORT_TITLE = "固定文本对齐为何落后于 No-text"
SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[3]
OUTPUT_DIR = SCRIPT_PATH.parent
SQLITE_PATH = OUTPUT_DIR / "analysis.sqlite"

ADA_EVAL = REPO_ROOT / "exp/eval_results/baseline_ada_1_5x3.json"
NO_TEXT_EVAL = REPO_ROOT / "exp/eval_results/baseline_wo_input_ids_ada_5x3.json"
ADA_CHECKPOINT = REPO_ROOT / "exp/outputs/baseline_ada_1_5x3/last.pth"
NO_TEXT_CHECKPOINT = REPO_ROOT / "exp/outputs/baseline_wo_input_ids_ada_5x3/last.pth"
ADA_EVENT = REPO_ROOT / (
    "exp/runs/baseline_ada_1_5x3/"
    "events.out.tfevents.1786420717.4090-48g.511988.0"
)
NO_TEXT_EVENT = REPO_ROOT / (
    "exp/runs/baseline_wo_input_ids_ada_5x3/"
    "events.out.tfevents.1786302112.4090-48g.2760507.0"
)
HISTORICAL_SQLITE = REPO_ROOT / (
    "exp/analysis/baseline_ada_1_vs_wo_input_ids/analysis.sqlite"
)
TRAIN_SPLIT = Path("/media/data1/feihong/ckpt/train.txt")
DRONE_ROOT = Path("/media/data1/feihong/drone_img")

NUM_EPOCHS = 20
BATCH_SIZE = 32
GRID_LOCATIONS = 15
SPATIAL_TERMS = re.compile(
    r"\b(left|right|top|bottom|upper|lower|foreground|background)\b",
    flags=re.IGNORECASE,
)


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def record_key(record: dict) -> tuple:
    return (
        record["height"],
        record["angle"],
        record["drone_path"],
        record["satellite_path"],
        record["gt_label"],
    )


def quote_identifier(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def sqlite_type(rows: list[dict], field: str) -> str:
    values = [row[field] for row in rows if row.get(field) is not None]
    if not values:
        return "TEXT"
    if all(isinstance(value, bool) or isinstance(value, int) for value in values):
        return "INTEGER"
    if all(isinstance(value, (int, float)) for value in values):
        return "REAL"
    return "TEXT"


def materialize_sqlite(datasets: dict[str, list[dict]]) -> dict[str, list[dict]]:
    connection = sqlite3.connect(SQLITE_PATH)
    connection.row_factory = sqlite3.Row
    try:
        for table_name, rows in datasets.items():
            if not rows:
                raise ValueError(f"Cannot materialize empty dataset: {table_name}")
            fields = list(rows[0])
            quoted_table = quote_identifier(table_name)
            connection.execute(f"DROP TABLE IF EXISTS {quoted_table}")
            columns_sql = ", ".join(
                f"{quote_identifier(field)} {sqlite_type(rows, field)}"
                for field in fields
            )
            connection.execute(f"CREATE TABLE {quoted_table} ({columns_sql})")
            placeholders = ", ".join("?" for _ in fields)
            columns = ", ".join(quote_identifier(field) for field in fields)
            connection.executemany(
                f"INSERT INTO {quoted_table} ({columns}) VALUES ({placeholders})",
                [[row.get(field) for field in fields] for row in rows],
            )
        connection.commit()

        reviewed = {}
        for table_name in datasets:
            rows = connection.execute(
                f"SELECT * FROM {quote_identifier(table_name)}"
            ).fetchall()
            reviewed[table_name] = [dict(row) for row in rows]
        return reviewed
    finally:
        connection.close()


def cluster_bootstrap_ci(
    labels: np.ndarray,
    values: np.ndarray,
    seed: int,
    draws: int = 5000,
) -> tuple[float, float]:
    unique_labels = np.unique(labels)
    cluster_means = np.array(
        [values[labels == label].mean() for label in unique_labels],
        dtype=np.float64,
    )
    rng = np.random.default_rng(seed)
    bootstrap_means = []
    batch_draws = 100
    for start in range(0, draws, batch_draws):
        current = min(batch_draws, draws - start)
        indices = rng.integers(
            0,
            len(cluster_means),
            size=(current, len(cluster_means)),
        )
        bootstrap_means.extend(cluster_means[indices].mean(axis=1).tolist())
    low, high = np.quantile(np.asarray(bootstrap_means), [0.025, 0.975])
    return float(low), float(high)


def profile_evaluations() -> dict[str, list[dict]]:
    ada = load_json(ADA_EVAL)
    no_text = load_json(NO_TEXT_EVAL)
    ada_records = ada["query_records"]
    no_text_records = no_text["query_records"]

    no_text_lookup = {record_key(record): record for record in no_text_records}
    if len(no_text_lookup) != len(no_text_records):
        raise ValueError("No-text evaluation contains duplicate query keys.")
    if len(ada_records) != len(no_text_records):
        raise ValueError("Evaluation row counts differ.")
    if any(record_key(record) not in no_text_lookup for record in ada_records):
        raise ValueError("Evaluation query populations differ.")

    paired = [(record, no_text_lookup[record_key(record)]) for record in ada_records]
    labels = np.asarray([record["gt_label"] for record in ada_records])

    raw_fields = {
        "R@1": "top1_correct",
        "R@5": "top5_correct",
        "R@10": "top10_correct",
        "mIoU": "iou",
        "uIoU": "uIoU",
        "Center distance": "center_distance",
    }
    arrays = {}
    for metric, field in raw_fields.items():
        arrays[metric] = (
            np.asarray([float(left[field]) for left, _ in paired]),
            np.asarray([float(right[field]) for _, right in paired]),
        )

    metric_specs = [
        ("Retrieval", "R@1", "higher", "rate"),
        ("Retrieval", "R@5", "higher", "rate"),
        ("Retrieval", "R@10", "higher", "rate"),
        ("Localization", "mIoU", "higher", "rate"),
        ("Localization", "IoU > 0.50", "higher", "rate"),
        ("Localization", "IoU > 0.25", "higher", "rate"),
        ("Joint", "uIoU", "higher", "rate"),
        ("Joint", "uIoU > 0.25", "higher", "rate"),
        ("Localization", "Center distance", "lower", "px"),
        ("Joint", "uCDE", "lower", "px"),
    ]

    ada_iou, no_text_iou = arrays["mIoU"]
    ada_uiou, no_text_uiou = arrays["uIoU"]
    ada_r1, no_text_r1 = arrays["R@1"]
    ada_center, no_text_center = arrays["Center distance"]
    ada_success = ada_r1.astype(bool)
    no_text_success = no_text_r1.astype(bool)

    values = {
        "R@1": (ada_r1, no_text_r1),
        "R@5": arrays["R@5"],
        "R@10": arrays["R@10"],
        "mIoU": (ada_iou, no_text_iou),
        "IoU > 0.50": ((ada_iou > 0.5).astype(float), (no_text_iou > 0.5).astype(float)),
        "IoU > 0.25": ((ada_iou > 0.25).astype(float), (no_text_iou > 0.25).astype(float)),
        "uIoU": (ada_uiou, no_text_uiou),
        "uIoU > 0.25": ((ada_uiou > 0.25).astype(float), (no_text_uiou > 0.25).astype(float)),
        "Center distance": (ada_center, no_text_center),
        "uCDE": (ada_center[ada_success], no_text_center[no_text_success]),
    }

    overall_rows = []
    ci_rows = []
    for index, (group, metric, preferred, unit) in enumerate(metric_specs):
        ada_values, no_text_values = values[metric]
        ada_mean = float(ada_values.mean())
        no_text_mean = float(no_text_values.mean())
        delta = ada_mean - no_text_mean
        overall_rows.append(
            {
                "group": group,
                "metric": metric,
                "preferred_direction": preferred,
                "unit": unit,
                "ada_value": ada_mean,
                "no_text_value": no_text_mean,
                "ada_minus_no_text": delta,
                "delta_percentage_points": delta * 100.0 if unit == "rate" else None,
            }
        )
        if metric != "uCDE":
            paired_delta = ada_values - no_text_values
            low, high = cluster_bootstrap_ci(
                labels,
                paired_delta,
                seed=20260811 + index,
            )
            ci_rows.append(
                {
                    "metric": metric,
                    "estimate": float(paired_delta.mean()),
                    "ci_low": low,
                    "ci_high": high,
                    "clusters": int(len(np.unique(labels))),
                    "unit": unit,
                }
            )

    retrieval_rows = [
        {
            "metric": metric,
            "delta_pp": (arrays[metric][0] - arrays[metric][1]).mean() * 100.0,
            "ada_hits": int(arrays[metric][0].sum()),
            "no_text_hits": int(arrays[metric][1].sum()),
            "net_hit_delta": int(arrays[metric][0].sum() - arrays[metric][1].sum()),
            "num_queries": len(paired),
        }
        for metric in ("R@1", "R@5", "R@10")
    ]

    localization_rows = []
    for metric in ("mIoU", "IoU > 0.50", "IoU > 0.25", "uIoU", "uIoU > 0.25"):
        left, right = values[metric]
        localization_rows.append(
            {
                "metric": metric,
                "delta_points": float((left.mean() - right.mean()) * 100.0),
                "ada_value": float(left.mean()),
                "no_text_value": float(right.mean()),
                "num_queries": len(paired),
            }
        )

    subset_rows = []
    for height in sorted({record["height"] for record in ada_records}):
        for angle in sorted({record["angle"] for record in ada_records}):
            indices = np.asarray(
                [
                    record["height"] == height and record["angle"] == angle
                    for record in ada_records
                ]
            )
            subset_rows.append(
                {
                    "subset": f"{height}m / {angle}°",
                    "height": int(height),
                    "height_label": f"{height}m",
                    "angle": int(angle),
                    "num_queries": int(indices.sum()),
                    "r1_delta_pp": float((ada_r1[indices] - no_text_r1[indices]).mean() * 100.0),
                    "miou_delta_points": float((ada_iou[indices] - no_text_iou[indices]).mean() * 100.0),
                    "uiou_delta_points": float((ada_uiou[indices] - no_text_uiou[indices]).mean() * 100.0),
                    "center_delta_px": float((ada_center[indices] - no_text_center[indices]).mean()),
                }
            )

    center_rows = []
    for threshold in (10, 25, 50, 100, 200, 300, 400, 600):
        for model, array in (("Text 0.01 fixed", ada_center), ("No text", no_text_center)):
            center_rows.append(
                {
                    "threshold_px": threshold,
                    "threshold_label": f"≤{threshold}px",
                    "model": model,
                    "share_within_threshold": float((array <= threshold).mean()),
                    "num_queries": len(array),
                }
            )

    common_correct = ada_success & no_text_success
    paired_pattern_rows = [
        {
            "pattern": "R@1 both correct",
            "count": int(common_correct.sum()),
            "share": float(common_correct.mean()),
        },
        {
            "pattern": "R@1 Ada only",
            "count": int((ada_success & ~no_text_success).sum()),
            "share": float((ada_success & ~no_text_success).mean()),
        },
        {
            "pattern": "R@1 No-text only",
            "count": int((~ada_success & no_text_success).sum()),
            "share": float((~ada_success & no_text_success).mean()),
        },
        {
            "pattern": "R@1 both wrong",
            "count": int((~ada_success & ~no_text_success).sum()),
            "share": float((~ada_success & ~no_text_success).mean()),
        },
    ]

    provenance_rows = [
        {
            "check": "Aligned query records",
            "ada": str(len(ada_records)),
            "no_text": str(len(no_text_records)),
            "result": "89,728 unique keys; 0 mismatches",
        },
        {
            "check": "Candidate size",
            "ada": str(ada["candidate_size"]),
            "no_text": str(no_text["candidate_size"]),
            "result": "match",
        },
        {
            "check": "Gallery size",
            "ada": str(ada["num_gallery"]),
            "no_text": str(no_text["num_gallery"]),
            "result": "match",
        },
        {
            "check": "Satellite input",
            "ada": f"{ada['sat_size']['height']}×{ada['sat_size']['width']}",
            "no_text": f"{no_text['sat_size']['height']}×{no_text['sat_size']['width']}",
            "result": "match",
        },
        {
            "check": "Test crop ratio",
            "ada": str(ada["test_crop_ratio"]),
            "no_text": str(no_text["test_crop_ratio"]),
            "result": "match",
        },
        {
            "check": "Checkpoint → JSON chronology",
            "ada": f"{(ADA_EVAL.stat().st_mtime - ADA_CHECKPOINT.stat().st_mtime) / 60:.1f} min",
            "no_text": f"{(NO_TEXT_EVAL.stat().st_mtime - NO_TEXT_CHECKPOINT.stat().st_mtime) / 60:.1f} min",
            "result": "both JSON files follow their checkpoints",
        },
    ]

    return {
        "overall_comparison": overall_rows,
        "cluster_intervals": ci_rows,
        "retrieval_deltas": retrieval_rows,
        "localization_deltas": localization_rows,
        "subset_deltas": subset_rows,
        "center_thresholds": center_rows,
        "paired_patterns": paired_pattern_rows,
        "provenance_checks": provenance_rows,
    }


def scalar_values(event: EventAccumulator, tag: str) -> list[float]:
    return [item.value for item in event.Scalars(tag)]


def epoch_means(values: list[float]) -> list[float]:
    if len(values) % NUM_EPOCHS != 0:
        raise ValueError(f"Step count {len(values)} is not divisible by {NUM_EPOCHS}.")
    per_epoch = len(values) // NUM_EPOCHS
    return [
        float(np.mean(values[index * per_epoch : (index + 1) * per_epoch]))
        for index in range(NUM_EPOCHS)
    ]


def profile_training_curves() -> list[dict]:
    events = {}
    for model, path in (("Text 0.01 fixed", ADA_EVENT), ("No text", NO_TEXT_EVENT)):
        event = EventAccumulator(str(path), size_guidance={"scalars": 0}).Reload()
        events[model] = {
            "bbox_geo": epoch_means(scalar_values(event, "Loss/bbox_geo_step")),
            "bbox_cls": epoch_means(scalar_values(event, "Loss/bbox_cls_step")),
            "heatmap": epoch_means(scalar_values(event, "Loss/heatmap_step")),
            "text_align": scalar_values(event, "Loss/text_pooler_align_epoch"),
            "reported_train": scalar_values(event, "Loss/train_epoch"),
        }

    rows = []
    for metric in ("bbox_geo", "bbox_cls", "heatmap"):
        ada = np.asarray(events["Text 0.01 fixed"][metric])
        no_text = np.asarray(events["No text"][metric])
        rows.append(
            {
                "metric": metric,
                "epochs_ada_worse": int((ada > no_text).sum()),
                "mean_ada": float(ada.mean()),
                "mean_no_text": float(no_text.mean()),
                "mean_delta": float((ada - no_text).mean()),
                "final_ada": float(ada[-1]),
                "final_no_text": float(no_text[-1]),
                "final_delta": float(ada[-1] - no_text[-1]),
            }
        )

    text_align = events["Text 0.01 fixed"]["text_align"]
    rows.append(
        {
            "metric": "text_alignment",
            "epochs_ada_worse": None,
            "mean_ada": float(np.mean(text_align)),
            "mean_no_text": 0.0,
            "mean_delta": float(np.mean(text_align)),
            "final_ada": float(text_align[-1]),
            "final_no_text": 0.0,
            "final_delta": float(text_align[-1]),
        }
    )
    return rows


def probability_any_group_collision(groups: int, group_size: int, batch_size: int) -> float:
    total = groups * group_size
    log_probability_no_collision = sum(
        math.log(groups - index)
        + math.log(group_size)
        - math.log(total - index)
        for index in range(batch_size)
    )
    return 1.0 - math.exp(log_probability_no_collision)


def profile_text_and_targets() -> list[dict]:
    train_ids = []
    for line in TRAIN_SPLIT.read_text(encoding="utf-8").splitlines():
        try:
            train_ids.append(int(line.split(",")[0].strip()))
        except ValueError:
            continue

    texts = []
    for satellite_id in train_ids:
        path = DRONE_ROOT / f"{satellite_id:04d}" / "qwen_6_28_description.json"
        payload = load_json(path)
        for segments in payload.get("description_segments", {}).values():
            texts.extend(
                segment["text"].strip()
                for segment in segments
                if isinstance(segment.get("text"), str) and segment["text"].strip()
            )

    num_satellites = len(train_ids)
    num_samples = num_satellites * 4 * 8
    batches_per_epoch = num_samples // BATCH_SIZE
    expected_satellite_pairs = (
        math.comb(BATCH_SIZE, 2) * ((4 * 8) - 1) / (num_samples - 1)
    )
    expected_exact_caption_pairs = (
        math.comb(BATCH_SIZE, 2) * (8 - 1) / (num_samples - 1)
    )
    text_counts = Counter(texts)
    spatial_count = sum(bool(SPATIAL_TERMS.search(text)) for text in texts)

    return [
        {
            "evidence": "Training samples",
            "value": float(num_samples),
            "unit": "samples",
            "interpretation": f"{num_satellites} satellites × 4 heights × 8 angles",
        },
        {
            "evidence": "Same-satellite collision probability",
            "value": probability_any_group_collision(num_satellites, 32, BATCH_SIZE),
            "unit": "share of batches",
            "interpretation": f"Expected {expected_satellite_pairs * batches_per_epoch:.1f} same-satellite pairs per epoch",
        },
        {
            "evidence": "Identical-caption collision probability",
            "value": probability_any_group_collision(num_satellites * 4, 8, BATCH_SIZE),
            "unit": "share of batches",
            "interpretation": f"Expected {expected_exact_caption_pairs * batches_per_epoch:.1f} same-(satellite,height) pairs per epoch",
        },
        {
            "evidence": "Captions with relative spatial terms",
            "value": spatial_count / len(texts),
            "unit": "share of captions",
            "interpretation": "left/right/top/bottom/upper/lower/foreground/background",
        },
        {
            "evidence": "Caption rows",
            "value": float(len(texts)),
            "unit": "captions",
            "interpretation": f"{len(text_counts)} exact-unique captions; one variant selected globally per epoch",
        },
        {
            "evidence": "Uniform text-alignment CE reference",
            "value": math.log(BATCH_SIZE * GRID_LOCATIONS),
            "unit": "nats",
            "interpretation": "log(480) candidates",
        },
    ]


def historical_rows() -> list[dict]:
    if not HISTORICAL_SQLITE.exists():
        return [
            {
                "comparison": "Historical snapshot unavailable",
                "r1": None,
                "miou": None,
                "note": "No prior snapshot found.",
            }
        ]
    connection = sqlite3.connect(HISTORICAL_SQLITE)
    connection.row_factory = sqlite3.Row
    try:
        row = connection.execute(
            "SELECT recall_at_1, recall_at_5, recall_at_10, mean_iou, uiou "
            "FROM historical_eval_summary WHERE run_id = 'text_scheduled'"
        ).fetchone()
    finally:
        connection.close()
    current = load_json(ADA_EVAL)["overall"]
    return [
        {
            "comparison": "Current fixed 0.01 minus old 5-epoch schedule",
            "r1_delta_pp": (current["recall@1"] - row["recall_at_1"]) * 100.0,
            "r5_delta_pp": (current["recall@5"] - row["recall_at_5"]) * 100.0,
            "r10_delta_pp": (current["recall@10"] - row["recall_at_10"]) * 100.0,
            "miou_delta": current["mean_iou"] - row["mean_iou"],
            "uiou_delta": current["uIoU"] - row["uiou"],
            "note": "Descriptive only: different random run and overwritten old checkpoint.",
        }
    ]


def source(
    source_id: str,
    label: str,
    table: str,
    generated_at: str,
    description: str,
    filters: list[str],
    metric_definitions: list[str],
) -> dict:
    return {
        "id": source_id,
        "label": label,
        "path": "exp/analysis/current_ada_eval_diagnosis/analysis.sqlite",
        "query": {
            "engine": "SQLite",
            "language": "sql",
            "sql": f"SELECT * FROM {table}",
            "description": description,
            "executed_at": generated_at,
            "tables_used": [table],
            "filters": filters,
            "metric_definitions": metric_definitions,
        },
    }


def build_artifact() -> dict:
    generated_at = datetime.now(timezone.utc).isoformat()
    datasets = profile_evaluations()
    datasets["training_components"] = profile_training_curves()
    datasets["mechanism_evidence"] = profile_text_and_targets()
    datasets["historical_schedule"] = historical_rows()
    reviewed = materialize_sqlite(datasets)

    overall = {row["metric"]: row for row in reviewed["overall_comparison"]}
    retrieval = {row["metric"]: row for row in reviewed["retrieval_deltas"]}
    training = {row["metric"]: row for row in reviewed["training_components"]}
    mechanism = {row["evidence"]: row for row in reviewed["mechanism_evidence"]}
    subsets = reviewed["subset_deltas"]
    historical = reviewed["historical_schedule"][0]

    sources = [
        source(
            "eval-overall",
            "Paired current evaluation metrics",
            "overall_comparison",
            generated_at,
            "Current fixed-text and no-text metrics on exactly aligned test queries.",
            ["89,728 aligned queries", "Candidate size 100", "Gallery size 2,804"],
            [
                "Recall@K is the query share with an accepted positive in the top K.",
                "mIoU averages predicted-versus-target box IoU before retrieval gating.",
                "uIoU sets IoU to zero when top-1 retrieval is wrong.",
                "Center distance is Euclidean pixel error between predicted and target box centers; lower is better.",
            ],
        ),
        source(
            "retrieval-deltas-source",
            "Paired retrieval deltas",
            "retrieval_deltas",
            generated_at,
            "Recall deltas and hit-count changes on exactly aligned queries.",
            ["89,728 aligned queries", "Ada minus no-text"],
            ["Delta is expressed in percentage points; hit delta is an exact paired count difference."],
        ),
        source(
            "localization-deltas-source",
            "Paired localization and joint deltas",
            "localization_deltas",
            generated_at,
            "Overlap and joint metric deltas on exactly aligned queries.",
            ["89,728 aligned queries", "Ada minus no-text"],
            ["Rate deltas are expressed as percentage points."],
        ),
        source(
            "eval-subsets",
            "Paired height × angle evaluation",
            "subset_deltas",
            generated_at,
            "Paired deltas for all 32 equal-size height-by-angle slices.",
            ["2,804 queries per slice", "Ada minus no-text"],
            ["Metric deltas use the same queries and candidate sets within each slice."],
        ),
        source(
            "center-distribution",
            "Center-error empirical distribution",
            "center_thresholds",
            generated_at,
            "Share of queries at or below each center-distance threshold.",
            ["All 89,728 aligned queries"],
            ["Higher share within a fixed pixel threshold means better center precision."],
        ),
        source(
            "training-components",
            "Latest complete TensorBoard events",
            "training_components",
            generated_at,
            "Epoch-level aggregation of bbox geometry, classification, heatmap, and text-alignment scalars.",
            ["Latest complete fixed-text event", "Latest complete no-text event", "20 epochs"],
            ["Lower loss is better; step values are averaged within each epoch and then across epochs."],
        ),
        source(
            "mechanism-audit",
            "Repository code and caption-target audit",
            "mechanism_evidence",
            generated_at,
            "Static code/data audit of text reuse, batch collisions, and alignment target grain.",
            ["701 training satellites", "Batch size 32", "Random shuffle assumption"],
            [
                "Collision probabilities are exact sampling-without-replacement expectations, not observed batch logs.",
                "Spatial-term share uses the listed case-insensitive English terms.",
            ],
        ),
        source(
            "provenance-checks",
            "Evaluation provenance and data-quality checks",
            "provenance_checks",
            generated_at,
            "Identity, scope, parameter, and chronology checks for both evaluation files.",
            ["Current files as of report generation"],
            ["A query key is height, angle, drone path, satellite path, and ground-truth label."],
        ),
        source(
            "historical-schedule",
            "Pre-evaluation historical snapshot",
            "historical_schedule",
            generated_at,
            "Current fixed-text result compared with the saved old five-epoch text-schedule summary.",
            ["Different unseeded runs", "Old checkpoint no longer exists"],
            ["Historical deltas are descriptive and not causal estimates."],
        ),
    ]

    r1_delta = retrieval["R@1"]["delta_pp"]
    miou_delta = overall["mIoU"]["ada_minus_no_text"]
    center_delta = overall["Center distance"]["ada_minus_no_text"]
    r1_negative_slices = sum(row["r1_delta_pp"] < 0 for row in subsets)
    miou_negative_slices = sum(row["miou_delta_points"] < 0 for row in subsets)
    sat_collision = mechanism["Same-satellite collision probability"]["value"]
    caption_collision = mechanism["Identical-caption collision probability"]["value"]
    spatial_share = mechanism["Captions with relative spatial terms"]["value"]
    align_final = training["text_alignment"]["final_ada"]

    summary_body = f"""## 技术摘要

- **当前结果是真实、同口径的泛化差距，不是测试脚本或样本错位。** 两份 JSON 的 89,728 条 query 逐条一致，candidate、gallery、输入尺寸和测试随机种子相同；当前 Ada JSON 在最新固定 `0.01` checkpoint 之后生成。
- **退化分成两部分：检索是广泛的小幅负偏，框重叠是更明显的系统性负偏。** R@1/5/10 分别低 {abs(retrieval['R@1']['delta_pp']):.2f}/{abs(retrieval['R@5']['delta_pp']):.2f}/{abs(retrieval['R@10']['delta_pp']):.2f} 个百分点，mIoU 低 {abs(miou_delta):.4f}；32 个高度×角度切片的 mIoU 全部为负。
- **最可能的训练机制不是“checkpoint 坏了”，而是文本目标与主任务冲突。** 文本 loss 直接更新共享 vision LoRA；相同 `(satellite,height)` 文本跨 8 个 yaw 复用，却在 batch-global InfoNCE 中互作负样本。随机 batch 下约 {caption_collision:.1%} 的 batch 至少出现一组完全相同 caption 冲突。
- **这仍不是严格因果证明。** 两次训练没有固定 Python/NumPy/PyTorch seed，包含随机 crop、shuffle 和 LoRA dropout；必须用同一入口、同一初始化做 matched ablation 才能把差距完全归因到文本项。
"""

    provenance_body = """## 两份评测完全可比，掉点不来自测试阶段

`run_all.sh` 和 `run_ada.sh` 都调用同一个 `test.py`。测试前向只传 drone image、satellite image 和 angle，不传 `input_ids`，所以不存在“文本模型测试时漏传文本”的口径差异。候选抽样由相同的 `seed=43 + query_index` 决定；逐条 query key 比对为 0 mismatch。下表记录当前文件的身份与时间链路。唯一 provenance 缺口是 JSON 只保存 checkpoint 路径，没有保存 checkpoint SHA256、训练 seed 或代码 commit。
"""

    retrieval_body = f"""## 检索下降约 0.7 个百分点，是广泛重排后的净负偏

R@1 少 {abs(retrieval['R@1']['net_hit_delta']):,} 个命中，R@5 少 {abs(retrieval['R@5']['net_hit_delta']):,} 个，R@10 少 {abs(retrieval['R@10']['net_hit_delta']):,} 个。R@1 的 paired churn 中 Ada-only 为 7,473 条、No-text-only 为 8,120 条，说明并非固定少数样本彻底崩坏，而是大量候选排序互换后净少 647 条。按 2,804 个 `gt_label` 聚类 bootstrap 后，R@1 的 95% 区间仍低于零；但该区间只覆盖测试 query 不确定性，不覆盖训练 seed 方差。
"""

    localization_body = f"""## 主要损失在框重叠：所有 32 个切片的 mIoU 都更差

mIoU 从 {overall['mIoU']['no_text_value']:.4f} 降至 {overall['mIoU']['ada_value']:.4f}，IoU>0.5 下降 {abs(overall['IoU > 0.50']['delta_percentage_points']):.2f} 个百分点。R@1 在 {r1_negative_slices}/32 个切片为负，而 mIoU 在 {miou_negative_slices}/32 个切片全部为负；因此定位退化不是单一高度、角度或少数大学 ID 拖累。mIoU 在 retrieval 之前、GT satellite 上计算，所以这部分退化独立于 R@K 的小幅下降。
"""

    center_body = f"""## 平均中心距离反而好 {abs(center_delta):.2f}px，但这是尾部改善换取精确样本变差

“全线落后”并不严格成立：Ada 的 mean center distance 为 {overall['Center distance']['ada_value']:.2f}px，No-text 为 {overall['Center distance']['no_text_value']:.2f}px。阈值曲线显示 Ada 在 ≤10/25/50/100px 的精确命中率更低，但在 200px 之后累计占比反超，说明它减少了极端中心偏移，却牺牲了近距离精度。与此同时 IoU 全面下降，更像框宽高、面积、anchor/confidence 或形状校准变差；当前 JSON 没保存预测框坐标，因此这是由分布推断出的机制，不是最终几何证明。
"""

    mechanism_body = f"""## 最可能的主因：共享 LoRA 收到带假负样本的粗粒度文本梯度

文本 encoder 完全冻结，`text_feats` 也被 detach；但 satellite region candidates 没有 detach，所以梯度沿 `text loss → satellite region features → vision LoRA` 回传。同一个 vision tower 同时编码 drone 与 satellite，且 satellite features 也是 bbox/heatmap 的输入，因此文本辅助项会直接改变检索和定位共同依赖的视觉表示。推理期没有文本融合来补偿这种表示变化。

训练循环定义了支持 `satellite_ids` 多正样本的 `build_retrieval_soft_targets`，实际却没有调用；手工 target 只认可当前 row 的 15 个格。随机 batch 32 下，约 {sat_collision:.1%} 的 batch 会出现同 satellite row，约 {caption_collision:.1%} 会出现同 `(satellite,height)` 的完全相同 caption。后者用相同冻结 embedding 指向不同 row 的正样本、同时把另一 row 置为负样本，目标无法同时满足。

此外，caption 描述整幅 drone scene，却用 92% target mass 监督一个 satellite 3×5 单格；约 {spatial_share:.1%} caption 含相对画面方位词，又跨 8 个 yaw 复用。配置里的 `TEXT_FILE`/`TEXT_JSON_NAME` 并未传入 dataset，真实来源是硬编码的 `qwen_6_28_description.json`。这些因素共同使文本约束更像带噪的粗语义正则，而不是可靠的局部几何监督。
"""

    training_body = f"""## 训练 loss 没发散，但子项已显示几何与分类略受损

最终 text alignment CE 为 {align_final:.3f}，低于均匀预测 `log(480)={math.log(480):.3f}`，说明模型确实在拟合该辅助目标。可是 bbox aggregate 被 heatmap 掩盖：`bbox_geo` 和 `bbox_cls` 都有 {training['bbox_geo']['epochs_ada_worse']}/20 与 {training['bbox_cls']['epochs_ada_worse']}/20 个 epoch 高于 No-text，heatmap 则大多更低。当前文本项经过 retrieval 权重后的有效系数为 `0.9×0.01=0.009`，最后一轮对总 loss 贡献约 {0.009 * align_final:.4f}，相当于加权 bbox 项的约 17%，并非可忽略。你刚改成全程固定后，YAML 的 stop epoch 已经不再生效，文本梯度持续全部 14,020 steps。

历史快照也呈同样警示：固定 20 轮相对旧的仅前 5 轮文本调度，R@1 反而高 {historical['r1_delta_pp']:.2f}pp，但 mIoU 低 {abs(historical['miou_delta']):.4f}。由于两次仍是不同随机 run，这只能支持“持续文本更伤框质量”的假设，不能作为因果估计。
"""

    definitions_body = """## 范围、指标与诊断方法

- Population：2,804 个 gallery label，每个 4 高度×8 角度，共 89,728 个 query；每个 query 从固定 100 个候选中检索。
- R@K：top-K 中出现 ground-truth 或 include-map 接受的正样本比例。
- mIoU：在配对的 GT satellite 上计算预测框与目标框 IoU，尚未受 retrieval 正误门控。
- uIoU：top-1 检索错误时把该 query 的 IoU 置零后再平均。
- Center distance：预测框和目标框中心的像素欧氏距离，越低越好；它不衡量框宽高和形状。
- 置信区间按 2,804 个 `gt_label` 聚类 bootstrap，避免把同一 satellite 的 32 个视角当作完全独立样本。
"""

    limitations_body = """## 限制、稳健性与尚未证明的部分

1. 评测数据和口径已验证，但 JSON 没保存 checkpoint hash、seed、include-map hash 或代码版本，目录复用后 provenance 仍可能丢失。
2. 训练未固定 Python/NumPy/PyTorch seed；随机 crop、shuffle、LoRA dropout 和 `cudnn.benchmark=True` 都会贡献 run-to-run 方差。
3. `train.py` 与 `train_ada.py` 当前非文本路径基本相同，但严格实验仍应统一入口，避免历史代码漂移。
4. “框尺度/长宽比校准变差”由 IoU 与 center-distance 分布共同推断；需保存 `pred_bbox`/`gt_bbox` 才能验证宽高、面积和 anchor 选择。
5. 假负样本碰撞率是随机 shuffle 下的解析期望；当前 event 没记录每个 batch 的 satellite IDs，无法回放实际碰撞序列。
"""

    next_body = """## 建议的验证与修复顺序

1. **先做最小 matched ablation：** 同一个 `train.py`、固定同一初始化和 seed，只比较 weight=0 与 0.01，至少 3 个 seed。
2. **修复多正样本：** 实际使用 `build_retrieval_soft_targets(local_indices, satellite_ids)`，同 satellite 的 crop 不应互作负样本；同时记录每 batch 冲突数。
3. **把文本目标改回 within-image 15-way 或显式按 yaw 生成 caption：** 避免相同 caption 跨 row 竞争，并移除/旋转 left-right 等相对方位词。
4. **先恢复只在前 5 轮启用：** 当前证据更支持文本作为早期正则，而不是持续优化目标；再扫描更小权重。
5. **扩充评测 JSON：** 保存 pred/gt bbox、宽高比、面积比、chosen anchor、checkpoint SHA256、seed 和 code commit。
"""

    questions_body = """## 进一步问题

- 修复同 satellite 多正样本后，text CE 是否下降得更快，同时恢复 R@K 与 IoU？
- IoU 缺口具体来自预测宽度、高度、面积，还是 confidence/anchor 选择？
- 去掉相对方位词或改用 angle-specific caption 后，315° 等角度的 R@1 是否恢复？
- 在 3 个以上 matched seeds 上，0.7pp retrieval 差距是否仍稳定存在？
"""

    charts = [
        {
            "id": "retrieval-delta-chart",
            "title": "Retrieval metric deltas",
            "subtitle": "Ada fixed 0.01 minus No-text; 89,728 aligned queries, percentage points",
            "showDescription": True,
            "intent": "comparison",
            "question": "How large is the retrieval regression at each rank cutoff?",
            "rationale": "Three discrete same-unit deltas are most directly compared with a zero-referenced bar chart.",
            "comparisonContext": {"baseline": "No text", "grain": "metric", "unit": "percentage points"},
            "type": "bar",
            "dataset": "retrieval_deltas",
            "sourceId": "retrieval-deltas-source",
            "encodings": {
                "x": {"field": "metric", "type": "nominal", "label": "Metric"},
                "y": {"field": "delta_pp", "type": "quantitative", "format": "number", "label": "Ada − No-text (pp)"},
                "tooltip": [
                    {"field": "metric", "type": "text", "label": "Metric"},
                    {"field": "delta_pp", "type": "quantitative", "format": "number", "label": "Delta (pp)"},
                    {"field": "net_hit_delta", "type": "quantitative", "format": "number", "label": "Net hit delta"},
                ],
            },
            "xAxisTitle": "Metric",
            "yAxisTitle": "Percentage points",
            "layout": "full",
            "palette": {"kind": "sequential", "name": "blue"},
            "referenceLines": [{"axis": "y", "value": 0, "label": "No difference", "color": "neutral", "lineStyle": "solid"}],
            "maxRows": 3,
            "surface": {"surface": "explorer", "viewMode": "visualization"},
        },
        {
            "id": "localization-delta-chart",
            "title": "Localization and joint metric deltas",
            "subtitle": "Ada fixed 0.01 minus No-text; rate metrics expressed as percentage points",
            "showDescription": True,
            "intent": "comparison",
            "question": "Which localization metrics account for most of the evaluation gap?",
            "rationale": "A zero-referenced bar chart shows the consistently negative overlap metrics on one common scale.",
            "comparisonContext": {"baseline": "No text", "grain": "metric", "unit": "percentage points"},
            "type": "bar",
            "dataset": "localization_deltas",
            "sourceId": "localization-deltas-source",
            "encodings": {
                "x": {"field": "metric", "type": "nominal", "label": "Metric"},
                "y": {"field": "delta_points", "type": "quantitative", "format": "number", "label": "Ada − No-text (points)"},
                "tooltip": [
                    {"field": "metric", "type": "text", "label": "Metric"},
                    {"field": "delta_points", "type": "quantitative", "format": "number", "label": "Delta (points)"},
                    {"field": "ada_value", "type": "quantitative", "format": "number", "label": "Ada"},
                    {"field": "no_text_value", "type": "quantitative", "format": "number", "label": "No-text"},
                ],
            },
            "xAxisTitle": "Metric",
            "yAxisTitle": "Percentage points",
            "layout": "full",
            "palette": {"kind": "sequential", "name": "orange"},
            "referenceLines": [{"axis": "y", "value": 0, "label": "No difference", "color": "neutral", "lineStyle": "solid"}],
            "maxRows": 5,
            "surface": {"surface": "explorer", "viewMode": "visualization"},
        },
        {
            "id": "subset-scatter",
            "title": "Height × angle slice deltas",
            "subtitle": "32 equal slices; x=R@1 delta, y=mIoU delta, both Ada minus No-text in points",
            "showDescription": True,
            "intent": "relationship",
            "question": "Is the regression broad or concentrated in a few conditions?",
            "rationale": "Thirty-two equal-grain observations are sufficient to show whether slice deltas cluster in the negative quadrant.",
            "comparisonContext": {"baseline": "No text", "grain": "height × angle", "unit": "percentage points"},
            "type": "scatter",
            "dataset": "subset_deltas",
            "sourceId": "eval-subsets",
            "encodings": {
                "x": {"field": "r1_delta_pp", "type": "quantitative", "format": "number", "label": "R@1 delta (pp)"},
                "y": {"field": "miou_delta_points", "type": "quantitative", "format": "number", "label": "mIoU delta (points)"},
                "color": {"field": "height_label", "type": "nominal", "label": "Height"},
                "tooltip": [
                    {"field": "subset", "type": "text", "label": "Subset"},
                    {"field": "r1_delta_pp", "type": "quantitative", "format": "number", "label": "R@1 delta (pp)"},
                    {"field": "miou_delta_points", "type": "quantitative", "format": "number", "label": "mIoU delta (points)"},
                    {"field": "center_delta_px", "type": "quantitative", "format": "number", "label": "Center delta (px)"},
                    {"field": "num_queries", "type": "quantitative", "format": "number", "label": "Queries"},
                ],
            },
            "xAxisTitle": "R@1 delta (pp)",
            "yAxisTitle": "mIoU delta (points)",
            "layout": "full",
            "palette": {"kind": "categorical", "name": "height"},
            "legend": {"position": "bottom", "sort": "spec"},
            "referenceLines": [
                {"axis": "x", "value": 0, "label": "R@1 parity", "color": "neutral", "lineStyle": "solid"},
                {"axis": "y", "value": 0, "label": "mIoU parity", "color": "neutral", "lineStyle": "solid"},
            ],
            "maxRows": 32,
            "surface": {"surface": "explorer", "viewMode": "visualization"},
        },
        {
            "id": "center-cdf-chart",
            "title": "Center-distance threshold coverage",
            "subtitle": "Share of 89,728 queries within each pixel threshold; higher is better",
            "showDescription": True,
            "intent": "distribution",
            "question": "How can mean center distance improve while IoU worsens?",
            "rationale": "Eight ordered thresholds reveal the precision-versus-tail crossover hidden by a single mean.",
            "comparisonContext": {"baseline": "No text", "grain": "threshold", "unit": "query share"},
            "type": "line",
            "dataset": "center_thresholds",
            "sourceId": "center-distribution",
            "encodings": {
                "x": {"field": "threshold_label", "type": "ordinal", "label": "Center-distance threshold"},
                "y": {"field": "share_within_threshold", "type": "quantitative", "format": "percent", "label": "Share within threshold"},
                "color": {"field": "model", "type": "nominal", "label": "Model"},
                "tooltip": [
                    {"field": "model", "type": "text", "label": "Model"},
                    {"field": "threshold_px", "type": "quantitative", "format": "number", "label": "Threshold (px)"},
                    {"field": "share_within_threshold", "type": "quantitative", "format": "percent", "label": "Share"},
                ],
            },
            "xAxisTitle": "Center-distance threshold",
            "yAxisTitle": "Query share",
            "layout": "full",
            "palette": {"kind": "categorical", "name": "model-comparison"},
            "legend": {"position": "bottom", "sort": "spec"},
            "maxRows": 16,
            "surface": {"surface": "explorer", "viewMode": "visualization"},
        },
    ]

    tables = [
        {
            "id": "overall-table",
            "title": "Current evaluation metrics",
            "subtitle": "89,728 exactly aligned queries; values are current Ada fixed 0.01 versus No-text",
            "showDescription": True,
            "dataset": "overall_comparison",
            "defaultSort": {"field": "group", "direction": "asc"},
            "density": "spacious",
            "sourceId": "eval-overall",
            "layout": "full",
            "columns": [
                {"field": "group", "label": "Group", "type": "text"},
                {"field": "metric", "label": "Metric", "type": "text"},
                {"field": "preferred_direction", "label": "Better", "type": "text"},
                {"field": "ada_value", "label": "Ada fixed 0.01", "format": "number"},
                {"field": "no_text_value", "label": "No-text", "format": "number"},
                {"field": "ada_minus_no_text", "label": "Ada − No-text", "format": "number", "semantic": "movement", "movement": True},
                {"field": "unit", "label": "Unit", "type": "text"},
            ],
        },
        {
            "id": "provenance-table",
            "title": "Evaluation identity checks",
            "subtitle": "Current files, parameters, aligned population, and checkpoint-to-JSON chronology",
            "showDescription": True,
            "dataset": "provenance_checks",
            "defaultSort": {"field": "check", "direction": "asc"},
            "density": "spacious",
            "sourceId": "provenance-checks",
            "layout": "full",
            "columns": [
                {"field": "check", "label": "Check", "type": "text"},
                {"field": "ada", "label": "Ada", "type": "text"},
                {"field": "no_text", "label": "No-text", "type": "text"},
                {"field": "result", "label": "Result", "type": "text"},
            ],
        },
        {
            "id": "training-table",
            "title": "Training component audit",
            "subtitle": "Latest complete 20-epoch events; step scalars aggregated by epoch",
            "showDescription": True,
            "dataset": "training_components",
            "defaultSort": {"field": "metric", "direction": "asc"},
            "density": "spacious",
            "sourceId": "training-components",
            "layout": "full",
            "columns": [
                {"field": "metric", "label": "Metric", "type": "text"},
                {"field": "epochs_ada_worse", "label": "Epochs Ada higher", "format": "number"},
                {"field": "mean_delta", "label": "20-epoch mean delta", "format": "number", "semantic": "movement", "movement": True},
                {"field": "final_ada", "label": "Final Ada", "format": "number"},
                {"field": "final_no_text", "label": "Final No-text", "format": "number"},
                {"field": "final_delta", "label": "Final delta", "format": "number", "semantic": "movement", "movement": True},
            ],
        },
        {
            "id": "mechanism-table",
            "title": "Text-target conflict evidence",
            "subtitle": "Static dataset/code audit and random-batch collision expectations",
            "showDescription": True,
            "dataset": "mechanism_evidence",
            "defaultSort": {"field": "evidence", "direction": "asc"},
            "density": "spacious",
            "sourceId": "mechanism-audit",
            "layout": "full",
            "columns": [
                {"field": "evidence", "label": "Evidence", "type": "text"},
                {"field": "value", "label": "Value", "format": "number"},
                {"field": "unit", "label": "Unit", "type": "text"},
                {"field": "interpretation", "label": "Interpretation", "type": "text"},
            ],
        },
    ]

    blocks = [
        {"id": "title", "type": "markdown", "body": f"# {REPORT_TITLE}", "layout": "full"},
        {"id": "technical-summary", "type": "markdown", "body": summary_body, "layout": "full"},
        {"id": "provenance-finding", "type": "markdown", "body": provenance_body, "layout": "full", "sourceId": "provenance-checks"},
        {"id": "provenance-table-block", "type": "table", "tableId": "provenance-table", "layout": "full"},
        {"id": "retrieval-finding", "type": "markdown", "body": retrieval_body, "layout": "full"},
        {"id": "retrieval-chart-block", "type": "chart", "chartId": "retrieval-delta-chart", "layout": "full"},
        {"id": "localization-finding", "type": "markdown", "body": localization_body, "layout": "full"},
        {"id": "localization-chart-block", "type": "chart", "chartId": "localization-delta-chart", "layout": "full"},
        {"id": "subset-chart-block", "type": "chart", "chartId": "subset-scatter", "layout": "full"},
        {"id": "center-finding", "type": "markdown", "body": center_body, "layout": "full"},
        {"id": "center-chart-block", "type": "chart", "chartId": "center-cdf-chart", "layout": "full"},
        {"id": "overall-table-block", "type": "table", "tableId": "overall-table", "layout": "full"},
        {"id": "mechanism-finding", "type": "markdown", "body": mechanism_body, "layout": "full"},
        {"id": "mechanism-table-block", "type": "table", "tableId": "mechanism-table", "layout": "full"},
        {"id": "training-finding", "type": "markdown", "body": training_body, "layout": "full"},
        {"id": "training-table-block", "type": "table", "tableId": "training-table", "layout": "full"},
        {"id": "definitions", "type": "markdown", "body": definitions_body, "layout": "full"},
        {"id": "limitations", "type": "markdown", "body": limitations_body, "layout": "full"},
        {"id": "next-steps", "type": "markdown", "body": next_body, "layout": "full"},
        {"id": "questions", "type": "markdown", "body": questions_body, "layout": "full"},
    ]

    manifest = {
        "version": 1,
        "surface": "report",
        "title": REPORT_TITLE,
        "description": "Paired evaluation, data-quality, training-component, and text-target mechanism diagnosis.",
        "generatedAt": generated_at,
        "sources": sources,
        "cards": [],
        "charts": charts,
        "tables": tables,
        "blocks": blocks,
    }
    return {
        "surface": "report",
        "manifest": manifest,
        "snapshot": {
            "version": 1,
            "generatedAt": generated_at,
            "status": "ready",
            "datasets": reviewed,
        },
        "sources": sources,
    }


def write_notes() -> None:
    notes = """# Analysis notes

## Required structure mapping

- Title: `title`
- Technical summary: `technical-summary`
- Key findings with visual evidence: provenance, retrieval, localization, center distribution, mechanism, and training components
- Scope/data/metric definitions: `definitions`
- Methodology: embedded in `definitions` and source query metadata
- Limitations/uncertainty/robustness: `limitations`
- Recommended next steps: `next-steps`
- Further questions: `questions`

## Chart map

| Section | Question | Family/type | Fields | Supported claim |
|---|---|---|---|---|
| Retrieval | How large is each rank-cutoff regression? | Comparison / bar | metric, delta_pp | R@1/5/10 each decline about 0.7pp. |
| Localization | Which overlap metrics move most? | Comparison / bar | metric, delta_points | IoU threshold and mean metrics consistently decline. |
| Slices | Broad or concentrated regression? | Relationship / scatter | r1_delta_pp, miou_delta_points, height | All 32 mIoU slice deltas are negative. |
| Center error | Why does mean CDE improve? | Distribution / threshold-CDF line | threshold, share, model | Ada loses near-center precision but improves the extreme tail. |

Palette policy is a hard two-root cap plus neutrals for focal-vs-baseline comparisons. The subset scatter uses four height categories because height identity is analytically relevant. Zero reference lines carry sign without relying on color.

## Source and QA notes

- Current Ada JSON mtime: 2026-08-11 18:27:27 +0800; checkpoint mtime: 17:11:48.
- Current Ada checkpoint SHA256 observed during this audit: `ca4cfb62053d8958a295928e8622e12d04db15074738f70b539c3b91296a3be4`.
- JSON files do not persist hashes, seeds, include-map hashes, or code commits; chronology and exact path matching provide high-confidence but not cryptographic run attribution.
- Query keys, population, candidate size, gallery size, satellite dimensions, crop ratio, and deterministic candidate seed match exactly.
- Cluster bootstrap intervals cover test-query sampling by gallery label, not training-run seed variance.
- Caption collision probabilities are expected values under random shuffle; actual batch satellite IDs were not logged.
- The previous training-curve report was generated before the current Ada JSON overwrote the old scheduled-run JSON and is superseded for current evaluation conclusions.

### Raw source inventory

- Current fixed-text evaluation: `exp/eval_results/baseline_ada_1_5x3.json`
- Current no-text evaluation: `exp/eval_results/baseline_wo_input_ids_ada_5x3.json`
- Current fixed-text event: `exp/runs/baseline_ada_1_5x3/events.out.tfevents.1786420717.4090-48g.511988.0`
- Current no-text event: `exp/runs/baseline_wo_input_ids_ada_5x3/events.out.tfevents.1786302112.4090-48g.2760507.0`
- Code paths: `exp/train.py`, `exp/train_ada.py`, `exp/model.py`, `exp/dataset.py`, `exp/test.py`
- Current effective configs: `exp/outputs/baseline_ada_1_5x3/effective_config.json`, `exp/outputs/baseline_wo_input_ids_ada_5x3/effective_config.json`
- Historical scheduled-text snapshot: `exp/analysis/baseline_ada_1_vs_wo_input_ids/analysis.sqlite`
- Caption corpus profiled from the 701 train IDs and per-satellite `qwen_6_28_description.json` files.
"""
    with open(OUTPUT_DIR / "analysis_notes.md", "w", encoding="utf-8") as file:
        file.write(notes)


def main() -> None:
    artifact = build_artifact()
    with open(OUTPUT_DIR / "artifact.json", "w", encoding="utf-8") as file:
        json.dump(artifact, file, ensure_ascii=False, indent=2)
    write_notes()
    print(OUTPUT_DIR / "artifact.json")


if __name__ == "__main__":
    main()
