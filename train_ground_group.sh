#!/bin/bash
set -uo pipefail

cd /media/data1/feihong/univerisity_dev

DRY_RUN=0
EXTRA_ARGS=()
for arg in "$@"; do
    if [ "$arg" = "--dry-run" ]; then
        DRY_RUN=1
    else
        EXTRA_ARGS+=("$arg")
    fi
done

CONFIGS=(
    "configs/grounding/siglip2_heat.yaml"
    "configs/grounding/siglip2_test.yaml"
    "configs/grounding/siglip_ground.yaml"
    "configs/grounding/lpn.yaml"
    "configs/grounding/sample4geo.yaml"
    "configs/grounding/smgeo.yaml"
    "configs/grounding/ocg.yaml"
    "configs/grounding/trogeolite.yaml"
    "configs/grounding/det.yaml"
)

SUMMARY_DIR="eval_results/grounding"
SUMMARY_PATH="${SUMMARY_DIR}/group_summary.jsonl"
FINAL_JSON="${SUMMARY_DIR}/group_summary.json"
mkdir -p "$SUMMARY_DIR"
: > "$SUMMARY_PATH"

for CONFIG_PATH in "${CONFIGS[@]}"; do
    echo "============================================================"
    echo "Running grounding config: ${CONFIG_PATH}"
    echo "Started at: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "============================================================"

    TRAIN_STATUS="ok"
    EVAL_STATUS="ok"
    TRAIN_ARGS=("${EXTRA_ARGS[@]}")
    EVAL_ARGS=("${EXTRA_ARGS[@]}")
    if [ "$DRY_RUN" -eq 1 ]; then
        TRAIN_ARGS+=("--dry-run")
        EVAL_ARGS+=("--dry-run")
    fi

    python grounding/train.py --config "$CONFIG_PATH" "${TRAIN_ARGS[@]}"
    if [ "$?" -ne 0 ]; then
        TRAIN_STATUS="failed"
    fi

    python grounding/eval.py --config "$CONFIG_PATH" "${EVAL_ARGS[@]}"
    if [ "$?" -ne 0 ]; then
        EVAL_STATUS="failed"
    fi

    printf '{"config":"%s","train_status":"%s","eval_status":"%s"}\n' \
        "$CONFIG_PATH" "$TRAIN_STATUS" "$EVAL_STATUS" >> "$SUMMARY_PATH"

    if [ "$TRAIN_STATUS" = "failed" ] || [ "$EVAL_STATUS" = "failed" ]; then
        echo "Config failed; continue to next config: ${CONFIG_PATH}"
        continue
    fi
done

python - "$SUMMARY_PATH" "$FINAL_JSON" <<'PY'
import json
import sys
from pathlib import Path

from grounding.config import load_config

summary_path = Path(sys.argv[1])
final_json = Path(sys.argv[2])
metric_keys = [
    "checkpoint",
    "mean_iou",
    "iou_at_0_5",
    "iou_at_0_25",
    "mean_center_distance",
]

items = []
for line in summary_path.read_text(encoding="utf-8").splitlines():
    if not line.strip():
        continue

    item = json.loads(line)
    cfg = load_config(item["config"])
    metrics_path = Path(cfg["eval"]["output_dir"]) / "metrics.json"
    metrics = {}
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

    for key in metric_keys:
        item[key] = metrics.get(key)
    items.append(item)

final_json.write_text(json.dumps(items, indent=2, sort_keys=True), encoding="utf-8")
print(f"Wrote {final_json}")
PY
