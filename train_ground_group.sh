#!/bin/bash
set -uo pipefail

cd /media/data1/feihong/univerisity_dev

DRY_RUN=0
GPUS_CSV="${CUDA_VISIBLE_DEVICES:-0}"
EXTRA_ARGS=()

while [ "$#" -gt 0 ]; do
    case "$1" in
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        --gpus)
            if [ "$#" -lt 2 ]; then
                echo "Missing value for --gpus, e.g. --gpus 0,1,2,3" >&2
                exit 2
            fi
            GPUS_CSV="$2"
            shift 2
            ;;
        --gpus=*)
            GPUS_CSV="${1#--gpus=}"
            shift
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

IFS=',' read -r -a GPUS <<< "$GPUS_CSV"
if [ "${#GPUS[@]}" -eq 0 ] || [ -z "${GPUS[0]}" ]; then
    echo "No GPUs provided. Use --gpus 0,1 or set CUDA_VISIBLE_DEVICES." >&2
    exit 2
fi
NUM_GPUS="${#GPUS[@]}"

CONFIGS=(
    "configs/grounding/smgeo.yaml"
    "configs/grounding/ocg.yaml"
    "configs/grounding/trogeolite.yaml"
    "configs/grounding/det.yaml"
    "configs/grounding/siglip_ground.yaml"
    "configs/grounding/siglip2_heat.yaml"
    "configs/grounding/siglip2_test.yaml"
    "configs/grounding/sample4geo.yaml"
    "configs/grounding/lpn.yaml"
)

SUMMARY_DIR="eval_results/grounding"
SUMMARY_PATH="${SUMMARY_DIR}/group_summary.jsonl"
FINAL_JSON="${SUMMARY_DIR}/group_summary.json"
mkdir -p "$SUMMARY_DIR"
: > "$SUMMARY_PATH"

run_worker() {
    local WORKER_ID="$1"
    local GPU_ID="$2"
    local WORKER_SUMMARY="${SUMMARY_DIR}/worker_${WORKER_ID}.jsonl"
    : > "$WORKER_SUMMARY"

    local CONFIG_INDEX
    for CONFIG_INDEX in "${!CONFIGS[@]}"; do
        if [ $((CONFIG_INDEX % NUM_GPUS)) -ne "$WORKER_ID" ]; then
            continue
        fi

        local CONFIG_PATH="${CONFIGS[$CONFIG_INDEX]}"
        echo "============================================================"
        echo "Worker ${WORKER_ID}/${NUM_GPUS} on GPU ${GPU_ID}: ${CONFIG_PATH}"
        echo "Started at: $(date '+%Y-%m-%d %H:%M:%S')"
        echo "============================================================"

        local TRAIN_STATUS="ok"
        local EVAL_STATUS="ok"
        local TRAIN_ARGS=("${EXTRA_ARGS[@]}")
        local EVAL_ARGS=("${EXTRA_ARGS[@]}")
        if [ "$DRY_RUN" -eq 1 ]; then
            TRAIN_ARGS+=("--dry-run")
            EVAL_ARGS+=("--dry-run")
        fi

        CUDA_VISIBLE_DEVICES="$GPU_ID" python grounding/train.py --config "$CONFIG_PATH" "${TRAIN_ARGS[@]}"
        if [ "$?" -ne 0 ]; then
            TRAIN_STATUS="failed"
        fi

        CUDA_VISIBLE_DEVICES="$GPU_ID" python grounding/eval.py --config "$CONFIG_PATH" "${EVAL_ARGS[@]}"
        if [ "$?" -ne 0 ]; then
            EVAL_STATUS="failed"
        fi

        printf '{"config_index":%s,"config":"%s","gpu":"%s","worker_id":%s,"train_status":"%s","eval_status":"%s"}\n' \
            "$CONFIG_INDEX" "$CONFIG_PATH" "$GPU_ID" "$WORKER_ID" "$TRAIN_STATUS" "$EVAL_STATUS" >> "$WORKER_SUMMARY"

        if [ "$TRAIN_STATUS" = "failed" ] || [ "$EVAL_STATUS" = "failed" ]; then
            echo "Config failed; worker ${WORKER_ID} continues: ${CONFIG_PATH}"
            continue
        fi
    done
}

echo "Running ${#CONFIGS[@]} grounding configs on ${NUM_GPUS} GPU(s): ${GPUS_CSV}"
PIDS=()
for WORKER_ID in "${!GPUS[@]}"; do
    GPU_ID="${GPUS[$WORKER_ID]}"
    run_worker "$WORKER_ID" "$GPU_ID" &
    PIDS+=("$!")
done

EXIT_STATUS=0
for PID in "${PIDS[@]}"; do
    wait "$PID"
    STATUS="$?"
    if [ "$STATUS" -ne 0 ]; then
        EXIT_STATUS="$STATUS"
    fi
done

for WORKER_ID in "${!GPUS[@]}"; do
    WORKER_SUMMARY="${SUMMARY_DIR}/worker_${WORKER_ID}.jsonl"
    if [ -f "$WORKER_SUMMARY" ]; then
        cat "$WORKER_SUMMARY" >> "$SUMMARY_PATH"
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

items.sort(key=lambda item: int(item.get("config_index", 0)))
final_json.write_text(json.dumps(items, indent=2, sort_keys=True), encoding="utf-8")
print(f"Wrote {final_json}")
PY

exit "$EXIT_STATUS"
