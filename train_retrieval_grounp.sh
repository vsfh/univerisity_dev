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
                echo "Missing value for --gpus, e.g. --gpus 0,1,2" >&2
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
    echo "No GPUs provided. Use --gpus 0,1,2 or set CUDA_VISIBLE_DEVICES." >&2
    exit 2
fi
FIRST_GPU="${GPUS[0]}"

CONFIGS=(
    # Skipped: last.pth generated after 2026-07-08.
    # "configs/retrieval/siglip.yaml"
    # "configs/retrieval/clip.yaml"
    "configs/retrieval/openclip.yaml"
    "configs/retrieval/evaclip.yaml"
    "configs/retrieval/sample_retrieval.yaml"
)

SUMMARY_DIR="eval_results/retrieval"
SUMMARY_PATH="${SUMMARY_DIR}/group_summary.jsonl"
FINAL_JSON="${SUMMARY_DIR}/group_summary.json"
mkdir -p "$SUMMARY_DIR"
: > "$SUMMARY_PATH"

echo "Running ${#CONFIGS[@]} retrieval configs sequentially on GPU(s): ${GPUS_CSV}"

for CONFIG_INDEX in "${!CONFIGS[@]}"; do
    CONFIG_PATH="${CONFIGS[$CONFIG_INDEX]}"
    echo "============================================================"
    echo "Running retrieval config: ${CONFIG_PATH}"
    echo "Visible physical GPUs: ${GPUS_CSV}"
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

    CUDA_VISIBLE_DEVICES="$GPUS_CSV" python retrieval/train.py \
        --config "$CONFIG_PATH" \
        --device cuda:0 \
        "${TRAIN_ARGS[@]}"
    if [ "$?" -ne 0 ]; then
        TRAIN_STATUS="failed"
        EVAL_STATUS="skipped"
        printf '{"config_index":%s,"config":"%s","gpus":"%s","train_status":"%s","eval_status":"%s"}\n' \
            "$CONFIG_INDEX" "$CONFIG_PATH" "$GPUS_CSV" "$TRAIN_STATUS" "$EVAL_STATUS" >> "$SUMMARY_PATH"
        echo "Training failed; skip eval for this config: ${CONFIG_PATH}"
        continue
    fi

    CUDA_VISIBLE_DEVICES="$FIRST_GPU" python retrieval/eval.py \
        --config "$CONFIG_PATH" \
        --device cuda:0 \
        "${EVAL_ARGS[@]}"
    if [ "$?" -ne 0 ]; then
        EVAL_STATUS="failed"
    fi

    printf '{"config_index":%s,"config":"%s","gpus":"%s","train_status":"%s","eval_status":"%s"}\n' \
        "$CONFIG_INDEX" "$CONFIG_PATH" "$GPUS_CSV" "$TRAIN_STATUS" "$EVAL_STATUS" >> "$SUMMARY_PATH"
done

python - "$SUMMARY_PATH" "$FINAL_JSON" <<'PY'
import json
import sys
from pathlib import Path

from retrieval.config import load_config

summary_path = Path(sys.argv[1])
final_json = Path(sys.argv[2])
metric_keys = ["checkpoint", "num_samples", "num_gallery", "overall", "retrieval"]

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
