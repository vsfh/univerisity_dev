#!/bin/bash
set -uo pipefail

cd /media/data1/feihong/univerisity_dev

DRY_RUN=0
GPUS_CSV="${CUDA_VISIBLE_DEVICES:-0}"
TRAIN_EXTRA_ARGS=()

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
            TRAIN_EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

IFS=',' read -r -a GPUS <<< "$GPUS_CSV"
if [ "${#GPUS[@]}" -eq 0 ] || [ -z "${GPUS[0]}" ]; then
    echo "No GPUs provided. Use --gpus 0,1,2 or set CUDA_VISIBLE_DEVICES." >&2
    exit 2
fi
NUM_GPUS="${#GPUS[@]}"
FIRST_GPU="${GPUS[0]}"

CONFIGS=(
    # "configs/grounding/smgeo.yaml"
    "configs/grounding/trogeolite.yaml"
    # "configs/grounding/det.yaml"
    # "configs/grounding/sample4geo.yaml"
    # "configs/grounding/lpn.yaml"
    # "configs/grounding/ocg.yaml"
)

SUMMARY_DIR="eval_results/grounding"
SUMMARY_PATH="${SUMMARY_DIR}/group_summary.jsonl"
FINAL_JSON="${SUMMARY_DIR}/group_summary.json"
mkdir -p "$SUMMARY_DIR"
: > "$SUMMARY_PATH"

echo "Running ${#CONFIGS[@]} grounding configs sequentially on ${NUM_GPUS} GPU(s): ${GPUS_CSV}"

for CONFIG_INDEX in "${!CONFIGS[@]}"; do
    CONFIG_PATH="${CONFIGS[$CONFIG_INDEX]}"
    echo "============================================================"
    echo "Running grounding config on ${NUM_GPUS} GPU(s): ${CONFIG_PATH}"
    echo "Visible physical GPUs: ${GPUS_CSV}"
    echo "Started at: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "============================================================"

    TRAIN_STATUS="ok"
    EVAL_STATUS="ok"
    MODEL_TYPE="$(basename "$CONFIG_PATH" .yaml)"

    # if [ "$DRY_RUN" -eq 1 ]; then
    #     echo "[dry-run] train: ${CONFIG_PATH}"
    #     echo "[dry-run] test_unify_ground: ${MODEL_TYPE}"
    #     printf '{"config_index":%s,"config":"%s","model_type":"%s","gpus":"%s","num_gpus":%s,"train_status":"dry_run","eval_status":"dry_run"}\n' \
    #         "$CONFIG_INDEX" "$CONFIG_PATH" "$MODEL_TYPE" "$GPUS_CSV" "$NUM_GPUS" >> "$SUMMARY_PATH"
    #     continue
    # fi

    # if [ "$NUM_GPUS" -eq 1 ]; then
    #     CUDA_VISIBLE_DEVICES="$FIRST_GPU" python grounding/train.py \
    #         --config "$CONFIG_PATH" \
    #         --device cuda:0 \
    #         "${TRAIN_EXTRA_ARGS[@]}"
    # else
    #     MASTER_PORT=$((29500 + CONFIG_INDEX))
    #     CUDA_VISIBLE_DEVICES="$GPUS_CSV" python -m torch.distributed.run \
    #         --master_addr 127.0.0.1 \
    #         --master_port "$MASTER_PORT" \
    #         --nproc_per_node="$NUM_GPUS" \
    #         grounding/train.py \
    #         --config "$CONFIG_PATH" \
    #         --device cuda:0 \
    #         "${TRAIN_EXTRA_ARGS[@]}"
    # fi
    # if [ "$?" -ne 0 ]; then
    #     TRAIN_STATUS="failed"
    #     EVAL_STATUS="skipped"
    #     printf '{"config_index":%s,"config":"%s","gpus":"%s","num_gpus":%s,"train_status":"%s","eval_status":"%s"}\n' \
    #         "$CONFIG_INDEX" "$CONFIG_PATH" "$GPUS_CSV" "$NUM_GPUS" "$TRAIN_STATUS" "$EVAL_STATUS" >> "$SUMMARY_PATH"
    #     echo "Training failed; skip eval for this config: ${CONFIG_PATH}"
    #     continue
    # fi

    CUDA_VISIBLE_DEVICES="$FIRST_GPU" python test_unify_ground.py \
        --model-types "$MODEL_TYPE" \
        --device cuda:0 \
        --output-dir "$SUMMARY_DIR" \
        --output-suffix "$MODEL_TYPE"
    if [ "$?" -ne 0 ]; then
        EVAL_STATUS="failed"
    fi

    printf '{"config_index":%s,"config":"%s","gpus":"%s","num_gpus":%s,"train_status":"%s","eval_status":"%s"}\n' \
        "$CONFIG_INDEX" "$CONFIG_PATH" "$GPUS_CSV" "$NUM_GPUS" "$TRAIN_STATUS" "$EVAL_STATUS" >> "$SUMMARY_PATH"

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
items = []
for line in summary_path.read_text(encoding="utf-8").splitlines():
    if not line.strip():
        continue

    item = json.loads(line)
    cfg = load_config(item["config"])
    model_type = str(cfg["model"]["type"])
    metrics_path = Path("eval_results/grounding") / f"test_unify_{model_type}_{model_type}.json"
    metrics = {}
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

    overall = metrics.get("overall", {})
    item["checkpoint"] = metrics.get("checkpoint")
    item["mean_iou"] = overall.get("mean_iou")
    item["iou_at_0_5"] = overall.get("ratio_iou_gt_0_5")
    item["iou_at_0_25"] = overall.get("ratio_iou_gt_0_25")
    item["mean_center_distance"] = overall.get("mean_center_distance")
    item["recall_at_1"] = overall.get("recall@1")
    item["unified_iou"] = overall.get("uIoU")
    items.append(item)

items.sort(key=lambda item: int(item.get("config_index", 0)))
final_json.write_text(json.dumps(items, indent=2, sort_keys=True), encoding="utf-8")
print(f"Wrote {final_json}")
PY
