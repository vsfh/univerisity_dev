#!/bin/bash
set -euo pipefail

cd /media/data1/feihong/univerisity_dev

CONFIG_PATH="${1:-configs/unified_siglip_supp/single_config/baseline_sat.yaml}"
shift || true

CHECKPOINT_NAME="${CHECKPOINT_NAME:-last.pth}"
OUTPUT_DIR="${OUTPUT_DIR:-/media/data1/feihong/univerisity_dev/eval_results/test_unify}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-8}"

read -r EXP_NAME SAVE_ROOT USE_ANGLE USE_HEATMAP CONFIG_MODEL_TYPE < <(
python - "$CONFIG_PATH" <<'PY'
import sys
import yaml

with open(sys.argv[1], "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f) or {}
config = cfg.get("config", {}) or {}
encoder_type = str(config.get("ENCODER_TYPE", "heat")).lower()
model_type = {
    "heat": "encoder_heat",
    "test": "encoder_test",
    "model_pre": "model_pre",
    "model_bi": "model_bi",
}.get(encoder_type)
if model_type is None:
    raise SystemExit(f"Unsupported ENCODER_TYPE: {encoder_type}")
print(
    cfg.get("exp_name"),
    cfg.get("save_root", "/media/data1/feihong/ckpt"),
    int(bool(config.get("USE_ANGLE_INPUT", True))),
    int(bool(config.get("USE_HEATMAP_LOSS", True))),
    model_type,
)
PY
)

if [ -z "${EXP_NAME}" ] || [ "${EXP_NAME}" = "None" ]; then
    echo "Cannot find exp_name in ${CONFIG_PATH}" >&2
    exit 1
fi

MODEL_TYPE="${MODEL_TYPE:-${CONFIG_MODEL_TYPE}}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-${SAVE_ROOT}/${EXP_NAME}/${CHECKPOINT_NAME}}"
ANGLE_FLAG="--no-encoder-heat-use-angle"
HEATMAP_FLAG="--no-encoder-heat-use-heatmap"
if [ "${USE_ANGLE}" = "1" ]; then
    ANGLE_FLAG="--encoder-heat-use-angle"
fi
if [ "${USE_HEATMAP}" = "1" ]; then
    HEATMAP_FLAG="--encoder-heat-use-heatmap"
fi

COMMON_ARGS=(
    --model-types "${MODEL_TYPE}"
    --checkpoint "${CHECKPOINT_PATH}"
    --output-dir "${OUTPUT_DIR}"
    --batch-size "${BATCH_SIZE}"
    --num-workers "${NUM_WORKERS}"
    --lora-rank 8
    --lora-alpha 16.0
    --lora-dropout 0.05
    --no-encoder-heat-use-text
    "${ANGLE_FLAG}"
    "${HEATMAP_FLAG}"
)

CANDIDATE_SIZES=(200 400 600)
TEST_CROP_RATIOS=(0.8 0.6 0.4)

echo "============================================================"
echo "Config: ${CONFIG_PATH}"
echo "Checkpoint: ${CHECKPOINT_PATH}"
echo "Started at: $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"

# for candidate_size in "${CANDIDATE_SIZES[@]}"; do
#     echo "Testing candidate_size=${candidate_size}, test_crop_ratio=1.0"
#     python test_unify.py \
#         "${COMMON_ARGS[@]}" \
#         --candidate-size "${candidate_size}" \
#         --test-crop-ratio 1.0 \
#         --output-suffix "${EXP_NAME}_candidate_sweep_candidate_${candidate_size}_crop_1.0" \
#         "$@"
# done

for test_crop_ratio in "${TEST_CROP_RATIOS[@]}"; do
    echo "Testing candidate_size=100, test_crop_ratio=${test_crop_ratio}"
    python test_unify.py \
        "${COMMON_ARGS[@]}" \
        --candidate-size 100 \
        --test-crop-ratio "${test_crop_ratio}" \
        --output-suffix "${EXP_NAME}_crop_sweep_candidate_100_crop_${test_crop_ratio}" \
        "$@"
done

echo "============================================================"
echo "Finished test_group sweep"
echo "Finished at: $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"
