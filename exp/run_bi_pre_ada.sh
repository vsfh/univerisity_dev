#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

PRETRAINED_CHECKPOINT="outputs/baseline_grounding_full_ada_5x3/last.pth"
if [[ ! -f "${PRETRAINED_CHECKPOINT}" ]]; then
    echo "Required ada checkpoint not found: ${PRETRAINED_CHECKPOINT}" >&2
    echo "Train configs_ada/baseline_grounding_full_ada.yaml first." >&2
    exit 1
fi

CONFIG_PATHS=(
    "configs_ada/baseline_pre_ada_2.yaml"
    "configs_ada/baseline_pre_ada.yaml"

)

for CONFIG_PATH in "${CONFIG_PATHS[@]}"; do
    CONFIG_NAME="${CONFIG_PATH##*/}"
    EXP_NAME="${CONFIG_NAME%.yaml}_5x3"
    CHECKPOINT_PATH="outputs/${EXP_NAME}/last.pth"

    echo "============================================================"
    echo "Training ${EXP_NAME}"
    echo "Config: ${CONFIG_PATH}"
    echo "Started at: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "============================================================"

    accelerate launch \
        --mixed_precision fp16 \
        --gradient_accumulation_steps 1 \
        train_ada.py \
        --config "${CONFIG_PATH}"

    echo "Testing ${EXP_NAME}"
    python test.py \
        --config "${CONFIG_PATH}" \
        --checkpoint "${CHECKPOINT_PATH}" \
        --output-dir eval_results \
        --batch-size "${TEST_BATCH_SIZE:-8}" \
        --num-workers "${TEST_NUM_WORKERS:-8}"
done

echo "============================================================"
echo "Finished ada bi/pre training and evaluation"
echo "Finished at: $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"
