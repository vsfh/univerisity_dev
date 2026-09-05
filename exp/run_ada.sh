#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

SAVE_ROOT="/media/data2/feihong/ckpt"

for CONFIG_PATH in configs/*.yaml; do
    CONFIG_NAME="${CONFIG_PATH##*/}"
    EXP_NAME="${CONFIG_NAME%.yaml}_5x3"

    # accelerate launch \
    #     --mixed_precision fp16 \
    #     --gradient_accumulation_steps 1 \
    #     train_ada.py \
    #     --config "${CONFIG_PATH}"

    python test.py \
        --config "${CONFIG_PATH}" \
        --checkpoint "outputs/${EXP_NAME}/last.pth" \
        --output-dir eval_results \
        --batch-size "${TEST_BATCH_SIZE:-8}" \
        --num-workers "${TEST_NUM_WORKERS:-8}"
done
