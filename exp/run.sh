#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

accelerate launch \
    --mixed_precision fp16 \
    --gradient_accumulation_steps 1 \
    train.py \
    --config configs/baseline_sat.yaml

python test.py \
    --config configs/baseline_sat.yaml \
    --checkpoint outputs/baseline_sat_5x3/last.pth \
    --output-dir eval_results \
    --batch-size "${TEST_BATCH_SIZE:-8}" \
    --num-workers "${TEST_NUM_WORKERS:-8}"
