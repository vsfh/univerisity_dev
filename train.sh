#!/bin/bash
set -euo pipefail

# Train unified_siglip_supp.py with accelerate.
# Usage:
#   bash train.sh configs/unified_siglip_supp/0.5_heat_text.yaml

CONFIG_PATH="${1:-}"
if [ -z "$CONFIG_PATH" ]; then
    echo "Usage: bash train.sh <config.yaml> [extra unified_siglip_supp.py args...]"
    exit 1
fi
shift

cd /media/data1/feihong/univerisity_dev

export CUDA_VISIBLE_DEVICES=0,1

accelerate launch \
    --multi_gpu \
    --num_processes 2 \
    --mixed_precision fp16 \
    --gradient_accumulation_steps 1 \
    unified_siglip_supp.py \
    --config "$CONFIG_PATH" \
    "$@"
