#!/bin/bash
set -euo pipefail

cd /media/data1/feihong/univerisity_dev

CONFIG_PATH="${1:-configs/grounding/siglip2_heat.yaml}"

python grounding/train.py --config "$CONFIG_PATH"

python grounding/eval.py --config "$CONFIG_PATH"
