#!/bin/bash
set -euo pipefail

cd /media/data1/feihong/univerisity_dev

accelerate launch --multi_gpu --num_processes 2 text.py

python test_text.py \
  --checkpoint /media/data1/feihong/ckpt/text_qwen3vl/last.pth