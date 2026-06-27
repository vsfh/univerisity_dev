#!/bin/bash
set -euo pipefail

cd /media/data1/feihong/remote/univerisity_dev

accelerate launch --multi_gpu --num_processes 2 text.py

python test_text.py \
  --checkpoint /media/data1/feihong/remote/ckpt/text_qwen3vl/last.pth