#!/bin/bash
set -euo pipefail

cd /media/data1/feihong/univerisity_dev

python grounding/train_siglip.py

python grounding/eval_ground.py