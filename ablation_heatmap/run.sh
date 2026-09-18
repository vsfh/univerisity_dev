#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
exec "${PYTHON:-/home/feihong/miniconda3/bin/python}" ablation_heatmap/study.py "$@"
