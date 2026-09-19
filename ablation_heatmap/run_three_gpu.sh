#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2}"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
exec "${PYTHON:-python}" ablation_heatmap/three_gpu.py "$@"
