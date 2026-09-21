#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if (( $# )); then
    echo "Usage: bash run_clip_siglip2.sh (no arguments)" >&2
    exit 2
fi
export PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
if [[ -n "${PYTHON:-}" ]]; then
    python_bin="$PYTHON"
elif command -v python >/dev/null 2>&1; then
    python_bin="$(command -v python)"
elif [[ -x /home/feihong/miniconda3/bin/python ]]; then
    python_bin=/home/feihong/miniconda3/bin/python
else
    echo "Set PYTHON to the Python executable for your training environment." >&2
    exit 1
fi
exec "$python_bin" run_clip_siglip2.py
