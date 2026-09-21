#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
if [[ -z "${PYTHON:-}" ]]; then
  PYTHON="$(command -v python || true)"
  if [[ -z "$PYTHON" ]]; then
    for candidate in /home/feihong/miniconda3/bin/python /data/feihong/miniconda3/bin/python; do
      if [[ -x "$candidate" ]]; then PYTHON="$candidate"; break; fi
    done
  fi
fi
if [[ -z "$PYTHON" ]]; then echo "Activate the training environment or set PYTHON." >&2; exit 1; fi
exec "$PYTHON" ablation_heatmap/study1.py "$@"
