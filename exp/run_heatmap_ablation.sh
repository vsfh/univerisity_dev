#!/usr/bin/env bash
set -euo pipefail
exec bash "$(dirname "$0")/../ablation_heatmap/run.sh" "$@"
