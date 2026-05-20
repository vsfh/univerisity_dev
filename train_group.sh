#!/bin/bash
set -euo pipefail

cd /media/data1/feihong/univerisity_dev

CONFIGS=(
    # Encoder_heat without geo input.
    # "configs/unified_siglip_supp/single_config/model_heat_no_geo.yaml"
    # Encoder_heat without input_ids.
    # "configs/unified_siglip_supp/single_config/model_heat_no_input_ids.yaml"
    # Encoder_test with text-guided height-aware refinement.
    "configs/unified_siglip_supp/single_config/model_test_geo_input_ids.yaml"
)

for CONFIG_PATH in "${CONFIGS[@]}"; do
    echo "============================================================"
    echo "Running unified_siglip_supp experiment: ${CONFIG_PATH}"
    echo "Started at: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "============================================================"

    bash train.sh "${CONFIG_PATH}"

    echo "============================================================"
    echo "Finished experiment: ${CONFIG_PATH}"
    echo "Finished at: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "============================================================"
done
