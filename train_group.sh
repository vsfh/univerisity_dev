#!/bin/bash
set -euo pipefail

cd /media/data1/feihong/univerisity_dev

CONFIGS=(
    # "configs/unified_siglip_supp/model_full.yaml"
    # "configs/unified_siglip_supp/model_retrieval_only.yaml"
    # "configs/unified_siglip_supp/model_bbox_only.yaml"
    # "configs/unified_siglip_supp/model_wo_angle.yaml"
    # "configs/unified_siglip_supp/model_wo_text.yaml"
    # "configs/unified_siglip_supp/model_wo_angle_wo_text.yaml"    
    # "configs/unified_siglip_supp/end_num/model_end1.yaml"
    # "configs/unified_siglip_supp/end_num/model_end2.yaml"
    # "configs/unified_siglip_supp/end_num/model_end3.yaml"
    # "configs/unified_siglip_supp/end_num/model_end4.yaml"
    "configs/unified_siglip_supp/end_num/model_end5.yaml"
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
