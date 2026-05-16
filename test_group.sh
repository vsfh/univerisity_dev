#!/bin/bash
set -euo pipefail

cd /media/data1/feihong/univerisity_dev

CHECKPOINT_NAME="${CHECKPOINT_NAME:-last.pth}"
OUTPUT_DIR="${OUTPUT_DIR:-/media/data1/feihong/univerisity_dev/eval_results/test_unify}"
MODEL_NAME="${MODEL_NAME:-google/siglip2-base-patch16-224}"

ENCODER_HEAT_CHECKPOINT="/media/data1/feihong/ckpt/model_1/${CHECKPOINT_NAME}"
ENCODER_LORA_CHECKPOINT="/media/data1/feihong/ckpt/model_lora_1/${CHECKPOINT_NAME}"

echo "============================================================"
echo "Testing encoder heat/sig checkpoint: ${ENCODER_HEAT_CHECKPOINT}"
echo "Started at: $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"

python test_unify.py \
    --model-types encoder_sig \
    --checkpoint "${ENCODER_HEAT_CHECKPOINT}" \
    --encoder-sig-model-name "${MODEL_NAME}" \
    --encoder-sig-use-text \
    --encoder-sig-use-angle \
    --encoder-sig-use-ap \
    --heatmap-confidence-weight 0.5 \
    --output-dir "${OUTPUT_DIR}" \
    "$@"

echo "============================================================"
echo "Testing encoder lora checkpoint: ${ENCODER_LORA_CHECKPOINT}"
echo "Started at: $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"

# python test_unify.py \
#     --model-types encoder_lora \
#     --checkpoint "${ENCODER_LORA_CHECKPOINT}" \
#     --encoder-sig-model-name "${MODEL_NAME}" \
#     --encoder-lora-rank 8 \
#     --encoder-lora-alpha 16.0 \
#     --encoder-lora-dropout 0.05 \
#     --encoder-lora-use-text \
#     --encoder-lora-use-angle \
#     --encoder-lora-use-ap \
#     --heatmap-confidence-weight 0.5 \
#     --output-dir "${OUTPUT_DIR}" \
#     "$@"

echo "============================================================"
echo "Finished all test_group evaluations"
echo "Finished at: $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"
