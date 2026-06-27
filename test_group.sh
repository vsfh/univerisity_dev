#!/bin/bash
set -euo pipefail

cd /media/data1/feihong/remote/univerisity_dev

CHECKPOINT_NAME="${CHECKPOINT_NAME:-last.pth}"
OUTPUT_DIR="${OUTPUT_DIR:-/media/data1/feihong/remote/univerisity_dev/eval_results/test_unify}"
COMMON_ARGS=(
    --output-dir "${OUTPUT_DIR}"
    --batch-size "${BATCH_SIZE:-8}"
    --num-workers "${NUM_WORKERS:-8}"
    --lora-rank 8
    --lora-alpha 16.0
    --lora-dropout 0.05
    --encoder-heat-text-score-weight "${TEXT_SCORE_WEIGHT:-0.0}"
    --encoder-heat-text-rerank-topk "${TEXT_RERANK_TOPK:-50}"
)

# echo "============================================================"
# echo "Testing Encoder_heat without geo input"
# echo "Started at: $(date '+%Y-%m-%d %H:%M:%S')"
# echo "============================================================"
# python test_unify.py \
#     --model-types encoder_heat \
#     --checkpoint "/media/data1/feihong/remote/ckpt/model_heat_no_geo/${CHECKPOINT_NAME}" \
#     --no-encoder-heat-use-angle \
#     --no-encoder-heat-use-text \
#     "${COMMON_ARGS[@]}" \
#     "$@"

# echo "============================================================"
# echo "Testing Encoder_heat without input_ids"
# echo "Started at: $(date '+%Y-%m-%d %H:%M:%S')"
# echo "============================================================"
# python test_unify.py \
#     --model-types encoder_heat \
#     --checkpoint "/media/data1/feihong/remote/ckpt/model_heat_no_input_ids/${CHECKPOINT_NAME}" \
#     --encoder-heat-use-angle \
#     --no-encoder-heat-use-text \
#     "${COMMON_ARGS[@]}" \
#     "$@"
# --checkpoint "/media/data1/feihong/remote/ckpt/model_test_geo_input_ids/${CHECKPOINT_NAME}" \


echo "============================================================"
echo "Testing Encoder_test with geo and input_ids"
echo "Started at: $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"
python test_unify.py \
    --model-types encoder_heat \
    --checkpoint "/media/data1/feihong/remote/ckpt/model_heat_no_input_ids/${CHECKPOINT_NAME}" \
    --encoder-heat-use-angle \
    --no-encoder-heat-use-text \
    "${COMMON_ARGS[@]}" \
    "$@"

echo "============================================================"
echo "Finished all test_group evaluations"
echo "Finished at: $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"
