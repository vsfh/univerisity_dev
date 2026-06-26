#!/bin/bash
set -euo pipefail

cd /media/data1/feihong/univerisity_dev

CONFIGS=(
    "configs/unified_siglip_supp/single_config/baseline_wo_input_ids.yaml"
    "configs/unified_siglip_supp/single_config/baseline.yaml"
)

MODEL_TYPES=(
    "encoder_test"
    "encoder_test"
)

CHECKPOINT_DIRS=(
    "/media/data1/feihong/ckpt/baseline_wo_input_ids"
    "/media/data1/feihong/ckpt/baseline"
)

TEXT_FLAGS=(
    "--no-encoder-heat-use-text"
    "--encoder-heat-use-text"
)

CHECKPOINT_NAME="${CHECKPOINT_NAME:-last.pth}"
OUTPUT_DIR="${OUTPUT_DIR:-/media/data1/feihong/univerisity_dev/eval_results/test_unify}"
COMMON_TEST_ARGS=(
    --output-dir "${OUTPUT_DIR}"
    --batch-size "${BATCH_SIZE:-8}"
    --num-workers "${NUM_WORKERS:-8}"
    --lora-rank 8
    --lora-alpha 16.0
    --lora-dropout 0.05
    --encoder-heat-text-score-weight "${TEXT_SCORE_WEIGHT:-0.0}"
    --encoder-heat-text-rerank-topk "${TEXT_RERANK_TOPK:-50}"
)

for IDX in "${!CONFIGS[@]}"; do
    CONFIG_PATH="${CONFIGS[$IDX]}"
    MODEL_TYPE="${MODEL_TYPES[$IDX]}"
    CHECKPOINT_PATH="${CHECKPOINT_DIRS[$IDX]}/${CHECKPOINT_NAME}"
    TEXT_FLAG="${TEXT_FLAGS[$IDX]}"

    echo "============================================================"
    echo "Running unified_siglip_supp experiment: ${CONFIG_PATH}"
    echo "Started at: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "============================================================"

    bash train.sh "${CONFIG_PATH}"

    echo "============================================================"
    echo "Finished experiment: ${CONFIG_PATH}"
    echo "Finished at: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "============================================================"

    echo "============================================================"
    echo "Testing ${MODEL_TYPE}: ${CHECKPOINT_PATH}"
    echo "Started at: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "============================================================"

    python test_unify.py \
        --model-types "${MODEL_TYPE}" \
        --checkpoint "${CHECKPOINT_PATH}" \
        --encoder-heat-use-angle \
        "${TEXT_FLAG}" \
        "${COMMON_TEST_ARGS[@]}" \
        "$@"

    echo "============================================================"
    echo "Finished test: ${MODEL_TYPE}"
    echo "Finished at: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "============================================================"
done

echo "============================================================"
echo "Finished all train_group experiments and test_unify evaluations"
echo "Finished at: $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"
