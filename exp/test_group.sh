#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

CONFIG_PATH="${CONFIG_PATH:-config_abla/baseline_wo_input_ids_ada_end_0_5.yaml}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-/media/data2/feihong/ckpt/baseline_wo_input_ids_ada_end_0_5_5x3_seed_43/last.pth}"
OUTPUT_DIR="${OUTPUT_DIR:-eval_results/baseline_wo_input_ids_ada_end_0_5_5x3_seed_43_candidate_crop_sweep}"
BATCH_SIZE="${TEST_BATCH_SIZE:-8}"
NUM_WORKERS="${TEST_NUM_WORKERS:-8}"

CANDIDATE_SIZES=(200 400 600)
TEST_CROP_RATIOS=(0.8 0.6 0.4)

if [[ ! -f "${CONFIG_PATH}" ]]; then
    echo "Config not found: ${CONFIG_PATH}" >&2
    exit 1
fi
if [[ ! -f "${CHECKPOINT_PATH}" ]]; then
    echo "Checkpoint not found: ${CHECKPOINT_PATH}" >&2
    exit 1
fi

echo "============================================================"
echo "Config: ${CONFIG_PATH}"
echo "Checkpoint: ${CHECKPOINT_PATH}"
echo "Output directory: ${OUTPUT_DIR}"
echo "Experiments: ${#CANDIDATE_SIZES[@]} candidate-size runs + ${#TEST_CROP_RATIOS[@]} crop-ratio runs"
echo "Started at: $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"

for candidate_size in "${CANDIDATE_SIZES[@]}"; do
    echo "Testing candidate_size=${candidate_size}, test_crop_ratio=1.0"
    python test.py \
        --config "${CONFIG_PATH}" \
        --checkpoint "${CHECKPOINT_PATH}" \
        --output-dir "${OUTPUT_DIR}" \
        --output-suffix "baseline_wo_input_ids_ada_5x3_candidate_${candidate_size}_crop_1.0" \
        --candidate-size "${candidate_size}" \
        --test-crop-ratio 1.0 \
        --batch-size "${BATCH_SIZE}" \
        --num-workers "${NUM_WORKERS}" \
        "$@"
done

for test_crop_ratio in "${TEST_CROP_RATIOS[@]}"; do
    echo "Testing candidate_size=100, test_crop_ratio=${test_crop_ratio}"
    python test.py \
        --config "${CONFIG_PATH}" \
        --checkpoint "${CHECKPOINT_PATH}" \
        --output-dir "${OUTPUT_DIR}" \
        --output-suffix "baseline_wo_input_ids_ada_5x3_candidate_100_crop_${test_crop_ratio}" \
        --candidate-size 100 \
        --test-crop-ratio "${test_crop_ratio}" \
        --batch-size "${BATCH_SIZE}" \
        --num-workers "${NUM_WORKERS}" \
        "$@"
done

echo "============================================================"
echo "Finished 6 candidate/crop evaluations"
echo "Finished at: $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"
