#!/bin/bash
set -euo pipefail

cd /media/data1/feihong/remote/univerisity_dev

CHECKPOINT="${CHECKPOINT:-/media/data1/feihong/remote/ckpt/model_test_geo_input_ids/last.pth}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/media/data1/feihong/remote/univerisity_dev/eval_results/test_unify/model_test_geo_input_ids_para}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-8}"

COMMON_ARGS=(
    --model-types encoder_test
    --checkpoint "${CHECKPOINT}"
    --encoder-heat-use-angle
    --encoder-heat-use-text
    --batch-size "${BATCH_SIZE}"
    --num-workers "${NUM_WORKERS}"
    --lora-rank 8
    --lora-alpha 16.0
    --lora-dropout 0.05
)

run_eval() {
    local candidate_size="$1"
    local test_crop_ratio="$2"
    shift 2
    local run_name="candidate_${candidate_size}_crop_${test_crop_ratio//./p}"
    local output_dir="${OUTPUT_ROOT}/${run_name}"

    mkdir -p "${output_dir}"
    echo "============================================================"
    echo "Testing model_test_geo_input_ids"
    echo "candidate_size=${candidate_size}, test_crop_ratio=${test_crop_ratio}"
    echo "output_dir=${output_dir}"
    echo "Started at: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "============================================================"

    python test_unify.py \
        "${COMMON_ARGS[@]}" \
        --candidate-size "${candidate_size}" \
        --test-crop-ratio "${test_crop_ratio}" \
        --output-dir "${output_dir}" \
        --output-suffix "${run_name}" \
        "$@"
}

if [ ! -f "${CHECKPOINT}" ]; then
    echo "Checkpoint not found: ${CHECKPOINT}" >&2
    exit 1
fi

for candidate_size in 50 100 200 500; do
    run_eval "${candidate_size}" "1.0" "$@"
done

for test_crop_ratio in 0.5 0.8; do
    run_eval "100" "${test_crop_ratio}" "$@"
done

echo "============================================================"
echo "Finished parameter sweep"
echo "Output root: ${OUTPUT_ROOT}"
echo "Finished at: $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"
