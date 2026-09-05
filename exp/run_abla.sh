#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

SAVE_ROOT="/media/data2/feihong/ckpt"
DEFAULT_SEEDS=(42 43 44)
if (( $# > 0 )); then
    SEEDS=("$@")
else
    SEEDS=("${DEFAULT_SEEDS[@]}")
fi

for SEED in "${SEEDS[@]}"; do
    if [[ ! "${SEED}" =~ ^[0-9]+$ ]] || (( SEED > 4294967295 )); then
        echo "Invalid seed '${SEED}'; expected an integer in [0, 4294967295]." >&2
        exit 2
    fi
done


for CONFIG_PATH in config_abla/*.yaml; do
    CONFIG_NAME="${CONFIG_PATH##*/}"
    BASE_EXP_NAME="${CONFIG_NAME%.yaml}_5x3"

    for SEED in "${SEEDS[@]}"; do
        EXP_NAME="${BASE_EXP_NAME}_seed_${SEED}"

        echo "============================================================"
        echo "Training ${EXP_NAME}"
        echo "Config: ${CONFIG_PATH}"
        echo "Seed: ${SEED}"
        echo "Checkpoint directory: ${SAVE_ROOT}/${EXP_NAME}"
        echo "Started at: $(date '+%Y-%m-%d %H:%M:%S')"
        echo "============================================================"

        PYTHONHASHSEED="${SEED}" CUBLAS_WORKSPACE_CONFIG=:4096:8 accelerate launch \
            --mixed_precision fp16 \
            --gradient_accumulation_steps 1 \
            train_ada.py \
            --config "${CONFIG_PATH}" \
            --exp-name "${EXP_NAME}" \
            --seed "${SEED}"

        echo "Testing ${EXP_NAME}"
        python test.py \
            --config "${CONFIG_PATH}" \
            --checkpoint "${SAVE_ROOT}/${EXP_NAME}/last.pth" \
            --seed "${SEED}" \
            --output-suffix "${EXP_NAME}" \
            --output-dir eval_results \
            --batch-size "${TEST_BATCH_SIZE:-8}" \
            --num-workers "${TEST_NUM_WORKERS:-8}"
    done
done

echo "============================================================"
echo "Finished all end_num ablations for seeds: ${SEEDS[*]}"
echo "Finished at: $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"
