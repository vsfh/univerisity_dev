#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

# Physical GPU 1 is exposed as cuda:0 inside this process.
export CUDA_VISIBLE_DEVICES=1

SAVE_ROOT="/media/data2/feihong/ckpt"
EVAL_ROOT="eval_results"

is_valid_result() {
    local result_path="$1"
    [[ -s "${result_path}" ]] && \
        python -c 'import json, sys; json.load(open(sys.argv[1], "r"))' \
            "${result_path}" >/dev/null 2>&1
}

train_job() {
    local config_stem="$1"
    local seed="$2"
    local config_path="config_abla/${config_stem}.yaml"
    local exp_name="${config_stem}_5x3_seed_${seed}"
    local checkpoint="${SAVE_ROOT}/${exp_name}/last.pth"
    local result_path="${EVAL_ROOT}/${exp_name}.json"

    if is_valid_result "${result_path}"; then
        echo "[skip train] Valid result already exists: ${result_path}"
        return
    fi
    if [[ -s "${checkpoint}" ]]; then
        echo "[skip train] Checkpoint already exists: ${checkpoint}"
        return
    fi

    echo "============================================================"
    echo "[GPU 1] Training ${exp_name}"
    echo "Started at: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "============================================================"

    PYTHONHASHSEED="${seed}" CUBLAS_WORKSPACE_CONFIG=:4096:8 accelerate launch \
        --num_processes 1 \
        --mixed_precision fp16 \
        --gradient_accumulation_steps 1 \
        train_ada.py \
        --config "${config_path}" \
        --exp-name "${exp_name}" \
        --seed "${seed}"
}

test_job() {
    local config_stem="$1"
    local seed="$2"
    local config_path="config_abla/${config_stem}.yaml"
    local exp_name="${config_stem}_5x3_seed_${seed}"
    local checkpoint="${SAVE_ROOT}/${exp_name}/last.pth"
    local result_path="${EVAL_ROOT}/${exp_name}.json"

    if is_valid_result "${result_path}"; then
        echo "[skip test] Valid result already exists: ${result_path}"
        return
    fi

    while [[ ! -s "${checkpoint}" ]]; do
        echo "[GPU 1] Waiting for checkpoint from GPU 0: ${checkpoint}"
        sleep 60
    done

    echo "============================================================"
    echo "[GPU 1] Testing ${exp_name}"
    echo "Started at: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "============================================================"

    python test.py \
        --config "${config_path}" \
        --checkpoint "${checkpoint}" \
        --seed "${seed}" \
        --output-suffix "${exp_name}" \
        --output-dir "${EVAL_ROOT}" \
        --batch-size "${TEST_BATCH_SIZE:-8}" \
        --num-workers "${TEST_NUM_WORKERS:-8}"
}

# Four training jobs and seven test jobs: about 30 h total.
train_job baseline_wo_input_ids_ada_end_0_5 43
test_job  baseline_wo_input_ids_ada_end_0_5 43

train_job baseline_wo_input_ids_ada_end_0_7 44
test_job  baseline_wo_input_ids_ada_end_0_7 44
test_job  baseline_wo_input_ids_ada_end_0_5 42

train_job baseline_wo_input_ids_ada_end_0_7 43
test_job  baseline_wo_input_ids_ada_end_0_7 43
test_job  baseline_wo_input_ids_ada_end_0_7 42

train_job baseline_wo_input_ids_ada_end_0_9 43
test_job  baseline_wo_input_ids_ada_end_0_9 42
test_job  baseline_wo_input_ids_ada_end_0_9 43

echo "============================================================"
echo "GPU 1 ablation queue finished at: $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"
