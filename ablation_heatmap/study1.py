"""Run missing heatmap experiments, preserving completed training and evaluation."""
import copy
import argparse
import csv
from contextlib import contextmanager
import io
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import time
import socket
import uuid
import zipfile

import yaml

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = [0.4, 0.2, 0.1, 0.05, 0.01]
SEEDS = [44, 43, 42]  # All seeds in the shared summary, independent of this worker.
BOX_WEIGHT = 0.5
METRICS = ["recall@1", "recall@5", "recall@10", "mean_iou",
           "ratio_iou_gt_0_5", "uIoU", "ratio_uIoU_gt_0_25",
           "mean_center_distance", "uCDE"]


def prepare_output(root, data_parallel_gpus=1):
    # Preserve existing runs, including interrupted experiments.
    outputs = (root / "outputs").resolve()
    if data_parallel_gpus not in (1, 3):
        raise ValueError("This study supports one or three GPUs.")
    folder = "heatmap_lambda_box_0p5"
    study = outputs / folder
    if study.is_symlink() or study.resolve().parent != outputs:
        raise ValueError("Refusing to use a redirected study directory.")
    for folder in ["configs", "checkpoints", "results", "logs", "runtime", "cache", "metadata", "locks"]:
        (study / folder).mkdir(parents=True, exist_ok=True)
    return study


def make_payload(base, study, weight, seed, data_parallel_gpus=1):
    payload = copy.deepcopy(base)
    name = f"heatmap_{weight:g}_seed_{seed}".replace(".", "p")
    payload.update(exp_name=name, end_num=BOX_WEIGHT, save_root=str(study / "checkpoints"))
    payload.pop("save_dir", None)
    payload["config"]["HEATMAP_LOSS_WEIGHT"] = weight / BOX_WEIGHT
    payload["config"]["DATA_PARALLEL_GPUS"] = data_parallel_gpus
    return payload


def comparable_config(config):
    config = dict(config)
    config.pop("DATA_PARALLEL_GPUS", None)  # Device count is recorded as provenance.
    return config


def atomic_text(path, text):
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def directory_lock(path, wait_seconds=0):
    # mkdir is atomic on the shared SFTP-backed filesystem; process-local flock is not.
    deadline = time.monotonic() + wait_seconds
    while True:
        try:
            path.mkdir()
            break
        except FileExistsError:
            if time.monotonic() >= deadline:
                raise RuntimeError(f"Busy lock: {path}. Check its owner.json before removing a stale lock.")
            time.sleep(0.2)
    try:
        (path / "owner.json").write_text(json.dumps({"host": socket.gethostname(), "pid": os.getpid()}))
        yield
    finally:
        (path / "owner.json").unlink(missing_ok=True)
        path.rmdir()


def checkpoint_ready(path):
    # train_ada writes last.pth only at the end. An interrupted torch.save lacks
    # the ZIP central directory; reject it without loading multi-GB model tensors.
    return path.is_file() and zipfile.is_zipfile(path)


def inspect_run(study, payload, seed):
    """Read-only classification; do not infer completion from directory existence."""
    name = payload["exp_name"]
    config_path = study / "configs" / (name + ".yaml")
    checkpoint = study / "checkpoints" / name / "last.pth"
    result_path = study / "results" / (name + ".json")
    if config_path.exists():
        previous = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        # Output paths can change when migrating a finished study to another host.
        for key in ("config", "use_ap", "end_num", "exp_name"):
            left, right = previous.get(key), payload.get(key)
            if key == "config":
                left, right = comparable_config(left), comparable_config(right)
            if left != right:
                raise ValueError(f"Existing config differs ({key}): {config_path}. Existing results preserved.")
    elif checkpoint.exists() or result_path.exists():
        raise ValueError(f"Cannot verify artifacts without their config: {config_path}")
    effective = checkpoint.parent / "effective_config.json"
    if effective.exists():
        try:
            previous_seed = json.loads(effective.read_text(encoding="utf-8")).get("seed")
        except ValueError:
            return "train", None  # Another host may just be writing startup metadata.
        if previous_seed is not None and int(previous_seed) != seed:
            raise ValueError(f"Existing checkpoint seed differs: {effective}")
    if result_path.exists():
        try:
            result = json.loads(result_path.read_text(encoding="utf-8"))
            overall = result["overall"]
            saved_checkpoint = Path(result["checkpoint"])
            valid = (saved_checkpoint.name == "last.pth" and saved_checkpoint.parent.name == name
                     and result["candidate_size"] == 100 and result["test_crop_ratio"] == 1.0
                     and result["sat_size"] == {"height": 432, "width": 768}
                     and overall["num_samples"] > 0
                     and all(isinstance(overall[key], (int, float)) and math.isfinite(overall[key])
                             for key in METRICS))
            if valid:
                return "done", overall
        except (ValueError, KeyError, TypeError):
            pass  # Missing/truncated/invalid evaluation output must be regenerated.
    return ("test" if checkpoint_ready(checkpoint) else "train"), None


def run_command(command, runtime, env, log_path):
    """Show progress in the terminal and retain a log."""
    with log_path.open("a", encoding="utf-8") as log:
        log.write("\nRUN " + json.dumps(command) + "\n")
        log.flush()
        process = subprocess.Popen(command, cwd=runtime, env=env,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, encoding="utf-8", errors="replace")
        try:
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            returncode = process.wait()
            if returncode:
                raise subprocess.CalledProcessError(returncode, command)
        except BaseException:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            raise
        finally:
            process.stdout.close()


def collect_completed(study, base):
    completed = []
    for weight in WEIGHTS:
        for seed in SEEDS:
            payload = make_payload(base, study, weight, seed)
            state, overall = inspect_run(study, payload, seed)
            if state == "done":
                completed.append({"lambda_heatmap": weight, "seed": seed, "overall": overall})
    return completed


def summarize(study, completed=None, base=None):
    # Always rebuild from ALL seeds on disk while holding the shared summary lock.
    if base is None:
        base = yaml.safe_load((ROOT / "exp/config_abla/baseline_wo_input_ids_ada_end_0_5.yaml").read_text())
    with directory_lock(study / "locks" / "summary", wait_seconds=60):
        _write_summary(study, collect_completed(study, base))


def _write_summary(study, completed):
    rows = []
    for weight in WEIGHTS:
        results = [item for item in completed if item["lambda_heatmap"] == weight]
        row = {"lambda_heatmap": weight, "lambda_box": BOX_WEIGHT,
               "n_completed": len(results), "n_expected": len(SEEDS)}
        for metric in METRICS:
            values = [item["overall"][metric] for item in results
                      if item["overall"][metric] is not None]
            row[metric + "_mean"] = statistics.mean(values) if values else None
            row[metric + "_std"] = statistics.stdev(values) if len(values) > 1 else None
        rows.append(row)
    handle = io.StringIO(newline="")
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    atomic_text(study / "summary.csv", handle.getvalue())
    atomic_text(study / "summary.json", json.dumps({"rows": rows, "completed_runs": completed}, indent=2) + "\n")


def main(data_parallel_gpus=1):
    parser = argparse.ArgumentParser(description="Run selected seeds in the shared heatmap study.")
    parser.add_argument("--seeds", nargs="+", type=int, choices=SEEDS)
    args = parser.parse_args()
    selected_seeds = args.seeds if args.seeds is not None else ([42, 43] if data_parallel_gpus == 3 else [44])
    if len(set(selected_seeds)) != len(selected_seeds):
        raise ValueError("Seeds must not repeat.")
    exp = ROOT / "exp"
    base = yaml.safe_load(
        (exp / "config_abla/baseline_wo_input_ids_ada_end_0_5.yaml").read_text(encoding="utf-8"))
    config = base["config"]
    if (config["OPTIMIZE_OBJECTIVE"] != "combined"
            or config["USE_HEATMAP_LOSS"] is not True
            or config.get("GROUNDING_START_EPOCH", 0) != 0
            or config.get("GROUNDING_WARMUP_EPOCHS", 0) != 0):
        raise ValueError("Expected combined baseline with heatmap enabled and no grounding warmup.")
    for name in ["train_ada.py", "test.py"]:
        if not (exp / name).is_file():
            raise FileNotFoundError(exp / name)
    if data_parallel_gpus == 3:
        # Validate before starting any work. No model is loaded here.
        sys.path.insert(0, str(exp))
        from batch_parallel import require_visible_gpus
        require_visible_gpus(3)
        if int(config["BATCH_SIZE"]) < 3:
            raise ValueError("Global batch size must be at least three.")
    study = prepare_output(ROOT, data_parallel_gpus)
    print(f"Continue study: {study}\nWeights={WEIGHTS}; worker seeds={selected_seeds}; lambda_box={BOX_WEIGHT}", flush=True)
    env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONDONTWRITEBYTECODE="1",
               HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               CUBLAS_WORKSPACE_CONFIG=":4096:8")
    # Unix sockets cannot live on the server's network-mounted outputs/.
    env["TMPDIR"] = "/tmp" if os.name == "posix" else tempfile.gettempdir()
    if data_parallel_gpus == 3:
        # One process owns the full batch and optimizer; all three GPUs do forward/backward.
        # Ignore inherited torchrun/Accelerate settings from the other machine.
        for key in list(env):
            if key.startswith("ACCELERATE_") or key in {
                "RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE",
                "MASTER_ADDR", "MASTER_PORT", "GROUP_RANK", "ROLE_RANK", "ROLE_WORLD_SIZE",
            }:
                env.pop(key)
        env.update(ACCELERATE_USE_CPU="false", ACCELERATE_DYNAMO_BACKEND="NO",
                   ACCELERATE_MIXED_PRECISION="fp16" if config["USE_AMP"] else "no",
                   ACCELERATE_GRADIENT_ACCUMULATION_STEPS=str(config["GRAD_ACCUMULATION_STEPS"]))
        manifest = {
            "parallelism": "single_process_data_parallel", "gpus": 3,
            "visible_devices": env.get("CUDA_VISIBLE_DEVICES"),
            "global_batch_size": config["BATCH_SIZE"],
            "gradient_accumulation_steps": config["GRAD_ACCUMULATION_STEPS"],
            "effective_batch_size": config["BATCH_SIZE"] * config["GRAD_ACCUMULATION_STEPS"],
            "weights": WEIGHTS, "seeds": selected_seeds, "lambda_box": BOX_WEIGHT,
            "loss_scope": "full batch after gathering outputs",
        }
        atomic_text(study / "metadata" / (f"launcher_{socket.gethostname()}_{os.getpid()}.json"), json.dumps(manifest, indent=2))
        print(json.dumps(manifest, indent=2), flush=True)
    for variable, folder in [("MPLCONFIGDIR", "matplotlib"),
                             ("TORCH_EXTENSIONS_DIR", "torch_extensions"),
                             ("TORCHINDUCTOR_CACHE_DIR", "torch_inductor"),
                             ("TRITON_CACHE_DIR", "triton")]:
        path = study / "cache" / folder
        path.mkdir(exist_ok=True)
        env[variable] = str(path)
    completed = []
    plans = []
    for weight in WEIGHTS:
        for seed in selected_seeds:
            name = f"heatmap_{weight:g}_seed_{seed}".replace(".", "p")
            payload = copy.deepcopy(base)
            payload.update(exp_name=name, end_num=BOX_WEIGHT,
                           save_root=str(study / "checkpoints"))
            payload.pop("save_dir", None)
            # train_ada nests heatmap loss inside lambda_box * grounding_loss.
            payload["config"]["HEATMAP_LOSS_WEIGHT"] = weight / BOX_WEIGHT
            if data_parallel_gpus == 3:
                payload["config"]["DATA_PARALLEL_GPUS"] = 3
            config_path = study / "configs" / (name + ".yaml")
            state, overall = inspect_run(study, payload, seed)
            plans.append((weight, seed, name, payload, config_path, state))
            if state == "done":
                completed.append({"lambda_heatmap": weight, "seed": seed, "overall": overall})
    summarize(study, completed)
    for weight, seed, name, payload, config_path, state in plans:
        if state == "done":
            print(f"SKIP completed train+test: {name}", flush=True)
            continue
        with directory_lock(study / "locks" / name):
            # Recheck after locking: another host may have completed it since planning.
            state, _ = inspect_run(study, payload, seed)
            if state == "done":
                print(f"SKIP completed train+test: {name}", flush=True)
                continue
            execute_run(study, exp, env.copy(), weight, seed, name, payload, config_path, state, data_parallel_gpus)
        summarize(study, base=base)
    summarize(study, base=base)
    print(f"All selected training and evaluation finished: {study / 'summary.csv'}", flush=True)


def execute_run(study, exp, env, weight, seed, name, payload, config_path, state, data_parallel_gpus):
    if state == "train":
        atomic_text(config_path, yaml.safe_dump(payload, sort_keys=False))
    # For test-only continuation retain the original training config/provenance.
    checkpoint_dir = study / "checkpoints" / name
    env["PYTHONHASHSEED"] = str(seed)
    train = [sys.executable, "-m", "accelerate.commands.launch",
             "--num_processes", "1", "--num_machines", "1",
             "--mixed_precision", "fp16", "--gradient_accumulation_steps", "1",
             str(exp / "train_ada.py"), "--config", str(config_path),
             "--exp-name", name, "--save-dir", str(checkpoint_dir),
             "--end-num", str(BOX_WEIGHT), "--seed", str(seed)]
    if data_parallel_gpus == 3:
        # Accelerator is configured inside train_ada; do not launch three data loaders.
        train = [sys.executable, *train[train.index(str(exp / "train_ada.py")):]]
    if state == "train":
        atomic_text(study / "metadata" / (name + ".json"), json.dumps({
            "host": socket.gethostname(), "training_gpus": data_parallel_gpus,
            "seed": seed, "lambda_heatmap": weight, "status": "training",
        }, indent=2))
        print(f"\nTRAIN {name}", flush=True)
        run_command(train, study / "runtime", env, study / "logs" / (name + ".train.log"))
    else:
        print(f"SKIP training; TEST existing last.pth: {name}", flush=True)
    checkpoint = checkpoint_dir / "last.pth"
    if not checkpoint_ready(checkpoint):
        raise FileNotFoundError(checkpoint)
    test = [sys.executable, str(exp / "test.py"), "--config", str(config_path),
            "--checkpoint", str(checkpoint), "--seed", str(seed),
            "--output-dir", str(study / "results"), "--output-suffix", name,
            "--batch-size", "8", "--num-workers", "8",
            "--candidate-size", "100", "--test-crop-ratio", "1.0"]
    if data_parallel_gpus == 3:
        test.extend(["--data-parallel-gpus", "3"])
    print(f"\nTEST {name}", flush=True)
    run_command(test, study / "runtime", env, study / "logs" / (name + ".test.log"))
    state, overall = inspect_run(study, payload, seed)
    if state != "done":
        raise ValueError(f"Test did not produce a complete matching result: {name}")
    metadata_path = study / "metadata" / (name + ".json")
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {
        "training_gpus": payload["config"].get("DATA_PARALLEL_GPUS", 1)}
    metadata.update(status="done", evaluation_gpus=data_parallel_gpus, evaluation_host=socket.gethostname())
    atomic_text(metadata_path, json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
