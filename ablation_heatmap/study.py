"""Train and test all heatmap ablations from scratch, overwriting this study."""
import copy
import csv
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import tempfile

import yaml

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = [0.01, 0.05, 0.1, 0.2, 0.4]
SEEDS = [42]
BOX_WEIGHT = 0.5
METRICS = ["recall@1", "recall@5", "recall@10", "mean_iou",
           "ratio_iou_gt_0_5", "uIoU", "ratio_uIoU_gt_0_25",
           "mean_center_distance", "uCDE"]


def reset_output(root):
    # Only this fixed study directory may be replaced; never delete outputs/.
    outputs = (root / "outputs").resolve()
    study = outputs / "heatmap_lambda_box_0p5_42"
    if study.is_symlink() or study.resolve().parent != outputs:
        raise ValueError("Refusing to overwrite a redirected study directory.")
    if study.exists():
        shutil.rmtree(study)
    for folder in ["configs", "checkpoints", "results", "logs", "runtime", "cache"]:
        (study / folder).mkdir(parents=True, exist_ok=True)
    return study


def run_command(command, runtime, env, log_path):
    """Show progress in the terminal and retain a log."""
    with log_path.open("w", encoding="utf-8") as log:
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


def summarize(study, completed):
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
    with (study / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (study / "summary.json").write_text(
        json.dumps({"rows": rows, "completed_runs": completed}, indent=2) + "\n",
        encoding="utf-8")


def main():
    if len(sys.argv) != 1:
        raise SystemExit("No arguments needed: bash ablation_heatmap/run.sh")
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
    study = reset_output(ROOT)
    print(f"Fresh run: {study}\nWeights={WEIGHTS}; seeds={SEEDS}; lambda_box={BOX_WEIGHT}", flush=True)
    env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONDONTWRITEBYTECODE="1",
               HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               CUBLAS_WORKSPACE_CONFIG=":4096:8")
    # Unix sockets cannot live on the server's network-mounted outputs/.
    env["TMPDIR"] = "/tmp" if os.name == "posix" else tempfile.gettempdir()
    for variable, folder in [("MPLCONFIGDIR", "matplotlib"),
                             ("TORCH_EXTENSIONS_DIR", "torch_extensions"),
                             ("TORCHINDUCTOR_CACHE_DIR", "torch_inductor"),
                             ("TRITON_CACHE_DIR", "triton")]:
        path = study / "cache" / folder
        path.mkdir()
        env[variable] = str(path)
    completed = []
    summarize(study, completed)
    for weight in WEIGHTS:
        for seed in SEEDS:
            name = f"heatmap_{weight:g}_seed_{seed}".replace(".", "p")
            payload = copy.deepcopy(base)
            payload.update(exp_name=name, end_num=BOX_WEIGHT,
                           save_root=str(study / "checkpoints"))
            payload.pop("save_dir", None)
            # train_ada nests heatmap loss inside lambda_box * grounding_loss.
            payload["config"]["HEATMAP_LOSS_WEIGHT"] = weight / BOX_WEIGHT
            config_path = study / "configs" / (name + ".yaml")
            config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
            checkpoint_dir = study / "checkpoints" / name
            env["PYTHONHASHSEED"] = str(seed)
            train = [sys.executable, "-m", "accelerate.commands.launch",
                     "--num_processes", "1", "--num_machines", "1",
                     "--mixed_precision", "fp16", "--gradient_accumulation_steps", "1",
                     str(exp / "train_ada.py"), "--config", str(config_path),
                     "--exp-name", name, "--save-dir", str(checkpoint_dir),
                     "--end-num", str(BOX_WEIGHT), "--seed", str(seed)]
            print(f"\n[{len(completed) + 1}/{len(WEIGHTS) * len(SEEDS)}] TRAIN {name}", flush=True)
            run_command(train, study / "runtime", env, study / "logs" / (name + ".train.log"))
            checkpoint = checkpoint_dir / "last.pth"
            if not checkpoint.is_file():
                raise FileNotFoundError(checkpoint)
            test = [sys.executable, str(exp / "test.py"), "--config", str(config_path),
                    "--checkpoint", str(checkpoint), "--seed", str(seed),
                    "--output-dir", str(study / "results"), "--output-suffix", name,
                    "--batch-size", "8", "--num-workers", "8",
                    "--candidate-size", "100", "--test-crop-ratio", "1.0"]
            print(f"\nTEST {name}", flush=True)
            run_command(test, study / "runtime", env, study / "logs" / (name + ".test.log"))
            result = json.loads((study / "results" / (name + ".json")).read_text(encoding="utf-8"))
            completed.append({"lambda_heatmap": weight, "seed": seed, "overall": result["overall"]})
            summarize(study, completed)
    print(f"All training and evaluation finished: {study / 'summary.csv'}", flush=True)


if __name__ == "__main__":
    main()
