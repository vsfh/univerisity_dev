"""Foreground baseline train/test suite. Invoked only by the user."""
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import yaml

ROOT = Path(__file__).resolve().parent
MODELS = {
    **{"retrieval_" + name: ("retrieval", name)
       for name in ("clip", "openclip", "evaclip", "siglip", "sample_retrieval")},
    **{"grounding_" + name: ("grounding", name)
       for name in ("ocg", "det", "sample4geo", "smgeo", "lpn", "trogeolite")},
}


def run_logged(command, log_path, cwd, env):
    with log_path.open("w", encoding="utf-8") as log:
        log.write(json.dumps(command) + "\n")
        log.flush()
        process = subprocess.Popen(command, cwd=cwd, env=env, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, errors="replace",
                                   bufsize=1)
        try:
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            return process.wait()
        except BaseException:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            raise


def save_summary(output, rows):
    (output / "summary.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with (output / "summary.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main(names=None):
    from grounding.config import load_config as load_grounding
    from retrieval.config import load_config as load_retrieval

    names = list(MODELS) if not names else list(names)
    if len(names) != len(set(names)) or any(name not in MODELS for name in names):
        raise ValueError("Choose distinct model names from: " + ", ".join(MODELS))
    # Read every source config before replacing the previous suite output.
    configs = {}
    for key in names:
        family, name = MODELS[key]
        loader = load_retrieval if family == "retrieval" else load_grounding
        configs[key] = loader(str(ROOT / "configs" / family / (name + ".yaml")))

    outputs = (ROOT / "outputs").resolve()
    output = ROOT / "outputs" / "baseline_recheck"
    if output.is_symlink() or output.resolve() != outputs / "baseline_recheck":
        raise ValueError("Refusing to replace a redirected suite directory.")
    if output.exists():
        shutil.rmtree(output)
    for directory in ("configs/retrieval", "configs/grounding", "logs",
                      "results", "runtime", "cache"):
        (output / directory).mkdir(parents=True, exist_ok=True)
    output = output.resolve()
    env = os.environ.copy()
    env.setdefault("CUDA_VISIBLE_DEVICES", "0")
    env.update(PYTHONUNBUFFERED="1", PYTHONDONTWRITEBYTECODE="1",
               PYTHONHASHSEED="43", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               TMPDIR="/tmp", MPLCONFIGDIR=str(output / "cache" / "matplotlib"),
               TRITON_CACHE_DIR=str(output / "cache" / "triton"))
    # /tmp holds short-lived IPC sockets; SSHFS outputs cannot host those sockets.
    rows = []
    for key in names:
        family, name = MODELS[key]
        cfg = configs[key]
        checkpoint = output / "checkpoints" / key / "last.pth"
        result_dir = output / "results" / key
        config_path = output / "configs" / family / (name + ".yaml")
        cfg.update(exp_name=key, save_dir=str(checkpoint.parent),
                   config_path=str(config_path))
        cfg["train"].update(device="cuda:0", resume_checkpoint=None, save_best=False)
        cfg["eval"].update(checkpoint="last.pth", output_dir=str(result_dir))
        config_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
        train_command = [sys.executable, str(ROOT / family / "train.py"),
                         "--config", str(config_path), "--device", "cuda:0"]
        if family == "retrieval":
            test_command = [sys.executable, str(ROOT / "retrieval" / "eval.py"),
                            "--config", str(config_path), "--checkpoint",
                            str(checkpoint), "--device", "cuda:0"]
            metrics_path = result_dir / "metrics.json"
        else:
            test_command = [
                sys.executable, str(ROOT / "test_unify_ground.py"),
                "--model-types", name, "--config-dir", str(config_path.parent),
                "--checkpoint", str(checkpoint), "--device", "cuda:0",
                "--candidate-size", "100", "--test-crop-ratio", "1",
                "--seed", "43", "--output-dir", str(result_dir),
                "--output-suffix", name,
            ]
            metrics_path = result_dir / ("test_unify_" + name + "_" + name + ".json")
        row = dict(model=key, status="running", phase="train",
                   checkpoint=str(checkpoint))
        rows.append(row)
        save_summary(output, rows)
        print("\n=== " + key + ": train, then test last.pth ===", flush=True)
        try:
            code = run_logged(train_command, output / "logs" / (key + ".train.log"),
                              output / "runtime", env)
            row["returncode"] = code
            if code:
                raise RuntimeError("Training failed; test skipped.")
            if not checkpoint.is_file():
                raise FileNotFoundError("Training did not produce last.pth.")
            row["phase"] = "test"
            save_summary(output, rows)
            code = run_logged(test_command, output / "logs" / (key + ".test.log"),
                              output / "runtime", env)
            row["returncode"] = code
            if code:
                raise RuntimeError("Testing failed.")
            result = json.loads(metrics_path.read_text(encoding="utf-8"))
            if Path(result["checkpoint"]).resolve() != checkpoint.resolve():
                raise ValueError("Metrics refer to a different checkpoint.")
            row.update(result["overall"])
            row.update(status="ok", phase="done")
        except KeyboardInterrupt:
            row.update(status="interrupted", error="Interrupted by user.")
            save_summary(output, rows)
            raise
        except Exception as error:
            row.update(status="failed", error=str(error))
            print(key + ": " + str(error), flush=True)
        save_summary(output, rows)
    print("\nResults: " + str(output / "summary.csv"), flush=True)
    return int(any(row["status"] != "ok" for row in rows))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
