"""Merge legacy folders without deleting originals or duplicating large checkpoints."""
import argparse
import json
import os
from pathlib import Path
import shutil

import yaml
import study


def merge_existing(root, apply=False):
    target = root / "outputs" / "heatmap_lambda_box_0p5"
    sources = [target, root / "outputs/heatmap_lambda_box_0p5_42",
               root / "outputs/heatmap_lambda_box_0p5_3gpu"]
    base = yaml.safe_load((root / "exp/config_abla/baseline_wo_input_ids_ada_end_0_5.yaml").read_text())
    plans = []
    for weight in study.WEIGHTS:
        for seed in study.SEEDS:
            expected = study.make_payload(base, target, weight, seed)
            name = expected["exp_name"]
            candidates = []
            for source in sources:
                config = source / "configs" / (name + ".yaml")
                checkpoint = source / "checkpoints" / name / "last.pth"
                effective = checkpoint.parent / "effective_config.json"
                if not config.exists() and not checkpoint.exists():
                    continue
                recovered = False
                if config.exists():
                    payload = yaml.safe_load(config.read_text())
                    # Verify the scientific setup before accepting an existing run.
                    for key in ("exp_name", "use_ap", "end_num"):
                        if payload.get(key) != expected.get(key):
                            raise ValueError(f"Config mismatch: {config} ({key})")
                    if study.comparable_config(payload["config"]) != study.comparable_config(expected["config"]):
                        raise ValueError(f"Training config mismatch: {config}")
                    state, overall = study.inspect_run(source, payload, seed)
                else:
                    if not effective.exists():
                        raise ValueError(f"Missing config and effective_config: {checkpoint}")
                    saved = json.loads(effective.read_text())
                    if saved.get("seed") != seed or saved.get("end_num") != study.BOX_WEIGHT or saved.get("use_ap") != expected["use_ap"]:
                        raise ValueError(f"Cannot recover config: {effective}")
                    for key, value in study.comparable_config(expected["config"]).items():
                        if saved["config"].get(key) != value:
                            raise ValueError(f"Effective config mismatch: {effective} ({key})")
                    payload = study.make_payload(base, target, weight, seed, saved["config"].get("DATA_PARALLEL_GPUS", 1))
                    state, overall = ("test" if study.checkpoint_ready(checkpoint) else "train"), None
                    recovered = True
                candidates.append(dict(source=source, payload=payload, state=state, overall=overall, recovered=recovered))
            if not candidates:
                continue
            completed = [item for item in candidates if item["state"] == "done"]
            if completed and any(item["overall"] != completed[0]["overall"] for item in completed[1:]):
                raise ValueError(f"Conflicting completed results for {name}; originals preserved for manual selection.")
            chosen = max(candidates, key=lambda item: {"train": 0, "test": 1, "done": 2}[item["state"]])
            plans.append((name, weight, seed, chosen, candidates))
    report = {"target": str(target), "runs": [
        {"run": name, "lambda_heatmap": weight, "seed": seed, "state": chosen["state"],
         "selected_source": chosen["source"].name, "config_recovered": chosen["recovered"],
         "all_sources": [item["source"].name for item in candidates]}
        for name, weight, seed, chosen, candidates in plans]}
    if not apply:
        return report
    target = study.prepare_output(root)
    if any(path.is_dir() for path in (target / "locks").iterdir()):
        raise RuntimeError("Stop active runners before merging; a shared lock exists.")
    with study.directory_lock(target / "locks" / "merge"):
        for name, weight, seed, chosen, candidates in plans:
            source = chosen["source"]
            dst_config = target / "configs" / (name + ".yaml")
            if not dst_config.exists():
                study.atomic_text(dst_config, yaml.safe_dump(chosen["payload"], sort_keys=False))
            if source != target:
                # Relative links work on both hosts despite different repository mount paths.
                for subdir, filename in [("checkpoints", name), ("results", name + ".json"),
                                         ("logs", name + ".train.log"), ("logs", name + ".test.log")]:
                    src, dst = source / subdir / filename, target / subdir / filename
                    if not src.exists():
                        continue
                    if dst.exists() or dst.is_symlink():
                        if src.resolve() == dst.resolve():
                            continue
                        raise ValueError(f"Target already exists: {dst}; no overwrite performed.")
                    if subdir == "results":
                        temporary = dst.with_name(dst.name + ".merge-tmp")
                        shutil.copy2(src, temporary)
                        os.replace(temporary, dst)
                    else:
                        dst.symlink_to(os.path.relpath(src, dst.parent), target_is_directory=src.is_dir())
            metadata = target / "metadata" / (name + ".json")
            if not metadata.exists():
                study.atomic_text(metadata, json.dumps({
                    "status": chosen["state"], "seed": seed, "lambda_heatmap": weight,
                    "training_gpus": chosen["payload"]["config"].get("DATA_PARALLEL_GPUS", 1),
                    "legacy_sources": [item["source"].name for item in candidates],
                    "selected_source": source.name,
                }, indent=2))
        study.atomic_text(target / "merge_report.json", json.dumps(report, indent=2))
        study.summarize(target, base=base)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true", help="Merge only; never train or test.")
    args = parser.parse_args()
    print(json.dumps(merge_existing(study.ROOT, args.apply), indent=2))
