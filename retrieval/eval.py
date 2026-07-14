import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

import torch

ROOT = Path(__file__).resolve().parents[1]
RETRIEVAL_DIR = Path(__file__).resolve().parent
for path in (ROOT, RETRIEVAL_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from retrieval.config import load_config


def _checkpoint_path(cfg: Dict[str, Any]) -> str:
    checkpoint = cfg["eval"]["checkpoint"]
    return str(checkpoint) if os.path.isabs(str(checkpoint)) else os.path.join(str(cfg["save_dir"]), str(checkpoint))


def _print_result(result: Dict[str, Any]) -> None:
    print(json.dumps(result, indent=2, sort_keys=True))


def evaluate(cfg: Dict[str, Any], dry_run: bool = False, max_batches: int = 0) -> Dict[str, Any]:
    checkpoint = _checkpoint_path(cfg)
    if dry_run:
        result = {
            "status": "dry_run",
            "config": cfg["config_path"],
            "checkpoint": checkpoint,
            "output_dir": cfg["eval"]["output_dir"],
        }
        _print_result(result)
        return result

    del max_batches
    from retrieval.eval_retrieval import eval as legacy_eval

    device = str(cfg["train"]["device"])
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    metrics = legacy_eval(
        model_type=str(cfg["model"]["type"]),
        model_config=dict(cfg["model"]),
        checkpoint_path=checkpoint,
        subset_heights=cfg["data"].get("subset_heights"),
        subset_angles=cfg["data"].get("subset_angles"),
        candidate_size=cfg["eval"].get("candidate_size"),
        include_file=str(cfg["eval"].get("include_file")),
        batch_size=int(cfg["eval"]["batch_size"]),
        num_workers=int(cfg["data"]["num_workers"]),
        device=device,
        output_dir=str(cfg["eval"]["output_dir"]),
    )
    result = {
        "status": "ok",
        "model_type": str(cfg["model"]["type"]),
        "checkpoint": checkpoint,
        "num_samples": metrics.get("num_queries"),
        "overall": {
            "recall@1": metrics.get("recall@1"),
            "recall@5": metrics.get("recall@5"),
            "recall@10": metrics.get("recall@10"),
        },
        "retrieval": metrics,
    }
    output_dir = Path(cfg["eval"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    _print_result(result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate retrieval-only models from YAML.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if args.device:
        cfg["train"]["device"] = args.device
    if args.checkpoint:
        cfg["eval"]["checkpoint"] = args.checkpoint
    evaluate(cfg, dry_run=args.dry_run, max_batches=args.max_batches)


if __name__ == "__main__":
    main()
