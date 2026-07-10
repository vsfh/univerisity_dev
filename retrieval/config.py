import copy
from pathlib import Path
from typing import Any, Dict

import yaml


DEFAULT_CONFIG: Dict[str, Any] = {
    "exp_name": "retrieval_exp",
    "save_dir": "/media/data1/feihong/ckpt/retrieval_exp",
    "model": {
        "type": "siglip",
        "model_name": "google/siglip-base-patch16-224",
        "pretrained": None,
        "cache_dir": "/media/data1/feihong/hf_cache",
        "proj_dim": 768,
        "emb_size": 1024,
        "pretrained_backbone": True,
    },
    "data": {
        "num_workers": 8,
        "sat_size": {"height": 432, "width": 768},
        "drone_size": {"height": 256, "width": 256},
        "test_crop_ratio": 1.0,
        "subset_heights": [150, 200, 250, 300],
        "subset_angles": [0, 45, 90, 135, 180, 225, 270, 315],
    },
    "train": {
        "epochs": 20,
        "batch_size": 32,
        "grad_accumulation_steps": 1,
        "lr": 1.0e-5,
        "weight_decay": 0.0,
        "amp": True,
        "enable_tf32": True,
        "grad_clip_norm": 0.0,
        "device": "cuda:0",
        "drop_last": True,
    },
    "loss": {
        "temperature": 0.07,
        "granularity": "auto",
        "use_text_loss": False,
    },
    "eval": {
        "batch_size": 8,
        "checkpoint": "last.pth",
        "output_dir": "eval_results/retrieval/retrieval_exp",
        "candidate_size": 100,
        "include_file": "/media/data1/feihong/ckpt/include1.json",
        "save_query_records": False,
    },
}


def _merge_dict(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge_dict(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path: str) -> Dict[str, Any]:
    config_path = Path(path)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    cfg = _merge_dict(DEFAULT_CONFIG, raw)
    cfg["config_path"] = str(config_path)
    cfg["save_dir"] = str(cfg["save_dir"])
    cfg["eval"]["output_dir"] = str(cfg["eval"]["output_dir"])
    return cfg
