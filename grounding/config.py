import copy
from pathlib import Path
from typing import Any, Dict

import yaml


DEFAULT_CONFIG: Dict[str, Any] = {
    "exp_name": "grounding_exp",
    "save_dir": "/media/data1/feihong/ckpt/grounding_exp",
    "model": {
        "type": "siglip2_heat",
        "checkpoint": None,
        "model_name": "google/siglip2-base-patch16-224",
        "cache_dir": "/media/data1/feihong/hf_cache",
        "use_angle": True,
        "use_heatmap": True,
        "use_text": False,
    },
    "data": {
        "num_workers": 8,
        "sat_size": {"height": 432, "width": 768},
        "drone_size": {"height": 256, "width": 256},
    },
    "train": {
        "epochs": 20,
        "batch_size": 32,
        "grad_accumulation_steps": 2,
        "lr": 5.0e-5,
        "weight_decay": 1.0e-4,
        "amp": True,
        "grad_clip_norm": 1.0,
        "device": "cuda:0",
    },
    "loss": {
        "bbox_weight": 1.0,
        "heatmap_weight": 0.2,
        "heatmap_confidence_weight": 0.5,
        "heatmap_loss_type": ["mse", "cross_entropy"],
        "heatmap_bbox_center_edge_value": 0.2,
        "heatmap_bbox_center_log_scale": 9.0,
    },
    "eval": {
        "batch_size": 8,
        "checkpoint": "last.pth",
        "output_dir": "eval_results/grounding/grounding_exp",
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
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    cfg = _merge_dict(DEFAULT_CONFIG, raw or {})
    cfg["config_path"] = str(config_path)
    cfg["save_dir"] = str(cfg["save_dir"])
    cfg["eval"]["output_dir"] = str(cfg["eval"]["output_dir"])
    return cfg
