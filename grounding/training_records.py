import json
import os
from pathlib import Path
from typing import Any, Dict

import torch


def _temporary_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".tmp")


def atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_path(path)
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(temporary, path)


def append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def atomic_save_checkpoint(path: Path, payload: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_path(path)
    torch.save(payload, temporary)
    os.replace(temporary, path)


def load_v2_resume_checkpoint(path: Path) -> Dict[str, Any]:
    path = Path(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("architecture_version") != 2:
        raise ValueError(
            f"Resume requires a checkpoint with architecture_version=2: {path}"
        )
    if not isinstance(payload.get("model"), dict):
        raise ValueError(f"Version-2 checkpoint has no model state: {path}")
    return payload
