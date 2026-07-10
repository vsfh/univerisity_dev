# Retrieval Config Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a YAML-driven retrieval train/eval workflow for CLIP, SigLIP, OpenCLIP, EVA-CLIP, and SampleGeoLite while preserving existing model implementations.

**Architecture:** Add retrieval modules that mirror the grounding package: config loading, lazy registry, adapters, loss helpers, train entry point, eval entry point, YAML configs, and a group script. Reuse `ShiftedSatelliteDroneDataset`, `unified_siglip_supp.build_retrieval_soft_targets`, and existing retrieval/test metric functions instead of rewriting model or dataset internals.

**Tech Stack:** Python, PyTorch, Hugging Face transformers, OpenCLIP, timm, PyYAML, pytest/unittest, bash.

---

## File Structure

- Create `retrieval/config.py`: default config and YAML merge logic.
- Create `retrieval/losses.py`: grid/global granularity inference and retrieval loss calculation.
- Create `retrieval/adapters.py`: feature payload dataclass and adapter classes around existing model methods.
- Create `retrieval/registry.py`: lazy model/processor/tokenizer builders for all five model types.
- Create `retrieval/train.py`: shared train loop with dry-run and max-steps support.
- Create `retrieval/eval.py`: shared eval entry point that reuses feature extraction/scoring concepts from current eval code.
- Create `configs/retrieval/{siglip,clip,openclip,evaclip,sample_retrieval}.yaml`: per-model configs.
- Create `train_retrieval_grounp.sh`: group train/eval runner using the requested spelling.
- Create `test_retrieval_config.py`: config defaults and YAML override coverage.
- Create `test_retrieval_losses.py`: grid/global loss selection coverage.
- Create `test_retrieval_registry.py`: lazy registry metadata and dry-run coverage without loading heavy weights.
- Create `test_train_retrieval_group.py`: static checks for the group script and device overrides.

Do not edit the old `retrieval/train_clip.py`, `retrieval/train_siglip.py`, `retrieval/train_openclip.py`, `retrieval/train_evaclip.py`, or `grounding/train_sample_retrieval.py` unless a later implementation step proves an import bug blocks the new registry.

### Task 1: Config Loader

**Files:**
- Create: `retrieval/config.py`
- Create: `test_retrieval_config.py`

- [ ] **Step 1: Write failing config tests**

Add `test_retrieval_config.py`:

```python
from pathlib import Path

from retrieval.config import load_config


def test_retrieval_config_defaults_keep_text_loss_off(tmp_path: Path) -> None:
    cfg_path = tmp_path / "siglip.yaml"
    cfg_path.write_text(
        """
exp_name: siglip
save_dir: /tmp/retrieval_siglip
model:
  type: siglip
""",
        encoding="utf-8",
    )

    cfg = load_config(str(cfg_path))

    assert cfg["config_path"] == str(cfg_path)
    assert cfg["model"]["type"] == "siglip"
    assert cfg["loss"]["use_text_loss"] is False
    assert cfg["loss"]["granularity"] == "auto"
    assert cfg["loss"]["temperature"] == 0.07
    assert cfg["data"]["sat_size"] == {"height": 432, "width": 768}
    assert cfg["eval"]["checkpoint"] == "last.pth"


def test_retrieval_config_merges_nested_overrides(tmp_path: Path) -> None:
    cfg_path = tmp_path / "clip.yaml"
    cfg_path.write_text(
        """
exp_name: clip
save_dir: /tmp/retrieval_clip
model:
  type: clip
train:
  batch_size: 4
loss:
  use_text_loss: true
data:
  subset_heights: [150, 300]
eval:
  candidate_size: 50
""",
        encoding="utf-8",
    )

    cfg = load_config(str(cfg_path))

    assert cfg["train"]["batch_size"] == 4
    assert cfg["train"]["epochs"] == 4
    assert cfg["loss"]["use_text_loss"] is True
    assert cfg["data"]["subset_heights"] == [150, 300]
    assert cfg["data"]["subset_angles"] == [0, 45, 90, 135, 180, 225, 270, 315]
    assert cfg["eval"]["candidate_size"] == 50
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest test_retrieval_config.py -q`

Expected: FAIL with `ModuleNotFoundError` or missing `retrieval.config`.

- [ ] **Step 3: Implement config loader**

Create `retrieval/config.py`:

```python
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
        "epochs": 4,
        "batch_size": 8,
        "grad_accumulation_steps": 1,
        "lr": 1.0e-5,
        "weight_decay": 0.0,
        "amp": False,
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
```

- [ ] **Step 4: Run config tests**

Run: `pytest test_retrieval_config.py -q`

Expected: PASS.

- [ ] **Step 5: Commit config loader**

```bash
git add retrieval/config.py test_retrieval_config.py
git commit -m "feat: add retrieval config loader"
```

### Task 2: Retrieval Loss Helpers

**Files:**
- Create: `retrieval/losses.py`
- Create: `test_retrieval_losses.py`

- [ ] **Step 1: Write failing loss tests**

Add `test_retrieval_losses.py`:

```python
import pytest
import torch

from retrieval.losses import (
    RetrievalLossResult,
    compute_retrieval_loss,
    infer_granularity,
    prepare_candidate_targets,
)


def test_infer_granularity_detects_grid_and_global() -> None:
    assert infer_granularity(torch.zeros(2, 9, 4), "auto") == "grid"
    assert infer_granularity(torch.zeros(2, 4), "auto") == "global"
    assert infer_granularity(torch.zeros(2, 9, 4), "grid") == "grid"
    assert infer_granularity(torch.zeros(2, 4), "global") == "global"


def test_infer_granularity_rejects_unsupported_token_count() -> None:
    with pytest.raises(ValueError, match="expected 9 grid locations"):
        infer_granularity(torch.zeros(2, 16, 4), "auto")


def test_prepare_grid_targets_uses_unified_soft_defaults() -> None:
    batch = {
        "index": torch.tensor([2, 5]),
        "satellite_id": torch.tensor([7, 7]),
    }
    candidates = torch.randn(2, 9, 4)

    flat_candidates, targets, granularity = prepare_candidate_targets(
        candidates,
        batch,
        requested_granularity="auto",
    )

    assert granularity == "grid"
    assert flat_candidates.shape == (18, 4)
    assert targets.shape == (2, 18)
    assert targets[0, 2] == torch.tensor(0.92)
    assert targets[1, 14] == torch.tensor(0.92)
    assert torch.allclose(targets.sum(dim=1), torch.ones(2))


def test_prepare_global_targets_ignores_index() -> None:
    batch = {
        "index": torch.tensor([8, 8]),
        "satellite_id": torch.tensor([7, 7]),
    }
    candidates = torch.randn(2, 4)

    flat_candidates, targets, granularity = prepare_candidate_targets(
        candidates,
        batch,
        requested_granularity="auto",
    )

    assert granularity == "global"
    assert flat_candidates.shape == (2, 4)
    assert torch.equal(targets, torch.tensor([0, 1]))


def test_compute_retrieval_loss_returns_image_only_by_default() -> None:
    query = torch.eye(2, 4)
    candidates = torch.eye(2, 4)
    batch = {"index": torch.tensor([8, 8])}

    result = compute_retrieval_loss(
        query_feats=query,
        candidate_feats=candidates,
        batch=batch,
        temperature=0.07,
        requested_granularity="auto",
        text_feats=None,
        use_text_loss=False,
    )

    assert isinstance(result, RetrievalLossResult)
    assert result.granularity == "global"
    assert result.text_loss is None
    assert result.total.item() >= 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest test_retrieval_losses.py -q`

Expected: FAIL with missing `retrieval.losses`.

- [ ] **Step 3: Implement loss helpers**

Create `retrieval/losses.py`:

```python
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from unified_siglip_supp import build_retrieval_soft_targets


@dataclass
class RetrievalLossResult:
    total: torch.Tensor
    image_loss: torch.Tensor
    text_loss: Optional[torch.Tensor]
    granularity: str


def infer_granularity(candidate_feats: torch.Tensor, requested: str = "auto") -> str:
    if requested not in {"auto", "grid", "global"}:
        raise ValueError(f"Unsupported retrieval granularity: {requested}")
    if requested == "grid":
        if candidate_feats.ndim != 3 or int(candidate_feats.shape[1]) != 9:
            raise ValueError("grid retrieval requires candidate_feats with shape [B, 9, D].")
        return "grid"
    if requested == "global":
        if candidate_feats.ndim != 2:
            raise ValueError("global retrieval requires candidate_feats with shape [B, D].")
        return "global"
    if candidate_feats.ndim == 2:
        return "global"
    if candidate_feats.ndim == 3 and int(candidate_feats.shape[1]) == 9:
        return "grid"
    if candidate_feats.ndim == 3:
        raise ValueError(
            "auto retrieval expected 9 grid locations for [B, L, D] candidates; "
            f"got L={int(candidate_feats.shape[1])}."
        )
    raise ValueError(f"Unsupported candidate_feats shape: {tuple(candidate_feats.shape)}")


def prepare_candidate_targets(
    candidate_feats: torch.Tensor,
    batch: Dict[str, Any],
    requested_granularity: str = "auto",
) -> Tuple[torch.Tensor, torch.Tensor, str]:
    granularity = infer_granularity(candidate_feats, requested_granularity)
    if granularity == "grid":
        local_indices = batch["index"].to(candidate_feats.device).long()
        satellite_ids = batch.get("satellite_id")
        if isinstance(satellite_ids, torch.Tensor):
            satellite_ids = satellite_ids.to(candidate_feats.device)
        targets = build_retrieval_soft_targets(local_indices, satellite_ids)
        return candidate_feats.reshape(-1, candidate_feats.shape[-1]), targets, granularity

    labels = torch.arange(candidate_feats.shape[0], device=candidate_feats.device)
    return candidate_feats, labels, granularity


def _contrastive_loss(
    query_feats: torch.Tensor,
    candidate_feats: torch.Tensor,
    targets: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    query_feats = F.normalize(query_feats, p=2, dim=1)
    candidate_feats = F.normalize(candidate_feats, p=2, dim=1)
    logits = torch.matmul(query_feats, candidate_feats.T) / float(temperature)
    if targets.ndim == 2:
        log_probs = F.log_softmax(logits, dim=1)
        return -(targets.to(logits.dtype) * log_probs).sum(dim=1).mean()
    return F.cross_entropy(logits, targets.long())


def compute_retrieval_loss(
    query_feats: torch.Tensor,
    candidate_feats: torch.Tensor,
    batch: Dict[str, Any],
    temperature: float = 0.07,
    requested_granularity: str = "auto",
    text_feats: Optional[torch.Tensor] = None,
    use_text_loss: bool = False,
) -> RetrievalLossResult:
    flat_candidates, targets, granularity = prepare_candidate_targets(
        candidate_feats,
        batch,
        requested_granularity=requested_granularity,
    )
    image_loss = _contrastive_loss(query_feats, flat_candidates, targets, temperature)
    text_loss = None
    total = image_loss
    if use_text_loss and text_feats is not None:
        text_loss = _contrastive_loss(text_feats, flat_candidates, targets, temperature)
        total = total + text_loss
    return RetrievalLossResult(
        total=total,
        image_loss=image_loss,
        text_loss=text_loss,
        granularity=granularity,
    )
```

- [ ] **Step 4: Run loss tests**

Run: `pytest test_retrieval_losses.py test_retrieval_soft_targets.py -q`

Expected: PASS.

- [ ] **Step 5: Commit loss helpers**

```bash
git add retrieval/losses.py test_retrieval_losses.py
git commit -m "feat: add retrieval loss helpers"
```

### Task 3: Adapters And Registry

**Files:**
- Create: `retrieval/adapters.py`
- Create: `retrieval/registry.py`
- Create: `test_retrieval_registry.py`

- [ ] **Step 1: Write failing adapter/registry tests**

Add `test_retrieval_registry.py`:

```python
import torch
import torch.nn as nn

from retrieval.adapters import FeaturePayload, ForwardMethodAdapter
from retrieval.registry import MODEL_TYPES, get_model_entry


class TinyGridModel(nn.Module):
    def query_forward(self, pixel_values):
        return pixel_values.mean(dim=(2, 3))

    def ref_forward(self, pixel_values):
        pooled = pixel_values.mean(dim=(2, 3))
        return pooled.unsqueeze(1).expand(pooled.shape[0], 9, pooled.shape[1])

    def text_forward(self, input_ids, attention_mask=None):
        del attention_mask
        return input_ids.float()


def test_registry_exposes_five_retrieval_model_types() -> None:
    assert set(MODEL_TYPES) == {
        "clip",
        "siglip",
        "openclip",
        "evaclip",
        "sample_retrieval",
    }
    assert get_model_entry("sample4geo").canonical_type == "sample_retrieval"


def test_forward_method_adapter_returns_feature_payload() -> None:
    model = TinyGridModel()
    adapter = ForwardMethodAdapter(model, {"loss": {"use_text_loss": True}})
    batch = {
        "target_pixel_values": torch.ones(2, 3, 2, 2),
        "search_pixel_values": torch.ones(2, 3, 2, 2) * 2,
        "input_ids": torch.ones(2, 3, dtype=torch.long),
        "attention_mask": torch.ones(2, 3, dtype=torch.long),
    }

    payload = adapter.forward(batch, torch.device("cpu"))

    assert isinstance(payload, FeaturePayload)
    assert payload.query_feats.shape == (2, 3)
    assert payload.candidate_feats.shape == (2, 9, 3)
    assert payload.text_feats is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest test_retrieval_registry.py -q`

Expected: FAIL with missing adapter or registry modules.

- [ ] **Step 3: Implement adapters**

Create `retrieval/adapters.py`:

```python
from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch


@dataclass
class FeaturePayload:
    query_feats: torch.Tensor
    candidate_feats: torch.Tensor
    text_feats: Optional[torch.Tensor] = None


class BaseRetrievalAdapter:
    def __init__(self, model: torch.nn.Module, cfg: Dict[str, Any]):
        self.model = model
        self.cfg = cfg

    def forward(self, batch: Dict[str, Any], device: torch.device) -> FeaturePayload:
        raise NotImplementedError


class ForwardMethodAdapter(BaseRetrievalAdapter):
    def forward(self, batch: Dict[str, Any], device: torch.device) -> FeaturePayload:
        query_imgs = batch["target_pixel_values"].to(device, non_blocking=True)
        search_imgs = batch["search_pixel_values"].to(device, non_blocking=True)
        query_feats = self.model.query_forward(query_imgs)
        candidate_feats = self.model.ref_forward(search_imgs)

        text_feats = None
        if bool(self.cfg.get("loss", {}).get("use_text_loss", False)) and hasattr(self.model, "text_forward"):
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch.get("attention_mask")
            if isinstance(attention_mask, torch.Tensor):
                attention_mask = attention_mask.to(device, non_blocking=True)
            try:
                text_feats = self.model.text_forward(input_ids, attention_mask)
            except TypeError:
                text_feats = self.model.text_forward(input_ids)

        return FeaturePayload(
            query_feats=query_feats,
            candidate_feats=candidate_feats,
            text_feats=text_feats,
        )


class SampleRetrievalAdapter(BaseRetrievalAdapter):
    def forward(self, batch: Dict[str, Any], device: torch.device) -> FeaturePayload:
        query_imgs = batch["target_pixel_values"].to(device, non_blocking=True)
        search_imgs = batch["search_pixel_values"].to(device, non_blocking=True)
        if hasattr(self.model, "retrieval_forward"):
            query_feats, candidate_feats = self.model.retrieval_forward(query_imgs, search_imgs)
        else:
            query_feats, candidate_feats = self.model(query_imgs, search_imgs)
        return FeaturePayload(query_feats=query_feats, candidate_feats=candidate_feats)
```

- [ ] **Step 4: Implement registry with lazy imports**

Create `retrieval/registry.py`:

```python
from dataclasses import dataclass
from typing import Any, Callable, Dict, Tuple

import torch.nn as nn

from retrieval.adapters import BaseRetrievalAdapter, ForwardMethodAdapter, SampleRetrievalAdapter


@dataclass
class ModelEntry:
    canonical_type: str
    builder: Callable[[Dict[str, Any]], nn.Module]
    adapter_cls: type[BaseRetrievalAdapter]
    io_builder: Callable[[Dict[str, Any]], Tuple[Any, Any]]


def _build_clip(cfg: Dict[str, Any]) -> nn.Module:
    from retrieval.train_clip import Encoder

    model_cfg = cfg["model"]
    return Encoder(
        model_name=str(model_cfg["model_name"]),
        proj_dim=int(model_cfg.get("proj_dim", 768)),
    )


def _build_siglip(cfg: Dict[str, Any]) -> nn.Module:
    from retrieval.train_siglip import Encoder

    model_cfg = cfg["model"]
    return Encoder(
        model_name=str(model_cfg["model_name"]),
        proj_dim=int(model_cfg.get("proj_dim", 768)),
    )


def _build_openclip(cfg: Dict[str, Any]) -> nn.Module:
    from retrieval.train_openclip import Encoder

    model_cfg = cfg["model"]
    return Encoder(
        model_name=str(model_cfg["model_name"]),
        pretrained=str(model_cfg.get("pretrained") or ""),
        proj_dim=int(model_cfg.get("proj_dim", 768)),
    )


def _build_evaclip(cfg: Dict[str, Any]) -> nn.Module:
    from retrieval.train_evaclip import Encoder

    model_cfg = cfg["model"]
    return Encoder(
        model_name=str(model_cfg["model_name"]),
        proj_dim=int(model_cfg.get("proj_dim", 768)),
    )


def _build_sample_retrieval(cfg: Dict[str, Any]) -> nn.Module:
    from grounding.train_sample_retrieval import EMB_SIZE, SampleGeoLite

    model_cfg = cfg["model"]
    return SampleGeoLite(
        emb_size=int(model_cfg.get("emb_size", EMB_SIZE)),
        pretrained=bool(model_cfg.get("pretrained_backbone", True)),
    )


def _build_hf_io(cfg: Dict[str, Any]) -> Tuple[Any, Any]:
    from transformers import AutoImageProcessor, AutoTokenizer, CLIPProcessor

    model_cfg = cfg["model"]
    model_name = str(model_cfg["model_name"])
    cache_dir = str(model_cfg["cache_dir"])
    sat_size = cfg["data"]["sat_size"]
    if cfg["model"]["type"] == "clip":
        processor = CLIPProcessor.from_pretrained(model_name, cache_dir=cache_dir)
        processor_sat = CLIPProcessor.from_pretrained(model_name, cache_dir=cache_dir)
    else:
        processor = AutoImageProcessor.from_pretrained(model_name, cache_dir=cache_dir)
        processor_sat = AutoImageProcessor.from_pretrained(model_name, cache_dir=cache_dir)
    if hasattr(processor_sat, "image_processor"):
        processor_sat.image_processor.size = dict(sat_size)
    elif hasattr(processor_sat, "size"):
        processor_sat.size = dict(sat_size)
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    return (processor, processor_sat), tokenizer


def _build_openclip_io(cfg: Dict[str, Any]) -> Tuple[Any, Any]:
    import open_clip
    from retrieval.train_openclip import OpenClipImageProcessorWrapper, OpenClipTokenizerWrapper

    model_cfg = cfg["model"]
    _, _, preprocess = open_clip.create_model_and_transforms(
        str(model_cfg["model_name"]),
        pretrained=model_cfg.get("pretrained"),
        cache_dir=str(model_cfg["cache_dir"]),
    )
    tokenizer = open_clip.get_tokenizer(str(model_cfg["model_name"]))
    processor = OpenClipImageProcessorWrapper(preprocess)
    return (processor, processor), OpenClipTokenizerWrapper(tokenizer)


def _build_evaclip_io(cfg: Dict[str, Any]) -> Tuple[Any, Any]:
    import open_clip
    from retrieval.train_evaclip import OpenClipImageProcessorWrapper, OpenClipTokenizerWrapper

    model_cfg = cfg["model"]
    _, _, preprocess = open_clip.create_model_and_transforms(
        str(model_cfg["model_name"]),
        cache_dir=str(model_cfg["cache_dir"]),
    )
    tokenizer = open_clip.get_tokenizer(str(model_cfg["model_name"]))
    processor = OpenClipImageProcessorWrapper(preprocess)
    return (processor, processor), OpenClipTokenizerWrapper(tokenizer)


def _build_sample_io(cfg: Dict[str, Any]) -> Tuple[Any, Any]:
    from grounding.train_sample_retrieval import CVOGL_TRANSFORM, DummyTokenizer, IMG_SIZE, TransformProcessorWrapper

    sat_size = cfg["data"]["sat_size"]
    drone_size = cfg["data"]["drone_size"]
    processor = TransformProcessorWrapper(CVOGL_TRANSFORM, (int(drone_size["width"]), int(drone_size["height"])))
    processor_sat = TransformProcessorWrapper(CVOGL_TRANSFORM, (int(sat_size["width"]), int(sat_size["height"])))
    return (processor, processor_sat), DummyTokenizer()


REGISTRY: Dict[str, ModelEntry] = {
    "clip": ModelEntry("clip", _build_clip, ForwardMethodAdapter, _build_hf_io),
    "siglip": ModelEntry("siglip", _build_siglip, ForwardMethodAdapter, _build_hf_io),
    "openclip": ModelEntry("openclip", _build_openclip, ForwardMethodAdapter, _build_openclip_io),
    "evaclip": ModelEntry("evaclip", _build_evaclip, ForwardMethodAdapter, _build_evaclip_io),
    "sample_retrieval": ModelEntry("sample_retrieval", _build_sample_retrieval, SampleRetrievalAdapter, _build_sample_io),
    "sample4geo": ModelEntry("sample_retrieval", _build_sample_retrieval, SampleRetrievalAdapter, _build_sample_io),
}

MODEL_TYPES = ("clip", "siglip", "openclip", "evaclip", "sample_retrieval")


def get_model_entry(model_type: str) -> ModelEntry:
    return REGISTRY[str(model_type).lower()]


def build_model_and_adapter(cfg: Dict[str, Any]):
    entry = get_model_entry(cfg["model"]["type"])
    model = entry.builder(cfg)
    adapter = entry.adapter_cls(model, cfg)
    return model, adapter


def build_processors_and_tokenizer(cfg: Dict[str, Any]):
    entry = get_model_entry(cfg["model"]["type"])
    return entry.io_builder(cfg)
```

- [ ] **Step 5: Run registry tests**

Run: `pytest test_retrieval_registry.py -q`

Expected: PASS.

- [ ] **Step 6: Commit adapters and registry**

```bash
git add retrieval/adapters.py retrieval/registry.py test_retrieval_registry.py
git commit -m "feat: add retrieval adapters and registry"
```

### Task 4: Dataset Loader And Training Entry Point

**Files:**
- Create: `retrieval/train.py`
- Modify: `test_retrieval_registry.py`

- [ ] **Step 1: Add dry-run tests for train entry point**

Append to `test_retrieval_registry.py`:

```python
from pathlib import Path

from retrieval.config import load_config
from retrieval.train import _checkpoint_path, train


def test_train_dry_run_returns_save_dir(tmp_path: Path) -> None:
    cfg_path = tmp_path / "train.yaml"
    cfg_path.write_text(
        """
exp_name: dry
save_dir: /tmp/retrieval_dry
model:
  type: siglip
""",
        encoding="utf-8",
    )
    cfg = load_config(str(cfg_path))

    result = train(cfg, dry_run=True)

    assert result["status"] == "dry_run"
    assert result["save_dir"] == "/tmp/retrieval_dry"
    assert result["config_path"] == str(cfg_path)
    assert _checkpoint_path(cfg, "last.pth") == "/tmp/retrieval_dry/last.pth"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest test_retrieval_registry.py::test_train_dry_run_returns_save_dir -q`

Expected: FAIL with missing `retrieval.train`.

- [ ] **Step 3: Implement train entry point**

Create `retrieval/train.py`:

```python
import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset import ShiftedSatelliteDroneDataset
from retrieval.config import load_config
from retrieval.losses import compute_retrieval_loss
from retrieval.registry import build_model_and_adapter, build_processors_and_tokenizer


def _device_from_config(cfg: Dict[str, Any]) -> torch.device:
    requested = str(cfg["train"]["device"])
    if requested.startswith("cuda") and torch.cuda.is_available():
        return torch.device(requested)
    return torch.device("cpu")


def _checkpoint_path(cfg: Dict[str, Any], name: str) -> str:
    if os.path.isabs(str(name)):
        return str(name)
    return os.path.join(str(cfg["save_dir"]), str(name))


def _build_loader(cfg: Dict[str, Any], split: str) -> DataLoader:
    (processor, processor_sat), tokenizer = build_processors_and_tokenizer(cfg)
    data_cfg = cfg["data"]
    sat_size = data_cfg["sat_size"]
    dataset = ShiftedSatelliteDroneDataset(
        processor=processor,
        processor_sat=processor_sat,
        tokenizer=tokenizer,
        split=split,
        sat_target_size=(int(sat_size["height"]), int(sat_size["width"])),
        test_crop_ratio=float(data_cfg["test_crop_ratio"]),
        subset_heights=data_cfg.get("subset_heights"),
        subset_angles=data_cfg.get("subset_angles"),
    )
    num_workers = int(data_cfg["num_workers"])
    return DataLoader(
        dataset,
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=split == "train",
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=bool(cfg["train"].get("drop_last", split == "train")),
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,
    )


def _save_checkpoint(model: torch.nn.Module, save_dir: str, name: str) -> str:
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, name)
    torch.save(model.state_dict(), path)
    return path


def train(cfg: Dict[str, Any], dry_run: bool = False, max_steps: int = 0) -> Dict[str, Any]:
    if dry_run:
        os.makedirs(cfg["save_dir"], exist_ok=True)
        return {"status": "dry_run", "save_dir": cfg["save_dir"], "config_path": cfg["config_path"]}

    device = _device_from_config(cfg)
    os.makedirs(cfg["save_dir"], exist_ok=True)
    model, adapter = build_model_and_adapter(cfg)
    model.to(device)
    loader = _build_loader(cfg, "train")
    optimizer = AdamW(
        model.parameters(),
        lr=float(cfg["train"]["lr"]),
        weight_decay=float(cfg["train"]["weight_decay"]),
    )
    amp_enabled = bool(cfg["train"]["amp"]) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    grad_accumulation_steps = max(1, int(cfg["train"]["grad_accumulation_steps"]))
    grad_clip_norm = float(cfg["train"]["grad_clip_norm"])
    writer = SummaryWriter(os.path.join("runs", "retrieval", str(cfg["exp_name"])))
    best_loss = float("inf")
    global_step = 0

    for epoch in range(int(cfg["train"]["epochs"])):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        count = 0
        progress = tqdm(loader, desc=f"Epoch {epoch + 1}/{cfg['train']['epochs']}")
        for batch_idx, batch in enumerate(progress):
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                payload = adapter.forward(batch, device)
                losses = compute_retrieval_loss(
                    query_feats=payload.query_feats,
                    candidate_feats=payload.candidate_feats,
                    batch=batch,
                    temperature=float(cfg["loss"]["temperature"]),
                    requested_granularity=str(cfg["loss"]["granularity"]),
                    text_feats=payload.text_feats,
                    use_text_loss=bool(cfg["loss"]["use_text_loss"]),
                )
                loss_to_backward = losses.total / grad_accumulation_steps

            scaler.scale(loss_to_backward).backward()
            should_step = (batch_idx + 1) % grad_accumulation_steps == 0
            should_stop = max_steps > 0 and global_step + 1 >= max_steps
            if should_step or should_stop or batch_idx + 1 == len(loader):
                if grad_clip_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

            total_loss += float(losses.total.detach().cpu())
            count += 1
            avg_loss = total_loss / max(count, 1)
            progress.set_postfix({"loss": f"{avg_loss:.4f}", "mode": losses.granularity})
            writer.add_scalar("Loss/train_batch", float(losses.total.detach().cpu()), global_step)
            if should_stop:
                break

        avg_loss = total_loss / max(count, 1)
        writer.add_scalar("Loss/train_epoch", avg_loss, epoch)
        if avg_loss < best_loss:
            best_loss = avg_loss
            _save_checkpoint(model, cfg["save_dir"], "best.pth")
        if max_steps > 0 and global_step >= max_steps:
            break

    last_path = _save_checkpoint(model, cfg["save_dir"], "last.pth")
    writer.close()
    return {"status": "ok", "save_dir": cfg["save_dir"], "checkpoint": last_path, "best_loss": best_loss}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train retrieval-only models from YAML.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if args.device:
        cfg["train"]["device"] = args.device
    result = train(cfg, dry_run=args.dry_run, max_steps=args.max_steps)
    print(result)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run train dry-run test**

Run: `pytest test_retrieval_registry.py::test_train_dry_run_returns_save_dir -q`

Expected: PASS.

- [ ] **Step 5: Commit train entry point**

```bash
git add retrieval/train.py test_retrieval_registry.py
git commit -m "feat: add retrieval training entry point"
```

### Task 5: Evaluation Entry Point

**Files:**
- Create: `retrieval/eval.py`
- Create: `test_retrieval_eval.py`

- [ ] **Step 1: Write failing eval dry-run tests**

Add `test_retrieval_eval.py`:

```python
from pathlib import Path

from retrieval.config import load_config
from retrieval.eval import _checkpoint_path, evaluate


def test_eval_dry_run_resolves_relative_checkpoint(tmp_path: Path) -> None:
    cfg_path = tmp_path / "eval.yaml"
    cfg_path.write_text(
        """
exp_name: eval
save_dir: /tmp/retrieval_eval
model:
  type: siglip
eval:
  output_dir: eval_results/retrieval/eval
  checkpoint: last.pth
""",
        encoding="utf-8",
    )
    cfg = load_config(str(cfg_path))

    result = evaluate(cfg, dry_run=True)

    assert result["status"] == "dry_run"
    assert result["checkpoint"] == "/tmp/retrieval_eval/last.pth"
    assert _checkpoint_path(cfg) == "/tmp/retrieval_eval/last.pth"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest test_retrieval_eval.py -q`

Expected: FAIL with missing `retrieval.eval`.

- [ ] **Step 3: Implement eval entry point**

Create `retrieval/eval.py`:

```python
import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from retrieval.config import load_config
from retrieval.eval_retrieval import eval_recall1_per_subset


def _checkpoint_path(cfg: Dict[str, Any]) -> str:
    checkpoint = cfg["eval"]["checkpoint"]
    if os.path.isabs(str(checkpoint)):
        return str(checkpoint)
    return os.path.join(str(cfg["save_dir"]), str(checkpoint))


def _print_result(result: Dict[str, Any]) -> None:
    print(json.dumps(result, indent=2, sort_keys=True))


def _build_legacy_eval_args(cfg: Dict[str, Any], checkpoint: str) -> SimpleNamespace:
    data_cfg = cfg["data"]
    eval_cfg = cfg["eval"]
    device = str(cfg["train"]["device"])
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    return SimpleNamespace(
        model_type=str(cfg["model"]["type"]),
        checkpoint=checkpoint,
        device=device,
        batch_size=int(eval_cfg["batch_size"]),
        num_workers=int(data_cfg["num_workers"]),
        candidate_size=int(eval_cfg["candidate_size"]) if eval_cfg.get("candidate_size") is not None else None,
        include_file=eval_cfg.get("include_file"),
        output_dir=str(eval_cfg["output_dir"]),
        subset_heights=data_cfg.get("subset_heights"),
        subset_angles=data_cfg.get("subset_angles"),
        test_crop_ratio=float(data_cfg["test_crop_ratio"]),
        save_query_records=bool(eval_cfg.get("save_query_records", False)),
    )


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
    args = _build_legacy_eval_args(cfg, checkpoint)
    result = eval_recall1_per_subset(
        model_type=args.model_type,
        checkpoint_path=args.checkpoint,
        subset_heights=args.subset_heights,
        subset_angles=args.subset_angles,
        candidate_size=args.candidate_size,
        include_file=args.include_file,
        output_dir=args.output_dir,
        device=args.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    output_dir = Path(cfg["eval"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
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
```

- [ ] **Step 4: Run eval dry-run tests**

Run: `pytest test_retrieval_eval.py -q`

Expected: PASS. If `retrieval.eval_retrieval.eval_recall1_per_subset` has a different callable signature, update only `_build_legacy_eval_args` and the `evaluate` call to match the existing function.

- [ ] **Step 5: Commit eval entry point**

```bash
git add retrieval/eval.py test_retrieval_eval.py
git commit -m "feat: add retrieval eval entry point"
```

### Task 6: YAML Configs

**Files:**
- Create: `configs/retrieval/siglip.yaml`
- Create: `configs/retrieval/clip.yaml`
- Create: `configs/retrieval/openclip.yaml`
- Create: `configs/retrieval/evaclip.yaml`
- Create: `configs/retrieval/sample_retrieval.yaml`
- Modify: `test_retrieval_config.py`

- [ ] **Step 1: Add test that all checked-in configs load**

Append to `test_retrieval_config.py`:

```python
import pytest


@pytest.mark.parametrize(
    "config_path,model_type",
    [
        ("configs/retrieval/siglip.yaml", "siglip"),
        ("configs/retrieval/clip.yaml", "clip"),
        ("configs/retrieval/openclip.yaml", "openclip"),
        ("configs/retrieval/evaclip.yaml", "evaclip"),
        ("configs/retrieval/sample_retrieval.yaml", "sample_retrieval"),
    ],
)
def test_checked_in_retrieval_configs_load(config_path: str, model_type: str) -> None:
    cfg = load_config(config_path)

    assert cfg["model"]["type"] == model_type
    assert cfg["loss"]["use_text_loss"] is False
    assert cfg["loss"]["granularity"] == "auto"
    assert cfg["eval"]["output_dir"].startswith("eval_results/retrieval/")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest test_retrieval_config.py::test_checked_in_retrieval_configs_load -q`

Expected: FAIL because `configs/retrieval/*.yaml` do not exist.

- [ ] **Step 3: Create model configs**

Create `configs/retrieval/siglip.yaml`:

```yaml
exp_name: retrieval_siglip
save_dir: /media/data1/feihong/ckpt/retrieval_siglip

model:
  type: siglip
  model_name: google/siglip-base-patch16-224
  proj_dim: 768

train:
  batch_size: 8
  lr: 1.0e-5

eval:
  output_dir: eval_results/retrieval/siglip
```

Create `configs/retrieval/clip.yaml`:

```yaml
exp_name: retrieval_clip
save_dir: /media/data1/feihong/ckpt/retrieval_clip

model:
  type: clip
  model_name: openai/clip-vit-base-patch16
  proj_dim: 768

train:
  batch_size: 8
  lr: 1.0e-5

eval:
  output_dir: eval_results/retrieval/clip
```

Create `configs/retrieval/openclip.yaml`:

```yaml
exp_name: retrieval_openclip
save_dir: /media/data1/feihong/ckpt/retrieval_openclip

model:
  type: openclip
  model_name: ViT-B-16
  pretrained: laion2b_s34b_b88k
  proj_dim: 768

train:
  batch_size: 8
  lr: 1.0e-5

eval:
  output_dir: eval_results/retrieval/openclip
```

Create `configs/retrieval/evaclip.yaml`:

```yaml
exp_name: retrieval_evaclip
save_dir: /media/data1/feihong/ckpt/retrieval_evaclip

model:
  type: evaclip
  model_name: EVA02-B-16
  proj_dim: 768

train:
  batch_size: 8
  lr: 1.0e-5

eval:
  output_dir: eval_results/retrieval/evaclip
```

Create `configs/retrieval/sample_retrieval.yaml`:

```yaml
exp_name: retrieval_sample4geo
save_dir: /media/data1/feihong/ckpt/retrieval_sample4geo

model:
  type: sample_retrieval
  proj_dim: 1024
  emb_size: 1024
  pretrained_backbone: true

train:
  batch_size: 8
  lr: 1.0e-4
  weight_decay: 1.0e-4

eval:
  output_dir: eval_results/retrieval/sample_retrieval
```

- [ ] **Step 4: Run config tests**

Run: `pytest test_retrieval_config.py -q`

Expected: PASS.

- [ ] **Step 5: Commit configs**

```bash
git add configs/retrieval test_retrieval_config.py
git commit -m "feat: add retrieval model configs"
```

### Task 7: Group Script

**Files:**
- Create: `train_retrieval_grounp.sh`
- Create: `test_train_retrieval_group.py`

- [ ] **Step 1: Write failing group script tests**

Add `test_train_retrieval_group.py`:

```python
from pathlib import Path
import unittest


SCRIPT = Path(__file__).resolve().parent / "train_retrieval_grounp.sh"


class TrainRetrievalGroupScriptTest(unittest.TestCase):
    def test_group_script_runs_all_retrieval_configs(self):
        source = SCRIPT.read_text(encoding="utf-8")

        self.assertIn("--gpus", source)
        self.assertIn("configs/retrieval/siglip.yaml", source)
        self.assertIn("configs/retrieval/clip.yaml", source)
        self.assertIn("configs/retrieval/openclip.yaml", source)
        self.assertIn("configs/retrieval/evaclip.yaml", source)
        self.assertIn("configs/retrieval/sample_retrieval.yaml", source)
        self.assertIn("retrieval/train.py", source)
        self.assertIn("retrieval/eval.py", source)
        self.assertIn("eval_results/retrieval", source)
        self.assertIn("--device cuda:0", source)

    def test_train_and_eval_support_device_override(self):
        train_source = (SCRIPT.parent / "retrieval" / "train.py").read_text(encoding="utf-8")
        eval_source = (SCRIPT.parent / "retrieval" / "eval.py").read_text(encoding="utf-8")

        self.assertIn('parser.add_argument("--device"', train_source)
        self.assertIn('cfg["train"]["device"] = args.device', train_source)
        self.assertIn('parser.add_argument("--device"', eval_source)
        self.assertIn('cfg["train"]["device"] = args.device', eval_source)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest test_train_retrieval_group.py -q`

Expected: FAIL because `train_retrieval_grounp.sh` does not exist.

- [ ] **Step 3: Create group script**

Create `train_retrieval_grounp.sh`:

```bash
#!/bin/bash
set -uo pipefail

cd /media/data1/feihong/univerisity_dev

DRY_RUN=0
GPUS_CSV="${CUDA_VISIBLE_DEVICES:-0}"
EXTRA_ARGS=()

while [ "$#" -gt 0 ]; do
    case "$1" in
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        --gpus)
            if [ "$#" -lt 2 ]; then
                echo "Missing value for --gpus, e.g. --gpus 0,1,2" >&2
                exit 2
            fi
            GPUS_CSV="$2"
            shift 2
            ;;
        --gpus=*)
            GPUS_CSV="${1#--gpus=}"
            shift
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

IFS=',' read -r -a GPUS <<< "$GPUS_CSV"
if [ "${#GPUS[@]}" -eq 0 ] || [ -z "${GPUS[0]}" ]; then
    echo "No GPUs provided. Use --gpus 0,1,2 or set CUDA_VISIBLE_DEVICES." >&2
    exit 2
fi
FIRST_GPU="${GPUS[0]}"

CONFIGS=(
    "configs/retrieval/siglip.yaml"
    "configs/retrieval/clip.yaml"
    "configs/retrieval/openclip.yaml"
    "configs/retrieval/evaclip.yaml"
    "configs/retrieval/sample_retrieval.yaml"
)

SUMMARY_DIR="eval_results/retrieval"
SUMMARY_PATH="${SUMMARY_DIR}/group_summary.jsonl"
FINAL_JSON="${SUMMARY_DIR}/group_summary.json"
mkdir -p "$SUMMARY_DIR"
: > "$SUMMARY_PATH"

echo "Running ${#CONFIGS[@]} retrieval configs sequentially on GPU(s): ${GPUS_CSV}"

for CONFIG_INDEX in "${!CONFIGS[@]}"; do
    CONFIG_PATH="${CONFIGS[$CONFIG_INDEX]}"
    echo "============================================================"
    echo "Running retrieval config: ${CONFIG_PATH}"
    echo "Visible physical GPUs: ${GPUS_CSV}"
    echo "Started at: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "============================================================"

    TRAIN_STATUS="ok"
    EVAL_STATUS="ok"
    TRAIN_ARGS=("${EXTRA_ARGS[@]}")
    EVAL_ARGS=("${EXTRA_ARGS[@]}")
    if [ "$DRY_RUN" -eq 1 ]; then
        TRAIN_ARGS+=("--dry-run")
        EVAL_ARGS+=("--dry-run")
    fi

    CUDA_VISIBLE_DEVICES="$GPUS_CSV" python retrieval/train.py \
        --config "$CONFIG_PATH" \
        --device cuda:0 \
        "${TRAIN_ARGS[@]}"
    if [ "$?" -ne 0 ]; then
        TRAIN_STATUS="failed"
        EVAL_STATUS="skipped"
        printf '{"config_index":%s,"config":"%s","gpus":"%s","train_status":"%s","eval_status":"%s"}\n' \
            "$CONFIG_INDEX" "$CONFIG_PATH" "$GPUS_CSV" "$TRAIN_STATUS" "$EVAL_STATUS" >> "$SUMMARY_PATH"
        echo "Training failed; skip eval for this config: ${CONFIG_PATH}"
        continue
    fi

    CUDA_VISIBLE_DEVICES="$FIRST_GPU" python retrieval/eval.py \
        --config "$CONFIG_PATH" \
        --device cuda:0 \
        "${EVAL_ARGS[@]}"
    if [ "$?" -ne 0 ]; then
        EVAL_STATUS="failed"
    fi

    printf '{"config_index":%s,"config":"%s","gpus":"%s","train_status":"%s","eval_status":"%s"}\n' \
        "$CONFIG_INDEX" "$CONFIG_PATH" "$GPUS_CSV" "$TRAIN_STATUS" "$EVAL_STATUS" >> "$SUMMARY_PATH"
done

python - "$SUMMARY_PATH" "$FINAL_JSON" <<'PY'
import json
import sys
from pathlib import Path

from retrieval.config import load_config

summary_path = Path(sys.argv[1])
final_json = Path(sys.argv[2])
metric_keys = ["checkpoint", "num_samples", "num_gallery", "overall", "retrieval"]

items = []
for line in summary_path.read_text(encoding="utf-8").splitlines():
    if not line.strip():
        continue
    item = json.loads(line)
    cfg = load_config(item["config"])
    metrics_path = Path(cfg["eval"]["output_dir"]) / "metrics.json"
    metrics = {}
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    for key in metric_keys:
        item[key] = metrics.get(key)
    items.append(item)

items.sort(key=lambda item: int(item.get("config_index", 0)))
final_json.write_text(json.dumps(items, indent=2, sort_keys=True), encoding="utf-8")
print(f"Wrote {final_json}")
PY
```

- [ ] **Step 4: Make script executable and run static tests**

Run:

```bash
chmod +x train_retrieval_grounp.sh
pytest test_train_retrieval_group.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit group script**

```bash
git add train_retrieval_grounp.sh test_train_retrieval_group.py
git commit -m "feat: add retrieval group runner"
```

### Task 8: Smoke Verification And Integration Fixes

**Files:**
- Modify only files created in Tasks 1-7 unless a concrete import or signature mismatch requires a small compatibility edit.

- [ ] **Step 1: Run focused unit tests**

Run:

```bash
pytest \
  test_retrieval_config.py \
  test_retrieval_losses.py \
  test_retrieval_registry.py \
  test_retrieval_eval.py \
  test_train_retrieval_group.py \
  test_retrieval_soft_targets.py \
  -q
```

Expected: PASS.

- [ ] **Step 2: Run retrieval dry runs**

Run:

```bash
python retrieval/train.py --config configs/retrieval/siglip.yaml --dry-run --device cpu
python retrieval/eval.py --config configs/retrieval/siglip.yaml --dry-run --device cpu
./train_retrieval_grounp.sh --dry-run --gpus 0
```

Expected:

- Train dry run prints `status: dry_run`.
- Eval dry run prints JSON with `status` equal to `dry_run`.
- Group dry run writes `eval_results/retrieval/group_summary.jsonl` and `eval_results/retrieval/group_summary.json`.

- [ ] **Step 3: Fix integration mismatches with minimal patches**

If dry run imports fail because `retrieval/eval_retrieval.py` exposes a different function signature, patch `retrieval/eval.py` by adapting `_build_legacy_eval_args` and the call site. Keep the public CLI unchanged:

```python
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

    args = _build_legacy_eval_args(cfg, checkpoint)
    result = eval_recall1_per_subset(
        model_type=args.model_type,
        checkpoint_path=args.checkpoint,
        subset_heights=args.subset_heights,
        subset_angles=args.subset_angles,
        candidate_size=args.candidate_size,
        include_file=args.include_file,
        output_dir=args.output_dir,
        device=args.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    output_dir = Path(cfg["eval"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    _print_result(result)
    return result
```

If `CLIPProcessor` or wrapper processors do not accept the exact `ShiftedSatelliteDroneDataset` call shape, patch only the relevant registry IO builder so every processor supports `processor(images=image, return_tensors="pt")`.

- [ ] **Step 4: Run one tiny training smoke if local dependencies are available**

Run:

```bash
python retrieval/train.py \
  --config configs/retrieval/siglip.yaml \
  --device cuda:0 \
  --max-steps 1
```

Expected: completes one optimizer step and writes `/media/data1/feihong/ckpt/retrieval_siglip/last.pth`. If CUDA, cached model weights, or dataset paths are unavailable, record the exact missing dependency and keep the dry-run verification as the completed local check.

- [ ] **Step 5: Run final status and commit integration fixes**

Run:

```bash
git status --short
```

Expected: only intended retrieval files, configs, group script, and tests are modified or untracked.

Commit if Task 8 made changes:

```bash
git add retrieval configs/retrieval train_retrieval_grounp.sh test_retrieval_*.py test_train_retrieval_group.py
git commit -m "test: verify retrieval config workflow"
```

### Task 9: Final Review

**Files:**
- Read-only review of changed files.

- [ ] **Step 1: Inspect final diff**

Run:

```bash
git diff HEAD~8..HEAD --stat
git diff HEAD~8..HEAD -- retrieval configs/retrieval train_retrieval_grounp.sh
```

Expected: changes match this plan and do not include unrelated edits to user-modified files.

- [ ] **Step 2: Run final test set**

Run:

```bash
pytest \
  test_retrieval_config.py \
  test_retrieval_losses.py \
  test_retrieval_registry.py \
  test_retrieval_eval.py \
  test_train_retrieval_group.py \
  test_retrieval_soft_targets.py \
  -q
```

Expected: PASS.

- [ ] **Step 3: Summarize result**

Report:

- New config-driven retrieval entry points.
- Five checked-in retrieval configs.
- `train_retrieval_grounp.sh` behavior and dry-run status.
- Tests run and any environment-limited checks that could not run.

