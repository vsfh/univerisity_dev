# Grounding Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a YAML-driven, grounding-only `grounding/` train/eval stack with shared bbox/heatmap loss, shared metrics, model adapters, legacy archiving, and a resilient group runner.

**Architecture:** Keep the new stack thin: `grounding/config.py` loads YAML, `grounding/registry.py` builds model+adapter pairs, `grounding/adapters.py` normalizes outputs, `grounding/losses.py` centralizes bbox/heatmap loss and decode helpers, and `grounding/train.py` / `grounding/eval.py` orchestrate loops. Old model code stays in place until each adapter is proven, then legacy scripts are moved without changing checkpoint formats.

**Tech Stack:** Python, PyYAML, PyTorch, Transformers `AutoImageProcessor`/`AutoTokenizer`, `dataset.ShiftedSatelliteDroneDataset`, `bbox.yolo_utils`, existing `model.py` and `model_ground.py`.

---

## File Structure

Create or modify these files:

- Create: `grounding/config.py`  
  Loads grounding YAML cards and exposes a simple dict-like config.

- Create: `grounding/losses.py`  
  Provides `build_geo_features`, `add_heatmap_to_confidence`, `heatmap_loss_fn`, `compute_grounding_loss`, `decode_anchor_prediction`, and metric helpers. Imports bbox helpers from `bbox.yolo_utils`.

- Create: `grounding/adapters.py`  
  Defines `GroundingOutput`, adapter base class, SigLIP2/unified adapters, and lightweight wrappers for old model outputs.

- Create: `grounding/registry.py`  
  Maps `model.type` to model builder and adapter class.

- Create: `grounding/train.py`  
  YAML-driven grounding-only training loop with AMP, grad accumulation, grad clipping, checkpoint save, `--dry-run`, and `--max-steps`.

- Create: `grounding/eval.py`  
  YAML-driven evaluation loop with shared metrics, JSON output, `--dry-run`, and `--max-batches`.

- Create: `configs/grounding/*.yaml`  
  One YAML card per model type.

- Create: `train_ground_group.sh`  
  Runs train/eval for all grounding YAML cards, continues after failures, and writes `eval_results/grounding/group_summary.json`.

- Move after adapters are working: selected old files into `grounding/legacy/`.

- Temporary tests during implementation only: `tmp_test_grounding_*.py`  
  These must be deleted before each task commit.

---

### Task 1: Config Loader And YAML Cards

**Files:**
- Create: `grounding/config.py`
- Create: `configs/grounding/siglip2_heat.yaml`
- Create: `configs/grounding/siglip2_test.yaml`
- Create: `configs/grounding/siglip_ground.yaml`
- Create: `configs/grounding/lpn.yaml`
- Create: `configs/grounding/sample4geo.yaml`
- Create: `configs/grounding/smgeo.yaml`
- Create: `configs/grounding/ocg.yaml`
- Create: `configs/grounding/trogeolite.yaml`
- Create: `configs/grounding/det.yaml`
- Temporary test: `tmp_test_grounding_config.py`

- [ ] **Step 1: Write the failing temporary config test**

Create `tmp_test_grounding_config.py`:

```python
import unittest
from pathlib import Path

from grounding.config import load_config


class GroundingConfigTest(unittest.TestCase):
    def test_loads_siglip2_heat_defaults(self):
        cfg = load_config("configs/grounding/siglip2_heat.yaml")

        self.assertEqual("siglip2_heat", cfg["model"]["type"])
        self.assertEqual(32, cfg["train"]["batch_size"])
        self.assertEqual(2, cfg["train"]["grad_accumulation_steps"])
        self.assertEqual(1.0, cfg["train"]["grad_clip_norm"])
        self.assertEqual(0.2, cfg["loss"]["heatmap_weight"])
        self.assertEqual("last.pth", cfg["eval"]["checkpoint"])

    def test_all_grounding_cards_exist(self):
        expected = {
            "siglip2_heat.yaml",
            "siglip2_test.yaml",
            "siglip_ground.yaml",
            "lpn.yaml",
            "sample4geo.yaml",
            "smgeo.yaml",
            "ocg.yaml",
            "trogeolite.yaml",
            "det.yaml",
        }
        actual = {path.name for path in Path("configs/grounding").glob("*.yaml")}
        self.assertEqual(expected, actual)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the test and verify it fails**

Run:

```bash
python tmp_test_grounding_config.py
```

Expected: FAIL or import error because `grounding.config` and YAML cards do not exist yet.

- [ ] **Step 3: Create `grounding/config.py`**

Create `grounding/config.py` with:

```python
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
```

- [ ] **Step 4: Create the nine YAML cards**

Use this shared shape, changing `exp_name`, `save_dir`, `model.type`, and model booleans.

`configs/grounding/siglip2_heat.yaml`:

```yaml
exp_name: siglip2_heat
save_dir: /media/data1/feihong/ckpt/ground_siglip2_heat

model:
  type: siglip2_heat
  use_angle: true
  use_heatmap: true
  use_text: false

eval:
  output_dir: eval_results/grounding/siglip2_heat
```

`configs/grounding/siglip2_test.yaml`:

```yaml
exp_name: siglip2_test
save_dir: /media/data1/feihong/ckpt/ground_siglip2_test

model:
  type: siglip2_test
  use_angle: true
  use_heatmap: true
  use_text: false

eval:
  output_dir: eval_results/grounding/siglip2_test
```

`configs/grounding/siglip_ground.yaml`:

```yaml
exp_name: siglip_ground
save_dir: /media/data1/feihong/ckpt/ground_siglip

model:
  type: siglip_ground
  use_angle: true
  use_heatmap: true
  use_text: false

eval:
  output_dir: eval_results/grounding/siglip_ground
```

`configs/grounding/lpn.yaml`:

```yaml
exp_name: lpn
save_dir: /media/data1/feihong/ckpt/ground_lpn

model:
  type: lpn
  use_angle: true
  use_heatmap: false
  use_text: false

eval:
  output_dir: eval_results/grounding/lpn
```

`configs/grounding/sample4geo.yaml`:

```yaml
exp_name: sample4geo
save_dir: /media/data1/feihong/ckpt/ground_sample

model:
  type: sample4geo
  use_angle: true
  use_heatmap: false
  use_text: false

eval:
  output_dir: eval_results/grounding/sample4geo
```

`configs/grounding/smgeo.yaml`:

```yaml
exp_name: smgeo
save_dir: /media/data1/feihong/ckpt/ground_sm

model:
  type: smgeo
  use_angle: true
  use_heatmap: false
  use_text: false

eval:
  output_dir: eval_results/grounding/smgeo
```

`configs/grounding/ocg.yaml`:

```yaml
exp_name: ocg
save_dir: /media/data1/feihong/ckpt/ground_ocg

model:
  type: ocg
  use_angle: true
  use_heatmap: false
  use_text: false

eval:
  output_dir: eval_results/grounding/ocg
```

`configs/grounding/trogeolite.yaml`:

```yaml
exp_name: trogeolite
save_dir: /media/data1/feihong/ckpt/ground_cvos

model:
  type: trogeolite
  use_angle: true
  use_heatmap: false
  use_text: false

eval:
  output_dir: eval_results/grounding/trogeolite
```

`configs/grounding/det.yaml`:

```yaml
exp_name: det
save_dir: /media/data1/feihong/ckpt/ground_det

model:
  type: det
  use_angle: true
  use_heatmap: false
  use_text: false

eval:
  output_dir: eval_results/grounding/det
```

- [ ] **Step 5: Run the config test and verify it passes**

Run:

```bash
python tmp_test_grounding_config.py
```

Expected: `Ran 2 tests` and `OK`.

- [ ] **Step 6: Delete the temporary config test**

Run:

```bash
rm tmp_test_grounding_config.py
```

- [ ] **Step 7: Compile the new config module**

Run:

```bash
python -m py_compile grounding/config.py
```

Expected: exit code 0.

- [ ] **Step 8: Commit config work**

Run:

```bash
git add grounding/config.py configs/grounding
git commit -m "Add grounding config cards"
```

---

### Task 2: Shared Grounding Losses And Metrics

**Files:**
- Create: `grounding/losses.py`
- Temporary test: `tmp_test_grounding_losses.py`

- [ ] **Step 1: Write the failing temporary losses test**

Create `tmp_test_grounding_losses.py`:

```python
import unittest

import torch

from grounding.losses import (
    GroundingLoss,
    build_geo_features,
    center_distance,
    compute_iou_metrics,
    decode_anchor_prediction,
)
from bbox.yolo_utils import get_tensor_anchors


class GroundingLossesTest(unittest.TestCase):
    def test_build_geo_features_uses_angle_height(self):
        batch = {
            "angle": torch.tensor([90.0, 180.0]),
            "height": torch.tensor([150.0, 200.0]),
        }
        geo = build_geo_features(batch, torch.device("cpu"))
        self.assertEqual((2, 2), tuple(geo.shape))
        self.assertAlmostEqual(0.25, float(geo[0, 0]))
        self.assertAlmostEqual(0.5, float(geo[1, 0]))

    def test_metrics_have_expected_fields(self):
        pred = torch.tensor([[0.0, 0.0, 10.0, 10.0], [0.0, 0.0, 5.0, 5.0]])
        target = torch.tensor([[0.0, 0.0, 10.0, 10.0], [10.0, 10.0, 20.0, 20.0]])
        metrics = compute_iou_metrics(pred, target)
        self.assertEqual({"mean_iou", "iou_at_0_5", "iou_at_0_25", "mean_center_distance"}, set(metrics))
        self.assertAlmostEqual(0.5, metrics["mean_iou"])
        self.assertAlmostEqual(0.5, metrics["iou_at_0_5"])

    def test_decode_anchor_prediction_returns_xyxy(self):
        anchors = get_tensor_anchors("cpu")
        pred_anchor = torch.zeros(1, 9, 5, 4, 4)
        pred_anchor[:, :, 4, :, :] = -10.0
        pred_anchor[0, 0, 4, 1, 2] = 10.0
        pred_anchor[0, 0, 2, 1, 2] = -2.0
        pred_anchor[0, 0, 3, 1, 2] = -2.0
        decoded = decode_anchor_prediction(pred_anchor, anchors, (768, 432))
        self.assertEqual((1, 4), tuple(decoded.shape))
        self.assertLess(decoded[0, 0], decoded[0, 2])
        self.assertLess(decoded[0, 1], decoded[0, 3])

    def test_grounding_loss_dataclass_fields(self):
        loss = GroundingLoss(
            total=torch.tensor(1.0),
            bbox=torch.tensor(0.8),
            geo=torch.tensor(0.3),
            cls=torch.tensor(0.5),
            heatmap=torch.tensor(0.2),
        )
        self.assertAlmostEqual(1.0, float(loss.total))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the test and verify it fails**

Run:

```bash
python tmp_test_grounding_losses.py
```

Expected: FAIL or import error because `grounding.losses` does not exist.

- [ ] **Step 3: Create `grounding/losses.py`**

Create `grounding/losses.py` with these public functions and dataclasses:

```python
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from bbox.yolo_utils import bbox_iou, build_target, eval_iou_acc, xywh2xyxy, yolo_loss


@dataclass
class GroundingLoss:
    total: torch.Tensor
    bbox: torch.Tensor
    geo: torch.Tensor
    cls: torch.Tensor
    heatmap: torch.Tensor


def build_geo_features(batch: Dict[str, Any], device: torch.device) -> Optional[torch.Tensor]:
    angle = batch.get("angle")
    height = batch.get("height")
    if angle is None or height is None:
        return None
    angle = angle.to(device=device, dtype=torch.float32).view(-1) / 360.0
    height = height.to(device=device, dtype=torch.float32).view(-1) / 300.0
    return torch.stack([angle, height], dim=1)


def build_heatmap_target(
    target_bbox: torch.Tensor,
    heatmap_hw: Tuple[int, int],
    image_wh: Tuple[int, int],
    edge_value: float = 0.2,
    log_scale: float = 9.0,
) -> torch.Tensor:
    grid_h, grid_w = int(heatmap_hw[0]), int(heatmap_hw[1])
    image_w, image_h = float(image_wh[0]), float(image_wh[1])
    device = target_bbox.device
    dtype = target_bbox.dtype

    x1 = torch.minimum(target_bbox[:, 0], target_bbox[:, 2]).clamp(0.0, image_w)
    y1 = torch.minimum(target_bbox[:, 1], target_bbox[:, 3]).clamp(0.0, image_h)
    x2 = torch.maximum(target_bbox[:, 0], target_bbox[:, 2]).clamp(0.0, image_w)
    y2 = torch.maximum(target_bbox[:, 1], target_bbox[:, 3]).clamp(0.0, image_h)

    center_x = ((x1 + x2) * 0.5 / max(image_w, 1.0) * grid_w).clamp(0.0, grid_w - 1e-6)
    center_y = ((y1 + y2) * 0.5 / max(image_h, 1.0) * grid_h).clamp(0.0, grid_h - 1e-6)

    ys = torch.arange(grid_h, device=device, dtype=dtype).view(1, grid_h, 1)
    xs = torch.arange(grid_w, device=device, dtype=dtype).view(1, 1, grid_w)
    x_img = (xs + 0.5) / max(float(grid_w), 1.0) * image_w
    y_img = (ys + 0.5) / max(float(grid_h), 1.0) * image_h

    inside_x = (x_img >= x1.view(-1, 1, 1)) & (x_img <= x2.view(-1, 1, 1))
    inside_y = (y_img >= y1.view(-1, 1, 1)) & (y_img <= y2.view(-1, 1, 1))
    inside = inside_x & inside_y

    center_img_x = (x1 + x2).view(-1, 1, 1) * 0.5
    center_img_y = (y1 + y2).view(-1, 1, 1) * 0.5
    half_w = ((x2 - x1) * 0.5).view(-1, 1, 1).clamp_min(1e-6)
    half_h = ((y2 - y1) * 0.5).view(-1, 1, 1).clamp_min(1e-6)

    norm_radius = torch.maximum(
        (x_img - center_img_x).abs() / half_w,
        (y_img - center_img_y).abs() / half_h,
    ).clamp(0.0, 1.0)
    decay = torch.log1p(norm_radius * float(log_scale)) / np.log1p(float(log_scale))
    target = 1.0 - (1.0 - float(edge_value)) * decay
    target = target * inside.to(dtype=dtype)

    center_x_idx = torch.floor(center_x).long().clamp(0, grid_w - 1)
    center_y_idx = torch.floor(center_y).long().clamp(0, grid_h - 1)
    batch_idx = torch.arange(target.shape[0], device=device)
    target[batch_idx, center_y_idx, center_x_idx] = 1.0
    return target.unsqueeze(1).clamp(0.0, 1.0)


def heatmap_loss_fn(
    heatmap: torch.Tensor,
    target_bbox: torch.Tensor,
    image_wh: Tuple[int, int],
    cfg: Dict[str, Any],
) -> torch.Tensor:
    heatmap_target = build_heatmap_target(
        target_bbox=target_bbox,
        heatmap_hw=heatmap.shape[-2:],
        image_wh=image_wh,
        edge_value=float(cfg["loss"]["heatmap_bbox_center_edge_value"]),
        log_scale=float(cfg["loss"]["heatmap_bbox_center_log_scale"]),
    )
    pred = heatmap.float().flatten(1)
    target = heatmap_target.float().flatten(1)
    target_prob = target / target.sum(dim=1, keepdim=True).clamp_min(1e-6)
    pred_prob = pred / pred.sum(dim=1, keepdim=True).clamp_min(1e-6)

    loss = heatmap.new_zeros(())
    loss_types = cfg["loss"]["heatmap_loss_type"]
    if "mse" in loss_types:
        loss = loss + F.mse_loss(pred_prob, target_prob, reduction="mean") * pred_prob.shape[1]
    if "cross_entropy" in loss_types:
        loss = loss + -(target_prob * pred_prob.clamp_min(1e-8).log()).sum(dim=1).mean()
    return loss


def add_heatmap_to_confidence(
    pred_anchor: torch.Tensor,
    heatmap: Optional[torch.Tensor],
    confidence_weight: float,
) -> torch.Tensor:
    if heatmap is None or confidence_weight <= 0:
        return pred_anchor
    if heatmap.shape[-2:] != pred_anchor.shape[-2:]:
        heatmap = F.interpolate(heatmap, size=pred_anchor.shape[-2:], mode="bilinear", align_corners=False)
    heat_confidence = float(confidence_weight) * heatmap.detach().to(dtype=pred_anchor.dtype).unsqueeze(1)
    return torch.cat([pred_anchor[:, :, :4, :, :], pred_anchor[:, :, 4:5, :, :] + heat_confidence], dim=2)


def compute_grounding_loss(output: Any, batch: Dict[str, Any], anchors_full: torch.Tensor, cfg: Dict[str, Any]) -> GroundingLoss:
    target_bbox = batch["bbox"].to(output.device)
    image_wh = output.image_wh
    pred_anchor = output.pred_anchor

    if pred_anchor is None:
        pred_bbox = output.pred_bbox
        iou = bbox_iou(pred_bbox, target_bbox, x1y1x2y2=True)
        bbox_loss = F.l1_loss(pred_bbox, target_bbox) + (1.0 - iou).mean()
        zero = bbox_loss.new_zeros(())
        total = float(cfg["loss"]["bbox_weight"]) * bbox_loss
        return GroundingLoss(total=total, bbox=bbox_loss, geo=bbox_loss, cls=zero, heatmap=zero)

    if pred_anchor.ndim == 4:
        pred_anchor = pred_anchor.view(pred_anchor.shape[0], 9, 5, pred_anchor.shape[2], pred_anchor.shape[3])
    pred_anchor = add_heatmap_to_confidence(
        pred_anchor,
        output.heatmap,
        float(cfg["loss"]["heatmap_confidence_weight"]),
    )
    grid_wh = (pred_anchor.shape[4], pred_anchor.shape[3])
    new_gt_bbox, best_anchor_gi_gj = build_target(target_bbox, anchors_full, image_wh, grid_wh)
    loss_geo, loss_cls = yolo_loss(pred_anchor, new_gt_bbox, anchors_full, best_anchor_gi_gj, image_wh)
    bbox_loss = loss_geo + loss_cls
    heatmap_loss = pred_anchor.new_zeros(())
    if output.heatmap is not None and cfg["model"]["use_heatmap"]:
        heatmap_loss = heatmap_loss_fn(output.heatmap, target_bbox, image_wh, cfg)
    total = float(cfg["loss"]["bbox_weight"]) * bbox_loss + float(cfg["loss"]["heatmap_weight"]) * heatmap_loss
    return GroundingLoss(total=total, bbox=bbox_loss, geo=loss_geo, cls=loss_cls, heatmap=heatmap_loss)


def decode_anchor_prediction(pred_anchor: torch.Tensor, anchors_full: torch.Tensor, image_wh: Tuple[int, int]) -> torch.Tensor:
    if pred_anchor.ndim == 4:
        pred_anchor = pred_anchor.view(pred_anchor.shape[0], 9, 5, pred_anchor.shape[2], pred_anchor.shape[3])
    batch_size = pred_anchor.shape[0]
    flat_conf = pred_anchor[:, :, 4, :, :].reshape(batch_size, -1)
    flat_idx = flat_conf.argmax(dim=1)
    anchor_idx = flat_idx // (pred_anchor.shape[3] * pred_anchor.shape[4])
    rem = flat_idx % (pred_anchor.shape[3] * pred_anchor.shape[4])
    gj = rem // pred_anchor.shape[4]
    gi = rem % pred_anchor.shape[4]
    selected = pred_anchor[torch.arange(batch_size, device=pred_anchor.device), anchor_idx, :, gj, gi]

    image_w, image_h = float(image_wh[0]), float(image_wh[1])
    grid_h, grid_w = pred_anchor.shape[3], pred_anchor.shape[4]
    stride_w = image_w / max(float(grid_w), 1.0)
    stride_h = image_h / max(float(grid_h), 1.0)
    anchors = anchors_full.to(pred_anchor.device)
    cx = (selected[:, 0].sigmoid() + gi.float()) * stride_w
    cy = (selected[:, 1].sigmoid() + gj.float()) * stride_h
    bw = torch.exp(selected[:, 2]).clamp(max=1e4) * anchors[anchor_idx, 0]
    bh = torch.exp(selected[:, 3]).clamp(max=1e4) * anchors[anchor_idx, 1]
    xywh = torch.stack([cx, cy, bw, bh], dim=1)
    xyxy = xywh2xyxy(xywh)
    xyxy[:, [0, 2]] = xyxy[:, [0, 2]].clamp(0.0, image_w)
    xyxy[:, [1, 3]] = xyxy[:, [1, 3]].clamp(0.0, image_h)
    return xyxy


def center_distance(pred_xyxy: torch.Tensor, target_xyxy: torch.Tensor) -> torch.Tensor:
    pred_center = torch.stack([(pred_xyxy[:, 0] + pred_xyxy[:, 2]) * 0.5, (pred_xyxy[:, 1] + pred_xyxy[:, 3]) * 0.5], dim=1)
    target_center = torch.stack([(target_xyxy[:, 0] + target_xyxy[:, 2]) * 0.5, (target_xyxy[:, 1] + target_xyxy[:, 3]) * 0.5], dim=1)
    return torch.linalg.norm(pred_center - target_center, dim=1)


def compute_iou_metrics(pred_xyxy: torch.Tensor, target_xyxy: torch.Tensor) -> Dict[str, float]:
    iou = bbox_iou(pred_xyxy, target_xyxy, x1y1x2y2=True).detach().float().cpu()
    distance = center_distance(pred_xyxy.detach().float().cpu(), target_xyxy.detach().float().cpu())
    return {
        "mean_iou": float(iou.mean().item()),
        "iou_at_0_5": float((iou >= 0.5).float().mean().item()),
        "iou_at_0_25": float((iou >= 0.25).float().mean().item()),
        "mean_center_distance": float(distance.mean().item()),
    }
```

- [ ] **Step 4: Run the losses test and verify it passes**

Run:

```bash
python tmp_test_grounding_losses.py
```

Expected: `Ran 4 tests` and `OK`.

- [ ] **Step 5: Delete the temporary losses test**

Run:

```bash
rm tmp_test_grounding_losses.py
```

- [ ] **Step 6: Compile losses**

Run:

```bash
python -m py_compile grounding/losses.py
```

Expected: exit code 0.

- [ ] **Step 7: Commit losses**

Run:

```bash
git add grounding/losses.py
git commit -m "Add shared grounding losses"
```

---

### Task 3: Adapters And Registry

**Files:**
- Create: `grounding/adapters.py`
- Create: `grounding/registry.py`
- Temporary test: `tmp_test_grounding_registry.py`

- [ ] **Step 1: Write the failing temporary registry test**

Create `tmp_test_grounding_registry.py`:

```python
import unittest

from grounding.config import load_config
from grounding.registry import MODEL_TYPES, get_model_entry


class GroundingRegistryTest(unittest.TestCase):
    def test_all_expected_model_types_are_registered(self):
        expected = {
            "siglip2_heat",
            "siglip2_test",
            "siglip_ground",
            "lpn",
            "sample4geo",
            "smgeo",
            "ocg",
            "trogeolite",
            "det",
        }
        self.assertEqual(expected, set(MODEL_TYPES))

    def test_each_config_resolves_registry_entry(self):
        for model_type in MODEL_TYPES:
            cfg = load_config(f"configs/grounding/{model_type if model_type != 'siglip_ground' else 'siglip_ground'}.yaml")
            entry = get_model_entry(cfg["model"]["type"])
            self.assertTrue(callable(entry.builder))
            self.assertIsNotNone(entry.adapter_cls)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the test and verify it fails**

Run:

```bash
python tmp_test_grounding_registry.py
```

Expected: FAIL or import error because registry and adapters do not exist.

- [ ] **Step 3: Create `grounding/adapters.py`**

Create `grounding/adapters.py` with:

```python
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch

from grounding.losses import build_geo_features, decode_anchor_prediction


@dataclass
class GroundingOutput:
    device: torch.device
    image_wh: Tuple[int, int]
    pred_anchor: Optional[torch.Tensor] = None
    pred_bbox: Optional[torch.Tensor] = None
    heatmap: Optional[torch.Tensor] = None


class BaseAdapter:
    def __init__(self, model: torch.nn.Module, cfg: Dict[str, Any]):
        self.model = model
        self.cfg = cfg

    def forward(self, batch: Dict[str, Any], device: torch.device) -> GroundingOutput:
        raise NotImplementedError

    def decode(self, output: GroundingOutput, batch: Dict[str, Any], anchors_full: torch.Tensor) -> torch.Tensor:
        if output.pred_bbox is not None:
            return output.pred_bbox
        return decode_anchor_prediction(output.pred_anchor, anchors_full, output.image_wh)


class SiglipTupleAdapter(BaseAdapter):
    def forward(self, batch: Dict[str, Any], device: torch.device) -> GroundingOutput:
        target = batch["target_pixel_values"].to(device, non_blocking=True)
        search = batch["search_pixel_values"].to(device, non_blocking=True)
        geo = build_geo_features(batch, device) if self.cfg["model"]["use_angle"] else None
        outputs = self.model(target, search, angle=geo)
        pred_anchor = outputs[0]
        heatmap = outputs[6] if len(outputs) > 6 else None
        return GroundingOutput(
            device=target.device,
            image_wh=(search.shape[-1], search.shape[-2]),
            pred_anchor=pred_anchor,
            heatmap=heatmap,
        )


class LegacyAnchorAdapter(BaseAdapter):
    def forward(self, batch: Dict[str, Any], device: torch.device) -> GroundingOutput:
        target = batch["target_pixel_values"].to(device, non_blocking=True)
        search = batch["search_pixel_values"].to(device, non_blocking=True)
        geo = build_geo_features(batch, device) if self.cfg["model"]["use_angle"] else None
        outputs = self.model(target, search, geo) if geo is not None else self.model(target, search)
        pred_anchor = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
        return GroundingOutput(
            device=target.device,
            image_wh=(search.shape[-1], search.shape[-2]),
            pred_anchor=pred_anchor,
        )


class DirectBboxAdapter(BaseAdapter):
    def forward(self, batch: Dict[str, Any], device: torch.device) -> GroundingOutput:
        target = batch["target_pixel_values"].to(device, non_blocking=True)
        search = batch["search_pixel_values"].to(device, non_blocking=True)
        geo = build_geo_features(batch, device) if self.cfg["model"]["use_angle"] else None
        outputs = self.model(target, search, geo) if geo is not None else self.model(target, search)
        pred_bbox = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
        return GroundingOutput(
            device=target.device,
            image_wh=(search.shape[-1], search.shape[-2]),
            pred_bbox=pred_bbox,
        )
```

- [ ] **Step 4: Create `grounding/registry.py`**

Create `grounding/registry.py` with:

```python
from dataclasses import dataclass
from typing import Any, Callable, Dict, Type

import torch.nn as nn

from grounding.adapters import DirectBboxAdapter, LegacyAnchorAdapter, SiglipTupleAdapter


@dataclass
class ModelEntry:
    builder: Callable[[Dict[str, Any]], nn.Module]
    adapter_cls: Type


def _build_siglip2_heat(cfg: Dict[str, Any]) -> nn.Module:
    from model import Encoder_heat

    return Encoder_heat(
        model_name=cfg["model"]["model_name"],
        usesg=True,
        useap=True,
    )


def _build_siglip2_test(cfg: Dict[str, Any]) -> nn.Module:
    from model import Encoder_test

    return Encoder_test(
        model_name=cfg["model"]["model_name"],
        usesg=True,
        useap=True,
        use_text_grounding_path=False,
    )


def _build_siglip_ground(cfg: Dict[str, Any]) -> nn.Module:
    from model_ground import Encoder_ground

    return Encoder_ground(
        model_name=cfg["model"]["model_name"],
        usesg=True,
        useap=True,
    )


def _build_lpn(cfg: Dict[str, Any]) -> nn.Module:
    from grounding.train_lpn import LPNGeoLite

    return LPNGeoLite()


def _build_sample4geo(cfg: Dict[str, Any]) -> nn.Module:
    from grounding.train_sample import SampleGeoLite

    return SampleGeoLite()


def _build_smgeo(cfg: Dict[str, Any]) -> nn.Module:
    from grounding.train_sm import SMGeoLite

    return SMGeoLite()


def _build_ocg(cfg: Dict[str, Any]) -> nn.Module:
    from grounding.train_ocg import OCGNetLite

    return OCGNetLite()


def _build_trogeolite(cfg: Dict[str, Any]) -> nn.Module:
    from grounding.ground_cvos import TROGeoLite

    return TROGeoLite()


def _build_det(cfg: Dict[str, Any]) -> nn.Module:
    from grounding.train_det import DetGeoLite

    return DetGeoLite()


REGISTRY: Dict[str, ModelEntry] = {
    "siglip2_heat": ModelEntry(_build_siglip2_heat, SiglipTupleAdapter),
    "siglip2_test": ModelEntry(_build_siglip2_test, SiglipTupleAdapter),
    "siglip_ground": ModelEntry(_build_siglip_ground, SiglipTupleAdapter),
    "lpn": ModelEntry(_build_lpn, LegacyAnchorAdapter),
    "sample4geo": ModelEntry(_build_sample4geo, DirectBboxAdapter),
    "smgeo": ModelEntry(_build_smgeo, DirectBboxAdapter),
    "ocg": ModelEntry(_build_ocg, LegacyAnchorAdapter),
    "trogeolite": ModelEntry(_build_trogeolite, LegacyAnchorAdapter),
    "det": ModelEntry(_build_det, LegacyAnchorAdapter),
}

MODEL_TYPES = tuple(REGISTRY.keys())


def get_model_entry(model_type: str) -> ModelEntry:
    return REGISTRY[model_type]


def build_model_and_adapter(cfg: Dict[str, Any]):
    entry = get_model_entry(cfg["model"]["type"])
    model = entry.builder(cfg)
    adapter = entry.adapter_cls(model, cfg)
    return model, adapter
```

- [ ] **Step 5: Run the registry test**

Run:

```bash
python tmp_test_grounding_registry.py
```

Expected: PASS for registry lookups. If an old class name does not exist, update only that builder import to the class name currently defined in the source file, then rerun.

- [ ] **Step 6: Delete the temporary registry test**

Run:

```bash
rm tmp_test_grounding_registry.py
```

- [ ] **Step 7: Compile adapters and registry**

Run:

```bash
python -m py_compile grounding/adapters.py grounding/registry.py
```

Expected: exit code 0.

- [ ] **Step 8: Commit adapters and registry**

Run:

```bash
git add grounding/adapters.py grounding/registry.py
git commit -m "Add grounding model adapters"
```

---

### Task 4: YAML-Driven Training Entry

**Files:**
- Create: `grounding/train.py`
- Temporary test: `tmp_test_grounding_train_static.py`

- [ ] **Step 1: Write the failing temporary static training test**

Create `tmp_test_grounding_train_static.py`:

```python
import ast
import unittest
from pathlib import Path


TRAIN_PATH = Path("grounding/train.py")


class GroundingTrainStaticTest(unittest.TestCase):
    def test_train_has_expected_cli_and_grounding_only_loss(self):
        source = TRAIN_PATH.read_text(encoding="utf-8")
        tree = ast.parse(source)
        calls = {
            node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, (ast.Attribute, ast.Name))
        }
        self.assertIn("--config", source)
        self.assertIn("--dry-run", source)
        self.assertIn("--max-steps", source)
        self.assertIn("compute_grounding_loss", source)
        self.assertIn("GradScaler", calls)
        self.assertIn("autocast", calls)
        self.assertNotIn("info_nce", source.lower())
        self.assertNotIn("retrieval_loss", source)
        self.assertNotIn("text_pooler_align", source)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the test and verify it fails**

Run:

```bash
python tmp_test_grounding_train_static.py
```

Expected: FAIL because `grounding/train.py` does not exist.

- [ ] **Step 3: Create `grounding/train.py`**

Create `grounding/train.py` with this structure:

```python
import argparse
import os
from pathlib import Path
from typing import Any, Dict

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoTokenizer

from bbox.yolo_utils import get_tensor_anchors
from dataset import ShiftedSatelliteDroneDataset
from grounding.config import load_config
from grounding.losses import compute_grounding_loss
from grounding.registry import build_model_and_adapter


def _device_from_config(cfg: Dict[str, Any]) -> torch.device:
    requested = str(cfg["train"]["device"])
    if requested.startswith("cuda") and torch.cuda.is_available():
        return torch.device(requested)
    return torch.device("cpu")


def _build_loader(cfg: Dict[str, Any], split: str) -> DataLoader:
    model_name = cfg["model"]["model_name"]
    cache_dir = cfg["model"]["cache_dir"]
    processor = AutoImageProcessor.from_pretrained(model_name, cache_dir=cache_dir)
    processor_sat = AutoImageProcessor.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        size=cfg["data"]["sat_size"],
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    dataset = ShiftedSatelliteDroneDataset(
        processor=processor,
        processor_sat=processor_sat,
        tokenizer=tokenizer,
        split=split,
    )
    return DataLoader(
        dataset,
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=(split == "train"),
        num_workers=int(cfg["data"]["num_workers"]),
        pin_memory=True,
        drop_last=False,
        persistent_workers=int(cfg["data"]["num_workers"]) > 0,
    )


def _save_checkpoint(model: torch.nn.Module, save_dir: str, name: str) -> str:
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, name)
    torch.save(model.state_dict(), path)
    return path


def train(cfg: Dict[str, Any], dry_run: bool = False, max_steps: int = 0) -> Dict[str, Any]:
    device = _device_from_config(cfg)
    model, adapter = build_model_and_adapter(cfg)
    model.to(device)
    anchors_full = get_tensor_anchors(str(device))

    if dry_run:
        return {"status": "dry_run", "save_dir": cfg["save_dir"]}

    loader = _build_loader(cfg, "train")
    optimizer = AdamW(model.parameters(), lr=float(cfg["train"]["lr"]), weight_decay=float(cfg["train"]["weight_decay"]))
    amp_enabled = bool(cfg["train"]["amp"]) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    grad_accumulation_steps = max(1, int(cfg["train"]["grad_accumulation_steps"]))
    grad_clip_norm = float(cfg["train"]["grad_clip_norm"])

    global_step = 0
    for epoch in range(int(cfg["train"]["epochs"])):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        progress = tqdm(loader, desc=f"Epoch {epoch + 1}/{cfg['train']['epochs']}")
        for batch_idx, batch in enumerate(progress):
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                output = adapter.forward(batch, device)
                losses = compute_grounding_loss(output, batch, anchors_full, cfg)
                loss_to_backward = losses.total / grad_accumulation_steps

            scaler.scale(loss_to_backward).backward()
            should_step = (batch_idx + 1) % grad_accumulation_steps == 0
            if should_step:
                if grad_clip_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            global_step += 1
            progress.set_postfix(
                {
                    "loss": f"{losses.total.item():.4f}",
                    "bbox": f"{losses.bbox.item():.4f}",
                    "heatmap": f"{losses.heatmap.item():.4f}",
                }
            )
            if max_steps > 0 and global_step >= max_steps:
                checkpoint = _save_checkpoint(model, cfg["save_dir"], "last.pth")
                return {"status": "max_steps", "checkpoint": checkpoint}

        _save_checkpoint(model, cfg["save_dir"], "last.pth")

    checkpoint = _save_checkpoint(model, cfg["save_dir"], "last.pth")
    return {"status": "ok", "checkpoint": checkpoint}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train grounding-only models from YAML.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-steps", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    result = train(cfg, dry_run=args.dry_run, max_steps=args.max_steps)
    print(result)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the static training test**

Run:

```bash
python tmp_test_grounding_train_static.py
```

Expected: `Ran 1 test` and `OK`.

- [ ] **Step 5: Run training dry-run**

Run:

```bash
python grounding/train.py --config configs/grounding/siglip2_heat.yaml --dry-run
```

Expected output contains `dry_run`.

- [ ] **Step 6: Delete the temporary training test**

Run:

```bash
rm tmp_test_grounding_train_static.py
```

- [ ] **Step 7: Compile training entry**

Run:

```bash
python -m py_compile grounding/train.py
```

Expected: exit code 0.

- [ ] **Step 8: Commit training entry**

Run:

```bash
git add grounding/train.py
git commit -m "Add grounding train entry"
```

---

### Task 5: YAML-Driven Evaluation Entry

**Files:**
- Create: `grounding/eval.py`
- Temporary test: `tmp_test_grounding_eval_static.py`

- [ ] **Step 1: Write the failing temporary eval test**

Create `tmp_test_grounding_eval_static.py`:

```python
import ast
import unittest
from pathlib import Path


EVAL_PATH = Path("grounding/eval.py")


class GroundingEvalStaticTest(unittest.TestCase):
    def test_eval_has_cli_metrics_and_json_write(self):
        source = EVAL_PATH.read_text(encoding="utf-8")
        tree = ast.parse(source)
        calls = {
            node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, (ast.Attribute, ast.Name))
        }
        self.assertIn("--config", source)
        self.assertIn("--dry-run", source)
        self.assertIn("--max-batches", source)
        self.assertIn("compute_iou_metrics", source)
        self.assertIn("metrics.json", source)
        self.assertIn("json.dump", source)
        self.assertIn("no_grad", calls)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the test and verify it fails**

Run:

```bash
python tmp_test_grounding_eval_static.py
```

Expected: FAIL because `grounding/eval.py` does not exist.

- [ ] **Step 3: Create `grounding/eval.py`**

Create `grounding/eval.py` with this structure:

```python
import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoTokenizer

from bbox.yolo_utils import get_tensor_anchors
from dataset import ShiftedSatelliteDroneDataset
from grounding.config import load_config
from grounding.losses import compute_iou_metrics
from grounding.registry import build_model_and_adapter


def _device_from_config(cfg: Dict[str, Any]) -> torch.device:
    requested = str(cfg["train"]["device"])
    if requested.startswith("cuda") and torch.cuda.is_available():
        return torch.device(requested)
    return torch.device("cpu")


def _build_loader(cfg: Dict[str, Any]) -> DataLoader:
    model_name = cfg["model"]["model_name"]
    cache_dir = cfg["model"]["cache_dir"]
    processor = AutoImageProcessor.from_pretrained(model_name, cache_dir=cache_dir)
    processor_sat = AutoImageProcessor.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        size=cfg["data"]["sat_size"],
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    dataset = ShiftedSatelliteDroneDataset(
        processor=processor,
        processor_sat=processor_sat,
        tokenizer=tokenizer,
        split="test",
    )
    return DataLoader(
        dataset,
        batch_size=int(cfg["eval"]["batch_size"]),
        shuffle=False,
        num_workers=int(cfg["data"]["num_workers"]),
        pin_memory=True,
        drop_last=False,
        persistent_workers=int(cfg["data"]["num_workers"]) > 0,
    )


def _checkpoint_path(cfg: Dict[str, Any]) -> str:
    checkpoint = cfg["eval"]["checkpoint"]
    if os.path.isabs(str(checkpoint)):
        return str(checkpoint)
    return os.path.join(cfg["save_dir"], str(checkpoint))


def _load_checkpoint(model: torch.nn.Module, path: str) -> None:
    state = torch.load(path, map_location="cpu")
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state, strict=False)


def _average_metrics(total: Dict[str, float], count: int) -> Dict[str, float]:
    return {key: float(value / max(count, 1)) for key, value in total.items()}


def evaluate(cfg: Dict[str, Any], dry_run: bool = False, max_batches: int = 0) -> Dict[str, Any]:
    device = _device_from_config(cfg)
    model, adapter = build_model_and_adapter(cfg)
    model.to(device)
    anchors_full = get_tensor_anchors(str(device))
    checkpoint = _checkpoint_path(cfg)

    if dry_run:
        return {"status": "dry_run", "checkpoint": checkpoint}

    _load_checkpoint(model, checkpoint)
    loader = _build_loader(cfg)
    model.eval()
    total = {
        "mean_iou": 0.0,
        "iou_at_0_5": 0.0,
        "iou_at_0_25": 0.0,
        "mean_center_distance": 0.0,
    }
    count = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(loader, desc=f"Eval {cfg['exp_name']}")):
            output = adapter.forward(batch, device)
            pred_bbox = adapter.decode(output, batch, anchors_full)
            target_bbox = batch["bbox"].to(pred_bbox.device)
            metrics = compute_iou_metrics(pred_bbox, target_bbox)
            batch_n = int(target_bbox.shape[0])
            for key, value in metrics.items():
                total[key] += float(value) * batch_n
            count += batch_n
            if max_batches > 0 and batch_idx + 1 >= max_batches:
                break

    metrics = _average_metrics(total, count)
    result = {
        "status": "ok",
        "config": cfg["config_path"],
        "checkpoint": checkpoint,
        "num_samples": count,
        **metrics,
    }
    output_dir = Path(cfg["eval"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, sort_keys=True)
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate grounding-only models from YAML.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-batches", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    evaluate(cfg, dry_run=args.dry_run, max_batches=args.max_batches)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the eval static test**

Run:

```bash
python tmp_test_grounding_eval_static.py
```

Expected: `Ran 1 test` and `OK`.

- [ ] **Step 5: Run eval dry-run**

Run:

```bash
python grounding/eval.py --config configs/grounding/siglip2_heat.yaml --dry-run
```

Expected output contains `dry_run`.

- [ ] **Step 6: Delete the temporary eval test**

Run:

```bash
rm tmp_test_grounding_eval_static.py
```

- [ ] **Step 7: Compile eval entry**

Run:

```bash
python -m py_compile grounding/eval.py
```

Expected: exit code 0.

- [ ] **Step 8: Commit eval entry**

Run:

```bash
git add grounding/eval.py
git commit -m "Add grounding eval entry"
```

---

### Task 6: Batch Runner With Failure Summary

**Files:**
- Create: `train_ground_group.sh`
- Temporary test: `tmp_test_grounding_group.py`

- [ ] **Step 1: Write the failing temporary group runner test**

Create `tmp_test_grounding_group.py`:

```python
import unittest
from pathlib import Path


class TrainGroundGroupScriptTest(unittest.TestCase):
    def test_script_has_dry_run_summary_and_no_exit_on_failure_loop(self):
        source = Path("train_ground_group.sh").read_text(encoding="utf-8")
        self.assertIn("CONFIGS=(", source)
        self.assertIn("configs/grounding/siglip2_heat.yaml", source)
        self.assertIn("--dry-run", source)
        self.assertIn("group_summary.json", source)
        self.assertIn("train_status", source)
        self.assertIn("eval_status", source)
        self.assertIn("continue", source)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the test and verify it fails**

Run:

```bash
python tmp_test_grounding_group.py
```

Expected: FAIL because `train_ground_group.sh` does not exist.

- [ ] **Step 3: Create `train_ground_group.sh`**

Create `train_ground_group.sh`:

```bash
#!/bin/bash
set -uo pipefail

cd /media/data1/feihong/univerisity_dev

DRY_RUN=0
EXTRA_ARGS=()
for arg in "$@"; do
    if [ "$arg" = "--dry-run" ]; then
        DRY_RUN=1
    else
        EXTRA_ARGS+=("$arg")
    fi
done

CONFIGS=(
    "configs/grounding/siglip2_heat.yaml"
    "configs/grounding/siglip2_test.yaml"
    "configs/grounding/siglip_ground.yaml"
    "configs/grounding/lpn.yaml"
    "configs/grounding/sample4geo.yaml"
    "configs/grounding/smgeo.yaml"
    "configs/grounding/ocg.yaml"
    "configs/grounding/trogeolite.yaml"
    "configs/grounding/det.yaml"
)

SUMMARY_DIR="eval_results/grounding"
SUMMARY_PATH="${SUMMARY_DIR}/group_summary.jsonl"
FINAL_JSON="${SUMMARY_DIR}/group_summary.json"
mkdir -p "$SUMMARY_DIR"
: > "$SUMMARY_PATH"

for CONFIG_PATH in "${CONFIGS[@]}"; do
    echo "============================================================"
    echo "Running grounding config: ${CONFIG_PATH}"
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

    python grounding/train.py --config "$CONFIG_PATH" "${TRAIN_ARGS[@]}"
    if [ "$?" -ne 0 ]; then
        TRAIN_STATUS="failed"
    fi

    python grounding/eval.py --config "$CONFIG_PATH" "${EVAL_ARGS[@]}"
    if [ "$?" -ne 0 ]; then
        EVAL_STATUS="failed"
    fi

    printf '{"config":"%s","train_status":"%s","eval_status":"%s"}\n' \
        "$CONFIG_PATH" "$TRAIN_STATUS" "$EVAL_STATUS" >> "$SUMMARY_PATH"

    if [ "$TRAIN_STATUS" = "failed" ] || [ "$EVAL_STATUS" = "failed" ]; then
        echo "Config failed; continue to next config: ${CONFIG_PATH}"
        continue
    fi
done

python - "$SUMMARY_PATH" "$FINAL_JSON" <<'PY'
import json
import sys
from pathlib import Path

summary_path = Path(sys.argv[1])
final_json = Path(sys.argv[2])
items = [json.loads(line) for line in summary_path.read_text(encoding="utf-8").splitlines() if line.strip()]
final_json.write_text(json.dumps(items, indent=2, sort_keys=True), encoding="utf-8")
print(f"Wrote {final_json}")
PY
```

- [ ] **Step 4: Run the group runner static test**

Run:

```bash
python tmp_test_grounding_group.py
```

Expected: `Ran 1 test` and `OK`.

- [ ] **Step 5: Run group dry-run**

Run:

```bash
bash train_ground_group.sh --dry-run
```

Expected: every config prints dry-run train/eval output, and `eval_results/grounding/group_summary.json` exists.

- [ ] **Step 6: Delete the temporary group test**

Run:

```bash
rm tmp_test_grounding_group.py
```

- [ ] **Step 7: Commit group runner**

Run:

```bash
git add train_ground_group.sh
git commit -m "Add grounding group runner"
```

---

### Task 7: Legacy Archive Pass

**Files:**
- Create directory: `grounding/legacy/`
- Move where safe: old `grounding/*.py`, `grounding/model/`, `grounding/utils/`
- Modify: `grounding/registry.py` imports for moved classes
- Create: `grounding/legacy/README.md`
- Temporary test: `tmp_test_grounding_legacy_imports.py`

- [ ] **Step 1: Write the failing temporary legacy import test**

Create `tmp_test_grounding_legacy_imports.py`:

```python
import unittest

from grounding.registry import MODEL_TYPES, get_model_entry


class GroundingLegacyImportsTest(unittest.TestCase):
    def test_registry_builders_are_importable_after_archive(self):
        for model_type in MODEL_TYPES:
            entry = get_model_entry(model_type)
            self.assertTrue(callable(entry.builder), model_type)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the test before moving files**

Run:

```bash
python tmp_test_grounding_legacy_imports.py
```

Expected: PASS before moving files.

- [ ] **Step 3: Create legacy directory and README**

Create `grounding/legacy/README.md`:

```markdown
# Grounding Legacy Scripts

This folder contains pre-refactor grounding training and evaluation scripts.

The new supported entry points are:

- `grounding/train.py`
- `grounding/eval.py`
- `train_ground_group.sh`

Legacy files are kept for checkpoint compatibility and experiment reference.
```

- [ ] **Step 4: Move legacy files in small groups**

Run these moves with `git mv`:

```bash
mkdir -p grounding/legacy
git mv grounding/eval_ground.py grounding/legacy/eval_ground.py
git mv grounding/train_siglip.py grounding/legacy/train_siglip.py
git mv grounding/train_lpn.py grounding/legacy/train_lpn.py
git mv grounding/train_sample.py grounding/legacy/train_sample.py
git mv grounding/train_sm.py grounding/legacy/train_sm.py
git mv grounding/train_ocg.py grounding/legacy/train_ocg.py
git mv grounding/train_det.py grounding/legacy/train_det.py
git mv grounding/train_wild.py grounding/legacy/train_wild.py
git mv grounding/ground_siglip.py grounding/legacy/ground_siglip.py
git mv grounding/ground_siglip_yolo_head.py grounding/legacy/ground_siglip_yolo_head.py
git mv grounding/ground_evaclip.py grounding/legacy/ground_evaclip.py
git mv grounding/ground_cvos.py grounding/legacy/ground_cvos.py
git mv grounding/out_model.py grounding/legacy/out_model.py
git mv grounding/util.py grounding/legacy/util.py
git mv grounding/yolo_utils.py grounding/legacy/yolo_utils.py
git mv grounding/model grounding/legacy/model
git mv grounding/utils grounding/legacy/utils
```

- [ ] **Step 5: Update registry imports to legacy paths**

Edit `grounding/registry.py`:

```python
def _build_lpn(cfg: Dict[str, Any]) -> nn.Module:
    from grounding.legacy.train_lpn import LPNGeoLite

    return LPNGeoLite()


def _build_sample4geo(cfg: Dict[str, Any]) -> nn.Module:
    from grounding.legacy.train_sample import SampleGeoLite

    return SampleGeoLite()


def _build_smgeo(cfg: Dict[str, Any]) -> nn.Module:
    from grounding.legacy.train_sm import SMGeoLite

    return SMGeoLite()


def _build_ocg(cfg: Dict[str, Any]) -> nn.Module:
    from grounding.legacy.train_ocg import OCGNetLite

    return OCGNetLite()


def _build_trogeolite(cfg: Dict[str, Any]) -> nn.Module:
    from grounding.legacy.ground_cvos import TROGeoLite

    return TROGeoLite()


def _build_det(cfg: Dict[str, Any]) -> nn.Module:
    from grounding.legacy.train_det import DetGeoLite

    return DetGeoLite()
```

- [ ] **Step 6: Run legacy import test**

Run:

```bash
python tmp_test_grounding_legacy_imports.py
```

Expected: PASS. If an import inside a legacy module references `grounding.model` or `grounding.utils`, patch that import to `grounding.legacy.model` or `grounding.legacy.utils` and rerun.

- [ ] **Step 7: Delete the temporary legacy test**

Run:

```bash
rm tmp_test_grounding_legacy_imports.py
```

- [ ] **Step 8: Compile new grounding modules after moves**

Run:

```bash
python -m py_compile grounding/config.py grounding/losses.py grounding/adapters.py grounding/registry.py grounding/train.py grounding/eval.py
```

Expected: exit code 0.

- [ ] **Step 9: Commit legacy archive**

Run:

```bash
git add grounding
git commit -m "Archive legacy grounding scripts"
```

---

### Task 8: Final Verification And Temporary Test Cleanup

**Files:**
- Verify: `grounding/config.py`
- Verify: `grounding/losses.py`
- Verify: `grounding/adapters.py`
- Verify: `grounding/registry.py`
- Verify: `grounding/train.py`
- Verify: `grounding/eval.py`
- Verify: `train_ground_group.sh`

- [ ] **Step 1: Confirm no temporary tests remain**

Run:

```bash
find . -maxdepth 2 -name 'tmp_test_grounding_*.py' -print
```

Expected: no output.

- [ ] **Step 2: Compile all new entry modules**

Run:

```bash
python -m py_compile grounding/config.py grounding/losses.py grounding/adapters.py grounding/registry.py grounding/train.py grounding/eval.py
```

Expected: exit code 0.

- [ ] **Step 3: Run required dry-runs**

Run:

```bash
python grounding/train.py --config configs/grounding/siglip2_heat.yaml --dry-run
python grounding/eval.py --config configs/grounding/siglip2_heat.yaml --dry-run
bash train_ground_group.sh --dry-run
```

Expected:

- train dry-run prints `dry_run`
- eval dry-run prints `dry_run`
- group dry-run writes `eval_results/grounding/group_summary.json`

- [ ] **Step 4: Run lightweight execution if GPU and model cache are available**

Run:

```bash
python grounding/train.py --config configs/grounding/siglip2_heat.yaml --max-steps 1
python grounding/eval.py --config configs/grounding/siglip2_heat.yaml --max-batches 1
```

Expected:

- train writes `/media/data1/feihong/ckpt/ground_siglip2_heat/last.pth`
- eval writes `eval_results/grounding/siglip2_heat/metrics.json`
- metrics JSON contains `mean_iou`, `iou_at_0_5`, `iou_at_0_25`, `mean_center_distance`

- [ ] **Step 5: Inspect git status for accidental temporary files**

Run:

```bash
git status --short
```

Expected: no `tmp_test_grounding_*.py` files. Existing unrelated dirty files may remain and must not be reverted.

- [ ] **Step 6: Commit final verification adjustments**

If Step 4 or Step 5 required small fixes, commit only those fixes:

```bash
git add grounding configs/grounding train_ground_group.sh
git commit -m "Verify grounding refactor"
```

If no files changed after the previous commits, skip this commit.

---

## Self-Review Checklist

- Spec coverage:
  - YAML cards: Task 1
  - shared loss and metrics: Task 2
  - adapters and registry: Task 3
  - shared train entry: Task 4
  - shared eval entry: Task 5
  - group runner with failure continuation: Task 6
  - legacy archive: Task 7
  - temporary tests removed and verification: Task 8

- No text/retrieval loss:
  - Task 4 static test checks absence of `info_nce`, `retrieval_loss`, and `text_pooler_align`.

- Temporary tests:
  - Each task deletes its temporary test before commit.
  - Task 8 verifies no `tmp_test_grounding_*.py` files remain.

- Known implementation attention:
  - If old model class names differ from the names in `grounding/registry.py`, update only the relevant builder import after checking the current class definition.
  - If moving a legacy module breaks its internal imports, patch those imports to `grounding.legacy.*` and keep the new entry modules unchanged.
