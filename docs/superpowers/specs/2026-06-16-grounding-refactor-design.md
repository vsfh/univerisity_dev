# Grounding Refactor Design

## Goal

Refactor the `grounding/` subproject into a small, shared grounding-only training and evaluation stack. The new stack should cover the models currently represented by `grounding/eval_ground.py`, use YAML cards under `configs/grounding/`, and keep grounding loss and evaluation consistent with the bbox/heatmap behavior used by `unified_siglip_supp.py`.

This refactor does not use text or retrieval losses.

## Scope

The first implementation phase covers:

- Unified training entry: `grounding/train.py`
- Unified evaluation entry: `grounding/eval.py`
- Model adapters with a shared output shape
- Grounding-only bbox and heatmap losses
- Shared mIoU, IoU@0.5, IoU@0.25, and center-distance evaluation
- Grounding YAML configs in `configs/grounding/`
- Legacy archive for old scripts under `grounding/legacy/`
- Batch runner `train_ground_group.sh`

Model classes should not be moved in the first phase. They may be imported from their current modules or from `grounding.legacy.*` after archive movement.

## Directory Layout

```text
grounding/
  train.py
  eval.py
  adapters.py
  losses.py
  registry.py
  config.py
  legacy/

configs/grounding/
  siglip2_heat.yaml
  siglip2_test.yaml
  siglip_ground.yaml
  lpn.yaml
  sample4geo.yaml
  smgeo.yaml
  ocg.yaml
  trogeolite.yaml
  det.yaml

train_ground_group.sh
```

Old scripts are archived rather than deleted. If moving a file into `grounding/legacy/` breaks imports for a model that is still needed, leave that file in place temporarily and record the reason in the implementation summary.

## Configuration

Each model has one YAML card under `configs/grounding/`.

Example:

```yaml
exp_name: siglip2_heat
save_dir: /media/data1/feihong/ckpt/ground_siglip2_heat

model:
  type: siglip2_heat
  checkpoint: null
  use_angle: true
  use_heatmap: true

train:
  epochs: 20
  batch_size: 32
  grad_accumulation_steps: 2
  lr: 5.0e-5
  weight_decay: 1.0e-4
  amp: true
  grad_clip_norm: 1.0

loss:
  bbox_weight: 1.0
  heatmap_weight: 0.2
  heatmap_confidence_weight: 0.5

eval:
  batch_size: 8
  checkpoint: last.pth
  output_dir: eval_results/grounding/siglip2_heat
```

The config loader should stay simple: read YAML, merge a small set of defaults, and expose values to train/eval. Avoid broad defensive parsing.

## Training Flow

```text
YAML
 -> grounding.config
 -> grounding.registry.build_model(...)
 -> dataset.ShiftedSatelliteDroneDataset
 -> adapter.forward(...)
 -> grounding.losses.compute_grounding_loss(...)
 -> save checkpoint
```

`grounding/train.py` trains only bbox/heatmap objectives. It must not compute text alignment, image retrieval, or text retrieval losses.

## Evaluation Flow

```text
YAML + checkpoint
 -> grounding.config
 -> grounding.registry.build_model(...)
 -> adapter.forward(...)
 -> adapter.decode(...)
 -> shared grounding metrics
 -> metrics.json
```

Metrics are:

- `mean_iou`
- `iou_at_0_5`
- `iou_at_0_25`
- `mean_center_distance`

All models should write metrics with the same JSON field names.

## Adapter Interface

Adapters normalize model outputs into one object:

```python
@dataclass
class GroundingOutput:
    pred_anchor: Optional[torch.Tensor] = None
    pred_bbox: Optional[torch.Tensor] = None
    heatmap: Optional[torch.Tensor] = None
```

Training uses:

```python
output = adapter.forward(batch)
losses = compute_grounding_loss(output, batch, config)
```

Evaluation uses:

```python
output = adapter.forward(batch)
pred_bbox = adapter.decode(output, batch, config)
```

Adapter rules:

- Anchor-based models return `pred_anchor`.
- Heatmap-capable models also return `heatmap`.
- Direct bbox or anchor-free models return `pred_bbox`.
- Each adapter supports only the model selected by its YAML card.
- Keep adapters short; do not add broad compatibility branches for unknown historical outputs.

## Model Mapping

Initial model types:

```text
siglip2_heat   -> model.Encoder_heat
siglip2_test   -> model.Encoder_test
siglip_ground  -> model_ground.Encoder_ground
lpn            -> LPNGeoLite from the legacy LPN script
sample4geo     -> SampleGeoLite from the legacy Sample4Geo script
smgeo          -> SMGeoLite from the legacy SMGeo script
ocg            -> OCGNetLite or OfficialOCGNet from the legacy OCG script
trogeolite     -> TROGeoLite from the legacy CVOGL script
det            -> DetGeoLite from the legacy DET script
```

The registry maps `model.type` to a builder and adapter. The first phase should favor minimal wrappers over moving model implementations.

## Losses

`grounding/losses.py` should reuse shared code where possible:

- `bbox.yolo_utils.build_target`
- `bbox.yolo_utils.yolo_loss`
- `bbox.yolo_utils.eval_iou_acc`
- `bbox.yolo_utils.bbox_iou`
- bbox/heatmap behavior aligned with `unified_siglip_supp.py`

For heatmap-enabled models:

```text
bbox_loss = loss_geo + loss_cls
loss = bbox_weight * bbox_loss + heatmap_weight * heatmap_loss
```

For non-heatmap models:

```text
loss = bbox_weight * bbox_loss
```

If an older model is anchor-free or direct-bbox, its adapter/loss path should stay minimal and grounding-only.

## Legacy Archive

Archive candidates:

```text
grounding/eval_ground.py
grounding/train_siglip.py
grounding/train_lpn.py
grounding/train_sample.py
grounding/train_sm.py
grounding/train_ocg.py
grounding/train_det.py
grounding/train_wild.py
grounding/ground_siglip.py
grounding/ground_siglip_yolo_head.py
grounding/ground_evaclip.py
grounding/ground_cvos.py
grounding/out_model.py
grounding/util.py
grounding/yolo_utils.py
grounding/model/
grounding/utils/
```

The implementation should not delete these files in the first phase. It should move what is safe and leave a short note for any file that cannot be moved without breaking imports.

## Batch Runner

`train_ground_group.sh` runs each YAML card, evaluates after training, and continues after failures.

Behavior:

- Train each config with `python grounding/train.py --config <path>`.
- Evaluate with `python grounding/eval.py --config <path>`.
- Record train/eval success or failure for each config.
- Continue to the next config even if the current one fails.
- Write `eval_results/grounding/group_summary.json`.

Summary fields:

- `config`
- `train_status`
- `eval_status`
- `checkpoint`
- `mean_iou`
- `iou_at_0_5`
- `iou_at_0_25`
- `mean_center_distance`

## Testing

Implementation should use temporary unit tests or scripts, then delete those test files after verification.

Temporary tests should cover:

- YAML loading and defaults
- registry lookup for every model type
- adapter output accepted by `compute_grounding_loss`
- anchor-based bbox decoding to `xyxy`
- evaluation metric field names
- `train_ground_group.sh` dry-run failure continuation and summary writing

Verification commands:

```bash
python -m py_compile grounding/train.py grounding/eval.py grounding/adapters.py grounding/losses.py grounding/registry.py grounding/config.py
python grounding/train.py --config configs/grounding/siglip2_heat.yaml --dry-run
python grounding/eval.py --config configs/grounding/siglip2_heat.yaml --dry-run
bash train_ground_group.sh --dry-run
```

If lightweight GPU execution is practical:

```bash
python grounding/train.py --config configs/grounding/siglip2_heat.yaml --max-steps 1
python grounding/eval.py --config configs/grounding/siglip2_heat.yaml --max-batches 1
```

## Acceptance Criteria

- New `grounding/train.py` and `grounding/eval.py` are YAML-driven.
- Grounding loss/eval use bbox/heatmap behavior consistent with `unified_siglip_supp.py`.
- No text or retrieval losses are used.
- Model differences are isolated in adapters.
- Old files are archived to `grounding/legacy/` or explicitly left in place with a reason.
- `train_ground_group.sh` continues after per-model failures and writes a summary.
- Temporary test files are removed after verification.
