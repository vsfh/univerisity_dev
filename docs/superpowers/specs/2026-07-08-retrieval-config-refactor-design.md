# Retrieval Config Refactor Design

## Goal

Refactor retrieval training and testing to follow the grounding package pattern while keeping the model implementations unchanged. The refactor covers five retrieval model families:

- `clip`
- `siglip`
- `openclip`
- `evaclip`
- `sample_retrieval` / `sample4geo`

The new retrieval workflow should reuse `dataset.py`, `unified_siglip_supp.py`, and `test_unify.py` logic where practical, replace duplicated training loops with a configuration-driven entry point, and add a group script named `train_retrieval_grounp.sh` that trains and evaluates all five models.

## Non-Goals

- Do not change the architecture or forward shape of the existing model classes.
- Do not replace existing grounding workflows.
- Do not fold `encoder_heat`, `encoder_test`, `unify_geo`, or `trans_geo` into this retrieval group.
- Do not add bbox or grounding losses to retrieval-only training.

## Architecture

Add a retrieval package structure parallel to `grounding/`:

- `retrieval/config.py`: load YAML configs, merge defaults, normalize paths.
- `retrieval/registry.py`: map `model.type` to model builders and adapters.
- `retrieval/adapters.py`: expose a uniform retrieval interface over models with different forward methods.
- `retrieval/train.py`: shared training loop for retrieval-only InfoNCE.
- `retrieval/eval.py`: shared evaluation entry point for retrieval metrics.
- `configs/retrieval/*.yaml`: one config per retrieval model.
- `train_retrieval_grounp.sh`: sequential group train/eval runner.

The registry should reuse existing model classes instead of copying them:

- `retrieval/train_clip.py::Encoder`
- `retrieval/train_siglip.py::Encoder`
- `retrieval/train_openclip.py::Encoder`
- `retrieval/train_evaclip.py::Encoder`
- `grounding/train_sample_retrieval.py::SampleGeoLite`

The old scripts may remain for compatibility, but the new group workflow should use the new YAML-driven entry points.

## Dataset And Inputs

Training and evaluation should use `ShiftedSatelliteDroneDataset` from `dataset.py`. The new retrieval code should not maintain the old duplicated `TargetSearchDataset` classes.

Each batch is expected to provide:

- `target_pixel_values`: drone query image tensor.
- `search_pixel_values`: satellite search image or crop tensor.
- `input_ids` and `attention_mask`: text tokens when the model supports text.
- `index`: 3x3 grid positive location for grid retrieval.
- `satellite_id`: same-location grouping for soft targets.
- `bbox`, `height`, `angle`, `drone_path`, and `satellite_path`: metadata for evaluation and reporting.

Config-controlled dataset options should include:

- `data.sat_size`
- `data.drone_size`
- `data.test_crop_ratio`
- `data.subset_heights`
- `data.subset_angles`
- `data.num_workers`

Model-specific processor/tokenizer construction belongs in the registry or adapter setup. SigLIP/CLIP models use Hugging Face processors. OpenCLIP/EVA-CLIP and SampleGeoLite reuse their existing wrapper classes.

## Retrieval Adapter Contract

Adapters should convert a batch into a common feature payload without changing the underlying model:

- `query_feats`: `[B, D]`
- `candidate_feats`: either `[B, 9, D]` for grid retrieval or `[B, D]` for global retrieval
- `text_feats`: optional `[B, D]`

Adapters also declare or infer retrieval granularity:

- `grid`: candidate features are `[B, 9, D]` and use `batch["index"]`.
- `global`: candidate features are `[B, D]` and use `torch.arange(B)` labels.
- `auto`: default behavior; infer from `candidate_feats.ndim` and shape.

If auto mode sees `[B, 9, D]`, use grid retrieval. If it sees `[B, D]`, use global retrieval. If it sees `[B, L, D]` with `L != 9`, fail with a clear error unless the adapter explicitly converts the output to global features or the config selects a supported mode.

This matters because some future or external models may only support whole-image features. Those models must not use `index` labels.

## Loss

The training objective is retrieval-only InfoNCE.

For grid retrieval:

- Flatten candidates from `[B, 9, D]` to `[B * 9, D]`.
- Use `unified_siglip_supp.build_retrieval_soft_targets` with its defaults:
  - `num_locations=9`
  - `positive_weight=0.92`
- Pass `batch["satellite_id"]` when available so same-satellite support matches unified SigLIP behavior.
- Use `batch["index"]` as the local positive grid index.

For global retrieval:

- Use candidates as `[B, D]`.
- Use `torch.arange(B)` as hard labels.
- Do not read or use `batch["index"]`.

`loss.use_text_loss` defaults to `false`. CLIP, SigLIP, OpenCLIP, and EVA-CLIP may enable text retrieval loss through config. SampleGeoLite ignores text loss because it has no text branch. If text loss is enabled, it follows the same grid or global label rule as the image query loss.

Other loss config:

- `loss.temperature`: default `0.07`.
- `loss.granularity`: default `auto`.
- `loss.use_text_loss`: default `false`.

## Training

`retrieval/train.py` should follow the grounding training entry point style:

- `--config` selects a YAML config.
- `--device` overrides the configured device.
- `--dry-run` validates config and output paths without training.
- `--max-steps` optionally limits training for smoke tests.

The training loop should support:

- AMP via `train.amp`.
- Gradient accumulation via `train.grad_accumulation_steps`.
- Gradient clipping via `train.grad_clip_norm`.
- TensorBoard logging under `runs/retrieval/<exp_name>`.
- Saving `best.pth` and `last.pth` under `save_dir`.

Default checkpoint directories:

- `/media/data1/feihong/ckpt/retrieval_siglip`
- `/media/data1/feihong/ckpt/retrieval_clip`
- `/media/data1/feihong/ckpt/retrieval_openclip`
- `/media/data1/feihong/ckpt/retrieval_evaclip`
- `/media/data1/feihong/ckpt/retrieval_sample4geo`

## Evaluation

`retrieval/eval.py` should load the same YAML config and evaluate a checkpoint. By default it evaluates `cfg["eval"]["checkpoint"]`, resolved relative to `save_dir`, with `last.pth` as the default. `--checkpoint` can override this path.

Evaluation should reuse existing retrieval metric logic from `retrieval/eval_retrieval.py` and `test_unify.py` where practical:

- candidate sampling
- include map matching
- recall@1, recall@5, recall@10
- per-height summaries
- per-angle summaries
- per-subset summaries
- `top1_success_drone_names`

The output JSON should include:

- `model_type`
- `checkpoint`
- `num_gallery`
- `num_samples`
- `overall`
- `retrieval`
- `per_subset`
- `per_height`
- `per_angle`
- `top1_success_drone_names`

Evaluation must respect the same granularity rules as training. Grid models can score local grid candidates. Global models score whole-image candidates and do not depend on `index`.

## Group Script

Create `train_retrieval_grounp.sh` with behavior matching the grounding group script style:

- Change to repo root.
- Accept `--dry-run`.
- Accept `--gpus 0,1,2` or use `CUDA_VISIBLE_DEVICES`.
- Pass remaining arguments through to train and eval.
- Run five configs sequentially:
  - `configs/retrieval/siglip.yaml`
  - `configs/retrieval/clip.yaml`
  - `configs/retrieval/openclip.yaml`
  - `configs/retrieval/evaclip.yaml`
  - `configs/retrieval/sample_retrieval.yaml`
- Train each config first, then evaluate it if training succeeds.
- Record failures and continue to the next config.
- Write:
  - `eval_results/retrieval/group_summary.jsonl`
  - `eval_results/retrieval/group_summary.json`

The script name intentionally follows the requested spelling: `train_retrieval_grounp.sh`.

## Testing And Verification

Add focused tests for:

- Config loading and default values.
- Registry construction for each model type in dry-run or mocked mode.
- Loss target selection:
  - grid features use `index` and unified soft target defaults.
  - global features use `torch.arange(B)` and ignore `index`.
  - `[B, L, D]` with unsupported `L` errors clearly.
- Group script dry run produces summary files without starting full training.

Run existing relevant tests when possible:

- `test_retrieval_soft_targets.py`
- `test_denseuav_retrieval.py`
- Any existing dry-run tests around group scripts.

Because full model construction may need cached model weights and GPU memory, smoke tests should prefer dry-run paths or small `--max-steps` runs.

## Risks

- Importing old training modules may execute module-level setup or require optional dependencies such as `open_clip` or `timm`. The registry should import lazily so unsupported model dependencies fail only when that model is selected.
- Evaluation logic is currently large and duplicated. The first implementation should reuse stable functions instead of rewriting all metrics at once.
- Existing worktree changes touch shared files such as `dataset.py` and `unified_siglip_supp.py`. Implementation must build on the current content and avoid reverting unrelated edits.

