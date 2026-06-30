# NGCG-MLLMs Adaptation Design

## Goal

Adapt the NGCG-MLLMs idea from `yuqichen888/NGCG-MLLMs` into this drone-to-satellite retrieval and localization framework, with the practical goal of building a stronger model than the current SigLIP2-only encoder.

The target model should keep our existing strengths:

- drone-to-satellite retrieval with batch InfoNCE
- satellite grid retrieval over 3x3 candidates
- bbox/heatmap localization on the satellite image
- LoRA-style efficient fine-tuning
- Qwen-generated height-specific text descriptions

and add the NGCG-style strengths:

- MLLM text-image joint encoding instead of text and vision being mostly separate
- stronger natural-language grounding from generated descriptions
- trainable register/query tokens for more stable pooled retrieval features
- optional attention/feature redistribution ideas for reducing sink-token dominance

## Source Method Summary

The NGCG-MLLMs repository trains multimodal embedding models for geo-localization. Its useful pieces for our framework are:

1. A paired query-positive training structure: collators produce `qry_inputs` and `pos_inputs`, and `MMEBModel.forward()` encodes both sides into normalized embeddings.
2. MLLM backbones: InternVL, Qwen VL, Phi3-V, SmolVLM are supported through a common encoder interface.
3. Feature extraction from hidden states: pooling can use EOS/last token, mean over all tokens, or register-token variants.
4. Trainable register tokens: `act_language_register_without_position()` prepends register tokens while handling attention masks and RoPE.
5. Efficient tuning: QLoRA/LoRA is applied to selected projection modules, with optional trainable register tokens saved alongside LoRA weights.
6. Retrieval loss: cosine-like dot product logits are divided by a temperature and trained with cross entropy.

Relevant upstream files:

- `train.py`: Lightning training entry, datamodule wiring, LoRA/temperature/config handling.
- `src/model.py`: `MMEBModel`, MLLM backbone loading, pooling, train/validation contrastive loss.
- `src/attn_hock.py`: register-token insertion and attention-hook experiments.
- `src/collator.py`: text-image query/positive batching.

## Current Framework Summary

Our current training path is centered on:

- `unified_siglip_supp.py`: training loop, config, retrieval loss, YOLO bbox loss, heatmap loss.
- `model.py`: `Encoder_heat` and `Encoder_test`.
- `dataset.py`: `ShiftedSatelliteDroneDataset`, which returns drone image, satellite image, text tokens, bbox, angle, height, satellite id, and 3x3 local index.
- `test_unify.py`: feature extraction and retrieval/localization evaluation.

Current model behavior:

- Drone image is encoded by SigLIP2 vision model into `anchor_feats`.
- Satellite image is encoded by SigLIP2 vision model into `sat_feats`.
- Query retrieval feature is `anchor_pooler`.
- Satellite retrieval candidates are `sat_feature_2d_pool` with 9 pooled features.
- Bbox head uses query tokens as context over satellite feature map.
- Text is currently used mostly as an auxiliary alignment signal through frozen SigLIP text features.

The main limitation is that generated text descriptions are not deeply fused into the query representation. They regularize only weakly, while NGCG treats text-image MLLM encoding as the central representation.

## Recommended Architecture

Build a new encoder class rather than replacing `Encoder_test` directly:

```text
EncoderNGCG
├── mllm_encoder: InternVL/Qwen-VL style backbone with LoRA
├── query_adapter: fuses drone image + generated text + geo tokens
├── satellite_adapter: extracts satellite global/grid features
├── register_pooler: optional trainable register/query tokens
├── retrieval_projector: maps MLLM hidden size to PROJECTION_DIM
├── bbox_bridge: maps MLLM/satellite features into existing YOLO/heatmap head
└── bbox_head: reuse current SpatialTransformer + YOLO output path
```

The key design choice is to treat MLLM features as a better retrieval and grounding representation, while keeping the existing bbox head and evaluation code initially.

### Query Side

For each drone sample, construct a multimodal query:

```text
<image>
Height: 150m. Heading: 90 degrees.
Description: {selected qwen_6_28_description segment}
Task: retrieve the matching satellite region.
```

The MLLM hidden state is pooled into `query_feat`.

Recommended pooling order:

1. `eos`: simplest and closest to NGCG baseline.
2. `register_eos`: prepend trainable text register tokens and concatenate/register-pool with EOS.
3. `mean+eos`: fallback if register tokens are unstable.

### Satellite Side

For each satellite crop, encode:

```text
<image>
Satellite view of the candidate search region.
```

Return:

- `sat_global_feat`: one global feature for whole-image retrieval.
- `sat_grid_feats`: 3x3 features for local retrieval, compatible with `build_retrieval_soft_targets()`.
- `sat_feature_map`: spatial feature map for bbox/heatmap localization.

The first implementation should not require the MLLM to output a dense feature map. Instead:

- use MLLM pooled features for retrieval;
- keep SigLIP2 vision tokens for bbox feature maps;
- add a learned bridge that lets MLLM query features condition the existing bbox head.

This hybrid approach is lower risk than trying to extract dense features from InternVL/Qwen immediately.

## Integration Phases

### Phase 1: NGCG Retrieval Teacher

Purpose: validate whether MLLM embeddings improve retrieval before touching bbox.

Implementation:

- Add `EncoderNGCGRetrieval`.
- It returns the same retrieval interface as `Encoder_test`:
  - `anchor_feats`: MLLM query embedding.
  - `grid_feats`: MLLM or SigLIP satellite 3x3 candidates.
  - `pred_anchor`: dummy or frozen current bbox path.
- Use current `info_nce_loss()` and `build_retrieval_soft_targets()`.
- Train only LoRA + projection heads.

Loss:

```text
L = retrieval_loss
```

Success criteria:

- Retrieval Recall@1 improves against current SigLIP baseline.
- Feature extraction still works in `test_unify.py`.
- Text ablation shows generated descriptions matter.

### Phase 2: Hybrid Retrieval + Existing Bbox

Purpose: improve both retrieval and localization without rewriting the detector.

Implementation:

- MLLM query feature replaces or augments `anchor_pooler`.
- SigLIP satellite feature map remains the bbox spatial backbone.
- Bbox context becomes:

```text
bbox_context = adapter(concat(siglip_anchor_pooler, mllm_query_feat, geo_feat))
```

or token-level:

```text
anchor_context = [siglip_anchor_tokens, projected_mllm_query_token, geo_token]
```

Loss:

```text
L = retrieval_weight * retrieval_loss
  + bbox_weight * (yolo_loss + heatmap_weight * heatmap_loss)
  + align_weight * mllm_siglip_align_loss
```

`mllm_siglip_align_loss` keeps MLLM and SigLIP spaces compatible:

```text
InfoNCE(mllm_query_feat, siglip_anchor_pooler.detach())
InfoNCE(mllm_sat_feat, siglip_sat_global.detach())
```

Success criteria:

- Retrieval does not regress from Phase 1.
- IoU/Acc@0.25 and Acc@0.5 improve or remain stable.
- Bbox head still trains with existing labels.

### Phase 3: Register-Token Pooling

Purpose: import the most NGCG-specific idea once the simple MLLM path works.

Implementation:

- Add trainable text/image register tokens to the MLLM encoder.
- Start with language/query side only, because drone + description is the side with richer text.
- Use `register + eos` pooling:

```text
query_feat = projector(concat(flatten(register_states), eos_state))
```

Avoid patching every attention path at first. A conservative implementation can add learnable query tokens after hidden states and cross-attend to the MLLM token sequence, which is easier to debug than monkey-patching RoPE/attention.

Success criteria:

- Register pooling improves recall or stabilizes training.
- Attention maps do not collapse into register tokens only.
- LoRA checkpoint includes register parameters.

### Phase 4: Full MLLM Satellite Grid

Purpose: replace SigLIP 3x3 satellite retrieval features with MLLM-derived local candidates.

Implementation options:

1. Use patch/image token hidden states from the MLLM vision tower and pool them to 3x3.
2. Prompt the MLLM with nine region tokens and extract hidden states for each region token.
3. Keep SigLIP grid features and use MLLM only as query/global teacher.

Recommendation: start with option 3, then option 1. Option 2 is attractive but harder to make numerically stable.

## Data Design

Extend `ShiftedSatelliteDroneDataset` to expose:

- `description_text`: selected Qwen description for the current height.
- `all_description_texts`: optional list of 5 variants.
- `geo_text`: deterministic string for height and angle.
- `satellite_id`: already available.
- `local_index`: already available as `index`.

Training text schedule:

- Epoch cycling through the 5 descriptions is good.
- Keep deterministic cycling for reproducibility:

```text
epoch 0 -> description 5
epoch 1 -> description 4
...
```

Negative handling:

- Keep current same-satellite soft targets so multiple drone views from the same satellite are not treated as hard negatives.

## Loss Design

Recommended starting weights:

```text
retrieval_weight = 0.8
bbox_weight = 0.2
heatmap_weight = 0.2
mllm_siglip_align_weight = 0.05 -> 0.0 over first 10 epochs
temperature = 0.03 to 0.07
```

For MLLM retrieval:

- Normalize all projected features.
- Use the same soft target matrix as current training.
- Add hard-negative mining only after the baseline is stable.

For bbox:

- Do not let bbox loss immediately dominate MLLM LoRA.
- In Phase 2, detach the MLLM query feature for the bbox path for the first few epochs, then unfreeze.
- Keep gradient clipping at 1.0.

## Training Configuration

Recommended first MLLM backbone:

- InternVL3.5-1B or Qwen2.5-VL small/medium if available locally.
- Avoid starting with very large Qwen3.6-35B for training; use it for caption generation or offline teacher labels.

Recommended memory setup:

- QLoRA / 4-bit for MLLM.
- LoRA target modules:

```text
q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj
```

or the NGCG-style target list for InternVL:

```text
qkv_proj,o_proj,gate_up_proj,down_proj,k_proj,q_proj,out_proj,v_proj
```

Trainable modules:

- LoRA weights
- retrieval projector
- bbox bridge
- optional register/query tokens
- existing bbox head

Frozen modules:

- base MLLM weights
- optionally SigLIP backbone at first

## Code Changes

### New Files

```text
ngcg/
├── __init__.py
├── encoder_ngcg.py
├── mllm_pooling.py
├── mllm_processors.py
└── config.py
```

### Modified Files

`unified_siglip_supp.py`

- Add `ENCODER_TYPE = "ngcg"`.
- Add MLLM config options.
- Allow model outputs to contain both MLLM retrieval features and SigLIP bbox features.

`model.py`

- Keep `Encoder_test` stable.
- Add adapters only if shared helper code is useful.

`dataset.py`

- Return raw text strings in addition to tokenized SigLIP text.
- Keep current tokenized fields for backward compatibility.

`test_unify.py`

- Add `encoder_ngcg` model type.
- Extract query/gallery features from the NGCG return tuple.
- Preserve bbox evaluation.

## Interface Contract

`EncoderNGCG.forward()` should return the same 7-tuple shape as `Encoder_test`:

```python
(
    pred_anchor,
    None,
    text_feats,
    anchor_feats,
    grid_feats,
    aux_outputs,
    heatmap_logits,
)
```

Where:

- `anchor_feats`: normalized MLLM query feature, shape `[B, D]`.
- `grid_feats`: normalized candidate features, shape `[B, 9, D]`.
- `text_feats`: optional text-only feature for diagnostics.
- `aux_outputs`: dict with `mllm_query`, `siglip_query`, `mllm_sat`, and debug info.

This lets the training loop reuse existing retrieval, bbox, heatmap, and logging code.

## Evaluation Plan

Run ablations in this order:

1. Current SigLIP baseline.
2. MLLM query feature + SigLIP satellite grid.
3. MLLM query + MLLM global satellite + SigLIP grid.
4. Hybrid bbox bridge.
5. Register pooling.
6. Full MLLM satellite grid.

Metrics:

- Retrieval Recall@1/5/10.
- Same-satellite false negative rate.
- IoU mean.
- Acc@0.25 and Acc@0.5.
- uIoU if using `test_unify.py`.
- Training loss decomposition.

Diagnostics:

- Feature norm distribution.
- Similarity matrix diagonal/off-diagonal.
- Same-satellite soft target rows.
- Text ablation: blank text vs Qwen description vs height/angle only.
- LoRA trainable parameter report.

## Risks

1. MLLM features may be strong globally but weak for precise 3x3 localization.
   Mitigation: keep SigLIP dense satellite features for bbox and grid initially.

2. Register-token monkey patches can be fragile across backbones.
   Mitigation: implement external cross-attention pooling first; patch RoPE only after baseline works.

3. Qwen-generated captions may contain noisy details.
   Mitigation: keep the `tool/check.py` validation/fix pipeline and run text ablations.

4. MLLM training may be slow.
   Mitigation: QLoRA, small backbone first, cache MLLM features for ablations.

5. Bbox loss may damage retrieval representation.
   Mitigation: staged training and small bbox weight at first.

## Recommended First Milestone

Build `EncoderNGCGRetrieval` as a retrieval-only model:

- Use Qwen/InternVL processor to encode drone image + generated text.
- Use SigLIP satellite 3x3 grid initially.
- Project MLLM query feature to `PROJECTION_DIM`.
- Train with existing `info_nce_loss()` and same-satellite soft labels.
- Evaluate retrieval only.

This milestone is the fastest way to answer the core question: does NGCG-style MLLM representation improve our retrieval problem?

If yes, move to hybrid bbox. If no, use the MLLM as an offline teacher or caption generator rather than making it the core encoder.

## References

- NGCG-MLLMs repository: https://github.com/yuqichen888/NGCG-MLLMs/tree/main
- NGCG project page: https://yuqichen888.github.io/NGCG-MLLMs-web/
- Local training entry: `unified_siglip_supp.py`
- Local encoder implementation: `model.py`
- Local dataset: `dataset.py`
- Local evaluation: `test_unify.py`
