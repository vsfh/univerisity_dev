# Query-Grounded Dense Local Contrastive Training Design

## Goal

Restore the query-dependent mechanisms that matter in the DetGeo, LPN,
Sample4Geo, TROGeoLite, OCGNet, and SMGeo grounding variants, while preserving
the repository's unified training entry point. Add an augmented-bbox-driven
dense local contrastive objective and a mandatory spatial matching gate so the
localization path cannot reduce to a satellite-only detector.

This is a key-mechanism restoration, not a line-for-line import of six official
projects. Existing unified adapters, image sizes, localization heads, and the
shared dataset remain the integration boundary.

## Problem

The current group training path supervises bbox and optional heatmap outputs but
does not penalize query-independent predictions. Random satellite bbox
augmentation changes the target location, but each satellite sample still has a
single supervised target. A model can therefore learn satellite saliency without
using the paired drone query.

The observed DetGeo checkpoint confirms this failure mode: shuffling queries
changes predicted boxes by only about 0.1 pixels, and different query embeddings
have near-unity cosine similarity. The affected group models report test mIoU
near 0.03.

`unified_siglip_supp.py` avoids the same degree of collapse in its combined
configuration because query features must retrieve the correct region from
`B * 9` satellite candidates through a dominant InfoNCE objective. Its
`bbox_only` configuration does not have the same guarantee.

## Scope

### Included

- Restore each model's query/reference encoding, prompt, fusion, or retrieval
  mechanism where it materially affects query dependence.
- Add a common pre-fusion query/local-feature contract.
- Add bbox-derived dense soft targets and local InfoNCE.
- Feed the same matcher into localization through a mandatory spatial gate.
- Add explicit query-click data flow.
- Extend `test_unify.py` to support all six grounding model types with its
  existing protocol and result schema.
- Record training losses, query-dependence diagnostics, configuration, and
  checkpoint metadata.

### Excluded

- Full verbatim ports of the six official repositories.
- Reproduction of every paper-specific dependency or preprocessing step.
- Loading old `ground_*/last.pth` checkpoints into the new training runs.
- Running full training or formal evaluation experiments as part of the code
  change. The user will run those experiments.

## Selected Architecture

Each model retains a paper-specific feature path but exposes a common output:

```python
@dataclass
class GroundingOutput:
    device: torch.device
    image_wh: tuple[int, int]
    pred_anchor: torch.Tensor | None = None
    pred_bbox: torch.Tensor | None = None
    heatmap: torch.Tensor | None = None
    bbox_raw: torch.Tensor | None = None
    moe_entropy: torch.Tensor | None = None
    query_embedding: torch.Tensor | None = None
    search_local_features: torch.Tensor | None = None
    search_grid_size: tuple[int, int] | None = None
    matcher_logits: torch.Tensor | None = None
    paper_aux_losses: dict[str, torch.Tensor] | None = None
```

`query_embedding` and `search_local_features` must be taken before cross-view
fusion. A fused query representation is forbidden because it can leak satellite
information into the contrastive query.

Every model owns lightweight projections to a configurable common dimension:

```text
query native feature     -> Linear(native_dim, projection_dim)
satellite native feature -> 1x1 Conv(native_dim, projection_dim)
```

Both outputs are L2 normalized before similarity calculation.

## Mandatory Query Matcher

An auxiliary projection alone is insufficient: the projection branch could use
the query while the bbox branch remains satellite-only. The matcher therefore
also gates the satellite localization feature.

For projected query `q` and projected satellite cells `s_n`:

```text
local_logits[n] = cosine(q, s_n) / temperature
gate = spatial_softmax(local_logits) * num_cells
gated_satellite = satellite_feature * gate
```

There is no ungated satellite residual around this operation. The gated feature
is passed to the paper-specific fusion and localization head. Multiplication by
the number of cells preserves an average gate magnitude near one.

The same local logits are used by the dense contrastive objective. This couples
the representation objective and localization path: local InfoNCE prevents a
uniform gate, and the no-residual gate prevents the bbox path from bypassing it.

For models that already contain query-aware spatial gating, such as DetGeo and
SMGeo, the common projected matcher replaces or supplies that gate rather than
adding an independent competing attention map.

## Augmented-Bbox-Driven Dense Targets

Targets use `batch["bbox"]` after satellite augmentation and resizing. No
pre-augmentation coordinates or fixed 3x3 index may be used.

Given an augmented bbox `[x1, y1, x2, y2]`, satellite image size `(W, H)`, and
local feature grid `(Hg, Wg)`, coordinates are mapped continuously:

```text
x_grid = x / W * Wg
y_grid = y / H * Hg
```

The per-reference target distribution is:

```text
target = center_weight * center_one_hot
       + (1 - center_weight) * normalized_cell_bbox_overlap
```

The default `center_weight` is `0.7`. The center cell is always positive, even
when the bbox is smaller than a grid cell. Invalid/reversed coordinates are
ordered and clipped to the image before target construction. A degenerate bbox
falls back to the clipped center cell.

For batch size `B` and `N` cells per reference, candidates are flattened to
`B * N`. Positive support for a query includes bbox cells from every batch row
with the same identity. Identity resolution is:

```text
object_id -> satellite_id -> row index
```

The current dataset supplies `satellite_id`; this represents the same physical
target across height/angle views. Same-identity cells outside their augmented
bboxes remain negatives. Rows with different identities are negatives.

The dense loss is soft-target cross entropy:

```text
L_dense = mean_i(-sum_j(target_ij * log_softmax(logits_ij)))
```

Total training loss is:

```text
L_total = L_grounding
        + dense_weight * L_dense
        + L_paper_aux
```

Defaults:

```yaml
query_guard:
  enabled: true
  projection_dim: 256
  temperature: 0.07
  weight: 0.2
  warmup_epochs: 2
  center_weight: 0.7
  identity_key: object_id
```

`dense_weight` increases linearly from zero to `0.2` over two epochs.

## Query Click Contract

The dataset returns an explicit normalized point:

```python
batch["query_click"]  # float tensor [B, 2], values in [0, 1]
```

The current drone crops designate a centered object, so the initial value is
`(0.5, 0.5)`. Models must not silently manufacture a center prompt. This makes
the assumption visible and permits future real click annotations without model
interface changes.

DetGeo, OCGNet, TROGeoLite, and SMGeo convert this point into a Gaussian click
map. Sample4Geo and LPN retain whole-image retrieval semantics and do not require
the click map.

## Per-Model Restoration

### DetGeo

- Restore distinct query ResNet18 and reference Darknet branches.
- Initialize the query branch from ImageNet and the reference branch from the
  configured YOLO/Darknet weights when available.
- Concatenate the explicit click map with the query RGB input through the
  original input adapter.
- Derive the contrastive query from the mapped query feature and satellite local
  candidates from the mapped Darknet feature.
- Use the common matcher as the original cosine spatial gate before the YOLO
  bbox head.

### OCGNet

- Restore separate query and reference encoders.
- Preserve Gaussian Knowledge Transfer, early/late Location Enhancement, and
  Multi-Head Cross Attention.
- Derive query/local contrastive features after GKT and reference encoding but
  before MHCA.
- Apply the mandatory matcher gate to reference features before MHCA/fusion.

### TROGeoLite

- Restore click-conditioned query input.
- Preserve its shared Swin transformer and Cross-View Object Perception-style
  query-context attention.
- Keep bbox and coordinate/segmentation-compatible auxiliary outputs.
- Extract both contrastive inputs before CVOPM and gate reference features
  before cross-attention.

### Sample4Geo

- Preserve the weight-shared ConvNeXt encoder.
- Restore global symmetric InfoNCE as a paper auxiliary objective.
- Use pooled query features for the global objective and the common query
  projection.
- Use the pre-fusion ConvNeXt satellite feature map for dense candidates.
- Gate the reference map before the grounding cross-attention extension.

### LPN

- Preserve the shared ResNet50 encoder.
- Restore local partition/ring pooling descriptors and their global aggregation.
- Use the aggregated query descriptor for matching and the final pre-fusion
  reference map as dense candidates.
- Gate the reference map before grounding cross-attention.

### SMGeo

- Preserve view-specific patch embeddings, the shared Swin stages, GMoE routing,
  query-guided fusion, and the anchor-free heatmap/bbox head.
- Add the explicit click prompt to the query encoding path.
- Expose the existing post-GMoE query vector and pre-conditioning satellite map.
- Replace the residual sigmoid condition with the mandatory matcher gate before
  the anchor-free head.

## Adapter and Training Flow

Adapters pass query images, satellite images, optional geo features, and the
explicit query click to the model. They normalize paper-specific return types
into `GroundingOutput` but do not recompute model features.

The unified training loop performs one backbone forward per image pair:

```text
batch -> adapter/model -> GroundingOutput
      -> existing grounding loss
      -> dense target construction from augmented batch bbox
      -> dense local contrastive loss
      -> paper auxiliary losses
      -> weighted total/backward
```

Only projections, a similarity matrix, and the spatial gate are added. No
shuffled-query second backbone forward is part of normal training.

## Checkpoints and Initialization

Existing config `save_dir` values remain unchanged; no `_dense_v2` directories
are introduced.

New training never reads the existing `last.pth` in those directories. It starts
from configured ImageNet, YOLO, Sample4Geo, or SMGeo paper/backbone pretraining.
The next training run directly overwrites the existing `last.pth`, as explicitly
approved by the user.

Checkpoint payloads contain:

```text
architecture_version = 2
model state
optimizer state
epoch and global step
merged config
query_guard config
training summary
```

Automatic resume is disabled by default. An explicit resume path is accepted
only when its payload has `architecture_version == 2`; old plain state dicts and
old grounding `last.pth` files are rejected as resume checkpoints.

Paper/backbone initialization is configured separately from resume state. A
load report lists loaded, missing, unexpected, and shape-incompatible tensors.

## Training Records

The implementation supplies the training process and records, while the user
runs full experiments.

TensorBoard records per-step and per-epoch:

- total loss;
- bbox, geo, classification, and heatmap losses;
- dense local contrastive loss and its scheduled weight;
- each paper auxiliary loss;
- learning rate;
- query encoder gradient norm;
- mean off-diagonal query cosine;
- matched-versus-deranged local score gap computed from the existing similarity
  matrix without another backbone forward.

Each save directory also contains:

- `train_history.jsonl`: one structured record per epoch;
- `training_summary.json`: model/config identity, initialization report, final
  epoch values, best observed training diagnostics, checkpoint path, start/end
  times, and status;
- `last.pth`: the current checkpoint, overwritten atomically after each epoch.

An interrupted run writes a failure/interruption status to the summary when the
training process can handle the exception or signal normally.

## `test_unify.py` Compatibility

Extend `test_unify.py` choices with:

```text
det, lpn, sample4geo, trogeolite, ocg, smgeo
```

The six models use the current test protocol without changing defaults:

```text
satellite size: 432 x 768
candidate size: 100
test crop ratio: 1.0
seed: 43
heights: 150, 200, 250, 300
angles: 0, 45, 90, 135, 180, 225, 270, 315
include file: /media/data1/feihong/ckpt/include2.json
```

Gallery features are the normalized projected satellite local features. Query
features are normalized projected query embeddings. Retrieval scores use the
same local matcher as training:

```text
score(query, satellite) = max_cell cosine(query, satellite_cell)
```

Paired-reference grounding predictions are decoded through the existing model
adapter. Results retain the current JSON/CSV schema, including Recall@1/5/10,
mean IoU, IoU threshold ratios, uIoU, center distance, and height/angle groups.
Formal experiments and interpretation are outside this implementation scope.

## Error Handling

- Missing configured paper/backbone weights fail with a path-specific error
  unless the config explicitly permits random initialization.
- Dense loss fails clearly when enabled but query/local features are missing,
  have inconsistent batch sizes, or contain non-finite values.
- Feature-grid/image-size mismatches are validated before bbox target mapping.
- Invalid identity tensor shapes fail instead of silently treating positives as
  negatives.
- Resume rejects architecture version mismatches.
- Training records include initialization and checkpoint write failures.

## Verification Strategy

Implementation follows test-first development for deterministic components and
CPU-scale smoke checks. It does not run the user's full training experiments.

Required automated coverage:

- bbox-to-grid center/overlap target construction;
- degenerate/small/clipped bboxes;
- normalized target rows;
- same-identity positive support;
- dense loss ordering for matched and mismatched features;
- mandatory gate shape, normalization, query dependence, and absence of an
  ungated residual;
- nonzero query-encoder gradients from the combined objective;
- explicit click propagation;
- output-shape/finite-value smoke tests for each model using lightweight or
  dependency-injected backbones;
- checkpoint version rejection and training record serialization;
- `test_unify.py` parser/model routing and output-schema compatibility.

## Acceptance Criteria

- All six registry models expose valid pre-fusion query and satellite-local
  features through the common adapter output.
- Dense targets come from the augmented bbox and handle repeated
  `satellite_id` values without false negatives.
- The similarity matrix is reused for both dense InfoNCE and mandatory spatial
  gating.
- The localization path has no ungated satellite feature bypass around the
  mandatory gate.
- Training starts without reading old grounding `last.pth` files and overwrites
  the configured `last.pth` only when saving the new run.
- Training writes TensorBoard, `train_history.jsonl`, and
  `training_summary.json` records with all specified loss and query diagnostics.
- `test_unify.py` accepts all six models and emits its existing comparable
  metric schema.
- Focused automated tests and import/compile checks pass. Full model training and
  official metric experiments remain the user's responsibility.
