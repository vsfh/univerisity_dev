# 六模型 Query 约束重构实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 恢复六个 grounding 模型的 query 关键路径，并加入增强 bbox 驱动的密集局部对比学习和不可绕过的空间门控。

**Architecture:** 模型在 fusion 前输出 query 向量和 satellite feature map；统一 `DenseQueryMatcher` 生成局部相似度、全 batch 相似度和无 residual 的门控特征。统一 loss 用增强后 bbox 和 `satellite_id` 构造 soft target。训练记录完整 loss 与 query 诊断，正式实验由用户执行。

**Tech Stack:** Python、PyTorch、pytest、TensorBoard、现有 timm/transformers 模型与 `test_unify.py` 辅助函数。

---

### Task 1：密集 matcher 与 bbox soft target

**Files:**
- Create: `grounding/query_guard.py`
- Create: `tests/grounding/test_query_guard.py`

- [ ] 写失败测试：验证 bbox overlap/中心混合、同 ID positives、matched loss、gate 形状和 query 变化。
- [ ] 运行 `pytest -q tests/grounding/test_query_guard.py`，确认因模块不存在失败。
- [ ] 实现并满足以下调用契约：

```python
matcher = DenseQueryMatcher(
    query_dim=4,
    search_dim=4,
    projection_dim=4,
    temperature=0.07,
)
result = matcher(query_embedding, search_features)
assert result.gated_search.shape == search_features.shape
assert result.all_logits.shape == (batch_size, batch_size * grid_h * grid_w)

targets = build_dense_bbox_targets(
    bboxes=bboxes,
    image_wh=(image_width, image_height),
    grid_hw=(grid_h, grid_w),
    identities=satellite_ids,
    center_weight=0.7,
)
loss = dense_local_contrastive_loss(result.all_logits, targets)
gap = deranged_score_gap(result.all_logits, satellite_ids, grid_h * grid_w)
```

- [ ] 运行测试，确认通过。

### Task 2：统一 output、配置与 click 数据

**Files:**
- Modify: `grounding/adapters.py`
- Modify: `grounding/config.py`
- Modify: `dataset.py`
- Create: `tests/grounding/test_contract.py`

- [ ] 写失败测试：`GroundingOutput` 新字段、默认 query_guard 配置、dataset item 包含 `[0.5, 0.5]` click。
- [ ] 运行目标测试，确认失败原因对应缺失字段。
- [ ] 扩展 `GroundingOutput`，增加 query/local/matcher/paper aux 字段。
- [ ] 增加默认 `query_guard` 配置和 loss 权重。
- [ ] dataset item 显式返回 normalized `query_click`。
- [ ] 更新 adapters，将 click/geo 传入模型并接收结构化 dict 输出；保留旧 tuple 解码兼容。
- [ ] 运行目标测试。

### Task 3：恢复 DetGeo、TROGeo、Sample4Geo、LPN 路径

**Files:**
- Modify: `grounding/legacy/ground_cvos.py`
- Modify: `grounding/registry.py`
- Create: `tests/grounding/test_ground_cvos_contract.py`

- [ ] 写使用轻量注入 backbone 的失败测试，逐个检查四模型返回 `pred_anchor/query_embedding/search_local_features/matcher_logits`。
- [ ] DetGeo 改为独立 query ResNet18/reference Darknet，显式 click，matcher gate 后进 bbox head。
- [ ] TROGeo 恢复 click input，matcher gate 后进 cross-attention。
- [ ] Sample4Geo 保持共享 ConvNeXt，增加 symmetric InfoNCE paper aux，matcher gate 后进 cross-attention。
- [ ] LPN 恢复 ring pooling query descriptor，matcher gate 后进 cross-attention。
- [ ] registry 只加载论文/backbone 预训练，不读取 save dir 的 `last.pth`。
- [ ] 运行四模型 contract 测试。

### Task 4：恢复 OCGNet 与 SMGeo 路径

**Files:**
- Modify: `grounding/legacy/train_ocg.py`
- Modify: `grounding/legacy/train_sm.py`
- Modify: `grounding/registry.py`
- Create: `tests/grounding/test_ocg_sm_contract.py`

- [ ] 写失败测试：OCG/SM 输出统一特征，click 改变 matcher，SM 不再有 residual condition bypass。
- [ ] OCG 使用独立 query/reference encoder，保留 GKT/MHCA，在 MHCA 前 mandatory gate。
- [ ] SM query 编码接收 click map，post-GMoE query/pre-condition satellite 进入 matcher，matcher gate 直接进入 anchor-free head。
- [ ] adapters/registry 接收两模型 dict 输出。
- [ ] 运行目标测试。

### Task 5：统一 loss、训练记录与 checkpoint

**Files:**
- Modify: `grounding/losses.py`
- Modify: `grounding/train.py`
- Create: `grounding/training_records.py`
- Create: `tests/grounding/test_training_pipeline.py`

- [ ] 写失败测试：combined loss、warmup、query gradient、旧 checkpoint resume 拒绝、JSONL/summary/checkpoint payload。
- [ ] `GroundingLoss` 增加 `dense` 和 `paper_aux`；`compute_grounding_loss` 计算 dense soft targets 和总损失。
- [ ] train loop 记录 dense weight、query grad norm、query cosine、deranged score gap、paper aux。
- [ ] 实现 `train_history.jsonl`、`training_summary.json` 和临时文件 + `os.replace` 的原子 checkpoint。
- [ ] checkpoint 写入 `architecture_version=2`；显式 resume 只接受 version 2；默认不 resume。
- [ ] 运行目标测试。

### Task 6：新增 ground 统一测试入口

**Files:**
- Create: `test_unify_ground.py`
- Create: `tests/grounding/test_unify_ground.py`
- Read-only: `test_unify.py`

- [ ] 记录 `test_unify.py` 修改前 hash。
- [ ] 写失败测试：六 model choices、checkpoint 路由、FeatureBundle/schema 构建。
- [ ] 新文件引用 `test_unify.py` 的 loader、`FeatureBundle`、grid scorer、汇总和保存函数；独立实现 ground feature extraction/retrieval routing。
- [ ] 支持 `det/lpn/sample4geo/trogeolite/ocg/smgeo` 和显式 config/checkpoint。
- [ ] 运行目标测试并确认 `test_unify.py` hash 未变化。

### Task 7：配置与整体验证

**Files:**
- Modify: `configs/grounding/det.yaml`
- Modify: `configs/grounding/lpn.yaml`
- Modify: `configs/grounding/sample4geo.yaml`
- Modify: `configs/grounding/trogeolite.yaml`
- Modify: `configs/grounding/ocg.yaml`
- Modify: `configs/grounding/smgeo.yaml`

- [ ] 六配置启用相同 query_guard 默认值，保持原 save dir。
- [ ] 将论文预训练字段明确为 `pretrained_checkpoint`，不配置 resume。
- [ ] 运行 `pytest -q tests/grounding`。
- [ ] 运行 `python -m compileall grounding test_unify_ground.py`。
- [ ] 运行六配置 `--dry-run` 或等价构建检查，不启动完整训练。
- [ ] 检查 `git diff --check`，确认 `test_unify.py` 零改动并汇总未执行的正式实验。
