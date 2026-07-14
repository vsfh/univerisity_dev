# 六模型 Query 约束重构规格

## 目标

重构 `det/lpn/sample4geo/trogeolite/ocg/smgeo`：

1. 恢复各论文中会影响 query 使用的编码、prompt 和融合机制；
2. 用增强后的 satellite bbox 构造密集局部对比标签；
3. 同一相似度矩阵同时用于 InfoNCE 和定位主干的强制空间门控，避免 bbox 分支绕过 query；
4. 保留统一训练入口，新增独立的 `test_unify_ground.py` 测试六个模型；
5. 只提供训练代码和记录能力，完整训练与正式测试由用户执行。

不做六个官方仓库的逐行移植，不加载旧 `ground_*/last.pth` 开始新训练。

## 统一输出接口

扩展 `grounding/adapters.py::GroundingOutput`：

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
    query_embedding: torch.Tensor | None = None        # [B, Cq]
    search_local_features: torch.Tensor | None = None  # [B, Cs, Hg, Wg]
    search_grid_size: tuple[int, int] | None = None
    matcher_logits: torch.Tensor | None = None
    paper_aux_losses: dict[str, torch.Tensor] | None = None
```

约束：

- `query_embedding` 和 `search_local_features` 必须来自 cross-view fusion 之前；
- 禁止用含有 satellite 信息的 fused feature 作为 query；
- adapter 只整理返回值，不重复跑 backbone 或重新提取特征。

## 局部 Matcher 和强制门控

新增通用组件，模型按自身通道数创建投影层：

```text
query feature     -> Linear(Cq, 256) -> L2 normalize
satellite feature -> Conv1x1(Cs, 256) -> L2 normalize
```

单张 satellite 内的局部相似度：

```python
local_logits = einsum("bd,bdhw->bhw", query_proj, search_proj) / temperature
gate = softmax(local_logits.flatten(1), dim=1).view(B, 1, Hg, Wg) * (Hg * Wg)
gated_search = search_features * gate
```

`gated_search` 进入各模型原有 fusion/head。不得添加 `search_features + gated_search` 或其他未门控的 satellite residual。

训练用的 `[B, B*N]` logits 由同一组投影特征计算：

```python
all_logits = query_proj @ search_proj.flatten(2).permute(0, 2, 1).reshape(B * N, D).T
all_logits = all_logits / temperature
```

这样 projection 不是独立辅助分支，定位路径和对比损失共用 matcher。

## 增强 bbox 密集标签

输入必须使用 dataset augment 和 resize 后的 `batch["bbox"]`。

对每个 bbox：

1. 修正 xyxy 顺序并裁剪到 `(W, H)`；
2. 连续映射到 `(Hg, Wg)`；
3. 计算每个 cell 与 bbox 的相交面积；
4. 构造：

```text
target = 0.7 * bbox中心cell one-hot + 0.3 * 归一化cell相交面积
```

小 bbox 或退化 bbox 至少保留中心 cell；每个 query 的最终 target 行归一化为 1。

batch 候选为所有 `B*N` cells。正样本身份优先级：

```text
object_id -> satellite_id -> batch row index
```

当前数据使用 `satellite_id`。同 ID 的其他增强样本只把其 bbox 覆盖 cells 设为正样本，其他 cells 仍为负样本。

损失：

```python
dense_loss = -(target * F.log_softmax(all_logits, dim=1)).sum(dim=1).mean()
```

总损失：

```text
total = grounding_loss
      + scheduled_dense_weight * dense_loss
      + sum(paper_aux_losses)
```

默认配置：

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

前两个 epoch 将 dense weight 从 0 线性增加到 0.2。

## Query click 数据流

`dataset.py` 增加：

```python
"query_click": torch.tensor([0.5, 0.5], dtype=torch.float32)
```

坐标归一化到 `[0, 1]`。当前 drone crop 的目标位于中心，因此使用 `(0.5, 0.5)`；模型内部不再隐式生成中心 prompt。

Det、OCG、TROGeoLite、SMGeo 将坐标转换成 Gaussian click map。Sample4Geo、LPN 不使用 click map。

## 六模型改动

### DetGeo

- query 使用 ResNet18，reference 使用 Darknet，不共享参数；
- query 输入显式 click map；
- matcher 输入：mapped query 全局池化、mapped Darknet feature map；
- matcher gate 后接现有 YOLO bbox head；
- reference Darknet 权重只从配置的 YOLO 预训练文件初始化。

### OCGNet

- query/reference encoder 不共享；
- 保留 GKT、early/late Location Enhancement、MHCA；
- matcher 输入取 GKT 后、MHCA 前的 query/reference feature；
- reference feature 先经过 mandatory gate，再进入 MHCA/fusion。

### TROGeoLite

- 恢复 click-conditioned query 输入；
- 保留共享 Swin 和 CVOPM 风格 cross-attention；
- matcher 输入取 cross-attention 前的 query/reference feature；
- reference feature 先门控，再进入 cross-attention；
- 保留 bbox 和 coordinate 辅助输出。

### Sample4Geo

- 保留共享 ConvNeXt；
- 恢复全局 symmetric InfoNCE，作为 paper auxiliary loss；
- dense matcher 使用 pooled query 和 pre-fusion reference feature map；
- reference feature 先门控，再进入 grounding cross-attention。

### LPN

- 保留共享 ResNet50；
- 恢复局部分块/环形 pooling descriptor；
- 聚合后的 query descriptor 用于 matcher，reference 最后一层 feature map 作为局部候选；
- reference feature 先门控，再进入 grounding cross-attention。

### SMGeo

- 保留 view-specific patch embedding、共享 Swin、GMoE 和 anchor-free head；
- query 编码加入显式 click prompt；
- matcher 使用 post-GMoE query vector 和 pre-conditioning satellite map；
- 移除可绕过的 residual sigmoid condition，改用 mandatory gate。

## 训练入口和 checkpoint

保持现有六个配置的 `save_dir`，不增加 `_dense_v2`。

初始化来源只允许：

- ImageNet/timm backbone；
- DetGeo 配置的 YOLO/Darknet 权重；
- Sample4Geo 论文预训练权重；
- SMGeo 论文预训练权重。

新训练不得自动读取 save dir 中已有的 `last.pth`。保存新训练时直接覆盖该文件。

checkpoint 改成包含：

```python
{
    "architecture_version": 2,
    "model": model.state_dict(),
    "optimizer": optimizer.state_dict(),
    "epoch": epoch,
    "global_step": global_step,
    "config": cfg,
    "training_summary": summary,
}
```

默认禁止 resume。显式 resume 时只接受 `architecture_version == 2`，拒绝旧 plain state dict 和旧 grounding checkpoint。

checkpoint 使用临时文件写入后 `os.replace`，避免中断留下半文件。

## 训练记录

TensorBoard 按 step/epoch 记录：

- total/bbox/geo/cls/heatmap loss；
- dense loss 和当前 dense weight；
- paper auxiliary losses；
- learning rate；
- query encoder gradient norm；
- query batch 的非对角平均 cosine；
- 利用现有 `[B, B*N]` 矩阵计算 matched 与 deranged score gap，不增加 backbone forward。

每个 save dir 写入：

- `train_history.jsonl`：每 epoch 一行；
- `training_summary.json`：模型、配置、初始化加载报告、最后指标、checkpoint、起止时间和状态；
- `last.pth`：每 epoch 原子覆盖。

## `test_unify_ground.py`

`test_unify.py` 不允许修改。新增 `test_unify_ground.py`，支持：

```text
det, lpn, sample4geo, trogeolite, ocg, smgeo
```

新文件尽量直接引用 `test_unify.py` 中已有的：

- dataset/loader 创建函数；
- `FeatureBundle`；
- `score_grid_encoder_query()`；
- `summarize_records()` 和 `group_summaries()`；
- 路径标签、include map、结果保存和指标打印辅助函数。

不能直接复用 `score_retrieval_and_uiou()`：它当前只区分 SigLIP grid scorer 和 `unify_geo` detail scorer，ground model type 会错误进入后者。新文件单独实现 ground retrieval 路由，不改变原函数。

新文件保持 `test_unify.py` 的默认测试参数和 JSON/CSV 指标结构。

特征：

```text
query_feats   = normalized projected query embedding
gallery_feats = normalized projected satellite local features
retrieval score = max_cell cosine(query, satellite_cell)
```

配对 satellite 的 grounding 结果由统一 adapter decode。正式训练、`test_unify_ground.py` 运行和指标分析由用户执行。

## 实现验证

只做重构所需的自动验证，不跑完整实验：

- bbox 到 grid soft target 的数值、归一化、裁剪和退化情况；
- 同 `satellite_id` 的正样本展开；
- matched dense loss 小于 mismatched loss；
- mandatory gate 的形状、均值和 query 依赖；
- combined loss 能向 query encoder 反传非零梯度；
- click 坐标从 dataset 传到模型；
- 六模型轻量 forward/output contract smoke test；
- 旧 checkpoint resume 被拒绝，新 checkpoint/JSONL/summary 可读写；
- `test_unify.py` 文件内容和行为完全不变；
- `test_unify_ground.py` 的 parser、六模型路由和输出 schema 与 `test_unify.py` 对齐。

## 完成条件

- 六模型都输出 fusion 前 query/local features；
- dense target 只使用增强后 bbox；
- InfoNCE 与定位 gate 共用相似度特征；
- 定位路径不存在未门控的 satellite bypass；
- 新训练不读取旧 `last.pth`，保存时按约定直接覆盖；
- 训练记录字段齐全；
- `test_unify.py` 零改动，`test_unify_ground.py` 支持六模型；
- 相关自动测试和 Python 编译检查通过。
