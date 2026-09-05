# unified_siglip_supp single_config 5×3

本目录独立复制 `configs/unified_siglip_supp/single_config/` 全部 YAML 所需的训练和测试代码。卫星图区域由 3×3 改为 5 列×3 行，共 15 个区域；row-major 中心索引为 7。

全部配置依次训练和测试：

```bash
bash exp/run_all.sh
```

只训练和测试 `baseline_sat`：

```bash
bash exp/run.sh
```

checkpoint 写入 `exp/outputs/<exp_name>_5x3/`，测试结果写入 `exp/eval_results/`。

训练和测试采用 3×5 分区、共享 SigLIP 原始 probe 的 attention pooling `Encoder_ada` 两个对照实验：

```bash
bash exp/run_ada.sh
```

`run_ada.sh` 使用 `train_ada.py`；文本只与同一张卫星图的 15 个区域做 InfoNCE，动态 bbox 区域为正样本，其余 14 个区域为负样本。

单独训练并测试 ada 版双塔（bi）与预训练初始化（pre）模型：

```bash
bash exp/run_bi_pre_ada.sh
```

- `model_bi_ada`：无人机和卫星图使用独立的 SigLIP vision encoder，卫星 patch 按 3 行×5 列切分后分别 attention pooling。
- `model_pre_ada`：使用 ada 结构，并从 `outputs/baseline_grounding_full_ada_5x3/last.pth` 初始化。

配置和输出分别为：

- `configs_ada/baseline_bi_ada.yaml` → `outputs/baseline_bi_ada_5x3/`
- `configs_ada/baseline_pre_ada.yaml` → `outputs/baseline_pre_ada_5x3/`

测试结果写入 `eval_results/baseline_bi_ada_5x3.json` 和 `eval_results/baseline_pre_ada_5x3.json`。

消融实验默认使用 seed 42、43、44 依次训练和测试：

```bash
bash exp/run_abla.sh
```

也可以通过位置参数指定一组 seed：

```bash
bash exp/run_abla.sh 7 21 100
```

每个实验的 checkpoint 和评测结果都会添加 `_seed_<seed>` 后缀，避免不同 seed 互相覆盖。
