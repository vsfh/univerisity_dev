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

对应结果为 `baseline_sat_ada_5x3.json` 和 `baseline_wo_input_ids_ada_5x3.json`。
