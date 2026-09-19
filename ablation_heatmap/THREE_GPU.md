# 同一个 heatmap 消融实验使用三张卡

在另一台 Linux 机器激活已有训练环境，进入 Git 同步后的仓库根目录，然后运行：

```bash
bash ablation_heatmap/run_three_gpu.sh
```

默认使用当前环境的 `python` 和物理 GPU 0、1、2。也可以指定：

```bash
CUDA_VISIBLE_DEVICES=1,2,3 PYTHON=/path/to/env/bin/python bash ablation_heatmap/run_three_gpu.sh
```

这是前台任务，不使用 nohup，不需要 prepare。每个 λ 的训练同时使用三张卡，完成后用三卡提取测试特征、统一计算指标，再开始下一个 λ。实验之间顺序执行。只测试 `last.pth`，保留原训练器不执行验证的行为。失败会停止；Ctrl+C 中断；重新运行从头训练。

参数直接沿用远端当前 `study.py` 和原 baseline YAML：λ_heatmap 为 0.01、0.05、0.1、0.2、0.4，seed 为 42，λ_box 固定为 0.5，每组 20 epochs。没有恢复成早先的 15 组设置。单 seed 的标准差为空。

## 与单卡的对应关系

| 设置 | 现有单卡 | 新三卡入口 |
|---|---|---|
| 单次前向的全局 batch | 32 | 32，分成 11 / 11 / 10 |
| 梯度累积 | 2 | 2 |
| 常规 optimizer step 的有效 batch | 64 | 64 |
| 每次 InfoNCE 的 query / 区域候选数 | 32 / 480 | 合并三卡特征后仍为 32 / 480 |
| loss 权重 | 0.5 L_retrieval + 0.5 L_box + λ_heatmap L_heatmap | 相同 |
| 学习率、scheduler、数据顺序、增强和 worker 设置 | 原配置 | 原配置 |

采用单进程 `DataParallel`：只切分模型前向和反向，所有输出按照原样本顺序合并到逻辑 cuda:0，随后运行原训练器中的完整 loss。没有对三份局部 InfoNCE 求平均，没有扩大 batch、补重复样本或改变梯度累积。模型、数据集类、损失函数及 checkpoint 参数名不变。Accelerate 仍管理 AMP、梯度累积和 checkpoint 解包，但不是三进程 DDP。

这里的“一样”指训练目标和 batch 语义一致，不是逐位一致或保证最终指标完全相同。LoRA dropout 使用各卡独立随机流，矩阵运算尺寸、浮点归约顺序和不同 GPU/CUDA 版本都可能造成差异。三卡耗时也不保证缩短到单卡的三分之一，主卡仍负责合并特征和 loss；主卡显存负担会较大。

## 输出和迁移

所有持久实验产物位于 `./outputs/heatmap_lambda_box_0p5_3gpu/`，包括 configs、checkpoints、results、logs、runtime、cache、summary.csv/json 和 parallel_config.json。再次运行只覆盖这个三卡目录，保留原单卡 `heatmap_lambda_box_0p5_42` 和其他实验目录。临时进程通信仍使用 /tmp。

Git 只同步代码。另一台机器还需具有相同数据和模型缓存，现有代码路径为 `/media/data1/feihong/` 下的 image_2048、img_test_2、drone_img、ckpt 和 hf_cache；ckpt 中需有原划分文件、bbox_test_2.json、include2.json。若实际挂载路径不同，可事先建立相应目录映射/软链接，保持原数据内容和划分。脚本继续离线读取缓存。

本次应同步：`ablation_heatmap/study.py`、`three_gpu.py`、`run_three_gpu.sh`、`THREE_GPU.md`、`test_study.py`、`test_three_gpu.py`，以及 `exp/batch_parallel.py`、`exp/train_ada.py`、`exp/test.py`。新增文件需加入 Git；只提交已跟踪文件不会带上新脚本。不要提交 outputs。

开发检查仅使用 CPU 合成张量和模拟训练/测试调用。没有启动真实训练、测试，也未执行三卡性能或最终指标对比。
