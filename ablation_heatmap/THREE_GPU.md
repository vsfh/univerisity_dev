# 三卡运行同一个实验

在训练环境中、仓库根目录运行：

```bash
CUDA_VISIBLE_DEVICES=0,1,2 bash ablation_heatmap/run_three_gpu.sh --seeds 42 43
```

另一台单卡运行：

```bash
CUDA_VISIBLE_DEVICES=1 bash ablation_heatmap/run.sh --seeds 44
```

两个入口统一输出到 ./outputs/heatmap_lambda_box_0p5/。目前两台服务器的 outputs 挂载到同一个共享盘，因此结果直接汇总，不需要 rsync。省略 --seeds 时，三卡默认 42、43，单卡默认 44；已有完成结果继续跳过。

每个 λ 的训练由三张卡共同完成，随后测试 last.pth，再继续下一组。λ_heatmap 为 0.01、0.05、0.1、0.2、0.4，λ_box 为 0.5，每组 20 epochs。

| 设置 | 单卡 | 三卡 |
|---|---|---|
| 全局 batch | 32 | 32，分成 11 / 11 / 10 |
| 梯度累积 | 2 | 2 |
| 常规 optimizer step 有效 batch | 64 | 64 |
| 每次 InfoNCE query / 区域候选数 | 32 / 480 | 合并后仍为 32 / 480 |
| loss | 0.5 L_retrieval + 0.5 L_box + λ_heatmap L_heatmap | 相同 |

采用单进程 DataParallel。只切分模型前向/反向，输出按样本顺序汇集后运行原 loss；未扩大 batch，也没有对三份局部 InfoNCE 求平均。数据集和网络结构不变。浮点归约和各卡 dropout 随机流不同，因此不保证与单卡逐位一致或最终指标完全相同，也不保证三倍加速。

已合并的单卡结果保留其原始来源，不会伪装成三卡重跑的结果。metadata/ 保存训练 GPU 数和来源。按 seed 划分任务后可同时运行；每个实验和总汇总分别加共享锁，汇总始终读取 42、43、44 的已完成结果。没有完整 last.pth 的中断训练从头开始。

旧目录保留。统一目录中的大权重和日志通过相对符号链接复用旧文件，**不要删除被链接的旧目录**。汇总和合并报告在统一目录内。详细规则见 README.md。

Git 只同步代码；数据、划分、候选集、模型缓存和 outputs 挂载需由机器环境提供。gpu-server2 现有 /data/ 路径修改应保留。本次合并和代码检查没有启动真实训练或测试。
