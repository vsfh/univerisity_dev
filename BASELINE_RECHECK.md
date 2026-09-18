# 基线重新训练和测试

在远端仓库根目录运行：

```bash
cd /media/data1/feihong/univerisity_dev
bash run_baselines.sh
```

前台顺序训练、测试 11 个模型：检索 CLIP、OpenCLIP、EVA-CLIP、SigLIP、Sample4Geo；定位 OCG、DetGeo、Sample4Geo、SMGeo、LPN、TROGeoLite。采用各自已有 YAML 超参数（默认 20 epochs），不验证、不续训、不选择 best；每个模型训练完只测试自己的 last.pth。一个模型失败会记录错误并继续其他模型；全部结束后只要有失败，脚本退出码就是 1。Ctrl+C 中断当前任务。

所有持久实验输出保存在 ./outputs/baseline_recheck：checkpoints/ 存 last.pth，configs/ 存实际配置，logs/ 存训练测试日志，results/ 存完整指标，runtime/ 存 TensorBoard 等运行产物，summary.csv 和 summary.json 汇总结果。指标保持原评测器的单位。每次运行覆盖整个 baseline_recheck 目录，其他 outputs 子目录不受影响。临时进程通信 socket 使用 /tmp，因为当前 outputs 挂载不支持此类 socket。

默认使用可见的第 0 张 GPU，可通过 CUDA_VISIBLE_DEVICES=1 bash run_baselines.sh 指定物理 GPU；PYTHON 环境变量可指定 Python。离线读取服务器上已有的模型缓存，不自动下载权重。缺失预训练文件或缓存会在对应模型日志中报告。

仅重跑指定模型也可以，例如 bash run_baselines.sh retrieval_clip grounding_ocg；这同样覆盖整个 baseline_recheck 目录。

## 修改范围

1. 定位的 anchor confidence 在训练 loss 和测试 decode 中使用同一个函数。use_heatmap=false 时两边都不融合；开启时两边都使用配置中的 heatmap_confidence_weight，保留 detached heatmap 的训练语义。anchor-free 分支不变。
2. CLIP/OpenCLIP/EVA 的额外中心裁剪改为整幅缩放，保留原归一化和原网络要求的输入规格；训练和测试共用 processor 构建入口。默认这些模型的张量是 224×224，并不是 768×432。这消除了视野与标签错配，但不代表不同编码器的实际像素分辨率相同，整幅缩放也会改变宽高比。
3. 训练和测试直接导入 train_ada 使用的 exp.dataset.ShiftedSatelliteDroneDataset，数据集源码不变。该类输出 3×5 index，既有基线网络输出 3×3 描述符，因此只在检索 loss 中从同一 bbox 中心换算 3×3 标签，避免直接误用 15 区域标签。

CLIP token 切片、所有网络结构、原训练超参数、数据划分及候选集采样逻辑保持原样。EVA 原配置未指定预训练权重的问题也没有扩大范围修改，本脚本不保证该配置已经预训练。UnifyGeo/TransGeo 不在本次两项修复对应的重测列表中。本次修复不能单独证明所有基线已完全公平，也不保证指标必然提升。

代码检查仅使用合成图像、合成张量和模拟子进程；没有自动启动真实训练或测试。
