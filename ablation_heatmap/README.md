# Heatmap ablation: one command

For one experiment using three GPUs together, run `bash ablation_heatmap/run_three_gpu.sh`.
See [THREE_GPU.md](THREE_GPU.md) for batch equivalence, migration, and output paths.

On the server, from /media/data1/feihong/univerisity_dev:

```bash
bash ablation_heatmap/run.sh
```

Runs in the foreground on GPU 0 using /home/feihong/miniconda3/bin/python.
No prepare step, arguments, resume, or completed-run skipping.
Every invocation replaces **only outputs/heatmap_lambda_box_0p5_42/** and
trains the current 5 experiments from scratch. Stop any previous runner before restarting.
Other output directories are untouched.

- Fixed lambda_box: 0.5.
- Outer lambda_heatmap: 0.01, 0.05, 0.1, 0.2, 0.4.
- Seed: 42; 20 epochs per experiment from the existing baseline.
- Loss: 0.5 L_retrieval + 0.5 L_box + lambda_heatmap L_heatmap.
- The exp trainer nests heatmap loss inside the box coefficient, so generated
  HEATMAP_LOSS_WEIGHT is lambda_heatmap / 0.5.
- USE_HEATMAP_LOSS remains true even at zero; confidence fusion is unchanged.
- Each training run is immediately followed by exp/test.py on last.pth.
  Evaluation uses 100 candidates, crop ratio 1.0, and the matching training seed.
- On error, execution stops and preserves the log; rerunning starts everything over.
- Ctrl+C stops the foreground job; no background launcher is used.

Training and evaluation output is shown in the terminal and saved under logs/.
Configs, checkpoints, evaluation JSON, TensorBoard events (runtime/runs/), and caches
are all under outputs/heatmap_lambda_box_0p5_42/. Ephemeral DataLoader IPC uses /tmp
because the server's outputs mount does not support Unix sockets.
summary.csv and summary.json update after each evaluation, with completed counts,
means and sample standard deviations. Metrics use the original test.py units;
an empty value means no result / insufficient seeds, not zero.

Optional environment overrides:

```bash
CUDA_VISIBLE_DEVICES=1 PYTHON=/path/to/python bash ablation_heatmap/run.sh
```

The original exp training, model, dataset, and evaluation implementations are reused.
The exp/run_heatmap_ablation.sh shortcut invokes this same script.
Code generation/deployment does not launch training.
