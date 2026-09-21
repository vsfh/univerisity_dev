# Heatmap lambda ablation: shared output

Both one-GPU and three-GPU launchers now use **outputs/heatmap_lambda_box_0p5/**.
On the currently configured servers, outputs is the SAME shared filesystem, so no rsync is required between them.

Three GPUs, seeds 42 and 43 (run in the activated training environment):

```bash
CUDA_VISIBLE_DEVICES=0,1,2 bash ablation_heatmap/run_three_gpu.sh --seeds 42 43
```

One GPU, seed 44:

```bash
CUDA_VISIBLE_DEVICES=1 bash ablation_heatmap/run.sh --seeds 44
```

The defaults are also 42/43 for run_three_gpu.sh and 44 for run.sh. Explicit --seeds is recommended when splitting work. PYTHON=/path/to/python can select an environment. Within each launcher experiments remain sequential; three GPUs cooperate on the same experiment.

- Outer lambda_heatmap: 0.01, 0.05, 0.1, 0.2, 0.4. lambda_box: 0.5.
- Same original 20 epochs, global batch 32, accumulation 2, learning rate and full-batch InfoNCE.
- The three-GPU path gathers outputs before loss computation; checkpoints retain ordinary parameter names.
- Skip completed train+test results. If last.pth is complete, only finish missing evaluation. Otherwise retrain that experiment from epoch 1.
- Existing scientific config mismatches stop rather than overwriting results. Device count alone does not force a completed experiment to rerun.
- Each result/checkpoint/config name contains BOTH lambda and seed, e.g. heatmap_0p1_seed_43.
- Per-experiment mkdir locks prevent duplicate writers across hosts. Summary updates use a separate shared lock and atomic replacement.
- summary.csv/json always include all available seeds 42,43,44, regardless of which seeds this worker executes. n_expected=3; missing seeds are not zero measurements.
- metadata/ records original training GPU count, legacy source and subsequent evaluation mode.
- Logs append. Ctrl+C stops the current foreground task. No training is launched by deployment or merging.

Existing directories were consolidated using:

```bash
python ablation_heatmap/merge_outputs.py --apply
```

This only merges artifacts, never trains/tests. Original folders remain intact; large checkpoints and logs use **relative symlinks** to avoid copies. **Do not delete the old _42/_3gpu directories while links reference them.** The report is outputs/heatmap_lambda_box_0p5/merge_report.json. Missing configs are recovered only when effective_config.json proves matching parameters. Conflicting completed metrics for the same lambda/seed stop the merge for manual selection.

Different-seed jobs can run concurrently. If a process was killed without cleanup, inspect locks/<run>/owner.json and confirm that host/process is no longer running before removing that stale lock directory. Never remove another active worker's lock.

If future servers do not share outputs, using the same relative directory alone does not synchronize files; explicitly synchronize artifacts and rebuild the global summary. No automatic cross-server transfer is configured.
