"""Shard only model forward/backward; preserve full-batch losses and optimizer steps."""
import os

import torch
from torch import nn


def require_visible_gpus(count):
    if count != 3:
        raise ValueError("The parallel ablation expects exactly three GPUs.")
    if any(int(os.environ.get(key, "1")) > 1 for key in ("WORLD_SIZE", "LOCAL_WORLD_SIZE")):
        raise ValueError("Use bash ablation_heatmap/run_three_gpu.sh, not torchrun.")
    if not torch.cuda.is_available() or torch.cuda.device_count() != count:
        raise RuntimeError("Expose exactly three CUDA GPUs with CUDA_VISIBLE_DEVICES=0,1,2.")


def wrap_batch_parallel(model, count, device, seed):
    if count == 1:
        return model
    require_visible_gpus(count)
    device = torch.device(device)
    if device.type != "cuda" or device.index not in (None, 0):
        raise ValueError("Gather full-batch outputs on logical cuda:0.")
    # Current Encoder_ada uses LayerNorm/GroupNorm. BatchNorm would change statistics
    # when sharded, so fail explicitly if a future encoder adds it.
    if any(isinstance(module, nn.modules.batchnorm._BatchNorm) for module in model.modules()):
        raise ValueError("BatchNorm is incompatible with this single-batch-equivalent path.")
    model = model.to(device)
    if model.training:
        # Avoid identical dropout masks across replicas; keep the primary RNG untouched.
        for index in range(1, count):
            with torch.cuda.device(index):
                torch.cuda.manual_seed((int(seed) + index) % (2**32))
    # DataParallel gathers nested tensors/None/dicts in original sample order.
    # The unchanged trainer computes InfoNCE and bbox/heatmap losses AFTER gather.
    # With B=32 the shards are 11/11/10; gradients sum into the original parameters.
    parallel = nn.DataParallel(model, device_ids=list(range(count)), output_device=0, dim=0)
    return parallel.train(model.training)
