# AGENTS.md - CVOS-Code Development Guidelines

PyTorch implementation for cross-view object segmentation (ICCV 2025).

## Build Commands

### Training
```bash
# Drone → Satellite
python train.py --emb_size 768 --img_size 1024 --max_epoch 25 --data_root /path/dataset/CVOGL --data_name CVOGL_DroneAerial --beta 1.0 --savename model_droneaerial --gpu 0,1 --batch_size 12 --num_workers 24 --print_freq 50

# Ground → Satellite
python train.py --emb_size 768 --img_size 1024 --max_epoch 25 --data_root /path/dataset/CVOGL --data_name CVOGL_SVI --beta 1.0 --savename model_svi --gpu 0,1 --batch_size 12 --num_workers 8 --print_freq 50
```

### Evaluation
```bash
# Validate
python train.py --val --pretrain saved_models/model_droneaerial_model_best.pth.tar --emb_size 768 --img_size 1024 --data_root /path/dataset/CVOGL --data_name CVOGL_DroneAerial --savename test_model --gpu 0,1 --batch_size 12

# Test
python train.py --test --pretrain saved_models/model_droneaerial_model_best.pth.tar --emb_size 768 --img_size 1024 --data_root /path/dataset/CVOGL --data_name CVOGL_DroneAerial --savename test_model --gpu 0,1 --batch_size 12
```

### Scripts
Use `scripts/run_train_*.sh` and `scripts/run_test_*.sh` for convenience.

### Linting
```bash
pip install ruff black mypy
ruff check . && black . && mypy .
```

## Code Style

### Imports (alphabetical within groups)
```python
# Standard lib
import os, sys, argparse, time, random, logging

# Numerical/ML
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import cv2, torchvision

# Third-party
from einops import rearrange
from shapely.geometry import Polygon
from albumentations import Compose, ToTensor, Normalize

# Local
from dataset.data_loader import RSDataset
from model.TROGeo import TROGeo
from utils.utils import AverageMeter, eval_iou_acc
```

### Conventions
| Component | Convention | Examples |
|-----------|------------|----------|
| Variables/Functions | snake_case | `query_imgs`, `train_epoch` |
| Classes | PascalCase | `RSDataset`, `TROGeo` |
| Constants | UPPER_SNAKE_CASE | `BOX_COLOR` |
| Private methods | `_prefix` | `_forward_impl` |

### Formatting
- Line length: 120 chars
- Indentation: 4 spaces
- No comments unless substantial documentation
- Two blank lines between classes, one between methods

### Error Handling
- Use `assert` for internal invariants and unreachable branches
- Use try/except for I/O operations
- Handle CUDA memory: `torch.cuda.empty_cache(); gc.collect()`

### PyTorch Patterns
```python
# Model definition
class TROGeo(nn.Module):
    def __init__(self):
        super().__init__()
        # components
    
    def forward(self, x):
        return output

# Multi-GPU
model = torch.nn.DataParallel(model).cuda()

# Inference
with torch.no_grad():
    output = model(input)

# Reproducibility
cudnn.benchmark = False
cudnn.deterministic = True
random.seed(args.seed)
np.random.seed(args.seed+1)
torch.manual_seed(args.seed+2)
torch.cuda.manual_seed_all(args.seed+3)
```

### Dataset
- Extend `torch.utils.data.Dataset`
- Use `cv2.imread` with `cv2.COLOR_BGR2RGB`
- ImageNet normalization: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]

### File Structure
```
CVOS-Code/
├── train.py              # Main entry point
├── sam_prompt.py         # SAM integration
├── dataset/data_loader.py
├── model/TROGeo.py       # Main model
├── model/attention.py    # Attention modules
├── model/loss.py         # Loss functions
├── utils/utils.py        # Metrics
└── utils/checkpoint.py   # Checkpoint I/O
```
