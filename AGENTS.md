# AGENTS.md - Guide for Coding Agents

## Build/Test Commands

### Training Scripts
Run training directly with Python:
```bash
python train_clip.py      # CLIP model training
python train_siglip.py    # SigLIP model training  
python train_evaclip.py   # EVA-CLIP model training
python train_openclip.py  # OpenCLIP model training
python finetune.py        # DINOv2 fine-tuning
```

### Feature Extraction
```bash
python feature_extraction.py
python download_model.py
```

### Inference/Testing
Scripts are standalone and run directly. No standard test framework (pytest) is used.
Monitor training with TensorBoard:
```bash
tensorboard --logdir runs/
```

## Code Style Guidelines

### Imports
- Standard library imports first (os, json, random, glob)
- Third-party imports second (torch, PIL, numpy, tqdm)
- Relative/local imports last
- Group imports logically with blank lines between groups

Example:
```python
import os
import json
import random
from glob import glob

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from tqdm import tqdm

from transformers import CLIPModel, CLIPProcessor
```

### Formatting & Naming
- Use 4-space indentation
- Variable names: snake_case (e.g., `image_paths`, `crop_size`)
- Class names: PascalCase (e.g., `TargetSearchDataset`, `Encoder`)
- Constants: UPPER_SNAKE_CASE at top of file (e.g., `MODEL_NAME`, `BATCH_SIZE`, `DEVICE`)
- Function names: snake_case (e.g., `get_data_loaders`, `train_one_epoch`)
- Private methods: single underscore prefix (e.g., `_process_batch`)

### Configuration
Place all configuration constants at the top of files, separated with comments:
```python
# --- Configuration ---
MODEL_NAME = "openai/clip-vit-base-patch32"
CACHE_DIR = "/data/feihong/hf_cache"
NUM_EPOCHS = 40
BATCH_SIZE = 20
LEARNING_RATE = 1e-5
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
PROJECTION_DIM = 768
```

### PyTorch Patterns
- Use `with torch.no_grad():` for inference/evaluation
- Use `.to(DEVICE)` after moving tensors to device
- Load models with `torch.load(model_path, map_location='cpu')` for safety
- Save with `torch.save(model.state_dict(), path)`
- Set model to train/eval mode explicitly: `model.train()` / `model.eval()`

### Datasets & DataLoaders
- Inherit from `torch.utils.data.Dataset`
- Implement `__len__` and `__getitem__`
- Use `num_workers=4` for DataLoader
- Handle FileNotFoundError in `__getitem__` gracefully

### Model Architecture
- Inherit from `nn.Module`
- Call `super().__init__()` in __init__
- Use `nn.Sequential` for simple layer stacks
- Projection heads typically: Linear -> ReLU -> Linear

### Loss Functions
- Define custom loss functions with clear docstrings
- Use InfoNCE/contrastive loss for retrieval tasks
- Normalize features before computing similarity: `F.normalize(feats, p=2, dim=1)`

### Logging & Progress
- Use `tqdm` for progress bars
- Use `torch.utils.tensorboard.SummaryWriter` for logging
- Log both batch-level and epoch-level metrics
- Print key metrics to console

### Error Handling
- Wrap model loading in try/except with descriptive error messages
- Handle missing files gracefully with FileNotFoundError
- Use assertions for shape validation in critical paths

### File I/O
- Use absolute paths (data paths are configured as constants)
- Load JSON with `json.load(open(path, 'r'))`
- Save NPZ files: `np.savez(path, **dict)`
- Use `os.path.exists()` before file operations

### Environment Variables
Set cache directories via environment variables before imports:
```python
os.environ['HF_HOME'] = '/path/to/cache'
os.environ['HF_CACHE'] = '/path/to/cache'
```

### Comments
- Keep comments minimal and concise
- Comment disabled code blocks with context
- Use inline comments for complex calculations

### Device Management
- Check CUDA availability: `torch.cuda.is_available()`
- Select specific GPU: `DEVICE = "cuda:0"` or `"cuda:2"`
- Use conditional: `DEVICE = "cuda:2" if torch.cuda.is_available() else "cpu"`

### Data Augmentation
- Random cropping during training, fixed crops during test
- Center crop for evaluation: `image.crop((left, top, right, bottom))`
- Resize to standard dimensions: `.resize((640, 640))`

### Early Stopping
Track validation loss and stop after N epochs without improvement:
```python
if not_update > 5:
    print("Validation loss not improving. Stopping early.")
    break
```
