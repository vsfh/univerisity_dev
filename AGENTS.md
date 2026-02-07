# AGENTS.md - Guide for Coding Agents

## Repository Overview

This codebase implements image retrieval and localization models (CLIP, SigLIP, EVA-CLIP, OpenCLIP) for drone-to-satellite image matching. Key components:
- **Vision encoders**: CLIP/SigLIP vision models for query (drone) and search (satellite) images
- **Text encoder**: CLIP text model for textual descriptions
- **Retrieval loss**: InfoNCE contrastive loss for cross-modal retrieval
- **Bbox prediction**: MLP-based bbox head for localization
- **Detection heads**: DETR, YOLO, RPN implementations in `bbox/` module

## Build/Test Commands

### Training Scripts
```bash
python train_clip.py           # CLIP model training
python train_siglip.py         # SigLIP model training
python train_evaclip.py        # EVA-CLIP model training
python train_openclip.py       # OpenCLIP model training
python unified_siglip.py       # Combined retrieval + bbox prediction
```

### Feature Extraction & Evaluation
```bash
python feature_extraction.py   # Extract features for evaluation
python download_model.py       # Download pretrained models
```

### Monitoring
```bash
tensorboard --logdir runs/     # Monitor training with TensorBoard
```

## Code Style Guidelines

### Imports Order
1. Standard library (os, json, random, glob, typing)
2. Third-party (torch, PIL, numpy, tqdm, transformers)
3. Local/relative imports

```python
import os
import json
from typing import List, Tuple
from glob import glob

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from tqdm import tqdm

from transformers import AutoModel, AutoTokenizer
```

### Naming Conventions
| Type | Convention | Example |
|------|------------|---------|
| Variables | snake_case | `image_paths`, `crop_size` |
| Classes | PascalCase | `TargetSearchDataset`, `Encoder` |
| Constants | UPPER_SNAKE_CASE | `MODEL_NAME`, `BATCH_SIZE` |
| Functions | snake_case | `compute_bbox_loss` |
| Private methods | `_single_underscore` | `_pool_grid_features` |

### Configuration Block
Place at file top with `--- Configuration ---` separator:
```python
# --- Configuration ---
MODEL_NAME = "google/siglip-base-patch16-224"
CACHE_DIR = "/data/feihong/hf_cache"
NUM_EPOCHS = 40
BATCH_SIZE = 20
LEARNING_RATE = 1e-5
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
PROJECTION_DIM = 768
```

### PyTorch Patterns
```python
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.load_state_dict(torch.load(path, map_location="cpu"), strict=False)
torch.save(model.state_dict(), path)
model.eval()
with torch.no_grad():
    outputs = model(inputs)
tensor = tensor.to(DEVICE)
```

### Dataset Structure
```python
class TargetSearchDataset(Dataset):
    def __init__(self, image_pairs, processor, tokenizer, img_to_text_dict, bbox_dict=None, mode="train"):
        pass

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        return {"target_pixel_values": ..., "bbox": ...}
```

### Model Architecture Patterns
```python
class Encoder(nn.Module):
    def __init__(self, model_name, proj_dim=768):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)
        self.pool = nn.AdaptiveAvgPool2d((3, 3))
        self.text_projector = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.ReLU(),
            nn.Linear(dim * 2, proj_dim),
        )

    def ref_forward(self, pixel_values):
        pass

    def bbox_forward(self, anchor, search):
        pass
```

### Loss Functions
```python
def compute_bbox_loss(pred_bbox: torch.Tensor, target_bbox: torch.Tensor) -> torch.Tensor:
    l1_loss = F.l1_loss(pred_bbox, target_bbox)
    # IoU calculation ...
    return 0.5 * l1_loss + 0.5 * iou_loss.mean()

def info_nce_loss(query_feats, candidate_feats, positive_indices, temperature=0.07):
    query_feats = F.normalize(query_feats, p=2, dim=1)
    candidate_feats = F.normalize(candidate_feats, p=2, dim=1)
    sim_matrix = torch.matmul(query_feats, candidate_feats.T) / temperature
    return F.cross_entropy(sim_matrix, positive_indices)
```

### Error Handling
```python
try:
    query_image = Image.open(query_path).convert("RGB")
except FileNotFoundError as e:
    print(f"Error loading image: {e}. Skipping this item.")
    return self.__getitem__((idx + 1) % len(self))
```

### File I/O
```python
img_to_text_dict = json.load(open(TEXT_FILE, "r"))
np.savez("eval_search_siglip.npz", **res_search)
if os.path.exists(search_path):
    pass
```

### Logging
```python
writer = SummaryWriter(f"runs/{exp_name}")
writer.add_scalar("Loss/train_epoch", avg_loss, epoch)
progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
```

### Bbox Module Structure
```
bbox/
├── __init__.py           # Exports DETRHead, YOLOHead, RPNHead
├── detr_head.py          # DETR-style transformer decoder
├── yolo_head.py          # YOLO anchor-based detection
└── rpn_head.py           # RPN + Fast R-CNN heads
```

### Training Loop Pattern
```python
for epoch in range(NUM_EPOCHS):
    model.train()
    for batch in dataloader:
        bbox_pred, anchor_feats, grid_feats = model.bbox_forward(...)
        retrieval_loss = info_nce_loss(...)
        bbox_loss = compute_bbox_loss(bbox_pred, batch["bbox"])
        loss = retrieval_loss + BBOX_LOSS_WEIGHT * bbox_loss
        loss.backward()
        optimizer.step()
```

### Data Augmentation
- Training: Random crop within bbox constraints
- Test: Fixed center crop `(840, 0, 3000, 2160)`
- Resize: Standard sizes (640x640, 512x512)
