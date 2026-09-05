from __future__ import annotations

import subprocess
import sys
from pathlib import Path


EXP_DIR = Path(__file__).resolve().parents[1]


def test_text_pooler_stops_at_region_but_other_losses_do_not() -> None:
    script = r"""
import torch
import torch.nn as nn
from model import Encoder_ada


class ToyPooling(nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = nn.Dropout(0.2)
        self.proj = nn.Linear(4, 4, bias=False)

    def forward(self, region, output_tokens):
        return self.proj(self.dropout(region.mean(dim=1, keepdim=True)))


encoder = Encoder_ada.__new__(Encoder_ada)
nn.Module.__init__(encoder)
encoder.attnPooling = ToyPooling()
sat_feats = torch.randn(2, 15, 4, requires_grad=True)

torch.manual_seed(7)
image_grid_without_text, _, _ = encoder._pool_satellite_regions(
    sat_feats,
    3,
    5,
    build_text_align_features=False,
)
torch.manual_seed(7)
image_grid, text_grid, sat_2d = encoder._pool_satellite_regions(
    sat_feats,
    3,
    5,
    build_text_align_features=True,
)
assert torch.equal(image_grid, image_grid_without_text)

text_sat_grad, text_pool_grad = torch.autograd.grad(
    text_grid.square().sum(),
    (sat_feats, encoder.attnPooling.proj.weight),
    allow_unused=True,
    retain_graph=True,
)
image_sat_grad_without_text = torch.autograd.grad(
    image_grid_without_text.square().sum(), sat_feats, retain_graph=True
)[0]
image_sat_grad = torch.autograd.grad(
    image_grid.square().sum(), sat_feats, retain_graph=True
)[0]
bbox_sat_grad = torch.autograd.grad(sat_2d.square().sum(), sat_feats)[0]

assert text_sat_grad is None
assert text_pool_grad.abs().sum() > 0
assert torch.equal(image_sat_grad, image_sat_grad_without_text)
assert image_sat_grad.abs().sum() > 0
assert bbox_sat_grad.abs().sum() > 0
"""
    subprocess.run([sys.executable, "-c", script], cwd=EXP_DIR, check=True)
