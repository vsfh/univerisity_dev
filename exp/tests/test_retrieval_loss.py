from __future__ import annotations

import importlib.util
from pathlib import Path

import torch
import torch.nn.functional as F


EXP_DIR = Path(__file__).resolve().parents[1]
MODULE_SPEC = importlib.util.spec_from_file_location(
    "exp_retrieval_loss",
    EXP_DIR / "retrieval_loss.py",
)
assert MODULE_SPEC is not None and MODULE_SPEC.loader is not None
RETRIEVAL_LOSS = importlib.util.module_from_spec(MODULE_SPEC)
MODULE_SPEC.loader.exec_module(RETRIEVAL_LOSS)
build_image_retrieval_candidate_mask = (
    RETRIEVAL_LOSS.build_image_retrieval_candidate_mask
)
info_nce_loss = RETRIEVAL_LOSS.info_nce_loss


def _paired_soft_targets(
    batch_size: int,
    num_locations: int,
    *,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    targets = torch.zeros(
        batch_size,
        batch_size * num_locations,
        dtype=dtype,
    )
    rows = torch.arange(batch_size)
    own_columns = rows[:, None] * num_locations + torch.arange(num_locations)[None, :]
    targets[rows[:, None], own_columns] = 0.08 / (num_locations - 1)
    targets[rows, rows * num_locations + rows.remainder(num_locations)] = 0.92
    return targets


def test_fp16_high_similarity_duplicate_satellite_is_finite() -> None:
    batch_size, num_locations, feature_dim = 40, 15, 32
    query = torch.ones(
        batch_size,
        feature_dim,
        dtype=torch.float16,
        requires_grad=True,
    )
    candidates = torch.ones(
        batch_size * num_locations,
        feature_dim,
        dtype=torch.float16,
        requires_grad=True,
    )
    satellite_ids = torch.arange(batch_size)
    satellite_ids[1] = satellite_ids[0]
    mask = build_image_retrieval_candidate_mask(satellite_ids, num_locations)

    loss = info_nce_loss(
        query,
        candidates,
        _paired_soft_targets(batch_size, num_locations),
        candidate_mask=mask,
    )
    loss.backward()

    assert loss.dtype == torch.float32
    assert torch.isfinite(loss)
    assert torch.isfinite(query.grad).all()
    assert torch.isfinite(candidates.grad).all()


def test_masked_candidates_have_zero_gradient() -> None:
    query = torch.tensor([[1.0, 0.0]], requires_grad=True)
    candidates = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5], [-1.0, 0.0]],
        requires_grad=True,
    )
    targets = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    mask = torch.tensor([[True, True, False, False]])

    loss = info_nce_loss(query, candidates, targets, candidate_mask=mask)
    loss.backward()

    assert torch.count_nonzero(candidates.grad[2:]) == 0
    assert torch.count_nonzero(candidates.grad[:2]) > 0


def test_all_rows_with_same_satellite_keep_their_paired_candidates() -> None:
    batch_size, num_locations, feature_dim = 4, 3, 8
    satellite_ids = torch.full((batch_size,), 7)
    mask = build_image_retrieval_candidate_mask(satellite_ids, num_locations)
    expected = torch.zeros_like(mask)
    for row in range(batch_size):
        expected[row, row * num_locations : (row + 1) * num_locations] = True

    assert torch.equal(mask, expected)

    query = torch.ones(batch_size, feature_dim, dtype=torch.float16, requires_grad=True)
    candidates = torch.ones(
        batch_size * num_locations,
        feature_dim,
        dtype=torch.float16,
        requires_grad=True,
    )
    loss = info_nce_loss(
        query,
        candidates,
        _paired_soft_targets(batch_size, num_locations),
        candidate_mask=mask,
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(query.grad).all()
    assert torch.isfinite(candidates.grad).all()


def test_unmasked_soft_targets_match_float32_cross_entropy() -> None:
    torch.manual_seed(3)
    query = torch.randn(3, 7, dtype=torch.float16)
    candidates = torch.randn(12, 7, dtype=torch.float16)
    targets = torch.rand(3, 12)
    targets /= targets.sum(dim=1, keepdim=True)

    actual = info_nce_loss(query, candidates, targets)
    expected_logits = F.normalize(query.float(), dim=1) @ F.normalize(
        candidates.float(),
        dim=1,
    ).T / 0.07
    expected = F.cross_entropy(expected_logits, targets)

    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)
