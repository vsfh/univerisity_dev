import pytest
import torch

from retrieval.losses import (
    RetrievalLossResult,
    compute_retrieval_loss,
    infer_granularity,
    prepare_candidate_targets,
)


def test_infer_granularity_detects_grid_and_global() -> None:
    assert infer_granularity(torch.zeros(2, 9, 4), "auto") == "grid"
    assert infer_granularity(torch.zeros(2, 4), "auto") == "global"
    assert infer_granularity(torch.zeros(2, 9, 4), "grid") == "grid"
    assert infer_granularity(torch.zeros(2, 4), "global") == "global"


def test_infer_granularity_rejects_unsupported_token_count() -> None:
    with pytest.raises(ValueError, match="expected 9 grid locations"):
        infer_granularity(torch.zeros(2, 16, 4), "auto")


def test_prepare_grid_targets_uses_unified_soft_defaults() -> None:
    batch = {"index": torch.tensor([2, 5]), "satellite_id": torch.tensor([7, 7])}
    flat_candidates, targets, granularity = prepare_candidate_targets(
        torch.randn(2, 9, 4),
        batch,
        requested_granularity="auto",
    )

    assert granularity == "grid"
    assert flat_candidates.shape == (18, 4)
    assert targets.shape == (2, 18)
    assert targets[0, 2] == torch.tensor(0.92)
    assert targets[1, 14] == torch.tensor(0.92)
    assert torch.allclose(targets.sum(dim=1), torch.ones(2))


def test_prepare_global_targets_ignores_index() -> None:
    batch = {"index": torch.tensor([8, 8]), "satellite_id": torch.tensor([7, 7])}
    flat_candidates, targets, granularity = prepare_candidate_targets(
        torch.randn(2, 4),
        batch,
        requested_granularity="auto",
    )

    assert granularity == "global"
    assert flat_candidates.shape == (2, 4)
    assert torch.equal(targets, torch.tensor([0, 1]))


def test_compute_retrieval_loss_returns_image_only_by_default() -> None:
    result = compute_retrieval_loss(
        query_feats=torch.eye(2, 4),
        candidate_feats=torch.eye(2, 4),
        batch={"index": torch.tensor([8, 8])},
        temperature=0.07,
        requested_granularity="auto",
        text_feats=None,
        use_text_loss=False,
    )

    assert isinstance(result, RetrievalLossResult)
    assert result.granularity == "global"
    assert result.text_loss is None
    assert result.total.item() >= 0.0
