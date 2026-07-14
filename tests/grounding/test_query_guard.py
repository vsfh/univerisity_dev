import torch

from grounding.query_guard import (
    DenseQueryMatcher,
    build_dense_bbox_targets,
    dense_local_contrastive_loss,
    deranged_score_gap,
)


def test_bbox_target_mixes_center_and_cell_overlap():
    targets = build_dense_bbox_targets(
        bboxes=torch.tensor([[25.0, 25.0, 75.0, 75.0]]),
        image_wh=(100, 100),
        grid_hw=(2, 2),
        identities=torch.tensor([7]),
        center_weight=0.7,
    )

    expected = torch.tensor([[0.075, 0.075, 0.075, 0.775]])
    torch.testing.assert_close(targets, expected)
    torch.testing.assert_close(targets.sum(dim=1), torch.ones(1))


def test_bbox_target_shares_positive_mass_across_same_identity_rows():
    targets = build_dense_bbox_targets(
        bboxes=torch.tensor(
            [
                [0.0, 0.0, 50.0, 50.0],
                [50.0, 50.0, 100.0, 100.0],
                [0.0, 50.0, 50.0, 100.0],
            ]
        ),
        image_wh=(100, 100),
        grid_hw=(2, 2),
        identities=torch.tensor([3, 3, 9]),
    )

    assert targets.shape == (3, 12)
    torch.testing.assert_close(targets.sum(dim=1), torch.ones(3))
    torch.testing.assert_close(targets[0, :4].sum(), torch.tensor(0.5))
    torch.testing.assert_close(targets[0, 4:8].sum(), torch.tensor(0.5))
    torch.testing.assert_close(targets[0, 8:].sum(), torch.tensor(0.0))
    torch.testing.assert_close(targets[2, :8].sum(), torch.tensor(0.0))
    torch.testing.assert_close(targets[2, 8:].sum(), torch.tensor(1.0))


def test_small_degenerate_bbox_keeps_one_positive_cell():
    targets = build_dense_bbox_targets(
        bboxes=torch.tensor([[120.0, -10.0, 120.0, -10.0]]),
        image_wh=(100, 100),
        grid_hw=(2, 2),
        identities=torch.tensor([1]),
    )

    torch.testing.assert_close(targets.sum(dim=1), torch.ones(1))
    assert torch.count_nonzero(targets) == 1
    assert targets[0, 1] == 1


def test_matcher_gate_depends_on_query_and_has_no_residual_path():
    matcher = DenseQueryMatcher(
        query_dim=2,
        search_dim=2,
        projection_dim=2,
        temperature=1.0,
    )
    with torch.no_grad():
        matcher.query_projection.weight.copy_(torch.eye(2))
        matcher.search_projection.weight.copy_(torch.eye(2).view(2, 2, 1, 1))

    search = torch.tensor([[[[1.0, 0.0]], [[0.0, 1.0]]]])
    first = matcher(torch.tensor([[1.0, 0.0]]), search)
    second = matcher(torch.tensor([[0.0, 1.0]]), search)

    assert first.gated_search.shape == search.shape
    assert first.all_logits.shape == (1, 2)
    torch.testing.assert_close(first.spatial_gate.mean(), torch.tensor(1.0))
    assert not torch.allclose(first.spatial_gate, second.spatial_gate)
    torch.testing.assert_close(first.gated_search, search * first.spatial_gate)


def test_dense_loss_prefers_matched_logits_and_backpropagates_to_query():
    query = torch.tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True)
    search = torch.tensor(
        [
            [[[1.0]], [[0.0]]],
            [[[0.0]], [[1.0]]],
        ]
    )
    matcher = DenseQueryMatcher(2, 2, projection_dim=2, temperature=0.1)
    with torch.no_grad():
        matcher.query_projection.weight.copy_(torch.eye(2))
        matcher.search_projection.weight.copy_(torch.eye(2).view(2, 2, 1, 1))

    output = matcher(query, search)
    matched_targets = torch.eye(2)
    mismatched_targets = torch.flip(matched_targets, dims=[1])
    matched_loss = dense_local_contrastive_loss(output.all_logits, matched_targets)
    mismatched_loss = dense_local_contrastive_loss(output.all_logits, mismatched_targets)

    assert matched_loss < mismatched_loss
    matched_loss.backward()
    assert query.grad is not None
    assert torch.isfinite(query.grad).all()


def test_deranged_score_gap_uses_different_identity_blocks():
    logits = torch.tensor(
        [
            [5.0, 4.0, 1.0, 0.0],
            [0.0, 1.0, 4.0, 5.0],
        ]
    )
    gap = deranged_score_gap(
        all_logits=logits,
        identities=torch.tensor([10, 20]),
        num_locations=2,
    )
    torch.testing.assert_close(gap, torch.tensor(4.0))


if __name__ == "__main__":
    tests = [value for name, value in globals().items() if name.startswith("test_") and callable(value)]
    for test in tests:
        test()
    print(f"passed {len(tests)} tests")
