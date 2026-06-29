import torch

from unified_siglip_supp import build_retrieval_soft_targets


def test_retrieval_soft_targets_share_mass_across_same_satellite() -> None:
    local_indices = torch.tensor([2, 5, 1])
    satellite_ids = torch.tensor([10, 10, 20])

    targets = build_retrieval_soft_targets(local_indices, satellite_ids)

    assert targets.shape == (3, 27)
    assert torch.allclose(targets.sum(dim=1), torch.ones(3))
    assert targets[0, 2] == torch.tensor(0.92)
    assert targets[1, 14] == torch.tensor(0.92)

    same_sat_other_sample_cols = torch.arange(9, 18)
    assert torch.all(targets[0, same_sat_other_sample_cols] > 0)

    different_sat_cols = torch.arange(18, 27)
    assert torch.all(targets[0, different_sat_cols] == 0)


def test_retrieval_soft_targets_matches_original_single_satellite_behavior() -> None:
    local_indices = torch.tensor([4, 0])
    satellite_ids = torch.tensor([10, 20])

    targets = build_retrieval_soft_targets(local_indices, satellite_ids)

    assert targets[0, 4] == torch.tensor(0.92)
    own_other_cols = [idx for idx in range(9) if idx != 4]
    assert torch.allclose(targets[0, own_other_cols], torch.full((8,), 0.01))
    assert torch.all(targets[0, 9:18] == 0)
