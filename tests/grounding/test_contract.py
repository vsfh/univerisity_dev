import torch
import torch.nn as nn

from dataset import build_default_query_click
from grounding.adapters import GroundingOutput, LegacyAnchorAdapter
from grounding.config import DEFAULT_CONFIG


class DictGroundingModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.received_click = None

    def forward(self, query, search, query_click=None, geo=None):
        self.received_click = query_click
        batch_size = query.shape[0]
        return {
            "pred_anchor": query.new_zeros((batch_size, 45, 2, 2)),
            "query_embedding": query.new_ones((batch_size, 4)),
            "search_local_features": search.new_ones((batch_size, 4, 2, 2)),
            "matcher_logits": query.new_zeros((batch_size, batch_size * 4)),
            "paper_aux_losses": {"retrieval": query.new_tensor(0.25)},
        }


def test_grounding_output_exposes_query_guard_fields():
    output = GroundingOutput(
        device=torch.device("cpu"),
        image_wh=(8, 8),
        query_embedding=torch.ones(2, 4),
        search_local_features=torch.ones(2, 4, 2, 2),
        search_grid_size=(2, 2),
        matcher_logits=torch.ones(2, 8),
        paper_aux_losses={"retrieval": torch.tensor(1.0)},
    )

    assert output.query_embedding.shape == (2, 4)
    assert output.search_grid_size == (2, 2)
    assert output.paper_aux_losses["retrieval"].item() == 1.0


def test_default_config_enables_query_guard():
    query_guard = DEFAULT_CONFIG["query_guard"]
    assert query_guard == {
        "enabled": True,
        "projection_dim": 256,
        "temperature": 0.07,
        "weight": 0.2,
        "warmup_epochs": 2,
        "center_weight": 0.7,
        "identity_key": "object_id",
    }


def test_default_query_click_is_explicit_center_coordinate():
    click = build_default_query_click()
    torch.testing.assert_close(click, torch.tensor([0.5, 0.5]))
    assert click.dtype == torch.float32


def test_legacy_adapter_passes_click_and_unpacks_mapping_output():
    model = DictGroundingModel()
    adapter = LegacyAnchorAdapter(
        model,
        {"model": {"use_angle": False}},
    )
    batch = {
        "target_pixel_values": torch.zeros(2, 3, 4, 4),
        "search_pixel_values": torch.zeros(2, 3, 8, 8),
        "query_click": torch.tensor([[0.25, 0.75], [0.5, 0.5]]),
    }

    output = adapter.forward(batch, torch.device("cpu"))

    torch.testing.assert_close(model.received_click, batch["query_click"])
    assert output.pred_anchor.shape == (2, 45, 2, 2)
    assert output.query_embedding.shape == (2, 4)
    assert output.search_local_features.shape == (2, 4, 2, 2)
    assert output.search_grid_size == (2, 2)
    assert output.paper_aux_losses["retrieval"].item() == 0.25


if __name__ == "__main__":
    tests = [value for name, value in globals().items() if name.startswith("test_") and callable(value)]
    for test in tests:
        test()
    print(f"passed {len(tests)} tests")
