from pathlib import Path

import torch
import torch.nn as nn

from retrieval.adapters import FeaturePayload, ForwardMethodAdapter
from retrieval.config import load_config
from retrieval.registry import MODEL_TYPES, get_model_entry
from retrieval.train import _checkpoint_path, train


class TinyGridModel(nn.Module):
    def query_forward(self, pixel_values):
        return pixel_values.mean(dim=(2, 3))

    def ref_forward(self, pixel_values):
        pooled = pixel_values.mean(dim=(2, 3))
        return pooled.unsqueeze(1).expand(pooled.shape[0], 9, pooled.shape[1])

    def text_forward(self, input_ids, attention_mask=None):
        del attention_mask
        return input_ids.float()


def test_registry_exposes_five_retrieval_model_types() -> None:
    assert set(MODEL_TYPES) == {"clip", "siglip", "openclip", "evaclip", "sample_retrieval"}
    assert get_model_entry("sample4geo").canonical_type == "sample_retrieval"


def test_forward_method_adapter_returns_feature_payload() -> None:
    adapter = ForwardMethodAdapter(TinyGridModel(), {"loss": {"use_text_loss": True}})
    payload = adapter.forward(
        {
            "target_pixel_values": torch.ones(2, 3, 2, 2),
            "search_pixel_values": torch.ones(2, 3, 2, 2) * 2,
            "input_ids": torch.ones(2, 3, dtype=torch.long),
            "attention_mask": torch.ones(2, 3, dtype=torch.long),
        },
        torch.device("cpu"),
    )

    assert isinstance(payload, FeaturePayload)
    assert payload.query_feats.shape == (2, 3)
    assert payload.candidate_feats.shape == (2, 9, 3)
    assert payload.text_feats is not None


def test_train_dry_run_returns_save_dir(tmp_path: Path) -> None:
    cfg_path = tmp_path / "train.yaml"
    cfg_path.write_text(
        """
exp_name: dry
save_dir: /tmp/retrieval_dry
model:
  type: siglip
""",
        encoding="utf-8",
    )
    cfg = load_config(str(cfg_path))

    result = train(cfg, dry_run=True)

    assert result["status"] == "dry_run"
    assert result["save_dir"] == "/tmp/retrieval_dry"
    assert result["config_path"] == str(cfg_path)
    assert _checkpoint_path(cfg, "last.pth") == "/tmp/retrieval_dry/last.pth"
