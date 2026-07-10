from pathlib import Path

import pytest

from retrieval.config import load_config


def test_retrieval_config_defaults_keep_text_loss_off(tmp_path: Path) -> None:
    cfg_path = tmp_path / "siglip.yaml"
    cfg_path.write_text(
        """
exp_name: siglip
save_dir: /tmp/retrieval_siglip
model:
  type: siglip
""",
        encoding="utf-8",
    )

    cfg = load_config(str(cfg_path))

    assert cfg["config_path"] == str(cfg_path)
    assert cfg["loss"]["use_text_loss"] is False
    assert cfg["loss"]["granularity"] == "auto"
    assert cfg["loss"]["temperature"] == 0.07
    assert cfg["train"]["batch_size"] == 32
    assert cfg["train"]["amp"] is True
    assert cfg["data"]["sat_size"] == {"height": 432, "width": 768}
    assert cfg["eval"]["checkpoint"] == "last.pth"


def test_retrieval_config_merges_nested_overrides(tmp_path: Path) -> None:
    cfg_path = tmp_path / "clip.yaml"
    cfg_path.write_text(
        """
exp_name: clip
save_dir: /tmp/retrieval_clip
model:
  type: clip
train:
  batch_size: 4
loss:
  use_text_loss: true
data:
  subset_heights: [150, 300]
eval:
  candidate_size: 50
""",
        encoding="utf-8",
    )

    cfg = load_config(str(cfg_path))

    assert cfg["train"]["batch_size"] == 4
    assert cfg["train"]["epochs"] == 4
    assert cfg["loss"]["use_text_loss"] is True
    assert cfg["data"]["subset_heights"] == [150, 300]
    assert cfg["data"]["subset_angles"] == [0, 45, 90, 135, 180, 225, 270, 315]
    assert cfg["eval"]["candidate_size"] == 50


@pytest.mark.parametrize(
    "config_path,model_type",
    [
        ("configs/retrieval/siglip.yaml", "siglip"),
        ("configs/retrieval/clip.yaml", "clip"),
        ("configs/retrieval/openclip.yaml", "openclip"),
        ("configs/retrieval/evaclip.yaml", "evaclip"),
        ("configs/retrieval/sample_retrieval.yaml", "sample_retrieval"),
    ],
)
def test_checked_in_retrieval_configs_load(config_path: str, model_type: str) -> None:
    cfg = load_config(config_path)

    assert cfg["model"]["type"] == model_type
    assert cfg["train"]["batch_size"] == 32
    assert cfg["train"]["amp"] is True
    assert cfg["loss"]["use_text_loss"] is False
    assert cfg["loss"]["granularity"] == "auto"
    assert cfg["eval"]["output_dir"].startswith("eval_results/retrieval/")
