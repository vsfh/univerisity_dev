import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from grounding.adapters import GroundingOutput
from grounding.config import DEFAULT_CONFIG
from grounding.losses import compute_grounding_loss, query_guard_weight
from grounding.eval import _load_checkpoint as load_eval_checkpoint
from grounding.training_records import (
    append_jsonl,
    atomic_save_checkpoint,
    atomic_write_json,
    load_v2_resume_checkpoint,
)


def build_cfg():
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg["model"]["type"] = "direct_test"
    cfg["model"]["use_heatmap"] = False
    cfg["loss"]["bbox_weight"] = 1.0
    return cfg


def test_query_guard_weight_warms_up_by_epoch():
    cfg = build_cfg()
    assert query_guard_weight(cfg, epoch=0) == 0.0
    assert query_guard_weight(cfg, epoch=1) == 0.1
    assert query_guard_weight(cfg, epoch=2) == 0.2
    assert query_guard_weight(cfg, epoch=20) == 0.2


def test_combined_loss_adds_dense_and_paper_aux_terms():
    cfg = build_cfg()
    query = torch.tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True)
    candidates = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    matcher_logits = query @ candidates.T / 0.1
    target_bbox = torch.tensor([[1.0, 1.0, 5.0, 5.0], [2.0, 2.0, 6.0, 6.0]])
    output = GroundingOutput(
        device=torch.device("cpu"),
        image_wh=(8, 8),
        pred_bbox=target_bbox.clone(),
        query_embedding=query,
        search_local_features=candidates.view(2, 2, 1, 1),
        search_grid_size=(1, 1),
        matcher_logits=matcher_logits,
        paper_aux_losses={"retrieval": torch.tensor(0.25)},
    )
    batch = {"bbox": target_bbox, "satellite_id": torch.tensor([10, 20])}

    losses = compute_grounding_loss(
        output,
        batch,
        anchors_full=torch.ones(9, 2),
        cfg=cfg,
        epoch=2,
    )

    torch.testing.assert_close(
        losses.total,
        losses.bbox + 0.2 * losses.dense + losses.paper_aux,
    )
    assert losses.paper_aux.item() == 0.25
    losses.total.backward()
    assert query.grad is not None
    assert torch.isfinite(query.grad).all()


def test_query_guard_requires_matcher_outputs_when_enabled():
    cfg = build_cfg()
    bbox = torch.tensor([[1.0, 1.0, 5.0, 5.0]])
    output = GroundingOutput(
        device=torch.device("cpu"),
        image_wh=(8, 8),
        pred_bbox=bbox.clone(),
    )
    try:
        compute_grounding_loss(
            output,
            {"bbox": bbox, "satellite_id": torch.tensor([1])},
            anchors_full=torch.ones(9, 2),
            cfg=cfg,
            epoch=2,
        )
    except ValueError as error:
        assert "matcher_logits" in str(error)
    else:
        raise AssertionError("Enabled query_guard accepted missing matcher outputs.")


def test_training_records_and_v2_checkpoint_are_atomic_and_readable():
    with TemporaryDirectory() as directory:
        root = Path(directory)
        summary_path = root / "training_summary.json"
        history_path = root / "train_history.jsonl"
        checkpoint_path = root / "last.pth"

        atomic_write_json(summary_path, {"status": "running", "epoch": 1})
        append_jsonl(history_path, {"epoch": 0, "loss": 2.0})
        append_jsonl(history_path, {"epoch": 1, "loss": 1.0})
        atomic_save_checkpoint(
            checkpoint_path,
            {
                "architecture_version": 2,
                "model": {"weight": torch.ones(1)},
                "optimizer": {},
                "epoch": 1,
                "global_step": 2,
                "config": {},
                "training_summary": {"status": "running"},
            },
        )

        assert json.loads(summary_path.read_text(encoding="utf-8"))["epoch"] == 1
        assert len(history_path.read_text(encoding="utf-8").strip().splitlines()) == 2
        payload = load_v2_resume_checkpoint(checkpoint_path)
        assert payload["architecture_version"] == 2
        assert not checkpoint_path.with_suffix(".pth.tmp").exists()


def test_old_checkpoint_is_rejected_for_resume():
    with TemporaryDirectory() as directory:
        path = Path(directory) / "old.pth"
        torch.save({"weight": torch.ones(1)}, path)
        try:
            load_v2_resume_checkpoint(path)
        except ValueError as error:
            assert "architecture_version=2" in str(error)
        else:
            raise AssertionError("Old checkpoint was accepted for resume.")


def test_eval_loader_reads_model_state_from_v2_payload():
    with TemporaryDirectory() as directory:
        path = Path(directory) / "last.pth"
        source = torch.nn.Linear(2, 1)
        target = torch.nn.Linear(2, 1)
        with torch.no_grad():
            source.weight.fill_(3.0)
            source.bias.fill_(2.0)
            target.weight.zero_()
            target.bias.zero_()
        atomic_save_checkpoint(
            path,
            {
                "architecture_version": 2,
                "model": source.state_dict(),
                "optimizer": {},
                "epoch": 0,
                "global_step": 0,
                "config": {},
                "training_summary": {},
            },
        )

        load_eval_checkpoint(target, str(path))
        torch.testing.assert_close(target.weight, source.weight)
        torch.testing.assert_close(target.bias, source.bias)


if __name__ == "__main__":
    tests = [value for name, value in globals().items() if name.startswith("test_") and callable(value)]
    for test in tests:
        test()
    print(f"passed {len(tests)} tests")
