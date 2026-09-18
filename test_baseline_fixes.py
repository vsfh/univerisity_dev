"""CPU regression checks only: no models, real images, or training subprocesses."""
import ast
import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
import yaml
from PIL import Image
from torchvision.transforms import CenterCrop, Compose, Normalize, Resize, ToTensor
from transformers import CLIPImageProcessor

import baseline_suite
from grounding.adapters import BaseAdapter
from grounding.losses import anchor_predictions_for_loss_and_decode, decode_anchor_prediction
from retrieval.losses import prepare_candidate_targets
from retrieval.processors import FullFrameHFProcessor, FullFrameTransformProcessor

ROOT = Path(__file__).resolve().parent


class BaselineFixTests(unittest.TestCase):
    def striped_image(self):
        image = Image.new("RGB", (768, 432), (0, 255, 0))
        image.paste((255, 0, 0), (0, 0, 100, 432))
        image.paste((0, 0, 255), (668, 0, 768, 432))
        return image

    def assert_edges(self, value):
        self.assertEqual(tuple(value.shape), (1, 3, 224, 224))
        self.assertGreater(value[0, 0, 112, 2].item(), value[0, 1, 112, 2].item())
        self.assertGreater(value[0, 2, 112, -3].item(), value[0, 1, 112, -3].item())

    def test_full_frame_transform_preserves_both_edges(self):
        norm = Normalize([0.5] * 3, [0.5] * 3)
        old = Compose([Resize(224), CenterCrop(224), ToTensor(), norm])
        proc = FullFrameTransformProcessor(old, {"height": 432, "width": 768})
        self.assertIs(proc.preprocess.transforms[-1], norm)
        self.assertEqual(proc.size, {"height": 432, "width": 768})
        self.assert_edges(proc(self.striped_image())["pixel_values"])
        cropped = old(self.striped_image())
        self.assertGreater(cropped[1, 112, 2].item(), cropped[0, 112, 2].item())

    def test_hf_full_frame_preserves_both_edges(self):
        proc = FullFrameHFProcessor(CLIPImageProcessor(),
                                    {"height": 432, "width": 768}, (224, 224))
        self.assertFalse(proc.processor.do_center_crop)
        self.assert_edges(proc(self.striped_image())["pixel_values"])

    def test_heatmap_gating_decode_and_detachment(self):
        raw = torch.zeros(1, 9, 5, 1, 2, requires_grad=True)
        heat = torch.tensor([[[[0., 10.]]]], requires_grad=True)
        output = SimpleNamespace(pred_anchor=raw, heatmap=heat, pred_bbox=None,
                                 image_wh=(200, 100))
        cfg = {"model": {"use_heatmap": False},
               "loss": {"heatmap_confidence_weight": 0.5}}
        anchors = torch.ones(9, 2) * 20
        off = anchor_predictions_for_loss_and_decode(output, cfg)
        self.assertTrue(torch.equal(off, raw))
        off_box = BaseAdapter(None, cfg).decode(output, {}, anchors)
        cfg["model"]["use_heatmap"] = True
        on = anchor_predictions_for_loss_and_decode(output, cfg)
        on_box = BaseAdapter(None, cfg).decode(output, {}, anchors)
        self.assertTrue(torch.equal(on_box, decode_anchor_prediction(on, anchors, output.image_wh)))
        self.assertGreater(on_box[0, 0].item(), off_box[0, 0].item())
        self.assertEqual(raw.detach().abs().sum().item(), 0)
        on.sum().backward()
        self.assertIsNone(heat.grad)
        self.assertIsNotNone(raw.grad)
        source = (ROOT / "grounding/losses.py").read_text()
        tree = ast.parse(source)
        compute = next(node for node in tree.body
                       if isinstance(node, ast.FunctionDef) and node.name == "compute_grounding_loss")
        self.assertTrue(any(isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                            and node.func.id == "anchor_predictions_for_loss_and_decode"
                            for node in ast.walk(compute)))

    def test_exp_labels_route_to_unchanged_nine_descriptors(self):
        batch = {"index": torch.tensor([14]),
                 "bbox": torch.tensor([[680., 380., 760., 430.]])}
        flat, target, granularity = prepare_candidate_targets(
            torch.zeros(1, 9, 4), batch, image_wh=(768, 432))
        self.assertEqual(tuple(flat.shape), (9, 4))
        self.assertEqual(target.argmax(1).item(), 8)
        self.assertAlmostEqual(target.sum().item(), 1., places=6)
        self.assertEqual(granularity, "grid")

    def test_all_entry_points_import_exp_dataset(self):
        for filename in ("retrieval/train.py", "retrieval/eval_retrieval.py",
                         "grounding/train.py", "grounding/eval.py", "test_unify_ground.py"):
            nodes = ast.walk(ast.parse((ROOT / filename).read_text()))
            sources = [node.module for node in nodes if isinstance(node, ast.ImportFrom)
                       and any(alias.name == "ShiftedSatelliteDroneDataset" for alias in node.names)]
            self.assertEqual(sources, ["exp.dataset"], filename)

    def test_suite_all_models_without_any_real_subprocess(self):
        (ROOT / "outputs").mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ROOT / "outputs", prefix="baseline_unit_") as folder:
            root = Path(folder)
            for family, name in baseline_suite.MODELS.values():
                config = root / "configs" / family / (name + ".yaml")
                config.parent.mkdir(parents=True, exist_ok=True)
                config.write_text("model:\n  type: " + name + "\n")
            unrelated = root / "outputs" / "other_experiment"
            unrelated.mkdir(parents=True)
            keep = unrelated / "keep.txt"
            keep.write_text("keep")
            stale = root / "outputs" / "baseline_recheck" / "stale.txt"
            stale.parent.mkdir()
            stale.write_text("old")
            calls = []
            def fake_run(command, log_path, cwd, env):
                calls.append(command)
                self.assertTrue(cwd.is_relative_to(root / "outputs"))
                self.assertTrue(log_path.is_relative_to(root / "outputs"))
                self.assertNotIn("--resume", command)
                self.assertFalse(any("val.py" == Path(part).name for part in command))
                if Path(command[1]).name == "train.py":
                    cfg = yaml.safe_load(Path(command[command.index("--config") + 1]).read_text())
                    self.assertIsNone(cfg["train"]["resume_checkpoint"])
                    self.assertFalse(cfg["train"]["save_best"])
                    checkpoint = Path(cfg["save_dir"]) / "last.pth"
                    checkpoint.parent.mkdir(parents=True)
                    checkpoint.write_bytes(b"mock checkpoint, never loaded")
                else:
                    checkpoint = Path(command[command.index("--checkpoint") + 1])
                    self.assertEqual(checkpoint.name, "last.pth")
                    self.assertTrue(checkpoint.is_file())
                    if Path(command[1]).name == "eval.py":
                        cfg = yaml.safe_load(Path(command[command.index("--config") + 1]).read_text())
                        result = Path(cfg["eval"]["output_dir"]) / "metrics.json"
                    else:
                        name = command[command.index("--model-types") + 1]
                        result = Path(command[command.index("--output-dir") + 1]) / (
                            "test_unify_" + name + "_" + name + ".json")
                    result.parent.mkdir(parents=True, exist_ok=True)
                    result.write_text(json.dumps({"checkpoint": str(checkpoint),
                                                  "overall": {"mock_metric": 0.5}}))
                return 0
            with patch.object(baseline_suite, "ROOT", root), \
                 patch.object(baseline_suite, "run_logged", fake_run), \
                 contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(baseline_suite.main(), 0)
                self.assertEqual(len(calls), 22)
                self.assertFalse(stale.exists())
                self.assertTrue(keep.exists())
                rows = json.loads((stale.parent / "summary.json").read_text())
                self.assertEqual(len(rows), 11)
                self.assertTrue(all(row["status"] == "ok" for row in rows))
                with patch.object(baseline_suite, "run_logged", return_value=3):
                    self.assertEqual(baseline_suite.main(["retrieval_clip", "grounding_ocg"]), 1)
                rows = json.loads((stale.parent / "summary.json").read_text())
                self.assertEqual(len(rows), 2)
                self.assertTrue(all(row["status"] == "failed" for row in rows))
                self.assertTrue(keep.exists())


if __name__ == "__main__":
    unittest.main()
