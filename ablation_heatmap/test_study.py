"""CPU-only checks. Training/evaluation processes are mocked, never launched."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import yaml
import study


class HeatmapSweepTests(unittest.TestCase):
    def test_overwrite_is_scoped(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = study.reset_output(root)
            (output / "stale").write_text("old")
            sibling = root / "outputs" / "unrelated"
            sibling.mkdir()
            (sibling / "keep").write_text("keep")
            self.assertEqual(study.reset_output(root), output)
            self.assertFalse((output / "stale").exists())
            self.assertEqual((sibling / "keep").read_text(), "keep")

    def test_symlink_cannot_redirect_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "outputs").mkdir()
            victim = root / "victim"
            victim.mkdir()
            (victim / "keep").write_text("keep")
            try:
                (root / "outputs/heatmap_lambda_box_0p5_42").symlink_to(victim, target_is_directory=True)
            except OSError:
                self.skipTest("Symlinks unavailable on this host")
            with self.assertRaises(ValueError):
                study.reset_output(root)
            self.assertTrue((victim / "keep").exists())

    def test_all_runs_train_then_test_and_repeat_from_scratch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            exp = root / "exp"
            (exp / "config_abla").mkdir(parents=True)
            base = {"use_ap": True, "config": {
                "OPTIMIZE_OBJECTIVE": "combined", "USE_HEATMAP_LOSS": True,
                "NUM_EPOCHS": 20, "HEATMAP_CONFIDENCE_WEIGHT": 0.5}}
            (exp / "config_abla/baseline_wo_input_ids_ada_end_0_5.yaml").write_text(yaml.safe_dump(base))
            for filename in ["train_ada.py", "test.py"]:
                (exp / filename).write_text("# mock only")
            calls = []

            def fake_run(command, runtime, env, log_path):
                calls.append(command)
                payload = yaml.safe_load(Path(command[command.index("--config") + 1]).read_text())
                self.assertEqual(payload["end_num"], 0.5)
                self.assertTrue(payload["config"]["USE_HEATMAP_LOSS"])
                self.assertEqual(payload["config"]["HEATMAP_CONFIDENCE_WEIGHT"], 0.5)
                self.assertTrue(runtime.is_relative_to(root / "outputs"))
                if "--save-dir" in command:
                    run_index = ((len(calls) - 1) // 2) % (len(study.WEIGHTS) * len(study.SEEDS))
                    expected_weight = study.WEIGHTS[run_index // len(study.SEEDS)]
                    self.assertEqual(payload["config"]["HEATMAP_LOSS_WEIGHT"], expected_weight / 0.5)
                    checkpoint_dir = Path(command[command.index("--save-dir") + 1])
                    self.assertFalse(checkpoint_dir.exists())
                    checkpoint_dir.mkdir()
                    (checkpoint_dir / "last.pth").write_text("mock checkpoint")
                else:
                    self.assertIn("--save-dir", calls[-2])
                    self.assertEqual(command[command.index("--seed") + 1],
                                     calls[-2][calls[-2].index("--seed") + 1])
                    name = command[command.index("--output-suffix") + 1]
                    output = Path(command[command.index("--output-dir") + 1]) / (name + ".json")
                    output.write_text(json.dumps({"overall": {key: 0.5 for key in study.METRICS}}))

            with patch.object(study, "ROOT", root), patch.object(study.sys, "argv", ["study.py"]), \
                    patch.object(study, "run_command", side_effect=fake_run), contextlib.redirect_stdout(io.StringIO()):
                study.main()
                study.main()
            expected_runs = len(study.WEIGHTS) * len(study.SEEDS)
            self.assertEqual(len(calls), 4 * expected_runs)
            result = json.loads((root / "outputs/heatmap_lambda_box_0p5_42/summary.json").read_text())
            self.assertEqual(len(result["completed_runs"]), expected_runs)
            self.assertTrue(all(row["n_completed"] == len(study.SEEDS) for row in result["rows"]))
            expected_std = 0 if len(study.SEEDS) > 1 else None
            self.assertTrue(all(row["recall@1_std"] == expected_std for row in result["rows"]))


if __name__ == "__main__":
    unittest.main()
