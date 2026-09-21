"""CPU-only checks. Training/evaluation processes are mocked, never launched."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
import zipfile
from unittest.mock import patch

import yaml
import study


class HeatmapSweepTests(unittest.TestCase):
    def test_existing_outputs_are_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = study.prepare_output(root)
            (output / "stale").write_text("old")
            sibling = root / "outputs" / "unrelated"
            sibling.mkdir()
            (sibling / "keep").write_text("keep")
            self.assertEqual(study.prepare_output(root), output)
            self.assertTrue((output / "stale").exists())
            self.assertEqual((sibling / "keep").read_text(), "keep")

    def test_symlink_cannot_redirect_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "outputs").mkdir()
            victim = root / "victim"
            victim.mkdir()
            (victim / "keep").write_text("keep")
            try:
                (root / "outputs/heatmap_lambda_box_0p5").symlink_to(victim, target_is_directory=True)
            except OSError:
                self.skipTest("Symlinks unavailable on this host")
            with self.assertRaises(ValueError):
                study.prepare_output(root)
            self.assertTrue((victim / "keep").exists())

    def test_completed_runs_skip_and_interrupted_tests_only_retest(self):
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
                    with zipfile.ZipFile(checkpoint_dir / "last.pth", "w") as archive:
                        archive.writestr("mock", "not a real model")
                else:
                    self.assertEqual(command[command.index("--seed") + 1], payload["exp_name"].split("_seed_")[1])
                    name = command[command.index("--output-suffix") + 1]
                    output = Path(command[command.index("--output-dir") + 1]) / (name + ".json")
                    output.write_text(json.dumps({
                        "checkpoint": command[command.index("--checkpoint") + 1],
                        "candidate_size": 100, "test_crop_ratio": 1.0,
                        "sat_size": {"height": 432, "width": 768},
                        "overall": {"num_samples": 32, **{key: 0.5 for key in study.METRICS}},
                    }))

            with patch.object(study, "ROOT", root), patch.object(study.sys, "argv", ["study.py", "--seeds", "42", "43", "44"]), \
                    patch.object(study, "run_command", side_effect=fake_run), contextlib.redirect_stdout(io.StringIO()):
                study.main()
                output_dir = root / "outputs/heatmap_lambda_box_0p5"
                before = {p.name: p.read_bytes() for p in (output_dir / "results").glob("*.json")}
                study.main()
                self.assertEqual(len(calls), 2 * len(study.WEIGHTS) * len(study.SEEDS))
                self.assertEqual(before, {p.name: p.read_bytes() for p in (output_dir / "results").glob("*.json")})
                result_path = next((output_dir / "results").glob("*.json"))
                result_path.write_text('{"overall":')  # Interrupted JSON write.
                study.main()
                self.assertNotIn("--save-dir", calls[-1])
                # A changed setup must not silently reuse or erase old results.
                base["config"]["NUM_EPOCHS"] = 99
                (exp / "config_abla/baseline_wo_input_ids_ada_end_0_5.yaml").write_text(yaml.safe_dump(base))
                with self.assertRaisesRegex(ValueError, "Existing config differs"):
                    study.main()
            expected_runs = len(study.WEIGHTS) * len(study.SEEDS)
            self.assertEqual(len(calls), 2 * expected_runs + 1)
            result = json.loads((root / "outputs/heatmap_lambda_box_0p5/summary.json").read_text())
            self.assertEqual(len(result["completed_runs"]), expected_runs)
            self.assertTrue(all(row["n_completed"] == len(study.SEEDS) for row in result["rows"]))
            expected_std = 0 if len(study.SEEDS) > 1 else None
            self.assertTrue(all(row["recall@1_std"] == expected_std for row in result["rows"]))

    def test_incomplete_checkpoint_is_not_reused(self):
        with tempfile.TemporaryDirectory() as directory:
            output = study.prepare_output(Path(directory))
            payload = {"exp_name": "heatmap_0p1_seed_42", "config": {}, "end_num": 0.5, "use_ap": True}
            (output / "configs/heatmap_0p1_seed_42.yaml").write_text(yaml.safe_dump(payload))
            checkpoint = output / "checkpoints/heatmap_0p1_seed_42/last.pth"
            checkpoint.parent.mkdir()
            checkpoint.write_bytes(b"PK incomplete save")
            self.assertEqual(study.inspect_run(output, payload, 42), ("train", None))
            with zipfile.ZipFile(checkpoint, "w") as archive:
                archive.writestr("mock", "complete container")
            self.assertEqual(study.inspect_run(output, payload, 42), ("test", None))


if __name__ == "__main__":
    unittest.main()
