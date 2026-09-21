"""Mock subprocess checks only; never run model training or evaluation."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import yaml

import baseline_suite


class PairRunnerTests(unittest.TestCase):
    def test_pair_uses_own_outputs_and_last_checkpoints(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            configs = root / "configs/retrieval"
            configs.mkdir(parents=True)
            for name in ("clip", "siglip"):
                (configs / (name + ".yaml")).write_text("model:\n  type: " + name + "\n")
            old = root / "outputs/baseline_recheck/keep.txt"
            old.parent.mkdir(parents=True)
            old.write_text("existing baseline results")
            pair = root / "outputs/clip_siglip2_recheck"
            pair.mkdir()
            (pair / "stale.txt").write_text("old pair run")
            calls = []

            def fake_run(command, log_path, cwd, env):
                calls.append(Path(command[1]).name)
                cfg = yaml.safe_load(Path(command[command.index("--config") + 1]).read_text())
                self.assertTrue(log_path.is_relative_to(pair))
                self.assertTrue(cwd.is_relative_to(pair))
                self.assertIsNone(cfg["train"]["resume_checkpoint"])
                self.assertFalse(cfg["train"]["save_best"])
                checkpoint = Path(cfg["save_dir"]) / "last.pth"
                if calls[-1] == "train.py":
                    checkpoint.parent.mkdir(parents=True)
                    checkpoint.write_text("mock")
                else:
                    self.assertEqual(calls[-1], "eval.py")
                    self.assertEqual(command[command.index("--checkpoint") + 1], str(checkpoint))
                    self.assertTrue(checkpoint.exists())
                    output = Path(cfg["eval"]["output_dir"])
                    output.mkdir(parents=True)
                    (output / "metrics.json").write_text(json.dumps({
                        "checkpoint": str(checkpoint), "overall": {"recall@1": 0.5}}))
                return 0

            with patch.object(baseline_suite, "ROOT", root), \
                 patch.object(baseline_suite, "run_logged", fake_run), \
                 contextlib.redirect_stdout(io.StringIO()):
                result = baseline_suite.main(
                    ["retrieval_clip", "retrieval_siglip"], "clip_siglip2_recheck")
            self.assertEqual(result, 0)
            self.assertEqual(calls, ["train.py", "eval.py", "train.py", "eval.py"])
            self.assertEqual(old.read_text(), "existing baseline results")
            self.assertFalse((pair / "stale.txt").exists())
            rows = json.loads((pair / "summary.json").read_text())
            self.assertEqual([row["model"] for row in rows], ["retrieval_clip", "retrieval_siglip"])
            self.assertTrue(all(row["status"] == "ok" for row in rows))

    def test_rejects_output_path_escape(self):
        with self.assertRaises(ValueError):
            baseline_suite.main(["retrieval_clip"], "../baseline_recheck")


if __name__ == "__main__":
    unittest.main()
