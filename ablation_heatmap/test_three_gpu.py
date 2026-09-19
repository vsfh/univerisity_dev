"""No real training/evaluation or CUDA kernels: algebra and mocked orchestration."""
import contextlib
import copy
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F
import yaml
from accelerate.utils import extract_model_from_parallel

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "exp"))
import batch_parallel
import study
from train_ada import info_nce_loss


class ThreeGpuTests(unittest.TestCase):
    def test_full_batch_loss_gradients_and_update_match_uneven_shards(self):
        torch.manual_seed(7)
        full = torch.nn.Linear(7, 4 * 16).double()
        split = copy.deepcopy(full)
        full_opt = torch.optim.AdamW(full.parameters(), lr=5e-5)
        split_opt = torch.optim.AdamW(split.parameters(), lr=5e-5)

        def objective(output):
            anchor = output[:, :4]
            grid = output[:, 4:].reshape(32, 15, 4)
            labels = torch.full((32, 32 * 15), 0., dtype=torch.double)
            for i in range(32):
                labels[i, i * 15:(i + 1) * 15] = 0.005714
                labels[i, i * 15 + (i % 15)] = 0.92
            retrieval = info_nce_loss(anchor, grid.reshape(-1, 4), labels)
            # Uneven-shard regression term must also be reduced over the full batch.
            regression = (output - 0.3).square().mean()
            return 0.5 * retrieval + 0.5 * regression

        for _ in range(2):  # Same accumulation as the real experiment.
            inputs = torch.randn(32, 7, dtype=torch.double)
            full_output = full(inputs)
            chunks = inputs.chunk(3)
            self.assertEqual([len(chunk) for chunk in chunks], [11, 11, 10])
            gathered = torch.cat([split(chunk) for chunk in chunks], dim=0)
            torch.testing.assert_close(full_output, gathered, atol=1e-12, rtol=1e-12)
            full_loss, split_loss = objective(full_output), objective(gathered)
            torch.testing.assert_close(full_loss, split_loss, atol=1e-12, rtol=1e-12)
            (full_loss / 2).backward()
            (split_loss / 2).backward()
        for left, right in zip(full.parameters(), split.parameters()):
            torch.testing.assert_close(left.grad, right.grad, atol=1e-10, rtol=1e-10)
        full_opt.step()
        split_opt.step()
        for left, right in zip(full.parameters(), split.parameters()):
            torch.testing.assert_close(left, right, atol=1e-10, rtol=1e-10)

    def test_checkpoint_unwrap_and_single_gpu_unchanged(self):
        model = torch.nn.Linear(3, 4)
        self.assertIs(batch_parallel.wrap_batch_parallel(model, 1, "cpu", 42), model)
        wrapped = torch.nn.DataParallel(model)  # CPU only; forward is never invoked.
        bare = extract_model_from_parallel(wrapped)
        self.assertIs(bare, model)
        torch.nn.Linear(3, 4).load_state_dict(bare.state_dict(), strict=True)
        with patch.object(batch_parallel, "require_visible_gpus"):
            with self.assertRaisesRegex(ValueError, "BatchNorm"):
                batch_parallel.wrap_batch_parallel(torch.nn.BatchNorm1d(4), 3, "cuda:0", 42)

    def test_gpu_count_validation(self):
        with patch.dict(os.environ, {"WORLD_SIZE": "1", "LOCAL_WORLD_SIZE": "1"}), \
             patch.object(torch.cuda, "is_available", return_value=True), \
             patch.object(torch.cuda, "device_count", return_value=2):
            with self.assertRaisesRegex(RuntimeError, "three CUDA"):
                batch_parallel.require_visible_gpus(3)

    def test_three_gpu_pipeline_is_sequential_experiments_and_shared_batch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            exp = root / "exp"
            (exp / "config_abla").mkdir(parents=True)
            base = {"use_ap": True, "config": {
                "OPTIMIZE_OBJECTIVE": "combined", "USE_HEATMAP_LOSS": True,
                "NUM_EPOCHS": 20, "BATCH_SIZE": 32, "GRAD_ACCUMULATION_STEPS": 2,
                "USE_AMP": True, "LEARNING_RATE": 5e-5, "LORA_DROPOUT": 0.05,
                "HEATMAP_CONFIDENCE_WEIGHT": 0.5}}
            (exp / "config_abla/baseline_wo_input_ids_ada_end_0_5.yaml").write_text(yaml.safe_dump(base))
            for name in ["train_ada.py", "test.py"]:
                (exp / name).write_text("# never executed")
            single = root / "outputs/heatmap_lambda_box_0p5_42"
            single.mkdir(parents=True)
            (single / "keep").write_text("single-card result")
            calls = []

            def fake_run(command, runtime, env, log_path):
                calls.append(command)
                cfg = yaml.safe_load(Path(command[command.index("--config") + 1]).read_text())
                for key in ["BATCH_SIZE", "GRAD_ACCUMULATION_STEPS", "NUM_EPOCHS",
                            "LEARNING_RATE", "USE_AMP", "LORA_DROPOUT", "HEATMAP_CONFIDENCE_WEIGHT"]:
                    self.assertEqual(cfg["config"][key], base["config"][key])
                self.assertEqual(cfg["config"]["DATA_PARALLEL_GPUS"], 3)
                self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "0,1,2")
                self.assertEqual(env["ACCELERATE_GRADIENT_ACCUMULATION_STEPS"], "2")
                self.assertNotIn("LOCAL_RANK", env)
                self.assertNotIn("WORLD_SIZE", env)
                self.assertTrue(runtime.is_relative_to(root / "outputs"))
                if "--save-dir" in command:
                    self.assertEqual(Path(command[1]).name, "train_ada.py")
                    self.assertNotIn("accelerate.commands.launch", command)
                    run_idx = (len(calls) - 1) // 2
                    weight = study.WEIGHTS[run_idx // len(study.SEEDS)]
                    self.assertEqual(cfg["config"]["HEATMAP_LOSS_WEIGHT"], weight / 0.5)
                    checkpoint_dir = Path(command[command.index("--save-dir") + 1])
                    checkpoint_dir.mkdir()
                    (checkpoint_dir / "last.pth").write_text("mock only")
                else:
                    self.assertIn("--save-dir", calls[-2])
                    self.assertEqual(command[command.index("--data-parallel-gpus") + 1], "3")
                    checkpoint = Path(command[command.index("--checkpoint") + 1])
                    self.assertEqual(checkpoint.name, "last.pth")
                    self.assertTrue(checkpoint.exists())
                    self.assertEqual(command[command.index("--seed") + 1],
                                     calls[-2][calls[-2].index("--seed") + 1])
                    name = command[command.index("--output-suffix") + 1]
                    path = Path(command[command.index("--output-dir") + 1]) / (name + ".json")
                    path.write_text(json.dumps({"overall": {key: 0.5 for key in study.METRICS}}))

            with patch.object(study, "ROOT", root), \
                 patch.object(study.sys, "argv", ["three_gpu.py"]), \
                 patch.object(batch_parallel, "require_visible_gpus"), \
                 patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0,1,2", "LOCAL_RANK": "0",
                                         "WORLD_SIZE": "1", "ACCELERATE_GRADIENT_ACCUMULATION_STEPS": "9"}), \
                 patch.object(study, "run_command", side_effect=fake_run), \
                 contextlib.redirect_stdout(io.StringIO()):
                study.main(3)
            self.assertEqual(len(calls), len(study.WEIGHTS) * len(study.SEEDS) * 2)
            self.assertTrue((single / "keep").exists())
            output = root / "outputs/heatmap_lambda_box_0p5_3gpu"
            manifest = json.loads((output / "parallel_config.json").read_text())
            self.assertEqual(manifest["effective_batch_size"], 64)
            summary = json.loads((output / "summary.json").read_text())
            self.assertEqual(len(summary["completed_runs"]), len(study.WEIGHTS) * len(study.SEEDS))

    def test_three_gpu_reset_cannot_delete_single_gpu_results(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            single = study.reset_output(root)
            (single / "keep").write_text("keep")
            parallel = study.reset_output(root, 3)
            (parallel / "old").write_text("old")
            self.assertEqual(study.reset_output(root, 3), parallel)
            self.assertFalse((parallel / "old").exists())
            self.assertTrue((single / "keep").exists())


if __name__ == "__main__":
    unittest.main()
