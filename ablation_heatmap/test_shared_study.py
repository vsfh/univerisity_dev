"""CPU-only merge, shared summary and lock checks."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

import yaml
import study
from merge_outputs import merge_existing


class SharedStudyTests(unittest.TestCase):
    def fixture(self, root):
        base = {"use_ap": True, "end_num": 0.5, "config": {
            "BATCH_SIZE": 32, "NUM_EPOCHS": 20, "HEATMAP_LOSS_WEIGHT": 0.2}}
        path = root / "exp/config_abla/baseline_wo_input_ids_ada_end_0_5.yaml"
        path.parent.mkdir(parents=True)
        path.write_text(yaml.safe_dump(base))
        return base

    def artifact(self, folder, base, weight, seed, completed=False, effective_only=False, gpus=1):
        for sub in ["configs", "results", "checkpoints"]:
            (folder / sub).mkdir(parents=True, exist_ok=True)
        payload = study.make_payload(base, folder, weight, seed, gpus)
        name = payload["exp_name"]
        checkpoint = folder / "checkpoints" / name / "last.pth"
        checkpoint.parent.mkdir()
        with zipfile.ZipFile(checkpoint, "w") as archive:
            archive.writestr("weights", "mock")
        if effective_only:
            (checkpoint.parent / "effective_config.json").write_text(json.dumps({**payload, "seed": seed}))
        else:
            (folder / "configs" / (name + ".yaml")).write_text(yaml.safe_dump(payload))
        if completed:
            (folder / "results" / (name + ".json")).write_text(json.dumps({
                "checkpoint": str(checkpoint), "candidate_size": 100, "test_crop_ratio": 1.,
                "sat_size": {"height": 432, "width": 768},
                "overall": {"num_samples": 32, **{key: seed / 100 for key in study.METRICS}}}))
        return payload

    def test_merge_recovers_configs_preserves_weights_and_aggregates_all_seeds(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = self.fixture(root)
            legacy = root / "outputs/heatmap_lambda_box_0p5_42"
            target = study.prepare_output(root)
            payload42 = self.artifact(legacy, base, 0.01, 42, completed=True)
            self.artifact(target, base, 0.01, 43, effective_only=True)
            report = merge_existing(root, apply=True)
            checkpoint = target / "checkpoints" / payload42["exp_name"]
            self.assertTrue(checkpoint.is_symlink())
            self.assertTrue((checkpoint / "last.pth").is_file())
            self.assertTrue((legacy / "results" / (payload42["exp_name"] + ".json")).exists())
            config43 = target / "configs/heatmap_0p01_seed_43.yaml"
            self.assertTrue(config43.exists())
            payload43 = study.make_payload(base, target, 0.01, 43, 3)
            self.assertEqual(study.inspect_run(target, payload43, 43)[0], "test")
            self.assertEqual(len(report["runs"]), 2)
            merge_existing(root, apply=True)  # Idempotent merge, no duplicate results.
            self.artifact(target, base, 0.01, 44, completed=True)
            study.summarize(target, base=base)
            summary = json.loads((target / "summary.json").read_text())
            row = summary["rows"][0]
            self.assertEqual(row["n_completed"], 2)
            self.assertEqual(row["n_expected"], 3)
            self.assertAlmostEqual(row["recall@1_mean"], .43)
            self.assertEqual({item["seed"] for item in summary["completed_runs"]}, {42, 44})

    def test_conflicting_completed_results_do_not_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = self.fixture(root)
            a = root / "outputs/heatmap_lambda_box_0p5_42"
            b = root / "outputs/heatmap_lambda_box_0p5_3gpu"
            self.artifact(a, base, .01, 42, completed=True)
            self.artifact(b, base, .01, 42, completed=True, gpus=3)
            result_path = b / "results/heatmap_0p01_seed_42.json"
            obj = json.loads(result_path.read_text())
            obj["overall"]["recall@1"] = .99
            result_path.write_text(json.dumps(obj))
            before = result_path.read_bytes()
            with self.assertRaisesRegex(ValueError, "Conflicting completed"):
                merge_existing(root, apply=True)
            self.assertEqual(result_path.read_bytes(), before)

    def test_lock_excludes_duplicate_workers_and_releases_after_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "lock"
            with self.assertRaisesRegex(ValueError, "simulated"):
                with study.directory_lock(path):
                    with self.assertRaisesRegex(RuntimeError, "Busy lock"):
                        with study.directory_lock(path):
                            self.fail("Lock admitted two writers")
                    raise ValueError("simulated")
            self.assertFalse(path.exists())
            with study.directory_lock(path):
                self.assertTrue(path.exists())


if __name__ == "__main__":
    unittest.main()
