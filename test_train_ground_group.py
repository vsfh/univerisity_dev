from pathlib import Path
import unittest


SCRIPT = Path(__file__).resolve().parent / "train_ground_group.sh"


class TrainGroundGroupScriptTest(unittest.TestCase):
    def test_supports_multi_gpu_workers(self):
        source = SCRIPT.read_text(encoding="utf-8")

        self.assertIn("--gpus", source)
        self.assertIn("IFS=',' read -r -a GPUS", source)
        self.assertIn("run_worker()", source)
        self.assertIn('CUDA_VISIBLE_DEVICES="$GPU_ID"', source)
        self.assertIn("CONFIG_INDEX % NUM_GPUS", source)
        self.assertIn('worker_${WORKER_ID}.jsonl', source)
        self.assertIn("wait", source)
        self.assertIn("--device cuda:0", source)

    def test_train_and_eval_support_device_override(self):
        train_source = (SCRIPT.parent / "grounding" / "train.py").read_text(encoding="utf-8")
        eval_source = (SCRIPT.parent / "grounding" / "eval.py").read_text(encoding="utf-8")

        self.assertIn('parser.add_argument("--device"', train_source)
        self.assertIn('cfg["train"]["device"] = args.device', train_source)
        self.assertIn('parser.add_argument("--device"', eval_source)
        self.assertIn('cfg["train"]["device"] = args.device', eval_source)


if __name__ == "__main__":
    unittest.main()
