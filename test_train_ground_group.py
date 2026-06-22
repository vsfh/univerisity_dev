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


if __name__ == "__main__":
    unittest.main()
