from pathlib import Path
import unittest


SCRIPT = Path(__file__).resolve().parent / "train_ground_group.sh"


class TrainGroundGroupScriptTest(unittest.TestCase):
    def test_runs_each_config_with_torchrun_over_requested_gpus(self):
        source = SCRIPT.read_text(encoding="utf-8")

        self.assertIn("--gpus", source)
        self.assertIn("IFS=',' read -r -a GPUS", source)
        self.assertIn("--nproc_per_node=\"$NUM_GPUS\"", source)
        self.assertIn("torch.distributed.run", source)
        self.assertIn('CUDA_VISIBLE_DEVICES="$GPUS_CSV"', source)
        self.assertIn('CUDA_VISIBLE_DEVICES="$FIRST_GPU"', source)
        self.assertNotIn("CONFIG_INDEX % NUM_GPUS", source)
        self.assertNotIn("run_worker()", source)
        self.assertIn("--device cuda:0", source)

    def test_train_supports_distributed_batch_splitting(self):
        train_source = (SCRIPT.parent / "grounding" / "train.py").read_text(encoding="utf-8")

        self.assertIn('parser.add_argument("--device"', train_source)
        self.assertIn('cfg["train"]["device"] = args.device', train_source)
        self.assertIn("DistributedDataParallel", train_source)
        self.assertIn("DistributedSampler", train_source)
        self.assertIn("math.ceil(global_batch_size / distributed.world_size)", train_source)
        self.assertIn("sampler.set_epoch(epoch)", train_source)

    def test_eval_supports_device_override(self):
        eval_source = (SCRIPT.parent / "grounding" / "eval.py").read_text(encoding="utf-8")

        self.assertIn('parser.add_argument("--device"', eval_source)
        self.assertIn('cfg["train"]["device"] = args.device', eval_source)


if __name__ == "__main__":
    unittest.main()
