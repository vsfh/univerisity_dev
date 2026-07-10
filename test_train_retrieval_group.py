from pathlib import Path
import unittest


SCRIPT = Path(__file__).resolve().parent / "train_retrieval_grounp.sh"


class TrainRetrievalGroupScriptTest(unittest.TestCase):
    def test_group_script_runs_all_retrieval_configs(self):
        source = SCRIPT.read_text(encoding="utf-8")

        self.assertIn("--gpus", source)
        self.assertIn("configs/retrieval/siglip.yaml", source)
        self.assertIn("configs/retrieval/clip.yaml", source)
        self.assertIn("configs/retrieval/openclip.yaml", source)
        self.assertIn("configs/retrieval/evaclip.yaml", source)
        self.assertIn("configs/retrieval/sample_retrieval.yaml", source)
        self.assertIn("retrieval/train.py", source)
        self.assertIn("retrieval/eval.py", source)
        self.assertIn("eval_results/retrieval", source)
        self.assertIn("--device cuda:0", source)

    def test_train_and_eval_support_device_override(self):
        train_source = (SCRIPT.parent / "retrieval" / "train.py").read_text(encoding="utf-8")
        eval_source = (SCRIPT.parent / "retrieval" / "eval.py").read_text(encoding="utf-8")

        self.assertIn('parser.add_argument("--device"', train_source)
        self.assertIn('cfg["train"]["device"] = args.device', train_source)
        self.assertIn('parser.add_argument("--device"', eval_source)
        self.assertIn('cfg["train"]["device"] = args.device', eval_source)


if __name__ == "__main__":
    unittest.main()
