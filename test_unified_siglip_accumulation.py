from pathlib import Path
import unittest


SOURCE = Path(__file__).resolve().parent / "unified_siglip_supp.py"


class UnifiedSiglipAccumulationTest(unittest.TestCase):
    def test_accelerate_accumulation_steps_only_when_gradients_sync(self):
        source = SOURCE.read_text(encoding="utf-8")

        self.assertIn("accelerator.backward(loss)", source)
        self.assertIn("if accelerator.sync_gradients:", source)
        self.assertIn("optimizer.zero_grad(set_to_none=True)", source)
        self.assertNotIn("scaler.scale(loss).backward()", source)
        self.assertNotIn("torch.amp.GradScaler", source)


if __name__ == "__main__":
    unittest.main()
