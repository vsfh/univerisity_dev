import ast
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parent / "grounding" / "train_siglip.py"


def _load_tree():
    return ast.parse(MODULE_PATH.read_text(encoding="utf-8"))


def _call_names(tree):
    names = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Attribute):
            names.append(func.attr)
        elif isinstance(func, ast.Name):
            names.append(func.id)
    return names


def _module_constants(tree):
    constants = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if isinstance(target, ast.Name):
            try:
                constants[target.id] = ast.literal_eval(node.value)
            except (ValueError, TypeError):
                continue
    return constants


class TrainSiglipAmpTest(unittest.TestCase):
    def test_ground_siglip_training_uses_amp_scaler_and_autocast(self):
        tree = _load_tree()
        calls = _call_names(tree)

        self.assertIn("GradScaler", calls)
        self.assertIn("autocast", calls)
        self.assertIn("scale", calls)
        self.assertIn("step", calls)
        self.assertIn("update", calls)

    def test_ground_siglip_training_uses_gradient_accumulation_steps(self):
        tree = _load_tree()
        constants = _module_constants(tree)
        source = MODULE_PATH.read_text(encoding="utf-8")

        self.assertIn("GRAD_ACCUMULATION_STEPS", constants)
        self.assertEqual(2, constants["GRAD_ACCUMULATION_STEPS"])
        self.assertIn("grad_accumulation_steps", source)
        self.assertIn("loss / grad_accumulation_steps", source)
        self.assertIn("should_step", source)

    def test_ground_siglip_training_uses_grad_clip(self):
        tree = _load_tree()
        constants = _module_constants(tree)
        source = MODULE_PATH.read_text(encoding="utf-8")
        calls = _call_names(tree)

        self.assertIn("GRAD_CLIP_NORM", constants)
        self.assertEqual(1.0, constants["GRAD_CLIP_NORM"])
        self.assertIn("unscale_", calls)
        self.assertIn("clip_grad_norm_", source)
        self.assertIn("grad_clip_norm", source)

    def test_ground_siglip_training_uses_weighted_heatmap_loss(self):
        tree = _load_tree()
        constants = _module_constants(tree)
        source = MODULE_PATH.read_text(encoding="utf-8")
        calls = _call_names(tree)

        self.assertEqual(True, constants["USE_HEATMAP_LOSS"])
        self.assertEqual(0.2, constants["HEATMAP_LOSS_WEIGHT"])
        self.assertEqual(0.5, constants["HEATMAP_CONFIDENCE_WEIGHT"])
        self.assertIn("heatmap_loss_fn", calls)
        self.assertIn("add_heatmap_to_confidence", calls)
        self.assertIn("heatmap_loss_weight * heatmap_loss", source)

    def test_ground_siglip_uses_siglip2_processor_wrapper(self):
        tree = _load_tree()
        source = MODULE_PATH.read_text(encoding="utf-8")
        class_names = {
            node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)
        }

        self.assertIn("SiglipProcessorWrapper", class_names)
        self.assertIn("AutoImageProcessor.from_pretrained", source)
        self.assertIn("SiglipProcessorWrapper(MODEL_NAME, cache_dir=CACHE_DIR)", source)
        self.assertIn("size={\"height\": UNIV_SAT_SIZE[0], \"width\": UNIV_SAT_SIZE[1]}", source)


if __name__ == "__main__":
    unittest.main()
