import ast
import importlib.util
import os
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch


MODULE_PATH = Path(__file__).resolve().parent / "tools" / "generate_qwen_7_8.py"


def _height_prompt() -> str:
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if isinstance(target, ast.Name) and target.id == "HEIGHT_PROMPT":
            return ast.literal_eval(node.value)
    raise AssertionError("HEIGHT_PROMPT not found")


def _constant_value(name: str):
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if isinstance(target, ast.Name) and target.id == name:
            return ast.literal_eval(node.value)
    raise AssertionError(f"{name} not found")


def _max_new_tokens_default() -> int:
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Attribute) or node.func.attr != "add_argument":
            continue
        if not node.args or not isinstance(node.args[0], ast.Constant):
            continue
        if node.args[0].value != "--max_new_tokens":
            continue
        for keyword in node.keywords:
            if keyword.arg == "default":
                return ast.literal_eval(keyword.value)
    raise AssertionError("--max_new_tokens default not found")


def _load_module():
    fake_torch = types.SimpleNamespace(
        bfloat16="bfloat16",
        float16="float16",
        float32="float32",
        device=object,
        cuda=types.SimpleNamespace(is_available=lambda: False),
        inference_mode=lambda: lambda fn: fn,
    )
    fake_pil = types.SimpleNamespace(Image=object)
    fake_hub = types.SimpleNamespace(snapshot_download=lambda **_: "")
    fake_transformers = types.SimpleNamespace(AutoProcessor=object)

    with patch.dict(
        sys.modules,
        {
            "torch": fake_torch,
            "PIL": fake_pil,
            "PIL.Image": fake_pil.Image,
            "huggingface_hub": fake_hub,
            "transformers": fake_transformers,
        },
    ):
        spec = importlib.util.spec_from_file_location("generate_qwen_7_8_for_test", MODULE_PATH)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module


class GenerateQwen78PromptTest(unittest.TestCase):
    def test_height_prompt_requests_one_dense_paragraph(self):
        prompt = _height_prompt()

        self.assertIn("Write ONE dense paragraph", prompt)
        self.assertIn("Output only the final caption", prompt)
        self.assertNotIn("exactly five captions", prompt)
        self.assertNotIn("numbered 1–5", prompt)

    def test_default_max_new_tokens_is_1000(self):
        self.assertEqual(_max_new_tokens_default(), 1000)

    def test_default_gpu_ids_expose_two_cards(self):
        self.assertEqual(_constant_value("DEFAULT_GPU_IDS"), "0,1")

    def test_configure_visible_gpus_uses_requested_ids(self):
        module = _load_module()

        with patch.dict(os.environ, {}, clear=True):
            parsed = module._configure_visible_gpus("2,3")
            self.assertEqual(parsed, [2, 3])
            self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "2,3")

    def test_normalizes_height_description_to_single_paragraph(self):
        module = _load_module()

        text = "  Red roof around central lawn.\nTree rows border the east road.  "

        self.assertEqual(
            module.structure_height_description(text),
            "Red roof around central lawn. Tree rows border the east road.",
        )

    def test_generate_result_includes_one_description_by_height(self):
        module = _load_module()
        output_text = "Red roof around central lawn. Tree rows border the east road."

        module.HEIGHTS = (150,)
        module.collect_height_groups = lambda _: {"150": ["0.png", "90.png", "180.png", "270.png"]}
        module._validate_images = lambda *_, **__: None
        module._build_message = lambda *_, **__: {"role": "user", "content": []}
        module._generate_batch = lambda **_: [output_text]
        module._sample_id_from_dir = lambda _: "sample-1"
        args = types.SimpleNamespace(
            generation_batch_size=1,
            image_width=256,
            image_height=256,
            image_field="image",
            prompt="prompt",
            max_new_tokens=1000,
            temperature=0.0,
            top_p=0.95,
            top_k=64,
            enable_thinking=False,
            model_name="model",
        )

        result = module.generate_descriptions_for_one_dir(
            resolved_image_dir="/tmp/sample-1",
            model=object(),
            processor=object(),
            args=args,
        )

        self.assertNotIn("description_segments", result)
        self.assertEqual(result["height_descriptions"], {"150": output_text})


if __name__ == "__main__":
    unittest.main()
