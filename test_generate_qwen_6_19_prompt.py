import ast
import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch


MODULE_PATH = Path(__file__).resolve().parent / "tools" / "generate_qwen_6_19.py"


def _height_prompt() -> str:
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if isinstance(target, ast.Name) and target.id == "HEIGHT_PROMPT":
            return ast.literal_eval(node.value)
    raise AssertionError("HEIGHT_PROMPT not found")


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


def _skip_existing_default() -> bool:
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Attribute) or node.func.attr != "add_argument":
            continue
        arg_values = [
            arg.value
            for arg in node.args
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str)
        ]
        if "--skip_existing" not in arg_values:
            continue
        for keyword in node.keywords:
            if keyword.arg == "default":
                return ast.literal_eval(keyword.value)
    raise AssertionError("--skip_existing default not found")


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
        spec = importlib.util.spec_from_file_location("generate_qwen_6_19_for_test", MODULE_PATH)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module


class GenerateQwen619PromptTest(unittest.TestCase):
    def test_height_prompt_requests_five_siglip2_segments(self):
        prompt = _height_prompt()

        self.assertIn("exactly 5 English retrieval segments", prompt)
        self.assertIn("separated by ||", prompt)
        self.assertIn("SigLIP2-style text tokens", prompt)
        self.assertIn("short clauses", prompt)
        self.assertNotIn("output one English sentence", prompt)

    def test_default_max_new_tokens_is_400(self):
        self.assertEqual(_max_new_tokens_default(), 400)

    def test_skip_existing_is_enabled_by_default(self):
        self.assertIs(_skip_existing_default(), True)

    def test_structures_five_pipe_separated_segments(self):
        module = _load_module()
        output_text = (
            "gray rectangular roof near a paved road; green trees line the east edge || "
            "white building blocks around a central court; asphalt paths border the south side || "
            "tan concrete yard beside dark road bends; parking pavement sits west of the target || "
            "grass fields and tree belts surround the place; open paved areas frame the roof || "
            "central building lies between roads and greenery; north edge meets a straight lane"
        )

        segments = module.structure_description_segments(output_text)

        self.assertEqual(len(segments), 5)
        self.assertEqual(segments[0]["index"], 1)
        self.assertEqual(segments[0]["text"], "gray rectangular roof near a paved road; green trees line the east edge")
        self.assertNotIn("clauses", segments[0])
        self.assertEqual(segments[4]["index"], 5)

    def test_generate_result_includes_structured_segments_by_height(self):
        module = _load_module()
        output_text = (
            "gray rectangular roof near a paved road; green trees line the east edge || "
            "white building blocks around a central court; asphalt paths border the south side || "
            "tan concrete yard beside dark road bends; parking pavement sits west of the target || "
            "grass fields and tree belts surround the place; open paved areas frame the roof || "
            "central building lies between roads and greenery; north edge meets a straight lane"
        )

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
            max_new_tokens=400,
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

        self.assertNotIn("descriptions", result)
        self.assertEqual(len(result["description_segments"]["150"]), 5)
        self.assertEqual(result["description_segments"]["150"][1]["index"], 2)
        self.assertEqual(
            result["description_segments"]["150"][1]["text"],
            "white building blocks around a central court; asphalt paths border the south side",
        )

    def test_completed_result_skip_uses_nonempty_file_without_json_load(self):
        module = _load_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "qwen_6_19_description.json"
            output_path.write_text("not valid json", encoding="utf-8")

            result = module._load_completed_result(
                image_dir=tmpdir,
                output_name="qwen_6_19_description.json",
                skip_existing=True,
            )

        self.assertEqual(
            result,
            {
                "image_dir": tmpdir,
                "output_path": str(output_path),
                "skipped": True,
            },
        )

    def test_empty_completed_file_is_not_skipped(self):
        module = _load_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "qwen_6_19_description.json"
            output_path.touch()

            result = module._load_completed_result(
                image_dir=tmpdir,
                output_name="qwen_6_19_description.json",
                skip_existing=True,
            )

        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
