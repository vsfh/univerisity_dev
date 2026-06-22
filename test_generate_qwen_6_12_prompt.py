import ast
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parent / "tools" / "generate_qwen_6_12.py"


def _height_prompt() -> str:
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if isinstance(target, ast.Name) and target.id == "HEIGHT_PROMPT":
            return ast.literal_eval(node.value)
    raise AssertionError("HEIGHT_PROMPT not found")


class GenerateQwenPromptTest(unittest.TestCase):
    def test_height_prompt_requests_one_sentence_about_32_tokens(self):
        prompt = _height_prompt()

        self.assertIn("same target", prompt)
        self.assertIn("one English sentence", prompt)
        self.assertIn("about 32 SigLIP2-style text tokens", prompt)
        self.assertIn("no markdown", prompt)
        self.assertNotIn("about 60 SigLIP2-style text tokens", prompt)
        self.assertNotIn("then 4 concise", prompt)


if __name__ == "__main__":
    unittest.main()
