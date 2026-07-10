import importlib
import sys
import unittest
from pathlib import Path


class RetrievalEvalImportTest(unittest.TestCase):
    def test_legacy_eval_import_does_not_require_unused_grounding_encoders(self):
        retrieval_dir = Path(__file__).resolve().parent / "retrieval"
        sys.path.insert(0, str(retrieval_dir))
        sys.modules.pop("retrieval.eval_retrieval", None)

        module = importlib.import_module("retrieval.eval_retrieval")

        self.assertTrue(hasattr(module, "_build_model_and_io"))


if __name__ == "__main__":
    unittest.main()
