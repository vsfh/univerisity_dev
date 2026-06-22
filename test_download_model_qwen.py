import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch


MODULE_PATH = Path(__file__).resolve().parent / "tools" / "download_model.py"


def _load_download_model_module():
    spec = importlib.util.spec_from_file_location("download_model_for_test", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class DownloadQwenTest(unittest.TestCase):
    def test_download_qwen_3_6_35b_a3b_uses_default_cache(self):
        module = _load_download_model_module()
        snapshot_download = Mock(return_value="/fake/qwen/snapshot")
        fake_hf_hub = types.SimpleNamespace(snapshot_download=snapshot_download)

        with patch.dict(sys.modules, {"huggingface_hub": fake_hf_hub}):
            with patch.object(module.os, "makedirs") as makedirs:
                with patch.dict(module.os.environ, {}, clear=True):
                    result = module.download_qwen_3_6_35b_a3b(direct_fallback=False)

        makedirs.assert_called_once_with("/media/4tb/feihong/hf_cache", exist_ok=True)
        snapshot_download.assert_called_once_with(
            repo_id="Qwen/Qwen3.6-35B-A3B",
            cache_dir="/media/4tb/feihong/hf_cache",
            revision=None,
            resume_download=True,
            max_workers=1,
        )
        self.assertEqual(result["model_id"], "Qwen/Qwen3.6-35B-A3B")
        self.assertEqual(result["cache_dir"], "/media/4tb/feihong/hf_cache")
        self.assertEqual(result["snapshot_dir"], "/fake/qwen/snapshot")
        self.assertEqual(result["hf_endpoint"], "https://hf-mirror.com")

    def test_cli_exposes_qwen_3_6_35b_a3b(self):
        source = MODULE_PATH.read_text(encoding="utf-8")

        self.assertIn('"qwen_3_6_35b_a3b"', source)
        self.assertIn('args.model == "qwen_3_6_35b_a3b"', source)
        self.assertIn("download_qwen_3_6_35b_a3b(", source)


if __name__ == "__main__":
    unittest.main()
