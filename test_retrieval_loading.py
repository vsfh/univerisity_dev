"""CPU regression checks for processor sizes and configured model loading."""
from pathlib import Path
from unittest import TestCase, main
from unittest.mock import patch

from transformers.image_utils import SizeDict

from retrieval.config import load_config
from retrieval.processors import image_hw
from retrieval.registry import _build_siglip


class RetrievalLoadingTests(TestCase):
    def test_transformers_sizes(self):
        cases = [
            (SizeDict(height=224, width=224), (224, 224)),
            (SizeDict(height=432, width=768), (432, 768)),
            (SizeDict(shortest_edge=224), (224, 224)),
            ({"height": 432, "width": 768}, (432, 768)),
            ({"shortest_edge": 224}, (224, 224)),
            ((432, 768), (432, 768)),
            ([224, 224], (224, 224)),
            (224, (224, 224)),
        ]
        for size, expected in cases:
            with self.subTest(size=size):
                self.assertEqual(image_hw(size), expected)
        with self.assertRaises(ValueError):
            image_hw(SizeDict())

    def test_siglip_builder_uses_configured_cache(self):
        cfg = {"model": {"model_name": "google/siglip2-base-patch16-224",
                         "cache_dir": "/custom/cache", "proj_dim": 768}}
        with patch("retrieval.train_siglip.Encoder") as encoder:
            self.assertIs(_build_siglip(cfg), encoder.return_value)
            encoder.assert_called_once_with(
                "google/siglip2-base-patch16-224", proj_dim=768,
                cache_dir="/custom/cache")

    def test_siglip_config_uses_cached_generation(self):
        path = Path(__file__).parent / "configs/retrieval/siglip.yaml"
        cfg = load_config(str(path))
        self.assertEqual(cfg["model"]["model_name"], "google/siglip2-base-patch16-224")


if __name__ == "__main__":
    main()
