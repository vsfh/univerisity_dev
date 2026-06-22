import json
import random
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from PIL import Image

from dataset import ShiftedSatelliteDroneDataset


class DummyProcessor:
    def __init__(self, size=None):
        self.size = size


class DummyTokenizer:
    pass


def _write_image(path: Path, size=(64, 64)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size=size, color=(0, 0, 0)).save(path)


def _write_legacy_description(path: Path, prefix: str) -> None:
    payload = {
        "descriptions": {
            "150": f"{prefix} height 150",
            "200": f"{prefix} height 200",
        }
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_qwen_6_19_description(path: Path) -> None:
    payload = {
        "description_segments": {
            "150": [
                {"index": 1, "text": "qwen619 height 150 index 1"},
                {"index": 2, "text": "qwen619 height 150 index 2"},
                {"index": 3, "text": "qwen619 height 150 index 3"},
                {"index": 4, "text": "qwen619 height 150 index 4"},
                {"index": 5, "text": "qwen619 height 150 index 5"},
            ],
            "200": [
                {"index": 1, "text": "qwen619 height 200 index 1"},
            ],
        }
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


class DatasetRandomTextTest(unittest.TestCase):
    def test_dataset_collects_qwen_6_19_index_options_for_patch_height(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            train_sat_root = root / "sat"
            test_sat_root = root / "test_sat"
            drone_root = root / "drone"
            bbox_file = root / "bbox.json"
            train_split_file = root / "train.txt"

            _write_image(train_sat_root / "0000.png")
            train_split_file.write_text("0000\n", encoding="utf-8")
            bbox_file.write_text("{}", encoding="utf-8")

            drone_dir = drone_root / "0000"
            _write_image(drone_dir / "150_0.png")
            for name, prefix in (
                ("gemma_5_28_description.json", "gemma"),
                ("qwen_6_4_description.json", "qwen64"),
                ("qwen_6_7_description.json", "qwen67"),
                ("qwen_6_12_description.json", "qwen612"),
            ):
                _write_legacy_description(drone_dir / name, prefix)
            _write_qwen_6_19_description(drone_dir / "qwen_6_19_description.json")

            dataset = ShiftedSatelliteDroneDataset(
                processor=DummyProcessor(),
                processor_sat=DummyProcessor(size={"height": 640, "width": 640}),
                tokenizer=DummyTokenizer(),
                split="train",
                train_satellite_root=str(train_sat_root),
                test_satellite_root=str(test_sat_root),
                drone_image_root=str(drone_root),
                bbox_file=str(bbox_file),
                train_split_file=str(train_split_file),
            )

            sample = dataset.samples[0]
            self.assertEqual(150, sample["height"])
            self.assertEqual(
                {
                    "qwen619 height 150 index 1",
                    "qwen619 height 150 index 2",
                    "qwen619 height 150 index 3",
                    "qwen619 height 150 index 4",
                    "qwen619 height 150 index 5",
                },
                set(sample["text_options"]),
            )
            self.assertNotIn("qwen619 height 200 index 1", sample["text_options"])
            self.assertNotIn("gemma height 150", sample["text_options"])

            rng = random.Random(0)
            selected = {
                dataset._choose_sample_text(sample, rng=rng)
                for _ in range(20)
            }
            self.assertGreater(len(selected), 1)
            self.assertTrue(selected.issubset(set(sample["text_options"])))


if __name__ == "__main__":
    unittest.main()
