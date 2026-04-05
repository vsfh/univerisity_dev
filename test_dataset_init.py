import json
from pathlib import Path

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


def _write_unified_text(drone_dir: Path, text: str = "test description") -> None:
    drone_dir.mkdir(parents=True, exist_ok=True)
    payload = {"unified_description": text}
    with open(drone_dir / "unified_description.json", "w", encoding="utf-8") as f:
        json.dump(payload, f)


def test_initialize_shifted_satellite_drone_dataset_train_split(tmp_path: Path) -> None:
    train_sat_root = tmp_path / "image_1024"
    test_sat_root = tmp_path / "image_1024_shifted"
    drone_root = tmp_path / "drone_img"
    bbox_file = tmp_path / "shifted_bboxes.json"

    _write_image(train_sat_root / "0000.png")

    drone_dir = drone_root / "0000"
    _write_unified_text(drone_dir)
    _write_image(drone_dir / "150_0.png")

    with open(bbox_file, "w", encoding="utf-8") as f:
        json.dump({}, f)

    dataset = ShiftedSatelliteDroneDataset(
        processor=DummyProcessor(),
        processor_sat=DummyProcessor(size={"height": 640, "width": 640}),
        tokenizer=DummyTokenizer(),
        split="train",
        train_satellite_root=str(train_sat_root),
        test_satellite_root=str(test_sat_root),
        drone_image_root=str(drone_root),
        bbox_file=str(bbox_file),
    )

    assert len(dataset) == 1
    assert dataset.samples[0]["satellite_id"] == 0
    assert dataset.samples[0]["height"] == 150


def test_initialize_shifted_satellite_drone_dataset_test_split(tmp_path: Path) -> None:
    train_sat_root = tmp_path / "image_1024"
    test_sat_root = tmp_path / "image_1024_shifted"
    drone_root = tmp_path / "drone_img"
    bbox_file = tmp_path / "shifted_bboxes.json"

    _write_image(test_sat_root / "1066.png")

    drone_dir = drone_root / "1066"
    _write_unified_text(drone_dir)
    _write_image(drone_dir / "150_0.png")

    with open(bbox_file, "w", encoding="utf-8") as f:
        json.dump({"1066": {"150": [10, 20, 30, 40]}}, f)

    dataset = ShiftedSatelliteDroneDataset(
        processor=DummyProcessor(),
        processor_sat=DummyProcessor(size={"height": 640, "width": 640}),
        tokenizer=DummyTokenizer(),
        split="test",
        train_satellite_root=str(train_sat_root),
        test_satellite_root=str(test_sat_root),
        drone_image_root=str(drone_root),
        bbox_file=str(bbox_file),
    )

    assert len(dataset) == 1
    assert dataset.samples[0]["satellite_id"] == 1066
    assert dataset.samples[0]["bbox"] == [10.0, 20.0, 30.0, 40.0]
