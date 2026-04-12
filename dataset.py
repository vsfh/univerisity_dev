import json
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import Dataset


# --- Configuration ---
TRAIN_SATELLITE_ROOT = "/data/feihong/image_2048"
TEST_SATELLITE_ROOT = "/data/feihong/img_test_1"
DRONE_IMAGE_ROOT = "/data/feihong/drone_img"
BBOX_FILE = "/data/feihong/ckpt/bbox_test_1.json"
TRAIN_MAX_SATELLITE_ID = 1065
TRAIN_SPLIT_FILE = "/data/feihong/ckpt/train.txt"
VAL_SPLIT_FILE = "/data/feihong/ckpt/val.txt"
TEST_SPLIT_FILE = "/data/feihong/ckpt/test_1.txt"
VAL_SPLIT_COUNT = 20
TEST_SPLIT_COUNT = 400
MAX_TEXT_LENGTH = 64
DEFAULT_SAT_TARGET_SIZE = (432, 768)
DEFAULT_TEST_CROP_RATIO = 1.0
DEFAULT_SUBSET_HEIGHTS = [150, 200, 250, 300]
DEFAULT_SUBSET_ANGLES = [0, 90, 180, 270]
DEFAULT_ENABLE_TIMING_LOG = False
DEFAULT_TIMING_LOG_INTERVAL = 200
DEFAULT_MODEL_NAME = "google/siglip-base-patch16-224"
DEFAULT_CACHE_DIR = "/data/feihong/hf_cache"
TRAIN_HEIGHT_TO_BOX_SIZE = {
	150: 330//2,
	200: 414//2,
	250: 527//2,
	300: 678//2,
}


def _parse_patch_name(path: Path) -> Optional[Tuple[int, int]]:
	"""Parse drone patch name like 150_315.png into (height, angle)."""
	parts = path.stem.split("_")
	if len(parts) != 2:
		return None
	try:
		height = int(parts[0])
		angle = int(parts[1])
	except ValueError:
		return None
	return height, angle


def _safe_int(text: str) -> Optional[int]:
	try:
		return int(text)
	except ValueError:
		return None


def _parse_split_ids_from_text_file(file_path: Path) -> Set[int]:
	"""Parse numeric IDs from a split file with one token per line or CSV first column."""
	if not file_path.exists():
		return set()

	ids: Set[int] = set()
	with open(file_path, "r", encoding="utf-8") as f:
		for raw_line in f:
			line = raw_line.strip()
			if not line:
				continue

			first_field = line.split(",")[0].strip()
			token = Path(first_field).name
			candidates = [first_field, token, Path(first_field).stem, Path(token).stem]

			parsed = None
			for candidate in candidates:
				parsed = _safe_int(candidate)
				if parsed is not None:
					break

			if parsed is not None:
				ids.add(parsed)

	return ids


def create_val_test_split_files(
	test_satellite_root: str = TEST_SATELLITE_ROOT,
	train_split_file: str = TRAIN_SPLIT_FILE,
	val_split_file: str = VAL_SPLIT_FILE,
	test_split_file: str = TEST_SPLIT_FILE,
	val_count: int = VAL_SPLIT_COUNT,
	test_count: int = TEST_SPLIT_COUNT,
	seed: int = 43,
) -> Tuple[int, int]:
	"""
	Create val/test split files from shifted satellite set, excluding train IDs.
	Returns (num_val_written, num_test_written).
	"""
	test_root = Path(test_satellite_root)
	if not test_root.is_dir():
		raise NotADirectoryError(f"Satellite directory does not exist: {test_root}")

	train_ids = _parse_split_ids_from_text_file(Path(train_split_file))

	all_ids: Set[int] = set()
	for ext in ("*.png", "*.jpg", "*.jpeg", "*.webp"):
		for path in test_root.glob(ext):
			parsed = _safe_int(path.stem)
			if parsed is not None:
				all_ids.add(parsed)

	candidate_ids = sorted(idx for idx in all_ids if idx not in train_ids)
	if not candidate_ids:
		raise ValueError(
			"No candidate IDs found in shifted satellite root after excluding train split IDs."
		)

	rng = random.Random(seed)
	rng.shuffle(candidate_ids)

	actual_val_count = min(max(0, int(val_count)), len(candidate_ids))
	remaining = candidate_ids[actual_val_count:]
	actual_test_count = min(max(0, int(test_count)), len(remaining))

	val_ids = sorted(candidate_ids[:actual_val_count])
	test_ids = sorted(remaining[:actual_test_count])

	for out_path in (Path(val_split_file), Path(test_split_file)):
		out_path.parent.mkdir(parents=True, exist_ok=True)

	with open(val_split_file, "w", encoding="utf-8") as f:
		for idx in val_ids:
			f.write(f"{idx:04d}\n")

	with open(test_split_file, "w", encoding="utf-8") as f:
		for idx in test_ids:
			f.write(f"{idx:04d}\n")

	return len(val_ids), len(test_ids)


def _resolve_processor_target_size(
	processor_sat: Any,
	fallback_size: Tuple[int, int],
) -> Tuple[int, int]:
	"""Infer target (height, width) used by the satellite image processor."""
	size = getattr(processor_sat, "size", None)
	if isinstance(size, dict):
		if "height" in size and "width" in size:
			return int(size["height"]), int(size["width"])
		if "shortest_edge" in size:
			edge = int(size["shortest_edge"])
			return edge, edge
	if isinstance(size, int):
		return size, size
	if isinstance(size, Sequence) and len(size) == 2:
		return int(size[0]), int(size[1])
	return fallback_size


def _normalize_subset_values(
	values: Optional[Sequence[int]],
	default_values: Sequence[int],
) -> Set[int]:
	if values is None:
		return {int(v) for v in default_values}
	return {int(v) for v in values}


def _sample_edge_biased(min_value: float, max_value: float, alpha: float = 0.35) -> float:
	"""Sample in [min_value, max_value] with higher probability near both edges."""
	if max_value <= min_value:
		return float(min_value)
	edge_alpha = max(1e-3, float(alpha))
	u = random.betavariate(edge_alpha, edge_alpha)
	return float(min_value) + (float(max_value) - float(min_value)) * u


def _build_center_bbox(
	image_width: int,
	image_height: int,
	bbox_width: float,
	bbox_height: float,
) -> List[float]:
	"""Build a center-aligned bbox in (x1, y1, x2, y2) format."""
	cx = image_width / 2.0
	cy = image_height / 2.0
	half_w = bbox_width / 2.0
	half_h = bbox_height / 2.0

	x1 = max(0.0, cx - half_w)
	y1 = max(0.0, cy - half_h)
	x2 = min(float(image_width), cx + half_w)
	y2 = min(float(image_height), cy + half_h)
	return [x1, y1, x2, y2]


def _augment_satellite_image_with_bbox(
	image: Image.Image,
	bbox: Sequence[float],
	mode: str,
	target_size: Optional[Tuple[int, int]] = None,
	test_crop_ratio: float = DEFAULT_TEST_CROP_RATIO,
) -> Tuple[Image.Image, List[float]]:
	"""Crop around bbox and resize using OpenCV, enforcing >=2.5x bbox coverage."""
	if target_size is None:
		target_h, target_w = image.height, image.width
	else:
		target_h, target_w = int(target_size[0]), int(target_size[1])

	orig_w, orig_h = image.size
	x1, y1, x2, y2 = [float(v) for v in bbox]

	if mode != "train":
		ratio = float(test_crop_ratio)
		# Keep original behavior when ratio >= 1.0 (no test-time crop).
		if ratio >= 1.0:
			if (orig_h, orig_w) == (target_h, target_w):
				return image, [float(v) for v in bbox]
			scale_x = target_w / max(float(orig_w), 1.0)
			scale_y = target_h / max(float(orig_h), 1.0)
			resized_bbox = [x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y]
			img_np = np.asarray(image, dtype=np.uint8)
			resized_np = cv2.resize(img_np, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
			return Image.fromarray(resized_np), resized_bbox

		ratio = max(1e-3, min(ratio, 1.0))
		bbox_w = max(1.0, x2 - x1)
		bbox_h = max(1.0, y2 - y1)

		# Ensure crop can contain full bbox and provide at least 3x bbox context when feasible.
		min_ratio_bbox = max(
			bbox_w / max(float(orig_w), 1.0),
			bbox_h / max(float(orig_h), 1.0),
		)
		min_ratio_context = max(
			2.5 * bbox_w / max(float(orig_w), 1.0),
			2.5 * bbox_h / max(float(orig_h), 1.0),
		)
		ratio = max(ratio, min_ratio_bbox, min_ratio_context)

		crop_w = max(1.0, min(float(orig_w), float(orig_w) * ratio))
		crop_h = max(1.0, min(float(orig_h), float(orig_h) * ratio))

		left_min = max(0.0, x2 - crop_w)
		left_max = min(float(orig_w) - crop_w, x1)
		top_min = max(0.0, y2 - crop_h)
		top_max = min(float(orig_h) - crop_h, y1)

		if left_max < left_min:
			left_min = left_max = min(max(0.0, x1), float(orig_w) - crop_w)
		if top_max < top_min:
			top_min = top_max = min(max(0.0, y1), float(orig_h) - crop_h)

		# Deterministic crop position at the center of the feasible region.
		left = 0.5 * (left_min + left_max)
		top = 0.5 * (top_min + top_max)
		right = left + crop_w
		bottom = top + crop_h

		img_np = np.asarray(image, dtype=np.uint8)
		left_i = int(max(0, min(orig_w - 1, round(left))))
		top_i = int(max(0, min(orig_h - 1, round(top))))
		right_i = int(max(left_i + 1, min(orig_w, round(right))))
		bottom_i = int(max(top_i + 1, min(orig_h, round(bottom))))

		cropped_np = img_np[top_i:bottom_i, left_i:right_i]
		resized_np = cv2.resize(cropped_np, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

		actual_crop_w = float(max(1, right_i - left_i))
		actual_crop_h = float(max(1, bottom_i - top_i))
		scale_x = target_w / actual_crop_w
		scale_y = target_h / actual_crop_h
		new_bbox = [
			(x1 - left_i) * scale_x,
			(y1 - top_i) * scale_y,
			(x2 - left_i) * scale_x,
			(y2 - top_i) * scale_y,
		]
		return Image.fromarray(resized_np), new_bbox

	bbox_w = max(1.0, x2 - x1)
	bbox_h = max(1.0, y2 - y1)

	# Ensure sampled ratio can contain bbox and at least 3x bbox context when feasible.
	min_ratio_context = max(
		3.0 * bbox_w / max(float(orig_w), 1.0),
		3.0 * bbox_h / max(float(orig_h), 1.0),
	)
	ratio = random.uniform(min_ratio_context, min_ratio_context*3/4)

	crop_w = max(1.0, min(float(orig_w), float(orig_w) * ratio))
	crop_h = max(1.0, min(float(orig_h), float(orig_h) * ratio))

	min_left = max(0.0, x2 - crop_w)
	max_left = min(float(orig_w) - crop_w, x1)
	min_top = max(0.0, y2 - crop_h)
	max_top = min(float(orig_h) - crop_h, y1)
	if max_left < min_left:
		max_left = min_left
	if max_top < min_top:
		max_top = min_top

	left = _sample_edge_biased(min_left, max_left)
	top = _sample_edge_biased(min_top, max_top)
	right = left + crop_w
	bottom = top + crop_h

	img_np = np.asarray(image, dtype=np.uint8)
	left_i = int(max(0, min(orig_w - 1, round(left))))
	top_i = int(max(0, min(orig_h - 1, round(top))))
	right_i = int(max(left_i + 1, min(orig_w, round(right))))
	bottom_i = int(max(top_i + 1, min(orig_h, round(bottom))))

	cropped_np = img_np[top_i:bottom_i, left_i:right_i]
	resized_np = cv2.resize(cropped_np, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

	actual_crop_w = float(max(1, right_i - left_i))
	actual_crop_h = float(max(1, bottom_i - top_i))
	scale_x = target_w / actual_crop_w
	scale_y = target_h / actual_crop_h
	new_x1 = (x1 - left_i) * scale_x
	new_y1 = (y1 - top_i) * scale_y
	new_x2 = (x2 - left_i) * scale_x
	new_y2 = (y2 - top_i) * scale_y

	new_bbox = [
		max(0.0, min(float(target_w), new_x1)),
		max(0.0, min(float(target_h), new_y1)),
		max(0.0, min(float(target_w), new_x2)),
		max(0.0, min(float(target_h), new_y2)),
	]

	return Image.fromarray(resized_np), new_bbox


def _augment_satellite_image_with_bbox_640(
	image: Image.Image,
	bbox: Sequence[float],
	mode: str,
	target_size: Tuple[int, int] = (640, 640),
	test_crop_ratio: float = DEFAULT_TEST_CROP_RATIO,
) -> Tuple[Image.Image, List[float]]:
	"""Aspect-ratio-aware crop path; keeps crop_w/crop_h aligned with target_w/target_h."""
	target_h, target_w = int(target_size[0]), int(target_size[1])
	orig_w, orig_h = image.size
	x1, y1, x2, y2 = [float(v) for v in bbox]
	bbox_w = max(1.0, x2 - x1)
	bbox_h = max(1.0, y2 - y1)

	target_aspect = float(target_w) / max(float(target_h), 1.0)

	if float(orig_w) / max(float(orig_h), 1.0) >= target_aspect:
		max_crop_h = float(orig_h)
		max_crop_w = max_crop_h * target_aspect
	else:
		max_crop_w = float(orig_w)
		max_crop_h = max_crop_w / max(target_aspect, 1e-6)

	if mode == "train":
		context_scale = 4.0
		min_crop_w = max(bbox_w, bbox_h * target_aspect) * context_scale
	else:
		ratio = max(1e-3, min(1.0, float(test_crop_ratio)))
		context_scale = 4.0
		min_crop_w = max(bbox_w, bbox_h * target_aspect) * context_scale
		# Keep ratio<1.0 as an optional lower bound; ratio>=1.0 should not force full-size crop.
		if ratio < 1.0:
			ratio_crop_w = max_crop_w * ratio
			min_crop_w = max(min_crop_w, ratio_crop_w)

	min_crop_w = min(max_crop_w, max(1.0, min_crop_w))
	min_crop_h = min(max_crop_h, min_crop_w / max(target_aspect, 1e-6))
	min_crop_w = min(max_crop_w, min_crop_h * target_aspect)

	if mode == "train":
		# scale_up = random.uniform(1.0, 1.25)
		scale_up = 1
		crop_w = min(max_crop_w, min_crop_w * scale_up)
		crop_h = min(max_crop_h, crop_w / max(target_aspect, 1e-6))
		crop_w = min(max_crop_w, crop_h * target_aspect)
	else:
		crop_w = min_crop_w
		crop_h = min_crop_h

	left_min = max(0.0, x2 - crop_w)
	left_max = min(float(orig_w) - crop_w, x1)
	top_min = max(0.0, y2 - crop_h)
	top_max = min(float(orig_h) - crop_h, y1)

	if left_max < left_min or top_max < top_min:
		cx = 0.5 * (x1 + x2)
		cy = 0.5 * (y1 + y2)
		left = min(max(0.0, cx - 0.5 * crop_w), float(orig_w) - crop_w)
		top = min(max(0.0, cy - 0.5 * crop_h), float(orig_h) - crop_h)
	else:
		if mode == "train":
			left = _sample_edge_biased(left_min, left_max)
			top = _sample_edge_biased(top_min, top_max)
		else:
			left = 0.5 * (left_min + left_max)
			top = 0.5 * (top_min + top_max)

	right = left + crop_w
	bottom = top + crop_h

	img_np = np.asarray(image, dtype=np.uint8)
	left_i = int(max(0, min(orig_w - 1, round(left))))
	top_i = int(max(0, min(orig_h - 1, round(top))))
	right_i = int(max(left_i + 1, min(orig_w, round(right))))
	bottom_i = int(max(top_i + 1, min(orig_h, round(bottom))))

	cropped_np = img_np[top_i:bottom_i, left_i:right_i]
	resized_np = cv2.resize(cropped_np, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

	actual_crop_w = float(max(1, right_i - left_i))
	actual_crop_h = float(max(1, bottom_i - top_i))
	scale_x = target_w / actual_crop_w
	scale_y = target_h / actual_crop_h
	new_bbox = [
		(x1 - left_i) * scale_x,
		(y1 - top_i) * scale_y,
		(x2 - left_i) * scale_x,
		(y2 - top_i) * scale_y,
	]
	new_bbox = [
		max(0.0, min(float(target_w), new_bbox[0])),
		max(0.0, min(float(target_h), new_bbox[1])),
		max(0.0, min(float(target_w), new_bbox[2])),
		max(0.0, min(float(target_h), new_bbox[3])),
	]

	return Image.fromarray(resized_np), new_bbox


def _sanitize_bbox_xyxy(
	bbox: Sequence[float],
	image_width: float,
	image_height: float,
	min_size: float = 1.0,
) -> List[float]:
	"""Clamp bbox to valid range [0, size) and enforce positive width/height."""
	max_x = max(0.0, float(image_width) - 1e-3)
	max_y = max(0.0, float(image_height) - 1e-3)

	x1, y1, x2, y2 = [float(v) for v in bbox]
	x1 = max(0.0, min(max_x, x1))
	y1 = max(0.0, min(max_y, y1))
	x2 = max(0.0, min(max_x, x2))
	y2 = max(0.0, min(max_y, y2))

	if x2 <= x1:
		x2 = min(max_x, x1 + min_size)
		if x2 <= x1:
			x1 = max(0.0, max_x - min_size)
			x2 = max_x

	if y2 <= y1:
		y2 = min(max_y, y1 + min_size)
		if y2 <= y1:
			y1 = max(0.0, max_y - min_size)
			y2 = max_y

	return [x1, y1, x2, y2]


class ShiftedSatelliteDroneDataset(Dataset):
	def __init__(
		self,
		processor,
		processor_sat,
		tokenizer,
		split: str,
		train_satellite_root: str = TRAIN_SATELLITE_ROOT,
		test_satellite_root: str = TEST_SATELLITE_ROOT,
		drone_image_root: str = DRONE_IMAGE_ROOT,
		bbox_file: str = BBOX_FILE,
		train_max_satellite_id: int = TRAIN_MAX_SATELLITE_ID,
		train_split_file: str = TRAIN_SPLIT_FILE,
		val_split_file: str = VAL_SPLIT_FILE,
		test_split_file: str = TEST_SPLIT_FILE,
		val_split_count: int = VAL_SPLIT_COUNT,
		max_text_length: int = MAX_TEXT_LENGTH,
		sat_target_size: Tuple[int, int] = DEFAULT_SAT_TARGET_SIZE,
		test_crop_ratio: float = DEFAULT_TEST_CROP_RATIO,
		subset_heights: Optional[Sequence[int]] = None,
		subset_angles: Optional[Sequence[int]] = None,
		enable_timing_log: bool = DEFAULT_ENABLE_TIMING_LOG,
		timing_log_interval: int = DEFAULT_TIMING_LOG_INTERVAL,
	):
		if split not in {"train", "val", "test"}:
			raise ValueError("split must be 'train', 'val', or 'test'.")

		self.processor = processor
		self.processor_sat = processor_sat
		self.tokenizer = tokenizer
		self.split = split
		self.train_max_satellite_id = train_max_satellite_id
		self.train_split_file = Path(train_split_file)
		self.val_split_file = Path(val_split_file)
		self.test_split_file = Path(test_split_file)
		self.val_split_count = max(0, int(val_split_count))
		self.max_text_length = max_text_length
		self.train_split_ids: Optional[Set[int]] = None
		self.val_split_ids: Optional[Set[int]] = None
		self.test_split_ids: Optional[Set[int]] = None
		if self.split == "train":
			self.train_split_ids = self._load_train_split_ids()
		elif self.split == "val":
			self.val_split_ids = self._load_optional_split_ids(self.val_split_file)
		elif self.split == "test":
			self.test_split_ids = self._load_optional_split_ids(self.test_split_file)
		self.sat_target_size = _resolve_processor_target_size(
			processor_sat,
			sat_target_size,
		)
		self.test_crop_ratio = max(0.0, min(1.0, float(test_crop_ratio)))
		self.subset_heights = _normalize_subset_values(
			subset_heights,
			DEFAULT_SUBSET_HEIGHTS,
		)
		self.subset_angles = _normalize_subset_values(
			subset_angles,
			DEFAULT_SUBSET_ANGLES,
		)
		self.enable_timing_log = bool(enable_timing_log)
		self.timing_log_interval = max(1, int(timing_log_interval))
		self._timing_acc: Dict[str, float] = {
			"io": 0.0,
			"query_proc": 0.0,
			"search_proc": 0.0,
			"total": 0.0,
		}
		self._timing_count = 0

		self.train_satellite_root = Path(train_satellite_root)
		self.test_satellite_root = Path(test_satellite_root)
		self.drone_image_root = Path(drone_image_root)

		with open(bbox_file, "r", encoding="utf-8") as f:
			self.bbox_dict: Dict[str, Dict[str, List[float]]] = json.load(f)

		self.samples: List[Dict[str, Any]] = self._build_samples()
		if not self.samples:
			raise ValueError(f"No valid samples found for split={split}.")

	def _tokenize_text(self, text: str) -> torch.Tensor:
		text_inputs = self.tokenizer(
			text,
			padding="max_length",
			truncation=True,
			max_length=self.max_text_length,
			return_tensors="pt",
		)
		input_ids = text_inputs["input_ids"]
		if not isinstance(input_ids, torch.Tensor):
			input_ids = torch.tensor(input_ids, dtype=torch.long)
		if input_ids.ndim > 1:
			input_ids = input_ids[0]
		return input_ids.long().clone()

	def _list_satellite_files(self) -> List[Path]:
		sat_root = self.train_satellite_root if self.split == "train" else self.test_satellite_root
		if not sat_root.is_dir():
			raise NotADirectoryError(f"Satellite directory does not exist: {sat_root}")

		files: List[Path] = []
		for ext in ("*.png", "*.jpg", "*.jpeg", "*.webp"):
			files.extend(sat_root.glob(ext))

		def sort_key(path: Path) -> Tuple[int, str]:
			parsed = _safe_int(path.stem)
			if parsed is None:
				return (10**9, path.name)
			return (parsed, path.name)

		sorted_files = sorted(files, key=sort_key)
		if self.split == "train":
			if not self.train_split_ids:
				raise ValueError("Train split IDs are empty.")
			return [
				path
				for path in sorted_files
				if _safe_int(path.stem) in self.train_split_ids
			]
		if self.split == "val":
			if self.val_split_ids:
				return [
					path
					for path in sorted_files
					if _safe_int(path.stem) in self.val_split_ids
				]
			return sorted_files[: self.val_split_count]
		if self.split == "test":
			if self.test_split_ids:
				return [
					path
					for path in sorted_files
					if _safe_int(path.stem) in self.test_split_ids
				]
			return sorted_files[self.val_split_count :]
		return sorted_files

	def _load_train_split_ids(self) -> Set[int]:
		if not self.train_split_file.exists():
			raise FileNotFoundError(
				f"Train split file not found: {self.train_split_file}"
			)

		train_ids = _parse_split_ids_from_text_file(self.train_split_file)

		if not train_ids:
			raise ValueError(
				f"No valid numeric folder names found in train split file: {self.train_split_file}"
			)

		return train_ids

	def _load_optional_split_ids(self, split_file: Path) -> Optional[Set[int]]:
		if not split_file.exists():
			return None
		parsed_ids = _parse_split_ids_from_text_file(split_file)
		if not parsed_ids:
			return None
		return parsed_ids

	def _select_drone_dir(self, satellite_id: str) -> Optional[Path]:
		candidate_1 = self.drone_image_root / satellite_id
		if candidate_1.is_dir():
			return candidate_1

		parsed = _safe_int(satellite_id)
		if parsed is None:
			return None

		candidate_2 = self.drone_image_root / f"{parsed:04d}"
		if candidate_2.is_dir():
			return candidate_2
		return None

	def _load_unified_text(self, drone_dir: Path) -> Optional[str]:
		text_path = drone_dir / "unified_description.json"
		if not text_path.exists():
			return None

		with open(text_path, "r", encoding="utf-8") as f:
			payload = json.load(f)

		text = payload.get("unified_description", "")
		if not isinstance(text, str):
			return None

		text = text.strip()
		if not text:
			return None
		return text

	def _build_samples(self) -> List[Dict[str, Any]]:
		samples: List[Dict[str, Any]] = []

		for sat_path in self._list_satellite_files():
			sat_id_int = _safe_int(sat_path.stem)
			if sat_id_int is None:
				continue

			if (
				self.split == "train"
				and self.train_split_ids is None
				and not (0 <= sat_id_int <= self.train_max_satellite_id)
			):
				continue

			sat_id = f"{sat_id_int:04d}"
			drone_dir = self._select_drone_dir(sat_id)
			if drone_dir is None:
				continue

			unified_text = self._load_unified_text(drone_dir)
			if unified_text is None:
				continue

			sat_bbox_dict: Dict[str, List[float]] = {}
			if self.split in {"val", "test"}:
				sat_bbox_dict = self.bbox_dict.get(sat_id, {})
				if not isinstance(sat_bbox_dict, dict):
					continue

			drone_files = sorted(drone_dir.glob("*.png"))
			if not drone_files:
				continue

			valid_count_for_sat = 0
			for drone_path in drone_files:
				parsed = _parse_patch_name(drone_path)
				if parsed is None:
					continue

				height, angle = parsed
				if int(height) not in self.subset_heights:
					continue
				if int(angle) not in self.subset_angles:
					continue
				if str(angle).endswith("5"):
					continue

				sample: Dict[str, Any] = {
					"satellite_path": str(sat_path),
					"drone_path": str(drone_path),
					"satellite_id": sat_id_int,
					"height": height,
					"angle": angle,
					"text": unified_text,
				}

				if self.split in {"val", "test"}:
					bbox = sat_bbox_dict.get(str(height))
					if bbox is None or len(bbox) != 4:
						continue
					sample["bbox"] = [float(v) for v in bbox]
				else:
					bbox_size = TRAIN_HEIGHT_TO_BOX_SIZE.get(height)
					if bbox_size is None:
						continue
					sample["bbox_size"] = float(bbox_size)

				samples.append(sample)
				valid_count_for_sat += 1

			# Remove satellite images that do not have any matched drone samples.
			if valid_count_for_sat == 0:
				continue

		return samples

	def __len__(self) -> int:
		return len(self.samples)

	def __getitem__(self, idx: int) -> Dict[str, Any]:
		sample = self.samples[idx]

		t_io_start = time.perf_counter()
		try:
			query_image = Image.open(sample["drone_path"]).convert("RGB")
			search_image = Image.open(sample["satellite_path"]).convert("RGB")
		except FileNotFoundError:
			return self.__getitem__((idx + 1) % len(self.samples))

		orig_w, orig_h = search_image.size

		if self.split == "train":
			bbox_size = float(sample["bbox_size"])
			base_bbox = _build_center_bbox(
				image_width=orig_w,
				image_height=orig_h,
				bbox_width=bbox_size,
				bbox_height=bbox_size,
			)
		else:
			base_bbox = [float(v) for v in sample["bbox"]]

		augmented_search_image, augmented_bbox = _augment_satellite_image_with_bbox_640(
			image=search_image,
			bbox=base_bbox,
			mode=self.split,
			target_size=self.sat_target_size,
			test_crop_ratio=self.test_crop_ratio,
		)

		target_h, target_w = self.sat_target_size
		resized_bbox = [float(v) for v in augmented_bbox]
		resized_bbox = _sanitize_bbox_xyxy(
			resized_bbox,
			image_width=float(target_w),
			image_height=float(target_h),
		)

		center_x = (resized_bbox[0] + resized_bbox[2]) / 2.0 / max(float(target_w), 1.0)
		center_y = (resized_bbox[1] + resized_bbox[3]) / 2.0 / max(float(target_h), 1.0)
		col_idx = min(int(center_x * 3), 2)
		row_idx = min(int(center_y * 3), 2)
		index = row_idx * 3 + col_idx

		input_ids = self._tokenize_text(sample["text"])
		query_inputs = self.processor(images=query_image, return_tensors="pt")
		search_inputs = self.processor_sat(
			images=augmented_search_image,
			return_tensors="pt",
		)

		return {
			"target_pixel_values": query_inputs["pixel_values"][0],
			"search_pixel_values": search_inputs["pixel_values"][0],
			"input_ids": input_ids,
			"index": index,
			"bbox": torch.tensor(resized_bbox, dtype=torch.float32),
			"height": torch.tensor(sample["height"], dtype=torch.long),
			"angle": torch.tensor(sample["angle"], dtype=torch.long),
			"drone_path": sample["drone_path"],
			"satellite_path": sample["satellite_path"],
		}


def initialize_shifted_dataset_like_unified_siglip_supp(
	split: str,
	model_name: str = DEFAULT_MODEL_NAME,
	cache_dir: str = DEFAULT_CACHE_DIR,
	train_satellite_root: str = TRAIN_SATELLITE_ROOT,
	test_satellite_root: str = TEST_SATELLITE_ROOT,
	drone_image_root: str = DRONE_IMAGE_ROOT,
	bbox_file: str = BBOX_FILE,
	train_max_satellite_id: int = TRAIN_MAX_SATELLITE_ID,
	train_split_file: str = TRAIN_SPLIT_FILE,
	val_split_file: str = VAL_SPLIT_FILE,
	test_split_file: str = TEST_SPLIT_FILE,
	val_split_count: int = VAL_SPLIT_COUNT,
	max_text_length: int = MAX_TEXT_LENGTH,
	sat_target_size: Tuple[int, int] = DEFAULT_SAT_TARGET_SIZE,
	test_crop_ratio: float = DEFAULT_TEST_CROP_RATIO,
	subset_heights: Optional[Sequence[int]] = None,
	subset_angles: Optional[Sequence[int]] = None,
	enable_timing_log: bool = DEFAULT_ENABLE_TIMING_LOG,
	timing_log_interval: int = DEFAULT_TIMING_LOG_INTERVAL,
) -> ShiftedSatelliteDroneDataset:
	"""Initialize dataset with SigLIP processors/tokenizer like unified_siglip_supp.py."""
	from transformers import AutoImageProcessor, AutoTokenizer

	processor = AutoImageProcessor.from_pretrained(
		model_name,
		cache_dir=cache_dir,
	)
	processor_sat = AutoImageProcessor.from_pretrained(
		model_name,
		cache_dir=cache_dir,
		size={"height": sat_target_size[0], "width": sat_target_size[1]},
	)
	tokenizer = AutoTokenizer.from_pretrained(model_name)

	return ShiftedSatelliteDroneDataset(
		processor=processor,
		processor_sat=processor_sat,
		tokenizer=tokenizer,
		split=split,
		train_satellite_root=train_satellite_root,
		test_satellite_root=test_satellite_root,
		drone_image_root=drone_image_root,
		bbox_file=bbox_file,
		train_max_satellite_id=train_max_satellite_id,
		train_split_file=train_split_file,
		val_split_file=val_split_file,
		test_split_file=test_split_file,
		val_split_count=val_split_count,
		max_text_length=max_text_length,
		sat_target_size=sat_target_size,
		test_crop_ratio=test_crop_ratio,
		subset_heights=subset_heights,
		subset_angles=subset_angles,
		enable_timing_log=enable_timing_log,
		timing_log_interval=timing_log_interval,
	)


def _draw_bbox_on_image(
	image: Image.Image,
	bbox: Sequence[float],
	label: str,
	color: Tuple[int, int, int],
) -> Image.Image:
	canvas = image.convert("RGB").copy()
	draw = ImageDraw.Draw(canvas)
	x1, y1, x2, y2 = [float(v) for v in bbox]
	draw.rectangle((x1, y1, x2, y2), outline=color, width=4)
	draw.text((8, 8), label, fill=color)
	return canvas


def _resize_to_height(image: Image.Image, target_height: int) -> Image.Image:
	if image.height == target_height:
		return image
	new_width = max(1, int(round(image.width * target_height / max(1, image.height))))
	return image.resize((new_width, target_height), Image.Resampling.BILINEAR)


def visualize_shifted_dataset_sample(
	dataset: ShiftedSatelliteDroneDataset,
	sample_index: int = 0,
	output_path: str = "runs/visualizations/shifted_dataset_sample.png",
	random_seed: Optional[int] = None,
) -> str:
	"""Visualize satellite bbox before/after augmentation and corresponding drone image."""
	if sample_index < 0 or sample_index >= len(dataset.samples):
		raise IndexError(f"sample_index={sample_index} is out of range.")

	rng_state = random.getstate()
	try:
		if random_seed is not None:
			random.seed(random_seed)

		sample = dataset.samples[sample_index]
		satellite_image = Image.open(sample["satellite_path"]).convert("RGB")
		drone_image = Image.open(sample["drone_path"]).convert("RGB")

		orig_w, orig_h = satellite_image.size
		if dataset.split == "train":
			bbox_size = float(sample["bbox_size"])
			base_bbox = _build_center_bbox(
				image_width=orig_w,
				image_height=orig_h,
				bbox_width=bbox_size,
				bbox_height=bbox_size,
			)
		else:
			base_bbox = [float(v) for v in sample["bbox"]]

		augmented_image, augmented_bbox = _augment_satellite_image_with_bbox_640(
			image=satellite_image,
			bbox=base_bbox,
			mode=dataset.split,
			target_size=dataset.sat_target_size,
			test_crop_ratio=dataset.test_crop_ratio,
		)
		augmented_bbox = _sanitize_bbox_xyxy(
			augmented_bbox,
			image_width=float(augmented_image.width),
			image_height=float(augmented_image.height),
		)

		panel_sat_before = _draw_bbox_on_image(
			satellite_image,
			base_bbox,
			label="satellite before aug",
			color=(255, 80, 80),
		)
		panel_sat_after = _draw_bbox_on_image(
			augmented_image,
			augmented_bbox,
			label="satellite after aug",
			color=(80, 255, 120),
		)
		panel_drone = drone_image.convert("RGB").copy()
		ImageDraw.Draw(panel_drone).text((8, 8), "corresponding drone", fill=(80, 180, 255))

		target_height = 420
		panels = [
			_resize_to_height(panel_sat_before, target_height),
			_resize_to_height(panel_sat_after, target_height),
			_resize_to_height(panel_drone, target_height),
		]

		padding = 12
		canvas_width = sum(img.width for img in panels) + padding * (len(panels) + 1)
		canvas_height = target_height + 2 * padding
		canvas = Image.new("RGB", (canvas_width, canvas_height), color=(18, 18, 18))

		x_offset = padding
		for panel in panels:
			canvas.paste(panel, (x_offset, padding))
			x_offset += panel.width + padding

		output_file = Path(output_path)
		output_file.parent.mkdir(parents=True, exist_ok=True)
		canvas.save(output_file)
		return str(output_file)
	finally:
		if random_seed is not None:
			random.setstate(rng_state)


if __name__ == "__main__":
    for index in range(10):
        sample_index = index * 8
        # dataset = initialize_shifted_dataset_like_unified_siglip_supp(split="train")
        # visualize_shifted_dataset_sample(dataset, sample_index=sample_index, output_path=f"runs/visualizations/shifted_train_sample_{sample_index}.png", random_seed=42)

        dataset = initialize_shifted_dataset_like_unified_siglip_supp(split="test")
        visualize_shifted_dataset_sample(dataset, sample_index=sample_index, output_path=f"runs/visualizations/shifted_test_sample_{sample_index}.png", random_seed=42)