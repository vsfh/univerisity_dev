import json
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from PIL import Image, ImageDraw
from torch.utils.data import Dataset


# --- Configuration ---
TRAIN_SATELLITE_ROOT = "/data/feihong/image_1024"
TEST_SATELLITE_ROOT = "/data2/feihong/image_1024_shifted"
DRONE_IMAGE_ROOT = "/data2/feihong/drone_img"
BBOX_FILE = "/data2/feihong/ckpt/shifted_bboxes.json"
TRAIN_MAX_SATELLITE_ID = 1065
VAL_SPLIT_COUNT = 30
MAX_TEXT_LENGTH = 64
DEFAULT_SAT_TARGET_SIZE = (640, 640)
DEFAULT_ENABLE_TIMING_LOG = False
DEFAULT_TIMING_LOG_INTERVAL = 200
DEFAULT_MODEL_NAME = "google/siglip-base-patch16-224"
DEFAULT_CACHE_DIR = "/data/feihong/hf_cache"
TRAIN_HEIGHT_TO_BOX_SIZE = {
	150: 330,
	200: 414,
	250: 527,
	300: 678,
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
) -> Tuple[Image.Image, List[float]]:
	"""Randomly crop around bbox (keeping bbox inside) and resize to target size."""
	if target_size is None:
		target_h, target_w = image.height, image.width
	else:
		target_h, target_w = int(target_size[0]), int(target_size[1])

	if mode != "train":
		orig_w, orig_h = image.size
		if (orig_h, orig_w) == (target_h, target_w):
			return image, [float(v) for v in bbox]

		x1, y1, x2, y2 = [float(v) for v in bbox]
		scale_x = target_w / max(float(orig_w), 1.0)
		scale_y = target_h / max(float(orig_h), 1.0)
		resized_bbox = [x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y]
		resized_image = image.resize((target_w, target_h), Image.Resampling.BILINEAR)
		return resized_image, resized_bbox

	x1, y1, x2, y2 = [float(v) for v in bbox]
	orig_w, orig_h = image.size

	bbox_w = max(1.0, x2 - x1)
	bbox_h = max(1.0, y2 - y1)
	min_crop = max(bbox_w, bbox_h) * 1.2
	max_crop = float(min(orig_w, orig_h))
	crop_size = random.uniform(min_crop, max(min_crop, max_crop))
	crop_size = min(crop_size, float(orig_w), float(orig_h))

	min_left = max(0.0, x2 - crop_size)
	max_left = min(float(orig_w) - crop_size, x1)
	min_top = max(0.0, y2 - crop_size)
	max_top = min(float(orig_h) - crop_size, y1)

	if max_left < min_left:
		max_left = min_left
	if max_top < min_top:
		max_top = min_top

	left = random.uniform(min_left, max_left)
	top = random.uniform(min_top, max_top)
	right = left + crop_size
	bottom = top + crop_size

	cropped = image.crop((left, top, right, bottom))
	augmented = cropped.resize((target_w, target_h), Image.Resampling.BILINEAR)

	scale_x = target_w / max(crop_size, 1.0)
	scale_y = target_h / max(crop_size, 1.0)
	new_x1 = (x1 - left) * scale_x
	new_y1 = (y1 - top) * scale_y
	new_x2 = (x2 - left) * scale_x
	new_y2 = (y2 - top) * scale_y

	new_bbox = [
		max(0.0, min(float(target_w), new_x1)),
		max(0.0, min(float(target_h), new_y1)),
		max(0.0, min(float(target_w), new_x2)),
		max(0.0, min(float(target_h), new_y2)),
	]

	return augmented, new_bbox


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
		val_split_count: int = VAL_SPLIT_COUNT,
		max_text_length: int = MAX_TEXT_LENGTH,
		sat_target_size: Tuple[int, int] = DEFAULT_SAT_TARGET_SIZE,
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
		self.val_split_count = max(0, int(val_split_count))
		self.max_text_length = max_text_length
		self.sat_target_size = _resolve_processor_target_size(
			processor_sat,
			sat_target_size,
		)
		self.enable_timing_log = bool(enable_timing_log)
		self.timing_log_interval = max(1, int(timing_log_interval))
		self._timing_acc: Dict[str, float] = {
			"io": 0.0,
			"augment": 0.0,
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

		# Tokenization in __getitem__ is a CPU hotspot; cache per unique text once.
		self._token_cache: Dict[str, torch.Tensor] = {}
		for sample in self.samples:
			text = sample["text"]
			if text not in self._token_cache:
				self._token_cache[text] = self._tokenize_text(text)

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
		if self.split == "val":
			return sorted_files[: self.val_split_count]
		if self.split == "test":
			return sorted_files[self.val_split_count :]
		return sorted_files

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

			if self.split == "train" and not (0 <= sat_id_int <= self.train_max_satellite_id):
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
		t_start = time.perf_counter()
		sample = self.samples[idx]

		t_io_start = time.perf_counter()
		try:
			query_image = Image.open(sample["drone_path"]).convert("RGB")
			search_image = Image.open(sample["satellite_path"]).convert("RGB")
		except FileNotFoundError:
			return self.__getitem__((idx + 1) % len(self.samples))
		t_io = time.perf_counter() - t_io_start

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

		t_aug_start = time.perf_counter()
		augmented_search_image, augmented_bbox = _augment_satellite_image_with_bbox(
			image=search_image,
			bbox=base_bbox,
			mode=self.split,
			target_size=self.sat_target_size,
		)
		t_aug = time.perf_counter() - t_aug_start

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

		input_ids = self._token_cache[sample["text"]]
		t_qproc_start = time.perf_counter()
		query_inputs = self.processor(images=query_image, return_tensors="pt")
		t_qproc = time.perf_counter() - t_qproc_start
		t_sproc_start = time.perf_counter()
		search_inputs = self.processor_sat(
			images=augmented_search_image,
			return_tensors="pt",
		)
		t_sproc = time.perf_counter() - t_sproc_start

		t_total = time.perf_counter() - t_start
		if self.enable_timing_log:
			self._timing_acc["io"] += t_io
			self._timing_acc["augment"] += t_aug
			self._timing_acc["query_proc"] += t_qproc
			self._timing_acc["search_proc"] += t_sproc
			self._timing_acc["total"] += t_total
			self._timing_count += 1

			if self._timing_count % self.timing_log_interval == 0:
				den = float(self._timing_count)
				worker_info = torch.utils.data.get_worker_info()
				worker_id = worker_info.id if worker_info is not None else 0
				print(
					"[DatasetTiming] "
					f"split={self.split} worker={worker_id} samples={self._timing_count} "
					f"io={self._timing_acc['io'] / den * 1000.0:.2f}ms "
					f"aug={self._timing_acc['augment'] / den * 1000.0:.2f}ms "
					f"query_proc={self._timing_acc['query_proc'] / den * 1000.0:.2f}ms "
					f"search_proc={self._timing_acc['search_proc'] / den * 1000.0:.2f}ms "
					f"total={self._timing_acc['total'] / den * 1000.0:.2f}ms"
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
	val_split_count: int = VAL_SPLIT_COUNT,
	max_text_length: int = MAX_TEXT_LENGTH,
	sat_target_size: Tuple[int, int] = DEFAULT_SAT_TARGET_SIZE,
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
		val_split_count=val_split_count,
		max_text_length=max_text_length,
		sat_target_size=sat_target_size,
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

		augmented_image, augmented_bbox = _augment_satellite_image_with_bbox(
			image=satellite_image,
			bbox=base_bbox,
			mode=dataset.split,
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