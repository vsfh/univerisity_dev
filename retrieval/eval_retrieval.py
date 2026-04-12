import json
import os
import random
import importlib.util
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoTokenizer, CLIPProcessor
from PIL import Image, ImageDraw
import numpy as np

import open_clip

from train_clip import (
	CACHE_DIR as CLIP_CACHE_DIR,
	MODEL_NAME as CLIP_MODEL_NAME,
	PROJECTION_DIM as CLIP_PROJECTION_DIM,
	Encoder as ClipEncoder,
	_set_processor_size as set_clip_processor_size,
)
from train_evaclip import (
	CACHE_DIR as EVACLIP_CACHE_DIR,
	MODEL_NAME as EVACLIP_MODEL_NAME,
	PROJECTION_DIM as EVACLIP_PROJECTION_DIM,
	Encoder as EvaClipEncoder,
	OpenClipImageProcessorWrapper as EvaImageProcessorWrapper,
	OpenClipTokenizerWrapper as EvaTokenizerWrapper,
)
from train_openclip import (
	CACHE_DIR as OPENCLIP_CACHE_DIR,
	MODEL_NAME as OPENCLIP_MODEL_NAME,
	PRETRAINED as OPENCLIP_PRETRAINED,
	PROJECTION_DIM as OPENCLIP_PROJECTION_DIM,
	Encoder as OpenClipEncoder,
	OpenClipImageProcessorWrapper,
	OpenClipTokenizerWrapper,
)
from train_siglip import (
	CACHE_DIR as SIGLIP_CACHE_DIR,
	DEFAULT_SUBSET_ANGLES,
	DEFAULT_SUBSET_HEIGHTS,
	MODEL_NAME as SIGLIP_MODEL_NAME,
	PROCESSOR_IMAGE_SIZE,
	PROJECTION_DIM as SIGLIP_PROJECTION_DIM,
	Encoder as SiglipEncoder,
	_set_processor_size as set_siglip_processor_size,
)


SEED = 43
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
	sys.path.insert(0, str(REPO_ROOT))

from dataset import ShiftedSatelliteDroneDataset
from unified_siglip_supp import visualize_batch

def _load_encoder_abla_class():
	"""Load root model.py explicitly to avoid import ambiguity."""
	repo_root_str = str(REPO_ROOT)
	if repo_root_str not in sys.path:
		sys.path.insert(0, repo_root_str)
	model_path = REPO_ROOT / "model.py"
	spec = importlib.util.spec_from_file_location("root_model_module", str(model_path))
	if spec is None or spec.loader is None:
		raise ImportError(f"Unable to load model module from: {model_path}")
	module = importlib.util.module_from_spec(spec)
	spec.loader.exec_module(module)
	return module.Encoder_abla


EncoderAbla = _load_encoder_abla_class()


EVAL_CONFIG = {
	"device": None,
	"include_file": "/data/feihong/ckpt/include_train.json",
	"subset_heights": [250],
	"subset_angles": [0],
	"candidate_size": 100,
	"batch_size": 16,
	"num_workers": 8,
	"pca_visualization": {
		"run": False,
		"output_dir": "runs/visualizations/pca_search",
		"max_batches": 1,
		"max_images_per_batch": 8,
	},
	"error_analysis": {
		"run": True,
		"output_dir": "runs/visualizations/retrieval_top1_errors",
	},
	"models": {
		"siglip": {"run": False, "checkpoint": '/data/feihong/ckpt/retrieval_siglip/best.pth'},
		"encoder_abla": {"run": True, "checkpoint": '/data/feihong/ckpt/rope_0.1/last.pth'},
		"clip": {"run": False, "checkpoint": None},
		"openclip": {"run": False, "checkpoint": None},
		"evaclip": {"run": False, "checkpoint": None},
	},
}


def _canonical_model_type(model_type: str) -> str:
	name = model_type.lower()
	if name in {"abla", "encoder_abla"}:
		return "encoder_abla"
	return name


def _is_encoder_abla(model_type: str) -> bool:
	return _canonical_model_type(model_type) == "encoder_abla"


def _normalize_subset(values: Optional[Sequence[int]], defaults: Sequence[int]) -> Set[int]:
	if values is None:
		return {int(v) for v in defaults}
	return {int(v) for v in values}


def _normalize_label(label: str) -> str:
	return str(label).strip().split(".")[0]


def _load_include_map(include_file: Optional[str]) -> Dict[str, Set[str]]:
	include_map: Dict[str, Set[str]] = {}
	if not include_file:
		return include_map
	if not os.path.exists(include_file):
		print(f"Warning: include file not found: {include_file}. Fallback to exact-match GT.")
		return include_map

	with open(include_file, "r", encoding="utf-8") as f:
		include_raw = json.load(f)

	for key, values in include_raw.items():
		norm_key = _normalize_label(key)
		if isinstance(values, list):
			include_map[norm_key] = {_normalize_label(v) for v in values}
		else:
			include_map[norm_key] = {_normalize_label(values)}

	return include_map


def _default_checkpoint_for(model_type: str) -> str:
	model_type = _canonical_model_type(model_type)
	return str(REPO_ROOT / "ckpt" / f"retrieval_{model_type}" / "best.pth")


def _infer_token_grid_hw(n_tokens: int, input_h: int, input_w: int) -> Tuple[int, int]:
	"""Infer token grid (h, w) from token count using input image aspect ratio."""
	if n_tokens <= 0:
		return 1, 1

	target_ratio = float(input_w) / max(float(input_h), 1.0)
	best_h, best_w = 1, int(n_tokens)
	best_score = abs((best_w / max(float(best_h), 1.0)) - target_ratio)

	max_factor = int(n_tokens**0.5)
	for h in range(1, max_factor + 1):
		if n_tokens % h != 0:
			continue
		w = n_tokens // h
		for hh, ww in ((h, w), (w, h)):
			ratio = float(ww) / max(float(hh), 1.0)
			score = abs(ratio - target_ratio)
			if score < best_score:
				best_h, best_w = int(hh), int(ww)
				best_score = score

	return best_h, best_w


def _load_rgb_or_blank(path: Optional[str], fallback_size: Tuple[int, int] = (640, 640)) -> Image.Image:
	if path and os.path.exists(path):
		try:
			return Image.open(path).convert("RGB")
		except Exception:
			pass
	return Image.new("RGB", fallback_size, color=(20, 20, 20))


def _save_correct_pair_image(
	drone_path: Optional[str],
	gt_satellite_path: Optional[str],
	pred_satellite_index: int,
	pred_satellite_path: Optional[str],
	gt_label: str,
	pred_label: str,
	output_path: str,
) -> None:
	"""Save one image containing the correct drone-satellite pair, with wrong top1 index metadata."""
	drone_img = _load_rgb_or_blank(drone_path, fallback_size=(256, 256))
	gt_sat_img = _load_rgb_or_blank(gt_satellite_path, fallback_size=(640, 640))

	target_h = 512
	drone_w = max(1, int(round(drone_img.width * target_h / max(1, drone_img.height))))
	gt_w = max(1, int(round(gt_sat_img.width * target_h / max(1, gt_sat_img.height))))
	drone_img = drone_img.resize((drone_w, target_h), Image.Resampling.BILINEAR)
	gt_sat_img = gt_sat_img.resize((gt_w, target_h), Image.Resampling.BILINEAR)

	pad = 12
	header_h = 70
	canvas_w = pad * 3 + drone_w + gt_w
	canvas_h = header_h + pad * 2 + target_h
	canvas = Image.new("RGB", (canvas_w, canvas_h), color=(15, 15, 15))

	canvas.paste(drone_img, (pad, header_h + pad))
	canvas.paste(gt_sat_img, (pad * 2 + drone_w, header_h + pad))

	draw = ImageDraw.Draw(canvas)
	draw.text(
		(pad, 8),
		f"Top1 Miss | wrong_sat_index={pred_satellite_index} | pred={pred_label} | gt={gt_label}",
		fill=(240, 240, 240),
	)
	draw.text(
		(pad, 30),
		"Left: Drone query (correct) | Right: GT satellite (correct pair)",
		fill=(180, 220, 180),
	)
	if pred_satellite_path:
		draw.text(
			(pad, 50),
			f"Pred satellite path: {pred_satellite_path}",
			fill=(220, 160, 160),
		)

	os.makedirs(os.path.dirname(output_path), exist_ok=True)
	canvas.save(output_path)


def pca_visualization(
	batch,
	search_grid_feats: torch.Tensor,
	output_dir: str = "runs/visualizations/pca_search",
	max_images: int = 8,
	file_prefix: str = "",
	interp_scale: float = 9.0,
) -> List[str]:
	"""Save PCA RGB maps from search feature tokens shaped [B, N, D]."""
	output_root = Path(output_dir)
	output_root.mkdir(parents=True, exist_ok=True)

	if search_grid_feats.ndim != 3:
		raise ValueError(
			f"search_grid_feats must be [B, N, D], got shape={tuple(search_grid_feats.shape)}"
		)

	saved_paths: List[str] = []
	features_cpu = search_grid_feats.detach().float().cpu()
	sat_paths = list(batch.get("satellite_path", []))
	search_pixels = batch.get("search_pixel_values", None)
	batch_count = min(int(max_images), int(features_cpu.shape[0]))

	for i in range(batch_count):
		token_feats = features_cpu[i]  # [N, D]
		n_tokens = int(token_feats.shape[0])

		if isinstance(search_pixels, torch.Tensor) and search_pixels.ndim == 4 and i < search_pixels.shape[0]:
			input_h = int(search_pixels[i].shape[-2])
			input_w = int(search_pixels[i].shape[-1])
		else:
			side = int(n_tokens**0.5)
			if side * side == n_tokens:
				input_h, input_w = side, side
			else:
				input_h, input_w = 1, n_tokens

		h, w = _infer_token_grid_hw(n_tokens=n_tokens, input_h=input_h, input_w=input_w)

		# Interpolate the spatial token grid before PCA so visualization has higher native resolution.
		token_grid = token_feats.reshape(h, w, -1).permute(2, 0, 1).unsqueeze(0)
		up_h = max(1, int(round(float(h) * float(interp_scale))))
		up_w = max(1, int(round(float(w) * float(interp_scale))))
		if up_h != h or up_w != w:
			token_grid = F.interpolate(
				token_grid,
				size=(up_h, up_w),
				mode="bilinear",
				align_corners=False,
			)
		else:
			up_h, up_w = h, w

		x = token_grid.squeeze(0).permute(1, 2, 0).reshape(up_h * up_w, -1)
		x = x - x.mean(dim=0, keepdim=True)
		q = max(1, min(3, int(x.shape[0]), int(x.shape[1])))
		_, _, v = torch.pca_lowrank(x, q=q)
		proj = x @ v[:, :q]
		if q < 3:
			pad = torch.zeros((proj.shape[0], 3 - q), dtype=proj.dtype)
			proj = torch.cat([proj, pad], dim=1)

		proj = proj.reshape(up_h, up_w, 3).numpy()
		for c in range(3):
			channel = proj[:, :, c]
			ch_min = float(channel.min())
			ch_max = float(channel.max())
			if ch_max > ch_min:
				proj[:, :, c] = (channel - ch_min) / (ch_max - ch_min)
			else:
				proj[:, :, c] = 0.0

		rgb_uint8 = (proj * 255.0).clip(0.0, 255.0).astype(np.uint8)
		image = Image.fromarray(rgb_uint8, mode="RGB")
		# image = image.resize((640, 640), Image.Resampling.BILINEAR)

		sat_stem = f"sample_{i}"
		if i < len(sat_paths):
			sat_stem = Path(str(sat_paths[i])).stem

		out_path = output_root / f"{file_prefix}{sat_stem}_pca.png"
		image.save(out_path)
		saved_paths.append(str(out_path))

	return saved_paths


def _build_model_and_io(model_type: str, device: str):
	model_type = _canonical_model_type(model_type)

	if model_type == "siglip":
		processor = AutoImageProcessor.from_pretrained(
			SIGLIP_MODEL_NAME,
			cache_dir=SIGLIP_CACHE_DIR,
		)
		processor_sat = AutoImageProcessor.from_pretrained(
			SIGLIP_MODEL_NAME,
			cache_dir=SIGLIP_CACHE_DIR,
		)
		set_siglip_processor_size(processor_sat, PROCESSOR_IMAGE_SIZE)
		tokenizer = AutoTokenizer.from_pretrained(SIGLIP_MODEL_NAME)
		model = SiglipEncoder(SIGLIP_MODEL_NAME, proj_dim=SIGLIP_PROJECTION_DIM).to(device)
		return model, processor, processor_sat, tokenizer

	if model_type == "encoder_abla":
		processor = AutoImageProcessor.from_pretrained(
			SIGLIP_MODEL_NAME,
			cache_dir=SIGLIP_CACHE_DIR,
		)
		processor_sat = AutoImageProcessor.from_pretrained(
			SIGLIP_MODEL_NAME,
			cache_dir=SIGLIP_CACHE_DIR,
		)
		set_siglip_processor_size(processor_sat, {"height": 640, "width": 640})
		tokenizer = AutoTokenizer.from_pretrained(SIGLIP_MODEL_NAME)
		model = EncoderAbla(
			model_name=SIGLIP_MODEL_NAME,
			proj_dim=SIGLIP_PROJECTION_DIM,
			usesg=True,
			useap=True,
		).to(device)
		return model, processor, processor_sat, tokenizer

	if model_type == "clip":
		processor = CLIPProcessor.from_pretrained(
			CLIP_MODEL_NAME,
			cache_dir=CLIP_CACHE_DIR,
		)
		processor_sat = CLIPProcessor.from_pretrained(
			CLIP_MODEL_NAME,
			cache_dir=CLIP_CACHE_DIR,
		)
		set_clip_processor_size(processor_sat, PROCESSOR_IMAGE_SIZE)
		tokenizer = AutoTokenizer.from_pretrained(CLIP_MODEL_NAME, cache_dir=CLIP_CACHE_DIR)
		model = ClipEncoder(CLIP_MODEL_NAME, proj_dim=CLIP_PROJECTION_DIM).to(device)
		return model, processor, processor_sat, tokenizer

	if model_type == "openclip":
		_, _, preprocess = open_clip.create_model_and_transforms(
			OPENCLIP_MODEL_NAME,
			pretrained=OPENCLIP_PRETRAINED,
			cache_dir=OPENCLIP_CACHE_DIR,
		)
		tokenizer = open_clip.get_tokenizer(OPENCLIP_MODEL_NAME)
		processor_wrapper = OpenClipImageProcessorWrapper(preprocess)
		tokenizer_wrapper = OpenClipTokenizerWrapper(tokenizer)
		model = OpenClipEncoder(
			model_name=OPENCLIP_MODEL_NAME,
			pretrained=OPENCLIP_PRETRAINED,
			proj_dim=OPENCLIP_PROJECTION_DIM,
		).to(device)
		return model, processor_wrapper, processor_wrapper, tokenizer_wrapper

	if model_type == "evaclip":
		_, _, preprocess = open_clip.create_model_and_transforms(
			EVACLIP_MODEL_NAME,
			cache_dir=EVACLIP_CACHE_DIR,
		)
		tokenizer = open_clip.get_tokenizer(EVACLIP_MODEL_NAME)
		processor_wrapper = EvaImageProcessorWrapper(preprocess)
		tokenizer_wrapper = EvaTokenizerWrapper(tokenizer)
		model = EvaClipEncoder(
			model_name=EVACLIP_MODEL_NAME,
			proj_dim=EVACLIP_PROJECTION_DIM,
		).to(device)
		return model, processor_wrapper, processor_wrapper, tokenizer_wrapper

	raise ValueError(f"Unsupported model type: {model_type}")


def _extract_features(
	model_type: str,
	checkpoint_path: Optional[str],
	device: str,
	batch_size: int,
	num_workers: int,
	subset_heights: Sequence[int],
	subset_angles: Sequence[int],
):
	model_type = _canonical_model_type(model_type)
	model, processor, processor_sat, tokenizer = _build_model_and_io(model_type, device)

	resolved_checkpoint = checkpoint_path or _default_checkpoint_for(model_type)
	if not os.path.exists(resolved_checkpoint):
		raise FileNotFoundError(f"Checkpoint not found: {resolved_checkpoint}")

	model.load_state_dict(torch.load(resolved_checkpoint, map_location="cpu"), strict=True)
	model.eval()

	dataset = ShiftedSatelliteDroneDataset(
		processor=processor,
		processor_sat=processor_sat,
		tokenizer=tokenizer,
		split="train",
		subset_heights=list(subset_heights),
		subset_angles=list(subset_angles),
	)
	loader = DataLoader(
		dataset,
		batch_size=batch_size,
		shuffle=False,
		num_workers=num_workers,
		pin_memory=device.startswith("cuda"),
		persistent_workers=num_workers > 0,
		prefetch_factor=4 if num_workers > 0 else None,
	)

	query_feats: List[torch.Tensor] = []
	query_labels: List[str] = []
	query_drone_paths: List[str] = []
	query_satellite_paths: List[str] = []
	gallery_feat_dict: Dict[str, torch.Tensor] = {}
	gallery_path_dict: Dict[str, str] = {}

	with torch.inference_mode():
		for batch in tqdm(loader, desc=f"Extract features [{model_type}]"):
			query_inputs = batch["target_pixel_values"].to(device, non_blocking=True)
			search_inputs = batch["search_pixel_values"].to(device, non_blocking=True)
			input_ids = batch["input_ids"].to(device, non_blocking=True)
			angles = batch["angle"].to(device, non_blocking=True)
			satellite_paths = batch["satellite_path"]

			if _is_encoder_abla(model_type):
				_, _, _, _, grid_feats, fused_feats = model(query_inputs, search_inputs, input_ids, angles)
				if fused_feats is None:
					raise RuntimeError("Encoder_abla did not return fused query features.")
				fused_query_feats = F.normalize(fused_feats, p=2, dim=1)
				gallery_grid_batch = F.normalize(grid_feats, p=2, dim=2)
			else:
				attention_mask = (input_ids != 0).long()
				anchor_feats = model.query_forward(query_inputs)
				# Different model wrappers expose text_forward with or without attention_mask.
				try:
					text_feats = model.text_forward(input_ids, attention_mask)
				except TypeError:
					text_feats = model.text_forward(input_ids)
				fused_query_feats = F.normalize(anchor_feats + text_feats, p=2, dim=1)
				gallery_grid_batch = None
			query_feats.append(fused_query_feats.cpu())

			batch_labels = [
				os.path.splitext(os.path.basename(path))[0] for path in satellite_paths
			]
			query_labels.extend(batch_labels)
			query_drone_paths.extend([str(p) for p in batch["drone_path"]])
			query_satellite_paths.extend([str(p) for p in satellite_paths])

			unseen_indices = []
			unseen_labels = []
			for i, label in enumerate(batch_labels):
				if label not in gallery_feat_dict:
					unseen_indices.append(i)
					unseen_labels.append(label)

			if unseen_indices:
				if _is_encoder_abla(model_type):
					gallery_grid = gallery_grid_batch[unseen_indices]
				else:
					if gallery_grid_batch is not None:
						gallery_grid = gallery_grid_batch[unseen_indices]
					else:
						gallery_batch = search_inputs[unseen_indices]
						with torch.amp.autocast("cuda", enabled=False):
							gallery_grid = model.ref_forward(gallery_batch)
							gallery_grid = F.normalize(gallery_grid, p=2, dim=2)
				gallery_grid = gallery_grid.cpu()
				for i, label in enumerate(unseen_labels):
					gallery_feat_dict[label] = gallery_grid[i]
					gallery_path_dict[label] = str(satellite_paths[unseen_indices[i]])

	if not query_feats or not gallery_feat_dict:
		raise RuntimeError("No valid query/gallery features extracted.")

	query_tensor = torch.cat(query_feats, dim=0)
	gallery_labels = sorted(gallery_feat_dict.keys())
	gallery_tensor = torch.stack([gallery_feat_dict[label] for label in gallery_labels], dim=0)
	gallery_satellite_paths = [gallery_path_dict.get(label, "") for label in gallery_labels]

	return (
		query_tensor,
		query_labels,
		query_drone_paths,
		query_satellite_paths,
		gallery_tensor,
		gallery_labels,
		gallery_satellite_paths,
	)


def _score_recall(
	query_feats: torch.Tensor,
	query_labels: List[str],
	query_drone_paths: List[str],
	query_satellite_paths: List[str],
	gallery_feats: torch.Tensor,
	gallery_labels: List[str],
	gallery_satellite_paths: List[str],
	include_map: Dict[str, Set[str]],
	device: str,
	candidate_size: Optional[int] = None,
	error_analysis: Optional[Dict[str, object]] = None,
) -> Dict[str, float]:
	label_to_gallery_index = {label: idx for idx, label in enumerate(gallery_labels)}
	norm_gallery_labels = [_normalize_label(label) for label in gallery_labels]

	query_feats = query_feats.to(device)
	gallery_feats = gallery_feats.to(device)

	valid_queries = 0
	top1 = 0
	top5 = 0
	top10 = 0

	all_indices = list(range(len(gallery_labels)))
	error_enabled = bool((error_analysis or {}).get("run", False))
	error_output_dir = str((error_analysis or {}).get("output_dir", "runs/visualizations/retrieval_top1_errors"))
	error_saved = 0
	error_records: List[Dict[str, object]] = []
	if candidate_size is not None:
		candidate_size = int(candidate_size)
		if candidate_size <= 0:
			raise ValueError("candidate_size must be a positive integer.")

	for q_idx, gt_label in enumerate(query_labels):
		gt_gallery_index = label_to_gallery_index.get(gt_label)
		if gt_gallery_index is None:
			continue

		gt_norm = _normalize_label(gt_label)
		positive_label_set = set(include_map.get(gt_norm, set()))
		positive_label_set.add(gt_norm)

		if candidate_size is None or candidate_size >= len(all_indices):
			candidate_indices = all_indices
		else:
			negative_pool = [idx for idx in all_indices if idx != gt_gallery_index]
			sampled_negatives = random.sample(negative_pool, candidate_size - 1)
			candidate_indices = sampled_negatives + [gt_gallery_index]
			# random.shuffle(candidate_indices)

		candidate_gallery_feats = gallery_feats[candidate_indices]
		score_vec = torch.einsum("d,knd->kn", query_feats[q_idx], candidate_gallery_feats).max(dim=1)[0]
		k_max = min(10, score_vec.shape[0])
		topk_local_indices = torch.topk(score_vec, k=k_max, dim=0).indices.tolist()
		top1_local_idx = topk_local_indices[0]
		top1_global_idx = candidate_indices[top1_local_idx]
		top1_pred_label = gallery_labels[top1_global_idx]

		positive_local_indices = [
			local_idx
			for local_idx, global_idx in enumerate(candidate_indices)
			if norm_gallery_labels[global_idx] in positive_label_set
		]

		valid_queries += 1
		top1_correct = any(idx in topk_local_indices[:1] for idx in positive_local_indices)
		if top1_correct:
			top1 += 1
		if any(idx in topk_local_indices[: min(5, k_max)] for idx in positive_local_indices):
			top5 += 1
		if any(idx in topk_local_indices[: min(10, k_max)] for idx in positive_local_indices):
			top10 += 1

		if error_enabled and (not top1_correct):
			drone_path = query_drone_paths[q_idx] if q_idx < len(query_drone_paths) else ""
			gt_sat_path = query_satellite_paths[q_idx] if q_idx < len(query_satellite_paths) else ""
			pred_sat_path = (
				gallery_satellite_paths[top1_global_idx]
				if top1_global_idx < len(gallery_satellite_paths)
				else ""
			)

			out_file = os.path.join(
				error_output_dir,
				f"q{q_idx:06d}_gt_{gt_norm}_pred_{_normalize_label(top1_pred_label)}_errIdx_{top1_global_idx}.jpg",
			)
			_save_correct_pair_image(
				drone_path=drone_path,
				gt_satellite_path=gt_sat_path,
				pred_satellite_index=top1_global_idx,
				pred_satellite_path=pred_sat_path,
				gt_label=gt_norm,
				pred_label=_normalize_label(top1_pred_label),
				output_path=out_file,
			)

			error_records.append(
				{
					"query_index": q_idx,
					"gt_label": gt_norm,
					"pred_label": _normalize_label(top1_pred_label),
					"error_satellite_index": int(top1_global_idx),
					"drone_path": drone_path,
					"gt_satellite_path": gt_sat_path,
					"pred_satellite_path": pred_sat_path,
					"saved_image": out_file,
				}
			)
			error_saved += 1

	if valid_queries == 0:
		raise RuntimeError("No valid queries matched gallery labels.")

	if error_enabled and error_records:
		os.makedirs(error_output_dir, exist_ok=True)
		error_jsonl = os.path.join(error_output_dir, "top1_error_cases.jsonl")
		with open(error_jsonl, "w", encoding="utf-8") as f:
			for item in error_records:
				f.write(json.dumps(item, ensure_ascii=False) + "\n")
		print(f"Saved {len(error_records)} top1 error pair images and index log to: {error_output_dir}")

	return {
		"num_queries": valid_queries,
		"recall@1": top1 / valid_queries,
		"recall@5": top5 / valid_queries,
		"recall@10": top10 / valid_queries,
		"top1_hits": top1,
		"top5_hits": top5,
		"top10_hits": top10,
		"top1_errors": valid_queries - top1,
	}


def eval(
	model_type: str,
	checkpoint_path: Optional[str] = None,
	subset_heights: Optional[Sequence[int]] = None,
	subset_angles: Optional[Sequence[int]] = None,
	candidate_size: Optional[int] = None,
	include_file: str = "/data/feihong/ckpt/include.json",
	batch_size: int = 16,
	num_workers: int = 8,
	device: Optional[str] = None,
	error_analysis: Optional[Dict[str, object]] = None,
) -> Dict[str, float]:
	if device is None:
		device = "cuda" if torch.cuda.is_available() else "cpu"

	subset_height_set = _normalize_subset(subset_heights, DEFAULT_SUBSET_HEIGHTS)
	subset_angle_set = _normalize_subset(subset_angles, DEFAULT_SUBSET_ANGLES)
	include_map = _load_include_map(include_file)

	print(
		f"Evaluating {model_type} with heights={sorted(subset_height_set)} "
		f"angles={sorted(subset_angle_set)} candidate_size={candidate_size}"
	)

	(
		query_feats,
		query_labels,
		query_drone_paths,
		query_satellite_paths,
		gallery_feats,
		gallery_labels,
		gallery_satellite_paths,
	) = _extract_features(
		model_type=model_type,
		checkpoint_path=checkpoint_path,
		device=device,
		batch_size=batch_size,
		num_workers=num_workers,
		subset_heights=sorted(subset_height_set),
		subset_angles=sorted(subset_angle_set),
	)

	metrics = _score_recall(
		query_feats=query_feats,
		query_labels=query_labels,
		query_drone_paths=query_drone_paths,
		query_satellite_paths=query_satellite_paths,
		gallery_feats=gallery_feats,
		gallery_labels=gallery_labels,
		gallery_satellite_paths=gallery_satellite_paths,
		include_map=include_map,
		device=device,
		candidate_size=candidate_size,
		error_analysis=error_analysis,
	)

	n = metrics["num_queries"]
	print(
		f"Recall@1: {metrics['top1_hits']} / {n} ({metrics['recall@1'] * 100:.2f}%)\n"
		f"Recall@5: {metrics['top5_hits']} / {n} ({metrics['recall@5'] * 100:.2f}%)\n"
		f"Recall@10: {metrics['top10_hits']} / {n} ({metrics['recall@10'] * 100:.2f}%)"
	)
	return metrics



if __name__ == "__main__":
    model_cfg = EVAL_CONFIG["models"]
    selected_models = [name for name, cfg in model_cfg.items() if cfg.get("run", False)]

    if not selected_models:
        raise ValueError("No model selected. Set EVAL_CONFIG['models'][name]['run'] = True.")


    for model_type in selected_models:
        eval(
            model_type=model_type,
            checkpoint_path=model_cfg[model_type].get("checkpoint"),
            subset_heights=EVAL_CONFIG["subset_heights"],
            subset_angles=EVAL_CONFIG["subset_angles"],
            candidate_size=EVAL_CONFIG["candidate_size"],
            include_file=EVAL_CONFIG["include_file"],
            batch_size=EVAL_CONFIG["batch_size"],
            num_workers=EVAL_CONFIG["num_workers"],
            device=EVAL_CONFIG["device"],
			error_analysis=EVAL_CONFIG.get("error_analysis", None),
        )
