import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor


# --- Configuration ---
MODEL_NAME = "Qwen/Qwen3-VL-4B-Instruct"
IMAGE_SIZE = (256, 256)
NUM_PATCHES = 32
ANGLE_ORDER = (0, 45, 90, 135, 180, 225, 270, 315)
DEFAULT_IMAGE_ROOT = "/data/feihong/drone_img"
CACHE_DIR = "/data/feihong/hf_cache"
DEFAULT_GPU_ID = 1


def _patch_sort_key(path_str: str) -> Tuple[int, int, str]:
	"""Sort patches as radius-angle (e.g., 150_0, 150_45, ..., 300_315)."""
	stem = Path(path_str).stem
	parts = stem.split("_")
	if len(parts) != 2:
		return (10**9, 10**9, path_str)

	try:
		radius = int(parts[0])
		angle = int(parts[1])
	except ValueError:
		return (10**9, 10**9, path_str)

	try:
		angle_idx = ANGLE_ORDER.index(angle)
	except ValueError:
		angle_idx = 10**9

	return (radius, angle_idx, path_str)


def load_qwen_vl(
	model_name: str = MODEL_NAME,
	use_flash_attention: bool = False,
	cache_dir: str = CACHE_DIR,
	gpu_id: Optional[int] = DEFAULT_GPU_ID,
	torch_dtype: torch.dtype = torch.bfloat16,
) -> Tuple[Any, AutoProcessor]:
	"""Load an open-source Qwen-VL model from Hugging Face."""
	os.makedirs(cache_dir, exist_ok=True)
	os.environ["HF_HOME"] = cache_dir
	os.environ["HUGGINGFACE_HUB_CACHE"] = cache_dir
	os.environ["TRANSFORMERS_CACHE"] = cache_dir

	if gpu_id is None:
		device_map: str | Dict[str, str] = "auto"
	else:
		if not torch.cuda.is_available():
			raise RuntimeError("CUDA is not available, but --gpu_id was provided.")
		num_cuda = torch.cuda.device_count()
		if gpu_id < 0 or gpu_id >= num_cuda:
			raise ValueError(
				f"Invalid gpu_id={gpu_id}. Available CUDA devices: 0 to {num_cuda - 1}."
			)
		device_map = {"": f"cuda:{gpu_id}"}

	model_kwargs = {
		"device_map": device_map,
		"torch_dtype": torch_dtype,
		"cache_dir": cache_dir,
	}
	if use_flash_attention:
		model_kwargs["attn_implementation"] = "flash_attention_2"

	model = AutoModelForImageTextToText.from_pretrained(
		model_name,
		**model_kwargs,
	)
	processor = AutoProcessor.from_pretrained(model_name, cache_dir=cache_dir)
	model.eval()
	return model, processor


def _get_model_device(model: Any) -> torch.device:
	return next(model.parameters()).device


def _validate_images(image_paths: Sequence[str]) -> None:
	if len(image_paths) != NUM_PATCHES:
		raise ValueError(
			f"Expected exactly {NUM_PATCHES} images, but got {len(image_paths)}."
		)

	for path in image_paths:
		if not os.path.exists(path):
			raise FileNotFoundError(f"Image path not found: {path}")
		with Image.open(path) as img:
			if img.size != IMAGE_SIZE:
				raise ValueError(
					f"Image {path} has size {img.size}, expected {IMAGE_SIZE}."
				)


def _build_message(prompt: str, image_paths: Sequence[str]) -> Dict:
	content = [{"type": "text", "text": prompt}]
	for path in image_paths:
		content.append({"type": "image", "image": path})
	return {"role": "user", "content": content}


@torch.inference_mode()
def _generate(
	model: Any,
	processor: AutoProcessor,
	messages: List[Dict],
	max_new_tokens: int,
	temperature: float,
) -> str:
	inputs = processor.apply_chat_template(
		messages,
		tokenize=True,
		add_generation_prompt=True,
		return_dict=True,
		return_tensors="pt",
	)
	inputs = inputs.to(_get_model_device(model))

	do_sample = temperature > 0.0
	generated_ids = model.generate(
		**inputs,
		max_new_tokens=max_new_tokens,
		do_sample=do_sample,
		temperature=temperature if do_sample else None,
	)
	generated_ids_trimmed = [
		out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
	]
	output_text = processor.batch_decode(
		generated_ids_trimmed,
		skip_special_tokens=True,
		clean_up_tokenization_spaces=False,
	)
	return output_text[0].strip()


def generate_unified_description_from_32_drone_images(
	image_paths: Sequence[str],
	model: Any,
	processor: AutoProcessor,
	chunk_size: int = 8,
	max_new_tokens_stage1: int = 128,
	max_new_tokens_stage2: int = 96,
) -> Dict[str, List[str] | str]:
	"""
	Mimic a Fig.4-like two-stage text generation pipeline.

	Stage 1: generate chunk-level descriptions from 32 drone patches.
	Stage 2: fuse chunk-level text into one unified description.
	"""
	if chunk_size <= 0:
		raise ValueError("chunk_size must be positive.")

	image_paths = list(image_paths)
	_validate_images(image_paths)

	stage1_prompt = (
		"All images are 256x256 drone patches of the same region. "
		"Describe stable, distinctive details of the target area, emphasizing roof "
		"patterns, geometry, materials, and nearby context. Avoid uncertain claims. "
		"Return 4-6 concise bullet points."
	)

	chunk_descriptions: List[str] = []
	for start in range(0, len(image_paths), chunk_size):
		chunk = image_paths[start : start + chunk_size]
		message = _build_message(stage1_prompt, chunk)
		chunk_text = _generate(
			model=model,
			processor=processor,
			messages=[message],
			max_new_tokens=max_new_tokens_stage1,
			temperature=0.0,
		)
		chunk_descriptions.append(chunk_text)

	fused_notes = "\n".join(
		[f"Chunk {idx + 1}: {desc}" for idx, desc in enumerate(chunk_descriptions)]
	)
	stage2_prompt = (
		"Fuse the chunk descriptions into one unified geo-localization description. "
		"Output two short parts in plain text: "
		"(1) building profile in <=32 tokens as noun phrases; "
		"(2) surrounding environment in <=32 tokens. "
		"Do not include headings, markdown, or explanations.\n\n"
		f"Chunk descriptions:\n{fused_notes}"
	)

	unified_description = _generate(
		model=model,
		processor=processor,
		messages=[{"role": "user", "content": [{"type": "text", "text": stage2_prompt}]}],
		max_new_tokens=max_new_tokens_stage2,
		temperature=0.0,
	)

	return {
		"chunk_descriptions": chunk_descriptions,
		"unified_description": unified_description,
	}


def collect_patch_paths_from_dir(image_dir: str) -> List[str]:
	exts = ("*.jpg", "*.jpeg", "*.png", "*.webp")
	paths: List[str] = []
	for ext in exts:
		paths.extend(str(p) for p in Path(image_dir).glob(ext))

	paths = sorted(paths, key=_patch_sort_key)

	if len(paths) != NUM_PATCHES:
		raise ValueError(
			f"Expected {NUM_PATCHES} patches in {image_dir}, but found {len(paths)}."
		)
	return paths


def resolve_image_dir(
	image_dir: Optional[str],
	image_root: Optional[str],
	sample_id: Optional[str],
	one_image: Optional[str],
) -> str:
	"""Resolve a sample directory from explicit dir, root+id, or one image path."""
	if one_image:
		if not os.path.exists(one_image):
			raise FileNotFoundError(f"Image path not found: {one_image}")
		resolved = str(Path(one_image).resolve().parent)
	elif image_dir:
		resolved = str(Path(image_dir).resolve())
	elif image_root and sample_id:
		resolved = str((Path(image_root) / sample_id).resolve())
	else:
		raise ValueError(
			"Provide one of: --image_dir, --one_image, or both --image_root and --sample_id."
		)

	if not os.path.isdir(resolved):
		raise NotADirectoryError(f"Resolved image directory does not exist: {resolved}")

	return resolved


def list_sample_dirs(image_root: str) -> List[str]:
	"""List sample directories under root (e.g., /data/feihong/drone_img/0000)."""
	root = Path(image_root).resolve()
	if not root.is_dir():
		raise NotADirectoryError(f"Image root does not exist: {root}")

	sample_dirs = [str(p) for p in sorted(root.iterdir()) if p.is_dir()]
	if not sample_dirs:
		raise ValueError(f"No sample directories found under: {root}")

	return sample_dirs


def generate_and_save_for_one_dir(
	resolved_image_dir: str,
	output_name: str,
	chunk_size: int,
	model: Any,
	processor: AutoProcessor,
) -> Dict[str, Any]:
	"""Generate and save unified description for one sample directory."""
	output_path = Path(output_name)
	if not output_path.is_absolute():
		output_path = Path(resolved_image_dir) / output_path

	if output_path.exists():
		try:
			with open(output_path, "r", encoding="utf-8") as f:
				existing_result = json.load(f)
			if isinstance(existing_result, dict):
				existing_result.setdefault("image_dir", resolved_image_dir)
				existing_result.setdefault("num_patches", NUM_PATCHES)
				existing_result["skipped"] = True
				return existing_result
		except json.JSONDecodeError:
			# If the existing file is invalid JSON, regenerate it below.
			pass

	image_paths = collect_patch_paths_from_dir(resolved_image_dir)
	result = generate_unified_description_from_32_drone_images(
		image_paths=image_paths,
		model=model,
		processor=processor,
		chunk_size=chunk_size,
	)
	result["image_dir"] = resolved_image_dir
	result["num_patches"] = len(image_paths)
	result["skipped"] = False

	with open(output_path, "w", encoding="utf-8") as f:
		json.dump(result, f, indent=2, ensure_ascii=False)

	return result


def main() -> None:
	parser = argparse.ArgumentParser(
		description=(
			"Generate unified description from 32 drone patches using Qwen-VL. "
			"If no image arguments are provided, process all folders under "
			f"{DEFAULT_IMAGE_ROOT}."
		)
	)
	parser.add_argument("--image_dir", type=str, default=None)
	parser.add_argument("--image_root", type=str, default=None)
	parser.add_argument("--sample_id", type=str, default=None)
	parser.add_argument("--one_image", type=str, default=None)
	parser.add_argument("--default_image_root", type=str, default=DEFAULT_IMAGE_ROOT)
	parser.add_argument("--output", type=str, default="udes_version1.json")
	parser.add_argument("--model_name", type=str, default=MODEL_NAME)
	parser.add_argument("--cache_dir", type=str, default=CACHE_DIR)
	parser.add_argument(
		"--gpu_id",
		type=int,
		default=DEFAULT_GPU_ID,
		help="CUDA GPU index. Use 1 for the second GPU; use -1 to enable auto device_map.",
	)
	parser.add_argument("--chunk_size", type=int, default=32)
	parser.add_argument("--flash_attn", action="store_true")
	args = parser.parse_args()

	selected_gpu_id: Optional[int] = None if args.gpu_id == -1 else args.gpu_id

	has_single_input = any(
		[arg is not None for arg in [args.image_dir, args.one_image]]
	) or (args.image_root is not None and args.sample_id is not None)

	model, processor = load_qwen_vl(
		model_name=args.model_name,
		use_flash_attention=args.flash_attn,
		cache_dir=args.cache_dir,
		gpu_id=selected_gpu_id,
	)

	if has_single_input:
		resolved_image_dir = resolve_image_dir(
			image_dir=args.image_dir,
			image_root=args.image_root,
			sample_id=args.sample_id,
			one_image=args.one_image,
		)
		result = generate_and_save_for_one_dir(
			resolved_image_dir=resolved_image_dir,
			output_name=args.output,
			chunk_size=args.chunk_size,
			model=model,
			processor=processor,
		)
		print(result["unified_description"])
		return

	sample_dirs = list_sample_dirs(args.default_image_root)[:1652][::-1]
	if Path(args.output).is_absolute():
		output_name = Path(args.output).name
	else:
		output_name = args.output

	num_success = 0
	failed_samples: List[Dict[str, str]] = []

	for idx, sample_dir in enumerate(sample_dirs, start=1):
		print(f"[{idx}/{len(sample_dirs)}] Processing {sample_dir}")
		try:
			result = generate_and_save_for_one_dir(
				resolved_image_dir=sample_dir,
				output_name=output_name,
				chunk_size=args.chunk_size,
				model=model,
				processor=processor,
			)
			num_success += 1
			print(result["unified_description"])
		except Exception as exc:  # noqa: BLE001
			failed_samples.append({"image_dir": sample_dir, "error": str(exc)})
			print(f"Failed: {sample_dir} | {exc}")

	summary = {
		"image_root": str(Path(args.default_image_root).resolve()),
		"num_samples": len(sample_dirs),
		"num_success": num_success,
		"num_failed": len(failed_samples),
		"failed_samples": failed_samples,
	}
	summary_path = Path(args.default_image_root).resolve() / "unified_description_version1.json"
	with open(summary_path, "w", encoding="utf-8") as f:
		json.dump(summary, f, indent=2, ensure_ascii=False)

	print(
		f"Done. success={num_success}/{len(sample_dirs)}, "
		f"failed={len(failed_samples)}. summary={summary_path}"
	)


if __name__ == "__main__":
	main()
