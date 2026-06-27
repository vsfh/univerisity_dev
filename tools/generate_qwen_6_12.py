import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from PIL import Image
from huggingface_hub import snapshot_download
from transformers import AutoProcessor

try:
    from transformers import AutoModelForImageTextToText
except ImportError:
    AutoModelForImageTextToText = None

try:
    from transformers import Qwen3VLForConditionalGeneration
except ImportError:
    Qwen3VLForConditionalGeneration = None


# --- Configuration ---
MODEL_NAME = "Qwen/Qwen3.6-35B-A3B-FP8"
CACHE_DIR = "/media/4tb/feihong/hf_cache"
DEFAULT_GPU_ID = -1
DEFAULT_GPU_IDS = "1"
DEFAULT_GPU_MEMORY_UTILIZATION = 0.92
DEFAULT_GPU_MEMORY_RESERVE_GIB = 2.0
DEFAULT_FREE_MEMORY_RESERVE_GIB = 3.0
DEFAULT_CUDA_ALLOC_CONF = "expandable_segments:True"
DEFAULT_HF_ENDPOINT = os.environ.get("HF_ENDPOINT", "")
DEFAULT_IMAGE_ROOT = "/media/data1/feihong/remote/data/drone_img"
IMAGE_SIZE = (256, 256)
HEIGHTS = (150, 200, 250, 300)
CARDINAL_ANGLES = (0, 90, 180, 270)
DEFAULT_OUTPUT_NAME = "qwen_6_12_description.json"
DEFAULT_SUMMARY_NAME = "qwen_6_12_description_summary.json"
DEFAULT_GENERATION_BATCH_SIZE = 4
REQUIRED_LOCAL_FILES = (
    "config.json",
    "tokenizer_config.json",
)
HEIGHT_PROMPT = (
    "You are a remote-sensing image description expert. "
    "The four drone images show the same target area at one flight height. "
    "Image order is angle 0 north-facing, angle 90 east-facing, angle 180 "
    "south-facing, and angle 270 west-facing. Use these orientations only as "
    "clues for inferring the unified overhead layout. The top of the satellite "
    "map is north.\n\n"
    "Write for a SigLIP2 text encoder and satellite-map search: output one "
    "English sentence that describes the same target as a single unified place, "
    "not four separate views. The sentence should be about 32 SigLIP2-style text "
    "tokens. Make it a compact retrieval description, not a report, checklist, "
    "caption, or per-image comparison.\n\n"
    "Use only stable, satellite-visible evidence shared or strongly supported "
    "across the four images: building footprint, roof color and shape, facade or "
    "pavement color when visible, road color and geometry, parking lots, "
    "playgrounds or sports fields, water, green space, plazas, intersections, "
    "walls or fences, tree belts, shadows, and adjacent land cover. Include "
    "concrete color and layout words when useful, such as gray, white, red, blue, "
    "green, tan, black, light, dark, brick, asphalt, concrete, grass, north, "
    "south, east, west, center, near, along, adjacent to, between, or surrounded "
    "by.\n\n"
    "Do not mention the source view or camera orientation in the final output. "
    "Forbidden wording includes north-facing image, east-facing image, "
    "south-facing image, west-facing image, this view, that view, angle 0, "
    "angle 90, angle 180, angle 270, photo, and image. Because all inputs are "
    "drone aerial images, also avoid generic capture-mode phrases such as "
    "aerial view of, drone view of, overhead view of, satellite view of, or "
    "top-down view of; these do not help retrieval. Do not write separate "
    "sentences for separate views.\n\n"
    "Do not invent place names, school names, city names, coordinates, functions, "
    "or facts not visible from the images. Do not make temporary details the main "
    "signal: vehicles, pedestrians, construction material, and lighting changes "
    "should be ignored unless they are unusually prominent and consistent across "
    "all four views. If views disagree, silently discard conflicting, weak, or "
    "unstable cues; keep only shared and stable evidence in the retrieval text.\n\n"
    "Output exactly one line: one English sentence, about 32 SigLIP2-style text "
    "tokens, no markdown, no numbering, no comma-separated phrase list."
)


def _dtype_from_name(dtype_name: str) -> Any:
    name = dtype_name.strip().lower()
    if name == "auto":
        return "auto"
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if name in {"fp16", "float16", "half"}:
        return torch.float16
    if name in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype_name}")


def _parse_gpu_ids(gpu_ids: str) -> Optional[List[int]]:
    gpu_ids = gpu_ids.strip()
    if not gpu_ids:
        return None
    parsed: List[int] = []
    for item in gpu_ids.split(","):
        item = item.strip()
        if not item:
            continue
        parsed.append(int(item))
    return parsed or None


def _configure_visible_gpus(gpu_ids: str) -> Optional[List[int]]:
    parsed = _parse_gpu_ids(gpu_ids)
    if parsed is None:
        return None
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(item) for item in parsed)
    print(f"Using CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}")
    return parsed


def _normalize_memory_value(value: str) -> str:
    value = value.strip()
    if not value:
        raise ValueError("Empty GPU memory value.")
    suffixes = ("GiB", "MiB", "GB", "MB")
    if value.endswith(suffixes):
        return value
    return f"{value}GiB"


def _memory_value_to_gib(value: str) -> float:
    normalized = _normalize_memory_value(value)
    for suffix, divisor in (
        ("GiB", 1.0),
        ("GB", 1024 ** 3 / 1000 ** 3),
        ("MiB", 1024.0),
        ("MB", 1024 ** 3 / 1000 ** 2),
    ):
        if normalized.endswith(suffix):
            return float(normalized[: -len(suffix)].strip()) / divisor
    raise ValueError(f"Unsupported memory value: {value}")


def _free_cuda_memory_gib(device_idx: int) -> float:
    free_bytes, _ = torch.cuda.mem_get_info(device_idx)
    return free_bytes / (1024 ** 3)


def _build_max_memory(
    max_memory_per_gpu: Optional[str],
    gpu_memory_utilization: float,
    gpu_memory_reserve_gib: float,
    cpu_max_memory: Optional[str],
    respect_free_memory: bool,
    free_memory_reserve_gib: float,
) -> Optional[Dict[Any, str]]:
    if not torch.cuda.is_available():
        return None

    num_devices = torch.cuda.device_count()
    if num_devices <= 0:
        return None

    max_memory: Dict[Any, str] = {}
    reserve_free_gib = max(float(free_memory_reserve_gib), 0.0)
    if max_memory_per_gpu:
        values = [
            _normalize_memory_value(item)
            for item in max_memory_per_gpu.split(",")
            if item.strip()
        ]
        if len(values) == 1:
            values = values * num_devices
        if len(values) != num_devices:
            raise ValueError(
                f"--max_memory_per_gpu should provide 1 value or {num_devices} values, "
                f"got {len(values)}."
            )
        for idx, value in enumerate(values):
            requested_gib = _memory_value_to_gib(value)
            usable_gib = requested_gib
            if respect_free_memory:
                free_gib = _free_cuda_memory_gib(idx)
                usable_gib = min(requested_gib, max(free_gib - reserve_free_gib, 1.0))
                if int(usable_gib) < int(requested_gib):
                    print(
                        f"Capping visible GPU {idx} max_memory from {value} to "
                        f"{int(usable_gib)}GiB based on current free memory "
                        f"({free_gib:.1f}GiB free)."
                    )
            max_memory[idx] = f"{int(usable_gib)}GiB"
    else:
        utilization = min(max(float(gpu_memory_utilization), 0.1), 1.0)
        reserve_gib = max(float(gpu_memory_reserve_gib), 0.0)
        for idx in range(num_devices):
            props = torch.cuda.get_device_properties(idx)
            total_gib = props.total_memory / (1024 ** 3)
            usable_gib = max(total_gib * utilization - reserve_gib, 1.0)
            if respect_free_memory:
                free_gib = _free_cuda_memory_gib(idx)
                usable_gib = min(usable_gib, max(free_gib - reserve_free_gib, 1.0))
            max_memory[idx] = f"{int(usable_gib)}GiB"

    if cpu_max_memory:
        max_memory["cpu"] = _normalize_memory_value(cpu_max_memory)
    return max_memory


def _snapshot_download_with_retries(
    model_name: str,
    cache_dir: str,
    hf_endpoint: Optional[str],
    hf_token: Optional[str],
    retries: int,
    max_workers: int,
) -> str:
    endpoint = hf_endpoint.strip().rstrip("/") if hf_endpoint else None
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

    last_error: Optional[Exception] = None
    for attempt in range(1, max(int(retries), 1) + 1):
        try:
            return snapshot_download(
                repo_id=model_name,
                cache_dir=cache_dir,
                endpoint=endpoint,
                token=hf_token,
                max_workers=max(int(max_workers), 1),
                etag_timeout=30,
                force_download=False,
            )
        except Exception as exc:
            last_error = exc
            if attempt >= max(int(retries), 1):
                break
            wait_seconds = min(60, 2 ** attempt)
            print(
                f"Download attempt {attempt}/{retries} failed: {exc}. "
                f"Retrying in {wait_seconds}s..."
            )
            time.sleep(wait_seconds)

    raise RuntimeError(f"Failed to download {model_name}: {last_error}") from last_error


def _hf_cache_repo_dir(cache_dir: str, model_name: str) -> Path:
    return Path(cache_dir) / f"models--{model_name.replace('/', '--')}"


def _has_model_weights(snapshot_dir: Path) -> bool:
    return (
        (snapshot_dir / "model.safetensors.index.json").exists()
        or (snapshot_dir / "model.safetensors").exists()
        or any(snapshot_dir.glob("*.safetensors"))
    )


def _has_processor_files(snapshot_dir: Path) -> bool:
    processor_names = (
        "preprocessor_config.json",
        "processor_config.json",
        "chat_template.json",
    )
    return any((snapshot_dir / name).exists() for name in processor_names)


def _is_valid_local_snapshot(snapshot_dir: Path) -> bool:
    return (
        snapshot_dir.is_dir()
        and all((snapshot_dir / name).exists() for name in REQUIRED_LOCAL_FILES)
        and _has_model_weights(snapshot_dir)
        and _has_processor_files(snapshot_dir)
    )


def resolve_local_snapshot(model_name: str, cache_dir: str) -> Optional[Path]:
    model_path = Path(model_name)
    if model_path.is_dir():
        if not _is_valid_local_snapshot(model_path):
            raise FileNotFoundError(f"Local model directory is incomplete: {model_path}")
        return model_path.resolve()

    repo_dir = _hf_cache_repo_dir(cache_dir, model_name)
    snapshots_dir = repo_dir / "snapshots"
    if not snapshots_dir.is_dir():
        return None

    refs_main = repo_dir / "refs" / "main"
    if refs_main.exists():
        commit_hash = refs_main.read_text(encoding="utf-8").strip()
        ref_snapshot = snapshots_dir / commit_hash
        if _is_valid_local_snapshot(ref_snapshot):
            return ref_snapshot.resolve()

    candidates = sorted(
        [path for path in snapshots_dir.iterdir() if path.is_dir()],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for candidate in candidates:
        if _is_valid_local_snapshot(candidate):
            return candidate.resolve()
    return None


def load_qwen36(
    model_name: str = MODEL_NAME,
    cache_dir: str = CACHE_DIR,
    gpu_id: Optional[int] = None,
    dtype: str = "auto",
    use_flash_attention: bool = False,
    hf_token: Optional[str] = None,
    hf_endpoint: Optional[str] = DEFAULT_HF_ENDPOINT,
    pre_download: bool = True,
    download_retries: int = 5,
    download_max_workers: int = 8,
    max_memory: Optional[Dict[Any, str]] = None,
) -> Tuple[Any, AutoProcessor]:
    os.makedirs(cache_dir, exist_ok=True)
    os.environ["HF_HOME"] = cache_dir
    os.environ["HUGGINGFACE_HUB_CACHE"] = cache_dir
    os.environ["TRANSFORMERS_CACHE"] = cache_dir
    if hf_endpoint:
        os.environ["HF_ENDPOINT"] = hf_endpoint.strip().rstrip("/")

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

    model_cls = AutoModelForImageTextToText or Qwen3VLForConditionalGeneration
    if model_cls is None:
        raise ImportError(
            "This Transformers install has neither AutoModelForImageTextToText nor "
            "Qwen3VLForConditionalGeneration. Install/upgrade with: "
            "pip install -U transformers"
        )

    model_kwargs: Dict[str, Any] = {
        "device_map": device_map,
        "cache_dir": cache_dir,
        "dtype": _dtype_from_name(dtype),
    }
    if max_memory is not None and gpu_id is None:
        model_kwargs["max_memory"] = max_memory
        print(f"Using device_map='auto' max_memory={max_memory}")
    if hf_token:
        model_kwargs["token"] = hf_token
    if use_flash_attention:
        model_kwargs["attn_implementation"] = "flash_attention_2"

    load_path = model_name
    if pre_download:
        load_path = _snapshot_download_with_retries(
            model_name=model_name,
            cache_dir=cache_dir,
            hf_endpoint=hf_endpoint,
            hf_token=hf_token,
            retries=download_retries,
            max_workers=download_max_workers,
        )

    try:
        model = model_cls.from_pretrained(load_path, **model_kwargs)
    except TypeError:
        torch_dtype = model_kwargs.pop("dtype")
        model_kwargs["torch_dtype"] = torch_dtype
        model = model_cls.from_pretrained(load_path, **model_kwargs)

    processor_kwargs: Dict[str, Any] = {"cache_dir": cache_dir}
    if hf_token:
        processor_kwargs["token"] = hf_token
    processor = AutoProcessor.from_pretrained(load_path, **processor_kwargs)
    model.eval()
    return model, processor


def _get_input_device(model: Any) -> torch.device:
    hf_device_map = getattr(model, "hf_device_map", None)
    if isinstance(hf_device_map, dict):
        for device in hf_device_map.values():
            if isinstance(device, int):
                return torch.device(f"cuda:{device}")
            if isinstance(device, str) and device.startswith("cuda"):
                return torch.device(device)
    device = getattr(model, "device", None)
    if device is not None:
        return torch.device(device)
    return next(model.parameters()).device


def _validate_images(image_paths: Sequence[str], expected_size: Tuple[int, int]) -> None:
    for path in image_paths:
        if not Path(path).exists():
            raise FileNotFoundError(f"Image path not found: {path}")
        with Image.open(path) as img:
            if img.size != expected_size:
                raise ValueError(
                    f"Image {path} has size {img.size}, expected {expected_size}."
                )


def _image_content(path: str, image_field: str) -> Dict[str, Any]:
    resolved = Path(path).resolve()
    if image_field == "pil":
        image = Image.open(resolved).convert("RGB")
        return {"type": "image", "image": image}
    if image_field == "image":
        return {"type": "image", "image": str(resolved)}
    if image_field == "url":
        return {"type": "image", "url": resolved.as_uri()}
    raise ValueError(f"Unsupported image_field={image_field}")


def _build_message(prompt: str, image_paths: Sequence[str], image_field: str) -> Dict[str, Any]:
    content: List[Dict[str, Any]] = [
        _image_content(path, image_field=image_field) for path in image_paths
    ]
    content.append({"type": "text", "text": prompt})
    return {"role": "user", "content": content}


def _height_image_paths(image_dir: str, height: int) -> List[str]:
    base = Path(image_dir)
    paths = [base / f"{int(height)}_{int(angle)}.png" for angle in CARDINAL_ANGLES]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing cardinal-angle images for height={height} in {image_dir}: {missing}"
        )
    return [str(path) for path in paths]


def collect_height_groups(image_dir: str) -> Dict[str, List[str]]:
    return {str(height): _height_image_paths(image_dir, height) for height in HEIGHTS}


def list_sample_dirs(image_root: str) -> List[str]:
    root = Path(image_root).resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"Image root does not exist: {root}")

    sample_dirs = [str(path) for path in sorted(root.iterdir()) if path.is_dir()]
    if not sample_dirs:
        raise ValueError(f"No sample directories found under: {root}")
    return sample_dirs


def resolve_image_dir(
    image_dir: Optional[str],
    image_root: Optional[str],
    sample_id: Optional[str],
    one_image: Optional[str],
) -> str:
    if one_image:
        if not Path(one_image).exists():
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

    if not Path(resolved).is_dir():
        raise NotADirectoryError(f"Resolved image directory does not exist: {resolved}")
    return resolved


def _sample_id_from_dir(image_dir: str) -> str:
    return Path(image_dir).resolve().name


def _apply_chat_template(
    processor: AutoProcessor,
    messages: Any,
    model: Any,
    enable_thinking: bool,
) -> Any:
    kwargs = {
        "tokenize": True,
        "return_dict": True,
        "return_tensors": "pt",
        "add_generation_prompt": True,
    }
    if not enable_thinking:
        kwargs["enable_thinking"] = False

    try:
        inputs = processor.apply_chat_template(messages, **kwargs)
    except TypeError:
        kwargs.pop("enable_thinking", None)
        inputs = processor.apply_chat_template(messages, **kwargs)

    return inputs.to(_get_input_device(model))


@torch.inference_mode()
def _generate_batch(
    model: Any,
    processor: AutoProcessor,
    conversations: Sequence[List[Dict[str, Any]]],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    enable_thinking: bool,
) -> List[str]:
    inputs = _apply_chat_template(
        processor=processor,
        messages=list(conversations),
        model=model,
        enable_thinking=enable_thinking,
    )
    input_len = inputs["input_ids"].shape[-1]

    do_sample = temperature > 0.0
    generate_kwargs: Dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
    }
    if do_sample:
        generate_kwargs.update({"temperature": temperature, "top_p": top_p, "top_k": top_k})

    outputs = model.generate(**inputs, **generate_kwargs)
    generated_ids_trimmed = outputs[:, input_len:]
    output_texts = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return [text.replace("\n", " ").strip() for text in output_texts]


@torch.inference_mode()
def _generate(
    model: Any,
    processor: AutoProcessor,
    messages: List[Dict[str, Any]],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    enable_thinking: bool,
) -> str:
    return _generate_batch(
        model=model,
        processor=processor,
        conversations=[messages],
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        enable_thinking=enable_thinking,
    )[0]


def generate_height_description(
    image_paths: Sequence[str],
    model: Any,
    processor: Any,
    prompt: str,
    image_size: Tuple[int, int],
    image_field: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    enable_thinking: bool,
) -> str:
    _validate_images(image_paths, expected_size=image_size)
    message = _build_message(prompt, image_paths, image_field=image_field)
    return _generate(
        model=model,
        processor=processor,
        messages=[message],
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        enable_thinking=enable_thinking,
    )


def generate_descriptions_for_one_dir(
    resolved_image_dir: str,
    model: Any,
    processor: Any,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    height_groups = collect_height_groups(resolved_image_dir)
    descriptions: Dict[str, str] = {}
    generation_batch_size = max(int(args.generation_batch_size), 1)
    height_items = [(str(height), height_groups[str(height)]) for height in HEIGHTS]

    for start in range(0, len(height_items), generation_batch_size):
        batch_items = height_items[start : start + generation_batch_size]
        conversations: List[List[Dict[str, Any]]] = []
        for _, image_paths in batch_items:
            _validate_images(
                image_paths,
                expected_size=(args.image_width, args.image_height),
            )
            conversations.append(
                [_build_message(args.prompt, image_paths, image_field=args.image_field)]
            )

        batch_outputs = _generate_batch(
            model=model,
            processor=processor,
            conversations=conversations,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            enable_thinking=args.enable_thinking,
        )
        for (height, _), output_text in zip(batch_items, batch_outputs):
            descriptions[height] = output_text

    return {
        "satellite_id": _sample_id_from_dir(resolved_image_dir),
        "model": args.model_name,
        "prompt": args.prompt,
        "image_dir": str(Path(resolved_image_dir).resolve()),
        "image_paths": {
            height: [str(Path(path).resolve()) for path in paths]
            for height, paths in height_groups.items()
        },
        "descriptions": descriptions,
    }


def generate_and_save_for_one_dir(
    resolved_image_dir: str,
    output_name: str,
    model: Any,
    processor: Any,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    output_path = Path(output_name)
    if not output_path.is_absolute():
        output_path = Path(resolved_image_dir) / output_path

    if output_path.exists() and args.skip_existing:
        try:
            with output_path.open("r", encoding="utf-8") as f:
                existing_result = json.load(f)
            if isinstance(existing_result, dict):
                existing_result.setdefault("image_dir", resolved_image_dir)
                existing_result["skipped"] = True
                return existing_result
        except json.JSONDecodeError:
            pass

    result = generate_descriptions_for_one_dir(
        resolved_image_dir=resolved_image_dir,
        model=model,
        processor=processor,
        args=args,
    )
    result["skipped"] = False

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    return result


def _output_path_for_dir(image_dir: str, output_name: str) -> Path:
    output_path = Path(output_name)
    if output_path.is_absolute():
        return output_path
    return Path(image_dir) / output_path


def _load_completed_result(
    image_dir: str,
    output_name: str,
    skip_existing: bool,
) -> Optional[Dict[str, Any]]:
    if not skip_existing:
        return None
    output_path = _output_path_for_dir(image_dir, output_name)
    if not output_path.exists():
        return None
    try:
        with output_path.open("r", encoding="utf-8") as f:
            result = json.load(f)
    except json.JSONDecodeError:
        return None
    if not isinstance(result, dict):
        return None
    result.setdefault("image_dir", image_dir)
    result["skipped"] = True
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate four height-specific phrase descriptions from cardinal-angle "
            "drone images using Qwen/Qwen3.6-35B-A3B-FP8."
        )
    )
    parser.add_argument("--image_dir", type=str, default=None)
    parser.add_argument("--image_root", type=str, default=None)
    parser.add_argument("--sample_id", type=str, default=None)
    parser.add_argument("--one_image", type=str, default=None)
    parser.add_argument("--default_image_root", type=str, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT_NAME)
    parser.add_argument("--model_name", type=str, default=MODEL_NAME)
    parser.add_argument("--cache_dir", type=str, default=CACHE_DIR)
    parser.add_argument(
        "--gpu_id",
        type=int,
        default=DEFAULT_GPU_ID,
        help=(
            "CUDA GPU index for single-GPU loading. Use -1 for device_map='auto' "
            "across the GPUs exposed by --gpu_ids."
        ),
    )
    parser.add_argument(
        "--gpu_ids",
        type=str,
        default=DEFAULT_GPU_IDS,
        help=(
            "Physical CUDA GPU ids exposed to this process for auto sharding. "
            "Default uses two cards: 0,1. Set to '' to keep the current environment."
        ),
    )
    parser.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=DEFAULT_GPU_MEMORY_UTILIZATION,
        help=(
            "Fraction of each visible GPU's total memory used for max_memory when "
            "--max_memory_per_gpu is not set."
        ),
    )
    parser.add_argument(
        "--gpu_memory_reserve_gib",
        type=float,
        default=DEFAULT_GPU_MEMORY_RESERVE_GIB,
        help="Extra GiB reserved on each visible GPU after applying utilization.",
    )
    parser.add_argument(
        "--max_memory_per_gpu",
        type=str,
        default=None,
        help=(
            "Override max_memory per visible GPU, e.g. '42GiB' for all visible GPUs "
            "or '42GiB,44GiB'. Plain numbers are treated as GiB."
        ),
    )
    parser.add_argument(
        "--cpu_max_memory",
        type=str,
        default=None,
        help="Optional CPU offload memory limit for device_map, e.g. '64GiB'.",
    )
    parser.add_argument(
        "--respect_free_memory",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Cap max_memory by currently free CUDA memory. This avoids overcommitting "
            "a visible GPU that already has another process on it."
        ),
    )
    parser.add_argument(
        "--free_memory_reserve_gib",
        type=float,
        default=DEFAULT_FREE_MEMORY_RESERVE_GIB,
        help="Additional GiB kept free on each visible GPU when capping by free memory.",
    )
    parser.add_argument(
        "--cuda_alloc_conf",
        type=str,
        default=DEFAULT_CUDA_ALLOC_CONF,
        help=(
            "Default PYTORCH_CUDA_ALLOC_CONF set before CUDA initialization. Use '' "
            "to leave the environment unchanged."
        ),
    )
    parser.add_argument("--dtype", type=str, default="auto", choices=["auto", "bf16", "fp16", "fp32"])
    parser.add_argument("--flash_attn", action="store_true")
    parser.add_argument("--hf_token", type=str, default=None)
    parser.add_argument(
        "--hf_endpoint",
        type=str,
        default=DEFAULT_HF_ENDPOINT,
        help="Optional Hugging Face mirror endpoint, e.g. https://hf-mirror.com.",
    )
    parser.add_argument(
        "--skip_pre_download",
        action="store_true",
        help="Skip snapshot_download and let from_pretrained download files directly.",
    )
    parser.add_argument("--download_retries", type=int, default=5)
    parser.add_argument("--download_max_workers", type=int, default=8)
    parser.add_argument(
        "--prefer_local_snapshot",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Prefer an existing complete Hugging Face snapshot under --cache_dir "
            "before trying snapshot_download."
        ),
    )
    parser.add_argument(
        "--local_files_only",
        action="store_true",
        help="Require a complete local snapshot and never try to download.",
    )
    parser.add_argument("--image_width", type=int, default=IMAGE_SIZE[0])
    parser.add_argument("--image_height", type=int, default=IMAGE_SIZE[1])
    parser.add_argument("--image_field", choices=["pil", "url", "image"], default="image")
    parser.add_argument("--prompt", type=str, default=HEIGHT_PROMPT)
    parser.add_argument("--max_new_tokens", type=int, default=96)
    parser.add_argument(
        "--generation_batch_size",
        type=int,
        default=DEFAULT_GENERATION_BATCH_SIZE,
        help=(
            "Number of height prompts generated together per sample. Use 1 for "
            "lowest memory, 2 or 4 for better GPU utilization."
        ),
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=64)
    parser.add_argument("--enable_thinking", action="store_true")
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip a case when its output JSON already exists. Default regenerates and overwrites.",
    )
    parser.add_argument(
        "--num_shards",
        type=int,
        default=1,
        help="Split sample directories across this many independent processes.",
    )
    parser.add_argument(
        "--shard_id",
        type=int,
        default=0,
        help="Shard index for this process, in [0, num_shards).",
    )
    args = parser.parse_args()

    _configure_visible_gpus(args.gpu_ids)
    if args.cuda_alloc_conf:
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", args.cuda_alloc_conf)
        print(f"Using PYTORCH_CUDA_ALLOC_CONF={os.environ['PYTORCH_CUDA_ALLOC_CONF']}")
    selected_gpu_id: Optional[int] = None if args.gpu_id == -1 else args.gpu_id
    max_memory = _build_max_memory(
        max_memory_per_gpu=args.max_memory_per_gpu,
        gpu_memory_utilization=args.gpu_memory_utilization,
        gpu_memory_reserve_gib=args.gpu_memory_reserve_gib,
        cpu_max_memory=args.cpu_max_memory,
        respect_free_memory=args.respect_free_memory,
        free_memory_reserve_gib=args.free_memory_reserve_gib,
    )
    has_single_input = any(
        item is not None for item in [args.image_dir, args.one_image]
    ) or (args.image_root is not None and args.sample_id is not None)

    resolved_image_dir: Optional[str] = None
    all_sample_dirs: List[str] = []
    shard_sample_dirs: List[str] = []
    sample_dirs: List[str] = []
    output_name = Path(args.output).name if Path(args.output).is_absolute() else args.output

    if has_single_input:
        resolved_image_dir = resolve_image_dir(
            image_dir=args.image_dir,
            image_root=args.image_root,
            sample_id=args.sample_id,
            one_image=args.one_image,
        )
        existing_result = _load_completed_result(
            image_dir=resolved_image_dir,
            output_name=args.output,
            skip_existing=args.skip_existing,
        )
        if existing_result is not None:
            print(f"Skipping completed case: {resolved_image_dir}")
            if "descriptions" in existing_result:
                print(json.dumps(existing_result["descriptions"], indent=2, ensure_ascii=False))
            return
    else:
        if args.num_shards < 1:
            raise ValueError("--num_shards must be >= 1.")
        if args.shard_id < 0 or args.shard_id >= args.num_shards:
            raise ValueError("--shard_id must be in [0, num_shards).")

        all_sample_dirs = list_sample_dirs(args.default_image_root)
        shard_sample_dirs = [
            sample_dir
            for idx, sample_dir in enumerate(all_sample_dirs)
            if idx % args.num_shards == args.shard_id
        ]
        sample_dirs = [
            sample_dir
            for sample_dir in shard_sample_dirs
            if _load_completed_result(sample_dir, output_name, args.skip_existing) is None
        ]
        num_skipped_existing = len(shard_sample_dirs) - len(sample_dirs)
        if num_skipped_existing > 0:
            print(
                f"Skipping {num_skipped_existing} completed cases with existing "
                f"{output_name}."
            )
        if not sample_dirs:
            summary_name = DEFAULT_SUMMARY_NAME
            if args.num_shards > 1:
                summary_name = (
                    f"{Path(DEFAULT_SUMMARY_NAME).stem}_shard{args.shard_id}"
                    f"_of_{args.num_shards}{Path(DEFAULT_SUMMARY_NAME).suffix}"
                )
            summary_path = Path(args.default_image_root).resolve() / summary_name
            summary = {
                "image_root": str(Path(args.default_image_root).resolve()),
                "model_name": args.model_name,
                "output_name": output_name,
                "heights": list(HEIGHTS),
                "angles_per_height": list(CARDINAL_ANGLES),
                "num_total_samples": len(all_sample_dirs),
                "num_samples": len(shard_sample_dirs),
                "num_pending": 0,
                "num_skipped_existing": num_skipped_existing,
                "num_shards": args.num_shards,
                "shard_id": args.shard_id,
                "num_success": 0,
                "num_failed": 0,
                "failed_samples": [],
            }
            with summary_path.open("w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            print(f"Done. No pending cases. summary={summary_path}")
            return

    load_model_name = args.model_name
    pre_download = not args.skip_pre_download
    if args.prefer_local_snapshot or args.local_files_only:
        local_snapshot = resolve_local_snapshot(args.model_name, args.cache_dir)
        if local_snapshot is not None:
            load_model_name = str(local_snapshot)
            pre_download = False
            print(f"Using local model snapshot: {load_model_name}")
        elif args.local_files_only:
            raise FileNotFoundError(
                f"No complete local snapshot found for {args.model_name} under {args.cache_dir}."
            )

    model, processor = load_qwen36(
        model_name=load_model_name,
        cache_dir=args.cache_dir,
        gpu_id=selected_gpu_id,
        dtype=args.dtype,
        use_flash_attention=args.flash_attn,
        hf_token=args.hf_token,
        hf_endpoint=args.hf_endpoint,
        pre_download=pre_download,
        download_retries=args.download_retries,
        download_max_workers=args.download_max_workers,
        max_memory=max_memory,
    )

    if has_single_input:
        assert resolved_image_dir is not None
        result = generate_and_save_for_one_dir(
            resolved_image_dir=resolved_image_dir,
            output_name=args.output,
            model=model,
            processor=processor,
            args=args,
        )
        print(json.dumps(result["descriptions"], indent=2, ensure_ascii=False))
        return

    num_success = 0
    failed_samples: List[Dict[str, str]] = []
    for idx, sample_dir in enumerate(sample_dirs, start=1):
        print(f"[{idx}/{len(sample_dirs)}] Processing {sample_dir}")
        try:
            result = generate_and_save_for_one_dir(
                resolved_image_dir=sample_dir,
                output_name=output_name,
                model=model,
                processor=processor,
                args=args,
            )
            num_success += 1
            print(json.dumps(result["descriptions"], indent=2, ensure_ascii=False))
        except Exception as exc:  # noqa: BLE001
            failed_samples.append({"image_dir": sample_dir, "error": str(exc)})
            print(f"Failed: {sample_dir} | {exc}")

    summary = {
        "image_root": str(Path(args.default_image_root).resolve()),
        "model_name": args.model_name,
        "output_name": output_name,
        "heights": list(HEIGHTS),
        "angles_per_height": list(CARDINAL_ANGLES),
        "num_total_samples": len(all_sample_dirs),
        "num_samples": len(shard_sample_dirs),
        "num_pending": len(sample_dirs),
        "num_skipped_existing": len(shard_sample_dirs) - len(sample_dirs),
        "num_shards": args.num_shards,
        "shard_id": args.shard_id,
        "num_success": num_success,
        "num_failed": len(failed_samples),
        "failed_samples": failed_samples,
    }
    summary_name = DEFAULT_SUMMARY_NAME
    if args.num_shards > 1:
        summary_name = (
            f"{Path(DEFAULT_SUMMARY_NAME).stem}_shard{args.shard_id}"
            f"_of_{args.num_shards}{Path(DEFAULT_SUMMARY_NAME).suffix}"
        )
    summary_path = Path(args.default_image_root).resolve() / summary_name
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(
        f"Done. success={num_success}/{len(sample_dirs)}, "
        f"failed={len(failed_samples)}. summary={summary_path}"
    )


if __name__ == "__main__":
    main()
