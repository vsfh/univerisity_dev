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
CACHE_DIR = "/media/data1/feihong/hf_cache"
DEFAULT_GPU_ID = -1
DEFAULT_HF_ENDPOINT = os.environ.get("HF_ENDPOINT", "")
DEFAULT_IMAGE_ROOT = "/media/data1/feihong/drone_img"
IMAGE_SIZE = (256, 256)
HEIGHTS = (150, 200, 250, 300)
CARDINAL_ANGLES = (0, 90, 180, 270)
DEFAULT_OUTPUT_NAME = "qwen_6_4_description.json"
DEFAULT_SUMMARY_NAME = "qwen_6_4_description_summary.json"
REQUIRED_LOCAL_FILES = (
    "config.json",
    "tokenizer_config.json",
)
HEIGHT_PROMPT = (
    "The four drone images show the same target area at one flight height from "
    "north/east/south/west views.\n\n"
    "Write exactly one satellite-map retrieval description as exactly 4 "
    "comma-separated English noun phrases.\n\n"
    "Phrase 1 is mandatory: it must be 8-12 words and describe the single most "
    "distinctive building or environmental feature that would distinguish this "
    "target area on a satellite map.\n"
    "Phrases 2-4 should be short supporting cues about roof shape, footprint, "
    "open space, road geometry, water, sports fields, parking, tree belts, or "
    "relative layout.\n\n"
    "Use stable overhead-visible evidence only. Avoid angle-specific language, "
    "uncertain claims, markdown, numbering, headings, and full explanatory "
    "sentences. Output only the 4 comma-separated phrases."
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
    messages: List[Dict[str, Any]],
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
    inputs = _apply_chat_template(
        processor=processor,
        messages=messages,
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
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return output_text[0].replace("\n", " ").strip()


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

    for height in (str(item) for item in HEIGHTS):
        descriptions[height] = generate_height_description(
            image_paths=height_groups[height],
            model=model,
            processor=processor,
            prompt=args.prompt,
            image_size=(args.image_width, args.image_height),
            image_field=args.image_field,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            enable_thinking=args.enable_thinking,
        )

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

    if output_path.exists() and not args.overwrite:
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
        help="CUDA GPU index. Use -1 for device_map='auto' across available GPUs.",
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
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=64)
    parser.add_argument("--enable_thinking", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    selected_gpu_id: Optional[int] = None if args.gpu_id == -1 else args.gpu_id
    has_single_input = any(
        item is not None for item in [args.image_dir, args.one_image]
    ) or (args.image_root is not None and args.sample_id is not None)

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
            model=model,
            processor=processor,
            args=args,
        )
        print(json.dumps(result["descriptions"], indent=2, ensure_ascii=False))
        return

    sample_dirs = list_sample_dirs(args.default_image_root)
    output_name = Path(args.output).name if Path(args.output).is_absolute() else args.output

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
        "num_samples": len(sample_dirs),
        "num_success": num_success,
        "num_failed": len(failed_samples),
        "failed_samples": failed_samples,
    }
    summary_path = Path(args.default_image_root).resolve() / DEFAULT_SUMMARY_NAME
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(
        f"Done. success={num_success}/{len(sample_dirs)}, "
        f"failed={len(failed_samples)}. summary={summary_path}"
    )


if __name__ == "__main__":
    main()
