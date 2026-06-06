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
    from transformers import AutoModelForMultimodalLM
except ImportError:  # Older Transformers fallback.
    AutoModelForMultimodalLM = None

try:
    from transformers import AutoModelForImageTextToText
except ImportError:
    AutoModelForImageTextToText = None


# --- Configuration ---
MODEL_NAME = "google/gemma-4-31B-it"
IMAGE_SIZE = (256, 256)
NUM_IMAGES = 4
ANGLE_ORDER = (0, 45, 90, 135, 180, 225, 270, 315)
DEFAULT_IMAGE_ROOT = "/data/feihong/drone_img"
CACHE_DIR = "/data/feihong/hf_cache"
DEFAULT_GPU_ID = -1
DEFAULT_HF_ENDPOINT = os.environ.get("HF_ENDPOINT", "")
RETRIEVAL_PROMPT = (
    "All images show the same place from drone views.\n\n"
    "Write a satellite-retrieval description as 8-12 comma-separated noun phrases. "
    "Use only stable visual cues visible from above: roof geometry, footprint shape, "
    "courtyard/open space, sports field/water/parking, road layout, tree belt, "
    "and relative positions.\n\n"
    "Avoid generic words unless paired with a distinctive shape or location. "
    "Avoid markdown, headings, full sentences, and uncertain claims. "
    "Output only the comma-separated phrases."
)
NAMES = ['250_0.png', '250_90.png', '250_180.png', '250_270.png',]

def _patch_sort_key(path_str: str) -> Tuple[int, int, str]:
    """Sort patches as radius-angle, e.g. 150_0, 150_45, ..., 300_315."""
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


def load_gemma4(
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
    """Load Gemma 4 with the official multimodal Transformers API."""
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

    model_cls = AutoModelForMultimodalLM or AutoModelForImageTextToText
    if model_cls is None:
        raise ImportError(
            "This Transformers install has neither AutoModelForMultimodalLM nor "
            "AutoModelForImageTextToText. Install/upgrade with: pip install -U transformers"
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
        if not os.path.exists(path):
            raise FileNotFoundError(f"Image path not found: {path}")
        with Image.open(path) as img:
            if img.size != expected_size:
                raise ValueError(
                    f"Image {path} has size {img.size}, expected {expected_size}."
                )


def _image_content(path: str, image_field: str) -> Dict[str, str]:
    resolved = Path(path).resolve()
    if image_field == "url":
        return {"type": "image", "url": resolved.as_uri()}
    if image_field == "image":
        return {"type": "image", "image": str(resolved)}
    raise ValueError(f"Unsupported image_field={image_field}")


def _build_message(prompt: str, image_paths: Sequence[str], image_field: str) -> Dict[str, Any]:
    content: List[Dict[str, str]] = [
        _image_content(path, image_field=image_field) for path in image_paths
    ]
    content.append({"type": "text", "text": prompt})
    return {"role": "user", "content": content}


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


def _parse_response(processor: AutoProcessor, response: str) -> str:
    if hasattr(processor, "parse_response"):
        parsed = processor.parse_response(response)
        if isinstance(parsed, str):
            return parsed.strip()
        if isinstance(parsed, dict):
            for key in ("response", "content", "answer", "text"):
                value = parsed.get(key)
                if value:
                    return str(value).strip()
            return json.dumps(parsed, ensure_ascii=False)
        if isinstance(parsed, (list, tuple)) and parsed:
            return str(parsed[-1]).strip()
    return response.strip()


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
    response = processor.decode(
        outputs[0][input_len:],
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    parsed = _parse_response(processor, response)
    return parsed.replace("\n", " ").strip()


def generate_description_from_drone_images(
    image_paths: Sequence[str],
    model: Any,
    processor: AutoProcessor,
    model_name: str,
    prompt: str = RETRIEVAL_PROMPT,
    image_size: Tuple[int, int] = IMAGE_SIZE,
    image_field: str = "url",
    max_new_tokens: int = 128,
    temperature: float = 0.0,
    top_p: float = 0.95,
    top_k: int = 64,
    enable_thinking: bool = False,
) -> Dict[str, Any]:
    image_paths = list(image_paths)
    _validate_images(image_paths, expected_size=image_size)

    message = _build_message(prompt, image_paths, image_field=image_field)
    description = _generate(
        model=model,
        processor=processor,
        messages=[message],
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        enable_thinking=enable_thinking,
    )

    return {
        "prompt": prompt,
        "model_name": model_name,
        "image_paths": [str(Path(path).resolve()) for path in image_paths],
        "unified_description": description,
    }


def collect_image_paths_from_dir(
    image_dir: str,
    num_images: int,
    strict_num_images: bool = False,
) -> List[str]:
    exts = ("*.jpg", "*.jpeg", "*.png", "*.webp")
    paths: List[str] = []
    for ext in exts:
        paths.extend(str(p) for p in Path(image_dir).glob(ext) if str(p) in NAMES)
    paths = sorted(paths, key=_patch_sort_key)

    if strict_num_images and len(paths) != num_images:
        raise ValueError(
            f"Expected exactly {num_images} images in {image_dir}, but found {len(paths)}."
        )
    if len(paths) < num_images:
        raise ValueError(
            f"Expected at least {num_images} images in {image_dir}, but found {len(paths)}."
        )
    return paths[:num_images]


def resolve_image_dir(
    image_dir: Optional[str],
    image_root: Optional[str],
    sample_id: Optional[str],
    one_image: Optional[str],
) -> str:
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
    num_images: int,
    strict_num_images: bool,
    model: Any,
    processor: AutoProcessor,
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
                existing_result.setdefault("num_images", num_images)
                existing_result["skipped"] = True
                return existing_result
        except json.JSONDecodeError:
            pass

    image_paths = collect_image_paths_from_dir(
        resolved_image_dir,
        num_images=num_images,
        strict_num_images=strict_num_images,
    )
    result = generate_description_from_drone_images(
        image_paths=image_paths,
        model=model,
        processor=processor,
        model_name=args.model_name,
        prompt=args.prompt,
        image_size=(args.image_width, args.image_height),
        image_field=args.image_field,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        enable_thinking=args.enable_thinking,
    )
    result["image_dir"] = resolved_image_dir
    result["num_images"] = len(image_paths)
    result["skipped"] = False

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate retrieval-oriented text from local drone images using "
            "google/gemma-4-31B-it. If no single input is provided, process all "
            f"folders under {DEFAULT_IMAGE_ROOT}."
        )
    )
    parser.add_argument("--image_dir", type=str, default=None)
    parser.add_argument("--image_root", type=str, default=None)
    parser.add_argument("--sample_id", type=str, default=None)
    parser.add_argument("--one_image", type=str, default=None)
    parser.add_argument("--default_image_root", type=str, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--output", type=str, default="gemma4_description.json")
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
    parser.add_argument("--num_images", type=int, default=NUM_IMAGES)
    parser.add_argument("--strict_num_images", action="store_true")
    parser.add_argument("--image_width", type=int, default=IMAGE_SIZE[0])
    parser.add_argument("--image_height", type=int, default=IMAGE_SIZE[1])
    parser.add_argument("--image_field", choices=["url", "image"], default="url")
    parser.add_argument("--prompt", type=str, default=RETRIEVAL_PROMPT)
    parser.add_argument("--max_new_tokens", type=int, default=128)
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

    model, processor = load_gemma4(
        model_name=args.model_name,
        cache_dir=args.cache_dir,
        gpu_id=selected_gpu_id,
        dtype=args.dtype,
        use_flash_attention=args.flash_attn,
        hf_token=args.hf_token,
        hf_endpoint=args.hf_endpoint,
        pre_download=not args.skip_pre_download,
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
            num_images=args.num_images,
            strict_num_images=args.strict_num_images,
            model=model,
            processor=processor,
            args=args,
        )
        print(result["unified_description"])
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
                num_images=args.num_images,
                strict_num_images=args.strict_num_images,
                model=model,
                processor=processor,
                args=args,
            )
            num_success += 1
            print(result["unified_description"])
        except Exception as exc:  # noqa: BLE001
            failed_samples.append({"image_dir": sample_dir, "error": str(exc)})
            print(f"Failed: {sample_dir} | {exc}")

    summary = {
        "image_root": str(Path(args.default_image_root).resolve()),
        "model_name": args.model_name,
        "num_images_per_sample": args.num_images,
        "num_samples": len(sample_dirs),
        "num_success": num_success,
        "num_failed": len(failed_samples),
        "failed_samples": failed_samples,
    }
    summary_path = Path(args.default_image_root).resolve() / "gemma4_description_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(
        f"Done. success={num_success}/{len(sample_dirs)}, "
        f"failed={len(failed_samples)}. summary={summary_path}"
    )


if __name__ == "__main__":
    main()
