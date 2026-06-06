import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from PIL import Image
from transformers import AutoConfig, AutoModelForMultimodalLM, AutoProcessor, AutoTokenizer


# --- Configuration ---
GEMMA_MODEL_NAME = "google/gemma-4-31B-it"
SIGLIP2_MODEL_NAME = "google/siglip2-base-patch16-224"
CACHE_DIR = "/data/feihong/hf_cache"
DEFAULT_IMAGE_ROOT = "/data/feihong/drone_img"
DEFAULT_OUTPUT_NAME = "udes_siglip2_gemma4.json"
DEFAULT_SUMMARY_NAME = "gemma4_siglip2_text_summary.json"
IMAGE_SIZE = (256, 256)
NUM_IMAGES_PER_SAMPLE = 4
ANGLE_ORDER = (0, 45, 90, 135, 180, 225, 270, 315)


def _patch_sort_key(path_str: str) -> Tuple[int, int, str]:
    stem = Path(path_str).stem
    parts = stem.split("_")
    if len(parts) != 2:
        return (10**9, 10**9, path_str)

    try:
        height = int(parts[0])
        angle = int(parts[1])
    except ValueError:
        return (10**9, 10**9, path_str)

    try:
        angle_idx = ANGLE_ORDER.index(angle)
    except ValueError:
        angle_idx = 10**9

    return (height, angle_idx, path_str)


def _setup_hf_cache(
    cache_dir: str,
    use_mirror: bool,
    mirror_endpoint: str,
) -> Optional[str]:
    os.makedirs(cache_dir, exist_ok=True)
    os.environ["HF_HOME"] = cache_dir
    os.environ["HUGGINGFACE_HUB_CACHE"] = cache_dir
    os.environ["TRANSFORMERS_CACHE"] = cache_dir

    previous_endpoint = os.environ.get("HF_ENDPOINT")
    if use_mirror:
        os.environ["HF_ENDPOINT"] = mirror_endpoint.rstrip("/")
    return previous_endpoint


def _restore_hf_endpoint(previous_endpoint: Optional[str]) -> None:
    if previous_endpoint is None:
        os.environ.pop("HF_ENDPOINT", None)
    else:
        os.environ["HF_ENDPOINT"] = previous_endpoint


def load_gemma4(
    model_name: str,
    cache_dir: str,
    dtype: str = "auto",
    device_map: str = "auto",
    attn_implementation: Optional[str] = None,
) -> Tuple[Any, AutoProcessor]:
    model_kwargs: Dict[str, Any] = {
        "cache_dir": cache_dir,
        "dtype": _resolve_torch_dtype(dtype),
        "device_map": device_map,
    }
    if attn_implementation:
        model_kwargs["attn_implementation"] = attn_implementation

    processor = AutoProcessor.from_pretrained(model_name, cache_dir=cache_dir)
    model = AutoModelForMultimodalLM.from_pretrained(model_name, **model_kwargs)
    model.eval()
    return model, processor


def _resolve_torch_dtype(dtype: str) -> Any:
    if dtype == "auto":
        return "auto"
    dtype_map = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    if dtype not in dtype_map:
        raise ValueError(f"Unsupported dtype={dtype}. Choose auto, float32, float16, or bfloat16.")
    return dtype_map[dtype]


def load_siglip2_tokenizer(
    model_name: str,
    cache_dir: str,
) -> Tuple[Any, int]:
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    max_length = getattr(tokenizer, "model_max_length", None)
    if max_length is None or int(max_length) > 100000:
        config = AutoConfig.from_pretrained(model_name, cache_dir=cache_dir)
        text_config = getattr(config, "text_config", None)
        max_length = getattr(text_config, "max_position_embeddings", None)
        if max_length is None:
            max_length = getattr(config, "max_position_embeddings", None)
    if max_length is None or int(max_length) <= 0:
        max_length = 64
    return tokenizer, int(max_length)


def _get_model_device(model: Any) -> torch.device:
    return next(model.parameters()).device


def collect_patch_paths_from_dir(
    image_dir: str,
    num_images: int = NUM_IMAGES_PER_SAMPLE,
    check_size: bool = True,
) -> List[str]:
    exts = ("*.png", "*.jpg", "*.jpeg", "*.webp")
    paths: List[str] = []
    for ext in exts:
        paths.extend(str(path) for path in Path(image_dir).glob(ext))

    paths = sorted(paths, key=_patch_sort_key)
    if len(paths) < num_images:
        raise ValueError(f"Expected at least {num_images} images in {image_dir}, got {len(paths)}.")

    selected = paths[:num_images]
    if check_size:
        for path in selected:
            with Image.open(path) as image:
                if image.size != IMAGE_SIZE:
                    raise ValueError(f"Image {path} has size {image.size}, expected {IMAGE_SIZE}.")
    return selected


def list_sample_dirs(image_root: str) -> List[str]:
    root = Path(image_root).resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"Image root does not exist: {root}")

    sample_dirs = [str(path) for path in sorted(root.iterdir()) if path.is_dir()]
    if not sample_dirs:
        raise ValueError(f"No sample directories found under: {root}")
    return sample_dirs


def _build_grounding_prompt(siglip2_max_tokens: int, image_paths: Sequence[str]) -> str:
    image_names = ", ".join(Path(path).name for path in image_paths)
    target_min_tokens = max(1, siglip2_max_tokens - 8)
    return (
        "Describe the satellite-map region that would match these drone-view patches.\n"
        f"Images: {image_names}.\n"
        f"Output one single English line with no markdown, no numbering, no labels, and no newline.\n"
        f"Use close to the full SigLIP2 text budget: aim for {target_min_tokens}-{siglip2_max_tokens} "
        f"SigLIP2 tokens, never exceed {siglip2_max_tokens} tokens.\n"
        "Use compact noun phrases separated by commas.\n"
        "Focus only on stable overhead-visible cues useful for retrieval and grounding: "
        "road layout, building footprints, roof color and shape, vegetation, open ground, "
        "water, parking lots, intersections, field boundaries, and relative spatial arrangement.\n"
        "Avoid camera-view words, drone-view words, altitude, angle, weather, lighting, people, "
        "vehicles unless clearly structural, uncertain object names, and narrative sentences."
    )


def _build_message(prompt: str, image_paths: Sequence[str]) -> Dict[str, Any]:
    content: List[Dict[str, str]] = []
    for image_path in image_paths:
        content.append({"type": "image", "url": str(Path(image_path).resolve())})
    content.append({"type": "text", "text": prompt})
    return {"role": "user", "content": content}


def _build_text_message(prompt: str) -> Dict[str, Any]:
    return {"role": "user", "content": [{"type": "text", "text": prompt}]}


def _apply_chat_template(
    processor: AutoProcessor,
    messages: Sequence[Dict[str, Any]],
    model: Any,
) -> Any:
    kwargs = {
        "tokenize": True,
        "add_generation_prompt": True,
        "return_dict": True,
        "return_tensors": "pt",
    }
    try:
        inputs = processor.apply_chat_template(
            messages,
            enable_thinking=False,
            **kwargs,
        )
    except TypeError:
        inputs = processor.apply_chat_template(messages, **kwargs)
    return inputs.to(_get_model_device(model))


@torch.inference_mode()
def _generate(
    model: Any,
    processor: AutoProcessor,
    messages: Sequence[Dict[str, Any]],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
) -> str:
    inputs = _apply_chat_template(processor, messages, model)
    input_len = inputs["input_ids"].shape[-1]
    do_sample = temperature > 0.0
    generation_kwargs: Dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
    }
    if do_sample:
        generation_kwargs.update(
            {
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
            }
        )

    outputs = model.generate(**inputs, **generation_kwargs)
    response = processor.decode(
        outputs[0][input_len:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return _normalize_single_line(response)


def _normalize_single_line(text: str) -> str:
    text = re.sub(r"<\|channel\>thought.*?<channel\|>", " ", text, flags=re.DOTALL)
    text = re.sub(r"^[\s\-*\d.)]+", "", text.strip())
    text = re.sub(r"\s+", " ", text)
    lines = [line.strip(" -\t") for line in text.splitlines() if line.strip()]
    if len(lines) > 1:
        text = ", ".join(lines)
    else:
        text = text.strip()
    text = text.replace("**", "").replace("`", "")
    return text.strip(" \n\t\"'")


def siglip2_token_count(tokenizer: Any, text: str) -> int:
    encoded = tokenizer(text, add_special_tokens=True, truncation=False)
    input_ids = encoded["input_ids"]
    if input_ids and isinstance(input_ids[0], list):
        input_ids = input_ids[0]
    return int(len(input_ids))


def truncate_to_siglip2_max_length(tokenizer: Any, text: str, max_length: int) -> str:
    encoded = tokenizer(
        text,
        add_special_tokens=True,
        truncation=True,
        max_length=max_length,
    )
    input_ids = encoded["input_ids"]
    if input_ids and isinstance(input_ids[0], list):
        input_ids = input_ids[0]
    return tokenizer.decode(input_ids, skip_special_tokens=True).strip()


def fit_text_to_siglip2_length(
    text: str,
    tokenizer: Any,
    siglip2_max_length: int,
    model: Any,
    processor: AutoProcessor,
    max_refine_rounds: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
) -> Tuple[str, int, bool]:
    target_min_tokens = max(1, siglip2_max_length - 8)
    text = _normalize_single_line(text)
    token_count = siglip2_token_count(tokenizer, text)
    if target_min_tokens <= token_count <= siglip2_max_length:
        return text, token_count, False

    for _ in range(max_refine_rounds):
        if token_count > siglip2_max_length:
            rewrite_prompt = (
                f"Compress this geo-localization description into one English line with "
                f"{target_min_tokens}-{siglip2_max_length} SigLIP2 text tokens. Keep only overhead-visible "
                "retrieval and grounding cues. No markdown, no labels, no newline.\n\n"
                f"Description: {text}"
            )
        else:
            rewrite_prompt = (
                f"Expand this geo-localization description into one English line with "
                f"{target_min_tokens}-{siglip2_max_length} SigLIP2 text tokens. Add only concrete, "
                "overhead-visible retrieval and grounding cues such as road geometry, roof shape, "
                "vegetation, open ground, parking, intersections, and relative layout. "
                "No markdown, no labels, no newline.\n\n"
                f"Description: {text}"
            )
        text = _generate(
            model=model,
            processor=processor,
            messages=[_build_text_message(rewrite_prompt)],
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )
        token_count = siglip2_token_count(tokenizer, text)
        if target_min_tokens <= token_count <= siglip2_max_length:
            return text, token_count, False

    if token_count <= siglip2_max_length:
        return text, token_count, False

    text = truncate_to_siglip2_max_length(tokenizer, text, siglip2_max_length)
    token_count = siglip2_token_count(tokenizer, text)
    return text, token_count, True


def generate_description_from_4_drone_images(
    image_paths: Sequence[str],
    model: Any,
    processor: AutoProcessor,
    siglip2_tokenizer: Any,
    siglip2_max_length: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    max_refine_rounds: int,
) -> Dict[str, Any]:
    prompt = _build_grounding_prompt(siglip2_max_length, image_paths)
    raw_description = _generate(
        model=model,
        processor=processor,
        messages=[_build_message(prompt, image_paths)],
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
    )
    final_description, token_count, truncated = fit_text_to_siglip2_length(
        text=raw_description,
        tokenizer=siglip2_tokenizer,
        siglip2_max_length=siglip2_max_length,
        model=model,
        processor=processor,
        max_refine_rounds=max_refine_rounds,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
    )
    return {
        "unified_description": final_description,
        "raw_description": raw_description,
        "siglip2_token_count": token_count,
        "siglip2_max_token_length": siglip2_max_length,
        "siglip2_target_min_token_count": max(1, siglip2_max_length - 8),
        "truncated_to_siglip2_max_length": truncated,
    }


def resolve_image_dir(
    image_dir: Optional[str],
    image_root: Optional[str],
    sample_id: Optional[str],
    one_image: Optional[str],
) -> str:
    if one_image:
        image_path = Path(one_image).resolve()
        if not image_path.exists():
            raise FileNotFoundError(f"Image path not found: {image_path}")
        resolved = image_path.parent
    elif image_dir:
        resolved = Path(image_dir).resolve()
    elif image_root and sample_id:
        resolved = (Path(image_root) / sample_id).resolve()
    else:
        raise ValueError("Provide one of: --image_dir, --one_image, or both --image_root and --sample_id.")

    if not resolved.is_dir():
        raise NotADirectoryError(f"Resolved image directory does not exist: {resolved}")
    return str(resolved)


def generate_and_save_for_one_dir(
    resolved_image_dir: str,
    output_name: str,
    gemma_model_name: str,
    siglip2_model_name: str,
    model: Any,
    processor: AutoProcessor,
    siglip2_tokenizer: Any,
    siglip2_max_length: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    max_refine_rounds: int,
    check_size: bool,
    overwrite: bool,
) -> Dict[str, Any]:
    output_path = Path(output_name)
    if not output_path.is_absolute():
        output_path = Path(resolved_image_dir) / output_path

    if output_path.exists() and not overwrite:
        with open(output_path, "r", encoding="utf-8") as f:
            existing_result = json.load(f)
        if not isinstance(existing_result, dict):
            raise ValueError(f"Existing output is not a JSON object: {output_path}")
        existing_result.setdefault("image_dir", resolved_image_dir)
        existing_result["skipped"] = True
        return existing_result

    image_paths = collect_patch_paths_from_dir(
        resolved_image_dir,
        num_images=NUM_IMAGES_PER_SAMPLE,
        check_size=check_size,
    )
    result = generate_description_from_4_drone_images(
        image_paths=image_paths,
        model=model,
        processor=processor,
        siglip2_tokenizer=siglip2_tokenizer,
        siglip2_max_length=siglip2_max_length,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_refine_rounds=max_refine_rounds,
    )
    result.update(
        {
            "image_dir": resolved_image_dir,
            "selected_images": image_paths,
            "num_images": len(image_paths),
            "gemma_model_name": gemma_model_name,
            "siglip2_model_name": siglip2_model_name,
            "skipped": False,
        }
    )

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    return result


def _select_sample_dirs(args: argparse.Namespace) -> List[str]:
    has_single_input = any(
        arg is not None for arg in [args.image_dir, args.one_image]
    ) or (args.image_root is not None and args.sample_id is not None)
    if has_single_input:
        return [
            resolve_image_dir(
                image_dir=args.image_dir,
                image_root=args.image_root,
                sample_id=args.sample_id,
                one_image=args.one_image,
            )
        ]

    sample_dirs = list_sample_dirs(args.default_image_root)
    if args.reverse:
        sample_dirs = sample_dirs[::-1]
    if args.start_index > 0:
        sample_dirs = sample_dirs[args.start_index :]
    if args.limit is not None:
        sample_dirs = sample_dirs[: args.limit]
    return sample_dirs


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate SigLIP2-length grounding/retrieval text from four drone images "
            "using google/gemma-4-31B-it."
        )
    )
    parser.add_argument("--image_dir", type=str, default=None)
    parser.add_argument("--image_root", type=str, default=None)
    parser.add_argument("--sample_id", type=str, default=None)
    parser.add_argument("--one_image", type=str, default=None)
    parser.add_argument("--default_image_root", type=str, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT_NAME)
    parser.add_argument("--summary", type=str, default=DEFAULT_SUMMARY_NAME)
    parser.add_argument("--model_name", type=str, default=GEMMA_MODEL_NAME)
    parser.add_argument("--siglip2_model_name", type=str, default=SIGLIP2_MODEL_NAME)
    parser.add_argument("--cache_dir", type=str, default=CACHE_DIR)
    parser.add_argument("--device_map", type=str, default="auto")
    parser.add_argument("--dtype", type=str, default="auto")
    parser.add_argument("--attn_implementation", type=str, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=96)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=64)
    parser.add_argument("--max_refine_rounds", type=int, default=2)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--reverse", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip_size_check", action="store_true")
    parser.add_argument("--use_mirror", action="store_true")
    parser.add_argument("--mirror_endpoint", type=str, default="https://hf-mirror.com")
    args = parser.parse_args()

    previous_endpoint = _setup_hf_cache(
        cache_dir=args.cache_dir,
        use_mirror=args.use_mirror,
        mirror_endpoint=args.mirror_endpoint,
    )
    try:
        sample_dirs = _select_sample_dirs(args)
        siglip2_tokenizer, siglip2_max_length = load_siglip2_tokenizer(
            model_name=args.siglip2_model_name,
            cache_dir=args.cache_dir,
        )
        model, processor = load_gemma4(
            model_name=args.model_name,
            cache_dir=args.cache_dir,
            dtype=args.dtype,
            device_map=args.device_map,
            attn_implementation=args.attn_implementation,
        )

        output_name = Path(args.output).name if len(sample_dirs) > 1 else args.output
        failed_samples: List[Dict[str, str]] = []
        num_success = 0

        for idx, sample_dir in enumerate(sample_dirs, start=1):
            print(f"[{idx}/{len(sample_dirs)}] Processing {sample_dir}")
            try:
                result = generate_and_save_for_one_dir(
                    resolved_image_dir=sample_dir,
                    output_name=output_name,
                    gemma_model_name=args.model_name,
                    siglip2_model_name=args.siglip2_model_name,
                    model=model,
                    processor=processor,
                    siglip2_tokenizer=siglip2_tokenizer,
                    siglip2_max_length=siglip2_max_length,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    top_k=args.top_k,
                    max_refine_rounds=args.max_refine_rounds,
                    check_size=not args.skip_size_check,
                    overwrite=args.overwrite,
                )
                num_success += 1
                print(
                    f"{result.get('siglip2_token_count', 'unknown')}/"
                    f"{result.get('siglip2_max_token_length', siglip2_max_length)} "
                    f"tokens: {result['unified_description']}"
                )
            except Exception as exc:  # noqa: BLE001
                failed_samples.append({"image_dir": sample_dir, "error": str(exc)})
                print(f"Failed: {sample_dir} | {exc}")

        summary = {
            "image_root": str(Path(args.default_image_root).resolve()),
            "output_name": output_name,
            "gemma_model_name": args.model_name,
            "siglip2_model_name": args.siglip2_model_name,
            "siglip2_max_token_length": siglip2_max_length,
            "num_samples": len(sample_dirs),
            "num_success": num_success,
            "num_failed": len(failed_samples),
            "failed_samples": failed_samples,
        }
        summary_path = Path(args.summary)
        if not summary_path.is_absolute():
            summary_path = Path(args.default_image_root).resolve() / summary_path
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        print(
            f"Done. success={num_success}/{len(sample_dirs)}, "
            f"failed={len(failed_samples)}. summary={summary_path}"
        )
    finally:
        if args.use_mirror:
            _restore_hf_endpoint(previous_endpoint)


if __name__ == "__main__":
    main()
