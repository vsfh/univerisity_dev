import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from PIL import Image

from generate_gamma_4_text import (
    CACHE_DIR,
    DEFAULT_GPU_ID,
    DEFAULT_HF_ENDPOINT,
    DEFAULT_IMAGE_ROOT,
    IMAGE_SIZE,
    MODEL_NAME,
    _generate,
    load_gemma4,
    resolve_image_dir,
)


# --- Configuration ---
HEIGHTS = (150, 200, 250, 300)
CARDINAL_ANGLES = (0, 90, 180, 270)
DEFAULT_OUTPUT_NAME = "gemma_5_28_description.json"
DEFAULT_SUMMARY_NAME = "gemma_5_28_description_summary.json"
REQUIRED_LOCAL_FILES = (
    "config.json",
    "model.safetensors.index.json",
    "model-00001-of-00002.safetensors",
    "model-00002-of-00002.safetensors",
    "processor_config.json",
    "tokenizer.json",
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


def _hf_cache_repo_dir(cache_dir: str, model_name: str) -> Path:
    return Path(cache_dir) / f"models--{model_name.replace('/', '--')}"


def _is_valid_local_snapshot(snapshot_dir: Path) -> bool:
    return snapshot_dir.is_dir() and all((snapshot_dir / name).exists() for name in REQUIRED_LOCAL_FILES)


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


def _sample_id_from_dir(image_dir: str) -> str:
    return Path(image_dir).resolve().name


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
            "drone images using google/gemma-4-31B-it."
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
    parser.add_argument("--image_field", choices=["pil", "url", "image"], default="pil")
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

    model, processor = load_gemma4(
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
