import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_ROOT = Path("/media/data1/feihong/drone_img")
DEFAULT_JSON_NAME = "qwen_6_28_description.json"
DEFAULT_OUTPUT = REPO_ROOT / "error_text.txt"
DEFAULT_MODEL_NAME = "Qwen/Qwen3.6-35B-A3B-FP8"
DEFAULT_CACHE_DIR = "/media/4tb/feihong/hf_cache"
EXPECTED_HEIGHTS = ("150", "200", "250", "300")
EXPECTED_COUNT = 5
MAX_WORDS = 52
LOW_INFO_PHRASES = (
    "an aerial view of",
    "aerial view of",
    "a view of",
    "an image of",
    "image of",
)
ARTICLES = ("a", "an", "the")
PREPOSITIONS = ("with", "of", "for", "in", "on", "at", "by", "from", "to", "and", "but")


def word_count(text: str) -> int:
    return len(text.strip().split())


def clean_caption(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", text)
    return " ".join(text.split())


def remove_words(text: str, words: Tuple[str, ...]) -> str:
    pattern = "|".join(re.escape(word) for word in words)
    return clean_caption(re.sub(rf"\b(?:{pattern})\b\s*", "", text, flags=re.IGNORECASE))


def local_compress_caption(text: str) -> str:
    result = clean_caption(text)
    for phrase in LOW_INFO_PHRASES:
        result = re.sub(rf"\b{re.escape(phrase)}\b", "", result, flags=re.IGNORECASE)
    result = remove_words(result, ARTICLES)
    if word_count(result) < MAX_WORDS:
        return result

    result_without_prepositions = remove_words(result, PREPOSITIONS)
    if word_count(result_without_prepositions) < MAX_WORDS:
        return result_without_prepositions
    return text


def split_captions(text: str) -> List[str]:
    if "||" in text:
        parts = text.split("||")
    else:
        parts = text.splitlines()
    return [clean_caption(part) for part in parts if clean_caption(part)]


def case_id(path: Path, data: Dict[str, Any]) -> str:
    return str(data.get("satellite_id") or path.parent.name)


def backup_path_for(path: Path) -> Path:
    return path.with_name(f"old_{path.name}")


def bad_json(path: Path) -> List[str]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return [f"{path.parent.name} - - invalid json"]

    errors: List[str] = []
    cid = case_id(path, data)
    segments = data.get("description_segments")
    if not isinstance(segments, dict):
        return [f"{cid} - - missing description_segments"]

    for height in EXPECTED_HEIGHTS:
        items = segments.get(height)
        if not isinstance(items, list) or len(items) != EXPECTED_COUNT:
            got = len(items) if isinstance(items, list) else 0
            errors.append(f"{cid} {height} - expected 5 descriptions (got {got})")
            continue
        for pos, item in enumerate(items, start=1):
            index = item.get("index", pos) if isinstance(item, dict) else pos
            text = item.get("text") if isinstance(item, dict) else None
            if not isinstance(text, str) or not text.strip():
                errors.append(f"{cid} {height} {index} missing text")
            elif word_count(text) >= MAX_WORDS:
                errors.append(f"{cid} {height} {index} exceed word length (limit={MAX_WORDS})")
    return errors


def qwen_text(model: Any, processor: Any, prompt: str, max_new_tokens: int = 260) -> str:
    from tools.generate_qwen_6_25 import _generate

    message = {"role": "user", "content": [{"type": "text", "text": prompt}]}
    return _generate(
        model=model,
        processor=processor,
        messages=[message],
        max_new_tokens=max_new_tokens,
        temperature=0.0,
        top_p=0.95,
        top_k=64,
        enable_thinking=False,
    ).strip()


def compress_caption(model: Any, processor: Any, text: str) -> str:
    local_result = local_compress_caption(text)
    if word_count(local_result) < MAX_WORDS:
        return local_result

    prompt = f"""
Rewrite the caption below to fewer than {MAX_WORDS} words.
Do not change meaning. Do not remove any distinctive visual feature.
Prefer removing low-information words such as "a", "the", "a view of",
"image of", "aerial view of", and similar filler.
Return only the rewritten caption.

Caption:
{text}
""".strip()
    result = clean_caption(qwen_text(model, processor, prompt))
    return result if result and word_count(result) < MAX_WORDS else text


def make_variants(
    model: Any,
    processor: Any,
    base_text: str,
    count: int,
) -> List[str]:
    prompt = f"""
Create exactly {count} new captions based on the caption below.
Each new caption must express exactly the same meaning and keep all original
visual features. Only change word order and replace a few words with synonyms.
Each caption must be fewer than {MAX_WORDS} words.
Separate captions with ||. Return captions only.

Caption:
{base_text}
""".strip()
    variants = split_captions(qwen_text(model, processor, prompt, max_new_tokens=320))
    return variants[:count]


def fix_items(model: Any, processor: Any, items: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], bool]:
    changed = False
    fixed = [item for item in items if isinstance(item, dict)]

    if len(fixed) > EXPECTED_COUNT:
        fixed = fixed[:EXPECTED_COUNT]
        changed = True

    for idx, item in enumerate(fixed, start=1):
        item["index"] = idx
        text = str(item.get("text", "")).strip()
        if text and word_count(text) >= MAX_WORDS:
            item["text"] = compress_caption(model, processor, text)
            changed = True

    base_text = str(fixed[0].get("text", "")).strip() if fixed else ""
    if base_text and len(fixed) < EXPECTED_COUNT:
        needed = EXPECTED_COUNT - len(fixed)
        for text in make_variants(model, processor, base_text, needed):
            if word_count(text) >= MAX_WORDS:
                text = compress_caption(model, processor, text)
            fixed.append({"index": len(fixed) + 1, "text": text})
            changed = True
    return fixed, changed


def fix_json(path: Path, model: Any, processor: Any) -> bool:
    original_text = path.read_text(encoding="utf-8")
    data = json.loads(original_text)
    segments = data.setdefault("description_segments", {})
    if not isinstance(segments, dict):
        segments = {}
        data["description_segments"] = segments

    changed = False
    for height in EXPECTED_HEIGHTS:
        items = segments.get(height)
        if not isinstance(items, list):
            items = []
        fixed, height_changed = fix_items(model, processor, items)
        segments[height] = fixed
        changed = changed or height_changed or fixed != items

    if changed:
        backup_path_for(path).write_text(original_text, encoding="utf-8")
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return changed


def load_fix_model(args: argparse.Namespace) -> Tuple[Any, Any]:
    from tools.generate_qwen_6_25 import load_qwen36, resolve_local_snapshot

    model_name = args.model_name
    pre_download = not args.local_files_only
    local_snapshot = resolve_local_snapshot(args.model_name, args.cache_dir)
    if local_snapshot is not None:
        model_name = str(local_snapshot)
        pre_download = False
        print(f"Using local model snapshot: {model_name}")
    elif args.local_files_only:
        raise FileNotFoundError(
            f"No complete local snapshot found for {args.model_name} under {args.cache_dir}."
        )

    return load_qwen36(
        model_name=model_name,
        cache_dir=args.cache_dir,
        gpu_id=args.gpu_id if args.gpu_id >= 0 else None,
        dtype=args.dtype,
        pre_download=pre_download,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--json-name", default=DEFAULT_JSON_NAME)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--fix", action="store_true")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    parser.add_argument("--gpu-id", type=int, default=-1)
    parser.add_argument("--dtype", default="auto", choices=["auto", "bf16", "fp16", "fp32"])
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--max-fix-cases", type=int, default=0)
    args = parser.parse_args()

    paths = sorted(args.root.glob(f"*/{args.json_name}"))

    if args.fix:
        model, processor = load_fix_model(args)
        fixed = 0
        for path in paths:
            if bad_json(path):
                if args.max_fix_cases > 0 and fixed >= args.max_fix_cases:
                    break
                if fix_json(path, model, processor):
                    fixed += 1
                    print(f"fixed {path}")
        print(f"fixed_cases={fixed}")

    errors: List[str] = []
    for path in paths:
        errors.extend(bad_json(path))

    args.output.write_text("\n".join(errors) + ("\n" if errors else ""), encoding="utf-8")
    print(f"checked={len(paths)} bad={len(errors)} output={args.output}")


if __name__ == "__main__":
    main()
