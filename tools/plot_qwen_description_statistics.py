"""Plot category coverage and a word cloud for Qwen description JSON files.

The category bars report the fraction of descriptions that contain at least one
keyword from each category. Categories are not mutually exclusive, so the bars
do not sum to one. The word cloud uses token frequency after stop-word removal.

Example:
    python tools/plot_qwen_description_statistics.py
    python tools/plot_qwen_description_statistics.py \
        --input-root /media/data1/feihong/drone_img \
        --output tools/qwen_description_statistics --formats png pdf
"""

import argparse
import csv
import json
import math
import os
import re
import tempfile
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont


# --- Configuration ---
DEFAULT_INPUT_ROOT = "/media/data1/feihong/drone_img"
DEFAULT_JSON_NAME = "qwen_6_28_description.json"
DEFAULT_OUTPUT = "tools/qwen_description_statistics"
DEFAULT_HEIGHTS = ("150", "200", "250", "300")
DEFAULT_FORMATS = ("png", "pdf")
DEFAULT_DPI = 300
RANDOM_SEED = 28

# A description contributes at most once to each category. Keyword overlap
# between categories is intentional (for example, "tower" is both a structure
# and a landmark). Add project-specific terms here when needed.
CATEGORY_PATTERNS: Mapping[str, Sequence[str]] = {
    # "Sign / Name": (
    #     r"sign(?:age)?", r"logo", r"lettering", r"label(?:ed)?", r"banner",
    #     r"billboard", r"inscription", r"brand(?:ing)?",
    # ),
    "Landmark": (
        r"tower", r"spire", r"dome", r"monument", r"statue", r"clock",
        r"stadium", r"cathedral", r"church", r"chapel", r"temple",
        r"castle", r"bridge", r"fountain", r"lighthouse", r"arch",
    ),
    "Road Elements": (
        r"road", r"street", r"lane", r"highway", r"intersection",
        r"roundabout", r"sidewalk", r"pavement", r"path", r"driveway",
        r"parking(?:\s+lot|\s+area)?", r"plaza", r"curb", r"crosswalk",
    ),
    "Buildings": (
        r"building", r"complex", r"structure", r"facility", r"block",
        r"wing", r"hall", r"house", r"roof", r"fa[cç]ade", r"courtyard",
        r"campus", r"warehouse", r"annex", r"tower",
    ),
    "Water &\nVegetation": (
        r"tree", r"lawn", r"grass", r"garden", r"vegetation", r"greenery",
        r"bush", r"shrub", r"forest", r"wooded", r"pond", r"lake",
        r"river", r"canal", r"water", r"fountain",
    ),
    "Direction": (
        r"left", r"right", r"cent(?:er|re)(?:ed)?", r"central(?:ly)?", r"upper",
        r"lower", r"bottom", r"top", r"corner", r"edge", r"north",
        r"south", r"east", r"west", r"beside", r"adjacent", r"flank(?:ed|ing)?",
        r"between", r"surround(?:ed|ing)?",
    ),
    "Vehicles": (
        r"car", r"vehicle", r"bus", r"truck", r"van", r"bicycle",
        r"bike", r"motorcycle",
    ),
    # "Sky &\nWeather": (
    #     r"sky", r"cloud", r"cloudy", r"sunny", r"shadow", r"snow",
    #     r"rain", r"fog", r"overcast",
    # ),
}

STOP_WORDS = {
    "a", "about", "above", "across", "after", "against", "all", "along",
    "also", "an", "and", "another", "are", "around", "as", "at", "be",
    "because", "been", "before", "being", "below", "between", "both", "but",
    "by", "can", "centered", "characterized", "contains", "dominated", "drone",
    "each", "edge", "features", "featuring", "few", "for", "frame", "from",
    "front", "has", "have", "having", "image", "in", "into", "is", "it",
    "its", "large", "main", "more", "most", "near", "next", "of", "on",
    "one", "other", "over", "perspective", "prominent", "scene", "several",
    "shot", "shows", "site", "sits", "small", "smaller", "some", "sprawling",
    "surrounded", "than", "that", "the", "their", "these", "this", "through",
    "to", "toward", "two", "under", "view", "visible", "was", "while", "with",
    "within", "whose", "aerial", "oblique", "area", "areas", "located",
    "positioned", "forming", "set", "multiple", "separate", "distinctive",
}

PLURAL_NORMALIZATION = {
    "buildings": "building", "roofs": "roof", "trees": "tree",
    "roads": "road", "streets": "street", "paths": "path", "wings": "wing",
    "windows": "window", "courtyards": "courtyard", "lawns": "lawn",
    "lots": "lot", "structures": "structure", "blocks": "block",
    "facilities": "facility", "panels": "panel", "skylights": "skylight",
    "cars": "car", "vehicles": "vehicle", "towers": "tower",
    "gardens": "garden", "fields": "field", "walls": "wall",
}

BAR_COLORS = (
    "#F3A47E", "#DEAD92", "#CDB6A2", "#BEB4A5",
    "#AAB7A8", "#96B9AE", "#82BDC0", "#70C5CE",
)
WORD_COLORS = (
    "#E88964", "#EE9C75", "#DDA788", "#C5B09D", "#9FA99D",
    "#79AEB0", "#63B7C0", "#B7A58F",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot category coverage and word frequency for Qwen captions."
    )
    parser.add_argument("--input-root", default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--json-name", default=DEFAULT_JSON_NAME)
    parser.add_argument(
        "--heights", nargs="*", default=list(DEFAULT_HEIGHTS),
        help="description_segments keys to include; pass no values to include all keys.",
    )
    parser.add_argument(
        "--unit", choices=("caption", "file"), default="caption",
        help="Use individual captions or one concatenated document per JSON as denominator.",
    )
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output path without extension.")
    parser.add_argument(
        "--formats", nargs="+", choices=("png", "pdf", "svg"),
        default=list(DEFAULT_FORMATS),
    )
    parser.add_argument("--csv", default=None, help="Optional statistics CSV path.")
    parser.add_argument("--top-words", type=int, default=80)
    parser.add_argument("--min-word-count", type=int, default=5)
    parser.add_argument("--dpi", type=int, default=DEFAULT_DPI)
    parser.add_argument("--title", default=None)
    parser.add_argument("--show-values", action="store_true")
    parser.add_argument("--font-family", default="DejaVu Serif")
    return parser.parse_args()


def extract_texts(path: Path, heights: Sequence[str]) -> List[str]:
    """Extract non-empty caption text from one generated-description JSON."""
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    segments = data.get("description_segments", {})
    if not isinstance(segments, dict):
        return []

    selected_keys = list(heights) if heights else list(segments)
    texts: List[str] = []
    for height in selected_keys:
        items = segments.get(str(height), [])
        if not isinstance(items, list):
            continue
        for item in items:
            text = item.get("text", "") if isinstance(item, dict) else item
            if isinstance(text, str) and text.strip():
                texts.append(" ".join(text.split()))
    return texts


def load_corpus(
    input_root: Path,
    json_name: str,
    heights: Sequence[str],
    unit: str,
) -> Tuple[List[str], int, List[Tuple[Path, str]]]:
    paths = sorted(input_root.rglob(json_name))
    if not paths:
        raise FileNotFoundError(f"No {json_name!r} found under {input_root}")

    documents: List[str] = []
    errors: List[Tuple[Path, str]] = []
    valid_files = 0
    for path in paths:
        try:
            texts = extract_texts(path, heights)
        except (OSError, json.JSONDecodeError) as exc:
            errors.append((path, str(exc)))
            continue
        if not texts:
            continue
        valid_files += 1
        documents.extend(texts if unit == "caption" else [" ".join(texts)])
    if not documents:
        raise ValueError("JSON files were found, but no valid descriptions were extracted.")
    return documents, valid_files, errors


def compile_categories() -> Dict[str, re.Pattern[str]]:
    return {
        name: re.compile(
            r"\b(?:" + "|".join(patterns) + r")(?:s|es)?\b",
            flags=re.IGNORECASE,
        )
        for name, patterns in CATEGORY_PATTERNS.items()
    }


def category_coverage(documents: Sequence[str]) -> Tuple[Dict[str, int], Dict[str, float]]:
    patterns = compile_categories()
    counts = {
        name: sum(bool(pattern.search(document)) for document in documents)
        for name, pattern in patterns.items()
    }
    total = len(documents)
    ratios = {name: count / total for name, count in counts.items()}
    return counts, ratios


def tokenize(documents: Iterable[str]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for document in documents:
        clean_text = document.lower().replace("-", " ").replace("’", "'")
        for token in re.findall(r"[a-z]+(?:'[a-z]+)?", clean_text):
            token = token.removesuffix("'s")
            token = PLURAL_NORMALIZATION.get(token, token)
            if len(token) >= 3 and token not in STOP_WORDS:
                counts[token] += 1
    return counts


def _font(font_path: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(font_path, size=max(8, int(size)))


def make_word_cloud(
    word_counts: Counter[str],
    top_words: int,
    min_count: int,
    font_path: str,
    width: int = 1300,
    height: int = 1350,
) -> Image.Image:
    """Create a deterministic dependency-free word cloud using Pillow."""
    words = [(word, count) for word, count in word_counts.most_common(top_words * 3)
             if count >= min_count][:top_words]
    if not words:
        raise ValueError("No words remain for the word cloud; lower --min-word-count.")

    rng = np.random.default_rng(RANDOM_SEED)
    image = Image.new("RGB", (width, height), "white")
    occupied = np.zeros((height, width), dtype=np.uint8)
    max_count = words[0][1]
    min_freq = words[-1][1]

    for rank, (word, count) in enumerate(words):
        if max_count == min_freq:
            scale = 1.0
        else:
            scale = (math.sqrt(count) - math.sqrt(min_freq)) / (
                math.sqrt(max_count) - math.sqrt(min_freq)
            )
        font_size = int(25 + scale * 91)
        color = WORD_COLORS[rank % len(WORD_COLORS)]
        rotate = bool(rank >= 12 and rng.random() < 0.14)

        font = _font(font_path, font_size)
        probe = Image.new("RGBA", (1, 1), (255, 255, 255, 0))
        probe_draw = ImageDraw.Draw(probe)
        bbox = probe_draw.textbbox((0, 0), word, font=font, stroke_width=0)
        text_w = max(1, bbox[2] - bbox[0])
        text_h = max(1, bbox[3] - bbox[1])
        tile = Image.new("RGBA", (text_w + 12, text_h + 12), (255, 255, 255, 0))
        tile_draw = ImageDraw.Draw(tile)
        tile_draw.text((6 - bbox[0], 6 - bbox[1]), word, font=font, fill=color)
        if rotate:
            tile = tile.rotate(90, expand=True, resample=Image.Resampling.BICUBIC)

        tile_w, tile_h = tile.size
        placed = False
        phase = rng.uniform(0.0, 2.0 * math.pi)
        for step in range(2200):
            theta = phase + step * 0.21
            radius = 0.48 * step
            x = int(width / 2 + radius * math.cos(theta) - tile_w / 2)
            y = int(height / 2 + 0.82 * radius * math.sin(theta) - tile_h / 2)
            if x < 5 or y < 5 or x + tile_w >= width - 5 or y + tile_h >= height - 5:
                continue
            if not occupied[y:y + tile_h, x:x + tile_w].any():
                image.paste(tile, (x, y), tile)
                occupied[max(0, y - 3):min(height, y + tile_h + 3),
                         max(0, x - 3):min(width, x + tile_w + 3)] = 1
                placed = True
                break
        if not placed:
            continue
    return image


def rounded_axis_limit(values: Sequence[float]) -> float:
    maximum = max(values, default=0.0)
    return min(1.0, max(0.1, math.ceil((maximum + 0.025) * 10.0) / 10.0))


def plot_statistics(
    ratios: Mapping[str, float],
    word_cloud: Image.Image,
    output_base: Path,
    formats: Sequence[str],
    dpi: int,
    title: str | None,
    show_values: bool,
    font_family: str,
) -> List[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FormatStrFormatter, MultipleLocator

    plt.rcParams.update({
        "font.family": font_family,
        "font.size": 10,
        "axes.linewidth": 1.2,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    names = list(ratios)
    values = [ratios[name] for name in names]
    y_positions = np.arange(len(names))
    fig = plt.figure(figsize=(10.6, 5.4), facecolor="white")
    grid = fig.add_gridspec(1, 2, width_ratios=(1.34, 1.0), wspace=0.01)
    ax_bar = fig.add_subplot(grid[0, 0])
    ax_cloud = fig.add_subplot(grid[0, 1])

    ax_bar.barh(
        y_positions,
        values,
        height=0.72,
        color=BAR_COLORS[:len(values)],
        edgecolor="white",
        linewidth=0.6,
    )
    ax_bar.set_yticks(y_positions, labels=names, fontweight="semibold")
    ax_bar.invert_yaxis()
    x_limit = rounded_axis_limit(values)
    ax_bar.set_xlim(0.0, x_limit)
    ax_bar.xaxis.set_major_locator(MultipleLocator(0.1))
    ax_bar.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax_bar.tick_params(axis="x", direction="out", length=5, width=1.1)
    ax_bar.tick_params(axis="y", length=4, width=1.0)
    ax_bar.spines["top"].set_visible(False)
    ax_bar.spines["right"].set_visible(False)
    ax_bar.grid(False)
    ax_bar.set_xlabel("Proportion of descriptions", labelpad=7)

    if show_values:
        for y, value in zip(y_positions, values):
            ax_bar.text(
                min(value + x_limit * 0.012, x_limit * 0.985), y,
                f"{value:.3f}", va="center", ha="left", fontsize=8.5,
            )

    ax_cloud.imshow(word_cloud)
    ax_cloud.set_axis_off()
    if title:
        fig.suptitle(title, y=0.985, fontsize=13, fontweight="semibold")
    fig.subplots_adjust(left=0.145, right=0.985, bottom=0.13, top=0.95)

    output_base.parent.mkdir(parents=True, exist_ok=True)
    saved_paths: List[Path] = []
    for output_format in dict.fromkeys(formats):
        path = output_base.with_suffix(f".{output_format}")
        save_kwargs = {"bbox_inches": "tight", "facecolor": "white"}
        if output_format == "png":
            save_kwargs["dpi"] = dpi
        fig.savefig(path, **save_kwargs)
        saved_paths.append(path)
    plt.close(fig)
    return saved_paths


def write_statistics_csv(
    path: Path,
    counts: Mapping[str, int],
    ratios: Mapping[str, float],
    denominator: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(("category", "matched_documents", "total_documents", "proportion"))
        for name in ratios:
            writer.writerow((name.replace("\n", " "), counts[name], denominator, f"{ratios[name]:.8f}"))


def main() -> None:
    args = parse_args()
    input_root = Path(args.input_root).expanduser()
    output_base = Path(args.output).expanduser()
    if output_base.suffix.lower().lstrip(".") in {"png", "pdf", "svg"}:
        output_base = output_base.with_suffix("")

    documents, valid_files, errors = load_corpus(
        input_root=input_root,
        json_name=args.json_name,
        heights=args.heights,
        unit=args.unit,
    )
    counts, ratios = category_coverage(documents)
    word_counts = tokenize(documents)

    # Some training servers expose a read-only home directory. Keep Matplotlib's
    # cache writable without overriding a user-provided MPLCONFIGDIR.
    os.environ.setdefault(
        "MPLCONFIGDIR",
        str(Path(tempfile.gettempdir()) / "qwen_description_matplotlib"),
    )
    from matplotlib import font_manager

    font_path = font_manager.findfont(args.font_family, fallback_to_default=True)
    cloud = make_word_cloud(
        word_counts=word_counts,
        top_words=max(1, args.top_words),
        min_count=max(1, args.min_word_count),
        font_path=font_path,
    )
    saved_paths = plot_statistics(
        ratios=ratios,
        word_cloud=cloud,
        output_base=output_base,
        formats=args.formats,
        dpi=args.dpi,
        title=args.title,
        show_values=args.show_values,
        font_family=args.font_family,
    )

    csv_path = Path(args.csv).expanduser() if args.csv else output_base.with_suffix(".csv")
    write_statistics_csv(csv_path, counts, ratios, len(documents))

    print(f"Scanned JSON files: {valid_files}")
    print(f"Analyzed {args.unit}s: {len(documents)}")
    if errors:
        print(f"Skipped invalid JSON files: {len(errors)}")
        for path, message in errors[:5]:
            print(f"  {path}: {message}")
    for name, ratio in ratios.items():
        print(f"  {name.replace(chr(10), ' '):20s} {ratio:.4f} ({counts[name]}/{len(documents)})")
    for path in saved_paths:
        print(f"Saved figure: {path}")
    print(f"Saved statistics: {csv_path}")


if __name__ == "__main__":
    main()
