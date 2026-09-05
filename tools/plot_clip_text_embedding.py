#!/usr/bin/env python3
"""Visualize one CLIP text embedding per drone location.

For every description JSON, the script takes the first caption at 250 m,
encodes it with the CLIP text encoder, and projects the pooled embeddings to
two dimensions.  PCA is the dependency-free default; t-SNE is available when
scikit-learn is installed.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-clip-text-embedding")

import matplotlib.pyplot as plt
from transformers import AutoModel, AutoTokenizer


# --- Configuration ---
DEFAULT_INPUT_ROOT = Path("/media/data1/feihong/drone_img")
DEFAULT_CACHE_DIR = Path("/media/data1/feihong/hf_cache")
DEFAULT_MODEL = "openai/clip-vit-base-patch16"
DEFAULT_JSON_NAME = "qwen_6_28_description.json"
DEFAULT_OUTPUT = Path(__file__).resolve().with_name("clip_250_text_embedding.png")
SEED = 28
COLORS = ("#3567A8", "#E28743", "#4B9B73", "#A467A7", "#C7A23A", "#5A9AA8")
STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "by", "for", "from", "in",
    "is", "it", "of", "on", "the", "to", "with", "view", "aerial",
    "large", "small", "featuring", "features", "surrounded", "sits",
    "building", "buildings", "area", "areas", "central", "centrally",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--json-name", default=DEFAULT_JSON_NAME)
    parser.add_argument("--height", default="250")
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--projection", choices=("pca", "tsne"), default="pca")
    parser.add_argument("--clusters", type=int, default=6)
    parser.add_argument("--pair-samples", type=int, default=100_000)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--pdf-output", type=Path, default=None)
    parser.add_argument("--csv-output", type=Path, default=None)
    parser.add_argument("--summary-output", type=Path, default=None)
    parser.add_argument("--notes-output", type=Path, default=None)
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def load_first_captions(root: Path, json_name: str, height: str) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    for path in sorted(root.rglob(json_name)):
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        items = data.get("description_segments", {}).get(str(height), [])
        if not items:
            continue
        first = items[0]
        text = first.get("text", "") if isinstance(first, dict) else first
        if not isinstance(text, str) or not text.strip():
            continue
        location_id = path.parent.name
        preferred_image = path.parent / f"{height}_0.png"
        if not preferred_image.is_file():
            images = sorted(path.parent.glob(f"{height}_*"))
            preferred_image = images[0] if images else None
        records.append(
            {
                "location_id": location_id,
                "description_path": str(path),
                "image_path": str(preferred_image) if preferred_image else "",
                "text": " ".join(text.split()),
            }
        )
    if not records:
        raise RuntimeError(f"No first {height} m captions found below {root}")
    return records


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def pooled_tensor(output: object) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    value = getattr(output, "pooler_output", None)
    if isinstance(value, torch.Tensor):
        return value
    raise TypeError("The selected model did not return pooled CLIP text features.")


def encode_texts(
    texts: Sequence[str],
    model_name: str,
    cache_dir: Path,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, cache_dir=cache_dir, local_files_only=True
    )
    model = AutoModel.from_pretrained(
        model_name, cache_dir=cache_dir, local_files_only=True
    ).to(device)
    model.eval()

    batches: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            tokens = tokenizer(
                list(texts[start : start + batch_size]),
                padding=True,
                truncation=True,
                return_tensors="pt",
            ).to(device)
            features = pooled_tensor(model.get_text_features(**tokens)).float()
            features = torch.nn.functional.normalize(features, dim=1)
            batches.append(features.cpu().numpy())
    return np.concatenate(batches, axis=0)


def pca_projection(features: np.ndarray) -> tuple[np.ndarray, list[float]]:
    centered = features - features.mean(axis=0, keepdims=True)
    _, singular_values, right = np.linalg.svd(centered, full_matrices=False)
    coordinates = centered @ right[:2].T
    variance = singular_values**2
    explained = (variance[:2] / variance.sum()).tolist()
    return coordinates, explained


def project(features: np.ndarray, method: str) -> tuple[np.ndarray, dict[str, object]]:
    if method == "pca":
        coordinates, explained = pca_projection(features)
        return coordinates, {"method": "PCA", "explained_variance_ratio": explained}

    try:
        from sklearn.manifold import TSNE
    except ImportError as exc:
        raise RuntimeError(
            "--projection tsne requires scikit-learn; use --projection pca "
            "for the dependency-free projection."
        ) from exc
    coordinates = TSNE(
        n_components=2,
        perplexity=min(30.0, (len(features) - 1) / 3),
        init="pca",
        learning_rate="auto",
        random_state=SEED,
    ).fit_transform(features)
    return coordinates, {"method": "t-SNE", "perplexity": min(30.0, (len(features) - 1) / 3)}


def spherical_kmeans(features: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(SEED)
    centers = [features[rng.integers(len(features))]]
    for _ in range(1, k):
        distance = 1.0 - np.max(features @ np.stack(centers).T, axis=1)
        weights = np.maximum(distance, 0.0) ** 2
        centers.append(features[rng.choice(len(features), p=weights / weights.sum())])
    centers_array = np.stack(centers)

    labels = np.zeros(len(features), dtype=np.int64)
    for _ in range(50):
        updated_labels = np.argmax(features @ centers_array.T, axis=1)
        if np.array_equal(updated_labels, labels):
            break
        labels = updated_labels
        for cluster in range(k):
            members = features[labels == cluster]
            if len(members):
                center = members.mean(axis=0)
                centers_array[cluster] = center / np.linalg.norm(center)
    return labels, centers_array


def cluster_keywords(records: Sequence[dict[str, str]], labels: np.ndarray, k: int) -> list[str]:
    names: list[str] = []
    for cluster in range(k):
        words: Counter[str] = Counter()
        for record, label in zip(records, labels):
            if label != cluster:
                continue
            tokens = re.findall(r"[a-z][a-z-]+", record["text"].lower())
            words.update(token for token in tokens if token not in STOP_WORDS and len(token) > 2)
        names.append(", ".join(word for word, _ in words.most_common(3)))
    return names


def sampled_cosine_distances(features: np.ndarray, count: int) -> np.ndarray:
    rng = np.random.default_rng(SEED)
    left = rng.integers(0, len(features), size=count)
    right = rng.integers(0, len(features), size=count)
    same = left == right
    while same.any():
        right[same] = rng.integers(0, len(features), size=int(same.sum()))
        same = left == right
    return 1.0 - np.sum(features[left] * features[right], axis=1)


def create_figure(
    records: Sequence[dict[str, str]],
    coordinates: np.ndarray,
    labels: np.ndarray,
    distances: np.ndarray,
) -> plt.Figure:
    figure, (scatter_ax, hist_ax) = plt.subplots(
        1, 2, figsize=(13.2, 7.3), gridspec_kw={"width_ratios": (1.75, 1.0)}
    )
    figure.patch.set_facecolor("#FCFCFA")
    for axis in (scatter_ax, hist_ax):
        axis.set_facecolor("#FCFCFA")

    for cluster in range(int(labels.max()) + 1):
        selected = labels == cluster
        scatter_ax.scatter(
            coordinates[selected, 0],
            coordinates[selected, 1],
            s=15,
            alpha=0.72,
            color=COLORS[cluster % len(COLORS)],
            edgecolors="white",
            linewidths=0.18,
            label=f"C{cluster + 1}",
            rasterized=True,
        )

    scatter_ax.set_title("(T1)", loc="left", fontsize=12, fontweight="bold")
    scatter_ax.set_xlabel("Dimension 1")
    scatter_ax.set_ylabel("Dimension 2")
    scatter_ax.grid(True, color="#D9DEE3", linewidth=0.55, alpha=0.65)
    scatter_ax.legend(
        loc="upper center", bbox_to_anchor=(0.5, -0.105), ncol=3,
        frameon=False, fontsize=7.7, handletextpad=0.4, columnspacing=1.0,
    )

    hist_ax.hist(distances, bins=36, color="#6F91B7", edgecolor="white", linewidth=0.5)
    median = float(np.median(distances))
    hist_ax.axvline(median, color="#C85C3C", linewidth=1.8)
    hist_ax.set_title("(T2)", loc="left", fontsize=12, fontweight="bold")
    hist_ax.set_xlabel("Cosine distance")
    hist_ax.set_ylabel("Pairs")
    hist_ax.grid(axis="y", color="#D9DEE3", linewidth=0.55, alpha=0.65)
    hist_ax.spines[["top", "right"]].set_visible(False)
    scatter_ax.spines[["top", "right"]].set_visible(False)

    figure.suptitle(
        "Text feature distribution",
        x=0.055, y=0.975, ha="left", fontsize=17, fontweight="bold", color="#20262E",
    )
    figure.subplots_adjust(left=0.07, right=0.985, top=0.89, bottom=0.16, wspace=0.22)
    return figure


def main() -> int:
    args = parse_args()
    records = load_first_captions(args.input_root, args.json_name, args.height)
    device = resolve_device(args.device)
    features = encode_texts(
        [record["text"] for record in records],
        args.model_name,
        args.cache_dir,
        device,
        args.batch_size,
    )
    coordinates, projection_info = project(features, args.projection)
    labels, _ = spherical_kmeans(features, args.clusters)
    names = cluster_keywords(records, labels, args.clusters)
    distances = sampled_cosine_distances(features, args.pair_samples)

    figure = create_figure(
        records, coordinates, labels, distances,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=args.dpi, bbox_inches="tight", pad_inches=0.12)
    if args.pdf_output:
        args.pdf_output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(args.pdf_output, bbox_inches="tight", pad_inches=0.12)
    plt.close(figure)

    csv_output = args.csv_output or args.output.with_suffix(".csv")
    csv_output.parent.mkdir(parents=True, exist_ok=True)
    with csv_output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "location_id", "image_path", "description_path", "text",
                "projection_x", "projection_y", "cluster",
            ),
        )
        writer.writeheader()
        for record, xy, label in zip(records, coordinates, labels):
            writer.writerow({**record, "projection_x": xy[0], "projection_y": xy[1], "cluster": int(label) + 1})

    summary = {
        "input_root": str(args.input_root),
        "json_name": args.json_name,
        "height_m": args.height,
        "model_name": args.model_name,
        "device": str(device),
        "num_locations": len(records),
        "num_unique_descriptions": len({record["text"] for record in records}),
        "feature_dimension": int(features.shape[1]),
        "projection": projection_info,
        "num_clusters": args.clusters,
        "cluster_sizes": {str(i + 1): int((labels == i).sum()) for i in range(args.clusters)},
        "cluster_keywords": {str(i + 1): name for i, name in enumerate(names)},
        "pairwise_cosine_distance": {
            "num_sampled_pairs": len(distances),
            "mean": float(distances.mean()),
            "median": float(np.median(distances)),
            "p10": float(np.quantile(distances, 0.10)),
            "p90": float(np.quantile(distances, 0.90)),
        },
    }
    summary_output = args.summary_output or args.output.with_suffix(".json")
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    notes_output = args.notes_output or args.output.with_name(f"{args.output.stem}_notes.txt")
    cluster_lines = "\n".join(
        f"C{i + 1}：无监督语义分组；高频词：{name}；n={(labels == i).sum():,}。"
        for i, name in enumerate(names)
    )
    notes_output.write_text(
        "文本特征分布——标识说明\n"
        "========================\n\n"
        f"数据范围：每个无人机地点在 {args.height} 米高度的第一条描述，共 {len(records):,} 个地点、"
        f"{len(set(record['text'] for record in records)):,} 条不同文本。\n"
        f"编码器：{args.model_name}；每个地点得到一个 {features.shape[1]} 维、经过 L2 归一化的 pooled 文本特征。\n\n"
        f"T1：文本特征的 {projection_info['method']} 二维投影。图上越接近的点，CLIP 文本特征通常越相似。\n"
        "C1–C6：用 spherical k-means 从特征中自动得到的分组，仅用于显示特征结构，不是真实类别标签。\n"
        f"{cluster_lines}\n\n"
        f"T2：从 {len(distances):,} 对随机文本中得到的余弦距离分布。"
        f"红色竖线表示中位数（{np.median(distances):.3f}）；"
        f"P10={np.quantile(distances, 0.10):.3f}, P90={np.quantile(distances, 0.90):.3f}.\n",
        encoding="utf-8",
    )
    print(f"Saved {args.output}, {csv_output}, {summary_output}, and {notes_output} ({len(records):,} locations).")
    if args.pdf_output:
        print(f"Saved {args.pdf_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
