#!/usr/bin/env python3
"""Visualize fixed-view drone images in a pretrained vision-feature space.

Each location contributes exactly one ``<height>_<angle>.png`` image. Images
are encoded with the pooled SigLIP2 vision output, L2-normalized, projected by
t-SNE, and grouped by spherical k-means for visual separation only.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-drone-vision-embedding")

import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnnotationBbox, OffsetImage, TextArea, VPacker


# --- Configuration ---
DEFAULT_IMAGE_ROOT = Path("/media/data1/feihong/drone_img")
DEFAULT_CACHE_DIR = Path("/media/data1/feihong/hf_cache")
DEFAULT_MODEL = "google/siglip2-base-patch16-224"
DEFAULT_HEIGHT = 250
DEFAULT_ANGLE = 0
DEFAULT_FEATURE_CACHE = Path(__file__).resolve().with_name(
    "siglip2_250_0_vision_features.npz"
)
DEFAULT_TSNE_CACHE = Path(__file__).resolve().with_name(
    "siglip2_250_0_vision_tsne.npz"
)
DEFAULT_OUTPUT = Path(__file__).resolve().with_name(
    "siglip2_250_0_vision_embedding.png"
)
SEED = 28
COLORS = ("#3567A8", "#E28743", "#4B9B73", "#A467A7", "#C7A23A", "#5A9AA8")
MEDOID_IDS = ("2220", "1928", "1569", "0111", "0187", "1140")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    parser.add_argument("--angle", type=int, default=DEFAULT_ANGLE)
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--clusters", type=int, default=6)
    parser.add_argument("--thumbnail-zoom", type=float, default=0.48)
    parser.add_argument("--feature-cache", type=Path, default=DEFAULT_FEATURE_CACHE)
    parser.add_argument("--tsne-cache", type=Path, default=DEFAULT_TSNE_CACHE)
    parser.add_argument("--recompute-features", action="store_true")
    parser.add_argument("--recompute-tsne", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--pdf-output", type=Path, default=None)
    parser.add_argument("--csv-output", type=Path, default=None)
    parser.add_argument("--summary-output", type=Path, default=None)
    parser.add_argument("--notes-output", type=Path, default=None)
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if args.clusters <= 1:
        parser.error("--clusters must be greater than one")
    if args.clusters != len(MEDOID_IDS):
        parser.error(
            f"This figure defines {len(MEDOID_IDS)} medoid thumbnails; "
            f"--clusters must be {len(MEDOID_IDS)}."
        )
    if args.thumbnail_zoom <= 0:
        parser.error("--thumbnail-zoom must be positive")
    return args


def load_image_records(root: Path, height: int, angle: int) -> list[dict[str, str]]:
    if not root.is_dir():
        raise NotADirectoryError(f"Drone image root does not exist: {root}")
    records: list[dict[str, str]] = []
    filename = f"{height}_{angle}.png"
    for path in root.glob(f"*/{filename}"):
        try:
            location_id = f"{int(path.parent.name):04d}"
        except ValueError:
            continue
        records.append({"location_id": location_id, "image_path": str(path)})
    records.sort(key=lambda record: record["location_id"])
    location_ids = [record["location_id"] for record in records]
    if len(location_ids) != len(set(location_ids)):
        raise ValueError("Duplicate location IDs found for the fixed-view drone images.")
    if not records:
        raise RuntimeError(f"No {filename} images found below {root}")
    return records


def records_signature(records: Sequence[dict[str, str]]) -> str:
    digest = hashlib.sha256()
    for record in records:
        path = Path(record["image_path"])
        stat = path.stat()
        digest.update(
            f"{record['location_id']}\0{path}\0{stat.st_size}\0{stat.st_mtime_ns}\n".encode()
        )
    return digest.hexdigest()


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def pooled_image_tensor(output: Any) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    pooled = getattr(output, "pooler_output", None)
    if isinstance(pooled, torch.Tensor):
        return pooled
    raise TypeError("The selected model did not return pooled vision features.")


def encode_images(
    image_paths: Sequence[str],
    model_name: str,
    cache_dir: Path,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    processor = AutoImageProcessor.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        local_files_only=True,
    )
    model = AutoModel.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        local_files_only=True,
    ).to(device)
    model.eval()

    batches: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[start : start + batch_size]
            images: list[Image.Image] = []
            for raw_path in batch_paths:
                with Image.open(raw_path) as image:
                    images.append(image.convert("RGB"))
            inputs = processor(images=images, return_tensors="pt")
            inputs = {
                key: value.to(device)
                for key, value in inputs.items()
                if isinstance(value, torch.Tensor)
            }
            output = model.get_image_features(**inputs)
            features = F.normalize(pooled_image_tensor(output).float(), dim=1)
            batches.append(features.cpu().numpy())
            done = min(start + len(batch_paths), len(image_paths))
            print(f"Encoded {done:,}/{len(image_paths):,} images", flush=True)
    return np.concatenate(batches, axis=0)


def scalar_string(payload: Any, key: str) -> str:
    return str(np.asarray(payload[key]).item())


def cache_matches(
    cached: Any,
    records: Sequence[dict[str, str]],
    signature: str,
    model_name: str,
    height: int,
    angle: int,
) -> bool:
    required = {"location_ids", "signature", "model_name", "height", "angle"}
    if not required.issubset(cached.files):
        return False
    ids = np.asarray([record["location_id"] for record in records])
    return (
        np.array_equal(cached["location_ids"], ids)
        and scalar_string(cached, "signature") == signature
        and scalar_string(cached, "model_name") == model_name
        and int(np.asarray(cached["height"]).item()) == height
        and int(np.asarray(cached["angle"]).item()) == angle
    )


def load_or_encode_features(
    records: Sequence[dict[str, str]],
    model_name: str,
    cache_dir: Path,
    device: torch.device,
    batch_size: int,
    height: int,
    angle: int,
    feature_cache: Path,
    recompute: bool = False,
) -> tuple[np.ndarray, str]:
    signature = records_signature(records)
    if feature_cache.is_file() and not recompute:
        with np.load(feature_cache, allow_pickle=False) as cached:
            if cache_matches(cached, records, signature, model_name, height, angle):
                return np.asarray(cached["features"], dtype=np.float32), signature

    features = encode_images(
        [record["image_path"] for record in records],
        model_name,
        cache_dir,
        device,
        batch_size,
    )
    feature_cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        feature_cache,
        location_ids=np.asarray([record["location_id"] for record in records]),
        features=features.astype(np.float32),
        signature=np.asarray(signature),
        model_name=np.asarray(model_name),
        height=np.asarray(height),
        angle=np.asarray(angle),
    )
    return features, signature


def spherical_kmeans(features: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    if k > len(features):
        raise ValueError(f"clusters ({k}) cannot exceed samples ({len(features)})")
    rng = np.random.default_rng(SEED)
    centers = [features[rng.integers(len(features))]]
    for _ in range(1, k):
        distance = 1.0 - np.max(features @ np.stack(centers).T, axis=1)
        weights = np.maximum(distance, 0.0) ** 2
        if float(weights.sum()) == 0.0:
            centers.append(features[rng.integers(len(features))])
        else:
            centers.append(features[rng.choice(len(features), p=weights / weights.sum())])
    centers_array = np.stack(centers)

    labels = np.full(len(features), -1, dtype=np.int64)
    for _ in range(50):
        updated_labels = np.argmax(features @ centers_array.T, axis=1)
        if np.array_equal(updated_labels, labels):
            break
        labels = updated_labels
        for cluster in range(k):
            members = features[labels == cluster]
            if len(members):
                center = members.mean(axis=0)
                centers_array[cluster] = center / max(np.linalg.norm(center), 1e-12)
    return labels, centers_array


def compute_tsne(
    features: np.ndarray,
    records: Sequence[dict[str, str]],
    signature: str,
    model_name: str,
    height: int,
    angle: int,
    clusters: int,
    tsne_cache: Path,
    recompute: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    if tsne_cache.is_file() and not recompute:
        with np.load(tsne_cache, allow_pickle=False) as cached:
            if (
                cache_matches(cached, records, signature, model_name, height, angle)
                and "coordinates" in cached.files
                and "labels" in cached.files
                and "clusters" in cached.files
                and int(np.asarray(cached["clusters"]).item()) == clusters
            ):
                return cached["coordinates"], cached["labels"]

    try:
        from sklearn.manifold import TSNE
    except ImportError as exc:
        raise RuntimeError(
            "Computing a new t-SNE cache requires scikit-learn. Install it, then rerun."
        ) from exc

    coordinates = TSNE(
        n_components=2,
        perplexity=min(30.0, (len(features) - 1) / 3),
        init="pca",
        learning_rate="auto",
        max_iter=1000,
        random_state=SEED,
    ).fit_transform(features)
    labels, _ = spherical_kmeans(features, clusters)
    tsne_cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        tsne_cache,
        location_ids=np.asarray([record["location_id"] for record in records]),
        coordinates=coordinates.astype(np.float32),
        labels=labels,
        signature=np.asarray(signature),
        model_name=np.asarray(model_name),
        height=np.asarray(height),
        angle=np.asarray(angle),
        clusters=np.asarray(clusters),
    )
    return coordinates, labels


def sampled_cosine_distances(features: np.ndarray, count: int) -> np.ndarray:
    rng = np.random.default_rng(SEED)
    left = rng.integers(0, len(features), size=count)
    right = rng.integers(0, len(features), size=count)
    same = left == right
    while same.any():
        right[same] = rng.integers(0, len(features), size=int(same.sum()))
        same = left == right
    return 1.0 - np.sum(features[left] * features[right], axis=1)


def resolve_medoid_records(
    records: Sequence[dict[str, str]],
    labels: np.ndarray,
    medoid_ids: Sequence[str],
) -> dict[int, dict[str, str]]:
    id_to_index = {
        record["location_id"]: index
        for index, record in enumerate(records)
    }
    resolved: dict[int, dict[str, str]] = {}
    for cluster, location_id in enumerate(medoid_ids):
        if location_id not in id_to_index:
            raise FileNotFoundError(
                f"C{cluster + 1} medoid ID {location_id} is absent from the fixed-view records."
            )
        index = id_to_index[location_id]
        actual_cluster = int(labels[index])
        if actual_cluster != cluster:
            raise ValueError(
                f"Medoid ID {location_id} was assigned to C{actual_cluster + 1}, "
                f"not expected C{cluster + 1}."
            )
        image_path = Path(records[index]["image_path"])
        if not image_path.is_file():
            raise FileNotFoundError(f"Medoid thumbnail does not exist: {image_path}")
        resolved[cluster] = {
            **records[index],
            "record_index": str(index),
        }
    return resolved


def cluster_anchor_coordinates(coordinates: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Use the coordinate-wise median as a robust visual anchor per cluster."""
    anchors = []
    for cluster in range(int(labels.max()) + 1):
        selected = coordinates[labels == cluster]
        if not len(selected):
            raise ValueError(f"Cluster C{cluster + 1} has no projected points.")
        anchors.append(np.median(selected, axis=0))
    return np.asarray(anchors, dtype=np.float64)


def peripheral_thumbnail_positions(
    coordinates: np.ndarray,
    anchors: np.ndarray,
) -> tuple[dict[int, tuple[float, float]], tuple[float, float], tuple[float, float]]:
    """Place three thumbnails on each side while preserving vertical order."""
    x_min, y_min = coordinates.min(axis=0)
    x_max, y_max = coordinates.max(axis=0)
    width = max(float(x_max - x_min), 1.0)
    height = max(float(y_max - y_min), 1.0)
    left_clusters = sorted(np.argsort(anchors[:, 0])[:3], key=lambda c: anchors[c, 1], reverse=True)
    right_clusters = sorted(np.argsort(anchors[:, 0])[3:], key=lambda c: anchors[c, 1], reverse=True)
    y_slots = np.linspace(y_max - 0.12 * height, y_min + 0.12 * height, 3)
    left_x = x_min - 0.19 * width
    right_x = x_max + 0.19 * width
    positions: dict[int, tuple[float, float]] = {}
    for cluster, y_value in zip(left_clusters, y_slots):
        positions[int(cluster)] = (float(left_x), float(y_value))
    for cluster, y_value in zip(right_clusters, y_slots):
        positions[int(cluster)] = (float(right_x), float(y_value))
    x_limits = (float(x_min - 0.34 * width), float(x_max + 0.34 * width))
    y_limits = (float(y_min - 0.08 * height), float(y_max + 0.08 * height))
    return positions, x_limits, y_limits


def add_medoid_thumbnail(
    axis: plt.Axes,
    image_path: Path,
    cluster: int,
    location_id: str,
    anchor: np.ndarray,
    position: tuple[float, float],
    zoom: float,
) -> None:
    with Image.open(image_path) as image:
        thumbnail = np.asarray(image.convert("RGB"))
    image_box = OffsetImage(thumbnail, zoom=zoom, interpolation="bilinear")
    label_box = TextArea(
        f"C{cluster + 1} · ID {location_id}",
        textprops={
            "fontsize": 8.2,
            "fontweight": "bold",
            "color": COLORS[cluster % len(COLORS)],
            "ha": "center",
        },
    )
    packed = VPacker(children=(image_box, label_box), align="center", pad=0, sep=3)
    artist = AnnotationBbox(
        packed,
        xy=(float(anchor[0]), float(anchor[1])),
        xybox=position,
        xycoords="data",
        boxcoords="data",
        box_alignment=(0.5, 0.5),
        frameon=True,
        bboxprops={
            "boxstyle": "round,pad=0.20",
            "facecolor": "white",
            "edgecolor": COLORS[cluster % len(COLORS)],
            "linewidth": 1.35,
        },
        arrowprops={
            "arrowstyle": "-",
            "color": COLORS[cluster % len(COLORS)],
            "linewidth": 0.85,
            "alpha": 0.78,
            "shrinkA": 5,
            "shrinkB": 3,
        },
        zorder=12,
    )
    axis.add_artist(artist)


def create_figure(
    coordinates: np.ndarray,
    labels: np.ndarray,
    records: Sequence[dict[str, str]],
    medoid_ids: Sequence[str],
    model_name: str,
    height: int,
    angle: int,
    thumbnail_zoom: float,
) -> tuple[plt.Figure, list[dict[str, Any]]]:
    figure, scatter_axis = plt.subplots(figsize=(13.2, 8.3))
    figure.patch.set_facecolor("#FCFCFA")
    scatter_axis.set_facecolor("#FCFCFA")

    for cluster in range(int(labels.max()) + 1):
        selected = labels == cluster
        scatter_axis.scatter(
            coordinates[selected, 0],
            coordinates[selected, 1],
            s=14,
            alpha=0.76,
            color=COLORS[cluster % len(COLORS)],
            edgecolors="white",
            linewidths=0.18,
            rasterized=True,
            zorder=3,
        )

    medoid_records = resolve_medoid_records(records, labels, medoid_ids)
    anchors = cluster_anchor_coordinates(coordinates, labels)
    positions, x_limits, y_limits = peripheral_thumbnail_positions(coordinates, anchors)
    medoid_summary: list[dict[str, Any]] = []
    for cluster in range(len(medoid_ids)):
        record = medoid_records[cluster]
        record_index = int(record["record_index"])
        color = COLORS[cluster % len(COLORS)]
        scatter_axis.scatter(
            [anchors[cluster, 0]],
            [anchors[cluster, 1]],
            s=48,
            facecolors="white",
            edgecolors=color,
            linewidths=1.25,
            zorder=8,
        )
        add_medoid_thumbnail(
            scatter_axis,
            Path(record["image_path"]),
            cluster,
            record["location_id"],
            anchors[cluster],
            positions[cluster],
            thumbnail_zoom,
        )
        medoid_summary.append(
            {
                "cluster": cluster + 1,
                "location_id": record["location_id"],
                "image_path": record["image_path"],
                "medoid_projection": [
                    float(coordinates[record_index, 0]),
                    float(coordinates[record_index, 1]),
                ],
                "cluster_anchor_projection": [
                    float(anchors[cluster, 0]),
                    float(anchors[cluster, 1]),
                ],
            }
        )

    scatter_axis.set_xlim(*x_limits)
    scatter_axis.set_ylim(*y_limits)
    scatter_axis.set_aspect("equal", adjustable="box")
    scatter_axis.set_xticks([])
    scatter_axis.set_yticks([])
    scatter_axis.spines[["top", "right", "left", "bottom"]].set_visible(False)

    figure.suptitle(
        "Fixed-view UAV vision feature t-SNE",
        x=0.055,
        y=0.972,
        ha="left",
        fontsize=17,
        fontweight="bold",
        color="#20262E",
    )
    figure.text(
        0.055,
        0.932,
        f"{height} m, {angle}° · {model_name} · n={len(coordinates):,} · thumbnails show cluster-medoid UAV views",
        ha="left",
        fontsize=9,
        color="#5F6772",
    )
    figure.subplots_adjust(left=0.035, right=0.985, top=0.89, bottom=0.035)
    return figure, medoid_summary


def main() -> int:
    args = parse_args()
    records = load_image_records(args.image_root, args.height, args.angle)
    device = resolve_device(args.device)
    features, signature = load_or_encode_features(
        records,
        args.model_name,
        args.cache_dir,
        device,
        args.batch_size,
        args.height,
        args.angle,
        args.feature_cache,
        args.recompute_features,
    )
    coordinates, labels = compute_tsne(
        features,
        records,
        signature,
        args.model_name,
        args.height,
        args.angle,
        args.clusters,
        args.tsne_cache,
        args.recompute_tsne,
    )
    figure, medoid_summary = create_figure(
        coordinates,
        labels,
        records,
        MEDOID_IDS,
        args.model_name,
        args.height,
        args.angle,
        args.thumbnail_zoom,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=args.dpi, bbox_inches="tight", pad_inches=0.12)
    if args.pdf_output:
        args.pdf_output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(args.pdf_output, bbox_inches="tight", pad_inches=0.12)
    plt.close(figure)

    csv_output = args.csv_output or args.output.with_suffix(".csv")
    with csv_output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "location_id", "image_path", "projection_x", "projection_y",
                "cluster", "is_cluster_medoid_thumbnail",
            ),
        )
        writer.writeheader()
        for record, coordinate, label in zip(records, coordinates, labels):
            writer.writerow(
                {
                    **record,
                    "projection_x": coordinate[0],
                    "projection_y": coordinate[1],
                    "cluster": int(label) + 1,
                    "is_cluster_medoid_thumbnail": record["location_id"] == MEDOID_IDS[int(label)],
                }
            )

    summary = {
        "image_root": str(args.image_root),
        "height_m": args.height,
        "angle_deg": args.angle,
        "model_name": args.model_name,
        "device": str(device),
        "num_locations": len(records),
        "feature_dimension": int(features.shape[1]),
        "feature_cache": str(args.feature_cache),
        "tsne_cache": str(args.tsne_cache),
        "num_clusters": args.clusters,
        "cluster_sizes": {
            str(index + 1): int((labels == index).sum())
            for index in range(args.clusters)
        },
        "cluster_medoid_thumbnails": medoid_summary,
    }
    summary_output = args.summary_output or args.output.with_suffix(".json")
    summary_output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    notes_output = args.notes_output or args.output.with_name(f"{args.output.stem}_notes.txt")
    notes_output.write_text(
        "无人机视觉特征分布——标识说明\n"
        "==============================\n\n"
        f"数据范围：每个地点固定使用 {args.height} 米、{args.angle}° 的无人机图像，共 {len(records):,} 个地点。\n"
        f"编码器：{args.model_name}；每张图得到一个 {features.shape[1]} 维、L2 归一化的 pooled vision feature。\n"
        "主图：视觉特征的 t-SNE 二维投影；C1–C6 为 spherical k-means 无监督视觉簇，不是真实类别。\n"
        "点云外围的六张本地 UAV 图是各簇的代表缩略图，细线指向对应簇的二维中位数位置：\n"
        + "\n".join(
            f"  C{item['cluster']}：ID {item['location_id']}，{item['image_path']}"
            for item in medoid_summary
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        f"Saved {args.output}, {csv_output}, {summary_output}, and {notes_output} "
        f"({len(records):,} locations)."
    )
    if args.pdf_output:
        print(f"Saved {args.pdf_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
