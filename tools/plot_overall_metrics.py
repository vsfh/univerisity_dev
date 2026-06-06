import argparse
import csv
from pathlib import Path
from typing import Dict, List


DEFAULT_CSV = "/data/feihong/univerisity_dev/eval_results/test_unify/overall_metrics.csv"
DEFAULT_OUTPUT = "/data/feihong/univerisity_dev/eval_results/test_unify/overall_metrics_lines.png"
METRIC_COLUMNS = ("recall@1", "mean_iou", "uIoU")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot recall@1, mean IoU, and uIoU from overall_metrics.csv."
    )
    parser.add_argument("--csv", type=str, default=DEFAULT_CSV)
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--x-column",
        type=str,
        default=None,
        help="Optional numeric CSV column for the x-axis, e.g. candidate_size.",
    )
    parser.add_argument(
        "--label-column",
        type=str,
        default="result_name",
        help="CSV column used for x tick labels when --x-column is not set.",
    )
    parser.add_argument("--title", type=str, default="Overall Metrics")
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def load_rows(csv_path: str) -> List[Dict[str, str]]:
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        raise ValueError(f"No rows found in CSV: {path}")

    missing = [column for column in METRIC_COLUMNS if column not in rows[0]]
    if missing:
        raise KeyError(f"Missing metric column(s): {missing}")
    return rows


def to_float(value: str, column: str, row_idx: int) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid numeric value for {column} at row {row_idx + 1}: {value!r}") from exc


def main() -> None:
    args = parse_args()
    rows = load_rows(args.csv)

    if args.x_column is not None:
        if args.x_column not in rows[0]:
            raise KeyError(f"x-axis column not found: {args.x_column}")
        x_values = [to_float(row[args.x_column], args.x_column, idx) for idx, row in enumerate(rows)]
        x_tick_labels = [str(row[args.x_column]) for row in rows]
        x_label = args.x_column
    else:
        x_values = list(range(1, len(rows) + 1))
        x_tick_labels = [
            row.get(args.label_column, "") or str(idx + 1)
            for idx, row in enumerate(rows)
        ]
        x_label = args.label_column if args.label_column in rows[0] else "row"

    metric_values = {
        column: [to_float(row[column], column, idx) for idx, row in enumerate(rows)]
        for column in METRIC_COLUMNS
    }

    display_names = {
        "recall@1": "Recall@1",
        "mean_iou": "Mean IoU",
        "uIoU": "uIoU",
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib.pyplot as plt

        fig_width = max(10.0, min(24.0, len(rows) * 1.2))
        fig, ax = plt.subplots(figsize=(fig_width, 6.0))
        for column in METRIC_COLUMNS:
            ax.plot(
                x_values,
                metric_values[column],
                marker="o",
                linewidth=2.0,
                markersize=5.0,
                label=display_names[column],
            )

        ax.set_title(args.title)
        ax.set_xlabel(x_label)
        ax.set_ylabel("Score")
        ax.set_ylim(0.0, 1.0)
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.legend()
        ax.set_xticks(x_values)
        ax.set_xticklabels(x_tick_labels, rotation=35, ha="right")
        fig.tight_layout()
        fig.savefig(output_path, dpi=int(args.dpi))
    except ModuleNotFoundError:
        save_with_pil(
            output_path=output_path,
            title=args.title,
            x_label=x_label,
            x_tick_labels=x_tick_labels,
            metric_values=metric_values,
            display_names=display_names,
        )
    print(f"Saved plot: {output_path}")


def save_with_pil(
    output_path: Path,
    title: str,
    x_label: str,
    x_tick_labels: List[str],
    metric_values: Dict[str, List[float]],
    display_names: Dict[str, str],
) -> None:
    from PIL import Image, ImageDraw, ImageFont

    row_count = len(x_tick_labels)
    width = max(1200, min(2600, 260 + row_count * 130))
    height = 760
    margin_left = 90
    margin_right = 40
    margin_top = 90
    margin_bottom = 220
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom

    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 18)
        small_font = ImageFont.truetype("DejaVuSans.ttf", 14)
        title_font = ImageFont.truetype("DejaVuSans-Bold.ttf", 24)
    except OSError:
        font = ImageFont.load_default()
        small_font = ImageFont.load_default()
        title_font = ImageFont.load_default()

    colors = {
        "recall@1": (39, 101, 186),
        "mean_iou": (214, 111, 36),
        "uIoU": (38, 147, 92),
    }

    def x_pos(idx: int) -> float:
        if row_count <= 1:
            return margin_left + plot_w / 2
        return margin_left + idx / (row_count - 1) * plot_w

    def y_pos(value: float) -> float:
        value = max(0.0, min(1.0, float(value)))
        return margin_top + (1.0 - value) * plot_h

    draw.text((margin_left, 25), title, fill=(20, 20, 20), font=title_font)
    draw.line((margin_left, margin_top, margin_left, margin_top + plot_h), fill=(0, 0, 0), width=2)
    draw.line((margin_left, margin_top + plot_h, margin_left + plot_w, margin_top + plot_h), fill=(0, 0, 0), width=2)

    for tick_idx in range(6):
        value = tick_idx / 5
        y = y_pos(value)
        draw.line((margin_left, y, margin_left + plot_w, y), fill=(225, 225, 225), width=1)
        draw.text((18, y - 9), f"{value:.1f}", fill=(80, 80, 80), font=small_font)

    for idx, label in enumerate(x_tick_labels):
        x = x_pos(idx)
        draw.line((x, margin_top + plot_h, x, margin_top + plot_h + 6), fill=(0, 0, 0), width=1)
        short_label = label if len(label) <= 28 else label[:25] + "..."
        draw.text((x - 45, margin_top + plot_h + 14), short_label, fill=(60, 60, 60), font=small_font)

    for column in METRIC_COLUMNS:
        values = metric_values[column]
        points = [(x_pos(idx), y_pos(value)) for idx, value in enumerate(values)]
        color = colors[column]
        if len(points) >= 2:
            draw.line(points, fill=color, width=4)
        for x, y in points:
            draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=color, outline="white", width=1)

    legend_x = margin_left + plot_w - 360
    legend_y = 28
    for idx, column in enumerate(METRIC_COLUMNS):
        y = legend_y + idx * 26
        draw.line((legend_x, y + 10, legend_x + 34, y + 10), fill=colors[column], width=4)
        draw.text((legend_x + 45, y), display_names[column], fill=(30, 30, 30), font=font)

    draw.text((margin_left + plot_w / 2 - 50, height - 36), x_label, fill=(30, 30, 30), font=font)
    image.save(output_path)


if __name__ == "__main__":
    main()
