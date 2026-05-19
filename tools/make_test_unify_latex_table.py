import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple, Union


DEFAULT_INPUT = (
    # "eval_results/test_unify/test_unify_encoder_heat_configs_summary.json",
    "eval_results/test_unify/single_config/test_unify_encoder_sig_model_1.json",
    "eval_results/test_unify/single_config/test_unify_encoder_sig_model_9.json",
    # "eval_results/test_unify/test_unify_encoder_heat_model_end1.json",
    # "eval_results/test_unify/test_unify_encoder_heat_model_end2.json",
    # "eval_results/test_unify/test_unify_encoder_heat_model_end3.json",
    # "eval_results/test_unify/test_unify_encoder_heat_model_end4.json",
    # "eval_results/test_unify/test_unify_encoder_heat_model_end5.json"
)

InputPath = Union[str, Path]
InputPaths = Union[InputPath, Sequence[InputPath]]

RETRIEVAL_COLUMNS: Sequence[Tuple[str, str]] = (
    ("recall@1", "R@1"),
    ("recall@5", "R@5"),
    ("recall@10", "R@10"),
)

GROUNDING_COLUMNS: Sequence[Tuple[str, str]] = (
    ("mean_iou", "mIoU"),
    ("ratio_iou_gt_0_25", "IoU@.25"),
    ("ratio_iou_gt_0_5", "IoU@.50"),
    ("mean_center_distance", "C.Dist"),
)

UNIFY_COLUMNS: Sequence[Tuple[str, str]] = (
    ("uIoU", "uIoU"),
    ("ratio_uIoU_gt_0_25", "uIoU@.25"),
)


def load_results(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if isinstance(payload, dict):
        return [payload]
    if isinstance(payload, list):
        return payload
    raise ValueError(f"Expected JSON list or object, got {type(payload).__name__}.")


def load_all_results(inputs: InputPaths) -> List[Dict[str, Any]]:
    if isinstance(inputs, (str, Path)):
        input_paths = [inputs]
    else:
        input_paths = list(inputs)

    results: List[Dict[str, Any]] = []
    for path in input_paths:
        results.extend(load_results(Path(path)))
    return results


def latex_escape(value: Any) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(ch, ch) for ch in text)


def model_name(item: Dict[str, Any]) -> str:
    for key in ("exp_name", "config_name"):
        value = item.get(key)
        if value:
            return str(value)

    checkpoint = item.get("checkpoint")
    if checkpoint:
        path = Path(str(checkpoint))
        if path.parent.name:
            return path.parent.name
        return path.stem

    return str(item.get("model_type", "model"))


def format_metric(value: Any, precision: int, percent: bool) -> str:
    if value is None:
        return "--"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return latex_escape(value)
    if percent:
        number *= 100.0
    return f"{number:.{precision}f}"


def metric_value(item: Dict[str, Any], key: str) -> Any:
    overall = item.get("overall", {}) or {}
    retrieval = item.get("retrieval", {}) or {}
    if key in retrieval:
        return retrieval[key]
    return overall.get(key)


def build_latex_table(
    results: Iterable[Dict[str, Any]],
    precision: int = 2,
    percent: bool = True,
    caption: str = "Overall results on test_unify.",
    label: str = "tab:test-unify-overall",
) -> str:
    columns = [
        *RETRIEVAL_COLUMNS,
        *GROUNDING_COLUMNS,
        *UNIFY_COLUMNS,
    ]
    num_cols = 1 + len(columns)

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\resizebox{\linewidth}{!}{%",
        r"\begin{tabular}{l" + "c" * len(columns) + r"}",
        r"\toprule",
        (
            r"\multirow{2}{*}{Model} & "
            r"\multicolumn{3}{c}{Retrieval} & "
            r"\multicolumn{4}{c}{Grounding} & "
            r"\multicolumn{2}{c}{Unify} \\"
        ),
        (
            r"\cmidrule(lr){2-4} "
            r"\cmidrule(lr){5-8} "
            r"\cmidrule(lr){9-10}"
        ),
        " & ".join([""] + [label for _, label in columns]) + r" \\",
        r"\midrule",
    ]

    for item in results:
        row = [latex_escape(model_name(item))]
        for key, _ in columns:
            is_distance = key == "mean_center_distance"
            row.append(
                format_metric(
                    metric_value(item, key),
                    precision=precision,
                    percent=percent and not is_distance,
                )
            )
        lines.append(" & ".join(row) + r" \\")

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}%",
            r"}",
            rf"\caption{{{latex_escape(caption)}}}",
            rf"\label{{{latex_escape(label)}}}",
            r"\end{table}",
        ]
    )
    if num_cols != 10:
        raise AssertionError(f"Unexpected LaTeX column count: {num_cols}")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert test_unify encoder_heat config summary JSON into a LaTeX "
            "table with retrieval, grounding, and unify metrics."
        )
    )
    parser.add_argument(
        "--input",
        nargs="+",
        default=DEFAULT_INPUT,
        help="Input summary JSON path(s).",
    )
    parser.add_argument("--output", default='eval_results/test_unify/res_tab.tex', help="Optional output .tex path.")
    parser.add_argument("--precision", type=int, default=2, help="Decimal places.")
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Print raw metric values instead of percentages for ratio metrics.",
    )
    parser.add_argument(
        "--caption",
        default="Overall results on test_unify.",
        help="LaTeX table caption.",
    )
    parser.add_argument(
        "--label",
        default="tab:test-unify-overall",
        help="LaTeX table label.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = load_all_results(args.input)
    table = build_latex_table(
        results,
        precision=args.precision,
        percent=not args.raw,
        caption=args.caption,
        label=args.label,
    )

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(table + "\n", encoding="utf-8")
    else:
        print(table)


if __name__ == "__main__":
    main()
