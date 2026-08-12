import json
import math
import sqlite3
import statistics
from datetime import datetime, timezone
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


# --- Configuration ---
REPORT_TITLE = "Ada 文本对齐训练曲线诊断"
NUM_EPOCHS = 20
NUM_CANDIDATES_PER_BATCH = 32 * 15

SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[3]
OUTPUT_DIR = SCRIPT_PATH.parent
SQLITE_PATH = OUTPUT_DIR / "analysis.sqlite"

RUNS = {
    "text_scheduled": {
        "label": "Text 0.01（旧调度）",
        "line_style": "dotted",
        "event": "exp/runs/baseline_ada_1_5x3/events.out.tfevents.1786381215.4090-48g.4013619.0",
        "role": "historical",
    },
    "text_constant": {
        "label": "Text 0.01（全程固定）",
        "line_style": "solid",
        "event": "exp/runs/baseline_ada_1_5x3/events.out.tfevents.1786420717.4090-48g.511988.0",
        "role": "current",
    },
    "no_text": {
        "label": "No text",
        "line_style": "dashed",
        "event": "exp/runs/baseline_wo_input_ids_ada_5x3/events.out.tfevents.1786302112.4090-48g.2760507.0",
        "role": "baseline",
    },
}

TEXT_EVAL_PATH = "exp/eval_results/baseline_ada_1_5x3.json"
NO_TEXT_EVAL_PATH = "exp/eval_results/baseline_wo_input_ids_ada_5x3.json"


def load_event(relative_path: str) -> EventAccumulator:
    event = EventAccumulator(
        str(REPO_ROOT / relative_path),
        size_guidance={"scalars": 0},
    )
    return event.Reload()


def scalar_values(event: EventAccumulator, tag: str) -> list[float]:
    return [item.value for item in event.Scalars(tag)]


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def quote_identifier(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def sqlite_type(rows: list[dict], field: str) -> str:
    values = [row[field] for row in rows if row.get(field) is not None]
    if not values:
        return "TEXT"
    if all(isinstance(value, bool) or isinstance(value, int) for value in values):
        return "INTEGER"
    if all(isinstance(value, (int, float)) for value in values):
        return "REAL"
    return "TEXT"


def materialize_sqlite(datasets: dict[str, list[dict]]) -> dict[str, list[dict]]:
    connection = sqlite3.connect(SQLITE_PATH)
    connection.row_factory = sqlite3.Row
    try:
        for table_name, rows in datasets.items():
            if not rows:
                raise ValueError(f"Cannot materialize empty dataset {table_name}.")
            fields = list(rows[0])
            quoted_table = quote_identifier(table_name)
            connection.execute(f"DROP TABLE IF EXISTS {quoted_table}")
            columns_sql = ", ".join(
                f"{quote_identifier(field)} {sqlite_type(rows, field)}"
                for field in fields
            )
            connection.execute(f"CREATE TABLE {quoted_table} ({columns_sql})")
            placeholders = ", ".join("?" for _ in fields)
            columns = ", ".join(quote_identifier(field) for field in fields)
            connection.executemany(
                f"INSERT INTO {quoted_table} ({columns}) VALUES ({placeholders})",
                [[row.get(field) for field in fields] for row in rows],
            )
        connection.commit()

        reviewed = {}
        for table_name in datasets:
            query = f"SELECT * FROM {quote_identifier(table_name)}"
            reviewed[table_name] = [
                dict(row) for row in connection.execute(query).fetchall()
            ]
        return reviewed
    finally:
        connection.close()


def extract_epoch_rows() -> list[dict]:
    rows = []
    for run_id, spec in RUNS.items():
        event = load_event(spec["event"])
        train_epoch = scalar_values(event, "Loss/train_epoch")
        bbox_epoch = scalar_values(event, "Loss/bbox_epoch")
        heatmap_epoch = scalar_values(event, "Loss/heatmap_epoch")
        text_epoch = scalar_values(event, "Loss/text_pooler_align_epoch")
        retrieval_step = scalar_values(event, "Loss/retrieval_step")
        text_step = scalar_values(event, "Loss/text_pooler_align_step")
        weight_step = scalar_values(event, "Weight/text_pooler_align")

        if len(train_epoch) != NUM_EPOCHS:
            raise ValueError(
                f"{run_id} has {len(train_epoch)} epochs; expected {NUM_EPOCHS}."
            )
        if not (len(retrieval_step) == len(text_step) == len(weight_step)):
            raise ValueError(f"Step scalar lengths disagree for {run_id}.")
        if len(retrieval_step) % NUM_EPOCHS != 0:
            raise ValueError(f"Step count is not divisible by epochs for {run_id}.")

        steps_per_epoch = len(retrieval_step) // NUM_EPOCHS
        for epoch_index in range(NUM_EPOCHS):
            start = epoch_index * steps_per_epoch
            stop = (epoch_index + 1) * steps_per_epoch
            retrieval_values = retrieval_step[start:stop]
            text_values = text_step[start:stop]
            weight_values = weight_step[start:stop]
            pure_image_values = [
                retrieval - weight * text_loss
                for retrieval, weight, text_loss in zip(
                    retrieval_values,
                    weight_values,
                    text_values,
                )
            ]
            active_weight = mean(weight_values)
            active_text_loss = text_epoch[epoch_index] if active_weight > 0 else None
            rows.append(
                {
                    "epoch": epoch_index + 1,
                    "run_id": run_id,
                    "run": spec["label"],
                    "run_role": spec["role"],
                    "line_style": spec["line_style"],
                    "reported_train_loss": train_epoch[epoch_index],
                    "reported_retrieval_loss": mean(retrieval_values),
                    "pure_image_retrieval_loss": mean(pure_image_values),
                    "bbox_loss": bbox_epoch[epoch_index],
                    "heatmap_loss": heatmap_epoch[epoch_index],
                    "text_alignment_loss": active_text_loss,
                    "mean_logged_alignment_weight": active_weight,
                    "steps_per_epoch": steps_per_epoch,
                    "source_event": spec["event"],
                }
            )
    return rows


def last_row(rows: list[dict], run_id: str) -> dict:
    matching = [row for row in rows if row["run_id"] == run_id]
    return max(matching, key=lambda row: row["epoch"])


def load_json(relative_path: str) -> dict:
    with open(REPO_ROOT / relative_path, "r", encoding="utf-8") as file:
        return json.load(file)


def cluster_mean_ci(
    text_records: list[dict],
    no_text_records: list[dict],
    field: str,
) -> tuple[float, float, float]:
    by_label = {}
    for text_record, no_text_record in zip(text_records, no_text_records):
        text_key = (text_record["drone_path"], text_record["satellite_path"])
        no_text_key = (no_text_record["drone_path"], no_text_record["satellite_path"])
        if text_key != no_text_key:
            raise ValueError("Evaluation query order differs between runs.")
        by_label.setdefault(text_record["gt_label"], []).append(
            float(text_record[field]) - float(no_text_record[field])
        )

    cluster_means = [mean(values) for values in by_label.values()]
    estimate = mean(cluster_means)
    standard_error = statistics.stdev(cluster_means) / math.sqrt(len(cluster_means))
    return estimate, estimate - 1.96 * standard_error, estimate + 1.96 * standard_error


def extract_eval_rows() -> tuple[list[dict], dict]:
    text_eval = load_json(TEXT_EVAL_PATH)
    no_text_eval = load_json(NO_TEXT_EVAL_PATH)
    rows = []
    for run_id, label, payload, schedule in [
        (
            "text_scheduled",
            "Text 0.01（旧调度）",
            text_eval,
            "旧 run：第 1–5 epoch 调度后归零",
        ),
        ("no_text", "No text", no_text_eval, "无文本输入"),
    ]:
        overall = payload["overall"]
        rows.append(
            {
                "run_id": run_id,
                "run": label,
                "schedule": schedule,
                "num_samples": overall["num_samples"],
                "recall_at_1": overall["recall@1"],
                "recall_at_5": overall["recall@5"],
                "recall_at_10": overall["recall@10"],
                "mean_iou": overall["mean_iou"],
                "uiou": overall["uIoU"],
                "mean_center_distance": overall["mean_center_distance"],
                "checkpoint_path": payload["checkpoint"],
            }
        )

    text_records = text_eval["query_records"]
    no_text_records = no_text_eval["query_records"]
    r1_delta, r1_low, r1_high = cluster_mean_ci(
        text_records,
        no_text_records,
        "top1_correct",
    )
    iou_delta, iou_low, iou_high = cluster_mean_ci(
        text_records,
        no_text_records,
        "iou",
    )
    paired = {
        "r1_delta": r1_delta,
        "r1_ci_low": r1_low,
        "r1_ci_high": r1_high,
        "iou_delta": iou_delta,
        "iou_ci_low": iou_low,
        "iou_ci_high": iou_high,
        "num_clusters": len({record["gt_label"] for record in text_records}),
    }
    for row in rows:
        row.update(paired)
    return rows, paired


def build_artifact() -> dict:
    generated_at = datetime.now(timezone.utc).isoformat()
    epoch_rows = extract_epoch_rows()
    eval_rows, paired = extract_eval_rows()

    constant_final = last_row(epoch_rows, "text_constant")
    scheduled_final = last_row(epoch_rows, "text_scheduled")
    no_text_final = last_row(epoch_rows, "no_text")
    constant_first = min(
        (row for row in epoch_rows if row["run_id"] == "text_constant"),
        key=lambda row: row["epoch"],
    )

    reported_gap = (
        constant_final["reported_train_loss"] - no_text_final["reported_train_loss"]
    )
    pure_gap = (
        constant_final["pure_image_retrieval_loss"]
        - no_text_final["pure_image_retrieval_loss"]
    )
    bbox_gap = constant_final["bbox_loss"] - no_text_final["bbox_loss"]
    heatmap_gap = constant_final["heatmap_loss"] - no_text_final["heatmap_loss"]
    alignment_drop = (
        constant_final["text_alignment_loss"]
        / constant_first["text_alignment_loss"]
        - 1.0
    )
    uniform_alignment_loss = math.log(NUM_CANDIDATES_PER_BATCH)
    constant_rows = sorted(
        (row for row in epoch_rows if row["run_id"] == "text_constant"),
        key=lambda row: row["epoch"],
    )
    no_text_by_epoch = {
        row["epoch"]: row for row in epoch_rows if row["run_id"] == "no_text"
    }
    heatmap_deltas = [
        row["heatmap_loss"] - no_text_by_epoch[row["epoch"]]["heatmap_loss"]
        for row in constant_rows
    ]
    bbox_deltas = [
        row["bbox_loss"] - no_text_by_epoch[row["epoch"]]["bbox_loss"]
        for row in constant_rows
    ]
    heatmap_better_epochs = sum(delta < 0 for delta in heatmap_deltas)
    heatmap_last_ten_delta = mean(heatmap_deltas[-10:])
    bbox_last_five_delta = mean(bbox_deltas[-5:])

    final_rows = []
    for row in [scheduled_final, constant_final, no_text_final]:
        final_rows.append(
            {
                "run": row["run"],
                "run_role": row["run_role"],
                "reported_train_loss": row["reported_train_loss"],
                "pure_image_retrieval_loss": row["pure_image_retrieval_loss"],
                "bbox_loss": row["bbox_loss"],
                "heatmap_loss": row["heatmap_loss"],
                "text_alignment_loss": row["text_alignment_loss"],
                "effective_alignment_weight": row[
                    "mean_logged_alignment_weight"
                ],
            }
        )

    alignment_rows = [
        {
            "epoch": row["epoch"],
            "text_alignment_loss": row["text_alignment_loss"],
            "uniform_prediction_loss": uniform_alignment_loss,
        }
        for row in epoch_rows
        if row["run_id"] == "text_constant"
    ]

    headline_rows = [
        {
            "reported_loss_gap": reported_gap,
            "pure_image_loss_gap": pure_gap,
            "bbox_loss_gap": bbox_gap,
            "heatmap_loss_gap": heatmap_gap,
            "final_text_alignment_loss": constant_final["text_alignment_loss"],
            "text_alignment_change": alignment_drop,
        }
    ]

    reviewed_datasets = materialize_sqlite(
        {
            "epoch_curves": epoch_rows,
            "alignment_curve": alignment_rows,
            "headline_metrics": headline_rows,
            "final_epoch_summary": final_rows,
            "historical_eval_summary": eval_rows,
        }
    )

    curve_source = {
        "id": "curve-analysis",
        "label": "TensorBoard scalar extraction and loss decomposition",
        "path": "exp/analysis/baseline_ada_1_vs_wo_input_ids/analysis.sqlite",
        "query": {
            "engine": "SQLite",
            "language": "sql",
            "sql": "SELECT * FROM epoch_curves",
            "description": "All 20 epoch rows for the three selected complete event streams.",
            "executed_at": generated_at,
            "tables_used": ["epoch_curves"],
            "filters": [
                "Latest complete text event selected for the current checkpoint",
                "Latest complete no-text event selected for its current checkpoint",
                "Three earlier complete no-text reruns excluded",
                "Earlier scheduled-text event retained as historical",
                "Empty event excluded",
            ],
            "metric_definitions": [
                "Reported train loss = 0.9 × reported retrieval objective + 0.1 × bbox loss.",
                "Pure image retrieval loss = reported retrieval objective − logged weight × logged text alignment loss, computed stepwise then averaged by epoch.",
                "Lower loss is better.",
            ],
        },
    }
    alignment_source = {
        "id": "alignment-analysis",
        "label": "Fixed-weight text alignment curve",
        "path": "exp/analysis/baseline_ada_1_vs_wo_input_ids/analysis.sqlite",
        "query": {
            "engine": "SQLite",
            "language": "sql",
            "sql": "SELECT * FROM alignment_curve",
            "description": "Text alignment cross-entropy for the latest fixed-weight run.",
            "executed_at": generated_at,
            "tables_used": ["alignment_curve"],
            "filters": ["20 epochs", "Alignment weight fixed at 0.01"],
            "metric_definitions": [
                "Text alignment loss is cross-entropy over 32 × 15 satellite region candidates.",
                "Uniform-prediction reference is log(480).",
            ],
        },
    }
    final_source = {
        "id": "final-components",
        "label": "Final epoch component summary",
        "path": "exp/analysis/baseline_ada_1_vs_wo_input_ids/analysis.sqlite",
        "query": {
            "engine": "SQLite",
            "language": "sql",
            "sql": "SELECT * FROM final_epoch_summary",
            "description": "Epoch 20 component values for the three selected event streams.",
            "executed_at": generated_at,
            "tables_used": ["final_epoch_summary"],
            "filters": ["Epoch 20 only"],
            "metric_definitions": [
                "Pure image retrieval excludes the weighted text alignment term.",
                "Null text alignment means the text objective was inactive at epoch 20.",
            ],
        },
    }
    eval_source = {
        "id": "historical-eval",
        "label": "Historical test evaluation JSON files",
        "path": "exp/analysis/baseline_ada_1_vs_wo_input_ids/analysis.sqlite",
        "query": {
            "engine": "SQLite",
            "language": "sql",
            "sql": "SELECT * FROM historical_eval_summary",
            "description": "Top-line test metrics for the historical scheduled-text run and no-text baseline.",
            "executed_at": generated_at,
            "tables_used": ["historical_eval_summary"],
            "filters": [
                "89,728 aligned test queries",
                "Candidate set size 100",
                "2,804 gallery images",
            ],
            "metric_definitions": [
                "Recall@K is the fraction of queries with an accepted positive in the top K candidates.",
                "mIoU is mean predicted-versus-target box intersection over union.",
                "uIoU sets IoU to zero when top-1 retrieval is incorrect before averaging.",
                "Center distance is pixel distance between predicted and target box centers; lower is better.",
            ],
        },
    }
    config_source = {
        "id": "effective-configs",
        "label": "Effective experiment configuration snapshots",
        "path": "exp/analysis/baseline_ada_1_vs_wo_input_ids/build_report.py",
    }
    sources = [
        curve_source,
        alignment_source,
        final_source,
        eval_source,
        config_source,
    ]

    summary_body = f"""## 技术摘要

- **最新固定权重 run 没有出现图像检索或定位训练退化。** 第 20 epoch 的报告总 loss 比 no-text 高 {reported_gap:.4f}，但扣除持续加入的文本项后，纯图像检索 loss 反而低 {abs(pure_gap):.4f}；bbox loss 低 {abs(bbox_gap):.4f}，heatmap loss 低 {abs(heatmap_gap):.4f}。
- **总 loss 的高位主要是口径效应。** 文本实验把 `0.01 × text_alignment_loss` 加进 retrieval objective；第 20 epoch 该项仍约贡献 {0.01 * constant_final['text_alignment_loss']:.4f}，再乘 retrieval 权重 0.9 后约贡献 {0.009 * constant_final['text_alignment_loss']:.4f}，足以解释观察到的总 loss 差距。
- **文本对齐确实在学习，但最新模型尚未有测试结果。** 对齐交叉熵从 {constant_first['text_alignment_loss']:.3f} 降至 {constant_final['text_alignment_loss']:.3f}（{alignment_drop:.1%}）；不过当前固定权重 checkpoint 覆盖了旧 checkpoint，而现有测试 JSON 对应的是旧的“第 5 epoch 后关闭文本项”run。
- **因此现在可以下优化结论，不能下泛化结论。** 曲线支持“0.01 全程文本约束没有破坏训练收敛”；是否提升 R@1、mIoU 或 uIoU，必须重新测试当前 checkpoint，并最好做多随机种子复验。
"""

    scope_body = """## 比较口径：同为 20 epoch，但两个目录都混有完整重跑

本报告把三个完整事件流分别标记：当前 `Text 0.01（全程固定）`、历史 `Text 0.01（旧调度）`、以及与当前 no-text checkpoint 对应的最新 `No text`。三者均记录 20 个 epoch、每 epoch 701 个 step。`baseline_ada_1_5x3` 目录有两次完整 run 和一个 88-byte 空 event；`baseline_wo_input_ids_ada_5x3` 目录则有四次完整 run。若直接让 TensorBoard 合并整个目录，会在相同 step 上叠加多次训练，曲线视觉上会混淆。

“纯图像检索 loss”按每个 step 从已记录 retrieval objective 中扣除 `alignment_weight × text_alignment_loss` 后再按 epoch 平均。No-text 虽然日志里曾记录非零权重，但 `USE_TEXT_INPUT=false` 导致文本 loss 恒为零，因此有效文本约束为零。
"""

    reported_body = f"""## 报告总 loss 的差距主要由持续加入的文本项造成

固定权重文本 run 在第 20 epoch 为 {constant_final['reported_train_loss']:.4f}，no-text 为 {no_text_final['reported_train_loss']:.4f}。这 {reported_gap:.4f} 的差距不表示图像分支更差：日志字段 `Loss/image_retrieval_step` 已经包含文本对齐项，名称容易造成误读。旧调度 run 在第 5 epoch 后关闭文本项，所以后半程会自然回到 no-text 附近。
"""

    pure_body = f"""## 扣除文本项后，图像检索轨迹几乎重合

固定文本与 no-text 的 20-epoch 纯图像检索 loss 平均绝对差为 {mean([abs(row['pure_image_retrieval_loss'] - next(item['pure_image_retrieval_loss'] for item in epoch_rows if item['run_id'] == 'no_text' and item['epoch'] == row['epoch'])) for row in epoch_rows if row['run_id'] == 'text_constant']):.4f}。第 20 epoch 分别为 {constant_final['pure_image_retrieval_loss']:.4f} 与 {no_text_final['pure_image_retrieval_loss']:.4f}。这说明 0.01 文本约束在训练集上既没有带来明显的图像检索优化，也没有造成明显负迁移；可见差异处在单次训练波动量级，尚不能解释泛化表现。
"""

    alignment_body = f"""## 文本对齐持续改善，但下降幅度仍温和

固定权重 run 的文本–卫星交叉熵在 20 个 epoch 内下降 {abs(alignment_drop):.1%}，最终为 {constant_final['text_alignment_loss']:.3f}，低于 480 个候选均匀预测的参考值 `log(480)={uniform_alignment_loss:.3f}`。曲线末段仍缓慢下降，未出现发散；但最终值仍明显高于理想分离状态，所以“学到可用对齐信号”需要用当前 checkpoint 的检索指标验证，不能只看训练 CE。
"""

    component_body = f"""## Heatmap 出现稳定小幅改善，bbox 总项基本持平

固定文本 run 的 heatmap loss 在 20 个 epoch 中有 {heatmap_better_epochs} 个低于 no-text，后 10 个 epoch 全部更低，后 10 epoch 平均差为 {heatmap_last_ten_delta:.4f}。bbox 则在最后 5 epoch 平均只差 {bbox_last_five_delta:.4f}，基本持平。这是曲线中最一致的正向信号，但两次训练没有固定或记录随机种子，仍不能把小幅改善视为确定的文本收益。下表用于精确核对末 epoch，不作为因果估计。
"""

    historical_body = f"""## 历史测试偏向 no-text，但它不是当前固定权重模型的结果

现有 `baseline_ada_1_5x3.json` 在第一次训练结束后、第二次固定权重训练开始前生成。历史旧调度模型相对 no-text 的 R@1 低 {abs(paired['r1_delta']):.2%}，mIoU 低 {abs(paired['iou_delta']):.4f}；按 2,804 个 `gt_label` 聚类的近似 95% 区间分别为 [{paired['r1_ci_low']:.2%}, {paired['r1_ci_high']:.2%}] 和 [{paired['iou_ci_low']:.4f}, {paired['iou_ci_high']:.4f}]。不过旧 checkpoint 已被第二次训练覆盖，JSON 又没有保存 checkpoint hash，因此这些数字只能说明旧调度 run 的历史表现，不能用于评价当前固定 0.01 的模型。
"""

    method_body = """## 实验与计算方法

- 两个实验的模型、batch size、epoch 数、学习率、检索/定位权重和主要数据配置一致。
- 当前文本实验使用 `train.py`，全程输入文本，并对 32×15=480 个候选区域计算跨 batch 文本–卫星 InfoNCE；no-text 实验使用 `train_ada.py`，文本分支因 `USE_TEXT_INPUT=false` 实际不参与梯度。
- 每个 epoch 的纯图像检索 loss 由全部 701 个 step 逐点扣除文本项后求均值，避免使用“均值的乘积”近似。
- 历史评测的区间按 2,804 个 gallery/ground-truth label 聚类计算，用于避免把同一卫星图的 32 个视角完全当作独立样本。
"""

    limitation_body = """## 限制与稳健性：当前证据不足以隔离文本输入的因果效应

1. `baseline_ada_1_5x3` 目录复用了两次，no-text 目录复用了四次；TensorBoard 目录级曲线会混合相同步数。
2. 当前固定权重 checkpoint 尚未测试；历史测试对应已被覆盖的旧 checkpoint。
3. 训练代码没有固定或记录随机种子，且两个实验入口分别是 `train.py` 与 `train_ada.py`；单次 run 的小差异可能来自初始化、数据顺序或代码路径。
4. `Loss/image_retrieval_step` 的命名与实际口径不符，因为文本开启时它是“图像检索 + 文本对齐”的合计。
5. 训练 loss 只能描述拟合与优化稳定性，不能替代 R@K、mIoU、uIoU 等测试指标。
"""

    next_body = """## 建议的下一步

1. 立即用当前 `baseline_ada_1_5x3/last.pth` 重新运行 `test.py`，并把结果写到不会与旧 JSON 混淆的新名字。
2. 给每次实验使用独立 run/output 目录，并在结果 JSON 中保存 checkpoint SHA256、effective config 和代码 commit/hash。
3. 做严格 matched ablation：同一 `train.py`、同一 seed，唯一差异设为 `USE_TEXT_INPUT`/文本权重；至少 3 个 seed，报告均值与标准差。
4. 将日志拆成 `Loss/image_retrieval_pure_step` 与 `Loss/retrieval_with_text_step`，避免以后把加法项误判为收敛退化。
5. 若当前固定模型仍不提升测试检索，优先扫描更小权重或只在早期启用，并同时观察文本 CE 与 R@1，而不是按总 train loss 选择模型。
"""

    questions_body = """## 仍需回答的问题

- 当前固定 0.01 checkpoint 的 R@1、mIoU、uIoU 是否优于 no-text？
- 文本对齐改善是否集中在特定高度、角度或大学场景？
- 文本监督改善的是区域排序，还是只改变候选特征尺度/校准？
- 在固定 seed 的情况下，纯图像检索 loss 的约 0.01 波动是否属于正常 run-to-run 方差？
"""

    manifest = {
        "version": 1,
        "surface": "report",
        "title": REPORT_TITLE,
        "description": "TensorBoard training-curve comparison with loss decomposition and historical evaluation caveats.",
        "generatedAt": generated_at,
        "sources": sources,
        "cards": [],
        "charts": [
            {
                "id": "reported-train-loss-chart",
                "title": "报告训练总 loss",
                "subtitle": "20 epochs；固定文本 run 的 retrieval objective 持续包含文本对齐项",
                "showDescription": True,
                "intent": "trend",
                "question": "三条训练总 loss 曲线为何在后半程分离？",
                "rationale": "20 个有序 epoch 足以显示收敛形状与旧调度关闭文本项后的拐点。",
                "comparisonContext": {
                    "baseline": "No text",
                    "grain": "epoch",
                    "unit": "loss",
                },
                "type": "line",
                "dataset": "epoch_curves",
                "sourceId": "curve-analysis",
                "encodings": {
                    "x": {
                        "field": "epoch",
                        "type": "ordinal",
                        "label": "Epoch",
                    },
                    "y": {
                        "field": "reported_train_loss",
                        "type": "quantitative",
                        "format": "number",
                        "label": "Reported train loss",
                    },
                    "color": {
                        "field": "run",
                        "type": "nominal",
                        "label": "Run",
                    },
                    "lineStyle": {
                        "field": "line_style",
                        "type": "nominal",
                    },
                    "tooltip": [
                        {"field": "run", "type": "text", "label": "Run"},
                        {"field": "epoch", "type": "ordinal", "label": "Epoch"},
                        {
                            "field": "reported_train_loss",
                            "type": "quantitative",
                            "format": "number",
                            "label": "Reported train loss",
                        },
                        {
                            "field": "mean_logged_alignment_weight",
                            "type": "quantitative",
                            "format": "number",
                            "label": "Logged alignment weight",
                        },
                    ],
                },
                "xAxisTitle": "Epoch",
                "yAxisTitle": "Loss",
                "layout": "full",
                "palette": {"kind": "categorical", "name": "training-runs"},
                "legend": {"position": "bottom", "sort": "spec"},
                "maxRows": 60,
                "surface": {"surface": "explorer", "viewMode": "visualization"},
            },
            {
                "id": "pure-image-loss-chart",
                "title": "扣除文本项后的纯图像检索 loss",
                "subtitle": "逐 step 扣除 alignment_weight × text_alignment_loss 后按 epoch 平均",
                "showDescription": True,
                "intent": "trend",
                "question": "文本约束是否改变图像检索分支自身的收敛？",
                "rationale": "损失分解消除了报告口径差异，使三次训练可直接比较。",
                "comparisonContext": {
                    "baseline": "No text",
                    "grain": "epoch",
                    "unit": "loss",
                },
                "type": "line",
                "dataset": "epoch_curves",
                "sourceId": "curve-analysis",
                "encodings": {
                    "x": {
                        "field": "epoch",
                        "type": "ordinal",
                        "label": "Epoch",
                    },
                    "y": {
                        "field": "pure_image_retrieval_loss",
                        "type": "quantitative",
                        "format": "number",
                        "label": "Pure image retrieval loss",
                    },
                    "color": {
                        "field": "run",
                        "type": "nominal",
                        "label": "Run",
                    },
                    "lineStyle": {
                        "field": "line_style",
                        "type": "nominal",
                    },
                    "tooltip": [
                        {"field": "run", "type": "text", "label": "Run"},
                        {"field": "epoch", "type": "ordinal", "label": "Epoch"},
                        {
                            "field": "pure_image_retrieval_loss",
                            "type": "quantitative",
                            "format": "number",
                            "label": "Pure image retrieval loss",
                        },
                        {
                            "field": "bbox_loss",
                            "type": "quantitative",
                            "format": "number",
                            "label": "BBox loss",
                        },
                    ],
                },
                "xAxisTitle": "Epoch",
                "yAxisTitle": "Loss",
                "layout": "full",
                "palette": {"kind": "categorical", "name": "training-runs"},
                "legend": {"position": "bottom", "sort": "spec"},
                "maxRows": 60,
                "surface": {"surface": "explorer", "viewMode": "visualization"},
            },
            {
                "id": "alignment-loss-chart",
                "title": "固定权重 run 的文本–卫星对齐交叉熵",
                "subtitle": "20 epochs；虚线参考为 480 个候选的均匀预测 loss",
                "showDescription": True,
                "intent": "trend",
                "question": "全程固定 0.01 时，文本对齐是否持续学习并保持稳定？",
                "rationale": "20 个 epoch 显示持续下降与相对均匀预测参考线的位置。",
                "comparisonContext": {
                    "baseline": "Uniform prediction log(480)",
                    "grain": "epoch",
                    "unit": "cross-entropy",
                },
                "type": "line",
                "dataset": "alignment_curve",
                "sourceId": "alignment-analysis",
                "encodings": {
                    "x": {
                        "field": "epoch",
                        "type": "ordinal",
                        "label": "Epoch",
                    },
                    "y": {
                        "field": "text_alignment_loss",
                        "type": "quantitative",
                        "format": "number",
                        "label": "Text alignment CE",
                    },
                    "tooltip": [
                        {"field": "epoch", "type": "ordinal", "label": "Epoch"},
                        {
                            "field": "text_alignment_loss",
                            "type": "quantitative",
                            "format": "number",
                            "label": "Text alignment CE",
                        },
                    ],
                },
                "xAxisTitle": "Epoch",
                "yAxisTitle": "Cross-entropy",
                "layout": "full",
                "palette": {"kind": "sequential", "name": "blue"},
                "referenceLines": [
                    {
                        "axis": "y",
                        "value": uniform_alignment_loss,
                        "label": "Uniform log(480)",
                        "color": "neutral",
                        "lineStyle": "dashed",
                    }
                ],
                "maxRows": 20,
                "surface": {"surface": "explorer", "viewMode": "visualization"},
            },
        ],
        "tables": [
            {
                "id": "final-component-table",
                "title": "第 20 epoch 训练项",
                "subtitle": "三次完整训练；纯图像项已扣除文本对齐贡献",
                "showDescription": True,
                "dataset": "final_epoch_summary",
                "defaultSort": {"field": "reported_train_loss", "direction": "asc"},
                "density": "spacious",
                "sourceId": "final-components",
                "layout": "full",
                "columns": [
                    {"field": "run", "label": "Run", "type": "text"},
                    {
                        "field": "reported_train_loss",
                        "label": "Reported train",
                        "format": "number",
                    },
                    {
                        "field": "pure_image_retrieval_loss",
                        "label": "Pure image retrieval",
                        "format": "number",
                    },
                    {"field": "bbox_loss", "label": "BBox", "format": "number"},
                    {
                        "field": "heatmap_loss",
                        "label": "Heatmap",
                        "format": "number",
                    },
                    {
                        "field": "text_alignment_loss",
                        "label": "Text alignment",
                        "format": "number",
                    },
                    {
                        "field": "effective_alignment_weight",
                        "label": "Alignment weight",
                        "format": "number",
                    },
                ],
            },
            {
                "id": "historical-eval-table",
                "title": "历史测试结果",
                "subtitle": "89,728 queries；文本结果对应旧调度 run，不对应当前固定权重 checkpoint",
                "showDescription": True,
                "dataset": "historical_eval_summary",
                "defaultSort": {"field": "recall_at_1", "direction": "desc"},
                "density": "spacious",
                "sourceId": "historical-eval",
                "layout": "full",
                "columns": [
                    {"field": "run", "label": "Run", "type": "text"},
                    {
                        "field": "recall_at_1",
                        "label": "R@1",
                        "format": "percent",
                    },
                    {
                        "field": "recall_at_5",
                        "label": "R@5",
                        "format": "percent",
                    },
                    {
                        "field": "recall_at_10",
                        "label": "R@10",
                        "format": "percent",
                    },
                    {"field": "mean_iou", "label": "mIoU", "format": "number"},
                    {"field": "uiou", "label": "uIoU", "format": "number"},
                    {
                        "field": "mean_center_distance",
                        "label": "Center distance (px)",
                        "format": "number",
                    },
                ],
            },
        ],
        "blocks": [
            {"id": "title", "type": "markdown", "body": f"# {REPORT_TITLE}", "layout": "full"},
            {"id": "technical-summary", "type": "markdown", "body": summary_body, "layout": "full"},
            {"id": "scope", "type": "markdown", "body": scope_body, "layout": "full", "sourceId": "curve-analysis"},
            {"id": "reported-loss-finding", "type": "markdown", "body": reported_body, "layout": "full", "sourceId": "curve-analysis"},
            {"id": "reported-loss-visual", "type": "chart", "chartId": "reported-train-loss-chart", "layout": "full"},
            {"id": "pure-loss-finding", "type": "markdown", "body": pure_body, "layout": "full", "sourceId": "curve-analysis"},
            {"id": "pure-loss-visual", "type": "chart", "chartId": "pure-image-loss-chart", "layout": "full"},
            {"id": "alignment-finding", "type": "markdown", "body": alignment_body, "layout": "full", "sourceId": "alignment-analysis"},
            {"id": "alignment-visual", "type": "chart", "chartId": "alignment-loss-chart", "layout": "full"},
            {"id": "component-finding", "type": "markdown", "body": component_body, "layout": "full", "sourceId": "curve-analysis"},
            {"id": "component-table", "type": "table", "tableId": "final-component-table", "layout": "full"},
            {"id": "historical-eval-finding", "type": "markdown", "body": historical_body, "layout": "full", "sourceId": "historical-eval"},
            {"id": "historical-eval-table-block", "type": "table", "tableId": "historical-eval-table", "layout": "full"},
            {"id": "method", "type": "markdown", "body": method_body, "layout": "full"},
            {"id": "limitations", "type": "markdown", "body": limitation_body, "layout": "full"},
            {"id": "next-steps", "type": "markdown", "body": next_body, "layout": "full"},
            {"id": "questions", "type": "markdown", "body": questions_body, "layout": "full"},
        ],
    }

    artifact = {
        "surface": "report",
        "manifest": manifest,
        "snapshot": {
            "version": 1,
            "generatedAt": generated_at,
            "status": "ready",
            "datasets": reviewed_datasets,
        },
        "sources": sources,
    }
    return artifact


def write_notes(artifact: dict) -> None:
    notes = """# Analysis notes

## Report structure mapping

- Title: `title`
- Technical summary: `technical-summary`
- Key findings with visual evidence: reported loss, pure image loss, alignment loss, component table, historical evaluation
- Scope/data/definitions: `scope`
- Methodology and experiment specification: `method`
- Limitations/robustness: `limitations`
- Recommended next steps: `next-steps`
- Further questions: `questions`

## Chart map

| Section | Question | Family/type | Fields | Supported claim |
|---|---|---|---|---|
| Reported total loss | Why does the latest text run remain higher? | Trend / line | epoch, reported_train_loss, run | The gap is expected while the text term stays active. |
| Pure image retrieval | Did the image branch converge differently? | Trend / line | epoch, pure_image_retrieval_loss, run | Curves nearly overlap after loss decomposition. |
| Text alignment | Does fixed text supervision learn stably? | Trend / line + reference | epoch, text_alignment_loss | Alignment CE falls steadily below the uniform baseline. |

Repeated line charts are intentional because all three questions concern continuous 20-epoch optimization paths; the second is a decomposition validation of the first, and the third is a distinct auxiliary objective. Exact final values and historical evaluation use tables rather than additional charts.

## Source and QA notes

- The latest complete event is selected for the current `baseline_ada_1_5x3` checkpoint.
- The earlier complete event is retained and explicitly labelled historical because its evaluation JSON exists.
- The 88-byte empty event is omitted.
- The latest of four complete no-text events is selected because it matches the current checkpoint timestamp; three earlier reruns are omitted.
- Latest fixed-weight model evaluation is omitted because no matching result exists.
- Historical clustered confidence intervals are descriptive robustness checks; no causal claim is made.

### Raw source inventory

- Current fixed text event: `exp/runs/baseline_ada_1_5x3/events.out.tfevents.1786420717.4090-48g.511988.0`
- Historical scheduled text event: `exp/runs/baseline_ada_1_5x3/events.out.tfevents.1786381215.4090-48g.4013619.0`
- Current no-text event: `exp/runs/baseline_wo_input_ids_ada_5x3/events.out.tfevents.1786302112.4090-48g.2760507.0`
- Historical text evaluation: `exp/eval_results/baseline_ada_1_5x3.json`
- No-text evaluation: `exp/eval_results/baseline_wo_input_ids_ada_5x3.json`
- Effective configs: `exp/outputs/baseline_ada_1_5x3/effective_config.json`, `exp/outputs/baseline_wo_input_ids_ada_5x3/effective_config.json`
"""
    with open(OUTPUT_DIR / "analysis_notes.md", "w", encoding="utf-8") as file:
        file.write(notes)


def main() -> None:
    artifact = build_artifact()
    with open(OUTPUT_DIR / "artifact.json", "w", encoding="utf-8") as file:
        json.dump(artifact, file, ensure_ascii=False, indent=2)
    write_notes(artifact)
    print(OUTPUT_DIR / "artifact.json")


if __name__ == "__main__":
    main()
