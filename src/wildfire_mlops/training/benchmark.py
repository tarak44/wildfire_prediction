from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class BenchmarkRecord:
    model_name: str
    metrics_path: Path
    accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    roc_auc: float | None


def _extract_metric(metrics: dict, *keys: str, default: float | None = None) -> float | None:
    current = metrics
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return float(current)


def load_benchmark_record(model_name: str, metrics_path: Path) -> BenchmarkRecord:
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    latest_metrics = payload.get("test") or payload["history"][-1]["val"]

    if "macro_precision" in latest_metrics:
        macro_precision = float(latest_metrics["macro_precision"])
        macro_recall = float(latest_metrics["macro_recall"])
        macro_f1 = float(latest_metrics["macro_f1"])
    elif "macro_avg" in latest_metrics:
        macro_precision = float(latest_metrics["macro_avg"]["precision"])
        macro_recall = float(latest_metrics["macro_avg"]["recall"])
        macro_f1 = float(latest_metrics["macro_avg"]["f1"])
    else:
        precision_values = list(latest_metrics["precision"].values())
        recall_values = list(latest_metrics["recall"].values())
        f1_values = list(latest_metrics["f1"].values())
        macro_precision = sum(precision_values) / len(precision_values)
        macro_recall = sum(recall_values) / len(recall_values)
        macro_f1 = sum(f1_values) / len(f1_values)

    return BenchmarkRecord(
        model_name=model_name,
        metrics_path=metrics_path,
        accuracy=float(latest_metrics["accuracy"]),
        macro_precision=macro_precision,
        macro_recall=macro_recall,
        macro_f1=macro_f1,
        roc_auc=_extract_metric(latest_metrics, "roc_auc"),
    )


def render_markdown_table(records: list[BenchmarkRecord]) -> str:
    lines = [
        "| Model | Accuracy | Macro Precision | Macro Recall | Macro F1 | ROC-AUC |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for record in records:
        roc_auc = f"{record.roc_auc:.4f}" if record.roc_auc is not None else "N/A"
        lines.append(
            f"| {record.model_name} | {record.accuracy:.4f} | {record.macro_precision:.4f} | "
            f"{record.macro_recall:.4f} | {record.macro_f1:.4f} | {roc_auc} |"
        )
    return "\n".join(lines)


def save_benchmark_table(records: list[BenchmarkRecord], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_markdown_table(records), encoding="utf-8")
