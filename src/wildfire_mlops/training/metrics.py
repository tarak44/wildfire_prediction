from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import torch


@dataclass
class Metrics:
    accuracy: float
    precision: Dict[str, float]
    recall: Dict[str, float]
    f1: Dict[str, float]
    confusion_matrix: List[List[int]]


def _safe_div(numer: float, denom: float) -> float:
    return float(numer / denom) if denom else 0.0


def compute_metrics(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    class_names: list[str],
) -> Metrics:
    n = len(class_names)
    cm = [[0 for _ in range(n)] for _ in range(n)]

    for t, p in zip(y_true.tolist(), y_pred.tolist()):
        cm[t][p] += 1

    total = sum(sum(row) for row in cm)
    correct = sum(cm[i][i] for i in range(n))
    accuracy = _safe_div(correct, total)

    precision: Dict[str, float] = {}
    recall: Dict[str, float] = {}
    f1: Dict[str, float] = {}

    for i, name in enumerate(class_names):
        tp = cm[i][i]
        fp = sum(cm[r][i] for r in range(n)) - tp
        fn = sum(cm[i][c] for c in range(n)) - tp

        p = _safe_div(tp, tp + fp)
        r = _safe_div(tp, tp + fn)
        f = _safe_div(2 * p * r, p + r)

        precision[name] = p
        recall[name] = r
        f1[name] = f

    return Metrics(
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        confusion_matrix=cm,
    )
