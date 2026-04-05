from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import torch


@dataclass
class Metrics:
    accuracy: float
    balanced_accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    roc_auc: float | None
    precision: Dict[str, float]
    recall: Dict[str, float]
    f1: Dict[str, float]
    confusion_matrix: List[List[int]]


def _safe_div(numer: float, denom: float) -> float:
    return float(numer / denom) if denom else 0.0


def _binary_roc_auc(y_true: torch.Tensor, y_score: torch.Tensor) -> float | None:
    positives = y_score[y_true == 1]
    negatives = y_score[y_true == 0]
    if positives.numel() == 0 or negatives.numel() == 0:
        return None

    wins = 0.0
    ties = 0.0
    for positive_score in positives.tolist():
        for negative_score in negatives.tolist():
            if positive_score > negative_score:
                wins += 1.0
            elif positive_score == negative_score:
                ties += 1.0

    total_pairs = float(positives.numel() * negatives.numel())
    return (wins + (0.5 * ties)) / total_pairs


def compute_metrics(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    class_names: list[str],
    y_score: torch.Tensor | None = None,
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

    recalls = list(recall.values())
    precisions = list(precision.values())
    f1_scores = list(f1.values())

    return Metrics(
        accuracy=accuracy,
        balanced_accuracy=_safe_div(sum(recalls), len(recalls)),
        macro_precision=_safe_div(sum(precisions), len(precisions)),
        macro_recall=_safe_div(sum(recalls), len(recalls)),
        macro_f1=_safe_div(sum(f1_scores), len(f1_scores)),
        roc_auc=_binary_roc_auc(y_true, y_score) if y_score is not None and n == 2 else None,
        precision=precision,
        recall=recall,
        f1=f1,
        confusion_matrix=cm,
    )
