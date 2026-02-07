from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from PIL import Image, ImageFile

from wildfire_mlops.constants import SUPPORTED_IMAGE_EXTS
from wildfire_mlops.inference import predict_image

# Handle occasional truncated images in real-world datasets
ImageFile.LOAD_TRUNCATED_IMAGES = True

logger = logging.getLogger(__name__)


def _iter_labeled_images(root_dir: Path, class_names: List[str]) -> List[Tuple[Path, int]]:
    exts = SUPPORTED_IMAGE_EXTS
    items: List[Tuple[Path, int]] = []
    for idx, name in enumerate(class_names):
        class_dir = root_dir / name
        if not class_dir.exists():
            logger.warning("Missing class directory: %s", class_dir)
            continue
        for path in class_dir.rglob("*"):
            if path.is_file() and path.suffix.lower() in exts:
                items.append((path, idx))
    return items


def _confusion_matrix(n: int) -> List[List[int]]:
    return [[0 for _ in range(n)] for _ in range(n)]


def _safe_div(numer: float, denom: float) -> float:
    return float(numer / denom) if denom else 0.0


def _limit_items(
    items: List[Tuple[Path, int]], max_samples: int | None, seed: int
) -> List[Tuple[Path, int]]:
    if max_samples is None or max_samples <= 0 or len(items) <= max_samples:
        return items
    g = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(items), generator=g)[:max_samples].tolist()
    return [items[i] for i in indices]


def evaluate_dataset(
    root_dir: Path,
    model,
    class_names: List[str],
    device: str = "cpu",
    image_size: int = 224,
    max_samples: int | None = None,
    seed: int = 42,
) -> Dict[str, object]:
    labeled = _iter_labeled_images(root_dir, class_names)
    labeled = _limit_items(labeled, max_samples, seed)
    if not labeled:
        raise ValueError("No labeled images found under dataset root")

    n_classes = len(class_names)
    cm = _confusion_matrix(n_classes)

    for img_path, true_idx in labeled:
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            logger.exception("Failed to open image: %s", img_path)
            continue

        pred = predict_image(
            image=image,
            model=model,
            class_names=class_names,
            device=device,
            image_size=image_size,
        )
        pred_idx = class_names.index(pred.class_name)
        cm[true_idx][pred_idx] += 1

    total = sum(sum(row) for row in cm)
    correct = sum(cm[i][i] for i in range(n_classes))
    accuracy = _safe_div(correct, total)

    per_class = {}
    precisions = []
    recalls = []
    f1s = []

    for i, name in enumerate(class_names):
        tp = cm[i][i]
        fp = sum(cm[r][i] for r in range(n_classes)) - tp
        fn = sum(cm[i][c] for c in range(n_classes)) - tp

        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        f1 = _safe_div(2 * precision * recall, precision + recall)

        per_class[name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": sum(cm[i]),
        }

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    macro = {
        "precision": _safe_div(sum(precisions), n_classes),
        "recall": _safe_div(sum(recalls), n_classes),
        "f1": _safe_div(sum(f1s), n_classes),
    }

    return {
        "accuracy": accuracy,
        "macro_avg": macro,
        "per_class": per_class,
        "confusion_matrix": cm,
        "num_samples": total,
    }


def save_metrics(metrics: Dict[str, object], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
