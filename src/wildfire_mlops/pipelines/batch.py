from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Iterable

from PIL import Image

from wildfire_mlops.constants import SUPPORTED_IMAGE_EXTS
from wildfire_mlops.inference import Prediction, predict_image

logger = logging.getLogger(__name__)


def _iter_images(input_dir: Path) -> Iterable[Path]:
    exts = SUPPORTED_IMAGE_EXTS
    for path in input_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in exts:
            yield path


def run_batch_inference(
    input_dir: Path,
    output_csv: Path,
    model,
    class_names: list[str],
    device: str = "cpu",
    image_size: int = 224,
) -> int:
    rows: list[dict[str, str]] = []
    count = 0

    for img_path in _iter_images(input_dir):
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            logger.exception("Failed to open image: %s", img_path)
            continue

        pred: Prediction = predict_image(
            image=image,
            model=model,
            class_names=class_names,
            device=device,
            image_size=image_size,
        )

        row = {
            "path": str(img_path),
            "class": pred.class_name,
            "confidence": f"{pred.confidence:.6f}",
        }
        for k, v in pred.probabilities.items():
            row[f"prob_{k}"] = f"{v:.6f}"
        rows.append(row)
        count += 1

    if rows:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        with output_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

    return count
