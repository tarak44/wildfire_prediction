from __future__ import annotations

import argparse
import base64
import csv
import json
from io import BytesIO
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image

from wildfire_mlops.constants import SUPPORTED_IMAGE_EXTS
from wildfire_mlops.core import get_settings, setup_logging
from wildfire_mlops.inference import EnvironmentalFeatures, predict_image
from wildfire_mlops.modeling import load_checkpoint


def _safe_div(numer: float, denom: float) -> float:
    return numer / denom if denom else 0.0


def _binary_roc_points(
    y_true: list[int], y_score: list[float]
) -> tuple[list[float], list[float], float | None]:
    positives = sum(y_true)
    negatives = len(y_true) - positives
    if positives == 0 or negatives == 0:
        return [0.0, 1.0], [0.0, 1.0], None

    thresholds = sorted(set(y_score), reverse=True)
    thresholds = [float("inf")] + thresholds + [float("-inf")]
    tprs: list[float] = []
    fprs: list[float] = []

    for threshold in thresholds:
        tp = fp = tn = fn = 0
        for label, score in zip(y_true, y_score):
            predicted = 1 if score >= threshold else 0
            if label == 1 and predicted == 1:
                tp += 1
            elif label == 1:
                fn += 1
            elif predicted == 1:
                fp += 1
            else:
                tn += 1
        tprs.append(_safe_div(tp, tp + fn))
        fprs.append(_safe_div(fp, fp + tn))

    auc = 0.0
    for index in range(1, len(fprs)):
        auc += (fprs[index] - fprs[index - 1]) * (tprs[index] + tprs[index - 1]) * 0.5

    ordered = sorted(zip(fprs, tprs), key=lambda item: item[0])
    ordered_fprs = [point[0] for point in ordered]
    ordered_tprs = [point[1] for point in ordered]
    return ordered_fprs, ordered_tprs, auc


def _plot_roc_curve(y_true: list[int], y_score: list[float], output_path: Path) -> None:
    fprs, tprs, auc = _binary_roc_points(y_true, y_score)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fprs, tprs, label=f"ROC AUC = {auc:.4f}" if auc is not None else "ROC")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Wildfire ROC Curve")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _plot_confusion_matrix(
    confusion_matrix: list[list[int]], class_names: list[str], output_path: Path
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(confusion_matrix, cmap="OrRd")
    ax.set_xticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=20, ha="right")
    ax.set_yticks(range(len(class_names)))
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Wildfire Confusion Matrix")

    for row_index, row in enumerate(confusion_matrix):
        for col_index, value in enumerate(row):
            ax.text(col_index, row_index, str(value), ha="center", va="center", color="black")

    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _decode_gradcam_overlay(overlay_base64: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.open(BytesIO(base64.b64decode(overlay_base64))).convert("RGB")
    image.save(output_path)


def _parse_temporal_sequence(raw_value: str | None):
    if not raw_value:
        return None
    return json.loads(raw_value)


def _predict_from_manifest(
    manifest_path: Path,
    model,
    class_names: list[str],
    settings,
    gradcam_image_output: Path | None = None,
) -> tuple[list[int], list[int], list[float], list[list[int]]]:
    y_true: list[int] = []
    y_pred: list[int] = []
    y_score: list[float] = []
    confusion_matrix = [[0 for _ in class_names] for _ in class_names]

    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        first_row = None
        for row in reader:
            if first_row is None:
                first_row = row

            image = Image.open(Path(row["image_path"])).convert("RGB")
            features = EnvironmentalFeatures(
                latitude=float(row["latitude"]) if row.get("latitude") else None,
                longitude=float(row["longitude"]) if row.get("longitude") else None,
                temperature_c=float(row["temperature_c"]),
                humidity_pct=float(row["humidity_pct"]),
                wind_speed_kph=float(row["wind_speed_kph"]),
                drought_index=float(row["drought_index"]),
                vegetation_dryness=float(row["vegetation_dryness"]),
                days_since_rain=float(row["days_since_rain"]),
            )
            prediction = predict_image(
                image=image,
                model=model,
                class_names=class_names,
                device=settings.device,
                image_size=settings.image_size,
                reference_stats_path=settings.reference_stats_path,
                environmental_features=features,
                temporal_sequence=_parse_temporal_sequence(row.get("temporal_sequence")),
                include_explainability=gradcam_image_output is not None and len(y_true) == 0,
            )

            true_index = class_names.index(row["label"])
            pred_index = class_names.index(prediction.class_name)
            y_true.append(true_index)
            y_pred.append(pred_index)
            y_score.append(prediction.image_wildfire_probability)
            confusion_matrix[true_index][pred_index] += 1

            if (
                gradcam_image_output is not None
                and prediction.explainability is not None
                and prediction.explainability.overlay_base64 is not None
                and len(y_true) == 1
            ):
                _decode_gradcam_overlay(
                    prediction.explainability.overlay_base64, gradcam_image_output
                )

    return y_true, y_pred, y_score, confusion_matrix


def _iter_labeled_images(root_dir: Path, class_names: list[str]) -> list[tuple[Path, int]]:
    items: list[tuple[Path, int]] = []
    for class_index, class_name in enumerate(class_names):
        class_dir = root_dir / class_name
        if not class_dir.exists():
            continue
        for image_path in sorted(class_dir.rglob("*")):
            if image_path.is_file() and image_path.suffix.lower() in SUPPORTED_IMAGE_EXTS:
                items.append((image_path, class_index))
    return items


def _predict_from_dir(
    eval_dir: Path,
    model,
    class_names: list[str],
    settings,
    gradcam_image_output: Path | None = None,
) -> tuple[list[int], list[int], list[float], list[list[int]]]:
    y_true: list[int] = []
    y_pred: list[int] = []
    y_score: list[float] = []
    confusion_matrix = [[0 for _ in class_names] for _ in class_names]

    for index, (image_path, true_index) in enumerate(_iter_labeled_images(eval_dir, class_names)):
        image = Image.open(image_path).convert("RGB")
        prediction = predict_image(
            image=image,
            model=model,
            class_names=class_names,
            device=settings.device,
            image_size=settings.image_size,
            reference_stats_path=settings.reference_stats_path,
            include_explainability=gradcam_image_output is not None and index == 0,
        )
        pred_index = class_names.index(prediction.class_name)
        y_true.append(true_index)
        y_pred.append(pred_index)
        y_score.append(prediction.image_wildfire_probability)
        confusion_matrix[true_index][pred_index] += 1

        if (
            gradcam_image_output is not None
            and prediction.explainability is not None
            and prediction.explainability.overlay_base64 is not None
            and index == 0
        ):
            _decode_gradcam_overlay(prediction.explainability.overlay_base64, gradcam_image_output)

    return y_true, y_pred, y_score, confusion_matrix


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate wildfire evaluation visualizations")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--manifest", help="CSV multimodal manifest to evaluate")
    group.add_argument("--eval-dir", help="Folder dataset with class subdirectories")
    parser.add_argument("--model-path", default=None, help="Model checkpoint path")
    parser.add_argument("--model-arch", default=None, help="Model architecture")
    parser.add_argument("--output-dir", default="outputs/figures", help="Directory for plot output")
    args = parser.parse_args()

    settings = get_settings()
    setup_logging(settings.log_level)

    model_arch = args.model_arch or settings.model_arch
    model_path = args.model_path or settings.resolve_model_path(model_arch)
    model, class_names = load_checkpoint(
        model_path, model_arch=model_arch, pretrained=settings.pretrained
    )

    output_dir = Path(args.output_dir)
    roc_path = output_dir / f"{model_arch}_roc_curve.png"
    cm_path = output_dir / f"{model_arch}_confusion_matrix.png"
    gradcam_path = output_dir / f"{model_arch}_gradcam_example.png"

    if args.manifest:
        y_true, _, y_score, confusion_matrix = _predict_from_manifest(
            manifest_path=Path(args.manifest),
            model=model,
            class_names=class_names,
            settings=settings,
            gradcam_image_output=gradcam_path,
        )
    else:
        y_true, _, y_score, confusion_matrix = _predict_from_dir(
            eval_dir=Path(args.eval_dir),
            model=model,
            class_names=class_names,
            settings=settings,
            gradcam_image_output=gradcam_path,
        )

    _plot_roc_curve(y_true=y_true, y_score=y_score, output_path=roc_path)
    _plot_confusion_matrix(
        confusion_matrix=confusion_matrix, class_names=class_names, output_path=cm_path
    )

    print(f"roc_curve={roc_path}")
    print(f"confusion_matrix={cm_path}")
    print(f"gradcam_example={gradcam_path}")


if __name__ == "__main__":
    main()
