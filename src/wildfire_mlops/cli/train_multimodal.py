from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import yaml

from wildfire_mlops.core import setup_logging
from wildfire_mlops.training import MultimodalTrainConfig, train_multimodal_model


def _load_config(path: Path) -> MultimodalTrainConfig:
    with path.open("r", encoding="utf-8") as file_handle:
        data = yaml.safe_load(file_handle)

    multimodal = data["multimodal"]
    train = data["train"]
    return MultimodalTrainConfig(
        train_manifest=Path(multimodal["train_manifest"]),
        val_manifest=Path(multimodal["val_manifest"]),
        test_manifest=(
            Path(multimodal["test_manifest"]) if multimodal.get("test_manifest") else None
        ),
        image_size=int(multimodal.get("image_size") or 224),
        batch_size=int(train["batch_size"]),
        num_workers=int(train["num_workers"]),
        epochs=int(train["epochs"]),
        lr=float(train["lr"]),
        weight_decay=float(train["weight_decay"]),
        device=str(train["device"]),
        output_dir=Path(train["output_dir"]),
        seed=int(train["seed"]),
        max_train_samples=int(train.get("max_train_samples") or 0) or None,
        max_val_samples=int(train.get("max_val_samples") or 0) or None,
        mlflow_tracking_uri=str(
            os.environ.get("WILDFIRE_MLFLOW_TRACKING_URI")
            or train.get("mlflow_tracking_uri")
            or "sqlite:///mlflow.db"
        ),
        mlflow_experiment=str(train.get("mlflow_experiment") or "wildfire-multimodal"),
        mlflow_run_name=str(train.get("mlflow_run_name") or "multimodal-run"),
        model_arch=str(train.get("model_arch") or "multimodal_efficientnet_b0"),
        pretrained=bool(train.get("pretrained") if train.get("pretrained") is not None else True),
        register_model=bool(
            train.get("register_model") if train.get("register_model") is not None else True
        ),
        model_registry_name=str(train.get("model_registry_name") or "wildfire-multimodal"),
        tabular_feature_names=list(multimodal["tabular_feature_names"]),
        temporal_feature_names=list(multimodal.get("temporal_feature_names") or []),
        temporal_sequence_column=str(
            multimodal.get("temporal_sequence_column") or "temporal_sequence"
        ),
        temporal_encoder_arch=str(multimodal.get("temporal_encoder_arch") or "lstm"),
        temporal_hidden_dim=int(multimodal.get("temporal_hidden_dim") or 64),
        temporal_max_sequence_length=int(multimodal.get("temporal_max_sequence_length") or 24),
        image_column=str(multimodal.get("image_column") or "image_path"),
        label_column=str(multimodal.get("label_column") or "label"),
        auxiliary_context_loss_weight=float(train.get("auxiliary_context_loss_weight") or 0.2),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train multimodal wildfire model")
    parser.add_argument("--config", default="configs/experiments/multimodal.yaml")
    args = parser.parse_args()

    setup_logging("INFO")
    logging.getLogger("PIL").setLevel(logging.WARNING)

    cfg = _load_config(Path(args.config))
    results = train_multimodal_model(cfg)
    print(f"best_val_macro_f1={results['best_val_macro_f1']:.4f}")


if __name__ == "__main__":
    main()
