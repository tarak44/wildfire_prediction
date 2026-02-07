from __future__ import annotations

import argparse
import logging
from pathlib import Path

import yaml

from wildfire_mlops.core import setup_logging
from wildfire_mlops.training import TrainConfig, train_model


def _load_config(path: Path) -> TrainConfig:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    return TrainConfig(
        train_dir=Path(data["data"]["train_dir"]),
        val_dir=Path(data["data"]["val_dir"]),
        test_dir=Path(data["data"].get("test_dir")) if data["data"].get("test_dir") else None,
        image_size=int(data["data"]["image_size"]),
        batch_size=int(data["train"]["batch_size"]),
        num_workers=int(data["train"]["num_workers"]),
        epochs=int(data["train"]["epochs"]),
        lr=float(data["train"]["lr"]),
        weight_decay=float(data["train"]["weight_decay"]),
        device=str(data["train"]["device"]),
        output_dir=Path(data["train"]["output_dir"]),
        seed=int(data["train"]["seed"]),
        max_train_samples=int(data["train"].get("max_train_samples") or 0) or None,
        max_val_samples=int(data["train"].get("max_val_samples") or 0) or None,
        mlflow_tracking_uri=str(data["train"].get("mlflow_tracking_uri") or "./mlruns"),
        mlflow_experiment=str(data["train"].get("mlflow_experiment") or "wildfire"),
        mlflow_run_name=str(data["train"].get("mlflow_run_name") or "run"),
        model_arch=str(data["train"].get("model_arch") or "custom_cnn"),
        pretrained=bool(
            data["train"].get("pretrained")
            if data["train"].get("pretrained") is not None
            else True
        ),
        register_model=bool(
            data["train"].get("register_model")
            if data["train"].get("register_model") is not None
            else True
        ),
        model_registry_name=str(data["train"].get("model_registry_name") or "wildfire-classifier"),
        max_stat_batches=int(data["train"].get("max_stat_batches") or 10),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Wildfire model")
    parser.add_argument("--config", default="params.yaml", help="Path to params yaml")
    args = parser.parse_args()

    setup_logging("INFO")
    logging.getLogger("PIL").setLevel(logging.WARNING)

    cfg = _load_config(Path(args.config))
    results = train_model(cfg)
    print(f"best_val_accuracy={results['best_val_accuracy']:.4f}")


if __name__ == "__main__":
    main()
