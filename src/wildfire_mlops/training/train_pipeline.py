from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import torch
from torch import nn
from torch.optim import Adam

import mlflow

from wildfire_mlops.data import DataConfig, build_dataloaders
from wildfire_mlops.training.metrics import compute_metrics
from wildfire_mlops.modeling import build_model
from wildfire_mlops.monitoring import compute_reference_stats

logger = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    train_dir: Path
    val_dir: Path
    test_dir: Path | None
    image_size: int
    batch_size: int
    num_workers: int
    epochs: int
    lr: float
    weight_decay: float
    device: str
    output_dir: Path
    seed: int
    max_train_samples: int | None
    max_val_samples: int | None
    mlflow_tracking_uri: str
    mlflow_experiment: str
    mlflow_run_name: str
    model_arch: str
    pretrained: bool
    register_model: bool
    model_registry_name: str
    max_stat_batches: int


def _set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _run_epoch(
    model: nn.Module,
    loader,
    criterion,
    optimizer,
    device: str,
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    total = 0
    correct = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item()) * images.size(0)
        _, preds = torch.max(outputs, 1)
        total += labels.size(0)
        correct += int((preds == labels).sum().item())

    return {
        "loss": total_loss / max(total, 1),
        "accuracy": correct / max(total, 1),
    }


def _evaluate(model: nn.Module, loader, device: str, class_names: list[str]) -> Dict[str, object]:
    model.eval()
    all_true = []
    all_pred = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_true.append(labels.cpu())
            all_pred.append(preds.cpu())

    y_true = torch.cat(all_true)
    y_pred = torch.cat(all_pred)
    metrics = compute_metrics(y_true, y_pred, class_names)

    return {
        "accuracy": metrics.accuracy,
        "precision": metrics.precision,
        "recall": metrics.recall,
        "f1": metrics.f1,
        "confusion_matrix": metrics.confusion_matrix,
    }


def train_model(cfg: TrainConfig) -> Dict[str, object]:
    _set_seed(cfg.seed)

    data_cfg = DataConfig(
        train_dir=cfg.train_dir,
        val_dir=cfg.val_dir,
        test_dir=cfg.test_dir,
        image_size=cfg.image_size,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        max_train_samples=cfg.max_train_samples,
        max_val_samples=cfg.max_val_samples,
        seed=cfg.seed,
    )

    train_loader, val_loader, test_loader, class_names = build_dataloaders(data_cfg)

    model = build_model(cfg.model_arch, num_classes=len(class_names), pretrained=cfg.pretrained).to(cfg.device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    best_acc = -1.0
    history: List[Dict[str, object]] = []

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = cfg.output_dir / "model_best.pth"
    latest_path = cfg.output_dir / "model_latest.pth"
    stats_path = cfg.output_dir / "reference_stats.json"

    # Compute reference stats for drift monitoring
    ref_stats = compute_reference_stats(train_loader, max_batches=cfg.max_stat_batches, device=cfg.device)
    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(ref_stats, f, indent=2)

    mlflow.set_tracking_uri(cfg.mlflow_tracking_uri)
    mlflow.set_experiment(cfg.mlflow_experiment)

    with mlflow.start_run(run_name=cfg.mlflow_run_name):
        mlflow.log_params(
            {
                "train_dir": str(cfg.train_dir),
                "val_dir": str(cfg.val_dir),
                "test_dir": str(cfg.test_dir) if cfg.test_dir else "",
                "image_size": cfg.image_size,
                "batch_size": cfg.batch_size,
                "num_workers": cfg.num_workers,
                "epochs": cfg.epochs,
                "lr": cfg.lr,
                "weight_decay": cfg.weight_decay,
                "device": cfg.device,
                "seed": cfg.seed,
                "max_train_samples": cfg.max_train_samples or 0,
                "max_val_samples": cfg.max_val_samples or 0,
                "model_arch": cfg.model_arch,
                "pretrained": cfg.pretrained,
            }
        )

        for epoch in range(1, cfg.epochs + 1):
            train_metrics = _run_epoch(model, train_loader, criterion, optimizer, cfg.device)
            val_metrics = _evaluate(model, val_loader, cfg.device, class_names)

            record = {
                "epoch": epoch,
                "train": train_metrics,
                "val": val_metrics,
            }
            history.append(record)

            val_acc = float(val_metrics["accuracy"])
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "class_names": class_names,
                "epoch": epoch,
                "val_accuracy": val_acc,
            }
            torch.save(checkpoint, latest_path)

            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(checkpoint, checkpoint_path)

            mlflow.log_metric("train_loss", float(train_metrics["loss"]), step=epoch)
            mlflow.log_metric("train_accuracy", float(train_metrics["accuracy"]), step=epoch)
            mlflow.log_metric("val_accuracy", val_acc, step=epoch)

            logger.info("epoch=%s train_acc=%.4f val_acc=%.4f", epoch, train_metrics["accuracy"], val_acc)

        results = {
            "best_val_accuracy": best_acc,
            "history": history,
        }

        if test_loader is not None:
            test_metrics = _evaluate(model, test_loader, cfg.device, class_names)
            results["test"] = test_metrics
            mlflow.log_metric("test_accuracy", float(test_metrics["accuracy"]))

        metrics_path = cfg.output_dir / "metrics.json"
        with metrics_path.open("w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        mlflow.log_artifact(str(metrics_path))
        if stats_path.exists():
            mlflow.log_artifact(str(stats_path))
        if checkpoint_path.exists():
            mlflow.log_artifact(str(checkpoint_path))
        if latest_path.exists():
            mlflow.log_artifact(str(latest_path))

        # Optional: register model in MLflow Model Registry
        if cfg.register_model and checkpoint_path.exists():
            try:
                import mlflow.pytorch as mlflow_pytorch

                model_uri = mlflow_pytorch.log_model(model, artifact_path="model")
                mlflow.register_model(model_uri.model_uri, cfg.model_registry_name)
            except Exception:
                logger.exception("Failed to register model in MLflow registry")

    return results
