from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import mlflow
import torch
from torch import nn
from torch.optim import AdamW

from wildfire_mlops.data.multimodal import MultimodalDataConfig, build_multimodal_dataloaders
from wildfire_mlops.modeling import build_model
from wildfire_mlops.training.metrics import compute_metrics

logger = logging.getLogger(__name__)


@dataclass
class MultimodalTrainConfig:
    train_manifest: Path
    val_manifest: Path
    test_manifest: Path | None
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
    tabular_feature_names: list[str]
    temporal_feature_names: list[str]
    temporal_sequence_column: str
    temporal_encoder_arch: str
    temporal_hidden_dim: int
    temporal_max_sequence_length: int
    image_column: str
    label_column: str
    auxiliary_context_loss_weight: float


def _set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _compute_loss(
    outputs,
    labels: torch.Tensor,
    criterion: nn.Module,
    auxiliary_context_loss_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    main_loss = criterion(outputs.logits, labels)
    aux_loss = torch.tensor(0.0, device=labels.device)
    if outputs.context_logits is not None:
        aux_loss = criterion(outputs.context_logits, labels)
    total_loss = main_loss + (auxiliary_context_loss_weight * aux_loss)
    return total_loss, {
        "main_loss": float(main_loss.detach().item()),
        "context_loss": float(aux_loss.detach().item()),
        "total_loss": float(total_loss.detach().item()),
    }


def _run_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: str,
    auxiliary_context_loss_weight: float,
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    total = 0
    correct = 0

    for batch in loader:
        images = batch["image"].to(device)
        tabular = batch["tabular"].to(device)
        temporal = batch["temporal"].to(device) if batch["temporal"].numel() else None
        temporal_lengths = batch["temporal_lengths"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        outputs = model(
            image=images,
            tabular=tabular,
            temporal=temporal,
            temporal_lengths=temporal_lengths if temporal is not None else None,
        )
        loss, _ = _compute_loss(
            outputs=outputs,
            labels=labels,
            criterion=criterion,
            auxiliary_context_loss_weight=auxiliary_context_loss_weight,
        )
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item()) * images.size(0)
        predictions = torch.argmax(outputs.logits, dim=1)
        total += labels.size(0)
        correct += int((predictions == labels).sum().item())

    return {
        "loss": total_loss / max(total, 1),
        "accuracy": correct / max(total, 1),
    }


def _evaluate(
    model: nn.Module,
    loader,
    device: str,
    class_names: list[str],
) -> Dict[str, object]:
    model.eval()
    true_labels = []
    predicted_labels = []
    positive_scores = []
    context_scores = []

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            tabular = batch["tabular"].to(device)
            temporal = batch["temporal"].to(device) if batch["temporal"].numel() else None
            temporal_lengths = batch["temporal_lengths"].to(device)
            labels = batch["label"].to(device)

            outputs = model(
                image=images,
                tabular=tabular,
                temporal=temporal,
                temporal_lengths=temporal_lengths if temporal is not None else None,
            )
            probabilities = torch.softmax(outputs.logits, dim=1)
            predictions = torch.argmax(probabilities, dim=1)

            true_labels.append(labels.cpu())
            predicted_labels.append(predictions.cpu())
            positive_scores.append(probabilities[:, -1].cpu())

            if outputs.context_logits is not None:
                context_probabilities = torch.softmax(outputs.context_logits, dim=1)
                context_scores.append(context_probabilities[:, -1].cpu())

    y_true = torch.cat(true_labels)
    y_pred = torch.cat(predicted_labels)
    y_score = torch.cat(positive_scores)
    metrics = compute_metrics(
        y_true=y_true, y_pred=y_pred, class_names=class_names, y_score=y_score
    )

    return {
        "accuracy": metrics.accuracy,
        "balanced_accuracy": metrics.balanced_accuracy,
        "macro_precision": metrics.macro_precision,
        "macro_recall": metrics.macro_recall,
        "macro_f1": metrics.macro_f1,
        "roc_auc": metrics.roc_auc,
        "precision": metrics.precision,
        "recall": metrics.recall,
        "f1": metrics.f1,
        "confusion_matrix": metrics.confusion_matrix,
        "mean_context_score": (
            float(torch.cat(context_scores).mean().item()) if context_scores else None
        ),
    }


def train_multimodal_model(cfg: MultimodalTrainConfig) -> Dict[str, object]:
    _set_seed(cfg.seed)

    data_cfg = MultimodalDataConfig(
        train_manifest=cfg.train_manifest,
        val_manifest=cfg.val_manifest,
        test_manifest=cfg.test_manifest,
        image_size=cfg.image_size,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        tabular_feature_names=cfg.tabular_feature_names,
        temporal_feature_names=cfg.temporal_feature_names,
        temporal_sequence_column=cfg.temporal_sequence_column,
        image_column=cfg.image_column,
        label_column=cfg.label_column,
        max_train_samples=cfg.max_train_samples,
        max_val_samples=cfg.max_val_samples,
        seed=cfg.seed,
    )
    train_loader, val_loader, test_loader, class_names, class_weights = (
        build_multimodal_dataloaders(data_cfg)
    )

    model = build_model(
        cfg.model_arch,
        num_classes=len(class_names),
        pretrained=cfg.pretrained,
        tabular_feature_dim=len(cfg.tabular_feature_names),
        temporal_feature_dim=len(cfg.temporal_feature_names),
        temporal_encoder_arch=cfg.temporal_encoder_arch,
        temporal_hidden_dim=cfg.temporal_hidden_dim,
        temporal_max_sequence_length=cfg.temporal_max_sequence_length,
        tabular_feature_names=cfg.tabular_feature_names,
        temporal_feature_names=cfg.temporal_feature_names,
    ).to(cfg.device)

    criterion = nn.CrossEntropyLoss(weight=class_weights.to(cfg.device))
    optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = cfg.output_dir / "multimodal_model_best.pth"
    latest_path = cfg.output_dir / "multimodal_model_latest.pth"
    metrics_path = cfg.output_dir / "multimodal_metrics.json"
    best_macro_f1 = -1.0
    history: List[Dict[str, object]] = []

    mlflow.set_tracking_uri(cfg.mlflow_tracking_uri)
    mlflow.set_experiment(cfg.mlflow_experiment)

    with mlflow.start_run(run_name=cfg.mlflow_run_name):
        mlflow.log_params(
            {
                "train_manifest": str(cfg.train_manifest),
                "val_manifest": str(cfg.val_manifest),
                "test_manifest": str(cfg.test_manifest) if cfg.test_manifest else "",
                "image_size": cfg.image_size,
                "batch_size": cfg.batch_size,
                "epochs": cfg.epochs,
                "lr": cfg.lr,
                "weight_decay": cfg.weight_decay,
                "model_arch": cfg.model_arch,
                "tabular_feature_names": json.dumps(cfg.tabular_feature_names),
                "temporal_feature_names": json.dumps(cfg.temporal_feature_names),
                "temporal_encoder_arch": cfg.temporal_encoder_arch,
                "auxiliary_context_loss_weight": cfg.auxiliary_context_loss_weight,
                "class_weights": json.dumps(class_weights.tolist()),
            }
        )

        for epoch in range(1, cfg.epochs + 1):
            train_metrics = _run_epoch(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                criterion=criterion,
                device=cfg.device,
                auxiliary_context_loss_weight=cfg.auxiliary_context_loss_weight,
            )
            val_metrics = _evaluate(
                model=model, loader=val_loader, device=cfg.device, class_names=class_names
            )

            history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "class_names": class_names,
                "epoch": epoch,
                "model_arch": cfg.model_arch,
                "tabular_feature_names": cfg.tabular_feature_names,
                "tabular_feature_dim": len(cfg.tabular_feature_names),
                "temporal_feature_names": cfg.temporal_feature_names,
                "temporal_feature_dim": len(cfg.temporal_feature_names),
                "temporal_encoder_arch": cfg.temporal_encoder_arch,
                "temporal_hidden_dim": cfg.temporal_hidden_dim,
                "temporal_max_sequence_length": cfg.temporal_max_sequence_length,
                "auxiliary_context_loss_weight": cfg.auxiliary_context_loss_weight,
            }
            torch.save(checkpoint, latest_path)

            macro_f1 = float(val_metrics["macro_f1"])
            if macro_f1 > best_macro_f1:
                best_macro_f1 = macro_f1
                torch.save(checkpoint, checkpoint_path)

            mlflow.log_metric("train_loss", float(train_metrics["loss"]), step=epoch)
            mlflow.log_metric("train_accuracy", float(train_metrics["accuracy"]), step=epoch)
            mlflow.log_metric("val_accuracy", float(val_metrics["accuracy"]), step=epoch)
            mlflow.log_metric("val_macro_f1", macro_f1, step=epoch)
            if val_metrics["roc_auc"] is not None:
                mlflow.log_metric("val_roc_auc", float(val_metrics["roc_auc"]), step=epoch)

            logger.info(
                "epoch=%s train_acc=%.4f val_acc=%.4f val_macro_f1=%.4f",
                epoch,
                train_metrics["accuracy"],
                val_metrics["accuracy"],
                macro_f1,
            )

        results: Dict[str, object] = {
            "best_val_macro_f1": best_macro_f1,
            "history": history,
        }

        if test_loader is not None:
            test_metrics = _evaluate(
                model=model,
                loader=test_loader,
                device=cfg.device,
                class_names=class_names,
            )
            results["test"] = test_metrics
            mlflow.log_metric("test_accuracy", float(test_metrics["accuracy"]))
            if test_metrics["roc_auc"] is not None:
                mlflow.log_metric("test_roc_auc", float(test_metrics["roc_auc"]))

        with metrics_path.open("w", encoding="utf-8") as file_handle:
            json.dump(results, file_handle, indent=2)

        mlflow.log_artifact(str(metrics_path))
        if latest_path.exists():
            mlflow.log_artifact(str(latest_path))
        if checkpoint_path.exists():
            mlflow.log_artifact(str(checkpoint_path))

        if cfg.register_model and checkpoint_path.exists():
            try:
                import mlflow.pytorch as mlflow_pytorch

                model_uri = mlflow_pytorch.log_model(model, artifact_path="multimodal_model")
                mlflow.register_model(model_uri.model_uri, cfg.model_registry_name)
            except Exception:
                logger.exception("Failed to register multimodal model in MLflow registry")

    return results
