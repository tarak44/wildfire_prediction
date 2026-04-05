from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset

from wildfire_mlops.data.loader import build_transforms

ImageFile.LOAD_TRUNCATED_IMAGES = True


@dataclass
class MultimodalDataConfig:
    train_manifest: Path
    val_manifest: Path
    test_manifest: Path | None
    image_size: int
    batch_size: int
    num_workers: int
    tabular_feature_names: list[str]
    temporal_feature_names: list[str]
    temporal_sequence_column: str
    image_column: str
    label_column: str
    max_train_samples: int | None
    max_val_samples: int | None
    seed: int


@dataclass
class MultimodalRecord:
    image_path: Path
    label: str
    tabular: list[float]
    temporal: list[list[float]] | None


def _parse_temporal_row(
    row: dict[str, str],
    temporal_feature_names: list[str],
    temporal_sequence_column: str,
) -> list[list[float]] | None:
    raw_value = row.get(temporal_sequence_column)
    if not raw_value:
        return None

    parsed = json.loads(raw_value)
    if not isinstance(parsed, list):
        raise ValueError(f"Temporal sequence must be a list, received: {type(parsed)!r}")

    if not parsed:
        return []

    if isinstance(parsed[0], dict):
        return [
            [float(step.get(feature_name, 0.0)) for feature_name in temporal_feature_names]
            for step in parsed
        ]

    return [[float(value) for value in step] for step in parsed]


def load_multimodal_manifest(
    manifest_path: Path,
    tabular_feature_names: list[str],
    temporal_feature_names: list[str],
    image_column: str = "image_path",
    label_column: str = "label",
    temporal_sequence_column: str = "temporal_sequence",
) -> list[MultimodalRecord]:
    records: list[MultimodalRecord] = []
    with manifest_path.open("r", encoding="utf-8", newline="") as file_handle:
        reader = csv.DictReader(file_handle)
        for row in reader:
            tabular = [float(row[feature_name]) for feature_name in tabular_feature_names]
            temporal = _parse_temporal_row(row, temporal_feature_names, temporal_sequence_column)
            records.append(
                MultimodalRecord(
                    image_path=Path(row[image_column]),
                    label=row[label_column],
                    tabular=tabular,
                    temporal=temporal,
                )
            )
    return records


def _limit_records(records: list[MultimodalRecord], max_samples: int | None, seed: int):
    if max_samples is None or max_samples <= 0 or len(records) <= max_samples:
        return records
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(records), generator=generator)[:max_samples].tolist()
    return [records[index] for index in indices]


class MultimodalWildfireDataset(Dataset):
    def __init__(
        self,
        records: list[MultimodalRecord],
        class_names: list[str],
        image_size: int,
        train: bool = False,
    ) -> None:
        super().__init__()
        self.records = records
        self.class_names = class_names
        train_transform, eval_transform = build_transforms(image_size)
        self.transform = train_transform if train else eval_transform
        self.class_to_index = {name: index for index, name in enumerate(class_names)}

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | int]:
        record = self.records[index]
        image = Image.open(record.image_path).convert("RGB")
        image_tensor = self.transform(image)
        label_index = self.class_to_index[record.label]
        temporal = (
            torch.tensor(record.temporal, dtype=torch.float32)
            if record.temporal is not None
            else torch.empty(0, 0, dtype=torch.float32)
        )
        return {
            "image": image_tensor,
            "tabular": torch.tensor(record.tabular, dtype=torch.float32),
            "temporal": temporal,
            "temporal_length": int(temporal.shape[0]) if temporal.numel() else 0,
            "label": label_index,
        }


def _multimodal_collate(batch: list[dict[str, torch.Tensor | int]]) -> dict[str, torch.Tensor]:
    images = torch.stack([item["image"] for item in batch])  # type: ignore[index]
    tabular = torch.stack([item["tabular"] for item in batch])  # type: ignore[index]
    labels = torch.tensor([int(item["label"]) for item in batch], dtype=torch.long)
    lengths = torch.tensor([int(item["temporal_length"]) for item in batch], dtype=torch.long)

    max_length = int(lengths.max().item()) if lengths.numel() else 0
    if max_length > 0:
        feature_dim = next(
            int(item["temporal"].shape[1])  # type: ignore[index]
            for item in batch
            if int(item["temporal_length"]) > 0
        )
        temporal_batch = torch.zeros(len(batch), max_length, feature_dim, dtype=torch.float32)
        for row_index, item in enumerate(batch):
            current_length = int(item["temporal_length"])
            if current_length:
                temporal_batch[row_index, :current_length] = item["temporal"]  # type: ignore[index]
    else:
        temporal_batch = torch.empty(len(batch), 0, 0, dtype=torch.float32)

    return {
        "image": images,
        "tabular": tabular,
        "temporal": temporal_batch,
        "temporal_lengths": lengths,
        "label": labels,
    }


def compute_record_class_weights(
    records: list[MultimodalRecord], class_names: list[str]
) -> torch.Tensor:
    class_to_index = {name: index for index, name in enumerate(class_names)}
    counts = torch.zeros(len(class_names), dtype=torch.float32)
    for record in records:
        counts[class_to_index[record.label]] += 1.0
    counts = torch.where(counts == 0, torch.ones_like(counts), counts)
    weights = counts.sum() / (counts * len(class_names))
    return weights / weights.mean()


def build_multimodal_dataloaders(
    cfg: MultimodalDataConfig,
) -> tuple[DataLoader, DataLoader, DataLoader | None, list[str], torch.Tensor]:
    train_records = load_multimodal_manifest(
        cfg.train_manifest,
        tabular_feature_names=cfg.tabular_feature_names,
        temporal_feature_names=cfg.temporal_feature_names,
        image_column=cfg.image_column,
        label_column=cfg.label_column,
        temporal_sequence_column=cfg.temporal_sequence_column,
    )
    val_records = load_multimodal_manifest(
        cfg.val_manifest,
        tabular_feature_names=cfg.tabular_feature_names,
        temporal_feature_names=cfg.temporal_feature_names,
        image_column=cfg.image_column,
        label_column=cfg.label_column,
        temporal_sequence_column=cfg.temporal_sequence_column,
    )
    test_records = (
        load_multimodal_manifest(
            cfg.test_manifest,
            tabular_feature_names=cfg.tabular_feature_names,
            temporal_feature_names=cfg.temporal_feature_names,
            image_column=cfg.image_column,
            label_column=cfg.label_column,
            temporal_sequence_column=cfg.temporal_sequence_column,
        )
        if cfg.test_manifest is not None
        else None
    )

    train_records = _limit_records(train_records, cfg.max_train_samples, cfg.seed)
    val_records = _limit_records(val_records, cfg.max_val_samples, cfg.seed)
    class_names = sorted({record.label for record in train_records})

    train_dataset = MultimodalWildfireDataset(
        records=train_records,
        class_names=class_names,
        image_size=cfg.image_size,
        train=True,
    )
    val_dataset = MultimodalWildfireDataset(
        records=val_records,
        class_names=class_names,
        image_size=cfg.image_size,
        train=False,
    )
    test_dataset = (
        MultimodalWildfireDataset(
            records=test_records,
            class_names=class_names,
            image_size=cfg.image_size,
            train=False,
        )
        if test_records is not None
        else None
    )

    class_weights = compute_record_class_weights(train_records, class_names)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=_multimodal_collate,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=_multimodal_collate,
    )
    test_loader = (
        DataLoader(
            test_dataset,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            collate_fn=_multimodal_collate,
        )
        if test_dataset is not None
        else None
    )

    return train_loader, val_loader, test_loader, class_names, class_weights
