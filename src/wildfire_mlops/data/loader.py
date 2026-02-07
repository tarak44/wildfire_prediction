from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import ImageFile

from wildfire_mlops.constants import IMAGENET_MEAN, IMAGENET_STD

# Handle occasional truncated images in real-world datasets
ImageFile.LOAD_TRUNCATED_IMAGES = True
logger = logging.getLogger(__name__)


@dataclass
class DataConfig:
    train_dir: Path
    val_dir: Path
    test_dir: Path | None
    image_size: int
    batch_size: int
    num_workers: int
    max_train_samples: int | None
    max_val_samples: int | None
    seed: int


def build_transforms(image_size: int) -> Tuple[transforms.Compose, transforms.Compose]:
    train_tf = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    eval_tf = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    return train_tf, eval_tf


def _limit_dataset(dataset, max_samples: int | None, seed: int):
    if max_samples is None or max_samples <= 0:
        return dataset
    if len(dataset) <= max_samples:
        return dataset
    g = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=g)[:max_samples].tolist()
    return torch.utils.data.Subset(dataset, indices)


def build_dataloaders(cfg: DataConfig) -> Tuple[DataLoader, DataLoader, DataLoader | None, list[str]]:
    train_tf, eval_tf = build_transforms(cfg.image_size)

    train_ds = datasets.ImageFolder(root=str(cfg.train_dir), transform=train_tf)
    val_ds = datasets.ImageFolder(root=str(cfg.val_dir), transform=eval_tf)

    test_ds = None
    if cfg.test_dir is not None:
        test_ds = datasets.ImageFolder(root=str(cfg.test_dir), transform=eval_tf)

    train_ds = _limit_dataset(train_ds, cfg.max_train_samples, cfg.seed)
    val_ds = _limit_dataset(val_ds, cfg.max_val_samples, cfg.seed)

    class_names = train_ds.dataset.classes if hasattr(train_ds, "dataset") else train_ds.classes

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
    )
    test_loader = None
    if test_ds is not None:
        test_loader = DataLoader(
            test_ds,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
        )

    return train_loader, val_loader, test_loader, class_names


def save_class_names(path: Path, class_names: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(class_names, f, indent=2)
