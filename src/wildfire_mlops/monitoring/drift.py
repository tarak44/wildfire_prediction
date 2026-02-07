from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import torch


def compute_reference_stats(
    loader, max_batches: int = 10, device: str = "cpu"
) -> Dict[str, list[float]]:
    """Compute per-channel mean/std over a few batches."""
    means = []
    stds = []
    count = 0
    for images, _ in loader:
        images = images.to(device)
        # images: [B, C, H, W]
        means.append(images.mean(dim=[0, 2, 3]).cpu())
        stds.append(images.std(dim=[0, 2, 3]).cpu())
        count += 1
        if count >= max_batches:
            break
    mean = torch.stack(means).mean(dim=0)
    std = torch.stack(stds).mean(dim=0)
    return {"mean": mean.tolist(), "std": std.tolist()}


def compute_sample_stats(image_tensor: torch.Tensor) -> Dict[str, list[float]]:
    """Compute per-channel mean/std for a single image tensor [1,C,H,W] or [C,H,W]."""
    if image_tensor.dim() == 3:
        image_tensor = image_tensor.unsqueeze(0)
    mean = image_tensor.mean(dim=[0, 2, 3]).cpu()
    std = image_tensor.std(dim=[0, 2, 3]).cpu()
    return {"mean": mean.tolist(), "std": std.tolist()}


def drift_score(ref: Dict[str, list[float]], sample: Dict[str, list[float]]) -> float:
    """Compute a simple drift score as L1 distance between mean/std vectors."""
    ref_vec = torch.tensor(ref["mean"] + ref["std"])
    sample_vec = torch.tensor(sample["mean"] + sample["std"])
    return float(torch.abs(ref_vec - sample_vec).mean().item())


def load_stats(path: Path) -> Dict[str, list[float]] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)
