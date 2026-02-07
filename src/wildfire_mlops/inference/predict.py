from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import torch
from PIL import Image
from torchvision import transforms

from wildfire_mlops.constants import IMAGENET_MEAN, IMAGENET_STD
from wildfire_mlops.monitoring import compute_sample_stats, drift_score, load_stats


@dataclass
class Prediction:
    class_name: str
    confidence: float
    probabilities: Dict[str, float]


def get_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=IMAGENET_MEAN,
                std=IMAGENET_STD,
            ),
        ]
    )


def predict_image(
    image: Image.Image,
    model: torch.nn.Module,
    class_names: list[str],
    device: str = "cpu",
    image_size: int = 224,
    reference_stats_path: str | None = None,
) -> Prediction:
    model.eval()
    model.to(device)

    transform = get_transform(image_size)
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1).squeeze(0)

    pred_idx = int(torch.argmax(probs).item())
    conf = float(probs[pred_idx].item())
    prob_map = {class_names[i]: float(probs[i].item()) for i in range(len(class_names))}

    # Optional drift score logging (non-blocking)
    if reference_stats_path:
        ref = load_stats(Path(reference_stats_path))
        if ref:
            sample = compute_sample_stats(tensor.detach().cpu())
            _ = drift_score(ref, sample)

    return Prediction(
        class_name=class_names[pred_idx],
        confidence=conf,
        probabilities=prob_map,
    )
