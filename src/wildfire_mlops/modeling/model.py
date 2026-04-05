from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import EfficientNet_B0_Weights, ResNet18_Weights

from .multimodal import WildfireMultimodalModel


class WildfireCNN(nn.Module):
    def __init__(self, num_classes: int = 2) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)

        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)

        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)

        self.fc1 = nn.Linear(256 * 14 * 14, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, num_classes)

        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        x = self.pool(self.relu(self.bn4(self.conv4(x))))
        x = x.view(x.size(0), -1)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.fc3(x)
        return x


def _torch_load(path: str):
    try:
        return torch.load(path, map_location=torch.device("cpu"), weights_only=False)
    except TypeError:
        return torch.load(path, map_location=torch.device("cpu"))


def load_checkpoint(
    path: str, model_arch: str = "custom_cnn", pretrained: bool = True
) -> Tuple[nn.Module, list[str]]:
    checkpoint = _torch_load(path)

    if (
        isinstance(checkpoint, dict)
        and "model" in checkpoint
        and isinstance(checkpoint["model"], nn.Module)
    ):
        model = checkpoint["model"]
        class_names = checkpoint.get("class_names", ["class0", "class1"])
        model.eval()
        return model, class_names

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        class_names = checkpoint.get("class_names", ["class0", "class1"])
        resolved_arch = checkpoint.get("model_arch", model_arch)
        model = build_model(
            resolved_arch,
            num_classes=len(class_names),
            pretrained=pretrained,
            tabular_feature_dim=int(checkpoint.get("tabular_feature_dim", 0) or 0),
            temporal_feature_dim=int(checkpoint.get("temporal_feature_dim", 0) or 0),
            temporal_encoder_arch=str(checkpoint.get("temporal_encoder_arch") or "lstm"),
            temporal_hidden_dim=int(checkpoint.get("temporal_hidden_dim", 64) or 64),
            temporal_max_sequence_length=int(
                checkpoint.get("temporal_max_sequence_length", 24) or 24
            ),
            tabular_feature_names=checkpoint.get("tabular_feature_names") or [],
            temporal_feature_names=checkpoint.get("temporal_feature_names") or [],
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        return model, class_names

    if isinstance(checkpoint, dict):
        model = build_model(model_arch, num_classes=2, pretrained=pretrained)
        model.load_state_dict(checkpoint)
        model.eval()
        return model, ["nowildfire", "wildfire"]

    raise RuntimeError("Unsupported checkpoint format")


def build_model(
    arch: str,
    num_classes: int,
    pretrained: bool = True,
    tabular_feature_dim: int = 0,
    temporal_feature_dim: int = 0,
    temporal_encoder_arch: str = "lstm",
    temporal_hidden_dim: int = 64,
    temporal_max_sequence_length: int = 24,
    tabular_feature_names: list[str] | None = None,
    temporal_feature_names: list[str] | None = None,
) -> nn.Module:
    if arch == "custom_cnn":
        return WildfireCNN(num_classes=num_classes)
    if arch == "resnet18":
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.resnet18(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        return model
    if arch == "efficientnet_b0":
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
        return model
    if arch in {"multimodal_efficientnet_b0", "temporal_multimodal_efficientnet_b0"}:
        return WildfireMultimodalModel(
            num_classes=num_classes,
            tabular_feature_dim=tabular_feature_dim,
            pretrained=pretrained,
            temporal_feature_dim=temporal_feature_dim,
            temporal_encoder_arch=temporal_encoder_arch,
            temporal_hidden_dim=temporal_hidden_dim,
            temporal_max_sequence_length=temporal_max_sequence_length,
            tabular_feature_names=tabular_feature_names,
            temporal_feature_names=temporal_feature_names,
        )
    raise ValueError(f"Unsupported model arch: {arch}")
