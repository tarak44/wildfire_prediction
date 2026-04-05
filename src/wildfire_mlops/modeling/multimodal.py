from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import EfficientNet_B0_Weights


@dataclass
class MultimodalOutput:
    logits: torch.Tensor
    context_logits: torch.Tensor | None = None
    image_embedding: torch.Tensor | None = None
    context_embedding: torch.Tensor | None = None


class TabularMLPEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: tuple[int, ...] = (64, 32),
        embedding_dim: int = 32,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        current_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(current_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            current_dim = hidden_dim

        layers.append(nn.Linear(current_dim, embedding_dim))
        self.network = nn.Sequential(*layers)
        self.output_dim = embedding_dim

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)


class TemporalSequenceEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        output_dim: int = 32,
        num_layers: int = 1,
        dropout: float = 0.1,
        arch: str = "lstm",
        max_sequence_length: int = 24,
    ) -> None:
        super().__init__()
        self.arch = arch
        self.output_dim = output_dim
        self.max_sequence_length = max_sequence_length

        if arch == "lstm":
            self.encoder = nn.LSTM(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
            self.projection = nn.Linear(hidden_dim, output_dim)
        elif arch == "temporal_mlp":
            flattened_dim = input_dim * max_sequence_length
            self.encoder = nn.Sequential(
                nn.Linear(flattened_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, output_dim),
            )
            self.projection = nn.Identity()
        else:
            raise ValueError(f"Unsupported temporal encoder architecture: {arch}")

    def forward(
        self,
        sequence: torch.Tensor,
        lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.arch == "lstm":
            if lengths is not None:
                packed = nn.utils.rnn.pack_padded_sequence(
                    sequence,
                    lengths=lengths.cpu(),
                    batch_first=True,
                    enforce_sorted=False,
                )
                _, (hidden, _) = self.encoder(packed)
            else:
                _, (hidden, _) = self.encoder(sequence)
            return self.projection(hidden[-1])

        batch_size, seq_len, feature_dim = sequence.shape
        if seq_len > self.max_sequence_length:
            sequence = sequence[:, : self.max_sequence_length, :]
            seq_len = self.max_sequence_length
        elif seq_len < self.max_sequence_length:
            padding = torch.zeros(
                batch_size,
                self.max_sequence_length - seq_len,
                feature_dim,
                device=sequence.device,
                dtype=sequence.dtype,
            )
            sequence = torch.cat([sequence, padding], dim=1)

        flattened = sequence.reshape(batch_size, -1)
        return self.projection(self.encoder(flattened))


class EfficientNetImageEncoder(nn.Module):
    def __init__(
        self,
        pretrained: bool = True,
        embedding_dim: int = 128,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.efficientnet_b0(weights=weights)
        self.features = backbone.features
        self.pool = backbone.avgpool
        self.projection = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(1280, embedding_dim),
            nn.ReLU(),
        )
        self.output_dim = embedding_dim

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        features = self.features(image)
        pooled = self.pool(features).flatten(1)
        return self.projection(pooled)


class WildfireMultimodalModel(nn.Module):
    expects_tabular = True

    def __init__(
        self,
        num_classes: int,
        tabular_feature_dim: int,
        pretrained: bool = True,
        temporal_feature_dim: int = 0,
        temporal_encoder_arch: str = "lstm",
        temporal_hidden_dim: int = 64,
        temporal_max_sequence_length: int = 24,
        tabular_feature_names: list[str] | None = None,
        temporal_feature_names: list[str] | None = None,
    ) -> None:
        super().__init__()
        if tabular_feature_dim <= 0:
            raise ValueError("tabular_feature_dim must be greater than zero for multimodal models")

        self.tabular_feature_names = tabular_feature_names or []
        self.temporal_feature_names = temporal_feature_names or []
        self.expects_temporal = temporal_feature_dim > 0
        self.temporal_encoder_arch = temporal_encoder_arch
        self.temporal_max_sequence_length = temporal_max_sequence_length

        self.image_encoder = EfficientNetImageEncoder(pretrained=pretrained)
        self.tabular_encoder = TabularMLPEncoder(input_dim=tabular_feature_dim)
        context_dim = self.tabular_encoder.output_dim

        self.temporal_encoder: TemporalSequenceEncoder | None = None
        if self.expects_temporal:
            self.temporal_encoder = TemporalSequenceEncoder(
                input_dim=temporal_feature_dim,
                hidden_dim=temporal_hidden_dim,
                output_dim=32,
                arch=temporal_encoder_arch,
                max_sequence_length=temporal_max_sequence_length,
            )
            context_dim += self.temporal_encoder.output_dim

        fusion_dim = self.image_encoder.output_dim + context_dim
        self.fusion_head = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        self.classifier = nn.Linear(64, num_classes)
        self.context_head = nn.Sequential(
            nn.Linear(context_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes),
        )

    def forward(
        self,
        image: torch.Tensor,
        tabular: torch.Tensor,
        temporal: torch.Tensor | None = None,
        temporal_lengths: torch.Tensor | None = None,
    ) -> MultimodalOutput:
        image_embedding = self.image_encoder(image)
        tabular_embedding = self.tabular_encoder(tabular)

        context_parts = [tabular_embedding]
        if self.temporal_encoder is not None:
            if temporal is None:
                raise ValueError("Temporal tensor is required for temporal multimodal models")
            temporal_embedding = self.temporal_encoder(temporal, lengths=temporal_lengths)
            context_parts.append(temporal_embedding)

        context_embedding = torch.cat(context_parts, dim=1)
        fusion_embedding = torch.cat([image_embedding, context_embedding], dim=1)
        fusion_hidden = self.fusion_head(fusion_embedding)

        return MultimodalOutput(
            logits=self.classifier(fusion_hidden),
            context_logits=self.context_head(context_embedding),
            image_embedding=image_embedding,
            context_embedding=context_embedding,
        )
