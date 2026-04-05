from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict

import torch
from PIL import Image
from torchvision import transforms

from wildfire_mlops.constants import IMAGENET_MEAN, IMAGENET_STD
from wildfire_mlops.inference.gradcam import generate_gradcam_overlay
from wildfire_mlops.monitoring import compute_sample_stats, drift_score, load_stats

DEFAULT_TABULAR_FEATURE_NAMES = [
    "temperature_c",
    "humidity_pct",
    "wind_speed_kph",
    "drought_index",
    "vegetation_dryness",
    "days_since_rain",
]


class PredictionError(RuntimeError):
    """Base inference error."""


class MissingContextError(PredictionError):
    """Raised when a multimodal model is missing required context."""


@dataclass
class EnvironmentalFeatures:
    latitude: float | None = None
    longitude: float | None = None
    temperature_c: float | None = None
    humidity_pct: float | None = None
    wind_speed_kph: float | None = None
    drought_index: float | None = None
    vegetation_dryness: float | None = None
    days_since_rain: float | None = None

    def has_values(self) -> bool:
        return any(value is not None for value in self.to_mapping().values())

    def to_mapping(self) -> dict[str, float | None]:
        return {
            "latitude": self.latitude,
            "longitude": self.longitude,
            "temperature_c": self.temperature_c,
            "humidity_pct": self.humidity_pct,
            "wind_speed_kph": self.wind_speed_kph,
            "drought_index": self.drought_index,
            "vegetation_dryness": self.vegetation_dryness,
            "days_since_rain": self.days_since_rain,
        }

    def to_vector(
        self,
        feature_names: list[str],
        strict: bool = False,
        device: str = "cpu",
    ) -> torch.Tensor:
        values = []
        missing: list[str] = []
        mapping = self.to_mapping()
        for feature_name in feature_names:
            value = mapping.get(feature_name)
            if value is None:
                if strict:
                    missing.append(feature_name)
                values.append(0.0)
            else:
                values.append(float(value))

        if missing:
            raise MissingContextError(
                "Missing environmental features for multimodal inference: "
                + ", ".join(sorted(missing))
            )

        return torch.tensor(values, dtype=torch.float32, device=device).unsqueeze(0)


@dataclass
class RiskContributor:
    factor: str
    impact: float
    rationale: str
    raw_value: float | None = None


@dataclass
class Explainability:
    method: str
    overlay_base64: str | None = None
    summary: str | None = None


@dataclass
class Prediction:
    class_name: str
    confidence: float
    probabilities: Dict[str, float]
    image_wildfire_probability: float
    context_risk_score: float | None
    overall_risk_score: float
    risk_level: str
    recommended_action: str
    environmental_context: EnvironmentalFeatures | None = None
    top_contributors: list[RiskContributor] = field(default_factory=list)
    explainability: Explainability | None = None


def get_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def _normalize_feature(
    value: float | None,
    low: float,
    high: float,
    invert: bool = False,
) -> float:
    if value is None:
        return 0.0
    clipped = max(low, min(high, float(value)))
    score = (clipped - low) / (high - low)
    return 1.0 - score if invert else score


def _positive_class_index(class_names: list[str]) -> int:
    for index, name in enumerate(class_names):
        normalized = name.lower().replace("-", "").replace("_", "").replace(" ", "")
        if normalized == "wildfire":
            return index
        if "wildfire" in normalized and not normalized.startswith(("no", "non", "without")):
            return index
    return len(class_names) - 1


def _risk_level(score: float) -> str:
    if score >= 0.65:
        return "high"
    if score >= 0.35:
        return "moderate"
    return "low"


def _recommended_action(risk_level: str) -> str:
    if risk_level == "high":
        return (
            "Escalate for rapid human review and correlate with live weather and satellite feeds."
        )
    if risk_level == "moderate":
        return "Queue for analyst review and monitor the next weather update window."
    return "Keep under passive monitoring and continue collecting context."


def _rank_context_factors(features: EnvironmentalFeatures | None) -> list[RiskContributor]:
    if features is None or not features.has_values():
        return []

    scored = [
        RiskContributor(
            factor="temperature_c",
            impact=_normalize_feature(features.temperature_c, low=20.0, high=45.0),
            rationale="Higher temperatures dry fuel and make ignition more likely.",
            raw_value=features.temperature_c,
        ),
        RiskContributor(
            factor="humidity_pct",
            impact=_normalize_feature(features.humidity_pct, low=15.0, high=85.0, invert=True),
            rationale="Lower humidity increases fuel dryness and flame persistence.",
            raw_value=features.humidity_pct,
        ),
        RiskContributor(
            factor="wind_speed_kph",
            impact=_normalize_feature(features.wind_speed_kph, low=0.0, high=60.0),
            rationale="Higher wind speeds accelerate fire spread and spotting risk.",
            raw_value=features.wind_speed_kph,
        ),
        RiskContributor(
            factor="drought_index",
            impact=_normalize_feature(features.drought_index, low=0.0, high=800.0),
            rationale=(
                "Drought pressure is a strong proxy for dry vegetation "
                "and sustained burn risk."
            ),
            raw_value=features.drought_index,
        ),
        RiskContributor(
            factor="vegetation_dryness",
            impact=_normalize_feature(features.vegetation_dryness, low=0.0, high=1.0),
            rationale="Dry vegetation means ignition is easier to sustain once flames appear.",
            raw_value=features.vegetation_dryness,
        ),
        RiskContributor(
            factor="days_since_rain",
            impact=_normalize_feature(features.days_since_rain, low=0.0, high=30.0),
            rationale="Longer dry spells usually increase surface fuel flammability.",
            raw_value=features.days_since_rain,
        ),
    ]
    ranked = [item for item in scored if item.raw_value is not None]
    ranked.sort(key=lambda item: item.impact, reverse=True)
    return ranked[:3]


def _prepare_temporal_tensor(
    temporal_sequence: list[dict[str, float]] | list[list[float]] | None,
    feature_names: list[str],
    device: str,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if temporal_sequence is None:
        return None, None
    if not temporal_sequence:
        return torch.empty(1, 0, len(feature_names), device=device), torch.tensor(
            [0], device=device
        )

    if isinstance(temporal_sequence[0], dict):
        rows = [
            [float(step.get(feature_name, 0.0)) for feature_name in feature_names]
            for step in temporal_sequence
        ]
    else:
        rows = [[float(value) for value in step] for step in temporal_sequence]

    tensor = torch.tensor(rows, dtype=torch.float32, device=device).unsqueeze(0)
    lengths = torch.tensor([tensor.shape[1]], dtype=torch.long, device=device)
    return tensor, lengths


def _model_forward(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
    environmental_features: EnvironmentalFeatures | None,
    temporal_sequence: list[dict[str, float]] | list[list[float]] | None,
    device: str,
) -> tuple[object, dict]:
    model_kwargs: dict[str, torch.Tensor] = {}
    if getattr(model, "expects_tabular", False):
        feature_names = (
            getattr(model, "tabular_feature_names", None) or DEFAULT_TABULAR_FEATURE_NAMES
        )
        features = environmental_features or EnvironmentalFeatures()
        model_kwargs["tabular"] = features.to_vector(
            feature_names=feature_names, strict=True, device=device
        )

    if getattr(model, "expects_temporal", False):
        temporal_feature_names = (
            getattr(model, "temporal_feature_names", None) or DEFAULT_TABULAR_FEATURE_NAMES
        )
        temporal_tensor, temporal_lengths = _prepare_temporal_tensor(
            temporal_sequence=temporal_sequence,
            feature_names=temporal_feature_names,
            device=device,
        )
        if temporal_tensor is None or temporal_lengths is None:
            raise MissingContextError("Temporal context is required for the configured model")
        model_kwargs["temporal"] = temporal_tensor
        model_kwargs["temporal_lengths"] = temporal_lengths

    outputs = model(image_tensor, **model_kwargs) if model_kwargs else model(image_tensor)
    return outputs, model_kwargs


def _build_explanation_summary(
    class_name: str,
    risk_level: str,
    contributors: list[RiskContributor],
) -> str:
    if contributors:
        factors = ", ".join(contributor.factor for contributor in contributors[:2])
        return (
            f"The model predicts `{class_name}` with `{risk_level}` wildfire risk. "
            f"Most influential contextual factors: {factors}."
        )
    return (
        f"The model predicts `{class_name}` with `{risk_level}` wildfire risk "
        "based on image evidence."
    )


def predict_image(
    image: Image.Image,
    model: torch.nn.Module,
    class_names: list[str],
    device: str = "cpu",
    image_size: int = 224,
    reference_stats_path: str | None = None,
    environmental_features: EnvironmentalFeatures | None = None,
    temporal_sequence: list[dict[str, float]] | list[list[float]] | None = None,
    include_explainability: bool = False,
) -> Prediction:
    model.eval()
    model.to(device)

    transform = get_transform(image_size)
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs, model_kwargs = _model_forward(
            model=model,
            image_tensor=tensor,
            environmental_features=environmental_features,
            temporal_sequence=temporal_sequence,
            device=device,
        )

    logits = outputs.logits if hasattr(outputs, "logits") else outputs
    context_logits = outputs.context_logits if hasattr(outputs, "context_logits") else None

    probs = torch.softmax(logits, dim=1).squeeze(0)
    pred_idx = int(torch.argmax(probs).item())
    wildfire_idx = _positive_class_index(class_names)
    confidence = float(probs[pred_idx].item())
    probabilities = {
        class_names[index]: float(probs[index].item()) for index in range(len(class_names))
    }
    image_wildfire_probability = float(probabilities[class_names[wildfire_idx]])
    context_risk_score = None
    if context_logits is not None:
        context_probs = torch.softmax(context_logits, dim=1).squeeze(0)
        context_risk_score = float(context_probs[wildfire_idx].item())

    overall_risk_score = float(image_wildfire_probability)
    risk_level = _risk_level(overall_risk_score)
    top_contributors = _rank_context_factors(environmental_features)

    explainability = None
    if include_explainability:
        overlay = generate_gradcam_overlay(
            image=image,
            model=model,
            tensor=tensor.clone(),
            target_index=pred_idx,
            model_kwargs=model_kwargs,
        )
        explainability = Explainability(
            method="gradcam",
            overlay_base64=overlay,
            summary=_build_explanation_summary(
                class_name=class_names[pred_idx],
                risk_level=risk_level,
                contributors=top_contributors,
            ),
        )

    if reference_stats_path:
        ref = load_stats(Path(reference_stats_path))
        if ref:
            sample = compute_sample_stats(tensor.detach().cpu())
            _ = drift_score(ref, sample)

    return Prediction(
        class_name=class_names[pred_idx],
        confidence=confidence,
        probabilities=probabilities,
        image_wildfire_probability=image_wildfire_probability,
        context_risk_score=context_risk_score,
        overall_risk_score=overall_risk_score,
        risk_level=risk_level,
        recommended_action=_recommended_action(risk_level),
        environmental_context=(
            environmental_features
            if environmental_features and environmental_features.has_values()
            else None
        ),
        top_contributors=top_contributors,
        explainability=explainability,
    )
