import torch

from wildfire_mlops.modeling import build_model


def test_multimodal_model_forward():
    model = build_model(
        "temporal_multimodal_efficientnet_b0",
        num_classes=2,
        pretrained=False,
        tabular_feature_dim=6,
        temporal_feature_dim=5,
        temporal_encoder_arch="lstm",
        temporal_feature_names=[
            "temperature_c",
            "humidity_pct",
            "wind_speed_kph",
            "drought_index",
            "vegetation_dryness",
        ],
        tabular_feature_names=[
            "temperature_c",
            "humidity_pct",
            "wind_speed_kph",
            "drought_index",
            "vegetation_dryness",
            "days_since_rain",
        ],
    )

    image = torch.randn(2, 3, 224, 224)
    tabular = torch.randn(2, 6)
    temporal = torch.randn(2, 4, 5)
    lengths = torch.tensor([4, 3], dtype=torch.long)

    outputs = model(image=image, tabular=tabular, temporal=temporal, temporal_lengths=lengths)

    assert outputs.logits.shape == (2, 2)
    assert outputs.context_logits is not None
    assert outputs.context_logits.shape == (2, 2)
