import numpy as np
from PIL import Image

from wildfire_mlops.inference import EnvironmentalFeatures, predict_image
from wildfire_mlops.modeling import WildfireCNN


def test_predict_image_shape():
    model = WildfireCNN(num_classes=2)
    class_names = ["nowildfire", "wildfire"]
    data = (np.random.rand(224, 224, 3) * 255).astype("uint8")
    image = Image.fromarray(data, mode="RGB")

    pred = predict_image(
        image=image,
        model=model,
        class_names=class_names,
        environmental_features=EnvironmentalFeatures(
            temperature_c=35.0,
            humidity_pct=20.0,
            wind_speed_kph=24.0,
        ),
        include_explainability=True,
    )

    assert pred.class_name in class_names
    assert 0.0 <= pred.confidence <= 1.0
    assert set(pred.probabilities.keys()) == set(class_names)
    assert 0.0 <= pred.overall_risk_score <= 1.0
    assert pred.explainability is not None
