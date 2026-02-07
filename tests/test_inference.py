from PIL import Image
import numpy as np

from wildfire_mlops.inference import predict_image
from wildfire_mlops.modeling import WildfireCNN


def test_predict_image_shape():
    model = WildfireCNN(num_classes=2)
    class_names = ["nowildfire", "wildfire"]
    data = (np.random.rand(224, 224, 3) * 255).astype("uint8")
    image = Image.fromarray(data, mode="RGB")

    pred = predict_image(image=image, model=model, class_names=class_names)

    assert pred.class_name in class_names
    assert 0.0 <= pred.confidence <= 1.0
    assert set(pred.probabilities.keys()) == set(class_names)
