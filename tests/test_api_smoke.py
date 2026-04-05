import base64
import io

from fastapi.testclient import TestClient
from PIL import Image

from api.main import app


def test_api_predict_smoke():
    client = TestClient(app)
    img = Image.new("RGB", (224, 224), color=(128, 64, 32))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)

    files = {"file": ("test.png", buf, "image/png")}
    resp = client.post("/predict", files=files)

    assert resp.status_code == 200
    data = resp.json()
    assert "class_name" in data
    assert "confidence" in data
    assert "probabilities" in data
    assert "overall_risk_score" in data
    assert "risk_level" in data


def test_api_predict_batch_and_model_info():
    client = TestClient(app)
    img = Image.new("RGB", (224, 224), color=(200, 90, 20))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    payload = {
        "items": [
            {
                "image_base64": base64.b64encode(buf.getvalue()).decode("utf-8"),
                "environmental_context": {
                    "temperature_c": 34.0,
                    "humidity_pct": 22.0,
                    "wind_speed_kph": 20.0,
                },
            }
        ]
    }

    batch_response = client.post("/predict-batch", json=payload)
    assert batch_response.status_code == 200
    batch_data = batch_response.json()
    assert len(batch_data["predictions"]) == 1
    assert "overall_risk_score" in batch_data["predictions"][0]

    info_response = client.get("/model-info")
    assert info_response.status_code == 200
    info = info_response.json()
    assert "model_arch" in info
    assert "class_names" in info
