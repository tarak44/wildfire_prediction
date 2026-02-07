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
