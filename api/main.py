import logging
from io import BytesIO

from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image

from wildfire_mlops.constants import SUPPORTED_CONTENT_TYPES
from wildfire_mlops.core import PredictionResponse, get_settings, setup_logging
from wildfire_mlops.inference import predict_image
from wildfire_mlops.modeling import load_checkpoint

settings = get_settings()
setup_logging(settings.log_level)
logger = logging.getLogger("api")

app = FastAPI(title="Wildfire Prediction API", version="0.1.0")

model, class_names = load_checkpoint(
    settings.resolve_model_path(), model_arch=settings.model_arch, pretrained=settings.pretrained
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)) -> PredictionResponse:
    if file.content_type not in SUPPORTED_CONTENT_TYPES:
        raise HTTPException(status_code=400, detail="Unsupported file type")

    data = await file.read()
    try:
        image = Image.open(BytesIO(data)).convert("RGB")
    except Exception as exc:
        logger.exception("Failed to load image")
        raise HTTPException(status_code=400, detail="Invalid image") from exc

    pred = predict_image(
        image=image,
        model=model,
        class_names=class_names,
        device=settings.device,
        image_size=settings.image_size,
        reference_stats_path=settings.reference_stats_path,
    )

    return PredictionResponse(
        class_name=pred.class_name,
        confidence=pred.confidence,
        probabilities=pred.probabilities,
    )
