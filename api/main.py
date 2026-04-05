import asyncio
import base64
import json
import logging
from contextlib import asynccontextmanager
from io import BytesIO

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from PIL import Image

from wildfire_mlops.constants import SUPPORTED_CONTENT_TYPES
from wildfire_mlops.core import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    ErrorResponse,
    ModelInfoResponse,
    PredictionResponse,
    get_settings,
    setup_logging,
)
from wildfire_mlops.inference import (
    EnvironmentalFeatures,
    PredictionError,
    predict_image,
)
from wildfire_mlops.modeling import load_checkpoint
from wildfire_mlops.version import __version__

settings = get_settings()
setup_logging(settings.log_level)
logger = logging.getLogger("api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(
        "Loading model",
        extra={
            "model_arch": settings.model_arch,
            "model_path": settings.resolve_model_path(),
            "device": settings.device,
        },
    )
    loaded_model, loaded_class_names = load_checkpoint(
        settings.resolve_model_path(),
        model_arch=settings.model_arch,
        pretrained=settings.pretrained,
    )
    app.state.model = loaded_model
    app.state.class_names = loaded_class_names
    logger.info("Model loaded and ready for inference")
    yield


app = FastAPI(
    title="Wildfire Risk Intelligence API",
    version=__version__,
    description="Production-style wildfire screening API with multimodal inference support.",
    lifespan=lifespan,
)


def _get_model_bundle(request: Request) -> tuple[object, list[str]]:
    if not hasattr(request.app.state, "model") or not hasattr(request.app.state, "class_names"):
        loaded_model, loaded_class_names = load_checkpoint(
            settings.resolve_model_path(),
            model_arch=settings.model_arch,
            pretrained=settings.pretrained,
        )
        request.app.state.model = loaded_model
        request.app.state.class_names = loaded_class_names
    return request.app.state.model, request.app.state.class_names


def _build_environmental_features(
    latitude: float | None = None,
    longitude: float | None = None,
    temperature_c: float | None = None,
    humidity_pct: float | None = None,
    wind_speed_kph: float | None = None,
    drought_index: float | None = None,
    vegetation_dryness: float | None = None,
    days_since_rain: float | None = None,
) -> EnvironmentalFeatures:
    return EnvironmentalFeatures(
        latitude=latitude,
        longitude=longitude,
        temperature_c=temperature_c,
        humidity_pct=humidity_pct,
        wind_speed_kph=wind_speed_kph,
        drought_index=drought_index,
        vegetation_dryness=vegetation_dryness,
        days_since_rain=days_since_rain,
    )


def _parse_temporal_context(
    raw_value: str | None,
) -> list[dict[str, float]] | list[list[float]] | None:
    if not raw_value:
        return None
    try:
        parsed = json.loads(raw_value)
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=400, detail="Invalid temporal_context_json payload"
        ) from exc
    if not isinstance(parsed, list):
        raise HTTPException(status_code=400, detail="Temporal context must be a JSON list")
    return parsed


def _load_pil_image(data: bytes) -> Image.Image:
    try:
        return Image.open(BytesIO(data)).convert("RGB")
    except Exception as exc:
        logger.exception("Failed to load image bytes")
        raise HTTPException(status_code=400, detail="Invalid image") from exc


def _prediction_response_from_result(prediction) -> PredictionResponse:
    return PredictionResponse(
        class_name=prediction.class_name,
        confidence=prediction.confidence,
        probabilities=prediction.probabilities,
        image_wildfire_probability=prediction.image_wildfire_probability,
        context_risk_score=prediction.context_risk_score,
        overall_risk_score=prediction.overall_risk_score,
        risk_level=prediction.risk_level,
        recommended_action=prediction.recommended_action,
        environmental_context=(
            {
                **prediction.environmental_context.to_mapping(),
                "temporal_sequence": None,
            }
            if prediction.environmental_context is not None
            else None
        ),
        top_contributors=[
            {
                "factor": contributor.factor,
                "impact": contributor.impact,
                "rationale": contributor.rationale,
                "raw_value": contributor.raw_value,
            }
            for contributor in prediction.top_contributors
        ],
        explainability=(
            {
                "method": prediction.explainability.method,
                "overlay_base64": prediction.explainability.overlay_base64,
                "summary": prediction.explainability.summary,
            }
            if prediction.explainability is not None
            else None
        ),
    )


def _run_prediction(
    request: Request,
    image: Image.Image,
    environmental_features: EnvironmentalFeatures | None,
    temporal_context: list[dict[str, float]] | list[list[float]] | None,
    include_explainability: bool,
) -> PredictionResponse:
    model, class_names = _get_model_bundle(request)
    prediction = predict_image(
        image=image,
        model=model,
        class_names=class_names,
        device=settings.device,
        image_size=settings.image_size,
        reference_stats_path=settings.reference_stats_path,
        environmental_features=environmental_features,
        temporal_sequence=temporal_context,
        include_explainability=include_explainability,
    )
    return _prediction_response_from_result(prediction)


@app.exception_handler(PredictionError)
async def prediction_error_handler(_: Request, exc: PredictionError) -> JSONResponse:
    return JSONResponse(status_code=400, content=ErrorResponse(detail=str(exc)).model_dump())


@app.exception_handler(RequestValidationError)
async def validation_error_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
    return JSONResponse(status_code=422, content=ErrorResponse(detail=str(exc)).model_dump())


@app.exception_handler(Exception)
async def unhandled_error_handler(_: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled API error")
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            detail=f"Internal server error: {exc.__class__.__name__}"
        ).model_dump(),
    )


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "version": __version__,
        "model_arch": settings.model_arch,
        "model_loaded": hasattr(app.state, "model"),
    }


@app.get("/model-info", response_model=ModelInfoResponse)
def model_info(request: Request) -> ModelInfoResponse:
    model, class_names = _get_model_bundle(request)
    return ModelInfoResponse(
        model_arch=settings.model_arch,
        model_path=settings.resolve_model_path(),
        class_names=class_names,
        version=settings.model_version,
        supports_tabular=bool(getattr(model, "expects_tabular", False)),
        supports_temporal=bool(getattr(model, "expects_temporal", False)),
        tabular_feature_names=list(
            getattr(model, "tabular_feature_names", []) or settings.tabular_feature_names
        ),
        temporal_feature_names=list(
            getattr(model, "temporal_feature_names", []) or settings.temporal_feature_names
        ),
        mlflow_tracking_uri=settings.mlflow_tracking_uri,
    )


@app.post(
    "/predict",
    response_model=PredictionResponse,
    responses={400: {"model": ErrorResponse}, 422: {"model": ErrorResponse}},
)
async def predict(
    request: Request,
    file: UploadFile = File(...),
    latitude: float | None = Form(default=None),
    longitude: float | None = Form(default=None),
    temperature_c: float | None = Form(default=None),
    humidity_pct: float | None = Form(default=None),
    wind_speed_kph: float | None = Form(default=None),
    drought_index: float | None = Form(default=None),
    vegetation_dryness: float | None = Form(default=None),
    days_since_rain: float | None = Form(default=None),
    temporal_context_json: str | None = Form(default=None),
    include_explainability: bool = Form(default=False),
) -> PredictionResponse:
    if file.content_type not in SUPPORTED_CONTENT_TYPES:
        raise HTTPException(status_code=400, detail="Unsupported file type")

    data = await file.read()
    image = _load_pil_image(data)
    temporal_context = _parse_temporal_context(temporal_context_json)
    environmental_features = _build_environmental_features(
        latitude=latitude,
        longitude=longitude,
        temperature_c=temperature_c,
        humidity_pct=humidity_pct,
        wind_speed_kph=wind_speed_kph,
        drought_index=drought_index,
        vegetation_dryness=vegetation_dryness,
        days_since_rain=days_since_rain,
    )
    return _run_prediction(
        request=request,
        image=image,
        environmental_features=environmental_features,
        temporal_context=temporal_context,
        include_explainability=include_explainability,
    )


@app.post(
    "/predict-batch",
    response_model=BatchPredictionResponse,
    responses={400: {"model": ErrorResponse}, 422: {"model": ErrorResponse}},
)
async def predict_batch(
    request: Request,
    batch_request: BatchPredictionRequest,
) -> BatchPredictionResponse:
    if not batch_request.items:
        raise HTTPException(status_code=400, detail="Batch request must contain at least one item")

    semaphore = asyncio.Semaphore(4)

    async def run_item(item) -> PredictionResponse:
        async with semaphore:
            try:
                image_bytes = base64.b64decode(item.image_base64)
            except Exception as exc:
                raise HTTPException(status_code=400, detail="Invalid base64 image payload") from exc

            image = _load_pil_image(image_bytes)
            context = item.environmental_context
            environmental_features = (
                _build_environmental_features(
                    latitude=context.latitude,
                    longitude=context.longitude,
                    temperature_c=context.temperature_c,
                    humidity_pct=context.humidity_pct,
                    wind_speed_kph=context.wind_speed_kph,
                    drought_index=context.drought_index,
                    vegetation_dryness=context.vegetation_dryness,
                    days_since_rain=context.days_since_rain,
                )
                if context is not None
                else None
            )

            return await asyncio.to_thread(
                _run_prediction,
                request,
                image,
                environmental_features,
                context.temporal_sequence if context is not None else None,
                item.include_explainability,
            )

    predictions = await asyncio.gather(*(run_item(item) for item in batch_request.items))
    return BatchPredictionResponse(predictions=predictions)
