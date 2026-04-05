from pydantic import BaseModel, Field


class ContextContributorResponse(BaseModel):
    factor: str
    impact: float = Field(..., ge=0.0, le=1.0)
    rationale: str
    raw_value: float | None = None


class ExplainabilityResponse(BaseModel):
    method: str
    overlay_base64: str | None = None
    summary: str | None = None


class EnvironmentalContextPayload(BaseModel):
    latitude: float | None = None
    longitude: float | None = None
    temperature_c: float | None = None
    humidity_pct: float | None = None
    wind_speed_kph: float | None = None
    drought_index: float | None = None
    vegetation_dryness: float | None = None
    days_since_rain: float | None = None
    temporal_sequence: list[dict[str, float]] | list[list[float]] | None = None


class PredictionResponse(BaseModel):
    class_name: str
    confidence: float
    probabilities: dict[str, float]
    image_wildfire_probability: float = Field(..., ge=0.0, le=1.0)
    context_risk_score: float | None = Field(default=None, ge=0.0, le=1.0)
    overall_risk_score: float = Field(..., ge=0.0, le=1.0)
    risk_level: str
    recommended_action: str
    environmental_context: EnvironmentalContextPayload | None = None
    top_contributors: list[ContextContributorResponse] = Field(default_factory=list)
    explainability: ExplainabilityResponse | None = None


class BatchPredictionItem(BaseModel):
    image_base64: str
    filename: str | None = None
    environmental_context: EnvironmentalContextPayload | None = None
    include_explainability: bool = False


class BatchPredictionRequest(BaseModel):
    items: list[BatchPredictionItem]


class BatchPredictionResponse(BaseModel):
    predictions: list[PredictionResponse]


class ModelInfoResponse(BaseModel):
    model_arch: str
    model_path: str
    class_names: list[str]
    version: str
    supports_tabular: bool
    supports_temporal: bool
    tabular_feature_names: list[str] = Field(default_factory=list)
    temporal_feature_names: list[str] = Field(default_factory=list)
    mlflow_tracking_uri: str


class ErrorResponse(BaseModel):
    detail: str
