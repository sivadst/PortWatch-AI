from datetime import date
from enum import Enum

from pydantic import BaseModel, Field


class CongestionLevel(str, Enum):
    LOW = "LOW"
    NORMAL = "NORMAL"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class FeatureDriver(BaseModel):
    feature: str
    shap_value: float
    direction: str


class PredictRequest(BaseModel):
    port_id: str = Field(..., min_length=3, max_length=15, description="Port identifier e.g. PORT001")
    forecast_horizon_days: int = Field(7, ge=1, le=14, description="Forecasting horizon in days (1, 7, or 14)")
    include_explanation: bool = True


class PredictResponse(BaseModel):
    port_id: str
    forecast_date: date
    congestion_index: float
    congestion_level: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    explanation: list[FeatureDriver] | None = None
    model_version: str
    request_id: str


class BatchPredictRequest(BaseModel):
    requests: list[PredictRequest] = Field(..., max_length=100)


class ErrorResponse(BaseModel):
    error_code: str
    message: str
    request_id: str
    timestamp: str
