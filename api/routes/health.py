from fastapi import APIRouter
from pydantic import BaseModel

from src.utils.config import settings


class HealthResponse(BaseModel):
    status: str
    version: str


class ReadinessResponse(BaseModel):
    status: str
    version: str
    models_ready: bool
    data_ready: bool


router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    return HealthResponse(status="ok", version=settings.app_version)


@router.get("/ready", response_model=ReadinessResponse)
def readiness_check() -> ReadinessResponse:
    models_ready = (
        (settings.model_dir / "scaler.pkl").exists()
        and (settings.model_dir / "label_encoder.pkl").exists()
    )
    data_ready = (settings.data_dir / "processed" / "features.parquet").exists() or (
        settings.data_dir / "processed" / "portwatch.db"
    ).exists()
    status = "ready" if (models_ready or data_ready) else "initializing"

    return ReadinessResponse(
        status=status,
        version=settings.app_version,
        models_ready=models_ready,
        data_ready=data_ready,
    )
