import uuid
from datetime import datetime, timedelta

import pandas as pd
import structlog
from fastapi import APIRouter, HTTPException, Request

from api.schemas.predictions import (
    BatchPredictRequest,
    FeatureDriver,
    PredictRequest,
    PredictResponse,
)
from api.services.data_service import data_service
from src.models.explainability import ModelExplainer
from src.models.predict import Predictor

logger = structlog.get_logger()

router = APIRouter()
predictor = Predictor()
explainer = ModelExplainer()


@router.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest, req: Request) -> PredictResponse:
    request_id = str(uuid.uuid4())

    features = data_service.get_port_features(request.port_id)
    if not features:
        raise HTTPException(
            status_code=404,
            detail=f"Port '{request.port_id}' not found or no activity data available",
        )

    pred = predictor.predict(features, horizon_days=request.forecast_horizon_days)

    explanation = None
    if request.include_explanation:
        feature_names = [
            "port_calls_lag_1d",
            "port_calls_lag_2d",
            "port_calls_lag_7d",
            "global_chokepoint_transit",
            "active_disasters_7d",
        ]
        X = pd.DataFrame([features])
        for col in feature_names:
            if col not in X.columns:
                X[col] = 0.0
        X_scaled = predictor.scaler.transform(X[feature_names].fillna(0))
        X_df = pd.DataFrame(X_scaled, columns=feature_names)

        raw_exp = explainer.explain_prediction(X_df, horizon_days=request.forecast_horizon_days)
        explanation = [FeatureDriver(**e) for e in raw_exp]

    actual_horizon = pred.get("forecast_horizon_days", request.forecast_horizon_days)
    target_date = datetime.now().date() + timedelta(days=actual_horizon)

    return PredictResponse(
        port_id=request.port_id,
        forecast_date=target_date,
        congestion_index=pred["congestion_index"],
        congestion_level=pred["congestion_level"],
        confidence=pred["confidence"],
        explanation=explanation,
        model_version=pred["model_version"],
        request_id=request_id,
    )


@router.post("/batch-predict", response_model=list[PredictResponse])
async def batch_predict(
    request: BatchPredictRequest, req: Request
) -> list[PredictResponse]:
    results = []
    for r in request.requests:
        try:
            res = await predict(r, req)
            results.append(res)
        except HTTPException:
            continue
        except Exception as e:
            logger.warning(f"Batch prediction error for port {r.port_id}: {e}")
            continue
    return results
