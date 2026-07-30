import pytest
from pydantic import ValidationError

from api.schemas.predictions import PredictRequest


def test_predict_schema_valid():
    req = PredictRequest(port_id="PORT001", forecast_horizon_days=7, include_explanation=True)
    assert req.port_id == "PORT001"
    assert req.forecast_horizon_days == 7


def test_predict_schema_invalid():
    with pytest.raises(ValidationError):
        PredictRequest(port_id="P1", forecast_horizon_days=20)  # min len 3, max horizon 14


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_ready_endpoint(client):
    response = client.get("/ready")
    assert response.status_code == 200
    assert "status" in response.json()
