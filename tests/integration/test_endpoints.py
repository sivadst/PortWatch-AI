from api.services.data_service import DataService


def test_predict_endpoint_not_found(client, monkeypatch):
    def mock_get_features(self, port_id):
        return {}

    monkeypatch.setattr(DataService, "get_port_features", mock_get_features)

    response = client.post(
        "/predict",
        json={"port_id": "MISSING_PORT", "forecast_horizon_days": 7, "include_explanation": False},
    )
    assert response.status_code == 404
    data = response.json()
    assert data["error_code"] == "HTTP_404"


def test_predict_endpoint_success(client, monkeypatch):
    def mock_get_features(self, port_id):
        if port_id == "PORT_TEST":
            return {
                "port_calls_lag_1d": 100.0,
                "port_calls_lag_2d": 90.0,
                "port_calls_lag_7d": 95.0,
                "global_chokepoint_transit": 50.0,
                "active_disasters_7d": 0.0,
            }
        return {}

    monkeypatch.setattr(DataService, "get_port_features", mock_get_features)

    response = client.post(
        "/predict",
        json={"port_id": "PORT_TEST", "forecast_horizon_days": 7, "include_explanation": True},
    )
    assert response.status_code == 200
    data = response.json()
    assert "congestion_index" in data
    assert "explanation" in data
    assert len(data["explanation"]) > 0


def test_batch_predict_endpoint(client, monkeypatch):
    def mock_get_features(self, port_id):
        return {
            "port_calls_lag_1d": 50.0,
            "port_calls_lag_2d": 40.0,
            "port_calls_lag_7d": 45.0,
            "global_chokepoint_transit": 100.0,
            "active_disasters_7d": 0.0,
        }

    monkeypatch.setattr(DataService, "get_port_features", mock_get_features)

    payload = {
        "requests": [
            {"port_id": "PORT001", "forecast_horizon_days": 1},
            {"port_id": "PORT002", "forecast_horizon_days": 7},
            {"port_id": "PORT003", "forecast_horizon_days": 14},
        ]
    }
    response = client.post("/batch-predict", json=payload)
    assert response.status_code == 200
    results = response.json()
    assert len(results) == 3
