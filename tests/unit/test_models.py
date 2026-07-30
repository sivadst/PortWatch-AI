from src.models.explainability import ModelExplainer
from src.models.predict import Predictor


def test_predictor_default_prediction():
    predictor = Predictor()
    features = {
        "port_calls_lag_1d": 50.0,
        "port_calls_lag_2d": 45.0,
        "port_calls_lag_7d": 48.0,
        "global_chokepoint_transit": 120.0,
        "active_disasters_7d": 0.0,
    }
    res = predictor.predict(features, horizon_days=7)
    assert "congestion_index" in res
    assert "congestion_level" in res
    assert res["forecast_horizon_days"] == 7
    assert 0.0 <= res["confidence"] <= 1.0


def test_explainer_fallback():
    explainer = ModelExplainer()
    import pandas as pd
    X_scaled = pd.DataFrame([[0.0, 0.0, 0.0, 0.0, 0.0]])
    exps = explainer.explain_prediction(X_scaled, horizon_days=7)
    assert isinstance(exps, list)
    assert len(exps) > 0
    assert "feature" in exps[0]
    assert "direction" in exps[0]
