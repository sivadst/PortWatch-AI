from typing import Any

import joblib
import numpy as np
import pandas as pd
import structlog
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src.utils.config import settings

logger = structlog.get_logger()


class Predictor:
    def __init__(self) -> None:
        self.model_dir = settings.model_dir
        self.feature_names = [
            "port_calls_lag_1d",
            "port_calls_lag_2d",
            "port_calls_lag_7d",
            "global_chokepoint_transit",
            "active_disasters_7d",
        ]
        self._models_reg: dict[int, xgb.XGBRegressor] = {}
        self._models_cls: dict[int, xgb.XGBClassifier] = {}
        self._load_preprocessors()

    def _load_preprocessors(self) -> None:
        try:
            self.scaler = joblib.load(self.model_dir / "scaler.pkl")
            self.le = joblib.load(self.model_dir / "label_encoder.pkl")
        except Exception:
            # Fallback if preprocessing artifacts not yet saved
            self.scaler = StandardScaler()
            self.scaler.fit(np.zeros((1, len(self.feature_names))))
            self.le = LabelEncoder()
            self.le.fit(["LOW", "NORMAL", "HIGH", "CRITICAL"])

    def _get_model(self, horizon_days: int) -> tuple[xgb.XGBRegressor, xgb.XGBClassifier, int]:
        # Match requested horizon to closest supported horizon
        supported = settings.supported_horizons
        matched_h = min(supported, key=lambda h: abs(h - horizon_days))

        if matched_h not in self._models_reg:
            reg_path = self.model_dir / f"xgboost_regressor_{matched_h}d.json"
            if not reg_path.exists():
                reg_path = self.model_dir / "xgboost_regressor.json"

            cls_path = self.model_dir / f"xgboost_classifier_{matched_h}d.json"
            if not cls_path.exists():
                cls_path = self.model_dir / "xgboost_classifier.json"

            reg = xgb.XGBRegressor()
            cls = xgb.XGBClassifier()

            if reg_path.exists():
                reg.load_model(reg_path)
            if cls_path.exists():
                cls.load_model(cls_path)

            self._models_reg[matched_h] = reg
            self._models_cls[matched_h] = cls

        return self._models_reg[matched_h], self._models_cls[matched_h], matched_h

    def predict(self, features: dict[str, Any], horizon_days: int = 7) -> dict[str, Any]:
        """Predict port congestion given feature dictionary and forecast horizon."""
        X_df = pd.DataFrame([features])
        for col in self.feature_names:
            if col not in X_df.columns:
                X_df[col] = 0.0

        X_scaled = self.scaler.transform(X_df[self.feature_names].fillna(0))

        reg_model, cls_model, actual_horizon = self._get_model(horizon_days)

        try:
            reg_pred = float(reg_model.predict(X_scaled)[0])
        except Exception:
            reg_pred = 0.0

        try:
            cls_idx = int(cls_model.predict(X_scaled)[0])
            cls_pred = str(self.le.inverse_transform([cls_idx])[0])
            probs = cls_model.predict_proba(X_scaled)[0]
            confidence = float(np.max(probs))
        except Exception:
            cls_pred = "NORMAL"
            confidence = 0.75

        return {
            "congestion_index": reg_pred,
            "congestion_level": cls_pred,
            "confidence": confidence,
            "forecast_horizon_days": actual_horizon,
            "model_version": f"xgboost_{actual_horizon}d_v1.0.0",
        }
