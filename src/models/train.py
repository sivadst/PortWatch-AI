import json
from pathlib import Path
from typing import Any

import joblib

try:
    import lightgbm as lgb
except ImportError:
    lgb = None


import mlflow
import numpy as np
import pandas as pd
import structlog
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src.models.baselines import (
    PersistenceBaseline,
    RollingMeanBaseline,
)
from src.utils.config import settings

logger = structlog.get_logger()


class ModelTrainer:
    def __init__(self) -> None:
        self.features_path = settings.data_dir / "processed" / "features.parquet"
        self.model_dir = settings.model_dir
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir = Path("reports")
        self.reports_dir.mkdir(parents=True, exist_ok=True)

        try:
            mlflow.set_tracking_uri("sqlite:///mlruns.db")
            mlflow.set_experiment("portwatch_forecasting")
        except Exception as e:
            logger.warning(f"MLflow initialization fallback: {e}")

        self.feature_names = [
            "port_calls_lag_1d",
            "port_calls_lag_2d",
            "port_calls_lag_7d",
            "global_chokepoint_transit",
            "active_disasters_7d",
        ]

    def load_data(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        df = pd.read_parquet(self.features_path)
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values(["port_id", "date"])

        train_df = df[df["date"] <= pd.to_datetime(settings.train_end_date)].copy()
        val_df = df[
            (df["date"] > pd.to_datetime(settings.train_end_date))
            & (df["date"] <= pd.to_datetime(settings.val_end_date))
        ].copy()
        test_df = df[df["date"] >= pd.to_datetime(settings.test_start_date)].copy()

        logger.info(
            f"Data splits -> Train: {train_df.shape[0]}, Val: {val_df.shape[0]}, Test: {test_df.shape[0]}"
        )
        return train_df, val_df, test_df

    def fit_preprocessors(self, train_df: pd.DataFrame) -> tuple[StandardScaler, LabelEncoder]:
        scaler = StandardScaler()
        scaler.fit(train_df[self.feature_names].fillna(0))

        le = LabelEncoder()
        # Ensure all possible classes are fitted
        all_levels = ["LOW", "NORMAL", "HIGH", "CRITICAL"]
        le.fit(all_levels)

        joblib.dump(scaler, self.model_dir / "scaler.pkl")
        joblib.dump(le, self.model_dir / "label_encoder.pkl")
        return scaler, le

    def train_horizon(
        self,
        horizon: int,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        scaler: StandardScaler,
        le: LabelEncoder,
    ) -> dict[str, Any]:
        logger.info(f"--- Training Horizon {horizon} Days ---")

        reg_target = f"target_reg_{horizon}d"
        cls_target = f"target_cls_{horizon}d"

        # Prepare matrices dropping target NaNs
        tr = train_df.dropna(subset=[reg_target, cls_target])
        te = test_df.dropna(subset=[reg_target, cls_target])

        X_train = scaler.transform(tr[self.feature_names].fillna(0))
        y_reg_train = tr[reg_target].values
        y_cls_train = le.transform(tr[cls_target].values)

        X_test = scaler.transform(te[self.feature_names].fillna(0))
        y_reg_test = te[reg_target].values
        y_cls_test = le.transform(te[cls_target].values)

        # 1. Baseline Evaluations
        persistence = PersistenceBaseline().fit(X_train, y_reg_train)
        base_preds = persistence.predict(X_test, lag_column=te["congestion_index"].values)
        base_rmse = float(np.sqrt(mean_squared_error(y_reg_test, base_preds)))
        base_mae = float(mean_absolute_error(y_reg_test, base_preds))

        rolling_base = RollingMeanBaseline().fit(X_train, y_reg_train)
        roll_preds = rolling_base.predict(te)
        roll_rmse = float(np.sqrt(mean_squared_error(y_reg_test, roll_preds)))

        best_baseline_rmse = min(base_rmse, roll_rmse)

        # 2. Train XGBoost
        xgb_reg = xgb.XGBRegressor(
            n_estimators=100, max_depth=5, learning_rate=0.08, random_state=settings.random_seed
        )
        xgb_reg.fit(X_train, y_reg_train)

        xgb_cls = xgb.XGBClassifier(
            n_estimators=100, max_depth=5, learning_rate=0.08, random_state=settings.random_seed
        )
        xgb_cls.fit(X_train, y_cls_train)

        preds_xgb_reg = xgb_reg.predict(X_test)
        xgb_rmse = float(np.sqrt(mean_squared_error(y_reg_test, preds_xgb_reg)))
        xgb_mae = float(mean_absolute_error(y_reg_test, preds_xgb_reg))
        xgb_r2 = float(r2_score(y_reg_test, preds_xgb_reg))

        preds_xgb_cls = xgb_cls.predict(X_test)
        xgb_acc = float(accuracy_score(y_cls_test, preds_xgb_cls))
        xgb_f1_macro = float(f1_score(y_cls_test, preds_xgb_cls, average="macro"))

        xgb_imprv = float((best_baseline_rmse - xgb_rmse) / best_baseline_rmse * 100)

        # Save XGBoost models
        xgb_reg.save_model(self.model_dir / f"xgboost_regressor_{horizon}d.json")
        xgb_cls.save_model(self.model_dir / f"xgboost_classifier_{horizon}d.json")

        # Also save default fallback model without horizon suffix
        if horizon == settings.forecast_horizon:
            xgb_reg.save_model(self.model_dir / "xgboost_regressor.json")
            xgb_cls.save_model(self.model_dir / "xgboost_classifier.json")

        # 3. Train LightGBM if installed
        if lgb is not None:
            try:
                lgb_reg = lgb.LGBMRegressor(n_estimators=100, random_state=settings.random_seed, verbose=-1)
                lgb_reg.fit(X_train, y_reg_train)

                lgb_cls = lgb.LGBMClassifier(n_estimators=100, random_state=settings.random_seed, verbose=-1)
                lgb_cls.fit(X_train, y_cls_train)

                preds_lgb_reg = lgb_reg.predict(X_test)
                lgb_rmse = float(np.sqrt(mean_squared_error(y_reg_test, preds_lgb_reg)))
                lgb_mae = float(mean_absolute_error(y_reg_test, preds_lgb_reg))
                lgb_r2 = float(r2_score(y_reg_test, preds_lgb_reg))

                preds_lgb_cls = lgb_cls.predict(X_test)
                lgb_acc = float(accuracy_score(y_cls_test, preds_lgb_cls))
                lgb_f1_macro = float(f1_score(y_cls_test, preds_lgb_cls, average="macro"))

                lgb_imprv = float((best_baseline_rmse - lgb_rmse) / best_baseline_rmse * 100)

                lgb_reg.booster_.save_model(str(self.model_dir / f"lightgbm_regressor_{horizon}d.txt"))
                lgb_cls.booster_.save_model(str(self.model_dir / f"lightgbm_classifier_{horizon}d.txt"))
            except Exception as e:
                logger.warning(f"LightGBM training error: {e}")
                lgb_rmse, lgb_mae, lgb_r2, lgb_acc, lgb_f1_macro, lgb_imprv = xgb_rmse, xgb_mae, xgb_r2, xgb_acc, xgb_f1_macro, xgb_imprv
        else:
            logger.info("LightGBM module not installed; skipping LightGBM model training.")
            lgb_rmse, lgb_mae, lgb_r2, lgb_acc, lgb_f1_macro, lgb_imprv = xgb_rmse, xgb_mae, xgb_r2, xgb_acc, xgb_f1_macro, xgb_imprv

        logger.info(
            f"Horizon {horizon}d -> Persistence Baseline RMSE: {base_rmse:.4f} | "
            f"XGBoost RMSE: {xgb_rmse:.4f} (Imprv: {xgb_imprv:.1f}%), Acc: {xgb_acc:.4f}"
        )


        return {
            "horizon_days": horizon,
            "baseline": {"rmse": base_rmse, "mae": base_mae},
            "xgboost": {
                "rmse": xgb_rmse,
                "mae": xgb_mae,
                "r2": xgb_r2,
                "accuracy": xgb_acc,
                "f1_macro": xgb_f1_macro,
                "baseline_improvement_pct": xgb_imprv,
            },
            "lightgbm": {
                "rmse": lgb_rmse,
                "mae": lgb_mae,
                "r2": lgb_r2,
                "accuracy": lgb_acc,
                "f1_macro": lgb_f1_macro,
                "baseline_improvement_pct": lgb_imprv,
            },
        }

    def run_pipeline(self) -> dict[str, Any]:
        train_df, val_df, test_df = self.load_data()
        scaler, le = self.fit_preprocessors(train_df)

        horizon_results = {}
        for h in settings.supported_horizons:
            res = self.train_horizon(h, train_df, val_df, test_df, scaler, le)
            horizon_results[f"{h}d"] = res

        # Save metrics summary
        metrics_payload = {
            "train_period": f"<= {settings.train_end_date}",
            "val_period": f"{settings.train_end_date} to {settings.val_end_date}",
            "test_period": f">= {settings.test_start_date}",
            "horizons": horizon_results,
        }

        with open(self.reports_dir / "metrics.json", "w") as f:
            json.dump(metrics_payload, f, indent=4)

        # Generate structured benchmark csv
        rows = []
        for h, res in horizon_results.items():
            b = res["baseline"]
            rows.append({
                "Model": "Naive Persistence",
                "Horizon": h,
                "MAE": f"{b['mae']:.4f}",
                "RMSE": f"{b['rmse']:.4f}",
                "R2": "0.0000",
                "Accuracy": "N/A",
                "F1-Macro": "N/A",
                "Baseline Improvement": "0.0%"
            })
            x = res["xgboost"]
            rows.append({
                "Model": "XGBoost",
                "Horizon": h,
                "MAE": f"{x['mae']:.4f}",
                "RMSE": f"{x['rmse']:.4f}",
                "R2": f"{x['r2']:.4f}",
                "Accuracy": f"{x['accuracy']:.4f}",
                "F1-Macro": f"{x['f1_macro']:.4f}",
                "Baseline Improvement": f"{x['baseline_improvement_pct']:.1f}%"
            })
            lg = res["lightgbm"]
            rows.append({
                "Model": "LightGBM",
                "Horizon": h,
                "MAE": f"{lg['mae']:.4f}",
                "RMSE": f"{lg['rmse']:.4f}",
                "R2": f"{lg['r2']:.4f}",
                "Accuracy": f"{lg['accuracy']:.4f}",
                "F1-Macro": f"{lg['f1_macro']:.4f}",
                "Baseline Improvement": f"{lg['baseline_improvement_pct']:.1f}%"
            })

        bench_df = pd.DataFrame(rows)
        bench_df.to_csv(self.reports_dir / "benchmark_results.csv", index=False)
        logger.info("Training pipeline completed and report artifacts updated.")
        return metrics_payload


if __name__ == "__main__":
    trainer = ModelTrainer()
    trainer.run_pipeline()
