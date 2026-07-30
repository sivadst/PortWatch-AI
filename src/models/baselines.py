"""Baseline forecasting models for port congestion evaluation."""

from typing import Self

import numpy as np
import pandas as pd


class PersistenceBaseline:
    """Persistence baseline: predicts future target using last observed value at t."""

    def __init__(self) -> None:
        self.last_value_: float = 0.0

    def fit(self, X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray) -> Self:
        if len(y) > 0:
            self.last_value_ = float(np.mean(y))
        return self

    def predict(self, X: pd.DataFrame | np.ndarray, lag_column: np.ndarray | None = None) -> np.ndarray:
        if lag_column is not None:
            return np.asarray(lag_column, dtype=float)
        if isinstance(X, pd.DataFrame) and "port_calls_lag_1d" in X.columns:
            return X["port_calls_lag_1d"].to_numpy(dtype=float)
        if isinstance(X, np.ndarray) and X.ndim == 2 and X.shape[1] > 0:
            return X[:, 0]
        return np.full(len(X), self.last_value_)


class RollingMeanBaseline:
    """Rolling mean baseline: predicts future target using 7-day average of historical features."""

    def __init__(self, window: int = 7) -> None:
        self.window = window
        self.mean_value_: float = 0.0

    def fit(self, X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray) -> Self:
        if len(y) > 0:
            self.mean_value_ = float(np.mean(y))
        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            lag_cols = [c for c in ["port_calls_lag_1d", "port_calls_lag_2d", "port_calls_lag_7d"] if c in X.columns]
            if lag_cols:
                return X[lag_cols].mean(axis=1).to_numpy(dtype=float)
        if isinstance(X, np.ndarray) and X.ndim == 2 and X.shape[1] >= 3:
            return np.mean(X[:, :3], axis=1)
        return np.full(len(X), self.mean_value_)


class SeasonalNaiveBaseline:
    """Seasonal naive baseline: predicts value using 7-day lag (weekly cycle)."""

    def __init__(self, season_lag: int = 7) -> None:
        self.season_lag = season_lag
        self.fallback_: float = 0.0

    def fit(self, X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray) -> Self:
        if len(y) > 0:
            self.fallback_ = float(np.mean(y))
        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        if isinstance(X, pd.DataFrame) and "port_calls_lag_7d" in X.columns:
            return X["port_calls_lag_7d"].to_numpy(dtype=float)
        if isinstance(X, np.ndarray) and X.ndim == 2 and X.shape[1] >= 3:
            return X[:, 2]
        return np.full(len(X), self.fallback_)
