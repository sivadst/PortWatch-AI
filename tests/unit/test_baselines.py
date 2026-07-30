import numpy as np
import pandas as pd

from src.models.baselines import (
    PersistenceBaseline,
    RollingMeanBaseline,
    SeasonalNaiveBaseline,
)


def test_persistence_baseline():
    base = PersistenceBaseline()
    base.fit(np.zeros((10, 2)), np.array([1.0] * 10))

    X_df = pd.DataFrame({"port_calls_lag_1d": [5.0, 10.0, 15.0]})
    preds = base.predict(X_df)
    assert np.array_equal(preds, np.array([5.0, 10.0, 15.0]))


def test_rolling_mean_baseline():
    base = RollingMeanBaseline(window=7)
    base.fit(np.zeros((10, 2)), np.array([1.0] * 10))

    X_df = pd.DataFrame({
        "port_calls_lag_1d": [10.0, 20.0],
        "port_calls_lag_2d": [20.0, 30.0],
        "port_calls_lag_7d": [30.0, 40.0],
    })
    preds = base.predict(X_df)
    assert np.allclose(preds, np.array([20.0, 30.0]))


def test_seasonal_naive_baseline():
    base = SeasonalNaiveBaseline()
    base.fit(np.zeros((10, 2)), np.array([1.0] * 10))

    X_df = pd.DataFrame({"port_calls_lag_7d": [100.0, 200.0]})
    preds = base.predict(X_df)
    assert np.array_equal(preds, np.array([100.0, 200.0]))
