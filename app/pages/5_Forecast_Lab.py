import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Add project root to path for imports
sys.path.append(str(Path(__file__).resolve().parents[2]))

from api.services.data_service import data_service
from src.models.explainability import ModelExplainer
from src.models.predict import Predictor

st.set_page_config(page_title="Forecast Lab | PortWatch AI", layout="wide")
st.title("Forecast Lab — Multi-Horizon Port Congestion Forecasting")

predictor = Predictor()
explainer = ModelExplainer()

ports_list = data_service.get_ports()

if not ports_list:
    st.warning("⚠️ No processed port data found. Generating interactive demo ports...")
    ports_df = pd.DataFrame([
        {"port_id": f"PORT{str(i).zfill(3)}", "port_name": f"Port of PORT{str(i).zfill(3)}"}
        for i in range(1, 11)
    ])
else:
    ports_df = pd.DataFrame(ports_list)

col_sel1, col_sel2 = st.columns(2)

with col_sel1:
    port_sel = st.selectbox("Select Port to Forecast", ports_df["port_id"].tolist())

with col_sel2:
    horizon_sel = st.selectbox("Select Forecasting Horizon", [1, 7, 14], index=1)

if st.button("Generate Forecast", type="primary"):
    features = data_service.get_port_features(port_sel)
    if not features:
        features = {
            "port_calls_lag_1d": 42.0,
            "port_calls_lag_2d": 40.0,
            "port_calls_lag_7d": 45.0,
            "global_chokepoint_transit": 120.0,
            "active_disasters_7d": 0.0,
        }

    pred = predictor.predict(features, horizon_days=horizon_sel)

    # Explanation dataframe
    feature_names = [
        "port_calls_lag_1d",
        "port_calls_lag_2d",
        "port_calls_lag_7d",
        "global_chokepoint_transit",
        "active_disasters_7d",
    ]
    X_df = pd.DataFrame([features])
    for col in feature_names:
        if col not in X_df.columns:
            X_df[col] = 0.0

    X_scaled = predictor.scaler.transform(X_df[feature_names].fillna(0))
    raw_exp = explainer.explain_prediction(pd.DataFrame(X_scaled, columns=feature_names), horizon_days=horizon_sel)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Forecast Summary")
        st.metric("Forecast Horizon Target", f"{pred['forecast_horizon_days']} Days Ahead")
        st.metric("Predicted Congestion Index", f"{pred['congestion_index']:.2f}")

        level = pred["congestion_level"]
        color = "🟢" if level == "LOW" else "🟡" if level == "NORMAL" else "🟠" if level == "HIGH" else "🔴"
        st.metric("Risk Category", f"{color} {level}")
        st.metric("Model Confidence", f"{pred['confidence']:.0%}")
        st.caption(f"Model Version: `{pred['model_version']}`")

    with col2:
        st.subheader("Top Feature Drivers (SHAP Explanations)")
        exp_df = pd.DataFrame(raw_exp)
        st.dataframe(exp_df, use_container_width=True)
