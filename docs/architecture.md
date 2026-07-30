# Architecture Document

This project features a pipeline composed of:
1. **Data Engineering:** Extracting public trade datasets from the IMF, UNCTAD, World Bank, and GDACS. Cleaned using Pandas and loaded into DuckDB or PostgreSQL.
2. **Machine Learning:** Features engineered into a time-series view are fed to an XGBoost multi-output model tracking congestion indices and categorical congestion levels. Interpretability is added via SHAP.
3. **API Layer:** FastAPI provides a JSON REST interface, tracking model predictions on a per-port basis.
4. **App Layer:** Streamlit consumes the FastAPI metrics and DuckDB cache to visualize congestion maps and forecast details.

See the README.md for a mermaid flowchart representing the system.
