import sqlite3
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Add project root to path for imports
sys.path.append(str(Path(__file__).resolve().parents[1]))

try:
    import duckdb
except ImportError:
    duckdb = None

from src.utils.config import settings

st.set_page_config(page_title="PortWatch AI", page_icon="🚢", layout="wide")


def load_data() -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    duck_path = settings.data_dir / "processed" / "portwatch.duckdb"
    sqlite_path = settings.data_dir / "processed" / "portwatch.db"
    parquet_path = settings.data_dir / "processed" / "features.parquet"

    # 1. Try DuckDB
    if duckdb is not None and duck_path.exists():
        try:
            con = duckdb.connect(str(duck_path))
            ports = con.execute("SELECT * FROM ports").df()
            activity = con.execute("SELECT * FROM port_activity ORDER BY date DESC LIMIT 1000").df()
            con.close()
            return ports, activity
        except Exception:
            pass

    # 2. Try SQLite
    if sqlite_path.exists():
        try:
            conn = sqlite3.connect(sqlite_path)
            ports = pd.read_sql_query("SELECT * FROM ports", conn)
            activity = pd.read_sql_query("SELECT * FROM port_activity ORDER BY date DESC LIMIT 1000", conn)
            conn.close()
            return ports, activity
        except Exception:
            pass

    # 3. Try Parquet
    if parquet_path.exists():
        try:
            df = pd.read_parquet(parquet_path)
            ports = df[["port_id", "port_name", "country_code"]].drop_duplicates()
            activity = df.sort_values("date", ascending=False).head(1000)
            return ports, activity
        except Exception:
            pass

    return None, None


def main() -> None:
    if settings.use_sample_data or not (settings.data_dir / "raw" / "portwatch").exists():
        st.warning("⚠️ Running with synthetic sample data for demonstration.")

    st.title("PortWatch AI — Global Supply Chain Intelligence")
    st.markdown("Satellite-derived port congestion forecasting and trade flow risk monitoring.")

    ports, activity = load_data()
    if ports is None or activity is None:
        st.info("Pipeline initializing. Please run `make data` and `make train` to generate datasets.")
        return

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Ports Monitored", len(ports))
    with col2:
        st.metric("Global Congestion Avg", f"{activity['congestion_index'].mean():.2f}")
    with col3:
        st.metric("Active Disruption Alerts", int(activity['active_disasters_7d'].sum()))
    with col4:
        st.metric("Supported Forecast Horizons", "1d | 7d | 14d")

    st.markdown("### Recent Monitored Port Activity")
    st.dataframe(activity.head(10), use_container_width=True)

    # Show model benchmark report if available
    bench_file = Path("reports/benchmark_results.csv")
    if bench_file.exists():
        st.markdown("### Model Benchmark Comparison")
        bench_df = pd.read_csv(bench_file)
        st.dataframe(bench_df, use_container_width=True)


if __name__ == "__main__":
    main()
