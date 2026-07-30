import duckdb
import pandas as pd
import streamlit as st

from src.utils.config import settings

st.set_page_config(page_title="Overview | PortWatch AI", layout="wide")
st.title("Global Port Overview")

if settings.use_sample_data or not (settings.data_dir / "raw" / "portwatch").exists():
    st.warning("⚠️ Running in demo mode with synthetic sample data.")

@st.cache_data(ttl=3600)
def get_overview_data() -> pd.DataFrame:
    try:
        con = duckdb.connect(str(settings.data_dir / "processed" / "portwatch.duckdb"))
        query = """
            SELECT p.port_id, p.port_name, p.country_code, a.date, a.congestion_level, a.congestion_index, a.daily_port_calls
            FROM ports p
            JOIN port_activity a ON p.port_id = a.port_id
            WHERE a.date = (SELECT MAX(date) FROM port_activity)
        """
        df = con.execute(query).df()
        return df
    except Exception:
        return pd.DataFrame()

df = get_overview_data()
if df.empty:
    st.error("No data found.")
else:
    st.dataframe(df, use_container_width=True)
