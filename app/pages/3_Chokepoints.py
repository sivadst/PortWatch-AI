import duckdb
import plotly.express as px
import streamlit as st

from src.utils.config import settings

st.set_page_config(page_title="Chokepoints | PortWatch AI", layout="wide")
st.title("Chokepoint Monitoring")

if settings.use_sample_data or not (settings.data_dir / "raw" / "portwatch").exists():
    st.warning("⚠️ Running in demo mode with synthetic sample data.")

con = duckdb.connect(str(settings.data_dir / "processed" / "portwatch.duckdb"))
query = "SELECT date, global_chokepoint_transit FROM port_activity GROUP BY date, global_chokepoint_transit ORDER BY date DESC LIMIT 90"
data = con.execute(query).df()

st.subheader("Global Chokepoint Transit Trend (Last 90 Days)")
fig = px.line(data, x='date', y='global_chokepoint_transit', title="Transit Volume")
st.plotly_chart(fig, use_container_width=True)
