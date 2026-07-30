import duckdb
import plotly.express as px
import streamlit as st

from src.utils.config import settings

st.set_page_config(page_title="Port Congestion | PortWatch AI", layout="wide")
st.title("Port Congestion Tracker")

if settings.use_sample_data or not (settings.data_dir / "raw" / "portwatch").exists():
    st.warning("⚠️ Running in demo mode with synthetic sample data.")

con = duckdb.connect(str(settings.data_dir / "processed" / "portwatch.duckdb"))
ports_df = con.execute("SELECT port_id, port_name FROM ports").df()

port_sel = st.selectbox("Select Port", ports_df['port_id'].tolist(), format_func=lambda x: f"{x} - {ports_df[ports_df['port_id']==x]['port_name'].values[0]}")

if port_sel:
    query = "SELECT date, daily_port_calls, congestion_index, congestion_level FROM port_activity WHERE port_id = ? ORDER BY date DESC LIMIT 90"
    data = con.execute(query, [port_sel]).df()

    st.subheader(f"90-Day Trend for {port_sel}")
    fig = px.line(data, x='date', y='daily_port_calls', title="Daily Port Calls")
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(data.head())
