import folium
import streamlit as st
from streamlit_folium import st_folium

from src.utils.config import settings

st.set_page_config(page_title="Disruption Map | PortWatch AI", layout="wide")
st.title("Disruption Map")

if settings.use_sample_data or not (settings.data_dir / "raw" / "portwatch").exists():
    st.warning("⚠️ Running in demo mode with synthetic sample data.")

# For the map, we need actual lat/lons. In our sample we just have string IDs,
# so we will generate random coordinates for the visualization demo
import numpy as np

m = folium.Map(location=[0, 0], zoom_start=2)
# Dummy points
for _ in range(50):
    folium.CircleMarker(
        location=[np.random.uniform(-50, 50), np.random.uniform(-100, 100)],
        radius=5,
        color="red",
        fill=True
    ).add_to(m)

st_folium(m, width=1200, height=600)
