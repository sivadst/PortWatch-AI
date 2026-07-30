import streamlit as st

st.set_page_config(page_title="About | PortWatch AI", layout="wide")
st.title("About PortWatch AI")

st.markdown("""
### Project Methodology
PortWatch AI uses machine learning to forecast port congestion and supply chain disruptions up to 14 days ahead, integrating various data sources.

### Data Sources
* IMF PortWatch
* UNCTAD LSCI
* World Bank LPI 2.0
* GDACS Disaster Alerts

### Limitations
Forecasts are research-grade and not intended for immediate operational routing without human validation.
Data may experience lags, and unpredicted global events (e.g. pandemics) may not be captured by historical models.
""")
