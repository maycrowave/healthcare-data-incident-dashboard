import sys
sys.path.append("..")

import streamlit as st
import pandas as pd
import numpy as np
from src.constants import (PROCESSED_DATA_PATH)

st.title("UK Data Breach Incident Dashboard")

st.set_page_config(
    page_title="UK Data Breach Incident Dashboard",
    page_icon=":bar_chart:",
    layout="wide",
    initial_sidebar_state="auto",
    menu_items={
        'Get Help': 'mailto:29172807@students.lincoln.ac.uk',
        'Report a bug': 'mailto:29172807@students.lincoln.ac.uk',
        'About': "This dashboard provides insights into data breach incidents in the UK health sector."
    }
)

st.sidebar.success("Pages")

@st.cache_data
def load_data():
    data = pd.read_csv(PROCESSED_DATA_PATH)
    return data

data = load_data()
st.subheader("Overview of Data Breach Incidents")
st.write(data.head())

