import sys
sys.path.append("..")

from dashboard.components.header import render_header
from dashboard.components.disclaimer import render_disclaimer

import streamlit as st
import pandas as pd
import numpy as np
from src.constants import (PROCESSED_DATA_PATH)

st.set_page_config(
    page_title="UK Healthcare Data Breach Outcome Simulator",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'mailto:29172807@students.lincoln.ac.uk',
        'Report a bug': 'mailto:29172807@students.lincoln.ac.uk',
        'About': "This dashboard provides insights into data breach incidents in the UK health sector."
    }
)

def main () -> None:
    render_header(
        title="UK Healthcare Data Breach Outcome Simulator",
        subtitle=(
            "Explore predicted regulatory outcomes for hypothetical UK healthcare "
            "data breaches, using the Information Commissioner's Office (ICO) "
            "Data Security Incident Trends dataset."
        ),
    )
    render_disclaimer()
    
    st.markdown(
        """
        ### Where to go from here

        - **Simulator**: describe a hypothetical breach and see the predicted regulatory outcome and similar past breaches.
        - **Explore**: browse the historical breach data the model is trained on.
        - **About**: methodology, limitations, and the regulatory decision glossary.
        """
    )

if __name__ == "__main__":
    main()