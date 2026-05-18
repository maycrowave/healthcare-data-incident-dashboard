from components.header import render_header
from components.disclaimer import render_disclaimer

import streamlit as st

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
        subtitle=("Explore predicted regulatory outcomes for hypothetical UK healthcare data breaches, using the Information Commissioner's Office (ICO) Data Security Incident Trends dataset.")
    )
    render_disclaimer()
    
    st.markdown("""### Where to go from here""", text_alignment="center")
    nav_col1, nav_col2, nav_col3 = st.columns(3)
    
    with nav_col1:
        with st.container(border=True):
            st.markdown("#### About", text_alignment="center")
            st.caption("Methodology, limitations, and the regulatory decision glossary.", text_alignment="center")
            st.page_link("pages/About.py", label="See About the Dashboard ->", use_container_width=True)
        
    with nav_col2:
        with st.container(border=True):
            st.markdown("#### Explore", text_alignment="center")
            st.caption("Browse the historical breach data the model is trained on.", text_alignment="center")
            st.page_link("pages/Explore.py", label="Explore the Data ->", use_container_width=True)
            
    with nav_col3:
        with st.container(border=True):
            st.markdown("#### Simulator", text_alignment="center")
            st.caption("Describe a hypothetical breach and see predicted regulatory outcomes.", text_alignment="center")
            st.page_link("pages/Simulator.py", label="Simulate a Breach ->", use_container_width=True)

if __name__ == "__main__":
    main()