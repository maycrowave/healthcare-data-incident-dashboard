import streamlit as st

DISCLAIMER_TEXT = (
    "**For exploratory and educational use only.** "
    "This dashboard is a research artefact and is not validated for operational decision-making. "
    "Predictions are based on patterns in publicly reported, anonymised historical breaches and are not a substitute for professional regulatory or legal advice."
)

def render_disclaimer() -> None:
    """Render the standard non-operational-use disclaimer"""
    st.warning(DISCLAIMER_TEXT)