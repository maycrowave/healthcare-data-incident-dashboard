import streamlit as st

from utils.constants import (
    TRAINING_DATA_RANGE,
    DASHBOARD_LAST_UPDATED,
    DATA_SOURCE_NAME
)

def render_header(title: str, subtitle: str | None = None) -> None:
    """
    Render the page header
    """
    st.title(title, text_alignment="center")
    if subtitle:
        st.markdown(subtitle, text_alignment="center")
    st.caption(
        f"Trained on {DATA_SOURCE_NAME} data, {TRAINING_DATA_RANGE}. Dashboard last updated {DASHBOARD_LAST_UPDATED}.",
        text_alignment="center"
    )
    st.divider()