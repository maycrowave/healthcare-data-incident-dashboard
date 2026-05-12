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
    st.title(title)

    if subtitle:
        st.markdown(subtitle)

    st.caption(
        f"Trained on {DATA_SOURCE_NAME} data, {TRAINING_DATA_RANGE}. "
        f"Dashboard last updated {DASHBOARD_LAST_UPDATED}."
    )
    st.divider()