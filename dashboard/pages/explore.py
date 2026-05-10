import streamlit as st

from dashboard.components.disclaimer import render_disclaimer
from dashboard.components.header import render_header
from utils.artefacts import load_processed_dataset


def main() -> None:
    render_header(
        title="Explore the Data",
        subtitle="Browse the historical UK healthcare data breach incidents the model is trained on.",
    )
    render_disclaimer()
    st.info("Explore page — not yet implemented.")


if __name__ == "__main__":
    main()