import streamlit as st

from dashboard.components.disclaimer import render_disclaimer
from dashboard.components.header import render_header
from utils.artefacts import (
    load_classifier_model_type,
    load_classifier_results,
    load_kmeans_summary,
)


def main() -> None:
    render_header(
        title="About",
        subtitle="Methodology, limitations, and the regulatory decision glossary.",
    )
    render_disclaimer()
    st.info("About page — not yet implemented.")


if __name__ == "__main__":
    main()