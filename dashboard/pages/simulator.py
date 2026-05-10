import streamlit as st

from dashboard.components.disclaimer import render_disclaimer
from dashboard.components.header import render_header
from utils.artefacts import load_processed_dataset


def main() -> None:
    render_header(
        title="Simulator",
        subtitle="Describe a hypothetical breach to see the predicted regulatory outcome and similar past breaches.",
    )
    render_disclaimer()
    st.info("Simulator page — not yet implemented.")


if __name__ == "__main__":
    main()