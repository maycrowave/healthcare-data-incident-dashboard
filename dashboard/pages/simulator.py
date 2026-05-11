from time import time
import streamlit as st

from dashboard.components import inputs
from dashboard.components.classifier_output import render_classifier_panel
from dashboard.components.disclaimer import render_disclaimer
from dashboard.components.header import render_header
from dashboard.components.inputs import render_input_form
from utils.artefacts import load_processed_dataset
from components.classifier_output import render_classifier_panel
from components.cluster_output import render_cluster_panel
from components.output_footer import render_output_footer


def main() -> None:
    render_header(
        title="Simulator",
        subtitle=(
            "Describe a hypothetical breach to see the predicted regulatory decision taken and similar past breaches."
        ),
    )
    render_disclaimer()

    inputs = render_input_form()
    # Start timer after users input is submitted for testing purposes later
    if inputs is not None:
        time_start = time.perf_counter()
        
    if inputs is None:
        st.info(
            "Fill in all six fields above and click **Simulate outcome** to see predicted probabilities, historical baselines and similar past breaches."
        )
        return

    render_classifier_panel(inputs)
    render_cluster_panel(inputs)
    render_output_footer()
    
    time_elapsed = time.perf_counter() - time_start
    st.caption(f"Inference + Render: {time_elapsed*1000:.0f}ms")

if __name__ == "__main__":
    main()