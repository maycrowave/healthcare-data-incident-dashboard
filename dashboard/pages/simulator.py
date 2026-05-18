import time
import streamlit as st

from components.classifier_output import render_classifier_panel
from components.cluster_output import render_cluster_panel
from components.disclaimer import render_disclaimer
from components.header import render_header
from components.inputs import render_input_form
from components.output_footer import render_output_footer

def main() -> None:
    render_header(
        title="Simulator",
        subtitle=(
            "Describe a hypothetical breach to see the predicted regulatory decision taken and similar past breaches."
        )
    )
    render_disclaimer()

    inputs = render_input_form()
    if inputs is None:
        st.info(
            "Fill in all six fields above and click **Simulate outcome** to see predicted probabilities, historical baselines and similar past breaches."
        )
        return
    
    # Start timer after users input is submitted for testing purposes later
    time_start = time.perf_counter()#
    render_classifier_panel(inputs)
    render_cluster_panel(inputs)
    render_output_footer()
    
    time_elapsed = time.perf_counter() - time_start
    st.caption(f"Inference + Render: {time_elapsed*1000:.0f}ms")

if __name__ == "__main__":
    main()