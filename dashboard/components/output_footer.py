import streamlit as st

def render_output_footer() -> None:
    """Render the inline limitations footer for the simulator"""
    st.divider()

    st.markdown(
        """
        ##### How to read these outputs

        The probabilities above reflect patterns the model learned from publicly reported, anonymised historical breaches. 
        They are not a prediction of what the ICO will decide for any specific real-world breach.
        The cluster panel shows what kinds of breaches look similar to the one you described, not what their decision implies for yours.

        For full methodology, training data range, model performance and limitations, see the **About** page in the sidebar or click the button below.
        """
    )
    st.page_link("pages/About.py", label="Learn about the methodology and limitations ->")