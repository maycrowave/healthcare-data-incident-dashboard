from typing import Dict, List
import streamlit as st
from components.charts import horizontal_probability_bar
from utils.artefacts import (
    load_baseline_probabilities,
    load_classifier_results,
)
from utils.inference import InferenceError, predict_classifier_probabilities
from utils.interpretation import (
    CalibrationTier,
    calibration_tiers_from_results,
    likelihood_label,
)

_DISPLAY_ORDER: List[str] = [
    "Investigation Pursued",
    "Informal Action Taken",
    "No Further Action",
]

_TIER_BADGE_COLOUR: Dict[str, str] = {
    "high": "green",
    "moderate": "blue",
    "low": "orange",
    "limited": "grey",
}

def _render_reliability_badge(tier: CalibrationTier) -> None:
    """Render the reliability badge using Streamlit's coloured badge syntax."""
    colour = _TIER_BADGE_COLOUR.get(tier.tier, "grey")
    st.markdown(f":{colour}-badge[{tier.label}]")


def _render_class_row(class_label: str, probability: float, baseline: float, tier: CalibrationTier) -> None:
    """Render a single class's row in the panel."""
    # Header line: class name on the left, reliability badge on the right.
    header_left, header_right = st.columns([3, 1])
    with header_left:
        st.markdown(f"**{class_label}**")
    with header_right:
        _render_reliability_badge(tier)

    # Likelihood label and percentage
    label = likelihood_label(probability)
    st.markdown(
        f"<span style='color: var(--text-color); opacity: 0.7;'>"
        f"{label} · {probability:.1%}"
        f"</span>",
        unsafe_allow_html=True
    )

    # The bar itself
    fig = horizontal_probability_bar(probability=probability, baseline=baseline)
    st.plotly_chart(
        fig,
        width='stretch',
        config={"displayModeBar": False}
    )

    # Inline reliability explanation. Always visible (not in expander) so the user encounters it without an extra click — matches Lesson 7.
    st.caption(tier.explanation)


def render_classifier_panel(inputs: Dict[str, object]) -> None:
    """
    Render the full classifier output panel.

    Args:
        inputs: The validated input dict from render_input_form.
    """
    try: 
        with st.spinner("Generating prediction..."):
            probabilities = predict_classifier_probabilities(inputs)
    except InferenceError as e:
        st.error(f"Sorry, could not generate a prediction for this breach: {e}")
        return
    
    baselines = load_baseline_probabilities()
    classifier_results = load_classifier_results()
    tiers = calibration_tiers_from_results(
        classifier_results=classifier_results,
        base_rates=baselines,
        class_labels=_DISPLAY_ORDER
    )

    st.subheader("Predicted regulatory outcome")
    st.caption(
        "Predicted probability for each outcome based on the breach you described, "
        "compared to the historical baseline rate (vertical tick) for that outcome."
    )

    for class_label in _DISPLAY_ORDER:
        with st.container(border=True):
            _render_class_row(
                class_label=class_label,
                probability=probabilities.get(class_label, 0.0),
                baseline=baselines.get(class_label, 0.0),
                tier=tiers[class_label]
            )
            