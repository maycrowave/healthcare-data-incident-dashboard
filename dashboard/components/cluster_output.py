from typing import Dict, List
import streamlit as st
from components.charts import horizontal_stacked_proportions
from utils.artefacts import load_cluster_characteristics
from utils.inference import InferenceError, predict_cluster

# Same display order as the classifier panel for visual consistency
_DISPLAY_ORDER: List[str] = [
    "Investigation Pursued",
    "Informal Action Taken",
    "No Further Action"
]

# Map raw feature column names to user-facing labels, so the description reads as natural prose rather than column names
_FEATURE_LABELS: Dict[str, str] = {
    "incident_category": "incident category",
    "incident_type": "incident type",
    "data_subject_type": "data subjects",
    "data_type": "data types",
    "no_data_subjects_affected": "scale",
    "time_taken_to_report": "reporting time"
}

# Features prioritised in the description in order
# Surface the three most distinctive ones rather than all six to keep the description short
_DESCRIPTION_FEATURES: List[str] = [
    "incident_category",
    "incident_type",
    "no_data_subjects_affected",
    "time_taken_to_report"
]

def _describe_cluster(cluster_data: Dict) -> str:
    """
    Build a short  description of a cluster from its dominant feature values

    Picks the top value for each of the three most distinctive features and renders them as a sentence
    """
    top_features = cluster_data.get("top_features", {})
    parts: List[str] = []

    for feature in _DESCRIPTION_FEATURES[:3]:
        feature_data = top_features.get(feature, {})
        if not feature_data:
            continue

        # Take the highest % value
        top_value = max(feature_data.items(), key=lambda kv: kv[1])
        value, percentage = top_value
        label = _FEATURE_LABELS.get(feature, feature)
        parts.append(f"{percentage:.0f}% are {value.lower()} ({label})")

    if not parts:
        return "Detailed cluster characteristics unavailable."

    return "In this group of similar past breaches: " + "; ".join(parts) + "."


def render_cluster_panel(inputs: Dict[str, object]) -> None:
    """
    Render the cluster context panel

    Args:
        inputs: the validated input dict from render_input_form
    """
    try:
        with st.spinner("Finding similar past breaches..."):
            cluster_id = predict_cluster(inputs)
    except InferenceError as e:
        st.error(f"Sorry, could not find similar past breaches: {e}")
        return

    characteristics = load_cluster_characteristics()
    cluster_data = characteristics.get(str(cluster_id))

    if cluster_data is None:
        st.error(f"Cluster characteristics not found for cluster {cluster_id}.")
        return

    st.subheader("Similar past breaches")

    # persistent framing every render
    st.info(
        "**This is a similarity finding, not a prediction.** "
        "Clustering groups breaches by their characteristics. " 
        "The decision distribution below shows what the ICO historically decided for breaches that look similar to yours, not what they would decide for your specific breach. "
        "Statistical analysis showed clusters do not meaningfully align with regulatory decisions taken by the ICO."
    )

    with st.container(border=True):
        # Header= cluster identifier + size context
        header_left, header_right = st.columns([3, 1])
        with header_left:
            st.markdown(f"**Cluster {cluster_id}**")
        with header_right:
            size = cluster_data.get("size", 0)
            percent = cluster_data.get("percentage_of_dataset", 0.0)
            st.markdown(
                f"<div style='text-align: right; color: var(--text-color); opacity: 0.7;'>"
                f"{size:,} breaches · {percent:.1f}% of dataset"
                f"</div>",
                unsafe_allow_html=True
            )

        # Description = top feature values for the cluster
        st.markdown(_describe_cluster(cluster_data))

        # Within cluster decision distribution
        st.caption("Historical decision distribution within this group:")
        outcome_distribution: Dict[str, float] = cluster_data.get("outcome_distribution", {})
        # Convert from percentages (out of 100) to proportions (normalised) for the chart
        proportions = {label: outcome_distribution.get(label, 0.0) / 100.0 for label in _DISPLAY_ORDER}

        fig = horizontal_stacked_proportions(proportions=proportions, class_order=_DISPLAY_ORDER)
        
        st.plotly_chart(
            fig,
            width='stretch',
            config={"displayModeBar": False}
        )