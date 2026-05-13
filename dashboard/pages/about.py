from pathlib import Path

import streamlit as st

from components.header import render_header
from components.disclaimer import render_disclaimer
from utils.artefacts import (
    load_classifier_model_type,
    load_classifier_results,
    load_kmeans_summary
)

from utils.constants import (
    TRAINING_DATA_RANGE,
    DASHBOARD_LAST_UPDATED,
    DATA_SOURCE_NAME,
    DATA_SOURCE_URL
)

CONTENT_PATH = Path(__file__).resolve().parents[1] / "content"

def _load_markdown(filename: str) -> str:
    """Load a markdown file from dashboard/content/"""
    path = CONTENT_PATH / filename
    if not path.exists():
        return f"Content file missing: {filename}"
    return path.read_text(encoding="utf-8")


def main() -> None:
    render_header(
        title="About",
        subtitle="Methodology, limitations and the regulatory decision taken glossary."
    )

    # Load dynamic content from artefacts
    classifier_meta = load_classifier_model_type()
    classifier_results = load_classifier_results()
    kmeans_summary = load_kmeans_summary()

    # Introduction and purpose
    st.header("What this dashboard is")
    st.markdown(
        f"""
        This dashboard is a research artefact developed as part of a final year computer science dissertation. 
        It explores whether regulatory decision outcomes for UK healthcare data breaches can be modelled from the publicly available {DATA_SOURCE_NAME} dataset.

        The dashboard offers two main capabilities.
        - The **Simulator** lets users describe a hypothetical breach and see the model's predicted probability of each regulatory outcome, alongside historical baseline rates and the cluster of similar past breaches.
        - The **Explore** screen presents descriptive views of the underlying training data so people can see what the model has learned from.
        """
    )

    # Disclaimer and limitations
    st.header("What this dashboard is not")
    st.markdown(
        """
        This dashboard is **not validated for operational decision making**.
        It is not an advice tool, not a real-time monitoring system and not authoritative.
        Predictions are based on patterns in publicly reported, anonymised historical breaches and cannot account for the specifics of any real-world incident.
        """
    )

    # Methodology and performance
    st.header("How predictions are made")
    st.markdown(
        f"""
        The Simulator's outcome prediction comes from a **gradient-boosted tree classifier** ({classifier_meta.get("selected_model", "CatBoost")}) trained on labelled past breaches in the {DATA_SOURCE_NAME} dataset.
        The model takes the breach characteristics you provide as input and outputs a probability for each of the three decision taken outcomes.

        The model was trained on data from {TRAINING_DATA_RANGE} and evaluated on a held-out test set from a later date.
        
        Main test-set performance:
        """
    )

    # Display main performance metrics in a row of metric cards
    col_acc, col_f1, col_logloss = st.columns(3)
    col_acc.metric(
        label="Accuracy",
        value=f"{classifier_results.get('accuracy', 0.0):.1%}",
    )
    col_f1.metric(
        label="Macro F1",
        value=f"{classifier_results.get('macro_f1', 0.0):.4f}",
    )
    col_logloss.metric(
        label="Log Loss",
        value=f"{classifier_results.get('log_loss', 0.0):.4f}",
    )

    # Note on reliability and calibration
    st.caption(
        "These are aggregate metrics. Per-class reliability varies, see the Simulator's calibration indicators for class-specific guidance."
    )

    # Note on clusters and ari
    st.header("How similarity is determined")
    st.markdown(f"""
        The Simulator's cluster panel uses a **K-Means clustering model** with K={kmeans_summary.get("k", 7)} clusters trained on the same dataset.
        Each cluster group shares breach characteristics such as data types involved, incident type, breach scale and reporting time.

        An important finding from the modelling work: these clusters do **not** meaningfully align with regulatory decision outcomes (Adjusted Rand Index = {kmeans_summary.get("ari", 0.0):.3f}).
        This means that breaches similar in shape do not necessarily share regulatory decisions, the cluster information tells you that based on previous data "what kinds of breaches look like this simulated one" not "what decision is likely".
        The Simulator's cluster panel is framed accordingly.
        """
    )

    # Limitations
    st.markdown(_load_markdown("limitations.md"))

    # Glossary
    st.markdown(_load_markdown("glossary.md"))

    # Source and attribution
    st.header("Source and attribution")
    st.markdown(
        f"""
        - **Data source**: [{DATA_SOURCE_NAME}]({DATA_SOURCE_URL}), published by the Information Commissioner's Office under the Open Government Licence.
        - **Dashboard last updated**: {DASHBOARD_LAST_UPDATED}.
        """
    )

if __name__ == "__main__":
    main()