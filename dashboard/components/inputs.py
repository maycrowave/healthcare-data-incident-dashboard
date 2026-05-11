from typing import Dict, List, Optional
import streamlit as st
from utils.artefacts import load_allowed_values

# Tooltips for the most ambiguous fields. Other fields are self-describing.
_HELP_TEXT = {
    "incident_category": (
        "Cyber covers electronic causes (phishing, ransomware, unauthorised access). "
        "Non Cyber covers human or process causes (misdirected post, lost paperwork)."
    ),
    "no_data_subjects_affected": (
        "Approximate scale of the breach. Select 'Unknown' if the count is "
        "genuinely unknown rather than guessing a band."
    ),
    "time_taken_to_report": (
        "Under UK GDPR, controllers must notify the ICO within 72 hours of "
        "becoming aware of a personal data breach where feasible."
    ),
}

def _validate(inputs: Dict[str, object]) -> List[str]:
    """Return a list of human-readable error messages, empty if all valid."""
    errors: List[str] = []

    if not inputs.get("data_subject_type"):
        errors.append("Select at least one **Data Subject Type**.")
    if not inputs.get("data_type"):
        errors.append("Select at least one **Data Type**.")

    # Single-selects use index=None to start empty, an unsubmitted single-select will return None, which we treat as missing.
    for field, label in [
        ("incident_category", "Incident Category"),
        ("incident_type", "Incident Type"),
        ("no_data_subjects_affected", "No. Data Subjects Affected"),
        ("time_taken_to_report", "Time Taken to Report"),
    ]:
        if inputs.get(field) is None:
            errors.append(f"Select a value for **{label}**.")

    return errors


def render_input_form() -> Optional[Dict[str, object]]:
    """
    Render the Simulator input form.

    Returns:
        A dict of validated inputs if the user submitted and inputs are valid. Otherwise None.
        Returned keys:
            incident_category: str
            incident_type: str
            data_subject_type: List[str]
            data_type: List[str]
            no_data_subjects_affected: str
            time_taken_to_report: str
    """
    allowed = load_allowed_values()

    with st.form(key="simulator_form", clear_on_submit=False):
        st.markdown("#### Describe the breach")

        # Two-column layout for the six inputs
        # Left column: what kind of incident
        # Right column: who and what was affected, when, at what scale
        col_left, col_right = st.columns(2)

        with col_left:
            incident_category = st.selectbox(
                "Incident Category",
                options=allowed["incident_category"],
                index=None,
                placeholder="Select an option",
                help=_HELP_TEXT["incident_category"],
            )
            incident_type = st.selectbox(
                "Incident Type",
                options=allowed["incident_type"],
                index=None,
                placeholder="Select an option",
            )
            no_data_subjects_affected = st.selectbox(
                "No. Data Subjects Affected",
                options=allowed["no_data_subjects_affected"],
                index=None,
                placeholder="Select a band",
                help=_HELP_TEXT["no_data_subjects_affected"],
            )

        with col_right:
            data_subject_type = st.multiselect(
                "Data Subject Type",
                options=allowed["data_subject_type"],
                placeholder="Select one or more",
            )
            data_type = st.multiselect(
                "Data Type",
                options=allowed["data_type"],
                placeholder="Select one or more",
            )
            time_taken_to_report = st.selectbox(
                "Time Taken to Report",
                options=allowed["time_taken_to_report"],
                index=None,
                placeholder="Select a band",
                help=_HELP_TEXT["time_taken_to_report"],
            )

        submitted = st.form_submit_button("Simulate outcome", type="primary")

    if not submitted:
        return None

    inputs: Dict[str, object] = {
        "incident_category": incident_category,
        "incident_type": incident_type,
        "data_subject_type": data_subject_type,
        "data_type": data_type,
        "no_data_subjects_affected": no_data_subjects_affected,
        "time_taken_to_report": time_taken_to_report,
    }

    errors = _validate(inputs)
    if errors:
        for err in errors:
            st.error(err)
        return None

    return inputs