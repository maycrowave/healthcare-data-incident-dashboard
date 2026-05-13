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
        "Approximate scale of the breach. Select 'Unknown' if you don't know or aren't sure."
    ),
    "time_taken_to_report": (
        "Under UK GDPR, controllers must notify the ICO within 72 hours of becoming aware of a personal data breach where possible."
    ),
    "incident_type": (
        "The ICO's taxonomy of incident types is quite granular. If you're not sure, select the one that seems closest or 'Other cyber/ non-cyber incident'."
    ),
    "data_subject_type": (
        "The types of individuals affected by the breach. Select all that apply. Select 'Unknown' if you don't know or aren't sure."
    ),
    "data_type": (
        "The types of personal data involved in the breach. Select all that apply. Select 'Unknown' if you don't know or aren't sure."
    )
}


def _validate(inputs_dict: Dict[str, object]) -> List[str]:
    """Return a list of error messages, empty if all are valid"""
    errors: List[str] = []

    if not inputs_dict.get("data_subject_type"):
        errors.append("Select at least one **Data Subject Type**.")
    if not inputs_dict.get("data_type"):
        errors.append("Select at least one **Data Type**.")

    # Single selects use index = None to start empty
    # An unsubmitted single select will return None which is treated as missing
    for field, label in [
        ("incident_category", "Incident Category"),
        ("incident_type", "Incident Type"),
        ("no_data_subjects_affected", "No. Data Subjects Affected"),
        ("time_taken_to_report", "Time Taken to Report"),
    ]:
        if inputs_dict.get(field) is None:
            errors.append(f"Select a value for **{label}**.")

    return errors


def render_input_form() -> Optional[Dict[str, object]]:
    """
    Render the Simulator input form
    """
    allowed = load_allowed_values()

    with st.form(key="simulator_form", clear_on_submit=False):
        st.markdown("#### Describe the breach")

        # Two column layout for the six inputs
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
                help=_HELP_TEXT["incident_type"],
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
                help=_HELP_TEXT["data_subject_type"],
            )
            data_type = st.multiselect(
                "Data Type",
                options=allowed["data_type"],
                placeholder="Select one or more",
                help=_HELP_TEXT["data_type"],
            )
            time_taken_to_report = st.selectbox(
                "Time Taken to Report",
                options=allowed["time_taken_to_report"],
                index=None,
                placeholder="Select a time range",
                help=_HELP_TEXT["time_taken_to_report"],
            )

        submitted = st.form_submit_button("Simulate outcome", type="primary")

    if not submitted:
        return None

    inputs_dict: Dict[str, object] = {
        "incident_category": incident_category,
        "incident_type": incident_type,
        "data_subject_type": data_subject_type,
        "data_type": data_type,
        "no_data_subjects_affected": no_data_subjects_affected,
        "time_taken_to_report": time_taken_to_report,
    }

    errors = _validate(inputs_dict)
    if errors:
        for err in errors:
            st.error(err)
        return None

    return inputs_dict