from pathlib import Path
from typing import Dict, List
# Training data range used by the models
# Update if models are retrained on a different range i.e. future ICO data releases
TRAINING_DATA_RANGE = "2021 Q2 to 2025 Q4"

# Date the dashboard build was finalised and can be updated for future releases
DASHBOARD_LAST_UPDATED = "May 2026"

# Source citation displayed in the footer/ header
DATA_SOURCE_NAME = "ICO Data Security Incident Trends"
DATA_SOURCE_URL = "https://ico.org.uk/action-weve-taken/complaints-and-concerns-data-sets/data-security-incident-trends/"

DECISION_DISPLAY_ORDER = [
    "Investigation Pursued",
    "Informal Action Taken",
    "No Further Action"
]

# Probability to plain text likelihood
_LIKELIHOODS = [
    (0.00, 0.10, "Very unlikely"),
    (0.10, 0.33, "Unlikely"),
    (0.33, 0.66, "About as likely as not"),
    (0.66, 0.90, "Likely"),
    (0.90, 1.01, "Very likely")
]

_LIMITED_RELIABILITY_CLASSES = {
    "Investigation Pursued": ("This model rarely predicts Investigation Pursued. Treat low values for this outcome as 'unable to assess' rather than 'unlikely'.")
}

# Directory paths to saved artefacts
ARTEFACTS_PATH = Path(__file__).resolve().parents[2] / "artefacts"
CLASSIFIER_PATH = ARTEFACTS_PATH / "classifier"
CLUSTER_PATH = ARTEFACTS_PATH / "clustering"

# Badge colours used in the simulator output panel
_TIER_BADGE_COLOUR: Dict[str, str] = {
    "high": "green",
    "moderate": "blue",
    "low": "orange",
    "limited": "red"
}

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
_DESCRIPTION_FEATURES: List[str] = [
    "incident_category",
    "incident_type",
    "no_data_subjects_affected",
    "time_taken_to_report", 
    "data_subject_type",
    "data_type"
]

# Per-class colours, matching src/constants.PER_CLASS_COLOURS so dashboard colours are similar to the dissertation figures
_PER_CLASS_COLOURS: Dict[str, str] = {
    "Investigation Pursued": "#78ADED",
    "Informal Action Taken": "#ADED78",
    "No Further Action": "#ED78AD"
}