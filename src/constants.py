# File paths
ORIGINAL_DATA_PATH = "../data/data-security-incidents-trends-q1-2019-to-q4-2025.csv"
PROCESSED_DATA_PATH = "../data/new-data-security-incident-trends-health-sector.csv"
ARTIFACTS_PATH = "../artifacts/"

PREPROCESSING_CATEGORICAL_COLS = [
    "bi_reference",
    "year",
    "quarter",
    "data_subject_type",
    "data_type",
    "decision_taken",
    "incident_category",
    "incident_type",
    "no_data_subjects_affected",
    "time_taken_to_report"
]

# Target feature
TARGET_COL = "decision_taken"

# Multilabel columns contain a comma separated list of values and need multi label encoding
MULTILABEL_COLS = ["data_subject_type", "data_type"]

# Categorical columns contain single values
CATEGORICAL_COLS = ["incident_category", "incident_type", "no_data_subjects_affected", "time_taken_to_report"]

# All feature columns except the target column
FEATURE_COLS = MULTILABEL_COLS + CATEGORICAL_COLS

# Number of splits for cross validation
CV_SPLITS = int(5)

# Random state for reproducibility
RANDOM_STATE = int(42)

# Test year for train/val/test split
TEST_YEAR = int(2025)
VAL_YEAR = int(2024)
VAL_QUARTERS = ["Qtr 3", "Qtr 4"]

# Class labels
CLASS_LABELS = ["Informal Action Taken", "Investigation Pursued", "No Further Action"]
EXCLUDED_DECISION_LABEL = ["Not Yet Assigned"]

# Ordered No. Data Subjects Affected
NO_DATA_SUBJECTS_AFFECTED_ORDER = [
    "1 to 9",
    "10 to 99",
    "100 to 1k",
    "1k to 10k",
    "10k to 100k",
    "100k and above",
    "Unknown"
]

COLOUR_PALETTE = ["#f47d9b","#7be289","#f3dc4a","#8097e8","#e8a16e","#d585ed","#60e1e1"]
PER_CLASS_COLOURS = ["#7db8c7", "#e07b6b", "#abc77d"]

