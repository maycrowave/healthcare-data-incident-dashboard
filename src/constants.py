# File paths
ORIGINAL_DATA_PATH = "../data/data-security-incidents-trends-q1-2019-to-q4-2025.csv"
PROCESSED_DATA_PATH = "../data/new-data-security-incident-trends-health-sector.csv"
ARTEFACTS_PATH = "../artefacts/"

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
TRAIN_YEARS = list(range(2019, 2024))
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

# Colour palette for charts/ graphs
COLOUR_PALETTE = ["#ee7878","#ADED78","#efdf79","#78ADED","#e8a16e","#ED78AD","#73e8e8"]
PER_CLASS_COLOURS = {
    "Investigation Pursued": "#78ADED",
    "Informal Action Taken": "#ADED78",
    "No Further Action": "#ED78AD"
}

