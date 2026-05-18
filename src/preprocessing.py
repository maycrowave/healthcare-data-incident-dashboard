import pandas as pd
from typing import Optional

from src.constants import (
    ORIGINAL_DATA_PATH,
    PROCESSED_DATA_PATH,
    PREPROCESSING_CATEGORICAL_COLS,
    NO_DATA_SUBJECTS_AFFECTED_ORDER,
    EXCLUDED_DECISION_LABEL,
    TEST_YEAR,
    VAL_YEAR,
    VAL_QUARTERS
)

def load_original_dataset(file_path: str = ORIGINAL_DATA_PATH) -> pd.DataFrame:
    """Load the original ICO dataset"""
    return pd.read_csv(file_path)

def load_processed_dataset(file_path: str = PROCESSED_DATA_PATH) -> pd.DataFrame:
    """Load the processed dataset"""
    return pd.read_csv(file_path)

def filter_to_health_sector(df_ico: pd.DataFrame) -> pd.DataFrame:
    """Filter the dataset to include only the healthcare sector"""
    df_health = df_ico[df_ico["Sector"] == "Health"].copy(deep=True)
    df_health = df_health.drop(columns=["Sector"])
    
    # Standardise column names
    df_health.columns = (
        df_health.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace(".", "")
    )
    
    return df_health

def filter_to_timeframe(
    df_health: pd.DataFrame,
    start_year: int = 2021,
    start_quarter: int = 2,
    end_year: int = 2025,
    end_quarter: int = 4
    ) -> pd.DataFrame:
    """Filter the dataset to include only incidents from 2021 Q2 to 2025 Q4."""
    
    df_filtered = df_health.copy()
    df_filtered["quarter_num"] = df_filtered["quarter"].str.extract(r"(\d)").astype(int)
    
    df_filtered = df_filtered[(
        (df_filtered["year"] > start_year) |
        ((df_filtered["year"] == start_year) & (df_filtered["quarter_num"] >= start_quarter))
    ) & (
        (df_filtered["year"] < end_year) |
        ((df_filtered["year"] == end_year) & (df_filtered["quarter_num"] <= end_quarter))
    )]

    return df_filtered.drop(columns=["quarter_num"])

def set_categorical_dtypes(df_health: pd.DataFrame) -> pd.DataFrame:
    """Set the categorical data types for all columns"""
    df_dtyped = df_health.copy()
    for col in PREPROCESSING_CATEGORICAL_COLS:
        if col in df_dtyped.columns:
            df_dtyped[col] = df_dtyped[col].astype("category")
    
    # Set the order of the no_data_subjects_affected column to be ordinal
    if "no_data_subjects_affected" in df_dtyped.columns:
        df_dtyped["no_data_subjects_affected"] = pd.Categorical(
            df_dtyped["no_data_subjects_affected"],
            categories=NO_DATA_SUBJECTS_AFFECTED_ORDER,
            ordered=True
        )
        
    return df_dtyped

def aggregate_to_unique_breaches(df_health: pd.DataFrame) -> pd.DataFrame:
    """Aggregate the dataset to include only unique breaches by grouping on bi reference"""
    df_aggregated = df_health.copy()
    df_aggregated = df_aggregated.groupby("bi_reference").agg({
        "year": "first",
        "quarter": "first",
        "data_subject_type": lambda data: ', '.join(sorted(data.unique())),
        "data_type": lambda data: ', '.join(sorted(data.unique())),
        "decision_taken": "first",
        "incident_category": "first",
        "incident_type": "first",
        "no_data_subjects_affected": "first",
        "time_taken_to_report": "first"
    }).reset_index()
    
    df_aggregated = df_aggregated.drop(columns=["bi_reference"])
    # Remove breaches with the Not Yet Assigned label
    df_aggregated = df_aggregated[~df_aggregated["decision_taken"].isin(EXCLUDED_DECISION_LABEL)].reset_index(drop=True)
    
    return df_aggregated

def temporal_train_val_test_split(
    df: pd.DataFrame,
    test_year: int = TEST_YEAR,
    val_year: Optional[int] = VAL_YEAR,
    val_quarters: Optional[list] = VAL_QUARTERS,
    sort_chronologically: bool = True):
    """Split the dataset rows into training and testing sets based on the year"""
    
    # Filter the dataset to the specified years and quarters for test and validation sets
    df_test = df[df["year"] == test_year].copy()
    val_mask = ((df["year"] == val_year) & (df["quarter"].isin(val_quarters)))
    df_val = df[val_mask].copy()

    # The training set is everything that isn't in the test or validation sets
    train_mask = ((df["year"] < test_year) & ~val_mask)
    df_train = df[train_mask].copy()
    
    if sort_chronologically:
        df_train = df_train.sort_values(by=["year", "quarter"]).reset_index(drop=True)
        df_val = df_val.sort_values(by=["year", "quarter"]).reset_index(drop=True)
        df_test = df_test.sort_values(by=["year", "quarter"]).reset_index(drop=True)
    
    return df_train, df_val, df_test

def preprocess_original_dataset(file_path: str = ORIGINAL_DATA_PATH) -> pd.DataFrame:
    """Load and preprocess the original dataset and return the final processed dataset"""
    df_original = load_original_dataset(file_path)
    df_health = filter_to_health_sector(df_original)
    df_filtered = filter_to_timeframe(df_health)
    df_dtyped = set_categorical_dtypes(df_filtered)
    df_aggregated = aggregate_to_unique_breaches(df_dtyped)
    return df_aggregated

def validate_processed_dataset(df_processed: pd.DataFrame) -> None:
    """Validate the processed dataset to ensure it meets expected structure and content"""
    required_cols = [
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
    
    # Check for missing required columns
    missing_cols = [col for col in required_cols if col not in df_processed.columns]
    if missing_cols:
        raise ValueError(f"Processed dataset is missing the required column(s): {missing_cols}")
    
    missing_values = df_processed.isnull().sum()
    missing_values = missing_values[missing_values > 0]
    if not missing_values.empty:
        raise ValueError(f"Processed dataset contains missing value(s):\n{missing_values}")
    
    # Check that the decision taken column does not contain Not Yet Assigned
    if df_processed["decision_taken"].isin(EXCLUDED_DECISION_LABEL).any():
        raise ValueError(f"Processed dataset contains the excluded decision labels: {EXCLUDED_DECISION_LABEL}")
    
    # Check that the dataset is not empty after preprocessing
    if df_processed.empty:
        raise ValueError("Processed dataset is empty.")