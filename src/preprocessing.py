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
    """Load the original ICO dataset from the specified path."""
    return pd.read_csv(file_path)

def load_processed_dataset(file_path: str = PROCESSED_DATA_PATH) -> pd.DataFrame:
    """Load the processed dataset from the specified path."""
    return pd.read_csv(file_path)

def filter_to_health_sector(df_ico: pd.DataFrame) -> pd.DataFrame:
    """Filter the dataset to include only healthcare related incidents."""
    df_health = df_ico[df_ico["Sector"] == "Health"].copy(deep=True)
    df_health = df_health.drop(columns=["Sector"])
    
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
    """Set the categorical data types for all columns."""
    df_dtyped = df_health.copy()
    for col in PREPROCESSING_CATEGORICAL_COLS:
        if col in df_dtyped.columns:
            df_dtyped[col] = df_dtyped[col].astype("category")
    
    if "no_data_subjects_affected" in df_dtyped.columns:
        df_dtyped["no_data_subjects_affected"] = pd.Categorical(
            df_dtyped["no_data_subjects_affected"],
            categories=NO_DATA_SUBJECTS_AFFECTED_ORDER,
            ordered=True
        )
        
    return df_dtyped

def aggregate_to_unique_breaches(df_health: pd.DataFrame) -> pd.DataFrame:
    """Aggregate the dataset to include only unique breaches by grouping on bi_reference."""
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
    df_aggregated = df_aggregated[~df_aggregated["decision_taken"].isin(EXCLUDED_DECISION_LABEL)].reset_index(drop=True)
    
    return df_aggregated

def temporal_train_val_test_split(
    df: pd.DataFrame,
    test_year: int = TEST_YEAR,
    val_year: Optional[int] = VAL_YEAR,
    val_quarters: Optional[list] = VAL_QUARTERS,
    sort_chronologically: bool = True):
    """Split the dataset rows into training and testing sets based on the year."""
    
    df_test = df[df["year"] == test_year].copy()
    val_mask = ((df["year"] == val_year) & (df["quarter"].isin(val_quarters)))
    df_val = df[val_mask].copy()

    train_mask = ((df["year"] < test_year) & ~val_mask)
    df_train = df[train_mask].copy()
    
    if sort_chronologically:
        df_train = df_train.sort_values(by=["year", "quarter"]).reset_index(drop=True)
        df_val = df_val.sort_values(by=["year", "quarter"]).reset_index(drop=True)
        df_test = df_test.sort_values(by=["year", "quarter"]).reset_index(drop=True)
    
    return df_train, df_val, df_test