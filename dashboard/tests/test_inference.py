from unittest.mock import patch
import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder

from utils.inference import (InferenceError, _build_classifier_feature_row, _build_kmeans_feature_row)

# Mock encoders 
def _fixture_mlb_encoders():
    """Build two MLB encoders with known values"""
    mlb_subject = MultiLabelBinarizer()
    mlb_subject.fit([["Patients", "Employees", "Children"]])
    mlb_data = MultiLabelBinarizer()
    mlb_data.fit([["Basic personal identifiers", "Health data", "Financial data"]])
    return {"data_subject_type": mlb_subject, "data_type": mlb_data}

def _fixture_ohe_encoder():
    """Build a OHE over the classifier categorical features"""
    ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    df = pd.DataFrame({
        "incident_category": ["Cyber", "Non Cyber"],
        "incident_type": ["Phishing", "Data emailed to incorrect recipient"],
        "no_data_subjects_affected": ["1 to 9", "10 to 99"],
        "time_taken_to_report": ["Less than 24 hours", "More than 1 week"]
    })
    ohe.fit(df)
    return ohe

# Mock catboost metadata
def _fixture_classifier_metadata():
    """Mock classifier metadata JSON contents"""
    mlb_subject_cols = [
        "data_subject_type__Children",
        "data_subject_type__Employees",
        "data_subject_type__Patients"
    ]
    mlb_data_cols = [
        "data_type__Basic personal identifiers",
        "data_type__Financial data",
        "data_type__Health data"
    ]
    cat_cols = [
        "incident_category",
        "incident_type",
        "no_data_subjects_affected",
        "time_taken_to_report"
    ]
    return {
        "feature_columns": mlb_subject_cols + mlb_data_cols + cat_cols,
        "categorical_cols": cat_cols,
        "multilabel_cols": ["data_subject_type", "data_type"]
    }


# Shared sample input for these tests
_SAMPLE_INPUT = {
    "incident_category": "Cyber",
    "incident_type": "Phishing",
    "data_subject_type": ["Patients", "Children"],
    "data_type": ["Health data"],
    "no_data_subjects_affected": "1 to 9",
    "time_taken_to_report": "Less than 24 hours"
}


# Classifier feature row tests
class TestBuildClassifierFeatureRow:
    def test_returns_df_with_expected_columns(self):
        with patch("utils.inference.load_classifier_feature_metadata") as classifier_meta, \
            patch("utils.inference.load_encoders") as enc:
            classifier_meta.return_value = _fixture_classifier_metadata()
            enc.return_value = (_fixture_mlb_encoders(), _fixture_ohe_encoder())
            row = _build_classifier_feature_row(_SAMPLE_INPUT)

        assert isinstance(row, pd.DataFrame)
        assert row.shape[0] == 1
        expected_cols = _fixture_classifier_metadata()["feature_columns"]
        assert list(row.columns) == expected_cols

    def test_multilabel_encoding_correct(self):
        with patch("utils.inference.load_classifier_feature_metadata") as classifier_meta, \
            patch("utils.inference.load_encoders") as enc:
            classifier_meta.return_value = _fixture_classifier_metadata()
            enc.return_value = (_fixture_mlb_encoders(), _fixture_ohe_encoder())
            row = _build_classifier_feature_row(_SAMPLE_INPUT)

        assert row["data_subject_type__Patients"].iloc[0] == 1
        assert row["data_subject_type__Children"].iloc[0] == 1
        assert row["data_subject_type__Employees"].iloc[0] == 0
        assert row["data_type__Health data"].iloc[0] == 1
        assert row["data_type__Financial data"].iloc[0] == 0

    def test_categorical_columns_strings(self):
        with patch("utils.inference.load_classifier_feature_metadata") as classifier_meta, \
            patch("utils.inference.load_encoders") as enc:
            classifier_meta.return_value = _fixture_classifier_metadata()
            enc.return_value = (_fixture_mlb_encoders(), _fixture_ohe_encoder())

            row = _build_classifier_feature_row(_SAMPLE_INPUT)

        assert row["incident_category"].iloc[0] == "Cyber"
        assert row["incident_type"].iloc[0] == "Phishing"

    def test_empty_multilabel_yields_zeros(self):
        inputs = {**_SAMPLE_INPUT, "data_subject_type": []}
        with patch("utils.inference.load_classifier_feature_metadata") as classifier_meta, \
            patch("utils.inference.load_encoders") as enc:
            classifier_meta.return_value = _fixture_classifier_metadata()
            enc.return_value = (_fixture_mlb_encoders(), _fixture_ohe_encoder())
            row = _build_classifier_feature_row(inputs)

        for col in [
            "data_subject_type__Children",
            "data_subject_type__Employees",
            "data_subject_type__Patients"
        ]: assert row[col].iloc[0] == 0


# Mock kmeans metadata
def _fixture_kmeans_metadata():
    mlb_subject_cols = [
        "data_subject_type__Children",
        "data_subject_type__Employees",
        "data_subject_type__Patients"
    ]
    mlb_data_cols = [
        "data_type__Basic personal identifiers",
        "data_type__Financial data",
        "data_type__Health data"
    ]
    ohe_cols = [
        "incident_category_Cyber",
        "incident_category_Non Cyber",
        "incident_type_Data emailed to incorrect recipient",
        "incident_type_Phishing",
        "no_data_subjects_affected_1 to 9",
        "no_data_subjects_affected_10 to 99",
        "time_taken_to_report_Less than 24 hours",
        "time_taken_to_report_More than 1 week"
    ]
    return {
        "feature_columns_order": mlb_subject_cols + mlb_data_cols + ohe_cols,
        "categorical_columns": [
            "incident_category",
            "incident_type",
            "no_data_subjects_affected",
            "time_taken_to_report"
        ],
        "multilabel_columns": ["data_subject_type", "data_type"]
    }


class TestBuildKmeansFeatureRow:
    def test_returns_df_with_expected_columns(self):
        with patch("utils.inference.load_kmeans_feature_metadata") as kmeans_meta, \
            patch("utils.inference.load_encoders") as enc:
            kmeans_meta.return_value = _fixture_kmeans_metadata()
            enc.return_value = (_fixture_mlb_encoders(), _fixture_ohe_encoder())
            row = _build_kmeans_feature_row(_SAMPLE_INPUT)

        assert isinstance(row, pd.DataFrame)
        assert row.shape[0] == 1
        expected_cols = _fixture_kmeans_metadata()["feature_columns_order"]
        assert list(row.columns) == expected_cols

    def test_columns_numeric(self):
        with patch("utils.inference.load_kmeans_feature_metadata") as kmeans_meta, \
            patch("utils.inference.load_encoders") as enc:
            kmeans_meta.return_value = _fixture_kmeans_metadata()
            enc.return_value = (_fixture_mlb_encoders(), _fixture_ohe_encoder())
            row = _build_kmeans_feature_row(_SAMPLE_INPUT)

        for col in row.columns:
            assert pd.api.types.is_numeric_dtype(row[col]), (f"Column {col} is not numeric: {row[col].dtype}")

    def test_ohe_encodes_category(self):
        with patch("utils.inference.load_kmeans_feature_metadata") as kmeans_meta, \
            patch("utils.inference.load_encoders") as enc:
            kmeans_meta.return_value = _fixture_kmeans_metadata()
            enc.return_value = (_fixture_mlb_encoders(), _fixture_ohe_encoder())
            row = _build_kmeans_feature_row(_SAMPLE_INPUT)

        assert row["incident_category_Cyber"].iloc[0] == 1
        assert row["incident_category_Non Cyber"].iloc[0] == 0