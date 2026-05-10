from pathlib import Path
import json
import pickle
from typing import Any, Dict, List, Tuple

import pandas as pd
import streamlit as st
from catboost import CatBoostClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder

ARTIFACTS_PATH = Path(__file__).resolve().parents[2] / "artifacts"
CLASSIFIER_PATH = ARTIFACTS_PATH / "classifier"
CLUSTER_PATH = ARTIFACTS_PATH / "clustering"

def _classifier_path(filename: str) -> Path:
    """Return the absolute path to a classifier artefact file, raising if missing."""
    path = CLASSIFIER_PATH / filename
    if not path.exists():
        raise FileNotFoundError(
            f"Classifier artefact not found: {path}. "
            f"Re-run the relevant model notebook to regenerate artefacts."
        )
    return path

def _cluster_path(filename: str) -> Path:
    """Return the absolute path to a clustering artefact file, raising if missing."""
    path = CLUSTER_PATH / filename
    if not path.exists():
        raise FileNotFoundError(
            f"Clustering artefact not found: {path}. "
            f"Re-run the relevant model notebook to regenerate artefacts."
        )
    return path

@st.cache_resource(show_spinner="Loading classifier model...")
def load_classifier() -> CatBoostClassifier:
    """Load the trained CatBoost classifier."""
    model = CatBoostClassifier()
    model.load_model(str(_classifier_path("catboost_final.cbm")))
    return model


@st.cache_resource(show_spinner="Loading clustering model...")
def load_kmeans() -> KMeans:
    """Load the trained KMeans clusterer."""
    with open(_cluster_path("kmeans_final.pkl"), "rb") as f:
        return pickle.load(f)


@st.cache_resource(show_spinner="Loading encoders...")
def load_encoders() -> Tuple[Dict[str, MultiLabelBinarizer], OneHotEncoder]:
    """
    Load the fitted encoders.

    Returns a tuple of (mlb_encoders_by_column, ohe_encoder).
    The MLB dict has one entry per multilabel column (data_subject_type, data_type).
    The OHE encoder is only used by the KMeans pipeline, not the classifier.
    """
    mlb_encoders: Dict[str, MultiLabelBinarizer] = {}
    for col in ("data_subject_type", "data_type"):
        with open(_classifier_path(f"mlb_{col}.pkl"), "rb") as f:
            mlb_encoders[col] = pickle.load(f)

    with open(_classifier_path("ohe.pkl"), "rb") as f:
        ohe_encoder = pickle.load(f)

    return mlb_encoders, ohe_encoder


@st.cache_data
def load_allowed_values() -> Dict[str, List[str]]:
    """Valid input values per feature, derived from train+validation data."""
    with open(_classifier_path("allowed_values.json")) as f:
        return json.load(f)


@st.cache_data
def load_classifier_feature_metadata() -> Dict[str, Any]:
    """Feature ordering and categorical indices required by the classifier."""
    with open(_classifier_path("classifier_feature_metadata.json")) as f:
        return json.load(f)


@st.cache_data
def load_classifier_model_type() -> Dict[str, Any]:
    """Metadata about the selected classifier (model name, encoding, etc.)."""
    with open(_classifier_path("classifier_model_type.json")) as f:
        return json.load(f)


@st.cache_data
def load_baseline_probabilities() -> Dict[str, float]:
    """Historical decision rates from train+validation, for baseline comparison."""
    with open(_classifier_path("baseline_probabilities.json")) as f:
        return json.load(f)


@st.cache_data
def load_classifier_results() -> Dict[str, Any]:
    """
    Classifier test-set evaluation results.

    Used by the dashboard to surface per-class Brier scores (NF1, calibration indicator) and to display headline metrics on the About screen.
    """
    with open(_classifier_path("classifier_results.json")) as f:
        return json.load(f)


@st.cache_data
def load_cluster_characteristics() -> Dict[str, Any]:
    """
    Per-cluster size, outcome distribution, and dominant feature values.

    Drives the cluster context panel (F4) and the natural-language cluster description.
    """
    with open(_cluster_path("kmeans_cluster_characteristics.json")) as f:
        return json.load(f)


@st.cache_data
def load_kmeans_feature_metadata() -> Dict[str, Any]:
    """Feature ordering required by the KMeans matrix builder."""
    with open(_cluster_path("kmeans_feature_metadata.json")) as f:
        return json.load(f)


@st.cache_data
def load_kmeans_summary() -> Dict[str, Any]:
    """Aggregate clustering metrics (k, silhouette, ARI, NMI)."""
    with open(_cluster_path("kmeans_summary.json")) as f:
        return json.load(f)


@st.cache_data
def load_processed_dataset() -> pd.DataFrame:
    """
    Full processed dataset for the Explore screen.

    Loaded from data/ rather than artifacts/ since this is the source data, not a model output.
    """
    data_path = ARTIFACTS_PATH.parent / "data" / "new-data-security-incident-trends-health-sector.csv"
    if not data_path.exists():
        raise FileNotFoundError(f"Processed dataset not found: {data_path}")
    return pd.read_csv(data_path)
