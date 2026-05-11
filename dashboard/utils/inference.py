from typing import Dict, List

import numpy as np
import pandas as pd

from utils.artefacts import (
    load_classifier,
    load_classifier_feature_metadata,
    load_encoders,
    load_kmeans,
    load_kmeans_feature_metadata,
)

class InferenceError(Exception):
    """Raised when classifier or cluster inference fails for any reason."""


def _wrap_errors(operation: str):
    """
    Decorator to wrap inference functions, converting any unexpected
    error into a typed InferenceError with a user-friendly message.
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except FileNotFoundError as e:
                raise InferenceError(
                    f"Could not load required model files for {operation}. "
                    f"The dashboard's model artefacts may be missing or corrupt. "
                    f"Details: {e}"
                ) from e
            except (ValueError, KeyError) as e:
                raise InferenceError(
                    f"Could not run {operation} on the input you provided. "
                    f"This may indicate a mismatch between your input and "
                    f"the values the model was trained on. Details: {e}"
                ) from e
            except Exception as e:
                raise InferenceError(
                    f"An unexpected error occurred during {operation}. "
                    f"Details: {e}"
                ) from e
        return wrapper
    return decorator

def _build_classifier_feature_row(inputs: Dict[str, object]) -> pd.DataFrame:
    """
    Build a single-row DataFrame in the exact column order the CatBoost classifier was trained on.
    
    The column order is read from classifier_feature_metadata.json (feature_columns) rather than reconstructed, so this stays correct even if the upstream encoder vocabulary changes.
    """
    metadata = load_classifier_feature_metadata()
    mlb_encoders, _ = load_encoders()
    feature_columns: List[str] = metadata["feature_columns"]
    categorical_cols: List[str] = metadata["categorical_cols"]
    multilabel_cols: List[str] = metadata["multilabel_cols"]

    # Build the multi-label binarised columns. MLB.transform expects a list of lists (one inner list of selected labels per row).
    mlb_columns: Dict[str, int] = {}
    for col in multilabel_cols:
        encoder = mlb_encoders[col]
        selected_labels: List[str] = list(inputs.get(col) or [])
        # Single row, wrap in a list-of-lists for transform()
        encoded = encoder.transform([selected_labels])[0]
        for cls, value in zip(encoder.classes_, encoded):
            mlb_columns[f"{col}__{cls}"] = int(value)

    # Build the raw categorical columns (CatBoost handles these natively).
    categorical_columns: Dict[str, str] = {
        col: str(inputs[col]) for col in categorical_cols
    }

    # Combine and reindex to the trained column order. Any column missing from this row would raise here, which is the right failure mode.
    row_data: Dict[str, object] = {**mlb_columns, **categorical_columns}
    feature_row = pd.DataFrame([row_data])

    missing = set(feature_columns) - set(feature_row.columns)
    if missing:
        raise ValueError(
            f"Feature row is missing expected columns: {sorted(missing)}"
        )

    return feature_row[feature_columns]

@_wrap_errors("classifier prediction")
def predict_classifier_probabilities(inputs: Dict[str, object]) -> Dict[str, float]:
    """
    Run the classifier on a single user input and return per-class probabilities keyed by class label.

    Returns:
        Dict mapping class label (e.g. "Informal Action Taken") to its
        predicted probability in [0, 1]. Probabilities sum to ~1.0.
    """
    model = load_classifier()
    feature_row = _build_classifier_feature_row(inputs)

    # CatBoost returns shape (1, n_classes), take the single row.
    probs = model.predict_proba(feature_row)[0]

    # model.classes_ may be a numpy array of dtype object.
    # Coerce to str to keep keys consistent with the JSON-loaded class labels elsewhere.
    class_labels = [str(label) for label in model.classes_]

    return {label: float(p) for label, p in zip(class_labels, probs)}

def _build_kmeans_feature_row(inputs: Dict[str, object]) -> pd.DataFrame:
    """
    Build a single-row DataFrame in the column order the KMeans clusterer
    was trained on (MLB + OHE representation).

    Different from _build_classifier_feature_row: KMeans was trained on the
    one-hot encoded categorical columns, not the native categorical ones.
    """
    metadata = load_kmeans_feature_metadata()
    mlb_encoders, ohe_encoder = load_encoders()
    feature_columns: List[str] = metadata["feature_columns_order"]
    categorical_cols: List[str] = metadata["categorical_columns"]
    multilabel_cols: List[str] = metadata["multilabel_columns"]

    # Multi-label binarised columns (same as classifier path).
    mlb_columns: Dict[str, int] = {}
    for col in multilabel_cols:
        encoder = mlb_encoders[col]
        selected_labels: List[str] = list(inputs.get(col) or [])
        encoded = encoder.transform([selected_labels])[0]
        for cls, value in zip(encoder.classes_, encoded):
            mlb_columns[f"{col}__{cls}"] = int(value)

    # One-hot encoded categorical columns (different from classifier path).
    # OHE expects a 2D array-like with named columns, so we build a one-row
    # DataFrame in the order the OHE was fit on.
    cat_row = pd.DataFrame([{col: str(inputs[col]) for col in categorical_cols}])
    ohe_array = ohe_encoder.transform(cat_row[categorical_cols])
    ohe_column_names = ohe_encoder.get_feature_names_out(categorical_cols)
    ohe_columns: Dict[str, int] = {
        name: int(value) for name, value in zip(ohe_column_names, ohe_array[0])
    }

    # Combine and reindex to the trained column order.
    row_data: Dict[str, object] = {**mlb_columns, **ohe_columns}
    feature_row = pd.DataFrame([row_data])

    missing = set(feature_columns) - set(feature_row.columns)
    if missing:
        raise ValueError(
            f"KMeans feature row is missing expected columns: {sorted(missing)}"
        )

    return feature_row[feature_columns]

@_wrap_errors("cluster assignment")
def predict_cluster(inputs: Dict[str, object]) -> int:
    """
    Run the KMeans clusterer on a single user input and return the cluster
    index the breach is assigned to.

    Returns:
        Integer cluster index in [0, k).
    """
    model = load_kmeans()
    feature_row = _build_kmeans_feature_row(inputs)
    cluster_id = model.predict(feature_row)[0]
    return int(cluster_id)