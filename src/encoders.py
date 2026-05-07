from pathlib import Path
from typing import Dict, List, Tuple
import pickle
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder

from src.constants import (MULTILABEL_COLS, CATEGORICAL_COLS, ARTIFACTS_PATH)

def split_multilabel_values(series: pd.Series):
    """Split mutli-valued features into lists for MultiLabelBinarizer."""
    return series.fillna("").astype(str).apply(lambda value: [item.strip() for item in value.split(",")])

def fit_mlb_encoder(df_train: pd.DataFrame, multilabel_cols: List[str] = MULTILABEL_COLS) -> Dict[str, MultiLabelBinarizer]:
    """Fit MultiLabelBinarizer encoders for the multilabel columns."""
    mlb_encoders = {}
    for col in multilabel_cols:
        mlb = MultiLabelBinarizer()
        mlb.fit(split_multilabel_values(df_train[col]))
        mlb_encoders[col] = mlb
        
    return mlb_encoders

def transform_mlb_columns(df: pd.DataFrame, mlb_encoders: Dict[str, MultiLabelBinarizer], multilabel_cols: List[str] = MULTILABEL_COLS) -> pd.DataFrame:
    """Transform the multilabel columns using the fitted MultiLabelBinarizer encoders."""
    encoded_dframes = []
    for col in multilabel_cols:
        values_list = split_multilabel_values(df[col])
        encoder = mlb_encoders[col]
        encoded_col = pd.DataFrame(
            encoder.transform(values_list),
            columns=[f"{col}__{cls}" for cls in encoder.classes_],
            index=df.index
        )
        encoded_dframes.append(encoded_col)
        
    return pd.concat(encoded_dframes, axis=1)

def fit_ohe_encoder(df_train: pd.DataFrame, categorical_cols: List[str] = CATEGORICAL_COLS) -> OneHotEncoder:
    """Fit a OneHotEncoder for the categorical columns."""
    ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    ohe.fit(df_train[categorical_cols])
    return ohe

def transform_ohe_columns(df: pd.DataFrame, ohe_encoder: OneHotEncoder, categorical_cols: List[str] = CATEGORICAL_COLS) -> pd.DataFrame:
    """Transform the categorical columns using the fitted OneHotEncoder."""
    return pd.DataFrame(
        ohe_encoder.transform(df[categorical_cols]),
        columns=ohe_encoder.get_feature_names_out(categorical_cols),
        index=df.index
    )
    
def build_mlb_cat_matrix(df: pd.DataFrame, mlb_encoders: Dict[str, MultiLabelBinarizer], multilabel_cols: List[str] = MULTILABEL_COLS, categorical_cols: List[str] = CATEGORICAL_COLS) -> Tuple[pd.DataFrame, List[int]]:
    """Build the full encoded feature matrix by combining the multilabel and categorical encodings."""
    raw_categorical = df[categorical_cols].reset_index(drop=True)
    mlb_encoded = transform_mlb_columns(df, mlb_encoders, multilabel_cols).reset_index(drop=True)
    feature_matrix = pd.concat([raw_categorical, mlb_encoded], axis=1).reset_index(drop=True)
    cat_feature_indices = [feature_matrix.columns.get_loc(col) for col in categorical_cols]
    
    return feature_matrix, cat_feature_indices

def build_mlb_ohe_matrix(df: pd.DataFrame, mlb_encoders: Dict[str, MultiLabelBinarizer], ohe_encoder: OneHotEncoder, multilabel_cols: List[str] = MULTILABEL_COLS, categorical_cols: List[str] = CATEGORICAL_COLS) -> pd.DataFrame:
    """Build the full encoded feature matrix by combining the multilabel and one-hot encodings."""
    mlb_encoded = transform_mlb_columns(df, mlb_encoders, multilabel_cols).reset_index(drop=True)
    ohe_encoded = transform_ohe_columns(df, ohe_encoder, categorical_cols).reset_index(drop=True)
    feature_matrix = pd.concat([ohe_encoded, mlb_encoded], axis=1).reset_index(drop=True)
    return feature_matrix

def save_encoders(mlb_encoders: Dict[str, MultiLabelBinarizer], ohe_encoder: OneHotEncoder, output_dir: str = ARTIFACTS_PATH) -> None:
    """Save the fitted encoders to disk using pickle."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    for col, encoder in mlb_encoders.items():
        with open(output_path / f"mlb_{col}.pkl", "wb") as f:
            pickle.dump(encoder, f)
    
    with open(output_path / "ohe.pkl", "wb") as f:
        pickle.dump(ohe_encoder, f)
        
def load_encoders(multilabel_cols: List[str] = MULTILABEL_COLS, output_dir: str = ARTIFACTS_PATH) -> Tuple[Dict[str, MultiLabelBinarizer], OneHotEncoder]:
    """Load the fitted encoders from disk using pickle."""
    input_path = Path(output_dir)
    
    mlb_encoders = {}
    for col in multilabel_cols:
        with open(input_path / f"mlb_{col}.pkl", "rb") as f:
            mlb_encoders[col] = pickle.load(f)
    
    with open(input_path / "ohe.pkl", "rb") as f:
        ohe_encoder = pickle.load(f)
        
    return mlb_encoders, ohe_encoder
