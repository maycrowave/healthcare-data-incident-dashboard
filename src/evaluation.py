from typing import List, Tuple, Dict, Any, Optional, Union
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    average_precision_score,
    log_loss,
    ConfusionMatrixDisplay,
)
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
from src.constants import (CLASS_LABELS, RANDOM_STATE)

DEFAULT_CATBOOST_HYPERPARAMS = {
    "iterations": 2000,
    "learning_rate": 0.03,
    "depth": 6,
    "l2_leaf_reg": 3,
    "random_seed": RANDOM_STATE,
    "loss_function": "MultiClass",
    "eval_metric": "TotalF1:average=Macro",
    "early_stopping_rounds": 50,
    "verbose": False
}

DEFAULT_RF_HYPERPARAMS = {
    "n_estimators": 100,
    "max_depth": None,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
    "class_weight": "balanced",
    "max_features": "sqrt"
}

def evaluate_classification_model(
    model_name: str,
    model, 
    X_eval: pd.DataFrame,
    y_eval: pd.Series,
    class_labels: List[str] = CLASS_LABELS,
    show_plot: bool = True,
) -> Dict[str, Any]:
    """Evaluate a classification model using various metrics and visualizations."""
    
    print(f"Evaluating {model_name}\n")
    
    # Generate predictions and probabilities
    y_pred = model.predict(X_eval)
    if y_pred.ndim > 1:
        y_pred = y_pred.flatten()
    y_probs = model.predict_proba(X_eval)
    
    # Align probabilities with the target class labels
    probs_class_order = list(model.classes_)
    probs_alignment = np.column_stack([y_probs[:, probs_class_order.index(c)] for c in class_labels])
    
    # Convert y_eval to binary format for multi-class ROC AUC and PR AUC calculations
    y_true_binary = np.zeros((len(y_eval), len(class_labels)), dtype=int)
    for i, label in enumerate(class_labels):
        y_true_binary[:, i] = (y_eval == label).astype(int)
        
    # Calculate metrics
    accuracy = accuracy_score(y_eval, y_pred)
    macro_f1 = f1_score(y_eval, y_pred, average='macro', zero_division=0)
    macro_precision = precision_score(y_eval, y_pred, average='macro', zero_division=0)
    macro_recall = recall_score(y_eval, y_pred, average='macro', zero_division=0)
    
    # Handle cases where ROC AUC or PR AUC cannot be calculated due to a lack of positive samples
    try:
        macro_roc_auc = roc_auc_score(y_true_binary, probs_alignment, average='macro', multi_class='ovr')
    except ValueError:
        macro_roc_auc = float('nan')
    try:
        macro_pr_auc = average_precision_score(y_true_binary, probs_alignment, average='macro')
    except ValueError:
        macro_pr_auc = float('nan')
    try:
        logloss = log_loss(y_eval, probs_alignment, labels=class_labels)
    except ValueError:
        logloss = float('nan')
        
    # Calculate per-class F1 scores and ROC AUC
    per_class_f1 = f1_score(y_eval, y_pred, labels=class_labels, average=None, zero_division=0)
    per_class_roc_auc = []
    for i, label in enumerate(class_labels):
        try:
            auc = roc_auc_score(y_true_binary[:, i], probs_alignment[:, i])
        except ValueError:
            auc = float('nan')
        per_class_roc_auc.append(auc)
    
    # Print results    
    print("\nClassification Report:")
    print(classification_report(y_eval, y_pred, labels=class_labels, zero_division=0))
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro F1 Score: {macro_f1:.4f}")
    print(f"Macro Precision: {macro_precision:.4f}")
    print(f"Macro Recall: {macro_recall:.4f}")
    print(f"Macro ROC AUC: {macro_roc_auc:.4f}")
    print(f"Macro PR AUC: {macro_pr_auc:.4f}")
    print(f"Log Loss: {logloss:.4f}")
    
    print("\nPer-Class F1 Scores:")
    for label, f1, roc_auc in zip(class_labels, per_class_f1, per_class_roc_auc):
        print(f"\t{label}: F1 Score: {f1:.4f}, ROC AUC: {roc_auc:.4f}")
    
    # Generate confusion matrix
    cm = confusion_matrix(y_eval, y_pred, labels=class_labels)
    if show_plot:
        fig, ax = plt.subplots(figsize=(8, 6))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
        disp.plot(cmap=plt.cm.plasma, ax=ax, colorbar=False)
        ax.set_title(f"Confusion Matrix: {model_name}", fontsize=14, fontweight='bold', pad=12)
        plt.xticks(rotation=20, ha='right')
        plt.tight_layout()
        plt.show()
    
    # Print confusion matrix table
    print(f"Confusion Matrix table:\n")
    print(pd.DataFrame(cm, index=class_labels, columns=class_labels))
    
    # Compile results into a dict
    result = {
        "model": model_name,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_roc_auc": macro_roc_auc,
        "macro_pr_auc": macro_pr_auc,
        "log_loss": logloss,
    }
    
    # Add per-class metrics to the result dict
    for label, f1, roc_auc in zip(class_labels, per_class_f1, per_class_roc_auc):
        result[f"f1_{label}"] = round(f1, 4)
        result[f"roc_auc_{label}"] = round(roc_auc, 4)

    return result

def fit_catboost_classifier(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    X_val,
    y_val,
    class_labels: List[str] = CLASS_LABELS,
    cat_feature_indices: Optional[List[int]] = None,
    class_weights: Optional[Union[str, Dict[str, float]]] = None,
    hyperparameters: Optional[Dict[str, Any]] = None,
) -> Tuple[CatBoostClassifier, Dict[str, Any]]:
    """Fit a CatBoostClassifier and evaluate it on the test set."""
    
    kwargs = DEFAULT_CATBOOST_HYPERPARAMS.copy()
    if hyperparameters is not None:
        kwargs.update(hyperparameters) 
    
    if cat_feature_indices is not None:
        kwargs["cat_features"] = cat_feature_indices
    if class_weights == "Balanced":
        kwargs["auto_class_weights"] = "Balanced"
    elif isinstance(class_weights, dict):
        kwargs["class_weights"] = class_weights
        
    model = CatBoostClassifier(**kwargs)
    model.fit(X_train, y_train, eval_set=(X_val, y_val))
    
    evaluation_results = evaluate_classification_model(
        model_name=model_name,
        model=model,
        X_eval=X_test,
        y_eval=y_test,
        class_labels=class_labels
    )
    return model, evaluation_results
    
def fit_random_forest_classifier(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    class_labels: List[str] = CLASS_LABELS,
    class_weights: Optional[Union[str, Dict[str, float]]] = None,
    hyperparameters: Optional[Dict[str, Any]] = None,
) -> Tuple[RandomForestClassifier, Dict[str, Any]]:
    """Fit a RandomForestClassifier and evaluate it on the test set."""
    
    kwargs = DEFAULT_RF_HYPERPARAMS.copy()
    if hyperparameters is not None:
        kwargs.update(hyperparameters)
    if class_weights is not None:
        if class_weights == "balanced":
            kwargs["class_weight"] = "balanced"
        elif isinstance(class_weights, dict):   
            kwargs["class_weight"] = class_weights
    else:
        kwargs["class_weight"] = None
    
    model = RandomForestClassifier(**kwargs)
    model.fit(X_train, y_train)
    
    evaluation_results = evaluate_classification_model(
        model_name=model_name,
        model=model,
        X_eval=X_test,
        y_eval=y_test,
        class_labels=class_labels
    )
    return model, evaluation_results

def compare_results_table(
    results_list: List[Dict[str, Any]],
    class_labels: List[str] = CLASS_LABELS,
    print_table: bool = True
) -> pd.DataFrame:
    """Compare evaluation results from multiple models in a single table."""
    
    results_df = pd.DataFrame(results_list)
    col_headings = ["model", "accuracy", "macro_f1", "macro_precision", "macro_recall", "macro_roc_auc", "macro_pr_auc", "log_loss"]
    
    if print_table:
        print("Metric Headings:")
        print(results_df[col_headings].to_string(index=False))
        
        print("Ranked by macro F1 Score:")
        print(results_df[col_headings].sort_values(by="macro_f1", ascending=False).to_string(index=False))
        
        print("Ranked by macro ROC-AUC:")
        print(results_df[col_headings].sort_values(by="macro_roc_auc", ascending=False).to_string(index=False))
        
        print("Ranked by Log Loss:")
        print(results_df[col_headings].sort_values(by="log_loss", ascending=True).to_string(index=False))
        
        per_class_f1_cols = ["model"] + [f"f1_{label}" for label in class_labels]
        print("Per-Class F1 Scores:")
        print(results_df[per_class_f1_cols].to_string(index=False))
        
        per_class_auc_cols = ["model"] + [f"roc_auc_{label}" for label in class_labels]
        print("Per-Class ROC-AUC Scores:")
        print(results_df[per_class_auc_cols].to_string(index=False))
        
    return results_df

