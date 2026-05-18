from typing import List, Optional, Optional, Tuple
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.calibration import calibration_curve
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.constants import CLASS_LABELS, COLOUR_PALETTE, PER_CLASS_COLOURS

def get_original_feature(column_name: str, categorical_cols: Optional[List[str]] = None) -> str:
    """Returns the original feature name from a potentially encoded column"""
    if "__" in column_name:
        return column_name.split("__")[0]
    if categorical_cols is not None:
        for original_col in sorted(categorical_cols, key=len, reverse=True):
            if column_name.startswith(original_col + "_"):
                return original_col
    return column_name


def plot_vertical_bar_chart(
    series: pd.Series,
    title: str,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    sort_by: str = "value",
    figsize: Tuple[int, int] = (10, 6),
    annotate: str = "both",
    total: Optional[int] = None,
    rotation: int = 45,
    colour: List[str] = COLOUR_PALETTE,
    step: int = 100
) -> pd.DataFrame:
    """Plots a vertical bar chart"""

    # Sort the series by value or index as specified
    if sort_by == "value":
        series = series.sort_values(ascending=False)
    elif sort_by == "index":
        series = series.sort_index()
        
    categories = series.index.astype(str)
    counts = series.values
    denominator = total if total is not None else counts.sum()
    percentages = (counts / denominator) * 100 if denominator else counts * 0
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Handle both single colour and list of colours for bars
    if isinstance(colour, list):
        bar_colours = colour[:len(categories)]
    else:
        bar_colours = colour
    bars = ax.bar(categories, counts, color=bar_colours, edgecolor="white")
    
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel(xlabel if xlabel is not None else (series.index.name or ""))
    ax.set_ylabel(ylabel if ylabel is not None else "Count")
    ax.set_yticks(np.arange(0, counts.max() * 1.2, step))
    ax.set_xticks(range(len(categories)))
    ax.set_xticklabels(categories, ha="right", rotation=rotation if rotation else "center")
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Set y-axis limit slightly above max count
    if counts.max() > 0:
        ax.set_ylim(0, counts.max() * 1.15)
    
    if annotate and annotate != "none":
        for bar, count, percentage in zip(bars, counts, percentages):
            if annotate == "count":
                label = f"{count:,}"
            elif annotate == "percentage":
                label = f"{percentage:.1f}%"
            else:
                label = f"{percentage:.1f}%"
            
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                label,
                ha="center",
                va="bottom",
                fontsize=9
            )
            
    plt.tight_layout()
    plt.show()
    
    return pd.DataFrame({"count": counts, "percentage": percentages}, index=categories)


def plot_pie_chart(
    series: pd.Series,
    title: str,
    colours: List[str] = COLOUR_PALETTE,
    total: Optional[int] = None
) -> pd.DataFrame:
    """Plots a pie chart"""
    
    counts = series.values
    categories = series.index.astype(str)
    denominator = total if total is not None else counts.sum() 
    percentages = (counts / denominator) * 100 if denominator else counts * 0
    
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.pie(
        counts,
        labels=categories,
        autopct="%1.1f%%",
        startangle=90,
        colors=colours[:len(categories)],
        labeldistance=1.1,
        wedgeprops={"edgecolor": "white", "linewidth": 1},
        textprops={"fontsize": 10}
    )
    ax.axis("equal")
    plt.tight_layout()
    plt.show()
    
    return pd.DataFrame({"count": counts, "percentage": percentages.round(1)}, index=categories)


def plot_horizontal_stacked_crosstab(
    rows: pd.Series,
    cols: pd.Series,
    title: str,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    figsize: Tuple[int, int] = (16, 8),
    sort_by_size: bool = True,
    show_table: bool = True
):
    """Plots a horizontal stacked bar chart from a crosstab of two features"""
    xlabel = xlabel if xlabel is not None else ("Column")
    ylabel = ylabel if ylabel is not None else ("Row") 
    
    counts = pd.crosstab(rows, cols)
    proportions = pd.crosstab(rows, cols, normalize="index") 
    
    if sort_by_size:
        order = counts.sum(axis=1).sort_values(ascending=True).index
        counts = counts.loc[order]
        proportions = proportions.loc[order] 
    
    ax = proportions.plot(
        kind="barh",
        stacked=True,
        figsize=figsize,
        colormap="plasma",
        edgecolor="white",
        linewidth=0.5
    )
    
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel(f"Proportion of {xlabel}", fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_xticks(np.arange(0, 1.1, 0.1))
    ax.set_xticklabels([f"{int(x*100)}%" for x in ax.get_xticks()])
    ax.legend(title=ylabel, bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="x", linestyle="--", alpha=0.5)
    
    plt.tight_layout()
    plt.show()
    
    if show_table:
        return proportions.round(3), counts
    return None


def plot_vertical_stacked_crosstab(
    rows: pd.Series,
    cols: pd.Series,
    title: str,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    figsize: Tuple[int, int] = (16, 8),
    sort_by_size: bool = True,
    show_table: bool = True
):
    """Plots a vertical stacked bar chart from a crosstab of two features"""
    
    xlabel = xlabel if xlabel is not None else ("Row")
    ylabel = ylabel if ylabel is not None else ("Column")
    
    counts = pd.crosstab(rows, cols)
    proportions = pd.crosstab(rows, cols, normalize="index")
    
    if sort_by_size:
        order = counts.sum(axis=1).sort_values(ascending=True).index
        counts = counts.loc[order]
        proportions = proportions.loc[order]
    
    ax = proportions.plot(
        kind="bar",
        stacked=True,
        figsize=figsize,
        colormap="plasma",
        edgecolor="white",
        linewidth=0.5
    )
    
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_xlabel(ylabel, fontsize=14)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    ax.set_ylabel(xlabel, fontsize=14)
    ax.set_yticks(np.arange(0, 1.1, 0.1))
    ax.set_yticklabels([f"{int(x*100)}%" for x in np.arange(0, 1.1, 0.1)])
    ax.legend(title=xlabel, bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    
    plt.tight_layout()
    plt.show()
    
    if show_table:
        return proportions.round(3), counts
    return None


def plot_confusion_matrix(
    y_true,
    y_pred,
    class_labels: List[str] = CLASS_LABELS,
    title: str = "Confusion Matrix",
    figsize: Tuple[int, int] = (8, 6)
) -> None:
    """Plots a confusion matrix"""
    
    cm = confusion_matrix(y_true, y_pred, labels=class_labels)
    fig, ax = plt.subplots(figsize=figsize)
    display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
    display.plot(cmap="plasma", ax=ax, colorbar=False)
    ax.set_title(title, fontsize=12, fontweight="bold")
    plt.xticks(rotation=20, ha="right")
    
    plt.tight_layout()
    plt.show()


def plot_feature_importance(
    importances: np.ndarray,
    feature_names: List[str],
    title: str,
    categorical_cols: Optional[List[str]] = None,
    group_by_original_feature: bool = True,
    top_n: Optional[int] = None,
    figsize: Tuple[int, int] = (10, 6)
) -> pd.DataFrame:
    """Plots feature importances as a vertical bar chart"""
    importance_df = pd.DataFrame({"feature": feature_names, "importance": importances})

    # If grouping by original feature, sum importances of encoded features back to the original feature level
    if group_by_original_feature:
        importance_df["original_feature"] = importance_df["feature"].apply(lambda c: get_original_feature(c, categorical_cols))
        plot_df = importance_df.groupby("original_feature")["importance"].sum().sort_values(ascending=True)
    else:
        plot_df = importance_df.sort_values("importance", ascending=True)
        if top_n is not None:
            plot_df = plot_df.head(top_n)
        plot_df = plot_df.set_index("feature")["importance"]
    
    fig, ax = plt.subplots(figsize=figsize)
    plot_df.iloc[::-1].plot(kind="bar", ax=ax)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xticklabels(plot_df.index[::-1], rotation=45, ha="right")
    ax.set_xlabel("Original Feature", fontsize=10)
    ax.set_ylabel("Importance", fontsize=10)
    
    plt.tight_layout()
    plt.show()
    
    return importance_df

def plot_calibration_curve(
    y_true: pd.Series,
    y_prob_aligned: np.ndarray,
    class_labels: List[str] = CLASS_LABELS,
    n_bins: int = 10,
    title: str = "Calibration Plots (per-class)",
    figsize: Tuple[int, int] = (20, 16)
) -> None:
    """Plots calibration curves for each class"""
    fig, axes = plt.subplots(1, len(class_labels), figsize=figsize)
    if len(class_labels) == 1:
        axes = [axes]
        
    # Loop through each class and plot its calibration curve
    for i, (class_name, ax) in enumerate(zip(class_labels, axes)):
        y_true_binary = (y_true.values == class_name).astype(int)
        y_pred_prob = y_prob_aligned[:, i]
        
        try:
            prob_true, prob_pred = calibration_curve(y_true_binary, y_pred_prob, n_bins=n_bins, strategy="quantile")
            ax.plot(prob_pred, prob_true, marker="o", label="Model", linewidth=2)
        except ValueError as e:
            ax.text(0.5, 0.5, f"Insufficient data for calibration curve:\n{e}", ha="center", va="center", transform=ax.transAxes, fontsize=10)
            continue
        ax.plot([0, 1], [0, 1], linestyle="--", color="blue", label="Perfectly Calibrated", alpha=0.5)
            
        ax_histogram = ax.twinx()
        ax_histogram.hist(y_pred_prob, bins=n_bins, alpha=0.2, color="blue")
        ax_histogram.set_ylabel("Count of predictions", fontsize=9)
        ax_histogram.tick_params(axis="y", labelsize=9)
        
        ax.set_xlabel("Predicted Probability", fontsize=10)
        ax.set_ylabel("Actual Frequency", fontsize=10)
        ax.set_title(f"{class_name}", fontsize=12, fontweight="bold")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend(loc="upper left", fontsize=9)
        ax.grid(True, linestyle="--", alpha=0.5)
    
    plt.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.show()


def plot_cluster_distribution(
    cluster_labels: np.ndarray,
    decision_labels: pd.Series,
    class_labels: List[str] = CLASS_LABELS,
    title: str = "Decision Taken Distribution by Cluster",
    figsize: Tuple[int, int] = (10, 6)
) -> pd.DataFrame:
    """Plots the distribution of decision outcomes across clusters as a vertical stacked bar chart"""
    
    df = pd.DataFrame({"cluster": cluster_labels, "decision_taken": decision_labels})
    row_proportions = pd.crosstab(df["cluster"], df["decision_taken"], normalize="index").reindex(columns=class_labels, fill_value=0)
    
    fig, ax = plt.subplots(figsize=figsize)
    row_proportions.mul(100).plot(
        kind="bar",
        stacked=True,
        ax=ax,
        color=PER_CLASS_COLOURS,
        edgecolor="white"
    )
    
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Cluster", fontsize=10)
    ax.set_ylabel("Proportion of breaches (%)", fontsize=10)
    ax.set_xticklabels([f"Cluster {i}" for i in row_proportions.index], rotation=0)
    ax.legend(title="Decision Taken", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.show()
    return row_proportions

def plot_partition_bar_chart(
    df: pd.DataFrame,
    train_years: List[int],
    val_year: int,
    val_quarters: List[str],
    test_year: int,
    title: str = "Breaches per quarter, coloured by partition",
    figsize: Tuple[int, int] = (12, 6),
    step: int = 100
) -> pd.DataFrame:
    """
    Vertical bar chart of breach counts per quarter, with each bar coloured by which partition (train/ validation /test) the quarter belongs to
    """
    # Build chronological period label and count breaches per period
    df = df.copy()
    df["period"] = df["year"].astype(str) + " " + df["quarter"].astype(str)

    counts = (
        df.groupby(["year", "quarter", "period"], observed=True).size()
        .reset_index(name="count")
        .sort_values(["year", "quarter"])
    )

    # Assign each period to a partition
    def assign_partition(row):
        if row["year"] == test_year:
            return "Test"
        if row["year"] == val_year and row["quarter"] in val_quarters:
            return "Validation"
        return "Training"

    counts["partition"] = counts.apply(assign_partition, axis=1)

    # Map partitions to colours from your existing palette
    partition_colours = {
        "Training": COLOUR_PALETTE[3],
        "Validation": COLOUR_PALETTE[2],
        "Test": COLOUR_PALETTE[1],
    }
    bar_colours = [partition_colours[p] for p in counts["partition"]]

    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.bar(
        counts["period"],
        color=bar_colours,
        height=counts["count"],
        edgecolor="white"
    )

    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Quarter")
    ax.set_ylabel("Breaches reported")
    ax.set_yticks(np.arange(0, counts["count"].max() * 1.2, step))
    ax.set_xticks(range(len(counts["period"])))
    ax.set_xticklabels(counts["period"], rotation=45, ha="right")
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Count labels above each bar
    for bar, count in zip(bars, counts["count"]):
        if count > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{count:,}",
                ha="center", va="bottom", fontsize=9,
            )

    handles = [plt.Rectangle((0, 0), 1, 1, color=partition_colours[p]) for p in partition_colours]
    ax.legend(handles=handles, loc="best", frameon=False, labels=partition_colours.keys())

    plt.tight_layout()
    plt.show()

    return counts