from typing import List

import pandas as pd
import streamlit as st

from components.disclaimer import render_disclaimer
from components.explore_filter import apply_filter, render_filter
from components.header import render_header
from components.charts import (
    horizontal_stacked_proportions,
    time_trend_counts,
    time_trend_proportions,
    vertical_stacked_proportions_by_feature,
)
from utils.artefacts import (
    load_baseline_probabilities,
    load_processed_dataset,
)


_DISPLAY_ORDER: List[str] = [
    "Investigation Pursued",
    "Informal Action Taken",
    "No Further Action",
]

# Feature display order on the page, with user-facing labels.
_FEATURE_PANELS: List[tuple] = [
    ("incident_category", "Incident Category"),
    ("incident_type", "Incident Type"),
    ("data_subject_type", "Data Subject Type"),
    ("data_type", "Data Type"),
    ("no_data_subjects_affected", "Number of Data Subjects Affected"),
    ("time_taken_to_report", "Time Taken to Report"),
]

# Multi-valued features: their cells are comma-separated strings. 
# Need to explode before charting, otherwise "Patients, Employees" shows as its own bar separate from "Patients".
_MULTILABEL_COLS = {"data_subject_type", "data_type"}


def _explode_multilabel(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """
    Split a comma-separated multilabel column into individual rows.

    A breach with data_subject_type="Patients, Employees" becomes two rows
    in the returned dataframe — one tagged Patients, one tagged Employees.
    Other columns are duplicated.
    """
    exploded = df.copy()
    exploded[col] = exploded[col].astype(str).str.split(", ")
    return exploded.explode(col).reset_index(drop=True)


def _quarter_period_order(df: pd.DataFrame) -> List[str]:
    """
    Build the chronological order of (year, quarter) labels present in the
    filtered data, formatted as e.g. '2023 Qtr 2'.
    """
    pairs = (
        df[["year", "quarter"]].drop_duplicates()
        .sort_values(["year", "quarter"])
    )
    return [f"{row.year} {row.quarter}" for row in pairs.itertuples(index=False)]


def _attach_period(df: pd.DataFrame) -> pd.DataFrame:
    """Attach a chronological 'period' column combining year and quarter."""
    out = df.copy()
    out["period"] = out["year"].astype(str) + " " + out["quarter"].astype(str)
    return out


def main() -> None:
    render_header(
        title="Explore the Data",
        subtitle="Browse the historical UK healthcare data breach incidents the model is trained on.",
    )
    render_disclaimer()

    df_full = load_processed_dataset()
    selection = render_filter(df_full)
    df = apply_filter(df_full, selection)
    
    # CSV export of the currently filtered dataset
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download current view as CSV",
        data=csv_bytes,
        file_name=f"breach_data_{len(df)}_rows.csv",
        mime="text/csv",
        help="Download the breaches matching the current year/quarter filter.",
    )

    if df.empty:
        st.warning("No breaches match the current filter. Adjust the filter to see data.")
        return

    # About the data
    st.subheader("About the data in view")

    col_total, col_filtered, col_outcomes = st.columns(3)
    col_total.metric("Total breaches in dataset", f"{len(df_full):,}")
    col_filtered.metric("Breaches in current view", f"{len(df):,}")
    col_outcomes.metric("Distinct decision outcomes", len(_DISPLAY_ORDER))

    st.markdown("**Decision distribution in current view**")
    decision_counts = df["decision_taken"].value_counts()
    decision_proportions = {
        label: float(decision_counts.get(label, 0)) / len(df)
        for label in _DISPLAY_ORDER
    }

    chart_tab, table_tab = st.tabs(["Chart", "Table"])
    with chart_tab:
        fig = horizontal_stacked_proportions(
            proportions=decision_proportions,
            class_order=_DISPLAY_ORDER,
        )
        st.plotly_chart(fig, width='stretch', config={"displayModeBar": False})
    with table_tab:
        decision_table = pd.DataFrame({
            "Decision": _DISPLAY_ORDER,
            "Breaches": [decision_counts.get(label, 0) for label in _DISPLAY_ORDER],
            "Share": [f"{decision_proportions[label]:.1%}" for label in _DISPLAY_ORDER],
        })
        st.dataframe(decision_table, width='stretch', hide_index=True)

    st.divider()

    # Decision outcomes by feature
    st.subheader("Decision outcomes by feature")
    st.caption(
        "Each bar shows the proportion of regulatory decisions for breaches "
        "with that feature value. Bars sum to 100% within each value, so "
        "tall sections of one colour indicate that feature value's "
        "characteristic decision pattern."
    )

    for feature_col, feature_label in _FEATURE_PANELS:
        st.markdown(f"#### {feature_label}")

        # Multi-valued features need exploding before counting.
        df_for_feature = (
            _explode_multilabel(df, feature_col)
            if feature_col in _MULTILABEL_COLS
            else df
        )

        chart_tab, table_tab = st.tabs(["Chart", "Table"])
        with chart_tab:
            fig = vertical_stacked_proportions_by_feature(
                df=df_for_feature,
                feature_col=feature_col,
                target_col="decision_taken",
                class_order=_DISPLAY_ORDER,
            )
            st.plotly_chart(fig, width='stretch', config={"displayModeBar": False})
        with table_tab:
            crosstab = pd.crosstab(
                df_for_feature[feature_col],
                df_for_feature["decision_taken"],
            ).reindex(columns=_DISPLAY_ORDER, fill_value=0)
            crosstab["Total"] = crosstab.sum(axis=1)
            crosstab = crosstab.sort_values("Total", ascending=False)
            st.dataframe(crosstab, width='stretch')

    st.divider()

    # Breaches over time
    st.subheader("Breaches over time")
    st.caption(
        "How the volume and decision mix of reported breaches has changed "
        "across the training period."
    )

    df_time = _attach_period(df)
    period_order = _quarter_period_order(df)

    view_mode = st.radio(
        "Show",
        options=["Proportions", "Counts"],
        horizontal=True,
        label_visibility="collapsed",
    )

    chart_tab, table_tab = st.tabs(["Chart", "Table"])
    with chart_tab:
        if view_mode == "Counts":
            fig = time_trend_counts(
                df=df_time,
                period_col="period",
                target_col="decision_taken",
                class_order=_DISPLAY_ORDER,
                period_order=period_order,
            )
        else:
            fig = time_trend_proportions(
                df=df_time,
                period_col="period",
                target_col="decision_taken",
                class_order=_DISPLAY_ORDER,
                period_order=period_order,
            )
        st.plotly_chart(fig, width='stretch', config={"displayModeBar": False})
    with table_tab:
        time_table = (
            pd.crosstab(df_time["period"], df_time["decision_taken"])
            .reindex(index=period_order, columns=_DISPLAY_ORDER, fill_value=0)
        )
        time_table["Total"] = time_table.sum(axis=1)
        st.dataframe(time_table, width='stretch')


if __name__ == "__main__":
    main()