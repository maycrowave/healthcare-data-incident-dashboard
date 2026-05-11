from typing import List, NamedTuple, Tuple

import pandas as pd
import streamlit as st


class ExploreFilter(NamedTuple):
    """The current filter state."""
    years: List[int]
    quarters: List[str]


def render_filter(df: pd.DataFrame) -> ExploreFilter:
    """
    Render the year/quarter filter and return the selection.

    Defaults to all years and all quarters selected.
    """
    all_years = sorted(df["year"].unique().tolist())
    all_quarters = sorted(df["quarter"].unique().tolist())

    with st.expander("Filter the data", expanded=False):
        col_year, col_quarter = st.columns(2)
        with col_year:
            selected_years = st.multiselect(
                "Years",
                options=all_years,
                default=all_years,
                placeholder="Select one or more years",
            )
        with col_quarter:
            selected_quarters = st.multiselect(
                "Quarters",
                options=all_quarters,
                default=all_quarters,
                placeholder="Select one or more quarters",
            )

    return ExploreFilter(
        years=selected_years or all_years,
        quarters=selected_quarters or all_quarters,
    )


def apply_filter(df: pd.DataFrame, selection: ExploreFilter) -> pd.DataFrame:
    """Apply the filter selection to a dataframe."""
    return df[
        df["year"].isin(selection.years)
        & df["quarter"].isin(selection.quarters)
    ].copy()