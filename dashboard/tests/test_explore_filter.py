import pandas as pd
import pytest
from components.explore_filter import ExploreFilter, apply_filter

@pytest.fixture
def sample_df():
    """A small dataset spanning two years and several quarters"""
    return pd.DataFrame({
        "year": [2022, 2022, 2023, 2023, 2024, 2024],
        "quarter": ["Qtr 1", "Qtr 4", "Qtr 1", "Qtr 4", "Qtr 1", "Qtr 4"],
        "decision_taken": ["Informal Action Taken"] * 6
    })

class TestApplyFilter:
    def test_no_filtering_returns_all_rows(self, sample_df):
        selection = ExploreFilter(years=[2022, 2023, 2024], quarters=["Qtr 1", "Qtr 4"])
        result = apply_filter(sample_df, selection)
        assert len(result) == len(sample_df)

    def test_year_filtering_excludes_other_years(self, sample_df):
        selection = ExploreFilter(years=[2023], quarters=["Qtr 1", "Qtr 4"])
        result = apply_filter(sample_df, selection)
        assert len(result) == 2
        assert set(result["year"].unique()) == {2023}

    def test_quarter_filtering_excludes_other_quarters(self, sample_df):
        selection = ExploreFilter(years=[2022, 2023, 2024], quarters=["Qtr 1"])
        result = apply_filter(sample_df, selection)
        assert len(result) == 3
        assert set(result["quarter"].unique()) == {"Qtr 1"}

    def test_both_filters_combine(self, sample_df):
        selection = ExploreFilter(years=[2023, 2024], quarters=["Qtr 4"])
        result = apply_filter(sample_df, selection)
        assert len(result) == 2
        assert set(result["year"].unique()) == {2023, 2024}
        assert set(result["quarter"].unique()) == {"Qtr 4"}

    def test_returns_copy_not_view(self, sample_df):
        original_len = len(sample_df)
        selection = ExploreFilter(years=[2022], quarters=["Qtr 1"])
        _ = apply_filter(sample_df, selection)
        assert len(sample_df) == original_len