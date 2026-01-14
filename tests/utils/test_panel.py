"""Tests for panel data utilities in yohou/utils/panel.py."""

from datetime import datetime, timedelta

import polars as pl
import pytest

from yohou.utils.panel import dict_to_panel, inspect_locality, select_panel_columns


def test_inspect_locality_global_data_single_column():
    """Test inspect_locality with global data (single non-panel column)."""
    df = pl.DataFrame({"time": [1, 2, 3], "value": [10.0, 20.0, 30.0]})

    global_names, panel_groups = inspect_locality(df)

    assert global_names == ["value"]
    assert panel_groups == {}


def test_inspect_locality_global_data_multiple_columns():
    """Test inspect_locality with multiple global columns."""
    df = pl.DataFrame(
        {
            "time": [1, 2, 3],
            "feature_1": [10.0, 20.0, 30.0],
            "feature_2": [100.0, 200.0, 300.0],
        }
    )

    global_names, panel_groups = inspect_locality(df)

    assert set(global_names) == {"feature_1", "feature_2"}
    assert panel_groups == {}


def test_inspect_locality_panel_data_single_group():
    """Test inspect_locality with panel data (with __ separator)."""
    df = pl.DataFrame(
        {
            "time": [1, 2, 3],
            "sales__store_1": [100, 110, 120],
            "sales__store_2": [150, 160, 170],
        }
    )

    global_names, panel_groups = inspect_locality(df)

    assert global_names == []
    assert panel_groups == {"sales": ["sales__store_1", "sales__store_2"]}


def test_inspect_locality_panel_data_multiple_groups():
    """Test inspect_locality with multiple panel groups."""
    df = pl.DataFrame(
        {
            "time": [1, 2, 3],
            "sales__store_1": [100, 110, 120],
            "sales__store_2": [150, 160, 170],
            "inventory__store_1": [50, 55, 60],
            "inventory__store_2": [75, 80, 85],
        }
    )

    global_names, panel_groups = inspect_locality(df)

    assert global_names == []
    assert panel_groups == {
        "sales": ["sales__store_1", "sales__store_2"],
        "inventory": ["inventory__store_1", "inventory__store_2"],
    }


def test_inspect_locality_mixed_global_and_panel_data():
    """Test inspect_locality with mix of global columns and panel columns."""
    df = pl.DataFrame(
        {
            "time": [1, 2, 3],
            "global_feature": [10.0, 20.0, 30.0],
            "sales__store_1": [100, 110, 120],
            "sales__store_2": [150, 160, 170],
        }
    )

    global_names, panel_groups = inspect_locality(df)

    assert global_names == ["global_feature"]
    assert panel_groups == {"sales": ["sales__store_1", "sales__store_2"]}


def test_inspect_locality_time_column_excluded():
    """Test that 'time' column is always excluded from inspect_locality results."""
    df = pl.DataFrame(
        {
            "time": [1, 2, 3],
            "value": [10.0, 20.0, 30.0],
            "feature": [100.0, 200.0, 300.0],
        }
    )

    global_names, panel_groups = inspect_locality(df)

    assert "time" not in global_names
    assert "time" not in panel_groups


def test_select_panel_columns_target_exclude_global():
    """Test filtering for target (y) - should exclude global features."""
    df = pl.DataFrame(
        {
            "time": [1, 2, 3],
            "global_feature": [10.0, 20.0, 30.0],
            "sales__store_1": [100, 110, 120],
            "sales__store_2": [150, 160, 170],
        }
    )

    result = select_panel_columns(df, panel_group_names=["sales"], include_global=False)

    assert set(result.columns) == {"time", "sales__store_1", "sales__store_2"}
    assert len(result) == 3


def test_select_panel_columns_features_include_global():
    """Test filtering for features (X) - should include global features."""
    df = pl.DataFrame(
        {
            "time": [1, 2, 3],
            "global_feature": [10.0, 20.0, 30.0],
            "sales": pl.Series(
                [
                    {"store_1": 100, "store_2": 150},
                    {"store_1": 110, "store_2": 160},
                    {"store_1": 120, "store_2": 170},
                ]
            ),
        }
    )

    result = select_panel_columns(df, panel_group_names=["sales"], include_global=True)

    assert set(result.columns) == {"time", "global_feature", "sales"}
    assert len(result) == 3


def test_select_panel_columns_multiple_panel_columns():
    """Test filtering when multiple panel groups exist - returns all groups."""
    df = pl.DataFrame(
        {
            "time": [1, 2, 3],
            "global_feature": [10.0, 20.0, 30.0],
            "sales__store_1": [100, 110, 120],
            "sales__store_2": [150, 160, 170],
            "inventory__store_1": [50, 55, 60],
            "inventory__store_2": [75, 80, 85],
        }
    )

    result = select_panel_columns(
        df,
        panel_group_names=["sales", "inventory"],
        include_global=False,
    )

    # Should keep time and ALL panel group columns (both sales and inventory)
    assert set(result.columns) == {
        "time",
        "sales__store_1",
        "sales__store_2",
        "inventory__store_1",
        "inventory__store_2",
    }
    assert len(result) == 3


def test_select_panel_columns_multiple_global_columns():
    """Test filtering preserves all global columns when include_global=True."""
    df = pl.DataFrame(
        {
            "time": [1, 2, 3],
            "global_1": [10.0, 20.0, 30.0],
            "global_2": [100.0, 200.0, 300.0],
            "sales": pl.Series(
                [
                    {"store_1": 100, "store_2": 150},
                    {"store_1": 110, "store_2": 160},
                    {"store_1": 120, "store_2": 170},
                ]
            ),
        }
    )

    result = select_panel_columns(df, panel_group_names=["sales"], include_global=True)

    assert set(result.columns) == {"time", "global_1", "global_2", "sales"}
    assert len(result) == 3


def test_select_panel_columns_none_panel_group_names():
    """Test that None panel_group_names returns DataFrame unchanged."""
    df = pl.DataFrame(
        {
            "time": [1, 2, 3],
            "value": [10.0, 20.0, 30.0],
            "feature": [100.0, 200.0, 300.0],
        }
    )

    result = select_panel_columns(df, panel_group_names=None, include_global=True)

    # Should return unchanged DataFrame
    assert result.columns == df.columns
    assert result.equals(df)


def test_select_panel_columns_empty_dataframe():
    """Test filtering with empty DataFrame."""
    df = pl.DataFrame(
        {
            "time": pl.Series([], dtype=pl.Int64),
            "global_feature": pl.Series([], dtype=pl.Float64),
            "sales__store_1": pl.Series([], dtype=pl.Int64),
            "sales__store_2": pl.Series([], dtype=pl.Int64),
        }
    )

    result = select_panel_columns(df, panel_group_names=["sales"], include_global=False)

    assert set(result.columns) == {"time", "sales__store_1", "sales__store_2"}
    assert len(result) == 0


def test_select_panel_columns_preserves_data_values():
    """Test that filtering preserves actual data values correctly."""
    df = pl.DataFrame(
        {
            "time": [1, 2, 3],
            "global_feature": [10.0, 20.0, 30.0],
            "sales": pl.Series(
                [
                    {"store_1": 100, "store_2": 150},
                    {"store_1": 110, "store_2": 160},
                    {"store_1": 120, "store_2": 170},
                ]
            ),
        }
    )

    result = select_panel_columns(df, panel_group_names=["sales"], include_global=True)

    # Verify data values are preserved
    assert result["time"].to_list() == [1, 2, 3]
    assert result["global_feature"].to_list() == [10.0, 20.0, 30.0]
    # Panel column preserved
    assert "sales" in result.columns


def test_select_panel_columns_datetime_time_column():
    """Test filtering with datetime time column (real-world scenario)."""
    time = pl.datetime_range(
        start=datetime(2021, 1, 1),
        end=datetime(2021, 1, 1) + timedelta(seconds=2),
        interval="1s",
        eager=True,
    )
    df = pl.DataFrame(
        {
            "time": time,
            "global_feature": [10.0, 20.0, 30.0],
            "sales__store_1": [100, 110, 120],
            "sales__store_2": [150, 160, 170],
        }
    )

    result = select_panel_columns(df, panel_group_names=["sales"], include_global=False)

    assert set(result.columns) == {"time", "sales__store_1", "sales__store_2"}
    assert result["time"].dtype == pl.Datetime


def test_select_panel_columns_forecaster_y_filtering_pattern():
    """Test filtering pattern used for _y_observed in forecasters - returns all groups."""
    # Simulate _y_observed with multiple panel groups (flat columns)
    df = pl.DataFrame(
        {
            "time": [1, 2, 3],
            "store_sales__store_1": [100, 110, 120],
            "store_sales__store_2": [150, 160, 170],
            "store_sales__store_3": [200, 210, 220],
            "online_sales__web": [50, 55, 60],
            "online_sales__app": [30, 35, 40],
        }
    )

    # Filter with all panel groups - should keep ALL group columns
    result = select_panel_columns(
        df,
        panel_group_names=["store_sales", "online_sales"],
        include_global=False,
    )

    assert set(result.columns) == {
        "time",
        "store_sales__store_1",
        "store_sales__store_2",
        "store_sales__store_3",
        "online_sales__web",
        "online_sales__app",
    }


def test_select_panel_columns_forecaster_x_filtering_pattern():
    """Test filtering pattern used for X in forecasters."""
    # Simulate X features with global and local columns
    df = pl.DataFrame(
        {
            "time": [1, 2, 3],
            "holiday": [0, 1, 0],  # Global feature
            "temperature": [20.0, 22.0, 21.0],  # Global feature
            "store_promotions": pl.Series(
                [
                    {"store_1": 0, "store_2": 1},
                    {"store_1": 1, "store_2": 0},
                    {"store_1": 0, "store_2": 1},
                ]
            ),
        }
    )

    # Filter (should keep global features)
    result = select_panel_columns(
        df,
        panel_group_names=["store_promotions"],
        include_global=True,
    )

    assert set(result.columns) == {"time", "holiday", "temperature", "store_promotions"}


def test_select_panel_columns_no_filtering_when_no_panel_data():
    """Test that global data passes through unchanged."""
    df = pl.DataFrame(
        {
            "time": [1, 2, 3],
            "value": [10.0, 20.0, 30.0],
            "feature": [100.0, 200.0, 300.0],
        }
    )

    # No panel data (panel_group_names=None or empty)
    result = select_panel_columns(df, panel_group_names=None, include_global=True)

    assert result.equals(df)


def test_inspect_locality_conflict_panel_and_global_same_name():
    """Test that inspect_locality raises error when panel and global columns conflict."""
    # Panel column 'x__a' has unprefixed name 'a' which conflicts with global 'a'
    df = pl.DataFrame(
        {
            "time": [1, 2, 3],
            "x__a": [1, 2, 3],
            "x__b": [4, 5, 6],
            "a": [7, 8, 9],  # Conflicts with unprefixed 'a' from 'x__a'
        }
    )

    with pytest.raises(
        ValueError,
        match=r"Panel column names .* conflict with global column names: \['a'\]",
    ):
        inspect_locality(df)


def test_inspect_locality_conflict_multiple_conflicts():
    """Test that inspect_locality detects multiple name conflicts."""
    df = pl.DataFrame(
        {
            "time": [1, 2, 3],
            "x__a": [1, 2, 3],
            "x__b": [4, 5, 6],
            "y__a": [7, 8, 9],
            "y__c": [10, 11, 12],
            "a": [13, 14, 15],  # Conflicts with 'a'
            "c": [16, 17, 18],  # Conflicts with 'c'
        }
    )

    with pytest.raises(
        ValueError,
        match=r"Panel column names .* conflict with global column names",
    ):
        result = inspect_locality(df)
        # Should contain both 'a' and 'c' in error


def test_inspect_locality_no_conflict_different_names():
    """Test that inspect_locality works when panel and global have different names."""
    df = pl.DataFrame(
        {
            "time": [1, 2, 3],
            "x__sales": [1, 2, 3],
            "x__inventory": [4, 5, 6],
            "temperature": [20, 21, 22],  # No conflict
            "holiday": [0, 1, 0],  # No conflict
        }
    )

    # Should not raise
    global_names, panel_groups = inspect_locality(df)

    assert set(global_names) == {"temperature", "holiday"}
    assert panel_groups == {"x": ["x__sales", "x__inventory"]}


# ============================================================================
# Tests for dict_to_panel
# ============================================================================


def test_dict_to_panel_single_group_single_column():
    """Test dict_to_panel with single group and single column."""
    data_dict = {"sales": pl.DataFrame({"time": [1, 2, 3], "store_1": [100, 110, 120]})}

    result = dict_to_panel(data_dict)

    assert set(result.columns) == {"time", "sales__store_1"}
    assert result["time"].to_list() == [1, 2, 3]
    assert result["sales__store_1"].to_list() == [100, 110, 120]


def test_dict_to_panel_single_group_multiple_columns():
    """Test dict_to_panel with single group and multiple columns."""
    data_dict = {
        "sales": pl.DataFrame(
            {
                "time": [1, 2, 3],
                "store_1": [100, 110, 120],
                "store_2": [150, 160, 170],
            }
        )
    }

    result = dict_to_panel(data_dict)

    assert set(result.columns) == {"time", "sales__store_1", "sales__store_2"}
    assert result["time"].to_list() == [1, 2, 3]
    assert result["sales__store_1"].to_list() == [100, 110, 120]
    assert result["sales__store_2"].to_list() == [150, 160, 170]


def test_dict_to_panel_multiple_groups():
    """Test dict_to_panel with multiple groups."""
    data_dict = {
        "sales": pl.DataFrame(
            {
                "time": [1, 2, 3],
                "store_1": [100, 110, 120],
                "store_2": [150, 160, 170],
            }
        ),
        "inventory": pl.DataFrame(
            {"time": [1, 2, 3], "warehouse_1": [50, 55, 60], "warehouse_2": [75, 80, 85]}
        ),
    }

    result = dict_to_panel(data_dict)

    assert set(result.columns) == {
        "time",
        "sales__store_1",
        "sales__store_2",
        "inventory__warehouse_1",
        "inventory__warehouse_2",
    }
    assert result["time"].to_list() == [1, 2, 3]
    assert result["sales__store_1"].to_list() == [100, 110, 120]
    assert result["inventory__warehouse_1"].to_list() == [50, 55, 60]


def test_dict_to_panel_dataframe_passthrough():
    """Test dict_to_panel returns DataFrame unchanged when given a DataFrame."""
    df = pl.DataFrame(
        {"time": [1, 2, 3], "sales__store_1": [100, 110, 120], "sales__store_2": [150, 160, 170]}
    )

    result = dict_to_panel(df)

    # Should return the exact same DataFrame
    assert result.equals(df)
    assert result is df  # Same object


def test_dict_to_panel_global_data_passthrough():
    """Test dict_to_panel with global data (non-prefixed columns)."""
    df = pl.DataFrame({"time": [1, 2, 3], "value": [10, 20, 30], "feature": [100, 200, 300]})

    result = dict_to_panel(df)

    # Should return unchanged
    assert result.equals(df)


def test_dict_to_panel_preserves_data_types():
    """Test dict_to_panel preserves column data types."""
    data_dict = {
        "sales": pl.DataFrame(
            {
                "time": pl.Series([1, 2, 3], dtype=pl.Int64),
                "store_1": pl.Series([100.5, 110.5, 120.5], dtype=pl.Float64),
                "store_2": pl.Series([150, 160, 170], dtype=pl.Int32),
            }
        )
    }

    result = dict_to_panel(data_dict)

    assert result["time"].dtype == pl.Int64
    assert result["sales__store_1"].dtype == pl.Float64
    assert result["sales__store_2"].dtype == pl.Int32


def test_dict_to_panel_datetime_time_column():
    """Test dict_to_panel with datetime time column."""
    time = pl.datetime_range(
        start=datetime(2021, 1, 1),
        end=datetime(2021, 1, 3),
        interval="1d",
        eager=True,
    )
    data_dict = {
        "sales": pl.DataFrame(
            {"time": time, "store_1": [100, 110, 120], "store_2": [150, 160, 170]}
        )
    }

    result = dict_to_panel(data_dict)

    assert result["time"].dtype == pl.Datetime
    assert result["time"].to_list() == time.to_list()


def test_dict_to_panel_empty_dict_returns_none():
    """Test dict_to_panel returns None for empty dict (vacuous truth in all())."""
    result = dict_to_panel({})
    assert result is None


def test_dict_to_panel_none_input_returns_none():
    """Test dict_to_panel returns None when input is None."""
    result = dict_to_panel(None)
    assert result is None


def test_dict_to_panel_dict_with_none_values_returns_none():
    """Test dict_to_panel returns None when all dict values are None."""
    result = dict_to_panel({"sales": None, "inventory": None})
    assert result is None


def test_dict_to_panel_time_alignment():
    """Test dict_to_panel correctly joins on time column."""
    data_dict = {
        "sales": pl.DataFrame({"time": [1, 2, 3, 4], "store_1": [100, 110, 120, 130]}),
        "inventory": pl.DataFrame({"time": [1, 2, 3, 4], "warehouse_1": [50, 55, 60, 65]}),
    }

    result = dict_to_panel(data_dict)

    # Should have inner join on time (all 4 rows)
    assert len(result) == 4
    assert result["time"].to_list() == [1, 2, 3, 4]


def test_dict_to_panel_inverse_of_get_group_df():
    """Test dict_to_panel is inverse operation of get_group_df pattern."""
    from yohou.utils.panel import get_group_df

    # Start with panel DataFrame
    original_panel = pl.DataFrame(
        {
            "time": [1, 2, 3],
            "sales__store_1": [100, 110, 120],
            "sales__store_2": [150, 160, 170],
        }
    )

    # Extract group
    schema = {"store_1": pl.Int64, "store_2": pl.Int64}
    group_df = get_group_df(original_panel, "sales", schema)

    # Convert back to panel
    data_dict = {"sales": group_df}
    result = dict_to_panel(data_dict)

    # Should match original (column order may differ)
    assert set(result.columns) == set(original_panel.columns)
    assert result["time"].to_list() == original_panel["time"].to_list()
    assert result["sales__store_1"].to_list() == original_panel["sales__store_1"].to_list()
    assert result["sales__store_2"].to_list() == original_panel["sales__store_2"].to_list()


def test_dict_to_panel_complex_multi_group_scenario():
    """Test dict_to_panel with realistic multi-group scenario."""
    data_dict = {
        "store_sales": pl.DataFrame(
            {
                "time": [1, 2, 3],
                "location_1": [100, 110, 120],
                "location_2": [150, 160, 170],
                "location_3": [200, 210, 220],
            }
        ),
        "online_sales": pl.DataFrame({"time": [1, 2, 3], "web": [50, 55, 60], "app": [30, 35, 40]}),
        "inventory": pl.DataFrame({"time": [1, 2, 3], "warehouse": [500, 510, 520]}),
    }

    result = dict_to_panel(data_dict)

    assert set(result.columns) == {
        "time",
        "store_sales__location_1",
        "store_sales__location_2",
        "store_sales__location_3",
        "online_sales__web",
        "online_sales__app",
        "inventory__warehouse",
    }
    assert len(result) == 3


def test_dict_to_panel_handles_different_dtypes_across_groups():
    """Test dict_to_panel with different data types in different groups."""
    data_dict = {
        "sales": pl.DataFrame(
            {"time": [1, 2, 3], "amount": pl.Series([100.5, 110.5, 120.5], dtype=pl.Float64)}
        ),
        "inventory": pl.DataFrame(
            {"time": [1, 2, 3], "count": pl.Series([50, 55, 60], dtype=pl.Int32)}
        ),
    }

    result = dict_to_panel(data_dict)

    assert result["sales__amount"].dtype == pl.Float64
    assert result["inventory__count"].dtype == pl.Int32


def test_dict_to_panel_use_case_from_decomposer():
    """Test dict_to_panel with exact pattern from decomposer.py fit() method."""
    # Simulate y_t from _pre_fit (could be dict or DataFrame)

    # Scenario 1: Panel data (dict from _pre_fit)
    y_t_dict = {
        "sales": pl.DataFrame(
            {"time": [1, 2, 3], "store_1": [100, 110, 120], "store_2": [150, 160, 170]}
        )
    }
    y_t_panel = dict_to_panel(y_t_dict)
    assert set(y_t_panel.columns) == {"time", "sales__store_1", "sales__store_2"}

    # Scenario 2: Global data (DataFrame from _pre_fit)
    y_t_global = pl.DataFrame({"time": [1, 2, 3], "value": [10, 20, 30]})
    y_t_result = dict_to_panel(y_t_global)
    assert y_t_result.equals(y_t_global)


def test_dict_to_panel_column_order_deterministic():
    """Test dict_to_panel produces consistent column order within groups."""
    data_dict = {
        "sales": pl.DataFrame({"time": [1, 2, 3], "a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]})
    }

    result1 = dict_to_panel(data_dict)
    result2 = dict_to_panel(data_dict)

    # Column order should be deterministic
    assert result1.columns == result2.columns


def test_dict_to_panel_empty_dataframe_in_dict():
    """Test dict_to_panel with empty DataFrame in dict."""
    data_dict = {
        "sales": pl.DataFrame(
            {
                "time": pl.Series([], dtype=pl.Int64),
                "store_1": pl.Series([], dtype=pl.Int64),
            }
        )
    }

    result = dict_to_panel(data_dict)

    assert set(result.columns) == {"time", "sales__store_1"}
    assert len(result) == 0
