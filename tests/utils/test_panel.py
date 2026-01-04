"""Tests for panel data utilities in yohou/utils/panel.py."""

from datetime import datetime, timedelta

import polars as pl

from yohou.utils.panel import filter_panel_columns, inspect_locality


def test_inspect_locality_global_data_single_column():
    """Test inspect_locality with global data (single non-panel column)."""
    df = pl.DataFrame({"time": [1, 2, 3], "value": [10.0, 20.0, 30.0]})

    global_names, local_groups = inspect_locality(df)

    assert global_names == ["value"]
    assert local_groups == {}


def test_inspect_locality_global_data_multiple_columns():
    """Test inspect_locality with multiple global columns."""
    df = pl.DataFrame(
        {
            "time": [1, 2, 3],
            "feature_1": [10.0, 20.0, 30.0],
            "feature_2": [100.0, 200.0, 300.0],
        }
    )

    global_names, local_groups = inspect_locality(df)

    assert set(global_names) == {"feature_1", "feature_2"}
    assert local_groups == {}


def test_inspect_locality_panel_data_single_group():
    """Test inspect_locality with panel data (with __ separator)."""
    df = pl.DataFrame(
        {
            "time": [1, 2, 3],
            "sales__store_1": [100, 110, 120],
            "sales__store_2": [150, 160, 170],
        }
    )

    global_names, local_groups = inspect_locality(df)

    assert global_names == []
    assert local_groups == {"sales": ["sales__store_1", "sales__store_2"]}


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

    global_names, local_groups = inspect_locality(df)

    assert global_names == []
    assert local_groups == {
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

    global_names, local_groups = inspect_locality(df)

    assert global_names == ["global_feature"]
    assert local_groups == {"sales": ["sales__store_1", "sales__store_2"]}


def test_inspect_locality_time_column_excluded():
    """Test that 'time' column is always excluded from inspect_locality results."""
    df = pl.DataFrame(
        {
            "time": [1, 2, 3],
            "value": [10.0, 20.0, 30.0],
            "feature": [100.0, 200.0, 300.0],
        }
    )

    global_names, local_groups = inspect_locality(df)

    assert "time" not in global_names
    assert "time" not in local_groups


def test_filter_panel_columns_target_exclude_global():
    """Test filtering for target (y) - should exclude global features."""
    df = pl.DataFrame(
        {
            "time": [1, 2, 3],
            "global_feature": [10.0, 20.0, 30.0],
            "sales__store_1": [100, 110, 120],
            "sales__store_2": [150, 160, 170],
        }
    )

    result = filter_panel_columns(
        df, cross_learning_group="sales", local_group_names=["sales"], include_global=False
    )

    assert set(result.columns) == {"time", "sales__store_1", "sales__store_2"}
    assert len(result) == 3


def test_filter_panel_columns_features_include_global():
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

    result = filter_panel_columns(
        df, cross_learning_group="sales", local_group_names=["sales"], include_global=True
    )

    assert set(result.columns) == {"time", "global_feature", "sales"}
    assert len(result) == 3


def test_filter_panel_columns_multiple_panel_columns():
    """Test filtering when multiple panel groups exist (flat columns)."""
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

    result = filter_panel_columns(
        df,
        cross_learning_group="sales",
        local_group_names=["sales", "inventory"],
        include_global=False,
    )

    # Should keep only time and sales columns (not inventory)
    assert set(result.columns) == {"time", "sales__store_1", "sales__store_2"}
    assert len(result) == 3


def test_filter_panel_columns_multiple_global_columns():
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

    result = filter_panel_columns(
        df, cross_learning_group="sales", local_group_names=["sales"], include_global=True
    )

    assert set(result.columns) == {"time", "global_1", "global_2", "sales"}
    assert len(result) == 3


def test_filter_panel_columns_none_local_group_names():
    """Test that None local_group_names returns DataFrame unchanged."""
    df = pl.DataFrame(
        {
            "time": [1, 2, 3],
            "value": [10.0, 20.0, 30.0],
            "feature": [100.0, 200.0, 300.0],
        }
    )

    result = filter_panel_columns(
        df, cross_learning_group="value", local_group_names=None, include_global=True
    )

    # Should return unchanged DataFrame
    assert result.columns == df.columns
    assert result.equals(df)


def test_filter_panel_columns_empty_dataframe():
    """Test filtering with empty DataFrame."""
    df = pl.DataFrame(
        {
            "time": pl.Series([], dtype=pl.Int64),
            "global_feature": pl.Series([], dtype=pl.Float64),
            "sales__store_1": pl.Series([], dtype=pl.Int64),
            "sales__store_2": pl.Series([], dtype=pl.Int64),
        }
    )

    result = filter_panel_columns(
        df, cross_learning_group="sales", local_group_names=["sales"], include_global=False
    )

    assert set(result.columns) == {"time", "sales__store_1", "sales__store_2"}
    assert len(result) == 0


def test_filter_panel_columns_preserves_data_values():
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

    result = filter_panel_columns(
        df, cross_learning_group="sales", local_group_names=["sales"], include_global=True
    )

    # Verify data values are preserved
    assert result["time"].to_list() == [1, 2, 3]
    assert result["global_feature"].to_list() == [10.0, 20.0, 30.0]
    # Panel column preserved
    assert "sales" in result.columns


def test_filter_panel_columns_datetime_time_column():
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

    result = filter_panel_columns(
        df, cross_learning_group="sales", local_group_names=["sales"], include_global=False
    )

    assert set(result.columns) == {"time", "sales__store_1", "sales__store_2"}
    assert result["time"].dtype == pl.Datetime


def test_filter_panel_columns_forecaster_y_filtering_pattern():
    """Test filtering pattern used for _y_observed in forecasters."""
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

    # Filter for cross-learning on store_sales
    result = filter_panel_columns(
        df,
        cross_learning_group="store_sales",
        local_group_names=["store_sales", "online_sales"],
        include_global=False,
    )

    assert set(result.columns) == {"time", "store_sales__store_1", "store_sales__store_2", "store_sales__store_3"}
    assert "online_sales__web" not in result.columns
    assert "online_sales__app" not in result.columns
    assert "online_sales" not in result.columns


def test_filter_panel_columns_forecaster_x_filtering_pattern():
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

    # Filter for cross-learning (should keep global features)
    result = filter_panel_columns(
        df,
        cross_learning_group="store_promotions",
        local_group_names=["store_promotions"],
        include_global=True,
    )

    assert set(result.columns) == {"time", "holiday", "temperature", "store_promotions"}


def test_filter_panel_columns_no_filtering_when_no_panel_data():
    """Test that global data passes through unchanged."""
    df = pl.DataFrame(
        {
            "time": [1, 2, 3],
            "value": [10.0, 20.0, 30.0],
            "feature": [100.0, 200.0, 300.0],
        }
    )

    # No panel data (local_group_names=None or empty)
    result = filter_panel_columns(
        df, cross_learning_group="value", local_group_names=None, include_global=True
    )

    assert result.equals(df)
