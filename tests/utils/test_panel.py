"""Tests for panel data utilities in yohou/utils/panel.py."""

from datetime import datetime, timedelta

import polars as pl

from yohou.utils.panel import filter_panel_columns, inspect_locality


def test_inspect_locality_global_data_single_column():
    """Test inspect_locality with global data (single non-struct column)."""
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


def test_inspect_locality_panel_data_single_struct():
    """Test inspect_locality with panel data (single struct column)."""
    df = pl.DataFrame(
        {
            "time": [1, 2, 3],
            "sales": pl.Series(
                [
                    {"store_1": 100, "store_2": 150},
                    {"store_1": 110, "store_2": 160},
                    {"store_1": 120, "store_2": 170},
                ]
            ),
        }
    )

    global_names, local_groups = inspect_locality(df)

    assert global_names == []
    assert local_groups == {"sales": ["store_1", "store_2"]}


def test_inspect_locality_panel_data_multiple_structs():
    """Test inspect_locality with multiple struct columns."""
    df = pl.DataFrame(
        {
            "time": [1, 2, 3],
            "sales": pl.Series(
                [
                    {"store_1": 100, "store_2": 150},
                    {"store_1": 110, "store_2": 160},
                    {"store_1": 120, "store_2": 170},
                ]
            ),
            "inventory": pl.Series(
                [
                    {"store_1": 50, "store_2": 75},
                    {"store_1": 55, "store_2": 80},
                    {"store_1": 60, "store_2": 85},
                ]
            ),
        }
    )

    global_names, local_groups = inspect_locality(df)

    assert global_names == []
    assert local_groups == {
        "sales": ["store_1", "store_2"],
        "inventory": ["store_1", "store_2"],
    }


def test_inspect_locality_mixed_global_and_panel_data():
    """Test inspect_locality with mix of global columns and struct columns."""
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

    global_names, local_groups = inspect_locality(df)

    assert global_names == ["global_feature"]
    assert local_groups == {"sales": ["store_1", "store_2"]}


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
        df, cross_learning_group="sales", local_group_names=["sales"], include_global=False
    )

    assert result.columns == ["time", "sales"]
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


def test_filter_panel_columns_multiple_struct_columns():
    """Test filtering when multiple struct columns exist."""
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
            "inventory": pl.Series(
                [
                    {"store_1": 50, "store_2": 75},
                    {"store_1": 55, "store_2": 80},
                    {"store_1": 60, "store_2": 85},
                ]
            ),
        }
    )

    result = filter_panel_columns(
        df,
        cross_learning_group="sales",
        local_group_names=["sales", "inventory"],
        include_global=False,
    )

    # Should keep only time and sales (not inventory)
    assert result.columns == ["time", "sales"]
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
    schema = {
        "time": pl.Int64,
        "global_feature": pl.Float64,
        "sales": pl.Struct([pl.Field("store_1", pl.Int64), pl.Field("store_2", pl.Int64)]),
    }
    df = pl.DataFrame(schema=schema)

    result = filter_panel_columns(
        df, cross_learning_group="sales", local_group_names=["sales"], include_global=False
    )

    assert result.columns == ["time", "sales"]
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
    # Struct column preserved
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
        df, cross_learning_group="sales", local_group_names=["sales"], include_global=False
    )

    assert result.columns == ["time", "sales"]
    assert result["time"].dtype == pl.Datetime


def test_filter_panel_columns_forecaster_y_filtering_pattern():
    """Test filtering pattern used for _y_observed in forecasters."""
    # Simulate _y_observed with multiple struct columns
    df = pl.DataFrame(
        {
            "time": [1, 2, 3],
            "store_sales": pl.Series(
                [
                    {"store_1": 100, "store_2": 150, "store_3": 200},
                    {"store_1": 110, "store_2": 160, "store_3": 210},
                    {"store_1": 120, "store_2": 170, "store_3": 220},
                ]
            ),
            "online_sales": pl.Series(
                [
                    {"web": 50, "app": 30},
                    {"web": 55, "app": 35},
                    {"web": 60, "app": 40},
                ]
            ),
        }
    )

    # Filter for cross-learning on store_sales
    result = filter_panel_columns(
        df,
        cross_learning_group="store_sales",
        local_group_names=["store_sales", "online_sales"],
        include_global=False,
    )

    assert result.columns == ["time", "store_sales"]
    assert "online_sales" not in result.columns


def test_filter_panel_columns_forecaster_x_filtering_pattern():
    """Test filtering pattern used for X_post/X_ante in forecasters."""
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
