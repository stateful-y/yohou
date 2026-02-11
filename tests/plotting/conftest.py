"""Shared fixtures for plotting tests."""

import numpy as np
import polars as pl
import pytest


@pytest.fixture
def panel_daily_df():
    """Create a daily panel DataFrame with two groups (y__a, y__b).

    366 rows (2020 full year), daily frequency.
    Both panel columns have seasonal-like patterns.
    """
    dates = pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 12, 31), "1d", eager=True)
    n = len(dates)
    return pl.DataFrame({
        "time": dates,
        "y__a": [100 + i % 30 + (i % 7) * 3 for i in range(n)],
        "y__b": [200 + i % 20 + (i % 5) * 5 for i in range(n)],
    })


@pytest.fixture
def panel_monthly_df():
    """Create a monthly panel DataFrame with two groups (y__a, y__b).

    12 rows, monthly frequency.
    """
    return pl.DataFrame({
        "time": pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 12, 31), "1mo", eager=True),
        "y__a": [100, 120, 115, 130, 140, 135, 150, 160, 155, 170, 180, 175],
        "y__b": [200, 210, 205, 220, 230, 225, 240, 250, 245, 260, 270, 265],
    })


@pytest.fixture
def panel_monthly_3groups():
    """Create a monthly panel with three groups (sales__a, sales__b, sales__c).

    12 rows, monthly frequency. Uses 'sales' as the group prefix.
    """
    return pl.DataFrame({
        "time": pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 12, 31), "1mo", eager=True),
        "sales__a": [10, 12, 11, 13, 14, 13, 15, 16, 15, 17, 18, 17],
        "sales__b": [20, 22, 21, 23, 24, 23, 25, 26, 25, 27, 28, 27],
        "sales__c": [30, 32, 31, 33, 34, 33, 35, 36, 35, 37, 38, 37],
    })


@pytest.fixture
def panel_with_nulls():
    """Create a daily panel DataFrame with missing values.

    366 rows, daily frequency. Both columns have periodic nulls.
    """
    dates = pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 12, 31), "1d", eager=True)
    n = len(dates)
    return pl.DataFrame({
        "time": dates,
        "y__a": [100 + i if i % 5 != 0 else None for i in range(n)],
        "y__b": [200 + i if i % 7 != 0 else None for i in range(n)],
    })


@pytest.fixture
def panel_two_groups_df():
    """Create a panel DataFrame with two group prefixes (temp and wind).

    Used for testing panel_group_names filtering.
    """
    dates = pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 3, 31), "1d", eager=True)
    n = len(dates)
    return pl.DataFrame({
        "time": dates,
        "temp__city_a": [20.0 + i * 0.1 for i in range(n)],
        "temp__city_b": [15.0 + i * 0.1 for i in range(n)],
        "wind__city_a": [5.0 + np.sin(i * 0.3) for i in range(n)],
        "wind__city_b": [3.0 + np.cos(i * 0.3) for i in range(n)],
    })
