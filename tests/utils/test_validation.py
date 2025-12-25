"""Tests for interval validation and manipulation utilities."""

import calendar
import random
from datetime import datetime, timedelta

import polars as pl
import pytest

from yohou.utils.validation import (
    add_interval,
    check_interval_consistency,
    interval_to_timedelta,
    parse_interval,
)


def random_start_date(seed=None):
    """Generate random datetime between 1950 and 2030."""
    if seed is not None:
        random.seed(seed)
    year = random.randint(1950, 2030)
    month = random.randint(1, 12)
    day = random.randint(1, 28)  # Safe day for all months
    hour = random.randint(0, 23)
    minute = random.randint(0, 59)
    return datetime(year, month, day, hour, minute)


@pytest.mark.parametrize(
    "interval_str,n_periods,seed",
    [
        # Fixed intervals - daily
        ("1d", 100, 42),
        ("1d", 365, 123),
        ("7d", 52, 456),
        ("14d", 26, 789),
        # Fixed intervals - sub-daily
        ("1h", 168, 111),
        ("6h", 240, 222),
        ("12h", 120, 333),
        # Variable intervals - monthly
        ("1mo", 120, 666),
        ("2mo", 60, 777),
        ("3mo", 40, 888),
        ("6mo", 20, 999),
        # Variable intervals - yearly
        ("1y", 10, 2222),
    ],
)
def test_check_interval_consistency_parametrized(interval_str, n_periods, seed):
    """Test check_interval_consistency with wide range of intervals and random start dates."""
    start = random_start_date(seed)
    time_series = [add_interval(start, interval_str, i) for i in range(n_periods)]
    df = pl.DataFrame({"time": time_series})

    detected_interval = check_interval_consistency(df)
    assert detected_interval == interval_str, (
        f"Expected interval '{interval_str}', but detected '{detected_interval}' "
        f"(start={start}, n_periods={n_periods}, seed={seed})"
    )


@pytest.mark.parametrize(
    "start_day,interval_str,n_periods,seed",
    [
        # Month-end edge cases - start on day 31
        (31, "1mo", 12, 4444),
        (31, "2mo", 6, 5555),
        (31, "3mo", 4, 6666),
        # Month-end edge cases - start on day 30
        (30, "1mo", 12, 7777),
        (30, "2mo", 6, 8888),
        # Month-end edge cases - start on day 29 (leap year handling)
        (29, "1mo", 24, 9999),
        (29, "2mo", 12, 10000),
    ],
)
def test_check_interval_consistency_month_end(start_day, interval_str, n_periods, seed):
    """Test monthly intervals with month-end edge cases (Jan 31 → Feb 28/29 → Mar 31)."""
    random.seed(seed)
    year = random.randint(2000, 2030)

    # Pick a month that has the required start_day
    if start_day == 31:
        month = random.choice([1, 3, 5, 7, 8, 10, 12])
    elif start_day == 30:
        month = random.choice([1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    elif start_day == 29:
        # For day 29, avoid February in non-leap years
        # Ensure we have a valid date
        if calendar.isleap(year):
            month = random.randint(1, 12)
        else:
            month = random.choice([1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])  # Exclude February
    else:
        month = random.randint(1, 12)

    start = datetime(year, month, start_day)
    time_series = [add_interval(start, interval_str, i) for i in range(n_periods)]
    df = pl.DataFrame({"time": time_series})

    detected_interval = check_interval_consistency(df)
    assert detected_interval == interval_str


@pytest.mark.parametrize(
    "td,expected_str",
    [
        (timedelta(days=1), "1d"),
        (timedelta(days=7), "7d"),
        (timedelta(days=14), "14d"),
        (timedelta(days=30), "30d"),
        (timedelta(hours=1), "1h"),
        (timedelta(hours=6), "6h"),
        (timedelta(days=1, hours=0), "1d"),
        (timedelta(minutes=30), "30min"),
        (timedelta(minutes=1), "1min"),
    ],
)
def test_timedelta_to_string(td, expected_str):
    """Test conversion from timedelta to string format via check_interval_consistency."""
    # Create a simple time series with the given timedelta
    start = datetime(2020, 1, 1)
    time_series = [start + td * i for i in range(10)]
    df = pl.DataFrame({"time": time_series})

    detected = check_interval_consistency(df)
    assert detected == expected_str


@pytest.mark.parametrize(
    "interval_str,expected_multiplier,expected_unit",
    [
        ("1d", 1, "d"),
        ("7d", 7, "d"),
        ("1w", 1, "w"),
        ("2w", 2, "w"),
        ("1h", 1, "h"),
        ("12h", 12, "h"),
        ("1mo", 1, "mo"),
        ("3mo", 3, "mo"),
        ("6mo", 6, "mo"),
        ("1q", 1, "q"),
        ("1y", 1, "y"),
        ("2y", 2, "y"),
    ],
)
def test_parse_interval(interval_str, expected_multiplier, expected_unit):
    """Test parsing interval strings into (multiplier, unit) tuples."""
    multiplier, unit = parse_interval(interval_str)
    assert multiplier == expected_multiplier
    assert unit == expected_unit


@pytest.mark.parametrize(
    "interval_str,expected_td",
    [
        ("1d", timedelta(days=1)),
        ("7d", timedelta(days=7)),
        ("1w", timedelta(weeks=1)),
        ("2w", timedelta(weeks=2)),
        ("1h", timedelta(hours=1)),
        ("12h", timedelta(hours=12)),
        ("30min", timedelta(minutes=30)),
        ("60s", timedelta(seconds=60)),
        ("1mo", None),  # Variable interval
        ("1q", None),  # Variable interval
        ("1y", None),  # Variable interval
    ],
)
def test_interval_to_timedelta(interval_str, expected_td):
    """Test conversion from fixed interval strings back to timedelta."""
    result = interval_to_timedelta(interval_str)
    assert result == expected_td


@pytest.mark.parametrize(
    "start_seed,interval_str,n,expected_months_added",
    [
        (42, "1mo", 1, 1),
        (42, "1mo", 12, 12),
        (123, "2mo", 6, 12),
        (456, "3mo", 4, 12),
        (789, "6mo", 2, 12),
        (111, "1q", 4, 12),
        (222, "1y", 2, 24),
    ],
)
def test_add_interval_monthly(start_seed, interval_str, n, expected_months_added):
    """Test add_interval for monthly patterns with random start dates."""
    start = random_start_date(start_seed)
    result = add_interval(start, interval_str, n)

    months_diff = (result.year - start.year) * 12 + (result.month - start.month)
    assert months_diff == expected_months_added


@pytest.mark.parametrize(
    "start_seed,interval_str,n_periods",
    [
        (333, "1d", 100),
        (444, "7d", 52),
        (555, "1mo", 24),
        (666, "3mo", 12),
        (777, "3mo", 8),
        (888, "1y", 5),
    ],
)
def test_add_interval_roundtrip(start_seed, interval_str, n_periods):
    """Test that repeatedly adding intervals creates consistent time series."""
    start = random_start_date(start_seed)
    time_series = [add_interval(start, interval_str, i) for i in range(n_periods)]
    df = pl.DataFrame({"time": time_series})

    detected = check_interval_consistency(df)
    assert detected == interval_str


def test_add_interval_month_end_rollover():
    """Test that month-end dates roll over correctly."""
    # Jan 31 + 1mo = Feb 29 (2020 is leap year)
    start = datetime(2020, 1, 31)
    result = add_interval(start, "1mo", 1)
    assert result == datetime(2020, 2, 29)

    # Jan 31 + 1mo = Feb 28 (2021 is not leap year)
    start = datetime(2021, 1, 31)
    result = add_interval(start, "1mo", 1)
    assert result == datetime(2021, 2, 28)

    # Jan 31 + 2mo = Mar 31
    start = datetime(2020, 1, 31)
    result = add_interval(start, "2mo", 1)
    assert result == datetime(2020, 3, 31)


def test_add_interval_year_boundary():
    """Test year boundary crossing."""
    start = datetime(2020, 11, 15)
    result = add_interval(start, "1mo", 3)  # Nov + 3mo = Feb next year
    assert result == datetime(2021, 2, 15)

    start = datetime(2020, 12, 31)
    result = add_interval(start, "1mo", 1)
    assert result == datetime(2021, 1, 31)


def test_check_interval_consistency_empty_fails():
    """Test that empty DataFrame raises error."""
    df = pl.DataFrame({"time": []})
    with pytest.raises(ValueError, match="Need at least 2 time points"):
        check_interval_consistency(df)


def test_check_interval_consistency_single_point_fails():
    """Test that single point raises error."""
    df = pl.DataFrame({"time": [datetime(2020, 1, 1)]})
    with pytest.raises(ValueError, match="Need at least 2 time points"):
        check_interval_consistency(df)


def test_check_interval_consistency_truly_inconsistent():
    """Test that truly inconsistent intervals raise error."""
    df = pl.DataFrame(
        {
            "time": [
                datetime(2020, 1, 1),
                datetime(2020, 1, 2),  # 1 day
                datetime(2020, 1, 5),  # 3 days
                datetime(2020, 1, 6),  # 1 day
            ]
        }
    )
    with pytest.raises(ValueError, match="Cannot infer a regular frequency pattern"):
        check_interval_consistency(df)


def test_parse_interval_invalid():
    """Test that invalid interval strings raise error."""
    with pytest.raises(ValueError, match="Invalid interval format"):
        parse_interval("invalid")

    with pytest.raises(ValueError, match="Invalid interval format"):
        parse_interval("1x")


def test_add_interval_unsupported_unit():
    """Test that unsupported units raise error."""
    with pytest.raises(ValueError, match="Invalid interval format"):
        add_interval(datetime(2020, 1, 1), "1x", 1)


def test_air_passengers_monthly_data():
    """Integration test with air passengers-like monthly data."""
    # Create 12 years of monthly data starting Jan 1949
    dates = pl.date_range(datetime(1949, 1, 1), datetime(1960, 12, 1), interval="1mo", eager=True)

    df = pl.DataFrame({"time": dates, "passengers": range(len(dates))})

    # Should detect as monthly
    interval = check_interval_consistency(df)
    assert interval == "1mo"

    # Verify length (12 years * 12 months = 144)
    assert len(df) == 144


def test_quarterly_business_data():
    """Test quarterly data detection."""
    # Create quarterly dates: Q1, Q2, Q3, Q4
    dates = [datetime(2020, month, 1) for month in [1, 4, 7, 10]]
    dates += [datetime(2021, month, 1) for month in [1, 4, 7, 10]]

    df = pl.DataFrame({"time": dates, "value": range(len(dates))})

    interval = check_interval_consistency(df)
    assert interval == "3mo"


def test_bimonthly_data():
    """Test detection of 2-month intervals."""
    start = datetime(2020, 1, 15)
    time_series = [add_interval(start, "2mo", i) for i in range(12)]
    df = pl.DataFrame({"time": time_series, "value": range(len(time_series))})

    interval = check_interval_consistency(df)
    assert interval == "2mo"


def test_semiannual_data():
    """Test detection of 6-month intervals."""
    start = datetime(2020, 1, 1)
    time_series = [add_interval(start, "6mo", i) for i in range(8)]
    df = pl.DataFrame({"time": time_series, "value": range(len(time_series))})

    interval = check_interval_consistency(df)
    assert interval == "6mo"


def test_leap_year_february():
    """Test that leap year February is handled correctly."""
    # Leap year (2020)
    start = datetime(2020, 1, 29)
    result = add_interval(start, "1mo", 1)
    assert result == datetime(2020, 2, 29)

    # Non-leap year (2021)
    start = datetime(2021, 1, 29)
    result = add_interval(start, "1mo", 1)
    assert result == datetime(2021, 2, 28)


def test_multi_year_interval():
    """Test multi-year intervals."""
    start = datetime(2000, 3, 15)
    result = add_interval(start, "2y", 1)
    assert result == datetime(2002, 3, 15)

    # Across century boundary
    start = datetime(1998, 6, 30)
    result = add_interval(start, "5y", 1)
    assert result == datetime(2003, 6, 30)
