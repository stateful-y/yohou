"""Tests for interval validation and manipulation utilities."""

import calendar
import random
from datetime import datetime, timedelta

import polars as pl
import pytest
from polars.exceptions import ColumnNotFoundError

from yohou.utils.validation import (
    _check_day_of_month_consistency,
    _infer_monthly_freq,
    _normalize_interval,
    _timedelta_to_string,
    add_interval,
    check_continuity,
    check_forecasting_horizon_positive,
    check_groups,
    check_groups_exist,
    check_inputs,
    check_interval_consistency,
    check_schema,
    check_scorer_column_selection,
    check_time_column,
    check_X_actual_required,
    interval_to_timedelta,
    parse_interval,
    validate_column_names,
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


class TestCheckIntervalConsistency:
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
    def test_check_interval_consistency_parametrized(self, interval_str, n_periods, seed):
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
    def test_check_interval_consistency_month_end(self, start_day, interval_str, n_periods, seed):
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
            month = (
                random.randint(1, 12)
                if calendar.isleap(year)
                else random.choice([1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])  # Exclude February
            )
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
            (timedelta(minutes=30), "30m"),
            (timedelta(minutes=1), "1m"),
        ],
    )
    def test_timedelta_to_string(self, td, expected_str):
        """Test conversion from timedelta to string format via check_interval_consistency."""
        # Create a simple time series with the given timedelta
        start = datetime(2020, 1, 1)
        time_series = [start + td * i for i in range(10)]
        df = pl.DataFrame({"time": time_series})

        detected = check_interval_consistency(df)
        assert detected == expected_str

    def test_check_interval_consistency_empty_fails(self):
        """Test that empty DataFrame raises error."""
        df = pl.DataFrame({"time": []})
        with pytest.raises(ValueError, match="Need at least 2 time points"):
            check_interval_consistency(df)

    def test_check_interval_consistency_single_point_fails(self):
        """Test that single point raises error."""
        df = pl.DataFrame({"time": [datetime(2020, 1, 1)]})
        with pytest.raises(ValueError, match="Need at least 2 time points"):
            check_interval_consistency(df)

    def test_check_interval_consistency_truly_inconsistent(self):
        """Test that truly inconsistent intervals raise error."""
        df = pl.DataFrame({
            "time": [
                datetime(2020, 1, 1),
                datetime(2020, 1, 2),  # 1 day
                datetime(2020, 1, 5),  # 3 days
                datetime(2020, 1, 6),  # 1 day
            ]
        })
        with pytest.raises(ValueError, match="Cannot infer a regular frequency pattern"):
            check_interval_consistency(df)

    def test_air_passengers_monthly_data(self):
        """Integration test with air passengers-like monthly data."""
        # Create 12 years of monthly data starting Jan 1949
        dates = pl.date_range(datetime(1949, 1, 1), datetime(1960, 12, 1), interval="1mo", eager=True)

        df = pl.DataFrame({"time": dates, "passengers": range(len(dates))})

        # Should detect as monthly
        interval = check_interval_consistency(df)
        assert interval == "1mo"

        # Verify length (12 years * 12 months = 144)
        assert len(df) == 144

    def test_quarterly_business_data(self):
        """Test quarterly data detection."""
        # Create quarterly dates: Q1, Q2, Q3, Q4
        dates = [datetime(2020, month, 1) for month in [1, 4, 7, 10]]
        dates += [datetime(2021, month, 1) for month in [1, 4, 7, 10]]

        df = pl.DataFrame({"time": dates, "value": range(len(dates))})

        interval = check_interval_consistency(df)
        assert interval == "3mo"

    def test_bimonthly_data(self):
        """Test detection of 2-month intervals."""
        start = datetime(2020, 1, 15)
        time_series = [add_interval(start, "2mo", i) for i in range(12)]
        df = pl.DataFrame({"time": time_series, "value": range(len(time_series))})

        interval = check_interval_consistency(df)
        assert interval == "2mo"

    def test_semiannual_data(self):
        """Test detection of 6-month intervals."""
        start = datetime(2020, 1, 1)
        time_series = [add_interval(start, "6mo", i) for i in range(8)]
        df = pl.DataFrame({"time": time_series, "value": range(len(time_series))})

        interval = check_interval_consistency(df)
        assert interval == "6mo"


class TestParseInterval:
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
    def test_parse_interval(self, interval_str, expected_multiplier, expected_unit):
        """Test parsing interval strings into (multiplier, unit) tuples."""
        multiplier, unit = parse_interval(interval_str)
        assert multiplier == expected_multiplier
        assert unit == expected_unit

    def test_parse_interval_invalid(self):
        """Test that invalid interval strings raise error."""
        with pytest.raises(ValueError, match="Invalid interval format"):
            parse_interval("invalid")

        with pytest.raises(ValueError, match="Invalid interval format"):
            parse_interval("1x")


class TestIntervalToTimedelta:
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
            ("30m", timedelta(minutes=30)),
            ("60s", timedelta(seconds=60)),
            ("1mo", None),  # Variable interval
            ("1q", None),  # Variable interval
            ("1y", None),  # Variable interval
        ],
    )
    def test_interval_to_timedelta(self, interval_str, expected_td):
        """Test conversion from fixed interval strings back to timedelta."""
        result = interval_to_timedelta(interval_str)
        assert result == expected_td


class TestAddInterval:
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
    def test_add_interval_monthly(self, start_seed, interval_str, n, expected_months_added):
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
    def test_add_interval_roundtrip(self, start_seed, interval_str, n_periods):
        """Test that repeatedly adding intervals creates consistent time series."""
        start = random_start_date(start_seed)
        time_series = [add_interval(start, interval_str, i) for i in range(n_periods)]
        df = pl.DataFrame({"time": time_series})

        detected = check_interval_consistency(df)
        assert detected == interval_str

    def test_add_interval_month_end_rollover(self):
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

    def test_add_interval_year_boundary(self):
        """Test year boundary crossing."""
        start = datetime(2020, 11, 15)
        result = add_interval(start, "1mo", 3)  # Nov + 3mo = Feb next year
        assert result == datetime(2021, 2, 15)

        start = datetime(2020, 12, 31)
        result = add_interval(start, "1mo", 1)
        assert result == datetime(2021, 1, 31)

    def test_add_interval_unsupported_unit(self):
        """Test that unsupported units raise error."""
        with pytest.raises(ValueError, match="Invalid interval format"):
            add_interval(datetime(2020, 1, 1), "1x", 1)

    def test_leap_year_february(self):
        """Test that leap year February is handled correctly."""
        # Leap year (2020)
        start = datetime(2020, 1, 29)
        result = add_interval(start, "1mo", 1)
        assert result == datetime(2020, 2, 29)

        # Non-leap year (2021)
        start = datetime(2021, 1, 29)
        result = add_interval(start, "1mo", 1)
        assert result == datetime(2021, 2, 28)

    def test_multi_year_interval(self):
        """Test multi-year intervals."""
        start = datetime(2000, 3, 15)
        result = add_interval(start, "2y", 1)
        assert result == datetime(2002, 3, 15)

        # Across century boundary
        start = datetime(1998, 6, 30)
        result = add_interval(start, "5y", 1)
        assert result == datetime(2003, 6, 30)


class TestCheckTimeColumn:
    def test_check_time_column_valid(self):
        """Test check_time_column with valid DataFrame."""
        from yohou.utils.validation import check_time_column

        df = pl.DataFrame({
            "time": pl.datetime_range(datetime(2023, 1, 1), datetime(2023, 1, 10), "1d", eager=True),
            "value": range(10),
        })
        # Should not raise
        check_time_column(df)

    def test_check_time_column_missing(self):
        """Test check_time_column raises error when time column is missing."""
        from yohou.utils.validation import check_time_column

        df = pl.DataFrame({"value": [1, 2, 3]})

        with pytest.raises(ValueError, match="DataFrame must contain a 'time' column"):
            check_time_column(df)

    def test_check_time_column_wrong_type(self):
        """Test check_time_column raises error when time column is not Datetime."""
        from yohou.utils.validation import check_time_column

        # String type
        df = pl.DataFrame({"time": ["2023-01-01", "2023-01-02"], "value": [1, 2]})

        with pytest.raises(ValueError, match="'time' column in DataFrame must have dtype pl.Datetime or pl.Date"):
            check_time_column(df)

    def test_check_time_column_int_type(self):
        """Test check_time_column raises error when time column is Int64."""
        from yohou.utils.validation import check_time_column

        df = pl.DataFrame({"time": [1, 2, 3], "value": [10, 20, 30]})

        with pytest.raises(ValueError, match="'time' column in DataFrame must have dtype pl.Datetime or pl.Date"):
            check_time_column(df)

    def test_check_time_column_custom_name(self):
        """Test check_time_column with custom DataFrame name in error message."""
        from yohou.utils.validation import check_time_column

        df = pl.DataFrame({"value": [1, 2, 3]})

        with pytest.raises(ValueError, match="X must contain a 'time' column"):
            check_time_column(df, df_name="X")


class TestCheckSchema:
    def test_check_schema_non_panel_correct_order(self):
        """Test check_schema with non-panel data and correct column order."""
        df = pl.DataFrame({
            "time": [1, 2, 3],
            "value": [10, 20, 30],
        })
        expected_schema = {"value": pl.Int64}

        result = check_schema(df, expected_schema)

        assert result.columns == ["time", "value"]
        assert result.equals(df)

    def test_check_schema_non_panel_wrong_order(self):
        """Test check_schema reorders columns when they're in wrong order."""
        df = pl.DataFrame({
            "value": [10, 20, 30],
            "time": [1, 2, 3],
        })
        expected_schema = {"value": pl.Int64}

        result = check_schema(df, expected_schema)

        assert result.columns == ["time", "value"]
        assert result["value"].to_list() == [10, 20, 30]
        assert result["time"].to_list() == [1, 2, 3]

    def test_check_schema_non_panel_multiple_columns(self):
        """Test check_schema with multiple columns in non-panel data."""
        df = pl.DataFrame({
            "col_b": [1.5, 2.5, 3.5],
            "time": [1, 2, 3],
            "col_a": [10, 20, 30],
        })
        expected_schema = {"col_a": pl.Int64, "col_b": pl.Float64}

        result = check_schema(df, expected_schema)

        # Should preserve schema dict order
        assert result.columns == ["time", "col_a", "col_b"]

    def test_check_schema_non_panel_schema_mismatch_dtype(self):
        """Test check_schema raises error when dtypes don't match."""
        df = pl.DataFrame({
            "time": [1, 2, 3],
            "value": [10.0, 20.0, 30.0],  # Float64 instead of Int64
        })
        expected_schema = {"value": pl.Int64}

        with pytest.raises(ValueError, match="Schema mismatch"):
            check_schema(df, expected_schema)

    def test_check_schema_non_panel_missing_column(self):
        """Test check_schema raises error when column is missing."""
        df = pl.DataFrame({
            "time": [1, 2, 3],
            "value": [10, 20, 30],
        })
        expected_schema = {"value": pl.Int64, "extra_col": pl.Float64}

        with pytest.raises(ColumnNotFoundError):
            check_schema(df, expected_schema)

    def test_check_schema_non_panel_missing_time(self):
        """Test check_schema raises error when time column is missing."""
        df = pl.DataFrame({
            "value": [10, 20, 30],
        })
        expected_schema = {"value": pl.Int64}

        with pytest.raises(ColumnNotFoundError):
            check_schema(df, expected_schema)

    def test_check_schema_panel_correct_order(self):
        """Test check_schema with panel data and correct column order."""
        df = pl.DataFrame({
            "time": [1, 2, 3],
            "sales__store_1": [100, 110, 120],
            "sales__store_2": [150, 160, 170],
        })
        expected_schema = {"store_1": pl.Int64, "store_2": pl.Int64}

        result = check_schema(df, expected_schema, groups=["sales"])

        assert result.columns == ["time", "sales__store_1", "sales__store_2"]
        assert result.equals(df)

    def test_check_schema_panel_wrong_order(self):
        """Test check_schema reorders panel columns when they're in wrong order."""
        df = pl.DataFrame({
            "sales__store_2": [150, 160, 170],
            "time": [1, 2, 3],
            "sales__store_1": [100, 110, 120],
        })
        expected_schema = {"store_1": pl.Int64, "store_2": pl.Int64}

        result = check_schema(df, expected_schema, groups=["sales"])

        # Should reorder according to schema dict order
        assert result.columns == ["time", "sales__store_1", "sales__store_2"]
        assert result["sales__store_1"].to_list() == [100, 110, 120]
        assert result["sales__store_2"].to_list() == [150, 160, 170]

    def test_check_schema_panel_multiple_groups(self):
        """Test check_schema with multiple panel groups."""
        df = pl.DataFrame({
            "time": [1, 2, 3],
            "sales__store_1": [100, 110, 120],
            "sales__store_2": [150, 160, 170],
            "inventory__store_1": [50, 55, 60],
            "inventory__store_2": [75, 80, 85],
        })
        expected_schema = {"store_1": pl.Int64, "store_2": pl.Int64}

        result = check_schema(df, expected_schema, groups=["sales", "inventory"])

        # Should construct prefixes for all groups in order
        assert result.columns == [
            "time",
            "sales__store_1",
            "sales__store_2",
            "inventory__store_1",
            "inventory__store_2",
        ]

    def test_check_schema_panel_schema_mismatch(self):
        """Test check_schema raises error when panel data has wrong dtype."""
        df = pl.DataFrame({
            "time": [1, 2, 3],
            "sales__store_1": [100.0, 110.0, 120.0],  # Float64 instead of Int64
            "sales__store_2": [150, 160, 170],
        })
        expected_schema = {"store_1": pl.Int64, "store_2": pl.Int64}

        with pytest.raises(ValueError, match="Schema mismatch"):
            check_schema(df, expected_schema, groups=["sales"])

    def test_check_schema_panel_missing_prefixed_column(self):
        """Test check_schema raises error when prefixed column is missing."""
        df = pl.DataFrame({
            "time": [1, 2, 3],
            "sales__store_1": [100, 110, 120],
            # Missing sales__store_2
        })
        expected_schema = {"store_1": pl.Int64, "store_2": pl.Int64}

        with pytest.raises(ColumnNotFoundError):
            check_schema(df, expected_schema, groups=["sales"])

    def test_check_schema_panel_empty_group_names(self):
        """Test check_schema with empty groups list behaves like non-panel."""
        df = pl.DataFrame({
            "time": [1, 2, 3],
            "value": [10, 20, 30],
        })
        expected_schema = {"value": pl.Int64}

        # Empty list should behave like None (non-panel)
        result = check_schema(df, expected_schema, groups=[])

        assert result.columns == ["time"]  # Only time because empty group list means no columns

    def test_check_schema_panel_selects_only_specified_columns(self):
        """Test check_schema selects only columns matching the schema with prefixes."""
        df = pl.DataFrame({
            "time": [1, 2, 3],
            "sales__store_1": [100, 110, 120],
            "sales__store_2": [150, 160, 170],
            "extra_column": [999, 888, 777],  # Should be excluded
        })
        expected_schema = {"store_1": pl.Int64, "store_2": pl.Int64}

        result = check_schema(df, expected_schema, groups=["sales"])

        # Should only include time and the two prefixed columns from schema
        assert result.columns == ["time", "sales__store_1", "sales__store_2"]
        assert "extra_column" not in result.columns

    def test_check_schema_non_panel_selects_only_specified_columns(self):
        """Test check_schema selects only columns in the schema (non-panel)."""
        df = pl.DataFrame({
            "time": [1, 2, 3],
            "value": [10, 20, 30],
            "extra_column": [999, 888, 777],  # Should be excluded
        })
        expected_schema = {"value": pl.Int64}

        result = check_schema(df, expected_schema)

        # Should only include time and value
        assert result.columns == ["time", "value"]
        assert "extra_column" not in result.columns

    def test_check_schema_preserves_data(self):
        """Test that check_schema preserves the actual data values."""
        df = pl.DataFrame({
            "col_b": [1.5, 2.5, 3.5],
            "time": [10, 20, 30],
            "col_a": [100, 200, 300],
        })
        expected_schema = {"col_a": pl.Int64, "col_b": pl.Float64}

        result = check_schema(df, expected_schema)

        # Verify data is preserved
        assert result["time"].to_list() == [10, 20, 30]
        assert result["col_a"].to_list() == [100, 200, 300]
        assert result["col_b"].to_list() == [1.5, 2.5, 3.5]

    def test_check_schema_panel_preserves_data(self):
        """Test that check_schema preserves data values in panel data."""
        df = pl.DataFrame({
            "sales__store_2": [250, 260, 270],
            "time": [10, 20, 30],
            "sales__store_1": [100, 110, 120],
        })
        expected_schema = {"store_1": pl.Int64, "store_2": pl.Int64}

        result = check_schema(df, expected_schema, groups=["sales"])

        # Verify data is preserved after reordering
        assert result["time"].to_list() == [10, 20, 30]
        assert result["sales__store_1"].to_list() == [100, 110, 120]
        assert result["sales__store_2"].to_list() == [250, 260, 270]


class TestCheckPanelGroupNames:
    def test_check_groups_global_data_no_request(self):
        """Test check_groups with global data and no requested groups."""
        result = check_groups(fitted_panel_groups=None, requested_panel_groups=None)

        assert result is None

    def test_check_groups_global_data_with_request(self):
        """Test check_groups raises error when requesting groups on global data."""
        with pytest.raises(
            ValueError,
            match="The forecaster was fitted on global data, but `groups` were provided",
        ):
            check_groups(fitted_panel_groups=None, requested_panel_groups=["sales"])

    def test_check_groups_panel_data_no_request(self):
        """Test check_groups returns all fitted groups when none requested."""
        fitted = ["sales", "inventory"]
        result = check_groups(fitted_panel_groups=fitted, requested_panel_groups=None)

        assert result == fitted
        assert result is fitted  # Should return the same object

    def test_check_groups_panel_data_single_request(self):
        """Test check_groups validates single requested group."""
        fitted = ["sales", "inventory", "revenue"]
        requested = ["sales"]

        result = check_groups(fitted_panel_groups=fitted, requested_panel_groups=requested)

        assert result == requested

    def test_check_groups_panel_data_multiple_requests(self):
        """Test check_groups validates multiple requested groups."""
        fitted = ["sales", "inventory", "revenue", "profit"]
        requested = ["sales", "revenue"]

        result = check_groups(fitted_panel_groups=fitted, requested_panel_groups=requested)

        assert result == requested

    def test_check_groups_panel_data_all_requests(self):
        """Test check_groups when requesting all fitted groups explicitly."""
        fitted = ["sales", "inventory"]
        requested = ["sales", "inventory"]

        result = check_groups(fitted_panel_groups=fitted, requested_panel_groups=requested)

        assert result == requested

    def test_check_groups_missing_single_group(self):
        """Test check_groups raises error for single missing group."""
        fitted = ["sales", "inventory"]
        requested = ["revenue"]

        with pytest.raises(
            ValueError,
            match=r"Panel group\(s\) \['revenue'\] not found in fitted forecaster\. "
            r"Available groups: \['inventory', 'sales'\]\.",
        ):
            check_groups(fitted_panel_groups=fitted, requested_panel_groups=requested)

    def test_check_groups_missing_multiple_groups(self):
        """Test check_groups raises error for multiple missing groups."""
        fitted = ["sales", "inventory"]
        requested = ["revenue", "profit", "margin"]

        with pytest.raises(
            ValueError,
            match=r"Panel group\(s\) \['margin', 'profit', 'revenue'\] not found in fitted "
            r"forecaster\. Available groups: \['inventory', 'sales'\]\.",
        ):
            check_groups(fitted_panel_groups=fitted, requested_panel_groups=requested)

    def test_check_groups_partial_missing(self):
        """Test check_groups raises error when some groups are valid but others missing."""
        fitted = ["sales", "inventory", "revenue"]
        requested = ["sales", "profit", "inventory"]  # profit is missing

        with pytest.raises(
            ValueError,
            match=r"Panel group\(s\) \['profit'\] not found in fitted forecaster\. "
            r"Available groups: \['inventory', 'revenue', 'sales'\]\.",
        ):
            check_groups(fitted_panel_groups=fitted, requested_panel_groups=requested)

    def test_check_groups_empty_fitted_list(self):
        """Test check_groups with empty fitted groups list."""
        fitted = []
        requested = None

        result = check_groups(fitted_panel_groups=fitted, requested_panel_groups=requested)

        assert result == []

    def test_check_groups_empty_request_list(self):
        """Test check_groups with empty requested groups list."""
        fitted = ["sales", "inventory"]
        requested = []

        result = check_groups(fitted_panel_groups=fitted, requested_panel_groups=requested)

        # Empty request is valid - returns empty list
        assert result == []

    def test_check_groups_order_preserved(self):
        """Test check_groups preserves order of requested groups."""
        fitted = ["a", "b", "c", "d"]
        requested = ["d", "a", "c"]  # Non-alphabetical order

        result = check_groups(fitted_panel_groups=fitted, requested_panel_groups=requested)

        # Should preserve the order in requested_panel_groups
        assert result == ["d", "a", "c"]

    def test_check_groups_duplicate_requests(self):
        """Test check_groups with duplicate requested groups."""
        fitted = ["sales", "inventory"]
        requested = ["sales", "sales", "inventory"]

        result = check_groups(fitted_panel_groups=fitted, requested_panel_groups=requested)

        # Should preserve duplicates as provided (validation only checks existence)
        assert result == ["sales", "sales", "inventory"]

    def test_check_groups_case_sensitive(self):
        """Test check_groups is case-sensitive."""
        fitted = ["sales", "inventory"]
        requested = ["Sales"]  # Different case

        with pytest.raises(
            ValueError,
            match=r"Panel group\(s\) \['Sales'\] not found in fitted forecaster\. "
            r"Available groups: \['inventory', 'sales'\]\.",
        ):
            check_groups(fitted_panel_groups=fitted, requested_panel_groups=requested)

    def test_check_groups_single_fitted_group(self):
        """Test check_groups with only one fitted group."""
        fitted = ["sales"]
        requested = ["sales"]

        result = check_groups(fitted_panel_groups=fitted, requested_panel_groups=requested)

        assert result == ["sales"]

    def test_check_groups_single_fitted_group_wrong_request(self):
        """Test check_groups with one fitted group but different request."""
        fitted = ["sales"]
        requested = ["inventory"]

        with pytest.raises(
            ValueError,
            match=r"Panel group\(s\) \['inventory'\] not found in fitted forecaster\. "
            r"Available groups: \['sales'\]\.",
        ):
            check_groups(fitted_panel_groups=fitted, requested_panel_groups=requested)


class TestCheckPanelInternalConsistency:
    def test_check_panel_internal_consistency_no_panel(self):
        """Test check_panel_internal_consistency with global data (no panel)."""
        from yohou.utils.validation import check_panel_internal_consistency

        time = pl.datetime_range(
            start=datetime(2020, 1, 1),
            end=datetime(2020, 1, 10),
            interval="1d",
            eager=True,
        )
        df = pl.DataFrame({"time": time, "value": range(10)})

        # Should not raise (no panel data)
        check_panel_internal_consistency(df, "y")

    def test_check_panel_internal_consistency_valid(self):
        """Test check_panel_internal_consistency with valid panel data."""
        from yohou.utils.validation import check_panel_internal_consistency

        time = pl.datetime_range(
            start=datetime(2020, 1, 1),
            end=datetime(2020, 1, 10),
            interval="1d",
            eager=True,
        )
        df = pl.DataFrame({
            "time": time,
            "sales__store_1": range(10),
            "sales__store_2": range(10, 20),
        })

        # Should not raise (consistent structure)
        check_panel_internal_consistency(df, "y")

    def test_check_panel_internal_consistency_invalid(self):
        """Test check_panel_internal_consistency with mismatched columns."""
        from yohou.utils.validation import check_panel_internal_consistency

        time = pl.datetime_range(
            start=datetime(2020, 1, 1),
            end=datetime(2020, 1, 10),
            interval="1d",
            eager=True,
        )
        df = pl.DataFrame({
            "time": time,
            "sales__store_1": range(10),
            "sales__store_2": range(10, 20),
            "revenue__store_1": range(20, 30),
            # Missing revenue__store_2 - inconsistent!
        })

        with pytest.raises(ValueError, match="Panel structure mismatch in `y`"):
            check_panel_internal_consistency(df, "y")

    def test_check_panel_internal_consistency_multiple_groups(self):
        """Test check_panel_internal_consistency with multiple valid groups."""
        from yohou.utils.validation import check_panel_internal_consistency

        time = pl.datetime_range(
            start=datetime(2020, 1, 1),
            end=datetime(2020, 1, 10),
            interval="1d",
            eager=True,
        )
        df = pl.DataFrame({
            "time": time,
            "sales__store_1": range(10),
            "sales__store_2": range(10, 20),
            "revenue__store_1": range(20, 30),
            "revenue__store_2": range(30, 40),
        })

        # Should not raise (both groups have same local columns)
        check_panel_internal_consistency(df, "y")

    def test_check_panel_internal_consistency_custom_name(self):
        """Test check_panel_internal_consistency with custom DataFrame name."""
        from yohou.utils.validation import check_panel_internal_consistency

        time = pl.datetime_range(
            start=datetime(2020, 1, 1),
            end=datetime(2020, 1, 10),
            interval="1d",
            eager=True,
        )
        df = pl.DataFrame({
            "time": time,
            "sales__store_1": range(10),
            "revenue__store_2": range(10, 20),
        })

        with pytest.raises(ValueError, match="Panel structure mismatch"):
            check_panel_internal_consistency(df, "X_pred")


class TestCheckPanelGroupsMatch:
    def test_check_panel_groups_match_both_none(self):
        """Test check_panel_groups_match when both are None."""
        from yohou.utils.validation import check_panel_groups_match

        # Should not raise
        check_panel_groups_match(None, None)

    def test_check_panel_groups_match_y_none(self):
        """Test check_panel_groups_match when y is None."""
        from yohou.utils.validation import check_panel_groups_match

        time = pl.datetime_range(
            start=datetime(2020, 1, 1),
            end=datetime(2020, 1, 10),
            interval="1d",
            eager=True,
        )
        X = pl.DataFrame({"time": time, "feature": range(10)})

        # Should not raise (can't check if one is None)
        check_panel_groups_match(None, X)

    def test_check_panel_groups_match_X_none(self):
        """Test check_panel_groups_match when X is None."""
        from yohou.utils.validation import check_panel_groups_match

        time = pl.datetime_range(
            start=datetime(2020, 1, 1),
            end=datetime(2020, 1, 10),
            interval="1d",
            eager=True,
        )
        y = pl.DataFrame({"time": time, "target": range(10)})

        # Should not raise (can't check if one is None)
        check_panel_groups_match(y, None)

    def test_check_panel_groups_match_both_global(self):
        """Test check_panel_groups_match with both global data."""
        from yohou.utils.validation import check_panel_groups_match

        time = pl.datetime_range(
            start=datetime(2020, 1, 1),
            end=datetime(2020, 1, 10),
            interval="1d",
            eager=True,
        )
        y = pl.DataFrame({"time": time, "target": range(10)})
        X = pl.DataFrame({"time": time, "feature": range(10, 20)})

        # Should not raise (both are global)
        check_panel_groups_match(y, X)

    def test_check_panel_groups_match_both_panel_same_groups(self):
        """Test check_panel_groups_match with matching panel groups."""
        from yohou.utils.validation import check_panel_groups_match

        time = pl.datetime_range(
            start=datetime(2020, 1, 1),
            end=datetime(2020, 1, 10),
            interval="1d",
            eager=True,
        )
        y = pl.DataFrame({
            "time": time,
            "sales__store_1": range(10),
            "sales__store_2": range(10, 20),
        })
        X = pl.DataFrame({
            "time": time,
            "sales__store_1": range(20, 30),
            "sales__store_2": range(30, 40),
        })

        # Should not raise (both have "sales" group)
        check_panel_groups_match(y, X)

    def test_check_panel_groups_match_different_groups(self):
        """Test check_panel_groups_match with different panel groups."""
        from yohou.utils.validation import check_panel_groups_match

        time = pl.datetime_range(
            start=datetime(2020, 1, 1),
            end=datetime(2020, 1, 10),
            interval="1d",
            eager=True,
        )
        y = pl.DataFrame({
            "time": time,
            "sales__store_1": range(10),
            "sales__store_2": range(10, 20),
        })
        X = pl.DataFrame({
            "time": time,
            "temp__sensor_1": range(20, 30),
            "temp__sensor_2": range(30, 40),
        })

        with pytest.raises(ValueError, match="Panel groups mismatch between `y` and `X_actual`"):
            check_panel_groups_match(y, X)

    def test_check_panel_groups_match_y_panel_X_global(self):
        """Test check_panel_groups_match when y is panel but X is global."""
        from yohou.utils.validation import check_panel_groups_match

        time = pl.datetime_range(
            start=datetime(2020, 1, 1),
            end=datetime(2020, 1, 10),
            interval="1d",
            eager=True,
        )
        y = pl.DataFrame({
            "time": time,
            "sales__store_1": range(10),
            "sales__store_2": range(10, 20),
        })
        X = pl.DataFrame({"time": time, "feature": range(20, 30)})

        # Global-only X is valid with panel y (broadcast to all groups)
        check_panel_groups_match(y, X)

    def test_check_panel_groups_match_y_global_X_panel(self):
        """Test check_panel_groups_match when y is global but X is panel."""
        from yohou.utils.validation import check_panel_groups_match

        time = pl.datetime_range(
            start=datetime(2020, 1, 1),
            end=datetime(2020, 1, 10),
            interval="1d",
            eager=True,
        )
        y = pl.DataFrame({"time": time, "target": range(10)})
        X = pl.DataFrame({
            "time": time,
            "sales__store_1": range(20, 30),
            "sales__store_2": range(30, 40),
        })

        with pytest.raises(ValueError, match="Panel groups mismatch between `y` and `X_actual`"):
            check_panel_groups_match(y, X)

    def test_check_panel_groups_match_multiple_groups(self):
        """Test check_panel_groups_match with multiple matching groups."""
        from yohou.utils.validation import check_panel_groups_match

        time = pl.datetime_range(
            start=datetime(2020, 1, 1),
            end=datetime(2020, 1, 10),
            interval="1d",
            eager=True,
        )
        y = pl.DataFrame({
            "time": time,
            "sales__store_1": range(10),
            "sales__store_2": range(10, 20),
            "revenue__store_1": range(20, 30),
            "revenue__store_2": range(30, 40),
        })
        X = pl.DataFrame({
            "time": time,
            "sales__store_1": range(40, 50),
            "sales__store_2": range(50, 60),
            "revenue__store_1": range(60, 70),
            "revenue__store_2": range(70, 80),
        })

        # Should not raise (both have "sales" and "revenue" groups)
        check_panel_groups_match(y, X)


class TestCheckScorerColumnSelection:
    def test_check_scorer_column_selection_no_filtering(self):
        """Test that no filtering is applied when scorer has no specifications."""
        from yohou.metrics import MeanAbsoluteError

        scorer = MeanAbsoluteError()

        times = pl.datetime_range(datetime(2020, 1, 1), datetime(2020, 1, 10), "1d", eager=True)
        y_true = pl.DataFrame({"time": times, "value": range(10)})
        y_pred = pl.DataFrame({"time": times, "value": range(10, 20)})

        y_true_out, y_pred_out = check_scorer_column_selection(
            scorer=scorer,
            y_true=y_true,
            y_pred=y_pred,
            pred_type="point",
            coverage_rates=None,
            interval_pattern=None,
        )

        # Should return unchanged DataFrames
        assert y_true_out.equals(y_true)
        assert y_pred_out.equals(y_pred)

    def test_check_scorer_column_selection_panel_groups_point(self):
        """Test panel group filtering for point forecasts."""
        from yohou.metrics import MeanAbsoluteError

        times = pl.datetime_range(datetime(2020, 1, 1), datetime(2020, 1, 5), "1d", eager=True)
        y_true = pl.DataFrame({
            "time": times,
            "sales__store_1": range(5),
            "sales__store_2": range(5, 10),
            "revenue__store_1": range(10, 15),
            "revenue__store_2": range(15, 20),
        })
        y_pred = pl.DataFrame({
            "time": times,
            "sales__store_1": range(100, 105),
            "sales__store_2": range(105, 110),
            "revenue__store_1": range(110, 115),
            "revenue__store_2": range(115, 120),
        })

        scorer = MeanAbsoluteError(groups=["sales"])

        y_true_out, y_pred_out = check_scorer_column_selection(
            scorer=scorer,
            y_true=y_true,
            y_pred=y_pred,
            pred_type="point",
            coverage_rates=None,
            interval_pattern=None,
        )

        # Should only have sales columns
        assert set(y_true_out.columns) == {"time", "sales__store_1", "sales__store_2"}
        assert set(y_pred_out.columns) == {"time", "sales__store_1", "sales__store_2"}

    def test_check_scorer_column_selection_component_names_panel(self):
        """Test component filtering for panel data."""
        from yohou.metrics import MeanAbsoluteError

        times = pl.datetime_range(datetime(2020, 1, 1), datetime(2020, 1, 5), "1d", eager=True)
        y_true = pl.DataFrame({
            "time": times,
            "sales__store_1": range(5),
            "sales__store_2": range(5, 10),
            "revenue__store_1": range(10, 15),
            "revenue__store_2": range(15, 20),
        })
        y_pred = pl.DataFrame({
            "time": times,
            "sales__store_1": range(100, 105),
            "sales__store_2": range(105, 110),
            "revenue__store_1": range(110, 115),
            "revenue__store_2": range(115, 120),
        })

        scorer = MeanAbsoluteError(components=["store_1"])

        y_true_out, y_pred_out = check_scorer_column_selection(
            scorer=scorer,
            y_true=y_true,
            y_pred=y_pred,
            pred_type="point",
            coverage_rates=None,
            interval_pattern=None,
        )

        # Should only have store_1 columns
        assert set(y_true_out.columns) == {"time", "sales__store_1", "revenue__store_1"}
        assert set(y_pred_out.columns) == {"time", "sales__store_1", "revenue__store_1"}

    def test_check_scorer_column_selection_both_filters_panel(self):
        """Test both groups and component_names filters."""
        from yohou.metrics import MeanAbsoluteError

        times = pl.datetime_range(datetime(2020, 1, 1), datetime(2020, 1, 5), "1d", eager=True)
        y_true = pl.DataFrame({
            "time": times,
            "sales__store_1": range(5),
            "sales__store_2": range(5, 10),
            "revenue__store_1": range(10, 15),
            "revenue__store_2": range(15, 20),
        })
        y_pred = pl.DataFrame({
            "time": times,
            "sales__store_1": range(100, 105),
            "sales__store_2": range(105, 110),
            "revenue__store_1": range(110, 115),
            "revenue__store_2": range(115, 120),
        })

        scorer = MeanAbsoluteError(groups=["sales"], components=["store_1"])

        y_true_out, y_pred_out = check_scorer_column_selection(
            scorer=scorer,
            y_true=y_true,
            y_pred=y_pred,
            pred_type="point",
            coverage_rates=None,
            interval_pattern=None,
        )

        # Should only have sales__store_1
        assert set(y_true_out.columns) == {"time", "sales__store_1"}
        assert set(y_pred_out.columns) == {"time", "sales__store_1"}

    def test_check_scorer_column_selection_component_names_global(self):
        """Test component filtering for global (non-panel) data."""
        from yohou.metrics import MeanAbsoluteError

        times = pl.datetime_range(datetime(2020, 1, 1), datetime(2020, 1, 5), "1d", eager=True)
        y_true = pl.DataFrame({"time": times, "sales": range(5), "revenue": range(5, 10)})
        y_pred = pl.DataFrame({"time": times, "sales": range(100, 105), "revenue": range(105, 110)})

        scorer = MeanAbsoluteError(components=["sales"])

        y_true_out, y_pred_out = check_scorer_column_selection(
            scorer=scorer,
            y_true=y_true,
            y_pred=y_pred,
            pred_type="point",
            coverage_rates=None,
            interval_pattern=None,
        )

        # Should only have sales column
        assert set(y_true_out.columns) == {"time", "sales"}
        assert set(y_pred_out.columns) == {"time", "sales"}

    def test_check_scorer_column_selection_interval_with_coverage_rates(self):
        """Test interval forecast filtering by coverage rates."""
        import re

        from yohou.metrics import IntervalScore

        times = pl.datetime_range(datetime(2020, 1, 1), datetime(2020, 1, 5), "1d", eager=True)
        y_true = pl.DataFrame({"time": times, "value": range(5)})
        y_pred = pl.DataFrame({
            "time": times,
            "value_lower_0.9": range(10, 15),
            "value_upper_0.9": range(15, 20),
            "value_lower_0.95": range(8, 13),
            "value_upper_0.95": range(17, 22),
        })

        scorer = IntervalScore(coverage_rates=[0.9])

        interval_pattern = re.compile(r"^(.+)_(lower|upper)_([\d.]+)$")
        y_true_out, y_pred_out = check_scorer_column_selection(
            scorer=scorer,
            y_true=y_true,
            y_pred=y_pred,
            pred_type="interval",
            coverage_rates=[0.9],
            interval_pattern=interval_pattern,
        )

        # Should only have 0.9 coverage rate columns
        assert set(y_true_out.columns) == {"time", "value"}
        assert set(y_pred_out.columns) == {"time", "value_lower_0.9", "value_upper_0.9"}

    def test_check_scorer_column_selection_interval_panel_with_coverage(self):
        """Test interval forecast filtering for panel data with coverage rates."""
        import re

        from yohou.metrics import IntervalScore

        times = pl.datetime_range(datetime(2020, 1, 1), datetime(2020, 1, 3), "1d", eager=True)
        y_true = pl.DataFrame({
            "time": times,
            "sales__store_1": range(3),
            "sales__store_2": range(3, 6),
        })
        y_pred = pl.DataFrame({
            "time": times,
            "sales__store_1_lower_0.9": range(10, 13),
            "sales__store_1_upper_0.9": range(13, 16),
            "sales__store_1_lower_0.95": range(8, 11),
            "sales__store_1_upper_0.95": range(15, 18),
            "sales__store_2_lower_0.9": range(20, 23),
            "sales__store_2_upper_0.9": range(23, 26),
            "sales__store_2_lower_0.95": range(18, 21),
            "sales__store_2_upper_0.95": range(25, 28),
        })

        scorer = IntervalScore(groups=["sales"], components=["store_1"], coverage_rates=[0.95])

        interval_pattern = re.compile(r"^(.+)_(lower|upper)_([\d.]+)$")
        y_true_out, y_pred_out = check_scorer_column_selection(
            scorer=scorer,
            y_true=y_true,
            y_pred=y_pred,
            pred_type="interval",
            coverage_rates=[0.95],
            interval_pattern=interval_pattern,
        )

        # Should only have sales__store_1 with 0.95 coverage
        assert set(y_true_out.columns) == {"time", "sales__store_1"}
        assert set(y_pred_out.columns) == {
            "time",
            "sales__store_1_lower_0.95",
            "sales__store_1_upper_0.95",
        }

    def test_check_scorer_column_selection_invalid_panel_group(self):
        """Test error when requesting non-existent panel group."""
        from yohou.metrics import MeanAbsoluteError

        times = pl.datetime_range(datetime(2020, 1, 1), datetime(2020, 1, 5), "1d", eager=True)
        y_true = pl.DataFrame({"time": times, "sales__store_1": range(5)})
        y_pred = pl.DataFrame({"time": times, "sales__store_1": range(5)})

        scorer = MeanAbsoluteError(groups=["revenue"])

        with pytest.raises(ValueError, match="Invalid groups.*revenue.*not found"):
            check_scorer_column_selection(
                scorer=scorer,
                y_true=y_true,
                y_pred=y_pred,
                pred_type="point",
                coverage_rates=None,
                interval_pattern=None,
            )

    def test_check_scorer_column_selection_invalid_component_global(self):
        """Test error when requesting non-existent component in global data."""
        from yohou.metrics import MeanAbsoluteError

        times = pl.datetime_range(datetime(2020, 1, 1), datetime(2020, 1, 5), "1d", eager=True)
        y_true = pl.DataFrame({"time": times, "sales": range(5)})
        y_pred = pl.DataFrame({"time": times, "sales": range(5)})

        scorer = MeanAbsoluteError(components=["revenue"])

        with pytest.raises(ValueError, match="Invalid components.*revenue.*not found"):
            check_scorer_column_selection(
                scorer=scorer,
                y_true=y_true,
                y_pred=y_pred,
                pred_type="point",
                coverage_rates=None,
                interval_pattern=None,
            )

    def test_check_scorer_interval_component_global(self):
        """Test interval component filtering on global (non-panel) data."""
        import re

        from yohou.metrics import IntervalScore

        times = pl.datetime_range(datetime(2020, 1, 1), datetime(2020, 1, 5), "1d", eager=True)
        y_true = pl.DataFrame({"time": times, "a": range(5), "b": range(5, 10)})
        y_pred = pl.DataFrame({
            "time": times,
            "a_lower_0.9": [0.0] * 5,
            "a_upper_0.9": [10.0] * 5,
            "a_lower_0.95": [0.0] * 5,
            "a_upper_0.95": [10.0] * 5,
            "b_lower_0.9": [0.0] * 5,
            "b_upper_0.9": [20.0] * 5,
        })

        scorer = IntervalScore(components=["a"], coverage_rates=[0.9])
        interval_pattern = re.compile(r"^(.+)_(lower|upper)_([\d.]+)$")
        y_true_out, y_pred_out = check_scorer_column_selection(
            scorer=scorer,
            y_true=y_true,
            y_pred=y_pred,
            pred_type="interval",
            coverage_rates=[0.9],
            interval_pattern=interval_pattern,
        )
        assert set(y_true_out.columns) == {"time", "a"}
        assert set(y_pred_out.columns) == {"time", "a_lower_0.9", "a_upper_0.9"}

    def test_check_scorer_interval_no_component_coverage_filter(self):
        """Test coverage rate filtering without component_names on global data."""
        import re

        from yohou.metrics import IntervalScore

        times = pl.datetime_range(datetime(2020, 1, 1), datetime(2020, 1, 5), "1d", eager=True)
        y_true = pl.DataFrame({"time": times, "value": range(5)})
        y_pred = pl.DataFrame({
            "time": times,
            "value_lower_0.9": [0.0] * 5,
            "value_upper_0.9": [10.0] * 5,
            "value_lower_0.95": [0.0] * 5,
            "value_upper_0.95": [10.0] * 5,
        })
        scorer = IntervalScore(coverage_rates=[0.95])
        interval_pattern = re.compile(r"^(.+)_(lower|upper)_([\d.]+)$")
        y_true_out, y_pred_out = check_scorer_column_selection(
            scorer=scorer,
            y_true=y_true,
            y_pred=y_pred,
            pred_type="interval",
            coverage_rates=[0.95],
            interval_pattern=interval_pattern,
        )
        assert set(y_pred_out.columns) == {"time", "value_lower_0.95", "value_upper_0.95"}

    def test_check_scorer_class_proba_panel(self):
        """Test class_proba column selection on panel data."""
        from yohou.metrics import LogLoss

        times = pl.datetime_range(datetime(2020, 1, 1), datetime(2020, 1, 3), "1d", eager=True)
        y_true = pl.DataFrame({
            "time": times,
            "grpA__weather": ["sunny", "rainy", "cloudy"],
            "grpB__weather": ["cloudy", "sunny", "rainy"],
        })
        y_pred = pl.DataFrame({
            "time": times,
            "grpA__weather_proba_sunny": [0.7, 0.1, 0.2],
            "grpA__weather_proba_rainy": [0.2, 0.8, 0.1],
            "grpA__weather_proba_cloudy": [0.1, 0.1, 0.7],
            "grpB__weather_proba_sunny": [0.2, 0.7, 0.1],
            "grpB__weather_proba_rainy": [0.1, 0.2, 0.8],
            "grpB__weather_proba_cloudy": [0.7, 0.1, 0.1],
        })
        scorer = LogLoss(groups=["grpA"])
        y_true_out, y_pred_out = check_scorer_column_selection(
            scorer=scorer,
            y_true=y_true,
            y_pred=y_pred,
            pred_type="class_proba",
            coverage_rates=None,
            interval_pattern=None,
        )
        assert "grpA__weather" in y_true_out.columns
        assert "grpB__weather" not in y_true_out.columns
        assert any("grpA__weather_proba_" in c for c in y_pred_out.columns)
        assert not any("grpB__weather_proba_" in c for c in y_pred_out.columns)

    def test_check_scorer_class_proba_component_global(self):
        """Test class_proba column selection with component_names on global data."""
        from yohou.metrics import LogLoss

        times = pl.datetime_range(datetime(2020, 1, 1), datetime(2020, 1, 3), "1d", eager=True)
        y_true = pl.DataFrame({
            "time": times,
            "weather": ["sunny", "rainy", "cloudy"],
            "mood": ["happy", "sad", "happy"],
        })
        y_pred = pl.DataFrame({
            "time": times,
            "weather_proba_sunny": [0.7, 0.1, 0.2],
            "weather_proba_rainy": [0.2, 0.8, 0.1],
            "weather_proba_cloudy": [0.1, 0.1, 0.7],
            "mood_proba_happy": [0.8, 0.2, 0.9],
            "mood_proba_sad": [0.2, 0.8, 0.1],
        })
        scorer = LogLoss(components=["weather"])
        y_true_out, y_pred_out = check_scorer_column_selection(
            scorer=scorer,
            y_true=y_true,
            y_pred=y_pred,
            pred_type="class_proba",
            coverage_rates=None,
            interval_pattern=None,
        )
        assert set(y_true_out.columns) == {"time", "weather"}
        assert all("weather_proba_" in c for c in y_pred_out.columns if c != "time")
        assert not any("mood" in c for c in y_pred_out.columns)


class TestCheckTimeColumnEdgeCases:
    """Additional edge case tests for check_time_column."""

    def test_null_time_raises(self):
        """Time column with nulls raises ValueError."""
        df = pl.DataFrame({"time": [datetime(2020, 1, 1), None, datetime(2020, 1, 3)]})
        with pytest.raises(ValueError, match="null values"):
            check_time_column(df)

    def test_unsorted_time_raises(self):
        """Unsorted time column raises ValueError."""
        df = pl.DataFrame({"time": [datetime(2020, 1, 3), datetime(2020, 1, 1), datetime(2020, 1, 2)]})
        with pytest.raises(ValueError, match="sorted"):
            check_time_column(df)


class TestCheckPanelGroupNamesExistEdgeCases:
    """Additional tests for check_groups_exist."""

    def test_none_returns_early(self):
        """Passing None for requested_panel_groups returns without error."""
        check_groups_exist(
            fitted_panel_groups=["a", "b"],
            requested_panel_groups=None,
            context="predict",
        )

    def test_missing_groups_raises(self):
        """Requesting non-existent groups raises ValueError."""
        with pytest.raises(ValueError, match="not found in fitted"):
            check_groups_exist(
                fitted_panel_groups=["a", "b"],
                requested_panel_groups=["c"],
                context="predict",
            )


class TestCheckIntervalSubDay:
    """Tests for check_interval_consistency with sub-day data."""

    def test_hourly_data(self):
        """Hourly data returns correct interval string."""
        times = pl.datetime_range(datetime(2020, 1, 1, 0, 0), datetime(2020, 1, 1, 23, 0), "1h", eager=True)
        df = pl.DataFrame({"time": times})
        result = check_interval_consistency(df)
        assert "h" in result

    def test_inconsistent_interval_raises(self):
        """Inconsistent intervals raise ValueError."""
        df = pl.DataFrame({
            "time": [
                datetime(2020, 1, 1),
                datetime(2020, 1, 2),
                datetime(2020, 1, 5),
                datetime(2020, 1, 9),
                datetime(2020, 1, 20),
            ]
        })
        with pytest.raises(ValueError, match="inconsistent"):
            check_interval_consistency(df)


class TestCheckContinuityBranchCoverage:
    """Tests for check_continuity branch coverage."""

    def test_none_interval_skips_validation(self):
        """Passing expected_interval=None skips all checks."""
        df_p = pl.DataFrame({"time": [datetime(2020, 1, 1)]})
        df_n = pl.DataFrame({"time": [datetime(2020, 6, 1)]})
        check_continuity(df_p, df_n, expected_interval=None)

    def test_next_df_interval_mismatch_raises(self):
        """Next df with different interval from expected raises ValueError."""
        df_p = pl.DataFrame({
            "time": [datetime(2020, 1, 1)],
        })
        df_n = pl.DataFrame({
            "time": pl.datetime_range(datetime(2020, 1, 2), datetime(2020, 1, 6), "2d", eager=True),
        })
        with pytest.raises(ValueError, match="interval"):
            check_continuity(df_p, df_n, expected_interval="1d")


class TestCheckForecastingHorizonPositive:
    """Tests for check_forecasting_horizon_positive."""

    def test_none_with_allow_none_false_raises(self):
        """None horizon with allow_none=False raises ValueError."""
        with pytest.raises(ValueError, match="cannot be None"):
            check_forecasting_horizon_positive(None, allow_none=False)

    def test_none_with_allow_none_true_passes(self):
        """None horizon with allow_none=True returns without error."""
        check_forecasting_horizon_positive(None, allow_none=True)

    def test_zero_raises(self):
        """Zero horizon raises ValueError."""
        with pytest.raises(ValueError, match="must be >= 1"):
            check_forecasting_horizon_positive(0)

    def test_negative_raises(self):
        """Negative horizon raises ValueError."""
        with pytest.raises(ValueError, match="must be >= 1"):
            check_forecasting_horizon_positive(-5)

    def test_positive_passes(self):
        """Positive horizon passes without error."""
        check_forecasting_horizon_positive(10)


class TestCheckXActualRequired:
    """Tests for check_X_actual_required."""

    def test_none_X_with_positive_horizon_raises(self):
        """X_actual=None with observation_horizon > 0 raises ValueError."""
        with pytest.raises(ValueError):
            check_X_actual_required(None, observation_horizon=5, context="predict")

    def test_none_X_with_zero_horizon_passes(self):
        """X_actual=None with observation_horizon=0 passes without error."""
        check_X_actual_required(None, observation_horizon=0, context="predict")

    def test_provided_X_with_positive_horizon_passes(self):
        """X_actual provided with positive horizon passes without error."""
        X = pl.DataFrame({"time": [datetime(2020, 1, 1)], "val": [1.0]})
        check_X_actual_required(X, observation_horizon=5, context="predict")


class TestCheckPanelGroupNamesExist:
    """Tests for check_groups_exist."""

    def test_missing_group_raises(self):
        """Requesting a non-existent panel group raises ValueError."""
        with pytest.raises(ValueError, match="not found"):
            check_groups_exist(["g1"], ["g2", "g3"], "predict")

    def test_valid_groups_pass(self):
        """Requesting existing groups passes without error."""
        check_groups_exist(["g1", "g2", "g3"], ["g1", "g2"], "predict")


class TestValidateColumnNames:
    """Tests for validate_column_names."""

    def test_multiple_separators_raises(self):
        """Column with multiple __ separators raises ValueError."""
        df = pl.DataFrame({
            "time": [datetime(2020, 1, 1)],
            "a__b__c": [1.0],
        })
        with pytest.raises(ValueError, match="multiple __ separators"):
            validate_column_names(df)

    def test_leading_separator_raises(self):
        """Column starting with __ raises ValueError."""
        df = pl.DataFrame({
            "time": [datetime(2020, 1, 1)],
            "__sales": [1.0],
        })
        with pytest.raises(ValueError, match="beginning or end"):
            validate_column_names(df)

    def test_trailing_separator_raises(self):
        """Column ending with __ raises ValueError."""
        df = pl.DataFrame({
            "time": [datetime(2020, 1, 1)],
            "sales__": [1.0],
        })
        with pytest.raises(ValueError, match="beginning or end"):
            validate_column_names(df)

    def test_underscore_adjacent_to_separator_raises(self):
        """Column with underscore adjacent to __ separator raises ValueError."""
        df = pl.DataFrame({
            "time": [datetime(2020, 1, 1)],
            "store___sales": [1.0],
        })
        with pytest.raises(ValueError, match="underscores adjacent"):
            validate_column_names(df)

    def test_valid_panel_column_passes(self):
        """Valid panel column with group__series passes."""
        df = pl.DataFrame({
            "time": [datetime(2020, 1, 1)],
            "sales__store_1": [1.0],
        })
        validate_column_names(df)

    def test_global_column_passes(self):
        """Global column without __ passes."""
        df = pl.DataFrame({
            "time": [datetime(2020, 1, 1)],
            "sales": [1.0],
        })
        validate_column_names(df)


class TestNormalizeInterval:
    """Tests for _normalize_interval day-to-month conversion."""

    @pytest.mark.parametrize(
        ("interval", "expected"),
        [
            ("30d", "1mo"),
            ("28d", "1mo"),
            ("31d", "1mo"),
            ("60d", "2mo"),
            ("91d", "3mo"),
            ("182d", "6mo"),
            ("365d", "1y"),
            ("366d", "1y"),
            ("15d", "15d"),
            ("1h", "1h"),
            ("1mo", "1mo"),
        ],
    )
    def test_day_to_month_normalization(self, interval, expected):
        """Day intervals in month ranges normalize correctly."""
        assert _normalize_interval(interval) == expected


class TestTimedeltaToStringSubsecond:
    """Tests for _timedelta_to_string with subsecond precision."""

    def test_milliseconds(self):
        """Millisecond timedelta converts to ms string."""
        assert _timedelta_to_string(timedelta(milliseconds=5)) == "5ms"

    def test_microseconds(self):
        """Microsecond timedelta converts to us string."""
        assert _timedelta_to_string(timedelta(microseconds=500)) == "500us"

    def test_exact_millisecond_boundary(self):
        """Exactly 1000 microseconds converts to 1ms."""
        assert _timedelta_to_string(timedelta(microseconds=1000)) == "1ms"


class TestInferMonthlyFreqEdgeCases:
    """Tests for _infer_monthly_freq edge cases."""

    def test_single_element_returns_none(self):
        """Single-element list returns None."""
        assert _infer_monthly_freq([datetime(2024, 1, 1)]) is None

    def test_empty_list_returns_none(self):
        """Empty list returns None."""
        assert _infer_monthly_freq([]) is None


class TestCheckDayOfMonthConsistency:
    """Tests for _check_day_of_month_consistency."""

    def test_empty_list_returns_false(self):
        """Empty list returns False."""
        assert _check_day_of_month_consistency([]) is False

    def test_end_of_month_consistency(self):
        """End-of-month dates are consistent (Jan 31, Feb 29, Mar 31)."""
        dates = [datetime(2024, 1, 31), datetime(2024, 2, 29), datetime(2024, 3, 31)]
        assert _check_day_of_month_consistency(dates) is True

    def test_inconsistent_dates_return_false(self):
        """Dates with inconsistent day-of-month return False."""
        dates = [datetime(2024, 1, 15), datetime(2024, 2, 20), datetime(2024, 3, 15)]
        assert _check_day_of_month_consistency(dates) is False


class TestIntervalToTimedeltaMicroseconds:
    """Tests for interval_to_timedelta with subsecond units."""

    def test_microseconds(self):
        """Microsecond interval converts to correct timedelta."""
        result = interval_to_timedelta("100us")
        assert result == timedelta(microseconds=100)

    def test_milliseconds(self):
        """Millisecond interval converts to correct timedelta."""
        result = interval_to_timedelta("50ms")
        assert result == timedelta(milliseconds=50)


class TestCheckContinuity:
    """Tests for check_continuity gap and overlap detection."""

    def test_fixed_interval_gap_raises(self):
        """Gap in fixed-interval data raises ValueError."""
        df_p = pl.DataFrame({
            "time": pl.datetime_range(datetime(2020, 1, 1), datetime(2020, 1, 5), "1d", eager=True),
        })
        df_n = pl.DataFrame({
            "time": pl.datetime_range(datetime(2020, 1, 8), datetime(2020, 1, 12), "1d", eager=True),
        })
        with pytest.raises(ValueError, match="[Gg]ap"):
            check_continuity(df_p, df_n, expected_interval="1d")

    def test_fixed_interval_overlap_raises(self):
        """Overlap in fixed-interval data raises ValueError."""
        df_p = pl.DataFrame({
            "time": pl.datetime_range(datetime(2020, 1, 1), datetime(2020, 1, 5), "1d", eager=True),
        })
        df_n = pl.DataFrame({
            "time": pl.datetime_range(datetime(2020, 1, 4), datetime(2020, 1, 8), "1d", eager=True),
        })
        with pytest.raises(ValueError, match="[Oo]verlap"):
            check_continuity(df_p, df_n, expected_interval="1d")

    def test_continuous_data_passes(self):
        """Continuous data passes without error."""
        df_p = pl.DataFrame({
            "time": pl.datetime_range(datetime(2020, 1, 1), datetime(2020, 1, 5), "1d", eager=True),
        })
        df_n = pl.DataFrame({
            "time": pl.datetime_range(datetime(2020, 1, 6), datetime(2020, 1, 10), "1d", eager=True),
        })
        check_continuity(df_p, df_n, expected_interval="1d")

    def test_monthly_gap_raises(self):
        """Gap in monthly (variable-length) data raises ValueError."""
        df_p = pl.DataFrame({
            "time": [datetime(2024, 1, 1), datetime(2024, 2, 1)],
        })
        df_n = pl.DataFrame({
            "time": [datetime(2024, 5, 1), datetime(2024, 6, 1)],
        })
        with pytest.raises(ValueError, match="[Gg]ap"):
            check_continuity(df_p, df_n, expected_interval="1mo")

    def test_monthly_overlap_raises(self):
        """Overlap in monthly data raises ValueError."""
        df_p = pl.DataFrame({
            "time": [datetime(2024, 1, 1), datetime(2024, 2, 1)],
        })
        df_n = pl.DataFrame({
            "time": [datetime(2024, 2, 1), datetime(2024, 3, 1)],
        })
        with pytest.raises(ValueError, match="[Oo]verlap"):
            check_continuity(df_p, df_n, expected_interval="1mo")

    def test_previous_df_interval_mismatch_raises(self):
        """Previous df with wrong interval raises ValueError."""
        df_p = pl.DataFrame({
            "time": pl.datetime_range(datetime(2020, 1, 1), datetime(2020, 1, 5), "1h", eager=True),
        })
        df_n = pl.DataFrame({
            "time": pl.datetime_range(datetime(2020, 1, 6), datetime(2020, 1, 10), "1d", eager=True),
        })
        with pytest.raises(ValueError, match="interval"):
            check_continuity(df_p, df_n, expected_interval="1d")

    def test_interval_mismatch_between_frames_raises(self):
        """DataFrames with different internal intervals raise ValueError."""
        df_p = pl.DataFrame({
            "time": pl.datetime_range(datetime(2020, 1, 1), datetime(2020, 1, 3), "1h", eager=True),
        })
        df_n = pl.DataFrame({
            "time": pl.datetime_range(datetime(2020, 1, 4), datetime(2020, 1, 6), "1d", eager=True),
        })
        with pytest.raises(ValueError, match="[Ii]nterval|mismatch"):
            check_continuity(df_p, df_n, expected_interval="1h")


class TestCheckInputsIntervalMismatch:
    """Tests for check_inputs with mismatched y/X time intervals."""

    def test_different_intervals_raises(self):
        """Y and X with different time intervals raises ValueError."""
        y = pl.DataFrame({
            "time": pl.datetime_range(datetime(2020, 1, 1), datetime(2020, 1, 10), "1d", eager=True),
            "target": range(10),
        })
        X = pl.DataFrame({
            "time": pl.datetime_range(datetime(2020, 1, 1, 0), datetime(2020, 1, 1, 9), "1h", eager=True),
            "feature": range(10),
        })
        with pytest.raises(ValueError, match="[Ii]nterval mismatch"):
            check_inputs(y, X)


class TestAddIntervalAllUnits:
    """Tests for add_interval with various time units."""

    def test_hours(self):
        """Adding hours works correctly."""
        start = datetime(2020, 1, 1, 0, 0)
        result = add_interval(start, "3h", 1)
        assert result == datetime(2020, 1, 1, 3, 0)

    def test_minutes(self):
        """Adding minutes works correctly."""
        start = datetime(2020, 1, 1, 0, 0)
        result = add_interval(start, "30m", 1)
        assert result == datetime(2020, 1, 1, 0, 30)

    def test_seconds(self):
        """Adding seconds works correctly."""
        start = datetime(2020, 1, 1, 0, 0)
        result = add_interval(start, "10s", 1)
        assert result == datetime(2020, 1, 1, 0, 0, 10)

    def test_weeks(self):
        """Adding weeks works correctly."""
        start = datetime(2020, 1, 1)
        result = add_interval(start, "1w", 2)
        assert result == datetime(2020, 1, 15)

    def test_quarters(self):
        """Adding quarters (3 months) works correctly."""
        start = datetime(2020, 1, 1)
        result = add_interval(start, "1q", 1)
        assert result == datetime(2020, 4, 1)

    def test_years(self):
        """Adding years works correctly."""
        start = datetime(2020, 1, 1)
        result = add_interval(start, "1y", 1)
        assert result == datetime(2021, 1, 1)

    def test_month_day_preservation(self):
        """Adding months preserves day-of-month when possible."""
        start = datetime(2020, 1, 15)
        result = add_interval(start, "1mo", 1)
        assert result == datetime(2020, 2, 15)

    def test_month_day_clamping(self):
        """Adding months clamps day to month end when needed."""
        start = datetime(2020, 1, 31)
        result = add_interval(start, "1mo", 1)
        assert result == datetime(2020, 2, 29)

    def test_invalid_unit_raises(self):
        """Invalid unit raises ValueError."""
        with pytest.raises(ValueError):
            add_interval(datetime(2020, 1, 1), "1x", 1)

    def test_days(self):
        """Adding days works correctly."""
        start = datetime(2020, 1, 1)
        result = add_interval(start, "2d", 3)
        assert result == datetime(2020, 1, 7)
