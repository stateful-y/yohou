"""Tests for train_test_split and _split_X_forecast."""

from __future__ import annotations

from datetime import date, datetime, timedelta

import numpy as np
import polars as pl
import pytest

from yohou.model_selection import train_test_split
from yohou.model_selection.utils import _split_X_forecast


@pytest.fixture()
def y_daily():
    """10-row daily DataFrame with time and value columns."""
    return pl.DataFrame({
        "time": pl.date_range(date(2020, 1, 1), date(2020, 1, 10), eager=True),
        "value": list(range(10)),
    })


@pytest.fixture()
def X_daily():
    """10-row daily DataFrame with time and a feature column."""
    return pl.DataFrame({
        "time": pl.date_range(date(2020, 1, 1), date(2020, 1, 10), eager=True),
        "feature": list(range(100, 110)),
    })


@pytest.fixture()
def X_forecast_daily():
    """X_forecast with 3 vintages issued on Jan 3, 6, 9 (each covers 3 time steps)."""
    rows = []
    for v_day in [3, 6, 9]:
        vt = date(2020, 1, v_day)
        for offset in range(1, 4):
            rows.append({
                "vintage_time": vt,
                "time": date(2020, 1, v_day + offset),
                "temperature": float(v_day * 10 + offset),
            })
    return pl.DataFrame(rows).cast({"temperature": pl.Float64})


class TestTrainTestSplitRowIndex:
    """Row-index splitting of positional arrays."""

    def test_single_array_int_test_size(self, y_daily):
        y_train, y_test = train_test_split(y_daily, test_size=3)
        assert len(y_train) == 7
        assert len(y_test) == 3
        assert y_train["value"].to_list() == list(range(7))
        assert y_test["value"].to_list() == list(range(7, 10))

    def test_single_array_float_test_size(self, y_daily):
        y_train, y_test = train_test_split(y_daily, test_size=0.3)
        assert len(y_train) == 7
        assert len(y_test) == 3

    def test_two_arrays(self, y_daily, X_daily):
        y_train, y_test, X_actual_train, X_actual_test = train_test_split(
            y_daily,
            X_daily,
            test_size=2,
        )
        assert len(y_train) == 8
        assert len(y_test) == 2
        assert len(X_actual_train) == 8
        assert len(X_actual_test) == 2
        # Temporal order preserved
        assert y_train["time"][-1] < y_test["time"][0]
        assert X_actual_train["time"][-1] < X_actual_test["time"][0]

    def test_three_arrays(self, y_daily, X_daily):
        third = pl.DataFrame({"z": list(range(10))})
        result = train_test_split(y_daily, X_daily, third, test_size=4)
        assert len(result) == 6
        for i in range(0, 6, 2):
            assert len(result[i]) == 6  # trains
            assert len(result[i + 1]) == 4  # tests

    def test_temporal_order_preserved(self, y_daily):
        y_train, y_test = train_test_split(y_daily, test_size=3)
        assert y_train["time"].is_sorted()
        assert y_test["time"].is_sorted()
        assert y_train["time"][-1] < y_test["time"][0]

    def test_float_rounds_correctly(self):
        y = pl.DataFrame({
            "time": pl.date_range(date(2020, 1, 1), date(2020, 4, 9), eager=True),
            "v": list(range(100)),
        })
        y_train, y_test = train_test_split(y, test_size=0.15)
        assert len(y_test) == 15
        assert len(y_train) == 85

    def test_minimum_test_size_float(self):
        """Very small float still produces at least 1 test row."""
        y = pl.DataFrame({
            "time": pl.date_range(date(2020, 1, 1), date(2020, 1, 5), eager=True),
            "v": list(range(5)),
        })
        y_train, y_test = train_test_split(y, test_size=0.01)
        assert len(y_test) >= 1


class TestTrainTestSplitValidation:
    """Input validation for train_test_split."""

    def test_no_arrays_raises(self):
        with pytest.raises(ValueError, match="At least one array"):
            train_test_split(test_size=2)

    def test_mismatched_lengths_raises(self, y_daily):
        short = pl.DataFrame({"a": [1, 2, 3]})
        with pytest.raises(ValueError, match="same number of rows"):
            train_test_split(y_daily, short, test_size=2)

    def test_float_out_of_range_raises(self, y_daily):
        with pytest.raises(ValueError, match="must be in \\(0.0, 1.0\\)"):
            train_test_split(y_daily, test_size=0.0)
        with pytest.raises(ValueError, match="must be in \\(0.0, 1.0\\)"):
            train_test_split(y_daily, test_size=1.0)
        with pytest.raises(ValueError, match="must be in \\(0.0, 1.0\\)"):
            train_test_split(y_daily, test_size=-0.5)

    def test_int_too_large_raises(self, y_daily):
        with pytest.raises(ValueError, match="must be in \\[1, 9\\]"):
            train_test_split(y_daily, test_size=10)

    def test_int_zero_raises(self, y_daily):
        with pytest.raises(ValueError, match="must be in \\[1, 9\\]"):
            train_test_split(y_daily, test_size=0)

    def test_invalid_type_raises(self, y_daily):
        with pytest.raises(TypeError, match="must be int or float"):
            train_test_split(y_daily, test_size="half")


class TestTrainTestSplitXForecast:
    """X_forecast splitting by vintage_time range."""

    def test_x_forecast_split(self, y_daily, X_forecast_daily):
        # Split at row 7 → cutoff = Jan 7, test_end = Jan 10
        result = train_test_split(
            y_daily,
            test_size=3,
            X_forecast=X_forecast_daily,
        )
        assert len(result) == 4  # y_train, y_test, Xf_train, Xf_test
        y_train, y_test, Xf_train, Xf_test = result

        assert len(y_train) == 7
        assert len(y_test) == 3

        # Vintage Jan 3 (<= Jan 7 cutoff) → train
        # Vintage Jan 6 (<= Jan 7 cutoff) → train
        # Vintage Jan 9 (> Jan 7 cutoff and <= Jan 10 test_end) → test
        assert Xf_train["vintage_time"].unique().sort().to_list() == [
            date(2020, 1, 3),
            date(2020, 1, 6),
        ]
        assert Xf_test["vintage_time"].unique().sort().to_list() == [
            date(2020, 1, 9),
        ]

    def test_x_forecast_with_multiple_arrays(self, y_daily, X_daily, X_forecast_daily):
        result = train_test_split(
            y_daily,
            X_daily,
            test_size=3,
            X_forecast=X_forecast_daily,
        )
        # y_train, y_test, X_actual_train, X_actual_test, Xf_train, Xf_test
        assert len(result) == 6
        y_train, y_test, X_actual_train, X_actual_test, Xf_train, Xf_test = result
        assert len(y_train) == 7
        assert len(X_actual_train) == 7
        assert len(Xf_train) > 0
        assert len(Xf_test) > 0

    def test_x_forecast_no_time_column_raises(self):
        y = pl.DataFrame({"value": [1, 2, 3, 4, 5]})
        xf = pl.DataFrame({
            "vintage_time": [date(2020, 1, 1)],
            "time": [date(2020, 1, 2)],
            "temp": [10.0],
        })
        with pytest.raises(ValueError, match="must contain a 'time' column"):
            train_test_split(y, test_size=2, X_forecast=xf)

    def test_x_forecast_missing_vintage_time_raises(self, y_daily):
        bad_xf = pl.DataFrame({
            "time": [date(2020, 1, 2)],
            "temp": [10.0],
        })
        with pytest.raises(ValueError, match="vintage_time"):
            train_test_split(y_daily, test_size=2, X_forecast=bad_xf)

    def test_x_forecast_empty_test_when_all_before_cutoff(self, y_daily):
        """All vintages before cutoff → empty test split."""
        xf = pl.DataFrame({
            "vintage_time": [date(2020, 1, 1), date(2020, 1, 2)],
            "time": [date(2020, 1, 2), date(2020, 1, 3)],
            "temp": [1.0, 2.0],
        })
        _, _, Xf_train, Xf_test = train_test_split(
            y_daily,
            test_size=3,
            X_forecast=xf,
        )
        assert len(Xf_train) == 2
        assert len(Xf_test) == 0

    def test_x_forecast_empty_train_when_all_after_cutoff(self, y_daily):
        """All vintages after test end → empty train and test."""
        xf = pl.DataFrame({
            "vintage_time": [date(2020, 1, 11), date(2020, 1, 12)],
            "time": [date(2020, 1, 12), date(2020, 1, 13)],
            "temp": [1.0, 2.0],
        })
        _, _, Xf_train, Xf_test = train_test_split(
            y_daily,
            test_size=3,
            X_forecast=xf,
        )
        assert len(Xf_train) == 0
        assert len(Xf_test) == 0

    def test_x_forecast_boundary_vintage_goes_to_train(self, y_daily):
        """Vintage exactly at cutoff_time goes to train (<=)."""
        cutoff = date(2020, 1, 7)  # split_idx=7, cutoff = y["time"][6]
        xf = pl.DataFrame({
            "vintage_time": [cutoff, cutoff],
            "time": [date(2020, 1, 8), date(2020, 1, 9)],
            "temp": [1.0, 2.0],
        })
        _, _, Xf_train, Xf_test = train_test_split(
            y_daily,
            test_size=3,
            X_forecast=xf,
        )
        assert len(Xf_train) == 2
        assert len(Xf_test) == 0


class TestSplitForecastData:
    """Tests for the internal _split_X_forecast function."""

    @pytest.fixture()
    def y_100(self):
        return pl.DataFrame({
            "time": pl.date_range(date(2020, 1, 1), date(2020, 4, 9), eager=True),
            "value": list(range(100)),
        })

    @pytest.fixture()
    def xf_multi_vintage(self):
        """X_forecast with vintages at days 20, 40, 60, 80."""
        rows = []
        for v_offset in [20, 40, 60, 80]:
            vt = date(2020, 1, 1) + timedelta(days=v_offset - 1)
            for step in range(1, 4):
                rows.append({
                    "vintage_time": vt,
                    "time": date(2020, 1, 1) + timedelta(days=v_offset - 1 + step),
                    "temp": float(v_offset + step),
                })
        return pl.DataFrame(rows).cast({"temp": pl.Float64})

    def test_basic_split(self, y_100, xf_multi_vintage):
        # Train indices 0..59, test 60..79 → cutoff = day 60, test_end = day 80
        train = np.arange(60, dtype=np.intp)
        test = np.arange(60, 80, dtype=np.intp)
        Xf_train, Xf_test = _split_X_forecast(xf_multi_vintage, y_100, train, test)

        # Vintages: day 20 (Jan 20), day 40 (Feb 9), day 60 (Feb 29) → train (<=cutoff)
        # day 80 (Mar 20) → test (> cutoff and <= test_end)
        assert Xf_train is not None
        assert Xf_test is not None
        train_vintages = Xf_train["vintage_time"].unique().sort().to_list()
        test_vintages = Xf_test["vintage_time"].unique().sort().to_list()
        assert len(train_vintages) == 3  # days 20, 40, 60
        assert len(test_vintages) == 1  # day 80

    def test_none_returns_none(self, y_100):
        train = np.arange(60, dtype=np.intp)
        test = np.arange(60, 80, dtype=np.intp)
        result = _split_X_forecast(None, y_100, train, test)
        assert result == (None, None)

    def test_no_future_leakage(self, y_100, xf_multi_vintage):
        """Training set must not contain vintages after the cutoff."""
        train = np.arange(30, dtype=np.intp)  # cutoff = day 30
        test = np.arange(30, 50, dtype=np.intp)
        Xf_train, Xf_test = _split_X_forecast(xf_multi_vintage, y_100, train, test)

        cutoff = y_100["time"][29]
        assert Xf_train is not None
        assert (Xf_train["vintage_time"] <= cutoff).all()

    def test_test_excludes_future_folds(self, y_100, xf_multi_vintage):
        """Test set must not contain vintages beyond test_end."""
        train = np.arange(30, dtype=np.intp)
        test = np.arange(30, 50, dtype=np.intp)  # test_end = day 50
        Xf_train, Xf_test = _split_X_forecast(xf_multi_vintage, y_100, train, test)

        test_end = y_100["time"][49]
        assert Xf_test is not None
        assert (Xf_test["vintage_time"] <= test_end).all()
        assert (Xf_test["vintage_time"] > y_100["time"][29]).all()

    def test_cutoff_boundary_inclusive(self, y_100):
        """Vintage exactly at cutoff goes to train."""
        cutoff_date = y_100["time"][49]  # day 50
        xf = pl.DataFrame({
            "vintage_time": [cutoff_date],
            "time": [cutoff_date + timedelta(days=1)],
            "temp": [99.0],
        })
        train = np.arange(50, dtype=np.intp)
        test = np.arange(50, 70, dtype=np.intp)
        Xf_train, Xf_test = _split_X_forecast(xf, y_100, train, test)
        assert Xf_train is not None
        assert len(Xf_train) == 1
        assert Xf_test is not None
        assert len(Xf_test) == 0

    def test_preserves_all_columns(self, y_100):
        """All original columns are preserved in both splits."""
        xf = pl.DataFrame({
            "vintage_time": [date(2020, 1, 10), date(2020, 2, 15), date(2020, 3, 10)],
            "time": [date(2020, 1, 11), date(2020, 2, 16), date(2020, 3, 11)],
            "temp": [1.0, 2.0, 3.0],
            "wind": [10.0, 20.0, 30.0],
        })
        train = np.arange(50, dtype=np.intp)
        test = np.arange(50, 80, dtype=np.intp)
        Xf_train, Xf_test = _split_X_forecast(xf, y_100, train, test)
        assert Xf_train is not None
        assert Xf_test is not None
        assert set(Xf_train.columns) == {"vintage_time", "time", "temp", "wind"}
        assert set(Xf_test.columns) == {"vintage_time", "time", "temp", "wind"}

    def test_datetime_types(self):
        """Works with pl.Datetime (not just pl.Date)."""
        base = datetime(2020, 1, 1)
        y = pl.DataFrame({
            "time": [base + timedelta(hours=i) for i in range(100)],
            "v": list(range(100)),
        })
        xf = pl.DataFrame({
            "vintage_time": [base + timedelta(hours=30), base + timedelta(hours=70)],
            "time": [base + timedelta(hours=31), base + timedelta(hours=71)],
            "temp": [1.0, 2.0],
        })
        train = np.arange(50, dtype=np.intp)
        test = np.arange(50, 80, dtype=np.intp)
        Xf_train, Xf_test = _split_X_forecast(xf, y, train, test)
        assert Xf_train is not None
        assert len(Xf_train) == 1  # hour 30 <= hour 49 cutoff
        assert Xf_test is not None
        assert len(Xf_test) == 1  # hour 70 in (49, 79]
