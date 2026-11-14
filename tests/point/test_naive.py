from datetime import date, datetime, timedelta

import numpy as np
import polars as pl
import polars.testing
import pytest
from sklearn.base import clone
from sklearn.model_selection import train_test_split

from conftest import run_checks
from yohou.compose import DecompositionPipeline
from yohou.point import MeanSeasonalNaive, SeasonalNaive, SegmentProfileForecaster
from yohou.segmenting import CalendarSegmenter, HolidaySegmenter, UnionSegmenter
from yohou.stationarity import PolynomialTrendForecaster
from yohou.testing import _yield_yohou_forecaster_checks


@pytest.fixture(scope="module")
def naive_data():
    """Deterministic data shared by SeasonalNaive tests."""
    length = 22
    y = pl.DataFrame({
        "time": pl.datetime_range(
            start=datetime(2021, 12, 16),
            end=datetime(2021, 12, 16, 0, 0, length - 1),
            interval="1s",
            eager=True,
        ),
        "a": range(length),
        "b": range(10, length + 10),
    })
    X_actual = pl.DataFrame({
        "time": pl.datetime_range(
            start=datetime(2021, 12, 16),
            end=datetime(2021, 12, 16, 0, 0, length - 1),
            interval="1s",
            eager=True,
        ),
        "c": range(length),
        "d": range(10, length + 10),
        "e": range(20, length + 20),
    })
    return y, X_actual


class TestSeasonalNaive:
    @pytest.mark.parametrize(
        "forecaster,expected_failures",
        [
            (
                SeasonalNaive(seasonality=1),
                [],
            ),
            (
                SeasonalNaive(seasonality=7),
                [],
            ),
        ],
    )
    def test_seasonal_naive_checks(self, forecaster, expected_failures, y_X_factory):
        """Run systematic checks on SeasonalNaive forecaster."""
        y, X_actual = y_X_factory(length=200, seed=42)
        y_train, y_test = y[:160], y[160:]
        X_actual_train, X_actual_test = (X_actual[:160], X_actual[160:]) if X_actual is not None else (None, None)

        forecaster_fitted = clone(forecaster)
        forecaster_fitted.fit(y_train, X_actual_train, forecasting_horizon=3)

        run_checks(
            forecaster_fitted,
            _yield_yohou_forecaster_checks(forecaster_fitted, y_train, X_actual_train, y_test, X_actual_test),
            expected_failures=set(expected_failures),
        )

    @pytest.mark.parametrize(
        "seasonality, fit_forecasting_horizon, predict_forecasting_horizon, expected_a",
        [
            (1, 1, 5, [16, 16, 16, 16, 16]),
            (1, 3, 5, [16, 16, 16, 16, 16]),
            (1, 3, 2, [16, 16]),
            (2, 1, 5, [15, 16, 15, 16, 15]),
            (2, 3, 5, [15, 16, 15, 16, 15]),
            (2, 3, 2, [15, 16]),
            (8, 1, 5, [9, 10, 11, 12, 13]),
            (8, 3, 5, [9, 10, 11, 12, 13]),
            (8, 3, 2, [9, 10]),
        ],
    )
    def test_predict(self, seasonality, fit_forecasting_horizon, predict_forecasting_horizon, expected_a, naive_data):
        """Test specific prediction behavior of SeasonalNaive."""
        y, X_actual = naive_data
        y_train, y_test, X_actual_train, X_actual_test = train_test_split(y, X_actual, test_size=0.2, shuffle=False)
        forecaster = SeasonalNaive(seasonality=seasonality)

        forecaster.fit(y=y_train, X_actual=X_actual_train, forecasting_horizon=fit_forecasting_horizon)

        y_pred = forecaster.predict(forecasting_horizon=predict_forecasting_horizon)

        expected_y_pred = pl.DataFrame({
            "vintage_time": [y_train["time"][-1]] * predict_forecasting_horizon,
            "time": pl.datetime_range(
                start=datetime(2021, 12, 16, 0, 0, len(y_train)),
                end=datetime(2021, 12, 16, 0, 0, len(y_train) + predict_forecasting_horizon - 1),
                interval="1s",
                eager=True,
            ),
            "a": expected_a,
            "b": np.array(expected_a) + 10,
        })
        pl.testing.assert_frame_equal(y_pred, expected_y_pred)


class TestSeasonalNaiveWithoutExogenous:
    """Tests for SeasonalNaive with X_actual=None through the stateful lifecycle.

    The plain fit/predict X_actual=None path and the requires_exogenous tag are
    already validated by the systematic suite (check_fit_predict_without_exogenous),
    so only the observe/rewind variants, which the systematic checks run with
    non-None X_actual, are exercised here.
    """

    def test_observe_predict_without_exogenous(self, naive_data):
        """SeasonalNaive observe_predict should work without exogenous features."""
        y, _ = naive_data
        forecaster = SeasonalNaive(seasonality=3)
        forecaster.fit(y[:17], X_actual=None, forecasting_horizon=3)
        y_pred = forecaster.observe_predict(y[17:19], X_actual=None)

        assert isinstance(y_pred, pl.DataFrame)
        assert "vintage_time" in y_pred.columns
        assert "time" in y_pred.columns

    def test_rewind_without_exogenous(self, naive_data):
        """SeasonalNaive rewind should work without exogenous features."""
        y, _ = naive_data
        forecaster = SeasonalNaive(seasonality=3)
        forecaster.fit(y[:17], X_actual=None, forecasting_horizon=3)
        forecaster.observe(y[17:19], X_actual=None)
        forecaster.rewind(y[:17], X_actual=None)
        y_pred = forecaster.predict(forecasting_horizon=3)

        assert isinstance(y_pred, pl.DataFrame)
        assert len(y_pred) == 3

    def test_seasonal_naive_checks_with_step_data(self, y_X_factory):
        """Systematic checks include requires_exogenous warning when step data provided."""
        y, X_actual, X_future, X_forecast = y_X_factory(
            length=200,
            seed=42,
            n_future_features=2,
            n_forecast_features=2,
            return_exogenous=True,
        )
        y_train, y_test = y[:160], y[160:]
        X_actual_train, X_actual_test = (X_actual[:160], X_actual[160:]) if X_actual is not None else (None, None)

        forecaster = SeasonalNaive(seasonality=7)
        forecaster_fitted = clone(forecaster)
        forecaster_fitted.fit(y_train, X_actual_train, forecasting_horizon=3)

        run_checks(
            forecaster_fitted,
            _yield_yohou_forecaster_checks(
                forecaster_fitted,
                y_train,
                X_actual_train,
                y_test,
                X_actual_test,
                X_future_train=X_future,
                X_future_test=X_future,
                X_forecast_train=X_forecast,
                X_forecast_test=X_forecast,
            ),
            expected_failures=set(),
        )


class TestSeasonalNaivePanel:
    """Tests for SeasonalNaive with panel data."""

    def test_panel_predict(self):
        """Panel predict produces correct columns and repeats seasonal pattern."""
        length = 20
        y = pl.DataFrame({
            "time": pl.datetime_range(
                start=datetime(2021, 1, 1),
                end=datetime(2021, 1, 1, 0, 0, length - 1),
                interval="1s",
                eager=True,
            ),
            "grp__a": list(range(length)),
            "grp__b": list(range(10, length + 10)),
        })
        forecaster = SeasonalNaive(seasonality=3)
        forecaster.fit(y=y[:15], forecasting_horizon=5)
        y_pred = forecaster.predict(forecasting_horizon=5)

        assert "grp__a" in y_pred.columns
        assert "grp__b" in y_pred.columns
        assert len(y_pred) == 5
        # Last 3 values: [12, 13, 14], repeated => [12, 13, 14, 12, 13]
        assert y_pred["grp__a"].to_list() == [12, 13, 14, 12, 13]

    def test_panel_predict_fh_less_than_seasonality(self):
        """Panel predict works when fh <= seasonality (no repetition needed)."""
        length = 20
        y = pl.DataFrame({
            "time": pl.datetime_range(
                start=datetime(2021, 1, 1),
                end=datetime(2021, 1, 1, 0, 0, length - 1),
                interval="1s",
                eager=True,
            ),
            "grp__a": list(range(length)),
        })
        forecaster = SeasonalNaive(seasonality=5)
        forecaster.fit(y=y[:15], forecasting_horizon=3)
        y_pred = forecaster.predict(forecasting_horizon=3)

        assert len(y_pred) == 3
        # Last 5 values: [10, 11, 12, 13, 14], take first 3 => [10, 11, 12]
        assert y_pred["grp__a"].to_list() == [10, 11, 12]


@pytest.fixture(scope="module")
def mean_naive_data():
    """Deterministic data shared by MeanSeasonalNaive tests (longer than naive_data)."""
    length = 50
    y = pl.DataFrame({
        "time": pl.datetime_range(
            start=datetime(2021, 12, 16),
            end=datetime(2021, 12, 16, 0, 0, length - 1),
            interval="1s",
            eager=True,
        ),
        "a": [float(i) for i in range(length)],
        "b": [float(i) for i in range(10, length + 10)],
    })
    X_actual = pl.DataFrame({
        "time": pl.datetime_range(
            start=datetime(2021, 12, 16),
            end=datetime(2021, 12, 16, 0, 0, length - 1),
            interval="1s",
            eager=True,
        ),
        "c": range(length),
        "d": range(10, length + 10),
        "e": range(20, length + 20),
    })
    return y, X_actual


class TestMeanSeasonalNaive:
    @pytest.mark.parametrize(
        "forecaster,expected_failures",
        [
            (
                MeanSeasonalNaive(seasonality=1, n_seasons=1),
                [],
            ),
            (
                MeanSeasonalNaive(seasonality=3, n_seasons=3),
                [],
            ),
        ],
    )
    def test_mean_seasonal_naive_checks(self, forecaster, expected_failures, y_X_factory):
        """Run systematic checks on MeanSeasonalNaive forecaster."""
        y, X_actual = y_X_factory(length=200, seed=42)
        y_train, y_test = y[:160], y[160:]
        X_actual_train, X_actual_test = (X_actual[:160], X_actual[160:]) if X_actual is not None else (None, None)

        forecaster_fitted = clone(forecaster)
        forecaster_fitted.fit(y_train, X_actual_train, forecasting_horizon=3)

        run_checks(
            forecaster_fitted,
            _yield_yohou_forecaster_checks(forecaster_fitted, y_train, X_actual_train, y_test, X_actual_test),
            expected_failures=set(expected_failures),
        )

    @pytest.mark.parametrize(
        "seasonality, n_seasons, fit_fh, predict_fh, expected_a",
        [
            # n_seasons=1 should match SeasonalNaive exactly (dtype preserved)
            # train has 40 rows (a=0.0..39.0), last 1 value = [39.0]
            (1, 1, 5, 5, [39.0, 39.0, 39.0, 39.0, 39.0]),
            # seasonality=2, n_seasons=1: last 2 = [38.0, 39.0]
            (2, 1, 5, 5, [38.0, 39.0, 38.0, 39.0, 38.0]),
            # seasonality=2, n_seasons=2: last 4 = [36.0,37.0,38.0,39.0]
            #   reshape [[36,37],[38,39]], mean = [37.0, 38.0]
            (2, 2, 5, 5, [37.0, 38.0, 37.0, 38.0, 37.0]),
            # seasonality=3, n_seasons=2: last 6 = [34.0,35.0,36.0,37.0,38.0,39.0]
            #   reshape [[34,35,36],[37,38,39]], mean = [35.5, 36.5, 37.5]
            (3, 2, 5, 5, [35.5, 36.5, 37.5, 35.5, 36.5]),
            # seasonality=3, n_seasons=3: last 9 = [31.0..39.0]
            #   reshape [[31,32,33],[34,35,36],[37,38,39]], mean = [34.0, 35.0, 36.0]
            (3, 3, 3, 3, [34.0, 35.0, 36.0]),
            # predict_fh < seasonality
            (3, 2, 3, 2, [35.5, 36.5]),
        ],
    )
    def test_predict(self, seasonality, n_seasons, fit_fh, predict_fh, expected_a, mean_naive_data):
        """Test specific prediction behavior of MeanSeasonalNaive."""
        y, X_actual = mean_naive_data
        y_train, y_test, X_actual_train, X_actual_test = train_test_split(y, X_actual, test_size=0.2, shuffle=False)
        forecaster = MeanSeasonalNaive(seasonality=seasonality, n_seasons=n_seasons)

        forecaster.fit(y=y_train, X_actual=X_actual_train, forecasting_horizon=fit_fh)

        y_pred = forecaster.predict(forecasting_horizon=predict_fh)

        expected_b = [a + 10 for a in expected_a]

        expected_y_pred = pl.DataFrame({
            "vintage_time": [y_train["time"][-1]] * predict_fh,
            "time": pl.datetime_range(
                start=datetime(2021, 12, 16, 0, 0, len(y_train)),
                end=datetime(2021, 12, 16, 0, 0, len(y_train) + predict_fh - 1),
                interval="1s",
                eager=True,
            ),
            "a": expected_a,
            "b": expected_b,
        })
        pl.testing.assert_frame_equal(y_pred, expected_y_pred)

    def test_n_seasons_1_matches_seasonal_naive(self, mean_naive_data):
        """MeanSeasonalNaive(n_seasons=1) should produce identical output to SeasonalNaive."""
        y, X_actual = mean_naive_data
        y_train, _, X_actual_train, _ = train_test_split(y, X_actual, test_size=0.2, shuffle=False)

        for seasonality in [1, 3, 7]:
            naive = SeasonalNaive(seasonality=seasonality)
            naive.fit(y=y_train, X_actual=X_actual_train, forecasting_horizon=5)
            y_naive = naive.predict(forecasting_horizon=5)

            mean_naive = MeanSeasonalNaive(seasonality=seasonality, n_seasons=1)
            mean_naive.fit(y=y_train, X_actual=X_actual_train, forecasting_horizon=5)
            y_mean = mean_naive.predict(forecasting_horizon=5)

            pl.testing.assert_frame_equal(y_mean, y_naive)

    def test_n_seasons_1_preserves_integer_dtype(self):
        """n_seasons=1 returns the observed values verbatim, preserving the column dtype.

        With a Float64 fixture the dtype check is vacuous, so use an explicit
        Int64 column: averaging would upcast to Float64, the verbatim path must not.
        """
        length = 30
        y = pl.DataFrame({
            "time": pl.datetime_range(
                start=datetime(2021, 12, 16),
                end=datetime(2021, 12, 16, 0, 0, length - 1),
                interval="1s",
                eager=True,
            ),
            "a": pl.Series(range(length), dtype=pl.Int64),
        })

        mean_naive = MeanSeasonalNaive(seasonality=3, n_seasons=1)
        mean_naive.fit(y=y[:24], X_actual=None, forecasting_horizon=3)
        y_mean = mean_naive.predict(forecasting_horizon=3)

        assert y_mean.schema["a"] == pl.Int64


class TestMeanSeasonalNaiveWithoutExogenous:
    """Tests for MeanSeasonalNaive with X_actual=None through the stateful lifecycle.

    The plain fit/predict X_actual=None path and the requires_exogenous tag are
    already validated by the systematic suite (check_fit_predict_without_exogenous),
    so only the observe/rewind variants, which the systematic checks run with
    non-None X_actual, are exercised here.
    """

    def test_observe_predict_without_exogenous(self, mean_naive_data):
        """MeanSeasonalNaive observe_predict should work without exogenous features."""
        y, _ = mean_naive_data
        forecaster = MeanSeasonalNaive(seasonality=3, n_seasons=2)
        forecaster.fit(y[:40], X_actual=None, forecasting_horizon=3)
        y_pred = forecaster.observe_predict(y[40:42], X_actual=None)

        assert isinstance(y_pred, pl.DataFrame)
        assert "vintage_time" in y_pred.columns
        assert "time" in y_pred.columns

    def test_rewind_without_exogenous(self, mean_naive_data):
        """MeanSeasonalNaive rewind should work without exogenous features."""
        y, _ = mean_naive_data
        forecaster = MeanSeasonalNaive(seasonality=3, n_seasons=2)
        forecaster.fit(y[:40], X_actual=None, forecasting_horizon=3)
        forecaster.observe(y[40:42], X_actual=None)
        forecaster.rewind(y[:40], X_actual=None)
        y_pred = forecaster.predict(forecasting_horizon=3)

        assert isinstance(y_pred, pl.DataFrame)
        assert len(y_pred) == 3


class TestMeanSeasonalNaivePanel:
    """Tests for MeanSeasonalNaive with panel data."""

    @pytest.fixture()
    def panel_y(self):
        """Panel y with two groups, each with two series."""
        length = 30
        return pl.DataFrame({
            "time": pl.datetime_range(
                start=datetime(2021, 1, 1),
                end=datetime(2021, 1, 1, 0, 0, length - 1),
                interval="1s",
                eager=True,
            ),
            "grp__a": [float(i) for i in range(length)],
            "grp__b": [float(i + 10) for i in range(length)],
        })

    def test_panel_predict(self, panel_y):
        """Panel predict produces correct columns and length."""
        forecaster = MeanSeasonalNaive(seasonality=3, n_seasons=2)
        forecaster.fit(y=panel_y[:24], forecasting_horizon=3)
        y_pred = forecaster.predict(forecasting_horizon=3)

        assert "grp__a" in y_pred.columns
        assert "grp__b" in y_pred.columns
        assert len(y_pred) == 3

    def test_panel_predict_with_repeat(self, panel_y):
        """Panel predict repeats the seasonal pattern when fh > seasonality."""
        forecaster = MeanSeasonalNaive(seasonality=2, n_seasons=2)
        forecaster.fit(y=panel_y[:24], forecasting_horizon=5)
        y_pred = forecaster.predict(forecasting_horizon=5)

        assert len(y_pred) == 5
        assert "grp__a" in y_pred.columns
        # Last 4 values: [20,21,22,23], reshape [[20,21],[22,23]], mean=[21,22]
        # Repeated: [21,22,21,22,21]
        assert y_pred["grp__a"].to_list() == [21.0, 22.0, 21.0, 22.0, 21.0]

    def test_panel_observe_predict(self, panel_y):
        """Panel observe_predict works correctly."""
        forecaster = MeanSeasonalNaive(seasonality=3, n_seasons=2)
        forecaster.fit(y=panel_y[:24], forecasting_horizon=3)
        y_pred = forecaster.observe_predict(panel_y[24:26])

        assert isinstance(y_pred, pl.DataFrame)
        assert "grp__a" in y_pred.columns


def _frame(start: datetime, step: timedelta, values: dict, tz: str | None = None) -> pl.DataFrame:
    """Frame with a ``time`` column of ``len(values[...])`` steps and the given value columns."""
    n = len(next(iter(values.values())))
    times = pl.Series("time", [start + step * i for i in range(n)])
    if tz is not None:
        times = times.dt.replace_time_zone(tz)
    return pl.DataFrame({"time": times, **{k: pl.Series(v, dtype=pl.Float64) for k, v in values.items()}})


_HOUR = timedelta(hours=1)
_DAY = timedelta(days=1)


class TestSegmentProfileForecaster:
    """Tests for SegmentProfileForecaster, one per scenario of its spec."""

    def test_systematic_checks(self):
        """Run all forecaster checks on a local-hour profile with tz-aware hourly data."""
        rng = np.random.default_rng(0)
        y = _frame(datetime(2024, 3, 20), _HOUR, {"load": list(rng.normal(size=24 * 12))}, tz="UTC")
        y_train, y_test = y[: 24 * 9], y[24 * 9 :]
        forecaster = SegmentProfileForecaster(CalendarSegmenter("hour", time_zone="Europe/Paris"))
        forecaster.fit(y_train, forecasting_horizon=3)
        run_checks(forecaster, _yield_yohou_forecaster_checks(forecaster, y_train, None, y_test, None))

    def test_hourly_profile(self):
        """Four weeks where the value at hour h is h forecast each step's hour."""
        y = _frame(datetime(2024, 1, 1), _HOUR, {"value": [float(h % 24) for h in range(24 * 28)]})
        y_pred = SegmentProfileForecaster(CalendarSegmenter("hour")).fit(y, forecasting_horizon=24).predict()
        assert y_pred["value"].to_list() == [float(t.hour) for t in y_pred["time"].to_list()]

    def test_median(self):
        """With method='median' the 03:00 values 1, 2 and 100 forecast 2."""
        values = [0.0] * 72
        values[3], values[27], values[51] = 1.0, 2.0, 100.0
        y = _frame(datetime(2024, 1, 1), _HOUR, {"value": values})
        y_pred = SegmentProfileForecaster(CalendarSegmenter("hour"), method="median").fit(y, forecasting_horizon=4)
        assert y_pred.predict()["value"][3] == 2.0

    def test_missing_values_ignored(self):
        """Null and NaN 03:00 values are skipped, so 4 and 6 forecast 5."""
        values = [0.0] * 96
        values[3], values[27], values[51], values[75] = None, float("nan"), 4.0, 6.0
        y = _frame(datetime(2024, 1, 1), _HOUR, {"value": values})
        y_pred = SegmentProfileForecaster(CalendarSegmenter("hour")).fit(y, forecasting_horizon=4).predict()
        assert y_pred["value"][3] == 5.0

    def test_output_schema(self):
        """The output has vintage_time and time, then each value column in its training dtype."""
        y = pl.DataFrame({
            "time": pl.datetime_range(datetime(2024, 1, 1), datetime(2024, 1, 3), interval="1h", eager=True),
        }).with_columns(pl.col("time").dt.hour().cast(pl.Int64).alias("count"))
        y_pred = SegmentProfileForecaster(CalendarSegmenter("hour")).fit(y, forecasting_horizon=2).predict()
        assert y_pred.columns == ["vintage_time", "time", "count"]
        assert y_pred["count"].dtype == pl.Int64
        assert y_pred["count"].to_list() == [1, 2]

    @pytest.mark.parametrize(("values", "expected"), [([1, 2], 2), ([2, 3], 2)])
    def test_integer_target_rounds_half_to_even(self, values, expected):
        """Integer forecasts are rounded half to even, not truncated: means 1.5 and 2.5 both give 2."""
        y = pl.DataFrame({
            "time": [datetime(2024, 1, 1), datetime(2024, 1, 1, 1)],
            "count": pl.Series(values, dtype=pl.Int64),
        })
        y_pred = SegmentProfileForecaster().fit(y, forecasting_horizon=1).predict()
        assert y_pred["count"].to_list() == [expected]

    def test_peak_stays_at_local_time_across_daylight_saving(self):
        """18:00 Paris is 17:00 UTC before 2024-03-31 01:00 UTC and 16:00 UTC after."""
        times = [datetime(2024, 3, 2) + _HOUR * i for i in range(24 * 29)]
        values = [1.0 if t.hour == 17 else 0.0 for t in times]
        y = _frame(datetime(2024, 3, 2), _HOUR, {"value": values}, tz="UTC")
        assert y["time"][-1] == datetime(2024, 3, 30, 23).replace(tzinfo=y["time"][-1].tzinfo)
        forecaster = SegmentProfileForecaster(CalendarSegmenter("hour", time_zone="Europe/Paris"))
        y_pred = forecaster.fit(y, forecasting_horizon=48).predict()
        ones = y_pred.filter(pl.col("value") == 1.0)["time"].dt.replace_time_zone(None).to_list()
        assert ones == [datetime(2024, 3, 31, 16), datetime(2024, 4, 1, 16)]
        assert y_pred.filter(pl.col("value") != 1.0)["value"].to_list() == [0.0] * 46

    def test_calendar_aware_seasonal_naive(self):
        """n_recent=1 by weekday on daily data repeats the last week."""
        y = _frame(datetime(2024, 1, 1), _DAY, {"value": [float(i) for i in range(21)]})
        forecaster = SegmentProfileForecaster(CalendarSegmenter("day_of_week"), n_recent=1)
        y_pred = forecaster.fit(y, forecasting_horizon=7).predict()
        assert y_pred["value"].to_list() == y["value"][-7:].to_list()

    def test_latest_value_missing(self):
        """n_recent=1 skips a null latest Monday and uses the Monday before it."""
        values = [0.0] * 21
        values[7], values[14] = 7.0, None  # Mondays 2024-01-08 and 2024-01-15
        y = _frame(datetime(2024, 1, 1), _DAY, {"value": values})
        forecaster = SegmentProfileForecaster(CalendarSegmenter("day_of_week"), n_recent=1)
        y_pred = forecaster.fit(y, forecasting_horizon=1).predict()
        assert y_pred["time"][0] == datetime(2024, 1, 22)
        assert y_pred["value"][0] == 7.0

    def test_observe_updates_profile(self):
        """With window_size=48, two days of 0 then one observed day of 10 forecast 5."""
        y = _frame(datetime(2024, 1, 1), _HOUR, {"value": [0.0] * 48 + [10.0] * 24})
        forecaster = SegmentProfileForecaster(CalendarSegmenter("hour"), window_size=48)
        forecaster.fit(y[:48], forecasting_horizon=24)
        forecaster.observe(y[48:])
        assert forecaster.predict()["value"].to_list() == [5.0] * 24

    def test_default_cap_is_training_length(self):
        """Fitted on 100 rows, the memory stays at 100 rows after observing 24 more."""
        y = _frame(datetime(2024, 1, 1), _HOUR, {"value": [0.0] * 100 + [10.0] * 24})
        forecaster = SegmentProfileForecaster().fit(y[:100], forecasting_horizon=1)
        forecaster.observe(y[100:])
        assert forecaster.observation_horizon == 100
        assert forecaster.predict()["value"][0] == pytest.approx(24 * 10.0 / 100)

    def test_window_larger_than_training_data(self):
        """An integer window_size above the training length raises at fit."""
        y = _frame(datetime(2024, 1, 1), _HOUR, {"value": [0.0] * 100})
        with pytest.raises(ValueError, match="window_size=200"):
            SegmentProfileForecaster(window_size=200).fit(y, forecasting_horizon=1)

    def test_short_rewind_keeps_what_it_is_given_then_refills(self):
        """A 10-row rewind forecasts from those rows; observing 24 more keeps all 34."""
        values = [0.0] * 100 + [3.0] * 10 + [9.0] * 24
        y = _frame(datetime(2024, 1, 1), _HOUR, {"value": values})
        forecaster = SegmentProfileForecaster().fit(y[:100], forecasting_horizon=3)
        forecaster.rewind(y[100:110])
        assert forecaster.predict()["value"].to_list() == [3.0] * 3
        forecaster.observe(y[110:])
        assert forecaster.predict()["value"][0] == pytest.approx((10 * 3.0 + 24 * 9.0) / 34)

    def test_outside_the_segment(self):
        """A 03:00 step outside business hours takes the mean of all remembered values."""
        values = [float(h % 24) for h in range(48)]
        y = _frame(datetime(2024, 1, 1), _HOUR, {"value": values})
        forecaster = SegmentProfileForecaster(CalendarSegmenter("hour", values=list(range(8, 19))))
        y_pred = forecaster.fit(y, forecasting_horizon=4).predict()
        assert y_pred["value"][3] == pytest.approx(np.mean(values))

    def test_label_never_seen(self):
        """April, unseen when fitting on January to March, takes the pooled statistic."""
        days = pl.date_range(date(2024, 1, 1), date(2024, 3, 31), eager=True)
        y = pl.DataFrame({"time": days.cast(pl.Datetime("us"))}).with_columns(
            pl.col("time").dt.month().cast(pl.Float64).alias("value")
        )
        forecaster = SegmentProfileForecaster(CalendarSegmenter("month"))
        y_pred = forecaster.fit(y, forecasting_horizon=3).predict()
        assert y_pred["value"].to_list() == pytest.approx([y["value"].mean()] * 3)

    def test_no_segmenter(self):
        """Without a segmenter every forecast is the mean of the finite remembered values."""
        y = _frame(datetime(2024, 1, 1), _HOUR, {"value": [1.0, None, 3.0, float("nan"), 5.0]})
        y_pred = SegmentProfileForecaster().fit(y, forecasting_horizon=2).predict()
        assert y_pred["value"].to_list() == [3.0, 3.0]

    def test_column_with_no_usable_value(self):
        """A column whose memory is all null forecasts null; the other follows its profile."""
        hours = [float(h % 24) for h in range(72)]
        a = hours[:48] + [None] * 24
        y = _frame(datetime(2024, 1, 1), _HOUR, {"a": a, "b": hours})
        forecaster = SegmentProfileForecaster(CalendarSegmenter("hour"), window_size=24)
        forecaster.fit(y[:48], forecasting_horizon=24)
        forecaster.observe(y[48:])
        y_pred = forecaster.predict()
        assert y_pred["a"].null_count() == 24
        assert y_pred["b"].to_list() == [float(h) for h in range(24)]

    @pytest.mark.parametrize(
        "params",
        [{"method": "mode"}, {"n_recent": 0}, {"n_recent": 1.5}, {"window_size": 0}],
    )
    def test_invalid_parameters(self, params):
        """Invalid method, n_recent or window_size raise at fit, naming the parameter."""
        y = _frame(datetime(2024, 1, 1), _HOUR, {"value": [0.0] * 24})
        with pytest.raises(ValueError, match=next(iter(params))):
            SegmentProfileForecaster(**params).fit(y, forecasting_horizon=1)

    def test_infinite_values_ignored(self):
        """Infinite values are skipped like missing ones, so the profile stays finite."""
        values = [0.0] * 72
        values[3], values[27], values[51] = float("inf"), 4.0, 6.0
        y = _frame(datetime(2024, 1, 1), _HOUR, {"value": values})
        y_pred = SegmentProfileForecaster(CalendarSegmenter("hour")).fit(y, forecasting_horizon=4).predict()
        assert y_pred["value"][3] == 5.0
        y_pooled = SegmentProfileForecaster().fit(y, forecasting_horizon=1).predict()
        assert y_pooled["value"][0] == pytest.approx(10.0 / 71)

    def test_n_recent_counts_finite_values(self):
        """n_recent=2 averages the two latest finite values, skipping a missing one between them."""
        values = [0.0] * 28
        values[0], values[7], values[14], values[21] = 1.0, 3.0, None, 5.0  # Mondays
        y = _frame(datetime(2024, 1, 1), _DAY, {"value": values})
        forecaster = SegmentProfileForecaster(CalendarSegmenter("day_of_week"), n_recent=2)
        y_pred = forecaster.fit(y, forecasting_horizon=1).predict()
        assert y_pred["time"][0] == datetime(2024, 1, 29)
        assert y_pred["value"][0] == 4.0

    def test_long_rewind_keeps_at_most_the_cap(self):
        """A rewind longer than window_size keeps only the latest window_size rows."""
        y = _frame(datetime(2024, 1, 1), _HOUR, {"value": [1.0] * 48 + [0.0] * 48 + [10.0] * 24})
        forecaster = SegmentProfileForecaster(window_size=24).fit(y[:48], forecasting_horizon=1)
        forecaster.rewind(y[48:])
        assert forecaster.predict()["value"][0] == 10.0

    def test_short_panel_rewind(self):
        """A panel rewind shorter than the cap keeps each group's rows, then refills."""
        values = [0.0] * 100 + [3.0] * 10 + [9.0] * 24
        y = _frame(datetime(2024, 1, 1), _HOUR, {"a__value": values, "b__value": [2 * v for v in values]})
        forecaster = SegmentProfileForecaster().fit(y[:100], forecasting_horizon=1)
        forecaster.rewind(y[100:110])
        y_pred = forecaster.predict()
        assert (y_pred["a__value"][0], y_pred["b__value"][0]) == (3.0, 6.0)
        forecaster.observe(y[110:])
        expected = (10 * 3.0 + 24 * 9.0) / 34
        assert forecaster.predict()["a__value"][0] == pytest.approx(expected)

    def test_segmenter_with_wrong_label_count(self):
        """A segmenter returning the wrong number of labels raises a clear ValueError at fit."""

        class _DropsOne(CalendarSegmenter):
            def compute_labels(self, time):
                return super().compute_labels(time)[1:]

        y = _frame(datetime(2024, 1, 1), _HOUR, {"value": [0.0] * 24})
        with pytest.raises(ValueError, match="23 labels, expected 24"):
            SegmentProfileForecaster(_DropsOne("hour")).fit(y, forecasting_horizon=1)

    @pytest.fixture()
    def hourly_panel(self) -> pl.DataFrame:
        """Two groups with different hourly profiles."""
        hours = [float(h % 24) for h in range(24 * 7)]
        return _frame(datetime(2024, 1, 1), _HOUR, {"a__value": hours, "b__value": [10.0 * h for h in hours]})

    def test_panel_global(self, hourly_panel):
        """Under the global strategy each group follows its own hourly profile."""
        forecaster = SegmentProfileForecaster(CalendarSegmenter("hour"), panel_strategy="global")
        y_pred = forecaster.fit(hourly_panel, forecasting_horizon=24).predict()
        assert y_pred["a__value"].to_list() == [float(h) for h in range(24)]
        assert y_pred["b__value"].to_list() == [10.0 * h for h in range(24)]

    def test_panel_multivariate_equals_global(self, hourly_panel):
        """The multivariate strategy gives the same forecasts as the global one."""
        preds = [
            SegmentProfileForecaster(CalendarSegmenter("hour"), panel_strategy=strategy)
            .fit(hourly_panel, forecasting_horizon=24)
            .predict()
            for strategy in ("global", "multivariate")
        ]
        polars.testing.assert_frame_equal(preds[0], preds[1])

    def test_holiday_forecast_differs_from_ordinary_weekday(self):
        """A weekday holiday follows the holiday profile, an ordinary weekday the weekday profile."""
        holidays = pl.DataFrame({"date": [date(2024, 1, 10), date(2024, 2, 7)]})  # Wednesdays
        days = [datetime(2024, 1, 1) + _DAY * i for i in range(35)]  # Monday 2024-01-01 to Sunday 2024-02-04
        values = [10.0 if d.weekday() >= 5 or d.date() == date(2024, 1, 10) else 100.0 for d in days]
        y = _frame(datetime(2024, 1, 1), _DAY, {"value": values})
        segmenter = UnionSegmenter([
            ("hour", CalendarSegmenter("hour")),
            ("weekend", CalendarSegmenter("day_of_week", values=[6, 7])),
            ("holiday", HolidaySegmenter(holidays=holidays)),
        ])
        y_pred = SegmentProfileForecaster(segmenter).fit(y, forecasting_horizon=7).predict()
        by_day = dict(zip(y_pred["time"].dt.date().to_list(), y_pred["value"].to_list(), strict=True))
        assert by_day[date(2024, 2, 7)] == 10.0  # Wednesday holiday
        assert by_day[date(2024, 2, 6)] == 100.0  # ordinary Tuesday
        assert by_day[date(2024, 2, 10)] == 10.0  # Saturday

    def test_time_zone_on_naive_training_data(self):
        """A local-hour segmenter on naive timestamps makes fit raise."""
        y = _frame(datetime(2024, 1, 1), _HOUR, {"value": [0.0] * 24})
        with pytest.raises(ValueError, match="timezone-aware"):
            SegmentProfileForecaster(CalendarSegmenter("hour", time_zone="Europe/Paris")).fit(y, forecasting_horizon=1)

    def test_decomposition_pipeline_component(self):
        """As the seasonal component after a linear trend, it recovers trend plus hourly profile."""
        t = np.arange(24 * 14, dtype=float)
        truth = 0.05 * t + np.where((t % 24) == 18, 5.0, 0.0)
        y = _frame(datetime(2024, 1, 1), _HOUR, {"value": list(truth)})
        pipeline = DecompositionPipeline([
            ("trend", PolynomialTrendForecaster(degree=1)),
            ("seasonality", SegmentProfileForecaster(CalendarSegmenter("hour"))),
        ])
        pipeline.fit(y[: 24 * 13], forecasting_horizon=24)
        y_pred = pipeline.predict()
        np.testing.assert_allclose(y_pred["value"].to_numpy(), truth[24 * 13 :], atol=0.5)
        y_next = pipeline.observe_predict(y[24 * 13 : 24 * 13 + 1])
        assert y_next["value"].null_count() == 0
