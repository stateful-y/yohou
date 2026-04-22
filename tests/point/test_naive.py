from datetime import datetime

import numpy as np
import polars as pl
import polars.testing
import pytest
from sklearn.base import clone
from sklearn.model_selection import train_test_split

from conftest import run_checks
from yohou.point import MeanSeasonalNaive, SeasonalNaive
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
    X = pl.DataFrame({
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
    return y, X


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
        y, X = y_X_factory(length=200, seed=42)
        y_train, y_test = y[:160], y[160:]
        X_train, X_test = (X[:160], X[160:]) if X is not None else (None, None)

        forecaster_fitted = clone(forecaster)
        forecaster_fitted.fit(y_train, X_train, forecasting_horizon=3)

        run_checks(
            forecaster_fitted,
            _yield_yohou_forecaster_checks(forecaster_fitted, y_train, X_train, y_test, X_test),
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
        y, X = naive_data
        y_train, y_test, X_train, X_test = train_test_split(y, X, test_size=0.2, shuffle=False)
        forecaster = SeasonalNaive(seasonality=seasonality)

        forecaster.fit(y=y_train, X=X_train, forecasting_horizon=fit_forecasting_horizon)

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
    """Tests for SeasonalNaive with X=None throughout the lifecycle."""

    def test_fit_predict_without_exogenous(self, naive_data):
        """SeasonalNaive should work without exogenous features (ignores_exogenous=True)."""
        y, _ = naive_data
        forecaster = SeasonalNaive(seasonality=3)
        forecaster.fit(y[:17], X=None, forecasting_horizon=3)
        y_pred = forecaster.predict(forecasting_horizon=3)

        assert isinstance(y_pred, pl.DataFrame)
        assert "time" in y_pred.columns
        assert len(y_pred) == 3

    def test_observe_predict_without_exogenous(self, naive_data):
        """SeasonalNaive observe_predict should work without exogenous features."""
        y, _ = naive_data
        forecaster = SeasonalNaive(seasonality=3)
        forecaster.fit(y[:17], X=None, forecasting_horizon=3)
        y_pred = forecaster.observe_predict(y[17:19], X=None)

        assert isinstance(y_pred, pl.DataFrame)
        assert "time" in y_pred.columns

    def test_rewind_without_exogenous(self, naive_data):
        """SeasonalNaive rewind should work without exogenous features."""
        y, _ = naive_data
        forecaster = SeasonalNaive(seasonality=3)
        forecaster.fit(y[:17], X=None, forecasting_horizon=3)
        forecaster.observe(y[17:19], X=None)
        forecaster.rewind(y[:17], X=None)
        y_pred = forecaster.predict(forecasting_horizon=3)

        assert isinstance(y_pred, pl.DataFrame)
        assert len(y_pred) == 3

    def test_ignores_exogenous_tag(self):
        """SeasonalNaive should have ignores_exogenous=True tag."""
        forecaster = SeasonalNaive(seasonality=3)
        tags = forecaster.__sklearn_tags__()
        assert tags.forecaster_tags.ignores_exogenous is True


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
    X = pl.DataFrame({
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
    return y, X


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
        y, X = y_X_factory(length=200, seed=42)
        y_train, y_test = y[:160], y[160:]
        X_train, X_test = (X[:160], X[160:]) if X is not None else (None, None)

        forecaster_fitted = clone(forecaster)
        forecaster_fitted.fit(y_train, X_train, forecasting_horizon=3)

        run_checks(
            forecaster_fitted,
            _yield_yohou_forecaster_checks(forecaster_fitted, y_train, X_train, y_test, X_test),
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
        y, X = mean_naive_data
        y_train, y_test, X_train, X_test = train_test_split(y, X, test_size=0.2, shuffle=False)
        forecaster = MeanSeasonalNaive(seasonality=seasonality, n_seasons=n_seasons)

        forecaster.fit(y=y_train, X=X_train, forecasting_horizon=fit_fh)

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
        y, X = mean_naive_data
        y_train, _, X_train, _ = train_test_split(y, X, test_size=0.2, shuffle=False)

        for seasonality in [1, 3, 7]:
            naive = SeasonalNaive(seasonality=seasonality)
            naive.fit(y=y_train, X=X_train, forecasting_horizon=5)
            y_naive = naive.predict(forecasting_horizon=5)

            mean_naive = MeanSeasonalNaive(seasonality=seasonality, n_seasons=1)
            mean_naive.fit(y=y_train, X=X_train, forecasting_horizon=5)
            y_mean = mean_naive.predict(forecasting_horizon=5)

            pl.testing.assert_frame_equal(y_mean, y_naive)


class TestMeanSeasonalNaiveWithoutExogenous:
    """Tests for MeanSeasonalNaive with X=None throughout the lifecycle."""

    def test_fit_predict_without_exogenous(self, mean_naive_data):
        """MeanSeasonalNaive should work without exogenous features."""
        y, _ = mean_naive_data
        forecaster = MeanSeasonalNaive(seasonality=3, n_seasons=2)
        forecaster.fit(y[:40], X=None, forecasting_horizon=3)
        y_pred = forecaster.predict(forecasting_horizon=3)

        assert isinstance(y_pred, pl.DataFrame)
        assert "time" in y_pred.columns
        assert len(y_pred) == 3

    def test_observe_predict_without_exogenous(self, mean_naive_data):
        """MeanSeasonalNaive observe_predict should work without exogenous features."""
        y, _ = mean_naive_data
        forecaster = MeanSeasonalNaive(seasonality=3, n_seasons=2)
        forecaster.fit(y[:40], X=None, forecasting_horizon=3)
        y_pred = forecaster.observe_predict(y[40:42], X=None)

        assert isinstance(y_pred, pl.DataFrame)
        assert "time" in y_pred.columns

    def test_rewind_without_exogenous(self, mean_naive_data):
        """MeanSeasonalNaive rewind should work without exogenous features."""
        y, _ = mean_naive_data
        forecaster = MeanSeasonalNaive(seasonality=3, n_seasons=2)
        forecaster.fit(y[:40], X=None, forecasting_horizon=3)
        forecaster.observe(y[40:42], X=None)
        forecaster.rewind(y[:40], X=None)
        y_pred = forecaster.predict(forecasting_horizon=3)

        assert isinstance(y_pred, pl.DataFrame)
        assert len(y_pred) == 3

    def test_ignores_exogenous_tag(self):
        """MeanSeasonalNaive should have ignores_exogenous=True tag."""
        forecaster = MeanSeasonalNaive(seasonality=3, n_seasons=2)
        tags = forecaster.__sklearn_tags__()
        assert tags.forecaster_tags.ignores_exogenous is True


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
