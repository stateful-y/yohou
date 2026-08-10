"""Tests for SplitConformalForecaster."""

from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError

from conftest import run_checks
from yohou.interval import DistanceSimilarity, SplitConformalForecaster
from yohou.interval.similarity import SeasonalSimilarity
from yohou.metrics import AbsoluteResidual, Residual
from yohou.point import SeasonalNaive
from yohou.testing import _yield_yohou_forecaster_checks


@pytest.fixture
def conformal_data():
    """Create data for conformal forecaster tests."""
    length = 250
    time = pl.datetime_range(
        start=datetime(2021, 1, 1),
        end=datetime(2021, 1, 1) + timedelta(seconds=length - 1),
        interval="1s",
        eager=True,
    )
    import numpy as np

    rng = np.random.default_rng(42)
    values = np.cumsum(rng.standard_normal(length)) + 100
    y = pl.DataFrame({"time": time, "value": values})
    return y


class TestSplitConformalInit:
    """Test SplitConformalForecaster initialization."""

    def test_default_init(self):
        """Test default initialization."""
        scf = SplitConformalForecaster()
        assert isinstance(scf.point_forecaster, SeasonalNaive)
        assert scf.calibration_size == 100
        assert isinstance(scf.conformity_scorer, Residual)
        assert scf.similarity is None

    def test_custom_init(self):
        """Test custom initialization."""
        point = SeasonalNaive(seasonality=12)
        scorer = Residual()
        scf = SplitConformalForecaster(
            point_forecaster=point,
            calibration_size=50,
            conformity_scorer=scorer,
        )
        assert scf.calibration_size == 50
        assert scf.point_forecaster is point

    def test_clone(self):
        """Test that SplitConformalForecaster can be cloned."""
        scf = SplitConformalForecaster(calibration_size=50)
        scf_clone = clone(scf)
        assert scf_clone is not scf
        assert scf_clone.calibration_size == 50

    def test_tags(self):
        """Test that forecaster_type tag is both."""
        scf = SplitConformalForecaster()
        tags = scf.__sklearn_tags__()
        assert tags.forecaster_tags is not None
        assert tags.forecaster_tags.forecaster_type == frozenset({"point", "interval"})


class TestSplitConformalFitPredict:
    """Test fit and predict functionality."""

    def test_fit(self, conformal_data):
        """Test basic fit."""
        scf = SplitConformalForecaster(calibration_size=50)
        scf.fit(conformal_data, forecasting_horizon=1)
        assert hasattr(scf, "point_forecaster_")
        assert hasattr(scf, "conformity_scorers_")
        assert hasattr(scf, "conformity_scores_")

    def test_predict_after_fit(self, conformal_data):
        """Test predict after fit."""
        scf = SplitConformalForecaster(calibration_size=50)
        scf.fit(conformal_data, forecasting_horizon=1)
        y_pred = scf.predict()
        assert isinstance(y_pred, pl.DataFrame)
        assert len(y_pred) == 1

    def test_predict_not_fitted(self):
        """Test predict raises error when not fitted."""
        scf = SplitConformalForecaster()
        with pytest.raises(NotFittedError, match="fitted"):
            scf.predict()

    def test_predict_interval(self, conformal_data):
        """Test predict_interval produces interval bounds."""
        scf = SplitConformalForecaster(calibration_size=50)
        scf.fit(conformal_data, forecasting_horizon=1, coverage_rates=[0.9])
        y_pred_interval = scf.predict_interval(coverage_rates=[0.9])
        assert isinstance(y_pred_interval, pl.DataFrame)
        assert len(y_pred_interval) >= 1

    def test_predict_interval_multiple_coverage(self, conformal_data):
        """Test predict_interval with multiple coverage rates."""
        scf = SplitConformalForecaster(calibration_size=50)
        scf.fit(conformal_data, forecasting_horizon=1, coverage_rates=[0.8, 0.95])
        y_pred_interval = scf.predict_interval(coverage_rates=[0.8, 0.95])
        assert isinstance(y_pred_interval, pl.DataFrame)
        assert "time" in y_pred_interval.columns
        # Verify no duplicate columns
        assert len(y_pred_interval.columns) == len(set(y_pred_interval.columns))
        # Verify interval columns for both coverage rates
        interval_cols = [c for c in y_pred_interval.columns if c != "time"]
        assert any("_lower_0.8" in c for c in interval_cols)
        assert any("_upper_0.8" in c for c in interval_cols)
        assert any("_lower_0.95" in c for c in interval_cols)
        assert any("_upper_0.95" in c for c in interval_cols)

    @pytest.mark.slow
    def test_predict_multi_step(self, conformal_data):
        """Test prediction with multi-step horizon."""
        scf = SplitConformalForecaster(calibration_size=50)
        scf.fit(conformal_data, forecasting_horizon=3)
        y_pred = scf.predict(forecasting_horizon=3)
        assert isinstance(y_pred, pl.DataFrame)
        assert len(y_pred) == 3

    def test_predict_interval_zero_coverage_rate(self, conformal_data):
        """Test coverage_rate=0 produces zero-width intervals (lower == upper)."""
        scf = SplitConformalForecaster(calibration_size=50)
        scf.fit(conformal_data, forecasting_horizon=1, coverage_rates=[0.0])
        y_pred = scf.predict_interval(coverage_rates=[0.0])
        assert isinstance(y_pred, pl.DataFrame)
        lower_cols = [c for c in y_pred.columns if "_lower_0.0" in c]
        upper_cols = [c for c in y_pred.columns if "_upper_0.0" in c]
        assert len(lower_cols) > 0
        assert len(upper_cols) > 0
        for lower_col, upper_col in zip(lower_cols, upper_cols, strict=True):
            assert y_pred[lower_col].equals(y_pred[upper_col]), (
                f"Expected lower == upper for coverage_rate=0, "
                f"got lower={y_pred[lower_col].to_list()}, upper={y_pred[upper_col].to_list()}"
            )


class TestSplitConformalParameterValidation:
    """Test parameter validation."""

    def test_invalid_calibration_size_zero(self):
        """Test that calibration_size=0 raises error."""
        scf = SplitConformalForecaster(calibration_size=0)
        with pytest.raises(ValueError, match="calibration_size"):
            y = pl.DataFrame({
                "time": pl.datetime_range(
                    start=datetime(2021, 1, 1),
                    end=datetime(2021, 1, 1) + timedelta(seconds=99),
                    interval="1s",
                    eager=True,
                ),
                "value": list(range(100)),
            })
            scf.fit(y, forecasting_horizon=1)

    def test_invalid_point_type(self):
        """Test that non-forecaster point raises error."""
        with pytest.raises((TypeError, ValueError)):
            scf = SplitConformalForecaster(point_forecaster="not_a_forecaster")
            y = pl.DataFrame({
                "time": pl.datetime_range(
                    start=datetime(2021, 1, 1),
                    end=datetime(2021, 1, 1) + timedelta(seconds=249),
                    interval="1s",
                    eager=True,
                ),
                "value": list(range(250)),
            })
            scf.fit(y, forecasting_horizon=1)


class TestSplitConformalObserveRewind:
    """Test observe and rewind functionality."""

    def test_observe(self, conformal_data):
        """Test that observe updates the wrapped forecaster without error."""
        scf = SplitConformalForecaster(calibration_size=50)
        scf.fit(conformal_data[:200], forecasting_horizon=1)

        y_update = conformal_data[200:210]
        scf.observe(y_update)

        y_pred = scf.predict()
        assert isinstance(y_pred, pl.DataFrame)
        assert len(y_pred) == 1

    def test_observe_then_predict_interval(self, conformal_data):
        """Test predict_interval works after observe."""
        scf = SplitConformalForecaster(calibration_size=50)
        scf.fit(conformal_data[:200], forecasting_horizon=1, coverage_rates=[0.9])

        y_update = conformal_data[200:210]
        scf.observe(y_update)

        y_pred = scf.predict_interval(coverage_rates=[0.9])
        assert isinstance(y_pred, pl.DataFrame)
        assert len(y_pred) >= 1

    def test_observe_predict_interval(self, conformal_data):
        """Test observe_predict_interval composite method."""
        scf = SplitConformalForecaster(calibration_size=50)
        scf.fit(conformal_data[:200], forecasting_horizon=1, coverage_rates=[0.9])

        y_update = conformal_data[200:210]
        y_pred = scf.observe_predict_interval(
            y=y_update,
            coverage_rates=[0.9],
        )
        assert isinstance(y_pred, pl.DataFrame)
        assert len(y_pred) >= 1

    def test_observe_predict_interval_multi_step(self, conformal_data):
        """Test observe_predict_interval with forecasting_horizon > 1 produces sorted vintages."""
        scf = SplitConformalForecaster(calibration_size=50)
        fh = 3
        scf.fit(conformal_data[:200], forecasting_horizon=fh, coverage_rates=[0.9])

        y_test = conformal_data[200:210]
        y_pred = scf.observe_predict_interval(
            y=y_test,
            forecasting_horizon=fh,
            coverage_rates=[0.9],
        )
        assert isinstance(y_pred, pl.DataFrame)
        assert "vintage_time" in y_pred.columns
        assert len(y_pred) > fh

        # Each vintage must have sorted time (regression test for stale point_forecaster_ state)
        for vt in y_pred["vintage_time"].unique():
            vintage = y_pred.filter(pl.col("vintage_time") == vt)
            assert vintage["time"].is_sorted(), f"'time' within vintage_time={vt} is not sorted"

    def test_rewind(self, conformal_data):
        """Test that rewind delegates to the wrapped forecaster."""
        scf = SplitConformalForecaster(calibration_size=50)
        scf.fit(conformal_data[:200], forecasting_horizon=1)

        scf.rewind(conformal_data[190:200])

        y_pred = scf.predict()
        assert isinstance(y_pred, pl.DataFrame)
        assert len(y_pred) == 1

    def test_observe_syncs_observed_time(self, conformal_data):
        """Test that observe updates observed_time_ on the outer forecaster."""
        scf = SplitConformalForecaster(calibration_size=50)
        scf.fit(conformal_data[:200], forecasting_horizon=1)

        fit_time = scf.observed_time_

        y_update = conformal_data[200:210]
        scf.observe(y_update)

        assert scf.observed_time_ == scf.point_forecaster_.observed_time_
        assert scf.observed_time_ != fit_time
        assert scf.observed_time_ == y_update["time"][-1]

    def test_rewind_syncs_observed_time(self, conformal_data):
        """Test that rewind updates observed_time_ on the outer forecaster."""
        scf = SplitConformalForecaster(calibration_size=50)
        scf.fit(conformal_data[:200], forecasting_horizon=1)

        y_update = conformal_data[200:210]
        scf.observe(y_update)

        scf.rewind(conformal_data[190:200])

        assert scf.observed_time_ == scf.point_forecaster_.observed_time_
        assert scf.observed_time_ == conformal_data[199]["time"][0]

    def test_observe_syncs_observed_time_panel(self, y_X_factory):
        """Test that observe updates observed_time_ on panel data."""
        y, _ = y_X_factory(length=250, n_targets=1, n_features=0, seed=42, panel=True, n_groups=2)

        scf = SplitConformalForecaster(calibration_size=50)
        scf.fit(y[:200], forecasting_horizon=1)

        fit_time = scf.observed_time_

        y_update = y[200:210]
        scf.observe(y_update)

        assert scf.observed_time_ == scf.point_forecaster_.observed_time_
        assert scf.observed_time_ != fit_time
        expected_time = y_update["time"][-1]
        for group_time in scf.observed_time_.values():
            assert group_time == expected_time

    def test_rewind_syncs_observed_time_panel(self, y_X_factory):
        """Test that rewind updates observed_time_ on panel data."""
        y, _ = y_X_factory(length=250, n_targets=1, n_features=0, seed=42, panel=True, n_groups=2)

        scf = SplitConformalForecaster(calibration_size=50)
        scf.fit(y[:200], forecasting_horizon=1)

        y_update = y[200:210]
        scf.observe(y_update)

        scf.rewind(y[190:200])

        assert scf.observed_time_ == scf.point_forecaster_.observed_time_
        expected_time = y[199]["time"][0]
        for group_time in scf.observed_time_.values():
            assert group_time == expected_time

    def test_observe_updates_conformity_panel(self, y_X_factory):
        """observe on panel data must run the conformity/similarity update.

        Regression test: the panel branch dispatched through
        ``BasePanelForecaster._observe_panel`` previously skipped the
        conformity/similarity score update that the standard branch runs, so
        ``conformity_scores_`` stayed frozen at its fit-time size and prediction
        intervals went silently stale after the first panel ``observe``.
        """
        y, _ = y_X_factory(length=250, n_targets=1, n_features=0, seed=42, panel=True, n_groups=2)

        scf = SplitConformalForecaster(calibration_size=50, similarity=DistanceSimilarity())
        scf.fit(y[:200], forecasting_horizon=1, coverage_rates=[0.9])

        n_before = scf.conformity_scores_.height
        scf.observe(y[200:210])
        n_after_observe = scf.conformity_scores_.height
        assert n_after_observe > n_before, "panel observe did not append new conformity scores"

        scf.rewind(y[200:210])
        assert scf.conformity_scores_.height == n_before, "panel rewind did not roll back conformity scores"

    def test_observe_not_fitted(self):
        """Test observe raises error when not fitted."""
        scf = SplitConformalForecaster()
        with pytest.raises(NotFittedError, match="fitted"):
            scf.observe(pl.DataFrame({"time": [], "value": []}))


class TestSplitConformalSimilarity:
    """Test SplitConformalForecaster with DistanceSimilarity."""

    def test_fit_predict_interval_with_similarity(self, conformal_data):
        """Test that similarity-weighted intervals differ from unweighted."""
        horizon = 5

        scf_sim = SplitConformalForecaster(
            point_forecaster=SeasonalNaive(seasonality=1),
            calibration_size=50,
            conformity_scorer=AbsoluteResidual(),
            similarity=DistanceSimilarity(metric="euclidean"),
        )
        scf_sim.fit(conformal_data, forecasting_horizon=horizon, coverage_rates=[0.9])

        scf_no_sim = SplitConformalForecaster(
            point_forecaster=SeasonalNaive(seasonality=1),
            calibration_size=50,
            conformity_scorer=AbsoluteResidual(),
            similarity=None,
        )
        scf_no_sim.fit(conformal_data, forecasting_horizon=horizon, coverage_rates=[0.9])

        y_pred_sim = scf_sim.predict_interval(coverage_rates=[0.9])
        y_pred_no_sim = scf_no_sim.predict_interval(coverage_rates=[0.9])

        assert isinstance(y_pred_sim, pl.DataFrame)
        assert "time" in y_pred_sim.columns
        assert len(y_pred_sim) == horizon

        # Verify interval columns exist
        assert "value_lower_0.9" in y_pred_sim.columns
        assert "value_upper_0.9" in y_pred_sim.columns

        # Verify intervals differ (similarity should affect quantile estimation)
        lower_diff = np.abs(y_pred_sim["value_lower_0.9"].to_numpy() - y_pred_no_sim["value_lower_0.9"].to_numpy())
        upper_diff = np.abs(y_pred_sim["value_upper_0.9"].to_numpy() - y_pred_no_sim["value_upper_0.9"].to_numpy())
        assert np.sum(lower_diff > 1e-6) > 0 or np.sum(upper_diff > 1e-6) > 0

    def test_fit_stores_similarity_attributes(self, conformal_data):
        """Test that fit stores similarities_ when similarity is set."""
        horizon = 3

        scf = SplitConformalForecaster(
            point_forecaster=SeasonalNaive(seasonality=1),
            calibration_size=50,
            conformity_scorer=AbsoluteResidual(),
            similarity=DistanceSimilarity(metric="euclidean"),
        )
        scf.fit(conformal_data, forecasting_horizon=horizon)

        assert hasattr(scf, "similarities_")
        assert len(scf.similarities_) == horizon

    def test_fit_predict_interval_with_residual_scorer(self, conformal_data):
        """Test similarity with asymmetric (Residual) scorer."""
        horizon = 3

        scf = SplitConformalForecaster(
            point_forecaster=SeasonalNaive(seasonality=1),
            calibration_size=50,
            conformity_scorer=Residual(),
            similarity=DistanceSimilarity(metric="euclidean"),
        )
        scf.fit(conformal_data, forecasting_horizon=horizon, coverage_rates=[0.9])

        y_pred = scf.predict_interval(coverage_rates=[0.9])
        assert isinstance(y_pred, pl.DataFrame)
        assert len(y_pred) == horizon
        assert "value_lower_0.9" in y_pred.columns
        assert "value_upper_0.9" in y_pred.columns

    def test_no_similarity_attributes_when_none(self, conformal_data):
        """Test that similarities_ is not set when similarity is None."""
        scf = SplitConformalForecaster(
            point_forecaster=SeasonalNaive(seasonality=1),
            calibration_size=50,
        )
        scf.fit(conformal_data, forecasting_horizon=1)
        assert not hasattr(scf, "similarities_")

    def test_predict_interval_after_observe_with_similarity(self, conformal_data):
        """Test predict_interval after observe with similarity (vintage_time path)."""
        scf = SplitConformalForecaster(
            point_forecaster=SeasonalNaive(seasonality=1),
            calibration_size=50,
            conformity_scorer=AbsoluteResidual(),
            similarity=DistanceSimilarity(metric="euclidean"),
        )
        scf.fit(conformal_data[:200], forecasting_horizon=1, coverage_rates=[0.9])

        # Observe new data (this causes point_forecaster_ to track vintage_time)
        scf.observe(conformal_data[200:210])

        # predict_interval should handle vintage_time column from inner predict
        intervals = scf.predict_interval(coverage_rates=[0.9])
        assert isinstance(intervals, pl.DataFrame)
        assert len(intervals) >= 1
        assert "value_lower_0.9" in intervals.columns
        assert "value_upper_0.9" in intervals.columns

    def test_predict_interval_with_gamma_scorer(self, conformal_data):
        """Test that predict_interval uses multiplicative tag from GammaResidual."""
        from yohou.metrics import GammaResidual

        scf = SplitConformalForecaster(
            point_forecaster=SeasonalNaive(seasonality=1),
            calibration_size=50,
            conformity_scorer=GammaResidual(),
            similarity=DistanceSimilarity(metric="euclidean"),
        )
        scf.fit(conformal_data[:200], forecasting_horizon=1, coverage_rates=[0.9])

        y_pred = scf.predict_interval(coverage_rates=[0.9])
        assert isinstance(y_pred, pl.DataFrame)
        assert "value_lower_0.9" in y_pred.columns
        assert "value_upper_0.9" in y_pred.columns


class TestSplitConformalSystematicChecks:
    """Systematic checks for SplitConformalForecaster using the check generator."""

    @pytest.mark.slow
    def test_split_conformal_systematic_checks(self, y_X_factory):
        """Run all standard forecaster checks on SplitConformalForecaster.

        ``X_forecast`` is supplied because the generator gates a block of checks on
        its presence. Passing ``None`` yields a strictly smaller suite and reports
        nothing, so the run would pass while the checks covering this forecaster's
        heaviest input channel never executed.
        """
        y, _, _, X_forecast = y_X_factory(
            length=200,
            n_targets=1,
            n_features=0,
            seed=42,
            n_forecast_features=2,
            return_exogenous=True,
        )
        y_train, y_test = y[:180], y[180:]

        forecaster = SplitConformalForecaster(
            point_forecaster=SeasonalNaive(seasonality=7),
            calibration_size=50,
        )
        # Fit with X_forecast: the step-column checks require a forecaster that
        # derived step columns at fit, as check_observe_auto_rederives_step_columns
        # documents in its parameter description.
        forecaster.fit(y_train, forecasting_horizon=5, X_forecast=X_forecast)

        run_checks(
            forecaster,
            _yield_yohou_forecaster_checks(
                forecaster,
                y_train,
                None,
                y_test,
                None,
                X_forecast_train=X_forecast,
                X_forecast_test=X_forecast,
            ),
            expected_failures=set(),
        )


class TestSplitConformalObserveRewindSimilarity:
    """Tests for observe/rewind forwarding to similarity and conformity score updates."""

    @pytest.fixture
    def scf_with_similarity(self):
        """Create fitted SplitConformalForecaster with DistanceSimilarity."""
        n = 250
        dates = [datetime(2020, 1, 1) + timedelta(days=i) for i in range(n)]
        np.random.seed(42)
        values = [10.0 + 5.0 * np.sin(2 * np.pi * i / 7) + np.random.normal(0, 0.5) for i in range(n)]
        y = pl.DataFrame({"time": dates, "value": values})
        y_train = y[:200]
        y_test = y[200:]

        scf = SplitConformalForecaster(
            point_forecaster=SeasonalNaive(seasonality=7),
            calibration_size=50,
            conformity_scorer=AbsoluteResidual(),
            similarity=DistanceSimilarity(metric="euclidean"),
        )
        scf.fit(y_train, forecasting_horizon=1, coverage_rates=[0.9])
        return scf, y_train, y_test

    def test_observe_updates_similarity_state(self, scf_with_similarity):
        """Test that observe() forwards to similarity.observe() and grows state."""
        scf, y_train, y_test = scf_with_similarity
        sim_step = scf.similarities_["step_1"]
        n_before = len(sim_step._X_observed)

        scf.observe(y_test[:1])

        assert len(sim_step._X_observed) == n_before + 1

    def test_observe_updates_conformity_scores(self, scf_with_similarity):
        """Test that observe() appends new conformity scores."""
        scf, y_train, y_test = scf_with_similarity
        n_scores_before = len(scf.conformity_scores_)

        scf.observe(y_test[:1])

        assert len(scf.conformity_scores_) == n_scores_before + 1

    def test_rewind_restores_similarity_state(self, scf_with_similarity):
        """Test that observe() then rewind() restores original similarity state."""
        scf, y_train, y_test = scf_with_similarity
        sim_step = scf.similarities_["step_1"]
        n_before = len(sim_step._X_observed)
        x_observed_before = sim_step._X_observed.clone()

        # Observe new data one row at a time (standard streaming pattern)
        for i in range(10):
            scf.observe(y_test[i : i + 1])

        assert len(sim_step._X_observed) == n_before + 10

        # Rewind using training data (same pattern as existing rewind tests)
        scf.rewind(y_train[190:200])

        assert len(sim_step._X_observed) == n_before
        assert sim_step._X_observed.equals(x_observed_before)

    def test_rewind_restores_conformity_scores(self, scf_with_similarity):
        """Test that observe() then rewind() restores original conformity scores count."""
        scf, y_train, y_test = scf_with_similarity
        n_scores_before = len(scf.conformity_scores_)

        for i in range(10):
            scf.observe(y_test[i : i + 1])
        assert len(scf.conformity_scores_) == n_scores_before + 10

        scf.rewind(y_train[190:200])
        assert len(scf.conformity_scores_) == n_scores_before

    def test_observe_without_similarity_unchanged(self):
        """Test that observe() without similarity does not fail."""
        n = 200
        dates = [datetime(2020, 1, 1) + timedelta(days=i) for i in range(n)]
        np.random.seed(42)
        values = [10.0 + 5.0 * np.sin(2 * np.pi * i / 7) for i in range(n)]
        y = pl.DataFrame({"time": dates, "value": values})
        y_train = y[:180]
        y_test = y[180:]

        scf = SplitConformalForecaster(
            point_forecaster=SeasonalNaive(seasonality=7),
            calibration_size=50,
            conformity_scorer=AbsoluteResidual(),
        )
        scf.fit(y_train, forecasting_horizon=1, coverage_rates=[0.9])
        n_scores = len(scf.conformity_scores_)

        scf.observe(y_test[:1])
        # No similarity, so conformity scores should NOT be updated
        assert len(scf.conformity_scores_) == n_scores

    def test_observe_with_temporal_similarity(self):
        """Test that observe() works with SeasonalSimilarity."""
        n = 200
        dates = [datetime(2020, 1, 1) + timedelta(days=i) for i in range(n)]
        np.random.seed(42)
        values = [10.0 + 5.0 * np.sin(2 * np.pi * i / 7) + np.random.normal(0, 0.5) for i in range(n)]
        y = pl.DataFrame({"time": dates, "value": values})
        y_train = y[:180]
        y_test = y[180:]

        scf = SplitConformalForecaster(
            point_forecaster=SeasonalNaive(seasonality=7),
            calibration_size=50,
            conformity_scorer=AbsoluteResidual(),
            similarity=SeasonalSimilarity(seasonality=[7.0]),
        )
        scf.fit(y_train, forecasting_horizon=1, coverage_rates=[0.9])

        sim_step = scf.similarities_["step_1"]
        n_features_before = sim_step._features_observed.shape[0]

        scf.observe(y_test[:1])
        assert sim_step._features_observed.shape[0] == n_features_before + 1

    def test_multi_step_observe_rewind_symmetry(self, scf_with_similarity):
        """Test that multiple observe() then rewind() restores state."""
        scf, y_train, y_test = scf_with_similarity
        sim_step = scf.similarities_["step_1"]
        n_sim_before = len(sim_step._X_observed)
        n_scores_before = len(scf.conformity_scores_)

        # Observe 3 times
        for i in range(3):
            scf.observe(y_test[i : i + 1])

        assert len(sim_step._X_observed) == n_sim_before + 3
        assert len(scf.conformity_scores_) == n_scores_before + 3

        # Rewind using training data rows (at least observation_horizon rows)
        scf.rewind(y_train[190:200])

        assert len(sim_step._X_observed) == n_sim_before
        assert len(scf.conformity_scores_) == n_scores_before

    def test_rewind_without_prior_observe(self, scf_with_similarity):
        """Test that rewind with similarity when no data was observed is a no-op."""
        scf, y_train, _y_test = scf_with_similarity
        sim_step = scf.similarities_["step_1"]
        n_before = len(sim_step._X_observed)
        n_scores_before = len(scf.conformity_scores_)

        # Rewind without any prior observe: n_post_fit_removed == 0
        scf.rewind(y_train[190:200])

        assert len(sim_step._X_observed) == n_before
        assert len(scf.conformity_scores_) == n_scores_before

    def test_panel_observe_rewind_with_similarity(self):
        """Panel observe/rewind dispatches through the panel + similarity branch.

        With ``groups_`` set and ``similarity`` configured, observe and rewind
        route through ``BasePanelForecaster._observe_panel``/``_rewind_panel``;
        the round-trip must complete and rewind must restore the pre-observe
        intervals exactly. This panel + similarity path was previously
        unexercised (the panel observe tests use ``similarity=None``).
        """
        n = 250
        dates = [datetime(2020, 1, 1) + timedelta(days=i) for i in range(n)]
        np.random.seed(42)
        v1 = [10.0 + 5.0 * np.sin(2 * np.pi * i / 7) + np.random.normal(0, 0.5) for i in range(n)]
        v2 = [20.0 + 3.0 * np.sin(2 * np.pi * i / 7) + np.random.normal(0, 0.5) for i in range(n)]
        y = pl.DataFrame({"time": dates, "g1__value": v1, "g2__value": v2})
        y_train = y[:200]
        y_test = y[200:]

        scf = SplitConformalForecaster(
            point_forecaster=SeasonalNaive(seasonality=7),
            calibration_size=50,
            conformity_scorer=AbsoluteResidual(),
            similarity=DistanceSimilarity(metric="euclidean"),
        )
        scf.fit(y_train, forecasting_horizon=1, coverage_rates=[0.9])
        assert scf.groups_ == ["g1", "g2"]

        iv_before = scf.predict_interval(forecasting_horizon=1, coverage_rates=[0.9])

        for i in range(10):
            scf.observe(y_test[i : i + 1])
        scf.rewind(y_train[190:200])

        iv_after = scf.predict_interval(forecasting_horizon=1, coverage_rates=[0.9])

        for group in ["g1", "g2"]:
            lower = f"{group}__value_lower_0.9"
            upper = f"{group}__value_upper_0.9"
            assert lower in iv_after.columns
            assert upper in iv_after.columns
            assert bool((iv_after[upper] >= iv_after[lower]).all())
        assert iv_before.equals(iv_after)


class TestSplitConformalWithExogenousFeatures:
    """Tests for X-forwarding in fit, observe, and predict_interval."""

    @pytest.fixture
    def conformal_data_with_X(self):
        """Create data with exogenous features."""
        n = 250
        dates = [datetime(2020, 1, 1) + timedelta(days=i) for i in range(n)]
        np.random.seed(42)
        values = [10.0 + 5.0 * np.sin(2 * np.pi * i / 7) + np.random.normal(0, 0.5) for i in range(n)]
        y = pl.DataFrame({"time": dates, "value": values})
        X = pl.DataFrame({
            "time": dates,
            "feature_a": [float(i % 7) for i in range(n)],
        })
        return y, X

    def test_fit_with_X_and_similarity(self, conformal_data_with_X):
        """Test that fit with X and similarity stores similarities_ correctly."""
        y, X = conformal_data_with_X
        scf = SplitConformalForecaster(
            point_forecaster=SeasonalNaive(seasonality=7),
            calibration_size=50,
            conformity_scorer=AbsoluteResidual(),
            similarity=DistanceSimilarity(metric="euclidean"),
        )
        scf.fit(y[:200], X_actual=X[:200], forecasting_horizon=1, coverage_rates=[0.9])
        assert hasattr(scf, "similarities_")
        assert "step_1" in scf.similarities_

    def test_predict_interval_with_X_and_similarity(self, conformal_data_with_X):
        """Test predict_interval with X and similarity produces valid intervals."""
        y, X = conformal_data_with_X
        scf = SplitConformalForecaster(
            point_forecaster=SeasonalNaive(seasonality=7),
            calibration_size=50,
            conformity_scorer=AbsoluteResidual(),
            similarity=DistanceSimilarity(metric="euclidean"),
        )
        scf.fit(y[:200], X_actual=X[:200], forecasting_horizon=1, coverage_rates=[0.9])
        intervals = scf.predict_interval(coverage_rates=[0.9])
        assert isinstance(intervals, pl.DataFrame)
        assert "value_lower_0.9" in intervals.columns
        assert "value_upper_0.9" in intervals.columns

    def test_observe_with_X_and_similarity(self, conformal_data_with_X):
        """Test that observe with X forwards to similarity correctly."""
        y, X = conformal_data_with_X
        scf = SplitConformalForecaster(
            point_forecaster=SeasonalNaive(seasonality=7),
            calibration_size=50,
            conformity_scorer=AbsoluteResidual(),
            similarity=DistanceSimilarity(metric="euclidean"),
        )
        scf.fit(y[:200], X_actual=X[:200], forecasting_horizon=1, coverage_rates=[0.9])
        sim_step = scf.similarities_["step_1"]
        n_before = len(sim_step._X_observed)

        scf.observe(y[200:201], X_actual=X[200:201])
        assert len(sim_step._X_observed) == n_before + 1


class TestSplitConformalObservePredict:
    """Tests for the observe_predict rolling method."""

    def test_observe_predict_returns_point_predictions(self, conformal_data):
        """Test that observe_predict returns valid point predictions."""
        scf = SplitConformalForecaster(calibration_size=50)
        scf.fit(conformal_data[:200], forecasting_horizon=1)

        y_test = conformal_data[200:205]
        y_pred = scf.observe_predict(y=y_test)
        assert isinstance(y_pred, pl.DataFrame)
        # Initial prediction + one per stride step
        assert len(y_pred) >= 1

    def test_observe_predict_with_similarity(self, conformal_data):
        """Test observe_predict with similarity enabled."""
        scf = SplitConformalForecaster(
            point_forecaster=SeasonalNaive(seasonality=1),
            calibration_size=50,
            conformity_scorer=AbsoluteResidual(),
            similarity=DistanceSimilarity(metric="euclidean"),
        )
        scf.fit(conformal_data[:200], forecasting_horizon=1, coverage_rates=[0.9])

        y_test = conformal_data[200:205]
        y_pred = scf.observe_predict(y=y_test)
        assert isinstance(y_pred, pl.DataFrame)
        assert len(y_pred) >= 1

    def test_observe_predict_with_X(self):
        """Test observe_predict with exogenous features."""
        n = 250
        dates = [datetime(2020, 1, 1) + timedelta(days=i) for i in range(n)]
        np.random.seed(42)
        values = [10.0 + 5.0 * np.sin(2 * np.pi * i / 7) + np.random.normal(0, 0.5) for i in range(n)]
        y = pl.DataFrame({"time": dates, "value": values})
        X = pl.DataFrame({"time": dates, "feature_a": [float(i % 7) for i in range(n)]})

        scf = SplitConformalForecaster(
            point_forecaster=SeasonalNaive(seasonality=7),
            calibration_size=50,
            conformity_scorer=AbsoluteResidual(),
            similarity=DistanceSimilarity(metric="euclidean"),
        )
        scf.fit(y[:200], X_actual=X[:200], forecasting_horizon=1)

        y_pred = scf.observe_predict(y=y[200:205], X_actual=X[200:])
        assert isinstance(y_pred, pl.DataFrame)
        assert len(y_pred) >= 1

    def test_observe_predict_interval_with_similarity(self, conformal_data):
        """Test observe_predict_interval with similarity produces interval predictions."""
        scf = SplitConformalForecaster(
            point_forecaster=SeasonalNaive(seasonality=1),
            calibration_size=50,
            conformity_scorer=AbsoluteResidual(),
            similarity=DistanceSimilarity(metric="euclidean"),
        )
        scf.fit(conformal_data[:200], forecasting_horizon=1, coverage_rates=[0.9])

        y_test = conformal_data[200:205]
        y_pred = scf.observe_predict_interval(y=y_test, coverage_rates=[0.9])
        assert isinstance(y_pred, pl.DataFrame)
        assert len(y_pred) >= 1
        assert any("_lower_0.9" in c for c in y_pred.columns)
        assert any("_upper_0.9" in c for c in y_pred.columns)

    def test_observe_predict_with_explicit_stride(self, conformal_data):
        """Test observe_predict with an explicit stride parameter."""
        scf = SplitConformalForecaster(
            point_forecaster=SeasonalNaive(seasonality=1),
            calibration_size=50,
            conformity_scorer=AbsoluteResidual(),
        )
        scf.fit(conformal_data[:200], forecasting_horizon=1)

        y_test = conformal_data[200:205]
        y_pred = scf.observe_predict(y=y_test, stride=1)
        assert isinstance(y_pred, pl.DataFrame)
        assert len(y_pred) >= 1


class TestSimilarityStaysAlignedWithConformityScores:
    """Similarity state and conformity scores must grow together across observe.

    ``fit`` deliberately fits the similarity on the scored subset, because scoring
    aligns y and y_pred by an inner join on time and so can return fewer rows than
    it was given. ``observe`` did not repeat that alignment, so the similarity grew
    faster than the scores and ``predict_interval`` later paired an N-weight array
    with N-1 scores, raising "x and weights must have the same length".
    """

    @staticmethod
    def _fitted(horizon: int):
        n = 260
        dates = [datetime(2020, 1, 1) + timedelta(days=i) for i in range(n)]
        rng = np.random.default_rng(7)
        values = [10.0 + 5.0 * np.sin(2 * np.pi * i / 7) + rng.normal(0, 0.5) for i in range(n)]
        y = pl.DataFrame({"time": dates, "value": values})

        scf = SplitConformalForecaster(
            point_forecaster=SeasonalNaive(seasonality=7),
            calibration_size=50,
            conformity_scorer=AbsoluteResidual(),
            similarity=DistanceSimilarity(metric="euclidean"),
        )
        scf.fit(y[:200], forecasting_horizon=horizon, coverage_rates=[0.9])
        return scf, y[200:]

    @pytest.mark.parametrize("horizon", [1, 2])
    def test_state_lengths_track_each_other(self, horizon):
        scf, y_rest = self._fitted(horizon)

        for i in range(10):
            scf.observe(y_rest[i : i + 1])

        for step in range(1, 1 + horizon):
            n_weights = len(scf.similarities_[f"step_{step}"]._X_observed)
            n_scores = scf.conformity_scores_.filter(pl.col("step") == step).height
            assert n_weights == n_scores, (
                f"step {step}: similarity holds {n_weights} observations but "
                f"{n_scores} conformity scores exist; predict_interval pairs these"
            )

    @pytest.mark.parametrize("horizon", [1, 2])
    def test_predict_interval_after_observe_does_not_raise(self, horizon):
        scf, y_rest = self._fitted(horizon)

        for i in range(10):
            scf.observe(y_rest[i : i + 1])

        scf.predict_interval(forecasting_horizon=horizon, coverage_rates=[0.9])


class TestSplitConformalTransformerRouting:
    """Configuration reaches this forecaster through its inner, not through slots of its own.

    These assertions are deliberately ad-hoc rather than generic checks: they name
    ``point_forecaster__``, which the shared suite cannot express because it knows
    nothing about a meta's parameter names.
    """

    @staticmethod
    def _x_forecast(n: int = 60, n_steps: int = 3) -> pl.DataFrame:
        rows = []
        for i in range(n):
            vintage = datetime(2024, 1, 1) + timedelta(days=i)
            for step in range(1, n_steps + 1):
                rows.append({
                    "vintage_time": vintage,
                    "time": vintage + timedelta(days=step),
                    "load": float(i + step),
                })
        return pl.DataFrame(rows)

    @staticmethod
    def _y(n: int = 60) -> pl.DataFrame:
        times = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(n)]
        return pl.DataFrame({"time": times, "value": [float(i % 7) + i * 0.1 for i in range(n)]})

    def test_predict_returns_the_inner_forecasters_frame(self):
        """The meta's point prediction is the inner's.

        Asserted on the returned frames rather than by counting calls, so the test
        pins the delegation itself and survives any refactor that preserves it.
        Interval bounds are excluded on purpose: those are computed on the meta from
        its conformity scorers, so the meta is not a pure pass-through.
        """
        forecaster = SplitConformalForecaster(
            point_forecaster=SeasonalNaive(seasonality=7),
            calibration_size=20,
        )
        forecaster.fit(self._y(), forecasting_horizon=3)

        from_meta = forecaster.predict(forecasting_horizon=3)
        from_inner = forecaster.point_forecaster_.predict(forecasting_horizon=3)

        assert from_meta.equals(from_inner)

    def test_slots_are_reachable_only_through_the_inner(self):
        """The meta declares no slot; the inner's are reachable by nested path.

        The default ``SeasonalNaive`` fixes both slots to ``None`` without exposing
        them, so a slot-declaring inner is required for the nested paths to resolve.
        """
        from sklearn.linear_model import LinearRegression

        from yohou.point import PointReductionForecaster

        forecaster = SplitConformalForecaster(
            point_forecaster=PointReductionForecaster(LinearRegression()),
            calibration_size=20,
        )
        params = forecaster.get_params()

        assert "point_forecaster__forecast_transformer" in params
        assert "point_forecaster__actual_transformer" in params
        assert "forecast_transformer" not in params
        assert "actual_transformer" not in params

    def test_nested_forecast_transformer_path_is_tunable(self):
        """A search reaches the inner's slot through the nested path.

        ``cv_results_`` carries the grid and ``best_forecaster_`` the refit winner;
        it exposes no fitted forecaster per candidate, since only the best estimator
        is refitted.
        """
        from sklearn.linear_model import LinearRegression

        from yohou.compose import PerVintageActualTransformer
        from yohou.metrics import MeanAbsoluteError
        from yohou.model_selection import GridSearchCV
        from yohou.point import PointReductionForecaster
        from yohou.preprocessing import FunctionTransformer

        def _scaled(factor: float) -> PerVintageActualTransformer:
            # Stateless on purpose. A lifted lag consumes rows from the start of
            # every vintage, which leaves the nearest step columns null for every
            # row and fails the estimator before the grid can be compared. That
            # hazard is real but belongs to the transformer tests, not here.
            return PerVintageActualTransformer(
                FunctionTransformer(
                    func=lambda df, f=factor: df.select((pl.col("load") * f).alias("scaled")),
                    feature_names_out=lambda self, names: ["scaled"],
                )
            )

        inner = PointReductionForecaster(LinearRegression(), reduction_strategy="direct")
        search = GridSearchCV(
            forecaster=SplitConformalForecaster(point_forecaster=inner, calibration_size=15),
            param_grid={"point_forecaster__forecast_transformer": [_scaled(1.0), _scaled(2.0)]},
            scoring=MeanAbsoluteError(),
            cv=2,
        )
        search.fit(self._y(), forecasting_horizon=3, X_forecast=self._x_forecast())

        assert len(search.cv_results_["params"]) == 2
        best_slot = search.best_forecaster_.point_forecaster_.forecast_transformer_
        assert best_slot is not None


class TestCalibrationReplayStepColumns:
    """The calibration replay derives step columns once, not once per origin.

    ``_observe_predict_loop`` has two branches. Given ``X_future`` or ``X_forecast``
    it derives step columns once over every observation time and each origin selects
    its row; given neither it falls through to ``observe``, which re-derives them one
    timestamp at a time. ``SplitConformalForecaster.fit`` holds both frames, so it
    forwards them.

    Every forecaster here is a ``PointReductionForecaster``, not the ``SeasonalNaive``
    used elsewhere in this module. ``SeasonalNaive`` reports
    ``requires_exogenous=False`` and discards both frames, which would make every
    assertion below pass without exercising anything.
    """

    @staticmethod
    def _forecaster():
        from sklearn.linear_model import LinearRegression

        from yohou.point import PointReductionForecaster

        return PointReductionForecaster(LinearRegression(), reduction_strategy="direct", target_as_feature="raw")

    @pytest.fixture
    def replay_data(self, y_X_factory):
        y, _, X_future, X_forecast = y_X_factory(
            length=140,
            n_targets=1,
            n_features=0,
            seed=7,
            n_future_features=2,
            n_forecast_features=2,
            forecasting_horizon=4,
            return_exogenous=True,
        )
        return y, X_future, X_forecast

    def _replay(self, y_train, y_calib, X_forecast, *, forward):
        """Fit a point forecaster and replay the calibration block.

        ``forward=False`` reproduces the pre-change call, which withheld the frames
        and so took the per origin branch.
        """
        point = self._forecaster().fit(y_train, forecasting_horizon=4, X_forecast=X_forecast)
        return point.observe_predict(
            y=y_calib,
            forecasting_horizon=None,
            stride=1,
            predict_transformed=False,
            X_forecast=X_forecast if forward else None,
        )

    @staticmethod
    def _count_derivations(forecaster, y, X_forecast):
        """Count ``_derive_step_columns`` calls across one conformal fit."""
        import yohou.base.forecaster as forecaster_module
        import yohou.base.panel as panel_module
        import yohou.base.standard as standard_module

        modules = [forecaster_module, panel_module, standard_module]
        original = forecaster_module._derive_step_columns
        calls = []

        def counting(*args, **kwargs):
            calls.append(1)
            return original(*args, **kwargs)

        for module in modules:
            module._derive_step_columns = counting
        try:
            forecaster.fit(y, forecasting_horizon=4, X_forecast=X_forecast)
        finally:
            for module in modules:
                module._derive_step_columns = original
        return len(calls)

    def _conformal(self, calibration_size):
        return SplitConformalForecaster(point_forecaster=self._forecaster(), calibration_size=calibration_size)

    def test_derivation_count_does_not_grow_with_calibration_size(self, replay_data):
        """Two fits differing only in ``calibration_size`` derive step columns equally often.

        Asserted on call count rather than wall clock, which would be flaky on shared
        CI. Before the frames were forwarded the count grew by exactly one derivation
        per extra calibration origin.
        """
        y, _, X_forecast = replay_data

        small = self._count_derivations(self._conformal(20), y, X_forecast)
        large = self._count_derivations(self._conformal(50), y, X_forecast)

        assert small == large, (
            f"step column derivations scaled with calibration_size: {small} at 20 "
            f"origins, {large} at 50. The replay has fallen back to the per origin "
            f"branch of _observe_predict_loop, which means fit stopped forwarding "
            f"X_future / X_forecast to observe_predict."
        )

    def test_derivation_count_check_detects_the_per_origin_branch(self, replay_data):
        """Guard the count assertion against being unable to fail.

        Drives ``observe_predict`` directly with the frames withheld, which is exactly
        what ``fit`` did before this change, and confirms the count then does scale.
        """
        y, _, X_forecast = replay_data
        y_train = y[:100]

        def count_replay(calibration_rows):
            import yohou.base.forecaster as forecaster_module
            import yohou.base.panel as panel_module
            import yohou.base.standard as standard_module

            modules = [forecaster_module, panel_module, standard_module]
            original = forecaster_module._derive_step_columns
            calls = []

            def counting(*args, **kwargs):
                calls.append(1)
                return original(*args, **kwargs)

            point = self._forecaster().fit(y_train, forecasting_horizon=4, X_forecast=X_forecast)
            for module in modules:
                module._derive_step_columns = counting
            try:
                point.observe_predict(
                    y=y[100 : 100 + calibration_rows],
                    forecasting_horizon=None,
                    stride=1,
                    predict_transformed=False,
                )
            finally:
                for module in modules:
                    module._derive_step_columns = original
            return len(calls)

        assert count_replay(10) != count_replay(30), (
            "the derivation count is insensitive to how many origins the replay "
            "visits, so the assertion above could not detect a regression"
        )

    def test_forwarding_the_frames_changes_no_prediction(self, replay_data):
        """The two branches produce identical predictions, not merely close ones.

        ``_derive_step_columns`` resolves vintages as-of per observation time, so one
        call with N timestamps performs the same per-timestamp selection as N calls
        with one each. Nothing is reassociated, so this is exact equality; a tolerance
        here would mask a regression.
        """
        y, _, X_forecast = replay_data
        y_train, y_calib = y[:100], y[100:]

        forwarded = self._replay(y_train, y_calib, X_forecast, forward=True)
        withheld = self._replay(y_train, y_calib, X_forecast, forward=False)

        assert forwarded.equals(withheld), (
            "forwarding X_forecast changed the replay output; the two branches of "
            "_observe_predict_loop must resolve the same vintages per observation time"
        )

    def test_equivalence_check_detects_a_planted_difference(self, replay_data):
        """Guard the equality assertion against being unable to fail.

        A mutation confined to the final vintage row does NOT work here: that vintage
        serves at most the last origin, so the comparison passes whether or not the
        code is correct. The mutation must reach every origin's step features, so it
        perturbs a whole value column.
        """
        y, _, X_forecast = replay_data
        y_train, y_calib = y[:100], y[100:]

        value_col = next(c for c in X_forecast.columns if c not in ("vintage_time", "time"))
        mutated = X_forecast.with_columns((pl.col(value_col) * 1.5 + 1.0).alias(value_col))

        point = self._forecaster().fit(y_train, forecasting_horizon=4, X_forecast=X_forecast)
        baseline = point.observe_predict(
            y=y_calib,
            forecasting_horizon=None,
            stride=1,
            predict_transformed=False,
            X_forecast=X_forecast,
        )
        point = self._forecaster().fit(y_train, forecasting_horizon=4, X_forecast=X_forecast)
        perturbed = point.observe_predict(
            y=y_calib,
            forecasting_horizon=None,
            stride=1,
            predict_transformed=False,
            X_forecast=mutated,
        )

        assert not perturbed.equals(baseline), (
            "the equivalence comparison cannot fail: perturbing an entire X_forecast "
            "value column left the replay output unchanged"
        )

    def test_forwarding_x_future_changes_no_prediction(self, replay_data):
        """The same equality holds for the known-future channel, not only forecasts."""
        y, X_future, _ = replay_data
        y_train, y_calib = y[:100], y[100:]

        def replay(frame):
            point = self._forecaster().fit(y_train, forecasting_horizon=4, X_future=X_future)
            return point.observe_predict(
                y=y_calib,
                forecasting_horizon=None,
                stride=1,
                predict_transformed=False,
                X_future=frame,
            )

        assert replay(X_future).equals(replay(None))

    def test_no_calibration_origin_loses_its_forecast_features(self, replay_data):
        """Every origin resolves a vintage at or before it.

        A correctness guard, not a performance one. An origin whose forecast step
        columns are all null is scored against a degraded predictor, which silently
        widens the calibration distribution and therefore the served bands.
        """
        y, _, X_forecast = replay_data

        forecaster = self._conformal(30).fit(y, forecasting_horizon=4, X_forecast=X_forecast)

        observed = forecaster.point_forecaster_._X_t_observed
        frames = list(observed.values()) if isinstance(observed, dict) else [observed]
        for frame in frames:
            step_cols = [c for c in frame.columns if "_step_" in c]
            assert step_cols, "the fixture must produce step columns for this to test anything"
            all_null = [c for c in step_cols if frame[c].is_null().all()]
            assert not all_null, (
                f"forecast step columns {all_null} are entirely null on a calibration origin's observed feature row"
            )


class TestBatchedOriginPrediction:
    """The replay predicts every origin in one pass per horizon step.

    The point forecaster is frozen for the whole replay, so no origin depends on
    having predicted at the one before it. Recording each origin's feature row and
    predicting them together issues H estimator calls instead of H per origin, over
    the same rows through the same estimators.
    """

    @staticmethod
    def _panel(n=140, n_groups=3):
        rng = np.random.default_rng(11)
        time = pl.datetime_range(
            start=datetime(2024, 1, 1),
            end=datetime(2024, 1, 1) + timedelta(hours=n - 1),
            interval="1h",
            eager=True,
        )
        cols = {"time": time}
        for g in range(n_groups):
            cols[f"g{g}__value"] = np.cumsum(rng.standard_normal(n)) + 100.0 + g
        return pl.DataFrame(cols)

    @staticmethod
    def _forecaster():
        from sklearn.linear_model import LinearRegression

        from yohou.point import PointReductionForecaster
        from yohou.preprocessing import LagTransformer

        return PointReductionForecaster(
            LinearRegression(),
            reduction_strategy="direct",
            panel_strategy="global",
            target_as_feature="raw",
            actual_transformer=LagTransformer(lag=[1, 3]),
        )

    def _fitted(self, y_train, horizon=4):
        return self._forecaster().fit(y_train, forecasting_horizon=horizon)

    def test_batched_replay_matches_rolling(self):
        """Structure exactly, values to within floating-point reassociation.

        A step's estimator is applied row-wise, so stacking origins into one call
        computes the same numbers mathematically but not to the same bits: BLAS picks
        its kernel by matrix size, and the stacked call is hundreds of rows where the
        rolling call is one per group. Whether the last bit actually differs depends on
        the machine, so an exact comparison here passes and fails by runner. Same
        tolerance, and same reason for the ``abs_tol``, as
        `tests.point.test_reduction._assert_matches_per_group`, which covers the
        sibling case of batching across panel groups instead of across origins.

        A real mix-up between origins moves a prediction by orders of magnitude more,
        which is what the planted-difference test below pins.
        """
        y = self._panel()
        y_train, y_calib = y[:100], y[100:]

        rolling = self._fitted(y_train).observe_predict(
            y=y_calib, forecasting_horizon=None, stride=1, predict_transformed=False
        )
        batched_forecaster = self._fitted(y_train)
        batched = batched_forecaster._observe_predict_batched_origins(
            y=y_calib, X_actual=None, groups=batched_forecaster.groups_ or [], stride=1
        )

        assert batched.shape == rolling.shape
        assert batched.schema == rolling.schema
        pl.testing.assert_frame_equal(
            batched,
            rolling,
            check_exact=False,
            rel_tol=1e-12,
            abs_tol=1e-15,
        )

    def test_equivalence_check_detects_a_planted_difference(self):
        """Guard the equivalence assertion against being unable to fail.

        Mis-assigns each origin's predictions to a different origin, which is the
        failure mode that matters here: the batched pass slices one stacked result
        back apart by position, so an off-by-one there would be silent.
        """
        from yohou.base.reduction import BaseReductionForecaster

        y = self._panel()
        y_train, y_calib = y[:100], y[100:]

        rolling = self._fitted(y_train).observe_predict(
            y=y_calib, forecasting_horizon=None, stride=1, predict_transformed=False
        )

        original = BaseReductionForecaster._estimator_predict_direct_multi

        def shuffled(self, estimators, groups, X_tab_per_origin):
            return list(reversed(original(self, estimators, groups, X_tab_per_origin)))

        forecaster = self._fitted(y_train)
        BaseReductionForecaster._estimator_predict_direct_multi = shuffled
        try:
            mutated = forecaster._observe_predict_batched_origins(
                y=y_calib, X_actual=None, groups=forecaster.groups_ or [], stride=1
            )
        finally:
            BaseReductionForecaster._estimator_predict_direct_multi = original

        # The same tolerant comparison the real check uses, so this guards that check
        # rather than a stricter one no longer made anywhere.
        with pytest.raises(AssertionError):
            pl.testing.assert_frame_equal(mutated, rolling, check_exact=False, rel_tol=1e-12, abs_tol=1e-15)

    def test_estimator_calls_do_not_scale_with_origin_count(self):
        """One call per horizon step for the whole replay, not one per origin.

        Counted rather than timed: wall clock would be flaky on shared CI, and the
        call count is what actually changed.
        """
        import yohou.base.reduction as reduction_module

        y = self._panel()
        y_train = y[:100]
        horizon = 4

        def count(n_origins, *, batched):
            original = reduction_module._predict_direct_step
            calls = []

            def counting(*a, **k):
                calls.append(1)
                return original(*a, **k)

            forecaster = self._fitted(y_train, horizon)
            y_calib = y[100 : 100 + n_origins]
            reduction_module._predict_direct_step = counting
            try:
                if batched:
                    forecaster._observe_predict_batched_origins(
                        y=y_calib, X_actual=None, groups=forecaster.groups_ or [], stride=1
                    )
                else:
                    forecaster.observe_predict(y=y_calib, forecasting_horizon=None, stride=1, predict_transformed=False)
            finally:
                reduction_module._predict_direct_step = original
            return len(calls)

        assert count(10, batched=True) == count(30, batched=True) == horizon, (
            "the batched path must issue exactly one estimator call per horizon step, "
            "independent of how many origins the replay visits"
        )
        # The rolling contrast proves the assertion above has teeth.
        assert count(10, batched=False) != count(30, batched=False)

    def test_conformal_fit_reaches_the_batched_path(self):
        """The replay inside `fit` uses it, not only the method in isolation."""
        import yohou.base.reduction as reduction_module

        y = self._panel()
        original = reduction_module._predict_direct_step
        calls = []

        def counting(*a, **k):
            calls.append(1)
            return original(*a, **k)

        forecaster = SplitConformalForecaster(point_forecaster=self._forecaster(), calibration_size=25)
        reduction_module._predict_direct_step = counting
        try:
            forecaster.fit(y, forecasting_horizon=4)
        finally:
            reduction_module._predict_direct_step = original

        # A per origin replay at 25 origins would issue at least 26 * 4 = 104 calls.
        assert len(calls) < 104, (
            f"{len(calls)} estimator calls for a 25 origin conformal fit suggests the "
            f"replay is still predicting one origin at a time"
        )

    def test_non_direct_strategies_are_refused(self):
        """multi-output holds one estimator for all steps, so there is nothing to batch."""
        from sklearn.linear_model import LinearRegression

        from yohou.point import PointReductionForecaster
        from yohou.preprocessing import LagTransformer

        y = self._panel()
        forecaster = PointReductionForecaster(
            LinearRegression(),
            reduction_strategy="multi-output",
            panel_strategy="global",
            target_as_feature="raw",
            actual_transformer=LagTransformer(lag=[1, 3]),
        ).fit(y[:100], forecasting_horizon=4)

        with pytest.raises(ValueError, match="reduction_strategy='direct'"):
            forecaster._observe_predict_batched_origins(
                y=y[100:], X_actual=None, groups=forecaster.groups_ or [], stride=1
            )


class TestBulkObserveReplay:
    """The replay transforms the whole calibration block in one pass per group.

    `observe_transform` concatenates its buffer with the incoming rows, transforms the
    window, then keeps only the new part, so observing one row against a 168 row buffer
    transforms 169 rows to keep one. Transforming the block once does that work for
    every origin at once.

    Sound only where every transformer reports `batch_invariant`, which the replay
    checks. An undeclared stack keeps the rolling path, so a missing declaration costs
    a speedup and never a result.
    """

    @staticmethod
    def _panel(n=140, n_groups=3):
        rng = np.random.default_rng(23)
        time = pl.datetime_range(
            start=datetime(2024, 1, 1),
            end=datetime(2024, 1, 1) + timedelta(hours=n - 1),
            interval="1h",
            eager=True,
        )
        cols = {"time": time}
        for g in range(n_groups):
            cols[f"g{g}__value"] = np.cumsum(rng.standard_normal(n)) + 100.0 + g
        return pl.DataFrame(cols)

    @staticmethod
    def _forecaster(actual_transformer=None):
        from sklearn.linear_model import LinearRegression

        from yohou.compose import FeatureUnion
        from yohou.point import PointReductionForecaster
        from yohou.preprocessing import LagTransformer, RollingStatisticsTransformer

        if actual_transformer is None:
            actual_transformer = FeatureUnion([
                ("lags", LagTransformer(lag=[1, 3])),
                ("roll", RollingStatisticsTransformer(window_size=4, statistics=["mean", "std"])),
            ])
        return PointReductionForecaster(
            LinearRegression(),
            reduction_strategy="direct",
            panel_strategy="global",
            target_as_feature="raw",
            actual_transformer=actual_transformer,
        )

    def test_bulk_replay_matches_rolling_within_tolerance(self):
        """Compared on a relative tolerance, not bit equality.

        Batching a rolling accumulator reassociates it, so a rolling mean or standard
        deviation can differ between the two paths by about one ULP. Some stacks land
        exactly, but a bit-equality assertion would then be pinning a property of the
        data rather than of the code.
        """
        y = self._panel()
        y_train, y_calib = y[:100], y[100:]
        horizon = 4

        rolling = (
            self
            ._forecaster()
            .fit(y_train, forecasting_horizon=horizon)
            .observe_predict(y=y_calib, forecasting_horizon=None, stride=1, predict_transformed=False)
        )
        bulk_forecaster = self._forecaster().fit(y_train, forecasting_horizon=horizon)
        bulk = bulk_forecaster._observe_predict_bulk_origins(
            y=y_calib, X_actual=None, groups=bulk_forecaster.groups_ or []
        )

        assert bulk.shape == rolling.shape
        assert bulk["time"].equals(rolling["time"])
        assert bulk["vintage_time"].equals(rolling["vintage_time"])

        numeric = [c for c, dtype in rolling.schema.items() if dtype.is_numeric()]
        expected = rolling.select(numeric).to_numpy()
        actual = bulk.select(numeric).to_numpy()
        relative = np.abs(np.where(expected != 0, (actual - expected) / expected, actual - expected))
        assert np.nanmax(relative) < 1e-9, (
            f"bulk observe moved a prediction by {np.nanmax(relative):.3e} relative, far "
            f"beyond the floating point reassociation a batched accumulator explains"
        )

    def test_tolerance_check_detects_a_planted_difference(self):
        """Guard the tolerance assertion against being unable to fail.

        Shifts one origin's features by a row, which is the failure mode that matters:
        the bulk path reconstructs each origin's row by slicing rather than by rolling,
        so an off-by-one there would produce plausible values in the wrong places.
        """
        from yohou.base.reduction import BaseReductionForecaster

        y = self._panel()
        y_train, y_calib = y[:100], y[100:]

        rolling = (
            self
            ._forecaster()
            .fit(y_train, forecasting_horizon=4)
            .observe_predict(y=y_calib, forecasting_horizon=None, stride=1, predict_transformed=False)
        )

        original = BaseReductionForecaster._bulk_origin_features

        def shifted(self, **kwargs):
            X_tab, y_observed, observed_time = original(self, **kwargs)
            return X_tab[1:] + X_tab[:1], y_observed, observed_time

        forecaster = self._forecaster().fit(y_train, forecasting_horizon=4)
        BaseReductionForecaster._bulk_origin_features = shifted
        try:
            mutated = forecaster._observe_predict_bulk_origins(
                y=y_calib, X_actual=None, groups=forecaster.groups_ or []
            )
        finally:
            BaseReductionForecaster._bulk_origin_features = original

        numeric = [c for c, dtype in rolling.schema.items() if dtype.is_numeric()]
        expected = rolling.select(numeric).to_numpy()
        actual = mutated.select(numeric).to_numpy()
        relative = np.abs(np.where(expected != 0, (actual - expected) / expected, actual - expected))
        assert np.nanmax(relative) > 1e-9, (
            "the tolerance comparison cannot fail: rotating the per origin feature rows "
            "left every prediction within tolerance"
        )

    def test_transformer_calls_do_not_scale_with_origin_count(self):
        """One transform pass per group for the whole block, not one per origin."""
        import yohou.base.reduction as reduction_module

        y = self._panel()
        y_train = y[:100]

        def count(n_origins):
            original = reduction_module._observe_transformers_one
            calls = []

            def counting(*a, **k):
                calls.append(1)
                return original(*a, **k)

            forecaster = self._forecaster().fit(y_train, forecasting_horizon=4)
            reduction_module._observe_transformers_one = counting
            try:
                forecaster._observe_predict_bulk_origins(
                    y=y[100 : 100 + n_origins], X_actual=None, groups=forecaster.groups_ or []
                )
            finally:
                reduction_module._observe_transformers_one = original
            return len(calls)

        assert count(10) == count(30), (
            "the bulk path must transform each group once for the whole block, "
            "independent of how many origins the replay visits"
        )

    def test_an_undeclared_transformer_keeps_the_rolling_path(self):
        """A stack the gate cannot vouch for falls back rather than guessing."""
        from yohou.preprocessing import FunctionTransformer

        y = self._panel()
        declared = self._forecaster().fit(y[:100], forecasting_horizon=4)
        undeclared = self._forecaster(actual_transformer=FunctionTransformer(func=lambda frame: frame * 2.0)).fit(
            y[:100], forecasting_horizon=4
        )

        assert declared._chains_are_batch_invariant()
        assert not undeclared._chains_are_batch_invariant(), (
            "FunctionTransformer takes a caller-supplied func over the whole frame, so "
            "it cannot promise batch invariance and must not be vouched for"
        )

    def test_conformal_fit_reaches_the_bulk_path(self):
        """The replay inside `fit` uses it when the stack allows."""
        import yohou.base.reduction as reduction_module

        y = self._panel()
        original = reduction_module._observe_transformers_one
        calls = []

        def counting(*a, **k):
            calls.append(1)
            return original(*a, **k)

        forecaster = SplitConformalForecaster(point_forecaster=self._forecaster(), calibration_size=25)
        reduction_module._observe_transformers_one = counting
        try:
            forecaster.fit(y, forecasting_horizon=4)
        finally:
            reduction_module._observe_transformers_one = original

        # A per origin replay over 25 origins and 3 groups would issue at least 75 calls.
        assert len(calls) < 75, (
            f"{len(calls)} transformer passes for a 25 origin conformal fit suggests the "
            f"replay is still observing one row at a time"
        )

    def test_the_replay_leaves_the_forecaster_having_observed_the_block(self):
        """Both paths must agree on where they leave the forecaster, not just on frames.

        The rolling path observes its way through the calibration block and ends having
        observed all of it. The bulk path reconstructs each origin by slicing and never
        advances the buffer, so it has to land on that end state deliberately.

        Every equivalence test around this replay compares returned frames, and the two
        paths returned equivalent frames while disagreeing about the state left behind.
        That is why this asserts on the state instead.
        """
        from yohou.preprocessing import FunctionTransformer

        y = self._panel()
        calibration_size = 25

        bulk = SplitConformalForecaster(point_forecaster=self._forecaster(), calibration_size=calibration_size).fit(
            y, forecasting_horizon=4
        )
        rolling = SplitConformalForecaster(
            point_forecaster=self._forecaster(actual_transformer=FunctionTransformer(func=lambda frame: frame * 1.0)),
            calibration_size=calibration_size,
        ).fit(y, forecasting_horizon=4)

        assert bulk.point_forecaster_._chains_are_batch_invariant()
        assert not rolling.point_forecaster_._chains_are_batch_invariant()

        assert bulk.point_forecaster_.observed_time_ == rolling.point_forecaster_.observed_time_, (
            "the bulk replay left the forecaster at a different observation time than the "
            "rolling replay, so one of them did not observe the whole calibration block"
        )

    def test_the_replay_leaves_the_observation_time_at_the_end_of_the_data(self):
        """Pinned against the data rather than against the other path.

        Restoring the pre-replay state rewinds the forecaster by the whole calibration
        block, so this lands `calibration_size` rows early. Stating the expected time
        outright means the test still fails if both paths regress together.
        """
        y = self._panel()

        forecaster = SplitConformalForecaster(point_forecaster=self._forecaster(), calibration_size=25).fit(
            y, forecasting_horizon=4
        )

        observed = forecaster.point_forecaster_.observed_time_
        # Panel forecasters carry one observation time per group; they share a clock.
        if isinstance(observed, dict):
            assert len(set(observed.values())) == 1
            observed = next(iter(observed.values()))
        assert observed == y["time"][-1], (
            f"observation time is {observed} but the data ends at {y['time'][-1]}; a "
            f"forecaster rewound behind its own training data stitches a stale window "
            f"onto fresh forecasts"
        )

    def test_predicting_after_the_replay_sees_one_regular_time_axis(self):
        """The failure this actually caused, rather than the state that caused it.

        A forecaster left rewound by the calibration block produces a frame spanning its
        stale observation window and a fresh forecast window with nothing in between.
        Whichever transformer inverts first then rejects the two intervals, so a plain
        observe-then-predict is enough to catch it without reaching into any state.

        Two things make this bite, and both are properties of the production stack
        rather than of this test. The target transformer validates the frame handed to
        its inverse, so without one nothing inspects the time axis. And the observation
        horizon has to outlast the stride: the buffer keeps its most recent
        `observation_horizon` rows, so a stack that observes at least that many rows per
        origin flushes the hole away before anything looks at it. A short lookback is
        why the obvious version of this test passes against the bug.
        """
        from sklearn.linear_model import LinearRegression

        from yohou.compose import FeatureUnion
        from yohou.point import PointReductionForecaster
        from yohou.preprocessing import LagTransformer
        from yohou.stationarity import ASinhTransformer

        y = self._panel(n=180)
        train, later = y[:140], y[140:]

        point = PointReductionForecaster(
            LinearRegression(),
            reduction_strategy="direct",
            panel_strategy="global",
            target_as_feature="raw",
            actual_transformer=FeatureUnion([("lags", LagTransformer(lag=[1, 30]))]),
            target_transformer=ASinhTransformer(),
        )
        forecaster = SplitConformalForecaster(point_forecaster=point, calibration_size=25).fit(
            train, forecasting_horizon=4
        )

        predictions = forecaster.observe_predict_interval(y=later, forecasting_horizon=4, coverage_rates=[0.8])
        assert predictions.height > 0

    def test_the_batched_path_also_leaves_the_block_observed(self):
        """The third path, which the bulk-versus-rolling comparison does not reach.

        `_observe_predict_batched_origins` captures its saved state inside `assemble`,
        which runs as the `reduce_fn` after `_observe_predict_loop` has already rolled
        through the origins, so what it restores is a post-observe state. That is why it
        was never affected. Asserting it means the invariant covers all three paths
        rather than the two a bulk-versus-rolling comparison happens to exercise.
        """
        y = self._panel()
        forecaster = SplitConformalForecaster(point_forecaster=self._forecaster(), calibration_size=25)

        # The batched path is chosen when the stack is not batch invariant but the
        # forecaster still exposes batched origins, so deny the bulk guard to reach it.
        cls = type(forecaster.point_forecaster)
        saved = cls._chains_are_batch_invariant
        cls._chains_are_batch_invariant = lambda self: False
        try:
            fitted = forecaster.fit(y, forecasting_horizon=4)
        finally:
            cls._chains_are_batch_invariant = saved

        assert fitted.replay_path_ == "batched"
        observed = fitted.point_forecaster_.observed_time_
        stamps = set(observed.values()) if isinstance(observed, dict) else {observed}
        assert stamps == {y["time"].max()}, (
            "the batched replay left the forecaster short of the calibration block, so it "
            "has the defect the bulk path had"
        )

    def test_a_replay_that_rewinds_the_forecaster_fails_at_fit(self):
        """The invariant check fires where the state breaks, not four layers later.

        Before this check, a replay that rewound the forecaster surfaced as an
        inconsistent-interval error from whichever transformer inverted first, describing
        regular input data as irregular. Attributing that took a cloud step log and a
        bisect across two submodule bumps.
        """
        y = self._panel()
        forecaster = SplitConformalForecaster(point_forecaster=self._forecaster(), calibration_size=25).fit(
            y, forecasting_horizon=4
        )

        # Rewind the fitted point forecaster by the calibration block, which is exactly
        # what restoring the pre-replay state used to do.
        point = forecaster.point_forecaster_
        rewound = y["time"].max() - timedelta(hours=25)
        point.observed_time_ = (
            dict.fromkeys(point.observed_time_, rewound) if isinstance(point.observed_time_, dict) else rewound
        )

        with pytest.raises(RuntimeError, match="fitted and not predictable"):
            forecaster._check_replay_left_the_block_observed(y, "bulk")
