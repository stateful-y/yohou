"""Tests for SplitConformalForecaster."""

from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError

from conftest import run_checks
from yohou.interval import DistanceSimilarity, SplitConformalForecaster
from yohou.interval.similarity import TemporalSimilarity
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
        """Test that fit stores similarities_ and weights_ when similarity is set."""
        horizon = 3

        scf = SplitConformalForecaster(
            point_forecaster=SeasonalNaive(seasonality=1),
            calibration_size=50,
            conformity_scorer=AbsoluteResidual(),
            similarity=DistanceSimilarity(metric="euclidean"),
        )
        scf.fit(conformal_data, forecasting_horizon=horizon)

        assert hasattr(scf, "similarities_")
        assert hasattr(scf, "weights_")
        assert len(scf.similarities_) == horizon
        assert isinstance(scf.weights_, pl.DataFrame)
        assert "step" in scf.weights_.columns

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
        assert not hasattr(scf, "weights_")

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
        """Run all standard forecaster checks on SplitConformalForecaster."""
        y, _ = y_X_factory(length=200, n_targets=1, n_features=0, seed=42)
        y_train, y_test = y[:180], y[180:]

        forecaster = SplitConformalForecaster(
            point_forecaster=SeasonalNaive(seasonality=7),
            calibration_size=50,
        )
        forecaster.fit(y_train, forecasting_horizon=5)

        run_checks(
            forecaster,
            _yield_yohou_forecaster_checks(forecaster, y_train, None, y_test, None),
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
        """Test that observe() works with TemporalSimilarity."""
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
            similarity=TemporalSimilarity(seasonalities=[7.0]),
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
