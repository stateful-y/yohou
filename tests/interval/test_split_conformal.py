"""Tests for SplitConformalForecaster."""

from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError

from yohou.interval import DistanceSimilarity, SplitConformalForecaster
from yohou.metrics import AbsoluteResidual, Residual
from yohou.point import SeasonalNaive


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
        assert tags.forecaster_tags.forecaster_type == "both"


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
