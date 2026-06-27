"""Contract tests for BaseForecaster lifecycle methods.

Verifies fit, observe, rewind, predict, and observation_horizon
behaviour not covered by the systematic check suite, using a minimal
concrete forecaster.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from conftest import SimpleTransformer
from yohou.point import PointReductionForecaster


class TestBaseForecasterFit:
    """Tests for fit lifecycle and fitted attributes."""

    def test_fit_returns_self(self, y_X_factory):
        """Fit returns the forecaster instance."""
        y, X = y_X_factory(length=50, n_targets=1, n_features=0)
        f = PointReductionForecaster()
        result = f.fit(y, forecasting_horizon=3)
        assert result is f

    # The individual fitted attributes (fit_forecasting_horizon_, interval_,
    # _y_observed) are exhaustively covered by check_fit_sets_forecaster_attributes
    # in the TestForecasterCommon systematic suite (tests/test_common.py).


class TestBaseForecasterPredict:
    """Tests for predict lifecycle."""

    # Column presence (vintage_time/time), dtype, and length==forecasting_horizon
    # are exhaustively covered by check_predict_time_columns in the systematic
    # suite (tests/test_common.py); only the predict-time horizon override below
    # is unique to this file.

    def test_predict_different_horizon(self, y_X_factory):
        """Predict with different horizon than fit."""
        y, X = y_X_factory(length=50, n_targets=1, n_features=0)
        f = PointReductionForecaster()
        f.fit(y, forecasting_horizon=3)
        result = f.predict(forecasting_horizon=7)
        assert len(result) == 7


class TestBaseForecasterObservationHorizon:
    """Tests for observation_horizon property."""

    def test_horizon_after_fit(self, y_X_factory):
        """A PointReductionForecaster with no stateful transformer needs no buffer."""
        y, X = y_X_factory(length=50, n_targets=1, n_features=0)
        f = PointReductionForecaster()
        f.fit(y, forecasting_horizon=1)
        assert f.observation_horizon == 0

    def test_horizon_includes_transformer(self, y_X_factory):
        """Observation horizon includes target transformer horizon."""
        y, X = y_X_factory(length=50, n_targets=1, n_features=0)
        f = PointReductionForecaster(
            target_transformer=SimpleTransformer(observation_horizon=5),
        )
        f.fit(y, forecasting_horizon=1)
        assert f.observation_horizon >= 5


# estimator_type == "forecaster" and forecaster_type == frozenset({"point"})
# are covered by check_forecaster_tags_accessible_before_fit and
# check_forecaster_tags_match_capabilities in the systematic suite
# (tests/test_common.py).


class TestBaseForecasterWithExogenous:
    """Tests for forecasters with exogenous features."""

    def test_fit_with_X(self, y_X_factory):
        """Fit with exogenous features yields a usable forecaster."""
        y, X = y_X_factory(length=50, n_targets=1, n_features=2)
        f = PointReductionForecaster()
        f.fit(y, X, forecasting_horizon=1)
        # Verify the forecaster can produce predictions
        result = f.predict()
        assert len(result) == 1

    def test_observe_with_X(self, y_X_factory):
        """Observe with exogenous features advances the observation buffer."""
        y, X = y_X_factory(length=60, n_targets=1, n_features=2)
        f = PointReductionForecaster()
        f.fit(y[:50], X[:50], forecasting_horizon=1)
        result = f.observe(y[50:], X[50:])
        assert result is f
        # Non-panel buffer update: observed_time_ advances to the last new row.
        assert f.observed_time_ == y["time"][-1]


class TestBaseForecasterPreFitValidation:
    """Tests for _validate_pre_fit error paths."""

    def test_feature_transformer_without_X_raises(self, y_X_factory):
        """Feature transformer with target_as_feature=None and no X raises."""
        y, X = y_X_factory(length=50, n_targets=1, n_features=0)
        f = PointReductionForecaster(
            feature_transformer=SimpleTransformer(observation_horizon=0),
            target_as_feature=None,
        )
        with pytest.raises(ValueError, match="feature_transformer requires X"):
            f.fit(y, forecasting_horizon=1)

    def test_no_X_with_exogenous_forecaster_raises(self, y_X_factory):
        """Forecaster with requires_exogenous=True and no X raises."""
        y, X = y_X_factory(length=50, n_targets=1, n_features=0)
        f = PointReductionForecaster(target_as_feature=None)
        with pytest.raises(ValueError, match="target_as_feature=None requires X"):
            f.fit(y, forecasting_horizon=1)


class TestBaseForecasterRewindObservationHorizonZero:
    """Tests for rewind with observation_horizon == 0."""

    def test_rewind_zero_observation_horizon(self, y_X_factory):
        """Rewind special-cases observation_horizon == 0."""
        y, X = y_X_factory(length=50, n_targets=1, n_features=0)
        f = PointReductionForecaster()
        f.fit(y[:40], forecasting_horizon=3)
        assert f.observation_horizon == 0
        result = f.rewind(y[:40])
        assert result is f

    def test_rewind_zero_horizon_predict_matches(self, y_X_factory):
        """Predictions after rewind with horizon=0 remain consistent."""
        y, X = y_X_factory(length=50, n_targets=1, n_features=0)
        f = PointReductionForecaster()
        f.fit(y[:40], forecasting_horizon=3)
        pred_before = f.predict()
        f.rewind(y[:40])
        pred_after = f.predict()
        assert pred_before.equals(pred_after)


class TestBaseForecasterPanelObserve:
    """Tests for panel observe path."""

    def test_panel_observe(self, y_X_factory):
        """Observe advances observed_time_ for every panel group."""
        y, X = y_X_factory(length=60, n_targets=1, n_features=0, panel=True, n_groups=2)
        f = PointReductionForecaster()
        f.fit(y[:50], forecasting_horizon=1)
        observed_before = dict(f.observed_time_)

        result = f.observe(y[50:])

        assert result is f
        last_time = y["time"][-1]
        for group, before in observed_before.items():
            assert f.observed_time_[group] > before
            assert f.observed_time_[group] == last_time

    def test_panel_rewind(self, y_X_factory):
        """Rewind rolls observed_time_ back to the rewind boundary per group."""
        y, X = y_X_factory(length=60, n_targets=1, n_features=0, panel=True, n_groups=2)
        f = PointReductionForecaster()
        f.fit(y[:50], forecasting_horizon=1)
        f.observe(y[50:])

        result = f.rewind(y[:50])

        assert result is f
        boundary = y[:50]["time"][-1]
        for group_time in f.observed_time_.values():
            assert group_time == boundary
