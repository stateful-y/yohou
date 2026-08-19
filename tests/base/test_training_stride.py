"""Tests for the ``training_stride`` parameter of reduction forecasters.

The stride keeps one tabularized training instance every ``training_stride``
rows, tail-anchored: the most recent instance is always kept. The mask applies
to features, targets, and sample weights in lockstep, before NaN handling, and
per panel group in ``groups_`` order.
"""

from datetime import datetime

import numpy as np
import polars as pl
import pytest
from sklearn.base import BaseEstimator, RegressorMixin, clone

from yohou.interval import IntervalReductionForecaster
from yohou.point import PointReductionForecaster
from yohou.weighting import ExponentialDecayWeighter


class _RecordingRegressor(RegressorMixin, BaseEstimator):
    """Regressor that records the training data it receives."""

    def __init__(self):
        self.received_ = None

    def fit(self, X, y, sample_weight=None):
        self.received_ = (X, y, sample_weight)
        self.n_outputs_ = 1 if getattr(y, "ndim", 1) == 1 else y.shape[1]
        return self

    def predict(self, X):
        n_outputs = getattr(self, "n_outputs_", 1)
        if n_outputs == 1:
            return np.zeros(len(X))
        return np.zeros((len(X), n_outputs))


def _series(n: int, value_col: str = "value") -> pl.DataFrame:
    times = pl.datetime_range(
        datetime(2026, 1, 1),
        pl.select(pl.lit(datetime(2026, 1, 1)).dt.offset_by(f"{n - 1}h")).item(),
        interval="1h",
        eager=True,
    )
    return pl.DataFrame({"time": times, value_col: [float(i) for i in range(n)]})


class TestTrainingStrideSemantics:
    def test_default_is_a_no_op(self):
        """training_stride=1 trains on every tabularized instance."""
        y = _series(30)
        strided = PointReductionForecaster(estimator=_RecordingRegressor())
        strided.fit(y=y, forecasting_horizon=2)
        n_received = len(strided.estimator_.received_[0])
        assert n_received == len(y) - 2

    def test_tail_anchored_subsampling(self):
        """With stride k the kept rows are k apart and include the last instance."""
        y = _series(30)
        fc = PointReductionForecaster(estimator=_RecordingRegressor(), training_stride=7)
        fc.fit(y=y, forecasting_horizon=2)
        X_received, y_received, _ = fc.estimator_.received_

        n_instances = len(y) - 2  # 28
        expected_positions = [i for i in range(n_instances) if i % 7 == (n_instances - 1) % 7]
        assert len(X_received) == len(expected_positions)

        # The target for step 1 at kept instance i is y[i + 1]; the last
        # instance (origin y[27], targets y[28], y[29]) must be present.
        step1 = y_received["value_step_1"].to_list()
        assert step1 == [float(i + 1) for i in expected_positions]
        assert expected_positions[-1] == n_instances - 1

    def test_sample_weights_filtered_in_lockstep(self):
        """The weight vector matches the kept rows one to one."""
        y = _series(40)
        fc = PointReductionForecaster(
            estimator=_RecordingRegressor(),
            training_stride=5,
            time_weighter=ExponentialDecayWeighter(half_life=10),
        )
        fc.fit(y=y, forecasting_horizon=3)
        X_received, _, sw = fc.estimator_.received_
        assert sw is not None
        assert len(sw) == len(X_received)

    def test_panel_groups_masked_identically(self):
        """Equal-length groups contribute the same kept positions, stacked."""
        n = 26
        base = _series(n)
        y = pl.DataFrame({
            "time": base["time"],
            "a__value": base["value"],
            "b__value": base["value"] + 100.0,
        })
        fc = PointReductionForecaster(
            estimator=_RecordingRegressor(),
            training_stride=6,
            panel_strategy="global",
        )
        fc.fit(y=y, forecasting_horizon=2)
        X_received, y_received, _ = fc.estimator_.received_

        n_instances = n - 2
        kept_per_group = len([i for i in range(n_instances) if i % 6 == (n_instances - 1) % 6])
        assert len(X_received) == 2 * kept_per_group

        # Group order is groups_ order: group a's rows stack before group b's,
        # and group b's targets carry the +100 offset.
        step1 = y_received["value_step_1"].to_list()
        assert all(v < 100 for v in step1[:kept_per_group])
        assert all(v >= 100 for v in step1[kept_per_group:])

    @pytest.mark.parametrize("strategy", ["multi-output", "direct", "dir-rec"])
    def test_all_strategies_honor_the_stride(self, strategy):
        """Every reduction strategy fits on the same strided instance count."""
        y = _series(30)
        fc = PointReductionForecaster(
            estimator=_RecordingRegressor(),
            reduction_strategy=strategy,
            training_stride=4,
        )
        fc.fit(y=y, forecasting_horizon=2)
        estimators = fc.estimator_ if isinstance(fc.estimator_, list) else [fc.estimator_]
        n_instances = len(y) - 2
        kept = len([i for i in range(n_instances) if i % 4 == (n_instances - 1) % 4])
        for est in estimators:
            assert len(est.received_[0]) == kept

    def test_interval_per_quantile_estimators_are_strided(self):
        """Each per-quantile estimator of the interval model sees strided data."""
        y = _series(40)
        fc = IntervalReductionForecaster(training_stride=8)
        fc.fit(y=y, forecasting_horizon=2, coverage_rates=[0.5])
        n_instances = len(y) - 2
        kept = len([i for i in range(n_instances) if i % 8 == (n_instances - 1) % 8])
        assert isinstance(fc.estimator_, dict)
        for est in fc.estimator_.values():
            # MultiOutputRegressor exposes the fitted row count via its
            # estimators' n_features_in_ only; fit a recording check instead by
            # predicting on a single row and trusting the fit not to raise.
            assert est is not None
        # The stride reached the shared tabularization: refit with a recorder.
        rec = IntervalReductionForecaster(
            estimator=_MultiOutputRecording(),
            training_stride=8,
        )
        rec.fit(y=y, forecasting_horizon=2, coverage_rates=[0.5])
        for est in rec.estimator_.values():
            assert len(est.received_[0]) == kept


class _MultiOutputRecording(RegressorMixin, BaseEstimator):
    """Multi-output recorder accepting the quantile parameter."""

    def __init__(self, quantile=0.5):
        self.quantile = quantile
        self.received_ = None

    def fit(self, X, y, sample_weight=None):
        self.received_ = (X, y, sample_weight)
        self.n_outputs_ = 1 if getattr(y, "ndim", 1) == 1 else y.shape[1]
        return self

    def predict(self, X):
        n_outputs = getattr(self, "n_outputs_", 1)
        if n_outputs == 1:
            return np.zeros(len(X))
        return np.zeros((len(X), n_outputs))


class TestTrainingStrideValidation:
    def test_invalid_stride_rejected_before_fitting(self):
        y = _series(20)
        fc = PointReductionForecaster(training_stride=0)
        with pytest.raises(ValueError, match="training_stride"):
            fc.fit(y=y, forecasting_horizon=2)

    def test_clone_round_trip(self):
        fc = PointReductionForecaster(training_stride=24)
        assert clone(fc).training_stride == 24

    def test_all_nan_message_names_the_stride(self):
        """When NaN handling drops every kept instance, the stride is named."""
        n = 30
        times = pl.datetime_range(
            datetime(2026, 1, 1),
            pl.select(pl.lit(datetime(2026, 1, 1)).dt.offset_by(f"{n - 1}h")).item(),
            interval="1h",
            eager=True,
        )
        y = pl.DataFrame({"time": times, "value": [float("nan")] * n})
        fc = PointReductionForecaster(
            estimator=_RecordingRegressor(),
            training_stride=6,
            nan_handling="drop",
        )
        with pytest.raises(ValueError, match="training_stride=6"):
            fc.fit(y=y, forecasting_horizon=2)
