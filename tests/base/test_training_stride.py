"""Tests for the ``training_stride`` parameter of reduction forecasters.

The stride keeps one tabularized training instance every ``training_stride``
rows, anchored on the data tail: kept origins share the phase of the last row
passed to fit, a ``validation_size`` tail included. The mask applies to features, targets, and sample weights in
lockstep, before NaN handling, and per panel group in ``groups_`` order.
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
        # polars frames carry shape but not ndim; a (n, c) shape means c outputs.
        shape = getattr(y, "shape", None)
        self.n_outputs_ = shape[1] if shape is not None and len(shape) == 2 else 1
        return self

    def predict(self, X):
        n_outputs = getattr(self, "n_outputs_", 1)
        if n_outputs == 1:
            return np.zeros(len(X))
        return np.zeros((len(X), n_outputs))


def _tail_phase_mask(n_rows: int, horizon: int, stride: int) -> np.ndarray:
    """Expected keep mask: instance ``i`` (origin row ``i``) shares the last row's phase."""
    return np.arange(n_rows - horizon) % stride == (n_rows - 1) % stride


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

    def test_kept_origins_share_the_last_rows_phase(self):
        """With stride k the kept origins are k rows apart, in phase with the last row."""
        y = _series(30)
        fc = PointReductionForecaster(estimator=_RecordingRegressor(), training_stride=7)
        fc.fit(y=y, forecasting_horizon=2)
        X_received, y_received, _ = fc.estimator_.received_

        expected_positions = np.flatnonzero(_tail_phase_mask(len(y), 2, 7)).tolist()
        assert len(X_received) == len(expected_positions)

        # The target for step 1 at kept instance i is y[i + 1], so its value is i + 1.
        step1 = y_received["value_step_1"].to_list()
        assert step1 == [float(i + 1) for i in expected_positions]
        assert all((len(y) - 1 - i) % 7 == 0 for i in expected_positions)

    def test_daily_stride_keeps_the_last_rows_hour_at_a_40_step_horizon(self):
        """Hourly data ending 23:00, stride 24, horizon 40: every kept origin is at 23:00."""
        n = 24 * 10
        times = pl.datetime_range(datetime(2026, 1, 1), datetime(2026, 1, 10, 23), interval="1h", eager=True)
        y = pl.DataFrame({"time": times, "value": [float(i) for i in range(n)]})
        fc = PointReductionForecaster(estimator=_RecordingRegressor(), training_stride=24)
        fc.fit(y=y, forecasting_horizon=40)
        _, y_received, _ = fc.estimator_.received_

        origins = [int(v) - 1 for v in y_received["value_step_1"].to_list()]
        assert origins
        assert {times[i].hour for i in origins} == {23}
        # The last instance (origin row n - 41, at 07:00) and the seven after 23:00 are dropped.
        assert origins[-1] == n - 49

    def test_the_most_recent_instance_is_kept_when_the_horizon_is_a_multiple_of_the_stride(self):
        """Stride 5 at horizon 40: the kept set equals counting back from the last instance."""
        y = _series(100)
        fc = PointReductionForecaster(estimator=_RecordingRegressor(), training_stride=5)
        fc.fit(y=y, forecasting_horizon=40)
        _, y_received, _ = fc.estimator_.received_

        n_instances = len(y) - 40
        from_last_instance = [i for i in range(n_instances) if i % 5 == (n_instances - 1) % 5]
        origins = [int(v) - 1 for v in y_received["value_step_1"].to_list()]
        assert origins == from_last_instance
        assert origins[-1] == n_instances - 1

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

        kept_per_group = int(_tail_phase_mask(n, 2, 6).sum())
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
        kept = int(_tail_phase_mask(len(y), 2, 4).sum())
        for est in estimators:
            assert len(est.received_[0]) == kept

    def test_interval_per_quantile_estimators_are_strided(self):
        """Each per-quantile estimator of the interval model sees strided data."""
        y = _series(40)
        fc = IntervalReductionForecaster(training_stride=8)
        fc.fit(y=y, forecasting_horizon=2, coverage_rates=[0.5])
        kept = int(_tail_phase_mask(len(y), 2, 8).sum())
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
        # polars frames carry shape but not ndim; a (n, c) shape means c outputs.
        shape = getattr(y, "shape", None)
        self.n_outputs_ = shape[1] if shape is not None and len(shape) == 2 else 1
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


class _EvalSetRecorder(_RecordingRegressor):
    """Recording regressor whose fit accepts an ``eval_set``, as ``validation_size`` requires."""

    def fit(self, X, y, eval_set=None, sample_weight=None):
        self.eval_set_ = eval_set
        return super().fit(X, y, sample_weight=sample_weight)


class TestTrainingStrideWithValidationHoldout:
    """A ``validation_size`` tail is held back from training but still ends on the anchor row."""

    @pytest.mark.parametrize("validation_size", [40, 48, 50])
    def test_kept_origins_share_the_last_rows_phase(self, validation_size):
        """Hourly, stride 24, horizon 40: kept origins stay in phase with the last row."""
        y = _series(24 * 10)
        fc = PointReductionForecaster(estimator=_EvalSetRecorder(), training_stride=24, validation_size=validation_size)
        fc.fit(y=y, forecasting_horizon=40)
        _, y_received, _ = fc.estimator_.received_

        origins = [int(v) - 1 for v in y_received["value_step_1"].to_list()]
        assert origins
        assert all((len(y) - 1 - i) % 24 == 0 for i in origins)
        # Early stopping still sees every origin in the tail, unstrided.
        X_eval, _ = fc.estimator_.eval_set_[0]
        assert len(X_eval) == validation_size - 40 + 1

    def test_head_check_counts_the_instances_the_stride_skips(self):
        """A head of horizon + 1 rows is too short once the stride skips 6 instances."""
        fc = PointReductionForecaster(estimator=_EvalSetRecorder(), training_stride=24, validation_size=50)
        with pytest.raises(ValueError, match=r"at least 47 are needed .* training_stride=24, which skips the 6"):
            fc.fit(y=_series(91), forecasting_horizon=40)

    def test_shortest_accepted_head_keeps_one_instance(self):
        """At exactly the required head length the fit keeps one instance, in phase."""
        y = _series(97)
        fc = PointReductionForecaster(estimator=_EvalSetRecorder(), training_stride=24, validation_size=50)
        fc.fit(y=y, forecasting_horizon=40)
        _, y_received, _ = fc.estimator_.received_

        origins = [int(v) - 1 for v in y_received["value_step_1"].to_list()]
        assert origins == [0]
        assert (len(y) - 1) % 24 == 0

    def test_empty_after_stride_without_holdout_names_the_skip(self):
        """Without a holdout, a series the stride leaves empty fails with the skip count."""
        fc = PointReductionForecaster(estimator=_RecordingRegressor(), training_stride=24)
        with pytest.raises(ValueError, match=r"Training dataset is empty .* skips the 8 most recent"):
            fc.fit(y=_series(45), forecasting_horizon=40)


def _weight_correspondence(weighter_kwargs: dict, stride: int, n: int = 60, horizon: int = 3) -> None:
    """Assert strided sample weights are the strided subset of the full ones.

    Weights are computed on the full instance set and masked in lockstep with
    the training rows, so the strided fit's weight vector must equal the
    unstrided fit's vector filtered by the stride mask, for any weighter.
    """
    y = _series(n)
    full = PointReductionForecaster(estimator=_RecordingRegressor(), **weighter_kwargs)
    full.fit(y=y, forecasting_horizon=horizon)
    full_weights = full.estimator_.received_[2]
    assert full_weights is not None

    strided = PointReductionForecaster(estimator=_RecordingRegressor(), training_stride=stride, **weighter_kwargs)
    strided.fit(y=y, forecasting_horizon=horizon)
    strided_weights = strided.estimator_.received_[2]
    assert strided_weights is not None

    mask = _tail_phase_mask(n, horizon, stride)
    np.testing.assert_allclose(strided_weights, full_weights[mask])


class TestTrainingStrideWeighterCompatibility:
    """The stride composes with every weighter family and alignment.

    The invariant under test is positional: the mask filters instances,
    targets, and weights together, so the strided weight vector is exactly
    the strided subset of the full one, whatever produced the weights.
    """

    @pytest.mark.parametrize(
        "alignment",
        ["first_step", "mean_step", "weighted_mean_step", "max_weight_step", "min_weight_step"],
    )
    def test_time_weighter_alignments(self, alignment):
        from yohou.weighting import ExponentialDecayWeighter

        _weight_correspondence(
            {"time_weighter": ExponentialDecayWeighter(half_life=12), "sample_weight_alignment": alignment},
            stride=5,
        )

    def test_linear_decay_weighter(self):
        from yohou.weighting import LinearDecayWeighter

        _weight_correspondence({"time_weighter": LinearDecayWeighter(max_steps=48)}, stride=6)

    def test_seasonal_emphasis_weighter(self):
        from yohou.weighting import SeasonalEmphasisWeighter

        _weight_correspondence({"time_weighter": SeasonalEmphasisWeighter(seasonality=24, emphasis=3.0)}, stride=4)

    def test_vintage_weighter(self):
        from yohou.weighting import ExponentialDecayWeighter

        _weight_correspondence({"vintage_weighter": ExponentialDecayWeighter(half_life=24)}, stride=5)

    def test_combined_time_and_vintage_weighters(self):
        from yohou.weighting import ExponentialDecayWeighter, LinearDecayWeighter

        _weight_correspondence(
            {
                "time_weighter": LinearDecayWeighter(max_steps=72),
                "vintage_weighter": ExponentialDecayWeighter(half_life=18),
            },
            stride=7,
        )

    def test_table_weighter_covering_every_timestamp(self):
        """A join-keyed weighter needs the full time axis; the stride subsets after."""
        from yohou.weighting import TableWeighter

        n = 60
        y = _series(n)
        frame = pl.DataFrame({"time": y["time"], "weight": [1.0 + (i % 5) for i in range(n)]})
        _weight_correspondence({"vintage_weighter": TableWeighter(frame=frame)}, stride=5, n=n)

    def test_panel_weighted_stride(self):
        """Panel stacking keeps per-group weights aligned with per-group masks."""
        from yohou.weighting import ExponentialDecayWeighter

        n, horizon, stride = 40, 2, 6
        base = _series(n)
        y = pl.DataFrame({
            "time": base["time"],
            "a__value": base["value"],
            "b__value": base["value"] + 100.0,
        })

        def fit(training_stride):
            fc = PointReductionForecaster(
                estimator=_RecordingRegressor(),
                panel_strategy="global",
                time_weighter=ExponentialDecayWeighter(half_life=10),
                training_stride=training_stride,
            )
            fc.fit(y=y, forecasting_horizon=horizon)
            return fc.estimator_.received_

        full_X, _, full_w = fit(1)
        strided_X, _, strided_w = fit(stride)

        n_instances = n - horizon
        group_mask = _tail_phase_mask(n, horizon, stride)
        mask = np.concatenate([group_mask, group_mask])
        assert len(full_X) == 2 * n_instances
        assert len(strided_X) == int(mask.sum())
        np.testing.assert_allclose(strided_w, full_w[mask])

    def test_interval_reduction_weighted_stride(self):
        """Every per-quantile estimator sees the same strided weight subset."""
        from yohou.weighting import ExponentialDecayWeighter

        n, horizon, stride = 50, 2, 5
        y = _series(n)
        fc = IntervalReductionForecaster(
            estimator=_MultiOutputRecording(),
            time_weighter=ExponentialDecayWeighter(half_life=12),
            training_stride=stride,
        )
        fc.fit(y=y, forecasting_horizon=horizon, coverage_rates=[0.5])
        kept = int(_tail_phase_mask(n, horizon, stride).sum())
        weight_vectors = [est.received_[2] for est in fc.estimator_.values()]
        for w in weight_vectors:
            assert w is not None and len(w) == kept
        for w in weight_vectors[1:]:
            np.testing.assert_allclose(w, weight_vectors[0])

    def test_conformal_wrapper_with_weighted_strided_point_forecaster(self):
        """Both strides and a weighter compose end to end through fit and predict."""
        import yohou.interval.split_conformal as sc_module
        from yohou.interval import SplitConformalForecaster
        from yohou.weighting import ExponentialDecayWeighter

        point = PointReductionForecaster(
            estimator=_RecordingRegressor(),
            time_weighter=ExponentialDecayWeighter(half_life=24),
            training_stride=3,
        )
        fc = SplitConformalForecaster(
            point_forecaster=point,
            calibration_size=24,
            calibration_stride=3,
        )
        y = _series(120)
        original_floor = sc_module.MIN_TAIL_SAMPLES
        sc_module.MIN_TAIL_SAMPLES = 1
        try:
            fc.fit(y=y, forecasting_horizon=3, coverage_rates=[0.5])
        finally:
            sc_module.MIN_TAIL_SAMPLES = original_floor
        intervals = fc.predict_interval(coverage_rates=[0.5])
        assert len(intervals) == 3
        assert fc.point_forecaster_.estimator_.received_[2] is not None
