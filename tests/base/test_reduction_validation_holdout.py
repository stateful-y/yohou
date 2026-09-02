"""Tests for the validation_size holdout on reduction forecasters."""

from datetime import datetime, timedelta
from unittest import mock

import numpy as np
import polars as pl
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, clone
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler as SkStandardScaler
from sklearn.utils.validation import check_is_fitted

from yohou.class_proba import ClassProbaReductionForecaster
from yohou.compose import FeaturePipeline
from yohou.point import PointReductionForecaster
from yohou.preprocessing import (
    ExponentialMovingAverage,
    LagTransformer,
    MinMaxScaler,
    RollingStatisticsTransformer,
)
from yohou.stationarity import SeasonalDifferencing
from yohou.weighting import ExponentialDecayWeighter

LENGTH = 50
HORIZON = 3
VAL_SIZE = 10
# Strict mode: the last head anchor plus the tail anchors whose full target
# window fits, so VAL_SIZE - HORIZON + 1 evaluation rows.
STRICT_ROWS = VAL_SIZE - HORIZON + 1


class RecordingRegressor(RegressorMixin, BaseEstimator):
    """Stub with an eval_set fit parameter that records what it received."""

    def fit(self, X, y, eval_set=None, sample_weight=None):
        self.received_eval_set_ = eval_set
        self.received_sample_weight_ = sample_weight
        self.train_X_ = X
        self.train_y_ = y
        arr = np.asarray(y, dtype=float)
        self._ncols = 1 if arr.ndim == 1 else arr.shape[1]
        self._mean = float(np.nanmean(arr))
        return self

    def predict(self, X):
        out = np.full((len(X), self._ncols), self._mean)
        return out.ravel() if self._ncols == 1 else out


class RecordingClassifier(ClassifierMixin, BaseEstimator):
    """Classifier stub with an eval_set fit parameter."""

    def fit(self, X, y, eval_set=None, sample_weight=None):
        self.received_eval_set_ = eval_set
        arr = np.asarray(y, dtype=float)
        self.classes_ = np.unique(arr.ravel())
        self._ncols = 1 if arr.ndim == 1 else arr.shape[1]
        return self

    def predict(self, X):
        out = np.zeros((len(X), self._ncols))
        return out.ravel() if self._ncols == 1 else out

    def predict_proba(self, X):
        return np.tile(np.full(len(self.classes_), 1.0 / len(self.classes_)), (len(X), 1))


class EarlyStoppingStub(RegressorMixin, BaseEstimator):
    """Fake boosting estimator: iterates, scores eval_set, stops on plateau.

    Mimics the boosting-library contract yohou relies on: it consumes
    ``eval_set`` in fit, stops once the validation loss stops improving, and
    exposes the chosen iteration as ``best_iteration_``.
    """

    def __init__(self, max_iterations: int = 50, patience: int = 3):
        self.max_iterations = max_iterations
        self.patience = patience

    def fit(self, X, y, eval_set=None, sample_weight=None):
        arr = np.asarray(y, dtype=float)
        self._ncols = 1 if arr.ndim == 1 else arr.shape[1]
        self._mean = float(np.nanmean(arr))
        self.best_iteration_ = self.max_iterations
        self.evals_result_: list[float] = []
        if eval_set:
            # Each iteration ramps the prediction through and past the training
            # mean, and the loss is scored against the delivered evaluation
            # targets. The curve therefore bottoms out where the prediction
            # best matches those targets, so a wrong evaluation set stops at a
            # different iteration instead of silently passing.
            y_eval = np.asarray(eval_set[0][1], dtype=float).ravel()
            best, best_iter, stale = np.inf, 0, 0
            for i in range(1, self.max_iterations + 1):
                prediction = self._mean * i / 10.0
                loss = float(np.nanmean(np.abs(y_eval - prediction)))
                self.evals_result_.append(loss)
                if loss < best:
                    best, best_iter, stale = loss, i, 0
                else:
                    stale += 1
                    if stale >= self.patience:
                        break
            self.best_iteration_ = best_iter
        return self

    def predict(self, X):
        out = np.full((len(X), self._ncols), self._mean)
        return out.ravel() if self._ncols == 1 else out


def _times(length: int) -> pl.Series:
    return pl.datetime_range(
        start=datetime(2021, 1, 1),
        end=datetime(2021, 1, 1) + timedelta(seconds=length - 1),
        interval="1s",
        eager=True,
    )


def _make_y(length: int = LENGTH) -> pl.DataFrame:
    return pl.DataFrame({
        "time": _times(length),
        "value": [float(i) for i in range(length)],
    })


def _make_y_panel(length: int = LENGTH) -> pl.DataFrame:
    return pl.DataFrame({
        "time": _times(length),
        "a__value": [float(i) for i in range(length)],
        "b__value": [float(2 * i) for i in range(length)],
    })


def _eval_pair(estimator):
    (X_eval, y_eval) = estimator.received_eval_set_[0]
    return X_eval, y_eval


class TestDeliveryShape:
    """Task 5.1: delivery shape per strategy, on standard and panel data."""

    @pytest.mark.parametrize("panel", [False, True], ids=["standard", "panel"])
    @pytest.mark.parametrize("strategy", ["multi-output", "direct", "dir-rec"])
    def test_strategy_delivery(self, strategy, panel):
        y = _make_y_panel() if panel else _make_y()
        n_groups = 2 if panel else 1
        forecaster = PointReductionForecaster(
            estimator=RecordingRegressor(),
            reduction_strategy=strategy,
            validation_size=VAL_SIZE,
        )
        forecaster.fit(y=y, forecasting_horizon=HORIZON)

        estimators = forecaster.estimator_ if isinstance(forecaster.estimator_, list) else [forecaster.estimator_]
        assert len(estimators) == (HORIZON if strategy != "multi-output" else 1)

        for step, est in enumerate(estimators):
            X_eval, y_eval = _eval_pair(est)
            assert len(X_eval) == STRICT_ROWS * n_groups
            assert list(X_eval.columns) == list(est.train_X_.columns)
            if strategy == "multi-output":
                # The global panel strategy stacks groups vertically, so the
                # target keeps one column per (base target, step) pair.
                assert y_eval.shape == (STRICT_ROWS * n_groups, HORIZON)
            else:
                # Per-step targets are a single column, delivered as a series
                # to mirror training.
                assert isinstance(y_eval, pl.Series)
            if strategy == "dir-rec":
                aug = [c for c in X_eval.columns if c.startswith("__aug_")]
                assert len(aug) == step

    def test_panel_tail_membership(self):
        y = _make_y_panel()
        forecaster = PointReductionForecaster(estimator=RecordingRegressor(), validation_size=VAL_SIZE)
        forecaster.fit(y=y, forecasting_horizon=HORIZON)
        X_eval, y_eval = _eval_pair(forecaster.estimator_)
        # Per-group blocks stacked in groups_ order: group a rows, then group b.
        assert len(X_eval) == 2 * STRICT_ROWS
        # Group b values are 2x group a; targets of the b block must all be
        # even and >= 2 * (head length), proving the tail split ran per group.
        head_len = LENGTH - VAL_SIZE
        b_block = y_eval[STRICT_ROWS:]
        assert b_block.to_numpy().min() >= 2 * head_len

    def test_step_columns_reach_eval_rows(self):
        y = _make_y()
        t_ext = pl.datetime_range(
            start=datetime(2021, 1, 1),
            end=datetime(2021, 1, 1, 0, 0, LENGTH + HORIZON - 1),
            interval="1s",
            eager=True,
        )
        X_future = pl.DataFrame({"time": t_ext, "temp": [float(i % 7) for i in range(len(t_ext))]})
        forecaster = PointReductionForecaster(estimator=RecordingRegressor(), validation_size=VAL_SIZE)
        forecaster.fit(y=y, forecasting_horizon=HORIZON, X_future=X_future)
        X_eval, _ = _eval_pair(forecaster.estimator_)
        step_cols = [c for c in X_eval.columns if "_step_" in c]
        assert step_cols == [f"temp_step_{k}" for k in range(1, HORIZON + 1)]
        assert list(X_eval.columns) == list(forecaster.estimator_.train_X_.columns)
        assert X_eval.select(step_cols).null_count().sum_horizontal().item() == 0

    def test_stride_does_not_thin_eval(self):
        y = _make_y()
        forecaster = PointReductionForecaster(
            estimator=RecordingRegressor(), validation_size=VAL_SIZE, training_stride=3
        )
        forecaster.fit(y=y, forecasting_horizon=HORIZON)
        X_eval, _ = _eval_pair(forecaster.estimator_)
        assert len(X_eval) == STRICT_ROWS
        # Training rows, by contrast, are thinned by the stride.
        assert len(forecaster.estimator_.train_X_) < LENGTH - VAL_SIZE - HORIZON

    def test_all_eval_rows_nan_error_is_validation_specific(self):
        """training_stride never thins eval rows, so its hint must not appear."""
        values = [float(i) for i in range(LENGTH)]
        for i in range(LENGTH - VAL_SIZE - 1, LENGTH):
            values[i] = float("nan")
        y = _make_y().with_columns(pl.Series("value", values))
        forecaster = PointReductionForecaster(
            estimator=RecordingRegressor(),
            validation_size=VAL_SIZE,
            nan_handling="drop",
            training_stride=3,
        )
        with pytest.raises(ValueError) as excinfo:
            forecaster.fit(y=y, forecasting_horizon=HORIZON)
        message = str(excinfo.value)
        assert "validation instances" in message
        assert "training_stride" not in message

    def test_nan_eval_row_dropped_with_validation_context(self):
        values = [float(i) for i in range(LENGTH)]
        values[LENGTH - 1] = float("nan")
        y = _make_y().with_columns(pl.Series("value", values))
        forecaster = PointReductionForecaster(
            estimator=RecordingRegressor(), validation_size=VAL_SIZE, nan_handling="drop"
        )
        with pytest.warns(UserWarning, match=r"validation instances"):
            forecaster.fit(y=y, forecasting_horizon=HORIZON)
        X_eval, _ = _eval_pair(forecaster.estimator_)
        assert len(X_eval) < STRICT_ROWS


class TestBoundaryPolicy:
    """Task 5.2: strict versus overlap anchor selection."""

    def test_strict_no_training_target_overlap(self):
        y = _make_y()
        forecaster = PointReductionForecaster(estimator=RecordingRegressor(), validation_size=VAL_SIZE)
        forecaster.fit(y=y, forecasting_horizon=HORIZON)
        est = forecaster.estimator_
        head_len = LENGTH - VAL_SIZE
        train_targets = np.asarray(est.train_y_, dtype=float)
        eval_targets = np.asarray(_eval_pair(est)[1], dtype=float)
        # Values equal their index, so target values identify target times.
        assert train_targets.max() == head_len - 1
        assert eval_targets.min() >= head_len
        assert len(_eval_pair(est)[0]) == STRICT_ROWS

    def test_overlap_adds_straddling_rows(self):
        y = _make_y()
        forecaster = PointReductionForecaster(
            estimator=RecordingRegressor(), validation_size=VAL_SIZE, validation_overlap=True
        )
        forecaster.fit(y=y, forecasting_horizon=HORIZON)
        est = forecaster.estimator_
        X_eval, y_eval = _eval_pair(est)
        assert len(X_eval) == VAL_SIZE
        head_len = LENGTH - VAL_SIZE
        eval_targets = np.asarray(y_eval, dtype=float)
        # The straddling anchors pull head time points into the eval targets.
        assert eval_targets.min() < head_len

    def test_overlap_allows_small_holdout(self):
        y = _make_y()
        forecaster = PointReductionForecaster(
            estimator=RecordingRegressor(), validation_size=2, validation_overlap=True
        )
        forecaster.fit(y=y, forecasting_horizon=HORIZON)
        assert len(_eval_pair(forecaster.estimator_)[0]) == 2


class TestLeakage:
    """Task 5.3: nothing fitted sees the holdout tail."""

    def test_transformer_statistics_head_only(self):
        y = _make_y()
        forecaster = PointReductionForecaster(
            estimator=RecordingRegressor(),
            validation_size=VAL_SIZE,
            target_transformer=MinMaxScaler(),
        )
        forecaster.fit(y=y, forecasting_horizon=HORIZON)
        head_ref = MinMaxScaler().fit(y[:-VAL_SIZE]).instance_
        got = forecaster.target_transformer_.instance_
        np.testing.assert_allclose(got.data_max_, head_ref.data_max_)
        # The head maximum, not the full-series maximum.
        assert got.data_max_[0] == float(LENGTH - VAL_SIZE - 1)

    def test_sample_weights_head_only(self):
        y = _make_y()
        with_holdout = PointReductionForecaster(
            estimator=RecordingRegressor(),
            validation_size=VAL_SIZE,
            time_weighter=ExponentialDecayWeighter(half_life=5),
        )
        with_holdout.fit(y=y, forecasting_horizon=HORIZON)
        head_only = PointReductionForecaster(
            estimator=RecordingRegressor(),
            time_weighter=ExponentialDecayWeighter(half_life=5),
        )
        head_only.fit(y=y[:-VAL_SIZE], forecasting_horizon=HORIZON)
        np.testing.assert_allclose(
            with_holdout.estimator_.received_sample_weight_,
            head_only.estimator_.received_sample_weight_,
        )

    def test_eval_lag_warmup_from_head(self):
        y = _make_y()
        forecaster = PointReductionForecaster(
            estimator=RecordingRegressor(),
            validation_size=VAL_SIZE,
            actual_transformer=LagTransformer(lag=[1, 2]),
        )
        forecaster.fit(y=y, forecasting_horizon=HORIZON)
        X_eval, _ = _eval_pair(forecaster.estimator_)
        lag_cols = [c for c in X_eval.columns if "lag" in c]
        assert lag_cols, X_eval.columns
        # The first eval anchor is the last head row; its lag features reach
        # further into the head and must be present, not null.
        assert X_eval.select(lag_cols).null_count().sum_horizontal().item() == 0
        head_len = LENGTH - VAL_SIZE
        first_row_lags = sorted(X_eval[0].select(lag_cols).row(0))
        assert first_row_lags == [float(head_len - 3), float(head_len - 2)]

    def test_eval_rows_respect_vintage_availability(self):
        y = _make_y()
        boundary_idx = LENGTH - VAL_SIZE
        times = y["time"]
        # Vintage 1 published before the boundary, vintage 2 inside the tail.
        v1_time = times[boundary_idx - 5]
        v2_time = times[boundary_idx + 4]
        horizon_times = pl.datetime_range(
            start=datetime(2021, 1, 1),
            end=datetime(2021, 1, 1, 0, 0, LENGTH + HORIZON - 1),
            interval="1s",
            eager=True,
        )
        X_forecast = pl.concat([
            pl.DataFrame({
                "vintage_time": [vt] * len(horizon_times),
                "time": horizon_times,
                "fx": [val] * len(horizon_times),
            })
            for vt, val in [(v1_time, 1.0), (v2_time, 2.0)]
        ])
        forecaster = PointReductionForecaster(estimator=RecordingRegressor(), validation_size=VAL_SIZE)
        forecaster.fit(y=y, forecasting_horizon=HORIZON, X_forecast=X_forecast)
        X_eval, _ = _eval_pair(forecaster.estimator_)
        col = "fx_step_1"
        assert col in X_eval.columns
        # Strict eval anchors are the last head row then the tail rows: anchor
        # times boundary_idx - 1 .. boundary_idx + VAL_SIZE - HORIZON - 1.
        anchor_indices = list(range(boundary_idx - 1, boundary_idx + VAL_SIZE - HORIZON))
        expected = [2.0 if times[i] >= v2_time else 1.0 for i in anchor_indices]
        assert X_eval[col].to_list() == expected


class TestErrorContract:
    """Task 5.4: the six ValueError cases."""

    def test_estimator_without_eval_set(self):
        with pytest.raises(ValueError, match="does not support an eval_set"):
            PointReductionForecaster(estimator=LinearRegression(), validation_size=VAL_SIZE).fit(
                y=_make_y(), forecasting_horizon=HORIZON
            )

    def test_multioutput_wrapper_rejected(self):
        with pytest.raises(ValueError, match="MultiOutputRegressor"):
            PointReductionForecaster(estimator=MultiOutputRegressor(LinearRegression()), validation_size=VAL_SIZE).fit(
                y=_make_y(), forecasting_horizon=HORIZON
            )

    def test_head_too_small(self):
        with pytest.raises(ValueError, match="head rows"):
            PointReductionForecaster(estimator=RecordingRegressor(), validation_size=LENGTH - 2).fit(
                y=_make_y(), forecasting_horizon=HORIZON
            )

    def test_strict_holdout_too_small(self):
        with pytest.raises(ValueError, match="validation_overlap"):
            PointReductionForecaster(estimator=RecordingRegressor(), validation_size=HORIZON - 1).fit(
                y=_make_y(), forecasting_horizon=HORIZON
            )

    def test_tail_only_class(self):
        base = ["lo", "hi"] * ((LENGTH - 5) // 2)
        states = base + ["new"] * (LENGTH - len(base))
        y = _make_y().with_columns(pl.Series("state", states)).drop("value")
        with pytest.raises(ValueError, match="only inside the validation_size holdout"):
            ClassProbaReductionForecaster(estimator=RecordingClassifier(), validation_size=VAL_SIZE).fit(
                y=y, forecasting_horizon=2
            )

    def test_raw_eval_set_passthrough_conflict(self):
        with pytest.raises(ValueError, match="raw eval_set"):
            PointReductionForecaster(estimator=RecordingRegressor(), validation_size=VAL_SIZE).fit(
                y=_make_y(), forecasting_horizon=HORIZON, eval_set="raw"
            )

    def test_pipeline_final_step_without_eval_set_fails_before_mutation(self):
        """A Pipeline that cannot evaluate must fail before any state is touched."""
        forecaster = PointReductionForecaster(
            estimator=Pipeline([("scaler", SkStandardScaler()), ("lr", LinearRegression())]),
            validation_size=VAL_SIZE,
        )
        with pytest.raises(ValueError, match="Pipeline's final step LinearRegression"):
            forecaster.fit(y=_make_y(), forecasting_horizon=HORIZON)
        assert not hasattr(forecaster, "observed_time_")
        assert not hasattr(forecaster, "estimator_")

    def test_pipeline_ending_in_passthrough_rejected(self):
        forecaster = PointReductionForecaster(
            estimator=Pipeline([("scaler", SkStandardScaler()), ("nothing", "passthrough")]),
            validation_size=VAL_SIZE,
        )
        with pytest.raises(ValueError, match="passthrough"):
            forecaster.fit(y=_make_y(), forecasting_horizon=HORIZON)


class TestPipelineEstimator:
    """Task 1: a wrapped Pipeline evaluates in its own transformed space."""

    def test_eval_set_is_transformed_like_training(self):
        forecaster = PointReductionForecaster(
            estimator=Pipeline([("scaler", SkStandardScaler()), ("rec", RecordingRegressor())]),
            validation_size=VAL_SIZE,
        )
        forecaster.fit(y=_make_y(), forecasting_horizon=HORIZON)

        assert isinstance(forecaster.estimator_, Pipeline)
        check_is_fitted(forecaster.estimator_)
        rec = forecaster.estimator_.named_steps["rec"]
        X_eval = np.asarray(_eval_pair(rec)[0])
        X_train = np.asarray(rec.train_X_)
        assert len(X_eval) == STRICT_ROWS
        assert X_eval.shape[1] == X_train.shape[1]
        # Both sides are standardized: raw targets run to LENGTH, so an
        # untransformed eval matrix would carry values far outside this band.
        assert np.abs(X_train).max() < 6.0
        assert np.abs(X_eval).max() < 6.0
        assert forecaster.predict().height == HORIZON

    def test_pipeline_transformers_fitted_on_head_only(self):
        forecaster = PointReductionForecaster(
            estimator=Pipeline([("scaler", SkStandardScaler()), ("rec", RecordingRegressor())]),
            validation_size=VAL_SIZE,
        )
        forecaster.fit(y=_make_y(), forecasting_horizon=HORIZON)
        fitted_mean = forecaster.estimator_.named_steps["scaler"].mean_

        head_only = PointReductionForecaster(
            estimator=Pipeline([("scaler", SkStandardScaler()), ("rec", RecordingRegressor())]),
        )
        head_only.fit(y=_make_y()[: LENGTH - VAL_SIZE], forecasting_horizon=HORIZON)
        assert np.allclose(fitted_mean, head_only.estimator_.named_steps["scaler"].mean_)

    def test_sample_weight_and_eval_set_reach_final_step(self):
        forecaster = PointReductionForecaster(
            estimator=Pipeline([("scaler", SkStandardScaler()), ("rec", RecordingRegressor())]),
            validation_size=VAL_SIZE,
            time_weighter=ExponentialDecayWeighter(half_life=5),
        )
        forecaster.fit(y=_make_y(), forecasting_horizon=HORIZON)
        rec = forecaster.estimator_.named_steps["rec"]
        assert rec.received_eval_set_ is not None
        assert rec.received_sample_weight_ is not None
        assert len(rec.received_sample_weight_) == len(rec.train_X_)

    def test_single_step_pipeline(self):
        """A pipeline with no transformer prefix still delivers an eval set."""
        forecaster = PointReductionForecaster(
            estimator=Pipeline([("rec", RecordingRegressor())]),
            validation_size=VAL_SIZE,
        )
        forecaster.fit(y=_make_y(), forecasting_horizon=HORIZON)
        rec = forecaster.estimator_.named_steps["rec"]
        assert len(_eval_pair(rec)[0]) == STRICT_ROWS


class TestParameterOwnership:
    """The holdout parameters exist only on the families that expose them."""

    def test_interval_family_carries_no_trace(self):
        from yohou.interval import IntervalReductionForecaster

        forecaster = IntervalReductionForecaster()
        for name in ("validation_size", "validation_overlap"):
            assert name not in forecaster.get_params(deep=False)
            assert name not in forecaster._parameter_constraints
            assert name not in forecaster.__dict__

    @pytest.mark.parametrize(
        "cls, estimator",
        [(PointReductionForecaster, RecordingRegressor()), (ClassProbaReductionForecaster, RecordingClassifier())],
        ids=["point", "class_proba"],
    )
    def test_exposing_families_round_trip(self, cls, estimator):
        forecaster = cls(estimator=estimator, validation_size=VAL_SIZE, validation_overlap=True)
        params = forecaster.get_params(deep=False)
        assert params["validation_size"] == VAL_SIZE
        assert params["validation_overlap"] is True
        assert clone(forecaster).get_params(deep=False)["validation_size"] == VAL_SIZE


class TestDefaultNoOp:
    """Task 5.5: validation_size=None is byte-equivalent to omitting it."""

    def test_no_op_equivalence_and_no_holdout_code_paths(self, mocker):
        y = _make_y()
        spy_prepare = mocker.spy(PointReductionForecaster, "_prepare_validation_fit")
        spy_build = mocker.spy(PointReductionForecaster, "_build_validation_eval_data")
        spy_resolve = mocker.spy(PointReductionForecaster, "_check_eval_set_support")

        explicit = PointReductionForecaster(estimator=LinearRegression(), validation_size=None)
        explicit.fit(y=y, forecasting_horizon=HORIZON)
        omitted = PointReductionForecaster(estimator=LinearRegression())
        omitted.fit(y=y, forecasting_horizon=HORIZON)

        assert spy_prepare.call_count == 0
        assert spy_build.call_count == 0
        assert spy_resolve.call_count == 0
        assert explicit.predict().equals(omitted.predict())


class TestPostFitState:
    """Task 5.6: observation state ends at the end of all provided data."""

    @pytest.mark.parametrize("panel", [False, True], ids=["standard", "panel"])
    def test_predict_starts_after_all_data(self, panel):
        y = _make_y_panel() if panel else _make_y()
        forecaster = PointReductionForecaster(estimator=RecordingRegressor(), validation_size=VAL_SIZE)
        forecaster.fit(y=y, forecasting_horizon=HORIZON)
        last_time = y["time"][-1]
        observed = forecaster.observed_time_
        if panel:
            assert all(t == last_time for t in observed.values())
        else:
            assert observed == last_time
        prediction = forecaster.predict()
        assert prediction["time"].min() > last_time


class TestEarlyStopping:
    """Tasks 5.7 and 5.8: stopping behavior, faked and real."""

    def test_fake_estimator_stops_below_maximum(self):
        forecaster = PointReductionForecaster(
            estimator=EarlyStoppingStub(max_iterations=50, patience=3),
            validation_size=VAL_SIZE,
        )
        forecaster.fit(y=_make_y(), forecasting_horizon=HORIZON)
        estimator = forecaster.estimator_
        assert estimator.best_iteration_ < 50
        # The stop lands on the iteration whose loss is the minimum actually
        # observed, so the reported best iteration is the scored one.
        assert estimator.evals_result_[estimator.best_iteration_ - 1] == min(estimator.evals_result_)

    def test_stopping_iteration_depends_on_eval_content(self):
        """A different evaluation set must move the stopping iteration."""
        X = np.zeros((8, 1))
        y = np.arange(8.0)
        near = EarlyStoppingStub(max_iterations=50, patience=3).fit(X, y, eval_set=[(X, np.full(8, 4.0))])
        far = EarlyStoppingStub(max_iterations=50, patience=3).fit(X, y, eval_set=[(X, np.full(8, 20.0))])
        assert near.best_iteration_ != far.best_iteration_

    def test_fake_estimator_without_holdout_runs_to_maximum(self):
        forecaster = PointReductionForecaster(estimator=EarlyStoppingStub(max_iterations=50))
        forecaster.fit(y=_make_y(), forecasting_horizon=HORIZON)
        assert forecaster.estimator_.best_iteration_ == 50

    def test_lightgbm_early_stopping_triggers(self):
        # In the tests group, so normally present; macOS wheels still need the
        # libomp runtime at import, which a presence check cannot detect.
        lightgbm = pytest.importorskip("lightgbm")
        LGBMRegressor = lightgbm.LGBMRegressor

        rng = np.random.default_rng(0)
        length = 400
        y = pl.DataFrame({
            "time": pl.datetime_range(
                start=datetime(2021, 1, 1),
                end=datetime(2021, 1, 1, 0, 6, length - 1 - 360),
                interval="1s",
                eager=True,
            ),
            "value": (np.sin(np.arange(length) / 5.0) + rng.normal(0, 0.05, length)).tolist(),
        })
        estimator = LGBMRegressor(
            n_estimators=300,
            early_stopping_round=10,
            min_child_samples=5,
            verbose=-1,
        )
        forecaster = PointReductionForecaster(
            estimator=estimator,
            reduction_strategy="direct",
            actual_transformer=LagTransformer(lag=[1, 2, 3]),
            validation_size=60,
        )
        forecaster.fit(y=y, forecasting_horizon=2)
        for est in forecaster.estimator_:
            assert est.best_iteration_ is not None
            assert est.best_iteration_ < 300


class TestEquivalenceOracles:
    """Strong oracles for the evaluation-row construction.

    With data-independent transformers, the evaluation rows must equal the
    last rows of a no-holdout fit's tabularization on the full series
    (oracle A). With any data-dependent or stateful transformer, fitting with
    a holdout must be indistinguishable from fitting on the head and then
    observing the tail (oracle B).
    """

    @pytest.mark.parametrize(
        "kwargs",
        [
            {},
            {"actual_transformer": LagTransformer(lag=[1, 3])},
            {"reduction_strategy": "direct", "actual_transformer": LagTransformer(lag=[1, 2])},
        ],
        ids=["plain", "lags", "direct-lags"],
    )
    def test_eval_rows_equal_full_fit_tail(self, kwargs):
        y = _make_y()
        holdout = PointReductionForecaster(estimator=RecordingRegressor(), validation_size=VAL_SIZE, **kwargs)
        holdout.fit(y=y, forecasting_horizon=HORIZON)
        full = PointReductionForecaster(estimator=RecordingRegressor(), **kwargs)
        full.fit(y=y, forecasting_horizon=HORIZON)

        holdout_ests = holdout.estimator_ if isinstance(holdout.estimator_, list) else [holdout.estimator_]
        full_ests = full.estimator_ if isinstance(full.estimator_, list) else [full.estimator_]
        for est_holdout, est_full in zip(holdout_ests, full_ests, strict=True):
            X_eval, y_eval = _eval_pair(est_holdout)
            pl.testing.assert_frame_equal(X_eval, est_full.train_X_[-STRICT_ROWS:])
            y_eval_frame = y_eval.to_frame() if isinstance(y_eval, pl.Series) else y_eval
            y_full = est_full.train_y_
            y_full_frame = y_full.to_frame() if isinstance(y_full, pl.Series) else y_full
            pl.testing.assert_frame_equal(y_eval_frame, y_full_frame[-STRICT_ROWS:])

    def test_eval_rows_equal_full_fit_tail_with_x_future(self):
        y = _make_y()
        t_ext = pl.datetime_range(
            start=datetime(2021, 1, 1),
            end=datetime(2021, 1, 1, 0, 0, LENGTH + HORIZON - 1),
            interval="1s",
            eager=True,
        )
        X_future = pl.DataFrame({"time": t_ext, "temp": [float(i % 7) for i in range(len(t_ext))]})
        holdout = PointReductionForecaster(estimator=RecordingRegressor(), validation_size=VAL_SIZE)
        holdout.fit(y=y, forecasting_horizon=HORIZON, X_future=X_future)
        full = PointReductionForecaster(estimator=RecordingRegressor())
        full.fit(y=y, forecasting_horizon=HORIZON, X_future=X_future)
        X_eval, _ = _eval_pair(holdout.estimator_)
        pl.testing.assert_frame_equal(X_eval, full.estimator_.train_X_[-STRICT_ROWS:])

    def test_eval_rows_equal_full_fit_tail_panel(self):
        y = _make_y_panel()
        holdout = PointReductionForecaster(estimator=RecordingRegressor(), validation_size=VAL_SIZE)
        holdout.fit(y=y, forecasting_horizon=HORIZON)
        full = PointReductionForecaster(estimator=RecordingRegressor())
        full.fit(y=y, forecasting_horizon=HORIZON)
        X_eval, _ = _eval_pair(holdout.estimator_)
        X_full = full.estimator_.train_X_
        # Full-fit tabularization stacks whole per-group blocks, so the tail
        # comparison must slice per group.
        block = LENGTH - HORIZON
        pl.testing.assert_frame_equal(X_eval[:STRICT_ROWS], X_full[block - STRICT_ROWS : block])
        pl.testing.assert_frame_equal(X_eval[STRICT_ROWS:], X_full[2 * block - STRICT_ROWS :])

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"target_transformer": MinMaxScaler()},
            {"target_transformer": SeasonalDifferencing(seasonality=4)},
            {"actual_transformer": ExponentialMovingAverage(alpha=0.5)},
            {
                "target_transformer": MinMaxScaler(),
                "actual_transformer": FeaturePipeline(
                    steps=[
                        ("lag", LagTransformer(lag=[1, 2])),
                        ("roll", RollingStatisticsTransformer(window_size=3)),
                    ]
                ),
            },
        ],
        ids=["scaler", "seasonal-diff", "ema", "scaler+pipeline"],
    )
    def test_validation_fit_equals_fit_then_observe(self, kwargs):
        y = _make_y()
        holdout = PointReductionForecaster(estimator=RecordingRegressor(), validation_size=VAL_SIZE, **kwargs)
        holdout.fit(y=y, forecasting_horizon=HORIZON)

        reference = PointReductionForecaster(estimator=RecordingRegressor(), **kwargs)
        reference.fit(y=y[:-VAL_SIZE], forecasting_horizon=HORIZON)
        reference.observe(y[-VAL_SIZE:])

        assert holdout.observed_time_ == reference.observed_time_
        if holdout._y_observed is None:
            assert reference._y_observed is None
        else:
            pl.testing.assert_frame_equal(holdout._y_observed, reference._y_observed)
        pl.testing.assert_frame_equal(holdout.predict(), reference.predict())

    def test_validation_fit_equals_fit_then_observe_panel(self):
        y = _make_y_panel()
        kwargs = {"target_transformer": MinMaxScaler(), "actual_transformer": LagTransformer(lag=[1, 2])}
        holdout = PointReductionForecaster(estimator=RecordingRegressor(), validation_size=VAL_SIZE, **kwargs)
        holdout.fit(y=y, forecasting_horizon=HORIZON)

        reference = PointReductionForecaster(estimator=RecordingRegressor(), **kwargs)
        reference.fit(y=y[:-VAL_SIZE], forecasting_horizon=HORIZON)
        reference.observe(y[-VAL_SIZE:])

        assert holdout.observed_time_ == reference.observed_time_
        for group in holdout.groups_:
            pl.testing.assert_frame_equal(holdout._y_observed[group], reference._y_observed[group])
        pl.testing.assert_frame_equal(holdout.predict(), reference.predict())


class TestTransformerMatrix:
    """Specific configurations across the transformer and option matrix."""

    def test_matched_alignment_filters_eval_features(self):
        y = _make_y()
        t_ext = pl.datetime_range(
            start=datetime(2021, 1, 1),
            end=datetime(2021, 1, 1, 0, 0, LENGTH + HORIZON - 1),
            interval="1s",
            eager=True,
        )
        X_future = pl.DataFrame({"time": t_ext, "temp": [float(i % 7) for i in range(len(t_ext))]})
        forecaster = PointReductionForecaster(
            estimator=RecordingRegressor(),
            reduction_strategy="direct",
            step_feature_alignment="matched",
            validation_size=VAL_SIZE,
        )
        forecaster.fit(y=y, forecasting_horizon=HORIZON, X_future=X_future)
        for step, est in enumerate(forecaster.estimator_, start=1):
            X_eval, _ = _eval_pair(est)
            step_cols = [c for c in X_eval.columns if "_step_" in c]
            assert step_cols == [f"temp_step_{step}"]
            assert list(X_eval.columns) == list(est.train_X_.columns)

    @pytest.mark.parametrize("target_as_feature", ["raw", None], ids=["raw", "none"])
    def test_target_as_feature_variants(self, target_as_feature):
        y = _make_y()
        X_actual = _make_y().rename({"value": "exo"})
        forecaster = PointReductionForecaster(
            estimator=RecordingRegressor(),
            target_as_feature=target_as_feature,
            actual_transformer=LagTransformer(lag=[1, 2]),
            validation_size=VAL_SIZE,
        )
        forecaster.fit(y=y, X_actual=X_actual, forecasting_horizon=HORIZON)
        X_eval, y_eval = _eval_pair(forecaster.estimator_)
        assert len(X_eval) == STRICT_ROWS
        assert list(X_eval.columns) == list(forecaster.estimator_.train_X_.columns)
        # The cases differ in whether the target itself is carried as a feature.
        target_features = [c for c in X_eval.columns if "value" in c]
        if target_as_feature is None:
            assert target_features == []
        else:
            assert target_features != []

    def test_panel_multivariate_strategy(self):
        y = _make_y_panel()
        forecaster = PointReductionForecaster(
            estimator=RecordingRegressor(),
            panel_strategy="multivariate",
            validation_size=VAL_SIZE,
        )
        forecaster.fit(y=y, forecasting_horizon=HORIZON)
        X_eval, y_eval = _eval_pair(forecaster.estimator_)
        # Multivariate treats the panel as one wide series: no group stacking,
        # one eval block, H steps for each of the two targets.
        assert forecaster.groups_ is None
        assert len(X_eval) == STRICT_ROWS
        assert y_eval.shape == (STRICT_ROWS, HORIZON * 2)

    def test_multivariate_direct_delivers_two_column_targets(self):
        y = _make_y().with_columns((pl.col("value") * 2.0).alias("other"))
        forecaster = PointReductionForecaster(
            estimator=RecordingRegressor(),
            reduction_strategy="direct",
            validation_size=VAL_SIZE,
        )
        forecaster.fit(y=y, forecasting_horizon=HORIZON)
        for est in forecaster.estimator_:
            _, y_eval = _eval_pair(est)
            assert isinstance(y_eval, pl.DataFrame)
            assert y_eval.shape == (STRICT_ROWS, 2)

    def test_dir_rec_panel_with_x_future(self):
        y = _make_y_panel()
        t_ext = pl.datetime_range(
            start=datetime(2021, 1, 1),
            end=datetime(2021, 1, 1, 0, 0, LENGTH + HORIZON - 1),
            interval="1s",
            eager=True,
        )
        X_future = pl.DataFrame({"time": t_ext, "temp": [float(i % 7) for i in range(len(t_ext))]})
        forecaster = PointReductionForecaster(
            estimator=RecordingRegressor(),
            reduction_strategy="dir-rec",
            validation_size=VAL_SIZE,
        )
        forecaster.fit(y=y, forecasting_horizon=HORIZON, X_future=X_future)
        for step, est in enumerate(forecaster.estimator_):
            X_eval, _ = _eval_pair(est)
            assert len(X_eval) == 2 * STRICT_ROWS
            assert list(X_eval.columns) == list(est.train_X_.columns)
            aug = [c for c in X_eval.columns if c.startswith("__aug_")]
            assert len(aug) == step

    def test_class_proba_panel(self):
        labels = ["lo", "hi"]
        y = pl.DataFrame({
            "time": _make_y()["time"],
            "a__state": [labels[i % 2] for i in range(LENGTH)],
            "b__state": [labels[(i + 1) % 2] for i in range(LENGTH)],
        })
        forecaster = ClassProbaReductionForecaster(
            estimator=RecordingClassifier(), validation_size=VAL_SIZE, reduction_strategy="direct"
        )
        forecaster.fit(y=y, forecasting_horizon=2)
        for est in forecaster.estimator_:
            X_eval, _ = _eval_pair(est)
            assert len(X_eval) == 2 * (VAL_SIZE - 2 + 1)


class TestLifecycleComposition:
    """The holdout composes with the rest of the estimator lifecycle."""

    def test_observe_after_validation_fit(self):
        y = _make_y()
        forecaster = PointReductionForecaster(
            estimator=RecordingRegressor(),
            target_transformer=MinMaxScaler(),
            validation_size=VAL_SIZE,
        )
        forecaster.fit(y=y, forecasting_horizon=HORIZON)
        # Genuinely new data after the tail must observe cleanly: the tail was
        # observed exactly once during fit, so there is no overlap.
        extension = pl.DataFrame({
            "time": pl.datetime_range(
                start=datetime(2021, 1, 1, 0, 0, LENGTH),
                end=datetime(2021, 1, 1, 0, 0, LENGTH + 4),
                interval="1s",
                eager=True,
            ),
            "value": [float(LENGTH + i) for i in range(5)],
        })
        forecaster.observe(extension)
        assert forecaster.observed_time_ == extension["time"][-1]

    def test_rewind_after_validation_fit(self):
        y = _make_y()
        forecaster = PointReductionForecaster(
            estimator=RecordingRegressor(),
            target_transformer=MinMaxScaler(),
            validation_size=VAL_SIZE,
        )
        forecaster.fit(y=y, forecasting_horizon=HORIZON)
        # The seam a future calibration-reuse wrapper needs: rewinding to the
        # head/tail boundary after a validation fit.
        forecaster.rewind(y[:-VAL_SIZE])
        assert forecaster.observed_time_ == y["time"][LENGTH - VAL_SIZE - 1]

    def test_grid_search_composes_without_routing(self):
        from yohou.metrics import MeanAbsoluteError
        from yohou.model_selection import ExpandingWindowSplitter, GridSearchCV

        y = _make_y(length=120)
        forecaster = PointReductionForecaster(estimator=RecordingRegressor(), validation_size=VAL_SIZE)
        search = GridSearchCV(
            forecaster=forecaster,
            param_grid={"reduction_strategy": ["multi-output", "direct"]},
            scoring=MeanAbsoluteError(),
            cv=ExpandingWindowSplitter(n_splits=2, test_size=10),
        )
        fold_eval_sizes: list[int] = []
        original_fit = PointReductionForecaster.fit

        def record_fold(self, *args, **kwargs):
            fitted = original_fit(self, *args, **kwargs)
            estimators = fitted.estimator_ if isinstance(fitted.estimator_, list) else [fitted.estimator_]
            for est in estimators:
                assert est.received_eval_set_ is not None, "a fold fit received no eval_set"
                fold_eval_sizes.append(len(_eval_pair(est)[0]))
            return fitted

        with mock.patch.object(PointReductionForecaster, "fit", record_fold):
            search.fit(y, forecasting_horizon=HORIZON)

        # Every inner fit (each CV fold, plus the final refit) held out its own
        # tail, so the eval sets are the strict-mode size rather than absent.
        assert len(fold_eval_sizes) > 1, "expected per-fold fits, not just the final refit"
        assert set(fold_eval_sizes) == {STRICT_ROWS}
        best = search.best_forecaster_
        estimators = best.estimator_ if isinstance(best.estimator_, list) else [best.estimator_]
        assert all(est.received_eval_set_ is not None for est in estimators)

    def test_split_conformal_wraps_validation_fit(self):
        from yohou.interval import SplitConformalForecaster

        y = _make_y(length=120)
        forecaster = SplitConformalForecaster(
            point_forecaster=PointReductionForecaster(
                estimator=RecordingRegressor(),
                actual_transformer=LagTransformer(lag=[1, 2]),
                validation_size=VAL_SIZE,
            ),
            calibration_size=30,
        )
        forecaster.fit(y=y, forecasting_horizon=2)
        intervals = forecaster.predict_interval()
        assert len(intervals) > 0
        # The inner fit held out its own tail from the outer training split:
        # the eval targets must be the last values of the calibration-split
        # training window, not of the full series the outer forecaster saw.
        inner = forecaster.point_forecaster_
        X_eval, y_eval = _eval_pair(inner.estimator_)
        assert len(X_eval) == VAL_SIZE - 2 + 1
        inner_train_end = 120 - 30
        eval_targets = y_eval.to_numpy().ravel()
        assert eval_targets.max() == pytest.approx(float(inner_train_end - 1))
        assert eval_targets.min() >= float(inner_train_end - VAL_SIZE)
