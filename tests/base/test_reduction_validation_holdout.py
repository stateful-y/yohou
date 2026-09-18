"""Tests for the validation_size holdout on reduction forecasters."""

import warnings
from datetime import datetime, timedelta
from unittest import mock

import numpy as np
import polars as pl
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, clone
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import ClassifierChain, MultiOutputRegressor, RegressorChain
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler as SkStandardScaler
from sklearn.utils.validation import check_is_fitted

from yohou import UnweightedEvaluationSetWarning
from yohou.base.reduction import BaseReductionForecaster
from yohou.class_proba import ClassProbaReductionForecaster
from yohou.compose import FeaturePipeline
from yohou.interval import IntervalReductionForecaster
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


class EvalXRegressor(RegressorMixin, BaseEstimator):
    """Stub speaking LightGBM's newer keyword-only eval_X/eval_y convention."""

    def fit(self, X, y, sample_weight=None, *, eval_X=None, eval_y=None):
        self.received_eval_X_ = eval_X
        self.received_eval_y_ = eval_y
        self.train_X_ = X
        arr = np.asarray(y, dtype=float)
        self._ncols = 1 if arr.ndim == 1 else arr.shape[1]
        self._mean = float(np.nanmean(arr))
        return self

    def predict(self, X):
        out = np.full((len(X), self._ncols), self._mean)
        return out.ravel() if self._ncols == 1 else out


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


def _make_x_future(length: int = LENGTH, horizon: int = HORIZON) -> pl.DataFrame:
    t_ext = _times(length + horizon)
    return pl.DataFrame({"time": t_ext, "temp": [float(i % 7) for i in range(len(t_ext))]})


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
        X_future = _make_x_future()
        forecaster = PointReductionForecaster(estimator=RecordingRegressor(), validation_size=VAL_SIZE)
        forecaster.fit(y=y, forecasting_horizon=HORIZON, X_future=X_future)
        X_eval, _ = _eval_pair(forecaster.estimator_)
        step_cols = [c for c in X_eval.columns if "_step_" in c]
        assert step_cols == [f"temp_step_{k}" for k in range(1, HORIZON + 1)]
        assert list(X_eval.columns) == list(forecaster.estimator_.train_X_.columns)
        assert X_eval.select(step_cols).null_count().sum_horizontal().item() == 0

    def test_ragged_panel_splits_by_row_not_by_group_extent(self):
        """Groups with shorter extents share the split, which is by row count.

        ``_split_validation_tail`` slices the wide frame by absolute row, so a
        group whose series ends early contributes null evaluation targets
        rather than shifting the boundary. The training matrix carries the same
        nulls, so the holdout path stays consistent with the non-holdout path.
        """
        y = _make_y_panel()
        short = [float(2 * i) for i in range(LENGTH)]
        for i in range(LENGTH - 4, LENGTH):
            short[i] = float("nan")
        y = y.with_columns(pl.Series("b__value", short))
        forecaster = PointReductionForecaster(estimator=RecordingRegressor(), validation_size=VAL_SIZE)
        forecaster.fit(y=y, forecasting_horizon=HORIZON)
        X_eval, y_eval = _eval_pair(forecaster.estimator_)
        # The global strategy stacks both groups, so both contribute rows and
        # the boundary is unmoved by group b ending early.
        assert len(X_eval) == 2 * STRICT_ROWS
        assert np.isnan(np.asarray(y_eval, dtype=float)).any()
        assert list(X_eval.columns) == list(forecaster.estimator_.train_X_.columns)

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
        assert len(X_eval) == STRICT_ROWS - 1


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

    @pytest.mark.parametrize("overlap", [False, True], ids=["strict", "overlap"])
    def test_horizon_one_evaluates_the_whole_tail(self, overlap):
        """At horizon 1 the strict window is the full tail, so both modes agree.

        This is the boundary where VAL_SIZE - HORIZON + 1 == VAL_SIZE and the
        ``n < forecasting_horizon`` guard becomes vacuous.
        """
        forecaster = PointReductionForecaster(
            estimator=RecordingRegressor(), validation_size=VAL_SIZE, validation_overlap=overlap
        )
        forecaster.fit(y=_make_y(), forecasting_horizon=1)
        X_eval, y_eval = _eval_pair(forecaster.estimator_)
        assert len(X_eval) == VAL_SIZE
        # Values equal their index, so the tail targets are the last VAL_SIZE.
        eval_targets = sorted(np.asarray(y_eval, dtype=float).ravel().tolist())
        assert eval_targets == [float(i) for i in range(LENGTH - VAL_SIZE, LENGTH)]

    @pytest.mark.parametrize("validation_size", [LENGTH, LENGTH + 10], ids=["equal", "greater"])
    def test_validation_size_at_or_beyond_series_length(self, validation_size):
        """A holdout swallowing the series reports zero head rows, not a negative count."""
        with pytest.raises(ValueError, match="leaves 0 head rows"):
            PointReductionForecaster(estimator=RecordingRegressor(), validation_size=validation_size).fit(
                y=_make_y(), forecasting_horizon=HORIZON
            )


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
        with pytest.raises(ValueError, match="does not support an evaluation-set"):
            PointReductionForecaster(estimator=LinearRegression(), validation_size=VAL_SIZE).fit(
                y=_make_y(), forecasting_horizon=HORIZON
            )

    def test_multioutput_wrapper_rejected(self):
        with pytest.raises(ValueError, match="MultiOutputRegressor"):
            PointReductionForecaster(estimator=MultiOutputRegressor(LinearRegression()), validation_size=VAL_SIZE).fit(
                y=_make_y(), forecasting_horizon=HORIZON
            )

    @pytest.mark.parametrize(
        "wrapper",
        [RegressorChain(LinearRegression()), ClassifierChain(LinearRegression())],
        ids=["regressor-chain", "classifier-chain"],
    )
    def test_multioutput_chain_rejected(self, wrapper):
        """Chains are sklearn.multioutput wrappers too, so they raise here.

        Their fit takes ``**fit_params``, so without an explicit rejection they
        pass the support check and fail later inside sklearn's routing with a
        message that never mentions validation_size.
        """
        with pytest.raises(ValueError, match="cannot be used with .*Chain"):
            BaseReductionForecaster._check_eval_set_support(wrapper)

    def test_regressor_chain_rejected_at_fit(self):
        """The chain rejection reaches a real fit, not just the helper."""
        with pytest.raises(ValueError, match="RegressorChain"):
            PointReductionForecaster(
                estimator=RegressorChain(LinearRegression()),
                reduction_strategy="multi-output",
                validation_size=VAL_SIZE,
            ).fit(y=_make_y(), forecasting_horizon=HORIZON)

    def test_nested_pipeline_final_step_rejected(self):
        """A Pipeline ending in a Pipeline is rejected with our own message."""
        inner = Pipeline([("scale", SkStandardScaler()), ("model", RecordingRegressor())])
        estimator = Pipeline([("outer", SkStandardScaler()), ("inner", inner)])
        with pytest.raises(ValueError, match="final step is itself a Pipeline"):
            PointReductionForecaster(estimator=estimator, validation_size=VAL_SIZE).fit(
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
        forecaster = ClassProbaReductionForecaster(estimator=RecordingClassifier(), validation_size=VAL_SIZE)
        with pytest.raises(ValueError, match="only inside the validation_size holdout"):
            forecaster.fit(y=y, forecasting_horizon=2)
        # A rejected holdout must leave no fitted state behind, so that
        # check_is_fitted does not report a half-fitted instance as ready.
        for attribute in ("classes_", "n_classes_", "label_to_code_", "estimator_"):
            assert not hasattr(forecaster, attribute)

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"eval_set": "raw"}, "raw eval_set"),
            ({"eval_X": "raw"}, "raw eval_X"),
            ({"eval_y": "raw"}, "raw eval_y"),
            ({"eval_X": "raw", "eval_y": "raw"}, "raw eval_X, eval_y"),
        ],
        ids=["eval_set", "eval_X", "eval_y", "eval_X+eval_y"],
    )
    def test_raw_eval_passthrough_conflict(self, kwargs, match):
        """Both eval delivery conventions must be rejected identically.

        The internally built pair is spread last over the caller's fit params,
        so an unguarded key would be silently overwritten rather than honoured.
        """
        with pytest.raises(ValueError, match=match):
            PointReductionForecaster(estimator=RecordingRegressor(), validation_size=VAL_SIZE).fit(
                y=_make_y(), forecasting_horizon=HORIZON, **kwargs
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
        # An exact oracle, not a plausibility band: the delivered matrix must
        # equal the fitted prefix applied to the raw evaluation window. A band
        # check passes even when the prefix was fitted on the wrong rows.
        bare = PointReductionForecaster(estimator=RecordingRegressor(), validation_size=VAL_SIZE)
        bare.fit(y=_make_y(), forecasting_horizon=HORIZON)
        raw_eval = _eval_pair(bare.estimator_)[0]
        prefix = forecaster.estimator_.named_steps["scaler"]
        np.testing.assert_allclose(X_eval, np.asarray(prefix.transform(raw_eval)))
        # And the prefix itself must have seen training rows only: refitting it
        # on the bare training matrix reproduces the same transform.
        fresh = SkStandardScaler().fit(bare.estimator_.train_X_)
        np.testing.assert_allclose(X_eval, np.asarray(fresh.transform(raw_eval)))
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
    """All three reduction families expose and round-trip the holdout parameters."""

    @pytest.mark.parametrize(
        "cls, estimator",
        [
            (PointReductionForecaster, RecordingRegressor()),
            (ClassProbaReductionForecaster, RecordingClassifier()),
            (IntervalReductionForecaster, RecordingRegressor()),
        ],
        ids=["point", "class_proba", "interval"],
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
        X_future = _make_x_future()
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
            # The systematic-check matrix exercises these strategy and
            # alignment dimensions, but that check only compares shapes. The
            # oracle is what actually pins the evaluation rows, so it has to
            # reach the same configurations.
            {"reduction_strategy": "dir-rec", "actual_transformer": LagTransformer(lag=[1, 2])},
            {"reduction_strategy": "direct", "step_feature_alignment": "matched"},
            {"target_as_feature": "raw", "actual_transformer": LagTransformer(lag=1)},
            {"training_stride": 2, "nan_handling": "drop", "actual_transformer": LagTransformer(lag=[1, 2])},
        ],
        ids=[
            "scaler",
            "seasonal-diff",
            "ema",
            "scaler+pipeline",
            "dir-rec",
            "direct-matched",
            "target-as-feature-raw",
            "stride+drop",
        ],
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
        X_future = _make_x_future()
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
        X_future = _make_x_future()
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


class QuantileStub(RegressorMixin, BaseEstimator):
    """Quantile-parameterized stub recording the eval_set it received."""

    def __init__(self, quantile=0.5):
        self.quantile = quantile

    def fit(self, X, y, eval_set=None, sample_weight=None):
        self.received_eval_set_ = eval_set
        self.train_X_ = X
        arr = np.asarray(y, dtype=float)
        self._ncols = 1 if arr.ndim == 1 else arr.shape[1]
        self._value = float(np.nanquantile(arr, self.quantile))
        return self

    def predict(self, X):
        out = np.full((len(X), self._ncols), self._value)
        return out.ravel() if self._ncols == 1 else out


class MultiQuantileStub(RegressorMixin, BaseEstimator):
    """Stand-in for a CatBoost MultiQuantile estimator recording its eval_set.

    One model covers every quantile, so ``predict`` returns a column per
    quantile parsed from ``loss_function``.
    """

    def __init__(self, loss_function="MultiQuantile:alpha=0.05,0.95"):
        self.loss_function = loss_function

    def _quantiles(self):
        return [float(q) for q in self.loss_function.split("alpha=")[1].split(",")]

    def fit(self, X, y, eval_set=None, sample_weight=None):
        self.received_eval_set_ = eval_set
        self.train_X_ = X
        arr = np.asarray(y, dtype=float).ravel()
        self._values = [float(np.nanquantile(arr, q)) for q in self._quantiles()]
        return self

    def predict(self, X):
        return np.tile(np.asarray(self._values), (len(X), 1))


class TestErrorOrdering:
    """Configuration and shape errors fire before any tail row is observed.

    The guarantee is deliberately partial. Whether the evaluation rows are
    usable can only be known after they are built, and building them requires
    observing the tail, so content-dependent failures necessarily land after
    the observation. Those leave the same partial state any failed fit leaves.
    """

    def test_transformed_head_too_short_raises_before_tail_observed(self):
        y = _make_y()
        head_len = LENGTH - VAL_SIZE
        # Seasonality head_len - 2 leaves a 2-row transformed head; overlap
        # mode needs HORIZON (3) anchor rows, so the eval-window check fires.
        forecaster = PointReductionForecaster(
            estimator=RecordingRegressor(),
            target_transformer=SeasonalDifferencing(seasonality=head_len - 2),
            validation_size=VAL_SIZE,
            validation_overlap=True,
        )
        with pytest.raises(ValueError, match="transformed head has"):
            forecaster.fit(y=y, forecasting_horizon=HORIZON)
        # The failure precedes the tail observation: the observation state is
        # exactly what the head fit left, not advanced into the holdout.
        head_end = y["time"][head_len - 1]
        assert forecaster.observed_time_ == head_end
        assert forecaster._y_observed["time"][-1] == head_end

    def test_content_error_fires_after_tail_observed(self):
        """A content-dependent failure lands after the tail is observed.

        The counterpart to the check above, pinning the boundary of the
        guarantee rather than leaving it to chance: NaN handling runs on rows
        that only exist once the tail has been observed, so it cannot fail
        early. The forecaster is left unfitted, exactly as any failed fit
        leaves it.
        """
        values = [float(i) for i in range(LENGTH)]
        for i in range(LENGTH - VAL_SIZE - 1, LENGTH):
            values[i] = float("nan")
        y = _make_y().with_columns(pl.Series("value", values))
        forecaster = PointReductionForecaster(
            estimator=RecordingRegressor(),
            validation_size=VAL_SIZE,
            nan_handling="drop",
        )
        with pytest.raises(ValueError, match="validation instances contain NaN"):
            forecaster.fit(y=y, forecasting_horizon=HORIZON)

        assert not hasattr(forecaster, "estimator_")
        # The tail was already observed when the failure fired, so the
        # observation state sits at the series end rather than the head end.
        assert forecaster.observed_time_ == y["time"][-1]


class TestClassProbaPanelGuard:
    """The tail-only-class guard strips panel prefixes before comparing."""

    def test_panel_tail_only_class(self):
        base = ["lo", "hi"] * ((LENGTH - 5) // 2)
        b_states = base + ["new"] * (LENGTH - len(base))
        y = pl.DataFrame({
            "time": _times(LENGTH),
            "a__state": ["lo", "hi"] * (LENGTH // 2),
            "b__state": b_states,
        })
        with pytest.raises(ValueError, match="'b__state'.*only inside the validation_size holdout"):
            ClassProbaReductionForecaster(estimator=RecordingClassifier(), validation_size=VAL_SIZE).fit(
                y=y, forecasting_horizon=2
            )


class TestIntervalFamily:
    """The interval family shares one holdout across all quantile fits."""

    def test_quantile_estimators_share_the_pair(self):
        forecaster = IntervalReductionForecaster(estimator=QuantileStub(), validation_size=VAL_SIZE)
        forecaster.fit(y=_make_y(), forecasting_horizon=HORIZON, coverage_rates=[0.8, 0.9])
        estimators = list(forecaster.estimator_.values())
        assert len(estimators) == 4
        X_first, y_first = _eval_pair(estimators[0])
        assert len(X_first) == STRICT_ROWS
        for est in estimators:
            X_eval, y_eval = _eval_pair(est)
            assert X_eval.equals(X_first) and y_eval.equals(y_first)
            assert list(X_eval.columns) == list(est.train_X_.columns)

    def test_multiquantile_single_fit_receives_the_pair(self):
        """The MultiQuantile branch fits one model and must still get the eval pair.

        MultiQuantile requires a single target column at horizon 1, so this is
        the one interval path that never splits into lower/upper estimators.
        """
        forecaster = IntervalReductionForecaster(
            estimator=MultiQuantileStub(), validation_size=VAL_SIZE, validation_overlap=True
        )
        forecaster.fit(y=_make_y(), forecasting_horizon=1, coverage_rates=[0.9])
        assert list(forecaster.estimator_) == ["_multiquantile"]
        X_eval, _ = _eval_pair(forecaster.estimator_["_multiquantile"])
        # Horizon 1: strict and overlap both evaluate the whole tail.
        assert len(X_eval) == VAL_SIZE

    def test_validation_fit_equals_fit_then_observe(self):
        # The scaler rides the actual-transformer slot: interval's
        # predict_interval cannot inverse a target transformer's scaling of
        # the bound columns (pre-existing, independent of the holdout).
        kwargs = {
            "actual_transformer": FeaturePipeline(
                steps=[("lag", LagTransformer(lag=[1, 2])), ("scale", MinMaxScaler())]
            )
        }
        y = _make_y()
        holdout = IntervalReductionForecaster(estimator=QuantileStub(), validation_size=VAL_SIZE, **kwargs)
        holdout.fit(y=y, forecasting_horizon=HORIZON, coverage_rates=[0.9])

        reference = IntervalReductionForecaster(estimator=QuantileStub(), **kwargs)
        reference.fit(y=y[:-VAL_SIZE], forecasting_horizon=HORIZON, coverage_rates=[0.9])
        reference.observe(y[-VAL_SIZE:])

        assert holdout.observed_time_ == reference.observed_time_
        pl.testing.assert_frame_equal(holdout._y_observed, reference._y_observed)
        pl.testing.assert_frame_equal(holdout.predict_interval(), reference.predict_interval())

    def test_predict_interval_after_validation_fit(self):
        y = _make_y()
        forecaster = IntervalReductionForecaster(estimator=QuantileStub(), validation_size=VAL_SIZE)
        forecaster.fit(y=y, forecasting_horizon=HORIZON, coverage_rates=[0.9])
        intervals = forecaster.predict_interval()
        assert intervals["time"].min() > y["time"][-1]

    def test_default_wrapper_estimator_rejected(self):
        with pytest.raises(ValueError, match="MultiOutputRegressor"):
            IntervalReductionForecaster(validation_size=VAL_SIZE).fit(y=_make_y(), forecasting_horizon=HORIZON)

    def test_lightgbm_quantile_early_stopping_triggers(self):
        lightgbm = pytest.importorskip("lightgbm")
        LGBMRegressor = lightgbm.LGBMRegressor

        rng = np.random.default_rng(0)
        length = 400
        y = pl.DataFrame({
            "time": _times(length),
            "value": (np.sin(np.arange(length) / 5.0) + rng.normal(0, 0.05, length)).tolist(),
        })
        estimator = LGBMRegressor(
            objective="quantile",
            alpha=0.5,
            n_estimators=300,
            early_stopping_round=10,
            min_child_samples=5,
            verbose=-1,
        )
        forecaster = IntervalReductionForecaster(
            estimator=estimator,
            reduction_strategy="direct",
            actual_transformer=LagTransformer(lag=[1, 2, 3]),
            validation_size=60,
        )
        forecaster.fit(y=y, forecasting_horizon=2, coverage_rates=[0.9])
        assert len(forecaster.estimator_) == 2
        for est_list in forecaster.estimator_.values():
            for est in est_list:
                assert est.best_iteration_ is not None
                assert est.best_iteration_ < 300


class TestSystematicCheckShapes:
    """The shared holdout checks handle every ``estimator_`` shape."""

    @pytest.mark.parametrize("strategy", ["multi-output", "direct", "dir-rec"])
    def test_interval_strategies_through_holdout_checks(self, strategy):
        from yohou.testing.reduction import (
            check_validation_holdout_default_noop,
            check_validation_holdout_fit,
        )

        forecaster = IntervalReductionForecaster(reduction_strategy=strategy)
        y = _make_y(90)
        check_validation_holdout_fit(forecaster, y)
        check_validation_holdout_default_noop(forecaster, y)


class TestEvalSetDeliveryConvention:
    """The evaluation pair follows the estimator's keyword, not a fixed one.

    LightGBM has deprecated ``eval_set`` in favour of the keyword-only
    ``eval_X``/``eval_y`` pair while XGBoost and CatBoost still take
    ``eval_set``, so the keyword is read from the estimator's fit signature.
    """

    def test_eval_x_estimator_receives_eval_x(self):
        forecaster = PointReductionForecaster(estimator=EvalXRegressor(), validation_size=VAL_SIZE)
        forecaster.fit(y=_make_y(), forecasting_horizon=HORIZON)

        estimator = forecaster.estimator_
        assert estimator.received_eval_X_ is not None, "eval_X was not delivered"
        assert estimator.received_eval_y_ is not None, "eval_y was not delivered"
        assert len(estimator.received_eval_X_) == STRICT_ROWS
        assert list(estimator.received_eval_X_.columns) == list(estimator.train_X_.columns)

    def test_eval_set_estimator_still_receives_eval_set(self):
        forecaster = PointReductionForecaster(estimator=RecordingRegressor(), validation_size=VAL_SIZE)
        forecaster.fit(y=_make_y(), forecasting_horizon=HORIZON)

        assert forecaster.estimator_.received_eval_set_ is not None
        assert len(forecaster.estimator_.received_eval_set_) == 1

    def test_no_delivery_without_holdout(self):
        forecaster = PointReductionForecaster(estimator=EvalXRegressor())
        forecaster.fit(y=_make_y(), forecasting_horizon=HORIZON)

        assert forecaster.estimator_.received_eval_X_ is None
        assert forecaster.estimator_.received_eval_y_ is None

    def test_lightgbm_delivery_emits_no_deprecation_warning(self):
        """The whole point of following the estimator's keyword.

        Delivering ``eval_set`` to a current LightGBM works but warns on every
        fit, once per step estimator.
        """
        lightgbm = pytest.importorskip("lightgbm")
        estimator = lightgbm.LGBMRegressor(n_estimators=20, early_stopping_round=5, min_child_samples=2, verbose=-1)
        forecaster = PointReductionForecaster(
            estimator=estimator, reduction_strategy="direct", validation_size=VAL_SIZE
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            forecaster.fit(y=_make_y(), forecasting_horizon=HORIZON)

        deprecations = [w for w in caught if "eval_set" in str(w.message) and "deprecated" in str(w.message)]
        assert not deprecations, f"delivery raised {len(deprecations)} eval_set deprecation warnings"
        # The evaluation set really arrived: LightGBM only records a validation
        # curve when it is given one.
        assert all(est.evals_result_ for est in forecaster.estimator_)


class XValDialectStub(RegressorMixin, BaseEstimator):
    """Stub speaking scikit-learn's ``X_val``/``y_val`` convention."""

    def fit(self, X, y, sample_weight=None, X_val=None, y_val=None, sample_weight_val=None):
        self.received_X_val_ = X_val
        self.received_y_val_ = y_val
        self.received_sample_weight_val_ = sample_weight_val
        self.train_X_ = X
        arr = np.asarray(y, dtype=float)
        self._ncols = 1 if arr.ndim == 1 else arr.shape[1]
        self._mean = float(np.nanmean(arr))
        return self

    def predict(self, X):
        out = np.full((len(X), self._ncols), self._mean)
        return out.ravel() if self._ncols == 1 else out


class XValKwargsStub(XValDialectStub):
    """Declares the ``X_val`` pair *and* ``**kwargs``; the declaration must win."""

    def fit(self, X, y, X_val=None, y_val=None, **kwargs):
        return super().fit(X, y, X_val=X_val, y_val=y_val)


class BothDialectsStub(RegressorMixin, BaseEstimator):
    """Declares ``eval_set`` and the ``X_val`` pair; ``eval_set`` comes first."""

    def fit(self, X, y, eval_set=None, X_val=None, y_val=None, sample_weight=None):
        self.received_eval_set_ = eval_set
        self.received_X_val_ = X_val
        arr = np.asarray(y, dtype=float)
        self._ncols = 1 if arr.ndim == 1 else arr.shape[1]
        self._mean = float(np.nanmean(arr))
        return self

    def predict(self, X):
        out = np.full((len(X), self._ncols), self._mean)
        return out.ravel() if self._ncols == 1 else out


def _histgb(**kwargs):
    """A HistGradientBoosting regressor configured for these small fixtures."""
    sklearn_ensemble = pytest.importorskip("sklearn.ensemble")
    params = {"max_iter": 20, "early_stopping": True, "n_iter_no_change": 50, "min_samples_leaf": 2}
    params.update(kwargs)
    return sklearn_ensemble.HistGradientBoostingRegressor(**params)


class TestXValDialect:
    """Task 2: the third evaluation-set dialect, ``X_val``/``y_val``."""

    def test_histgb_receives_x_val_and_no_eval_set(self):
        forecaster = PointReductionForecaster(
            estimator=_histgb(), reduction_strategy="direct", validation_size=VAL_SIZE
        )
        forecaster.fit(y=_make_y(), forecasting_horizon=HORIZON)
        for estimator in forecaster.estimator_:
            # The curve is only recorded when the pair actually arrived.
            assert len(estimator.validation_score_) == estimator.n_iter_ + 1
            assert estimator.n_iter_ == 20

    def test_stub_receives_the_pair_with_training_columns(self):
        forecaster = PointReductionForecaster(
            estimator=XValDialectStub(), reduction_strategy="direct", validation_size=VAL_SIZE
        )
        forecaster.fit(y=_make_y(), forecasting_horizon=HORIZON)
        for estimator in forecaster.estimator_:
            assert estimator.received_X_val_ is not None
            assert not hasattr(estimator, "received_eval_set_")
            assert len(estimator.received_X_val_) == STRICT_ROWS
            assert list(estimator.received_X_val_.columns) == list(estimator.train_X_.columns)

    def test_explicit_window_uses_the_dialect(self):
        y = _make_y()
        head, tail = y[: LENGTH - VAL_SIZE], y[LENGTH - VAL_SIZE :]
        forecaster = PointReductionForecaster(estimator=XValDialectStub(), reduction_strategy="direct")
        forecaster.fit(y=head, forecasting_horizon=HORIZON, y_val=tail)
        for estimator in forecaster.estimator_:
            assert estimator.received_X_val_ is not None
            assert len(estimator.received_X_val_) == STRICT_ROWS

    def test_pipeline_final_step_uses_the_dialect(self):
        forecaster = PointReductionForecaster(
            estimator=Pipeline([("scaler", SkStandardScaler()), ("model", XValDialectStub())]),
            reduction_strategy="direct",
            validation_size=VAL_SIZE,
        )
        forecaster.fit(y=_make_y(), forecasting_horizon=HORIZON)
        for pipeline in forecaster.estimator_:
            model = pipeline.named_steps["model"]
            assert model.received_X_val_ is not None
            assert np.asarray(model.received_X_val_).shape[1] == np.asarray(model.train_X_).shape[1]

    def test_declared_pair_beats_the_kwargs_fallback(self):
        forecaster = PointReductionForecaster(
            estimator=XValKwargsStub(), reduction_strategy="direct", validation_size=VAL_SIZE
        )
        forecaster.fit(y=_make_y(), forecasting_horizon=HORIZON)
        for estimator in forecaster.estimator_:
            assert estimator.received_X_val_ is not None

    def test_eval_set_wins_when_both_are_declared(self):
        forecaster = PointReductionForecaster(
            estimator=BothDialectsStub(), reduction_strategy="direct", validation_size=VAL_SIZE
        )
        forecaster.fit(y=_make_y(), forecasting_horizon=HORIZON)
        for estimator in forecaster.estimator_:
            assert estimator.received_eval_set_ is not None
            assert estimator.received_X_val_ is None

    def test_classifier_under_class_proba(self):
        sklearn_ensemble = pytest.importorskip("sklearn.ensemble")
        y = _make_y().with_columns((pl.col("value").cast(pl.Int64) % 2).cast(pl.Utf8).alias("value"))
        forecaster = ClassProbaReductionForecaster(
            estimator=sklearn_ensemble.HistGradientBoostingClassifier(
                max_iter=10, early_stopping=True, n_iter_no_change=50, min_samples_leaf=2
            ),
            reduction_strategy="direct",
            validation_size=VAL_SIZE,
        )
        forecaster.fit(y=y, forecasting_horizon=HORIZON)
        assert forecaster.predict_class_proba().height == HORIZON


class TestXValRejections:
    """Task 2: configurations the ``X_val`` dialect must refuse."""

    @pytest.mark.parametrize("key", ["X_val", "sample_weight_val", "eval_sample_weight", "sample_weight_eval_set"])
    def test_raw_dialect_key_in_fit_params_rejected(self, key):
        forecaster = PointReductionForecaster(
            estimator=XValDialectStub(), reduction_strategy="direct", validation_size=VAL_SIZE
        )
        with pytest.raises(ValueError, match=key):
            forecaster.fit(y=_make_y(), forecasting_horizon=HORIZON, **{key: "anything"})
        assert not hasattr(forecaster, "estimator_")

    @pytest.mark.parametrize("value", ["auto", False])
    def test_early_stopping_not_true_rejected(self, value):
        forecaster = PointReductionForecaster(
            estimator=_histgb(early_stopping=value), reduction_strategy="direct", validation_size=VAL_SIZE
        )
        with pytest.raises(ValueError, match="early_stopping"):
            forecaster.fit(y=_make_y(), forecasting_horizon=HORIZON)
        assert not hasattr(forecaster, "estimator_")

    def test_early_stopping_true_fits(self):
        forecaster = PointReductionForecaster(
            estimator=_histgb(early_stopping=True), reduction_strategy="direct", validation_size=VAL_SIZE
        )
        forecaster.fit(y=_make_y(), forecasting_horizon=HORIZON)
        assert forecaster.predict().height == HORIZON

    def test_early_stopping_check_is_scoped_to_the_dialect(self):
        """An eval_set estimator carrying early_stopping=False is not rejected."""
        forecaster = PointReductionForecaster(
            estimator=EarlyStoppingStub(), reduction_strategy="direct", validation_size=VAL_SIZE
        )
        forecaster.estimator.early_stopping = False
        forecaster.fit(y=_make_y(), forecasting_horizon=HORIZON)
        assert forecaster.predict().height == HORIZON


class EvalXWeightRegressor(RegressorMixin, BaseEstimator):
    """LightGBM's dialect including its evaluation-weight keyword."""

    def fit(self, X, y, sample_weight=None, *, eval_X=None, eval_y=None, eval_sample_weight=None):
        self.received_eval_X_ = eval_X
        self.received_eval_sample_weight_ = eval_sample_weight
        self.train_X_ = X
        arr = np.asarray(y, dtype=float)
        self._ncols = 1 if arr.ndim == 1 else arr.shape[1]
        self._mean = float(np.nanmean(arr))
        return self

    def predict(self, X):
        out = np.full((len(X), self._ncols), self._mean)
        return out.ravel() if self._ncols == 1 else out


class WeightRecordingRegressor(RegressorMixin, BaseEstimator):
    """Records the evaluation weights delivered in each dialect."""

    def fit(
        self,
        X,
        y,
        eval_set=None,
        sample_weight=None,
        sample_weight_eval_set=None,
    ):
        self.received_eval_set_ = eval_set
        self.received_sample_weight_eval_set_ = sample_weight_eval_set
        self.train_X_ = X
        arr = np.asarray(y, dtype=float)
        self._ncols = 1 if arr.ndim == 1 else arr.shape[1]
        self._mean = float(np.nanmean(arr))
        return self

    def predict(self, X):
        out = np.full((len(X), self._ncols), self._mean)
        return out.ravel() if self._ncols == 1 else out


def _weighted(estimator, **kwargs):
    kwargs.setdefault("reduction_strategy", "direct")
    return PointReductionForecaster(
        estimator=estimator,
        validation_size=VAL_SIZE,
        time_weighter=ExponentialDecayWeighter(half_life=5),
        **kwargs,
    )


def _delivered_weights(estimator):
    """The evaluation weights an estimator received, whatever its dialect."""
    if getattr(estimator, "received_sample_weight_eval_set_", None) is not None:
        return np.asarray(estimator.received_sample_weight_eval_set_[0])
    if getattr(estimator, "received_sample_weight_val_", None) is not None:
        return np.asarray(estimator.received_sample_weight_val_)
    return None


class TestEvaluationWeights:
    """Task 4: the evaluation rows carry the forecaster's weights."""

    def test_weights_reach_the_estimator_with_one_entry_per_row(self):
        forecaster = _weighted(WeightRecordingRegressor())
        forecaster.fit(y=_make_y(), forecasting_horizon=HORIZON)
        for estimator in forecaster.estimator_:
            weights = _delivered_weights(estimator)
            assert weights is not None
            assert len(weights) == len(_eval_pair(estimator)[0])

    def test_x_val_dialect_receives_sample_weight_val(self):
        forecaster = _weighted(XValDialectStub())
        forecaster.fit(y=_make_y(), forecasting_horizon=HORIZON)
        for estimator in forecaster.estimator_:
            assert estimator.received_sample_weight_val_ is not None
            assert len(estimator.received_sample_weight_val_) == len(estimator.received_X_val_)

    def test_eval_x_dialect_receives_eval_sample_weight(self):
        forecaster = _weighted(EvalXWeightRegressor())
        forecaster.fit(y=_make_y(), forecasting_horizon=HORIZON)
        for estimator in forecaster.estimator_:
            assert estimator.received_eval_sample_weight_ is not None
            assert len(estimator.received_eval_sample_weight_[0]) == len(estimator.received_eval_X_)

    def test_dialect_without_a_weight_parameter_warns(self):
        """Speaking a dialect is not the same as accepting weights for it."""
        forecaster = _weighted(EvalXRegressor(), reduction_strategy="multi-output")
        with pytest.warns(UnweightedEvaluationSetWarning, match="EvalXRegressor"):
            forecaster.fit(y=_make_y(), forecasting_horizon=HORIZON)

    @pytest.mark.parametrize("overlap", [False, True])
    def test_weight_count_matches_rows_under_both_overlap_settings(self, overlap):
        forecaster = _weighted(WeightRecordingRegressor(), validation_overlap=overlap)
        forecaster.fit(y=_make_y(), forecasting_horizon=HORIZON)
        expected = VAL_SIZE if overlap else STRICT_ROWS
        for estimator in forecaster.estimator_:
            weights = _delivered_weights(estimator)
            assert len(weights) == len(_eval_pair(estimator)[0]) == expected

    def test_no_weighter_passes_no_weights(self):
        forecaster = PointReductionForecaster(
            estimator=WeightRecordingRegressor(), reduction_strategy="direct", validation_size=VAL_SIZE
        )
        forecaster.fit(y=_make_y(), forecasting_horizon=HORIZON)
        for estimator in forecaster.estimator_:
            assert estimator.received_sample_weight_eval_set_ is None

    def test_weights_are_normalized_over_the_evaluation_rows(self):
        forecaster = _weighted(WeightRecordingRegressor())
        forecaster.fit(y=_make_y(), forecasting_horizon=HORIZON)
        for estimator in forecaster.estimator_:
            weights = _delivered_weights(estimator)
            assert np.isclose(weights.sum(), len(weights))

    def test_later_evaluation_rows_weigh_more(self):
        """A decay weighter must produce an increasing vector over the window."""
        forecaster = _weighted(WeightRecordingRegressor())
        forecaster.fit(y=_make_y(), forecasting_horizon=HORIZON)
        weights = _delivered_weights(forecaster.estimator_[0])
        assert np.all(np.diff(weights) > 0), weights

    @pytest.mark.parametrize("strategy", ["multi-output", "direct", "dir-rec"])
    def test_every_strategy_delivers_weights(self, strategy):
        forecaster = _weighted(WeightRecordingRegressor(), reduction_strategy=strategy)
        forecaster.fit(y=_make_y(), forecasting_horizon=HORIZON)
        estimators = forecaster.estimator_ if isinstance(forecaster.estimator_, list) else [forecaster.estimator_]
        for estimator in estimators:
            weights = _delivered_weights(estimator)
            assert weights is not None
            assert len(weights) == len(_eval_pair(estimator)[0])

    def test_panel_weights_match_stacked_rows(self):
        forecaster = _weighted(WeightRecordingRegressor(), actual_transformer=LagTransformer(lag=[1, 2]))
        forecaster.fit(y=_make_y_panel(), forecasting_horizon=HORIZON)
        for estimator in forecaster.estimator_:
            weights = _delivered_weights(estimator)
            assert len(weights) == len(_eval_pair(estimator)[0])

    def test_pipeline_final_step_receives_weights(self):
        forecaster = _weighted(
            Pipeline([("scaler", SkStandardScaler()), ("model", WeightRecordingRegressor())]),
        )
        forecaster.fit(y=_make_y(), forecasting_horizon=HORIZON)
        for pipeline in forecaster.estimator_:
            model = pipeline.named_steps["model"]
            weights = _delivered_weights(model)
            assert weights is not None
            assert len(weights) == len(_eval_pair(model)[0])

    def test_explicit_window_delivers_weights(self):
        y = _make_y()
        head, tail = y[: LENGTH - VAL_SIZE], y[LENGTH - VAL_SIZE :]
        forecaster = PointReductionForecaster(
            estimator=WeightRecordingRegressor(),
            reduction_strategy="direct",
            time_weighter=ExponentialDecayWeighter(half_life=5),
        )
        forecaster.fit(y=head, forecasting_horizon=HORIZON, y_val=tail)
        for estimator in forecaster.estimator_:
            assert _delivered_weights(estimator) is not None


class TestEvaluationWeightsCannotBeDelivered:
    """Task 4: the warning path, for an estimator with nowhere to put them."""

    def test_warns_once_naming_the_estimator(self):
        forecaster = _weighted(RecordingRegressor(), reduction_strategy="multi-output")
        with pytest.warns(UnweightedEvaluationSetWarning, match="RecordingRegressor"):
            forecaster.fit(y=_make_y(), forecasting_horizon=HORIZON)

    def test_one_warning_per_fit_not_one_per_step(self):
        """A per-step strategy fits H estimators; the warning is about the fit."""
        forecaster = _weighted(RecordingRegressor(), reduction_strategy="direct")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            forecaster.fit(y=_make_y(), forecasting_horizon=HORIZON)
        unweighted = [w for w in caught if issubclass(w.category, UnweightedEvaluationSetWarning)]
        assert len(unweighted) == 1, f"{len(unweighted)} warnings for {HORIZON} estimators"

    def test_fit_still_succeeds_unweighted(self):
        forecaster = _weighted(RecordingRegressor(), reduction_strategy="multi-output")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UnweightedEvaluationSetWarning)
            forecaster.fit(y=_make_y(), forecasting_horizon=HORIZON)
        assert forecaster.predict().height == HORIZON

    def test_no_warning_when_the_dialect_can_carry_weights(self):
        forecaster = _weighted(XValDialectStub())
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            forecaster.fit(y=_make_y(), forecasting_horizon=HORIZON)
        assert not [w for w in caught if issubclass(w.category, UnweightedEvaluationSetWarning)]

    def test_no_warning_without_a_weighter(self):
        forecaster = PointReductionForecaster(
            estimator=RecordingRegressor(), reduction_strategy="direct", validation_size=VAL_SIZE
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            forecaster.fit(y=_make_y(), forecasting_horizon=HORIZON)
        assert not [w for w in caught if issubclass(w.category, UnweightedEvaluationSetWarning)]


class TestEvaluationWeightsAlignment:
    """Task 4: weights follow the rows through every filter."""

    @staticmethod
    def _with_null_target(row: int) -> pl.DataFrame:
        y = _make_y()
        values = y["value"].to_list()
        values[row] = None
        return y.with_columns(pl.Series("value", values))

    def test_weights_follow_dropped_evaluation_rows(self):
        clean = _weighted(WeightRecordingRegressor(), reduction_strategy="multi-output")
        clean.fit(y=_make_y(), forecasting_horizon=HORIZON)
        baseline = len(_delivered_weights(clean.estimator_))

        dropped = _weighted(WeightRecordingRegressor(), reduction_strategy="multi-output", nan_handling="drop")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dropped.fit(y=self._with_null_target(LENGTH - 2), forecasting_horizon=HORIZON)
        weights = _delivered_weights(dropped.estimator_)
        assert len(weights) == len(_eval_pair(dropped.estimator_)[0])
        assert len(weights) < baseline, "the null row should have been dropped"

    def test_direct_steps_keep_their_own_alignment_under_drop(self):
        forecaster = _weighted(WeightRecordingRegressor(), reduction_strategy="direct", nan_handling="drop")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            forecaster.fit(y=self._with_null_target(LENGTH - 2), forecasting_horizon=HORIZON)
        # Each step filters its own rows, so lengths may differ between steps;
        # what must hold is that each step's weights match its own rows.
        for estimator in forecaster.estimator_:
            assert len(_delivered_weights(estimator)) == len(_eval_pair(estimator)[0])

    @pytest.mark.parametrize("source", ["validation_size", "y_val"])
    def test_vintage_weighter_over_the_evaluation_window(self, source):
        """A vintage weighter looks weights up by time on the evaluation frame too.

        The lookup gives the last few timestamps five times the weight, so
        exactly the evaluation rows anchored there must carry the larger weight.
        """
        from yohou.weighting import LookupWeighter

        y = _make_y()
        times = y["time"].to_list()
        heavy = set(times[LENGTH - 4 :])
        weighter = LookupWeighter(mapping={t: (5.0 if t in heavy else 1.0) for t in times}, default=1.0)
        if source == "validation_size":
            forecaster = PointReductionForecaster(
                estimator=WeightRecordingRegressor(),
                reduction_strategy="direct",
                validation_size=VAL_SIZE,
                vintage_weighter=weighter,
            )
            forecaster.fit(y=y, forecasting_horizon=HORIZON)
        else:
            head, tail = y[: LENGTH - VAL_SIZE], y[LENGTH - VAL_SIZE :]
            forecaster = PointReductionForecaster(
                estimator=WeightRecordingRegressor(), reduction_strategy="direct", vintage_weighter=weighter
            )
            forecaster.fit(y=head, forecasting_horizon=HORIZON, y_val=tail)

        for estimator in forecaster.estimator_:
            weights = _delivered_weights(estimator)
            assert len(weights) == len(_eval_pair(estimator)[0])
            # Two distinct values: the heavily weighted anchors and the rest,
            # with the heavy ones strictly larger.
            assert len(set(np.round(weights, 9))) == 2
            assert weights.max() > weights.min()
            assert weights[-1] == weights.max()


class TestCatBoostEvaluationWeights:
    """Task 4: CatBoost has no weight keyword, so a Pool carries them."""

    def test_pool_delivery_changes_the_recorded_metric(self):
        catboost = pytest.importorskip("catboost")
        y = _make_y()

        def _fit(weighter):
            forecaster = PointReductionForecaster(
                estimator=catboost.CatBoostRegressor(
                    iterations=20, learning_rate=0.3, verbose=False, allow_writing_files=False
                ),
                reduction_strategy="direct",
                validation_size=VAL_SIZE,
                time_weighter=weighter,
            )
            forecaster.fit(y=y, forecasting_horizon=HORIZON)
            return forecaster.estimator_[0].get_evals_result()

        weighted = _fit(ExponentialDecayWeighter(half_life=3))
        unweighted = _fit(None)
        key = next(iter(weighted["validation"]))
        assert weighted["validation"][key] != unweighted["validation"][key], (
            "weights did not reach CatBoost's evaluation set"
        )

    def test_pool_is_the_delivered_eval_set(self):
        catboost = pytest.importorskip("catboost")
        estimator = catboost.CatBoostRegressor(iterations=5, verbose=False, allow_writing_files=False)
        params = BaseReductionForecaster._eval_set_fit_params(
            estimator,
            pl.DataFrame({"a": [1.0, 2.0, 3.0]}),
            pl.DataFrame({"t": [1.0, 2.0, 3.0]}),
            np.array([1.0, 2.0, 3.0]),
        )
        assert isinstance(params["eval_set"][0], catboost.Pool)
        np.testing.assert_allclose(params["eval_set"][0].get_weight(), [1.0, 2.0, 3.0])


class TestWeightSpellingPerLibrary:
    """Task 4.17: every library gets the keyword its own fit declares."""

    @staticmethod
    def _delivered(estimator):
        """The fit parameters the holdout would build for this estimator."""
        return BaseReductionForecaster._eval_set_fit_params(
            estimator,
            pl.DataFrame({"a": [1.0, 2.0, 3.0]}),
            pl.DataFrame({"t": [1.0, 2.0, 3.0]}),
            np.array([0.5, 1.0, 1.5]),
        )

    def test_lightgbm_gets_eval_sample_weight(self):
        lightgbm = pytest.importorskip("lightgbm")
        params = self._delivered(lightgbm.LGBMRegressor())
        assert "eval_sample_weight" in params
        np.testing.assert_allclose(params["eval_sample_weight"][0], [0.5, 1.0, 1.5])

    def test_xgboost_gets_sample_weight_eval_set(self):
        xgboost = pytest.importorskip("xgboost")
        params = self._delivered(xgboost.XGBRegressor())
        assert "sample_weight_eval_set" in params
        np.testing.assert_allclose(params["sample_weight_eval_set"][0], [0.5, 1.0, 1.5])

    def test_histgradientboosting_gets_sample_weight_val(self):
        from sklearn.ensemble import HistGradientBoostingRegressor

        params = self._delivered(HistGradientBoostingRegressor())
        assert "sample_weight_val" in params
        assert "X_val" in params and "eval_set" not in params
        np.testing.assert_allclose(params["sample_weight_val"], [0.5, 1.0, 1.5])


class TestPipelineEvaluationMatrixOracle:
    """Task 5: the delivered matrix equals the prefix, proven exactly."""

    @staticmethod
    def _pipeline(*steps):
        return Pipeline([*steps, ("rec", RecordingRegressor())])

    def _compare(self, forecaster, bare, panel=False):
        """Assert each delivered matrix equals its own fitted prefix's transform."""
        pipelines = forecaster.estimator_ if isinstance(forecaster.estimator_, list) else [forecaster.estimator_]
        bares = bare.estimator_ if isinstance(bare.estimator_, list) else [bare.estimator_]
        assert len(pipelines) == len(bares)
        for pipeline, plain in zip(pipelines, bares, strict=True):
            prefix = Pipeline(pipeline.steps[:-1])
            delivered = np.asarray(_eval_pair(pipeline.named_steps["rec"])[0])
            raw_eval = _eval_pair(plain)[0]
            np.testing.assert_allclose(delivered, np.asarray(prefix.transform(raw_eval)))
            # Leak check: a prefix refitted on the training rows alone agrees.
            fresh = Pipeline([(name, clone(step)) for name, step in pipeline.steps[:-1]])
            fresh.fit(plain.train_X_)
            np.testing.assert_allclose(delivered, np.asarray(fresh.transform(raw_eval)))

    @pytest.mark.parametrize("strategy", ["multi-output", "direct", "dir-rec"])
    def test_every_strategy(self, strategy):
        kwargs = {"reduction_strategy": strategy, "validation_size": VAL_SIZE}
        forecaster = PointReductionForecaster(estimator=self._pipeline(("scaler", SkStandardScaler())), **kwargs)
        forecaster.fit(y=_make_y(), forecasting_horizon=HORIZON)
        bare = PointReductionForecaster(estimator=RecordingRegressor(), **kwargs)
        bare.fit(y=_make_y(), forecasting_horizon=HORIZON)
        self._compare(forecaster, bare)

    def test_explicit_window(self):
        y = _make_y()
        head, tail = y[: LENGTH - VAL_SIZE], y[LENGTH - VAL_SIZE :]
        forecaster = PointReductionForecaster(estimator=self._pipeline(("scaler", SkStandardScaler())))
        forecaster.fit(y=head, forecasting_horizon=HORIZON, y_val=tail)
        bare = PointReductionForecaster(estimator=RecordingRegressor())
        bare.fit(y=head, forecasting_horizon=HORIZON, y_val=tail)
        self._compare(forecaster, bare)

    def test_two_step_prefix(self):
        kwargs = {"validation_size": VAL_SIZE, "actual_transformer": LagTransformer(lag=[1, 2, 3])}
        forecaster = PointReductionForecaster(
            estimator=self._pipeline(("scaler", SkStandardScaler()), ("second", SkStandardScaler())), **kwargs
        )
        forecaster.fit(y=_make_y(), forecasting_horizon=HORIZON)
        bare = PointReductionForecaster(estimator=RecordingRegressor(), **kwargs)
        bare.fit(y=_make_y(), forecasting_horizon=HORIZON)
        self._compare(forecaster, bare)

    @pytest.mark.parametrize("shaper", ["reduce", "expand"])
    def test_prefix_that_changes_the_column_count(self, shaper):
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import PolynomialFeatures

        step = ("pca", PCA(n_components=2)) if shaper == "reduce" else ("poly", PolynomialFeatures(2))
        kwargs = {"validation_size": VAL_SIZE, "actual_transformer": LagTransformer(lag=[1, 2, 3])}
        forecaster = PointReductionForecaster(estimator=self._pipeline(step), **kwargs)
        forecaster.fit(y=_make_y(), forecasting_horizon=HORIZON)
        bare = PointReductionForecaster(estimator=RecordingRegressor(), **kwargs)
        bare.fit(y=_make_y(), forecasting_horizon=HORIZON)
        self._compare(forecaster, bare)
        delivered = np.asarray(_eval_pair(forecaster.estimator_.named_steps["rec"])[0])
        train = np.asarray(forecaster.estimator_.named_steps["rec"].train_X_)
        assert delivered.shape[1] == train.shape[1]

    def test_panel_data(self):
        kwargs = {"validation_size": VAL_SIZE, "actual_transformer": LagTransformer(lag=[1, 2])}
        forecaster = PointReductionForecaster(estimator=self._pipeline(("scaler", SkStandardScaler())), **kwargs)
        forecaster.fit(y=_make_y_panel(), forecasting_horizon=HORIZON)
        bare = PointReductionForecaster(estimator=RecordingRegressor(), **kwargs)
        bare.fit(y=_make_y_panel(), forecasting_horizon=HORIZON)
        self._compare(forecaster, bare, panel=True)

    def test_pipeline_configuration_survives_the_two_phase_fit(self):
        """`memory` and `verbose` must not be dropped when the pipeline is rebuilt."""
        pipeline = Pipeline(
            [("scaler", SkStandardScaler()), ("rec", RecordingRegressor())],
            verbose=True,
        )
        forecaster = PointReductionForecaster(estimator=pipeline, validation_size=VAL_SIZE)
        forecaster.fit(y=_make_y(), forecasting_horizon=HORIZON)
        assert forecaster.estimator_.get_params()["verbose"] is True


class ExtraInputRegressor(RegressorMixin, BaseEstimator):
    """Records a caller-supplied routed input alongside the evaluation set."""

    def fit(self, X, y, eval_set=None, sample_weight=None, extra=None):
        self.received_eval_set_ = eval_set
        self.received_extra_ = extra
        self.train_X_ = X
        arr = np.asarray(y, dtype=float)
        self._ncols = 1 if arr.ndim == 1 else arr.shape[1]
        self._mean = float(np.nanmean(arr))
        return self

    def predict(self, X):
        out = np.full((len(X), self._ncols), self._mean)
        return out.ravel() if self._ncols == 1 else out


class TestPipelineTransformInput:
    """Task 5.8: a caller's transform_input is honoured, not silently ignored."""

    @staticmethod
    def _forecaster(transform_input):
        pipeline = Pipeline(
            [("scaler", SkStandardScaler()), ("model", ExtraInputRegressor())],
            transform_input=transform_input,
        )
        return PointReductionForecaster(estimator=pipeline, validation_size=VAL_SIZE)

    def _fit_with_extra(self, forecaster, extra):
        forecaster.estimator.steps[-1][1].set_fit_request(extra=True)
        forecaster.fit(y=_make_y(), forecasting_horizon=HORIZON, extra=extra)
        return forecaster.estimator_.named_steps["model"].received_extra_

    def test_named_input_is_transformed_by_the_fitted_prefix(self):
        extra = pl.DataFrame({"value": [10.0, 20.0, 30.0]})
        forecaster = self._forecaster(["extra"])
        received = self._fit_with_extra(forecaster, extra)
        expected = forecaster.estimator_.named_steps["scaler"].transform(extra)
        np.testing.assert_allclose(np.asarray(received), np.asarray(expected))

    def test_input_is_untouched_without_transform_input(self):
        extra = pl.DataFrame({"value": [10.0, 20.0, 30.0]})
        received = self._fit_with_extra(self._forecaster(None), extra)
        np.testing.assert_allclose(np.asarray(received), np.asarray(extra))

    def test_evaluation_pair_is_not_transformed_twice(self):
        """yohou builds the pair already transformed, so naming it changes nothing."""
        plain = self._forecaster(None)
        plain.fit(y=_make_y(), forecasting_horizon=HORIZON)
        named = self._forecaster(["X_val", "eval_set"])
        named.fit(y=_make_y(), forecasting_horizon=HORIZON)
        np.testing.assert_allclose(
            np.asarray(_eval_pair(named.estimator_.named_steps["model"])[0]),
            np.asarray(_eval_pair(plain.estimator_.named_steps["model"])[0]),
        )
