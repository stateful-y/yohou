"""Tests for the explicit evaluation window (``y_val``) on reduction forecasters.

The window path reuses the ``validation_size`` evaluation-row machinery, so
the strongest oracles compare against it: fitting on a head with the tail
passed as ``y_val`` must deliver exactly the evaluation rows that
``validation_size=len(tail)`` delivers on the concatenated series, while the
post-fit state must equal a plain fit on the head alone.
"""

from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression

from yohou.class_proba import ClassProbaReductionForecaster
from yohou.compose import FeaturePipeline
from yohou.interval import IntervalReductionForecaster
from yohou.point import PointReductionForecaster
from yohou.preprocessing import LagTransformer, MinMaxScaler, RollingStatisticsTransformer

from .test_reduction_validation_holdout import (
    HORIZON,
    LENGTH,
    VAL_SIZE,
    QuantileStub,
    RecordingClassifier,
    RecordingRegressor,
    _eval_pair,
    _make_x_future,
    _make_y,
    _make_y_panel,
)

STRICT_ROWS = VAL_SIZE - HORIZON + 1


class KwargsRecordingRegressor(RegressorMixin, BaseEstimator):
    """Stub recording every fit keyword it received."""

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha

    def fit(self, X, y, eval_set=None, **kwargs):
        self.received_keys_ = {"eval_set"} | set(kwargs) if eval_set is not None else set(kwargs)
        self.received_kwargs_ = kwargs
        arr = np.asarray(y, dtype=float)
        self._ncols = 1 if arr.ndim == 1 else arr.shape[1]
        self._mean = float(np.nanmean(arr))
        return self

    def predict(self, X):
        out = np.full((len(X), self._ncols), self._mean)
        return out.ravel() if self._ncols == 1 else out


def _estimators(forecaster):
    est = forecaster.estimator_
    if isinstance(est, dict):
        out = []
        for value in est.values():
            out.extend(value if isinstance(value, list) else [value])
        return out
    return est if isinstance(est, list) else [est]


def _split(y: pl.DataFrame, n: int = VAL_SIZE) -> tuple[pl.DataFrame, pl.DataFrame]:
    return y[:-n], y[-n:]


def _assert_same_state(a, b) -> None:
    assert a.observed_time_ == b.observed_time_
    if isinstance(a._y_observed, dict):
        for group in a._y_observed:
            pl.testing.assert_frame_equal(a._y_observed[group], b._y_observed[group])
    elif a._y_observed is None:
        assert b._y_observed is None
    else:
        pl.testing.assert_frame_equal(a._y_observed, b._y_observed)


def _as_frame(values):
    return values.to_frame() if isinstance(values, pl.Series) else values


def _vintage_forecast() -> pl.DataFrame:
    """Two vintages: one before the head/tail boundary, one inside the tail."""
    times = _make_y()["time"]
    boundary_idx = LENGTH - VAL_SIZE
    horizon_times = pl.datetime_range(
        start=datetime(2021, 1, 1),
        end=datetime(2021, 1, 1) + timedelta(seconds=LENGTH + HORIZON - 1),
        interval="1s",
        eager=True,
    )
    return pl.concat([
        pl.DataFrame({
            "vintage_time": [vt] * len(horizon_times),
            "time": horizon_times,
            "fx": [val] * len(horizon_times),
        })
        for vt, val in [(times[boundary_idx - 5], 1.0), (times[boundary_idx + 4], 2.0)]
    ])


class TestEvaluationRows:
    """Evaluation rows come from the supplied window, built without leakage."""

    @pytest.mark.parametrize(
        "kwargs",
        [
            {},
            {"actual_transformer": LagTransformer(lag=[1, 3])},
            {"reduction_strategy": "direct", "actual_transformer": LagTransformer(lag=[1, 2])},
            {
                "target_transformer": MinMaxScaler(),
                "actual_transformer": FeaturePipeline(
                    steps=[("lag", LagTransformer(lag=[1, 2])), ("roll", RollingStatisticsTransformer(window_size=3))]
                ),
            },
        ],
        ids=["plain", "lags", "direct-lags", "scaler+pipeline"],
    )
    def test_rows_equal_validation_size_oracle(self, kwargs):
        y = _make_y()
        head, tail = _split(y)
        window = PointReductionForecaster(estimator=RecordingRegressor(), **kwargs)
        window.fit(y=head, forecasting_horizon=HORIZON, y_val=tail)
        oracle = PointReductionForecaster(estimator=RecordingRegressor(), validation_size=VAL_SIZE, **kwargs)
        oracle.fit(y=y, forecasting_horizon=HORIZON)

        for est_window, est_oracle in zip(_estimators(window), _estimators(oracle), strict=True):
            X_w, y_w = _eval_pair(est_window)
            X_o, y_o = _eval_pair(est_oracle)
            assert len(X_w) == STRICT_ROWS
            pl.testing.assert_frame_equal(X_w, X_o)
            pl.testing.assert_frame_equal(_as_frame(y_w), _as_frame(y_o))
            pl.testing.assert_frame_equal(est_window.train_X_, est_oracle.train_X_)

    def test_eval_targets_lie_inside_the_window(self):
        y = _make_y()
        head, tail = _split(y)
        forecaster = PointReductionForecaster(estimator=RecordingRegressor(), reduction_strategy="multi-output")
        forecaster.fit(y=head, forecasting_horizon=HORIZON, y_val=tail)
        _, y_eval = _eval_pair(forecaster.estimator_)
        tail_values = set(tail["value"].to_list())
        assert set(np.asarray(y_eval).ravel().tolist()) <= tail_values
        train_targets = set(np.asarray(forecaster.estimator_.train_y_).ravel().tolist())
        assert not train_targets & tail_values

    def test_transformer_statistics_exclude_the_window(self):
        y = _make_y()
        head, tail = _split(y)
        window = PointReductionForecaster(estimator=RecordingRegressor(), target_transformer=MinMaxScaler())
        window.fit(y=head, forecasting_horizon=HORIZON, y_val=tail)
        plain = PointReductionForecaster(estimator=RecordingRegressor(), target_transformer=MinMaxScaler())
        plain.fit(y=head, forecasting_horizon=HORIZON)
        # The scaler's statistics shape every training target; equal training
        # targets mean the window never reached the scaler's fit.
        pl.testing.assert_frame_equal(_as_frame(window.estimator_.train_y_), _as_frame(plain.estimator_.train_y_))

    def test_eval_rows_respect_vintage_availability(self):
        y = _make_y()
        head, tail = _split(y)
        X_forecast = _vintage_forecast()
        cutoff = head["time"][-1]
        window = PointReductionForecaster(estimator=RecordingRegressor())
        window.fit(
            y=head,
            forecasting_horizon=HORIZON,
            X_forecast=X_forecast.filter(pl.col("vintage_time") <= cutoff),
            y_val=tail,
            X_forecast_val=X_forecast.filter(pl.col("vintage_time") > cutoff),
        )
        oracle = PointReductionForecaster(estimator=RecordingRegressor(), validation_size=VAL_SIZE)
        oracle.fit(y=y, forecasting_horizon=HORIZON, X_forecast=X_forecast)
        pl.testing.assert_frame_equal(_eval_pair(window.estimator_)[0], _eval_pair(oracle.estimator_)[0])

    def test_step_columns_from_x_future(self):
        y = _make_y()
        head, tail = _split(y)
        X_future = _make_x_future()
        window = PointReductionForecaster(estimator=RecordingRegressor(), reduction_strategy="direct")
        window.fit(y=head, forecasting_horizon=HORIZON, X_future=X_future, y_val=tail)
        oracle = PointReductionForecaster(
            estimator=RecordingRegressor(), reduction_strategy="direct", validation_size=VAL_SIZE
        )
        oracle.fit(y=y, forecasting_horizon=HORIZON, X_future=X_future)
        for est_window, est_oracle in zip(_estimators(window), _estimators(oracle), strict=True):
            pl.testing.assert_frame_equal(_eval_pair(est_window)[0], _eval_pair(est_oracle)[0])

    def test_panel_rows_equal_validation_size_oracle(self):
        y = _make_y_panel()
        head, tail = _split(y)
        kwargs = {"target_transformer": MinMaxScaler(), "actual_transformer": LagTransformer(lag=[1, 2])}
        window = PointReductionForecaster(estimator=RecordingRegressor(), **kwargs)
        window.fit(y=head, forecasting_horizon=HORIZON, y_val=tail)
        oracle = PointReductionForecaster(estimator=RecordingRegressor(), validation_size=VAL_SIZE, **kwargs)
        oracle.fit(y=y, forecasting_horizon=HORIZON)
        pl.testing.assert_frame_equal(_eval_pair(window.estimator_)[0], _eval_pair(oracle.estimator_)[0])

    def test_interval_rows_equal_validation_size_oracle(self):
        y = _make_y()
        head, tail = _split(y)
        window = IntervalReductionForecaster(estimator=QuantileStub(), reduction_strategy="direct")
        window.fit(y=head, forecasting_horizon=HORIZON, coverage_rates=[0.9], y_val=tail)
        oracle = IntervalReductionForecaster(
            estimator=QuantileStub(), reduction_strategy="direct", validation_size=VAL_SIZE
        )
        oracle.fit(y=y, forecasting_horizon=HORIZON, coverage_rates=[0.9])
        for est_window, est_oracle in zip(_estimators(window), _estimators(oracle), strict=True):
            pl.testing.assert_frame_equal(_eval_pair(est_window)[0], _eval_pair(est_oracle)[0])

    def test_class_proba_rows_equal_validation_size_oracle(self):
        times = _make_y()["time"]
        y = pl.DataFrame({"time": times, "state": [["a", "b", "c"][i % 3] for i in range(LENGTH)]})
        head, tail = _split(y)
        window = ClassProbaReductionForecaster(
            estimator=RecordingClassifier(), actual_transformer=LagTransformer(lag=[1, 2])
        )
        window.fit(y=head, forecasting_horizon=HORIZON, y_val=tail)
        oracle = ClassProbaReductionForecaster(
            estimator=RecordingClassifier(), actual_transformer=LagTransformer(lag=[1, 2]), validation_size=VAL_SIZE
        )
        oracle.fit(y=y, forecasting_horizon=HORIZON)
        X_w, y_w = _eval_pair(window.estimator_)
        X_o, y_o = _eval_pair(oracle.estimator_)
        pl.testing.assert_frame_equal(X_w, X_o)
        pl.testing.assert_frame_equal(_as_frame(y_w), _as_frame(y_o))

    def test_omitted_arguments_are_a_no_op(self):
        y = _make_y()
        forecaster = PointReductionForecaster(
            estimator=RecordingRegressor(), actual_transformer=LagTransformer(lag=[1])
        )
        forecaster.fit(y=y, forecasting_horizon=HORIZON, y_val=None)
        plain = PointReductionForecaster(estimator=RecordingRegressor(), actual_transformer=LagTransformer(lag=[1]))
        plain.fit(y=y, forecasting_horizon=HORIZON)
        assert forecaster.estimator_.received_eval_set_ is None
        _assert_same_state(forecaster, plain)
        pl.testing.assert_frame_equal(forecaster.predict(), plain.predict())


class TestPostFitState:
    """The observation state ends at the training data, not the window."""

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"target_transformer": MinMaxScaler(), "actual_transformer": LagTransformer(lag=[1, 2])},
            {
                "actual_transformer": FeaturePipeline(
                    steps=[("lag", LagTransformer(lag=[1, 2])), ("roll", RollingStatisticsTransformer(window_size=4))]
                )
            },
        ],
        ids=["scaler+lags", "warmup-pipeline"],
    )
    @pytest.mark.parametrize("panel", [False, True])
    def test_state_equals_plain_fit_on_training_data(self, kwargs, panel):
        y = _make_y_panel() if panel else _make_y()
        head, tail = _split(y)
        window = PointReductionForecaster(estimator=RecordingRegressor(), **kwargs)
        window.fit(y=head, forecasting_horizon=HORIZON, y_val=tail)
        plain = PointReductionForecaster(estimator=RecordingRegressor(), **kwargs)
        plain.fit(y=head, forecasting_horizon=HORIZON)
        _assert_same_state(window, plain)
        predicted = window.predict()
        pl.testing.assert_frame_equal(predicted, plain.predict())
        assert predicted["time"][0] == head["time"][-1] + timedelta(seconds=1)

    def test_state_with_x_forecast_equals_plain_fit(self):
        y = _make_y()
        head, tail = _split(y)
        X_forecast = _vintage_forecast()
        cutoff = head["time"][-1]
        X_train = X_forecast.filter(pl.col("vintage_time") <= cutoff)
        window = PointReductionForecaster(estimator=RecordingRegressor(), actual_transformer=LagTransformer(lag=[1]))
        window.fit(
            y=head,
            forecasting_horizon=HORIZON,
            X_forecast=X_train,
            y_val=tail,
            X_forecast_val=X_forecast.filter(pl.col("vintage_time") > cutoff),
        )
        plain = PointReductionForecaster(estimator=RecordingRegressor(), actual_transformer=LagTransformer(lag=[1]))
        plain.fit(y=head, forecasting_horizon=HORIZON, X_forecast=X_train)
        _assert_same_state(window, plain)
        pl.testing.assert_frame_equal(window.predict(), plain.predict())

    def test_observe_predict_over_the_window_matches_plain_fit(self):
        y = _make_y()
        head, tail = _split(y)
        window = PointReductionForecaster(estimator=RecordingRegressor(), actual_transformer=LagTransformer(lag=[1, 2]))
        window.fit(y=head, forecasting_horizon=HORIZON, y_val=tail)
        plain = PointReductionForecaster(estimator=RecordingRegressor(), actual_transformer=LagTransformer(lag=[1, 2]))
        plain.fit(y=head, forecasting_horizon=HORIZON)
        pl.testing.assert_frame_equal(window.observe_predict(tail), plain.observe_predict(tail))

    def test_interval_state_equals_plain_fit(self):
        y = _make_y()
        head, tail = _split(y)
        window = IntervalReductionForecaster(estimator=QuantileStub(), actual_transformer=LagTransformer(lag=[1, 2]))
        window.fit(y=head, forecasting_horizon=HORIZON, coverage_rates=[0.9], y_val=tail)
        plain = IntervalReductionForecaster(estimator=QuantileStub(), actual_transformer=LagTransformer(lag=[1, 2]))
        plain.fit(y=head, forecasting_horizon=HORIZON, coverage_rates=[0.9])
        _assert_same_state(window, plain)
        pl.testing.assert_frame_equal(
            window.predict_interval(coverage_rates=[0.9]), plain.predict_interval(coverage_rates=[0.9])
        )

    def test_class_proba_state_equals_plain_fit(self):
        times = _make_y()["time"]
        y = pl.DataFrame({"time": times, "state": [["a", "b", "c"][i % 3] for i in range(LENGTH)]})
        head, tail = _split(y)
        window = ClassProbaReductionForecaster(
            estimator=RecordingClassifier(), actual_transformer=LagTransformer(lag=[1])
        )
        window.fit(y=head, forecasting_horizon=HORIZON, y_val=tail)
        plain = ClassProbaReductionForecaster(
            estimator=RecordingClassifier(), actual_transformer=LagTransformer(lag=[1])
        )
        plain.fit(y=head, forecasting_horizon=HORIZON)
        _assert_same_state(window, plain)
        pl.testing.assert_frame_equal(window.predict_class_proba(), plain.predict_class_proba())


class TestNothingElseReachesTheEstimator:
    """The window path passes only the evaluation set and the caller's params."""

    def test_estimator_parameters_and_fit_keywords(self):
        y = _make_y()
        head, tail = _split(y)
        template = KwargsRecordingRegressor(alpha=0.5)
        before = template.get_params()
        forecaster = PointReductionForecaster(estimator=template, reduction_strategy="direct")
        forecaster.fit(y=head, forecasting_horizon=HORIZON, y_val=tail, marker="caller")
        assert template.get_params() == before
        for est in _estimators(forecaster):
            assert est.get_params() == before
            assert est.received_keys_ == {"eval_set", "marker"}
            assert est.received_kwargs_ == {"marker": "caller"}

    def test_interval_forwards_caller_params_to_every_estimator(self):
        y = _make_y()
        head, tail = _split(y)

        class QuantileKwargs(KwargsRecordingRegressor):
            def __init__(self, quantile: float = 0.5):
                self.quantile = quantile

        forecaster = IntervalReductionForecaster(estimator=QuantileKwargs())
        forecaster.fit(y=head, forecasting_horizon=HORIZON, coverage_rates=[0.9], y_val=tail, marker="caller")
        for est in _estimators(forecaster):
            assert est.received_kwargs_ == {"marker": "caller"}


class TestRejectedConfigurations:
    """Every rejection raises before any fitted or observation state exists."""

    @staticmethod
    def _assert_untouched(forecaster) -> None:
        assert not hasattr(forecaster, "estimator_")
        assert not hasattr(forecaster, "observed_time_")

    def _fit_raises(self, forecaster, match, **fit_kwargs):
        with pytest.raises(ValueError, match=match):
            forecaster.fit(forecasting_horizon=HORIZON, **fit_kwargs)
        self._assert_untouched(forecaster)

    def test_both_window_sources(self):
        head, tail = _split(_make_y())
        forecaster = PointReductionForecaster(estimator=RecordingRegressor(), validation_size=VAL_SIZE)
        self._fit_raises(forecaster, "validation_size.*y_val.*mutually exclusive", y=head, y_val=tail)

    @pytest.mark.parametrize("argument", ["X_actual_val", "X_forecast_val"])
    def test_window_features_without_window_target(self, argument):
        head, tail = _split(_make_y())
        forecaster = PointReductionForecaster(estimator=RecordingRegressor())
        self._fit_raises(forecaster, f"{argument} requires y_val", y=head, **{argument: tail})

    @pytest.mark.parametrize("offset", [2, 0], ids=["gap", "overlap"])
    def test_non_contiguous_window(self, offset):
        y = _make_y(LENGTH + 5)
        head = y[: LENGTH - VAL_SIZE]
        start = len(head) + offset - 1
        window = y[start : start + VAL_SIZE]
        forecaster = PointReductionForecaster(estimator=RecordingRegressor())
        self._fit_raises(forecaster, "must start one interval after", y=head, y_val=window)

    def test_column_mismatch(self):
        head, tail = _split(_make_y())
        forecaster = PointReductionForecaster(estimator=RecordingRegressor())
        self._fit_raises(forecaster, "same columns", y=head, y_val=tail.rename({"value": "other"}))

    def test_panel_group_mismatch(self):
        head, tail = _split(_make_y_panel())
        forecaster = PointReductionForecaster(estimator=RecordingRegressor())
        self._fit_raises(forecaster, "same columns", y=head, y_val=tail.drop("b__value"))

    def test_x_actual_pairing(self):
        y = _make_y()
        head, tail = _split(y)
        X = y.select("time", pl.col("value").alias("feature"))
        X_head, X_tail = _split(X)
        forecaster = PointReductionForecaster(estimator=RecordingRegressor())
        self._fit_raises(forecaster, "X_actual_val is required", y=head, X_actual=X_head, y_val=tail)
        forecaster = PointReductionForecaster(estimator=RecordingRegressor())
        self._fit_raises(
            forecaster,
            "X_actual_val was given but X_actual was not",
            y=head,
            y_val=tail,
            X_actual_val=X_tail,
        )

    def test_estimator_without_eval_set_support(self):
        head, tail = _split(_make_y())
        forecaster = PointReductionForecaster(estimator=LinearRegression())
        self._fit_raises(forecaster, "does not support an evaluation-set", y=head, y_val=tail)

    def test_strict_window_too_small(self):
        head, tail = _split(_make_y(), n=HORIZON - 1)
        forecaster = PointReductionForecaster(estimator=RecordingRegressor())
        self._fit_raises(forecaster, "validation_overlap", y=head, y_val=tail)

    def test_raw_eval_set_conflict(self):
        head, tail = _split(_make_y())
        forecaster = PointReductionForecaster(estimator=RecordingRegressor())
        self._fit_raises(forecaster, "raw eval_set", y=head, y_val=tail, eval_set=[(None, None)])

    def test_class_seen_only_in_the_window(self):
        times = _make_y()["time"]
        states = [["a", "b"][i % 2] for i in range(LENGTH)]
        states[-2] = "c"
        y = pl.DataFrame({"time": times, "state": states})
        head, tail = _split(y)
        forecaster = ClassProbaReductionForecaster(estimator=RecordingClassifier())
        with pytest.raises(ValueError, match=r"\['c'\].*only inside the y_val window"):
            forecaster.fit(y=head, forecasting_horizon=HORIZON, y_val=tail)
        assert not hasattr(forecaster, "classes_")
        self._assert_untouched(forecaster)


class TestFittedEstimatorPositions:
    """Every fitted estimator is listed once, under the documented key."""

    def test_direct_list(self):
        forecaster = PointReductionForecaster(estimator=RecordingRegressor(), reduction_strategy="direct")
        forecaster.fit(y=_make_y(), forecasting_horizon=HORIZON)
        positions = forecaster._fitted_estimator_positions()
        assert [key for key, _ in positions] == ["step_1", "step_2", "step_3"]
        assert [est for _, est in positions] == forecaster.estimator_

    def test_single_estimator(self):
        forecaster = PointReductionForecaster(estimator=RecordingRegressor(), reduction_strategy="multi-output")
        forecaster.fit(y=_make_y(), forecasting_horizon=HORIZON)
        assert forecaster._fitted_estimator_positions() == [("multi_output", forecaster.estimator_)]

    def test_interval_dict_of_single_estimators(self):
        forecaster = IntervalReductionForecaster(estimator=QuantileStub(), reduction_strategy="multi-output")
        forecaster.fit(y=_make_y(), forecasting_horizon=HORIZON, coverage_rates=[0.9])
        positions = dict(forecaster._fitted_estimator_positions())
        assert list(positions) == ["coverage_rate_0.9_lower", "coverage_rate_0.9_upper"]
        assert positions["coverage_rate_0.9_lower"] is forecaster.estimator_["coverage_rate_0.9_lower"]

    def test_interval_dict_of_lists(self):
        forecaster = IntervalReductionForecaster(estimator=QuantileStub(), reduction_strategy="direct")
        forecaster.fit(y=_make_y(), forecasting_horizon=2, coverage_rates=[0.9])
        positions = dict(forecaster._fitted_estimator_positions())
        assert list(positions) == [
            "coverage_rate_0.9_lower/step_1",
            "coverage_rate_0.9_lower/step_2",
            "coverage_rate_0.9_upper/step_1",
            "coverage_rate_0.9_upper/step_2",
        ]
        assert positions["coverage_rate_0.9_upper/step_2"] is forecaster.estimator_["coverage_rate_0.9_upper"][1]

    def test_multiquantile_entry(self):
        from .test_reduction_validation_holdout import MultiQuantileStub

        forecaster = IntervalReductionForecaster(estimator=MultiQuantileStub())
        forecaster.fit(y=_make_y(), forecasting_horizon=1, coverage_rates=[0.9])
        assert forecaster._fitted_estimator_positions() == [("_multiquantile", forecaster.estimator_["_multiquantile"])]

    def test_pipeline_final_step(self):
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        forecaster = PointReductionForecaster(
            estimator=Pipeline([("scale", StandardScaler()), ("model", RecordingRegressor())]),
            reduction_strategy="direct",
        )
        forecaster.fit(y=_make_y(), forecasting_horizon=2)
        positions = forecaster._fitted_estimator_positions()
        assert [key for key, _ in positions] == ["step_1", "step_2"]
        for (_, est), pipeline in zip(positions, forecaster.estimator_, strict=True):
            assert est is pipeline.named_steps["model"]


class TestWindowLengthChecks:
    """The explicit window rejects training data too short to build rows."""

    def test_training_data_needs_two_rows(self):
        y = _make_y()
        forecaster = PointReductionForecaster(estimator=RecordingRegressor())
        with pytest.raises(ValueError, match="at least 2 rows"):
            forecaster.fit(y=y[:1], forecasting_horizon=1, y_val=y[1:2])
        assert not hasattr(forecaster, "estimator_")

    def test_training_data_shorter_than_one_training_row(self):
        y = _make_y()
        forecaster = PointReductionForecaster(estimator=RecordingRegressor())
        with pytest.raises(ValueError, match="at least 4 are needed"):
            forecaster.fit(y=y[:3], forecasting_horizon=HORIZON, y_val=y[3:10])
        assert not hasattr(forecaster, "estimator_")


class TestPipelineWithoutSampleWeight:
    """A Pipeline final step that cannot take sample weights is named in the error."""

    def test_final_step_without_sample_weight(self):
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        from yohou.weighting import ExponentialDecayWeighter

        class NoWeightRegressor(RecordingRegressor):
            def fit(self, X, y, eval_set=None):
                return super().fit(X, y, eval_set=eval_set)

        head, tail = _split(_make_y())
        forecaster = PointReductionForecaster(
            estimator=Pipeline([("scale", StandardScaler()), ("model", NoWeightRegressor())]),
            time_weighter=ExponentialDecayWeighter(half_life=5),
        )
        with pytest.raises(ValueError, match="final step NoWeightRegressor does not support sample_weight"):
            forecaster.fit(y=head, forecasting_horizon=HORIZON, y_val=tail)


class TestWindowForecastsRequireFitForecasts:
    """X_forecast_val without X_forecast is rejected, not ignored."""

    def test_window_forecasts_without_fit_forecasts_rejected(self):
        y = _make_y()
        head, tail = _split(y)
        X_forecast = _vintage_forecast()
        cutoff = head["time"][-1]
        forecaster = PointReductionForecaster(estimator=RecordingRegressor())
        with pytest.raises(ValueError, match="X_forecast_val was given but X_forecast was not"):
            forecaster.fit(
                y=head,
                forecasting_horizon=HORIZON,
                y_val=tail,
                X_forecast_val=X_forecast.filter(pl.col("vintage_time") > cutoff),
            )
        assert not hasattr(forecaster, "estimator_")

    def test_window_forecasts_stay_optional(self):
        """X_forecast alone still fits: its vintages may already cover the window."""
        y = _make_y()
        head, tail = _split(y)
        X_forecast = _vintage_forecast()
        forecaster = PointReductionForecaster(estimator=RecordingRegressor())
        forecaster.fit(y=head, forecasting_horizon=HORIZON, X_forecast=X_forecast, y_val=tail)
        assert forecaster.predict().height == HORIZON
