"""Tests for nan_handling parameter on BaseReductionForecaster."""

import warnings
from datetime import datetime

import numpy as np
import polars as pl
import pytest
from sklearn.base import BaseEstimator

from yohou.point import PointReductionForecaster
from yohou.weighting import TableWeighter


def _has_nan(data) -> bool:
    """Check if data (Polars or numpy) contains NaN/null."""
    if isinstance(data, pl.DataFrame):
        # Check nulls
        if any(c > 0 for c in data.null_count().row(0)):
            return True
        # Check NaN in float columns
        float_cols = data.select(pl.col(pl.Float32, pl.Float64))
        if float_cols.width > 0:
            return float_cols.select(pl.all().is_nan().any()).row(0) != tuple(False for _ in range(float_cols.width))
        return False
    elif isinstance(data, pl.Series):
        if data.null_count() > 0:
            return True
        if data.dtype.is_float():
            return data.is_nan().any()
        return False
    else:
        arr = np.asarray(data, dtype=float)
        return bool(np.any(np.isnan(arr)))


def _nrows(data) -> int:
    if hasattr(data, "shape"):
        return data.shape[0]
    return len(data)


def _make_y(length: int = 30) -> pl.DataFrame:
    return pl.DataFrame({
        "time": pl.datetime_range(
            start=datetime(2021, 1, 1),
            end=datetime(2021, 1, 1, 0, 0, length - 1),
            interval="1s",
            eager=True,
        ),
        "value": [float(i) for i in range(length)],
    })


def _make_y_with_nan(length: int = 30, nan_positions: list[int] | None = None) -> pl.DataFrame:
    values = [float(i) for i in range(length)]
    if nan_positions:
        for pos in nan_positions:
            values[pos] = float("nan")
    return pl.DataFrame({
        "time": pl.datetime_range(
            start=datetime(2021, 1, 1),
            end=datetime(2021, 1, 1, 0, 0, length - 1),
            interval="1s",
            eager=True,
        ),
        "value": values,
    })


class _RecordingEstimator(BaseEstimator):
    """Estimator that records what it receives for inspection."""

    def __init__(self):
        pass

    def fit(self, X, y, sample_weight=None, **kwargs):
        self.X_fit_ = X
        self.y_fit_ = y
        if sample_weight is not None:
            kwargs["sample_weight"] = sample_weight
        self.fit_kwargs_ = kwargs
        return self

    def predict(self, X):
        return np.zeros(_nrows(X))


class TestNanHandlingPass:
    def test_nan_passes_through_to_estimator(self):
        """4.1: NaN reaches estimator unchanged when nan_handling='pass'."""
        y = _make_y_with_nan(30, nan_positions=[10])
        forecaster = PointReductionForecaster(
            estimator=_RecordingEstimator(),
            nan_handling="pass",
        )
        forecaster.fit(y=y, forecasting_horizon=2)
        assert _has_nan(forecaster.estimator_.y_fit_), "NaN should pass through"


class TestNanHandlingDrop:
    def test_drop_removes_nan_in_y(self):
        """4.2: Rows contaminated by NaN in y are removed."""
        y = _make_y_with_nan(30, nan_positions=[10])
        forecaster = PointReductionForecaster(
            estimator=_RecordingEstimator(),
            nan_handling="drop",
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            forecaster.fit(y=y, forecasting_horizon=2)
        assert not _has_nan(forecaster.estimator_.y_fit_), "NaN should be removed"

    def test_drop_removes_nan_in_x_actual(self):
        """4.3: Rows with NaN in X_actual lag columns are removed."""
        y = _make_y(30)
        x_values = [float(i) for i in range(30)]
        x_values[10] = float("nan")
        X_actual = pl.DataFrame({"time": y["time"], "feature": x_values})
        forecaster = PointReductionForecaster(
            estimator=_RecordingEstimator(),
            nan_handling="drop",
            target_as_feature=None,
            feature_transformer=None,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            forecaster.fit(y=y, X_actual=X_actual, forecasting_horizon=2)
        assert not _has_nan(forecaster.estimator_.X_fit_), "NaN from X_actual should be removed"

    def test_drop_removes_nan_in_x_future(self):
        """4.4: Rows with NaN in X_future step columns are removed."""
        y = _make_y(30)
        future_length = 35
        future_values = [float(i) for i in range(future_length)]
        future_values[15] = float("nan")
        X_future = pl.DataFrame({
            "time": pl.datetime_range(
                start=datetime(2021, 1, 1),
                end=datetime(2021, 1, 1, 0, 0, future_length - 1),
                interval="1s",
                eager=True,
            ),
            "holiday": future_values,
        })
        forecaster = PointReductionForecaster(
            estimator=_RecordingEstimator(),
            nan_handling="drop",
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            forecaster.fit(y=y, X_future=X_future, forecasting_horizon=2)
        assert not _has_nan(forecaster.estimator_.X_fit_), "NaN from X_future should be removed"

    def test_drop_removes_nan_in_x_forecast(self):
        """4.5: Rows with NaN in X_forecast step columns are removed."""
        y = _make_y(30)
        vintage_times = []
        target_times = []
        temp_values = []
        for i in range(28):
            t = datetime(2021, 1, 1, 0, 0, i)
            for step in range(1, 3):
                vintage_times.append(t)
                target_times.append(datetime(2021, 1, 1, 0, 0, i + step))
                if i == 10 and step == 1:
                    temp_values.append(float("nan"))
                else:
                    temp_values.append(float(i + step))
        X_forecast = pl.DataFrame({
            "vintage_time": vintage_times,
            "time": target_times,
            "temp": temp_values,
        })
        forecaster = PointReductionForecaster(
            estimator=_RecordingEstimator(),
            nan_handling="drop",
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            forecaster.fit(y=y, X_forecast=X_forecast, forecasting_horizon=2)
        assert not _has_nan(forecaster.estimator_.X_fit_), "NaN from X_forecast should be removed"

    def test_sample_weight_aligned_after_nan_removal(self):
        """4.6: sample_weight is filtered in lockstep with dropped rows."""
        y = _make_y_with_nan(30, nan_positions=[10])
        time_weight = pl.DataFrame({"time": y["time"], "weight": [1.0] * len(y)})
        forecaster = PointReductionForecaster(
            estimator=_RecordingEstimator(),
            nan_handling="drop",
            time_weighter=TableWeighter(frame=time_weight, on="time"),
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            forecaster.fit(y=y, forecasting_horizon=2)

        est = forecaster.estimator_
        sw = est.fit_kwargs_.get("sample_weight")
        assert sw is not None
        assert len(sw) == _nrows(est.X_fit_)

    def test_warning_emitted_with_count(self):
        """4.7: Warning includes count and percentage."""
        y = _make_y_with_nan(30, nan_positions=[10])
        forecaster = PointReductionForecaster(
            estimator=_RecordingEstimator(),
            nan_handling="drop",
        )
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            forecaster.fit(y=y, forecasting_horizon=2)

        nan_warnings = [x for x in w if "NaN handling" in str(x.message)]
        assert len(nan_warnings) >= 1, "Should emit a NaN handling warning"
        assert "%" in str(nan_warnings[0].message)

    def test_no_warning_when_clean(self):
        """4.8: No warning emitted when data is NaN-free."""
        y = _make_y(30)
        forecaster = PointReductionForecaster(
            estimator=_RecordingEstimator(),
            nan_handling="drop",
        )
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            forecaster.fit(y=y, forecasting_horizon=2)

        nan_warnings = [x for x in w if "NaN handling" in str(x.message)]
        assert len(nan_warnings) == 0

    def test_direct_per_step_filtering(self):
        """4.9: Direct strategy fits all steps even when some have NaN."""
        y = _make_y(40)
        future_length = 45
        future_values = [float(i) for i in range(future_length)]
        future_values[20] = float("nan")
        X_future = pl.DataFrame({
            "time": pl.datetime_range(
                start=datetime(2021, 1, 1),
                end=datetime(2021, 1, 1, 0, 0, future_length - 1),
                interval="1s",
                eager=True,
            ),
            "holiday": future_values,
        })
        forecaster = PointReductionForecaster(
            estimator=_RecordingEstimator(),
            nan_handling="drop",
            reduction_strategy="direct",
            step_feature_alignment="matched",
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            forecaster.fit(y=y, X_future=X_future, forecasting_horizon=3)
        assert len(forecaster.estimator_) == 3

    def test_valueerror_when_all_nan(self):
        """4.10: ValueError raised when all rows contain NaN."""
        y = pl.DataFrame({
            "time": pl.datetime_range(
                start=datetime(2021, 1, 1),
                end=datetime(2021, 1, 1, 0, 0, 9),
                interval="1s",
                eager=True,
            ),
            "value": [float("nan")] * 10,
        })
        forecaster = PointReductionForecaster(
            estimator=_RecordingEstimator(),
            nan_handling="drop",
        )
        with pytest.raises((ValueError, Exception)), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            forecaster.fit(y=y, forecasting_horizon=2)

    def test_panel_data_nan_handling(self):
        """4.11: NaN in panel data is handled correctly."""
        length = 20
        times = pl.datetime_range(
            start=datetime(2021, 1, 1),
            end=datetime(2021, 1, 1, 0, 0, length - 1),
            interval="1s",
            eager=True,
        )
        y = pl.DataFrame({
            "time": pl.concat([times, times]),
            "value": [float(i) for i in range(length)] * 2,
            "group": ["A"] * length + ["B"] * length,
        })
        y = y.with_columns(
            pl
            .when((pl.col("group") == "A") & (pl.col("value") == 5.0))
            .then(None)
            .otherwise(pl.col("value"))
            .alias("value")
        )
        forecaster = PointReductionForecaster(
            estimator=_RecordingEstimator(),
            nan_handling="drop",
            panel_strategy="global",
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            forecaster.fit(y=y, forecasting_horizon=2)
        assert not _has_nan(forecaster.estimator_.y_fit_)


class TestNanHandlingEdgeBranches:
    """Cover edge branches in _apply_nan_handling for non-float data."""

    def test_x_tab_no_float_columns(self):
        """X_tab with only integer columns uses null-only mask (no NaN check)."""
        forecaster = PointReductionForecaster(
            estimator=_RecordingEstimator(),
            nan_handling="drop",
        )
        X_tab = pl.DataFrame({"a": [1, 2, None, 4], "b": [10, 20, 30, 40]})
        y_tab = pl.Series("value", [1.0, 2.0, 3.0, 4.0])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            X_out, y_out, _ = forecaster._apply_nan_handling(X_tab, y_tab, None)
        assert X_out.shape[0] == 3
        assert len(y_out) == 3

    def test_y_series_non_float_dtype(self):
        """y_tab as integer Series skips is_not_nan check."""
        forecaster = PointReductionForecaster(
            estimator=_RecordingEstimator(),
            nan_handling="drop",
        )
        X_tab = pl.DataFrame({"a": [1.0, 2.0, 3.0, 4.0]})
        y_tab = pl.Series("value", [10, None, 30, 40])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            X_out, y_out, _ = forecaster._apply_nan_handling(X_tab, y_tab, None)
        assert X_out.shape[0] == 3
        assert len(y_out) == 3

    def test_y_dataframe_no_float_columns(self):
        """y_tab as DataFrame with only integer columns uses null-only mask."""
        forecaster = PointReductionForecaster(
            estimator=_RecordingEstimator(),
            nan_handling="drop",
        )
        X_tab = pl.DataFrame({"a": [1.0, 2.0, 3.0, 4.0]})
        y_tab = pl.DataFrame({"t1": [10, 20, None, 40], "t2": [1, 2, 3, 4]})
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            X_out, y_out, _ = forecaster._apply_nan_handling(X_tab, y_tab, None)
        assert X_out.shape[0] == 3
        assert y_out.shape[0] == 3


class TestNanHandlingPredict:
    """Tests for NaN handling at predict time when nan_handling='drop'."""

    def test_multi_output_returns_nan_on_nan_features(self):
        """Predict returns NaN when features contain NaN at predict time (multi-output)."""
        y = _make_y_with_nan(30, nan_positions=[28, 29])
        forecaster = PointReductionForecaster(
            estimator=_RecordingEstimator(),
            nan_handling="drop",
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            forecaster.fit(y=y, forecasting_horizon=2)
        y_pred = forecaster.predict()
        value_col = [c for c in y_pred.columns if c not in ("time", "vintage_time")][0]
        assert y_pred[value_col].is_nan().all(), "Should return NaN predictions"

    def test_multi_output_panel_returns_nan_on_nan_features(self):
        """Panel data returns NaN predictions when features have NaN (multi-output)."""
        length = 30
        times = pl.datetime_range(
            start=datetime(2021, 1, 1),
            end=datetime(2021, 1, 1, 0, 0, length - 1),
            interval="1s",
            eager=True,
        )
        values_a = [float(i) for i in range(length)]
        values_b = [float(i) for i in range(length)]
        values_a[-1] = float("nan")
        values_b[-1] = float("nan")
        y = pl.DataFrame({
            "time": times,
            "x__value": values_a,
            "y__value": values_b,
        })
        forecaster = PointReductionForecaster(
            estimator=_RecordingEstimator(),
            nan_handling="drop",
            panel_strategy="global",
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            forecaster.fit(y=y, forecasting_horizon=2)
        y_pred = forecaster.predict()
        target_cols = [c for c in y_pred.columns if c not in ("time", "vintage_time")]
        for col in target_cols:
            assert y_pred[col].is_nan().all(), f"Column {col} should be NaN"

    def test_direct_returns_nan_on_nan_features(self):
        """Predict returns NaN when features contain NaN (direct strategy)."""
        y = _make_y_with_nan(30, nan_positions=[28, 29])
        forecaster = PointReductionForecaster(
            estimator=_RecordingEstimator(),
            nan_handling="drop",
            reduction_strategy="direct",
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            forecaster.fit(y=y, forecasting_horizon=2)
        y_pred = forecaster.predict()
        value_col = [c for c in y_pred.columns if c not in ("time", "vintage_time")][0]
        assert y_pred[value_col].is_nan().all(), "Should return NaN predictions"

    def test_dir_rec_returns_nan_on_nan_features(self):
        """Predict returns NaN when features contain NaN (dir-rec strategy)."""
        y = _make_y_with_nan(30, nan_positions=[28, 29])
        forecaster = PointReductionForecaster(
            estimator=_RecordingEstimator(),
            nan_handling="drop",
            reduction_strategy="dir-rec",
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            forecaster.fit(y=y, forecasting_horizon=2)
        y_pred = forecaster.predict()
        value_col = [c for c in y_pred.columns if c not in ("time", "vintage_time")][0]
        assert y_pred[value_col].is_nan().all(), "Should return NaN predictions"

    def test_dir_rec_panel_returns_nan_on_nan_features(self):
        """Panel data returns NaN predictions when features have NaN (dir-rec strategy)."""
        length = 30
        times = pl.datetime_range(
            start=datetime(2021, 1, 1),
            end=datetime(2021, 1, 1, 0, 0, length - 1),
            interval="1s",
            eager=True,
        )
        values_a = [float(i) for i in range(length)]
        values_b = [float(i) for i in range(length)]
        values_a[-1] = float("nan")
        values_b[-1] = float("nan")
        y = pl.DataFrame({
            "time": times,
            "x__value": values_a,
            "y__value": values_b,
        })
        forecaster = PointReductionForecaster(
            estimator=_RecordingEstimator(),
            nan_handling="drop",
            reduction_strategy="dir-rec",
            panel_strategy="global",
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            forecaster.fit(y=y, forecasting_horizon=2)
        y_pred = forecaster.predict()
        target_cols = [c for c in y_pred.columns if c not in ("time", "vintage_time")]
        for col in target_cols:
            assert y_pred[col].is_nan().all(), f"Column {col} should be NaN"


class TestFeaturesHaveNanEdgeCases:
    """Cover edge branches in _features_have_nan for non-float data."""

    def test_no_float_columns_with_null(self):
        """_features_have_nan detects null in non-float columns."""
        forecaster = PointReductionForecaster(
            estimator=_RecordingEstimator(),
            nan_handling="drop",
        )
        X_tab = pl.DataFrame({"a": [1, None, 3], "b": [10, 20, 30]})
        assert forecaster._features_have_nan(X_tab) is True

    def test_no_float_columns_clean(self):
        """_features_have_nan returns False for clean non-float data."""
        forecaster = PointReductionForecaster(
            estimator=_RecordingEstimator(),
            nan_handling="drop",
        )
        X_tab = pl.DataFrame({"a": [1, 2, 3], "b": [10, 20, 30]})
        assert forecaster._features_have_nan(X_tab) is False


class TestResolveWeightParamsEdgeCases:
    """Cover edge branches in _resolve_sample_weight_params."""

    def test_var_keyword_fallback(self):
        """Estimator with **kwargs accepts sample_weight via VAR_KEYWORD fallback."""

        class KwargsEstimator(BaseEstimator):
            def fit(self, X, y, **kwargs):
                self.kwargs_ = kwargs
                return self

            def predict(self, X):
                return np.zeros(_nrows(X))

        from yohou.base.reduction import BaseReductionForecaster

        sw = np.array([1.0, 2.0, 3.0])
        result = BaseReductionForecaster._resolve_sample_weight_params(KwargsEstimator(), sw)
        assert "sample_weight" in result
        np.testing.assert_array_equal(result["sample_weight"], sw)

    def test_no_support_raises_valueerror(self):
        """Estimator without sample_weight or **kwargs raises ValueError."""

        class StrictEstimator(BaseEstimator):
            def fit(self, X, y):
                return self

            def predict(self, X):
                return np.zeros(_nrows(X))

        from yohou.base.reduction import BaseReductionForecaster

        sw = np.array([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="StrictEstimator"):
            BaseReductionForecaster._resolve_sample_weight_params(StrictEstimator(), sw)


class TestParameterValidation:
    def test_invalid_nan_handling_raises(self):
        """4.12: Invalid nan_handling value raises ValueError."""
        forecaster = PointReductionForecaster(nan_handling="raise")  # type: ignore[arg-type]
        with pytest.raises(ValueError):
            forecaster.fit(y=_make_y(20), forecasting_horizon=2)
