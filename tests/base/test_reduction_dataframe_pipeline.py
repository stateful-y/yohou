"""Tests verifying that the reduction pipeline passes Polars DataFrames to sklearn estimators.

The reduction pipeline passes Polars DataFrames (instead of numpy arrays) for
both X and y to the underlying sklearn estimator. This ensures that estimators
recording feature names during fit (e.g. LightGBM, XGBoost) see consistent
names at predict time.
"""

from datetime import datetime

import numpy as np
import polars as pl
import pytest
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression, Ridge

from yohou.point import PointReductionForecaster

LENGTH = 50


@pytest.fixture(scope="module")
def reduction_data():
    """Single-target linear trend with exogenous features."""
    time = pl.datetime_range(
        start=datetime(2021, 1, 1),
        end=datetime(2021, 1, 1, 0, 0, LENGTH - 1),
        interval="1s",
        eager=True,
    )
    y = pl.DataFrame({
        "time": time,
        "value": [2.0 * i + 10.0 for i in range(LENGTH)],
    })
    X = pl.DataFrame({
        "time": time,
        "feat_a": [float(i) for i in range(LENGTH)],
        "feat_b": [float(i) + 100.0 for i in range(LENGTH)],
    })
    return y[:40], y[40:], X[:40], X[40:]


@pytest.fixture(scope="module")
def multi_target_data():
    """Two-target dataset with exogenous features."""
    time = pl.datetime_range(
        start=datetime(2021, 1, 1),
        end=datetime(2021, 1, 1, 0, 0, LENGTH - 1),
        interval="1s",
        eager=True,
    )
    y = pl.DataFrame({
        "time": time,
        "sales": [float(i) for i in range(LENGTH)],
        "revenue": [float(i) * 2.5 for i in range(LENGTH)],
    })
    X = pl.DataFrame({
        "time": time,
        "promo": [float(i % 3) for i in range(LENGTH)],
        "weekday": [float(i % 7) for i in range(LENGTH)],
    })
    return y[:40], y[40:], X[:40], X[40:]


@pytest.fixture(scope="module")
def panel_data():
    """Two-group panel dataset."""
    time = pl.datetime_range(
        start=datetime(2021, 1, 1),
        end=datetime(2021, 1, 1, 0, 0, LENGTH - 1),
        interval="1s",
        eager=True,
    )
    y = pl.DataFrame({
        "time": time,
        "store_a__sales": [float(i) for i in range(LENGTH)],
        "store_b__sales": [float(i) + 5.0 for i in range(LENGTH)],
    })
    X = pl.DataFrame({
        "time": time,
        "store_a__promo": [float(i % 2) for i in range(LENGTH)],
        "store_b__promo": [float(i % 3) for i in range(LENGTH)],
        "weather": [float(i % 4) for i in range(LENGTH)],
    })
    return y[:40], y[40:], X[:40], X[40:]


class _SpyEstimator(BaseEstimator, RegressorMixin):
    """Estimator that records the types and column names of fit/predict inputs.

    Uses a class-level list to track predict calls because the forecaster's
    predict path deepcopies the forecaster (so instance state on the original
    is not updated).
    """

    predict_calls: list[dict] = []

    def __init__(self):
        self.fit_X_type = None
        self.fit_X_columns = None
        self.fit_y_type = None
        self.fit_y_columns = None
        self._inner = Ridge()

    @classmethod
    def reset_predict_calls(cls):
        cls.predict_calls = []

    def fit(self, X, y, **fit_params):
        self.fit_X_type = type(X)
        self.fit_y_type = type(y)
        self.fit_X_columns = list(X.columns) if hasattr(X, "columns") else None
        if isinstance(y, pl.DataFrame):
            self.fit_y_columns = list(y.columns)
        elif isinstance(y, pl.Series):
            self.fit_y_columns = [y.name]
        else:
            self.fit_y_columns = None
        self._inner.fit(X, y)
        return self

    def predict(self, X):
        _SpyEstimator.predict_calls.append({
            "X_type": type(X),
            "X_columns": list(X.columns) if hasattr(X, "columns") else None,
        })
        return self._inner.predict(X)


def _get_fitted_estimator(forecaster):
    """Return the first fitted estimator (handles both single and list)."""
    est = forecaster.estimator_
    return est[0] if isinstance(est, list) else est


class TestFitReceivesDataFrames:
    """Verify that the estimator receives DataFrames at fit time."""

    @pytest.mark.parametrize("strategy", ["multi-output", "direct", "dir-rec"])
    def test_X_is_dataframe(self, reduction_data, strategy):
        """The estimator's fit() receives a Polars DataFrame for X."""
        y_train, _, X_train, _ = reduction_data
        spy = _SpyEstimator()
        forecaster = PointReductionForecaster(estimator=spy, reduction_strategy=strategy)
        forecaster.fit(y=y_train, X=X_train, forecasting_horizon=3)

        assert _get_fitted_estimator(forecaster).fit_X_type is pl.DataFrame

    @pytest.mark.parametrize("strategy", ["multi-output", "direct", "dir-rec"])
    def test_y_is_polars_type(self, reduction_data, strategy):
        """The estimator's fit() receives a Polars DataFrame or Series for y."""
        y_train, _, X_train, _ = reduction_data
        spy = _SpyEstimator()
        forecaster = PointReductionForecaster(estimator=spy, reduction_strategy=strategy)
        forecaster.fit(y=y_train, X=X_train, forecasting_horizon=3)

        assert _get_fitted_estimator(forecaster).fit_y_type in (pl.DataFrame, pl.Series)

    def test_single_target_direct_y_is_series(self, reduction_data):
        """Single-target direct passes a Series for y (not a 1-column DataFrame)."""
        y_train, _, X_train, _ = reduction_data
        spy = _SpyEstimator()
        forecaster = PointReductionForecaster(estimator=spy, reduction_strategy="direct")
        forecaster.fit(y=y_train, X=X_train, forecasting_horizon=3)

        for est in forecaster.estimator_:
            assert est.fit_y_type is pl.Series

    def test_single_target_dir_rec_y_is_series(self, reduction_data):
        """Single-target dir-rec passes a Series for y."""
        y_train, _, X_train, _ = reduction_data
        spy = _SpyEstimator()
        forecaster = PointReductionForecaster(estimator=spy, reduction_strategy="dir-rec")
        forecaster.fit(y=y_train, X=X_train, forecasting_horizon=3)

        for est in forecaster.estimator_:
            assert est.fit_y_type is pl.Series

    def test_multi_target_multi_output_y_is_dataframe(self, multi_target_data):
        """Multi-target multi-output passes a DataFrame for y."""
        y_train, _, X_train, _ = multi_target_data
        spy = _SpyEstimator()
        forecaster = PointReductionForecaster(estimator=spy, reduction_strategy="multi-output")
        forecaster.fit(y=y_train, X=X_train, forecasting_horizon=3)

        assert forecaster.estimator_.fit_y_type is pl.DataFrame


class TestPredictReceivesDataFrames:
    """Verify that the estimator receives DataFrames at predict time."""

    @pytest.mark.parametrize("strategy", ["multi-output", "direct", "dir-rec"])
    def test_X_is_dataframe(self, reduction_data, strategy):
        """The estimator's predict() receives a Polars DataFrame for X."""
        y_train, _, X_train, X_test = reduction_data
        spy = _SpyEstimator()
        _SpyEstimator.reset_predict_calls()
        forecaster = PointReductionForecaster(estimator=spy, reduction_strategy=strategy)
        forecaster.fit(y=y_train, X=X_train, forecasting_horizon=3)
        forecaster.predict(X=X_test[:3], forecasting_horizon=3)

        assert len(_SpyEstimator.predict_calls) > 0
        for call in _SpyEstimator.predict_calls:
            assert call["X_type"] is pl.DataFrame


class TestFeatureNamesConsistency:
    """Verify that feature names are consistent between fit and predict."""

    @pytest.mark.parametrize("strategy", ["multi-output", "direct", "dir-rec"])
    def test_fit_predict_columns_match(self, reduction_data, strategy):
        """Feature columns passed to fit and predict are identical."""
        y_train, _, X_train, X_test = reduction_data
        spy = _SpyEstimator()
        _SpyEstimator.reset_predict_calls()
        forecaster = PointReductionForecaster(estimator=spy, reduction_strategy=strategy)
        forecaster.fit(y=y_train, X=X_train, forecasting_horizon=3)
        forecaster.predict(X=X_test[:3], forecasting_horizon=3)

        fitted = _get_fitted_estimator(forecaster)
        # The first predict call corresponds to the first estimator
        assert _SpyEstimator.predict_calls[0]["X_columns"] == fitted.fit_X_columns

    def test_columns_include_lagged_target(self, reduction_data):
        """Feature columns include lagged target when target_as_feature is set."""
        y_train, _, X_train, X_test = reduction_data
        spy = _SpyEstimator()
        forecaster = PointReductionForecaster(estimator=spy, target_as_feature="transformed")
        forecaster.fit(y=y_train, X=X_train, forecasting_horizon=2)

        columns = forecaster.estimator_.fit_X_columns
        assert any("value" in c for c in columns), f"Expected lagged target in columns: {columns}"

    def test_columns_exclude_time(self, reduction_data):
        """Feature columns never include the 'time' column."""
        y_train, _, X_train, _ = reduction_data
        spy = _SpyEstimator()
        forecaster = PointReductionForecaster(estimator=spy)
        forecaster.fit(y=y_train, X=X_train, forecasting_horizon=2)

        assert "time" not in _get_fitted_estimator(forecaster).fit_X_columns

    @pytest.mark.parametrize("strategy", ["multi-output", "direct", "dir-rec"])
    def test_panel_fit_predict_columns_match(self, panel_data, strategy):
        """Panel data: feature columns match between fit and predict."""
        y_train, _, X_train, X_test = panel_data
        spy = _SpyEstimator()
        _SpyEstimator.reset_predict_calls()
        forecaster = PointReductionForecaster(estimator=spy, reduction_strategy=strategy)
        forecaster.fit(y=y_train, X=X_train, forecasting_horizon=3)
        forecaster.predict(X=X_test[:3], forecasting_horizon=3)

        fitted = _get_fitted_estimator(forecaster)
        assert _SpyEstimator.predict_calls[0]["X_columns"] == fitted.fit_X_columns


class TestYTabColumnNames:
    """Verify that y passed to the estimator has meaningful column names."""

    def test_multi_output_y_has_step_columns(self, reduction_data):
        """Multi-output y has columns named {target}_step_{h}."""
        y_train, _, X_train, _ = reduction_data
        spy = _SpyEstimator()
        forecaster = PointReductionForecaster(estimator=spy, reduction_strategy="multi-output")
        forecaster.fit(y=y_train, X=X_train, forecasting_horizon=3)

        assert forecaster.estimator_.fit_y_columns == [
            "value_step_1", "value_step_2", "value_step_3",
        ]

    def test_direct_y_has_step_column_per_estimator(self, reduction_data):
        """Each direct estimator receives the column for its step."""
        y_train, _, X_train, _ = reduction_data
        spy = _SpyEstimator()
        forecaster = PointReductionForecaster(estimator=spy, reduction_strategy="direct")
        forecaster.fit(y=y_train, X=X_train, forecasting_horizon=3)

        for step, est in enumerate(forecaster.estimator_):
            assert est.fit_y_columns == [f"value_step_{step + 1}"]

    def test_dir_rec_y_has_step_column_per_estimator(self, reduction_data):
        """Each dir-rec estimator receives the column for its step."""
        y_train, _, X_train, _ = reduction_data
        spy = _SpyEstimator()
        forecaster = PointReductionForecaster(estimator=spy, reduction_strategy="dir-rec")
        forecaster.fit(y=y_train, X=X_train, forecasting_horizon=3)

        for step, est in enumerate(forecaster.estimator_):
            assert est.fit_y_columns == [f"value_step_{step + 1}"]

    def test_multi_target_multi_output_y_columns(self, multi_target_data):
        """Multi-target multi-output y has all target-step combinations."""
        y_train, _, X_train, _ = multi_target_data
        spy = _SpyEstimator()
        forecaster = PointReductionForecaster(estimator=spy, reduction_strategy="multi-output")
        forecaster.fit(y=y_train, X=X_train, forecasting_horizon=2)

        assert forecaster.estimator_.fit_y_columns == [
            "sales_step_1", "revenue_step_1", "sales_step_2", "revenue_step_2",
        ]

    def test_multi_target_direct_y_columns(self, multi_target_data):
        """Multi-target direct: each estimator gets its step for all targets."""
        y_train, _, X_train, _ = multi_target_data
        spy = _SpyEstimator()
        forecaster = PointReductionForecaster(estimator=spy, reduction_strategy="direct")
        forecaster.fit(y=y_train, X=X_train, forecasting_horizon=2)

        assert forecaster.estimator_[0].fit_y_columns == ["sales_step_1", "revenue_step_1"]
        assert forecaster.estimator_[1].fit_y_columns == ["sales_step_2", "revenue_step_2"]


class TestDirRecAugmentedFeatureNames:
    """Verify that dir-rec augmentation adds named columns progressively."""

    def test_augmented_columns_are_named_and_progressive(self, reduction_data):
        """Each subsequent dir-rec model has additional named columns."""
        y_train, _, X_train, _ = reduction_data
        spy = _SpyEstimator()
        forecaster = PointReductionForecaster(estimator=spy, reduction_strategy="dir-rec")
        forecaster.fit(y=y_train, X=X_train, forecasting_horizon=3)

        base_cols = forecaster.estimator_[0].fit_X_columns
        step1_cols = forecaster.estimator_[1].fit_X_columns
        step2_cols = forecaster.estimator_[2].fit_X_columns

        # Step 1 has more columns than step 0
        assert step1_cols[:len(base_cols)] == base_cols
        assert len(step1_cols) > len(base_cols)

        # Step 2 has more columns than step 1
        assert step2_cols[:len(base_cols)] == base_cols
        assert len(step2_cols) > len(step1_cols)

        # All columns are strings (no integer indices)
        for cols in [base_cols, step1_cols, step2_cols]:
            assert all(isinstance(c, str) for c in cols)

    def test_predict_passes_augmented_dataframe(self, reduction_data):
        """Dir-rec predict passes DataFrames with augmented columns to later estimators."""
        y_train, _, X_train, X_test = reduction_data
        spy = _SpyEstimator()
        _SpyEstimator.reset_predict_calls()
        forecaster = PointReductionForecaster(estimator=spy, reduction_strategy="dir-rec")
        forecaster.fit(y=y_train, X=X_train, forecasting_horizon=3)
        forecaster.predict(X=X_test[:3], forecasting_horizon=3)

        # All predict calls should receive DataFrames
        assert all(c["X_type"] is pl.DataFrame for c in _SpyEstimator.predict_calls)
        # The last estimator's predict call should have more columns than the first
        first_call = _SpyEstimator.predict_calls[0]
        last_call = _SpyEstimator.predict_calls[-1]
        assert len(last_call["X_columns"]) > len(first_call["X_columns"])

    def test_fit_predict_column_count_matches(self, reduction_data):
        """The last dir-rec estimator sees the same number of columns at fit and predict."""
        y_train, _, X_train, X_test = reduction_data
        spy = _SpyEstimator()
        _SpyEstimator.reset_predict_calls()
        forecaster = PointReductionForecaster(estimator=spy, reduction_strategy="dir-rec")
        forecaster.fit(y=y_train, X=X_train, forecasting_horizon=3)
        forecaster.predict(X=X_test[:3], forecasting_horizon=3)

        last_est = forecaster.estimator_[-1]
        last_predict_call = _SpyEstimator.predict_calls[-1]
        assert len(last_predict_call["X_columns"]) == len(last_est.fit_X_columns)


class TestNoFeatureNameWarning:
    """Verify no sklearn 'feature names' UserWarning is raised."""

    @pytest.mark.parametrize("strategy", ["multi-output", "direct", "dir-rec"])
    def test_no_warning_with_ridge(self, reduction_data, strategy):
        """Ridge predict emits no 'feature names' warning for any strategy."""
        import warnings

        y_train, _, X_train, X_test = reduction_data
        forecaster = PointReductionForecaster(estimator=Ridge(), reduction_strategy=strategy)
        forecaster.fit(y=y_train, X=X_train, forecasting_horizon=3)

        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            forecaster.predict(X=X_test[:3], forecasting_horizon=3)

        feature_warnings = [w for w in record if "feature names" in str(w.message).lower()]
        assert len(feature_warnings) == 0

    def test_no_warning_with_linear_regression(self, reduction_data):
        """LinearRegression predict emits no 'feature names' warning."""
        import warnings

        y_train, _, X_train, X_test = reduction_data
        forecaster = PointReductionForecaster(estimator=LinearRegression())
        forecaster.fit(y=y_train, X=X_train, forecasting_horizon=3)

        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            forecaster.predict(X=X_test[:3], forecasting_horizon=3)

        feature_warnings = [w for w in record if "feature names" in str(w.message).lower()]
        assert len(feature_warnings) == 0


class TestPanelDataFramePipeline:
    """Verify the DataFrame pipeline works with panel data."""

    @pytest.mark.parametrize("strategy", ["multi-output", "direct", "dir-rec"])
    def test_X_is_dataframe_at_fit(self, panel_data, strategy):
        """Panel: estimator's fit() receives a Polars DataFrame for X."""
        y_train, _, X_train, _ = panel_data
        spy = _SpyEstimator()
        forecaster = PointReductionForecaster(estimator=spy, reduction_strategy=strategy)
        forecaster.fit(y=y_train, X=X_train, forecasting_horizon=3)

        assert _get_fitted_estimator(forecaster).fit_X_type is pl.DataFrame

    @pytest.mark.parametrize("strategy", ["multi-output", "direct", "dir-rec"])
    def test_X_is_dataframe_at_predict(self, panel_data, strategy):
        """Panel: estimator's predict() receives a Polars DataFrame for X."""
        y_train, _, X_train, X_test = panel_data
        spy = _SpyEstimator()
        _SpyEstimator.reset_predict_calls()
        forecaster = PointReductionForecaster(estimator=spy, reduction_strategy=strategy)
        forecaster.fit(y=y_train, X=X_train, forecasting_horizon=3)
        forecaster.predict(X=X_test[:3], forecasting_horizon=3)

        assert len(_SpyEstimator.predict_calls) > 0
        for call in _SpyEstimator.predict_calls:
            assert call["X_type"] is pl.DataFrame

    @pytest.mark.parametrize("strategy", ["multi-output", "direct", "dir-rec"])
    def test_y_is_polars_type_at_fit(self, panel_data, strategy):
        """Panel: estimator's fit() receives a Polars DataFrame or Series for y."""
        y_train, _, X_train, _ = panel_data
        spy = _SpyEstimator()
        forecaster = PointReductionForecaster(estimator=spy, reduction_strategy=strategy)
        forecaster.fit(y=y_train, X=X_train, forecasting_horizon=3)

        assert _get_fitted_estimator(forecaster).fit_y_type in (pl.DataFrame, pl.Series)


class TestPredictionCorrectness:
    """Verify predictions remain numerically correct after the DataFrame change."""

    @pytest.mark.parametrize("strategy", ["multi-output", "direct", "dir-rec"])
    def test_linear_trend(self, strategy):
        """All strategies produce exact predictions on a perfect linear trend."""
        length = 60
        time = pl.datetime_range(
            start=datetime(2021, 1, 1),
            end=datetime(2021, 1, 1, 0, 0, length - 1),
            interval="1s",
            eager=True,
        )
        y = pl.DataFrame({"time": time, "value": [3.0 * i + 7.0 for i in range(length)]})

        forecaster = PointReductionForecaster(
            estimator=LinearRegression(),
            reduction_strategy=strategy,
        )
        forecaster.fit(y[:50], forecasting_horizon=3)
        y_pred = forecaster.predict(forecasting_horizon=3)

        expected = [3.0 * i + 7.0 for i in range(50, 53)]
        np.testing.assert_allclose(y_pred["value"].to_numpy(), expected, atol=1e-8)

    @pytest.mark.parametrize("strategy", ["multi-output", "direct", "dir-rec"])
    def test_constant_series(self, strategy):
        """All strategies predict the constant value for a constant series."""
        length = 40
        time = pl.datetime_range(
            start=datetime(2021, 1, 1),
            end=datetime(2021, 1, 1, 0, 0, length - 1),
            interval="1s",
            eager=True,
        )
        y = pl.DataFrame({"time": time, "value": [42.0] * length})

        forecaster = PointReductionForecaster(
            estimator=LinearRegression(),
            reduction_strategy=strategy,
        )
        forecaster.fit(y[:30], forecasting_horizon=3)
        y_pred = forecaster.predict(forecasting_horizon=3)

        np.testing.assert_allclose(y_pred["value"].to_numpy(), [42.0, 42.0, 42.0], atol=1e-8)

    def test_multi_target_shape_and_columns(self, multi_target_data):
        """Multi-target predictions have correct shape and column names."""
        y_train, _, X_train, X_test = multi_target_data
        forecaster = PointReductionForecaster(estimator=LinearRegression())
        forecaster.fit(y=y_train, X=X_train, forecasting_horizon=3)
        y_pred = forecaster.predict(X=X_test[:3], forecasting_horizon=3)

        assert y_pred.shape[0] == 3
        assert "sales" in y_pred.columns
        assert "revenue" in y_pred.columns
