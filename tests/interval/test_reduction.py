from datetime import datetime

import numpy as np
import polars as pl
import polars.testing
import pytest
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import train_test_split

from conftest import run_checks
from yohou.interval import IntervalReductionForecaster
from yohou.testing import _yield_yohou_forecaster_checks


@pytest.fixture(scope="module")
def standard_splits():
    """Deterministic standard data and train/test splits."""
    length = 22
    time = pl.DataFrame({
        "time": pl.datetime_range(
            start=datetime(2021, 12, 16),
            end=datetime(2021, 12, 16, 0, 0, length - 1),
            interval="1s",
            eager=True,
        ),
    })
    y = pl.DataFrame(
        {"a": range(length), "b": range(10, length + 10)},
        schema={"a": pl.Float64, "b": pl.Float64},
    )
    y = pl.concat([time, y], how="horizontal")
    X_actual = pl.DataFrame(
        {"c": range(length), "d": range(10, length + 10), "e": range(20, length + 20)},
        schema={"c": pl.Float64, "d": pl.Float64, "e": pl.Float64},
    )
    X_actual = pl.concat([time, X_actual], how="horizontal")
    return train_test_split(y, X_actual, test_size=0.2, shuffle=False)


@pytest.fixture(scope="module")
def panel_splits():
    """Deterministic panel data and train/test splits."""
    length = 22
    time = pl.DataFrame({
        "time": pl.datetime_range(
            start=datetime(2021, 12, 16),
            end=datetime(2021, 12, 16, 0, 0, length - 1),
            interval="1s",
            eager=True,
        ),
    })
    y_panel = pl.DataFrame({
        "x__a": range(length),
        "x__b": range(10, length + 10),
        "y__a": range(10, length + 10),
        "y__b": range(20, length + 20),
    })
    y_panel = pl.concat([time, y_panel], how="horizontal")
    X_actual_panel = pl.DataFrame({
        "x__c": range(length),
        "y__c": range(10, length + 10),
        "d": range(10, length + 10),
        "e": range(20, length + 20),
    })
    X_actual_panel = pl.concat([time, X_actual_panel], how="horizontal")
    return train_test_split(y_panel, X_actual_panel, test_size=0.2, shuffle=False)


class TestPredict:
    @pytest.mark.slow
    @pytest.mark.parametrize(
        "fit_forecasting_horizon, predict_forecasting_horizon",
        [
            (3, 2),
            (5, 5),
        ],
    )
    def test_predict(self, fit_forecasting_horizon, predict_forecasting_horizon, standard_splits):
        y_train, y_test, X_actual_train, X_actual_test = standard_splits
        coverage_rates = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        forecaster = IntervalReductionForecaster()

        forecaster.fit(
            y=y_train,
            X_actual=X_actual_train,
            forecasting_horizon=fit_forecasting_horizon,
            coverage_rates=coverage_rates,
        )

        y_pred = forecaster.predict_interval(
            forecasting_horizon=predict_forecasting_horizon,
            predict_transformed=False,
            coverage_rates=coverage_rates,
        )

        # Extract non-time column names from y_train
        y_columns = [col for col in y_train.columns if col != "time"]
        for col in y_columns:
            for coverage_rate in coverage_rates:
                assert all(y_pred[f"{col}_upper_{coverage_rate}"] + 1e-14 >= y_pred[f"{col}_lower_{coverage_rate}"])


class TestObservePredict:
    @pytest.mark.slow
    @pytest.mark.parametrize(
        "fit_forecasting_horizon, predict_forecasting_horizon, stride",
        [
            (1, 1, 1),
            (3, 3, 2),
            (3, 2, 1),
        ],
    )
    def test_observe_predict(self, fit_forecasting_horizon, predict_forecasting_horizon, stride, standard_splits):
        """Test interval observe_predict (non-recursive predict)."""
        y_train, y_test, X_actual_train, X_actual_test = standard_splits
        coverage_rates = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        forecaster = IntervalReductionForecaster()

        forecaster.fit(
            y=y_train,
            X_actual=X_actual_train,
            forecasting_horizon=fit_forecasting_horizon,
            coverage_rates=coverage_rates,
        )

        y_pred = forecaster.observe_predict_interval(
            y=y_test,
            X_actual=X_actual_test,
            forecasting_horizon=predict_forecasting_horizon,
            stride=stride,
            coverage_rates=coverage_rates,
        )

        # Extract non-time column names from y_train
        y_columns = [col for col in y_train.columns if col != "time"]
        for col in y_columns:
            for coverage_rate in coverage_rates:
                assert all(y_pred[f"{col}_upper_{coverage_rate}"] + 1e-14 >= y_pred[f"{col}_lower_{coverage_rate}"])


class TestObservePredictGlobal:
    @pytest.mark.slow
    @pytest.mark.parametrize(
        "fit_forecasting_horizon, predict_forecasting_horizon, stride",
        [
            (1, 1, 1),
            (3, 3, 2),
            (3, 2, 1),
        ],
    )
    def test_observe_predict_global(self, fit_forecasting_horizon, predict_forecasting_horizon, stride, panel_splits):
        """Test panel interval observe_predict (non-recursive predict)."""
        y_train_panel, y_test_panel, X_train_panel, X_test_panel = panel_splits
        coverage_rates = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        forecaster = IntervalReductionForecaster()

        forecaster.fit(
            y=y_train_panel,
            X_actual=X_train_panel,
            forecasting_horizon=fit_forecasting_horizon,
            coverage_rates=coverage_rates,
        )

        y_pred = forecaster.observe_predict_interval(
            y=y_test_panel,
            X_actual=X_test_panel,
            forecasting_horizon=predict_forecasting_horizon,
            stride=stride,
            coverage_rates=coverage_rates,
        )

        # Check that upper bounds >= lower bounds for columns with __ separator
        # Columns are like group1__a_lower_0.1, group1__a_upper_0.1, etc.
        for col in y_pred.columns:
            if col in ["time", "vintage_time"]:
                continue
            if "_upper_" in col:
                # Extract coverage rate and find corresponding lower bound
                parts = col.split("_upper_")
                if len(parts) == 2:
                    lower_col = f"{parts[0]}_lower_{parts[1]}"
                    if lower_col in y_pred.columns:
                        assert all(y_pred[col] + 1e-13 >= y_pred[lower_col]), (
                            f"Upper bound {col} should be >= lower bound {lower_col}"
                        )


class _MockMultiQuantileRegressor(BaseEstimator, RegressorMixin):
    """Lightweight stand-in for CatBoostRegressor with MultiQuantile loss.

    Stores quantiles parsed from ``loss_function`` and returns linear
    predictions (intercept + slope) per quantile.  This is intentionally
    simple: it just validates the code path through
    ``IntervalReductionForecaster``.
    """

    def __init__(self, loss_function="MultiQuantile:alpha=0.025,0.975"):
        self.loss_function = loss_function

    def _parse_quantiles(self):
        prefix = "MultiQuantile:alpha="
        alpha_str = self.loss_function[len(prefix) :]
        return [float(q) for q in alpha_str.split(",")]

    def fit(self, X_actual, y, **kwargs):
        self.quantiles_ = self._parse_quantiles()
        self.n_quantiles_ = len(self.quantiles_)
        # Crude linear fit: slope and intercept per quantile
        self.intercept_ = np.zeros(self.n_quantiles_)
        self.slope_ = np.zeros(self.n_quantiles_)
        y = np.asarray(y)
        if y.ndim == 2:
            y = y[:, 0]
        for i, q in enumerate(self.quantiles_):
            self.intercept_[i] = np.quantile(y, q)
            self.slope_[i] = 0.01 * q
        return self

    def predict(self, X_actual):
        n = X_actual.shape[0]
        preds = np.empty((n, self.n_quantiles_))
        for i in range(self.n_quantiles_):
            preds[:, i] = self.intercept_[i] + self.slope_[i] * np.mean(X_actual, axis=1)
        return preds


class TestDirectStrategyInterval:
    """Tests for direct reduction strategy on IntervalReductionForecaster."""

    @pytest.mark.slow
    def test_estimator_dict_contains_lists(self, standard_splits):
        """Direct strategy stores lists (of H estimators) in the estimator_ dict."""
        y_train, _y_test, X_actual_train, _X_test = standard_splits
        coverage_rates = [0.5, 0.9]
        forecaster = IntervalReductionForecaster(reduction_strategy="direct")
        forecaster.fit(y=y_train, X_actual=X_actual_train, forecasting_horizon=3, coverage_rates=coverage_rates)

        for key, value in forecaster.estimator_.items():
            assert isinstance(value, list), f"Expected list for key {key}, got {type(value)}"
            assert len(value) == 3, f"Expected 3 estimators for key {key}, got {len(value)}"

    @pytest.mark.slow
    @pytest.mark.parametrize(
        "fit_forecasting_horizon, predict_forecasting_horizon",
        [(3, 2), (5, 5)],
    )
    def test_predict_shape_and_bounds(
        self,
        fit_forecasting_horizon,
        predict_forecasting_horizon,
        standard_splits,
    ):
        """Direct interval predictions have correct shape and upper >= lower."""
        y_train, _y_test, X_actual_train, X_actual_test = standard_splits
        coverage_rates = [0.5, 0.9]
        forecaster = IntervalReductionForecaster(reduction_strategy="direct")
        forecaster.fit(
            y=y_train,
            X_actual=X_actual_train,
            forecasting_horizon=fit_forecasting_horizon,
            coverage_rates=coverage_rates,
        )

        y_pred = forecaster.predict_interval(
            forecasting_horizon=predict_forecasting_horizon,
            coverage_rates=coverage_rates,
        )

        assert y_pred.shape[0] == predict_forecasting_horizon
        y_cols = [c for c in y_train.columns if c != "time"]
        for col in y_cols:
            for cr in coverage_rates:
                assert all(y_pred[f"{col}_upper_{cr}"] + 1e-14 >= y_pred[f"{col}_lower_{cr}"])

    @pytest.mark.slow
    @pytest.mark.parametrize("reduction_strategy", ["direct", "dir-rec"])
    def test_predict_panel(self, reduction_strategy, panel_splits):
        """Direct and dir-rec strategies work with panel interval data."""
        y_train, _y_test, X_actual_train, X_actual_test = panel_splits
        coverage_rates = [0.9]
        forecaster = IntervalReductionForecaster(reduction_strategy=reduction_strategy)
        forecaster.fit(y=y_train, X_actual=X_actual_train, forecasting_horizon=3, coverage_rates=coverage_rates)

        y_pred = forecaster.predict_interval(
            forecasting_horizon=3,
            coverage_rates=coverage_rates,
        )

        assert y_pred.shape[0] == 3
        for group in ["x", "y"]:
            for target in ["a", "b"]:
                for cr in coverage_rates:
                    assert f"{group}__{target}_lower_{cr}" in y_pred.columns
                    assert f"{group}__{target}_upper_{cr}" in y_pred.columns


class TestDirRecStrategyInterval:
    """Tests for dir-rec reduction strategy on IntervalReductionForecaster."""

    @pytest.mark.slow
    def test_estimator_dict_contains_lists(self, standard_splits):
        """Dir-rec strategy stores lists (of H estimators) in the estimator_ dict."""
        y_train, _y_test, X_actual_train, _X_test = standard_splits
        coverage_rates = [0.5]
        forecaster = IntervalReductionForecaster(reduction_strategy="dir-rec")
        forecaster.fit(y=y_train, X_actual=X_actual_train, forecasting_horizon=3, coverage_rates=coverage_rates)

        for key, value in forecaster.estimator_.items():
            assert isinstance(value, list), f"Expected list for key {key}, got {type(value)}"
            assert len(value) == 3

    @pytest.mark.slow
    @pytest.mark.parametrize(
        "fit_forecasting_horizon, predict_forecasting_horizon",
        [(3, 2), (5, 5)],
    )
    def test_predict_shape_and_bounds(
        self,
        fit_forecasting_horizon,
        predict_forecasting_horizon,
        standard_splits,
    ):
        """Dir-rec interval predictions have correct shape and upper >= lower."""
        y_train, _y_test, X_actual_train, X_actual_test = standard_splits
        coverage_rates = [0.5, 0.9]
        forecaster = IntervalReductionForecaster(reduction_strategy="dir-rec")
        forecaster.fit(
            y=y_train,
            X_actual=X_actual_train,
            forecasting_horizon=fit_forecasting_horizon,
            coverage_rates=coverage_rates,
        )

        y_pred = forecaster.predict_interval(
            forecasting_horizon=predict_forecasting_horizon,
            coverage_rates=coverage_rates,
        )

        assert y_pred.shape[0] == predict_forecasting_horizon
        y_cols = [c for c in y_train.columns if c != "time"]
        for col in y_cols:
            for cr in coverage_rates:
                assert all(y_pred[f"{col}_upper_{cr}"] + 1e-14 >= y_pred[f"{col}_lower_{cr}"])


class TestObservePredictDirectDirRecInterval:
    """Tests for observe_predict_interval with direct and dir-rec strategies."""

    @pytest.mark.slow
    @pytest.mark.parametrize("strategy", ["direct", "dir-rec"])
    def test_observe_predict_interval(self, strategy, standard_splits):
        """observe_predict_interval works for direct and dir-rec."""
        y_train, y_test, X_actual_train, X_actual_test = standard_splits
        coverage_rates = [0.5, 0.9]
        forecaster = IntervalReductionForecaster(reduction_strategy=strategy)
        forecaster.fit(
            y=y_train,
            X_actual=X_actual_train,
            forecasting_horizon=3,
            coverage_rates=coverage_rates,
        )

        y_pred = forecaster.observe_predict_interval(
            y=y_test,
            X_actual=X_actual_test,
            forecasting_horizon=3,
            stride=1,
            coverage_rates=coverage_rates,
        )

        y_cols = [c for c in y_train.columns if c != "time"]
        for col in y_cols:
            for cr in coverage_rates:
                assert f"{col}_upper_{cr}" in y_pred.columns
                assert f"{col}_lower_{cr}" in y_pred.columns
                assert all(y_pred[f"{col}_upper_{cr}"] + 1e-14 >= y_pred[f"{col}_lower_{cr}"])


class TestMultiQuantile:
    """Tests for the multi-quantile (CatBoost-style) fast path."""

    def test_detect_multiquantile_loss(self):
        """_detect_multiquantile_loss returns the param name for MQ estimators."""
        est = _MockMultiQuantileRegressor()
        forecaster = IntervalReductionForecaster(estimator=est)
        assert forecaster._detect_multiquantile_loss() == "loss_function"

    def test_detect_multiquantile_loss_absent(self):
        """_detect_multiquantile_loss returns None for standard estimators."""
        forecaster = IntervalReductionForecaster()
        assert forecaster._detect_multiquantile_loss() is None

    @pytest.mark.slow
    def test_fit_predict_global(self, standard_splits):
        """Multi-quantile path produces correct interval columns (global, single target).

        A MultiQuantile estimator is single-output and single-step, so the
        path is only valid for one target column at ``forecasting_horizon=1``;
        it returns exactly one row.
        """
        y_train, y_test, X_actual_train, X_actual_test = standard_splits
        y_train = y_train.select(["time", "a"])
        coverage_rates = [0.5, 0.9]
        est = _MockMultiQuantileRegressor()
        forecaster = IntervalReductionForecaster(estimator=est)

        forecaster.fit(
            y=y_train,
            X_actual=X_actual_train,
            forecasting_horizon=1,
            coverage_rates=coverage_rates,
        )

        # Only one estimator should be stored
        assert "_multiquantile" in forecaster.estimator_
        assert len(forecaster.estimator_) == 1

        y_pred = forecaster.predict_interval(
            forecasting_horizon=1,
            X_actual=X_actual_test,
            coverage_rates=coverage_rates,
        )

        assert len(y_pred) == 1
        for cr in coverage_rates:
            assert f"a_lower_{cr}" in y_pred.columns
            assert f"a_upper_{cr}" in y_pred.columns
            assert all(y_pred[f"a_upper_{cr}"] + 1e-14 >= y_pred[f"a_lower_{cr}"])

    @pytest.mark.slow
    def test_fit_predict_panel_single_target(self, panel_splits):
        """Multi-quantile path is valid for one target per group, horizon=1."""
        y_train_panel, y_test_panel, X_train_panel, X_test_panel = panel_splits
        # One target ("a") per group keeps the per-group target count at 1.
        y_train_panel = y_train_panel.select(["time", "x__a", "y__a"])
        coverage_rates = [0.9]
        est = _MockMultiQuantileRegressor()
        forecaster = IntervalReductionForecaster(estimator=est)

        forecaster.fit(
            y=y_train_panel,
            X_actual=X_train_panel,
            forecasting_horizon=1,
            coverage_rates=coverage_rates,
        )

        y_pred = forecaster.predict_interval(
            forecasting_horizon=1,
            X_actual=X_test_panel,
            coverage_rates=coverage_rates,
        )

        for group in ["x", "y"]:
            for cr in coverage_rates:
                assert f"{group}__a_lower_{cr}" in y_pred.columns
                assert f"{group}__a_upper_{cr}" in y_pred.columns

    @pytest.mark.slow
    def test_multiquantile_rejects_multi_target(self, standard_splits):
        """Multi-target input is rejected at fit (single-output estimator).

        A single-output estimator cannot serve distinct quantiles per target,
        so multi-target input must be rejected at fit rather than silently
        reusing the first target's quantiles for every target.
        """
        y_train, _, X_actual_train, _ = standard_splits
        forecaster = IntervalReductionForecaster(estimator=_MockMultiQuantileRegressor())
        with pytest.raises(ValueError, match="single target column"):
            forecaster.fit(y=y_train, X_actual=X_actual_train, forecasting_horizon=1, coverage_rates=[0.9])

    @pytest.mark.slow
    def test_multiquantile_rejects_multi_step(self, standard_splits):
        """Horizon > 1 is rejected at fit (single-step estimator).

        A single-step estimator can only emit one row, so a horizon greater
        than 1 must be rejected at fit rather than silently returning a single
        row regardless of the fitted horizon.
        """
        y_train, _, X_actual_train, _ = standard_splits
        y_train = y_train.select(["time", "a"])
        forecaster = IntervalReductionForecaster(estimator=_MockMultiQuantileRegressor())
        with pytest.raises(ValueError, match="forecasting_horizon=1"):
            forecaster.fit(y=y_train, X_actual=X_actual_train, forecasting_horizon=2, coverage_rates=[0.9])

    @pytest.mark.slow
    def test_observe_predict_multiquantile(self, standard_splits):
        """Multi-quantile path works with observe_predict_interval (single target)."""
        y_train, y_test, X_actual_train, X_actual_test = standard_splits
        y_train = y_train.select(["time", "a"])
        y_test = y_test.select(["time", "a"])
        coverage_rates = [0.5, 0.9]
        est = _MockMultiQuantileRegressor()
        forecaster = IntervalReductionForecaster(estimator=est)

        forecaster.fit(
            y=y_train,
            X_actual=X_actual_train,
            forecasting_horizon=1,
            coverage_rates=coverage_rates,
        )

        y_test_truncated = y_test[:-1]
        y_pred = forecaster.observe_predict_interval(
            y=y_test_truncated,
            X_actual=X_actual_test,
            forecasting_horizon=1,
            stride=1,
            coverage_rates=coverage_rates,
        )

        for cr in coverage_rates:
            assert f"a_lower_{cr}" in y_pred.columns
            assert f"a_upper_{cr}" in y_pred.columns


class _MockLGBMQuantileRegressor(BaseEstimator, RegressorMixin):
    """Lightweight stand-in for LGBMRegressor with ``objective="quantile"``.

    Mimics the LightGBM pattern where the quantile level is controlled
    by an ``alpha`` parameter rather than a ``quantile`` parameter.
    """

    def __init__(self, objective="quantile", alpha=0.5):
        self.objective = objective
        self.alpha = alpha

    def fit(self, X_actual, y, **kwargs):
        y = np.asarray(y)
        if y.ndim == 2:
            y = y[:, 0]
        self.intercept_ = np.quantile(y, self.alpha)
        return self

    def predict(self, X_actual):
        return np.full(X_actual.shape[0], self.intercept_)


class TestLGBMQuantileAlpha:
    """Tests for LightGBM-style ``objective='quantile'`` + ``alpha`` detection."""

    def test_detect_lgbm_quantile_alpha(self):
        """_detect_lgbm_quantile_alpha returns alpha param for LGBM-style estimators."""
        est = _MockLGBMQuantileRegressor()
        forecaster = IntervalReductionForecaster(estimator=est)
        assert forecaster._detect_lgbm_quantile_alpha() == ["alpha"]

    def test_detect_lgbm_quantile_alpha_nested(self):
        """_detect_lgbm_quantile_alpha resolves nested alpha param path."""
        from sklearn.multioutput import MultiOutputRegressor

        est = MultiOutputRegressor(_MockLGBMQuantileRegressor())
        forecaster = IntervalReductionForecaster(estimator=est)
        assert forecaster._detect_lgbm_quantile_alpha() == ["estimator__alpha"]

    def test_detect_lgbm_quantile_alpha_absent(self):
        """_detect_lgbm_quantile_alpha returns empty list for standard estimators."""
        forecaster = IntervalReductionForecaster()
        assert forecaster._detect_lgbm_quantile_alpha() == []

    @pytest.mark.slow
    def test_fit_predict_direct(self, standard_splits):
        """LGBM-style alpha detection produces correct interval columns (direct)."""
        y_train, y_test, X_actual_train, X_actual_test = standard_splits
        coverage_rates = [0.5, 0.9]
        est = _MockLGBMQuantileRegressor()
        forecaster = IntervalReductionForecaster(
            estimator=est,
            reduction_strategy="direct",
        )

        forecaster.fit(
            y=y_train,
            X_actual=X_actual_train,
            forecasting_horizon=2,
            coverage_rates=coverage_rates,
        )

        y_pred = forecaster.predict_interval(
            forecasting_horizon=2,
            X_actual=X_actual_test,
            coverage_rates=coverage_rates,
        )

        y_cols = [c for c in y_train.columns if c != "time"]
        for col in y_cols:
            for cr in coverage_rates:
                assert f"{col}_lower_{cr}" in y_pred.columns
                assert f"{col}_upper_{cr}" in y_pred.columns
                assert all(y_pred[f"{col}_upper_{cr}"] + 1e-14 >= y_pred[f"{col}_lower_{cr}"])

    def test_no_quantile_param_raises(self):
        """Estimator without quantile or alpha raises ValueError."""

        class _NoQuantileEstimator(BaseEstimator, RegressorMixin):
            def fit(self, X_actual, y):
                return self

            def predict(self, X_actual):
                return np.zeros(X_actual.shape[0])

        forecaster = IntervalReductionForecaster(estimator=_NoQuantileEstimator())
        y = pl.DataFrame({
            "time": pl.datetime_range(
                start=datetime(2021, 1, 1),
                end=datetime(2021, 1, 1, 0, 0, 9),
                interval="1s",
                eager=True,
            ),
            "value": list(range(10)),
        }).cast({"value": pl.Float64})

        with pytest.raises(ValueError, match="No quantile parameter found"):
            forecaster.fit(y=y, forecasting_horizon=1, coverage_rates=[0.9])


class TestIntervalReductionNanHandling:
    """Tests for ``nan_handling='drop'`` on the interval reduction path."""

    @pytest.mark.slow
    def test_drop_fits_and_predicts_with_nan_features(self):
        """``nan_handling='drop'`` drops NaN training rows and still yields valid intervals.

        The drop path lives in ``base.reduction`` and is routed through
        ``IntervalReductionForecaster._estimator_predict_one`` only for interval
        forecasters; it is otherwise untested in ``tests/interval/``.
        """
        length = 30
        time = pl.datetime_range(
            start=datetime(2021, 1, 1),
            end=datetime(2021, 1, 1, 0, 0, length - 1),
            interval="1s",
            eager=True,
        )
        y = pl.DataFrame({"time": time, "a": [float(i) for i in range(length)]})

        feature = [float(i) for i in range(length)]
        feature[5] = float("nan")
        feature[12] = float("nan")
        X_actual = pl.DataFrame({"time": time, "c": feature})

        coverage_rates = [0.9]
        forecaster = IntervalReductionForecaster(nan_handling="drop")
        with pytest.warns(UserWarning, match="NaN handling dropped"):
            forecaster.fit(
                y=y[:24],
                X_actual=X_actual[:24],
                forecasting_horizon=2,
                coverage_rates=coverage_rates,
            )

        y_pred = forecaster.predict_interval(
            forecasting_horizon=2,
            X_actual=X_actual[24:],
            coverage_rates=coverage_rates,
        )

        assert y_pred.shape[0] == 2
        for cr in coverage_rates:
            assert f"a_lower_{cr}" in y_pred.columns
            assert f"a_upper_{cr}" in y_pred.columns
            assert all(y_pred[f"a_upper_{cr}"] + 1e-14 >= y_pred[f"a_lower_{cr}"])


class TestIntervalReductionWithFeaturesSystematicChecks:
    """Systematic checks for IntervalReductionForecaster with exogenous features."""

    @pytest.mark.slow
    def test_interval_reduction_with_features_checks(self, y_X_factory):
        """Run all standard forecaster checks on IntervalReductionForecaster with X_actual."""
        y, X_actual = y_X_factory(length=100, n_targets=1, n_features=2, seed=42)
        y_train, y_test = y[:80], y[80:]
        X_actual_train, X_actual_test = X_actual[:80], X_actual[80:]

        forecaster = IntervalReductionForecaster()
        forecaster.fit(y_train, X_actual_train, forecasting_horizon=3)

        run_checks(
            forecaster,
            _yield_yohou_forecaster_checks(forecaster, y_train, X_actual_train, y_test, X_actual_test),
        )


class TestMultiStepPanelInterval:
    """Tests for multi-step interval prediction with panel data."""

    @pytest.mark.slow
    def test_multi_step_panel_predict_interval(self, panel_splits):
        """Multi-step interval prediction works with panel data (recursive, no X_actual)."""
        y_train, _y_test, X_actual_train, X_actual_test = panel_splits
        coverage_rates = [0.9]
        forecaster = IntervalReductionForecaster()
        forecaster.fit(y=y_train, forecasting_horizon=1, coverage_rates=coverage_rates)

        y_pred = forecaster.predict_interval(
            forecasting_horizon=3,
            coverage_rates=coverage_rates,
        )

        assert y_pred.shape[0] == 3
        for group in ["x", "y"]:
            for target in ["a", "b"]:
                for cr in coverage_rates:
                    assert f"{group}__{target}_lower_{cr}" in y_pred.columns
                    assert f"{group}__{target}_upper_{cr}" in y_pred.columns

    @pytest.mark.slow
    @pytest.mark.parametrize("recursion_strategy", ["mean"])
    def test_multi_step_interval_strategy(self, recursion_strategy, standard_splits):
        """Multi-step interval prediction works with bound-derived strategies (no X_actual)."""
        y_train, _y_test, X_actual_train, X_actual_test = standard_splits
        coverage_rates = [0.5]
        forecaster = IntervalReductionForecaster()
        forecaster.fit(
            y=y_train,
            forecasting_horizon=1,
            coverage_rates=coverage_rates,
        )

        y_pred = forecaster.predict_interval(
            forecasting_horizon=3,
            coverage_rates=coverage_rates,
            recursion_strategy=recursion_strategy,
        )

        assert y_pred.shape[0] == 3

    @pytest.mark.slow
    def test_multi_step_interval_strategy_point_without_point_column_raises(self, standard_splits):
        """recursion_strategy='point' raises when the forecaster emits no bare point column."""
        y_train, _y_test, X_actual_train, X_actual_test = standard_splits
        coverage_rates = [0.5]
        forecaster = IntervalReductionForecaster()
        forecaster.fit(
            y=y_train,
            forecasting_horizon=1,
            coverage_rates=coverage_rates,
        )

        with pytest.raises(ValueError, match="recursion_strategy='point' requires a bare point column"):
            forecaster.predict_interval(
                forecasting_horizon=3,
                coverage_rates=coverage_rates,
                recursion_strategy="point",
            )


class TestIntervalReductionTargetTransformer:
    """IntervalReductionForecaster exposes a target_transformer parameter."""

    def test_target_transformer_in_params(self):
        """target_transformer is a constructor parameter, like the other reduction families."""
        forecaster = IntervalReductionForecaster()
        assert "target_transformer" in forecaster.get_params()
        assert forecaster.get_params()["target_transformer"] is None

    def test_target_transformer_passes_through_to_reduction_machinery(self):
        """A passed target_transformer is stored and threaded into the reduction pipeline."""
        from sklearn.linear_model import QuantileRegressor
        from sklearn.multioutput import MultiOutputRegressor

        from yohou.stationarity import SeasonalDifferencing

        transformer = SeasonalDifferencing(seasonality=7)
        forecaster = IntervalReductionForecaster(
            estimator=MultiOutputRegressor(QuantileRegressor(alpha=0.0)),
            target_transformer=transformer,
        )
        # Stored on the instance and visible through get_params (and clone).
        assert forecaster.target_transformer is transformer
        assert forecaster.get_params()["target_transformer"] is transformer
        from sklearn.base import clone

        assert clone(forecaster).get_params()["target_transformer"] is not None

        time = pl.datetime_range(start=datetime(2020, 1, 1), end=datetime(2020, 2, 10), interval="1d", eager=True)
        y = pl.DataFrame({"time": time, "value": pl.Series(range(41), dtype=pl.Float64)})
        forecaster.fit(y[:30], forecasting_horizon=3, coverage_rates=[0.5])

        # The transformed-space schema reflects the seasonal differencing transform,
        # proving the target_transformer flowed into the reduction pipeline at fit time.
        assert forecaster.local_y_t_schema_ == {"diff_s_7_value": pl.Float64}
        assert forecaster.target_transformer_ is not None
