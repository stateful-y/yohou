import warnings
from datetime import datetime, timedelta

import numpy as np
import polars as pl
import polars.selectors as cs
import pytest
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from conftest import run_checks
from yohou.point import PointReductionForecaster
from yohou.preprocessing import LagTransformer
from yohou.testing import _yield_yohou_forecaster_checks
from yohou.weighting import LookupWeighter, TableWeighter

LENGTH = 22


@pytest.fixture(scope="module")
def reduction_data():
    """Module-scoped fixture for standard reduction test data."""
    time = pl.DataFrame({
        "time": pl.datetime_range(
            start=datetime(2021, 12, 16),
            end=datetime(2021, 12, 16, 0, 0, LENGTH - 1),
            interval="1s",
            eager=True,
        ),
    })
    y = pl.DataFrame(
        {
            "a": range(LENGTH),
            "b": range(10, LENGTH + 10),
        },
        schema={
            "a": pl.Float64,
            "b": pl.Float64,
        },
    )
    y = pl.concat([time, y], how="horizontal")

    X_actual = pl.DataFrame(
        {
            "c": range(LENGTH),
            "d": range(10, LENGTH + 10),
            "e": range(20, LENGTH + 20),
        },
        schema={
            "c": pl.Float64,
            "d": pl.Float64,
            "e": pl.Float64,
        },
    )
    X_actual = pl.concat([time, X_actual], how="horizontal")

    y_train, y_test, X_actual_train, X_actual_test = train_test_split(y, X_actual, test_size=0.2, shuffle=False)
    return y_train, y_test, X_actual_train, X_actual_test


class TestPredict:
    @pytest.mark.parametrize(
        "fit_forecasting_horizon, predict_forecasting_horizon, expected_a",
        [
            (3, 2, [17.0, 18.0]),
            (5, 5, [17.0, 18.0, 19.0, 20.0, 21.0]),
        ],
    )
    def test_predict_with_x(self, reduction_data, fit_forecasting_horizon, predict_forecasting_horizon, expected_a):
        """Predict with X_actual (non-recursive: predict_fh <= fit_fh)."""
        y_train, y_test, X_actual_train, X_actual_test = reduction_data
        forecaster = PointReductionForecaster()

        forecaster.fit(y=y_train, X_actual=X_actual_train, forecasting_horizon=fit_forecasting_horizon)

        y_pred = forecaster.predict(
            forecasting_horizon=predict_forecasting_horizon,
        )

        expected_y_pred = pl.DataFrame(
            {
                "vintage_time": [y_train["time"][-1]] * predict_forecasting_horizon,
                "time": pl.datetime_range(
                    start=datetime(2021, 12, 16, 0, 0, len(y_train)),
                    end=datetime(2021, 12, 16, 0, 0, len(y_train) + predict_forecasting_horizon - 1),
                    interval="1s",
                    eager=True,
                ),
                "a": expected_a,
                "b": np.array(expected_a) + 10,
            },
            schema={
                "vintage_time": pl.Datetime(time_unit="us", time_zone=None),
                "time": pl.Datetime(time_unit="us", time_zone=None),
                "a": pl.Float64,
                "b": pl.Float64,
            },
        )
        pl.testing.assert_frame_equal(y_pred, expected_y_pred)

    def test_predict_recursive_no_x(self, reduction_data):
        """Recursive predict (predict_fh > fit_fh) without X_actual: shape-only check."""
        y_train, y_test, _, _ = reduction_data
        forecaster = PointReductionForecaster()

        forecaster.fit(y=y_train, forecasting_horizon=1)

        y_pred = forecaster.predict(forecasting_horizon=5)
        assert y_pred.shape[0] == 5
        assert set(y_pred.columns) == {"vintage_time", "time", "a", "b"}


class TestObservePredict:
    @pytest.mark.parametrize(
        "fit_forecasting_horizon, predict_forecasting_horizon, stride, expected_a",
        [
            (1, 1, 1, [22.0]),
            (3, 3, 2, [22.0, 23.0, 24.0]),
            (3, 2, 1, [22.0, 23.0]),
        ],
    )
    def test_observe_predict(
        self, reduction_data, fit_forecasting_horizon, predict_forecasting_horizon, stride, expected_a
    ):
        """Test observe_predict with exogenous features (non-recursive predict)."""
        y_train, y_test, X_actual_train, X_actual_test = reduction_data
        forecaster = PointReductionForecaster()

        forecaster.fit(y=y_train, X_actual=X_actual_train, forecasting_horizon=fit_forecasting_horizon)

        y_pred = forecaster.observe_predict(
            y=y_test,
            X_actual=X_actual_test,
            forecasting_horizon=predict_forecasting_horizon,
            stride=stride,
        )

        # Get the last set of predictions (after all observations)
        y_pred = y_pred.tail(predict_forecasting_horizon)

        expected_y_pred = pl.DataFrame(
            {
                "vintage_time": [y_test["time"][-1]] * predict_forecasting_horizon,
                "time": pl.datetime_range(
                    start=datetime(2021, 12, 16, 0, 0, len(y_train) + len(y_test)),
                    end=datetime(2021, 12, 16, 0, 0, len(y_train) + len(y_test) + predict_forecasting_horizon - 1),
                    interval="1s",
                    eager=True,
                ),
                "a": expected_a,
                "b": np.array(expected_a) + 10,
            },
            schema={
                "vintage_time": pl.Datetime(time_unit="us", time_zone=None),
                "time": pl.Datetime(time_unit="us", time_zone=None),
                "a": pl.Float64,
                "b": pl.Float64,
            },
        )
        pl.testing.assert_frame_equal(y_pred, expected_y_pred)

    def test_observe_predict_recursive_no_x(self, reduction_data):
        """Test recursive observe_predict without exogenous features.

        Recursive predict uses only target_as_feature for observation,
        so X_actual features are unavailable during recursive steps.
        """
        y_train, y_test, _, _ = reduction_data
        forecaster = PointReductionForecaster()

        forecaster.fit(y=y_train, forecasting_horizon=1)

        y_pred = forecaster.observe_predict(
            y=y_test,
            forecasting_horizon=5,
            stride=1,
        )

        # Verify shape: 6 vintages (1 initial + 5 observe steps), 5 rows each
        assert y_pred.shape[0] == (len(y_test) + 1) * 5
        assert "vintage_time" in y_pred.columns
        assert "time" in y_pred.columns


@pytest.fixture(scope="module")
def panel_reduction_data():
    """Module-scoped fixture for panel reduction test data."""
    time = pl.DataFrame({
        "time": pl.datetime_range(
            start=datetime(2021, 12, 16),
            end=datetime(2021, 12, 16, 0, 0, LENGTH - 1),
            interval="1s",
            eager=True,
        ),
    })
    y_panel = pl.DataFrame({
        "x__a": range(LENGTH),
        "x__b": range(10, LENGTH + 10),
        "y__a": range(10, LENGTH + 10),
        "y__b": range(20, LENGTH + 20),
    })
    y_panel = pl.concat([time, y_panel], how="horizontal")

    X_actual_panel = pl.DataFrame({
        "x__c": range(LENGTH),
        "y__c": range(10, LENGTH + 10),
        "d": range(10, LENGTH + 10),
        "e": range(20, LENGTH + 20),
    })
    X_actual_panel = pl.concat([time, X_actual_panel], how="horizontal")

    y_train_panel, y_test_panel, X_train_panel, X_test_panel = train_test_split(
        y_panel, X_actual_panel, test_size=0.2, shuffle=False
    )
    return y_train_panel, y_test_panel, X_train_panel, X_test_panel


class TestObservePredictGlobal:
    @pytest.mark.parametrize(
        "fit_forecasting_horizon, predict_forecasting_horizon, stride, expected_a",
        [
            (1, 1, 1, [22.0]),
            (3, 3, 2, [22.0, 23.0, 24.0]),
            (3, 2, 1, [22.0, 23.0]),
        ],
    )
    def test_observe_predict_global(
        self, panel_reduction_data, fit_forecasting_horizon, predict_forecasting_horizon, stride, expected_a
    ):
        """Test panel observe_predict with exogenous features (non-recursive predict)."""
        y_train_panel, y_test_panel, X_train_panel, X_test_panel = panel_reduction_data
        forecaster = PointReductionForecaster()

        forecaster.fit(
            y=y_train_panel,
            X_actual=X_train_panel,
            forecasting_horizon=fit_forecasting_horizon,
        )

        y_pred = forecaster.observe_predict(
            y=y_test_panel,
            X_actual=X_test_panel,
            forecasting_horizon=predict_forecasting_horizon,
            stride=stride,
        )

        # Get the last set of predictions (after all observations)
        y_pred = y_pred.tail(predict_forecasting_horizon)

        expected_y_pred = pl.DataFrame(
            {
                "vintage_time": [y_test_panel["time"][-1]] * predict_forecasting_horizon,
                "time": pl.datetime_range(
                    start=datetime(2021, 12, 16, 0, 0, len(y_train_panel) + len(y_test_panel)),
                    end=datetime(
                        2021,
                        12,
                        16,
                        0,
                        0,
                        len(y_train_panel) + len(y_test_panel) + predict_forecasting_horizon - 1,
                    ),
                    interval="1s",
                    eager=True,
                ),
                "x__a": [int(v) for v in expected_a],
                "x__b": [int(v) + 10 for v in expected_a],
                "y__a": [int(v) + 10 for v in expected_a],
                "y__b": [int(v) + 20 for v in expected_a],
            },
            schema={
                "vintage_time": pl.Datetime(time_unit="us", time_zone=None),
                "time": pl.Datetime(time_unit="us", time_zone=None),
                "x__a": pl.Int64,
                "x__b": pl.Int64,
                "y__a": pl.Int64,
                "y__b": pl.Int64,
            },
        )
        pl.testing.assert_frame_equal(y_pred, expected_y_pred)


class TestPointReductionChecks:
    @pytest.mark.parametrize(
        "forecaster,expected_failures",
        [
            (
                PointReductionForecaster(),
                [],
            ),
            (
                PointReductionForecaster(estimator=LinearRegression()),
                [],
            ),
        ],
    )
    def test_point_reduction_checks(self, forecaster, expected_failures, y_X_factory):
        """Run systematic checks on PointReductionForecaster."""
        y, X_actual, X_future, X_forecast = y_X_factory(
            length=100,
            seed=42,
            n_future_features=2,
            n_forecast_features=2,
            return_exogenous=True,
        )
        y_train, y_test = y[:80], y[80:]
        X_actual_train, X_actual_test = X_actual[:80], X_actual[80:]

        forecaster_fitted = clone(forecaster)
        forecaster_fitted.fit(y_train, X_actual_train, forecasting_horizon=3, X_future=X_future, X_forecast=X_forecast)

        run_checks(
            forecaster_fitted,
            _yield_yohou_forecaster_checks(
                forecaster_fitted,
                y_train,
                X_actual_train,
                y_test,
                X_actual_test,
                X_future_train=X_future,
                X_future_test=X_future,
                X_forecast_train=X_forecast,
                X_forecast_test=X_forecast,
            ),
            expected_failures=set(expected_failures),
        )


class TestLinearRegressionAnalytical:
    def test_linear_regression_perfect_linear_trend(self):
        """Test PointReductionForecaster with LinearRegression on perfect linear trend.

        With a perfect linear trend y = mx + b and default lag-1 features, LinearRegression
        should produce exact predictions since the relationship is exactly linear.
        """
        # Create perfect linear trend: y = 2*t + 10
        length = 50
        time = pl.datetime_range(
            start=datetime(2021, 1, 1),
            end=datetime(2021, 1, 1, 0, 0, length - 1),
            interval="1s",
            eager=True,
        )
        y = pl.DataFrame({
            "time": time,
            "value": [2.0 * i + 10.0 for i in range(length)],
        })

        # Train/test split
        y_train, _y_test = y[:40], y[40:]

        # Create forecaster with default feature transformer (LagTransformer(lag=[1])) and LinearRegression
        forecaster = PointReductionForecaster(
            estimator=LinearRegression(),
        )

        # Fit on training data with horizon=1 (one-step-ahead forecasting)
        forecaster.fit(y_train, X_actual=None, forecasting_horizon=1)

        # Predict one step ahead (exact prediction for linear trend with AR(1) structure)
        y_pred = forecaster.predict(forecasting_horizon=1)

        # Expected value: continue the linear trend
        expected_value = 2.0 * 40 + 10.0  # y_41 = 2*40 + 10 = 90

        # Check prediction is very close (numerical precision tolerance)
        predicted_value = y_pred["value"][0]
        np.testing.assert_allclose(predicted_value, expected_value, rtol=1e-5, atol=1e-5)

    def test_linear_regression_ar1_process(self):
        """Test PointReductionForecaster with LinearRegression on AR(1) process.

        For an AR(1) process y_t = phi * y_{t-1} + c, LinearRegression with default lag=1
        should recover the exact parameters and produce exact one-step-ahead predictions.
        """
        # Create AR(1) process: y_t = 0.8 * y_{t-1} + 5
        phi = 0.8
        c = 5.0
        length = 50

        time = pl.datetime_range(
            start=datetime(2021, 1, 1),
            end=datetime(2021, 1, 1, 0, 0, length - 1),
            interval="1s",
            eager=True,
        )

        # Generate AR(1) series
        values = [10.0]  # Initial value
        for _i in range(1, length):
            values.append(phi * values[-1] + c)

        y = pl.DataFrame({
            "time": time,
            "value": values,
        })

        # Train/test split
        y_train, _y_test = y[:40], y[40:]

        # Create forecaster with default feature transformer (LagTransformer(lag=[1])) and LinearRegression
        forecaster = PointReductionForecaster(
            estimator=LinearRegression(),
        )

        # Fit on training data with horizon=1
        forecaster.fit(y_train, X_actual=None, forecasting_horizon=1)

        # Check fitted coefficients are close to true values
        # The default LagTransformer creates lag=1 features
        # LinearRegression learns: y = coef * lag1 + intercept
        # Should have: coef ≈ phi, intercept ≈ c
        fitted_estimator = forecaster.estimator_
        np.testing.assert_allclose(fitted_estimator.coef_[0], phi, rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(fitted_estimator.intercept_, c, rtol=1e-10, atol=1e-10)

        # Predict one step ahead (should be exact given the model structure)
        y_pred = forecaster.predict(forecasting_horizon=1)

        # Expected value: phi * last_observed + c
        expected_value = phi * y_train["value"][-1] + c

        # Check prediction is exact
        predicted_value = y_pred["value"][0]
        np.testing.assert_allclose(predicted_value, expected_value, rtol=1e-10, atol=1e-10)


class TestDirectLinearRegressionAnalytical:
    """Analytical tests for direct strategy with LinearRegression."""

    def test_direct_perfect_linear_trend(self):
        """Direct strategy on perfect linear trend produces exact predictions.

        With y = 2*t + 10 and lag-1 features, each direct model should
        learn the one-step linear relationship exactly.
        """
        length = 50
        time = pl.datetime_range(
            start=datetime(2021, 1, 1),
            end=datetime(2021, 1, 1, 0, 0, length - 1),
            interval="1s",
            eager=True,
        )
        y = pl.DataFrame({"time": time, "value": [2.0 * i + 10.0 for i in range(length)]})
        y_train = y[:40]

        forecaster = PointReductionForecaster(
            estimator=LinearRegression(),
            reduction_strategy="direct",
        )
        forecaster.fit(y_train, X_actual=None, forecasting_horizon=3)

        y_pred = forecaster.predict(forecasting_horizon=3)

        # Perfect linear trend: predictions should be exact continuations
        expected = [2.0 * 40 + 10.0, 2.0 * 41 + 10.0, 2.0 * 42 + 10.0]
        np.testing.assert_allclose(y_pred["value"].to_numpy(), expected, rtol=1e-5, atol=1e-5)

    @pytest.mark.parametrize("strategy", ["direct", "dir-rec"])
    def test_ar1_horizon_1_recovers_exact_prediction(self, strategy):
        """Direct and dir-rec recover the exact AR(1) one-step prediction.

        At horizon=1 both strategies reduce to the multi-output one-step model
        (whose AR(1) correctness is asserted in
        TestLinearRegressionAnalytical.test_linear_regression_ar1_process), so a
        single parametrized check covers both rather than two separate copies.
        """
        phi = 0.8
        c = 5.0
        length = 50
        time = pl.datetime_range(
            start=datetime(2021, 1, 1),
            end=datetime(2021, 1, 1, 0, 0, length - 1),
            interval="1s",
            eager=True,
        )
        values = [10.0]
        for _ in range(1, length):
            values.append(phi * values[-1] + c)
        y = pl.DataFrame({"time": time, "value": values})
        y_train = y[:40]

        forecaster = PointReductionForecaster(
            estimator=LinearRegression(),
            reduction_strategy=strategy,
        )
        forecaster.fit(y_train, X_actual=None, forecasting_horizon=1)

        y_pred = forecaster.predict(forecasting_horizon=1)
        expected_value = phi * y_train["value"][-1] + c
        np.testing.assert_allclose(y_pred["value"][0], expected_value, rtol=1e-10, atol=1e-10)

    def test_direct_constant_series(self):
        """Direct strategy on a constant series predicts the constant."""
        length = 50
        time = pl.datetime_range(
            start=datetime(2021, 1, 1),
            end=datetime(2021, 1, 1, 0, 0, length - 1),
            interval="1s",
            eager=True,
        )
        y = pl.DataFrame({"time": time, "value": [42.0] * length})
        y_train = y[:40]

        forecaster = PointReductionForecaster(
            estimator=LinearRegression(),
            reduction_strategy="direct",
        )
        forecaster.fit(y_train, X_actual=None, forecasting_horizon=3)

        y_pred = forecaster.predict(forecasting_horizon=3)
        np.testing.assert_allclose(y_pred["value"].to_numpy(), [42.0, 42.0, 42.0], rtol=1e-5)


class TestDirRecLinearRegressionAnalytical:
    """Analytical tests for dir-rec strategy with LinearRegression."""

    def test_dir_rec_constant_series(self):
        """Dir-rec strategy on a constant series predicts the constant."""
        length = 50
        time = pl.datetime_range(
            start=datetime(2021, 1, 1),
            end=datetime(2021, 1, 1, 0, 0, length - 1),
            interval="1s",
            eager=True,
        )
        y = pl.DataFrame({"time": time, "value": [42.0] * length})
        y_train = y[:40]

        forecaster = PointReductionForecaster(
            estimator=LinearRegression(),
            reduction_strategy="dir-rec",
        )
        forecaster.fit(y_train, X_actual=None, forecasting_horizon=3)

        y_pred = forecaster.predict(forecasting_horizon=3)
        np.testing.assert_allclose(y_pred["value"].to_numpy(), [42.0, 42.0, 42.0], rtol=1e-5)

    def test_dir_rec_perfect_linear_trend(self):
        """Dir-rec on perfect linear trend produces near-exact predictions."""
        length = 50
        time = pl.datetime_range(
            start=datetime(2021, 1, 1),
            end=datetime(2021, 1, 1, 0, 0, length - 1),
            interval="1s",
            eager=True,
        )
        y = pl.DataFrame({"time": time, "value": [2.0 * i + 10.0 for i in range(length)]})
        y_train = y[:40]

        forecaster = PointReductionForecaster(
            estimator=LinearRegression(),
            reduction_strategy="dir-rec",
        )
        forecaster.fit(y_train, X_actual=None, forecasting_horizon=3)

        y_pred = forecaster.predict(forecasting_horizon=3)
        expected = [2.0 * 40 + 10.0, 2.0 * 41 + 10.0, 2.0 * 42 + 10.0]
        np.testing.assert_allclose(y_pred["value"].to_numpy(), expected, rtol=1e-4, atol=1e-4)


class TestDirectHorizonMismatch:
    """Recursive prediction (predict_fh > fit_fh) for direct/dir-rec strategies.

    The non-recursive predict_fh <= fit_fh shape cases are covered by
    TestDirectStrategy.test_predict_shape and TestDirRecStrategy.test_predict_shape;
    the recursive multi-output case by TestPredict.test_predict_recursive_no_x.
    Only the recursive scenario for the direct/dir-rec strategies is exercised here.
    """

    @pytest.mark.parametrize("strategy", ["direct", "dir-rec"])
    def test_predict_larger_horizon_than_fit(self, strategy):
        """When predict horizon > fit horizon, recursive application extends the forecast."""
        length = 50
        time = pl.datetime_range(
            start=datetime(2021, 1, 1),
            end=datetime(2021, 1, 1, 0, 0, length - 1),
            interval="1s",
            eager=True,
        )
        y = pl.DataFrame({"time": time, "value": [float(i) for i in range(length)]})
        y_train = y[:40]

        forecaster = PointReductionForecaster(
            estimator=LinearRegression(),
            reduction_strategy=strategy,
        )
        forecaster.fit(y_train, X_actual=None, forecasting_horizon=2)

        # Predict 5 steps even though fit horizon was 2.
        y_pred = forecaster.predict(forecasting_horizon=5)

        assert y_pred.shape[0] == 5
        assert "value" in y_pred.columns
        # Recursive extension must produce a contiguous 1s time range continuing
        # immediately after the last training timestamp.
        expected_time = pl.datetime_range(
            start=datetime(2021, 1, 1, 0, 0, 40),
            end=datetime(2021, 1, 1, 0, 0, 44),
            interval="1s",
            eager=True,
        )
        assert y_pred["time"].to_list() == expected_time.to_list()


class TestParameterValidation:
    """Tests for invalid parameter handling."""

    def test_invalid_strategy_rejected(self):
        """Invalid reduction_strategy raises at fit time."""
        forecaster = PointReductionForecaster()
        forecaster.set_params(reduction_strategy="foobar")

        length = 30
        time = pl.datetime_range(
            start=datetime(2021, 1, 1),
            end=datetime(2021, 1, 1, 0, 0, length - 1),
            interval="1s",
            eager=True,
        )
        y = pl.DataFrame({"time": time, "value": [float(i) for i in range(length)]})

        with pytest.raises(ValueError, match="reduction_strategy"):
            forecaster.fit(y, forecasting_horizon=3)


class TestNanHandlingDrop:
    """Point-unit coverage of nan_handling='drop' across reduction strategies.

    The full nan_handling matrix lives in tests/base/test_reduction_nan_handling.py;
    this asserts the user-facing contract (warning on fit, NaN-feature rows
    predicted as NaN) for both the multi-output and direct strategies through
    PointReductionForecaster.
    """

    @staticmethod
    def _series_with_trailing_nan(length: int = 30) -> pl.DataFrame:
        values = [float(i) for i in range(length)]
        values[-1] = float("nan")
        values[-2] = float("nan")
        return pl.DataFrame({
            "time": pl.datetime_range(
                start=datetime(2021, 1, 1),
                end=datetime(2021, 1, 1, 0, 0, length - 1),
                interval="1s",
                eager=True,
            ),
            "value": values,
        })

    @pytest.mark.parametrize("strategy", ["multi-output", "direct"])
    def test_drop_warns_and_predicts_nan_on_nan_features(self, strategy):
        """drop warns at fit and returns NaN predictions when features are NaN."""
        y = self._series_with_trailing_nan()
        forecaster = PointReductionForecaster(
            estimator=LinearRegression(),
            reduction_strategy=strategy,
            nan_handling="drop",
        )

        with pytest.warns(UserWarning):
            forecaster.fit(y=y, forecasting_horizon=2)

        y_pred = forecaster.predict(forecasting_horizon=2)
        assert y_pred["value"].is_nan().all()


class TestDtypePreservation:
    def test_dtype_preservation_single_column(self):
        """Test that predictions preserve Int32 dtype for single column."""
        time = pl.datetime_range(
            start=datetime(2020, 1, 1),
            end=datetime(2020, 1, 31),
            interval="1d",
            eager=True,
        )
        y = pl.DataFrame({
            "time": time,
            "value": pl.Series(range(31), dtype=pl.Int32),
        })

        forecaster = PointReductionForecaster(estimator=LinearRegression())
        forecaster.fit(y[:20], forecasting_horizon=3)

        # Check schema attributes are set correctly
        assert forecaster.local_y_schema_ == {"value": pl.Int32}
        assert forecaster.local_y_t_schema_ == {"value": pl.Int32}

        # Make predictions
        y_pred = forecaster.predict(forecasting_horizon=3)

        # Verify dtype is preserved
        assert y_pred.schema["value"] == pl.Int32
        assert all(isinstance(v, int) for v in y_pred["value"].to_list())

    def test_dtype_preservation_multiple_columns(self):
        """Test that predictions preserve different dtypes for multiple columns."""
        time = pl.datetime_range(
            start=datetime(2020, 1, 1),
            end=datetime(2020, 2, 29),
            interval="1d",
            eager=True,
        )
        y = pl.DataFrame({
            "time": time,
            "sales": pl.Series(range(len(time)), dtype=pl.Int8),
            "revenue": pl.Series([x * 10.5 for x in range(len(time))], dtype=pl.Float32),
        })

        forecaster = PointReductionForecaster(estimator=LinearRegression())
        forecaster.fit(y[:45], forecasting_horizon=5)

        # Check schema attributes
        assert forecaster.local_y_schema_ == {"sales": pl.Int8, "revenue": pl.Float32}
        assert forecaster.local_y_t_schema_ == {"sales": pl.Int8, "revenue": pl.Float32}

        # Make predictions
        y_pred = forecaster.predict(forecasting_horizon=5)

        # Verify both dtypes are preserved
        assert y_pred.schema["sales"] == pl.Int8
        assert y_pred.schema["revenue"] == pl.Float32
        assert all(isinstance(v, int) for v in y_pred["sales"].to_list())

    def test_dtype_preservation_with_transformer(self):
        """Test dtype preservation through target transformer and inverse transform."""
        from yohou.stationarity import SeasonalDifferencing

        time = pl.datetime_range(
            start=datetime(2020, 1, 1),
            end=datetime(2020, 2, 10),
            interval="1d",
            eager=True,
        )
        y = pl.DataFrame({
            "time": time,
            "value": pl.Series(range(41), dtype=pl.Int16),
        })

        forecaster = PointReductionForecaster(
            estimator=LinearRegression(),
            target_transformer=SeasonalDifferencing(seasonality=7),
        )
        forecaster.fit(y[:30], forecasting_horizon=3)

        # Check original and transformed schemas
        assert forecaster.local_y_schema_ == {"value": pl.Int16}
        assert forecaster.local_y_t_schema_ == {"diff_s_7_value": pl.Int16}

        # Make predictions (should go through inverse transform)
        y_pred = forecaster.predict(forecasting_horizon=3)

        # Verify dtype is preserved after inverse transform
        assert y_pred.schema["value"] == pl.Int16
        assert all(isinstance(v, int) for v in y_pred["value"].to_list())

    def test_dtype_preservation_int16_to_int64(self):
        """Test casting between different integer types."""
        time = pl.datetime_range(
            start=datetime(2020, 1, 1),
            end=datetime(2020, 1, 20),
            interval="1d",
            eager=True,
        )
        y = pl.DataFrame({
            "time": time,
            "small": pl.Series(range(20), dtype=pl.Int16),
            "large": pl.Series(range(1000, 1020), dtype=pl.Int64),
        })

        forecaster = PointReductionForecaster(estimator=LinearRegression())
        forecaster.fit(y[:15], forecasting_horizon=3)

        y_pred = forecaster.predict(forecasting_horizon=3)

        # Verify both integer dtypes are preserved
        assert y_pred.schema["small"] == pl.Int16
        assert y_pred.schema["large"] == pl.Int64


class TestDirectStrategy:
    """Tests for direct reduction strategy on PointReductionForecaster."""

    def test_estimator_is_list(self, reduction_data):
        """Direct strategy stores a list of H estimators."""
        y_train, _y_test, X_actual_train, _X_test = reduction_data
        forecaster = PointReductionForecaster(reduction_strategy="direct")
        forecaster.fit(y=y_train, X_actual=X_actual_train, forecasting_horizon=3)

        assert isinstance(forecaster.estimator_, list)
        assert len(forecaster.estimator_) == 3
        for est in forecaster.estimator_:
            assert isinstance(est, BaseEstimator)

    def test_estimators_are_independent_clones(self, reduction_data):
        """Each direct estimator is a distinct object."""
        y_train, _y_test, X_actual_train, _X_test = reduction_data
        forecaster = PointReductionForecaster(reduction_strategy="direct")
        forecaster.fit(y=y_train, X_actual=X_actual_train, forecasting_horizon=3)

        for i in range(len(forecaster.estimator_)):
            for j in range(i + 1, len(forecaster.estimator_)):
                assert forecaster.estimator_[i] is not forecaster.estimator_[j]

    @pytest.mark.parametrize(
        "fit_forecasting_horizon, predict_forecasting_horizon",
        [(3, 2), (5, 5)],
    )
    def test_predict_shape(self, reduction_data, fit_forecasting_horizon, predict_forecasting_horizon):
        """Direct predictions have correct shape (non-recursive: predict_fh <= fit_fh)."""
        y_train, _y_test, X_actual_train, X_actual_test = reduction_data
        forecaster = PointReductionForecaster(reduction_strategy="direct")
        forecaster.fit(y=y_train, X_actual=X_actual_train, forecasting_horizon=fit_forecasting_horizon)

        y_pred = forecaster.predict(
            forecasting_horizon=predict_forecasting_horizon,
        )

        assert y_pred.shape[0] == predict_forecasting_horizon
        assert "a" in y_pred.columns
        assert "b" in y_pred.columns

    def test_predict_matches_multi_output_on_linear(self):
        """On perfectly linear data, direct should match multi-output predictions."""
        length = 50
        time = pl.datetime_range(
            start=datetime(2021, 1, 1),
            end=datetime(2021, 1, 1, 0, 0, length - 1),
            interval="1s",
            eager=True,
        )
        y = pl.DataFrame({"time": time, "value": [2.0 * i + 10.0 for i in range(length)]})
        y_train = y[:40]

        forecaster_direct = PointReductionForecaster(
            estimator=LinearRegression(),
            reduction_strategy="direct",
        )
        forecaster_mo = PointReductionForecaster(
            estimator=LinearRegression(),
            reduction_strategy="multi-output",
        )

        forecaster_direct.fit(y_train, forecasting_horizon=3)
        forecaster_mo.fit(y_train, forecasting_horizon=3)

        y_pred_direct = forecaster_direct.predict(forecasting_horizon=3)
        y_pred_mo = forecaster_mo.predict(forecasting_horizon=3)

        np.testing.assert_allclose(
            y_pred_direct["value"].to_numpy(),
            y_pred_mo["value"].to_numpy(),
            rtol=1e-5,
        )


class TestDirectStrategyPanel:
    """Tests for direct strategy with panel data."""

    def test_predict_panel(self, panel_reduction_data):
        """Direct strategy works with panel data."""
        y_train, _y_test, X_actual_train, X_actual_test = panel_reduction_data
        forecaster = PointReductionForecaster(reduction_strategy="direct")
        forecaster.fit(y=y_train, X_actual=X_actual_train, forecasting_horizon=3)

        y_pred = forecaster.predict(forecasting_horizon=3)

        assert y_pred.shape[0] == 3
        assert "x__a" in y_pred.columns
        assert "y__b" in y_pred.columns


class TestDirRecStrategy:
    """Tests for dir-rec reduction strategy on PointReductionForecaster."""

    def test_estimator_is_list(self, reduction_data):
        """Dir-rec strategy stores a list of H estimators."""
        y_train, _y_test, X_actual_train, _X_test = reduction_data
        forecaster = PointReductionForecaster(reduction_strategy="dir-rec")
        forecaster.fit(y=y_train, X_actual=X_actual_train, forecasting_horizon=3)

        assert isinstance(forecaster.estimator_, list)
        assert len(forecaster.estimator_) == 3

    def test_progressive_feature_augmentation(self, reduction_data):
        """Dir-rec models should have progressively more features."""
        y_train, _y_test, X_actual_train, _X_test = reduction_data
        forecaster = PointReductionForecaster(reduction_strategy="dir-rec")
        forecaster.fit(y=y_train, X_actual=X_actual_train, forecasting_horizon=3)

        n_original = forecaster._dir_rec_n_original_features_
        # Model 0 (step 1): n_original features
        # Model 1 (step 2): n_original + n_targets features (augmented with model 0 preds)
        # Model 2 (step 3): n_original + 2 * n_targets features
        n_targets = len([c for c in y_train.columns if c != "time"])
        for step, est in enumerate(forecaster.estimator_):
            expected_features = n_original + step * n_targets
            assert est.n_features_in_ == expected_features, (
                f"Step {step}: expected {expected_features} features, got {est.n_features_in_}"
            )

    def test_stores_n_original_features(self, reduction_data):
        """Dir-rec fit stores _dir_rec_n_original_features_ attribute."""
        y_train, _y_test, X_actual_train, _X_test = reduction_data
        forecaster = PointReductionForecaster(reduction_strategy="dir-rec")
        forecaster.fit(y=y_train, X_actual=X_actual_train, forecasting_horizon=3)

        assert hasattr(forecaster, "_dir_rec_n_original_features_")
        assert forecaster._dir_rec_n_original_features_ > 0

    @pytest.mark.parametrize(
        "fit_forecasting_horizon, predict_forecasting_horizon",
        [(3, 2), (5, 5)],
    )
    def test_predict_shape(self, reduction_data, fit_forecasting_horizon, predict_forecasting_horizon):
        """Dir-rec predictions have correct shape (non-recursive: predict_fh <= fit_fh)."""
        y_train, _y_test, X_actual_train, X_actual_test = reduction_data
        forecaster = PointReductionForecaster(reduction_strategy="dir-rec")
        forecaster.fit(y=y_train, X_actual=X_actual_train, forecasting_horizon=fit_forecasting_horizon)

        y_pred = forecaster.predict(
            forecasting_horizon=predict_forecasting_horizon,
        )

        assert y_pred.shape[0] == predict_forecasting_horizon
        assert "a" in y_pred.columns
        assert "b" in y_pred.columns

    def test_horizon_1_matches_direct(self, reduction_data):
        """With horizon=1, dir-rec and direct should produce identical results."""
        y_train, _y_test, X_actual_train, X_actual_test = reduction_data
        forecaster_direct = PointReductionForecaster(reduction_strategy="direct")
        forecaster_dirrec = PointReductionForecaster(reduction_strategy="dir-rec")

        forecaster_direct.fit(y=y_train, X_actual=X_actual_train, forecasting_horizon=1)
        forecaster_dirrec.fit(y=y_train, X_actual=X_actual_train, forecasting_horizon=1)

        y_pred_direct = forecaster_direct.predict(forecasting_horizon=1)
        y_pred_dirrec = forecaster_dirrec.predict(forecasting_horizon=1)

        np.testing.assert_allclose(
            y_pred_direct.select(~cs.by_name("time", "vintage_time")).to_numpy(),
            y_pred_dirrec.select(~cs.by_name("time", "vintage_time")).to_numpy(),
            rtol=1e-10,
        )


class TestDirRecStrategyPanel:
    """Tests for dir-rec strategy with panel data."""

    def test_predict_panel(self, panel_reduction_data):
        """Dir-rec strategy works with panel data."""
        y_train, _y_test, X_actual_train, X_actual_test = panel_reduction_data
        forecaster = PointReductionForecaster(reduction_strategy="dir-rec")
        forecaster.fit(y=y_train, X_actual=X_actual_train, forecasting_horizon=3)

        y_pred = forecaster.predict(forecasting_horizon=3)

        assert y_pred.shape[0] == 3
        assert "x__a" in y_pred.columns
        assert "y__b" in y_pred.columns


class TestObservePredictDirectDirRec:
    """Tests for observe_predict with direct and dir-rec strategies."""

    @pytest.mark.parametrize("strategy", ["direct", "dir-rec"])
    def test_observe_predict(self, reduction_data, strategy):
        """observe_predict works for direct and dir-rec strategies."""
        y_train, y_test, X_actual_train, X_actual_test = reduction_data
        forecaster = PointReductionForecaster(reduction_strategy=strategy)
        forecaster.fit(y=y_train, X_actual=X_actual_train, forecasting_horizon=3)

        predict_forecasting_horizon = 3

        y_pred = forecaster.observe_predict(
            y=y_test,
            X_actual=X_actual_test,
            forecasting_horizon=predict_forecasting_horizon,
            stride=1,
        )

        y_pred_tail = y_pred.tail(predict_forecasting_horizon)
        assert y_pred_tail.shape[0] == predict_forecasting_horizon
        assert "a" in y_pred_tail.columns
        assert "b" in y_pred_tail.columns


class TestDirectDirRecChecks:
    """Run systematic checks for direct and dir-rec strategies."""

    @pytest.mark.parametrize(
        "forecaster,expected_failures",
        [
            (PointReductionForecaster(reduction_strategy="direct"), []),
            (PointReductionForecaster(reduction_strategy="dir-rec"), []),
        ],
    )
    def test_checks(self, forecaster, expected_failures, y_X_factory):
        """Run systematic checks on direct/dir-rec PointReductionForecaster."""
        y, X_actual, X_future, X_forecast = y_X_factory(
            length=100,
            seed=42,
            n_future_features=2,
            n_forecast_features=2,
            return_exogenous=True,
        )
        y_train, y_test = y[:80], y[80:]
        X_actual_train, X_actual_test = X_actual[:80], X_actual[80:]

        forecaster_fitted = clone(forecaster)
        forecaster_fitted.fit(y_train, X_actual_train, forecasting_horizon=3, X_future=X_future, X_forecast=X_forecast)

        run_checks(
            forecaster_fitted,
            _yield_yohou_forecaster_checks(
                forecaster_fitted,
                y_train,
                X_actual_train,
                y_test,
                X_actual_test,
                X_future_train=X_future,
                X_future_test=X_future,
                X_forecast_train=X_forecast,
                X_forecast_test=X_forecast,
            ),
            expected_failures=set(expected_failures),
        )


class TestPanelStepColumnChecks:
    """Run systematic checks for panel data with X_future / X_forecast."""

    @pytest.mark.parametrize(
        "forecaster,expected_failures",
        [
            (PointReductionForecaster(), []),
            (PointReductionForecaster(reduction_strategy="direct"), []),
        ],
    )
    def test_panel_checks_with_step_data(self, forecaster, expected_failures, y_X_factory):
        """Run systematic checks on panel PointReductionForecaster with step columns."""
        y, X_actual, X_future, X_forecast = y_X_factory(
            length=100,
            seed=42,
            panel=True,
            n_groups=2,
            n_future_features=2,
            n_forecast_features=2,
            return_exogenous=True,
        )
        y_train, y_test = y[:80], y[80:]
        X_actual_train, X_actual_test = X_actual[:80], X_actual[80:]

        forecaster_fitted = clone(forecaster)
        forecaster_fitted.fit(y_train, X_actual_train, forecasting_horizon=3, X_future=X_future, X_forecast=X_forecast)

        run_checks(
            forecaster_fitted,
            _yield_yohou_forecaster_checks(
                forecaster_fitted,
                y_train,
                X_actual_train,
                y_test,
                X_actual_test,
                X_future_train=X_future,
                X_future_test=X_future,
                X_forecast_train=X_forecast,
                X_forecast_test=X_forecast,
            ),
            expected_failures=set(expected_failures),
        )


class TestPanelGroupMismatchErrors:
    """Error paths for mismatched panel group names."""

    def test_fit_raises_on_mismatched_panel_groups(self):
        """Cover forecaster.py lines 251, 322: X_actual groups != y groups."""
        time = pl.DataFrame({
            "time": pl.datetime_range(
                start=datetime(2021, 12, 16),
                end=datetime(2021, 12, 16, 0, 0, 19),
                interval="1s",
                eager=True,
            )
        })
        y = pl.concat(
            [
                time,
                pl.DataFrame({
                    "group_a__y_0": np.random.default_rng(42).random(20),
                    "group_b__y_0": np.random.default_rng(42).random(20),
                }),
            ],
            how="horizontal",
        )
        X_actual = pl.concat(
            [
                time,
                pl.DataFrame({
                    "group_c__X_0": np.random.default_rng(42).random(20),
                    "group_d__X_0": np.random.default_rng(42).random(20),
                }),
            ],
            how="horizontal",
        )

        forecaster = PointReductionForecaster()
        # The canonical mismatch message lists both group sets.
        with pytest.raises(ValueError, match="Panel groups mismatch") as exc_info:
            forecaster.fit(y, X_actual=X_actual, forecasting_horizon=3)
        message = str(exc_info.value)
        assert "group_a" in message and "group_b" in message
        assert "group_c" in message and "group_d" in message


class TestEmptyTrainingData:
    """Tests for error handling when training data is too short."""

    def test_fit_raises_on_empty_tabularized_data(self):
        """Fitting with data too short for the horizon raises ValueError."""
        time = pl.datetime_range(
            start=datetime(2021, 1, 1),
            end=datetime(2021, 1, 1, 0, 0, 2),
            interval="1s",
            eager=True,
        )
        y = pl.DataFrame({"time": time, "value": [1.0, 2.0, 3.0]})
        forecaster = PointReductionForecaster()

        with pytest.raises(ValueError, match="Training dataset is empty"):
            forecaster.fit(y, forecasting_horizon=10)


class TestParameterDefaults:
    """Pin the public defaults the generic get/set-params round-trip cannot assert.

    The round-trip and clone-preservation contracts for every declared parameter
    are already covered by check_get_set_params_round_trip /
    check_clone_preserves_forecaster_params in the systematic suite; only the
    specific default values are pinned here.
    """

    @pytest.mark.parametrize(
        "param,expected_default",
        [("n_jobs", None), ("target_as_feature", "transformed")],
    )
    def test_param_default(self, param, expected_default):
        """The named parameter is exposed with its documented default."""
        params = PointReductionForecaster().get_params()
        assert param in params
        assert params[param] == expected_default


class TestNJobsParameter:
    """Tests for n_jobs parallel execution on direct strategy."""

    def test_n_jobs_direct_matches_sequential(self, reduction_data):
        """Direct strategy with n_jobs=2 gives same results as n_jobs=1."""
        y_train, _y_test, X_actual_train, X_actual_test = reduction_data

        forecaster_seq = PointReductionForecaster(
            reduction_strategy="direct",
            n_jobs=1,
        )
        forecaster_par = PointReductionForecaster(
            reduction_strategy="direct",
            n_jobs=2,
        )

        forecaster_seq.fit(y=y_train, X_actual=X_actual_train, forecasting_horizon=3)
        forecaster_par.fit(y=y_train, X_actual=X_actual_train, forecasting_horizon=3)

        y_seq = forecaster_seq.predict(forecasting_horizon=3)
        y_par = forecaster_par.predict(forecasting_horizon=3)

        np.testing.assert_allclose(
            y_seq.select(cs.numeric()).to_numpy(),
            y_par.select(cs.numeric()).to_numpy(),
        )

    @pytest.mark.parametrize("panel", [False, True])
    def test_predict_frames_are_identical_across_n_jobs(self, y_X_factory, panel):
        """n_jobs selects an execution strategy, never a result.

        Stronger than allclose: the frames must match exactly, including column
        names, dtypes, row order and vintage_time.
        """
        y, X_actual = y_X_factory(length=60, n_targets=1, n_features=2, panel=panel, n_groups=3)

        frames = []
        for n_jobs in (1, 2):
            forecaster = PointReductionForecaster(reduction_strategy="direct", n_jobs=n_jobs)
            forecaster.fit(y=y, X_actual=X_actual, forecasting_horizon=4)
            frames.append(forecaster.predict())

        seq, par = frames
        assert seq.columns == par.columns
        assert seq.dtypes == par.dtypes
        assert seq.equals(par)

    @pytest.mark.parametrize("nan_handling", ["pass", "drop"])
    def test_nan_handling_is_preserved_across_n_jobs(self, y_X_factory, nan_handling):
        """Both NaN modes behave the same however the work is dispatched.

        The estimator is NaN-tolerant because ``"pass"`` means exactly that: the nulls
        reach it untouched, which a ``LinearRegression`` would reject.
        """
        from sklearn.ensemble import HistGradientBoostingRegressor

        y, X_actual = y_X_factory(length=60, n_targets=1, n_features=2)
        # Null the last observation so the predict-time feature row carries a null.
        feature = [c for c in X_actual.columns if c != "time"][0]
        X_actual = X_actual.with_columns(
            pl
            .when(pl.arange(0, X_actual.height) == X_actual.height - 1)
            .then(None)
            .otherwise(pl.col(feature))
            .alias(feature)
        )

        frames = []
        for n_jobs in (1, 2):
            forecaster = PointReductionForecaster(
                estimator=HistGradientBoostingRegressor(max_iter=5, min_samples_leaf=2, random_state=0),
                reduction_strategy="direct",
                n_jobs=n_jobs,
                nan_handling=nan_handling,
            )
            forecaster.fit(y=y, X_actual=X_actual, forecasting_horizon=4)
            frames.append(forecaster.predict())

        seq, par = frames
        assert seq.equals(par)

        values = seq.select(cs.numeric()).to_numpy()
        assert values.size > 0
        if nan_handling == "drop":
            # The null reaches the predict feature row, so every step returns NaN.
            assert np.isnan(values).all()
        else:
            # A NaN-tolerant estimator sees the null and still predicts.
            assert not np.isnan(values).any()

    def test_dispatched_task_does_not_carry_the_forecaster(self, y_X_factory):
        """Regression guard for the closure capture.

        `_predict_direct_step` used to be nested inside `_estimator_predict_direct` and
        to read `self`, so cloudpickle captured the whole fitted forecaster into every
        dispatched task: for a 48 step horizon, all 48 models shipped to use one.

        This pickles what joblib actually ships, through the same cloudpickle loky uses.
        Pickling the module-level function directly would prove nothing, because it
        serializes by name under any implementation.
        """
        from joblib.externals import cloudpickle
        from sklearn.utils.parallel import delayed

        from yohou.base.reduction import _predict_direct_step

        # A re-nested function carries `<locals>` in its qualname and cells in its
        # closure, and cloudpickle serializes whatever those cells reach.
        assert "<locals>" not in _predict_direct_step.__qualname__
        assert _predict_direct_step.__closure__ is None

        y, X_actual = y_X_factory(length=120, n_targets=1, n_features=2)
        forecaster = PointReductionForecaster(reduction_strategy="direct")
        forecaster.fit(y=y, X_actual=X_actual, forecasting_horizon=24)

        X_tab = forecaster._get_predict_features()
        task = delayed(_predict_direct_step)(forecaster.estimator_[0], X_tab, 1, np.ones(X_tab.height, dtype=bool))
        payload = len(cloudpickle.dumps(task))
        one_model = len(cloudpickle.dumps(forecaster.estimator_[0]))
        whole_forecaster = len(cloudpickle.dumps(forecaster))

        # The task tracks the one model it needs, not the forecaster holding all 24.
        assert payload < one_model + 8192
        assert payload < whole_forecaster / 2


def _assert_matches_per_group(batched, reference):
    """Structure exactly, values to within floating-point reassociation.

    Batching predicts every panel group in one call, so a linear model's inference is a
    single ``(n_groups, n_features)`` matmul instead of ``n_groups`` separate
    ``(1, n_features)`` ones. BLAS reassociates and the last bit can differ, measured at
    1 ULP. Tree models come out bit-identical, but the assertion has to hold for both, so
    this is the one place the batched path is not exactly the per-group path.

    The small ``atol`` is not slack for the reassociation, which is relative. It covers a
    reference value of exactly zero, where a relative tolerance admits nothing at all and
    any nonzero batched value would fail.
    """
    assert batched.columns == reference.columns
    assert batched.dtypes == reference.dtypes
    np.testing.assert_allclose(batched.to_numpy(), reference.to_numpy(), rtol=1e-12, atol=1e-15)


def _reference_predict_per_group(forecaster):
    """Predict the pre-batching way: one estimator call per (group, step).

    The batched path stacks every panel group's feature row into one call per step.
    That is only sound if all groups share the step's estimator and see the same step
    columns, so this reconstructs the per-group algorithm as the thing to match.
    """
    from yohou.utils import cast as _cast

    y_cols = list(forecaster.local_y_t_schema_.keys())
    n_targets = len(y_cols)
    out = {}
    for group in forecaster.groups_:
        X_tab = forecaster._get_predict_features(group)
        rows = []
        for step, est in enumerate(forecaster.estimator_):
            X_step = forecaster._filter_step_features(X_tab, step + 1)
            if forecaster.nan_handling == "drop" and forecaster._features_have_nan(X_step):
                rows.append(np.full(n_targets, np.nan))
            else:
                rows.append(np.atleast_1d(np.asarray(est.predict(X_step)).ravel())[:n_targets])
        local = _cast(pl.DataFrame(np.vstack(rows), schema=y_cols), forecaster.local_y_t_schema_)
        out[group] = local.rename({c: f"{group}__{c}" for c in y_cols})
    return pl.concat(list(out.values()), how="horizontal")


class TestPanelBatchedPredict:
    """Stacking panel groups into one call per step must not change any number."""

    @pytest.mark.parametrize("alignment", ["all", "matched", "cumulative"])
    def test_batched_matches_per_group_with_forecast_step_columns(self, y_X_factory, alignment):
        """Equivalence under every step alignment, with X_forecast step columns.

        Vintage alignment is where a batching bug would hide: each step carries its own
        `*_step_h` columns, and stacking the groups must keep every group on its own row
        of the right step's frame.
        """
        y, X_actual, _X_future, X_forecast = y_X_factory(
            length=90,
            n_targets=1,
            n_features=2,
            panel=True,
            n_groups=3,
            n_forecast_features=2,
            forecasting_horizon=4,
            return_exogenous=True,
        )

        forecaster = PointReductionForecaster(reduction_strategy="direct", step_feature_alignment=alignment)
        forecaster.fit(y=y, X_actual=X_actual, forecasting_horizon=4, X_forecast=X_forecast)

        # Without step columns the alignment parameter is a no-op and this test would
        # pass under any implementation, so pin that they exist.
        assert forecaster._step_column_names_

        batched = forecaster.predict().drop("vintage_time", strict=False)
        reference = _reference_predict_per_group(forecaster)
        _assert_matches_per_group(batched.drop("time", strict=False), reference)

    def test_batched_matches_per_group_with_lags(self, y_X_factory):
        """Equivalence when features come from a lag transformer.

        Lags are rebuilt incrementally as observations arrive, so each group's feature
        row is its own history. Stacking must not let one group read another's.
        """
        from yohou.preprocessing import LagTransformer as _LagTransformer

        y, X_actual = y_X_factory(length=90, n_targets=1, n_features=2, panel=True, n_groups=3)

        forecaster = PointReductionForecaster(
            reduction_strategy="direct",
            actual_transformer=_LagTransformer(lag=[1, 2, 3]),
        )
        forecaster.fit(y=y, X_actual=X_actual, forecasting_horizon=4)

        batched = forecaster.predict().drop("vintage_time", strict=False)
        reference = _reference_predict_per_group(forecaster)
        _assert_matches_per_group(batched.drop("time", strict=False), reference)

    def test_batched_isolates_a_single_group_with_null_features(self, y_X_factory):
        """Under nan_handling="drop", one bad group must not poison the others.

        The per-group path checked nulls a row at a time. Batched, the mask is per row of
        the stacked frame, so this pins that a null in one group NaNs only that group.
        """
        y, X_actual = y_X_factory(length=90, n_targets=1, n_features=2, panel=True, n_groups=3)
        victim = [c for c in X_actual.columns if c.startswith("group_0__")][0]
        X_actual = X_actual.with_columns(
            pl
            .when(pl.arange(0, X_actual.height) == X_actual.height - 1)
            .then(None)
            .otherwise(pl.col(victim))
            .alias(victim)
        )

        forecaster = PointReductionForecaster(
            estimator=HistGradientBoostingRegressor(max_iter=5, min_samples_leaf=2, random_state=0),
            reduction_strategy="direct",
            nan_handling="drop",
        )
        forecaster.fit(y=y, X_actual=X_actual, forecasting_horizon=4)

        batched = forecaster.predict()
        _assert_matches_per_group(
            batched.drop("time", "vintage_time", strict=False),
            _reference_predict_per_group(forecaster),
        )

        poisoned = [c for c in batched.columns if c.startswith("group_0__")]
        healthy = [c for c in batched.columns if c not in poisoned and c not in ("time", "vintage_time")]
        assert np.isnan(batched.select(poisoned).to_numpy()).all()
        assert not np.isnan(batched.select(healthy).to_numpy()).any()


class TestPanelTimeWeight:
    """Tests for time_weight on panel data in reduction forecasters."""

    def test_panel_callable_time_weight(self, y_X_factory):
        """Callable time_weight is applied per panel group during fit."""
        y, X_actual = y_X_factory(length=60, n_targets=1, n_features=0, panel=True, n_groups=2)

        def constant_weight(t):
            return pl.Series("weight", [1.0] * len(t))

        f = PointReductionForecaster()
        f.set_params(time_weighter=LookupWeighter(mapping={}, default=1.0))
        f.fit(y[:50], forecasting_horizon=3)
        y_pred = f.predict()
        # The time-column contract is owned by check_predict_time_columns;
        # here we only confirm the weighted panel fit/predict round-trip succeeds.
        assert len(y_pred) == 3

    def test_panel_dataframe_time_weight(self, y_X_factory):
        """DataFrame time_weight with global weight column works on panel data."""
        y, X_actual = y_X_factory(length=60, n_targets=1, n_features=0, panel=True, n_groups=2)
        y_train = y[:50]
        weight_df = pl.DataFrame({
            "time": y_train["time"],
            "weight": [1.0] * len(y_train),
        })
        f = PointReductionForecaster()
        f.set_params(time_weighter=TableWeighter(frame=weight_df, on="time"))
        f.fit(y_train, forecasting_horizon=3)
        y_pred = f.predict()
        assert len(y_pred) == 3


@pytest.fixture(scope="module")
def step_alignment_data():
    """Data with X_future so step columns are created."""
    n = 30
    time = pl.datetime_range(
        start=datetime(2021, 1, 1),
        end=datetime(2021, 1, 1, 0, 0, n - 1),
        interval="1s",
        eager=True,
    )
    y = pl.DataFrame({"time": time, "value": np.random.default_rng(42).standard_normal(n)})

    # X_future covers training + forecast horizon
    future_time = pl.datetime_range(
        start=datetime(2021, 1, 1),
        end=datetime(2021, 1, 1, 0, 0, n + 4),
        interval="1s",
        eager=True,
    )
    rng = np.random.default_rng(123)
    X_future = pl.DataFrame({
        "time": future_time,
        "feat_a": rng.standard_normal(len(future_time)),
        "feat_b": rng.standard_normal(len(future_time)),
    })
    return y, X_future


class TestStepFeatureAlignment:
    """Tests for step_feature_alignment parameter on direct strategy."""

    def test_all_keeps_all_step_columns(self, step_alignment_data):
        """Default 'all' mode keeps every step column for every estimator."""
        y, X_future = step_alignment_data
        fh = 3
        f = PointReductionForecaster(
            reduction_strategy="direct",
            step_feature_alignment="all",
        )
        f.fit(y[:25], forecasting_horizon=fh, X_future=X_future)
        # Each estimator should have the same number of features
        n_features_list = [est.n_features_in_ for est in f.estimator_]
        assert len(set(n_features_list)) == 1, "all mode: estimators should have equal feature counts"

    def test_matched_filters_to_own_step(self, step_alignment_data):
        """'matched' mode gives each estimator only its own step columns."""
        y, X_future = step_alignment_data
        fh = 3
        f = PointReductionForecaster(
            reduction_strategy="direct",
            step_feature_alignment="matched",
        )
        f.fit(y[:25], forecasting_horizon=fh, X_future=X_future)
        # Each estimator should see the same (reduced) number of features
        n_features_list = [est.n_features_in_ for est in f.estimator_]
        assert len(set(n_features_list)) == 1, "matched mode: all estimators should have equal feature counts"
        # Fewer features than 'all' mode (2 step cols per step vs 6 total)
        f_all = PointReductionForecaster(
            reduction_strategy="direct",
            step_feature_alignment="all",
        )
        f_all.fit(y[:25], forecasting_horizon=fh, X_future=X_future)
        assert n_features_list[0] < f_all.estimator_[0].n_features_in_

    def test_cumulative_progressive_features(self, step_alignment_data):
        """'cumulative' mode gives step h columns 1..h."""
        y, X_future = step_alignment_data
        fh = 3
        f = PointReductionForecaster(
            reduction_strategy="direct",
            step_feature_alignment="cumulative",
        )
        f.fit(y[:25], forecasting_horizon=fh, X_future=X_future)
        # Estimators should have progressively more features
        n_features_list = [est.n_features_in_ for est in f.estimator_]
        for i in range(len(n_features_list) - 1):
            assert n_features_list[i] < n_features_list[i + 1], (
                f"cumulative: step {i + 1} should have fewer features than step {i + 2}"
            )

    def test_predict_consistent_with_fit(self, step_alignment_data):
        """Predictions succeed when step_feature_alignment filters at predict time too."""
        y, X_future = step_alignment_data
        fh = 3
        for mode in ("all", "matched", "cumulative"):
            f = PointReductionForecaster(
                reduction_strategy="direct",
                step_feature_alignment=mode,
            )
            f.fit(y[:25], forecasting_horizon=fh, X_future=X_future)
            y_pred = f.predict(forecasting_horizon=fh)
            assert len(y_pred) == fh, f"mode={mode}: expected {fh} rows"
            assert "value" in y_pred.columns

    def test_no_step_columns_passthrough(self, y_X_factory):
        """When no X_future is provided, all modes behave identically."""
        y, X_actual = y_X_factory(length=30, n_targets=1, n_features=2)
        fh = 3
        target_col = [c for c in y.columns if c != "time"][0]
        preds = {}
        for mode in ("all", "matched", "cumulative"):
            f = PointReductionForecaster(
                reduction_strategy="direct",
                step_feature_alignment=mode,
            )
            f.fit(y[:25], X_actual=X_actual[:25], forecasting_horizon=fh)
            preds[mode] = f.predict(forecasting_horizon=fh)
        # All predictions should be identical when no step columns exist
        for mode in ("matched", "cumulative"):
            assert preds["all"][target_col].to_list() == preds[mode][target_col].to_list()

    def test_multi_output_ignores_alignment(self, step_alignment_data):
        """multi-output feeds its estimator every step column regardless.

        Asserts the feature count, not the prediction length: a length check
        holds whether or not the parameter is honored, so it cannot detect the
        behaviour it names.
        """
        y, X_future = step_alignment_data
        fh = 3

        def n_features(alignment):
            f = PointReductionForecaster(
                reduction_strategy="multi-output",
                step_feature_alignment=alignment,
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                f.fit(y[:25], forecasting_horizon=fh, X_future=X_future)
            assert len(f.predict(forecasting_horizon=fh)) == fh
            return f.estimator_.n_features_in_

        assert n_features("matched") == n_features("all")

    def test_dir_rec_ignores_alignment(self, step_alignment_data):
        """dir-rec feeds every estimator every step column regardless."""
        y, X_future = step_alignment_data
        fh = 3

        def n_features(alignment):
            f = PointReductionForecaster(
                reduction_strategy="dir-rec",
                step_feature_alignment=alignment,
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                f.fit(y[:25], forecasting_horizon=fh, X_future=X_future)
            assert len(f.predict(forecasting_horizon=fh)) == fh
            return [est.n_features_in_ for est in f.estimator_]

        assert n_features("matched") == n_features("all")

    def test_inapplicable_alignment_warns(self, step_alignment_data):
        """A non-default alignment on a strategy that ignores it says so."""
        y, X_future = step_alignment_data
        for strategy in ("multi-output", "dir-rec"):
            f = PointReductionForecaster(
                reduction_strategy=strategy,
                step_feature_alignment="matched",
            )
            with pytest.warns(UserWarning, match="has no effect"):
                f.fit(y[:25], forecasting_horizon=3, X_future=X_future)

    def test_applicable_or_default_alignment_is_silent(self, step_alignment_data):
        """direct applies the parameter, and "all" is what everyone already does."""
        y, X_future = step_alignment_data
        cases = [("direct", "matched"), ("multi-output", "all"), ("dir-rec", "all")]
        for strategy, alignment in cases:
            f = PointReductionForecaster(
                reduction_strategy=strategy,
                step_feature_alignment=alignment,
            )
            with warnings.catch_warnings(record=True) as record:
                warnings.simplefilter("always")
                f.fit(y[:25], forecasting_horizon=3, X_future=X_future)
            assert [w for w in record if "has no effect" in str(w.message)] == [], (
                f"{strategy} + {alignment} should not warn"
            )


@pytest.fixture(scope="module")
def panel_step_alignment_data():
    """Panel target plus a global X_forecast, both reaching the fit horizon.

    Global (unprefixed) forecast channels on purpose: routed through a per-group
    ``forecast_transformer`` they come back prefixed, which is the naming split
    these tests exist to pin down.
    """
    n = 60
    horizon = 3
    time = pl.datetime_range(
        start=datetime(2021, 1, 1),
        end=datetime(2021, 1, 1, 0, 0, n - 1),
        interval="1s",
        eager=True,
    )
    rng = np.random.default_rng(7)
    y = pl.DataFrame({
        "time": time,
        "x__value": rng.standard_normal(n),
        "z__value": rng.standard_normal(n),
    })
    X_forecast = pl.concat([
        pl.DataFrame({
            "vintage_time": [t] * horizon,
            "time": [t + timedelta(seconds=h) for h in range(1, horizon + 1)],
            "load": rng.standard_normal(horizon),
            "wind": rng.standard_normal(horizon),
        })
        for t in time
    ])
    return y, X_forecast, horizon


def _passthrough_forecast_transformer():
    """A forecast_transformer that selects columns and derives nothing.

    Deliberately inert: it makes the fit take the per-group transform path (which
    re-prefixes every output column) without changing a single value, so a feature
    count that moves can only come from the alignment filter.
    """
    from yohou.compose import ColumnTransformer, PerVintageActualTransformer

    return PerVintageActualTransformer(
        transformer=ColumnTransformer(
            transformers=[("keep", "passthrough", ["load", "wind"])],
            remainder="drop",
            verbose_feature_names_out=False,
        )
    )


class TestStepFeatureAlignmentPanel:
    """Alignment must filter under panel naming, not just standard naming.

    Under ``panel_strategy="global"`` the fitted step column names are panel-wide
    (``{group}__{col}_step_{h}``) while the stacked matrix the estimators see uses
    the local spelling (``{col}_step_{h}``). Matching only the panel-wide names
    recognizes nothing, so the filter passes the matrix through and every per-step
    model silently trains on every step's columns.
    """

    def _fit(self, y, X_forecast, horizon, alignment, transformer=None, panel_strategy="global"):
        f = PointReductionForecaster(
            reduction_strategy="direct",
            step_feature_alignment=alignment,
            panel_strategy=panel_strategy,
            forecast_transformer=transformer,
            actual_transformer=LagTransformer(lag=[1]),
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            f.fit(y=y, forecasting_horizon=horizon, X_forecast=X_forecast)
        return f

    def test_matched_filters_through_a_forecast_transformer(self, panel_step_alignment_data):
        """The deployed shape: panel + X_forecast + per-group transformer."""
        y, X_forecast, horizon = panel_step_alignment_data
        matched = self._fit(y, X_forecast, horizon, "matched", _passthrough_forecast_transformer())
        every = self._fit(y, X_forecast, horizon, "all", _passthrough_forecast_transformer())

        matched_counts = [est.n_features_in_ for est in matched.estimator_]
        all_counts = [est.n_features_in_ for est in every.estimator_]

        assert len(set(matched_counts)) == 1, f"matched should be even across steps, got {matched_counts}"
        assert all(m < a for m, a in zip(matched_counts, all_counts, strict=True)), (
            f"matched={matched_counts} did not narrow against all={all_counts}; the filter recognized "
            f"no step column, which is the silent no-op"
        )
        # The transformer prefixes its outputs, so the two spellings must differ here.
        # Without that the test would pass on data that never exercises the mismatch.
        assert matched._step_column_names_ != matched._step_column_local_names_, (
            "fixture no longer produces prefixed step columns, so this test proves nothing"
        )

    def test_matched_keeps_only_its_own_step(self, panel_step_alignment_data):
        """Each estimator's features name its own step and no other."""
        y, X_forecast, horizon = panel_step_alignment_data
        f = self._fit(y, X_forecast, horizon, "matched", _passthrough_forecast_transformer())

        for step, est in enumerate(f.estimator_, start=1):
            names = list(getattr(est, "feature_names_in_", []))
            step_names = [n for n in names if "_step_" in n]
            assert step_names, f"step {step} estimator saw no step column at all"
            foreign = [n for n in step_names if not n.endswith(f"_step_{step}")]
            assert not foreign, f"step {step} estimator also saw {foreign}"

    def test_matched_filters_without_a_transformer(self, panel_step_alignment_data):
        """The mismatch is about prefixes, not about the transformer.

        Panel-shaped columns reach the same naming split with no transformer in
        play, so the fix must not be conditional on one being configured.
        """
        y, X_forecast, horizon = panel_step_alignment_data
        # Every group carries every channel: a panel column present for one group
        # only is a different (and invalid) shape, not the one under test.
        prefixed = X_forecast.select(
            "vintage_time",
            "time",
            pl.col("load").alias("x__load"),
            pl.col("wind").alias("x__wind"),
            pl.col("load").alias("z__load"),
            pl.col("wind").alias("z__wind"),
        )
        matched = self._fit(y, prefixed, horizon, "matched")
        every = self._fit(y, prefixed, horizon, "all")

        matched_counts = [est.n_features_in_ for est in matched.estimator_]
        all_counts = [est.n_features_in_ for est in every.estimator_]
        assert all(m < a for m, a in zip(matched_counts, all_counts, strict=True)), (
            f"matched={matched_counts} did not narrow against all={all_counts} on prefixed raw columns"
        )

    def test_cumulative_grows_with_the_step(self, panel_step_alignment_data):
        """Step h sees steps 1..h, so the counts must strictly increase."""
        y, X_forecast, horizon = panel_step_alignment_data
        f = self._fit(y, X_forecast, horizon, "cumulative", _passthrough_forecast_transformer())
        counts = [est.n_features_in_ for est in f.estimator_]
        assert all(a < b for a, b in zip(counts, counts[1:], strict=False)), (
            f"cumulative counts should increase with the step, got {counts}"
        )

    def test_predict_filters_as_fit_did(self, panel_step_alignment_data):
        """Predict must apply the same filter, and the narrowing must be visible."""
        y, X_forecast, horizon = panel_step_alignment_data
        preds = {}
        for alignment in ("all", "matched", "cumulative"):
            f = self._fit(y, X_forecast, horizon, alignment, _passthrough_forecast_transformer())
            y_pred = f.predict(forecasting_horizon=horizon)
            assert len(y_pred) == horizon, f"{alignment}: expected {horizon} rows"
            preds[alignment] = y_pred["x__value"].to_list()

        # Different feature sets, so different fitted models: identical predictions
        # would mean the filter never took effect.
        assert preds["matched"] != preds["all"], "matched predicted exactly as all did"

    def test_multivariate_is_unaffected(self, panel_step_alignment_data):
        """The multivariate strategy never had the split, and must keep working.

        It skips panel detection, so its two spellings coincide. This guards the
        fix against regressing the mode it was not written for.
        """
        y, X_forecast, horizon = panel_step_alignment_data
        matched = self._fit(y, X_forecast, horizon, "matched", panel_strategy="multivariate")
        every = self._fit(y, X_forecast, horizon, "all", panel_strategy="multivariate")

        assert matched._step_column_names_ == matched._step_column_local_names_, (
            "multivariate should not produce a naming split"
        )
        assert matched.estimator_[0].n_features_in_ < every.estimator_[0].n_features_in_
        assert len(matched.predict(forecasting_horizon=horizon)) == horizon

    def test_unrecognizable_step_columns_raise(self, panel_step_alignment_data):
        """A filter that cannot recognize any step column fails loudly.

        Unreachable through the public API once the naming is right, so the state
        is forced here. The point is that a future drift surfaces as an error
        rather than as models quietly trained on the wrong feature set.
        """
        y, X_forecast, horizon = panel_step_alignment_data
        f = self._fit(y, X_forecast, horizon, "matched", _passthrough_forecast_transformer())

        f._step_column_local_names_ = {"nothing_matches_step_1"}
        f._step_column_names_ = {"nothing_matches_step_1"}
        with pytest.raises(RuntimeError, match="step_feature_alignment='matched' cannot be applied"):
            f.predict(forecasting_horizon=horizon)


class TestPipelineSampleWeightRouting:
    """Tests for sample_weight routing through Pipeline and plain estimators."""

    @pytest.fixture()
    def simple_series(self):
        """Simple time series for weight routing tests."""
        n = 30
        time = pl.datetime_range(
            start=datetime(2021, 1, 1),
            end=datetime(2021, 1, 1, 0, 0, n - 1),
            interval="1s",
            eager=True,
        )
        y = pl.DataFrame({
            "time": time,
            "value": np.random.default_rng(42).standard_normal(n),
        })
        return y

    @staticmethod
    def _constant_weight(t):
        return pl.Series("weight", [1.0] * len(t))

    def test_pipeline_with_time_weight(self, simple_series):
        """Pipeline with Ridge final step accepts time_weight."""
        pipe = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge())])
        f = PointReductionForecaster(estimator=pipe)
        f.set_params(time_weighter=LookupWeighter(mapping={}, default=1.0))
        f.fit(simple_series, forecasting_horizon=3)
        y_pred = f.predict()
        # check_predict_time_columns owns the time-column contract; this test
        # only confirms the Ridge-final pipeline accepts the routed time weight.
        assert len(y_pred) == 3

    def test_pipeline_unsupported_final_step_raises(self, simple_series):
        """Pipeline whose final step lacks sample_weight raises ValueError."""

        class NoWeightEstimator(BaseEstimator):
            def fit(self, X, y):
                return self

            def predict(self, X):
                return np.zeros(X.shape[0])

        pipe = Pipeline([("scaler", StandardScaler()), ("noweight", NoWeightEstimator())])
        f = PointReductionForecaster(estimator=pipe, time_weighter=LookupWeighter(mapping={}, default=1.0))
        with pytest.raises(ValueError, match="NoWeightEstimator"):
            f.fit(simple_series, forecasting_horizon=3)

    def test_plain_ridge_with_time_weight(self, simple_series):
        """Plain Ridge estimator with time_weight works (regression guard)."""
        f = PointReductionForecaster(estimator=Ridge())
        f.set_params(time_weighter=LookupWeighter(mapping={}, default=1.0))
        f.fit(simple_series, forecasting_horizon=3)
        y_pred = f.predict()
        # check_predict_time_columns owns the time-column contract; this guard
        # only confirms a plain Ridge estimator accepts the routed time weight.
        assert len(y_pred) == 3

    def test_pipeline_with_non_weight_aware_intermediate_step(self, simple_series):
        """Pipeline with intermediate step that lacks set_fit_request for sample_weight."""
        from sklearn.impute import SimpleImputer

        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler()),
            ("ridge", Ridge()),
        ])
        f = PointReductionForecaster(estimator=pipe)
        f.set_params(time_weighter=LookupWeighter(mapping={}, default=1.0))
        with pytest.warns(UserWarning, match="Could not disable sample_weight routing"):
            f.fit(simple_series, forecasting_horizon=3)
        y_pred = f.predict()
        # The warning is the assertion under test; the predict round-trip only
        # confirms fit still produced a usable forecaster.
        assert len(y_pred) == 3


class TestDuckTypedEstimator:
    """The estimator is constrained by duck typing, so it need not subclass BaseEstimator.

    `_parameter_constraints` declares ``HasMethods(["fit", "predict"])``, which admits
    sklearn-compatible libraries that do not inherit from ``BaseEstimator``: CatBoost and
    XGBoost's native APIs are both in that position. Predict used to assert the nominal
    type instead, so such an estimator fitted happily and then failed at predict with a
    bare ``AssertionError``.
    """

    class _DuckMultiOutput:
        """Minimal sklearn-shaped multi-output regressor that is NOT a BaseEstimator."""

        def __init__(self, fill: float = 1.5):
            self.fill = fill

        def get_params(self, deep: bool = True):
            return {"fill": self.fill}

        def set_params(self, **params):
            for k, v in params.items():
                setattr(self, k, v)
            return self

        def fit(self, X, y):
            self.n_outputs_ = np.asarray(y).reshape(len(y), -1).shape[1]
            return self

        def predict(self, X):
            return np.full((len(X), self.n_outputs_), self.fill)

    @staticmethod
    def _series() -> pl.DataFrame:
        time = pl.datetime_range(datetime(2024, 1, 1), datetime(2024, 1, 5, 23), interval="1h", eager=True)
        return pl.DataFrame({"time": time, "v": [float(i % 24) for i in range(len(time))]})

    def test_not_a_base_estimator_still_predicts(self):
        est = self._DuckMultiOutput()
        assert not isinstance(est, BaseEstimator), "the point of this test"
        fc = PointReductionForecaster(estimator=est, reduction_strategy="multi-output")
        fc.fit(self._series(), forecasting_horizon=6)
        pred = fc.predict()
        assert pred.height == 6
        assert pred["v"].to_list() == pytest.approx([1.5] * 6)
