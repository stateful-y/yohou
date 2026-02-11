from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest
from sklearn.base import clone
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from conftest import run_checks
from yohou.point import PointReductionForecaster
from yohou.testing import _yield_yohou_forecaster_checks

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
    {
        "a": range(length),
        "b": range(10, length + 10),
    },
    schema={
        "a": pl.Float64,
        "b": pl.Float64,
    },
)
y = pl.concat([time, y], how="horizontal")

X = pl.DataFrame(
    {
        "c": range(length),
        "d": range(10, length + 10),
        "e": range(20, length + 20),
    },
    schema={
        "c": pl.Float64,
        "d": pl.Float64,
        "e": pl.Float64,
    },
)
X = pl.concat([time, X], how="horizontal")

y_train, y_test, X_train, X_test = train_test_split(y, X, test_size=0.2, shuffle=False)


class TestPredict:
    @pytest.mark.parametrize(
        "fit_forecasting_horizon, predict_forecasting_horizon, expected_a",
        [
            (1, 5, [17.0, 18.0, 19.0, 20.0, 21.0]),
            (3, 5, [17.0, 18.0, 19.0, 20.0, 21.0]),
            (3, 2, [17.0, 18.0]),
        ],
    )
    def test_predict(self, fit_forecasting_horizon, predict_forecasting_horizon, expected_a):
        forecaster = PointReductionForecaster()

        forecaster.fit(y=y_train, X=X_train, forecasting_horizon=fit_forecasting_horizon)

        y_pred = forecaster.predict(
            X=X_test[:predict_forecasting_horizon],
            forecasting_horizon=predict_forecasting_horizon,
        )

        expected_y_pred = pl.DataFrame(
            {
                "observed_time": [y_train["time"][-1]] * predict_forecasting_horizon,
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
                "observed_time": pl.Datetime(time_unit="us", time_zone=None),
                "time": pl.Datetime(time_unit="us", time_zone=None),
                "a": pl.Float64,
                "b": pl.Float64,
            },
        )
        pl.testing.assert_frame_equal(y_pred, expected_y_pred)


class TestObservePredict:
    @pytest.mark.parametrize(
        "fit_forecasting_horizon, predict_forecasting_horizon, stride, expected_a",
        [
            (1, 5, 1, [22.0, 23.0, 24.0, 25.0, 26.0]),
            (3, 5, 2, [22.0, 23.0, 24.0, 25.0, 26.0]),
            (3, 2, 1, [22.0, 23.0]),
        ],
    )
    def test_observe_predict(self, fit_forecasting_horizon, predict_forecasting_horizon, stride, expected_a):
        forecaster = PointReductionForecaster()

        forecaster.fit(y=y_train, X=X_train, forecasting_horizon=fit_forecasting_horizon)

        # Extend X_test for future horizon
        last_time = X_test["time"][-1]
        future_time = pl.datetime_range(
            start=last_time + timedelta(seconds=1),
            end=last_time + timedelta(seconds=predict_forecasting_horizon),
            interval="1s",
            eager=True,
        )

        # Calculate start values for future features
        # X_test ends at index 21 (length-1)
        # So next values start at 22
        start_val = length
        future_X = pl.DataFrame(
            {
                "time": future_time,
                "c": range(start_val, start_val + predict_forecasting_horizon),
                "d": range(start_val + 10, start_val + 10 + predict_forecasting_horizon),
                "e": range(start_val + 20, start_val + 20 + predict_forecasting_horizon),
            },
            schema=X_test.schema,
        )

        X_test_extended = pl.concat([X_test, future_X])

        y_pred = forecaster.observe_predict(
            y=y_test,
            X=X_test_extended,
            forecasting_horizon=predict_forecasting_horizon,
            stride=stride,
        )

        # Get the last set of predictions (after all observations)
        y_pred = y_pred.tail(predict_forecasting_horizon)

        expected_y_pred = pl.DataFrame(
            {
                "observed_time": [y_test["time"][-1]] * predict_forecasting_horizon,
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
                "observed_time": pl.Datetime(time_unit="us", time_zone=None),
                "time": pl.Datetime(time_unit="us", time_zone=None),
                "a": pl.Float64,
                "b": pl.Float64,
            },
        )
        pl.testing.assert_frame_equal(y_pred, expected_y_pred)


y_panel = pl.DataFrame({
    "x__a": range(length),
    "x__b": range(10, length + 10),
    "y__a": range(10, length + 10),
    "y__b": range(20, length + 20),
})
y_panel = pl.concat([time, y_panel], how="horizontal")

X_panel = pl.DataFrame({
    "x__c": range(length),
    "y__c": range(10, length + 10),
    "d": range(10, length + 10),
    "e": range(20, length + 20),
})
X_panel = pl.concat([time, X_panel], how="horizontal")

y_train_panel, y_test_panel, X_train_panel, X_test_panel = train_test_split(
    y_panel, X_panel, test_size=0.2, shuffle=False
)


class TestObservePredictGlobal:
    @pytest.mark.parametrize(
        "fit_forecasting_horizon, predict_forecasting_horizon, stride, expected_a",
        [
            (1, 5, 1, [22.0, 23.0, 24.0, 25.0, 26.0]),
            (3, 5, 2, [22.0, 23.0, 24.0, 25.0, 26.0]),
            (3, 2, 1, [22.0, 23.0]),
        ],
    )
    def test_observe_predict_global(self, fit_forecasting_horizon, predict_forecasting_horizon, stride, expected_a):
        forecaster = PointReductionForecaster()

        forecaster.fit(
            y=y_train_panel,
            X=X_train_panel,
            forecasting_horizon=fit_forecasting_horizon,
        )

        # Extend X_test_panel for future horizon
        last_time = X_test_panel["time"][-1]
        future_time = pl.datetime_range(
            start=last_time + timedelta(seconds=1),
            end=last_time + timedelta(seconds=predict_forecasting_horizon),
            interval="1s",
            eager=True,
        )

        start_val = length
        future_X_panel = pl.DataFrame(
            {
                "time": future_time,
                "x__c": range(start_val, start_val + predict_forecasting_horizon),
                "y__c": range(start_val + 10, start_val + 10 + predict_forecasting_horizon),
                "d": range(start_val + 10, start_val + 10 + predict_forecasting_horizon),
                "e": range(start_val + 20, start_val + 20 + predict_forecasting_horizon),
            },
        )

        X_test_panel_extended = pl.concat([X_test_panel, future_X_panel])

        y_pred = forecaster.observe_predict(
            y=y_test_panel,
            X=X_test_panel_extended,
            forecasting_horizon=predict_forecasting_horizon,
            stride=stride,
        )

        # Get the last set of predictions (after all observations)
        y_pred = y_pred.tail(predict_forecasting_horizon)

        expected_y_pred = pl.DataFrame(
            {
                "observed_time": [y_test_panel["time"][-1]] * predict_forecasting_horizon,
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
                "observed_time": pl.Datetime(time_unit="us", time_zone=None),
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
        y, X = y_X_factory(length=100, seed=42)
        y_train, y_test = y[:80], y[80:]
        X_train, X_test = X[:80], X[80:]

        forecaster_fitted = clone(forecaster)
        forecaster_fitted.fit(y_train, X_train, forecasting_horizon=3)

        run_checks(
            forecaster_fitted,
            _yield_yohou_forecaster_checks(forecaster_fitted, y_train, X_train, y_test, X_test),
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
        forecaster.fit(y_train, X=None, forecasting_horizon=1)

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
        forecaster.fit(y_train, X=None, forecasting_horizon=1)

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


class TestTargetAsFeatureParam:
    """Tests for target_as_feature parameter on PointReductionForecaster."""

    def test_target_as_feature_in_get_params(self):
        """target_as_feature should appear in get_params()."""
        forecaster = PointReductionForecaster()
        params = forecaster.get_params()
        assert "target_as_feature" in params
        assert params["target_as_feature"] == "transformed"

    def test_target_as_feature_set_params(self):
        """target_as_feature should be settable via set_params()."""
        forecaster = PointReductionForecaster()
        forecaster.set_params(target_as_feature=None)
        assert forecaster.target_as_feature is None

    def test_target_as_feature_constructor(self):
        """target_as_feature should be settable via constructor."""
        forecaster = PointReductionForecaster(target_as_feature="raw")
        assert forecaster.target_as_feature == "raw"
        params = forecaster.get_params()
        assert params["target_as_feature"] == "raw"

    def test_target_as_feature_clone(self):
        """target_as_feature should survive clone()."""
        from sklearn.base import clone

        forecaster = PointReductionForecaster(target_as_feature=None)
        cloned = clone(forecaster)
        assert cloned.target_as_feature is None
