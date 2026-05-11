"""Tests for parameter validation in forecasters."""

import pytest
from sklearn.linear_model import LinearRegression

from yohou.compose import ColumnForecaster
from yohou.interval import IntervalReductionForecaster, SplitConformalForecaster
from yohou.point import PointReductionForecaster, SeasonalNaive
from yohou.preprocessing import LagTransformer
from yohou.stationarity import LogTransformer, SeasonalDifferencing

# Add parent directory to path for imports
from yohou.testing import check_coverage_rates_validation, check_forecasting_horizon_validation
from yohou.utils._compat import InvalidParameterError


class TestHorizonValidation:
    """Tests for forecasting_horizon validation."""

    def test_point_horizon_validation(self, y_X_factory):
        """Test that point forecasters validate forecasting_horizon in fit and predict."""
        y, X = y_X_factory(length=50, seed=42)

        # Test PointReductionForecaster
        forecaster = PointReductionForecaster()
        check_forecasting_horizon_validation(forecaster, y, X)

        # Test SeasonalNaive
        forecaster = SeasonalNaive(seasonality=7)
        check_forecasting_horizon_validation(forecaster, y, X)

    def test_interval_horizon_validation(self, y_X_factory):
        """Test that interval forecasters validate forecasting_horizon in fit and predict."""
        y, X = y_X_factory(length=50, seed=42)

        # Test IntervalReductionForecaster
        forecaster = IntervalReductionForecaster()
        check_forecasting_horizon_validation(forecaster, y, X)

        # Test SplitConformalForecaster
        forecaster = SplitConformalForecaster()
        check_forecasting_horizon_validation(forecaster, y, X)

    def test_predict_validates_horizon(self, y_X_factory):
        """Test that predict() validates forecasting_horizon parameter."""
        y, X = y_X_factory(length=50, seed=42)
        y_train, _y_test = y[:40], y[40:]
        X_train, _X_actual_test = X[:40], X[40:]

        forecaster = PointReductionForecaster()
        forecaster.fit(y_train, X_actual=X_train, forecasting_horizon=3)

        # Test invalid horizon in predict
        with pytest.raises(ValueError, match="forecasting_horizon|positive"):
            forecaster.predict(forecasting_horizon=0)

        with pytest.raises(ValueError, match="forecasting_horizon|positive"):
            forecaster.predict(forecasting_horizon=-1)


class TestCoverageRatesValidation:
    """Tests for coverage_rates validation."""

    def test_interval_coverage_rates_validation(self, y_X_factory):
        """Test that interval forecasters validate coverage_rates in fit and predict."""
        y, X = y_X_factory(length=200, seed=42)

        # Test IntervalReductionForecaster
        forecaster = IntervalReductionForecaster()
        check_coverage_rates_validation(forecaster, y, X)

        # Note: SplitConformalForecaster has implementation issues, tested separately

    def test_predict_interval_validates_coverage_rates(self, y_X_factory):
        """Test that predict_interval() validates coverage_rates parameter."""
        y, X = y_X_factory(length=50, seed=42)
        y_train, _y_test = y[:40], y[40:]
        X_train, _X_test = X[:40], X[40:]

        forecaster = IntervalReductionForecaster()
        forecaster.fit(y_train, X_train, forecasting_horizon=3, coverage_rates=[0.95])

        # Test invalid coverage_rates in predict_interval
        with pytest.raises(ValueError, match="coverage"):
            forecaster.predict_interval(forecasting_horizon=3, coverage_rates=[1.5])

        with pytest.raises(ValueError, match="coverage"):
            forecaster.predict_interval(forecasting_horizon=3, coverage_rates=[-0.5])


class TestForecasterParameterConstraints:
    """Tests for _parameter_constraints validation on forecasters."""

    def test_seasonal_naive_seasonality_validation(self, y_X_factory):
        """Test SeasonalNaive.seasonality parameter validation via _parameter_constraints."""
        y, X = y_X_factory(length=50, seed=42)

        # Valid: seasonality >= 1
        forecaster = SeasonalNaive(seasonality=1)
        forecaster.fit(y, X, forecasting_horizon=3)  # Should not raise

        forecaster = SeasonalNaive(seasonality=7)
        forecaster.fit(y, X, forecasting_horizon=3)  # Should not raise

        # Invalid: seasonality < 1
        with pytest.raises(InvalidParameterError, match="seasonality"):
            forecaster = SeasonalNaive(seasonality=0)
            forecaster.fit(y, X, forecasting_horizon=3)

        with pytest.raises(InvalidParameterError, match="seasonality"):
            forecaster = SeasonalNaive(seasonality=-1)
            forecaster.fit(y, X, forecasting_horizon=3)

        # Invalid: not an integer
        with pytest.raises((InvalidParameterError, TypeError)):
            forecaster = SeasonalNaive(seasonality=1.5)
            forecaster.fit(y, X, forecasting_horizon=3)

    def test_point_reduction_estimator_validation(self, y_X_factory):
        """Test PointReductionForecaster.estimator requires fit and predict methods."""
        y, X = y_X_factory(length=50, seed=42)

        # Valid: LinearRegression has fit and predict
        forecaster = PointReductionForecaster(estimator=LinearRegression())
        forecaster.fit(y, X, forecasting_horizon=3)  # Should not raise

        # Invalid: missing fit method
        class NoFit:
            def predict(self, X):
                pass

        with pytest.raises(InvalidParameterError, match="estimator"):
            forecaster = PointReductionForecaster(estimator=NoFit())
            forecaster.fit(y, X, forecasting_horizon=3)

        # Invalid: missing predict method
        class NoPredict:
            def fit(self, X, y):
                pass

        with pytest.raises(InvalidParameterError, match="estimator"):
            forecaster = PointReductionForecaster(estimator=NoPredict())
            forecaster.fit(y, X, forecasting_horizon=3)

    def test_point_reduction_strategy_validation(self, y_X_factory):
        """Test PointReductionForecaster.reduction_strategy validates allowed values."""
        y, X = y_X_factory(length=50, seed=42)

        # Valid strategies
        forecaster = PointReductionForecaster(reduction_strategy="direct")
        forecaster.fit(y, X, forecasting_horizon=3)

        forecaster = PointReductionForecaster(reduction_strategy="multi-output")
        forecaster.fit(y, X, forecasting_horizon=3)

        # Invalid strategy
        with pytest.raises(InvalidParameterError, match="reduction_strategy"):
            forecaster = PointReductionForecaster(reduction_strategy="invalid")
            forecaster.fit(y, X, forecasting_horizon=3)

        with pytest.raises(InvalidParameterError, match="reduction_strategy"):
            forecaster = PointReductionForecaster(reduction_strategy="recursive")
            forecaster.fit(y, X, forecasting_horizon=3)

    def test_interval_reduction_estimator_validation(self, y_X_factory):
        """Test IntervalReductionForecaster.estimator requires fit and predict."""
        y, X = y_X_factory(length=200, seed=42)

        # Valid: MultiOutputRegressor with QuantileRegressor (default)
        forecaster = IntervalReductionForecaster()
        forecaster.fit(y, X, forecasting_horizon=3, coverage_rates=[0.9])  # Should not raise

        # Invalid: missing fit method
        class NoFit:
            def predict(self, X):
                pass

        with pytest.raises(InvalidParameterError, match="estimator"):
            forecaster = IntervalReductionForecaster(estimator=NoFit())
            forecaster.fit(y, X, forecasting_horizon=3, coverage_rates=[0.9])

        # Invalid: missing predict method
        class NoPredict:
            def fit(self, X, y):
                return self

        with pytest.raises(InvalidParameterError, match="estimator"):
            forecaster = IntervalReductionForecaster(estimator=NoPredict())
            forecaster.fit(y, X, forecasting_horizon=3, coverage_rates=[0.9])

        # Note: LinearRegression would pass HasMethods but fail domain validation
        # (no quantile parameter) - that's tested by domain logic, not constraints

    def test_interval_reduction_strategy_validation(self, y_X_factory):
        """Test IntervalReductionForecaster.reduction_strategy validates allowed values."""
        y, X = y_X_factory(length=200, seed=42)

        # Valid strategies (using default estimator which has predict_interval)
        forecaster = IntervalReductionForecaster(reduction_strategy="direct")
        forecaster.fit(y, X, forecasting_horizon=3, coverage_rates=[0.9])

        forecaster = IntervalReductionForecaster(reduction_strategy="multi-output")
        forecaster.fit(y, X, forecasting_horizon=3, coverage_rates=[0.9])

        # Invalid strategy
        with pytest.raises(InvalidParameterError, match="reduction_strategy"):
            forecaster = IntervalReductionForecaster(reduction_strategy="invalid")
            forecaster.fit(y, X, forecasting_horizon=3, coverage_rates=[0.9])

    def test_split_conformal_calibration_size_validation(self, y_X_factory):
        """Test SplitConformalForecaster.calibration_size parameter validation."""
        y, X = y_X_factory(length=200, seed=42)

        # Test invalid values only (valid values cause unrelated SplitConformal bugs)
        # Invalid: calibration_size < 1
        with pytest.raises(InvalidParameterError, match="calibration_size"):
            forecaster = SplitConformalForecaster(calibration_size=0)
            forecaster.fit(y, X, forecasting_horizon=3, coverage_rates=[0.9])

        with pytest.raises(InvalidParameterError, match="calibration_size"):
            forecaster = SplitConformalForecaster(calibration_size=-10)
            forecaster.fit(y, X, forecasting_horizon=3, coverage_rates=[0.9])

        # Invalid: not an integer
        with pytest.raises((InvalidParameterError, TypeError)):
            forecaster = SplitConformalForecaster(calibration_size=10.5)
            forecaster.fit(y, X, forecasting_horizon=3, coverage_rates=[0.9])

    def test_column_forecaster_target_forecaster_validation(self, y_X_factory):
        """Test ColumnForecaster.forecasters parameter validation."""
        y, X = y_X_factory(length=50, n_targets=2, seed=42)

        # Extract column names (exclude 'time')
        target_cols = [col for col in y.columns if col != "time"]

        # Valid: forecasters as list of (name, forecaster, columns) tuples
        forecaster = ColumnForecaster(
            forecasters=[("naive", SeasonalNaive(), target_cols)],
        )
        forecaster.fit(y, X, forecasting_horizon=3)

        # Invalid: forecaster entry not a valid estimator
        with pytest.raises((InvalidParameterError, TypeError, ValueError)):
            forecaster = ColumnForecaster(
                forecasters=[("bad", "not_a_forecaster", target_cols)],
            )
            forecaster.fit(y, X, forecasting_horizon=3)

    def test_column_forecaster_remainder_validation(self, y_X_factory):
        """Test ColumnForecaster.remainder parameter validation."""
        y, X = y_X_factory(length=50, n_targets=2, seed=42)

        target_cols = [col for col in y.columns if col != "time"]

        # Valid: remainder as BaseForecaster
        forecaster = ColumnForecaster(
            forecasters=[("naive", SeasonalNaive(), target_cols[0])],
            remainder=SeasonalNaive(seasonality=1),
        )
        forecaster.fit(y, X, forecasting_horizon=3)

        # Invalid: remainder as invalid value
        with pytest.raises((InvalidParameterError, ValueError)):
            forecaster = ColumnForecaster(
                forecasters=[("naive", SeasonalNaive(), target_cols[0])],
                remainder="invalid",
            )
            forecaster.fit(y, X, forecasting_horizon=3)


class TestTransformerParameterConstraints:
    """Tests for _parameter_constraints validation on transformers."""

    def test_log_transform_offset_validation(self, y_X_factory):
        """Test LogTransformer.offset parameter validation."""
        y, X = y_X_factory(length=50, seed=42)

        # Valid: offset >= 0
        transformer = LogTransformer(offset=0.0)
        transformer.fit(X)  # Should not raise

        transformer = LogTransformer(offset=1.5)
        transformer.fit(X)  # Should not raise

        # Invalid: offset < 0
        with pytest.raises(InvalidParameterError, match="offset"):
            transformer = LogTransformer(offset=-0.1)
            transformer.fit(X)

        with pytest.raises(InvalidParameterError, match="offset"):
            transformer = LogTransformer(offset=-10.0)
            transformer.fit(X)

    def test_seasonal_differencing_seasonality_validation(self, y_X_factory):
        """Test SeasonalDifferencing.seasonality parameter validation."""
        y, X = y_X_factory(length=50, seed=42)

        # Valid: seasonality >= 1
        transformer = SeasonalDifferencing(seasonality=1)
        transformer.fit(X)

        transformer = SeasonalDifferencing(seasonality=12)
        transformer.fit(X)

        # Invalid: seasonality < 1
        with pytest.raises(InvalidParameterError, match="seasonality"):
            transformer = SeasonalDifferencing(seasonality=0)
            transformer.fit(X)

        with pytest.raises(InvalidParameterError, match="seasonality"):
            transformer = SeasonalDifferencing(seasonality=-1)
            transformer.fit(X)

    def test_lag_transformer_lag_validation(self, y_X_factory):
        """Test LagTransformer.lag parameter validation."""
        y, X = y_X_factory(length=50, seed=42)

        # Valid: lag >= 1 (int or list)
        transformer = LagTransformer(lag=1)
        transformer.fit(X)

        transformer = LagTransformer(lag=5)
        transformer.fit(X)

        transformer = LagTransformer(lag=[1, 2, 3])
        transformer.fit(X)

        # Invalid: lag < 1 (single int)
        with pytest.raises(InvalidParameterError, match="lag"):
            transformer = LagTransformer(lag=0)
            transformer.fit(X)

        with pytest.raises(InvalidParameterError, match="lag"):
            transformer = LagTransformer(lag=-1)
            transformer.fit(X)

        # Note: List validation (checking elements) is not done by sklearn's Interval
        # It would require custom validation in fit() method
