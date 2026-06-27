"""Tests for point forecasting metrics."""

from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest
from sklearn.exceptions import NotFittedError

from conftest import run_checks as _run_checks_base
from yohou.metrics import (
    MaxAbsoluteError,
    MeanAbsoluteError,
    MeanAbsolutePercentageError,
    MeanAbsoluteScaledError,
    MeanDirectionalAccuracy,
    MeanSquaredError,
    MedianAbsoluteError,
    R2Score,
    RootMeanSquaredError,
    RootMeanSquaredScaledError,
    SymmetricMeanAbsolutePercentageError,
)
from yohou.testing import _yield_yohou_scorer_checks
from yohou.weighting import LookupWeighter, TableWeighter


@pytest.fixture
def scorer_data_factory():
    """Factory for generating time series data for scorer testing."""

    def _factory(length=100, n_targets=1, n_features=0, seed=42):
        """Generate synthetic time series data.

        Parameters
        ----------
        length : int
            Length of time series.
        n_targets : int
            Number of target columns.
        n_features : int
            Number of feature columns.
        seed : int
            Random seed.

        Returns
        -------
        y : pl.DataFrame
            Target time series with "time" column.
        X : pl.DataFrame or None
            Feature time series with "time" column (None if n_features=0).

        """
        np.random.seed(seed)

        time = [datetime(2020, 1, 1) + timedelta(days=i) for i in range(length)]

        # Generate target columns
        y_data = {"time": time}
        for i in range(n_targets):
            y_data[f"target_{i}"] = np.random.randn(length) * 10 + 100

        y = pl.DataFrame(y_data)

        # Generate feature columns if requested
        if n_features > 0:
            X_data = {"time": time}
            for i in range(n_features):
                X_data[f"feature_{i}"] = np.random.randn(length) * 5 + 50
            X = pl.DataFrame(X_data)
        else:
            X = None

        return y, X

    return _factory


@pytest.fixture
def y_true_y_pred():
    """Simple ground truth and prediction pair for basic tests."""
    y_true = pl.DataFrame({
        "time": [datetime(2020, 1, 1), datetime(2020, 1, 2), datetime(2020, 1, 3)],
        "value": [10.0, 20.0, 30.0],
    })
    y_pred = pl.DataFrame({
        "vintage_time": [datetime(2019, 12, 31)] * 3,
        "time": [datetime(2020, 1, 1), datetime(2020, 1, 2), datetime(2020, 1, 3)],
        "value": [12.0, 19.0, 28.0],
    })
    return y_true, y_pred


def run_checks(scorer, y_truth, y_pred):
    """Run all systematic checks for a scorer."""
    _run_checks_base(scorer, _yield_yohou_scorer_checks(scorer, y_truth, y_pred))


class TestSystematicChecks:
    def test_mae_systematic(self, y_true_y_pred):
        """Run systematic checks for MeanAbsoluteError."""
        y_truth, y_pred = y_true_y_pred
        scorer = MeanAbsoluteError()
        run_checks(scorer, y_truth, y_pred)

    def test_mse_systematic(self, y_true_y_pred):
        """Run systematic checks for MeanSquaredError."""
        y_truth, y_pred = y_true_y_pred
        scorer = MeanSquaredError()
        run_checks(scorer, y_truth, y_pred)

    def test_rmse_systematic(self, y_true_y_pred):
        """Run systematic checks for RootMeanSquaredError."""
        y_truth, y_pred = y_true_y_pred
        scorer = RootMeanSquaredError()
        run_checks(scorer, y_truth, y_pred)

    def test_rmsse_systematic(self, scorer_data_factory):
        """Run systematic checks for RootMeanSquaredScaledError."""
        y, _ = scorer_data_factory(length=100, n_targets=1, seed=42)
        y_train = y[:80]
        y_test = y[80:]

        # Create predictions
        y_pred = y_test.with_columns(pl.lit(datetime(2020, 1, 1)).alias("vintage_time"))

        # Fit scorer with training data
        scorer = RootMeanSquaredScaledError(seasonality=1)
        scorer.fit(y_train)

        run_checks(scorer, y_test, y_pred)

    def test_mape_systematic(self, y_true_y_pred):
        """Run systematic checks for MeanAbsolutePercentageError."""
        y_truth, y_pred = y_true_y_pred
        scorer = MeanAbsolutePercentageError()
        run_checks(scorer, y_truth, y_pred)

    def test_smape_systematic(self, y_true_y_pred):
        """Run systematic checks for SymmetricMeanAbsolutePercentageError."""
        y_truth, y_pred = y_true_y_pred
        scorer = SymmetricMeanAbsolutePercentageError()
        run_checks(scorer, y_truth, y_pred)

    def test_mase_systematic(self, scorer_data_factory):
        """Run systematic checks for MeanAbsoluteScaledError."""
        y, _ = scorer_data_factory(length=100, n_targets=1, seed=42)
        y_train = y[:80]
        y_test = y[80:]

        # Create predictions
        y_pred = y_test.with_columns(pl.lit(datetime(2020, 1, 1)).alias("vintage_time"))

        # Fit scorer with training data
        scorer = MeanAbsoluteScaledError(seasonality=1)
        scorer.fit(y_train)

        run_checks(scorer, y_test, y_pred)

    def test_median_absolute_error_systematic(self, y_true_y_pred):
        """Run systematic checks for MedianAbsoluteError."""
        y_truth, y_pred = y_true_y_pred
        scorer = MedianAbsoluteError()
        run_checks(scorer, y_truth, y_pred)


class TestRMSE:
    def test_rmse_equals_sqrt_mse(self, y_true_y_pred):
        """RMSE should equal square root of MeanSquaredError."""
        y_true, y_pred = y_true_y_pred

        mse = MeanSquaredError()
        rmse = RootMeanSquaredError()

        mse.fit(y_true)
        rmse.fit(y_true)
        mse_score = mse.score(y_true, y_pred)
        rmse_score = rmse.score(y_true, y_pred)

        assert np.isclose(rmse_score, np.sqrt(mse_score)), (
            f"RMSE ({rmse_score}) != sqrt(MeanSquaredError) ({np.sqrt(mse_score)})"
        )

    def test_rmse_units_match_target(self, y_true_y_pred):
        """RMSE should have same units as target (not squared like MeanSquaredError)."""
        y_true, y_pred = y_true_y_pred

        mae = MeanAbsoluteError()
        rmse = RootMeanSquaredError()
        mse = MeanSquaredError()

        mae.fit(y_true)
        rmse.fit(y_true)
        mse.fit(y_true)
        mae_score = mae.score(y_true, y_pred)
        rmse_score = rmse.score(y_true, y_pred)
        mse_score = mse.score(y_true, y_pred)

        # RMSE should be between MAE and sqrt(MeanSquaredError) for typical errors
        # RMSE should be closer in magnitude to MAE than to MeanSquaredError
        assert mae_score > 0, "MAE should be positive"
        assert rmse_score > 0, "RMSE should be positive"
        assert rmse_score < mse_score, f"RMSE ({rmse_score}) should be less than MeanSquaredError ({mse_score})"
        assert np.isclose(rmse_score, np.sqrt(mse_score)), "RMSE should equal sqrt(MeanSquaredError)"

    def test_rmse_perfect_prediction(self):
        """RMSE should be zero for perfect predictions."""
        y_true = pl.DataFrame({
            "time": [datetime(2020, 1, 1), datetime(2020, 1, 2)],
            "value": [10.0, 20.0],
        })
        y_pred = pl.DataFrame({
            "vintage_time": [datetime(2019, 12, 31)] * 2,
            "time": [datetime(2020, 1, 1), datetime(2020, 1, 2)],
            "value": [10.0, 20.0],
        })

        rmse = RootMeanSquaredError()
        rmse.fit(y_true)
        score = rmse.score(y_true, y_pred)

        assert np.isclose(score, 0.0), f"RMSE for perfect prediction should be 0, got {score}"

    def test_rmse_multiple_columns(self, scorer_data_factory):
        """RMSE should work with multiple target columns."""
        y, _ = scorer_data_factory(length=50, n_targets=3, n_features=0, seed=42)

        # Create predictions with small errors
        y_pred = y.clone()
        y_pred = y_pred.with_columns([pl.col(col) + np.random.randn(50) * 2 for col in y_pred.columns if col != "time"])
        y_pred = y_pred.with_columns(vintage_time=pl.lit(y["time"][0]))

        rmse = RootMeanSquaredError()
        rmse.fit(y)
        score = rmse.score(y, y_pred)

        assert isinstance(score, float), "RMSE should return float"
        assert score > 0, "RMSE should be positive for imperfect predictions"


class TestRMSSE:
    def test_rmsse_requires_fit(self):
        """RootMeanSquaredScaledError should raise error if used without fit()."""
        y_true = pl.DataFrame({
            "time": [datetime(2020, 1, 1), datetime(2020, 1, 2)],
            "value": [10.0, 20.0],
        })
        y_pred = pl.DataFrame({
            "vintage_time": [datetime(2019, 12, 31)] * 2,
            "time": [datetime(2020, 1, 1), datetime(2020, 1, 2)],
            "value": [12.0, 19.0],
        })

        rmsse = RootMeanSquaredScaledError(seasonality=1)

        with pytest.raises(NotFittedError, match="fitted"):
            rmsse.score(y_true, y_pred)

    def test_rmsse_requires_training_data(self):
        """RootMeanSquaredScaledError fit() should raise error if y_train is None."""
        rmsse = RootMeanSquaredScaledError(seasonality=1)

        with pytest.raises(ValueError, match="`y_train` is required for scorer.fit"):
            rmsse.fit(y_train=None)

    def test_rmsse_seasonality_too_large_error(self):
        """RootMeanSquaredScaledError should raise error if seasonality > training length - 1."""
        y_train = pl.DataFrame({
            "time": [datetime(2020, 1, 1), datetime(2020, 1, 2), datetime(2020, 1, 3)],
            "value": [10.0, 20.0, 30.0],
        })

        rmsse = RootMeanSquaredScaledError(seasonality=5)  # Too large for 3-row training data

        with pytest.raises(ValueError, match="Training data length.*must be greater than seasonality"):
            rmsse.fit(y_train)

    def test_rmsse_scaling_factors(self):
        """RootMeanSquaredScaledError should compute correct per-column scaling factors."""
        # Create training data with known seasonal pattern
        y_train = pl.DataFrame({
            "time": [datetime(2020, 1, 1) + timedelta(days=i) for i in range(10)],
            "value": [10.0, 12.0, 11.0, 13.0, 12.0, 14.0, 13.0, 15.0, 14.0, 16.0],
        })

        rmsse = RootMeanSquaredScaledError(seasonality=2)
        rmsse.fit(y_train)

        # Check that scales_ is a dict with expected column
        assert hasattr(rmsse, "scales_"), "RootMeanSquaredScaledError should have scales_ attribute after fit"
        assert isinstance(rmsse.scales_, dict), "scales_ should be a dictionary"
        assert "value" in rmsse.scales_, "scales_ should contain 'value' column"
        assert rmsse.scales_["value"] > 0, "Scaling factor should be positive"

    def test_rmsse_panel_data(self):
        """RootMeanSquaredScaledError should handle multiple columns with per-column scaling."""
        # Create training data with multiple columns
        y_train = pl.DataFrame({
            "time": [datetime(2020, 1, 1) + timedelta(days=i) for i in range(20)],
            "target_0": np.random.randn(20) * 10 + 100,
            "target_1": np.random.randn(20) * 5 + 50,
        })

        y_test = pl.DataFrame({
            "time": [datetime(2020, 1, 21), datetime(2020, 1, 22)],
            "target_0": [110.0, 115.0],
            "target_1": [52.0, 53.0],
        })

        y_pred = pl.DataFrame({
            "vintage_time": [datetime(2020, 1, 20)] * 2,
            "time": [datetime(2020, 1, 21), datetime(2020, 1, 22)],
            "target_0": [108.0, 113.0],
            "target_1": [51.0, 52.5],
        })

        rmsse = RootMeanSquaredScaledError(seasonality=2)
        rmsse.fit(y_train)

        # Check per-column scales
        assert len(rmsse.scales_) == 2, "Should have scales for both columns"
        assert "target_0" in rmsse.scales_, "Should have scale for target_0"
        assert "target_1" in rmsse.scales_, "Should have scale for target_1"

        # Compute score
        score = rmsse.score(y_test, y_pred)
        assert isinstance(score, float), "RootMeanSquaredScaledError should return float"
        assert score > 0, "RootMeanSquaredScaledError should be positive"

    def test_rmsse_different_seasonalities(self, scorer_data_factory):
        """RootMeanSquaredScaledError should work with different seasonality values."""
        y, _ = scorer_data_factory(length=100, n_targets=1, n_features=0, seed=42)

        y_train = y[:80]
        y_test = y[80:85]
        y_pred = y_test.clone().with_columns(vintage_time=pl.lit(y_train["time"][-1]))

        for seasonality in [1, 7, 12, 24]:
            rmsse = RootMeanSquaredScaledError(seasonality=seasonality)
            rmsse.fit(y_train)

            score = rmsse.score(y_test, y_pred)
            assert isinstance(score, float), (
                f"RootMeanSquaredScaledError with seasonality={seasonality} should return float"
            )
            assert score >= 0, f"RootMeanSquaredScaledError with seasonality={seasonality} should be non-negative"

    def test_rmsse_perfect_prediction_nonzero(self):
        """RootMeanSquaredScaledError can be zero for perfect predictions but depends on training scale."""
        y_train = pl.DataFrame({
            "time": [datetime(2020, 1, 1) + timedelta(days=i) for i in range(10)],
            "value": [10.0, 12.0, 11.0, 13.0, 12.0, 14.0, 13.0, 15.0, 14.0, 16.0],
        })

        y_test = pl.DataFrame({
            "time": [datetime(2020, 1, 11), datetime(2020, 1, 12)],
            "value": [15.0, 17.0],
        })

        # Perfect predictions
        y_pred = y_test.clone().with_columns(vintage_time=pl.lit(y_train["time"][-1]))

        rmsse = RootMeanSquaredScaledError(seasonality=2)
        rmsse.fit(y_train)
        score = rmsse.score(y_test, y_pred)

        # Perfect prediction should give RootMeanSquaredScaledError = 0
        assert np.isclose(score, 0.0, atol=1e-6), (
            f"RootMeanSquaredScaledError for perfect prediction should be ~0, got {score}"
        )

    def test_rmsse_vs_naive_baseline(self):
        """RootMeanSquaredScaledError < 1 means better than naive seasonal, RootMeanSquaredScaledError > 1 means worse."""
        # Create training data with clear seasonal pattern (period=2)
        y_train = pl.DataFrame({
            "time": [datetime(2020, 1, 1) + timedelta(days=i) for i in range(10)],
            "value": [10.0, 20.0, 10.0, 20.0, 10.0, 20.0, 10.0, 20.0, 10.0, 20.0],
        })

        y_test = pl.DataFrame({
            "time": [datetime(2020, 1, 11), datetime(2020, 1, 12)],
            "value": [10.0, 20.0],
        })

        # Good predictions (match pattern)
        y_pred_good = pl.DataFrame({
            "vintage_time": [datetime(2020, 1, 10)] * 2,
            "time": [datetime(2020, 1, 11), datetime(2020, 1, 12)],
            "value": [10.5, 19.5],  # Close to true pattern
        })

        # Bad predictions (opposite pattern)
        y_pred_bad = pl.DataFrame({
            "vintage_time": [datetime(2020, 1, 10)] * 2,
            "time": [datetime(2020, 1, 11), datetime(2020, 1, 12)],
            "value": [20.0, 10.0],  # Reversed pattern
        })

        rmsse = RootMeanSquaredScaledError(seasonality=2)
        rmsse.fit(y_train)

        score_good = rmsse.score(y_test, y_pred_good)
        score_bad = rmsse.score(y_test, y_pred_bad)

        # Good predictions should have lower RootMeanSquaredScaledError than bad predictions
        assert score_good < score_bad, (
            f"Good predictions RootMeanSquaredScaledError ({score_good}) should be < bad predictions ({score_bad})"
        )

    def test_rmsse_tags(self):
        """RootMeanSquaredScaledError should have requires_calibration=True tag."""
        rmsse = RootMeanSquaredScaledError(seasonality=1)
        tags = rmsse.__sklearn_tags__()

        assert tags.scorer_tags.requires_calibration is True, (
            "RootMeanSquaredScaledError should require calibration (fit)"
        )


class TestIntegration:
    def test_all_point_scorers_stateless_except_rmsse(self):
        """Verify that only RootMeanSquaredScaledError requires fit among point scorers."""
        mae = MeanAbsoluteError()
        mse = MeanSquaredError()
        rmse = RootMeanSquaredError()
        rmsse = RootMeanSquaredScaledError()

        assert mae.__sklearn_tags__().scorer_tags.requires_calibration is False
        assert mse.__sklearn_tags__().scorer_tags.requires_calibration is False
        assert rmse.__sklearn_tags__().scorer_tags.requires_calibration is False
        assert rmsse.__sklearn_tags__().scorer_tags.requires_calibration is True

    def test_point_scorers_comparison(self, scorer_data_factory):
        """Compare behavior of MeanAbsoluteError, MeanSquaredError, RMSE on same data."""
        y, _ = scorer_data_factory(length=50, n_targets=1, n_features=0, seed=42)

        y_train = y[:40]
        y_test = y[40:45]

        # Create predictions with controlled error
        y_pred = y_test.clone()
        errors = np.array([1.0, -2.0, 1.5, -1.5, 2.0])  # Known errors
        y_pred = y_pred.with_columns([(pl.col(col) + errors).alias(col) for col in y_pred.columns if col != "time"])
        y_pred = y_pred.with_columns(vintage_time=pl.lit(y_train["time"][-1]))

        mae = MeanAbsoluteError()
        mse = MeanSquaredError()
        rmse = RootMeanSquaredError()

        mae.fit(y_train)
        mse.fit(y_train)
        rmse.fit(y_train)
        mae_score = mae.score(y_test, y_pred)
        mse_score = mse.score(y_test, y_pred)
        rmse_score = rmse.score(y_test, y_pred)

        # Verify relationships
        assert np.isclose(rmse_score, np.sqrt(mse_score)), "RMSE should equal sqrt(MeanSquaredError)"
        assert mae_score == pytest.approx(np.mean(np.abs(errors)), abs=1e-5), "MAE should match mean absolute error"
        assert mse_score == pytest.approx(np.mean(errors**2), abs=1e-5), (
            "MeanSquaredError should match mean squared error"
        )


class TestComponentwise:
    def test_mae_per_step(self, y_true_y_pred):
        """MeanAbsoluteError with aggregation_method=['componentwise'] should return DataFrame with time column."""
        y_true, y_pred = y_true_y_pred

        mae = MeanAbsoluteError(aggregation_method=["componentwise"])
        mae.fit(y_true)
        result = mae.score(y_true, y_pred)

        # Check return type and structure
        assert isinstance(result, pl.DataFrame), "aggregation_method=['componentwise'] should return DataFrame"
        assert "time" in result.columns, "DataFrame should have 'time' column"
        assert "mae" in result.columns, "DataFrame should have 'mae' column"
        assert len(result) == 3, f"Should have 3 steps, got {len(result)}"

        # Verify time column contains datetime values
        assert result["time"].dtype == pl.Datetime, "Time column should be datetime type"

        # Verify per-step MAE values
        expected_errors = [2.0, 1.0, 2.0]  # |12-10|, |19-20|, |28-30|
        assert np.allclose(result["mae"].to_numpy(), expected_errors, atol=1e-5), "Per-step MAE values incorrect"

        # Verify aggregated mean matches 'all' aggregation
        mae_default = MeanAbsoluteError()
        mae_default.fit(y_true)
        default_score = mae_default.score(y_true, y_pred)
        componentwise_mean = result["mae"].mean()

        assert np.isclose(default_score, componentwise_mean, atol=1e-5), (
            f"Aggregated componentwise mean ({componentwise_mean}) should match 'all' ({default_score})"
        )

    def test_mse_per_step(self, y_true_y_pred):
        """MeanSquaredError with aggregation_method=['componentwise'] should return DataFrame with time column."""
        y_true, y_pred = y_true_y_pred

        mse = MeanSquaredError(aggregation_method=["componentwise"])
        mse.fit(y_true)
        result = mse.score(y_true, y_pred)

        # Check return type and structure
        assert isinstance(result, pl.DataFrame), "aggregation_method=['componentwise'] should return DataFrame"
        assert "time" in result.columns, "DataFrame should have 'time' column"
        assert "mse" in result.columns, "DataFrame should have 'mse' column"
        assert len(result) == 3, f"Should have 3 steps, got {len(result)}"

        # Verify time column contains datetime values
        assert result["time"].dtype == pl.Datetime, "Time column should be datetime type"

        # Verify per-step MeanSquaredError values
        expected_errors = [4.0, 1.0, 4.0]  # (12-10)^2, (19-20)^2, (28-30)^2
        assert np.allclose(result["mse"].to_numpy(), expected_errors, atol=1e-5), (
            "Per-step MeanSquaredError values incorrect"
        )

        # Verify aggregated mean matches 'all' aggregation
        mse_default = MeanSquaredError()
        mse_default.fit(y_true)
        default_score = mse_default.score(y_true, y_pred)
        componentwise_mean = result["mse"].mean()

        assert np.isclose(default_score, componentwise_mean, atol=1e-5), (
            f"Aggregated componentwise mean ({componentwise_mean}) should match 'all' ({default_score})"
        )

    def test_rmse_per_step(self, y_true_y_pred):
        """RMSE with aggregation_method=['componentwise'] should return DataFrame with time column."""
        y_true, y_pred = y_true_y_pred

        rmse = RootMeanSquaredError(aggregation_method=["componentwise"])
        rmse.fit(y_true)
        result = rmse.score(y_true, y_pred)

        # Check return type and structure
        assert isinstance(result, pl.DataFrame), "aggregation_method=['componentwise'] should return DataFrame"
        assert "time" in result.columns, "DataFrame should have 'time' column"
        assert "rmse" in result.columns, "DataFrame should have 'rmse' column"
        assert len(result) == 3, f"Should have 3 steps, got {len(result)}"

        # Verify time column contains datetime values
        assert result["time"].dtype == pl.Datetime, "Time column should be datetime type"

        # Verify per-step RMSE values (sqrt of MSE)
        expected_errors = [2.0, 1.0, 2.0]  # sqrt(4), sqrt(1), sqrt(4)
        assert np.allclose(result["rmse"].to_numpy(), expected_errors, atol=1e-5), "Per-step RMSE values incorrect"

        # Verify overall RMSE equals sqrt(mean(per_step_RMSE^2))
        # Note: mean(RMSE_per_step) != RMSE_overall due to non-linearity
        rmse_default = RootMeanSquaredError()
        rmse_default.fit(y_true)
        default_score = rmse_default.score(y_true, y_pred)

        # RMSE_overall = sqrt(mean(MSE_per_step)) = sqrt(mean([4, 1, 4])) = sqrt(3)
        expected_overall = np.sqrt(np.mean([4.0, 1.0, 4.0]))

        assert np.isclose(default_score, expected_overall, atol=1e-5), (
            f"Overall RMSE ({default_score}) should match sqrt(mean(MSE)) ({expected_overall})"
        )

    def test_rmsse_per_step(self):
        """RootMeanSquaredScaledError with aggregation_method=['componentwise'] should return DataFrame with time column."""
        # Create training data with known pattern
        y_train = pl.DataFrame({
            "time": [datetime(2020, 1, 1) + timedelta(days=i) for i in range(10)],
            "value": [10.0, 12.0, 11.0, 13.0, 12.0, 14.0, 13.0, 15.0, 14.0, 16.0],
        })

        y_test = pl.DataFrame({
            "time": [datetime(2020, 1, 11), datetime(2020, 1, 12), datetime(2020, 1, 13)],
            "value": [15.0, 17.0, 16.0],
        })

        y_pred = pl.DataFrame({
            "vintage_time": [datetime(2020, 1, 10)] * 3,
            "time": [datetime(2020, 1, 11), datetime(2020, 1, 12), datetime(2020, 1, 13)],
            "value": [15.5, 16.0, 16.5],
        })

        rmsse = RootMeanSquaredScaledError(seasonality=2, aggregation_method=["componentwise"])
        rmsse.fit(y_train)
        result = rmsse.score(y_test, y_pred)

        # Check return type and structure
        assert isinstance(result, pl.DataFrame), "aggregation_method=['componentwise'] should return DataFrame"
        assert "time" in result.columns, "DataFrame should have 'time' column"
        assert "rmsse" in result.columns, "DataFrame should have 'rmsse' column"
        assert len(result) == 3, f"Should have 3 steps, got {len(result)}"

        # Verify time column contains datetime values
        assert result["time"].dtype == pl.Datetime, "Time column should be datetime type"

        # Verify all componentwise RootMeanSquaredScaledError values are non-negative
        assert (result["rmsse"] >= 0).all(), (
            "All componentwise RootMeanSquaredScaledError values should be non-negative"
        )

        # Verify overall RootMeanSquaredScaledError is also computed correctly (uses sqrt, so mean(per_step) != overall)
        rmsse_default = RootMeanSquaredScaledError(seasonality=2)
        rmsse_default.fit(y_train)
        default_score = rmsse_default.score(y_test, y_pred)

        # Just verify the overall score is reasonable (positive, finite)
        assert np.isfinite(default_score), "Overall RootMeanSquaredScaledError should be finite"
        assert default_score > 0, "Overall RootMeanSquaredScaledError should be positive"

    def test_per_step_multiple_columns(self, scorer_data_factory):
        """Componentwise aggregate should average across multiple columns."""
        y, _ = scorer_data_factory(length=20, n_targets=2, n_features=0, seed=42)

        y_train = y[:15]
        y_test = y[15:18]

        # Create predictions with small errors
        y_pred = y_test.clone()
        y_pred = y_pred.with_columns([(pl.col(col) + 1.0).alias(col) for col in y_pred.columns if col != "time"])
        y_pred = y_pred.with_columns(vintage_time=pl.lit(y_train["time"][-1]))

        # Test MeanAbsoluteError componentwise with multiple columns
        mae = MeanAbsoluteError(aggregation_method=["componentwise"])
        mae.fit(y_train)
        result = mae.score(y_test, y_pred)

        assert "mae" in result.columns, "Should have mae column"
        assert "time" in result.columns, "Should have time column"
        assert len(result) == 3, "Should have 3 steps"

        # Verify all values are positive (since we added constant error)
        assert (result["mae"] > 0).all(), "All componentwise MAE values should be positive"

    def test_aggregate_parameter_validation(self):
        """aggregation_method parameter should accept valid list or string values."""
        # Valid: aggregation_method=['stepwise', 'vintagewise']
        mae_sv = MeanAbsoluteError(aggregation_method=["stepwise", "vintagewise"])
        assert mae_sv.aggregation_method == ["stepwise", "vintagewise"]

        # Valid: aggregation_method=['componentwise']
        mae_componentwise = MeanAbsoluteError(aggregation_method=["componentwise"])
        assert mae_componentwise.aggregation_method == ["componentwise"]

        # Valid: aggregation_method=['groupwise']
        mae_groupwise = MeanAbsoluteError(aggregation_method=["groupwise"])
        assert mae_groupwise.aggregation_method == ["groupwise"]

        # Valid: aggregation_method='all' (aggregates all dimensions)
        mae_all = MeanAbsoluteError(aggregation_method="all")
        assert mae_all.aggregation_method == "all"

        # Default should be 'all'
        mae_default = MeanAbsoluteError()
        assert mae_default.aggregation_method == "all"

        # Valid: multiple aggregation methods
        mae_multi = MeanAbsoluteError(aggregation_method=["stepwise", "vintagewise", "componentwise"])
        assert mae_multi.aggregation_method == ["stepwise", "vintagewise", "componentwise"]


class TestMAPE:
    def test_mape_basic_computation(self, y_true_y_pred):
        """MAPE should compute mean absolute percentage error."""
        y_true, y_pred = y_true_y_pred

        from yohou.metrics import MeanAbsolutePercentageError

        mape = MeanAbsolutePercentageError()
        mape.fit(y_true)
        score = mape.score(y_true, y_pred)

        # Expected: mean(abs([(10-12)/10, (20-19)/20, (30-28)/30])) * 100
        # = mean([0.2, 0.05, 0.0667]) * 100 ≈ 10.56%
        assert isinstance(score, float)
        assert 10 < score < 11  # Rough check

    def test_mape_epsilon_prevents_division_by_zero(self):
        """MAPE epsilon should handle zero actuals."""
        y_true = pl.DataFrame({
            "time": [datetime(2020, 1, 1)],
            "value": [0.0],  # Zero actual
        })
        y_pred = pl.DataFrame({
            "vintage_time": [datetime(2019, 12, 31)],
            "time": [datetime(2020, 1, 1)],
            "value": [10.0],
        })

        mape = MeanAbsolutePercentageError(epsilon=1e-8)
        mape.fit(y_true)
        score = mape.score(y_true, y_pred)

        # Should not raise, should be finite
        assert np.isfinite(score)

    def test_mape_default_epsilon(self):
        """MAPE should have default epsilon=1e-8."""
        mape = MeanAbsolutePercentageError()
        assert mape.epsilon == 1e-8

    def test_mape_scale_independent(self):
        """MAPE should be scale-independent."""
        y_true_1 = pl.DataFrame({
            "time": [datetime(2020, 1, 1), datetime(2020, 1, 2)],
            "value": [100.0, 200.0],
        })
        y_pred_1 = pl.DataFrame({
            "vintage_time": [datetime(2019, 12, 31)] * 2,
            "time": [datetime(2020, 1, 1), datetime(2020, 1, 2)],
            "value": [110.0, 210.0],
        })

        # Scale by 10x
        y_true_2 = y_true_1.with_columns(value=pl.col("value") * 10)
        y_pred_2 = y_pred_1.with_columns((pl.col("value") * 10).alias("value"))

        mape = MeanAbsolutePercentageError()
        mape.fit(y_true_1)
        score_1 = mape.score(y_true_1, y_pred_1)
        score_2 = mape.score(y_true_2, y_pred_2)

        # Scores should be identical (scale-independent)
        np.testing.assert_allclose(score_1, score_2, rtol=1e-5)


class TestSMAPE:
    def test_smape_basic_computation(self, y_true_y_pred):
        """SMAPE should compute symmetric mean absolute percentage error."""
        y_true, y_pred = y_true_y_pred

        from yohou.metrics import SymmetricMeanAbsolutePercentageError

        smape = SymmetricMeanAbsolutePercentageError()
        smape.fit(y_true)
        score = smape.score(y_true, y_pred)

        assert isinstance(score, float)
        assert 0 <= score <= 200  # sMAPE is bounded [0, 200]

    def test_smape_bounded(self):
        """SMAPE should be bounded between 0 and 200."""
        from yohou.metrics import SymmetricMeanAbsolutePercentageError

        y_true = pl.DataFrame({
            "time": [datetime(2020, 1, 1), datetime(2020, 1, 2)],
            "value": [100.0, 200.0],
        })
        # Perfect predictions
        y_pred_perfect = pl.DataFrame({
            "vintage_time": [datetime(2019, 12, 31)] * 2,
            "time": [datetime(2020, 1, 1), datetime(2020, 1, 2)],
            "value": [100.0, 200.0],
        })
        # Worst case: predict zero
        y_pred_worst = pl.DataFrame({
            "vintage_time": [datetime(2019, 12, 31)] * 2,
            "time": [datetime(2020, 1, 1), datetime(2020, 1, 2)],
            "value": [0.0, 0.0],
        })

        smape = SymmetricMeanAbsolutePercentageError()
        smape.fit(y_true)
        score_perfect = smape.score(y_true, y_pred_perfect)
        score_worst = smape.score(y_true, y_pred_worst)

        # Perfect should be 0, worst should be close to 200
        assert score_perfect == 0.0
        assert 195 < score_worst <= 200  # Close to theoretical max

    def test_smape_epsilon_handles_both_zero(self):
        """SMAPE epsilon should handle when both actual and predicted are zero."""
        from yohou.metrics import SymmetricMeanAbsolutePercentageError

        y_true = pl.DataFrame({
            "time": [datetime(2020, 1, 1)],
            "value": [0.0],
        })
        y_pred = pl.DataFrame({
            "vintage_time": [datetime(2019, 12, 31)],
            "time": [datetime(2020, 1, 1)],
            "value": [0.0],
        })

        smape = SymmetricMeanAbsolutePercentageError(epsilon=1e-8)
        smape.fit(y_true)
        score = smape.score(y_true, y_pred)

        # Should not raise, should be finite
        assert np.isfinite(score)


class TestMASE:
    def test_mase_basic_computation(self):
        """MASE should compute mean absolute scaled error."""
        from yohou.metrics import MeanAbsoluteScaledError

        y_train = pl.DataFrame({
            "time": [datetime(2020, 1, 1) + timedelta(days=i) for i in range(10)],
            "value": [10.0, 12.0, 11.0, 13.0, 12.0, 14.0, 13.0, 15.0, 14.0, 16.0],
        })

        y_true = pl.DataFrame({
            "time": [datetime(2020, 1, 11), datetime(2020, 1, 12)],
            "value": [15.0, 17.0],
        })

        y_pred = pl.DataFrame({
            "vintage_time": [datetime(2020, 1, 10)] * 2,
            "time": [datetime(2020, 1, 11), datetime(2020, 1, 12)],
            "value": [15.5, 16.5],
        })

        mase = MeanAbsoluteScaledError(seasonality=2)
        mase.fit(y_train)
        score = mase.score(y_true, y_pred)

        assert isinstance(score, float)
        assert score > 0

    def test_mase_requires_fit(self):
        """MASE should require fit() before score()."""
        from yohou.metrics import MeanAbsoluteScaledError

        y_true = pl.DataFrame({
            "time": [datetime(2020, 1, 1)],
            "value": [10.0],
        })
        y_pred = pl.DataFrame({
            "vintage_time": [datetime(2019, 12, 31)],
            "time": [datetime(2020, 1, 1)],
            "value": [12.0],
        })

        mase = MeanAbsoluteScaledError(seasonality=2)

        with pytest.raises(NotFittedError, match="fitted"):
            mase.score(y_true, y_pred)

    def test_mase_requires_training_data(self):
        """MASE should raise error if fit() called with None."""
        from yohou.metrics import MeanAbsoluteScaledError

        mase = MeanAbsoluteScaledError(seasonality=2)

        with pytest.raises(ValueError, match="`y_train` is required for scorer.fit"):
            mase.fit(y_train=None)

    def test_mase_seasonality_too_large_error(self):
        """MASE should raise error if seasonality > training length - 1."""
        from yohou.metrics import MeanAbsoluteScaledError

        y_train = pl.DataFrame({
            "time": [datetime(2020, 1, 1), datetime(2020, 1, 2), datetime(2020, 1, 3)],
            "value": [10.0, 20.0, 30.0],
        })

        mase = MeanAbsoluteScaledError(seasonality=5)

        with pytest.raises(ValueError, match="Training data length.*must be greater than seasonality"):
            mase.fit(y_train)

    def test_mase_scale_independent(self):
        """MASE should be scale-independent."""
        from yohou.metrics import MeanAbsoluteScaledError

        y_train_1 = pl.DataFrame({
            "time": [datetime(2020, 1, 1) + timedelta(days=i) for i in range(10)],
            "value": [10.0, 12.0, 11.0, 13.0, 12.0, 14.0, 13.0, 15.0, 14.0, 16.0],
        })
        y_true_1 = pl.DataFrame({
            "time": [datetime(2020, 1, 11), datetime(2020, 1, 12)],
            "value": [15.0, 17.0],
        })
        y_pred_1 = pl.DataFrame({
            "vintage_time": [datetime(2020, 1, 10)] * 2,
            "time": [datetime(2020, 1, 11), datetime(2020, 1, 12)],
            "value": [15.5, 16.5],
        })

        # Scale by 100x
        y_train_2 = y_train_1.with_columns(value=pl.col("value") * 100)
        y_true_2 = y_true_1.with_columns(value=pl.col("value") * 100)
        y_pred_2 = y_pred_1.with_columns(value=pl.col("value") * 100)

        mase = MeanAbsoluteScaledError(seasonality=2)
        mase.fit(y_train_1)
        score_1 = mase.score(y_true_1, y_pred_1)

        mase.fit(y_train_2)
        score_2 = mase.score(y_true_2, y_pred_2)

        # Scores should be identical (scale-independent)
        np.testing.assert_allclose(score_1, score_2, rtol=1e-5)


class TestMedianAbsoluteError:
    def test_median_ae_basic_computation(self):
        """MedianAbsoluteError should compute median of absolute errors."""
        from yohou.metrics import MedianAbsoluteError

        y_true = pl.DataFrame({
            "time": [datetime(2020, 1, 1), datetime(2020, 1, 2), datetime(2020, 1, 3)],
            "value": [10.0, 20.0, 30.0],
        })
        y_pred = pl.DataFrame({
            "vintage_time": [datetime(2019, 12, 31)] * 3,
            "time": [datetime(2020, 1, 1), datetime(2020, 1, 2), datetime(2020, 1, 3)],
            "value": [12.0, 19.0, 28.0],
        })

        medae = MedianAbsoluteError()
        medae.fit(y_true)
        score = medae.score(y_true, y_pred)

        # Errors: [2, 1, 2], median = 2
        assert isinstance(score, float)
        assert score == 2.0

    def test_median_ae_robust_to_outliers(self):
        """MedianAbsoluteError should be robust to outliers."""
        from yohou.metrics import MedianAbsoluteError

        y_true = pl.DataFrame({
            "time": [datetime(2020, 1, 1) + timedelta(days=i) for i in range(5)],
            "value": [10.0, 20.0, 30.0, 40.0, 50.0],
        })
        # One large outlier
        y_pred = pl.DataFrame({
            "vintage_time": [datetime(2019, 12, 31)] * 5,
            "time": [datetime(2020, 1, 1) + timedelta(days=i) for i in range(5)],
            "value": [11.0, 21.0, 31.0, 41.0, 100.0],  # Last one is outlier
        })

        medae = MedianAbsoluteError()
        medae.fit(y_true)
        score_median = medae.score(y_true, y_pred)

        mae = MeanAbsoluteError()
        mae.fit(y_true)
        score_mean = mae.score(y_true, y_pred)

        # Median should be less affected by outlier than mean
        # Errors: [1, 1, 1, 1, 50], median=1, mean=10.8
        assert score_median < score_mean


class TestRMSEStepwiseVintagewise:
    """Tests for RMSE with stepwise+vintagewise aggregation and sqrt."""

    def test_rmse_stepwise_vintagewise_returns_per_component(self, scorer_data_factory):
        """RMSE with stepwise+vintagewise agg returns per-component DataFrame with sqrt applied."""
        y, _ = scorer_data_factory(length=20, n_targets=3, n_features=0, seed=42)
        y_test = y[15:]
        y_pred = y_test.with_columns([(pl.col(c) + 2.0).alias(c) for c in y_test.columns if c != "time"]).with_columns(
            vintage_time=pl.lit(y["time"][14])
        )

        rmse = RootMeanSquaredError(aggregation_method=["stepwise", "vintagewise"])
        rmse.fit(y[:15])
        result = rmse.score(y_test, y_pred)

        assert isinstance(result, pl.DataFrame)
        assert result.shape == (1, 3)
        for col in result.columns:
            assert result[col][0] == pytest.approx(2.0, abs=1e-5)

    def test_rmsse_stepwise_vintagewise_returns_per_component(self):
        """RMSSE with stepwise+vintagewise agg returns per-component DataFrame with sqrt applied."""
        y_train = pl.DataFrame({
            "time": [datetime(2020, 1, 1) + timedelta(days=i) for i in range(10)],
            "value": [10.0, 12.0, 11.0, 13.0, 12.0, 14.0, 13.0, 15.0, 14.0, 16.0],
        })
        y_test = pl.DataFrame({
            "time": [datetime(2020, 1, 11), datetime(2020, 1, 12)],
            "value": [15.0, 17.0],
        })
        y_pred = pl.DataFrame({
            "vintage_time": [datetime(2020, 1, 10)] * 2,
            "time": [datetime(2020, 1, 11), datetime(2020, 1, 12)],
            "value": [15.5, 16.5],
        })

        rmsse = RootMeanSquaredScaledError(seasonality=2, aggregation_method=["stepwise", "vintagewise"])
        rmsse.fit(y_train)
        result = rmsse.score(y_test, y_pred)

        assert isinstance(result, pl.DataFrame)
        assert result.shape == (1, 1)
        assert result.columns[0] == "value"
        assert result["value"][0] > 0


class TestMAETimeWeight:
    """Tests for MAE score with time_weight parameter."""

    def test_mae_callable_time_weight(self):
        """MAE with callable time_weight applies weights to per-timestep errors."""
        y_true = pl.DataFrame({
            "time": [datetime(2020, 1, 1) + timedelta(days=i) for i in range(5)],
            "value": [10.0, 20.0, 30.0, 40.0, 50.0],
        })
        y_pred = pl.DataFrame({
            "vintage_time": [datetime(2019, 12, 31)] * 5,
            "time": [datetime(2020, 1, 1) + timedelta(days=i) for i in range(5)],
            "value": [11.0, 21.0, 31.0, 41.0, 51.0],
        })
        mae = MeanAbsoluteError(time_weighter=LookupWeighter(mapping={}, default=1.0))
        mae.fit(y_true)
        score = mae.score(y_true, y_pred)
        assert score == pytest.approx(1.0, abs=1e-5)

    def test_mae_dataframe_time_weight(self):
        """MAE with DataFrame time_weight joins on time column."""
        times = [datetime(2020, 1, 1) + timedelta(days=i) for i in range(3)]
        y_true = pl.DataFrame({
            "time": times,
            "value": [10.0, 20.0, 30.0],
        })
        y_pred = pl.DataFrame({
            "vintage_time": [datetime(2019, 12, 31)] * 3,
            "time": times,
            "value": [12.0, 19.0, 28.0],
        })
        weight_df = pl.DataFrame({
            "time": times,
            "weight": [1.0, 1.0, 1.0],
        })
        mae = MeanAbsoluteError(time_weighter=TableWeighter(frame=weight_df, on="time"))
        mae.fit(y_true)
        score = mae.score(y_true, y_pred)
        assert isinstance(score, float)
        assert score > 0


class TestMaxAbsoluteErrorSystematic:
    def test_max_ae_systematic(self, y_true_y_pred):
        """Run systematic checks for MaxAbsoluteError."""
        y_truth, y_pred = y_true_y_pred
        scorer = MaxAbsoluteError()
        run_checks(scorer, y_truth, y_pred)


class TestMaxAbsoluteError:
    def test_max_ae_basic_computation(self, y_true_y_pred):
        """MaxAbsoluteError should return the largest absolute error."""
        y_true, y_pred = y_true_y_pred
        # Errors: |10-12|=2, |20-19|=1, |30-28|=2
        max_ae = MaxAbsoluteError()
        max_ae.fit(y_true)
        score = max_ae.score(y_true, y_pred)
        assert score == 2.0

    def test_max_ae_large_outlier(self):
        """MaxAbsoluteError should capture the outlier error."""
        y_true = pl.DataFrame({
            "time": [datetime(2020, 1, 1) + timedelta(days=i) for i in range(5)],
            "value": [10.0, 20.0, 30.0, 40.0, 50.0],
        })
        y_pred = pl.DataFrame({
            "vintage_time": [datetime(2019, 12, 31)] * 5,
            "time": [datetime(2020, 1, 1) + timedelta(days=i) for i in range(5)],
            "value": [11.0, 21.0, 31.0, 41.0, 100.0],
        })

        max_ae = MaxAbsoluteError()
        max_ae.fit(y_true)
        score = max_ae.score(y_true, y_pred)

        # Max error is |50 - 100| = 50
        assert score == 50.0

    def test_max_ae_perfect_prediction(self):
        """MaxAbsoluteError should be zero for perfect predictions."""
        y_true = pl.DataFrame({
            "time": [datetime(2020, 1, 1), datetime(2020, 1, 2)],
            "value": [10.0, 20.0],
        })
        y_pred = pl.DataFrame({
            "vintage_time": [datetime(2019, 12, 31)] * 2,
            "time": [datetime(2020, 1, 1), datetime(2020, 1, 2)],
            "value": [10.0, 20.0],
        })

        max_ae = MaxAbsoluteError()
        max_ae.fit(y_true)
        score = max_ae.score(y_true, y_pred)
        assert np.isclose(score, 0.0)

    def test_max_ae_vs_mae(self, y_true_y_pred):
        """MaxAbsoluteError should always be >= MAE."""
        y_true, y_pred = y_true_y_pred

        max_ae = MaxAbsoluteError()
        max_ae.fit(y_true)
        max_score = max_ae.score(y_true, y_pred)

        mae = MeanAbsoluteError()
        mae.fit(y_true)
        mae_score = mae.score(y_true, y_pred)

        assert max_score >= mae_score

    def test_max_ae_componentwise(self, y_true_y_pred):
        """MaxAbsoluteError with componentwise returns per-timestep max."""
        y_true, y_pred = y_true_y_pred
        max_ae = MaxAbsoluteError(aggregation_method=["componentwise"])
        max_ae.fit(y_true)
        result = max_ae.score(y_true, y_pred)

        assert isinstance(result, pl.DataFrame)
        assert "time" in result.columns
        assert "max_ae" in result.columns

    def test_max_ae_stepwise_vintagewise(self, y_true_y_pred):
        """MaxAbsoluteError with stepwise+vintagewise returns per-component max."""
        y_true, y_pred = y_true_y_pred
        max_ae = MaxAbsoluteError(aggregation_method=["stepwise", "vintagewise"])
        max_ae.fit(y_true)
        result = max_ae.score(y_true, y_pred)

        assert isinstance(result, pl.DataFrame)
        assert result.shape == (1, 1)
        # Max of |2, 1, 2| = 2
        assert result["value"][0] == 2.0


class TestR2ScoreSystematic:
    def test_r2_systematic(self, y_true_y_pred):
        """Run systematic checks for R2Score."""
        y_truth, y_pred = y_true_y_pred
        scorer = R2Score()
        run_checks(scorer, y_truth, y_pred)


class TestR2Score:
    def test_r2_perfect_prediction(self):
        """R2 should be 1.0 for perfect predictions."""
        y_true = pl.DataFrame({
            "time": [datetime(2020, 1, 1), datetime(2020, 1, 2), datetime(2020, 1, 3)],
            "value": [10.0, 20.0, 30.0],
        })
        y_pred = pl.DataFrame({
            "vintage_time": [datetime(2019, 12, 31)] * 3,
            "time": [datetime(2020, 1, 1), datetime(2020, 1, 2), datetime(2020, 1, 3)],
            "value": [10.0, 20.0, 30.0],
        })
        r2 = R2Score()
        r2.fit(y_true)
        score = r2.score(y_true, y_pred)
        assert np.isclose(score, 1.0)

    def test_r2_mean_prediction(self):
        """R2 should be 0.0 when predictions equal the mean."""
        y_true = pl.DataFrame({
            "time": [datetime(2020, 1, 1), datetime(2020, 1, 2), datetime(2020, 1, 3)],
            "value": [10.0, 20.0, 30.0],
        })
        mean_val = 20.0
        y_pred = pl.DataFrame({
            "vintage_time": [datetime(2019, 12, 31)] * 3,
            "time": [datetime(2020, 1, 1), datetime(2020, 1, 2), datetime(2020, 1, 3)],
            "value": [mean_val, mean_val, mean_val],
        })
        r2 = R2Score()
        r2.fit(y_true)
        score = r2.score(y_true, y_pred)
        assert np.isclose(score, 0.0)

    def test_r2_worse_than_mean(self):
        """R2 should be negative when predictions are worse than the mean."""
        y_true = pl.DataFrame({
            "time": [datetime(2020, 1, 1), datetime(2020, 1, 2), datetime(2020, 1, 3)],
            "value": [10.0, 20.0, 30.0],
        })
        y_pred = pl.DataFrame({
            "vintage_time": [datetime(2019, 12, 31)] * 3,
            "time": [datetime(2020, 1, 1), datetime(2020, 1, 2), datetime(2020, 1, 3)],
            "value": [30.0, 10.0, 10.0],
        })
        r2 = R2Score()
        r2.fit(y_true)
        score = r2.score(y_true, y_pred)
        assert score < 0

    def test_r2_constant_true_values(self):
        """R2 should return 0.0 when true values are constant (SS_tot=0)."""
        y_true = pl.DataFrame({
            "time": [datetime(2020, 1, 1), datetime(2020, 1, 2), datetime(2020, 1, 3)],
            "value": [20.0, 20.0, 20.0],
        })
        y_pred = pl.DataFrame({
            "vintage_time": [datetime(2019, 12, 31)] * 3,
            "time": [datetime(2020, 1, 1), datetime(2020, 1, 2), datetime(2020, 1, 3)],
            "value": [19.0, 21.0, 20.0],
        })
        r2 = R2Score()
        r2.fit(y_true)
        score = r2.score(y_true, y_pred)
        assert score == 0.0

    def test_r2_higher_is_better(self):
        """R2Score should have lower_is_better=False."""
        r2 = R2Score()
        tags = r2.__sklearn_tags__()
        assert tags.scorer_tags.lower_is_better is False

    def test_r2_multiple_columns(self, scorer_data_factory):
        """R2 should work with multiple target columns."""
        y, _ = scorer_data_factory(length=50, n_targets=3, seed=42)
        y_pred = y.clone()
        y_pred = y_pred.with_columns([
            (pl.col(col) + np.random.randn(50) * 0.5).alias(col) for col in y_pred.columns if col != "time"
        ])
        y_pred = y_pred.with_columns(vintage_time=pl.lit(y["time"][0]))

        r2 = R2Score()
        r2.fit(y)
        score = r2.score(y, y_pred)
        assert isinstance(score, float)
        # Small noise → high R2
        assert score > 0.9

    def test_r2_stepwise_vintagewise(self, y_true_y_pred):
        """R2 with stepwise+vintagewise returns per-component DataFrame."""
        y_true, y_pred = y_true_y_pred
        r2 = R2Score(aggregation_method=["stepwise", "vintagewise"])
        r2.fit(y_true)
        result = r2.score(y_true, y_pred)

        assert isinstance(result, pl.DataFrame)
        assert result.shape == (1, 1)


class TestMDASystematic:
    def test_mda_systematic(self):
        """Run systematic checks for MeanDirectionalAccuracy."""
        y_truth = pl.DataFrame({
            "time": [datetime(2020, 1, 1) + timedelta(days=i) for i in range(10)],
            "value": np.cumsum(np.random.randn(10)).tolist(),
        })
        y_pred = y_truth.clone()
        y_pred = y_pred.with_columns((pl.col("value") + np.random.randn(10) * 0.1).alias("value"))
        y_pred = y_pred.with_columns(vintage_time=pl.lit(y_truth["time"][0]))
        scorer = MeanDirectionalAccuracy()
        run_checks(scorer, y_truth, y_pred)


class TestMDA:
    def test_mda_perfect_direction(self):
        """MDA should be 1.0 when all directions match."""
        y_true = pl.DataFrame({
            "time": [datetime(2020, 1, i) for i in range(1, 6)],
            "value": [10.0, 15.0, 12.0, 18.0, 20.0],
        })
        y_pred = pl.DataFrame({
            "vintage_time": [datetime(2019, 12, 31)] * 5,
            "time": [datetime(2020, 1, i) for i in range(1, 6)],
            "value": [10.0, 16.0, 11.0, 19.0, 21.0],
        })
        mda = MeanDirectionalAccuracy()
        mda.fit(y_true)
        score = mda.score(y_true, y_pred)
        assert score == 1.0

    def test_mda_all_wrong(self):
        """MDA should be 0.0 when all directions are wrong."""
        y_true = pl.DataFrame({
            "time": [datetime(2020, 1, i) for i in range(1, 6)],
            "value": [10.0, 15.0, 12.0, 18.0, 20.0],
        })
        # Reverse every direction
        y_pred = pl.DataFrame({
            "vintage_time": [datetime(2019, 12, 31)] * 5,
            "time": [datetime(2020, 1, i) for i in range(1, 6)],
            "value": [10.0, 8.0, 13.0, 11.0, 9.0],
        })
        mda = MeanDirectionalAccuracy()
        mda.fit(y_true)
        score = mda.score(y_true, y_pred)
        assert score == 0.0

    def test_mda_partial(self):
        """MDA should return correct fraction for partial match."""
        y_true = pl.DataFrame({
            "time": [datetime(2020, 1, i) for i in range(1, 6)],
            "value": [10.0, 15.0, 12.0, 18.0, 20.0],
        })
        # diffs: +5, -3, +6, +2
        y_pred = pl.DataFrame({
            "vintage_time": [datetime(2019, 12, 31)] * 5,
            "time": [datetime(2020, 1, i) for i in range(1, 6)],
            "value": [10.0, 14.0, 15.0, 17.0, 19.0],
        })
        # pred diffs: +4, +1, +2, +2  → match: [T, F, T, T] = 3/4
        mda = MeanDirectionalAccuracy()
        mda.fit(y_true)
        score = mda.score(y_true, y_pred)
        assert score == 0.75

    def test_mda_too_few_rows(self):
        """MDA should return 0.0 with fewer than 2 rows."""
        y_true = pl.DataFrame({
            "time": [datetime(2020, 1, 1)],
            "value": [10.0],
        })
        y_pred = pl.DataFrame({
            "vintage_time": [datetime(2019, 12, 31)],
            "time": [datetime(2020, 1, 1)],
            "value": [12.0],
        })
        mda = MeanDirectionalAccuracy()
        mda.fit(y_true)
        score = mda.score(y_true, y_pred)
        assert score == 0.0

    def test_mda_higher_is_better(self):
        """MDA should have lower_is_better=False."""
        mda = MeanDirectionalAccuracy()
        tags = mda.__sklearn_tags__()
        assert tags.scorer_tags.lower_is_better is False

    def test_mda_multiple_columns(self, scorer_data_factory):
        """MDA should work with multiple target columns."""
        y, _ = scorer_data_factory(length=50, n_targets=2, seed=42)
        y_pred = y.clone()
        y_pred = y_pred.with_columns([
            (pl.col(col) + np.random.randn(50) * 0.1).alias(col) for col in y_pred.columns if col != "time"
        ])
        y_pred = y_pred.with_columns(vintage_time=pl.lit(y["time"][0]))

        mda = MeanDirectionalAccuracy()
        mda.fit(y)
        score = mda.score(y, y_pred)
        assert isinstance(score, float)
        assert 0 <= score <= 1

    def test_mda_stepwise_vintagewise(self):
        """MDA with stepwise+vintagewise returns per-component DataFrame."""
        y_true = pl.DataFrame({
            "time": [datetime(2020, 1, i) for i in range(1, 6)],
            "value": [10.0, 15.0, 12.0, 18.0, 20.0],
        })
        y_pred = pl.DataFrame({
            "vintage_time": [datetime(2019, 12, 31)] * 5,
            "time": [datetime(2020, 1, i) for i in range(1, 6)],
            "value": [10.0, 16.0, 11.0, 19.0, 21.0],
        })
        mda = MeanDirectionalAccuracy(aggregation_method=["stepwise", "vintagewise"])
        mda.fit(y_true)
        result = mda.score(y_true, y_pred)
        assert isinstance(result, pl.DataFrame)
        assert result.shape == (1, 1)


class TestMaxAEPartialCollapse:
    def test_max_ae_stepwise_only(self):
        """MaxAbsoluteError with stepwise only exercises partial collapse."""
        y_true = pl.DataFrame({
            "time": [datetime(2020, 1, i) for i in range(1, 4)],
            "value": [10.0, 20.0, 30.0],
        })
        y_pred = pl.DataFrame({
            "vintage_time": [datetime(2019, 12, 31)] * 3,
            "time": [datetime(2020, 1, i) for i in range(1, 4)],
            "value": [12.0, 19.0, 28.0],
        })
        max_ae = MaxAbsoluteError(aggregation_method=["stepwise", "componentwise"])
        max_ae.fit(y_true)
        result = max_ae.score(y_true, y_pred)
        assert isinstance(result, (float, pl.DataFrame))

    def test_max_ae_vintagewise_only(self):
        """MaxAbsoluteError with vintagewise only exercises partial collapse."""
        y_true = pl.DataFrame({
            "time": [datetime(2020, 1, i) for i in range(1, 4)],
            "value": [10.0, 20.0, 30.0],
        })
        y_pred = pl.DataFrame({
            "vintage_time": [datetime(2019, 12, 31)] * 3,
            "time": [datetime(2020, 1, i) for i in range(1, 4)],
            "value": [12.0, 19.0, 28.0],
        })
        max_ae = MaxAbsoluteError(aggregation_method=["vintagewise", "componentwise"])
        max_ae.fit(y_true)
        result = max_ae.score(y_true, y_pred)
        assert isinstance(result, (float, pl.DataFrame))

    def test_max_ae_collapse_all_rows(self):
        """MaxAbsoluteError with stepwise+vintagewise collapses all rows."""
        y_true = pl.DataFrame({
            "time": [datetime(2020, 1, i) for i in range(1, 4)],
            "a": [10.0, 20.0, 30.0],
            "b": [15.0, 25.0, 35.0],
        })
        y_pred = pl.DataFrame({
            "vintage_time": [datetime(2019, 12, 31)] * 3,
            "time": [datetime(2020, 1, i) for i in range(1, 4)],
            "a": [12.0, 19.0, 28.0],
            "b": [14.0, 24.0, 34.0],
        })
        max_ae = MaxAbsoluteError(aggregation_method=["stepwise", "vintagewise"])
        max_ae.fit(y_true)
        result = max_ae.score(y_true, y_pred)
        assert isinstance(result, pl.DataFrame)


class TestMedianAEComponentwise:
    def test_median_ae_componentwise(self):
        """MedianAbsoluteError with componentwise returns per-timestep scores."""
        y_true = pl.DataFrame({
            "time": [datetime(2020, 1, i) for i in range(1, 4)],
            "a": [10.0, 20.0, 30.0],
            "b": [15.0, 25.0, 35.0],
        })
        y_pred = pl.DataFrame({
            "vintage_time": [datetime(2019, 12, 31)] * 3,
            "time": [datetime(2020, 1, i) for i in range(1, 4)],
            "a": [12.0, 19.0, 28.0],
            "b": [14.0, 24.0, 34.0],
        })
        medae = MedianAbsoluteError(aggregation_method=["componentwise"])
        medae.fit(y_true)
        result = medae.score(y_true, y_pred)
        assert isinstance(result, pl.DataFrame)
        assert "time" in result.columns

    def test_median_ae_per_component(self):
        """MedianAbsoluteError with stepwise+vintagewise returns per-component medians."""
        y_true = pl.DataFrame({
            "time": [datetime(2020, 1, i) for i in range(1, 4)],
            "a": [10.0, 20.0, 30.0],
            "b": [15.0, 25.0, 35.0],
        })
        y_pred = pl.DataFrame({
            "vintage_time": [datetime(2019, 12, 31)] * 3,
            "time": [datetime(2020, 1, i) for i in range(1, 4)],
            "a": [12.0, 19.0, 28.0],
            "b": [14.0, 24.0, 34.0],
        })
        medae = MedianAbsoluteError(aggregation_method=["stepwise", "vintagewise"])
        medae.fit(y_true)
        result = medae.score(y_true, y_pred)
        assert isinstance(result, pl.DataFrame)


class TestScoreOverrideStubs:
    def test_r2_compute_raw_errors_not_implemented(self):
        """R2Score fully overrides score(); _compute_raw_errors is unreachable."""
        y_true = pl.DataFrame({"value": [10.0, 20.0, 30.0]})
        y_pred = pl.DataFrame({"value": [12.0, 19.0, 28.0]})
        r2 = R2Score()
        with pytest.raises(NotImplementedError):
            r2._compute_raw_errors(y_true, y_pred)

    def test_mda_compute_raw_errors_not_implemented(self):
        """MDA fully overrides score(); _compute_raw_errors is unreachable."""
        y_true = pl.DataFrame({"value": [10.0, 20.0, 30.0]})
        y_pred = pl.DataFrame({"value": [12.0, 19.0, 28.0]})
        mda = MeanDirectionalAccuracy()
        with pytest.raises(NotImplementedError):
            mda._compute_raw_errors(y_true, y_pred)


class TestMapPerVintageEdgeCases:
    """Cover _map_per_vintage None return paths."""

    def test_mda_single_row_returns_zero(self):
        """MDA returns 0.0 for single-row data."""
        times = [datetime(2020, 1, 1)]
        y_true = pl.DataFrame({"time": times, "value": [10.0]})
        y_pred = pl.DataFrame({
            "vintage_time": [datetime(2019, 12, 31)],
            "time": times,
            "value": [12.0],
        })
        scorer = MeanDirectionalAccuracy()
        scorer.fit(y_true)
        score = scorer.score(y_true, y_pred)
        assert score == 0.0

    def test_mda_multi_vintage_one_skipped(self):
        """MDA skips vintages with <2 rows."""
        times_3 = [datetime(2020, 1, 1) + timedelta(days=i) for i in range(3)]
        vt1 = datetime(2019, 12, 31)
        vt2 = datetime(2019, 12, 30)

        y_true = pl.DataFrame({"time": times_3, "value": [10.0, 20.0, 30.0]})
        y_pred = pl.DataFrame({
            "vintage_time": [vt1] * 3 + [vt2],
            "time": times_3 + [times_3[0]],
            "value": [12.0, 22.0, 28.0, 11.0],
        })
        scorer = MeanDirectionalAccuracy()
        scorer.fit(y_true)
        score = scorer.score(y_true, y_pred)
        assert isinstance(score, float)

    def test_mda_all_vintages_skipped_raises(self):
        """All vintages skipped raises ValueError."""
        t1 = datetime(2020, 1, 1)
        t2 = datetime(2020, 1, 2)
        vt1 = datetime(2019, 12, 31)
        vt2 = datetime(2019, 12, 30)

        y_true = pl.DataFrame({"time": [t1, t2], "value": [10.0, 20.0]})
        y_pred = pl.DataFrame({
            "vintage_time": [vt1, vt2],
            "time": [t1, t2],
            "value": [12.0, 21.0],
        })
        scorer = MeanDirectionalAccuracy()
        scorer.fit(y_true)
        with pytest.raises(ValueError, match="All vintage groups were skipped"):
            scorer.score(y_true, y_pred)


class TestScaledMetricColumnMismatch:
    """Regression tests for clear errors when score-time columns are not fitted."""

    def test_rmsse_unfitted_column_raises_value_error(self):
        """RMSSE raises a descriptive ValueError for columns not seen during fit()."""
        y_train = pl.DataFrame({
            "time": [datetime(2020, 1, 1) + timedelta(days=i) for i in range(5)],
            "value": [1.0, 2.0, 3.0, 4.0, 5.0],
        })
        scorer = RootMeanSquaredScaledError(seasonality=1)
        scorer.fit(y_train)

        y_true = pl.DataFrame({"time": [datetime(2020, 1, 6)], "other": [6.0]})
        y_pred = pl.DataFrame({
            "vintage_time": [datetime(2020, 1, 5)],
            "time": [datetime(2020, 1, 6)],
            "other": [5.5],
        })
        with pytest.raises(ValueError, match="not seen during fit"):
            scorer.score(y_true, y_pred)

    def test_mase_unfitted_column_raises_value_error(self):
        """MASE raises a descriptive ValueError for columns not seen during fit()."""
        y_train = pl.DataFrame({
            "time": [datetime(2020, 1, 1) + timedelta(days=i) for i in range(5)],
            "value": [1.0, 2.0, 3.0, 4.0, 5.0],
        })
        scorer = MeanAbsoluteScaledError(seasonality=1)
        scorer.fit(y_train)

        y_true = pl.DataFrame({"time": [datetime(2020, 1, 6)], "other": [6.0]})
        y_pred = pl.DataFrame({
            "vintage_time": [datetime(2020, 1, 5)],
            "time": [datetime(2020, 1, 6)],
            "other": [5.5],
        })
        with pytest.raises(ValueError, match="not seen during fit"):
            scorer.score(y_true, y_pred)


class TestMDAShortInputAggregation:
    """Regression tests for MDA short-input handling across aggregation methods."""

    def test_mda_short_input_scalar_aggregation_returns_zero(self):
        """Fully collapsed aggregation returns 0.0 for too-short input."""
        y_true = pl.DataFrame({"time": [datetime(2020, 1, 1)], "value": [10.0]})
        y_pred = pl.DataFrame({
            "vintage_time": [datetime(2019, 12, 31)],
            "time": [datetime(2020, 1, 1)],
            "value": [12.0],
        })
        scorer = MeanDirectionalAccuracy(aggregation_method="all")
        scorer.fit(y_true)
        assert scorer.score(y_true, y_pred) == 0.0

    def test_mda_short_input_dataframe_aggregation_raises(self):
        """DataFrame-shaped aggregation raises instead of returning a bare scalar."""
        y_true = pl.DataFrame({"time": [datetime(2020, 1, 1)], "value": [10.0]})
        y_pred = pl.DataFrame({
            "vintage_time": [datetime(2019, 12, 31)],
            "time": [datetime(2020, 1, 1)],
            "value": [12.0],
        })
        scorer = MeanDirectionalAccuracy(aggregation_method="componentwise")
        scorer.fit(y_true)
        with pytest.raises(ValueError, match="at least 2 rows"):
            scorer.score(y_true, y_pred)
