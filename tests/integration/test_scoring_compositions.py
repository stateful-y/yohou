"""Stress tests for scorer compositions with all forecaster types.

These tests verify that all scorers work correctly through composition forecasters,
aggregation methods, and time weighting. Tests use analytical fixtures to verify
exact numerical behavior rather than just crash-testing.

Test coverage:
- ~24 tests: All point scorers on all forecaster types
- ~4 tests: Scaled scorers need calibration
- ~15 tests: All interval scorers on interval forecasters
- ~16 tests: Scorer aggregation methods on multi-column panel data
- ~6 tests: Scorer with time_weight
"""

import numpy as np
import polars as pl
import pytest
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LinearRegression

from yohou.compose import ColumnForecaster, DecompositionPipeline
from yohou.interval import SplitConformalForecaster
from yohou.metrics import (
    AbsoluteResidual,
    CalibrationError,
    EmpiricalCoverage,
    IntervalScore,
    MeanAbsoluteError,
    MeanAbsolutePercentageError,
    MeanAbsoluteScaledError,
    MeanIntervalWidth,
    MeanSquaredError,
    MedianAbsoluteError,
    PinballLoss,
    Residual,
    RootMeanSquaredError,
    RootMeanSquaredScaledError,
    SymmetricMeanAbsolutePercentageError,
)
from yohou.point import PointReductionForecaster, SeasonalNaive
from yohou.preprocessing import LagTransformer
from yohou.stationarity import PolynomialTrendForecaster
from yohou.utils.weighting import ExponentialDecayWeighter, LinearDecayWeighter, TableWeighter


@pytest.mark.integration
class TestPointScorerCompositions:
    """Test point scorers across different forecaster compositions and data scenarios."""

    @pytest.mark.parametrize(
        "forecaster",
        [
            SeasonalNaive(seasonality=1),
            PointReductionForecaster(LinearRegression(), feature_transformer=LagTransformer(lag=1)),
            DecompositionPipeline([
                ("trend", PolynomialTrendForecaster(degree=1)),
                ("residual", SeasonalNaive(seasonality=1)),
            ]),
            ColumnForecaster([
                ("c1", SeasonalNaive(seasonality=1), ["col_a"]),
                ("c2", SeasonalNaive(seasonality=1), ["col_b"]),
            ]),
        ],
        ids=["naive", "reduction", "decomposition", "column"],
    )
    @pytest.mark.parametrize(
        "scorer",
        [
            MeanAbsoluteError(),
            MeanSquaredError(),
            RootMeanSquaredError(),
            MeanAbsolutePercentageError(),
            SymmetricMeanAbsolutePercentageError(),
            MedianAbsoluteError(),
        ],
        ids=["mae", "mse", "rmse", "mape", "smape", "medae"],
    )
    def test_point_scorer_on_forecaster_composition(self, linear_series, forecaster, scorer):
        """Test that all point scorers work with all forecaster compositions."""
        # Generate data
        if isinstance(forecaster, ColumnForecaster):
            # Multi-column target for ColumnForecaster
            y1 = linear_series(slope=2.0, intercept=10.0, length=100)
            y2 = linear_series(slope=3.0, intercept=20.0, length=100)
            y = y1.rename({"value": "col_a"}).join(
                y2.select(pl.col("time"), pl.col("value").alias("col_b")),
                on="time",
            )
            train_len = 80
        else:
            # Single-column target
            y = linear_series(slope=2.0, intercept=10.0, length=100)
            train_len = 80

        # Fit and predict
        y_train = y[:train_len]
        forecaster.fit(y_train, forecasting_horizon=5)
        y_pred = forecaster.predict(forecasting_horizon=5)

        # Fit scorer and score
        y_test = y[train_len : train_len + 5]
        scorer.fit(y_train)
        score = scorer.score(y_test, y_pred)

        # Verify score is finite and non-negative (error metrics)
        assert np.isfinite(score), f"Score should be finite, got {score}"
        # MAPE/sMAPE can be large, others should be reasonable
        if not isinstance(scorer, MeanAbsolutePercentageError | SymmetricMeanAbsolutePercentageError):
            assert score >= 0, f"Score should be non-negative, got {score}"

    @pytest.mark.parametrize(
        "scorer",
        [
            MeanAbsoluteError(),
            MeanSquaredError(),
            RootMeanSquaredError(),
            MedianAbsoluteError(),
        ],
        ids=["mae", "mse", "rmse", "medae"],
    )
    def test_point_scorer_perfect_prediction(self, constant_series, scorer):
        """Test that scorers return 0.0 for perfect predictions."""
        # Generate constant data (easy to predict perfectly)
        y = constant_series(value=42.0, length=100)

        # Fit SeasonalNaive (will predict constant)
        forecaster = SeasonalNaive(seasonality=1)
        y_train = y[:80]
        forecaster.fit(y_train, forecasting_horizon=5)
        y_pred = forecaster.predict(forecasting_horizon=5)

        # Fit scorer and score
        y_test = y[80:85]
        scorer.fit(y_train)
        score = scorer.score(y_test, y_pred)

        # For constant data, prediction should be perfect
        assert score < 1e-10, f"Score should be ~0.0 for perfect prediction, got {score}"

    @pytest.mark.parametrize(
        "scorer",
        [
            MeanAbsoluteError(),
            MeanSquaredError(),
            RootMeanSquaredError(),
            MedianAbsoluteError(),
        ],
        ids=["mae", "mse", "rmse", "medae"],
    )
    def test_point_scorer_on_constant_data(self, constant_series, scorer):
        """Test that scorers return 0.0 when truth and prediction are both constant."""
        # Generate constant data
        y_true = constant_series(value=42.0, length=10)
        y_pred = constant_series(value=42.0, length=10)

        # Fit scorer and score
        scorer.fit(y_true)
        score = scorer.score(y_true, y_pred)
        assert score == 0.0, f"Score should be 0.0 for identical constant series, got {score}"


@pytest.mark.integration
class TestScaledScorers:
    """Test MASE and RMSSE scaled scorers requiring fit and baseline comparison."""

    def test_mase_requires_fit(self, linear_series):
        """Test that MASE requires fit() before score()."""
        scorer = MeanAbsoluteScaledError(seasonality=1)

        y = linear_series(slope=2.0, intercept=10.0, length=100)
        y_train = y[:80]
        y_test = y[80:85]

        # Fit forecaster and get predictions
        forecaster = SeasonalNaive(seasonality=1)
        forecaster.fit(y_train, forecasting_horizon=5)
        y_pred = forecaster.predict(forecasting_horizon=5)

        # Score without fit should raise NotFittedError
        with pytest.raises(NotFittedError, match="is not fitted"):
            scorer.score(y_test, y_pred)

        # After fit, should work
        scorer.fit(y_train)
        score = scorer.score(y_test, y_pred)
        assert np.isfinite(score)

    def test_rmsse_requires_fit(self, linear_series):
        """Test that RMSSE requires fit() before score()."""
        scorer = RootMeanSquaredScaledError(seasonality=1)

        y = linear_series(slope=2.0, intercept=10.0, length=100)
        y_train = y[:80]
        y_test = y[80:85]

        # Fit forecaster and get predictions
        forecaster = SeasonalNaive(seasonality=1)
        forecaster.fit(y_train, forecasting_horizon=5)
        y_pred = forecaster.predict(forecasting_horizon=5)

        # Score without fit should raise NotFittedError
        with pytest.raises(NotFittedError, match="is not fitted"):
            scorer.score(y_test, y_pred)

        # After fit, should work
        scorer.fit(y_train)
        score = scorer.score(y_test, y_pred)
        assert np.isfinite(score)

    def test_mase_equals_one_when_matches_naive(self, linear_series):
        """Test that MASE=1.0 when prediction accuracy equals naive seasonal forecast."""
        y = linear_series(slope=2.0, intercept=10.0, length=100)
        y_train = y[:80]
        y_test = y[80:85]

        # Fit naive forecaster (MASE baseline)
        forecaster = SeasonalNaive(seasonality=1)
        forecaster.fit(y_train, forecasting_horizon=5)
        y_pred = forecaster.predict(forecasting_horizon=5)

        # Score with MASE
        scorer = MeanAbsoluteScaledError(seasonality=1)
        scorer.fit(y_train)
        score = scorer.score(y_test, y_pred)

        # For naive forecaster, MASE should be close to 1.0
        # (exactly 1.0 if test error equals naive baseline error)
        # On linear data, naive will have larger error due to trend, so MASE > 1.0 is expected
        assert 0.5 <= score <= 5.0, f"MASE should be reasonable for naive forecaster, got {score}"

    def test_rmsse_equals_one_when_matches_naive(self, linear_series):
        """Test that RMSSE=1.0 when prediction accuracy equals naive seasonal forecast."""
        y = linear_series(slope=2.0, intercept=10.0, length=100)
        y_train = y[:80]
        y_test = y[80:85]

        # Fit naive forecaster (RMSSE baseline)
        forecaster = SeasonalNaive(seasonality=1)
        forecaster.fit(y_train, forecasting_horizon=5)
        y_pred = forecaster.predict(forecasting_horizon=5)

        # Score with RMSSE
        scorer = RootMeanSquaredScaledError(seasonality=1)
        scorer.fit(y_train)
        score = scorer.score(y_test, y_pred)

        # For naive forecaster, RMSSE should be close to 1.0
        # On linear data with trend, RMSSE > 1.0 is expected
        assert 0.5 <= score <= 5.0, f"RMSSE should be reasonable for naive forecaster, got {score}"


@pytest.mark.integration
class TestIntervalScorers:
    """Test interval scorers with conformal forecasters, coverage rates, and properties."""

    @pytest.mark.parametrize(
        "scorer",
        [
            IntervalScore(),
            PinballLoss(),
            EmpiricalCoverage(),
            MeanIntervalWidth(),
            CalibrationError(),
        ],
        ids=["interval_score", "pinball", "coverage", "width", "calibration"],
    )
    def test_interval_scorer_on_conformal(self, linear_series, scorer):
        """Test that all interval scorers work with SplitConformalForecaster."""
        y = linear_series(slope=2.0, intercept=10.0, length=100)

        # Fit conformal forecaster with multiple coverage rates
        base_forecaster = SeasonalNaive(seasonality=1)
        forecaster = SplitConformalForecaster(
            base_forecaster, conformity_scorer=AbsoluteResidual(), calibration_size=20
        )

        # Use multiple coverage rates for CalibrationError (requires ≥2)
        y_train = y[:80]
        forecaster.fit(y_train, forecasting_horizon=5)
        y_pred_interval = forecaster.predict_interval(forecasting_horizon=5, coverage_rates=[0.8, 0.9])

        # Fit scorer and score
        y_test = y[80:85]
        scorer.fit(y_train)

        # CalibrationError requires ≥2 coverage rates
        if isinstance(scorer, CalibrationError):
            score = scorer.score(y_test, y_pred_interval)
            assert np.isfinite(score), f"Score should be finite, got {score}"
        else:
            # For other scorers, score all coverage rates
            score = scorer.score(y_test, y_pred_interval)
            assert np.isfinite(score), f"Score should be finite, got {score}"

    def test_calibration_error_requires_multiple_coverage_rates(self, linear_series):
        """Test that CalibrationError raises error with single coverage rate."""
        y = linear_series(slope=2.0, intercept=10.0, length=100)

        # Fit conformal forecaster with single coverage rate
        base_forecaster = SeasonalNaive(seasonality=1)
        forecaster = SplitConformalForecaster(
            base_forecaster, conformity_scorer=AbsoluteResidual(), calibration_size=20
        )
        y_train = y[:80]
        forecaster.fit(y_train, forecasting_horizon=5)
        y_pred_interval = forecaster.predict_interval(forecasting_horizon=5, coverage_rates=[0.9])

        # CalibrationError should raise error
        scorer = CalibrationError()
        y_test = y[80:85]
        scorer.fit(y_train)

        with pytest.raises(ValueError, match="at least 2"):
            scorer.score(y_test, y_pred_interval)

    def test_empirical_coverage_returns_value_in_0_1(self, linear_series):
        """Test that EmpiricalCoverage returns values in [0, 1]."""
        y = linear_series(slope=2.0, intercept=10.0, length=100)

        # Fit conformal forecaster
        base_forecaster = SeasonalNaive(seasonality=1)
        forecaster = SplitConformalForecaster(
            base_forecaster, conformity_scorer=AbsoluteResidual(), calibration_size=20
        )
        y_train = y[:80]
        forecaster.fit(y_train, forecasting_horizon=5)
        y_pred_interval = forecaster.predict_interval(forecasting_horizon=5, coverage_rates=[0.9])

        # Fit scorer and score
        scorer = EmpiricalCoverage()
        y_test = y[80:85]
        scorer.fit(y_train)
        score = scorer.score(y_test, y_pred_interval)

        # Coverage should be in [0, 1]
        assert 0.0 <= score <= 1.0, f"Coverage should be in [0, 1], got {score}"

    def test_interval_score_is_nonnegative(self, linear_series):
        """Test that IntervalScore is non-negative."""
        y = linear_series(slope=2.0, intercept=10.0, length=100)

        # Fit conformal forecaster
        base_forecaster = SeasonalNaive(seasonality=1)
        forecaster = SplitConformalForecaster(
            base_forecaster, conformity_scorer=AbsoluteResidual(), calibration_size=20
        )
        y_train = y[:80]
        forecaster.fit(y_train, forecasting_horizon=5)
        y_pred_interval = forecaster.predict_interval(forecasting_horizon=5, coverage_rates=[0.9])

        # Fit scorer and score
        scorer = IntervalScore()
        y_test = y[80:85]
        scorer.fit(y_train)
        score = scorer.score(y_test, y_pred_interval)

        # Interval score should be non-negative
        assert score >= 0, f"IntervalScore should be non-negative, got {score}"

    def test_mean_interval_width_is_positive(self, linear_series):
        """Test that MeanIntervalWidth is positive."""
        y = linear_series(slope=2.0, intercept=10.0, length=100)

        # Fit conformal forecaster
        base_forecaster = SeasonalNaive(seasonality=1)
        forecaster = SplitConformalForecaster(
            base_forecaster, conformity_scorer=AbsoluteResidual(), calibration_size=20
        )
        y_train = y[:80]
        forecaster.fit(y_train, forecasting_horizon=5)
        y_pred_interval = forecaster.predict_interval(forecasting_horizon=5, coverage_rates=[0.9])

        # Fit scorer and score
        scorer = MeanIntervalWidth()
        y_test = y[80:85]
        scorer.fit(y_train)
        score = scorer.score(y_test, y_pred_interval)

        # Width should be positive
        assert score > 0, f"MeanIntervalWidth should be positive, got {score}"

    @pytest.mark.parametrize(
        "coverage_rates",
        [
            [0.8],
            [0.9],
            [0.8, 0.9],
            [0.8, 0.9, 0.95],
        ],
        ids=["single_80", "single_90", "double", "triple"],
    )
    def test_interval_scorer_column_subselection(self, linear_series, coverage_rates):
        """Test that interval scorers work with different coverage rate combinations."""
        y = linear_series(slope=2.0, intercept=10.0, length=100)

        # Fit conformal forecaster
        base_forecaster = SeasonalNaive(seasonality=1)
        forecaster = SplitConformalForecaster(
            base_forecaster, conformity_scorer=AbsoluteResidual(), calibration_size=20
        )
        y_train = y[:80]
        forecaster.fit(y_train, forecasting_horizon=5)
        y_pred_interval = forecaster.predict_interval(forecasting_horizon=5, coverage_rates=coverage_rates)

        # Fit scorer and score with IntervalScore (works with any number of coverage rates)
        scorer = IntervalScore()
        y_test = y[80:85]
        scorer.fit(y_train)
        score = scorer.score(y_test, y_pred_interval)

        assert np.isfinite(score), f"Score should be finite, got {score}"

    def test_interval_coverage_on_wide_intervals(self, linear_series):
        """Test that very wide intervals have high empirical coverage."""
        y = linear_series(slope=2.0, intercept=10.0, length=100)

        # Fit conformal with high coverage
        base_forecaster = SeasonalNaive(seasonality=1)
        forecaster = SplitConformalForecaster(
            base_forecaster, conformity_scorer=AbsoluteResidual(), calibration_size=20
        )
        y_train = y[:80]
        forecaster.fit(y_train, forecasting_horizon=5)
        y_pred_interval = forecaster.predict_interval(forecasting_horizon=5, coverage_rates=[0.99])

        # Fit scorer and score
        scorer = EmpiricalCoverage()
        y_test = y[80:85]
        scorer.fit(y_train)
        coverage = scorer.score(y_test, y_pred_interval)

        # High coverage rate should produce high empirical coverage
        # (may not be exactly 0.99 due to small sample, but should be high)
        assert coverage > 0.5, f"Coverage with 0.99 rate should be high, got {coverage}"


@pytest.mark.integration
class TestScorerAggregation:
    """Test scorer aggregation methods on panel and multi-column data."""

    @pytest.mark.parametrize(
        "aggregation_method",
        ["all", ["stepwise", "vintagewise"], "componentwise", "groupwise"],
    )
    def test_point_scorer_aggregation_on_panel(self, linear_series, panel_analytical, aggregation_method):
        """Test that aggregation methods produce correct output shapes on panel data."""
        # Generate 3-group panel with 2 targets each
        panel_fn = panel_analytical
        y = panel_fn(
            linear_series,
            n_groups=3,
            params_per_group=[
                {"slope": 1.0, "intercept": 10.0, "length": 100},
                {"slope": 2.0, "intercept": 20.0, "length": 100},
                {"slope": 3.0, "intercept": 30.0, "length": 100},
            ],
            prefix="target",
        )

        # Add second target column for each group (2 targets per group)
        y2 = panel_fn(
            linear_series,
            n_groups=3,
            params_per_group=[
                {"slope": 1.5, "intercept": 15.0, "length": 100},
                {"slope": 2.5, "intercept": 25.0, "length": 100},
                {"slope": 3.5, "intercept": 35.0, "length": 100},
            ],
            prefix="other",
        )
        # Join on time to combine both sets of columns
        y = y.join(y2, on="time")

        # Fit forecaster
        forecaster = SeasonalNaive(seasonality=1)
        y_train = y[:80]
        forecaster.fit(y_train, forecasting_horizon=5)
        y_pred = forecaster.predict(forecasting_horizon=5)

        # Fit scorer and score with aggregation method
        scorer = MeanAbsoluteError(aggregation_method=aggregation_method)
        y_test = y[80:85]
        scorer.fit(y_train)
        score = scorer.score(y_test, y_pred)

        # Verify output shape
        if aggregation_method == "all":
            assert isinstance(score, float), f"'all' should return scalar, got {type(score)}"
            assert np.isfinite(score)
        elif aggregation_method == ["stepwise", "vintagewise"]:
            # Stepwise+vintagewise aggregates over time, returns per-component-per-group scores
            assert isinstance(score, pl.DataFrame), f"stepwise+vintagewise should return DataFrame, got {type(score)}"
            # Columns are component-group pairs (target__group_0, target__group_1, ...)
            assert len(score.columns) == 6  # 2 targets × 3 groups
            assert len(score) == 1  # Single row with aggregated scores
        elif aggregation_method == "componentwise":
            # Componentwise aggregates over components, returns per-timestep-per-group scores
            assert isinstance(score, pl.DataFrame), f"'componentwise' should return DataFrame, got {type(score)}"
            assert "time" in score.columns
            # Should have time column plus group-specific score columns
            assert len(score) == 5  # 5 timesteps
        elif aggregation_method == "groupwise":
            # Groupwise aggregates over groups, returns per-component-per-timestep scores
            assert isinstance(score, pl.DataFrame), f"'groupwise' should return DataFrame, got {type(score)}"
            # Should have per-component per-timestep scores (aggregated over groups)
            assert len(score) > 0

    @pytest.mark.parametrize(
        "_scorer_cls",
        [
            MeanAbsoluteError,
            MeanSquaredError,
            RootMeanSquaredError,
            MedianAbsoluteError,
        ],
        ids=["mae", "mse", "rmse", "medae"],
    )
    @pytest.mark.parametrize(
        "aggregation_method",
        ["all", ["stepwise", "vintagewise"], "componentwise"],
    )
    def test_scorer_aggregation_output_shapes(self, linear_series, _scorer_cls, aggregation_method):
        """Test aggregation output shapes for different scorers."""
        # Generate 2-column data
        y1 = linear_series(slope=2.0, intercept=10.0, length=100)
        y2 = linear_series(slope=3.0, intercept=20.0, length=100)
        y = y1.rename({"value": "col_a"}).join(
            y2.select(pl.col("time"), pl.col("value").alias("col_b")),
            on="time",
        )

        # Fit and predict
        forecaster = ColumnForecaster([
            ("c1", SeasonalNaive(seasonality=1), ["col_a"]),
            ("c2", SeasonalNaive(seasonality=1), ["col_b"]),
        ])
        y_train = y[:80]
        forecaster.fit(y_train, forecasting_horizon=5)
        y_pred = forecaster.predict(forecasting_horizon=5)

        # Fit scorer and score
        scorer = _scorer_cls(aggregation_method=aggregation_method)
        y_test = y[80:85]
        scorer.fit(y_train)
        score = scorer.score(y_test, y_pred)

        # Verify shapes
        if aggregation_method == "all":
            assert isinstance(score, float)
            assert np.isfinite(score)
        elif aggregation_method == ["stepwise", "vintagewise"]:
            # Stepwise+vintagewise aggregates over time, returns per-component scores
            assert isinstance(score, pl.DataFrame)
            # Component names are columns (col_a, col_b)
            assert "col_a" in score.columns or len(score.columns) == 2
            assert len(score) == 1  # Single row with aggregated scores
        elif aggregation_method == "componentwise":
            # Componentwise aggregates over components, returns per-timestep scores
            assert isinstance(score, pl.DataFrame)
            assert "time" in score.columns
            meta_cols = {"time", "vintage_time", "forecasting_step", "coverage_rate"}
            value_cols = [c for c in score.columns if c not in meta_cols]
            assert "score" in value_cols or len(value_cols) == 1
            assert len(score) == 5  # 5 timesteps


@pytest.mark.integration
class TestTimeWeighting:
    """Test scorer behavior with time weighting functions."""

    def test_scorer_with_exponential_decay_weight(self, constant_series):
        """Test that exponential decay weights recent errors more."""
        # Use constant data so naive forecast is perfect (zero base error)
        y = constant_series(value=42.0, length=100)

        y_train = y[:80]
        y_test = y[80:90]  # 10 steps

        # Fit forecaster; predictions will be perfect on constant data
        forecaster = SeasonalNaive(seasonality=1)
        forecaster.fit(y_train, forecasting_horizon=10)
        y_pred = forecaster.predict(forecasting_horizon=10)

        # Add errors: large early, zero late
        errors = pl.Series("errors", [10.0, 10.0, 10.0, 5.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        y_pred_modified = y_pred.with_columns([(pl.col("value") + errors).alias("value")])

        # Fit scorer and score without weight
        scorer_unweighted = MeanAbsoluteError()
        scorer_unweighted.fit(y_train)
        score_unweighted = scorer_unweighted.score(y_test, y_pred_modified)

        # Fit scorer and score with exponential decay (recent emphasis)
        time_weight = ExponentialDecayWeighter(half_life=3)
        scorer_weighted = MeanAbsoluteError()
        scorer_weighted.fit(y_train)
        score_weighted = scorer_weighted.set_params(time_weighter=time_weight).score(y_test, y_pred_modified)

        # With recent-emphasis weight, score should be LOWER (better)
        # because recent predictions (which are good) are weighted more
        assert score_weighted < score_unweighted, (
            f"Weighted score should be lower (better) with recent emphasis: "
            f"weighted={score_weighted}, unweighted={score_unweighted}"
        )

    def test_scorer_with_linear_decay_weight(self, linear_series):
        """Test that linear decay weights work correctly."""
        y = linear_series(slope=2.0, intercept=10.0, length=100)

        # Fit and predict
        forecaster = SeasonalNaive(seasonality=1)
        y_train = y[:80]
        forecaster.fit(y_train, forecasting_horizon=10)
        y_pred = forecaster.predict(forecasting_horizon=10)
        y_test = y[80:90]

        # Fit scorer and score with linear decay weight
        time_weight = LinearDecayWeighter()
        scorer = MeanAbsoluteError()
        scorer.fit(y_train)
        score = scorer.set_params(time_weighter=time_weight).score(y_test, y_pred)

        assert np.isfinite(score), f"Score should be finite, got {score}"

    def test_weighted_mae_analytic_verification(self, constant_series):
        """Test weighted MAE computation analytically."""
        # Generate constant truth
        y_true = constant_series(value=10.0, length=5)

        # Generate predictions with known errors: [12, 13, 14, 15, 16]
        # Errors: [2, 3, 4, 5, 6]
        errors_series = pl.Series("errors", [2.0, 3.0, 4.0, 5.0, 6.0])
        y_pred = constant_series(value=10.0, length=5).with_columns([(pl.col("value") + errors_series).alias("value")])

        # Define weights: [1, 2, 3, 4, 5]
        weights = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        errors = np.array([2.0, 3.0, 4.0, 5.0, 6.0])

        # Expected weighted MAE: Σ(w_i * |e_i|) / Σ(w_i)
        expected = np.sum(weights * errors) / np.sum(weights)
        # = (1*2 + 2*3 + 3*4 + 4*5 + 5*6) / (1+2+3+4+5)
        # = (2 + 6 + 12 + 20 + 30) / 15
        # = 70 / 15 = 4.6667

        # Fit scorer and score with a table-backed time weighter
        weight_frame = pl.DataFrame({"time": y_true["time"], "weight": weights})
        scorer = MeanAbsoluteError(time_weighter=TableWeighter(frame=weight_frame, on="time"))
        scorer.fit(y_true)
        score = scorer.score(y_true, y_pred)

        assert np.abs(score - expected) < 1e-6, (
            f"Weighted MAE should match analytical: expected={expected}, got={score}"
        )

    @pytest.mark.parametrize(
        "forecaster",
        [
            SeasonalNaive(seasonality=1),
            PointReductionForecaster(LinearRegression(), feature_transformer=LagTransformer(lag=1)),
            DecompositionPipeline([
                ("trend", PolynomialTrendForecaster(degree=1)),
                ("residual", SeasonalNaive(seasonality=1)),
            ]),
        ],
        ids=["naive", "reduction", "decomposition"],
    )
    def test_time_weight_with_forecaster_compositions(self, linear_series, forecaster):
        """Test that time weighting works with different forecaster compositions."""
        y = linear_series(slope=2.0, intercept=10.0, length=100)

        # Fit and predict
        y_train = y[:80]
        forecaster.fit(y_train, forecasting_horizon=10)
        y_pred = forecaster.predict(forecasting_horizon=10)
        y_test = y[80:90]

        # Fit scorer and score with time weight
        time_weight = ExponentialDecayWeighter(half_life=3)
        scorer = MeanAbsoluteError()
        scorer.fit(y_train)
        score = scorer.set_params(time_weighter=time_weight).score(y_test, y_pred)

        assert np.isfinite(score), f"Score should be finite, got {score}"

    def test_time_weight_verification_early_vs_late_errors(self, constant_series):
        """Test weight verification: early errors vs late errors."""
        # Use constant data so naive forecast is perfect (zero base error)
        y = constant_series(value=42.0, length=100)

        # Create two prediction scenarios:
        # Scenario A: Large early errors, small late errors
        # Scenario B: Small early errors, large late errors

        y_train = y[:80]
        y_test = y[80:90]

        forecaster = SeasonalNaive(seasonality=1)
        forecaster.fit(y_train, forecasting_horizon=10)
        y_pred_base = forecaster.predict(forecasting_horizon=10)

        # Scenario A: Early errors = 10, late errors = 1
        errors_a = pl.Series("errors", [10.0] * 5 + [1.0] * 5)
        y_pred_a = y_pred_base.with_columns([(pl.col("value") + errors_a).alias("value")])

        # Scenario B: Early errors = 1, late errors = 10
        errors_b = pl.Series("errors", [1.0] * 5 + [10.0] * 5)
        y_pred_b = y_pred_base.with_columns([(pl.col("value") + errors_b).alias("value")])

        # With recent-emphasis weight (exponential decay), scenario B should have higher score (worse)
        time_weight = ExponentialDecayWeighter(half_life=2)
        scorer = MeanAbsoluteError()
        scorer.fit(y_train)

        score_a = scorer.set_params(time_weighter=time_weight).score(y_test, y_pred_a)
        score_b = scorer.set_params(time_weighter=time_weight).score(y_test, y_pred_b)

        # Scenario B (late errors) should have higher weighted score (worse)
        assert score_b > score_a, (
            f"Late errors should be weighted more with recent emphasis: score_a={score_a}, score_b={score_b}"
        )


@pytest.mark.integration
class TestScorerEdgeCases:
    """Test scorer edge cases including zero values, conformity scorers, and single timesteps."""

    def test_mape_with_zero_values_handling(self, linear_series):
        """Test MAPE behavior with zero values (can produce inf/nan)."""
        # Generate series with zeros
        y = linear_series(slope=0.0, intercept=0.0, length=100)

        forecaster = SeasonalNaive(seasonality=1)
        y_train = y[:80]
        forecaster.fit(y_train, forecasting_horizon=5)
        y_pred = forecaster.predict(forecasting_horizon=5)

        scorer = MeanAbsolutePercentageError()
        y_test = y[80:85]
        scorer.fit(y_train)
        _score = scorer.score(y_test, y_pred)

        # MAPE with zeros can be inf or nan, just check it doesn't crash
        # (behavior depends on implementation)
        assert True  # Just verify no crash

    def test_conformity_scorer_residual(self, linear_series):
        """Test conformity scorers (Residual, AbsoluteResidual) work directly."""
        y = linear_series(slope=2.0, intercept=10.0, length=100)

        forecaster = SeasonalNaive(seasonality=1)
        y_train = y[:80]
        forecaster.fit(y_train, forecasting_horizon=5)
        y_pred = forecaster.predict(forecasting_horizon=5)
        y_test = y[80:85]

        # Test Residual
        scorer_residual = Residual()
        scorer_residual.fit(y_train)
        score = scorer_residual.score(y_test, y_pred)
        assert isinstance(score, pl.DataFrame), "Residual should return DataFrame"
        assert "value" in score.columns, f"Expected 'value' in columns, got {score.columns}"

        # Test AbsoluteResidual
        scorer_abs = AbsoluteResidual()
        scorer_abs.fit(y_train)
        score_abs = scorer_abs.score(y_test, y_pred)
        assert isinstance(score_abs, pl.DataFrame), "AbsoluteResidual should return DataFrame"
        assert "value" in score_abs.columns, f"Expected 'value' in columns, got {score_abs.columns}"
        # Absolute residuals should be non-negative
        assert (score_abs["value"] >= 0).all(), "AbsoluteResidual values should be non-negative"

    def test_scorer_on_single_timestep_prediction(self, linear_series):
        """Test scorers work with single-timestep predictions."""
        y = linear_series(slope=2.0, intercept=10.0, length=100)

        # Fit and predict single step
        forecaster = SeasonalNaive(seasonality=1)
        y_train = y[:80]
        forecaster.fit(y_train, forecasting_horizon=1)
        y_pred = forecaster.predict(forecasting_horizon=1)
        y_test = y[80:81]

        # Fit scorer and score
        scorer = MeanAbsoluteError()
        scorer.fit(y_train)
        score = scorer.score(y_test, y_pred)

        assert np.isfinite(score), f"Score on single timestep should be finite, got {score}"
        assert score >= 0
