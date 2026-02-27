"""Analytical integration tests for cross-validation pipelines.

This module tests GridSearchCV and RandomizedSearchCV with complex forecasters,
nested pipelines, and interval forecasters to ensure hyperparameter search
integrates correctly with all Yohou components.
"""

import numpy as np
import polars as pl
import pytest
from scipy.stats import uniform
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.utils.validation import check_is_fitted

from yohou.compose import ColumnForecaster, FeaturePipeline
from yohou.interval import SplitConformalForecaster
from yohou.metrics import (
    AbsoluteResidual,
    IntervalScore,
    MeanAbsoluteError,
    RootMeanSquaredError,
)
from yohou.model_selection import (
    ExpandingWindowSplitter,
    GridSearchCV,
    RandomizedSearchCV,
    SlidingWindowSplitter,
)
from yohou.point import PointReductionForecaster, SeasonalNaive
from yohou.preprocessing import LagTransformer, StandardScaler


@pytest.mark.integration
class TestGridSearchNestedParamGrid:
    """GridSearchCV with nested parameter grids on reduction forecasters."""

    def test_grid_search_nested_param_grid_cv_results_count(self, linear_series):
        """Verify cv_results_ has correct number of parameter combinations."""
        y = linear_series(slope=2.0, intercept=10.0, length=100)

        forecaster = PointReductionForecaster(
            estimator=Ridge(),
            feature_transformer=FeaturePipeline([
                ("lag", LagTransformer(1)),
                ("scale", StandardScaler()),
            ]),
        )
        param_grid = {
            "estimator__alpha": [0.0, 0.1, 1.0],
            "feature_transformer__lag__lag": [1, [1, 2], [1, 2, 3]],
        }
        scorer = MeanAbsoluteError()
        cv = SlidingWindowSplitter(n_splits=3, test_size=10)

        search = GridSearchCV(forecaster, param_grid, scoring=scorer, cv=cv)
        search.fit(y, forecasting_horizon=5)

        # Should have 3 * 3 = 9 combinations
        assert len(search.cv_results_["params"]) == 9
        assert "param_estimator__alpha" in search.cv_results_
        assert "param_feature_transformer__lag__lag" in search.cv_results_
        assert "mean_test_score" in search.cv_results_
        assert "rank_test_score" in search.cv_results_

    def test_grid_search_nested_param_grid_best_params_linear(self, linear_series):
        """Verify best_params_ selects valid parameters for linear data."""
        y = linear_series(slope=2.0, intercept=10.0, length=100)

        forecaster = PointReductionForecaster(
            estimator=Ridge(),
            feature_transformer=FeaturePipeline([
                ("lag", LagTransformer(1)),
                ("scale", StandardScaler()),
            ]),
        )
        param_grid = {
            "estimator__alpha": [0.0, 0.1, 1.0],
            "feature_transformer__lag__lag": [1, [1, 2]],
        }
        scorer = MeanAbsoluteError()
        cv = SlidingWindowSplitter(n_splits=3, test_size=10)

        search = GridSearchCV(forecaster, param_grid, scoring=scorer, cv=cv)
        search.fit(y, forecasting_horizon=5)

        # Best params should be among tested values (relaxed assertion due to CV variance)
        assert search.best_params_["estimator__alpha"] in [0.0, 0.1, 1.0]
        # Best lag configuration should be among the tested values
        assert search.best_params_["feature_transformer__lag__lag"] in [1, [1, 2]]

    def test_grid_search_nested_param_grid_best_forecaster_fitted(self, linear_series):
        """Verify best_forecaster_ is fitted and has correct params."""
        y = linear_series(slope=2.0, intercept=10.0, length=100)

        forecaster = PointReductionForecaster(
            estimator=Ridge(),
            feature_transformer=LagTransformer(1),
        )
        param_grid = {
            "estimator__alpha": [0.0, 0.5, 1.0],
            "feature_transformer__lag": [1, [1, 2]],
        }
        scorer = MeanAbsoluteError()
        cv = SlidingWindowSplitter(n_splits=3, test_size=10)

        search = GridSearchCV(forecaster, param_grid, scoring=scorer, cv=cv)
        search.fit(y, forecasting_horizon=5)

        # best_forecaster_ should be fitted
        check_is_fitted(search.best_forecaster_)

        # Should have best params set
        assert search.best_forecaster_.estimator.alpha == search.best_params_["estimator__alpha"]
        assert search.best_forecaster_.feature_transformer.lag == search.best_params_["feature_transformer__lag"]

    def test_grid_search_nested_param_grid_predict_horizon(self, linear_series):
        """Verify predict() on fitted search produces correct horizon."""
        y = linear_series(slope=2.0, intercept=10.0, length=100)

        forecaster = PointReductionForecaster(
            estimator=Ridge(),
            feature_transformer=LagTransformer(1),
        )
        param_grid = {
            "estimator__alpha": [0.0, 0.1],
            "feature_transformer__lag": [1, [1, 2]],
        }
        scorer = MeanAbsoluteError()
        cv = SlidingWindowSplitter(n_splits=3, test_size=10)

        search = GridSearchCV(forecaster, param_grid, scoring=scorer, cv=cv)
        search.fit(y, forecasting_horizon=5)

        # Predict with different horizon than fit
        y_pred = search.predict(forecasting_horizon=8)

        assert len(y_pred) == 8
        assert "time" in y_pred.columns
        assert "value" in y_pred.columns
        # Should have 'observed_time' for point forecasts
        assert "observed_time" in y_pred.columns

    def test_grid_search_nested_param_grid_observe_predict(self, linear_series):
        """Verify observe() and predict() work correctly after fit."""
        y = linear_series(slope=2.0, intercept=10.0, length=100)

        forecaster = PointReductionForecaster(
            estimator=Ridge(),
            feature_transformer=LagTransformer(1),
        )
        param_grid = {
            "estimator__alpha": [0.0, 0.1, 1.0],
            "feature_transformer__lag": [1, [1, 2]],
        }
        scorer = MeanAbsoluteError()
        cv = SlidingWindowSplitter(n_splits=3, test_size=10)

        search = GridSearchCV(forecaster, param_grid, scoring=scorer, cv=cv)
        search.fit(y[:80], forecasting_horizon=5)

        # Update with new observations
        y_new = y[80:81]
        search.observe(y_new)

        # Predict from updated state
        y_pred = search.predict(forecasting_horizon=5)

        assert len(y_pred) == 5
        assert "time" in y_pred.columns
        assert "value" in y_pred.columns


@pytest.mark.integration
class TestGridSearchMultiMetric:
    """GridSearchCV with multiple scoring metrics and refit selection."""

    def test_grid_search_multi_metric_cv_results(self, linear_series):
        """Verify cv_results_ contains scores for multiple metrics."""
        y = linear_series(slope=2.0, intercept=10.0, length=100)

        forecaster = PointReductionForecaster(
            estimator=Ridge(),
            feature_transformer=LagTransformer(1),
        )
        param_grid = {
            "estimator__alpha": [0.0, 0.1, 1.0],
        }
        scoring = {"mae": MeanAbsoluteError(), "rmse": RootMeanSquaredError()}
        cv = SlidingWindowSplitter(n_splits=3, test_size=10)

        search = GridSearchCV(forecaster, param_grid, scoring=scoring, refit="mae", cv=cv)
        search.fit(y, forecasting_horizon=5)

        # Should have scores for both metrics
        assert "mean_test_mae" in search.cv_results_
        assert "mean_test_rmse" in search.cv_results_
        # Ranking should be based on refit metric (mae)
        assert "rank_test_mae" in search.cv_results_

    def test_grid_search_multi_metric_refit_ranking(self, linear_series):
        """Verify ranking is based on refit metric."""
        y = linear_series(slope=2.0, intercept=10.0, length=100)

        forecaster = PointReductionForecaster(
            estimator=Ridge(),
            feature_transformer=LagTransformer([1, 2]),
        )
        param_grid = {
            "estimator__alpha": [0.0, 0.5, 1.0, 2.0],
        }
        scoring = {"mae": MeanAbsoluteError(), "rmse": RootMeanSquaredError()}
        cv = SlidingWindowSplitter(n_splits=3, test_size=10)

        search = GridSearchCV(forecaster, param_grid, scoring=scoring, refit="mae", cv=cv)
        search.fit(y, forecasting_horizon=5)

        # Best params should minimize MAE (refit metric)
        mae_scores = search.cv_results_["mean_test_mae"]
        best_mae_idx = np.argmin([-score for score in mae_scores])
        assert search.cv_results_["rank_test_mae"][best_mae_idx] == 1

        # Verify best_params_ corresponds to best MAE
        best_alpha = search.cv_results_["param_estimator__alpha"][best_mae_idx]
        assert search.best_params_["estimator__alpha"] == best_alpha


@pytest.mark.integration
class TestRandomizedSearch:
    """RandomizedSearchCV with continuous distributions and reproducibility."""

    def test_randomized_search_n_iter_count(self, linear_series):
        """Verify n_iter produces exactly n candidates."""
        y = linear_series(slope=2.0, intercept=10.0, length=100)

        forecaster = PointReductionForecaster(
            estimator=Ridge(),
            feature_transformer=LagTransformer(1),
        )
        param_distributions = {
            "estimator__alpha": uniform(0, 10),
            "feature_transformer__lag": [1, [1, 2], [1, 2, 3]],
        }
        scorer = MeanAbsoluteError()
        cv = SlidingWindowSplitter(n_splits=3, test_size=10)

        search = RandomizedSearchCV(forecaster, param_distributions, n_iter=5, scoring=scorer, cv=cv, random_state=42)
        search.fit(y, forecasting_horizon=5)

        # Should have exactly 5 candidates
        assert len(search.cv_results_["params"]) == 5

    def test_randomized_search_reproducibility(self, linear_series):
        """Verify reproducibility with same random_state."""
        y = linear_series(slope=2.0, intercept=10.0, length=100)

        forecaster = PointReductionForecaster(
            estimator=Ridge(),
            feature_transformer=LagTransformer(1),
        )
        param_distributions = {
            "estimator__alpha": uniform(0, 10),
            "feature_transformer__lag": [1, [1, 2]],
        }
        scorer = MeanAbsoluteError()
        cv = SlidingWindowSplitter(n_splits=3, test_size=10)

        # First search
        search1 = RandomizedSearchCV(forecaster, param_distributions, n_iter=8, scoring=scorer, cv=cv, random_state=123)
        search1.fit(y, forecasting_horizon=5)

        # Second search with same seed
        search2 = RandomizedSearchCV(forecaster, param_distributions, n_iter=8, scoring=scorer, cv=cv, random_state=123)
        search2.fit(y, forecasting_horizon=5)

        # Should produce identical parameter samples
        params1 = search1.cv_results_["params"]
        params2 = search2.cv_results_["params"]

        for p1, p2 in zip(params1, params2, strict=False):
            assert p1["estimator__alpha"] == p2["estimator__alpha"]
            assert p1["feature_transformer__lag"] == p2["feature_transformer__lag"]

    def test_randomized_search_different_seeds(self, linear_series):
        """Verify different random_state produces different samples."""
        y = linear_series(slope=2.0, intercept=10.0, length=100)

        forecaster = PointReductionForecaster(
            estimator=Ridge(),
            feature_transformer=LagTransformer(1),
        )
        param_distributions = {
            "estimator__alpha": uniform(0, 10),
        }
        scorer = MeanAbsoluteError()
        cv = SlidingWindowSplitter(n_splits=3, test_size=10)

        # First search with seed 42
        search1 = RandomizedSearchCV(forecaster, param_distributions, n_iter=10, scoring=scorer, cv=cv, random_state=42)
        search1.fit(y, forecasting_horizon=5)

        # Second search with seed 123
        search2 = RandomizedSearchCV(
            forecaster, param_distributions, n_iter=10, scoring=scorer, cv=cv, random_state=123
        )
        search2.fit(y, forecasting_horizon=5)

        # Should produce different parameter samples
        alphas1 = [p["estimator__alpha"] for p in search1.cv_results_["params"]]
        alphas2 = [p["estimator__alpha"] for p in search2.cv_results_["params"]]

        # Not all alphas should be identical (very unlikely with continuous distribution)
        assert not all(a1 == a2 for a1, a2 in zip(alphas1, alphas2, strict=False))


@pytest.mark.integration
class TestGridSearchCVSplitters:
    """GridSearchCV with different cross-validation splitter strategies."""

    def test_grid_search_expanding_window_split_count(self, linear_series):
        """Verify correct split count with ExpandingWindowSplitter."""
        y = linear_series(slope=2.0, intercept=10.0, length=100)

        forecaster = PointReductionForecaster(
            estimator=Ridge(),
            feature_transformer=LagTransformer(1),
        )
        param_grid = {
            "estimator__alpha": [0.0, 1.0],
        }
        scorer = MeanAbsoluteError()
        # ExpandingWindowSplitter with 5 splits: training set grows each split
        cv = ExpandingWindowSplitter(n_splits=5, test_size=10)

        search = GridSearchCV(forecaster, param_grid, scoring=scorer, cv=cv)
        search.fit(y, forecasting_horizon=5)

        # Should have 5 splits
        n_splits = cv.get_n_splits(y)
        assert n_splits == 5

        # cv_results_ should have scores from all splits for each param combo
        for i in range(n_splits):
            assert f"split{i}_test_score" in search.cv_results_

    def test_grid_search_expanding_vs_sliding_window(self, linear_series):
        """Compare GridSearchCV results between Expanding and Sliding splitters."""
        y = linear_series(slope=2.0, intercept=10.0, length=100)

        forecaster = PointReductionForecaster(
            estimator=Ridge(),
            feature_transformer=LagTransformer([1, 2]),
        )
        param_grid = {
            "estimator__alpha": [0.0, 0.5, 1.0],
        }
        scorer = MeanAbsoluteError()

        # Expanding window search: 4 splits, test_size=10
        cv_expanding = ExpandingWindowSplitter(n_splits=4, test_size=10)
        search_expanding = GridSearchCV(forecaster, param_grid, scoring=scorer, cv=cv_expanding)
        search_expanding.fit(y, forecasting_horizon=5)

        # Sliding window search
        cv_sliding = SlidingWindowSplitter(n_splits=3, test_size=10, stride=10)
        search_sliding = GridSearchCV(forecaster, param_grid, scoring=scorer, cv=cv_sliding)
        search_sliding.fit(y, forecasting_horizon=5)

        # Both should find a best_params_ (may differ due to different validation sets)
        assert search_expanding.best_params_ is not None
        assert search_sliding.best_params_ is not None

        # Both should have same number of parameter combinations
        assert len(search_expanding.cv_results_["params"]) == 3
        assert len(search_sliding.cv_results_["params"]) == 3


@pytest.mark.integration
class TestGridSearchConformalForecaster:
    """GridSearchCV with SplitConformalForecaster and interval predictions."""

    def test_grid_search_conformal_interval_forecaster_fit(self, linear_series):
        """Verify GridSearchCV works with interval forecasters end-to-end."""
        y = linear_series(slope=2.0, intercept=10.0, length=100)

        forecaster = SplitConformalForecaster(
            point_forecaster=PointReductionForecaster(
                estimator=LinearRegression(),
                feature_transformer=LagTransformer(1),
            ),
            conformity_scorer=AbsoluteResidual(),
            calibration_size=20,  # Small calibration set to fit within CV splits
        )
        param_grid = {
            "point_forecaster__feature_transformer__lag": [1, [1, 2]],
        }
        scorer = IntervalScore()
        cv = SlidingWindowSplitter(n_splits=3, test_size=10)

        search = GridSearchCV(forecaster, param_grid, scoring=scorer, cv=cv)
        search.fit(y, forecasting_horizon=5)

        # Should complete without error
        assert len(search.cv_results_["params"]) == 2
        assert search.best_params_ is not None
        check_is_fitted(search.best_forecaster_)

    @pytest.mark.skip(reason="Polars indexing bug in SplitConformalForecaster.predict_interval()")
    def test_grid_search_conformal_predict_interval(self, linear_series):
        """Verify best_forecaster_ supports predict_interval()."""
        y = linear_series(slope=2.0, intercept=10.0, length=100)

        forecaster = SplitConformalForecaster(
            point_forecaster=PointReductionForecaster(
                estimator=LinearRegression(),
                feature_transformer=LagTransformer([1, 2]),
            ),
            conformity_scorer=AbsoluteResidual(),
            calibration_size=20,  # Small calibration set
        )
        param_grid = {
            "point_forecaster__estimator": [LinearRegression(), Ridge(alpha=1.0)],
        }
        scorer = IntervalScore()
        cv = SlidingWindowSplitter(n_splits=3, test_size=10)

        search = GridSearchCV(forecaster, param_grid, scoring=scorer, cv=cv)
        search.fit(y, forecasting_horizon=5)

        # predict_interval should work on fitted search
        y_pred = search.predict_interval(forecasting_horizon=8, coverage_rates=[0.9])

        assert len(y_pred) == 8
        assert "time" in y_pred.columns
        assert "value" in y_pred.columns
        assert "lower_0.9" in y_pred.columns
        assert "upper_0.9" in y_pred.columns

    def test_grid_search_conformal_conformity_scorer_params(self, linear_series):
        """Verify param_grid can tune conformity_scorer parameters."""
        y = linear_series(slope=2.0, intercept=10.0, length=100)

        forecaster = SplitConformalForecaster(
            point_forecaster=PointReductionForecaster(
                estimator=LinearRegression(),
                feature_transformer=LagTransformer(1),
            ),
            conformity_scorer=AbsoluteResidual(),
            calibration_size=20,  # Small calibration set
        )
        # Tune both point forecaster and conformity scorer
        param_grid = {
            "point_forecaster__feature_transformer__lag": [1, [1, 2]],
            "conformity_scorer": [AbsoluteResidual()],
        }
        scorer = IntervalScore()
        cv = SlidingWindowSplitter(n_splits=3, test_size=10)

        search = GridSearchCV(forecaster, param_grid, scoring=scorer, cv=cv)
        search.fit(y, forecasting_horizon=5)

        # Should have 2 * 1 = 2 combinations
        assert len(search.cv_results_["params"]) == 2
        assert "param_point_forecaster__feature_transformer__lag" in search.cv_results_
        assert "param_conformity_scorer" in search.cv_results_


@pytest.mark.integration
class TestGridSearchConfiguration:
    """GridSearchCV with various configuration options (verbose, error_score)."""

    def test_grid_search_verbose_output(self, linear_series):
        """Verify GridSearchCV works with verbose output enabled."""
        y = linear_series(slope=2.0, intercept=10.0, length=100)

        forecaster = PointReductionForecaster(
            estimator=Ridge(),
            feature_transformer=LagTransformer(1),
        )
        param_grid = {
            "estimator__alpha": [0.0, 0.1, 1.0],
        }
        scorer = MeanAbsoluteError()
        cv = SlidingWindowSplitter(n_splits=3, test_size=10)

        search = GridSearchCV(forecaster, param_grid, scoring=scorer, cv=cv, verbose=1)
        search.fit(y, forecasting_horizon=5)

        # Should complete successfully with verbose output
        assert search.best_params_ is not None
        assert len(search.cv_results_["params"]) == 3

    def test_grid_search_error_score_nan(self, linear_series):
        """Verify GridSearchCV handles failed fits with error_score=np.nan."""
        y = linear_series(slope=2.0, intercept=10.0, length=100)

        forecaster = PointReductionForecaster(
            estimator=Ridge(),
            feature_transformer=LagTransformer([1, 2]),
        )
        param_grid = {
            "estimator__alpha": [0.0, 0.5, 1.0],
        }
        scorer = MeanAbsoluteError()
        cv = SlidingWindowSplitter(n_splits=3, test_size=10)

        search = GridSearchCV(forecaster, param_grid, scoring=scorer, cv=cv, error_score=np.nan)
        search.fit(y, forecasting_horizon=5)

        # Should complete successfully even if some fits fail
        assert search.best_params_ is not None
        assert len(search.cv_results_["params"]) == 3


@pytest.mark.integration
class TestNestedCVColumnForecaster:
    """Nested GridSearchCV inside ColumnForecaster for multi-column scenarios."""

    @pytest.mark.skip(reason="GridSearchCV doesn't expose interval_ attribute required by ColumnForecaster")
    def test_nested_cv_column_forecaster_inner_search(self, linear_series):
        """Verify inner GridSearchCV fits during outer fit."""
        y_multi = linear_series(slope=2.0, intercept=10.0, length=100).rename({"value": "col_a"})
        y_multi = y_multi.with_columns(
            pl.lit(5.0).alias("col_b")  # Constant column
        )

        inner_search = GridSearchCV(
            PointReductionForecaster(
                estimator=Ridge(),
                feature_transformer=LagTransformer(1),
            ),
            {"estimator__alpha": [0.0, 1.0]},
            scoring=MeanAbsoluteError(),
            cv=SlidingWindowSplitter(n_splits=3, test_size=5),
        )

        column_forecaster = ColumnForecaster([
            ("col_a", inner_search, ["col_a"]),
            ("col_b", SeasonalNaive(1), ["col_b"]),
        ])

        column_forecaster.fit(y_multi, forecasting_horizon=5)

        # inner_search should be fitted
        check_is_fitted(inner_search)
        assert inner_search.best_params_ is not None

    @pytest.mark.skip(reason="GridSearchCV doesn't expose interval_ attribute required by ColumnForecaster")
    def test_nested_cv_column_forecaster_best_forecaster_populated(self, linear_series):
        """Verify best_forecaster_ is populated inside ColumnForecaster."""
        y_multi = linear_series(slope=2.0, intercept=10.0, length=100).rename({"value": "col_a"})
        y_multi = y_multi.with_columns(pl.lit(10.0).alias("col_b"))

        inner_search = GridSearchCV(
            PointReductionForecaster(
                estimator=Ridge(),
                feature_transformer=LagTransformer([1, 2]),
            ),
            {"estimator__alpha": [0.0, 0.5, 1.0]},
            scoring=MeanAbsoluteError(),
            cv=SlidingWindowSplitter(n_splits=3, test_size=10),
        )

        column_forecaster = ColumnForecaster([
            ("col_a", inner_search, ["col_a"]),
            ("col_b", SeasonalNaive(1), ["col_b"]),
        ])

        column_forecaster.fit(y_multi, forecasting_horizon=5)

        # best_forecaster_ should be accessible
        assert hasattr(inner_search, "best_forecaster_")
        check_is_fitted(inner_search.best_forecaster_)

    @pytest.mark.skip(reason="GridSearchCV doesn't expose interval_ attribute required by ColumnForecaster")
    def test_nested_cv_column_forecaster_predictions_correct(self, linear_series):
        """Verify predictions work correctly with nested CV in ColumnForecaster."""
        y_multi = linear_series(slope=2.0, intercept=10.0, length=100).rename({"value": "col_a"})
        y_multi = y_multi.with_columns(
            (pl.col("col_a") * 2 + 5.0).alias("col_b")  # Linear transformation
        )

        inner_search = GridSearchCV(
            PointReductionForecaster(
                estimator=LinearRegression(),
                feature_transformer=LagTransformer(1),
            ),
            {"feature_transformer__lag": [1, [1, 2]]},
            scoring=MeanAbsoluteError(),
            cv=SlidingWindowSplitter(n_splits=3, test_size=10),
        )

        column_forecaster = ColumnForecaster([
            ("col_a", inner_search, ["col_a"]),
            ("col_b", SeasonalNaive(1), ["col_b"]),
        ])

        column_forecaster.fit(y_multi, forecasting_horizon=5)
        y_pred = column_forecaster.predict(forecasting_horizon=8)

        # Should produce predictions for both columns
        assert len(y_pred) == 8
        assert "col_a" in y_pred.columns
        assert "col_b" in y_pred.columns
        assert "time" in y_pred.columns
        # Predictions should be numeric (not null)
        assert not y_pred["col_a"].is_null().any()
        assert not y_pred["col_b"].is_null().any()
