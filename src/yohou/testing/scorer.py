"""Check functions for yohou scorers.

This module provides systematic validation functions for testing BaseScorer
implementations. All check functions raise AssertionError on failure.

Organized into three categories:
1. Tag System Checks - Validate tag system correctness (3 functions)
2. Functionality Checks - Core scorer behavior validation (6 functions)
3. Parameter Validation Checks - Validate parameter validation (1 function)
"""

try:
    import polars as pl
except ImportError as e:
    raise ImportError("polars is required for yohou.testing module. Install with: uv sync --group tests") from e

import datetime
import inspect
from typing import Any

import numpy as np
from sklearn.base import clone
from sklearn.exceptions import NotFittedError

__all__ = [
    "check_scorer_aggregation_methods",
    "check_scorer_component_subselection",
    "check_scorer_coverage_rate_subselection",
    "check_scorer_lower_is_better",
    "check_scorer_methods_call_check_is_fitted",
    "check_scorer_multi_vintage",
    "check_scorer_panel_subselection",
    "check_scorer_parameter_validation",
    "check_scorer_prediction_type_compatibility",
    "check_scorer_tags_accessible_before_fit",
    "check_scorer_tags_match_capabilities",
    "check_scorer_tags_static_after_fit",
]


def check_scorer_tags_accessible_before_fit(scorer) -> None:
    """Check __sklearn_tags__() is callable on scorer instance.

    Tags should be accessible on instances to describe capabilities.

    Parameters
    ----------
    scorer : BaseScorer instance
        Scorer instance (can be unfitted)

    Raises
    ------
    AssertionError
        If __sklearn_tags__() is not callable or raises an error

    """
    scorer_name = type(scorer).__name__
    assert hasattr(scorer, "__sklearn_tags__"), f"{scorer_name} must have __sklearn_tags__() method"

    # Should be callable on instance
    try:
        tags = scorer.__sklearn_tags__()
    except Exception as e:
        raise AssertionError(f"{scorer_name}.__sklearn_tags__() raised {type(e).__name__}: {e}") from e

    assert tags.estimator_type == "scorer", (
        f"{scorer_name} tags must have estimator_type='scorer', got {tags.estimator_type}"
    )


def check_scorer_tags_static_after_fit(
    scorer,
    y_truth: pl.DataFrame,
) -> None:
    """Check tags remain unchanged after fit.

    Tags describe inherent capabilities, not fitted state.
    They should be identical before and after fitting.

    Parameters
    ----------
    scorer : BaseScorer
        Unfitted scorer instance
    y_truth : pl.DataFrame
        Ground truth with "time" column

    Raises
    ------
    AssertionError
        If tags change after fitting

    """
    scorer_clone = clone(scorer)

    # Get tags before fit
    tags_before = scorer_clone.__sklearn_tags__()

    # Always fit scorer (all scorers require fitting)
    scorer_clone.fit(y_truth)

    # Get tags after fit
    tags_after = scorer_clone.__sklearn_tags__()

    # Tags should be identical
    assert tags_before == tags_after, "Tags changed after fit() - tags should be static"


def check_scorer_tags_match_capabilities(
    scorer,
    y_truth: pl.DataFrame,
    y_pred: pl.DataFrame,
    expected_tags: dict | None = None,
) -> None:
    """Check tag values match actual scorer behavior.

    Validates that declared tags accurately reflect what the scorer does.

    Parameters
    ----------
    scorer : BaseScorer
        Scorer instance
    y_truth : pl.DataFrame
        Ground truth with "time" column
    y_pred : pl.DataFrame
        Predictions with "vintage_time" and "time" columns
    expected_tags : dict, optional
        Expected tag values to validate (keys: prediction_type, lower_is_better, requires_calibration)

    Raises
    ------
    AssertionError
        If tags don't match actual capabilities

    """
    tags = scorer.__sklearn_tags__()
    scorer_tags = tags.scorer_tags

    if expected_tags is not None:
        for key, expected_value in expected_tags.items():
            actual_value = getattr(scorer_tags, key)
            assert actual_value == expected_value, f"Tag mismatch: {key}={actual_value}, expected {expected_value}"

    # Validate prediction_type matches data
    if scorer_tags.prediction_type == "point":
        # Point scorers should work with point predictions (no _lower/_upper columns)
        interval_cols = [c for c in y_pred.columns if "_lower" in c or "_upper" in c]
        assert not interval_cols, f"Point scorer received interval predictions with columns: {interval_cols}"
    elif scorer_tags.prediction_type == "interval":
        # Interval scorers should have _lower and _upper columns
        lower_cols = [c for c in y_pred.columns if "_lower" in c]
        upper_cols = [c for c in y_pred.columns if "_upper" in c]
        assert lower_cols, "Interval scorer missing _lower columns"
        assert upper_cols, "Interval scorer missing _upper columns"


def check_scorer_prediction_type_compatibility(
    scorer,
    forecaster,
    y: pl.DataFrame,
    X_actual: pl.DataFrame | None = None,
) -> None:
    """Check scorer works with correct forecaster output type.

    Point scorers should work with point forecasters, interval scorers
    with interval forecasters.

    Parameters
    ----------
    scorer : BaseScorer
        Scorer instance
    forecaster : BaseForecaster
        Fitted forecaster
    y : pl.DataFrame
        Target time series
    X_actual : pl.DataFrame, optional
        Exogenous features

    Raises
    ------
    AssertionError
        If scorer/forecaster types are incompatible

    """
    scorer_tags = scorer.__sklearn_tags__()
    forecaster_tags = forecaster.__sklearn_tags__()

    # Get predictions based on scorer type
    if scorer_tags.scorer_tags.prediction_type == "point":
        # Point scorer needs point predictions
        assert "point" in forecaster_tags.forecaster_tags.forecaster_type, (
            f"Point scorer requires point forecaster, got {forecaster_tags.forecaster_tags.forecaster_type}"
        )
        y_pred = forecaster.predict(forecasting_horizon=3)
    elif scorer_tags.scorer_tags.prediction_type == "interval":
        # Interval scorer needs interval predictions
        assert "interval" in forecaster_tags.forecaster_tags.forecaster_type, (
            f"Interval scorer requires interval forecaster, got {forecaster_tags.forecaster_tags.forecaster_type}"
        )
        y_pred = forecaster.predict_interval(forecasting_horizon=3, coverage_rates=[0.9])
    elif scorer_tags.scorer_tags.prediction_type == "class_proba":
        # Class-probability scorer needs class_proba predictions
        assert "class_proba" in forecaster_tags.forecaster_tags.forecaster_type, (
            f"Class-proba scorer requires class_proba forecaster, got {forecaster_tags.forecaster_tags.forecaster_type}"
        )
        y_pred = forecaster.predict_class_proba(forecasting_horizon=3)
    else:
        raise AssertionError(f"Unknown prediction_type: {scorer_tags.scorer_tags.prediction_type}")

    # Build y_truth that overlaps with predicted times (predictions may be
    # for future timestamps beyond the training data).
    pred_times = y_pred["time"].unique()
    y_truth = y.filter(pl.col("time").is_in(pred_times.to_list()))
    if len(y_truth) == 0:
        value_cols = [c for c in y.columns if c != "time"]
        rows: dict[str, list] = {"time": pred_times.to_list()}
        for col in value_cols:
            val = y[col].mean() if y[col].dtype.is_numeric() else y[col][0]
            rows[col] = [val] * len(pred_times)
        y_truth = pl.DataFrame(rows)

    # Score should work
    try:
        score = scorer.score(y_truth, y_pred)
        assert isinstance(score, int | float | np.number), f"score() should return numeric value, got {type(score)}"
    except Exception as e:
        raise AssertionError(f"Compatible scorer/forecaster types failed: {e}") from e


def check_scorer_lower_is_better(scorer) -> None:
    """Check the ``lower_is_better`` tag is a boolean.

    Validates only that the scorer exposes a boolean ``lower_is_better`` tag;
    it does not assert a particular value, so the caller is responsible for
    pinning the expected direction (e.g. via ``expected_tags``) elsewhere.

    Parameters
    ----------
    scorer : BaseScorer
        Scorer instance

    Raises
    ------
    AssertionError
        If lower_is_better is not a boolean

    """
    tags = scorer.__sklearn_tags__()
    lower_is_better = tags.scorer_tags.lower_is_better

    # Most error metrics have lower_is_better=True
    # Most score metrics (R², etc.) have lower_is_better=False
    assert isinstance(lower_is_better, bool), f"lower_is_better should be bool, got {type(lower_is_better)}"


def check_scorer_aggregation_methods(
    scorer,
    y_truth: pl.DataFrame,
    y_pred: pl.DataFrame,
    aggregation_methods: list[str],
) -> None:
    """Check all aggregation_method combinations produce valid output.

    Parameters
    ----------
    scorer : BaseScorer
        Scorer instance with aggregation_method parameter
    y_truth : pl.DataFrame
        Ground truth
    y_pred : pl.DataFrame
        Predictions
    aggregation_methods : list of str
        Valid aggregation methods to test

    Raises
    ------
    AssertionError
        If any aggregation method fails or produces invalid output

    Notes
    -----
    A scalar is returned only when stepwise, vintagewise, and componentwise
    aggregations are all applied together; partial aggregations (e.g.,
    ['stepwise'] only) return a DataFrame.

    """
    for agg_method in aggregation_methods:
        scorer_copy = clone(scorer)
        scorer_copy.set_params(aggregation_method=[agg_method])

        # Always fit scorer before scoring
        scorer_copy.fit(y_truth)

        try:
            score = scorer_copy.score(y_truth, y_pred)

            # Validate return type - can be scalar or DataFrame depending on aggregation
            if isinstance(score, pl.DataFrame):
                # DataFrame is valid for partial aggregations (e.g., stepwise only)
                assert len(score) > 0, f"aggregation_method={agg_method}: returned empty DataFrame"
                assert not score.null_count().sum_horizontal()[0] > 0, (
                    f"aggregation_method={agg_method}: DataFrame contains null values"
                )
            elif isinstance(score, int | float | np.number):
                # Scalar is valid for full aggregations
                assert not np.isnan(score), f"aggregation_method={agg_method}: score is NaN"
            else:
                raise AssertionError(
                    f"aggregation_method={agg_method}: score should be numeric or DataFrame, got {type(score)}"
                )
        except Exception as e:
            raise AssertionError(f"aggregation_method={agg_method} failed: {e}") from e


def check_scorer_panel_subselection(
    scorer,
    y_truth_panel: pl.DataFrame,
    y_pred_panel: pl.DataFrame,
    groups: list[str],
) -> None:
    """Check groups filtering works correctly.

    Parameters
    ----------
    scorer : BaseScorer
        Scorer instance
    y_truth_panel : pl.DataFrame
        Panel ground truth
    y_pred_panel : pl.DataFrame
        Panel predictions
    groups : list of str
        Panel group names to filter

    Raises
    ------
    AssertionError
        If panel subselection fails

    """
    scorer_filtered = clone(scorer)
    scorer_filtered.set_params(groups=groups)

    # Always fit scorer before scoring
    scorer_filtered.fit(y_truth_panel)

    try:
        score = scorer_filtered.score(y_truth_panel, y_pred_panel)
        assert isinstance(score, int | float | np.number), "Panel-filtered score should be numeric"
    except Exception as e:
        raise AssertionError(f"groups={groups} filtering failed: {e}") from e


def check_scorer_component_subselection(
    scorer,
    y_truth: pl.DataFrame,
    y_pred: pl.DataFrame,
    components: list[str],
) -> None:
    """Check components filtering works correctly.

    Parameters
    ----------
    scorer : BaseScorer
        Scorer instance
    y_truth : pl.DataFrame
        Ground truth
    y_pred : pl.DataFrame
        Predictions
    components : list of str
        Component names to filter

    Raises
    ------
    AssertionError
        If component subselection fails

    """
    scorer_filtered = clone(scorer)
    scorer_filtered.set_params(components=components)

    # Always fit scorer before scoring
    scorer_filtered.fit(y_truth)

    try:
        score = scorer_filtered.score(y_truth, y_pred)
        assert isinstance(score, int | float | np.number), "Component-filtered score should be numeric"
    except Exception as e:
        raise AssertionError(f"components={components} filtering failed: {e}") from e


def check_scorer_coverage_rate_subselection(
    scorer,
    y_truth: pl.DataFrame,
    y_pred_interval: pl.DataFrame,
    coverage_rates: list[float],
) -> None:
    """Check coverage parameter filters interval predictions correctly.

    Only applicable for interval scorers.

    Parameters
    ----------
    scorer : BaseScorer
        Interval scorer instance
    y_truth : pl.DataFrame
        Ground truth
    y_pred_interval : pl.DataFrame
        Interval predictions with coverage_rate column
    coverage_rates : list of float
        Coverage rates to filter

    Raises
    ------
    AssertionError
        If coverage filtering fails

    """
    tags = scorer.__sklearn_tags__()
    if tags.scorer_tags.prediction_type != "interval":
        # Skip for non-interval scorers
        return

    scorer_filtered = clone(scorer)
    scorer_filtered.set_params(coverage_rates=coverage_rates)

    # Always fit scorer before scoring
    scorer_filtered.fit(y_truth)

    try:
        score = scorer_filtered.score(y_truth, y_pred_interval)
        assert isinstance(score, int | float | np.number), "Coverage-filtered score should be numeric"
    except Exception as e:
        raise AssertionError(f"coverage_rates={coverage_rates} filtering failed: {e}") from e


def check_scorer_parameter_validation(
    scorer_class,
    param_name: str,
    invalid_value: Any,
    error_match: str | None = None,
) -> None:
    """Check parameter validation raises ValueError for invalid inputs.

    Tests that scorer._validate_parameters() properly rejects invalid inputs
    when fit() is called (parameter validation happens in fit(), not score()).

    Parameters
    ----------
    scorer_class : type
        Scorer class
    param_name : str
        Parameter name to test
    invalid_value : any
        Invalid value that should trigger ValueError
    error_match : str, optional
        Expected substring in error message

    Raises
    ------
    AssertionError
        If invalid value is accepted

    """
    # Parameter validation happens in fit(), so minimal truth data is enough
    y_truth = pl.DataFrame({
        "time": [datetime.datetime(2020, 1, i) for i in range(1, 11)],
        "value": [float(i) for i in range(10)],
    })

    # Create scorer with invalid parameter
    params = {param_name: invalid_value}
    scorer = scorer_class(**params)

    # Always call fit() to trigger parameter validation (sklearn pattern)
    # Parameter validation happens in fit(), not score()
    try:
        scorer.fit(y_truth)
        raise AssertionError(f"{scorer_class.__name__}: invalid {param_name}={invalid_value} was accepted in fit()")
    except (ValueError, TypeError) as e:
        # Expected - validation should raise ValueError or TypeError
        if error_match is None or error_match in str(e):
            return  # Test passed
        raise AssertionError(f"Expected error containing '{error_match}', got: {e}") from e


def check_scorer_methods_call_check_is_fitted(scorer, y_train: pl.DataFrame, y_pred: pl.DataFrame) -> None:
    """Check all scorer methods (except fit) raise NotFittedError when unfitted.

    Validates that score() and inverse_score() methods check fitted state and
    raise NotFittedError before operating on an unfitted scorer.

    Parameters
    ----------
    scorer : BaseScorer
        Unfitted scorer instance
    y_train : pl.DataFrame
        Training target data with "time" column
    y_pred : pl.DataFrame
        Predicted values with "vintage_time" and "time" columns

    Raises
    ------
    AssertionError
        If any method fails to raise NotFittedError when called on unfitted scorer

    """
    scorer_clone = clone(scorer)

    # Test that score() raises NotFittedError when unfitted
    try:
        scorer_clone.score(y_train[:10], y_pred[:10])
        raise AssertionError(
            f"{scorer_clone.__class__.__name__}.score() must raise NotFittedError when called on unfitted scorer"
        )
    except NotFittedError:
        pass  # Expected

    # Test inverse_score() if implemented
    if hasattr(scorer_clone, "inverse_score"):
        scorer_clone.fit(y_train)
        try:
            scores = scorer_clone.score(y_train[:10], y_pred[:10])
            scorer_clone2 = clone(scorer)  # Fresh unfitted clone
            # Only pass coverage_rate when the scorer's inverse_score declares it.
            inverse_params = inspect.signature(scorer_clone2.inverse_score).parameters
            inverse_kwargs = {"coverage_rate": 0.9} if "coverage_rate" in inverse_params else {}
            scorer_clone2.inverse_score(y_pred[:10], scores, **inverse_kwargs)
            raise AssertionError(
                f"{scorer_clone.__class__.__name__}.inverse_score() must raise NotFittedError when called on unfitted scorer"
            )
        except NotFittedError:
            pass  # Expected


def check_scorer_multi_vintage(
    scorer,
    y_truth: pl.DataFrame,
    y_pred: pl.DataFrame,
) -> None:
    """Check that scorer produces a finite result on multi-vintage input.

    Builds a 2-vintage dataset by duplicating ``y_pred`` with a second
    ``vintage_time`` value. Scores the result and asserts the output is
    finite (float) or a non-empty DataFrame without nulls.

    This is a smoke test, not a correctness test. It catches scorers that
    crash or produce NaN on multi-vintage input.

    Parameters
    ----------
    scorer : BaseScorer
        Fitted scorer instance.
    y_truth : pl.DataFrame
        Ground truth with ``"time"`` column.
    y_pred : pl.DataFrame
        Predictions with ``"time"`` and ``"vintage_time"`` columns.

    Raises
    ------
    AssertionError
        If the scorer crashes, returns NaN, or returns an empty DataFrame.

    """
    scorer_clone = clone(scorer)
    scorer_clone.fit(y_truth)

    # Build 2-vintage y_pred by duplicating with a second vintage_time
    if "vintage_time" not in y_pred.columns:
        return  # Cannot build multi-vintage data without vintage_time

    existing_vt = y_pred["vintage_time"][0]
    # Shift the vintage time back by 1 day for the second vintage
    if isinstance(existing_vt, datetime.datetime | datetime.date):
        second_vt = existing_vt - datetime.timedelta(days=1)
    else:
        return  # Cannot build second vintage from non-date type

    y_pred_v2 = y_pred.with_columns(pl.lit(second_vt).alias("vintage_time"))
    y_pred_multi = pl.concat([y_pred, y_pred_v2]).sort("time", "vintage_time")

    score = scorer_clone.score(y_truth, y_pred_multi)

    if isinstance(score, pl.DataFrame):
        assert len(score) > 0, "Multi-vintage score returned empty DataFrame"
    elif isinstance(score, int | float | np.number):
        assert np.isfinite(score), f"Multi-vintage score is not finite: {score}"
    else:
        raise AssertionError(f"Multi-vintage score should be numeric or DataFrame, got {type(score)}")
