"""Check functions for yohou interval forecasters.

This module provides validation functions specific to interval forecasters
(BaseIntervalForecaster implementations).
"""

import math

import polars as pl
from sklearn.base import clone

from yohou.utils import inspect_panel

__all__ = [
    "check_coverage_rates_parameter",
    "check_coverage_rates_validation",
    "check_interval_bounds",
    "check_interval_prediction_columns",
    "check_interval_prediction_types",
    "check_per_column_calibration_independence",
]


def check_interval_prediction_columns(forecaster, y_test: pl.DataFrame) -> None:
    """Check interval predictions have {col}_lower_{rate} and {col}_upper_{rate} format.

    Parameters
    ----------
    forecaster : BaseIntervalForecaster
        Fitted interval forecaster instance
    y_test : pl.DataFrame
        Test target data

    Raises
    ------
    AssertionError
        If interval column naming is incorrect

    """
    forecasting_horizon = min(3, len(y_test))

    # Call predict_interval for interval forecasters
    y_pred = forecaster.predict_interval(forecasting_horizon=forecasting_horizon)

    # Get coverage rates - use fit_coverage_rates_ (set during fit)
    coverage_rates = forecaster.fit_coverage_rates_

    # Check if we have panel data (columns with __ separator)
    _, y_panel_groups = inspect_panel(y_test)

    if len(y_panel_groups) > 0:
        # For panel data, interval columns use __ separator
        # e.g., "stores__store_0_lower_0.1"
        for group_prefix in y_panel_groups:
            # Get fields from the original training data (full column names)
            expected_fields = y_panel_groups[group_prefix]

            for rate in coverage_rates:
                for field in expected_fields:
                    lower_col = f"{field}_lower_{rate}"
                    upper_col = f"{field}_upper_{rate}"

                    assert lower_col in y_pred.columns, f"Missing lower bound column: {lower_col}"
                    assert upper_col in y_pred.columns, f"Missing upper bound column: {upper_col}"
    else:
        # For global data, check individual column pattern: {col}_lower_{rate}
        target_cols = list(forecaster.local_y_schema_.keys())

        # Check each coverage rate has lower and upper bounds for each target
        for rate in coverage_rates:
            for col in target_cols:
                lower_col = f"{col}_lower_{rate}"
                upper_col = f"{col}_upper_{rate}"

                assert lower_col in y_pred.columns, f"Missing lower bound column: {lower_col}"
                assert upper_col in y_pred.columns, f"Missing upper bound column: {upper_col}"


def check_interval_bounds(forecaster, y_test: pl.DataFrame) -> None:
    """Check upper >= lower for all coverage rates and time steps.

    Parameters
    ----------
    forecaster : BaseIntervalForecaster
        Fitted interval forecaster instance
    y_test : pl.DataFrame
        Test target data

    Raises
    ------
    AssertionError
        If upper bounds are less than lower bounds

    """
    forecasting_horizon = min(3, len(y_test))
    y_pred = forecaster.predict_interval(forecasting_horizon=forecasting_horizon)

    coverage_rates = forecaster.fit_coverage_rates_

    # Check if we have panel data (columns with __ separator)
    _, y_panel_groups = inspect_panel(y_test)

    if len(y_panel_groups) > 0:
        # For panel data, interval columns use __ separator (e.g. "stores__store_0").
        fields = [field for group_prefix in y_panel_groups for field in y_panel_groups[group_prefix]]
    else:
        fields = list(forecaster.local_y_schema_.keys())

    # Build one violation flag per (field, rate) pair and evaluate in a single pass.
    violation_exprs = [
        (pl.col(f"{field}_lower_{rate}") > pl.col(f"{field}_upper_{rate}")).any().alias(f"{field}_lower_{rate}")
        for rate in coverage_rates
        for field in fields
    ]
    violations = y_pred.select(violation_exprs)
    offenders = [name for name, flagged in violations.row(0, named=True).items() if flagged]
    if offenders:
        raise AssertionError(f"Found bounds where lower > upper: {offenders}")


def check_interval_prediction_types(forecaster) -> None:
    """Check interval forecaster has 'interval' in forecaster_type tag.

    Parameters
    ----------
    forecaster : BaseIntervalForecaster
        Interval forecaster instance

    Raises
    ------
    AssertionError
        If forecaster_type doesn't indicate interval support

    """
    tags = forecaster.__sklearn_tags__()
    forecaster_type = tags.forecaster_tags.forecaster_type if tags.forecaster_tags else None

    assert forecaster_type is not None and "interval" in forecaster_type, (
        f"Interval forecaster should have 'interval' in forecaster_type tag, got {forecaster_type}"
    )


def check_coverage_rates_parameter(forecaster) -> None:
    """Check fit_coverage_rates_ fitted attribute is a non-empty list of floats in [0, 1].

    Inspects the post-fit resolved ``fit_coverage_rates_`` attribute (the
    constructor ``coverage_rates`` param may be ``None`` and is resolved to a
    concrete list during fit), not the constructor argument.

    Parameters
    ----------
    forecaster : BaseIntervalForecaster
        Interval forecaster instance

    Raises
    ------
    AssertionError
        If fit_coverage_rates_ is missing or invalid

    """
    coverage_rates = forecaster.fit_coverage_rates_

    assert isinstance(coverage_rates, list), f"coverage_rates should be list, got {type(coverage_rates)}"

    assert len(coverage_rates) > 0, "coverage_rates should not be empty"

    for rate in coverage_rates:
        assert isinstance(rate, int | float), f"Each coverage rate should be numeric, got {type(rate)} for {rate}"
        assert 0 <= rate <= 1, f"Coverage rates should be in [0, 1], got {rate}"


def check_coverage_rates_validation(
    forecaster,
    y: pl.DataFrame,
    X_actual: pl.DataFrame | None = None,
    X_future: pl.DataFrame | None = None,
    X_forecast: pl.DataFrame | None = None,
) -> None:
    """Check invalid coverage_rates raise ValueError during fit and predict.

    Parameters
    ----------
    forecaster : BaseIntervalForecaster
        Unfitted interval forecaster instance
    y : pl.DataFrame
        Training target data
    X_actual : pl.DataFrame, optional
        Training features
    X_future : pl.DataFrame, optional
        Known-future features forwarded to fit()
    X_forecast : pl.DataFrame, optional
        External forecast features forwarded to fit()

    Raises
    ------
    AssertionError
        If invalid coverage_rates don't raise ValueError

    """
    # Test rate = 1.5 (above 1 - invalid)
    forecaster_clone = clone(forecaster)
    try:
        forecaster_clone.fit(
            y, X_actual, forecasting_horizon=3, coverage_rates=[1.5], X_future=X_future, X_forecast=X_forecast
        )
        raise AssertionError(f"{forecaster_clone.__class__.__name__} should raise ValueError for coverage_rates=[1.5]")
    except ValueError as e:
        assert "coverage" in str(e).lower() or "1.5" in str(e), f"ValueError should mention coverage_rates, got: {e}"

    # Test negative rate (invalid)
    forecaster_clone = clone(forecaster)
    try:
        forecaster_clone.fit(
            y, X_actual, forecasting_horizon=3, coverage_rates=[-0.5], X_future=X_future, X_forecast=X_forecast
        )
        raise AssertionError(f"{forecaster_clone.__class__.__name__} should raise ValueError for coverage_rates=[-0.5]")
    except ValueError as e:
        assert "coverage" in str(e).lower() or "negative" in str(e).lower(), (
            f"ValueError should mention coverage_rates, got: {e}"
        )

    # Test predict_interval also validates
    forecaster_clone = clone(forecaster)
    forecaster_clone.fit(
        y, X_actual, forecasting_horizon=3, coverage_rates=[0.95], X_future=X_future, X_forecast=X_forecast
    )

    try:
        forecaster_clone.predict_interval(forecasting_horizon=3, coverage_rates=[1.5])
        raise AssertionError(
            f"{forecaster_clone.__class__.__name__}.predict_interval() should raise ValueError for coverage_rates=[1.5]"
        )
    except ValueError as e:
        assert "coverage" in str(e).lower(), f"ValueError should mention coverage_rates, got: {e}"


def check_per_column_calibration_independence(forecaster, y_train: pl.DataFrame) -> None:
    """Check each value column's interval is calibrated from that column alone.

    Multiplies one value column by a constant and asserts two things: that
    column's interval width scales with it, and every other column's width does
    not move. A forecaster that reduces a multi-column conformity frame to one
    shared quantile fails both halves, which is the defect this check exists to
    catch: before it, a frame holding a scale-1 and a scale-100 column gave both
    the same width, over-covering the small column and under-covering the large.

    The check builds its own inner point forecaster rather than reusing the one
    under test. The independence half is only well defined when the point model
    does not pool rows across columns: a shared global model legitimately moves
    every column's prediction when one column's data changes, so the width shift
    would not be attributable to the calibration axis.

    Parameters
    ----------
    forecaster : BaseIntervalForecaster
        Forecaster under test. Used for its class and constructor parameters;
        a fresh clone is fitted here.
    y_train : pl.DataFrame
        Training data. Ignored except for its time column, since the check
        generates its own two-column frame with a known scale relationship.

    Raises
    ------
    AssertionError
        If the scaled column's width does not track its data, or if another
        column's width moves in response.

    """
    from yohou.point import SeasonalNaive

    times = y_train["time"]
    if len(times) < 60:
        return

    seasonality = 7
    base = [10.0 + 5.0 * ((i % seasonality) - 3) / 3.0 for i in range(len(times))]
    jitter = [0.4 * (((i * 37) % 11) / 11.0 - 0.5) for i in range(len(times))]
    quiet = [b + j for b, j in zip(base, jitter, strict=True)]

    factor = 100.0
    frame = pl.DataFrame({"time": times, "a__value": quiet, "b__value": quiet})
    scaled = frame.with_columns(pl.col("b__value") * factor)

    def widths(y: pl.DataFrame) -> dict[str, float]:
        """Fit a fresh clone on ``y`` and return its interval width per column."""
        candidate = clone(forecaster)
        candidate.set_params(point_forecaster=SeasonalNaive(seasonality=seasonality))
        calibration_size = min(50, len(times) // 3)
        candidate.set_params(calibration_size=calibration_size)
        candidate.fit(y, forecasting_horizon=1, coverage_rates=[0.9])
        intervals = candidate.predict_interval(forecasting_horizon=1, coverage_rates=[0.9])
        return {
            column: float(intervals[f"{column}_upper_0.9"][0] - intervals[f"{column}_lower_0.9"][0])
            for column in ("a__value", "b__value")
        }

    before, after = widths(frame), widths(scaled)

    if before["b__value"] <= 0.0:
        return

    observed_factor = after["b__value"] / before["b__value"]
    assert 0.5 * factor < observed_factor < 2.0 * factor, (
        f"scaling column 'b__value' by {factor} changed its interval width by {observed_factor:.2f}x. "
        "A column's width must be calibrated from that column's own conformity scores."
    )
    assert math.isclose(after["a__value"], before["a__value"], rel_tol=1e-9), (
        f"scaling column 'b__value' moved column 'a__value' width from {before['a__value']} to "
        f"{after['a__value']}. Calibration must not be pooled across value columns."
    )
