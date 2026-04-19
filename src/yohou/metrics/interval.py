"""Metrics for evaluating interval forecast quality."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from .base import BaseIntervalScorer

if TYPE_CHECKING:
    from datetime import datetime

__all__ = [
    "CalibrationError",
    "EmpiricalCoverage",
    "IntervalScore",
    "MeanIntervalWidth",
    "PinballLoss",
]


class EmpiricalCoverage(BaseIntervalScorer):
    r"""Empirical coverage rate for prediction intervals.

    Measures the proportion of true values falling within the predicted
    intervals. A well-calibrated forecaster should achieve coverage close
    to the nominal rate (e.g., 90% for α=0.9).

    The empirical coverage for a given coverage rate α is:

    $$\\text{Coverage}(\\alpha) = \\frac{1}{n}\\sum_{i=1}^{n} \\mathbb{1}(y_i \\in [L_i(\\alpha), U_i(\\alpha)])$$

    where $L_i(\\alpha)$ and $U_i(\\alpha)$ are the lower and upper bounds at coverage $\\alpha$,
    $\\mathbb{1}(\\cdot)$ is the indicator function, and $n$ is the number of observations.

    Parameters
    ----------
    aggregation_method : list of str or str, default="all"
        Dimensions to collapse when aggregating scores. Orthogonal modes:

        - "stepwise": Collapse the forecasting-step dimension.
        - "vintagewise": Collapse the vintage/observed-time dimension.
        - "componentwise": Collapse components, return per-timestep scores.
        - "groupwise": Collapse panel groups (panel data only).
        - "coveragewise": Collapse coverage rates (return average coverage).

        - "all": Collapse all dimensions (returns scalar).
    coverage : list of float, dict of float to float, or None, default=None
        Coverage rate filter (list) or filter with weights (dict).
    groups : list of str, dict of str to float, or None, default=None
        Panel group filter (list) or filter with weights (dict).
    components : list of str, dict of str to float, or None, default=None
        Component filter (list) or filter with weights (dict).

    Attributes
    ----------
    lower_is_better : bool
        False for coverage (deviations from nominal rate in either direction are bad).

    Examples
    --------
    >>> import polars as pl
    >>> from datetime import datetime
    >>> from yohou.metrics import EmpiricalCoverage
    >>> y_true = pl.DataFrame({
    ...     "time": [datetime(2020, 1, 1), datetime(2020, 1, 2), datetime(2020, 1, 3)],
    ...     "value": [10.0, 20.0, 30.0],
    ... })
    >>> y_pred = pl.DataFrame({
    ...     "observed_time": [datetime(2019, 12, 31)] * 3,
    ...     "time": [datetime(2020, 1, 1), datetime(2020, 1, 2), datetime(2020, 1, 3)],
    ...     "value_lower_0.9": [8.0, 18.0, 28.0],
    ...     "value_upper_0.9": [12.0, 22.0, 32.0],
    ... })
    >>> coverage = EmpiricalCoverage()
    >>> _ = coverage.fit(y_true)
    >>> coverage.score(y_true, y_pred)
    1.0

    Notes
    -----
    - Perfect coverage = nominal rate (e.g., 0.9 for 90% intervals)
    - Over-coverage (> nominal) indicates conservative (wide) intervals
    - Under-coverage (< nominal) indicates poor calibration
    - Missing values are excluded from computation

    See Also
    --------
    `MeanIntervalWidth` : Evaluates interval sharpness
    `IntervalScore` : Combined coverage and sharpness metric
    `CalibrationError` : Aggregate miscalibration metric

    """

    _parameter_constraints: dict = {
        **BaseIntervalScorer._parameter_constraints,
    }

    _metric_name = "coverage"
    _lower_is_better = False

    def __init__(
        self,
        aggregation_method: list[str] | str = "all",
        coverage: list[float] | dict[float, float] | None = None,
        groups: list[str] | dict[str, float] | None = None,
        components: list[str] | dict[str, float] | None = None,
    ) -> None:
        agg_list = aggregation_method
        if aggregation_method == "all":
            agg_list = ["stepwise", "vintagewise", "componentwise", "groupwise", "coveragewise"]

        super().__init__(
            aggregation_method=agg_list,
            coverage=coverage,
            groups=groups,
            components=components,
        )

    def _compute_raw_scores(self, y_truth, y_pred, coverage_rates, target_columns):
        """Compute per-row empirical coverage indicators."""
        frames = []
        for rate in coverage_rates:
            rate_data = {}
            for col in target_columns:
                lower_col = f"{col}_lower_{rate}"
                upper_col = f"{col}_upper_{rate}"
                if lower_col in y_pred.columns and upper_col in y_pred.columns:
                    in_interval = (y_truth[col] >= y_pred[lower_col]) & (y_truth[col] <= y_pred[upper_col])
                    rate_data[col] = in_interval.cast(pl.Float64)
            frames.append(pl.DataFrame(rate_data).with_columns(pl.lit(rate).alias("coverage_rate")))
        return pl.concat(frames)


class MeanIntervalWidth(BaseIntervalScorer):
    r"""Mean width of prediction intervals.

    Measures the average width of prediction intervals. Narrower intervals are
    preferred (more informative), provided coverage is maintained.

    The mean interval width for coverage rate α is:

    $$\\text{Width}(\\alpha) = \\frac{1}{n}\\sum_{i=1}^{n} |U_i(\\alpha) - L_i(\\alpha)|$$

    Parameters
    ----------
    aggregation_method : list of str or str, default="all"
        Dimensions to collapse when aggregating scores. Orthogonal modes:

        - "stepwise": Collapse the forecasting-step dimension.
        - "vintagewise": Collapse the vintage/observed-time dimension.
        - "componentwise": Collapse components, return per-timestep scores.
        - "groupwise": Collapse panel groups (panel data only).
        - "coveragewise": Collapse coverage rates (return average coverage).

        - "all": Collapse all dimensions (returns scalar).
    coverage : list of float, dict of float to float, or None, default=None
        Coverage rate filter (list) or filter with weights (dict).
    groups : list of str, dict of str to float, or None, default=None
        Panel group filter (list) or filter with weights (dict).
    components : list of str, dict of str to float, or None, default=None
        Component filter (list) or filter with weights (dict).

    Attributes
    ----------
    lower_is_better : bool
        True for width (narrower intervals are better).

    Examples
    --------
    >>> import polars as pl
    >>> from datetime import datetime
    >>> from yohou.metrics import MeanIntervalWidth
    >>> y_true = pl.DataFrame({"time": [datetime(2020, 1, 1), datetime(2020, 1, 2)], "value": [10.0, 20.0]})
    >>> y_pred = pl.DataFrame({
    ...     "observed_time": [datetime(2019, 12, 31)] * 2,
    ...     "time": [datetime(2020, 1, 1), datetime(2020, 1, 2)],
    ...     "value_lower_0.9": [8.0, 18.0],
    ...     "value_upper_0.9": [12.0, 22.0],
    ... })
    >>> width = MeanIntervalWidth()
    >>> _ = width.fit(y_true)
    >>> width.score(y_true, y_pred)
    4.0

    Notes
    -----
    - Lower is better (sharper, more precise predictions)
    - Should only be compared when coverage is approximately equal
    - Scale-dependent (units match target variable)
    - Missing values are excluded from computation
    - Uses absolute width to handle bound inversions silently

    See Also
    --------
    `EmpiricalCoverage` : Evaluates interval calibration
    `IntervalScore` : Combined coverage and sharpness metric

    """

    _parameter_constraints: dict = {
        **BaseIntervalScorer._parameter_constraints,
    }

    _metric_name = "width"

    def __init__(
        self,
        aggregation_method: list[str] | str = "all",
        coverage: list[float] | dict[float, float] | None = None,
        groups: list[str] | dict[str, float] | None = None,
        components: list[str] | dict[str, float] | None = None,
    ) -> None:
        agg_list = aggregation_method
        if aggregation_method == "all":
            agg_list = ["stepwise", "vintagewise", "componentwise", "groupwise", "coveragewise"]

        super().__init__(
            aggregation_method=agg_list,
            coverage=coverage,
            groups=groups,
            components=components,
        )

    def _compute_raw_scores(self, y_truth, y_pred, coverage_rates, target_columns):
        """Compute per-row interval width values."""
        frames = []
        for rate in coverage_rates:
            rate_data = {}
            for col in target_columns:
                lower_col = f"{col}_lower_{rate}"
                upper_col = f"{col}_upper_{rate}"
                if lower_col in y_pred.columns and upper_col in y_pred.columns:
                    rate_data[col] = (y_pred[upper_col] - y_pred[lower_col]).abs()
            frames.append(pl.DataFrame(rate_data).with_columns(pl.lit(rate).alias("coverage_rate")))
        return pl.concat(frames)


class IntervalScore(BaseIntervalScorer):
    r"""Interval Score (Winkler Score) for prediction intervals.

    Combines interval width with penalties for observations falling outside
    the interval. Balances sharpness and coverage in a single metric.

    The interval score for coverage rate α is:

    $$\\text{IS}(\\alpha) = \\frac{1}{n}\\sum_{i=1}^{n} \\left[|U_i - L_i| + \\frac{2}{\\alpha}(L_i - y_i)\\mathbb{1}(y_i < L_i) + \\frac{2}{\\alpha}(y_i - U_i)\\mathbb{1}(y_i > U_i)\\right]$$

    where the first term is interval width and the second/third terms are penalties
    for under/over-coverage.

    Parameters
    ----------
    aggregation_method : list of str or str, default="all"
        Dimensions to collapse when aggregating scores. Orthogonal modes:

        - "stepwise": Collapse the forecasting-step dimension.
        - "vintagewise": Collapse the vintage/observed-time dimension.
        - "componentwise": Collapse components, return per-timestep scores.
        - "groupwise": Collapse panel groups (panel data only).
        - "coveragewise": Collapse coverage rates (return average interval score).

        - "all": Collapse all dimensions (returns scalar).
    coverage : list of float, dict of float to float, or None, default=None
        Coverage rate filter (list) or filter with weights (dict).
    groups : list of str, dict of str to float, or None, default=None
        Panel group filter (list) or filter with weights (dict).
    components : list of str, dict of str to float, or None, default=None
        Component filter (list) or filter with weights (dict).

    Attributes
    ----------
    lower_is_better : bool
        True for interval score (lower is better).

    Examples
    --------
    >>> import polars as pl
    >>> from datetime import datetime
    >>> from yohou.metrics import IntervalScore
    >>> y_true = pl.DataFrame({"time": [datetime(2020, 1, 1), datetime(2020, 1, 2)], "value": [10.0, 20.0]})
    >>> y_pred = pl.DataFrame({
    ...     "observed_time": [datetime(2019, 12, 31)] * 2,
    ...     "time": [datetime(2020, 1, 1), datetime(2020, 1, 2)],
    ...     "value_lower_0.9": [8.0, 18.0],
    ...     "value_upper_0.9": [12.0, 22.0],
    ... })
    >>> scorer = IntervalScore()
    >>> _ = scorer.fit(y_true)
    >>> scorer.score(y_true, y_pred)
    4.0

    Notes
    -----
    - Lower is better
    - Penalizes both wide intervals and poor coverage
    - Scale-dependent (same units as target)
    - Also known as "Winkler score" in literature
    - Widely used in forecasting competitions (M4, M5)

    See Also
    --------
    `EmpiricalCoverage` : Coverage-only metric
    `MeanIntervalWidth` : Width-only metric
    `PinballLoss` : Asymmetric quantile-based metric

    """

    _parameter_constraints: dict = {
        **BaseIntervalScorer._parameter_constraints,
    }

    _metric_name = "interval_score"

    def __init__(
        self,
        aggregation_method: list[str] | str = "all",
        coverage: list[float] | dict[float, float] | None = None,
        groups: list[str] | dict[str, float] | None = None,
        components: list[str] | dict[str, float] | None = None,
    ) -> None:
        agg_list = aggregation_method
        if aggregation_method == "all":
            agg_list = ["stepwise", "vintagewise", "componentwise", "groupwise", "coveragewise"]

        super().__init__(
            aggregation_method=agg_list,
            coverage=coverage,
            groups=groups,
            components=components,
        )

    def _compute_raw_scores(self, y_truth, y_pred, coverage_rates, target_columns):
        """Compute per-row interval score values."""
        frames = []
        for rate in coverage_rates:
            rate_data = {}
            for col in target_columns:
                lower_col = f"{col}_lower_{rate}"
                upper_col = f"{col}_upper_{rate}"
                if lower_col in y_pred.columns and upper_col in y_pred.columns:
                    width = (y_pred[upper_col] - y_pred[lower_col]).abs()

                    lower_penalty = (
                        pl
                        .when(y_truth[col] < y_pred[lower_col])
                        .then((2.0 / rate) * (y_pred[lower_col] - y_truth[col]))
                        .otherwise(0.0)
                    )
                    upper_penalty = (
                        pl
                        .when(y_truth[col] > y_pred[upper_col])
                        .then((2.0 / rate) * (y_truth[col] - y_pred[upper_col]))
                        .otherwise(0.0)
                    )

                    lower_penalty_series = y_pred.select(lower_penalty.alias("lp"))["lp"]
                    upper_penalty_series = y_pred.select(upper_penalty.alias("up"))["up"]

                    rate_data[col] = width + lower_penalty_series + upper_penalty_series
            frames.append(pl.DataFrame(rate_data).with_columns(pl.lit(rate).alias("coverage_rate")))
        return pl.concat(frames)


class PinballLoss(BaseIntervalScorer):
    r"""Pinball Loss (Quantile Score) for prediction intervals.

    Evaluates quantile forecasts with asymmetric penalty. Each interval bound
    corresponds to a quantile: lower bound = (1-α)/2, upper bound = (1+α)/2.

    The pinball loss for quantile τ is:

    $$\\text{Pinball}(\\tau, y, q) = \\begin{cases}
    \\tau(y - q) & \\text{if } y \\geq q \\\\
    (1-\\tau)(q - y) & \\text{if } y < q
    \\end{cases}$$

    For interval forecasts at coverage α, the total pinball loss is the sum of
    losses for lower (τ=(1-α)/2) and upper (τ=(1+α)/2) bounds.

    Parameters
    ----------
    aggregation_method : list of str or str, default="all"
        Dimensions to collapse when aggregating scores. Orthogonal modes:

        - "stepwise": Collapse the forecasting-step dimension.
        - "vintagewise": Collapse the vintage/observed-time dimension.
        - "componentwise": Collapse components, return per-timestep scores.
        - "groupwise": Collapse panel groups (panel data only).
        - "coveragewise": Collapse coverage rates (return average pinball loss).

        - "all": Collapse all dimensions (returns scalar).
    coverage : list of float, dict of float to float, or None, default=None
        Coverage rate filter (list) or filter with weights (dict).
    groups : list of str, dict of str to float, or None, default=None
        Panel group filter (list) or filter with weights (dict).
    components : list of str, dict of str to float, or None, default=None
        Component filter (list) or filter with weights (dict).

    Attributes
    ----------
    lower_is_better : bool
        True for pinball loss (lower is better).

    Examples
    --------
    >>> import polars as pl
    >>> from datetime import datetime
    >>> from yohou.metrics import PinballLoss
    >>> y_true = pl.DataFrame({
    ...     "time": [datetime(2020, 1, 1), datetime(2020, 1, 2)],
    ...     "value": [10.0, 20.0]
    ... })
    >>> y_pred = pl.DataFrame({
    ...     "observed_time": [datetime(2019, 12, 31)] * 2,
    ...     "time": [datetime(2020, 1, 1), datetime(2020, 1, 2)],
    ...     "value_lower_0.9": [8.0, 18.0],
    ...     "value_upper_0.9": [12.0, 22.0]
    ... })
    >>> loss = PinballLoss()
    >>> _ = loss.fit(y_true)
    >>> loss.score(y_true, y_pred)  # doctest: +ELLIPSIS
    0.2...

    Notes
    -----
    - Lower is better
    - Penalizes under-prediction differently than over-prediction
    - Scale-dependent
    - More relevant for quantile regression forecasters
    - Useful when asymmetric errors have different costs

    See Also
    --------
    `IntervalScore` : Symmetric penalty for coverage violations
    `EmpiricalCoverage` : Coverage-only metric

    """

    _parameter_constraints: dict = {
        **BaseIntervalScorer._parameter_constraints,
    }

    _metric_name = "loss"

    def __init__(
        self,
        aggregation_method: list[str] | str = "all",
        coverage: list[float] | dict[float, float] | None = None,
        groups: list[str] | dict[str, float] | None = None,
        components: list[str] | dict[str, float] | None = None,
    ) -> None:
        agg_list = aggregation_method
        if aggregation_method == "all":
            agg_list = ["stepwise", "vintagewise", "componentwise", "groupwise", "coveragewise"]

        super().__init__(
            aggregation_method=agg_list,
            coverage=coverage,
            groups=groups,
            components=components,
        )

    def _pinball(self, tau: float, y: pl.Series | float, q: pl.Series | float) -> pl.Expr | pl.Series | float:
        """Compute pinball loss for a single quantile.

        Parameters
        ----------
        tau : float
            Quantile level.
        y : pl.Series or float
            True values.
        q : pl.Series or float
            Quantile forecasts.

        Returns
        -------
        pl.Series or float
            Pinball loss.

        """
        if isinstance(y, pl.Series):
            loss = pl.when(y >= q).then(tau * (y - q)).otherwise((1 - tau) * (q - y))
        else:
            loss = tau * (y - q) if y >= q else (1 - tau) * (q - y)
        return loss

    def _compute_raw_scores(self, y_truth, y_pred, coverage_rates, target_columns):
        """Compute per-row pinball loss values."""
        frames = []
        for rate in coverage_rates:
            tau_lower = (1 - rate) / 2
            tau_upper = (1 + rate) / 2
            rate_data = {}
            for col in target_columns:
                lower_col = f"{col}_lower_{rate}"
                upper_col = f"{col}_upper_{rate}"
                if lower_col in y_pred.columns and upper_col in y_pred.columns:
                    loss_lower = self._pinball(tau_lower, y_truth[col], y_pred[lower_col])
                    loss_upper = self._pinball(tau_upper, y_truth[col], y_pred[upper_col])
                    loss_lower_series = y_pred.select(
                        loss_lower.alias("loss_lower")  # ty: ignore[unresolved-attribute]
                    )["loss_lower"]
                    loss_upper_series = y_pred.select(
                        loss_upper.alias("loss_upper")  # ty: ignore[unresolved-attribute]
                    )["loss_upper"]
                    rate_data[col] = loss_lower_series + loss_upper_series
            frames.append(pl.DataFrame(rate_data).with_columns(pl.lit(rate).alias("coverage_rate")))
        return pl.concat(frames)


class CalibrationError(BaseIntervalScorer):
    r"""Calibration Error for prediction intervals.

    Measures the discrepancy between nominal coverage rate and empirical coverage
    across different rates. Indicates if intervals are well-calibrated.

    The calibration error is:

    $$\\text{CalibError} = \\frac{1}{K}\\sum_{k=1}^{K} |\\text{Coverage}(\\alpha_k) - \\alpha_k|$$

    where K is the number of coverage rates.

    Parameters
    ----------
    aggregation_method : list of str or str, default="all"
        Dimensions to collapse when aggregating scores. Orthogonal modes:

        - "stepwise": Collapse the forecasting-step dimension.
        - "vintagewise": Collapse the vintage/observed-time dimension.
        - "componentwise": Collapse components, return per-timestep scores.
        - "groupwise": Collapse panel groups (panel data only).
        - "coveragewise": Collapse coverage rates (return average calibration error).

        - "all": Collapse all dimensions (returns scalar).
    coverage : list of float, dict of float to float, or None, default=None
        Coverage rate filter (list) or filter with weights (dict).
    groups : list of str, dict of str to float, or None, default=None
        Panel group filter (list) or filter with weights (dict).
    components : list of str, dict of str to float, or None, default=None
        Component filter (list) or filter with weights (dict).

    Attributes
    ----------
    lower_is_better : bool
        True for calibration error (lower is better).

    Examples
    --------
    >>> import polars as pl
    >>> from datetime import datetime
    >>> from yohou.metrics import CalibrationError
    >>> y_true = pl.DataFrame({"time": [datetime(2020, 1, 1), datetime(2020, 1, 2)], "value": [10.0, 20.0]})
    >>> y_pred = pl.DataFrame({
    ...     "observed_time": [datetime(2019, 12, 31)] * 2,
    ...     "time": [datetime(2020, 1, 1), datetime(2020, 1, 2)],
    ...     "value_lower_0.9": [8.0, 18.0],
    ...     "value_upper_0.9": [12.0, 22.0],
    ...     "value_lower_0.95": [7.0, 17.0],
    ...     "value_upper_0.95": [13.0, 23.0],
    ... })
    >>> error = CalibrationError()
    >>> _ = error.fit(y_true)
    >>> error.score(y_true, y_pred)  # doctest: +ELLIPSIS
    0.0...

    Notes
    -----
    - Lower is better (0 = perfect calibration)
    - Aggregates coverage errors across all rates
    - Scale-independent (always between 0 and 1)
    - Requires at least 2 coverage rates for meaningful metric
    - Missing values are excluded from computation

    See Also
    --------
    `EmpiricalCoverage` : Per-rate coverage metric
    `IntervalScore` : Combined coverage and sharpness metric

    """

    _parameter_constraints: dict = {
        **BaseIntervalScorer._parameter_constraints,
    }

    _metric_name = "calibration_error"

    def __init__(
        self,
        aggregation_method: list[str] | str = "all",
        coverage: list[float] | dict[float, float] | None = None,
        groups: list[str] | dict[str, float] | None = None,
        components: list[str] | dict[str, float] | None = None,
    ) -> None:
        agg_list = aggregation_method
        if aggregation_method == "all":
            agg_list = ["stepwise", "vintagewise", "componentwise", "groupwise", "coveragewise"]

        super().__init__(
            aggregation_method=agg_list,
            coverage=coverage,
            groups=groups,
            components=components,
        )

    def _compute_raw_scores(self, y_truth, y_pred, coverage_rates, target_columns):
        """Compute per-row calibration error values."""
        frames = []
        for rate in coverage_rates:
            rate_data = {}
            for col in target_columns:
                lower_col = f"{col}_lower_{rate}"
                upper_col = f"{col}_upper_{rate}"
                if lower_col in y_pred.columns and upper_col in y_pred.columns:
                    in_interval = (y_truth[col] >= y_pred[lower_col]) & (y_truth[col] <= y_pred[upper_col])
                    rate_data[col] = (in_interval.cast(pl.Float64) - rate).abs()
            frames.append(pl.DataFrame(rate_data).with_columns(pl.lit(rate).alias("coverage_rate")))
        return pl.concat(frames)

    def score(  # type: ignore
        self, y_truth: pl.DataFrame, y_pred: pl.DataFrame, /, **params
    ) -> pl.DataFrame | float | dict[str | float, float | pl.DataFrame]:
        """Compute calibration error.

        Parameters
        ----------
        y_truth : pl.DataFrame
            True values with "time" column.
        y_pred : pl.DataFrame
            Predicted intervals with "{col}_lower_{rate}", "{col}_upper_{rate}" columns.
        **params : dict
            Metadata to route to nested estimators.

        Returns
        -------
        float or pl.DataFrame
            Calibration error score.

        Raises
        ------
        ValueError
            If fewer than 2 coverage rates are provided.

        """
        # Validate minimum coverage rates before delegating
        rates = self._extract_coverage_rates(y_pred)
        if len(rates) < 2:
            msg = (
                f"CalibrationError requires at least 2 coverage rates, "
                f"but only {len(rates)} provided. "
                f"Use multiple coverage rates to compute calibration error."
            )
            raise ValueError(msg)

        return super().score(y_truth, y_pred, **params)
