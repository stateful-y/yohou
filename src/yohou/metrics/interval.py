"""Metrics for evaluating interval forecast quality."""

import polars as pl
from sklearn.utils.validation import check_is_fitted

from yohou.utils import validate_scorer_data

from .base import BaseIntervalScorer


class EmpiricalCoverage(BaseIntervalScorer):
    """Empirical coverage rate for prediction intervals.

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
        Dimensions to aggregate over. Options:
        - "timewise": Aggregate across time, return per-component DataFrame
        - "componentwise": Aggregate across components, return per-timestep DataFrame
        - "groupwise": Aggregate across panel groups (panel data only)
        - "coveragewise": Aggregate across coverage rates (return average coverage)
        - "all": Aggregate across all dimensions (returns scalar). Same as
          ["timewise", "componentwise", "groupwise", "coveragewise"].
        Example outputs:
        - "timewise" or ["timewise"]: Per-component (and per-group) DataFrame.
        - "componentwise" or ["componentwise"]: Per-timestep (and per-group) DataFrame.
        - "groupwise" or ["groupwise"]: Per-component per-timestep DataFrame (panel aggregated).
        - ["timewise", "componentwise"]: Scalar (global) or per-group DataFrame (panel).
        - "all": Scalar float (hierarchically aggregated for panel data).
    coverage_rates : list of float or None, default=None
        List of coverage rates to include in scoring. If None, all coverage rates
        are included. Rates are validated against actual prediction columns during scoring.
    panel_group_names : list of str or None, default=None
        List of panel group names to include in scoring. If None, all panel groups
        are included. Only applicable for panel data.
    component_names : list of str or None, default=None
        List of component (target column) names to include in scoring. If None, all
        components are included. For panel data, these are unprefixed column names.
    panel_group_weight : dict or None, default=None
        Dictionary mapping panel group names to weights for weighted aggregation.
        If None, all panel groups weighted equally. Only applicable for panel data.

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
    ...     "value": [10.0, 20.0, 30.0]
    ... })
    >>> y_pred = pl.DataFrame({
    ...     "observed_time": [datetime(2019, 12, 31)] * 3,
    ...     "time": [datetime(2020, 1, 1), datetime(2020, 1, 2), datetime(2020, 1, 3)],
    ...     "value_lower_0.9": [8.0, 18.0, 28.0],
    ...     "value_upper_0.9": [12.0, 22.0, 32.0]
    ... })
    >>> coverage = EmpiricalCoverage()
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
    MeanIntervalWidth : Evaluates interval sharpness
    IntervalScore : Combined coverage and sharpness metric
    CalibrationError : Aggregate miscalibration metric

    """

    _parameter_constraints: dict = {
        **BaseIntervalScorer._parameter_constraints,
    }

    def __init__(
        self,
        aggregation_method: list[str] | str = "all",
        panel_group_names: list[str] | None = None,
        component_names: list[str] | None = None,
        coverage_rates: list[float] | None = None,
        panel_group_weight: dict[str, float] | None = None,
    ) -> None:
        agg_list = aggregation_method
        if aggregation_method == "all":
            agg_list = ["timewise", "componentwise", "groupwise", "coveragewise"]

        super().__init__(
            aggregation_method=agg_list,
            panel_group_names=panel_group_names,
            component_names=component_names,
            coverage_rates=coverage_rates,
            panel_group_weight=panel_group_weight,
        )

    def score(
        self, y_truth: pl.DataFrame, y_pred: pl.DataFrame, **params
    ) -> float | dict[float, float] | pl.DataFrame:
        """Compute empirical coverage rate.

        Parameters
        ----------
        y_truth : pl.DataFrame
            True values with "time" column.
        y_pred : pl.DataFrame
            Predicted intervals with "observed_time", "time", and
            "{col}_lower_{rate}", "{col}_upper_{rate}" columns.
        **params : dict
            Metadata to route to nested estimators.

        Returns
        -------
        float or dict or pl.DataFrame
            If aggregation_method=["timewise"], returns DataFrame with per-component coverage.
            If aggregation_method=["componentwise"], returns DataFrame with "time" column and coverage column(s).
            If aggregation_method="all" (including "coveragewise"), returns average coverage (float).
            If aggregation_method excludes "coveragewise", returns dict {coverage_rate: empirical_coverage}.

        """
        check_is_fitted(self, ["_is_fitted"])

        y_truth, y_pred, time_values = validate_scorer_data(
            self,
            y_truth,
            y_pred,
        )

        coverage_rates = self._extract_coverage_rates(y_pred)
        target_columns = self._extract_target_columns(y_truth)

        # Compute raw per-timestep per-component per-rate coverage (0 or 1)
        scores_per_rate = {}
        for rate in coverage_rates:
            rate_data = {}
            for col in target_columns:
                lower_col = f"{col}_lower_{rate}"
                upper_col = f"{col}_upper_{rate}"

                if lower_col in y_pred.columns and upper_col in y_pred.columns:
                    # Compute binary coverage indicators
                    in_interval = (y_truth[col] >= y_pred[lower_col]) & (
                        y_truth[col] <= y_pred[upper_col]
                    )
                    # Convert boolean to float (0.0 or 1.0)
                    rate_data[col] = in_interval.cast(pl.Float64)

            scores_per_rate[rate] = pl.DataFrame(rate_data)

        # Apply aggregation strategy from base class
        result = self._aggregate_scores(scores_per_rate, time_values=time_values)

        # Customize column names for componentwise aggregation if needed
        if (
            "componentwise"
            in (self.aggregation_method if isinstance(self.aggregation_method, list) else [])
            and isinstance(result, pl.DataFrame)
            and not isinstance(result, dict)
        ):
            # Rename global and panel columns
            rename_map = {}
            if "score" in result.columns:
                rename_map["score"] = "coverage"
            for col in result.columns:
                if col.endswith("__score"):
                    rename_map[col] = col.replace("__score", "__coverage")

            if rename_map:
                result = result.rename(rename_map)

        # Customize column names for timewise aggregation if needed
        if (
            "timewise"
            in (self.aggregation_method if isinstance(self.aggregation_method, list) else [])
            and isinstance(result, pl.DataFrame)
            and not isinstance(result, dict)
        ):
            result = result.rename({col: f"coverage_{col}" for col in result.columns})

        return result


class MeanIntervalWidth(BaseIntervalScorer):
    """Mean width of prediction intervals.

    Measures the average width of prediction intervals. Narrower intervals are
    preferred (more informative), provided coverage is maintained.

    The mean interval width for coverage rate α is:

    $$\\text{Width}(\\alpha) = \\frac{1}{n}\\sum_{i=1}^{n} |U_i(\\alpha) - L_i(\\alpha)|$$

    Parameters
    ----------
    aggregation_method : list of str or str, default="all"
        Dimensions to aggregate over. Options:
        - "timewise": Aggregate across time, return per-component DataFrame
        - "componentwise": Aggregate across components, return per-timestep DataFrame
        - "groupwise": Aggregate across panel groups (panel data only)
        - "coveragewise": Aggregate across coverage rates (return average coverage)
        - "all": Aggregate across all dimensions (returns scalar). Same as
          ["timewise", "componentwise", "groupwise", "coveragewise"].
        Example outputs:
        - "timewise" or ["timewise"]: Per-component (and per-group) DataFrame.
        - "componentwise" or ["componentwise"]: Per-timestep (and per-group) DataFrame.
        - "groupwise" or ["groupwise"]: Per-component per-timestep DataFrame (panel aggregated).
        - ["timewise", "componentwise"]: Scalar (global) or per-group DataFrame (panel).
        - "all": Scalar float (hierarchically aggregated for panel data).
    coverage_rates : list of float or None, default=None
        List of coverage rates to include in scoring. If None, all coverage rates
        are included. Rates are validated against actual prediction columns during scoring.
    panel_group_names : list of str or None, default=None
        List of panel group names to include in scoring. If None, all panel groups
        are included. Only applicable for panel data.
    component_names : list of str or None, default=None
        List of component (target column) names to include in scoring. If None, all
        components are included. For panel data, these are unprefixed column names.
    panel_group_weight : dict or None, default=None
        Dictionary mapping panel group names to weights for weighted aggregation.
        If None, all panel groups weighted equally. Only applicable for panel data.

    Attributes
    ----------
    lower_is_better : bool
        True for width (narrower intervals are better).

    Examples
    --------
    >>> import polars as pl
    >>> from datetime import datetime
    >>> from yohou.metrics import MeanIntervalWidth
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
    >>> width = MeanIntervalWidth()
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
    EmpiricalCoverage : Evaluates interval calibration
    IntervalScore : Combined coverage and sharpness metric

    """

    _parameter_constraints: dict = {
        **BaseIntervalScorer._parameter_constraints,
    }

    def __init__(
        self,
        aggregation_method: list[str] | str = "all",
        panel_group_names: list[str] | None = None,
        component_names: list[str] | None = None,
        coverage_rates: list[float] | None = None,
        panel_group_weight: dict[str, float] | None = None,
    ) -> None:
        agg_list = aggregation_method
        if aggregation_method == "all":
            agg_list = ["timewise", "componentwise", "groupwise", "coveragewise"]

        super().__init__(
            aggregation_method=agg_list,
            panel_group_names=panel_group_names,
            component_names=component_names,
            coverage_rates=coverage_rates,
            panel_group_weight=panel_group_weight,
        )

    def score(
        self, y_truth: pl.DataFrame, y_pred: pl.DataFrame, **params
    ) -> float | dict[float, float] | pl.DataFrame:
        """Compute mean interval width.

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
        float or dict or pl.DataFrame
            If aggregation_method=["timewise"], returns DataFrame with per-component widths.
            If aggregation_method=["componentwise"], returns DataFrame with "time" column and width column(s).
            If aggregation_method="all" (including "coveragewise"), returns average width (float).
            If aggregation_method excludes "coveragewise", returns dict {coverage_rate: mean_width}.

        """
        check_is_fitted(self, ["_is_fitted"])

        y_truth, y_pred, time_values = validate_scorer_data(
            self,
            y_truth,
            y_pred,
        )

        coverage_rates = self._extract_coverage_rates(y_pred)
        target_columns = self._extract_target_columns(y_truth)

        # Compute raw per-timestep per-component per-rate widths
        scores_per_rate = {}
        for rate in coverage_rates:
            rate_data = {}
            for col in target_columns:
                lower_col = f"{col}_lower_{rate}"
                upper_col = f"{col}_upper_{rate}"

                if lower_col in y_pred.columns and upper_col in y_pred.columns:
                    # Compute interval width
                    widths = (y_pred[upper_col] - y_pred[lower_col]).abs()
                    rate_data[col] = widths

            scores_per_rate[rate] = pl.DataFrame(rate_data)

        # Apply aggregation strategy from base class
        result = self._aggregate_scores(scores_per_rate, time_values=time_values)

        # Customize column names for componentwise aggregation if needed
        if "componentwise" in (
            self.aggregation_method if isinstance(self.aggregation_method, list) else []
        ) and isinstance(result, pl.DataFrame):
            # Rename global and panel columns
            rename_map = {}
            if "score" in result.columns:
                rename_map["score"] = "width"
            for col in result.columns:
                if col.endswith("__score"):
                    rename_map[col] = col.replace("__score", "__width")

            if rename_map:
                result = result.rename(rename_map)

        # Customize column names for timewise aggregation if needed
        if "timewise" in (
            self.aggregation_method if isinstance(self.aggregation_method, list) else []
        ) and isinstance(result, pl.DataFrame):
            result = result.rename({col: f"width_{col}" for col in result.columns})

        return result


class IntervalScore(BaseIntervalScorer):
    """Interval Score (Winkler Score) for prediction intervals.

    Combines interval width with penalties for observations falling outside
    the interval. Balances sharpness and coverage in a single metric.

    The interval score for coverage rate α is:

    $$\\text{IS}(\\alpha) = \\frac{1}{n}\\sum_{i=1}^{n} \\left[|U_i - L_i| + \\frac{2}{\\alpha}(L_i - y_i)\\mathbb{1}(y_i < L_i) + \\frac{2}{\\alpha}(y_i - U_i)\\mathbb{1}(y_i > U_i)\\right]$$

    where the first term is interval width and the second/third terms are penalties
    for under/over-coverage.

    Parameters
    ----------
    aggregation_method : list of str or str, default="all"
        Dimensions to aggregate over. Options:
        - "timewise": Aggregate across time, return per-component DataFrame
        - "componentwise": Aggregate across components, return per-timestep DataFrame
        - "groupwise": Aggregate across panel groups (panel data only)
        - "coveragewise": Aggregate across coverage rates (return average interval score)
        - "all": Aggregate across all dimensions (returns scalar). Same as
          ["timewise", "componentwise", "groupwise", "coveragewise"].
        Example outputs:
        - "timewise" or ["timewise"]: Per-component (and per-group) DataFrame.
        - "componentwise" or ["componentwise"]: Per-timestep (and per-group) DataFrame.
        - "groupwise" or ["groupwise"]: Per-component per-timestep DataFrame (panel aggregated).
        - ["timewise", "componentwise"]: Scalar (global) or per-group DataFrame (panel).
        - "all": Scalar float (hierarchically aggregated for panel data).
    coverage_rates : list of float or None, default=None
        List of coverage rates to include in scoring. If None, all coverage rates
        are included. Rates are validated against actual prediction columns during scoring.
    panel_group_names : list of str or None, default=None
        List of panel group names to include in scoring. If None, all panel groups
        are included. Only applicable for panel data.
    component_names : list of str or None, default=None
        List of component (target column) names to include in scoring. If None, all
        components are included. For panel data, these are unprefixed column names.
    panel_group_weight : dict or None, default=None
        Dictionary mapping panel group names to weights for weighted aggregation.
        If None, all panel groups weighted equally. Only applicable for panel data.

    Attributes
    ----------
    lower_is_better : bool
        True for interval score (lower is better).

    Examples
    --------
    >>> import polars as pl
    >>> from datetime import datetime
    >>> from yohou.metrics import IntervalScore
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
    >>> scorer = IntervalScore()
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
    EmpiricalCoverage : Coverage-only metric
    MeanIntervalWidth : Width-only metric
    PinballLoss : Asymmetric quantile-based metric

    """

    _parameter_constraints: dict = {
        **BaseIntervalScorer._parameter_constraints,
    }

    def __init__(
        self,
        aggregation_method: list[str] | str = "all",
        panel_group_names: list[str] | None = None,
        component_names: list[str] | None = None,
        coverage_rates: list[float] | None = None,
        panel_group_weight: dict[str, float] | None = None,
    ) -> None:
        agg_list = aggregation_method
        if aggregation_method == "all":
            agg_list = ["timewise", "componentwise", "groupwise", "coveragewise"]

        super().__init__(
            aggregation_method=agg_list,
            panel_group_names=panel_group_names,
            component_names=component_names,
            coverage_rates=coverage_rates,
            panel_group_weight=panel_group_weight,
        )

    def score(
        self, y_truth: pl.DataFrame, y_pred: pl.DataFrame, **params
    ) -> float | dict[float, float] | pl.DataFrame:
        """Compute interval score.

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
        float or dict or pl.DataFrame
            If aggregation_method=["timewise"], returns DataFrame with per-component scores.
            If aggregation_method=["componentwise"], returns DataFrame with "time" column and score column(s).
            If aggregation_method="all" (including "coveragewise"), returns average score (float).
            If aggregation_method excludes "coveragewise", returns dict {coverage_rate: interval_score}.

        """
        check_is_fitted(self, ["_is_fitted"])

        y_truth, y_pred, time_values = validate_scorer_data(
            self,
            y_truth,
            y_pred,
        )

        coverage_rates = self._extract_coverage_rates(y_pred)
        target_columns = self._extract_target_columns(y_truth)

        # Compute raw per-timestep per-component per-rate interval scores
        scores_per_rate = {}
        for rate in coverage_rates:
            rate_data = {}
            for col in target_columns:
                lower_col = f"{col}_lower_{rate}"
                upper_col = f"{col}_upper_{rate}"

                if lower_col in y_pred.columns and upper_col in y_pred.columns:
                    # Width term
                    width = (y_pred[upper_col] - y_pred[lower_col]).abs()

                    # Penalty terms (need to extract series from expressions)
                    lower_penalty = (
                        pl.when(y_truth[col] < y_pred[lower_col])
                        .then((2.0 / rate) * (y_pred[lower_col] - y_truth[col]))
                        .otherwise(0.0)
                    )

                    upper_penalty = (
                        pl.when(y_truth[col] > y_pred[upper_col])
                        .then((2.0 / rate) * (y_truth[col] - y_pred[upper_col]))
                        .otherwise(0.0)
                    )

                    # Evaluate expressions to get Series
                    lower_penalty_series = y_pred.select(lower_penalty.alias("lower_penalty"))[
                        "lower_penalty"
                    ]
                    upper_penalty_series = y_pred.select(upper_penalty.alias("upper_penalty"))[
                        "upper_penalty"
                    ]

                    # Total interval score per timestep
                    rate_data[col] = width + lower_penalty_series + upper_penalty_series

            scores_per_rate[rate] = pl.DataFrame(rate_data)

        # Apply aggregation strategy from base class
        result = self._aggregate_scores(scores_per_rate, time_values=time_values)

        # Customize column names if needed
        if "componentwise" in (
            self.aggregation_method if isinstance(self.aggregation_method, list) else []
        ) and isinstance(result, pl.DataFrame):
            # Rename global and panel columns
            rename_map = {}
            if "score" in result.columns:
                rename_map["score"] = "interval_score"

            for col in result.columns:
                if col.endswith("__score"):
                    rename_map[col] = col.replace("__score", "__interval_score")

            if rename_map:
                result = result.rename(rename_map)
        elif "timewise" in (
            self.aggregation_method if isinstance(self.aggregation_method, list) else []
        ) and isinstance(result, pl.DataFrame):
            result = result.rename({col: f"interval_score_{col}" for col in result.columns})

        return result


class PinballLoss(BaseIntervalScorer):
    """Pinball Loss (Quantile Score) for prediction intervals.

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
        Dimensions to aggregate over. Options:
        - "timewise": Aggregate across time, return per-component DataFrame
        - "componentwise": Aggregate across components, return per-timestep DataFrame
        - "groupwise": Aggregate across panel groups (panel data only)
        - "coveragewise": Aggregate across coverage rates (return average pinball loss)
        - "all": Aggregate across all dimensions (returns scalar). Same as
          ["timewise", "componentwise", "groupwise", "coveragewise"].
        Example outputs:
        - "timewise" or ["timewise"]: Per-component (and per-group) DataFrame.
        - "componentwise" or ["componentwise"]: Per-timestep (and per-group) DataFrame.
        - "groupwise" or ["groupwise"]: Per-component per-timestep DataFrame (panel aggregated).
        - ["timewise", "componentwise"]: Scalar (global) or per-group DataFrame (panel).
        - "all": Scalar float (hierarchically aggregated for panel data).
    coverage_rates : list of float or None, default=None
        List of coverage rates to include in scoring. If None, all coverage rates
        are included. Rates are validated against actual prediction columns during scoring.
    panel_group_names : list of str or None, default=None
        List of panel group names to include in scoring. If None, all panel groups
        are included. Only applicable for panel data.
    component_names : list of str or None, default=None
        List of component (target column) names to include in scoring. If None, all
        components are included. For panel data, these are unprefixed column names.
    panel_group_weight : dict or None, default=None
        Dictionary mapping panel group names to weights for weighted aggregation.
        If None, all panel groups weighted equally. Only applicable for panel data.

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
    IntervalScore : Symmetric penalty for coverage violations
    EmpiricalCoverage : Coverage-only metric

    """

    _parameter_constraints: dict = {
        **BaseIntervalScorer._parameter_constraints,
    }

    def __init__(
        self,
        aggregation_method: list[str] | str = "all",
        panel_group_names: list[str] | None = None,
        component_names: list[str] | None = None,
        coverage_rates: list[float] | None = None,
        panel_group_weight: dict[str, float] | None = None,
    ) -> None:
        agg_list = aggregation_method
        if aggregation_method == "all":
            agg_list = ["timewise", "componentwise", "groupwise", "coveragewise"]

        super().__init__(
            aggregation_method=agg_list,
            panel_group_names=panel_group_names,
            component_names=component_names,
            coverage_rates=coverage_rates,
            panel_group_weight=panel_group_weight,
        )

    def _pinball(self, tau: float, y: pl.Series | float, q: pl.Series | float) -> pl.Series | float:
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

    def score(
        self, y_truth: pl.DataFrame, y_pred: pl.DataFrame, **params
    ) -> float | dict[float, float] | pl.DataFrame:
        """Compute pinball loss.

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
        float or dict or pl.DataFrame
            If aggregate="timewise", returns DataFrame with per-component losses.
            If aggregate="componentwise", returns DataFrame with "time" column and loss column(s).
            If aggregate="both" (including coveragewise), returns average loss (float).
            If aggregate excludes coveragewise, returns dict {coverage_rate: pinball_loss}.

        """
        check_is_fitted(self, ["_is_fitted"])

        y_truth, y_pred, time_values = validate_scorer_data(
            self,
            y_truth,
            y_pred,
        )

        coverage_rates = self._extract_coverage_rates(y_pred)
        target_columns = self._extract_target_columns(y_truth)

        # Compute raw per-timestep per-component per-rate pinball losses
        scores_per_rate = {}
        for rate in coverage_rates:
            tau_lower = (1 - rate) / 2
            tau_upper = (1 + rate) / 2
            rate_data = {}

            for col in target_columns:
                lower_col = f"{col}_lower_{rate}"
                upper_col = f"{col}_upper_{rate}"

                if lower_col in y_pred.columns and upper_col in y_pred.columns:
                    # Compute pinball loss for both bounds
                    loss_lower = self._pinball(tau_lower, y_truth[col], y_pred[lower_col])
                    loss_upper = self._pinball(tau_upper, y_truth[col], y_pred[upper_col])

                    # Extract series from polars expressions
                    loss_lower_series = y_pred.select(loss_lower.alias("loss_lower"))["loss_lower"]
                    loss_upper_series = y_pred.select(loss_upper.alias("loss_upper"))["loss_upper"]

                    # Sum losses for upper and lower bounds
                    rate_data[col] = loss_lower_series + loss_upper_series

            scores_per_rate[rate] = pl.DataFrame(rate_data)

        # Apply aggregation strategy from base class
        result = self._aggregate_scores(scores_per_rate, time_values=time_values)

        # Customize column names if needed
        if (
            "componentwise"
            in (self.aggregation_method if isinstance(self.aggregation_method, list) else [])
            and isinstance(result, pl.DataFrame)
            and not isinstance(result, dict)
        ):
            # Rename global and panel columns
            rename_map = {}
            if "score" in result.columns:
                rename_map["score"] = "loss"
            for col in result.columns:
                if col.endswith("__score"):
                    rename_map[col] = col.replace("__score", "__loss")

            if rename_map:
                result = result.rename(rename_map)
        elif (
            "timewise"
            in (self.aggregation_method if isinstance(self.aggregation_method, list) else [])
            and isinstance(result, pl.DataFrame)
            and not isinstance(result, dict)
        ):
            result = result.rename({col: f"loss_{col}" for col in result.columns})

        return result


class CalibrationError(BaseIntervalScorer):
    """Calibration Error for prediction intervals.

    Measures the discrepancy between nominal coverage rate and empirical coverage
    across different rates. Indicates if intervals are well-calibrated.

    The calibration error is:

    $$\\text{CalibError} = \\frac{1}{K}\\sum_{k=1}^{K} |\\text{Coverage}(\\alpha_k) - \\alpha_k|$$

    where K is the number of coverage rates.

    Parameters
    ----------
    aggregation_method : list of str or str, default="all"
        Dimensions to aggregate over. Options:
        - "timewise": Aggregate across time, return per-component DataFrame
        - "componentwise": Aggregate across components, return per-timestep DataFrame
        - "groupwise": Aggregate across panel groups (panel data only)
        - "coveragewise": Aggregate across coverage rates (return average calibration error)
        - "all": Aggregate across all dimensions (returns scalar). Same as
          ["timewise", "componentwise", "groupwise", "coveragewise"].
        Example outputs:
        - "timewise" or ["timewise"]: Per-component (and per-group) DataFrame.
        - "componentwise" or ["componentwise"]: Per-timestep (and per-group) DataFrame.
        - "groupwise" or ["groupwise"]: Per-component per-timestep DataFrame (panel aggregated).
        - ["timewise", "componentwise"]: Scalar (global) or per-group DataFrame (panel).
        - "all": Scalar float (hierarchically aggregated for panel data).
    coverage_rates : list of float or None, default=None
        List of coverage rates to include in scoring. If None, all coverage rates
        are included. Rates are validated against actual prediction columns during scoring.
    panel_group_names : list of str or None, default=None
        List of panel group names to include in scoring. If None, all panel groups
        are included. Only applicable for panel data.
    component_names : list of str or None, default=None
        List of component (target column) names to include in scoring. If None, all
        components are included. For panel data, these are unprefixed column names.
    panel_group_weight : dict or None, default=None
        Dictionary mapping panel group names to weights for weighted aggregation.
        If None, all panel groups weighted equally. Only applicable for panel data.

    Attributes
    ----------
    lower_is_better : bool
        True for calibration error (lower is better).

    Examples
    --------
    >>> import polars as pl
    >>> from datetime import datetime
    >>> from yohou.metrics import CalibrationError
    >>> y_true = pl.DataFrame({
    ...     "time": [datetime(2020, 1, 1), datetime(2020, 1, 2)],
    ...     "value": [10.0, 20.0]
    ... })
    >>> y_pred = pl.DataFrame({
    ...     "observed_time": [datetime(2019, 12, 31)] * 2,
    ...     "time": [datetime(2020, 1, 1), datetime(2020, 1, 2)],
    ...     "value_lower_0.9": [8.0, 18.0],
    ...     "value_upper_0.9": [12.0, 22.0],
    ...     "value_lower_0.95": [7.0, 17.0],
    ...     "value_upper_0.95": [13.0, 23.0]
    ... })
    >>> error = CalibrationError()
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
    EmpiricalCoverage : Per-rate coverage metric
    IntervalScore : Combined coverage and sharpness metric

    """

    _parameter_constraints: dict = {
        **BaseIntervalScorer._parameter_constraints,
    }

    def __init__(
        self,
        aggregation_method: list[str] | str = "all",
        panel_group_names: list[str] | None = None,
        component_names: list[str] | None = None,
        coverage_rates: list[float] | None = None,
        panel_group_weight: dict[str, float] | None = None,
    ) -> None:
        agg_list = aggregation_method
        if aggregation_method == "all":
            agg_list = ["timewise", "componentwise", "groupwise", "coveragewise"]

        super().__init__(
            aggregation_method=agg_list,
            panel_group_names=panel_group_names,
            component_names=component_names,
            coverage_rates=coverage_rates,
            panel_group_weight=panel_group_weight,
        )

    def score(
        self, y_truth: pl.DataFrame, y_pred: pl.DataFrame, **params
    ) -> float | dict[float, float] | pl.DataFrame:
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
            If aggregate="timewise", returns DataFrame with per-component calibration errors.
            If aggregate="componentwise", returns DataFrame with "time" and "calibration_error" columns.
            If aggregate="both", returns average calibration error (float).

        Raises
        ------
        ValueError
            If fewer than 2 coverage rates are provided.

        """
        check_is_fitted(self, ["_is_fitted"])

        y_truth, y_pred, time_values = validate_scorer_data(
            self,
            y_truth,
            y_pred,
        )

        coverage_rates = self._extract_coverage_rates(y_pred)

        if len(coverage_rates) < 2:
            msg = (
                f"CalibrationError requires at least 2 coverage rates, "
                f"but only {len(coverage_rates)} provided. "
                f"Use multiple coverage rates to compute calibration error."
            )
            raise ValueError(msg)

        target_columns = self._extract_target_columns(y_truth)

        # Compute raw per-timestep per-component per-rate calibration errors
        scores_per_rate = {}
        for rate in coverage_rates:
            rate_data = {}
            for col in target_columns:
                lower_col = f"{col}_lower_{rate}"
                upper_col = f"{col}_upper_{rate}"

                if lower_col in y_pred.columns and upper_col in y_pred.columns:
                    # Check if value is within interval
                    in_interval = (y_truth[col] >= y_pred[lower_col]) & (
                        y_truth[col] <= y_pred[upper_col]
                    )
                    # Compute empirical coverage across time for this timestep
                    # For per-timestep metrics, we need the absolute error
                    # Cast to float and compute |coverage - rate| per timestep
                    # Since coverage at single timestep is just 0 or 1, we compute |indicator - rate|
                    rate_data[col] = in_interval.cast(pl.Float64).abs() - rate
                    rate_data[col] = rate_data[col].abs()

            scores_per_rate[rate] = pl.DataFrame(rate_data)

        # Apply aggregation strategy from base class
        result = self._aggregate_scores(scores_per_rate, time_values=time_values)

        # Customize column names if needed
        if (
            "componentwise"
            in (self.aggregation_method if isinstance(self.aggregation_method, list) else [])
            and isinstance(result, pl.DataFrame)
            and not isinstance(result, dict)
        ):
            # Rename global and panel columns
            rename_map = {}
            if "score" in result.columns:
                rename_map["score"] = "calibration_error"
            for col in result.columns:
                if col.endswith("__score"):
                    rename_map[col] = col.replace("__score", "__calibration_error")

            if rename_map:
                result = result.rename(rename_map)
        elif (
            "timewise"
            in (self.aggregation_method if isinstance(self.aggregation_method, list) else [])
            and isinstance(result, pl.DataFrame)
            and not isinstance(result, dict)
        ):
            result = result.rename({col: f"calibration_error_{col}" for col in result.columns})

        return result
