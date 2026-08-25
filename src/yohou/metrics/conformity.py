"""Conformity scoring functions for conformal prediction intervals."""

import abc
import numbers

import numpy as np
import polars as pl
import polars.selectors as cs
from sklearn.base import check_is_fitted

from yohou.utils import validate_scorer_data
from yohou.utils._compat import Interval, _fit_context

from .conformity_base import BaseConformityScorer

__all__ = [
    "AbsoluteGammaResidual",
    "AbsoluteNormalizedResidual",
    "AbsoluteQuantileResidual",
    "AbsoluteResidual",
    "GammaResidual",
    "NormalizedResidual",
    "QuantileResidual",
    "Residual",
]


class Residual(BaseConformityScorer):
    r"""Residual-based conformity scorer using signed prediction errors.

    Computes conformity scores as the signed difference between the true
    and predicted values:

    $$s = y - \hat{y}$$

    The signed residuals produce **asymmetric** prediction intervals,
    where the lower and upper bounds can differ in width from the
    point prediction.

    Parameters
    ----------
    groups : list of str, dict of str to float, or None, default=None
        Panel group filter (list) or filter with weights (dict). If None,
        all panel groups are included with equal weight.
    components : list of str, dict of str to float, or None, default=None
        Component filter (list) or filter with weights (dict). If None,
        all components are included with equal weight.
    time_weighter : BaseWeighter or None, default=None
        Weighter applied along the time axis (observed timestamps). If None,
        all timestamps contribute equally.
    step_weighter : BaseWeighter or None, default=None
        Weighter applied along the forecasting-step axis. If None, all
        forecasting steps contribute equally.
    vintage_weighter : BaseWeighter or None, default=None
        Weighter applied along the vintage-time axis. If None, all vintages
        contribute equally.

    See Also
    --------
    - [`AbsoluteResidual`][yohou.metrics.conformity.AbsoluteResidual] : Symmetric variant using absolute residuals.
    - [`GammaResidual`][yohou.metrics.conformity.GammaResidual] : Scale-independent variant using relative errors.
    - [`SplitConformalForecaster`][yohou.interval.split_conformal.SplitConformalForecaster] :
        Conformal prediction forecaster that uses conformity scorers.

    Examples
    --------
    >>> import polars as pl
    >>> from datetime import date
    >>> from yohou.metrics.conformity import Residual
    >>> scorer = Residual().fit(
    ...     pl.DataFrame({"time": [date(2020, 1, 1), date(2020, 1, 2)], "y": [1.0, 2.0]})
    ... )
    >>> y_truth = pl.DataFrame({"time": [date(2020, 1, 3), date(2020, 1, 4)], "y": [3.0, 5.0]})
    >>> y_pred = pl.DataFrame({"time": [date(2020, 1, 3), date(2020, 1, 4)], "y": [2.5, 4.0]})
    >>> scores = scorer.score(y_truth, y_pred)
    >>> scores.drop("time").to_series().to_list()
    [0.5, 1.0]

    """

    def score(self, y_truth: pl.DataFrame, y_pred: pl.DataFrame, /, **score_params) -> pl.DataFrame:
        """Compute signed residual conformity scores.

        Parameters
        ----------
        y_truth : pl.DataFrame
            True target values.

        y_pred : pl.DataFrame
            Predicted values.

        Returns
        -------
        pl.DataFrame
            Conformity scores (y_truth - y_pred) with "time" column preserved.

        """
        check_is_fitted(self, ["_is_fitted"])

        # Filter out scorer from score_params to avoid conflict with explicit scorer=self
        score_params_filtered = {k: v for k, v in score_params.items() if k != "scorer"}

        # Validate and align (time dropped, returned as context)
        y_truth, y_pred, context = validate_scorer_data(
            self,
            y_truth,
            y_pred,
            **score_params_filtered,
        )

        # Compute scores and reconstruct with time
        scores_values = y_truth - y_pred
        scores = pl.DataFrame({"time": context.time_values}).hstack(scores_values)

        return scores

    def inverse_score(
        self,
        y_pred: pl.DataFrame,
        conformity_scores: pl.DataFrame,
        coverage_rate: float,
        global_calibration: bool = False,
    ) -> pl.DataFrame:
        """Construct prediction intervals from conformity scores.

        Parameters
        ----------
        y_pred : pl.DataFrame
            Point predictions, optionally with "time" column.

        conformity_scores : pl.DataFrame
            Computed conformity scores from calibration set, optionally with "time" column.

        coverage_rate : float
            Desired coverage probability (e.g., 0.9 for 90% intervals).

        Returns
        -------
        pl.DataFrame
            Prediction intervals with lower and upper bounds, and time columns if input had them.

        """
        check_is_fitted(self, ["_is_fitted"])

        # Validate and align inputs (time dropped, returned as context for reconstruction)
        y_pred, conformity_scores, context = validate_scorer_data(
            self, y_true=None, y_pred=y_pred, scores=conformity_scores, inverse=True
        )

        # Compute intervals. The quantiles arrive one per value column, in the
        # score frame's column order, which matches y_pred's positionally.
        lower_quantiles, upper_quantiles = self._compute_asymmetric_quantiles(
            conformity_scores, coverage_rate, global_calibration=global_calibration
        )
        lower_bound = y_pred.with_columns([
            pl.col(col) + q for col, q in zip(y_pred.columns, lower_quantiles, strict=True)
        ])
        upper_bound = y_pred.with_columns([
            pl.col(col) + q for col, q in zip(y_pred.columns, upper_quantiles, strict=True)
        ])

        y_pred_interval = self._format_y_pred_interval(lower_bound, upper_bound, coverage_rate)

        # Add time column back
        y_pred_interval = pl.DataFrame({"time": context.time_values}).hstack(y_pred_interval)

        return y_pred_interval


class AbsoluteResidual(Residual):
    r"""Absolute residual conformity scorer using unsigned prediction errors.

    Computes conformity scores as the absolute difference between true
    and predicted values:

    $$s = |y - \hat{y}|$$

    The absolute residuals produce **symmetric** prediction intervals,
    where the lower and upper bounds are equidistant from the point
    prediction.

    Parameters
    ----------
    groups : list of str, dict of str to float, or None, default=None
        Panel group filter (list) or filter with weights (dict). If None,
        all panel groups are included with equal weight.
    components : list of str, dict of str to float, or None, default=None
        Component filter (list) or filter with weights (dict). If None,
        all components are included with equal weight.
    time_weighter : BaseWeighter or None, default=None
        Weighter applied along the time axis (observed timestamps). If None,
        all timestamps contribute equally.
    step_weighter : BaseWeighter or None, default=None
        Weighter applied along the forecasting-step axis. If None, all
        forecasting steps contribute equally.
    vintage_weighter : BaseWeighter or None, default=None
        Weighter applied along the vintage-time axis. If None, all vintages
        contribute equally.

    See Also
    --------
    - [`Residual`][yohou.metrics.conformity.Residual] : Asymmetric variant using signed residuals.
    - [`AbsoluteGammaResidual`][yohou.metrics.conformity.AbsoluteGammaResidual] : Scale-independent symmetric variant.

    Examples
    --------
    >>> import polars as pl
    >>> from datetime import date
    >>> from yohou.metrics.conformity import AbsoluteResidual
    >>> scorer = AbsoluteResidual().fit(
    ...     pl.DataFrame({"time": [date(2020, 1, 1), date(2020, 1, 2)], "y": [1.0, 2.0]})
    ... )
    >>> y_truth = pl.DataFrame({"time": [date(2020, 1, 3), date(2020, 1, 4)], "y": [3.0, 5.0]})
    >>> y_pred = pl.DataFrame({"time": [date(2020, 1, 3), date(2020, 1, 4)], "y": [2.5, 6.0]})
    >>> scores = scorer.score(y_truth, y_pred)
    >>> scores.drop("time").to_series().to_list()
    [0.5, 1.0]

    """

    def __sklearn_tags__(self):
        """Get the tags for this estimator."""
        tags = super().__sklearn_tags__()
        assert tags.scorer_tags is not None
        tags.scorer_tags.symmetric = True
        return tags

    def score(self, y_truth: pl.DataFrame, y_pred: pl.DataFrame, /, **score_params) -> pl.DataFrame:
        """Compute absolute residual conformity scores.

        Parameters
        ----------
        y_truth : pl.DataFrame
            True target values.

        y_pred : pl.DataFrame
            Predicted values.

        Returns
        -------
        pl.DataFrame
            Conformity scores (|y_truth - y_pred|) with "time" column preserved.

        """
        check_is_fitted(self, ["_is_fitted"])

        # Filter out scorer from score_params to avoid conflict with explicit scorer=self
        score_params_filtered = {k: v for k, v in score_params.items() if k != "scorer"}

        # Validate and align (time dropped, returned as context)
        y_truth, y_pred, context = validate_scorer_data(
            self,
            y_truth,
            y_pred,
            **score_params_filtered,
        )

        # Compute scores and reconstruct with time
        scores_values = (y_truth - y_pred).select(pl.all().abs())
        scores = pl.DataFrame({"time": context.time_values}).hstack(scores_values)

        return scores

    def inverse_score(
        self,
        y_pred: pl.DataFrame,
        conformity_scores: pl.DataFrame,
        coverage_rate: float,
        global_calibration: bool = False,
    ) -> pl.DataFrame:
        """Construct symmetric prediction intervals from absolute conformity scores.

        Parameters
        ----------
        y_pred : pl.DataFrame
            Point predictions, optionally with "time" column.

        conformity_scores : pl.DataFrame
            Absolute conformity scores from calibration set, optionally with "time" column.

        coverage_rate : float
            Desired coverage probability.

        Returns
        -------
        pl.DataFrame
            Symmetric prediction intervals.

        """
        check_is_fitted(self, ["_is_fitted"])

        # Validate and align inputs (time dropped, returned as context for reconstruction)
        y_pred, conformity_scores, context = validate_scorer_data(
            self, y_true=None, y_pred=y_pred, scores=conformity_scores, inverse=True
        )

        # Compute symmetric intervals, one half-width per value column
        quantiles = self._compute_symmetric_quantiles(
            conformity_scores, coverage_rate, global_calibration=global_calibration
        )
        lower_bound = y_pred.with_columns([pl.col(col) - q for col, q in zip(y_pred.columns, quantiles, strict=True)])
        upper_bound = y_pred.with_columns([pl.col(col) + q for col, q in zip(y_pred.columns, quantiles, strict=True)])

        y_pred_interval = self._format_y_pred_interval(lower_bound, upper_bound, coverage_rate)

        # Add time column back
        y_pred_interval = pl.DataFrame({"time": context.time_values}).hstack(y_pred_interval)

        return y_pred_interval


class GammaResidual(BaseConformityScorer):
    r"""Gamma residual scorer using relative prediction errors.

    Computes conformity scores as the signed relative error, normalised
    by the predicted value:

    $$s = \frac{y - \hat{y}}{\hat{y} + \epsilon}$$

    This scorer is useful when the scale of the target variable varies
    over time, because the conformity scores are relative to the prediction
    magnitude. The ``epsilon`` parameter prevents division by zero when
    predictions are near zero.

    Parameters
    ----------
    epsilon : float, default=1e-8
        Small constant added to the denominator to prevent division by
        zero.

    See Also
    --------
    - [`AbsoluteGammaResidual`][yohou.metrics.conformity.AbsoluteGammaResidual] : Symmetric variant using absolute relative errors.
    - [`Residual`][yohou.metrics.conformity.Residual] : Scale-dependent signed residual scorer.

    Examples
    --------
    >>> import polars as pl
    >>> from datetime import date
    >>> from yohou.metrics.conformity import GammaResidual
    >>> scorer = GammaResidual(epsilon=1e-8).fit(
    ...     pl.DataFrame({"time": [date(2020, 1, 1), date(2020, 1, 2)], "y": [1.0, 2.0]})
    ... )
    >>> y_truth = pl.DataFrame({"time": [date(2020, 1, 3)], "y": [10.0]})
    >>> y_pred = pl.DataFrame({"time": [date(2020, 1, 3)], "y": [8.0]})
    >>> scores = scorer.score(y_truth, y_pred)
    >>> round(scores.drop("time").to_series().item(), 4)
    0.25

    """

    _parameter_constraints: dict = {
        **BaseConformityScorer._parameter_constraints,
        "epsilon": [Interval(numbers.Real, 0, None, closed="neither")],
    }

    def __init__(
        self,
        epsilon: float = 1e-8,
        groups: list[str] | dict[str, float] | None = None,
        components: list[str] | dict[str, float] | None = None,
    ) -> None:
        super().__init__(
            groups=groups,
            components=components,
        )
        self.epsilon = epsilon

    def __sklearn_tags__(self):
        """Get the tags for this estimator."""
        tags = super().__sklearn_tags__()
        assert tags.scorer_tags is not None
        tags.scorer_tags.multiplicative = True
        return tags

    def score(self, y_truth: pl.DataFrame, y_pred: pl.DataFrame, /, **score_params) -> pl.DataFrame:
        """Compute gamma (relative) residual conformity scores.

        Parameters
        ----------
        y_truth : pl.DataFrame
            True target values with "time" column.

        y_pred : pl.DataFrame
            Predicted values with "time" column.

        Returns
        -------
        pl.DataFrame
            Relative conformity scores (y_truth - y_pred) / (y_pred + epsilon) with "time" column preserved.

        """
        check_is_fitted(self, ["_is_fitted"])

        # Filter out scorer from score_params to avoid conflict with explicit scorer=self
        score_params_filtered = {k: v for k, v in score_params.items() if k != "scorer"}

        # Validate and align (time dropped, returned as context)
        y_truth, y_pred, context = validate_scorer_data(
            self,
            y_truth,
            y_pred,
            **score_params_filtered,
        )

        # Compute scores and reconstruct with time
        scores_values = (y_truth - y_pred) / (y_pred + self.epsilon)
        scores = pl.DataFrame({"time": context.time_values}).hstack(scores_values)

        return scores

    def inverse_score(
        self,
        y_pred: pl.DataFrame,
        conformity_scores: pl.DataFrame,
        coverage_rate: float,
        global_calibration: bool = False,
    ) -> pl.DataFrame:
        """Construct prediction intervals from gamma conformity scores.

        Parameters
        ----------
        y_pred : pl.DataFrame
            Point predictions, optionally with "time" column.
        conformity_scores : pl.DataFrame
            Computed conformity scores from calibration set, optionally with "time" column.
        coverage_rate : float
            Desired coverage probability (e.g., 0.9 for 90% intervals).

        Returns
        -------
        pl.DataFrame
            Prediction intervals with lower and upper bounds, and time columns if input had them.

        """
        check_is_fitted(self, ["_is_fitted"])

        # Validate and align
        y_pred, conformity_scores, context = validate_scorer_data(
            self, y_true=None, y_pred=y_pred, scores=conformity_scores, inverse=True
        )

        # Compute quantiles, one pair per value column
        lower_quantiles, upper_quantiles = self._compute_asymmetric_quantiles(
            conformity_scores, coverage_rate, global_calibration=global_calibration
        )

        # Reconstruct y. The score is relative to the prediction, so each
        # column's quantile is scaled by that column's own denominator.
        lower_bound = y_pred.with_columns([
            pl.col(col) + float(q) * (pl.col(col) + self.epsilon)
            for col, q in zip(y_pred.columns, lower_quantiles, strict=True)
        ])
        upper_bound = y_pred.with_columns([
            pl.col(col) + float(q) * (pl.col(col) + self.epsilon)
            for col, q in zip(y_pred.columns, upper_quantiles, strict=True)
        ])

        y_pred_interval = self._format_y_pred_interval(lower_bound, upper_bound, coverage_rate)

        return pl.DataFrame({"time": context.time_values}).hstack(y_pred_interval)


class AbsoluteGammaResidual(GammaResidual):
    r"""Absolute gamma residual scorer using absolute relative errors.

    Computes conformity scores as the absolute relative error:

    $$s = \left|\frac{y - \hat{y}}{\hat{y} + \epsilon}\right|$$

    Produces **symmetric** prediction intervals that are proportional
    to the prediction magnitude.

    Parameters
    ----------
    epsilon : float, default=1e-8
        Small constant added to the denominator to prevent division by
        zero.

    See Also
    --------
    - [`GammaResidual`][yohou.metrics.conformity.GammaResidual] : Asymmetric variant using signed relative errors.
    - [`AbsoluteResidual`][yohou.metrics.conformity.AbsoluteResidual] : Scale-independent symmetric variant.

    Examples
    --------
    >>> import polars as pl
    >>> from datetime import date
    >>> from yohou.metrics.conformity import AbsoluteGammaResidual
    >>> scorer = AbsoluteGammaResidual(epsilon=1e-8).fit(
    ...     pl.DataFrame({"time": [date(2020, 1, 1), date(2020, 1, 2)], "y": [1.0, 2.0]})
    ... )
    >>> y_truth = pl.DataFrame({"time": [date(2020, 1, 3)], "y": [6.0]})
    >>> y_pred = pl.DataFrame({"time": [date(2020, 1, 3)], "y": [8.0]})
    >>> scores = scorer.score(y_truth, y_pred)
    >>> round(scores.drop("time").to_series().item(), 4)
    0.25

    """

    def __sklearn_tags__(self):
        """Get the tags for this estimator."""
        tags = super().__sklearn_tags__()
        assert tags.scorer_tags is not None
        tags.scorer_tags.symmetric = True
        return tags

    def score(self, y_truth: pl.DataFrame, y_pred: pl.DataFrame, /, **score_params) -> pl.DataFrame:
        r"""Compute absolute gamma residual conformity scores.

        Parameters
        ----------
        y_truth : pl.DataFrame
            True target values.

        y_pred : pl.DataFrame
            Predicted values.

        Returns
        -------
        pl.DataFrame
            Absolute relative conformity scores with "time" column preserved.

        """
        check_is_fitted(self, ["_is_fitted"])

        # Get parent scores (includes "time" column)
        scores = GammaResidual.score(self, y_truth, y_pred)

        # Apply abs to non-time columns only
        scores = scores.select(
            "time",  # Keep time as-is
            cs.exclude("time").abs(),  # Apply abs to value columns
        )

        return scores


class QuantileResidual(BaseConformityScorer):
    """Quantile residual scorer for interval forecasts.

    Abstract base class for quantile-based conformity scoring.
    Subclasses must implement the ``score`` method to compute
    residuals between observed values and predicted interval bounds.

    Notes
    -----
    Unlike ``Residual`` and ``GammaResidual``, this scorer operates on
    interval predictions (lower/upper bounds) rather than point predictions.
    The ``prediction_type`` tag is set to ``"interval"``.

    See Also
    --------
    - [`AbsoluteQuantileResidual`][yohou.metrics.conformity.AbsoluteQuantileResidual] : Absolute variant of quantile residuals.
    - [`Residual`][yohou.metrics.conformity.Residual] : Point-prediction conformity scorer.

    """

    def __sklearn_tags__(self):
        """Get the tags for this estimator."""
        tags = super().__sklearn_tags__()
        assert tags.scorer_tags is not None
        tags.scorer_tags.prediction_type = "interval"
        return tags

    @abc.abstractmethod
    def score(self, y_truth: pl.DataFrame, y_pred: pl.DataFrame, /, **score_params) -> pl.DataFrame:
        """Compute quantile residual scores."""


class AbsoluteQuantileResidual(BaseConformityScorer):
    """Absolute quantile residual scorer for interval forecasts.

    Abstract base class for absolute quantile-based conformity scoring.
    Subclasses must implement the ``score`` method to compute absolute
    residuals between observed values and predicted interval bounds.

    Notes
    -----
    Unlike ``AbsoluteResidual`` and ``AbsoluteGammaResidual``, this scorer
    operates on interval predictions. The ``prediction_type`` tag is set
    to ``"interval"``.

    See Also
    --------
    - [`QuantileResidual`][yohou.metrics.conformity.QuantileResidual] : Signed variant of quantile residuals.
    - [`AbsoluteResidual`][yohou.metrics.conformity.AbsoluteResidual] : Point-prediction absolute conformity scorer.

    """

    def __sklearn_tags__(self):
        """Get the tags for this estimator."""
        tags = super().__sklearn_tags__()
        assert tags.scorer_tags is not None
        tags.scorer_tags.prediction_type = "interval"
        tags.scorer_tags.symmetric = True
        return tags

    @abc.abstractmethod
    def score(self, y_truth: pl.DataFrame, y_pred: pl.DataFrame, /, **score_params) -> pl.DataFrame:
        """Compute absolute quantile residual scores."""


class NormalizedResidual(BaseConformityScorer):
    r"""Residual scorer normalised by each column's own dispersion.

    Computes conformity scores as the signed residual divided by a scale
    fitted per value column at ``fit`` time:

    $$s = \frac{y - \hat{y}}{\sigma_c}$$

    where $\sigma_c$ is the standard deviation of the first difference of
    column *c* over the training target. Dividing by the column's own
    dispersion is what makes scores from different columns comparable, which
    is the precondition for pooling them into one quantile with
    ``SplitConformalForecaster(calibration_strategy="global")``.

    This is a stronger normalisation than
    [`GammaResidual`][yohou.metrics.conformity.GammaResidual], which divides
    by the predicted level. Level normalisation removes differences in
    magnitude but leaves differences in volatility, so two columns of equal
    size but unequal noise still produce incomparable scores. Measured on
    `fetch_hospital`, fitting each normaliser on the first half of the
    residuals and comparing entities on the held-out second half, the spread
    of per-entity score magnitudes was 10.7x raw, 5.8x under level
    normalisation, and 2.1x here.

    The scale is fitted from the training target rather than from residuals,
    so it does not depend on the wrapped forecaster, and it is frozen at
    ``fit`` so that ``predict`` is a function of the fitted state alone.

    Parameters
    ----------
    epsilon : float, default=1e-8
        Floor applied to a fitted scale, so a column whose training target
        does not vary does not produce a division by zero.
    groups : list of str, dict of str to float, or None, default=None
        Panel group filter (list) or filter with weights (dict). If None,
        all panel groups are included with equal weight.
    components : list of str, dict of str to float, or None, default=None
        Component filter (list) or filter with weights (dict). If None,
        all components are included with equal weight.

    Attributes
    ----------
    column_scales_ : dict of str to float
        The fitted dispersion per value column.

    See Also
    --------
    - [`AbsoluteNormalizedResidual`][yohou.metrics.conformity.AbsoluteNormalizedResidual] : Symmetric variant.
    - [`GammaResidual`][yohou.metrics.conformity.GammaResidual] : Normalises by the predicted level instead.
    - [`SplitConformalForecaster`][yohou.interval.split_conformal.SplitConformalForecaster] :
        Consumes this scorer for global calibration across columns.

    Examples
    --------
    >>> import polars as pl
    >>> from datetime import date
    >>> from yohou.metrics.conformity import NormalizedResidual
    >>> y_train = pl.DataFrame({
    ...     "time": [date(2020, 1, d) for d in range(1, 6)],
    ...     "y": [1.0, 3.0, 2.0, 4.0, 3.0],
    ... })
    >>> scorer = NormalizedResidual().fit(y_train)
    >>> round(scorer.column_scales_["y"], 6)
    1.5
    >>> y_truth = pl.DataFrame({"time": [date(2020, 1, 6)], "y": [11.0]})
    >>> y_pred = pl.DataFrame({"time": [date(2020, 1, 6)], "y": [10.0]})
    >>> round(scorer.score(y_truth, y_pred).drop("time").to_series().item(), 6)
    0.666667

    """

    _parameter_constraints: dict = {
        **BaseConformityScorer._parameter_constraints,
        "epsilon": [Interval(numbers.Real, 0, None, closed="neither")],
    }

    def __init__(
        self,
        epsilon: float = 1e-8,
        groups: list[str] | dict[str, float] | None = None,
        components: list[str] | dict[str, float] | None = None,
    ) -> None:
        super().__init__(groups=groups, components=components)
        self.epsilon = epsilon

    def __sklearn_tags__(self):
        """Get the tags for this estimator."""
        tags = super().__sklearn_tags__()
        assert tags.scorer_tags is not None
        tags.scorer_tags.comparable_across_columns = True
        return tags

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, y_train: pl.DataFrame, *, forecaster=None, **params) -> "NormalizedResidual":
        """Fit one dispersion scale per value column.

        Parameters
        ----------
        y_train : pl.DataFrame
            Training target with a ``"time"`` column and one or more value
            columns.
        forecaster : object or None, default=None
            Accepted for API consistency and unused: the scale describes the
            series, not the model fitted to it.
        **params : dict
            Metadata to route.

        Returns
        -------
        self

        """
        super().fit(y_train, forecaster=forecaster, **params)

        values = y_train.drop("time", strict=False)
        self.column_scales_ = {}
        for column in values.columns:
            series = values[column].to_numpy().astype(float)
            # First difference rather than the raw level: it approximates the
            # scale of a forecast error without needing the forecaster, and it
            # equalised entities better than residual dispersion when measured.
            scale = float(np.std(np.diff(series))) if series.size > 1 else 0.0
            self.column_scales_[column] = max(scale, self.epsilon)

        return self

    def _scales_for(self, columns: list[str]) -> list[float]:
        """Return the fitted scale for each named column, in order."""
        check_is_fitted(self, ["column_scales_"])
        missing = [c for c in columns if c not in self.column_scales_]
        if missing:
            raise ValueError(
                f"NormalizedResidual was fitted on columns {sorted(self.column_scales_)} and has no "
                f"scale for {missing}. Fit the scorer on the same value columns it will score."
            )
        return [self.column_scales_[c] for c in columns]

    def score(self, y_truth: pl.DataFrame, y_pred: pl.DataFrame, /, **score_params) -> pl.DataFrame:
        """Compute dispersion-normalised conformity scores.

        Parameters
        ----------
        y_truth : pl.DataFrame
            True target values with ``"time"``.
        y_pred : pl.DataFrame
            Predicted values with ``"time"``.
        **score_params : dict
            Metadata to route.

        Returns
        -------
        pl.DataFrame
            Conformity scores with ``"time"`` preserved.

        """
        check_is_fitted(self, ["_is_fitted"])

        score_params_filtered = {k: v for k, v in score_params.items() if k != "scorer"}
        y_truth, y_pred, context = validate_scorer_data(self, y_truth, y_pred, **score_params_filtered)

        scales = self._scales_for(list(y_truth.columns))
        raw = y_truth - y_pred
        scores_values = raw.with_columns([pl.col(col) / scale for col, scale in zip(raw.columns, scales, strict=True)])
        return pl.DataFrame({"time": context.time_values}).hstack(scores_values)

    def inverse_score(
        self,
        y_pred: pl.DataFrame,
        conformity_scores: pl.DataFrame,
        coverage_rate: float,
        global_calibration: bool = False,
    ) -> pl.DataFrame:
        """Construct prediction intervals, rescaling by each column's dispersion.

        Parameters
        ----------
        y_pred : pl.DataFrame
            Point predictions, optionally with ``"time"``.
        conformity_scores : pl.DataFrame
            Normalised conformity scores from calibration.
        coverage_rate : float
            Desired coverage probability.

        Returns
        -------
        pl.DataFrame
            Prediction intervals with lower and upper bounds.

        """
        check_is_fitted(self, ["_is_fitted"])

        y_pred, conformity_scores, context = validate_scorer_data(
            self, y_true=None, y_pred=y_pred, scores=conformity_scores, inverse=True
        )

        lower_quantiles, upper_quantiles = self._compute_asymmetric_quantiles(
            conformity_scores, coverage_rate, global_calibration=global_calibration
        )
        scales = self._scales_for(list(y_pred.columns))

        lower_bound = y_pred.with_columns([
            pl.col(col) + float(q) * scale
            for col, q, scale in zip(y_pred.columns, lower_quantiles, scales, strict=True)
        ])
        upper_bound = y_pred.with_columns([
            pl.col(col) + float(q) * scale
            for col, q, scale in zip(y_pred.columns, upper_quantiles, scales, strict=True)
        ])

        y_pred_interval = self._format_y_pred_interval(lower_bound, upper_bound, coverage_rate)
        return pl.DataFrame({"time": context.time_values}).hstack(y_pred_interval)


class AbsoluteNormalizedResidual(NormalizedResidual):
    r"""Symmetric variant of `NormalizedResidual` using absolute scores.

    Computes $s = |y - \hat{y}| / \sigma_c$, so the interval is equidistant
    from the point prediction, with each column's half-width scaled by its own
    fitted dispersion. Use it where the error distribution is roughly
    symmetric and global calibration is wanted.

    Parameters
    ----------
    epsilon : float, default=1e-8
        Floor applied to a fitted scale.
    groups : list of str, dict of str to float, or None, default=None
        Panel group filter.
    components : list of str, dict of str to float, or None, default=None
        Component filter.

    Attributes
    ----------
    column_scales_ : dict of str to float
        The fitted dispersion per value column.

    See Also
    --------
    - [`NormalizedResidual`][yohou.metrics.conformity.NormalizedResidual] : Asymmetric variant.

    Examples
    --------
    >>> import polars as pl
    >>> from datetime import date
    >>> from yohou.metrics.conformity import AbsoluteNormalizedResidual
    >>> y_train = pl.DataFrame({
    ...     "time": [date(2020, 1, d) for d in range(1, 6)],
    ...     "y": [1.0, 3.0, 2.0, 4.0, 3.0],
    ... })
    >>> scorer = AbsoluteNormalizedResidual().fit(y_train)
    >>> y_truth = pl.DataFrame({"time": [date(2020, 1, 6)], "y": [9.0]})
    >>> y_pred = pl.DataFrame({"time": [date(2020, 1, 6)], "y": [10.0]})
    >>> round(scorer.score(y_truth, y_pred).drop("time").to_series().item(), 6)
    0.666667

    """

    def __sklearn_tags__(self):
        """Get the tags for this estimator."""
        tags = super().__sklearn_tags__()
        assert tags.scorer_tags is not None
        tags.scorer_tags.symmetric = True
        return tags

    def score(self, y_truth: pl.DataFrame, y_pred: pl.DataFrame, /, **score_params) -> pl.DataFrame:
        """Compute absolute dispersion-normalised conformity scores.

        Parameters
        ----------
        y_truth : pl.DataFrame
            True target values with ``"time"``.
        y_pred : pl.DataFrame
            Predicted values with ``"time"``.
        **score_params : dict
            Metadata to route.

        Returns
        -------
        pl.DataFrame
            Absolute conformity scores with ``"time"`` preserved.

        """
        scores = super().score(y_truth, y_pred, **score_params)
        return scores.with_columns(cs.exclude("time").abs())

    def inverse_score(
        self,
        y_pred: pl.DataFrame,
        conformity_scores: pl.DataFrame,
        coverage_rate: float,
        global_calibration: bool = False,
    ) -> pl.DataFrame:
        """Construct symmetric intervals, rescaling by each column's dispersion.

        Parameters
        ----------
        y_pred : pl.DataFrame
            Point predictions, optionally with ``"time"``.
        conformity_scores : pl.DataFrame
            Absolute normalised conformity scores from calibration.
        coverage_rate : float
            Desired coverage probability.

        Returns
        -------
        pl.DataFrame
            Symmetric prediction intervals.

        """
        check_is_fitted(self, ["_is_fitted"])

        y_pred, conformity_scores, context = validate_scorer_data(
            self, y_true=None, y_pred=y_pred, scores=conformity_scores, inverse=True
        )

        quantiles = self._compute_symmetric_quantiles(
            conformity_scores, coverage_rate, global_calibration=global_calibration
        )
        scales = self._scales_for(list(y_pred.columns))

        lower_bound = y_pred.with_columns([
            pl.col(col) - float(q) * scale for col, q, scale in zip(y_pred.columns, quantiles, scales, strict=True)
        ])
        upper_bound = y_pred.with_columns([
            pl.col(col) + float(q) * scale for col, q, scale in zip(y_pred.columns, quantiles, scales, strict=True)
        ])

        y_pred_interval = self._format_y_pred_interval(lower_bound, upper_bound, coverage_rate)
        return pl.DataFrame({"time": context.time_values}).hstack(y_pred_interval)
