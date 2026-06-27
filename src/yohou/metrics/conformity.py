"""Conformity scoring functions for conformal prediction intervals."""

import abc
import numbers

import polars as pl
import polars.selectors as cs
from sklearn.base import check_is_fitted

from yohou.utils import validate_scorer_data
from yohou.utils._compat import Interval

from .conformity_base import BaseConformityScorer

__all__ = [
    "AbsoluteGammaResidual",
    "AbsoluteQuantileResidual",
    "AbsoluteResidual",
    "GammaResidual",
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

    See Also
    --------
    - [`AbsoluteResidual`][yohou.metrics.conformity.AbsoluteResidual] : Symmetric variant using absolute residuals.
    - [`GammaResidual`][yohou.metrics.conformity.GammaResidual] : Scale-dependent variant using relative errors.
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
        self, y_pred: pl.DataFrame, conformity_scores: pl.DataFrame, coverage_rate: float
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

        # Compute intervals
        lower_quantile, upper_quantile = self._compute_asymmetric_quantiles(conformity_scores, coverage_rate)
        lower_bound, upper_bound = y_pred + lower_quantile, y_pred + upper_quantile

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

    See Also
    --------
    - [`Residual`][yohou.metrics.conformity.Residual] : Asymmetric variant using signed residuals.
    - [`AbsoluteGammaResidual`][yohou.metrics.conformity.AbsoluteGammaResidual] : Scale-dependent symmetric variant.

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
        self, y_pred: pl.DataFrame, conformity_scores: pl.DataFrame, coverage_rate: float
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

        # Compute symmetric intervals
        quantile = self._compute_symmetric_quantiles(conformity_scores, coverage_rate)
        lower_bound, upper_bound = y_pred - quantile, y_pred + quantile

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
        self, y_pred: pl.DataFrame, conformity_scores: pl.DataFrame, coverage_rate: float
    ) -> pl.DataFrame:
        """Construct prediction intervals from gamma conformity scores.

        Parameters
        ----------
        y_pred : pl.DataFrame
             Point predictions.
        conformity_scores : pl.DataFrame
             Conformity scores.
        coverage_rate : float
             Coverage rate.

        Returns
        -------
        pl.DataFrame
             Prediction intervals.
        """
        check_is_fitted(self, ["_is_fitted"])

        # Validate and align
        y_pred, conformity_scores, context = validate_scorer_data(
            self, y_true=None, y_pred=y_pred, scores=conformity_scores, inverse=True
        )

        # Compute quantiles
        lower_q, upper_q = self._compute_asymmetric_quantiles(conformity_scores, coverage_rate)

        # Determine explicit float types for Polars
        lower_q, upper_q = float(lower_q), float(upper_q)

        # Reconstruct y
        denom = y_pred + self.epsilon
        lower_bound = y_pred + lower_q * denom
        upper_bound = y_pred + upper_q * denom

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
        return tags

    @abc.abstractmethod
    def score(self, y_truth: pl.DataFrame, y_pred: pl.DataFrame, /, **score_params) -> pl.DataFrame:
        """Compute absolute quantile residual scores."""
