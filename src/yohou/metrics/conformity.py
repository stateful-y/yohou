"""Conformity scoring functions for conformal prediction intervals."""

import abc

import polars as pl
import polars.selectors as cs
from sklearn.base import check_is_fitted

from yohou.utils import validate_scorer_data

from .base import BaseConformityScorer


class Residual(BaseConformityScorer):
    r"""Residual-based conformity scorer using signed prediction errors.

    Computes conformity scores as $y - \hat{y}$ for asymmetric intervals.
    """

    def score(self, y_truth: pl.DataFrame, y_pred: pl.DataFrame, **score_params) -> pl.DataFrame:
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

        # Validate and align (time dropped, returned as time_values)
        y_truth, y_pred, time_values = validate_scorer_data(
            self,
            y_truth,
            y_pred,
            **score_params_filtered,
        )

        # Compute scores and reconstruct with time
        scores_values = y_truth - y_pred
        scores = pl.DataFrame({"time": time_values}).hstack(scores_values)

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

        # Validate and align inputs (time dropped, returned as time_values for reconstruction)
        y_pred, conformity_scores, time_values = validate_scorer_data(
            self, y_true=None, y_pred=y_pred, scores=conformity_scores, inverse=True
        )

        # Compute intervals
        lower_quantile, upper_quantile = self._compute_assymetric_quantiles(
            conformity_scores, coverage_rate
        )
        lower_bound, upper_bound = y_pred + lower_quantile, y_pred + upper_quantile

        y_pred_interval = self._format_y_pred_interval(lower_bound, upper_bound, coverage_rate)

        # Add time column back
        y_pred_interval = pl.DataFrame({"time": time_values}).hstack(y_pred_interval)

        return y_pred_interval


class AbsoluteResidual(Residual):
    r"""Absolute residual conformity scorer using unsigned prediction errors.

    Computes conformity scores as $|y - \hat{y}|$ for symmetric intervals.
    """

    def score(self, y_truth: pl.DataFrame, y_pred: pl.DataFrame, **score_params) -> pl.DataFrame:
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

        # Validate and align (time dropped, returned as time_values)
        y_truth, y_pred, time_values = validate_scorer_data(
            self,
            y_truth,
            y_pred,
            **score_params_filtered,
        )

        # Compute scores and reconstruct with time
        scores_values = (y_truth - y_pred).select(pl.all().abs())
        scores = pl.DataFrame({"time": time_values}).hstack(scores_values)

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

        # Validate and align inputs (time dropped, returned as time_values for reconstruction)
        y_pred, conformity_scores, time_values = validate_scorer_data(
            self, y_true=None, y_pred=y_pred, scores=conformity_scores, inverse=True
        )

        # Compute symmetric intervals
        quantile = self._compute_symetric_quantiles(conformity_scores, coverage_rate)
        lower_bound, upper_bound = y_pred - quantile, y_pred + quantile

        y_pred_interval = self._format_y_pred_interval(lower_bound, upper_bound, coverage_rate)

        # Add time column back
        y_pred_interval = pl.DataFrame({"time": time_values}).hstack(y_pred_interval)

        return y_pred_interval


# TODO: Implement inverse_score
class GammaResidual(BaseConformityScorer):
    r"""Gamma residual scorer using relative prediction errors.

    Computes conformity scores as $(y - \hat{y}) / (\hat{y} + \epsilon)$.

    Parameters
    ----------
    epsilon : float, default=1e-8
        Small constant to prevent division by zero.

    """

    def __init__(
        self,
        epsilon: float = 1e-8,
        panel_group_names: list[str] | None = None,
        component_names: list[str] | None = None,
        panel_group_weight: dict[str, float] | None = None,
    ) -> None:
        super().__init__(
            panel_group_names=panel_group_names,
            component_names=component_names,
            panel_group_weight=panel_group_weight,
        )
        self.epsilon = epsilon

    def score(self, y_truth: pl.DataFrame, y_pred: pl.DataFrame, **score_params) -> pl.DataFrame:
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

        # Validate and align (time dropped, returned as time_values)
        y_truth, y_pred, time_values = validate_scorer_data(
            self,
            y_truth,
            y_pred,
            **score_params_filtered,
        )

        # Compute scores and reconstruct with time
        scores_values = (y_truth - y_pred) / (y_pred + self.epsilon)
        scores = pl.DataFrame({"time": time_values}).hstack(scores_values)

        return scores


# TODO: Implement inverse_score
class AbsoluteGammaResidual(GammaResidual):
    r"""Absolute gamma residual scorer using absolute relative errors.

    Computes conformity scores as $|(y - \hat{y}) / (\hat{y} + \epsilon)|$.
    """

    def score(self, y_truth: pl.DataFrame, y_pred: pl.DataFrame, **score_params) -> pl.DataFrame:
        """Compute absolute gamma residual conformity scores.

        Parameters
        ----------
        y_truth : pl.DataFrame
            True target values.

        y_pred : pl.DataFrame
            Predicted values.

        Returns
        -------
        pl.DataFrame
            Absolute relative conformity scores with \"time\" column preserved.

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


# TODO
class QuantileResidual(BaseConformityScorer):
    """Quantile residual scorer for interval forecasts."""

    def __sklearn_tags__(self):
        """Get the tags for this estimator."""
        tags = super().__sklearn_tags__()
        tags.scorer_tags.prediction_type = "interval"
        return tags

    @abc.abstractmethod
    def score(self, y_truth: pl.DataFrame, y_pred: pl.DataFrame, **score_params) -> pl.DataFrame:
        """Compute quantile residual scores."""


# TODO
class AbsoluteQuantileResidual(BaseConformityScorer):
    """Absolute quantile residual scorer for interval forecasts."""

    def __sklearn_tags__(self):
        """Get the tags for this estimator."""
        tags = super().__sklearn_tags__()
        tags.scorer_tags.prediction_type = "interval"
        return tags

    @abc.abstractmethod
    def score(self, y_truth: pl.DataFrame, y_pred: pl.DataFrame, **score_params) -> pl.DataFrame:
        """Compute absolute quantile residual scores."""
