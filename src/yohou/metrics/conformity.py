"""Conformity scoring functions for conformal prediction intervals."""

import abc

import polars as pl

from .base import BaseConformityScorer


class Residual(BaseConformityScorer):
    r"""Residual-based conformity scorer using signed prediction errors.

    Computes conformity scores as $y - \hat{y}$ for asymmetric intervals.
    """

    def __sklearn_tags__(self):
        """Get the tags for this estimator."""
        tags = super().__sklearn_tags__()
        tags.scorer_tags.prediction_type = "point"
        return tags

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
            Conformity scores (y_truth - y_pred).

        """
        self._validate_inputs(y_truth, y_pred)

        scores = y_truth - y_pred

        return scores

    def inverse_score(
        self, y_pred: pl.DataFrame, conformity_scores: pl.DataFrame, coverage_rate: float
    ) -> pl.DataFrame:
        """Construct prediction intervals from conformity scores.

        Parameters
        ----------
        y_pred : pl.DataFrame
            Point predictions.

        conformity_scores : pl.DataFrame
            Computed conformity scores from calibration set.

        coverage_rate : float
            Desired coverage probability (e.g., 0.9 for 90% intervals).

        Returns
        -------
        pl.DataFrame
            Prediction intervals with lower and upper bounds.

        """
        lower_quantile, upper_quantile = self._compute_assymetric_quantiles(
            conformity_scores, coverage_rate
        )
        lower_bound, upper_bound = y_pred + lower_quantile, y_pred + upper_quantile

        y_pred_interval = self._format_y_pred_interval(lower_bound, upper_bound, coverage_rate)

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
            Conformity scores (|y_truth - y_pred|).

        """
        self._validate_inputs(y_truth, y_pred)

        scores = (y_truth - y_pred).select(pl.all().abs())

        return scores

    def inverse_score(
        self, y_pred: pl.DataFrame, conformity_scores: pl.DataFrame, coverage_rate: float
    ) -> pl.DataFrame:
        """Construct symmetric prediction intervals from absolute conformity scores.

        Parameters
        ----------
        y_pred : pl.DataFrame
            Point predictions.

        conformity_scores : pl.DataFrame
            Absolute conformity scores from calibration set.

        coverage_rate : float
            Desired coverage probability.

        Returns
        -------
        pl.DataFrame
            Symmetric prediction intervals.

        """
        quantile = self._compute_symetric_quantiles(conformity_scores, coverage_rate)
        lower_bound, upper_bound = y_pred - quantile, y_pred + quantile

        y_pred_interval = self._format_y_pred_interval(lower_bound, upper_bound, coverage_rate)

        return y_pred_interval


class GammaResidual(BaseConformityScorer):
    r"""Gamma residual scorer using relative prediction errors.

    Computes conformity scores as $(y - \hat{y}) / (\hat{y} + \epsilon)$.

    Parameters
    ----------
    epsilon : float, default=1e-8
        Small constant to prevent division by zero.

    """

    def __sklearn_tags__(self):
        """Get the tags for this estimator."""
        tags = super().__sklearn_tags__()
        tags.scorer_tags.prediction_type = "point"
        return tags

    def __init__(self, epsilon: float = 1e-8) -> None:
        BaseConformityScorer.__init__(self)

        self.epsilon = epsilon

    def score(self, y_truth: pl.DataFrame, y_pred: pl.DataFrame, **score_params) -> pl.DataFrame:
        """Compute gamma (relative) residual conformity scores.

        Parameters
        ----------
        y_truth : pl.DataFrame
            True target values.

        y_pred : pl.DataFrame
            Predicted values.

        Returns
        -------
        pl.DataFrame
            Relative conformity scores (y_truth - y_pred) / (y_pred + epsilon).

        """
        self._validate_inputs(y_truth, y_pred)

        scores = (y_truth - y_pred) / (y_pred + self.epsilon)

        return scores


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
            Absolute relative conformity scores.

        """
        scores = GammaResidual.score(self, y_truth, y_pred).select(pl.all().abs())

        return scores


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
