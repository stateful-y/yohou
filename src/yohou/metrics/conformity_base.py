"""Base class for conformal prediction conformity scorers."""

import abc

import numpy as np
import polars as pl

from yohou.metrics.base import BaseScorer
from yohou.utils import Tags
from yohou.utils._compat import _fit_context

__all__ = ["BaseConformityScorer"]


class BaseConformityScorer(BaseScorer, metaclass=abc.ABCMeta):
    """Base class for conformal prediction conformity scorers.

    Conformity scorers quantify how "unusual" a prediction is compared to the
    calibration set. Used in conformal prediction to construct valid prediction
    intervals with coverage guarantees.

    See Also
    --------
    `Residual` : Concrete conformity scorer.
    `AbsoluteResidual` : Concrete conformity scorer.
    `SplitConformalForecaster` : Uses conformity scores.

    """

    def __sklearn_tags__(self) -> Tags:
        """Get estimator tags.

        Returns
        -------
        Tags
            Estimator tags with conformity scorer attributes.

        """
        tags = super().__sklearn_tags__()
        assert tags.scorer_tags is not None
        tags.scorer_tags.prediction_type = "conformity"  # ty: ignore[invalid-assignment]
        return tags

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, y_train: pl.DataFrame, **params) -> "BaseConformityScorer":
        """Fit the scorer on training data if needed."""
        # Conformity scorers typically don't aggregate results in the same way,
        # so they don't use aggregation_method, but they must implement fit.
        return super().fit(y_train, **params)

    @staticmethod
    def _compute_assymetric_quantiles(conformity_scores: pl.DataFrame, coverage_rate: float) -> tuple[float, float]:
        """Compute lower and upper quantiles for asymmetric intervals.

        Parameters
        ----------
        conformity_scores : pl.DataFrame
            Conformity scores from calibration.

        coverage_rate : float
            Target coverage rate.

        Returns
        -------
        lower_quantile : float
            Lower quantile value.

        upper_quantile : float
            Upper quantile value.

        """
        # Convert to numpy array for quantile computation
        if isinstance(conformity_scores, pl.DataFrame):
            scores_array = conformity_scores.to_numpy()
        else:
            scores_array = conformity_scores

        # Check if array is empty
        if hasattr(scores_array, "size") and scores_array.size == 0:
            raise ValueError(
                "Cannot compute quantile: conformity_scores is empty. "
                "This typically happens when the calibration set is too small. "
                "Increase calibration_size or reduce forecasting_horizon."
            )

        alpha = 1.0 - coverage_rate
        lower_quantile: float = np.quantile(scores_array, alpha / 2.0, method="lower")

        upper_quantile: float = np.quantile(scores_array, 1.0 - alpha / 2.0, method="higher")

        return lower_quantile, upper_quantile

    @staticmethod
    def _compute_symetric_quantiles(conformity_scores: pl.DataFrame, coverage_rate: float) -> float:
        """Compute quantile for symmetric intervals.

        Parameters
        ----------
        conformity_scores : pl.DataFrame
            Conformity scores from calibration.

        coverage_rate : float
            Target coverage rate.

        Returns
        -------
        float
            Quantile value for symmetric intervals.

        Raises
        ------
        ValueError
            If conformity_scores is empty.

        """
        # Convert to numpy array for quantile computation
        conformity_array = conformity_scores.to_numpy()

        # Check if array is empty
        if conformity_array.size == 0:
            raise ValueError(
                "Cannot compute quantile: conformity_scores is empty. "
                "This typically happens when the calibration set is too small. "
                "Increase calibration_size or reduce forecasting_horizon."
            )

        quantile: float = np.quantile(conformity_array, coverage_rate, method="lower")

        return quantile

    @staticmethod
    def _format_y_pred_interval(
        lower_bound: pl.DataFrame, upper_bound: pl.DataFrame, coverage_rate: float
    ) -> pl.DataFrame:
        """Format lower and upper bounds into interval DataFrame.

        Parameters
        ----------
        lower_bound : pl.DataFrame
            Lower bound predictions.

        upper_bound : pl.DataFrame
            Upper bound predictions.

        coverage_rate : float
            Coverage rate for labeling columns.

        Returns
        -------
        pl.DataFrame
            Formatted prediction intervals.

        """
        lower_bound.columns = [f"{col}_lower_{coverage_rate}" for col in lower_bound.columns]
        upper_bound.columns = [f"{col}_upper_{coverage_rate}" for col in upper_bound.columns]

        y_pred_interval = pl.concat([lower_bound, upper_bound], how="horizontal")

        return y_pred_interval

    @abc.abstractmethod
    def inverse_score(
        self, y_pred: pl.DataFrame, conformity_scores: pl.DataFrame, coverage_rate: float
    ) -> pl.DataFrame:
        """Transform conformity scores into prediction intervals.

        Parameters
        ----------
        y_pred : pl.DataFrame
            Point predictions.

        conformity_scores : pl.DataFrame
            Conformity scores from calibration.

        coverage_rate : float
            Target coverage probability.

        Returns
        -------
        pl.DataFrame
            Prediction intervals.

        """
