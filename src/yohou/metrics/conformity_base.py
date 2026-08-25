"""Base class for conformal prediction conformity scorers."""

import abc
from typing import Literal

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
    - [`Residual`][yohou.metrics.conformity.Residual] : Concrete conformity scorer.
    - [`AbsoluteResidual`][yohou.metrics.conformity.AbsoluteResidual] : Concrete conformity scorer.
    - [`SplitConformalForecaster`][yohou.interval.split_conformal.SplitConformalForecaster] : Uses conformity scores.

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
    def fit(self, y_train: pl.DataFrame, *, forecaster=None, **params) -> "BaseConformityScorer":
        """Fit the scorer on training data if needed."""
        # Conformity scorers typically don't aggregate results in the same way,
        # so they don't use aggregation_method, but they must implement fit.
        return super().fit(y_train, forecaster=forecaster, **params)

    @staticmethod
    def _global_calibration(calibration_strategy: Literal["local", "global"]) -> bool:
        """Validate ``calibration_strategy`` and map it to the internal pooling switch.

        An unrecognized value must raise rather than silently behave as
        ``"local"``.
        """
        if calibration_strategy not in ("local", "global"):
            raise ValueError(f"calibration_strategy must be 'local' or 'global', got {calibration_strategy!r}.")
        return calibration_strategy == "global"

    @staticmethod
    def _compute_asymmetric_quantiles(
        conformity_scores: pl.DataFrame, coverage_rate: float, global_calibration: bool = False
    ) -> tuple[list[float], list[float]]:
        """Compute lower and upper quantiles per value column.

        One quantile pair is derived per column, from that column's scores
        alone. Reducing the whole frame to a single pair gave every column the
        same interval width, which over-covers low-magnitude columns and
        under-covers high-magnitude ones.

        Parameters
        ----------
        conformity_scores : pl.DataFrame
            Conformity scores from calibration, one column per value column.

        coverage_rate : float
            Target coverage rate.

        Returns
        -------
        lower_quantiles : list of float
            Lower quantile per column, in column order. When
            ``coverage_rate == 0`` each is the median (quantile 0.5) of that
            column's scores.

        upper_quantiles : list of float
            Upper quantile per column, in column order. When
            ``coverage_rate == 0`` each is the median (quantile 0.5) of that
            column's scores.

        Raises
        ------
        ValueError
            If conformity_scores is empty.

        Notes
        -----
        When ``coverage_rate == 0`` both bounds are set to the median of the
        scores, producing a degenerate (zero-width) interval rather than an
        alpha-derived quantile.

        """
        # Convert to numpy array for quantile computation
        scores_array = conformity_scores.to_numpy()

        # Check if array is empty
        if scores_array.size == 0:
            raise ValueError(
                "Cannot compute quantile: conformity_scores is empty. "
                "This typically happens when the calibration set is too small. "
                "Increase calibration_size or reduce forecasting_horizon."
            )

        alpha = 1.0 - coverage_rate

        # axis=0 reduces down each column, so a frame of n columns yields n
        # quantiles rather than one over the flattened array.
        if coverage_rate == 0:
            medians = np.quantile(scores_array, 0.5, axis=0, method="lower")
            return list(np.atleast_1d(medians)), list(np.atleast_1d(medians))

        # Split conformal takes the ceil((n+1) * q)-th order statistic, not the
        # plain empirical quantile. The (n+1) accounts for the test point being
        # exchangeable with the calibration scores, and is what makes coverage
        # at least the nominal rate. Without it the bound is one order statistic
        # short and under-covers by construction, at every sample size.
        n_columns = scores_array.shape[1] if scores_array.ndim > 1 else 1
        if global_calibration:
            # One quantile over every column's scores, then handed back once per
            # column so each caller's reconstruction still applies that column's
            # own scale. Global calibration changes which scores the quantile is
            # drawn from, never the axis the bound is rebuilt on.
            scores_array = scores_array.reshape(-1, 1)

        n = scores_array.shape[0]
        ordered = np.sort(scores_array, axis=0)

        upper_index = int(np.ceil((n + 1) * (1.0 - alpha / 2.0)))
        lower_index = int(np.floor((n + 1) * (alpha / 2.0)))

        # Beyond the resolvable range the true bound is unbounded; the widest
        # observed score is the closest finite stand-in. Callers warn.
        upper_index = min(max(upper_index, 1), n)
        lower_index = min(max(lower_index, 1), n)

        upper_quantiles = ordered[upper_index - 1]
        lower_quantiles = ordered[lower_index - 1]

        lower = list(np.atleast_1d(lower_quantiles))
        upper = list(np.atleast_1d(upper_quantiles))
        if global_calibration:
            lower, upper = lower * n_columns, upper * n_columns
        return lower, upper

    @staticmethod
    def _compute_symmetric_quantiles(
        conformity_scores: pl.DataFrame, coverage_rate: float, global_calibration: bool = False
    ) -> list[float]:
        """Compute the symmetric half-width quantile per value column.

        As with the asymmetric variant, the reduction runs down each column so
        a column's interval is calibrated from its own scores alone.

        Parameters
        ----------
        conformity_scores : pl.DataFrame
            Conformity scores from calibration, one column per value column.

        coverage_rate : float
            Target coverage rate.

        Returns
        -------
        list of float
            Quantile per column, in column order.

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

        # Same (n+1) conformal correction as the asymmetric variant, applied to
        # the single half-width instead of two tails.
        n_columns = conformity_array.shape[1] if conformity_array.ndim > 1 else 1
        if global_calibration:
            conformity_array = conformity_array.reshape(-1, 1)

        n = conformity_array.shape[0]
        ordered = np.sort(conformity_array, axis=0)
        index = min(max(int(np.ceil((n + 1) * coverage_rate)), 1), n)
        quantiles = ordered[index - 1]

        result = list(np.atleast_1d(quantiles))
        return result * n_columns if global_calibration else result

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
        self,
        y_pred: pl.DataFrame,
        conformity_scores: pl.DataFrame,
        coverage_rate: float,
        calibration_strategy: Literal["local", "global"] = "local",
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

        calibration_strategy : {"local", "global"}, default="local"
            Which columns' scores the quantile is drawn from: that column's
            own (``"local"``) or every column's pooled (``"global"``), each
            applied on the column's own reconstruction. ``"global"`` is only
            meaningful for scorers declaring the
            ``supports_global_calibration`` tag. Matches
            ``SplitConformalForecaster``'s parameter of the same name.

        Returns
        -------
        pl.DataFrame
            Prediction intervals.

        """
