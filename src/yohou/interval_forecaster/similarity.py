"""Distance-based similarity measures for interval forecasting."""

from typing import Any

import numpy as np
import polars as pl
from scipy.spatial.distance import cdist

from .base import BaseSimilarity


class DistanceSimilarity(BaseSimilarity):
    """Distance-based similarity using scipy metrics for weighting observations.

    Parameters
    ----------
    metric : str, default="euclidean"
        Distance metric to use (e.g., "euclidean", "manhattan", "cosine").
        See scipy.spatial.distance.cdist for all options.

    metric_params : dict or None, default=None
        Additional parameters for the distance metric.

    Examples
    --------
    >>> from datetime import datetime
    >>> import polars as pl
    >>> import numpy as np
    >>> from yohou.interval_forecaster.similarity import DistanceSimilarity
    >>>
    >>> # Create training data
    >>> time_train = pl.datetime_range(
    ...     start=datetime(2021, 12, 16), end=datetime(2021, 12, 16, 0, 0, 7), interval="1s", eager=True
    ... )
    >>> y_train = pl.DataFrame({"time": time_train, "value": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]})
    >>> y_pred_train = pl.DataFrame({"time": time_train, "value": [1.1, 2.1, 2.9, 4.2, 4.8, 6.1, 7.0, 8.1]})
    >>>
    >>> # Fit similarity model
    >>> similarity = DistanceSimilarity(metric="euclidean")
    >>> _ = similarity.fit(y_train, y_pred_train)
    >>>
    >>> # Create new predictions to compute similarities for
    >>> time_test = pl.datetime_range(
    ...     start=datetime(2021, 12, 16, 0, 0, 8),
    ...     end=datetime(2021, 12, 16, 0, 0, 9),
    ...     interval="1s",
    ...     eager=True,
    ... )
    >>> y_pred_test = pl.DataFrame({"time": time_test, "value": [8.5, 9.2]})
    >>>
    >>> # Compute similarity weights
    >>> weights = similarity.predict(y_pred_test)
    >>> weights.shape
    (2, 8)
    >>> isinstance(weights, np.ndarray)
    True

    """

    def __init__(
        self,
        metric: str = "euclidean",
        metric_params: dict[str, object] | None = None,
    ) -> None:
        self.metric = metric
        self.metric_params = metric_params if metric_params is not None else {}

    @property
    def n_discarded_indices_(self) -> int:
        """Get number of discarded indices due to NaN values.

        Returns
        -------
        int
            Number of discarded observations.

        """
        return self._n_discarded_indices

    def _get_X(
        self,
        y_pred: pl.DataFrame,
        X: pl.DataFrame | None,
    ) -> pl.DataFrame:
        """Combine predictions and features into single feature matrix.

        Parameters
        ----------
        y_pred : pl.DataFrame
            Predictions.

        X : pl.DataFrame or None
            Exogenous features.

        Returns
        -------
        pl.DataFrame
            Combined feature matrix.

        """
        if X is not None:
            return pl.concat([y_pred, X], how="horizontal")
        return y_pred

    def fit(
        self,
        y: pl.DataFrame,
        y_pred: pl.DataFrame,
        X: pl.DataFrame | None = None,
    ) -> "DistanceSimilarity":
        """Fits the similarity model.

        Parameters
        ----------
        y : pl.DataFrame
            Target time series.

        y_pred : pl.DataFrame
            Point forecasts time series.

        X : pl.DataFrame or None, default=None
            Exogenous feature time series.

        """
        X_features = self._get_X(y_pred, X)
        self._X_observed = X_features.drop_nulls()

        self._n_discarded_indices = len(y_pred) - len(X_features)

        return self

    def update(
        self,
        y: pl.DataFrame,
        y_pred: pl.DataFrame,
        X: pl.DataFrame | None = None,
    ) -> "DistanceSimilarity":
        """Update similarity model with new observations.

        Parameters
        ----------
        y : pl.DataFrame
            New target values.

        y_pred : pl.DataFrame
            New predictions.

        X : pl.DataFrame or None, default=None
            New exogenous features.

        Returns
        -------
        self

        """
        X_features = self._get_X(y_pred, X)

        self._X_observed = pl.concat([self._X_observed, X_features])

        return self

    def predict(
        self,
        y_pred: pl.DataFrame,
        X: pl.DataFrame | None = None,
    ) -> np.ndarray[tuple[int, int], np.dtype[np.floating[Any]]]:
        """Compute similarity weights for new predictions.

        Parameters
        ----------
        y_pred : pl.DataFrame
            New predictions to compute similarities for.

        X : pl.DataFrame or None, default=None
            Exogenous features.

        Returns
        -------
        np.ndarray
            Similarity weight matrix.

        """
        X_features = self._get_X(y_pred, X)

        XA = X_features.select(pl.exclude("time")).to_numpy()
        XB = self._X_observed.select(pl.exclude("time")).to_numpy()
        distances: np.ndarray = cdist(XA, XB, metric=self.metric, **self.metric_params)  # type: ignore[arg-type]
        weights = np.reciprocal(np.exp(distances))

        weights = weights / np.sum(weights, axis=1)[:, np.newaxis] * self._X_observed.shape[1]
        weights = weights / (1 + np.sum(weights, axis=1)[:, np.newaxis])

        return weights
