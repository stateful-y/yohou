from typing import Optional

import numpy as np
import polars as pl
from scipy.spatial.distance import cdist

from .base import BaseSimilarity


class DistanceSimilarity(BaseSimilarity):
    def __init__(
        self,
        metric: str = "euclidean",
        metric_params: dict[str, object] | None = None,
    ) -> None:
        self.metric = metric
        self.metric_params = metric_params if metric_params is not None else {}

    @property
    def n_discarded_indices_(self) -> int:
        return self._n_discarded_indices

    def _get_X(
        self,
        y_pred: pl.DataFrame,
        X_ante: pl.DataFrame | None,
        X_post: pl.DataFrame | None,
    ) -> pl.DataFrame:
        X = y_pred
        if X_ante is not None:
            X = pl.concat([X, X_ante], how="horizontal")

        if X_post is not None:
            X = pl.concat([X, X_post], how="horizontal")

        return X

    def fit(
        self,
        y: pl.DataFrame,
        y_pred: pl.DataFrame,
        X_ante: Optional[pl.DataFrame] = None,
        X_post: Optional[pl.DataFrame] = None,
    ) -> "DistanceSimilarity":
        """Fits the similarity model.

        Parameters
        ----------
        y : pl.DataFrame
            Target time series.

        y_pred : pl.DataFrame
            Point forecasts time series.

        X_ante : pl.DataFrame or None, default=None
            Ex-ante feature time series.

        X_post : pl.DataFrame or None, default=None
            Ex-post feature time series.

        """
        X = self._get_X(y_pred, X_ante, X_post)
        self._X_observed = X.dropna()

        self._n_discarded_indices = len(y_pred) - len(X)

        return self

    def update(
        self,
        y: pl.DataFrame,
        y_pred: pl.DataFrame,
        X_ante: Optional[pl.DataFrame] = None,
        X_post: Optional[pl.DataFrame] = None,
    ) -> "DistanceSimilarity":
        X = self._get_X(y_pred, X_ante, X_post)

        self._X_observed = pl.concat([self._X_observed, X])

        return self

    def predict(
        self,
        y_pred: pl.DataFrame,
        X_ante: Optional[pl.DataFrame] = None,
        X_post: Optional[pl.DataFrame] = None,
    ) -> np.ndarray[tuple[int, int], np.dtype[np.floating[object]]]:
        X = self._get_X(y_pred, X_ante, X_post)

        distances = cdist(X, self._X_observed, self.metric, **self.metrics_params)
        weights = np.reciprocal(np.exp(distances))

        weights = weights / np.sum(weights, axis=1)[:, np.newaxis] * self._X_observed.shape[1]
        weights = weights / (1 + np.sum(weights, axis=1)[:, np.newaxis])

        return weights
