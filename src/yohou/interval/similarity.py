"""Similarity measures for interval forecasting."""

import math
from datetime import timedelta
from typing import Any, Literal

import numpy as np
import polars as pl
from scipy.spatial.distance import cdist
from sklearn.base import clone

from .base import BaseSimilarity

__all__ = ["CompositeSimilarity", "DistanceSimilarity", "TemporalSimilarity"]


class DistanceSimilarity(BaseSimilarity):
    r"""Distance-based similarity using scipy metrics for weighting observations.

    Computes observation weights by measuring the distance between new
    predictions and historical predictions in feature space. Closer
    historical observations receive higher weights, which are then used
    by interval forecasters to weight conformity scores when constructing
    prediction intervals.

    The weight for the *i*-th historical observation given prediction
    *j* is computed as:

    $$w_{ji} = \frac{\exp(-d(x_j, x_i))}{\sum_k \exp(-d(x_j, x_k))}$$

    where *d* is the chosen distance metric.

    Parameters
    ----------
    metric : str, default="euclidean"
        Distance metric to use (e.g., ``"euclidean"``, ``"cityblock"``,
        ``"cosine"``). Any metric supported by
        ``scipy.spatial.distance.cdist`` is accepted.

    metric_params : dict or None, default=None
        Additional keyword arguments forwarded to the distance metric
        function.

    Notes
    -----
    The distance-to-weight conversion uses the softmax of negative
    distances, so distant observations contribute exponentially less
    than nearby ones. The weights are further normalised so that each
    prediction row sums to a value in (0, 1).

    References
    ----------
    [1] Lei, J., G'Sell, M., Rinaldo, A., Tibshirani, R.J., &
        Wasserman, L. (2018). "Distribution-free predictive inference for
        regression." Journal of the American Statistical Association,
        113(523), 1094-1111.
        https://doi.org/10.1080/01621459.2017.1307116
    [2] Barber, R.F., Candes, E.J., Ramdas, A., & Tibshirani, R.J.
        (2023). "Conformal prediction beyond exchangeability." Annals of
        Statistics, 51(2), 816-845.
        https://doi.org/10.1214/23-AOS2276

    See Also
    --------
    - [`BaseSimilarity`][yohou.interval.base.BaseSimilarity] : Abstract similarity base class.
    - [`BaseIntervalForecaster`][yohou.interval.base.BaseIntervalForecaster] :
        Interval forecaster that can consume similarity weights.

    Examples
    --------
    >>> from datetime import datetime
    >>> import polars as pl
    >>> import numpy as np
    >>> from yohou.interval.similarity import DistanceSimilarity
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

    _parameter_constraints: dict = {
        "metric": [str],
        "metric_params": [dict, None],
    }

    def __init__(
        self,
        metric: str = "euclidean",
        metric_params: dict[str, object] | None = None,
    ) -> None:
        self.metric = metric
        self.metric_params = metric_params if metric_params is not None else {}

    def _get_X(
        self,
        y_pred: pl.DataFrame,
        X_actual: pl.DataFrame | None,
    ) -> pl.DataFrame:
        """Combine predictions and features into single feature matrix.

        Drops the ``"time"`` column from ``X`` before concatenation to
        avoid duplicate columns.  Validates that no column (except
        ``"time"``) contains null or NaN values.

        Parameters
        ----------
        y_pred : pl.DataFrame
            Predictions.

        X_actual : pl.DataFrame or None
            Exogenous features.

        Returns
        -------
        pl.DataFrame
            Combined feature matrix.

        Raises
        ------
        ValueError
            If any non-time column contains null or NaN values.

        """
        if X_actual is not None:
            X_no_time = X_actual.drop("time", strict=False)
            result = pl.concat([y_pred, X_no_time], how="horizontal")
        else:
            result = y_pred

        for col in result.columns:
            if col == "time":
                continue
            series = result[col]
            if series.null_count() > 0 or series.cast(pl.Float64, strict=False).is_nan().sum() > 0:
                raise ValueError(
                    f"Column '{col}' contains null or NaN values. DistanceSimilarity requires complete data."
                )
        return result

    def fit(
        self,
        y: pl.DataFrame,
        y_pred: pl.DataFrame,
        X_actual: pl.DataFrame | None = None,
    ) -> "DistanceSimilarity":
        """Fits the similarity model.

        Parameters
        ----------
        y : pl.DataFrame
            Target time series.

        y_pred : pl.DataFrame
            Point forecasts time series.

        X_actual : pl.DataFrame or None, default=None
            Exogenous feature time series.

        Returns
        -------
        self

        """
        X_features = self._get_X(y_pred, X_actual)
        self._X_observed = X_features

        self._n_discarded_indices = len(y_pred) - len(X_features)

        return self

    def observe(
        self,
        y: pl.DataFrame,
        y_pred: pl.DataFrame,
        X_actual: pl.DataFrame | None = None,
    ) -> "DistanceSimilarity":
        """Observe new data and update similarity model.

        Parameters
        ----------
        y : pl.DataFrame
            New target values.

        y_pred : pl.DataFrame
            New predictions.

        X_actual : pl.DataFrame or None, default=None
            New exogenous features.

        Returns
        -------
        self

        """
        X_features = self._get_X(y_pred, X_actual)

        self._X_observed = pl.concat([self._X_observed, X_features])

        return self

    def rewind(
        self,
        y: pl.DataFrame,
        y_pred: pl.DataFrame,
        X_actual: pl.DataFrame | None = None,
    ) -> "DistanceSimilarity":
        """Rewind the most recently observed data.

        Removes the last ``len(y)`` rows from the internal reference
        matrix, reversing the effect of the corresponding ``observe()``
        call.

        Parameters
        ----------
        y : pl.DataFrame
            Target observations to rewind (used only for row count).

        y_pred : pl.DataFrame
            Predictions to rewind (used only for row count).

        X_actual : pl.DataFrame or None, default=None
            Exogenous features to rewind (unused).

        Returns
        -------
        self

        """
        n_rewind = len(y)
        self._X_observed = self._X_observed[: len(self._X_observed) - n_rewind]
        return self

    def predict(
        self,
        y_pred: pl.DataFrame,
        X_actual: pl.DataFrame | None = None,
    ) -> np.ndarray[tuple[int, int], np.dtype[np.floating[Any]]]:
        """Compute similarity weights for new predictions.

        Parameters
        ----------
        y_pred : pl.DataFrame
            New predictions to compute similarities for.

        X_actual : pl.DataFrame or None, default=None
            Exogenous features.

        Returns
        -------
        np.ndarray
            Similarity weight matrix.

        """
        X_features = self._get_X(y_pred, X_actual)

        XA = X_features.select(pl.exclude("time")).to_numpy()
        XB = self._X_observed.select(pl.exclude("time")).to_numpy()
        distances: np.ndarray = cdist(XA, XB, metric=self.metric, **self.metric_params)  # ty: ignore[no-matching-overload]
        neg_d = -distances
        weights = np.exp(neg_d - np.max(neg_d, axis=1, keepdims=True))

        weights = weights / np.sum(weights, axis=1)[:, np.newaxis] * self._X_observed.shape[1]
        weights = weights / (1 + np.sum(weights, axis=1)[:, np.newaxis])

        return weights


class TemporalSimilarity(BaseSimilarity):
    r"""Temporal similarity using Fourier features for weighting observations.

    Computes observation weights by measuring the distance between
    cyclic temporal features extracted from prediction timestamps.
    Observations at similar seasonal positions (e.g. same day of week,
    same month of year) receive higher weights.

    Timestamps are converted to step indices relative to the first
    observed timestamp, then encoded as sin/cos pairs at the specified
    seasonal periods. Distances in this feature space are converted to
    weights using the same softmax formula as ``DistanceSimilarity``.

    Parameters
    ----------
    seasonalities : list of float
        Seasonal periods in time steps (e.g. ``[7.0, 365.25]`` for
        weekly and yearly cycles on daily data).
    harmonics : dict mapping float to list of int, or None, default=None
        Harmonics to include per seasonality period. Keys must match
        entries in ``seasonalities``. Each value is a list of positive
        integers specifying which harmonics to use. Defaults to
        ``{s: [1]}`` for each ``s`` in ``seasonalities``.
    metric : str, default="euclidean"
        Distance metric for ``scipy.spatial.distance.cdist``.
    metric_params : dict or None, default=None
        Additional keyword arguments forwarded to the distance metric.

    Attributes
    ----------
    first_time_ : datetime
        Reference timestamp from the first calibration prediction.
    interval_td_ : timedelta
        Time interval between consecutive timestamps, auto-detected
        from calibration data.

    Notes
    -----
    Sin/cos encoding ensures that cyclic distances are correctly
    captured (e.g. December 31 is close to January 1). Multiple
    seasonalities combine naturally by concatenating feature vectors.

    The weight normalisation matches ``DistanceSimilarity`` exactly:

    $$w_{ji} = \frac{\exp(-d(x_j, x_i))}{\sum_k \exp(-d(x_j, x_k))} \cdot n_{\text{features}}$$

    followed by

    $$w_{ji} \leftarrow \frac{w_{ji}}{1 + \sum_k w_{jk}}$$

    This reserves probability mass for a uniform component, following
    the conformal prediction literature.

    See Also
    --------
    - [`DistanceSimilarity`][yohou.interval.similarity.DistanceSimilarity] : Value-based distance similarity.
    - [`BaseSimilarity`][yohou.interval.base.BaseSimilarity] : Abstract similarity base class.

    Examples
    --------
    >>> from datetime import datetime, timedelta
    >>> import polars as pl
    >>> import numpy as np
    >>> from yohou.interval.similarity import TemporalSimilarity
    >>>
    >>> # Daily data with 3 weeks of calibration
    >>> dates = [datetime(2021, 1, 1) + timedelta(days=i) for i in range(21)]
    >>> y = pl.DataFrame({"time": dates, "value": np.random.randn(21)})
    >>> y_pred = pl.DataFrame({"time": dates, "value": np.random.randn(21)})
    >>>
    >>> # Fit with weekly seasonality
    >>> sim = TemporalSimilarity(seasonalities=[7.0])
    >>> _ = sim.fit(y, y_pred)
    >>>
    >>> # Predict weights for a new Monday
    >>> new_date = [datetime(2021, 1, 22)]
    >>> y_pred_new = pl.DataFrame({"time": new_date, "value": [0.5]})
    >>> weights = sim.predict(y_pred_new)
    >>> weights.shape
    (1, 21)

    """

    _parameter_constraints: dict = {
        "seasonalities": [list],
        "harmonics": [dict, None],
        "metric": [str],
        "metric_params": [dict, None],
    }

    def __init__(
        self,
        seasonalities: list[float] | None = None,
        harmonics: dict[float, list[int]] | None = None,
        metric: str = "euclidean",
        metric_params: dict[str, object] | None = None,
    ) -> None:
        self.seasonalities = seasonalities
        self.harmonics = harmonics
        self.metric = metric
        self.metric_params = metric_params if metric_params is not None else {}

    def _resolve_harmonics(self) -> dict[float, list[int]]:
        """Resolve harmonics, defaulting to first harmonic per seasonality.

        Returns
        -------
        dict[float, list[int]]
            Mapping from seasonality period to list of harmonic indices.

        """
        if self.harmonics is not None:
            return self.harmonics
        return {s: [1] for s in self.seasonalities}  # ty: ignore[not-iterable]

    def _extract_features(self, times: pl.Series) -> np.ndarray:
        """Extract Fourier features from a datetime series.

        Parameters
        ----------
        times : pl.Series
            Datetime series from which to compute features.

        Returns
        -------
        np.ndarray
            Feature matrix of shape ``(len(times), n_features)``.

        """
        time_diff = times - self.first_time_
        t = time_diff.dt.total_seconds().to_numpy().astype(np.float64)
        interval_seconds = self.interval_td_.total_seconds()
        if interval_seconds != 0:
            t = t / interval_seconds

        harmonics = self._resolve_harmonics()
        features = []
        for s in self.seasonalities:  # ty: ignore[not-iterable]
            for k in harmonics.get(s, [1]):
                angle = 2.0 * math.pi * k * t / s
                features.append(np.sin(angle))
                features.append(np.cos(angle))

        return np.column_stack(features)

    def fit(
        self,
        y: pl.DataFrame,
        y_pred: pl.DataFrame,
        X_actual: pl.DataFrame | None = None,
    ) -> "TemporalSimilarity":
        """Fit the temporal similarity from calibration predictions.

        Auto-detects the time interval from consecutive timestamps in
        ``y_pred`` and stores a reference timestamp and Fourier feature
        matrix for later distance computation.

        Parameters
        ----------
        y : pl.DataFrame
            Target time series (unused, accepted for API consistency).
        y_pred : pl.DataFrame
            Point forecast time series with a ``"time"`` column.
        X_actual : pl.DataFrame or None, default=None
            Exogenous features (unused, accepted for API consistency).

        Returns
        -------
        self

        """
        if self.seasonalities is None or len(self.seasonalities) == 0:
            raise ValueError("seasonalities must be a non-empty list of floats")

        times = y_pred["time"]
        self.first_time_ = times[0]

        if len(times) > 1:
            self.interval_td_ = times[1] - times[0]
        else:
            self.interval_td_ = timedelta(0)

        self._features_observed = self._extract_features(times)
        self._n_features = self._features_observed.shape[1]

        return self

    def observe(
        self,
        y: pl.DataFrame,
        y_pred: pl.DataFrame,
        X_actual: pl.DataFrame | None = None,
    ) -> "TemporalSimilarity":
        """Observe new data and extend the reference feature matrix.

        Parameters
        ----------
        y : pl.DataFrame
            New target observations (unused, accepted for API
            consistency).
        y_pred : pl.DataFrame
            New predictions with a ``"time"`` column.
        X_actual : pl.DataFrame or None, default=None
            Exogenous features (unused, accepted for API consistency).

        Returns
        -------
        self

        """
        new_features = self._extract_features(y_pred["time"])
        self._features_observed = np.vstack([self._features_observed, new_features])
        return self

    def rewind(
        self,
        y: pl.DataFrame,
        y_pred: pl.DataFrame,
        X_actual: pl.DataFrame | None = None,
    ) -> "TemporalSimilarity":
        """Rewind the most recently observed data.

        Removes the last ``len(y)`` rows from the internal feature
        matrix, reversing the effect of the corresponding ``observe()``
        call.

        Parameters
        ----------
        y : pl.DataFrame
            Target observations to rewind (used only for row count).
        y_pred : pl.DataFrame
            Predictions to rewind (used only for row count).
        X_actual : pl.DataFrame or None, default=None
            Exogenous features to rewind (unused).

        Returns
        -------
        self

        """
        n_rewind = len(y)
        self._features_observed = self._features_observed[: len(self._features_observed) - n_rewind]
        return self

    def predict(
        self,
        y_pred: pl.DataFrame,
        X_actual: pl.DataFrame | None = None,
    ) -> np.ndarray[tuple[int, int], np.dtype[np.floating[Any]]]:
        """Compute temporal similarity weights for new predictions.

        Parameters
        ----------
        y_pred : pl.DataFrame
            New predictions with a ``"time"`` column.
        X_actual : pl.DataFrame or None, default=None
            Exogenous features (unused).

        Returns
        -------
        np.ndarray
            Weight matrix of shape ``(n_predictions, n_calibration)``.

        """
        new_features = self._extract_features(y_pred["time"])

        distances: np.ndarray = cdist(
            new_features,
            self._features_observed,
            metric=self.metric,
            **self.metric_params,
        )  # ty: ignore[no-matching-overload]
        weights = np.exp(-distances)

        weights = weights / np.sum(weights, axis=1)[:, np.newaxis] * self._n_features
        weights = weights / (1 + np.sum(weights, axis=1)[:, np.newaxis])

        return weights


class CompositeSimilarity(BaseSimilarity):
    r"""Combine multiple similarity measures into a single weight vector.

    Delegates ``fit``, ``observe``, ``rewind``, and ``predict`` to each
    sub-similarity and then combines their weight matrices using either
    element-wise multiplication or weighted averaging.

    Parameters
    ----------
    similarities : list of BaseSimilarity
        At least two similarity instances to combine.
    combination : {"multiply", "mean"}, default="multiply"
        How to combine the individual weight matrices.

        ``"multiply"``
            Element-wise product with optional exponents:
            ``w_combined = prod(w_i ** alpha_i)``, then re-normalised
            so rows lie in (0, 1).
        ``"mean"``
            Weighted average: ``w_combined = sum(alpha_i * w_i)``.

    weights : list of float or None, default=None
        Per-similarity exponents (multiply) or mixing coefficients
        (mean). If ``None``, all similarities contribute equally
        (exponents/coefficients of 1.0).

    Attributes
    ----------
    similarities_ : list of BaseSimilarity
        Fitted copies of the sub-similarities (set after ``fit``).

    See Also
    --------
    - [`DistanceSimilarity`][yohou.interval.similarity.DistanceSimilarity] : Value-based distance similarity.
    - [`TemporalSimilarity`][yohou.interval.similarity.TemporalSimilarity] : Temporal Fourier feature similarity.

    Examples
    --------
    >>> from datetime import datetime, timedelta
    >>> import polars as pl
    >>> import numpy as np
    >>> from yohou.interval.similarity import (
    ...     CompositeSimilarity,
    ...     DistanceSimilarity,
    ...     TemporalSimilarity,
    ... )
    >>>
    >>> dates = [datetime(2021, 1, 1) + timedelta(days=i) for i in range(28)]
    >>> y = pl.DataFrame({"time": dates, "value": np.random.randn(28)})
    >>> y_pred = pl.DataFrame({"time": dates, "value": np.random.randn(28)})
    >>>
    >>> comp = CompositeSimilarity(
    ...     similarities=[
    ...         DistanceSimilarity(metric="euclidean"),
    ...         TemporalSimilarity(seasonalities=[7.0]),
    ...     ],
    ...     combination="multiply",
    ... )
    >>> _ = comp.fit(y, y_pred)
    >>> new_date = [datetime(2021, 1, 29)]
    >>> y_pred_new = pl.DataFrame({"time": new_date, "value": [0.5]})
    >>> weights = comp.predict(y_pred_new)
    >>> weights.shape
    (1, 28)

    """

    _parameter_constraints: dict = {
        "similarities": [list],
        "combination": [str],
        "weights": [list, None],
    }

    def __init__(
        self,
        similarities: list[BaseSimilarity] | None = None,
        combination: Literal["multiply", "mean"] = "multiply",
        weights: list[float] | None = None,
    ) -> None:
        self.similarities = similarities
        self.combination = combination
        self.weights = weights

    def _validate_params(self) -> None:
        """Validate constructor parameters."""
        if self.similarities is None or len(self.similarities) < 2:
            raise ValueError(
                "CompositeSimilarity requires at least 2 sub-similarities, "
                f"got {0 if self.similarities is None else len(self.similarities)}"
            )
        if self.combination not in ("multiply", "mean"):
            raise ValueError(f"combination must be 'multiply' or 'mean', got {self.combination!r}")
        if self.weights is not None and len(self.weights) != len(self.similarities):
            raise ValueError(
                f"weights length ({len(self.weights)}) must match similarities length ({len(self.similarities)})"
            )

    def _resolved_weights(self) -> list[float]:
        """Return per-similarity weights, defaulting to 1.0 each."""
        if self.weights is not None:
            return self.weights
        return [1.0] * len(self.similarities)  # ty: ignore[invalid-argument-type]

    def fit(
        self,
        y: pl.DataFrame,
        y_pred: pl.DataFrame,
        X_actual: pl.DataFrame | None = None,
    ) -> "CompositeSimilarity":
        """Fit all sub-similarities on the calibration data.

        Parameters
        ----------
        y : pl.DataFrame
            Target time series.
        y_pred : pl.DataFrame
            Point forecast time series.
        X_actual : pl.DataFrame or None, default=None
            Exogenous features.

        Returns
        -------
        self

        """
        self._validate_params()
        self.similarities_ = [clone(sim).fit(y=y, y_pred=y_pred, X_actual=X_actual) for sim in self.similarities]  # ty: ignore[not-iterable]
        return self

    def observe(
        self,
        y: pl.DataFrame,
        y_pred: pl.DataFrame,
        X_actual: pl.DataFrame | None = None,
    ) -> "CompositeSimilarity":
        """Forward observation to all sub-similarities.

        Parameters
        ----------
        y : pl.DataFrame
            New target observations.
        y_pred : pl.DataFrame
            New predictions.
        X_actual : pl.DataFrame or None, default=None
            New exogenous features.

        Returns
        -------
        self

        """
        for sim in self.similarities_:
            sim.observe(y=y, y_pred=y_pred, X_actual=X_actual)
        return self

    def rewind(
        self,
        y: pl.DataFrame,
        y_pred: pl.DataFrame,
        X_actual: pl.DataFrame | None = None,
    ) -> "CompositeSimilarity":
        """Forward rewind to all sub-similarities.

        Parameters
        ----------
        y : pl.DataFrame
            Target observations to rewind.
        y_pred : pl.DataFrame
            Predictions to rewind.
        X_actual : pl.DataFrame or None, default=None
            Exogenous features to rewind.

        Returns
        -------
        self

        """
        for sim in self.similarities_:
            sim.rewind(y=y, y_pred=y_pred, X_actual=X_actual)
        return self

    def predict(
        self,
        y_pred: pl.DataFrame,
        X_actual: pl.DataFrame | None = None,
    ) -> np.ndarray[tuple[int, int], np.dtype[np.floating[Any]]]:
        """Combine sub-similarity weights into a single weight matrix.

        Parameters
        ----------
        y_pred : pl.DataFrame
            Predictions to compute similarities for.
        X_actual : pl.DataFrame or None, default=None
            Exogenous features.

        Returns
        -------
        np.ndarray
            Combined weight matrix of shape
            ``(n_predictions, n_calibration)``.

        """
        alphas = self._resolved_weights()
        weight_matrices = [sim.predict(y_pred=y_pred, X_actual=X_actual) for sim in self.similarities_]

        if self.combination == "multiply":
            combined = np.ones_like(weight_matrices[0])
            for w, alpha in zip(weight_matrices, alphas, strict=True):
                combined *= np.power(w, alpha)
            # Re-normalise so rows lie in (0, 1)
            row_sums = combined.sum(axis=1, keepdims=True)
            row_sums = np.where(row_sums == 0, 1.0, row_sums)
            n_features = combined.shape[1]
            combined = combined / row_sums * n_features
            combined = combined / (1 + combined.sum(axis=1, keepdims=True))
        else:  # mean
            combined = np.zeros_like(weight_matrices[0])
            for w, alpha in zip(weight_matrices, alphas, strict=True):
                combined += alpha * w
            total_alpha = sum(alphas)
            if total_alpha != 0:
                combined /= total_alpha

        return combined
