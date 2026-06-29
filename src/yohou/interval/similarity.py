"""Similarity measures for interval forecasting."""

import math
from datetime import timedelta
from typing import Any, Literal

import numpy as np
import polars as pl
from scipy.spatial.distance import cdist
from sklearn.base import clone
from sklearn.utils.validation import check_is_fitted

from yohou.utils._compat import _BaseComposition, _fit_context

from .base import BaseSimilarity

__all__ = ["CompositeSimilarity", "DistanceSimilarity", "SeasonalSimilarity"]


class DistanceSimilarity(BaseSimilarity):
    r"""Distance-based similarity using scipy metrics for weighting observations.

    Computes observation weights by measuring the distance between new
    predictions and historical predictions in feature space. Closer
    historical observations receive higher weights, which are then used
    by interval forecasters to weight conformity scores when constructing
    prediction intervals.

    The weight for the *i*-th historical observation given prediction
    *j* is computed with a numerically-stable softmax of negative
    distances that reserves uniform mass for the (hypothetical) test
    point over the calibration axis:

    $$w_{ji} = \frac{\exp(-(d_{ji} - \max_k d_{jk}))}
    {1 + \sum_k \exp(-(d_{jk} - \max_k d_{jk}))}$$

    where $d_{ji} = d(x_j, x_i)$ for the chosen distance metric. The
    ``+1`` in the denominator reserves mass for the new test point
    (Barber et al., 2023), so each row sums to strictly less than 1.

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
        self.metric_params = metric_params

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

        value_cols = [col for col in result.columns if col != "time"]
        if value_cols:
            bad_mask = result.select(
                (pl.col(col).is_null() | pl.col(col).cast(pl.Float64, strict=False).is_nan()).any().alias(col)
                for col in value_cols
            )
            bad_cols = [col for col in value_cols if bad_mask[col][0]]
            if bad_cols:
                raise ValueError(
                    f"Columns {bad_cols} contain null or NaN values. DistanceSimilarity requires complete data."
                )
        return result

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(
        self,
        y: pl.DataFrame,
        y_pred: pl.DataFrame,
        X_actual: pl.DataFrame | None = None,
    ) -> "DistanceSimilarity":
        """Store the calibration feature matrix for distance computation.

        Combines ``y_pred`` and ``X_actual`` (if provided) via ``_get_X``
        and saves the result as ``_X_observed``. Subsequent ``predict``
        calls compute distances from new predictions to this stored matrix.

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
        check_is_fitted(self, "_X_observed")
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
        check_is_fitted(self, "_X_observed")
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
        check_is_fitted(self, "_X_observed")
        X_features = self._get_X(y_pred, X_actual)

        XA = X_features.select(pl.exclude("time")).to_numpy()
        XB = self._X_observed.select(pl.exclude("time")).to_numpy()
        distances: np.ndarray = cdist(XA, XB, metric=self.metric, **(self.metric_params or {}))  # ty: ignore[no-matching-overload]
        return self._to_weights(distances)


class SeasonalSimilarity(BaseSimilarity):
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
    seasonality : list of float or None, default=None
        Seasonal periods in time steps (e.g. ``[7.0, 365.25]`` for
        weekly and yearly cycles on daily data). ``None`` is accepted at
        construction but invalid; passing ``None`` raises ``ValueError``
        at ``fit`` time, so a list must be provided before fitting.
    harmonics : dict mapping float to list of int, or None, default=None
        Harmonics to include per seasonality period. Keys must match
        entries in ``seasonality``. Each value is a list of positive
        integers specifying which harmonics to use. Defaults to
        ``{s: [1]}`` for each ``s`` in ``seasonality``.
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
        from calibration data. When ``fit`` receives a single timestamp the
        interval cannot be inferred and is set to ``timedelta(0)``; in that
        case ``_extract_features`` leaves the time axis in raw seconds, so
        ``seasonality`` must be expressed in seconds to remain meaningful.

    Notes
    -----
    Sin/cos encoding ensures that cyclic distances are correctly
    captured (e.g. December 31 is close to January 1). Multiple
    seasonalities combine naturally by concatenating feature vectors.

    The weight normalisation matches ``DistanceSimilarity`` exactly, via the
    shared [`BaseSimilarity._to_weights`][yohou.interval.base.BaseSimilarity]:
    a numerically-stable softmax of negative distances with uniform mass
    reserved for the test point, ``w_{ji} = raw_{ji} / (1 + \sum_k raw_{jk})``
    where ``raw_{ji} = \exp(-(d_{ji} - \max_k d_{jk}))``. Each row is
    non-negative and sums below 1, following the non-exchangeable conformal
    construction (Barber et al., 2023).

    References
    ----------
    [1] Barber, R.F., Candes, E.J., Ramdas, A., & Tibshirani, R.J.
        (2023). "Conformal prediction beyond exchangeability." Annals of
        Statistics, 51(2), 816-845.
        https://doi.org/10.1214/23-AOS2276

    See Also
    --------
    - [`DistanceSimilarity`][yohou.interval.similarity.DistanceSimilarity] : Value-based distance similarity.
    - [`BaseSimilarity`][yohou.interval.base.BaseSimilarity] : Abstract similarity base class.

    Examples
    --------
    >>> from datetime import datetime, timedelta
    >>> import polars as pl
    >>> import numpy as np
    >>> from yohou.interval.similarity import SeasonalSimilarity
    >>>
    >>> # Daily data with 3 weeks of calibration
    >>> dates = [datetime(2021, 1, 1) + timedelta(days=i) for i in range(21)]
    >>> y = pl.DataFrame({"time": dates, "value": np.random.randn(21)})
    >>> y_pred = pl.DataFrame({"time": dates, "value": np.random.randn(21)})
    >>>
    >>> # Fit with weekly seasonality
    >>> sim = SeasonalSimilarity(seasonality=[7.0])
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
        "seasonality": [list],
        "harmonics": [dict, None],
        "metric": [str],
        "metric_params": [dict, None],
    }

    def __init__(
        self,
        seasonality: list[float] | None = None,
        harmonics: dict[float, list[int]] | None = None,
        metric: str = "euclidean",
        metric_params: dict[str, object] | None = None,
    ) -> None:
        self.seasonality = seasonality
        self.harmonics = harmonics
        self.metric = metric
        self.metric_params = metric_params

    def _resolve_harmonics(self) -> dict[float, list[int]]:
        """Resolve harmonics, defaulting to first harmonic per seasonality.

        Returns
        -------
        dict[float, list[int]]
            Mapping from seasonality period to list of harmonic indices.

        """
        if self.harmonics is not None:
            return self.harmonics
        return {s: [1] for s in self.seasonality}  # ty: ignore[not-iterable]

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
        for s in self.seasonality:  # ty: ignore[not-iterable]
            for k in harmonics.get(s, [1]):
                angle = 2.0 * math.pi * k * t / s
                features.append(np.sin(angle))
                features.append(np.cos(angle))

        return np.column_stack(features)

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(
        self,
        y: pl.DataFrame,
        y_pred: pl.DataFrame,
        X_actual: pl.DataFrame | None = None,
    ) -> "SeasonalSimilarity":
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

        Notes
        -----
        When ``y_pred`` contains only a single timestamp, the time interval
        cannot be inferred and ``interval_td_`` is set to ``timedelta(0)``;
        the feature extractor then works in raw seconds and ``seasonality``
        must also be expressed in seconds to remain meaningful.

        Raises
        ------
        ValueError
            If ``seasonality`` is ``None`` or empty.

        """
        if self.seasonality is None or len(self.seasonality) == 0:
            raise ValueError("seasonality must be a non-empty list of floats")

        times = y_pred["time"]
        self.first_time_ = times[0]

        if len(times) > 1:
            self.interval_td_ = times[1] - times[0]
        else:
            self.interval_td_ = timedelta(0)

        self._features_observed = self._extract_features(times)

        return self

    def observe(
        self,
        y: pl.DataFrame,
        y_pred: pl.DataFrame,
        X_actual: pl.DataFrame | None = None,
    ) -> "SeasonalSimilarity":
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
        check_is_fitted(self, "first_time_")
        new_features = self._extract_features(y_pred["time"])
        self._features_observed = np.vstack([self._features_observed, new_features])
        return self

    def rewind(
        self,
        y: pl.DataFrame,
        y_pred: pl.DataFrame,
        X_actual: pl.DataFrame | None = None,
    ) -> "SeasonalSimilarity":
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
        check_is_fitted(self, "first_time_")
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
        check_is_fitted(self, "first_time_")
        new_features = self._extract_features(y_pred["time"])

        distances: np.ndarray = cdist(
            new_features,
            self._features_observed,
            metric=self.metric,
            **(self.metric_params or {}),
        )  # ty: ignore[no-matching-overload]
        return self._to_weights(distances)


class CompositeSimilarity(BaseSimilarity, _BaseComposition):
    r"""Combine multiple named similarity measures into a single weight vector.

    Delegates ``fit``, ``observe``, ``rewind``, and ``predict`` to each
    sub-similarity and then combines their weight matrices using either
    element-wise multiplication or weighted averaging. Sub-similarities are
    named ``(name, similarity)`` tuples, so their parameters are tunable via
    the ``similarities__<name>__<param>`` syntax (sklearn ``_BaseComposition``).

    Parameters
    ----------
    similarities : list of (str, BaseSimilarity) tuples
        At least two named sub-similarities to combine, e.g.
        ``[("dist", DistanceSimilarity()), ("seasonal", SeasonalSimilarity([7.0]))]``.
    combination : {"multiply", "mean"}, default="multiply"
        How to combine the individual weight matrices.

        ``"multiply"``
            Element-wise product with optional exponents:
            ``w_combined = prod(w_i ** alpha_i)``, then re-normalised with the
            shared per-row mass reservation.
        ``"mean"``
            Weighted average: ``w_combined = sum(alpha_i * w_i) / sum(alpha_i)``.
            Unlike ``"multiply"``, the ``"mean"`` path does not re-apply the
            per-row mass reservation; each row of the result sums to the
            weighted average of the sub-similarity row sums, which may exceed
            the strict ``(0, 1)`` mass-reservation guarantee of the
            ``"multiply"`` path.

    weights : list of float or None, default=None
        Per-similarity exponents (multiply) or mixing coefficients
        (mean), aligned with ``similarities``. If ``None``, all
        similarities contribute equally (exponents/coefficients of 1.0).

    Attributes
    ----------
    similarities_ : list of (str, BaseSimilarity) tuples
        Fitted copies of the named sub-similarities (set after ``fit``).

    See Also
    --------
    - [`DistanceSimilarity`][yohou.interval.similarity.DistanceSimilarity] : Value-based distance similarity.
    - [`SeasonalSimilarity`][yohou.interval.similarity.SeasonalSimilarity] : Seasonal-phase Fourier feature similarity.

    Examples
    --------
    >>> from datetime import datetime, timedelta
    >>> import polars as pl
    >>> import numpy as np
    >>> from yohou.interval.similarity import (
    ...     CompositeSimilarity,
    ...     DistanceSimilarity,
    ...     SeasonalSimilarity,
    ... )
    >>>
    >>> dates = [datetime(2021, 1, 1) + timedelta(days=i) for i in range(28)]
    >>> y = pl.DataFrame({"time": dates, "value": np.random.randn(28)})
    >>> y_pred = pl.DataFrame({"time": dates, "value": np.random.randn(28)})
    >>>
    >>> comp = CompositeSimilarity(
    ...     similarities=[
    ...         ("dist", DistanceSimilarity(metric="euclidean")),
    ...         ("seasonal", SeasonalSimilarity(seasonality=[7.0])),
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
        similarities: list[tuple[str, BaseSimilarity]] | None = None,
        combination: Literal["multiply", "mean"] = "multiply",
        weights: list[float] | None = None,
    ) -> None:
        self.similarities = similarities
        self.combination = combination
        self.weights = weights

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Get parameters, including nested sub-similarity parameters.

        Parameters
        ----------
        deep : bool, default=True
            If True, include sub-similarity parameters as
            ``similarities__<name>__<param>``.

        Returns
        -------
        dict
            Parameter names mapped to values.

        """
        return self._get_params("similarities", deep=deep)

    def set_params(self, **params: Any) -> "CompositeSimilarity":
        """Set parameters, routing ``similarities__<name>__<param>`` to sub-similarities.

        Parameters
        ----------
        **params : dict
            Parameters to set.

        Returns
        -------
        self

        """
        self._set_params("similarities", **params)
        return self

    def _check_similarities(self) -> None:
        """Validate the composition parameters (not the sklearn ``_validate_params``)."""
        if self.similarities is None or len(self.similarities) < 2:
            raise ValueError(
                "CompositeSimilarity requires at least 2 sub-similarities, "
                f"got {0 if self.similarities is None else len(self.similarities)}"
            )
        for item in self.similarities:
            if not (isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], str)):
                raise ValueError(f"Each entry in `similarities` must be a (name, similarity) tuple, got {item!r}")
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

    @_fit_context(prefer_skip_nested_validation=True)
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
        self._check_similarities()
        self.similarities_ = [
            (name, clone(sim).fit(y=y, y_pred=y_pred, X_actual=X_actual))
            for name, sim in self.similarities  # ty: ignore[not-iterable]
        ]
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
        check_is_fitted(self, "similarities_")
        for _name, sim in self.similarities_:
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
        check_is_fitted(self, "similarities_")
        for _name, sim in self.similarities_:
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
        check_is_fitted(self, "similarities_")
        alphas = self._resolved_weights()
        weight_matrices = [sim.predict(y_pred=y_pred, X_actual=X_actual) for _name, sim in self.similarities_]

        if self.combination == "multiply":
            combined = np.ones_like(weight_matrices[0])
            for w, alpha in zip(weight_matrices, alphas, strict=True):
                combined *= np.power(w, alpha)
            # Shared per-row mass reservation (drops the former arbitrary feature-count factor).
            combined = self._reserve_mass(combined)
        else:  # mean
            combined = np.zeros_like(weight_matrices[0])
            for w, alpha in zip(weight_matrices, alphas, strict=True):
                combined += alpha * w
            total_alpha = sum(alphas)
            if total_alpha != 0:
                combined /= total_alpha

        return combined
