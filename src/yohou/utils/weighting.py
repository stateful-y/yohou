"""Estimator-based time-axis weighting for scorers and forecasters.

This module provides :class:`BaseWeighter` and concrete weighter estimators
that produce time-axis weights for evaluation (scorers) and training-sample
weighting (reduction forecasters). Weighters are scikit-learn estimators: their
parameters are introspectable via ``get_params``/``set_params`` and therefore
tunable with :class:`~yohou.model_selection.GridSearchCV`. Each weighter exposes
``compute_weights(key, group_name=None) -> pl.Series`` returning one weight per
element of the key series (datetime, vintage time, or integer forecasting step).
"""

from __future__ import annotations

import abc
import numbers
from datetime import timedelta
from typing import Any

import numpy as np
import polars as pl
from sklearn.base import BaseEstimator

from yohou.utils._compat import Interval, StrOptions, _BaseComposition

__all__ = [
    "BaseWeighter",
    "ExponentialDecayWeighter",
    "LinearDecayWeighter",
    "LookupWeighter",
    "ProductWeighter",
    "SeasonalEmphasisWeighter",
    "TableWeighter",
    "combine_weight_vectors",
    "normalize_weights",
    "resolve_weighter_to_array",
    "validate_weight_array",
]


class BaseWeighter(BaseEstimator, metaclass=abc.ABCMeta):
    """Base class for time-axis weighter estimators.

    A weighter maps a *key* series (datetime times, vintage times, or integer
    forecasting steps) to a series of non-negative weights. Concrete subclasses
    declare their tunable parameters in ``_parameter_constraints`` and implement
    :meth:`compute_weights`. Because weighters are scikit-learn estimators, their
    parameters can be addressed with the ``__`` syntax (e.g.
    ``time_weighter__half_life``) and tuned by search.

    Notes
    -----
    Weighters are stateless by default: :meth:`fit` is a no-op that exists so a
    future data-dependent weighter can hook into the host estimator's ``fit``
    without an API change.

    See Also
    --------
    - [`ExponentialDecayWeighter`][yohou.utils.weighting.ExponentialDecayWeighter] : Recency decay weighting.
    - [`LinearDecayWeighter`][yohou.utils.weighting.LinearDecayWeighter] : Linear recency weighting.
    - [`SeasonalEmphasisWeighter`][yohou.utils.weighting.SeasonalEmphasisWeighter] : Seasonal emphasis weighting.
    - [`LookupWeighter`][yohou.utils.weighting.LookupWeighter] : Explicit per-key weights.
    - [`TableWeighter`][yohou.utils.weighting.TableWeighter] : DataFrame-driven weights.
    - [`ProductWeighter`][yohou.utils.weighting.ProductWeighter] : Compose weighters by multiplication.

    """

    _parameter_constraints: dict = {}

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Merge parameter constraints from all classes in the MRO."""
        super().__init_subclass__(**kwargs)
        merged: dict = {}
        for klass in reversed(cls.__mro__):
            own = klass.__dict__.get("_parameter_constraints")
            if own and isinstance(own, dict):
                merged.update(own)
        cls._parameter_constraints = merged

    def fit(self, y: pl.DataFrame | None = None, **params: Any) -> BaseWeighter:
        """Fit the weighter (no-op by default).

        Parameters
        ----------
        y : pl.DataFrame or None, default=None
            Training data. Ignored by stateless weighters.
        **params : dict
            Ignored. Present for routing compatibility.

        Returns
        -------
        self

        """
        return self

    @abc.abstractmethod
    def compute_weights(self, key: pl.Series, group_name: str | None = None) -> pl.Series:
        """Compute one weight per element of ``key``.

        Parameters
        ----------
        key : pl.Series
            Key series to weight: datetime times, vintage times, or integer
            forecasting steps.
        group_name : str or None, default=None
            Panel group name for panel-aware weighting, or None for global data.

        Returns
        -------
        pl.Series
            Float64 weight series aligned to ``key`` (alias ``"weight"``).

        """
        raise NotImplementedError


class ExponentialDecayWeighter(BaseWeighter):
    r"""Exponential decay weights giving more weight to recent keys.

    Computes weights via exponential decay from the most recent key:

    $$w(k) = \exp\left(-\ln(2) \cdot \frac{d(k)}{\text{half\_life}}\right)$$

    where $d(k)$ is the distance from the most recent key. With
    ``scale="elapsed"`` the distance is the real elapsed time to the most recent
    datetime; with ``scale="position"`` it is the rank-index distance (usable for
    integer forecasting steps or regularly-spaced time).

    Parameters
    ----------
    half_life : int, float, or datetime.timedelta
        Distance over which weight decays to half. With ``scale="elapsed"`` an
        int/float is interpreted as a number of days and a ``timedelta`` as a
        duration. With ``scale="position"`` it is a number of steps (a
        ``timedelta`` is invalid and raises ``ValueError``).
    scale : {"elapsed", "position"} or None, default=None
        Decay basis. ``"elapsed"`` decays by real elapsed time (datetime keys);
        ``"position"`` decays by rank index. When None, inferred from the key
        dtype (datetime -> ``"elapsed"``, numeric -> ``"position"``).

    Notes
    -----
    The most recent (largest) key receives weight 1.0; smaller keys decay.

    References
    ----------
    [1] Hyndman, R.J., & Athanasopoulos, G. (2021). "Forecasting:
        principles and practice," 3rd edition, OTexts. Chapter 8.1.

    See Also
    --------
    - [`LinearDecayWeighter`][yohou.utils.weighting.LinearDecayWeighter] : Linear recency weighting.
    - [`SeasonalEmphasisWeighter`][yohou.utils.weighting.SeasonalEmphasisWeighter] : Seasonal emphasis weighting.
    - [`ProductWeighter`][yohou.utils.weighting.ProductWeighter] : Compose weighters by multiplication.

    Examples
    --------
    >>> import polars as pl
    >>> from datetime import datetime
    >>> times = pl.Series(
    ...     "time",
    ...     [datetime(2024, 1, 1), datetime(2024, 1, 2), datetime(2024, 1, 3)],
    ... )
    >>> ExponentialDecayWeighter(half_life=1).compute_weights(times)  # doctest: +NORMALIZE_WHITESPACE
    shape: (3,)
    Series: 'weight' [f64]
    [
        0.25
        0.5
        1.0
    ]

    """

    _parameter_constraints: dict = {
        "half_life": [Interval(numbers.Real, 0, None, closed="neither"), timedelta],
        "scale": [StrOptions({"elapsed", "position"}), None],
    }

    def __init__(self, half_life: int | float | timedelta = 1, scale: str | None = None) -> None:
        self.half_life = half_life
        self.scale = scale

    def _resolve_scale(self, key: pl.Series) -> str:
        """Resolve the effective decay scale, inferring from dtype if unset."""
        if self.scale is not None:
            return self.scale
        return "elapsed" if key.dtype.is_temporal() else "position"

    def compute_weights(self, key: pl.Series, group_name: str | None = None) -> pl.Series:
        """Compute exponential decay weights for ``key``."""
        self._validate_params()
        scale = self._resolve_scale(key)

        if scale == "elapsed":
            distances = (key.max() - key).dt.total_seconds().to_numpy().astype(np.float64)
            if isinstance(self.half_life, timedelta):
                half_life_seconds = self.half_life.total_seconds()
            else:
                half_life_seconds = timedelta(days=float(self.half_life)).total_seconds()
            weights = np.exp(-np.log(2) * distances / half_life_seconds)
        else:  # position
            if isinstance(self.half_life, timedelta):
                raise ValueError("timedelta half_life requires scale='elapsed', got scale='position'")
            ranks = (key.rank(method="ordinal") - 1).to_numpy().astype(np.float64)
            distances = ranks.max() - ranks
            weights = np.exp(-np.log(2) * distances / float(self.half_life))

        return pl.Series(weights, dtype=pl.Float64).alias("weight")


class LinearDecayWeighter(BaseWeighter):
    r"""Linear decay weights giving more weight to recent keys.

    Computes rank-based linear decay:

    $$w(k) = \frac{\text{rank}(k)}{n - 1}$$

    where $\text{rank}(k) = 0$ for the oldest key and $n - 1$ for the most
    recent. When ``max_steps`` is set, keys older than ``max_steps`` from the
    most recent receive weight 0. Rank-based, so usable for datetime or integer
    keys without a scale parameter.

    Parameters
    ----------
    max_steps : int or None, default=None
        Window over which to decay. If None, decays linearly across the whole
        range. If set, keys older than ``max_steps`` from the most recent receive
        weight 0.

    See Also
    --------
    - [`ExponentialDecayWeighter`][yohou.utils.weighting.ExponentialDecayWeighter] : Exponential recency weighting.
    - [`SeasonalEmphasisWeighter`][yohou.utils.weighting.SeasonalEmphasisWeighter] : Seasonal emphasis weighting.
    - [`ProductWeighter`][yohou.utils.weighting.ProductWeighter] : Compose weighters by multiplication.

    Examples
    --------
    >>> import polars as pl
    >>> from datetime import datetime
    >>> times = pl.Series("time", [datetime(2024, 1, d) for d in range(1, 5)])
    >>> LinearDecayWeighter().compute_weights(times)  # doctest: +NORMALIZE_WHITESPACE
    shape: (4,)
    Series: 'weight' [f64]
    [
        0.0
        0.333333
        0.666667
        1.0
    ]

    """

    _parameter_constraints: dict = {
        "max_steps": [Interval(numbers.Integral, 1, None, closed="left"), None],
    }

    def __init__(self, max_steps: int | None = None) -> None:
        self.max_steps = max_steps

    def compute_weights(self, key: pl.Series, group_name: str | None = None) -> pl.Series:
        """Compute linear decay weights for ``key``."""
        self._validate_params()
        n = len(key)
        if n == 1:
            return pl.Series([1.0], dtype=pl.Float64).alias("weight")

        ranks = (key.rank(method="ordinal") - 1).to_numpy()

        if self.max_steps is None:
            weights = ranks / (n - 1)
        else:
            cutoff = n - self.max_steps
            weights = np.where(
                ranks < cutoff,
                0.0,
                (ranks - cutoff) / (self.max_steps - 1) if self.max_steps > 1 else 1.0,
            )
            weights = np.minimum(1.0, weights)

        return pl.Series(weights, dtype=pl.Float64).alias("weight")


class SeasonalEmphasisWeighter(BaseWeighter):
    r"""Weights emphasizing keys in phase with the most recent seasonal position.

    Gives higher weight to keys matching the most recent seasonal position
    (e.g. same weekday). Rank-based, so usable for datetime or integer keys.

    Parameters
    ----------
    seasonality : int or list of int
        Seasonal period(s). If a list, keys matching ANY period receive emphasis.
    emphasis : float, default=2.0
        Weight multiplier for in-phase keys (out-of-phase keys receive 1.0).

    See Also
    --------
    - [`ExponentialDecayWeighter`][yohou.utils.weighting.ExponentialDecayWeighter] : Exponential recency weighting.
    - [`LinearDecayWeighter`][yohou.utils.weighting.LinearDecayWeighter] : Linear recency weighting.
    - [`ProductWeighter`][yohou.utils.weighting.ProductWeighter] : Compose weighters by multiplication.

    Examples
    --------
    >>> import polars as pl
    >>> from datetime import datetime
    >>> times = pl.Series(
    ...     "time",
    ...     [datetime(2024, 1, 1), datetime(2024, 1, 2), datetime(2024, 1, 8), datetime(2024, 1, 9)],
    ... )
    >>> SeasonalEmphasisWeighter(seasonality=7).compute_weights(times)  # doctest: +NORMALIZE_WHITESPACE
    shape: (4,)
    Series: 'weight' [f64]
    [
        1.0
        1.0
        1.0
        2.0
    ]

    """

    _parameter_constraints: dict = {
        "seasonality": [Interval(numbers.Integral, 1, None, closed="left"), list],
        "emphasis": [Interval(numbers.Real, 0, None, closed="neither")],
    }

    def __init__(self, seasonality: int | list[int] = 1, emphasis: float = 2.0) -> None:
        self.seasonality = seasonality
        self.emphasis = emphasis

    def compute_weights(self, key: pl.Series, group_name: str | None = None) -> pl.Series:
        """Compute seasonal emphasis weights for ``key``."""
        self._validate_params()
        seasonalities = [self.seasonality] if isinstance(self.seasonality, int) else self.seasonality

        n = len(key)
        if n == 1:
            return pl.Series([1.0], dtype=pl.Float64).alias("weight")

        ranks = key.rank(method="ordinal") - 1
        in_phase = np.zeros(n, dtype=bool)
        for s in seasonalities:
            most_recent_phase = int(ranks[-1]) % s
            phases = ranks.to_numpy() % s
            in_phase = in_phase | (phases == most_recent_phase)

        weights = np.where(in_phase, self.emphasis, 1.0)
        return pl.Series(weights, dtype=pl.Float64).alias("weight")


class LookupWeighter(BaseWeighter):
    r"""Explicit per-key weights from a mapping.

    Each key is looked up in ``mapping``; keys absent from ``mapping`` receive
    ``default``. Replaces the former ``{key: weight}`` dict input (the ``"*"``
    wildcard is replaced by the tunable ``default`` parameter, though a ``"*"``
    entry in ``mapping`` is still honored for compatibility).

    Parameters
    ----------
    mapping : dict
        Mapping from key value to weight.
    default : float, default=1.0
        Weight for keys absent from ``mapping``.

    See Also
    --------
    - [`TableWeighter`][yohou.utils.weighting.TableWeighter] : DataFrame-driven weights.
    - [`ProductWeighter`][yohou.utils.weighting.ProductWeighter] : Compose weighters by multiplication.

    Examples
    --------
    >>> import polars as pl
    >>> steps = pl.Series("forecasting_step", [1, 2, 3])
    >>> LookupWeighter(mapping={1: 1.0}, default=0.0).compute_weights(
    ...     steps
    ... )  # doctest: +NORMALIZE_WHITESPACE
    shape: (3,)
    Series: 'weight' [f64]
    [
        1.0
        0.0
        0.0
    ]

    """

    _parameter_constraints: dict = {
        "mapping": [dict, None],
        "default": [Interval(numbers.Real, 0, None, closed="left")],
    }

    def __init__(self, mapping: dict | None = None, default: float = 1.0) -> None:
        self.mapping = mapping
        self.default = default

    def compute_weights(self, key: pl.Series, group_name: str | None = None) -> pl.Series:
        """Compute lookup weights for ``key``."""
        self._validate_params()
        mapping = self.mapping or {}
        effective_default = mapping.get("*", self.default)
        lookup = {k: v for k, v in mapping.items() if k != "*"}
        weights = np.array([lookup.get(k, effective_default) for k in key.to_list()], dtype=np.float64)
        return pl.Series(weights, dtype=pl.Float64).alias("weight")


class TableWeighter(BaseWeighter):
    """DataFrame-driven weights resolved by joining on a key column.

    Replaces the former ``pl.DataFrame`` weight input. Weights are looked up by
    left-joining the key series to ``frame`` on column ``on``. For panel data, a
    ``{group_name}_weight`` column is used when present, otherwise ``weight``.
    A key with no match raises ``ValueError``.

    Parameters
    ----------
    frame : pl.DataFrame
        Weight table with the join column ``on`` and a ``"weight"`` column
        (and/or ``"{group}_weight"`` columns for panel data).
    on : str, default="time"
        Join column name (``"time"``, ``"vintage_time"``, or
        ``"forecasting_step"``).

    See Also
    --------
    - [`LookupWeighter`][yohou.utils.weighting.LookupWeighter] : Explicit per-key weights.
    - [`ProductWeighter`][yohou.utils.weighting.ProductWeighter] : Compose weighters by multiplication.

    """

    _parameter_constraints: dict = {
        "frame": [pl.DataFrame, None],
        "on": [str],
    }

    def __init__(self, frame: pl.DataFrame | None = None, on: str = "time") -> None:
        self.frame = frame
        self.on = on

    def compute_weights(self, key: pl.Series, group_name: str | None = None) -> pl.Series:
        """Compute join-based weights for ``key``."""
        self._validate_params()
        if self.frame is None:
            raise ValueError("TableWeighter requires a non-None `frame` to compute weights")
        key_df = pl.DataFrame({self.on: key})
        joined = key_df.join(self.frame, on=self.on, how="left")

        weight_col = None
        if group_name is not None:
            group_col = f"{group_name}_weight"
            if group_col in joined.columns:
                weight_col = group_col
        if weight_col is None and "weight" in joined.columns:
            weight_col = "weight"
        if weight_col is None:
            if group_name is not None:
                raise ValueError(
                    f"Weight DataFrame missing both '{group_name}_weight' and "
                    f"'weight' columns for panel group '{group_name}'"
                )
            raise ValueError("Weight DataFrame must have 'weight' column")

        weights_np = joined[weight_col].to_numpy().astype(np.float64)

        nan_mask = np.isnan(weights_np)
        if nan_mask.any():
            missing_keys = key.filter(pl.Series(nan_mask)).unique().to_list()
            raise ValueError(f"Weight DataFrame has no values for {self.on}s: {missing_keys}")

        return pl.Series(weights_np, dtype=pl.Float64).alias("weight")


class ProductWeighter(BaseWeighter, _BaseComposition):
    """Compose named weighters by element-wise multiplication.

    Multiplies the ``compute_weights`` outputs of its components. Sub-weighters
    are named ``(name, weighter)`` tuples, so their parameters are addressable
    as ``<name>__<param>`` (sklearn ``_BaseComposition``) and tunable — e.g.
    ``time_weighter__decay__half_life`` through a forecaster. This matches
    every other yohou compositor (``CompositeSimilarity``, ``FeaturePipeline``,
    ``ColumnForecaster``, the voting ensembles, …).

    Parameters
    ----------
    weighters : list of (str, BaseWeighter) tuples
        Named component weighters to multiply, e.g.
        ``[("decay", ExponentialDecayWeighter(7)), ("seasonal", SeasonalEmphasisWeighter(7))]``.
        Must be non-empty.

    See Also
    --------
    - [`ExponentialDecayWeighter`][yohou.utils.weighting.ExponentialDecayWeighter] : Exponential recency weighting.
    - [`SeasonalEmphasisWeighter`][yohou.utils.weighting.SeasonalEmphasisWeighter] : Seasonal emphasis weighting.

    Examples
    --------
    >>> import polars as pl
    >>> from datetime import datetime
    >>> times = pl.Series("time", [datetime(2024, 1, d) for d in range(1, 4)])
    >>> w = ProductWeighter([
    ...     ("decay", ExponentialDecayWeighter(2)),
    ...     ("seasonal", SeasonalEmphasisWeighter(2, 1.5)),
    ... ])
    >>> _ = w.compute_weights(times)

    """

    _parameter_constraints: dict = {
        "weighters": [list, None],
    }

    def __init__(self, weighters: list[tuple[str, BaseWeighter]] | None = None) -> None:
        self.weighters = weighters

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Get parameters, including nested component parameters.

        Parameters
        ----------
        deep : bool, default=True
            If True, include component parameters as ``<name>__<param>``.

        Returns
        -------
        dict
            Parameter names mapped to values.

        """
        return self._get_params("weighters", deep=deep)

    def set_params(self, **params: Any) -> ProductWeighter:
        """Set parameters, routing ``<name>__<param>`` to named sub-weighters.

        Parameters
        ----------
        **params : dict
            Parameters to set.

        Returns
        -------
        self

        """
        self._set_params("weighters", **params)
        return self

    def compute_weights(self, key: pl.Series, group_name: str | None = None) -> pl.Series:
        """Compute the product of all component weights for ``key``."""
        if not self.weighters:
            raise ValueError("ProductWeighter requires at least one weighter")
        result = pl.Series(np.ones(len(key)), dtype=pl.Float64)
        for item in self.weighters:
            if not (isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], str)):
                raise ValueError(f"Each entry in `weighters` must be a (name, weighter) tuple, got {item!r}")
            _name, weighter = item
            result = result * weighter.compute_weights(key, group_name)
        return result.alias("weight")


def resolve_weighter_to_array(
    weighter: BaseWeighter,
    key_series: pl.Series,
    group_name: str | None = None,
    name: str = "weight",
) -> np.ndarray:
    """Resolve a weighter to a validated numpy weight array aligned to ``key_series``.

    Parameters
    ----------
    weighter : BaseWeighter
        Weighter to evaluate.
    key_series : pl.Series
        Key series to weight (time, vintage time, or forecasting step).
    group_name : str or None, default=None
        Panel group name, or None for global data.
    name : str, default="weight"
        Descriptive name used in validation error messages.

    Returns
    -------
    numpy.ndarray
        Raw (unnormalized) float64 weight array of the same length as
        ``key_series``.

    Raises
    ------
    ValueError
        If the weighter returns a mis-sized series or the resolved weights fail
        validation (NaN, negative, infinite, or all-zero).

    """
    weights_series = weighter.compute_weights(key_series, group_name)
    if not isinstance(weights_series, pl.Series):
        raise ValueError(f"Weighter must return pl.Series, got {type(weights_series).__name__}")
    if len(weights_series) != len(key_series):
        raise ValueError(f"Weighter returned {len(weights_series)} weights, expected {len(key_series)} rows")
    weights_np = weights_series.to_numpy().astype(np.float64)
    validate_weight_array(weights_np, name=name)
    return weights_np


def normalize_weights(weights: np.ndarray) -> np.ndarray:
    """Normalize weights so they sum to the number of elements.

    Applies ``weights * (n / sum(weights))`` so the sum of the returned array
    equals ``len(weights)``. Preserves scale when weights are used as
    multiplicative factors on scores or loss values.

    Parameters
    ----------
    weights : numpy.ndarray
        Weight array. Must not sum to zero.

    Returns
    -------
    numpy.ndarray
        Normalized weight array with ``sum == len(weights)``.

    Raises
    ------
    ValueError
        If ``weights`` sums to zero.

    """
    total = weights.sum()
    if total == 0:
        raise ValueError("Cannot normalize weights: sum is zero")
    return weights * (len(weights) / total)


def validate_weight_array(weights: np.ndarray, name: str = "weights") -> None:
    """Validate a resolved weight array for NaN, negatives, infinities, and all-zero.

    Parameters
    ----------
    weights : numpy.ndarray
        Weight array to validate.
    name : str, default="weights"
        Descriptive name used in error messages.

    Raises
    ------
    ValueError
        If any element is NaN, negative, infinite, or if all elements are zero.

    """
    nan_mask = np.isnan(weights)
    if np.any(nan_mask):
        nan_indices = np.where(nan_mask)[0].tolist()
        raise ValueError(
            f"{name} contains NaN at {len(nan_indices)} position(s) "
            f"(indices {nan_indices[:10]}{'...' if len(nan_indices) > 10 else ''})"
        )
    if np.any(weights < 0):
        neg_indices = np.where(weights < 0)[0].tolist()
        raise ValueError(
            f"{name} contains negative values at indices {neg_indices[:10]}{'...' if len(neg_indices) > 10 else ''}"
        )
    if np.any(np.isinf(weights)):
        inf_indices = np.where(np.isinf(weights))[0].tolist()
        raise ValueError(
            f"{name} contains infinite values at indices {inf_indices[:10]}{'...' if len(inf_indices) > 10 else ''}"
        )
    if weights.sum() == 0:
        raise ValueError(f"All weights are zero for {name}")


def combine_weight_vectors(*arrays: np.ndarray | None, n: int) -> np.ndarray | None:
    """Combine weight vectors multiplicatively and normalize.

    Filters out ``None`` inputs, multiplies the remaining arrays element-wise,
    then normalizes so the result sums to ``n``.

    Parameters
    ----------
    *arrays : numpy.ndarray or None
        Weight arrays to combine. ``None`` entries are ignored.
    n : int
        Target sum after normalization (typically the number of rows).

    Returns
    -------
    numpy.ndarray or None
        Combined, normalized weight array, or ``None`` if all inputs are
        ``None``.

    Raises
    ------
    ValueError
        If the combined product sums to zero.

    """
    valid = [a for a in arrays if a is not None]
    if not valid:
        return None
    combined = valid[0].copy()
    for a in valid[1:]:
        combined = combined * a
    if combined.sum() == 0:
        raise ValueError("Combined weights sum to zero: all weighted elements have zero weight")
    return combined * (n / combined.sum())
