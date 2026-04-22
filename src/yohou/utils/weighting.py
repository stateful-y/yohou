"""Time-based weighting utilities for scorers and forecasters.

This module provides utilities for generating time-based weights that can be
used with scorers (evaluation weighting) and reduction forecasters (training
sample weighting). All functions return callables that accept time series and
return weight series.
"""

from __future__ import annotations

import inspect
import numbers
from collections.abc import Callable
from datetime import timedelta

import numpy as np
import polars as pl

__all__ = [
    "combine_weight_vectors",
    "compose_weights",
    "exponential_decay_weight",
    "linear_decay_weight",
    "normalize_weights",
    "resolve_dict_weights",
    "resolve_weight_to_array",
    "seasonal_emphasis_weight",
    "validate_callable_signature",
    "validate_weight_array",
]


def exponential_decay_weight(
    half_life: int | float | timedelta,
) -> Callable[[pl.Series], pl.Series]:
    r"""Generate exponential decay weights giving more weight to recent times.

    Creates a callable that computes weights using exponential decay:

    $$w(t) = \exp\left(-\ln(2) \cdot \frac{d(t)}{\text{half\_life}}\right)$$

    where $d(t)$ is the distance from the most recent time point.

    Parameters
    ----------
    half_life : int, float, or timedelta
        Time period over which weight decays to half. If int/float, interpreted
        as number of time steps. If timedelta, interpreted as time duration.

    Returns
    -------
    Callable[[pl.Series], pl.Series]
        Function accepting time series (datetime) and returning weight series
        (float64). Most recent time has weight 1.0, older times decay exponentially.

    References
    ----------
    .. [1] Hyndman, R.J., & Athanasopoulos, G. (2021). "Forecasting:
       principles and practice," 3rd edition, OTexts: Melbourne, Australia.
       OTexts.com/fpp3. Chapter 8.1.

    See Also
    --------
    `linear_decay_weight` : Linear decay weights for recent times.
    `seasonal_emphasis_weight` : Weights emphasizing seasonal positions.
    `compose_weights` : Compose multiple weight functions by multiplication.
    `validate_callable_signature` : Validate callable signature for time weighting.
    `BaseReductionForecaster` : Reduction forecaster supporting time_weight.

    Examples
    --------
    >>> import polars as pl
    >>> from datetime import datetime, timedelta
    >>> times = pl.Series(
    ...     "time",
    ...     [
    ...         datetime(2024, 1, 1),
    ...         datetime(2024, 1, 2),
    ...         datetime(2024, 1, 3),
    ...     ],
    ... )
    >>> weight_fn = exponential_decay_weight(half_life=1)
    >>> weights = weight_fn(times)
    >>> weights  # doctest: +NORMALIZE_WHITESPACE
    shape: (3,)
    Series: 'weight' [f64]
    [
        0.25
        0.5
        1.0
    ]

    """
    # Convert to timedelta if numeric (assume days for compatibility)
    half_life_td = half_life
    if isinstance(half_life, numbers.Number) and not isinstance(half_life, timedelta):
        # Type narrowing: exclude timedelta from union before converting
        half_life_td = timedelta(days=float(half_life))

    def _weight_fn(time: pl.Series) -> pl.Series:
        """Apply exponential decay weighting to time series.

        Parameters
        ----------
        time : pl.Series
            Time series (datetime type).

        Returns
        -------
        pl.Series
            Weight series with exponential decay from oldest to most recent.

        """
        # Compute distance from most recent time
        max_time = time.max()
        distances = (max_time - time).dt.total_seconds()

        # Convert half_life to seconds
        assert isinstance(half_life_td, timedelta)
        half_life_seconds = half_life_td.total_seconds()

        # Exponential decay: exp(-ln(2) * distance / half_life)
        weights = np.exp(-np.log(2) * distances / half_life_seconds)
        return pl.Series(weights, dtype=pl.Float64).alias("weight")

    return _weight_fn


def linear_decay_weight(
    max_steps: int | None = None,
) -> Callable[[pl.Series], pl.Series]:
    r"""Generate linear decay weights giving more weight to recent times.

    Creates a callable that computes weights using linear decay:

    $$w(t) = \frac{\text{rank}(t)}{n - 1}$$

    where $\text{rank}(t) = 0$ for the oldest observation and $n - 1$
    for the most recent. When ``max_steps`` is set, observations older
    than ``max_steps`` receive weight 0.

    Parameters
    ----------
    max_steps : int or None, default=None
        Maximum number of steps to decay over. If None, decays linearly across
        entire time range. If specified, times older than max_steps from the
        most recent time receive weight 0.

    Returns
    -------
    Callable[[pl.Series], pl.Series]
        Function accepting time series (datetime) and returning weight series
        (float64). Most recent time has weight 1.0, weights decrease linearly.

    See Also
    --------
    `exponential_decay_weight` : Exponential decay weights for recent times.
    `seasonal_emphasis_weight` : Weights emphasizing seasonal positions.
    `compose_weights` : Compose multiple weight functions by multiplication.
    `validate_callable_signature` : Validate callable signature for time weighting.
    `BaseReductionForecaster` : Reduction forecaster supporting time_weight.

    Examples
    --------
    >>> import polars as pl
    >>> from datetime import datetime
    >>> times = pl.Series(
    ...     "time",
    ...     [
    ...         datetime(2024, 1, 1),
    ...         datetime(2024, 1, 2),
    ...         datetime(2024, 1, 3),
    ...         datetime(2024, 1, 4),
    ...     ],
    ... )
    >>> weight_fn = linear_decay_weight(max_steps=None)
    >>> weights = weight_fn(times)
    >>> weights  # doctest: +NORMALIZE_WHITESPACE
    shape: (4,)
    Series: 'weight' [f64]
    [
        0.0
        0.333333
        0.666667
        1.0
    ]

    """

    def _weight_fn(time: pl.Series) -> pl.Series:
        """Apply linear decay weighting to time series.

        Parameters
        ----------
        time : pl.Series
            Time series (datetime type).

        Returns
        -------
        pl.Series
            Weight series with linear decay from oldest to most recent.

        """
        n = len(time)
        if n == 1:
            return pl.Series([1.0], dtype=pl.Float64).alias("weight")

        # Compute rank (0 = oldest, n-1 = newest) and convert to numpy
        ranks = (time.rank(method="ordinal") - 1).to_numpy()

        if max_steps is None:
            # Linear decay over entire range
            weights = ranks / (n - 1)
        else:
            # Linear decay over max_steps window
            # Older times (before the window) get weight 0
            # Recent max_steps times get linear increase from 0 to 1
            cutoff = n - max_steps  # Times with rank < cutoff get weight 0
            weights = np.where(ranks < cutoff, 0.0, (ranks - cutoff) / (max_steps - 1) if max_steps > 1 else 1.0)
            weights = np.minimum(1.0, weights)

        return pl.Series(weights, dtype=pl.Float64).alias("weight")

    return _weight_fn


def seasonal_emphasis_weight(
    seasonality: int | list[int],
    emphasis: float = 2.0,
) -> Callable[[pl.Series], pl.Series]:
    r"""Generate weights emphasizing specific seasonal positions.

    Creates a callable that gives higher weights to times matching the most
    recent seasonal position (e.g., same day of week, same day of month):

    $$w(t) = \begin{cases} \text{emphasis} & \text{if } t \bmod s \equiv t_n \bmod s \\ 1 & \text{otherwise} \end{cases}$$

    where $s$ is the seasonal period and $t_n$ is the most recent time.

    Parameters
    ----------
    seasonality : int or list of int
        Seasonal period(s) (e.g., 7 for weekly, 12 for monthly, or [7, 30] for
        both weekly and monthly patterns). Determines which times are in-phase
        with the most recent observation. If multiple seasonalities provided,
        times matching ANY pattern receive emphasis.
    emphasis : float, default=2.0
        Weight multiplier for in-phase times. In-phase times receive weight
        `emphasis`, out-of-phase times receive weight 1.0. Values > 1.0
        emphasize seasonality, values < 1.0 de-emphasize.

    Returns
    -------
    Callable[[pl.Series], pl.Series]
        Function accepting time series (datetime) and returning weight series
        (float64). Times matching the most recent seasonal position(s) receive
        higher weights.

    See Also
    --------
    `exponential_decay_weight` : Exponential decay weights for recent times.
    `linear_decay_weight` : Linear decay weights for recent times.
    `compose_weights` : Compose multiple weight functions by multiplication.
    `validate_callable_signature` : Validate callable signature for time weighting.
    `BaseReductionForecaster` : Reduction forecaster supporting time_weight.

    Examples
    --------
    >>> import polars as pl
    >>> from datetime import datetime
    >>> times = pl.Series(
    ...     "time",
    ...     [
    ...         datetime(2024, 1, 1),  # Monday
    ...         datetime(2024, 1, 2),  # Tuesday
    ...         datetime(2024, 1, 8),  # Monday (same weekday as most recent)
    ...         datetime(2024, 1, 9),  # Tuesday (most recent)
    ...     ],
    ... )
    >>> weight_fn = seasonal_emphasis_weight(seasonality=7, emphasis=2.0)
    >>> weights = weight_fn(times)
    >>> weights  # doctest: +NORMALIZE_WHITESPACE
    shape: (4,)
    Series: 'weight' [f64]
    [
        1.0
        1.0
        1.0
        2.0
    ]

    Multiple seasonalities (weekly + monthly):

    >>> times = pl.Series("time", [datetime(2024, 1, i) for i in range(1, 32)])
    >>> weight_fn = seasonal_emphasis_weight(seasonality=[7, 30], emphasis=2.0)
    >>> weights = weight_fn(times)
    >>> # Times matching day-of-week OR day-of-month get emphasized

    """
    # Normalize to list
    seasonalities = [seasonality] if isinstance(seasonality, int) else seasonality

    def _weight_fn(time: pl.Series) -> pl.Series:
        """Apply seasonal emphasis weighting to time series.

        Parameters
        ----------
        time : pl.Series
            Time series (datetime type).

        Returns
        -------
        pl.Series
            Weight series with emphasis on times matching seasonal pattern(s).

        """
        n = len(time)
        if n == 1:
            return pl.Series([1.0], dtype=pl.Float64).alias("weight")

        # Compute rank to determine position in sequence
        ranks = time.rank(method="ordinal") - 1

        # Check if any seasonality matches (OR logic)
        in_phase = np.zeros(n, dtype=bool)
        for s in seasonalities:
            # Most recent position modulo this seasonality
            most_recent_phase = int(ranks[-1]) % s
            # Check which positions match the most recent phase
            phases = ranks.to_numpy() % s
            in_phase = in_phase | (phases == most_recent_phase)

        # Apply emphasis to in-phase times
        weights = np.where(in_phase, emphasis, 1.0)
        return pl.Series(weights, dtype=pl.Float64).alias("weight")

    return _weight_fn


def compose_weights(
    *weight_fns: Callable[[pl.Series], pl.Series],
) -> Callable[[pl.Series], pl.Series]:
    """Compose multiple weight functions by multiplication.

    Creates a callable that applies multiple weight functions sequentially
    and multiplies the results. Useful for combining different weighting
    strategies (e.g., exponential decay + seasonal emphasis).

    Parameters
    ----------
    *weight_fns : Callable[[pl.Series], pl.Series]
        One or more weight functions to compose. Each must accept a time
        series and return a weight series.

    Returns
    -------
    Callable[[pl.Series], pl.Series]
        Composed function that multiplies weights from all input functions.

    See Also
    --------
    `exponential_decay_weight` : Exponential decay weights for recent times.
    `linear_decay_weight` : Linear decay weights for recent times.
    `seasonal_emphasis_weight` : Weights emphasizing seasonal positions.
    `validate_callable_signature` : Validate callable signature for time weighting.
    `BaseReductionForecaster` : Reduction forecaster supporting time_weight.

    Examples
    --------
    >>> import polars as pl
    >>> from datetime import datetime
    >>> times = pl.Series(
    ...     "time",
    ...     [
    ...         datetime(2024, 1, 1),
    ...         datetime(2024, 1, 2),
    ...         datetime(2024, 1, 3),
    ...     ],
    ... )
    >>> # Combine exponential decay with seasonal emphasis
    >>> weight_fn = compose_weights(
    ...     exponential_decay_weight(half_life=2), seasonal_emphasis_weight(seasonality=2, emphasis=1.5)
    ... )
    >>> weights = weight_fn(times)

    """
    if not weight_fns:
        raise ValueError("At least one weight function must be provided")

    def _weight_fn(time: pl.Series) -> pl.Series:
        """Apply composed weighting to time series.

        Parameters
        ----------
        time : pl.Series
            Time series (datetime type).

        Returns
        -------
        pl.Series
            Weight series from multiplying all component weight functions.

        """
        # Start with uniform weights
        result = pl.Series(np.ones(len(time)), dtype=pl.Float64)

        # Multiply by each weight function
        for fn in weight_fns:
            result = result * fn(time)

        return result.alias("weight")

    return _weight_fn


def validate_callable_signature(
    time_weight: Callable,
) -> int:
    """Validate that callable has valid signature for time weighting.

    Checks that the callable accepts either 1 parameter (global weighting)
    or 2 parameters (panel-aware weighting). Raises ValueError for invalid
    signatures.

    Parameters
    ----------
    time_weight : Callable
        Callable to validate. Must have signature (time: pl.Series) -> pl.Series
        or (time: pl.Series, group_name: str) -> pl.Series.

    Returns
    -------
    int
        Number of parameters (1 or 2) for valid signatures.

    Raises
    ------
    ValueError
        If callable signature is invalid (not 1 or 2 parameters).

    See Also
    --------
    `exponential_decay_weight` : Exponential decay weights for recent times.
    `linear_decay_weight` : Linear decay weights for recent times.
    `seasonal_emphasis_weight` : Weights emphasizing seasonal positions.
    `compose_weights` : Compose multiple weight functions by multiplication.
    `BaseReductionForecaster` : Reduction forecaster supporting time_weight.

    Examples
    --------
    >>> def global_weight(time):
    ...     return time * 0 + 1
    >>> validate_callable_signature(global_weight)
    1

    >>> def panel_weight(time, group_name):
    ...     return time * 0 + 1
    >>> validate_callable_signature(panel_weight)
    2

    """
    sig = inspect.signature(time_weight)
    n_params = len(sig.parameters)

    if n_params not in (1, 2):
        raise ValueError(
            f"time_weight callable must accept either 1 parameter "
            f"(time: pl.Series) or 2 parameters (time: pl.Series, group_name: str), "
            f"but got {n_params} parameters"
        )

    return n_params


def normalize_weights(weights: np.ndarray) -> np.ndarray:
    """Normalize weights so they sum to the number of elements.

    Applies the convention ``weights * (n / sum(weights))`` so the sum of
    the returned array equals ``len(weights)``.  This preserves scale when
    weights are used as multiplicative factors on scores or loss values.

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
        Descriptive name used in error messages (e.g., ``"step_weight"``,
        ``"vintage_weight"``).

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


def resolve_dict_weights(
    weight_dict: dict,
    keys: np.ndarray | list,
    default: float = 1.0,
) -> np.ndarray:
    """Map a ``{key: weight}`` dict to an aligned numpy array.

    For each element in ``keys``, looks up the corresponding weight in
    ``weight_dict``.  Missing keys receive ``default``.  If the dict
    contains the wildcard key ``"*"``, its value overrides ``default``
    (e.g., ``{"*": 0.0, 1: 2.0}`` gives step 1 weight 2.0 and all
    others weight 0.0).  The ``"*"`` key is consumed and never looked up
    in ``keys``.

    Parameters
    ----------
    weight_dict : dict
        Mapping from key values to weights.
    keys : numpy.ndarray or list
        Key values to look up, one per element.
    default : float, default=1.0
        Weight for keys not present in ``weight_dict``.

    Returns
    -------
    numpy.ndarray
        Weight array aligned to ``keys``.

    """
    effective_default = weight_dict.get("*", default)
    lookup = {k: v for k, v in weight_dict.items() if k != "*"}
    return np.array([lookup.get(k, effective_default) for k in keys], dtype=np.float64)


def combine_weight_vectors(*arrays: np.ndarray | None, n: int) -> np.ndarray | None:
    """Combine weight vectors multiplicatively and normalize.

    Filters out ``None`` inputs, multiplies the remaining arrays
    element-wise, then normalizes so the result sums to ``n``.

    Parameters
    ----------
    *arrays : numpy.ndarray or None
        Weight arrays to combine.  ``None`` entries are ignored.
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


def resolve_weight_to_array(
    weight: Callable | pl.DataFrame | dict,
    key_series: pl.Series,
    join_column: str,
    group_name: str | None = None,
) -> np.ndarray:
    """Resolve a weight specification to a raw numpy array.

    Accepts three formats (callable, DataFrame, or dict) and returns an
    unnormalized weight array aligned to ``key_series``.  Validation is
    performed via `validate_weight_array` after resolution.

    Parameters
    ----------
    weight : callable, pl.DataFrame, or dict
        Weight specification in one of the supported formats.
    key_series : polars.Series
        Data series whose values are used for alignment (e.g., time
        values, forecasting steps, vintage times).
    join_column : str
        Column name used for DataFrame joins (``"time"``,
        ``"forecasting_step"``, or ``"vintage_time"``).
    group_name : str or None, default=None
        Panel group name for 2-parameter callables and group-specific
        DataFrame columns.  ``None`` for global (non-panel) data.

    Returns
    -------
    numpy.ndarray
        Raw (unnormalized) weight array of same length as ``key_series``.

    Raises
    ------
    ValueError
        If ``weight`` is not callable, DataFrame, or dict, or if
        the resolved weights fail validation.

    """
    if isinstance(weight, dict):
        weights_np = resolve_dict_weights(weight, key_series.to_list())
    elif callable(weight):
        n_params = validate_callable_signature(weight)
        weights_series = weight(key_series) if n_params == 1 else weight(key_series, group_name)  # type: ignore[call-arg]  # ty: ignore[call-top-callable]

        if not isinstance(weights_series, pl.Series):
            raise ValueError(f"Weight callable must return pl.Series, got {type(weights_series).__name__}")
        if len(weights_series) != len(key_series):
            raise ValueError(f"Weight callable returned {len(weights_series)} weights, expected {len(key_series)} rows")
        weights_np = weights_series.to_numpy().astype(np.float64)
    elif isinstance(weight, pl.DataFrame):
        key_df = pl.DataFrame({join_column: key_series})
        joined = key_df.join(weight, on=join_column, how="left")

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

        # Check for NaN from unmatched keys (left join produces null for missing keys)
        nan_mask = np.isnan(weights_np)
        if nan_mask.any():
            missing_keys = key_series.filter(pl.Series(nan_mask)).unique().to_list()
            raise ValueError(f"Weight DataFrame has no values for {join_column}s: {missing_keys}")
    else:
        raise ValueError(f"Weight must be callable, pl.DataFrame, dict, or None, got {type(weight).__name__}")

    # Validate
    try:
        validate_weight_array(weights_np, name=f"{join_column} weight")
    except ValueError as exc:
        if "All weights are zero" in str(exc):
            # Re-raise with contextual info about which keys were requested
            if isinstance(weight, dict):
                requested = {k for k in weight if k != "*"}
                available = set(key_series.unique().to_list())
                raise ValueError(
                    f"All weights are zero for {join_column} weight. "
                    f"Requested keys {requested} vs available keys "
                    f"{available}"
                ) from exc
            raise
        raise

    return weights_np
