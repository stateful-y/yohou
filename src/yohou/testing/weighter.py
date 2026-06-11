"""Check functions for yohou weighters.

Systematic validation functions for the ``BaseWeighter`` estimator family
(time-axis weighters). All check functions raise AssertionError on failure.
"""

try:
    import polars as pl
except ImportError as e:
    raise ImportError("polars is required for yohou.testing module. Install with: uv sync --group tests") from e

import numpy as np

from yohou.weighting.weighters import BaseWeighter, _resolve_weighter_to_array

__all__ = [
    "check_weighter_compute_weights_alignment",
    "check_weighter_default_constructible",
    "check_weighter_fit_noop_returns_self",
    "check_weighter_resolved_array_validation",
    "check_weighter_tags_accessible_before_fit",
    "check_weighter_tags_static_after_fit",
]


def check_weighter_compute_weights_alignment(weighter, key: pl.Series) -> None:
    """Check ``compute_weights`` returns one weight per key element.

    Parameters
    ----------
    weighter : BaseWeighter
        Weighter instance.
    key : pl.Series
        Key series to weight.

    Raises
    ------
    AssertionError
        If the output is not a ``pl.Series`` aligned 1:1 with ``key``.

    """
    name = type(weighter).__name__
    weights = weighter.compute_weights(key)
    assert isinstance(weights, pl.Series), f"{name}.compute_weights must return pl.Series, got {type(weights)}"
    assert len(weights) == len(key), (
        f"{name}.compute_weights returned {len(weights)} weights, expected {len(key)} (one per key element)"
    )


def check_weighter_fit_noop_returns_self(weighter) -> None:
    """Check ``fit`` is a no-op returning self that leaves params unchanged.

    Parameters
    ----------
    weighter : BaseWeighter
        Weighter instance.

    Raises
    ------
    AssertionError
        If ``fit`` does not return ``self`` or alters constructor parameters.

    """
    name = type(weighter).__name__
    before = weighter.get_params(deep=False)
    returned = weighter.fit(None)
    assert returned is weighter, f"{name}.fit() must return self"
    after = weighter.get_params(deep=False)
    assert before.keys() == after.keys(), f"{name}.fit() changed parameter keys"


def check_weighter_resolved_array_validation(weighter, key: pl.Series) -> None:
    """Check resolved-array validation rejects NaN/negative/inf/all-zero.

    Drives ``_resolve_weighter_to_array`` (and ``_validate_weight_array``) with
    weighters whose resolution yields each invalid case and asserts a
    ``ValueError`` is raised.

    Parameters
    ----------
    weighter : BaseWeighter
        Weighter instance (unused; the check is family-level).
    key : pl.Series
        Key series with which to resolve the stub weighters.

    Raises
    ------
    AssertionError
        If any invalid resolved array is accepted.

    """
    n = len(key)

    class _BadWeighter(BaseWeighter):
        def __init__(self, values: np.ndarray) -> None:
            self._values = values

        def compute_weights(self, key: pl.Series, group_name: str | None = None) -> pl.Series:
            return pl.Series(self._values, dtype=pl.Float64).alias("weight")

    cases = {
        "NaN": np.array([np.nan] + [1.0] * (n - 1)),
        "negative": np.array([-1.0] + [1.0] * (n - 1)),
        "infinite": np.array([np.inf] + [1.0] * (n - 1)),
        "all-zero": np.zeros(n),
    }
    for label, values in cases.items():
        try:
            _resolve_weighter_to_array(_BadWeighter(values), key)
        except ValueError:
            continue
        raise AssertionError(f"resolved-array validation accepted {label} weights; expected ValueError")


def check_weighter_default_constructible(weighter) -> None:
    """Check the weighter class is constructible with no arguments.

    Parameters
    ----------
    weighter : BaseWeighter
        Weighter instance (used for its type).

    Raises
    ------
    AssertionError
        If ``type(weighter)()`` raises.

    """
    cls = type(weighter)
    try:
        cls()
    except Exception as e:  # noqa: BLE001
        raise AssertionError(f"{cls.__name__} is not default-constructible: {type(e).__name__}: {e}") from e


def check_weighter_tags_accessible_before_fit(weighter) -> None:
    """Check ``__sklearn_tags__()`` is callable on an unfitted weighter.

    Parameters
    ----------
    weighter : BaseWeighter
        Weighter instance.

    Raises
    ------
    AssertionError
        If tags are not accessible.

    """
    name = type(weighter).__name__
    assert hasattr(weighter, "__sklearn_tags__"), f"{name} must have __sklearn_tags__()"
    try:
        weighter.__sklearn_tags__()
    except Exception as e:  # noqa: BLE001
        raise AssertionError(f"{name}.__sklearn_tags__() raised {type(e).__name__}: {e}") from e


def check_weighter_tags_static_after_fit(weighter) -> None:
    """Check tags are unchanged by the no-op ``fit``.

    Parameters
    ----------
    weighter : BaseWeighter
        Unfitted weighter instance.

    Raises
    ------
    AssertionError
        If tags change after ``fit``.

    """
    name = type(weighter).__name__
    before = weighter.__sklearn_tags__()
    weighter.fit(None)
    after = weighter.__sklearn_tags__()
    assert before == after, f"{name} tags changed after fit() - tags should be static"
