"""Reducer and bare-list checks for ``{multiply, mean}`` compositors.

These checks cover the part of the composite contract that raises only at
compute/fit time (not at ``get_params``/``clone``) and therefore needs
family-specific data: bare-list rejection and ``combination``/``weights``
validation. They are consumed only by the weighter and similarity family
harnesses (``CompositeWeighter``, ``CompositeSimilarity``), which pass an
``operate`` callable that triggers the compositor's validation.

All check functions raise AssertionError on failure.
"""

from collections.abc import Callable, Generator

__all__ = [
    "check_composite_combination_validation",
    "check_composite_rejects_bare_list",
    "check_composite_weights_length_validation",
    "_yield_composite_reducer_checks",
]


def check_composite_rejects_bare_list(compositor, descriptor: dict, operate: Callable) -> None:
    """Check a bare ``[estimator, ...]`` list (no names) is rejected.

    Parameters
    ----------
    compositor : BaseEstimator
        A ``{multiply, mean}`` compositor instance (for its type).
    descriptor : dict
        Carries ``"attr"`` (composition-attribute name) and ``"components"``
        (valid named-tuple component list).
    operate : callable
        Invoked with a freshly-built compositor to trigger validation.

    Raises
    ------
    AssertionError
        If the bare list is accepted instead of raising.

    """
    name = type(compositor).__name__
    attr = descriptor["attr"]
    bare = [entry[1] for entry in descriptor["components"]]
    bad = type(compositor)(**{attr: bare})
    try:
        operate(bad)
    except (ValueError, TypeError):
        return
    raise AssertionError(f"{name}: a bare {attr} list (no names) was accepted; expected ValueError/TypeError")


def check_composite_combination_validation(compositor, descriptor: dict, operate: Callable) -> None:
    """Check an unknown ``combination`` raises.

    Parameters
    ----------
    compositor : BaseEstimator
        A ``{multiply, mean}`` compositor instance (for its type).
    descriptor : dict
        Carries ``"attr"`` and ``"components"``.
    operate : callable
        Invoked with a freshly-built compositor to trigger validation.

    Raises
    ------
    AssertionError
        If the invalid combination is accepted.

    """
    name = type(compositor).__name__
    attr = descriptor["attr"]
    bad = type(compositor)(**{attr: descriptor["components"], "combination": "__invalid__"})
    try:
        operate(bad)
    except ValueError:
        return
    raise AssertionError(f"{name}: unknown combination was accepted; expected ValueError")


def check_composite_weights_length_validation(compositor, descriptor: dict, operate: Callable) -> None:
    """Check a ``weights`` list of the wrong length raises.

    Parameters
    ----------
    compositor : BaseEstimator
        A ``{multiply, mean}`` compositor instance (for its type).
    descriptor : dict
        Carries ``"attr"`` and ``"components"``.
    operate : callable
        Invoked with a freshly-built compositor to trigger validation.

    Raises
    ------
    AssertionError
        If the mismatched-length weights are accepted.

    """
    name = type(compositor).__name__
    attr = descriptor["attr"]
    components = descriptor["components"]
    bad = type(compositor)(**{attr: components, "weights": [1.0] * (len(components) + 1)})
    try:
        operate(bad)
    except ValueError:
        return
    raise AssertionError(f"{name}: weights of wrong length were accepted; expected ValueError")


def _yield_composite_reducer_checks(
    compositor,
    descriptor: dict,
    operate: Callable,
) -> Generator[tuple[str, Callable, dict], None, None]:
    """Yield the reducer/bare-list checks for a ``{multiply, mean}`` compositor.

    Parameters
    ----------
    compositor : BaseEstimator
        A ``CompositeWeighter`` or ``CompositeSimilarity`` instance.
    descriptor : dict
        Per-compositor descriptor with ``"attr"`` and ``"components"``.
    operate : callable
        Triggers the compositor's validation at compute/fit time
        (e.g. ``lambda c: c.compute_weights(key)`` or ``lambda c: c.fit(y, y_pred)``).

    Yields
    ------
    tuple of (str, callable, dict)
        ``(check_name, check_func, check_kwargs)`` consumable by ``run_checks``.

    """
    kwargs = {"descriptor": descriptor, "operate": operate}
    yield "check_composite_rejects_bare_list", check_composite_rejects_bare_list, kwargs
    yield "check_composite_combination_validation", check_composite_combination_validation, kwargs
    yield "check_composite_weights_length_validation", check_composite_weights_length_validation, kwargs
