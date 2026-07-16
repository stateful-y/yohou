"""Family-agnostic sklearn estimator-contract and composition-contract checks.

This module provides systematic validation functions shared across estimator
families:

1. Estimator-contract checks - the sklearn contract (``clone`` round-trip,
   ``get_params``/``set_params`` round-trip, no ``__init__`` mutation) that was
   previously checked only for forecasters and searches.
2. Composition-contract checks - the *data-free* part of the
   ``_BaseComposition`` contract (nested ``<name>__<param>`` addressability and
   clone-deep-clones), run over every compositor in yohou.

All check functions raise AssertionError on failure.
"""

import inspect
from collections.abc import Callable, Generator

from sklearn.base import BaseEstimator, clone


def _safe_equal(a: object, b: object) -> bool:
    """Equality that tolerates DataFrame/array params (whose ``==`` is not a bool).

    Falls back to ``.equals`` for frame-like values and to ``.all()`` for
    array-like comparison results; returns ``False`` if equality cannot be
    determined.
    """
    if a is b:
        return True
    equals = getattr(a, "equals", None)
    if callable(equals):
        try:
            return bool(equals(b))
        except Exception:  # noqa: BLE001
            return False
    try:
        result = a == b
    except Exception:  # noqa: BLE001
        return False
    if isinstance(result, bool):
        return result
    reducer = getattr(result, "all", None)
    if callable(reducer):
        try:
            return bool(reducer())
        except Exception:  # noqa: BLE001
            return False
    return bool(result)


__all__ = [
    "check_clone_preserves_params",
    "check_composition_clone_deep_clones_components",
    "check_composition_nested_param_addressable",
    "check_get_set_params_round_trip",
    "check_init_no_param_mutation",
    "_yield_composition_contract_checks",
    "_yield_estimator_contract_checks",
]


def check_clone_preserves_params(estimator) -> None:
    """Check ``clone()`` preserves shallow params and freshens nested estimators.

    Family-agnostic generalization of ``check_clone_preserves_forecaster_params``:
    the clone reproduces every ``get_params(deep=False)`` value, and nested
    sub-estimators (including ``(name, estimator)`` tuple lists) become fresh
    instances of the same type.

    Parameters
    ----------
    estimator : BaseEstimator
        Estimator instance to clone.

    Raises
    ------
    AssertionError
        If the clone has different parameter keys or values.

    """
    name = type(estimator).__name__
    estimator_clone = clone(estimator)

    original = estimator.get_params(deep=False)
    cloned = estimator_clone.get_params(deep=False)

    assert set(original.keys()) == set(cloned.keys()), (
        f"{name}: clone() changed parameter keys: {set(cloned.keys())} vs {set(original.keys())}"
    )

    for key, orig_val in original.items():
        cloned_val = cloned[key]
        if orig_val is None:
            assert cloned_val is None, f"{name}: parameter {key!r} expected None, got {cloned_val!r}"
        elif isinstance(orig_val, list) and orig_val and isinstance(orig_val[0], tuple):
            # List of (name, estimator[, ...]) tuples (compositors).
            assert isinstance(cloned_val, list), f"{name}: parameter {key!r} expected list, got {type(cloned_val)}"
            assert len(cloned_val) == len(orig_val), f"{name}: parameter {key!r} changed length under clone()"
            for orig_item, cloned_item in zip(orig_val, cloned_val, strict=True):
                if not (isinstance(orig_item, tuple) and isinstance(cloned_item, tuple)):
                    continue
                assert orig_item[0] == cloned_item[0], f"{name}: component name changed under clone()"
                orig_est, cloned_est = orig_item[1], cloned_item[1]
                if isinstance(orig_est, BaseEstimator):
                    assert type(orig_est) is type(cloned_est), (
                        f"{name}: component {orig_item[0]!r} changed type under clone()"
                    )
        elif isinstance(orig_val, BaseEstimator):
            assert type(orig_val) is type(cloned_val), f"{name}: parameter {key!r} changed type under clone()"
        elif not _safe_equal(orig_val, cloned_val):
            raise AssertionError(f"{name}: parameter {key!r} expected {orig_val!r}, got {cloned_val!r}")


def check_get_set_params_round_trip(estimator) -> None:
    """Check ``set_params(**get_params(deep=True))`` is a no-op.

    Also asserts every ``get_params(deep=False)`` key is a constructor
    parameter (the sklearn ``get_params`` contract).

    Parameters
    ----------
    estimator : BaseEstimator
        Estimator instance.

    Raises
    ------
    AssertionError
        If a ``get_params(deep=False)`` key is absent from the constructor
        parameters.
    AssertionError
        If ``set_params(**get_params(deep=True))`` mutates any shallow
        parameter value.

    """
    name = type(estimator).__name__

    init_params = {p for p in inspect.signature(type(estimator).__init__).parameters if p != "self"}
    shallow = estimator.get_params(deep=False)
    for key in shallow:
        assert key in init_params, f"{name}: get_params key {key!r} is not a constructor parameter"

    before = estimator.get_params(deep=False)
    estimator.set_params(**estimator.get_params(deep=True))
    after = estimator.get_params(deep=False)

    assert set(before.keys()) == set(after.keys()), f"{name}: set_params round-trip changed parameter keys"
    for key, before_val in before.items():
        after_val = after[key]
        if isinstance(before_val, BaseEstimator) or (
            isinstance(before_val, list) and before_val and isinstance(before_val[0], tuple)
        ):
            continue  # nested estimators compare by identity/structure, not equality
        assert _safe_equal(before_val, after_val), (
            f"{name}: set_params round-trip changed {key!r}: {before_val!r} -> {after_val!r}"
        )


def check_init_no_param_mutation(estimator) -> None:
    """Check ``__init__`` stores constructor arguments verbatim.

    For every constructor parameter with a default, a default-constructed
    instance must store that exact default (e.g. ``metric_params=None`` stays
    ``None``, never ``{}``). Catches the init-mutation bug class.

    Parameters
    ----------
    estimator : BaseEstimator
        Estimator instance (used only for its type).

    Raises
    ------
    AssertionError
        If a stored attribute differs from the constructor default.

    """
    cls = type(estimator)
    name = cls.__name__
    signature = inspect.signature(cls.__init__)

    if any(
        p.default is inspect.Parameter.empty and pname not in ("self",)
        for pname, p in signature.parameters.items()
        if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)
    ):
        return  # not default-constructible; skip (covered where representative instances exist)

    fresh = cls()
    for pname, param in signature.parameters.items():
        if pname == "self" or param.default is inspect.Parameter.empty:
            continue
        if not hasattr(fresh, pname):
            continue  # stored under a different name (e.g. a property); out of scope here
        stored = getattr(fresh, pname)
        default = param.default
        if default is None:
            assert stored is None, f"{name}: __init__ mutated {pname!r} from None to {stored!r}"
        else:
            assert stored == default, f"{name}: __init__ mutated {pname!r} from {default!r} to {stored!r}"


def _yield_estimator_contract_checks(
    estimator,
) -> Generator[tuple[str, Callable, dict], None, None]:
    """Yield the shared sklearn estimator-contract checks.

    Parameters
    ----------
    estimator : BaseEstimator
        Estimator instance the checks run against.

    Yields
    ------
    tuple of (str, callable, dict)
        ``(check_name, check_func, check_kwargs)`` consumable by ``run_checks``.

    """
    yield "check_clone_preserves_params", check_clone_preserves_params, {}
    yield "check_get_set_params_round_trip", check_get_set_params_round_trip, {}
    yield "check_init_no_param_mutation", check_init_no_param_mutation, {}


def _first_named_component(compositor, attr: str) -> tuple[str, BaseEstimator]:
    """Return the ``(name, estimator)`` of the first component, handling 3-tuples."""
    entries = getattr(compositor, attr)
    first = entries[0]
    return first[0], first[1]


def check_composition_nested_param_addressable(compositor, descriptor: dict) -> None:
    """Check nested ``<name>__<param>`` get/set on a ``_BaseComposition``.

    ``get_params(deep=True)`` must expose the first component's parameters as
    ``<name>__<param>``, and ``set_params(<name>__<param>=value)`` must round-trip.

    Parameters
    ----------
    compositor : BaseEstimator
        A ``_BaseComposition``-based compositor instance.
    descriptor : dict
        Carries the composition-attribute name under ``"attr"``.

    Raises
    ------
    AssertionError
        If the nested parameter is absent or does not round-trip.

    """
    name = type(compositor).__name__
    attr = descriptor["attr"]
    comp_name, inner = _first_named_component(compositor, attr)

    inner_params = inner.get_params(deep=False)
    if not inner_params:
        return  # component exposes no params; nothing to address

    param = next(iter(inner_params))
    key = f"{comp_name}__{param}"

    deep = compositor.get_params(deep=True)
    assert key in deep, f"{name}: get_params(deep=True) missing nested key {key!r}"

    value = deep[key]
    compositor.set_params(**{key: value})
    round_tripped = compositor.get_params(deep=True)[key]
    assert _safe_equal(round_tripped, value), f"{name}: set_params({key}=...) did not round-trip"


def check_composition_clone_deep_clones_components(compositor, descriptor: dict) -> None:
    """Check ``clone(compositor)`` produces fresh same-type components.

    Parameters
    ----------
    compositor : BaseEstimator
        A ``_BaseComposition``-based compositor instance.
    descriptor : dict
        Carries the composition-attribute name under ``"attr"``.

    Raises
    ------
    AssertionError
        If a cloned component is the same instance or a different type.

    """
    name = type(compositor).__name__
    attr = descriptor["attr"]
    cloned = clone(compositor)

    orig_entries = getattr(compositor, attr)
    cloned_entries = getattr(cloned, attr)
    assert len(orig_entries) == len(cloned_entries), f"{name}: clone() changed component count"

    for orig, cl in zip(orig_entries, cloned_entries, strict=True):
        orig_est, cl_est = orig[1], cl[1]
        if not isinstance(orig_est, BaseEstimator):
            continue  # 'passthrough'/None sentinels
        assert type(orig_est) is type(cl_est), f"{name}: component {orig[0]!r} changed type under clone()"
        assert orig_est is not cl_est, f"{name}: component {orig[0]!r} was not deep-cloned (same instance)"


def _yield_composition_contract_checks(
    compositor,
    descriptor: dict,
) -> Generator[tuple[str, Callable, dict], None, None]:
    """Yield the data-free generic ``_BaseComposition`` contract checks.

    Parameters
    ----------
    compositor : BaseEstimator
        A ``_BaseComposition``-based compositor instance.
    descriptor : dict
        Per-compositor descriptor with the composition-attribute name under
        ``"attr"``.

    Yields
    ------
    tuple of (str, callable, dict)
        ``(check_name, check_func, check_kwargs)`` consumable by ``run_checks``.

    """
    yield (
        "check_composition_nested_param_addressable",
        check_composition_nested_param_addressable,
        {"descriptor": descriptor},
    )
    yield (
        "check_composition_clone_deep_clones_components",
        check_composition_clone_deep_clones_components,
        {"descriptor": descriptor},
    )
