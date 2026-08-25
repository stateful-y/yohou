"""Check functions for yohou conformal adapters.

Systematic validation functions for the ``BaseConformalAdapter`` estimator
family (adaptive conformal inference adapters). All check functions raise
AssertionError on failure.
"""

from sklearn.base import clone
from sklearn.exceptions import NotFittedError

__all__ = [
    "check_conformal_adapter_alpha_pooling_forwarded",
    "check_conformal_adapter_clipping",
    "check_conformal_adapter_methods_call_check_is_fitted",
    "check_conformal_adapter_observe_rewind_round_trip",
    "check_conformal_adapter_predict_returns_levels",
    "check_conformal_adapter_update_direction",
]

_RATES = [0.9, 0.8]


def _symmetric_errors(n: int) -> list[dict[float, object]]:
    """Build ``n`` per-row symmetric miscoverage dicts over ``_RATES``."""
    pattern = [1.0, 0.0]
    return [{rate: pattern[(i + j) % 2] for j, rate in enumerate(_RATES)} for i in range(n)]


def _asymmetric_errors(n: int) -> list[dict[float, object]]:
    """Build ``n`` per-row asymmetric ``(lower, upper)`` miscoverage dicts."""
    pattern = [(1.0, 0.0), (0.0, 1.0)]
    return [{rate: pattern[(i + j) % 2] for j, rate in enumerate(_RATES)} for i in range(n)]


def check_conformal_adapter_alpha_pooling_forwarded(adapter) -> None:
    """Check the constructor exposes ``alpha_pooling`` and ``clone`` preserves it.

    ``alpha_pooling`` is declared on ``BaseConformalAdapter``, but estimator
    parameter discovery reads the most derived constructor only. A subclass
    that does not accept it in its own ``__init__`` and forward it to
    ``super().__init__()`` silently drops the setting from ``get_params``,
    makes ``adapter__alpha_pooling`` unaddressable in a search, and lets
    ``clone`` reset a configured ``"shared"`` to the default.

    Parameters
    ----------
    adapter : BaseConformalAdapter
        Adapter instance.

    Raises
    ------
    AssertionError
        If ``alpha_pooling`` is missing from ``get_params``, or if cloning
        does not preserve a non-default ``alpha_pooling="shared"``.

    """
    name = type(adapter).__name__

    assert "alpha_pooling" in adapter.get_params(deep=False), (
        f"{name}.__init__ must accept alpha_pooling and forward it to super().__init__(); "
        f"without it the setting is missing from get_params and clone resets it"
    )

    configured = clone(adapter).set_params(alpha_pooling="shared")
    assert clone(configured).alpha_pooling == "shared", f"{name}: clone reset alpha_pooling='shared' to the default"


def check_conformal_adapter_predict_returns_levels(adapter) -> None:
    """Check ``predict`` returns one effective level per tracked coverage rate.

    Parameters
    ----------
    adapter : BaseConformalAdapter
        Adapter instance.

    Raises
    ------
    AssertionError
        If ``predict`` omits a tracked rate or returns a level outside
        ``[0, 1]`` (per tail for asymmetric scorers).

    """
    name = type(adapter).__name__

    fitted = clone(adapter).fit(_RATES, symmetric=True)
    levels = fitted.predict()
    assert set(levels) == set(_RATES), f"{name}.predict keys {set(levels)} != tracked rates {set(_RATES)}"
    for rate, level in levels.items():
        assert 0.0 <= float(level) <= 1.0, f"{name}.predict level {level} for rate {rate} outside [0, 1]"

    fitted_asym = clone(adapter).fit(_RATES, symmetric=False)
    levels_asym = fitted_asym.predict()
    for rate, level in levels_asym.items():
        lower, upper = level
        assert 0.0 <= float(lower) <= 1.0 and 0.0 <= float(upper) <= 1.0, (
            f"{name}.predict asymmetric level {level} for rate {rate} outside [0, 1]"
        )


def check_conformal_adapter_observe_rewind_round_trip(adapter) -> None:
    """Check ``observe`` then ``rewind`` of the same rows restores the levels.

    This is the core lifecycle invariant: a backtest that rolls forward then
    rewinds must land exactly on its pre-observe state.

    Parameters
    ----------
    adapter : BaseConformalAdapter
        Adapter instance.

    Raises
    ------
    AssertionError
        If observing then rewinding does not restore the effective levels, or
        if ``observe`` did not move them in the first place.

    """
    name = type(adapter).__name__

    for symmetric, errors in ((True, _symmetric_errors(4)), (False, _asymmetric_errors(4))):
        fitted = clone(adapter).fit(_RATES, symmetric=symmetric)
        before = fitted.predict()

        fitted.observe(errors)
        moved = fitted.predict()
        assert moved != before, f"{name}.observe (symmetric={symmetric}) did not change any level"

        fitted.rewind(len(errors))
        after = fitted.predict()
        assert after == before, (
            f"{name}: observe+rewind (symmetric={symmetric}) did not restore levels: {before} != {after}"
        )


def check_conformal_adapter_update_direction(adapter) -> None:
    """Check miscoverage lowers the level and coverage raises it (symmetric).

    Parameters
    ----------
    adapter : BaseConformalAdapter
        Adapter instance.

    Raises
    ------
    AssertionError
        If a fully-miscovered update does not decrease the level, or a
        fully-covered update does not increase it.

    """
    name = type(adapter).__name__
    rate = _RATES[0]

    miscover = clone(adapter).fit([rate], symmetric=True)
    seed = float(miscover.predict()[rate])
    miscover.observe([{rate: 1.0}])
    assert float(miscover.predict()[rate]) < seed, (
        f"{name}: a miscovered observation must lower the level (widen the interval)"
    )

    cover = clone(adapter).fit([rate], symmetric=True)
    cover.observe([{rate: 0.0}])
    assert float(cover.predict()[rate]) > seed, (
        f"{name}: a covered observation must raise the level (narrow the interval)"
    )


def check_conformal_adapter_clipping(adapter) -> None:
    """Check a positive ``epsilon`` keeps the level within ``[epsilon, 1 - epsilon]``.

    Skipped for adapters without an ``epsilon`` parameter.

    Parameters
    ----------
    adapter : BaseConformalAdapter
        Adapter instance.

    Raises
    ------
    AssertionError
        If a sustained miscoverage streak drives the level below ``epsilon``.

    """
    name = type(adapter).__name__
    if "epsilon" not in adapter.get_params(deep=False):
        return

    rate = _RATES[0]
    epsilon = 0.1
    clipped = clone(adapter).set_params(epsilon=epsilon).fit([rate], symmetric=True)
    clipped.observe([{rate: 1.0}] * 200)
    level = float(clipped.predict()[rate])
    assert level >= epsilon - 1e-9, f"{name}: level {level} fell below epsilon={epsilon} despite clipping"
    assert level <= 1.0 - epsilon + 1e-9, f"{name}: level {level} rose above 1 - epsilon={1 - epsilon}"


def check_conformal_adapter_methods_call_check_is_fitted(adapter) -> None:
    """Check ``predict``, ``observe``, and ``rewind`` raise ``NotFittedError`` before ``fit``.

    Parameters
    ----------
    adapter : BaseConformalAdapter
        Adapter instance.

    Raises
    ------
    AssertionError
        If any of ``predict``, ``observe``, or ``rewind`` does not raise
        ``NotFittedError`` when unfitted.

    """
    name = type(adapter).__name__

    unfitted = clone(adapter)
    try:
        unfitted.predict()
        raise AssertionError(f"{name}.predict() must raise NotFittedError when unfitted")
    except NotFittedError:
        pass

    unfitted2 = clone(adapter)
    try:
        unfitted2.observe([{_RATES[0]: 1.0}])
        raise AssertionError(f"{name}.observe() must raise NotFittedError when unfitted")
    except NotFittedError:
        pass

    unfitted3 = clone(adapter)
    try:
        unfitted3.rewind(1)
        raise AssertionError(f"{name}.rewind() must raise NotFittedError when unfitted")
    except NotFittedError:
        pass
