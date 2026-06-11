"""Unit tests for the shared estimator-contract and composition-contract checks."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from sklearn.base import BaseEstimator

from yohou.testing.contract import (
    _safe_equal,
    check_clone_preserves_params,
    check_composition_clone_deep_clones_components,
    check_composition_nested_param_addressable,
    check_get_set_params_round_trip,
    check_init_no_param_mutation,
)


class _Inner(BaseEstimator):
    def __init__(self, p=1):
        self.p = p


class _NoParams(BaseEstimator):
    """Estimator whose constructor takes no tunable parameters."""


class _Compo(BaseEstimator):
    def __init__(self, items=None, scalar=2, nothing=None, sub=None):
        self.items = items
        self.scalar = scalar
        self.nothing = nothing
        self.sub = sub


def test_safe_equal_identity():
    obj = object()
    assert _safe_equal(obj, obj)


def test_safe_equal_scalar():
    assert _safe_equal(3, 3)
    assert not _safe_equal(3, 4)


def test_safe_equal_dataframe_uses_equals():
    df = pl.DataFrame({"a": [1, 2]})
    assert _safe_equal(df, pl.DataFrame({"a": [1, 2]}))
    assert not _safe_equal(df, pl.DataFrame({"a": [1, 3]}))


def test_safe_equal_equals_method_raising_returns_false():
    class HasBadEquals:
        def equals(self, other):
            raise RuntimeError("boom")

    assert not _safe_equal(HasBadEquals(), HasBadEquals())


def test_safe_equal_numpy_array_reduces_with_all():
    a = np.array([1, 2, 3])
    assert _safe_equal(a, np.array([1, 2, 3]))
    assert not _safe_equal(a, np.array([1, 2, 4]))


def test_safe_equal_eq_raising_returns_false():
    class BadEq:
        __hash__ = object.__hash__

        def __eq__(self, other):
            raise RuntimeError("boom")

    assert not _safe_equal(BadEq(), BadEq())


def test_safe_equal_array_all_raising_returns_false():
    class WeirdResult:
        def all(self):
            raise RuntimeError("boom")

    class WeirdEq:
        __hash__ = object.__hash__

        def __eq__(self, other):
            return WeirdResult()

    assert not _safe_equal(WeirdEq(), WeirdEq())


def test_safe_equal_non_bool_non_array_result_falls_back_to_bool():
    class IntEq:
        __hash__ = object.__hash__

        def __eq__(self, other):
            return 1  # truthy, not a bool, no .all()

    assert _safe_equal(IntEq(), IntEq())


def test_clone_preserves_params_covers_all_branches():
    # items: list of (name, estimator) tuples (two components, so the loop
    # iterates more than once); sub: nested estimator; nothing: None;
    # scalar: plain value -> exercises every branch.
    check_clone_preserves_params(_Compo(items=[("a", _Inner()), ("b", _Inner(p=2))], sub=_Inner()))


def test_clone_preserves_params_handles_sentinels_and_non_tuples():
    # ('p', 'passthrough'): tuple whose component is not an estimator (skips the
    # type assertion); a bare estimator entry: not a tuple at all (skipped).
    check_clone_preserves_params(_Compo(items=[("p", "passthrough"), ("a", _Inner()), _Inner()]))


def test_clone_preserves_params_dataframe_param():
    class WithFrame(BaseEstimator):
        def __init__(self, frame=None):
            self.frame = frame

    check_clone_preserves_params(WithFrame(frame=pl.DataFrame({"a": [1, 2]})))


def test_get_set_params_round_trip_covers_nested_skip():
    check_get_set_params_round_trip(_Compo(items=[("a", _Inner())], sub=_Inner()))


def test_init_no_param_mutation_passes_for_verbatim_storage():
    check_init_no_param_mutation(_Compo())


def test_init_no_param_mutation_skips_non_default_constructible():
    class NeedsArg(BaseEstimator):
        def __init__(self, x):
            self.x = x

    # No default for x -> early return, no assertion error.
    check_init_no_param_mutation(NeedsArg(1))


def test_init_no_param_mutation_skips_attr_stored_elsewhere():
    class StoresElsewhere(BaseEstimator):
        def __init__(self, a=1):
            self._a = a  # not stored as self.a

    check_init_no_param_mutation(StoresElsewhere())


def test_init_no_param_mutation_detects_mutation():
    class Mutates(BaseEstimator):
        def __init__(self, a=None):
            self.a = a if a is not None else {}  # None -> {}

    with pytest.raises(AssertionError, match="mutated"):
        check_init_no_param_mutation(Mutates())


def test_composition_nested_param_addressable_skips_paramless_component():
    comp = _Compo(items=[("a", _NoParams())])
    # First component exposes no params -> the check returns without asserting.
    check_composition_nested_param_addressable(comp, {"attr": "items"})


def test_composition_clone_deep_clones_components_happy_path():
    check_composition_clone_deep_clones_components(_Compo(items=[("a", _Inner())]), {"attr": "items"})


def test_composition_clone_deep_clones_components_skips_sentinel():
    # Non-estimator components (e.g. 'passthrough') are skipped, not deep-clone checked.
    check_composition_clone_deep_clones_components(_Compo(items=[("a", "passthrough")]), {"attr": "items"})
