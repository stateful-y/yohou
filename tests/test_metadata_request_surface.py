"""Pinned per-class metadata request surfaces for the response methods.

The routable-key contract is three-layered: a key must be declarable on the
response method, mapped by the search router, and consumed by the dispatched
observe method. sklearn resolves ``__metadata_request__*`` class attributes by
substring over mangled names, so declarations leak onto shorter method names
of classes that define them. These tables pin the EFFECTIVE key sets,
including the deliberate leaks, so a declaration change or an sklearn
resolution change fails here rather than silently altering the surface.
"""

import pytest

from yohou.class_proba import ClassProbaReductionForecaster
from yohou.ensemble import VotingIntervalForecaster
from yohou.interval import IntervalReductionForecaster, SplitConformalForecaster
from yohou.point import PointReductionForecaster, SeasonalNaive


def _request_keys(estimator, method: str) -> set[str]:
    """Effective request keys of one method on one instance."""
    return set(getattr(estimator._get_metadata_request(), method).requests)


_DATA_ARGS_PREDICT = {"X_future", "X_forecast"}

# The pinned tables. Every entry is the COMPLETE effective set: data arguments
# that sklearn's exclusion list misses are listed too, so their later
# suppression (or a new leak) is a visible diff here, never a silent one.
_SURFACE: list[tuple[object, str, set[str]]] = [
    # Point family: stride declared on predict; predict_transformed and groups
    # are signature keys.
    (
        SeasonalNaive(seasonality=24),
        "predict",
        _DATA_ARGS_PREDICT | {"forecasting_horizon", "groups", "predict_transformed", "stride"},
    ),
    (
        SeasonalNaive(seasonality=24),
        "observe_predict",
        _DATA_ARGS_PREDICT | {"X_actual", "forecasting_horizon", "groups", "predict_transformed", "stride"},
    ),
    (
        PointReductionForecaster(),
        "predict",
        _DATA_ARGS_PREDICT | {"forecasting_horizon", "groups", "predict_transformed", "stride"},
    ),
    # Interval family: stride declared on predict_interval; strategy and groups
    # are signature keys; predict_transformed is NOT declarable there because
    # observe_predict_interval rejects it. IntervalReductionForecaster defines
    # no predict function, so nothing leaks onto predict and its set is empty.
    (
        IntervalReductionForecaster(),
        "predict_interval",
        _DATA_ARGS_PREDICT | {"forecasting_horizon", "coverage_rates", "strategy", "groups", "stride"},
    ),
    (IntervalReductionForecaster(), "predict", set()),
    # SplitConformalForecaster defines predict, so the predict_interval
    # declaration leaks stride onto it: pinned as deliberate.
    (
        SplitConformalForecaster(),
        "predict",
        _DATA_ARGS_PREDICT | {"forecasting_horizon", "groups", "predict_transformed", "stride"},
    ),
    (
        SplitConformalForecaster(),
        "predict_interval",
        _DATA_ARGS_PREDICT | {"forecasting_horizon", "coverage_rates", "strategy", "groups", "stride"},
    ),
    (
        VotingIntervalForecaster(forecasters=[("a", SeasonalNaive(seasonality=24))]),
        "predict",
        _DATA_ARGS_PREDICT | {"forecasting_horizon", "groups", "predict_transformed", "stride"},
    ),
    # Class-proba family: stride declared on predict_class_proba; the substring
    # resolution also places it on the family's predict, pinned as deliberate.
    (
        ClassProbaReductionForecaster(),
        "predict_class_proba",
        _DATA_ARGS_PREDICT | {"forecasting_horizon", "groups", "stride"},
    ),
    (
        ClassProbaReductionForecaster(),
        "predict",
        _DATA_ARGS_PREDICT | {"forecasting_horizon", "groups", "stride"},
    ),
]


@pytest.mark.parametrize(
    ("estimator", "method", "expected"),
    _SURFACE,
    ids=[f"{type(e).__name__}.{m}" for e, m, _ in _SURFACE],
)
def test_effective_request_surface_is_pinned(estimator, method, expected):
    """The effective request key set of each response method is the contract."""
    assert _request_keys(estimator, method) == expected


def test_all_declared_keys_default_to_unset():
    """Every pinned key carries alias None so default requests stay empty."""
    for estimator, method, expected in _SURFACE:
        requests = getattr(estimator._get_metadata_request(), method).requests
        assert set(requests) == expected
        assert all(alias is None for alias in requests.values()), (type(estimator).__name__, method, requests)


class TestRequestHygiene:
    """Data arguments are suppressed and dispatch registration is preserved."""

    def test_suppressed_data_argument_keys_are_absent(self):
        from yohou.metrics import MeanAbsoluteError
        from yohou.preprocessing import FunctionTransformer, LagTransformer

        assert _request_keys(LagTransformer(lag=[1]), "inverse_transform") == set()
        assert _request_keys(FunctionTransformer(), "transform") == set()
        scorer = MeanAbsoluteError()
        assert _request_keys(scorer, "fit") == set()
        assert _request_keys(scorer, "score") == set()

    def test_set_request_surfaces_do_not_offer_data_arguments(self):
        """The setters either vanish outright or reject the suppressed keys."""
        from yohou.metrics import MeanAbsoluteError
        from yohou.preprocessing import LagTransformer

        transformer = LagTransformer(lag=[1])
        setter = getattr(transformer, "set_inverse_transform_request", None)
        if setter is not None:
            with pytest.raises(TypeError):
                setter(X_t=True)
        scorer_setter = getattr(MeanAbsoluteError(), "set_score_request", None)
        if scorer_setter is not None:
            with pytest.raises(TypeError):
                scorer_setter(y_pred=True)

    def test_search_fit_absorbed_parameters_stay_explicit(self):
        """The absorbed dual-membership keys must remain named fit parameters.

        Demoting any of these into **params would create instant unmitigated
        routing hazards: they are request keys of both fit and
        predict_interval, protected today only by binding to the explicit
        signature.
        """
        import inspect

        from yohou.model_selection.search import BaseSearchCV

        params = inspect.signature(BaseSearchCV.fit).parameters
        for name in ("y", "X_actual", "forecasting_horizon", "X_future", "X_forecast"):
            assert name in params, f"search.fit lost explicit parameter {name!r}"
            assert params[name].kind is not inspect.Parameter.VAR_KEYWORD

    def test_dispatch_registration_mappings_are_present(self):
        """observe/rewind mappings are process_routing caller registration.

        They carry no keys today, but the stateful methods dispatch through
        process_routing, so removing the mapping breaks dispatch the moment
        any parameter flows. This pins them against a future cleanup.
        """
        from yohou.compose import FeaturePipeline, FeatureUnion
        from yohou.preprocessing import LagTransformer

        for composed in (
            FeaturePipeline(steps=[("lag", LagTransformer(lag=[1]))]),
            FeatureUnion(transformer_list=[("lag", LagTransformer(lag=[1]))]),
        ):
            mapping = composed.get_metadata_routing()._route_mappings["lag"].mapping
            pairs = {(m.caller, m.callee) for m in mapping}
            assert ("observe_transform", "observe_transform") in pairs
            assert ("rewind_transform", "rewind_transform") in pairs
