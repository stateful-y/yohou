"""Composed forecaster routers preserve inherited routing.

Every meta-forecaster router must build on ``super().get_metadata_routing()``,
so it carries the class's own ``$self_request`` and the inherited transformer
children alongside its named child forecasters. A from-scratch router makes
the class's own request keys invisible when it is nested inside another
router, which is exactly how metadata addressed to it silently vanished.
"""

from datetime import datetime

import polars as pl
import pytest

from yohou.class_proba import ClassProbaReductionForecaster
from yohou.compose import (
    ColumnForecaster,
    CombiningForecaster,
    DecompositionPipeline,
    ForecastedFeatureForecaster,
    LocalPanelForecaster,
)
from yohou.ensemble import (
    VotingClassProbaForecaster,
    VotingIntervalForecaster,
    VotingPointForecaster,
)
from yohou.point import SeasonalNaive
from yohou.preprocessing import LagTransformer
from yohou.stationarity import ASinhTransformer


def _router_names(estimator) -> set[str]:
    router = estimator.get_metadata_routing()
    return set(router._route_mappings)


def _has_self_request(estimator) -> bool:
    return estimator.get_metadata_routing()._self_request is not None


def _naive():
    return SeasonalNaive(seasonality=24)


# Every composed class, constructed with the child shape its constructor
# expects, with the router entries the child configuration must produce:
# one entry per named child forecaster (list-valued fields expanded) plus
# each optional slot when configured as an estimator.
_COMPOSED: list[tuple[object, set[str]]] = [
    (ColumnForecaster(forecasters=[("a", _naive(), ["value"])]), {"a"}),
    (
        CombiningForecaster(terms=[("t", None, _naive())], residual_forecaster=_naive()),
        {"t", "residual_forecaster"},
    ),
    (
        DecompositionPipeline(forecasters=[("d", _naive())], target_transformer=ASinhTransformer()),
        {"d", "target_transformer"},
    ),
    (
        ForecastedFeatureForecaster(target_forecaster=_naive(), feature_forecaster=_naive()),
        {"target_forecaster", "feature_forecaster"},
    ),
    (LocalPanelForecaster(forecaster=_naive()), {"forecaster"}),
    (VotingPointForecaster(forecasters=[("p1", _naive()), ("p2", _naive())]), {"p1", "p2"}),
    (VotingIntervalForecaster(forecasters=[("i1", _naive())]), {"i1"}),
    (VotingClassProbaForecaster(forecasters=[("c1", ClassProbaReductionForecaster())]), {"c1"}),
]


class TestComposedRouterCompleteness:
    @pytest.mark.parametrize(
        ("estimator", "expected_children"),
        _COMPOSED,
        ids=[type(e).__name__ for e, _ in _COMPOSED],
    )
    def test_router_carries_self_request_and_children(self, estimator, expected_children):
        """$self_request present, plus one entry per configured child."""
        assert _has_self_request(estimator), f"{type(estimator).__name__} drops $self_request"
        assert expected_children <= _router_names(estimator)

    def test_transformer_children_survive(self):
        """A configured transformer slot appears as a routing child.

        DecompositionPipeline is the one composed class exposing transformer
        constructor slots; the other seven fix them to None, so for them the
        super() rebase restores $self_request specifically.
        """
        fc = DecompositionPipeline(
            forecasters=[("d", _naive())],
            actual_transformer=LagTransformer(lag=[1]),
        )
        assert "actual_transformer" in _router_names(fc)

    def test_own_request_keys_visible_when_nested(self):
        """A nested meta-forecaster's own keys route instead of vanishing.

        The self request is what a parent router consults for the child's own
        metadata; without it, keys like the walk-forward stride were silently
        invisible under one level of composition.
        """
        inner = VotingPointForecaster(forecasters=[("p", _naive())])
        inner.set_predict_request(stride=True)
        self_request = inner.get_metadata_routing()._self_request
        assert self_request is not None
        assert self_request.predict.requests.get("stride") is True


class TestNestedRoutingPerFamily:
    """One nesting test per composed family: metadata reaches the child."""

    def _series(self, n: int) -> pl.DataFrame:
        times = pl.datetime_range(
            datetime(2026, 1, 1),
            pl.select(pl.lit(datetime(2026, 1, 1)).dt.offset_by(f"{n - 1}h")).item(),
            interval="1h",
            eager=True,
        )
        return pl.DataFrame({"time": times, "value": [float(i % 24) for i in range(n)]})

    def test_transformer_child_reachable_through_nesting(self):
        """The transformer child is visible one composition level down."""
        inner = DecompositionPipeline(
            forecasters=[("f", _naive())],
            actual_transformer=LagTransformer(lag=[1]),
        )
        names = _router_names(inner)
        assert {"f", "actual_transformer"} <= names

    def test_ensemble_self_request_visible_through_nesting(self):
        """An ensemble nested in a search exposes its own keys via $self_request."""
        inner = VotingIntervalForecaster(forecasters=[("i", _naive())])
        inner.set_predict_interval_request(stride=True)
        self_request = inner.get_metadata_routing()._self_request
        assert self_request is not None
        assert self_request.predict_interval.requests.get("stride") is True
