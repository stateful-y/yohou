"""SplitConformalForecaster routes received metadata to its point forecaster.

The wrapper's ``**params`` used to be swallowed while the docstrings claimed
routing; now the point forecaster is a routing child for the methods the
wrapper delegates, and everything else (conformity scorer, similarity,
adapters) stays deliberately outside the routing graph.
"""

from datetime import datetime
from typing import Any, ClassVar

import polars as pl

from yohou.interval import SplitConformalForecaster
from yohou.point import SeasonalNaive


def _series(n: int) -> pl.DataFrame:
    times = pl.datetime_range(
        datetime(2026, 1, 1),
        pl.select(pl.lit(datetime(2026, 1, 1)).dt.offset_by(f"{n - 1}h")).item(),
        interval="1h",
        eager=True,
    )
    return pl.DataFrame({"time": times, "value": [float(i % 24) for i in range(n)]})


class _TaggedNaive(SeasonalNaive):
    """Point forecaster whose fit accepts a routable metadata tag."""

    seen_tags: ClassVar[list[Any]] = []

    def fit(self, y, X_actual=None, forecasting_horizon=1, my_tag=None, X_future=None, X_forecast=None, **params):
        type(self).seen_tags.append(my_tag)
        return super().fit(
            y,
            X_actual=X_actual,
            forecasting_horizon=forecasting_horizon,
            X_future=X_future,
            X_forecast=X_forecast,
            **params,
        )


class TestRoutedMetadataReachesTheChild:
    def test_fit_metadata_routes_to_the_point_forecaster(self):
        point = _TaggedNaive(seasonality=24)
        point.set_fit_request(my_tag=True)
        _TaggedNaive.seen_tags.clear()

        fc = SplitConformalForecaster(point_forecaster=point, calibration_size=12)
        fc.fit(y=_series(60), forecasting_horizon=4, coverage_rates=[0.5], my_tag="routed")

        assert "routed" in _TaggedNaive.seen_tags

    def test_no_metadata_no_change(self):
        """A caller that routes nothing sees the previous behavior exactly."""
        fc = SplitConformalForecaster(point_forecaster=SeasonalNaive(seasonality=24), calibration_size=12)
        fc.fit(y=_series(60), forecasting_horizon=4, coverage_rates=[0.5])
        intervals = fc.predict_interval(coverage_rates=[0.5])
        assert len(intervals) == 4


class TestRouterShape:
    def test_router_contains_point_forecaster_and_inherited_slots_only(self):
        """The child roster: the point forecaster plus inherited transformer slots.

        No entry for the conformity scorer, similarity, or adapters: those are
        constructor-configured and documented as receiving no routed metadata.
        """
        fc = SplitConformalForecaster(point_forecaster=SeasonalNaive(seasonality=24))
        router = fc.get_metadata_routing()
        names = set(router._route_mappings)
        assert "point_forecaster" in names
        assert names <= {"point_forecaster", "target_transformer", "actual_transformer", "forecast_transformer"}
        assert router._self_request is not None

    def test_predict_family_maps_onto_child_predict(self):
        fc = SplitConformalForecaster(point_forecaster=SeasonalNaive(seasonality=24))
        mapping = fc.get_metadata_routing()._route_mappings["point_forecaster"].mapping
        pairs = {(m.caller, m.callee) for m in mapping}
        assert {
            ("fit", "fit"),
            ("predict", "predict"),
            ("predict_interval", "predict"),
            ("observe_predict", "predict"),
            ("observe_predict_interval", "predict"),
        } == pairs
