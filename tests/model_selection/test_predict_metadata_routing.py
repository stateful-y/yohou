"""End-to-end predict-side metadata routing for every scoring family.

The search's fit routes ``**params`` into the bucket named by the scorers'
resolved response method; ``_predict`` splats that bucket into the dispatched
observe method. These tests pin the whole chain for point-scored and
class-proba-scored searches (the interval chain is pinned by the downstream
spike test that introduced it) plus the per-callee discipline: unpaired
requests fail naming the unset method, a paired key routes cleanly into the
resolved response method's bucket, and plain (non-router) consumers keep
strict bucket exclusivity.
"""

from datetime import datetime
from typing import Any, ClassVar

import polars as pl
import pytest
from sklearn.utils._metadata_requests import UnsetMetadataPassedError, process_routing

from yohou.class_proba import ClassProbaReductionForecaster
from yohou.interval import SplitConformalForecaster
from yohou.metrics import MeanAbsoluteError
from yohou.metrics.class_proba import LogLoss
from yohou.model_selection import ExpandingWindowSplitter, GridSearchCV
from yohou.point import SeasonalNaive


def _series(n: int) -> pl.DataFrame:
    times = pl.datetime_range(
        datetime(2026, 1, 1),
        pl.select(pl.lit(datetime(2026, 1, 1)).dt.offset_by(f"{n - 1}h")).item(),
        interval="1h",
        eager=True,
    )
    return pl.DataFrame({"time": times, "value": [float(i % 24) for i in range(n)]})


def _pair_off_if_carried(forecaster, method: str, key: str) -> None:
    """Set ``key`` not-requested on ``method`` only where the key exists.

    The leaked sibling carrier exists only under substring resolution of
    request attributes (absent on exact-match sklearn versions such as 1.6),
    so the pairing is conditional, mirroring the discovery-based pairing a
    production caller uses.
    """
    if key in getattr(forecaster._get_metadata_request(), method).requests:
        getattr(forecaster, f"set_{method}_request")(**{key: False})


def _categorical_series(n: int) -> pl.DataFrame:
    times = pl.datetime_range(
        datetime(2026, 1, 1),
        pl.select(pl.lit(datetime(2026, 1, 1)).dt.offset_by(f"{n - 1}h")).item(),
        interval="1h",
        eager=True,
    )
    return pl.DataFrame({"time": times, "label": [("low", "mid", "high")[i % 3] for i in range(n)]})


class _RecordingNaive(SeasonalNaive):
    """Point forecaster recording the stride each observe_predict receives."""

    recorded_strides: ClassVar[list[Any]] = []

    def observe_predict(self, *args, **kwargs):
        type(self).recorded_strides.append(kwargs.get("stride"))
        return super().observe_predict(*args, **kwargs)


class _RecordingClassProba(ClassProbaReductionForecaster):
    """Class-proba forecaster recording the stride each observe call receives."""

    recorded_strides: ClassVar[list[Any]] = []

    def observe_predict_class_proba(self, *args, **kwargs):
        type(self).recorded_strides.append(kwargs.get("stride"))
        return super().observe_predict_class_proba(*args, **kwargs)


class TestStrideRoutesForEveryScoringFamily:
    def test_point_scored_search_receives_stride(self):
        forecaster = _RecordingNaive(seasonality=24)
        _RecordingNaive.recorded_strides.clear()
        forecaster.set_predict_request(stride=True)
        search = GridSearchCV(
            forecaster=forecaster,
            param_grid={"seasonality": [12, 24]},
            scoring=MeanAbsoluteError(),
            cv=ExpandingWindowSplitter(n_splits=2, test_size=8),
            refit=False,
        )
        search.fit(_series(96), forecasting_horizon=4, stride=4)
        assert _RecordingNaive.recorded_strides, "inner walk-forward never ran"
        assert all(s == 4 for s in _RecordingNaive.recorded_strides)

    def test_class_proba_scored_search_receives_stride(self):
        # Direct strategy: one classifier per horizon step, since the default
        # LogisticRegression cannot fit a multi-column target.
        forecaster = _RecordingClassProba(reduction_strategy="direct")
        _RecordingClassProba.recorded_strides.clear()
        forecaster.set_predict_class_proba_request(stride=True)
        # Under substring resolution the family's predict also carries the
        # leaked stride key and fit maps onto predict, so the sibling carrier
        # is paired off where it exists.
        _pair_off_if_carried(forecaster, "predict", "stride")
        search = GridSearchCV(
            forecaster=forecaster,
            param_grid={"target_as_feature": ["transformed", None]},
            scoring=LogLoss(),
            cv=ExpandingWindowSplitter(n_splits=2, test_size=6),
            refit=False,
        )
        search.fit(_categorical_series(60), forecasting_horizon=2, stride=2)
        assert _RecordingClassProba.recorded_strides, "inner walk-forward never ran"
        assert all(s == 2 for s in _RecordingClassProba.recorded_strides)


class TestPerCalleeDiscipline:
    def test_unpaired_request_fails_naming_the_unset_method(self):
        """SplitConformal carries stride on predict too; leaving it unset raises."""
        forecaster = SplitConformalForecaster(point_forecaster=SeasonalNaive(seasonality=24), calibration_size=12)
        if "stride" not in forecaster._get_metadata_request().predict.requests:
            pytest.skip("no substring leak on this sklearn: no sibling carrier exists to leave unpaired")
        forecaster.set_predict_interval_request(stride=True)
        search = GridSearchCV(
            forecaster=forecaster,
            param_grid={"point_forecaster__seasonality": [12, 24]},
            scoring=MeanAbsoluteError(),
            cv=ExpandingWindowSplitter(n_splits=2, test_size=8),
            refit=False,
        )
        with pytest.raises(UnsetMetadataPassedError, match="predict"):
            search.fit(_series(96), forecasting_horizon=4, stride=4)

    def test_paired_key_routes_cleanly_and_fit_bucket_stays_empty(self):
        """The paired request routes without error and reaches its bucket.

        SplitConformalForecaster is itself a router (its point forecaster is
        a routing child), and sklearn composes a router-consumer's bucket
        from everything its subtree knows: the predict bucket may therefore
        carry the key too, with the fine-grained True/False filtering applied
        at the inner level. The enforceable outer guarantees are: no
        unset-metadata error, the response-method bucket carries the value,
        and the fit bucket stays clean.
        """
        forecaster = SplitConformalForecaster(point_forecaster=SeasonalNaive(seasonality=24), calibration_size=12)
        forecaster.set_predict_interval_request(stride=True)
        _pair_off_if_carried(forecaster, "predict", "stride")
        search = GridSearchCV(
            forecaster=forecaster,
            param_grid={"point_forecaster__seasonality": [12, 24]},
            scoring=MeanAbsoluteError(),
            cv=ExpandingWindowSplitter(n_splits=2, test_size=8),
            refit=False,
        )
        routed = process_routing(search, "fit", stride=4)
        assert dict(routed.forecaster.predict_interval) == {"stride": 4}
        assert dict(routed.forecaster.fit) == {}

    def test_paired_key_bucket_exclusivity_for_plain_consumers(self):
        """A non-router consumer keeps strict bucket exclusivity."""
        from yohou.interval import IntervalReductionForecaster

        forecaster = IntervalReductionForecaster()
        forecaster.set_predict_interval_request(stride=True)
        search = GridSearchCV(
            forecaster=forecaster,
            param_grid={"estimator__estimator__alpha": [0.5, 1.0]},
            scoring=MeanAbsoluteError(),
            cv=ExpandingWindowSplitter(n_splits=2, test_size=8),
            refit=False,
        )
        routed = process_routing(search, "fit", stride=4)
        assert dict(routed.forecaster.predict_interval) == {"stride": 4}
        assert dict(routed.forecaster.predict) == {}
        assert dict(routed.forecaster.fit) == {}
