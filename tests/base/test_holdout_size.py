"""Tests for the ``holdout_size`` tag: how many trailing fit rows a forecaster does not learn from."""

import pytest
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeClassifier

from holdout_stub import _HoldoutStubForecaster, _SelectColumn
from yohou.base import BaseForecaster
from yohou.class_proba import ClassProbaReductionForecaster
from yohou.compose import (
    ColumnForecaster,
    CombiningForecaster,
    DecompositionPipeline,
    ForecastedFeatureForecaster,
    LocalPanelForecaster,
)
from yohou.ensemble import VotingClassProbaForecaster, VotingIntervalForecaster, VotingPointForecaster
from yohou.interval import IntervalReductionForecaster, SplitConformalForecaster
from yohou.metrics import MeanAbsoluteError
from yohou.metrics.interval import IntervalScore
from yohou.model_selection import GridSearchCV, RandomizedSearchCV
from yohou.point import MeanSeasonalNaive, PointReductionForecaster, SeasonalNaive
from yohou.stationarity import FourierSeasonalityForecaster, PatternSeasonalityForecaster, PolynomialTrendForecaster
from yohou.utils.discovery import all_estimators

FH = 3


def _declared(forecaster):
    return forecaster.__sklearn_tags__().forecaster_tags.holdout_size


@pytest.fixture
def y_X(y_X_factory):
    """Two-target standard series long enough for a calibration stretch."""
    return y_X_factory(length=200, n_targets=2, n_features=1)


class TestPlainForecaster:
    """A forecaster that learns from every row declares zero."""

    def test_declares_zero_before_and_after_fit(self, y_X):
        y, _ = y_X
        forecaster = PointReductionForecaster(Ridge())
        assert _declared(forecaster) == 0
        forecaster.fit(y, forecasting_horizon=FH)
        assert _declared(forecaster) == 0


class TestHoldoutStub:
    """The stub behaves as a point forecaster on both data shapes, so composites can use it."""

    def test_standard_data(self, y_X):
        y, _ = y_X
        stub = _HoldoutStubForecaster(holdout=5).fit(y[:150], forecasting_horizon=FH)
        assert _declared(stub) == 5
        assert len(stub.predict(forecasting_horizon=FH)) == FH
        stub.rewind(y[:120])
        assert len(stub.observe_predict(y[120:150])) > 0

    def test_panel_data(self, y_X_panel_factory):
        y, _ = y_X_panel_factory(n_groups=2, length=150, n_targets=1, n_features=1)
        stub = _HoldoutStubForecaster(holdout=5).fit(y[:120], forecasting_horizon=FH)
        assert len(stub.predict(forecasting_horizon=FH)) == FH
        stub.observe(y[120:130])
        assert _declared(stub) == 5


class TestSplitConformal:
    """A split-conformal forecaster declares its calibration stretch."""

    @pytest.mark.parametrize("calibration_size", [168, 720])
    def test_declares_calibration_size_before_and_after_fit(self, y_X_factory, calibration_size):
        y, _ = y_X_factory(length=calibration_size + 100, n_targets=1, n_features=1)
        forecaster = SplitConformalForecaster(
            point_forecaster=PointReductionForecaster(Ridge()), calibration_size=calibration_size
        )
        assert _declared(forecaster) == calibration_size
        forecaster.fit(y, forecasting_horizon=FH)
        assert _declared(forecaster) == calibration_size

    def test_nested_holdouts_add_up(self, y_X_factory):
        y, _ = y_X_factory(length=300, n_targets=1, n_features=1)
        forecaster = SplitConformalForecaster(point_forecaster=_HoldoutStubForecaster(holdout=24), calibration_size=168)
        assert _declared(forecaster) == 192
        forecaster.fit(y, forecasting_horizon=FH)
        assert _declared(forecaster) == 192

    def test_value_unchanged_by_observe_rewind_and_predict(self, y_X_factory):
        y, _ = y_X_factory(length=400, n_targets=1, n_features=1)
        forecaster = SplitConformalForecaster(point_forecaster=SeasonalNaive(), calibration_size=168)
        forecaster.fit(y[:300], forecasting_horizon=FH)
        assert _declared(forecaster) == 168
        forecaster.observe(y[300:350])
        assert _declared(forecaster) == 168
        forecaster.rewind(y[:320])
        assert _declared(forecaster) == 168
        forecaster.observe_predict_interval(y[320:360], coverage_rates=[0.9])
        assert _declared(forecaster) == 168


class TestSearch:
    """A search declares its base forecaster's value until refitted, then its best forecaster's."""

    def test_tuned_calibration_size_shows_after_refit(self, y_X_factory):
        y, _ = y_X_factory(length=1200, n_targets=1, n_features=1)
        search = GridSearchCV(
            forecaster=SplitConformalForecaster(point_forecaster=SeasonalNaive(), calibration_size=168),
            param_grid={"calibration_size": [720]},
            scoring=IntervalScore(coverage_rates=[0.9]),
            cv=2,
        )
        assert _declared(search) == 168
        search.fit(y, forecasting_horizon=FH)
        assert _declared(search) == 720


class TestRealConformalChildren:
    """Composites that accept an interval child find a real conformal model's stretch."""

    def test_column_forecaster(self, y_X):
        y, _ = y_X
        forecaster = ColumnForecaster(
            forecasters=[
                ("conformal", SplitConformalForecaster(point_forecaster=SeasonalNaive(), calibration_size=48), "y_0"),
                ("shorter", SplitConformalForecaster(point_forecaster=SeasonalNaive(), calibration_size=24), "y_1"),
            ]
        )
        assert _declared(forecaster) == 48
        forecaster.fit(y, forecasting_horizon=FH)
        assert _declared(forecaster) == 48

    def test_local_panel_forecaster(self, y_X_panel_factory):
        y, _ = y_X_panel_factory(n_groups=2, length=200, n_targets=1, n_features=1)
        forecaster = LocalPanelForecaster(
            SplitConformalForecaster(point_forecaster=SeasonalNaive(), calibration_size=48)
        )
        assert _declared(forecaster) == 48
        forecaster.fit(y, forecasting_horizon=FH)
        assert _declared(forecaster) == 48


# --------------------------------------------------------------------------------------
# Registry sweep: every forecaster yohou ships must be classified as plain or composite.
# --------------------------------------------------------------------------------------

_SWEEP_CALIBRATION = 24


def _standard(y_X_factory):
    return y_X_factory(length=200, n_targets=2, n_features=1)


def _plain(make):
    """Plain forecaster builder: fitted on standard data, expected to declare 0."""
    return lambda factories: (make(), _standard(factories["y_X"]), 0)


def _class_proba_forecaster():
    return ClassProbaReductionForecaster(estimator=DecisionTreeClassifier(random_state=0), reduction_strategy="direct")


_PLAIN_FORECASTERS = {
    # Returned by the registry although it is a base class; classified, never built.
    "BasePointForecaster": None,
    "ClassProbaReductionForecaster": lambda f: (
        _class_proba_forecaster(),
        f["class_proba"](length=200, n_targets=1, n_features=1),
        0,
    ),
    "FourierSeasonalityForecaster": _plain(lambda: FourierSeasonalityForecaster(seasonality=12, harmonics=[1, 2])),
    "IntervalReductionForecaster": _plain(IntervalReductionForecaster),
    "MeanSeasonalNaive": _plain(MeanSeasonalNaive),
    "PatternSeasonalityForecaster": _plain(lambda: PatternSeasonalityForecaster(seasonality=12)),
    "PointReductionForecaster": _plain(lambda: PointReductionForecaster(Ridge())),
    "PolynomialTrendForecaster": _plain(PolynomialTrendForecaster),
    "SeasonalNaive": _plain(SeasonalNaive),
}


def _stub(holdout):
    return _HoldoutStubForecaster(holdout=holdout)


def _conformal(calibration_size, point_forecaster=None):
    return SplitConformalForecaster(
        point_forecaster=point_forecaster if point_forecaster is not None else SeasonalNaive(),
        calibration_size=calibration_size,
    )


_COMPOSITE_BUILDERS = {
    "ColumnForecaster": lambda f: (
        ColumnForecaster(forecasters=[("a", _stub(168), "y_0"), ("b", _stub(0), "y_1")]),
        _standard(f["y_X"]),
        168,
    ),
    "CombiningForecaster": lambda f: (
        CombiningForecaster(
            terms=[
                ("a", _SelectColumn("y_0", "a_part"), _stub(168)),
                ("b", _SelectColumn("y_0", "b_part"), _stub(0)),
            ]
        ),
        f["y_X"](length=200, n_targets=1, n_features=1),
        168,
    ),
    "DecompositionPipeline": lambda f: (
        DecompositionPipeline(forecasters=[("a", _stub(168)), ("b", _stub(0))]),
        _standard(f["y_X"]),
        168,
    ),
    "ForecastedFeatureForecaster": lambda f: (
        ForecastedFeatureForecaster(target_forecaster=_stub(168), feature_forecaster=_stub(0)),
        _standard(f["y_X"]),
        168,
    ),
    "GridSearchCV": lambda f: (
        GridSearchCV(forecaster=_stub(168), param_grid={"seasonality": [1]}, scoring=MeanAbsoluteError(), cv=2),
        _standard(f["y_X"]),
        168,
    ),
    "LocalPanelForecaster": lambda f: (
        LocalPanelForecaster(_stub(168)),
        f["panel"](n_groups=2, length=200, n_targets=1, n_features=1),
        168,
    ),
    "RandomizedSearchCV": lambda f: (
        RandomizedSearchCV(
            forecaster=_stub(168),
            param_distributions={"seasonality": [1]},
            n_iter=1,
            scoring=MeanAbsoluteError(),
            cv=2,
        ),
        _standard(f["y_X"]),
        168,
    ),
    "SplitConformalForecaster": lambda f: (
        _conformal(_SWEEP_CALIBRATION, point_forecaster=_stub(168)),
        _standard(f["y_X"]),
        _SWEEP_CALIBRATION + 168,
    ),
    "VotingClassProbaForecaster": lambda f: (
        VotingClassProbaForecaster(forecasters=[("a", _class_proba_forecaster()), ("b", _class_proba_forecaster())]),
        f["class_proba"](length=200, n_targets=1, n_features=1),
        0,
    ),
    # Checks its children's family, so real conformal children stand in for stubs.
    "VotingIntervalForecaster": lambda f: (
        VotingIntervalForecaster(forecasters=[("a", _conformal(48)), ("b", _conformal(_SWEEP_CALIBRATION))]),
        _standard(f["y_X"]),
        48,
    ),
    "VotingPointForecaster": lambda f: (
        VotingPointForecaster(forecasters=[("a", _stub(168)), ("b", _stub(0))]),
        _standard(f["y_X"]),
        168,
    ),
}


def _registry_forecasters():
    return sorted(name for name, cls in all_estimators() if issubclass(cls, BaseForecaster))


def _unclassified(names, plain, composite):
    """Names in neither table, and names in both."""
    missing = [name for name in names if name not in plain and name not in composite]
    doubled = [name for name in names if name in plain and name in composite]
    return missing, doubled


class TestRegistrySweep:
    """Every forecaster yohou ships is classified, and declares its expected value."""

    def test_registry_is_collected_unfiltered(self):
        # The filtered call instantiates classes without arguments and drops every composite.
        names = _registry_forecasters()
        assert len(names) >= 20
        assert "ColumnForecaster" in names
        assert "LocalPanelForecaster" in names

    def test_every_forecaster_is_classified_exactly_once(self):
        missing, doubled = _unclassified(_registry_forecasters(), _PLAIN_FORECASTERS, _COMPOSITE_BUILDERS)
        assert not missing, f"Classify these forecasters in _PLAIN_FORECASTERS or _COMPOSITE_BUILDERS: {missing}"
        assert not doubled, f"Forecasters classified as both plain and composite: {doubled}"

    def test_unclassified_forecaster_is_reported(self):
        class _NewCompositeForecaster(SeasonalNaive):
            pass

        names = [*_registry_forecasters(), _NewCompositeForecaster.__name__]
        missing, _ = _unclassified(names, _PLAIN_FORECASTERS, _COMPOSITE_BUILDERS)
        assert missing == ["_NewCompositeForecaster"]

    def test_removed_entry_is_reported(self):
        plain = {k: v for k, v in _PLAIN_FORECASTERS.items() if k != "SeasonalNaive"}
        composite = {k: v for k, v in _COMPOSITE_BUILDERS.items() if k != "ColumnForecaster"}
        missing, _ = _unclassified(_registry_forecasters(), plain, composite)
        assert missing == ["ColumnForecaster", "SeasonalNaive"]

    @pytest.mark.parametrize("name", sorted({**_PLAIN_FORECASTERS, **_COMPOSITE_BUILDERS}))
    def test_declares_expected_value(self, name, y_X_factory, y_X_panel_factory, class_proba_y_X_factory):
        builder = _COMPOSITE_BUILDERS.get(name) or _PLAIN_FORECASTERS.get(name)
        if builder is None:
            pytest.skip(f"{name} is a base class and cannot be built")
        factories = {"y_X": y_X_factory, "panel": y_X_panel_factory, "class_proba": class_proba_y_X_factory}
        forecaster, (y, X_actual), expected = builder(factories)
        # Composites read constructor arguments before fit and fitted children after; both must declare it.
        assert _declared(forecaster) == expected
        fit_kwargs = {"coverage_rates": [0.9]} if _needs_coverage(forecaster) else {}
        forecaster.fit(y, X_actual, forecasting_horizon=FH, **fit_kwargs)
        assert _declared(forecaster) == expected


def _needs_coverage(forecaster):
    tags = forecaster.__sklearn_tags__().forecaster_tags
    return "interval" in (getattr(tags, "forecaster_type", None) or ())
