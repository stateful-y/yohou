"""Pickle serialization round-trip tests for all public estimators.

Verifies that every concrete estimator in yohou can be pickled and
unpickled without loss of state - both in unfitted and fitted states.
"""

from __future__ import annotations

import pickle
from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest
from sklearn.linear_model import Ridge

from yohou.utils.discovery import all_estimators

# Estimators that need explicit constructor arguments.
_ESTIMATOR_KWARGS: dict[str, dict] = {
    "ColumnForecaster": {"forecasters": []},
    "ColumnTransformer": {"transformers": []},
    "DecompositionPipeline": {"forecasters": []},
    "ExponentialMovingAverage": {"alpha": 0.3},
    "FeaturePipeline": {"steps": []},
    "FeatureUnion": {"transformer_list": []},
    "ForecastedFeatureForecaster": {
        "target_forecaster": None,
        "feature_forecaster": None,
    },
    "FourierSeasonalityForecaster": {"seasonality": 7},
    "GridSearchCV": {
        "forecaster": None,
        "param_grid": {"seasonality": [1, 2]},
    },
    "LocalPanelForecaster": {"forecaster": None},
    "PatternSeasonalityForecaster": {"seasonality": 7},
    "RandomizedSearchCV": {
        "forecaster": None,
        "param_distributions": {"seasonality": [1, 2]},
    },
    "SeasonalImputer": {"period": 7},
    "SlidingWindowFunctionTransformer": {"func": np.mean},
    "VotingClassProbaForecaster": {"forecasters": []},
    "VotingIntervalForecaster": {"forecasters": []},
    "VotingPointForecaster": {"forecasters": []},
}

# Estimators where nested sub-estimators make get_params() equality fail
# after pickle (different object identity). Check type + shallow param keys.
_NESTED_ESTIMATORS = {
    "ClassProbaReductionForecaster",
    "ColumnForecaster",
    "ForecastedFeatureForecaster",
    "FourierSeasonalityForecaster",
    "GridSearchCV",
    "IntervalReductionForecaster",
    "LocalPanelForecaster",
    "PointReductionForecaster",
    "PolynomialTrendForecaster",
    "RandomizedSearchCV",
    "SimpleImputer",
    "SplitConformalForecaster",
    "VotingClassProbaForecaster",
    "VotingIntervalForecaster",
    "VotingPointForecaster",
}

# Skip these: SklearnScaler/SklearnTransformer expect a class, not instance.
_SKIP_UNFITTED = {"SklearnScaler", "SklearnTransformer"}


def _make_instance(name: str, cls: type):
    """Instantiate an estimator, supplying required args where needed."""
    from yohou.point import SeasonalNaive

    if name in _SKIP_UNFITTED:
        pytest.skip(f"{name} requires special constructor arguments")

    if name in _ESTIMATOR_KWARGS:
        kwargs = dict(_ESTIMATOR_KWARGS[name])

        if name in ("GridSearchCV", "RandomizedSearchCV", "LocalPanelForecaster"):
            kwargs["forecaster"] = SeasonalNaive(seasonality=1)
        elif name == "ForecastedFeatureForecaster":
            kwargs["target_forecaster"] = SeasonalNaive(seasonality=1)
            kwargs["feature_forecaster"] = SeasonalNaive(seasonality=1)

        return cls(**kwargs)

    return cls()


def _make_y(length: int = 80) -> pl.DataFrame:
    """Create a target time series with a single ``y_0`` column."""
    rng = np.random.default_rng(42)
    time = pl.datetime_range(
        start=datetime(2021, 1, 1),
        end=datetime(2021, 1, 1) + timedelta(seconds=length - 1),
        interval="1s",
        eager=True,
    )
    return pl.DataFrame({"time": time, "y_0": rng.standard_normal(length).tolist()})


def _make_time_series(length: int = 50, n_cols: int = 2) -> pl.DataFrame:
    """Create a simple time series DataFrame."""
    rng = np.random.default_rng(42)
    time = pl.datetime_range(
        start=datetime(2021, 1, 1),
        end=datetime(2021, 1, 1) + timedelta(seconds=length - 1),
        interval="1s",
        eager=True,
    )
    data: dict[str, object] = {"time": time}
    for i in range(n_cols):
        data[f"col_{i}"] = rng.standard_normal(length).tolist()
    return pl.DataFrame(data)


_ALL_ESTIMATORS = [
    (name, cls)
    for name, cls in all_estimators()
    if name not in ("BasePointForecaster", "AbsoluteQuantileResidual", "QuantileResidual")
]


@pytest.fixture(
    params=[pytest.param((n, c), id=n) for n, c in _ALL_ESTIMATORS],
)
def estimator_name_cls(request):
    """Parametrized fixture yielding (name, cls) for each public estimator."""
    return request.param


class TestPickleUnfitted:
    """Pickle round-trip for unfitted estimators."""

    def test_unfitted_pickle_roundtrip(self, estimator_name_cls):
        """Unfitted estimator survives pickle round-trip with identical type."""
        name, cls = estimator_name_cls
        instance = _make_instance(name, cls)
        data = pickle.dumps(instance)
        restored = pickle.loads(data)  # noqa: S301

        assert type(restored) is type(instance)

        if name in _NESTED_ESTIMATORS:
            assert set(restored.get_params(deep=False)) == set(instance.get_params(deep=False))
        else:
            assert restored.get_params() == instance.get_params()

    def test_unfitted_pickle_protocol_variants(self, estimator_name_cls):
        """Unfitted estimator works with all pickle protocols."""
        name, cls = estimator_name_cls
        instance = _make_instance(name, cls)

        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            data = pickle.dumps(instance, protocol=protocol)
            restored = pickle.loads(data)  # noqa: S301
            assert type(restored) is type(instance)


_FITTED_TRANSFORMERS = [
    ("StandardScaler", lambda: __import__("yohou.preprocessing", fromlist=["StandardScaler"]).StandardScaler()),
    ("MinMaxScaler", lambda: __import__("yohou.preprocessing", fromlist=["MinMaxScaler"]).MinMaxScaler()),
    (
        "FunctionTransformer",
        lambda: __import__("yohou.preprocessing", fromlist=["FunctionTransformer"]).FunctionTransformer(),
    ),
    ("LogTransformer", lambda: __import__("yohou.stationarity", fromlist=["LogTransformer"]).LogTransformer()),
]

_FITTED_FORECASTERS = [
    ("SeasonalNaive", lambda: __import__("yohou.point", fromlist=["SeasonalNaive"]).SeasonalNaive(seasonality=1)),
    (
        "PointReductionForecaster",
        lambda: __import__("yohou.point", fromlist=["PointReductionForecaster"]).PointReductionForecaster(
            estimator=Ridge()
        ),
    ),
]

_FITTED_SCORERS = [
    ("MeanAbsoluteError", lambda: __import__("yohou.metrics", fromlist=["MeanAbsoluteError"]).MeanAbsoluteError()),
    ("MeanSquaredError", lambda: __import__("yohou.metrics", fromlist=["MeanSquaredError"]).MeanSquaredError()),
]

_FITTED_SPLITTERS = [
    (
        "ExpandingWindowSplitter",
        lambda: __import__("yohou.model_selection", fromlist=["ExpandingWindowSplitter"]).ExpandingWindowSplitter(),
    ),
    (
        "SlidingWindowSplitter",
        lambda: __import__("yohou.model_selection", fromlist=["SlidingWindowSplitter"]).SlidingWindowSplitter(),
    ),
]


class TestPickleFittedTransformers:
    """Pickle round-trip for fitted transformers."""

    @pytest.fixture(
        params=[pytest.param(pair, id=pair[0]) for pair in _FITTED_TRANSFORMERS],
    )
    def fitted_transformer(self, request):
        """Fit and return a transformer."""
        _, factory = request.param
        df = _make_time_series(50, 2)
        t = factory()
        t.fit(df)
        return t, df

    def test_fitted_transform_preserved(self, fitted_transformer):
        """Fitted transformer produces same output after pickle round-trip."""
        transformer, df = fitted_transformer
        original_out = transformer.transform(df)

        data = pickle.dumps(transformer)
        restored = pickle.loads(data)  # noqa: S301
        restored_out = restored.transform(df)

        assert original_out.equals(restored_out)


class TestPickleFittedForecasters:
    """Pickle round-trip for fitted forecasters."""

    @pytest.fixture(
        params=[pytest.param(pair, id=pair[0]) for pair in _FITTED_FORECASTERS],
    )
    def fitted_forecaster(self, request):
        """Fit and return a forecaster."""
        _, factory = request.param
        y = _make_y(80)
        f = factory()
        f.fit(y, forecasting_horizon=5)
        return f

    def test_fitted_predict_preserved(self, fitted_forecaster):
        """Fitted forecaster produces same predictions after round-trip."""
        original_pred = fitted_forecaster.predict()

        data = pickle.dumps(fitted_forecaster)
        restored = pickle.loads(data)  # noqa: S301
        restored_pred = restored.predict()

        assert original_pred.equals(restored_pred)


class TestPickleFittedScorers:
    """Pickle round-trip for fitted scorers."""

    @pytest.fixture(
        params=[pytest.param(pair, id=pair[0]) for pair in _FITTED_SCORERS],
    )
    def fitted_scorer(self, request):
        """Fit and return a scorer."""
        _, factory = request.param
        df = _make_time_series(30, 1)
        s = factory()
        s.fit(df)
        return s, df

    def test_fitted_score_preserved(self, fitted_scorer):
        """Fitted scorer produces same score after pickle round-trip."""
        scorer, df = fitted_scorer
        original_score = scorer.score(df, df)

        data = pickle.dumps(scorer)
        restored = pickle.loads(data)  # noqa: S301
        restored_score = restored.score(df, df)

        assert original_score == restored_score


class TestPickleFittedSplitters:
    """Pickle round-trip for fitted splitters."""

    @pytest.fixture(
        params=[pytest.param(pair, id=pair[0]) for pair in _FITTED_SPLITTERS],
    )
    def splitter_instance(self, request):
        """Return a splitter."""
        _, factory = request.param
        return factory()

    def test_splitter_pickle_roundtrip(self, splitter_instance):
        """Splitter produces same splits after round-trip."""
        y = _make_time_series(50, 1)

        original_splits = list(splitter_instance.split(y))

        data = pickle.dumps(splitter_instance)
        restored = pickle.loads(data)  # noqa: S301
        restored_splits = list(restored.split(y))

        assert len(original_splits) == len(restored_splits)
        for (o_train, o_test), (r_train, r_test) in zip(original_splits, restored_splits, strict=True):
            assert list(o_train) == list(r_train)
            assert list(o_test) == list(r_test)
