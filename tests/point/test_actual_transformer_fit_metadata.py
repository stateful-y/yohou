"""Reduction forecasters route fit metadata to the actual transformer that requests it."""

import pytest
from sklearn.linear_model import LinearRegression

from yohou.compose import FeatureUnion
from yohou.point import PointReductionForecaster
from yohou.preprocessing import LagTransformer


class _RecordingLag(LagTransformer):
    """LagTransformer that requests fit metadata and records what it receives."""

    __metadata_request__fit = {"forecasting_horizon": True, "marker": True}

    def fit(self, X, y=None, **params):
        """Record the fit metadata, then fit as a plain lag transformer."""
        self.seen_ = dict(params)
        return super().fit(X, y)


class _RecordingRegressor(LinearRegression):
    """LinearRegression that accepts and records a ``marker`` fit parameter."""

    def fit(self, X, y, sample_weight=None, marker=None):
        """Record ``marker``, then fit."""
        self.marker_ = marker
        return super().fit(X, y, sample_weight=sample_weight)


def _fitted_probes(forecaster):
    """Every fitted recording transformer, across panel groups when present."""
    fitted = forecaster.actual_transformer_
    unions = fitted.values() if isinstance(fitted, dict) else [fitted]
    return [union.transformer_list[0][1] for union in unions]


def test_horizon_reaches_nested_transformer_standard(y_X_factory):
    """The fit horizon reaches a transformer nested in a FeatureUnion."""
    y, X = y_X_factory(length=80, n_targets=1, n_features=2)
    forecaster = PointReductionForecaster(
        estimator=LinearRegression(),
        actual_transformer=FeatureUnion([("probe", _RecordingLag(lag=1))]),
    )
    forecaster.fit(y, X, forecasting_horizon=4)

    (probe,) = _fitted_probes(forecaster)
    assert probe.seen_ == {"forecasting_horizon": 4}


@pytest.mark.parametrize("panel_strategy", ["global", "multivariate"])
def test_horizon_reaches_nested_transformer_panel(y_X_panel_factory, panel_strategy):
    """The fit horizon reaches every group's transformer under both panel strategies."""
    y, X = y_X_panel_factory(n_groups=2, length=80, n_targets=1, n_features=1)
    forecaster = PointReductionForecaster(
        estimator=LinearRegression(),
        actual_transformer=FeatureUnion([("probe", _RecordingLag(lag=1))]),
        panel_strategy=panel_strategy,
    )
    forecaster.fit(y, X, forecasting_horizon=5)

    probes = _fitted_probes(forecaster)
    assert len(probes) == (2 if panel_strategy == "global" else 1)
    assert all(probe.seen_ == {"forecasting_horizon": 5} for probe in probes)


def test_caller_metadata_reaches_requesting_transformer_and_estimator(y_X_factory):
    """Caller metadata the transformer requests reaches it, and still reaches the estimator."""
    y, X = y_X_factory(length=80, n_targets=1, n_features=2)
    forecaster = PointReductionForecaster(
        estimator=_RecordingRegressor().set_fit_request(marker=True),
        actual_transformer=FeatureUnion([("probe", _RecordingLag(lag=1))]),
    )
    forecaster.fit(y, X, forecasting_horizon=3, marker="x")

    (probe,) = _fitted_probes(forecaster)
    assert probe.seen_ == {"forecasting_horizon": 3, "marker": "x"}
    assert forecaster.estimator_.marker_ == "x"


def test_transformers_without_requests_get_no_metadata(y_X_factory):
    """A union that requests nothing is fitted with a bare call and does not raise."""
    y, X = y_X_factory(length=80, n_targets=1, n_features=2)
    forecaster = PointReductionForecaster(
        estimator=LinearRegression(),
        actual_transformer=FeatureUnion([("lag", LagTransformer(lag=1))]),
    )
    forecaster.fit(y, X, forecasting_horizon=3)
    assert forecaster.predict(forecasting_horizon=3).height == 3
