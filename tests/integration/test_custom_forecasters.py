"""Integration tests for custom forecaster extensibility.

Verifies that every documented Tier 1 forecaster extension pattern
(point, interval, class-proba) works end to end with systematic check
generators and manual assertions. Subsumes ``test_tier1_fit_hook.py``.
"""

from __future__ import annotations

import numbers
from typing import Literal

import numpy as np
import polars as pl
import polars.selectors as cs
import pytest
from sklearn.base import clone
from sklearn.utils._param_validation import Interval
from sklearn.utils.validation import check_is_fitted

from conftest import run_checks
from yohou.class_proba.base import BaseClassProbaForecaster
from yohou.interval.base import BaseIntervalForecaster
from yohou.metrics import MeanAbsoluteError
from yohou.model_selection import GridSearchCV, SlidingWindowSplitter
from yohou.point.base import BasePointForecaster
from yohou.testing import _yield_yohou_forecaster_checks
from yohou.utils.tags import Tags


class _LastValueForecaster(BasePointForecaster):
    """Repeats the last observed value. Minimal Tier 1 pattern from docs."""

    _tags = {"ignores_exogenous": True, "stateful": True}

    def __init__(self):
        super().__init__(
            feature_transformer=None,
            target_transformer=None,
            target_as_feature=None,
        )

    def __sklearn_tags__(self) -> Tags:
        tags = super().__sklearn_tags__()
        assert tags.forecaster_tags is not None
        tags.forecaster_tags.requires_exogenous = False
        return tags

    @property
    def _observation_horizon(self):
        return 1

    def _predict_one(self, groups, **params):
        cols = [c for c in self.local_y_schema_ if c != "time"]
        h = self.fit_forecasting_horizon_

        if self.groups_ is None:
            last_value = self._y_observed.select(~cs.by_name("time")).row(-1)
            y_pred = pl.DataFrame({col: [float(last_value[i])] * h for i, col in enumerate(cols)})
        else:
            parts = []
            for group in groups:
                last_value = self._y_observed[group].select(~cs.by_name("time")).row(-1)
                part = pl.DataFrame({f"{group}__{col}": [float(last_value[i])] * h for i, col in enumerate(cols)})
                parts.append(part)
            y_pred = pl.concat(parts, how="horizontal")

        return self._add_time_columns(y_pred)


class _WindowMeanForecaster(BasePointForecaster):
    """Predicts the mean of recent values. Tier 1 with params, constraints, panel."""

    _parameter_constraints: dict = {
        **BasePointForecaster._parameter_constraints,
        "window": [Interval(numbers.Integral, 1, None, closed="left")],
    }

    def __init__(
        self,
        window: int = 5,
        panel_strategy: Literal["global", "multivariate"] = "global",
    ):
        super().__init__(
            feature_transformer=None,
            target_transformer=None,
            target_as_feature=None,
            panel_strategy=panel_strategy,
        )
        self.window = window

    def __sklearn_tags__(self) -> Tags:
        tags = super().__sklearn_tags__()
        assert tags.forecaster_tags is not None
        tags.forecaster_tags.requires_exogenous = False
        tags.forecaster_tags.stateful = True
        return tags

    @property
    def _observation_horizon(self) -> int:
        return self.window

    def _fit(self, y_t, X_t, forecasting_horizon):
        if isinstance(y_t, dict):
            self.means_ = {}
            for group, df in y_t.items():
                vals = df.select(pl.exclude("time")).to_numpy()
                self.means_[group] = np.mean(vals[-self.window :], axis=0)
        else:
            vals = y_t.select(pl.exclude("time")).to_numpy()
            self.mean_ = np.mean(vals[-self.window :], axis=0)

    def _predict_one(self, groups, **params):
        cols = [c for c in self.local_y_schema_ if c != "time"]
        h = self.fit_forecasting_horizon_

        if self.groups_ is None:
            y_pred = pl.DataFrame({col: [float(self.mean_[i])] * h for i, col in enumerate(cols)})
        else:
            parts = []
            for group in groups:
                part = pl.DataFrame({
                    f"{group}__{col}": [float(self.means_[group][i])] * h for i, col in enumerate(cols)
                })
                parts.append(part)
            y_pred = pl.concat(parts, how="horizontal")

        return self._add_time_columns(y_pred)


class _ConstantIntervalForecaster(BaseIntervalForecaster):
    """Returns fixed-width intervals around zero. Minimal interval Tier 1."""

    _tags = {"ignores_exogenous": True, "stateful": True}

    _parameter_constraints: dict = {
        **BaseIntervalForecaster._parameter_constraints,
        "half_width": [Interval(numbers.Real, 0, None, closed="neither")],
    }

    def __init__(self, half_width: float = 1.0):
        super().__init__(
            feature_transformer=None,
            target_as_feature=None,
        )
        self.half_width = half_width

    def __sklearn_tags__(self) -> Tags:
        tags = super().__sklearn_tags__()
        assert tags.forecaster_tags is not None
        tags.forecaster_tags.requires_exogenous = False
        return tags

    @property
    def _observation_horizon(self):
        return 1

    def _fit(self, y_t, X_t, forecasting_horizon):
        if isinstance(y_t, dict):
            self.centers_ = {}
            for group, df in y_t.items():
                self.centers_[group] = df.select(pl.exclude("time")).row(-1)
        else:
            self.center_ = y_t.select(pl.exclude("time")).row(-1)

    def _predict_one(self, groups, coverage_rates=None, **params):
        if coverage_rates is None:
            coverage_rates = self.fit_coverage_rates_

        h = self.fit_forecasting_horizon_
        cols = [c for c in self.local_y_schema_ if c != "time"]

        if self.groups_ is None:
            data: dict[str, list[float]] = {}
            for rate in coverage_rates:
                width = self.half_width * rate
                for i, col in enumerate(cols):
                    center = float(self.center_[i])
                    data[f"{col}_lower_{rate}"] = [center - width] * h
                    data[f"{col}_upper_{rate}"] = [center + width] * h
            y_pred = pl.DataFrame(data)
        else:
            parts = []
            for group in groups:
                data = {}
                for rate in coverage_rates:
                    width = self.half_width * rate
                    for i, col in enumerate(cols):
                        center = float(self.centers_[group][i])
                        data[f"{col}_lower_{rate}"] = [center - width] * h
                        data[f"{col}_upper_{rate}"] = [center + width] * h
                grp = pl.DataFrame(data)
                grp = grp.rename({c: f"{group}__{c}" for c in grp.columns})
                parts.append(grp)
            y_pred = pl.concat(parts, how="horizontal")

        return self._add_time_columns(y_pred)


class _UniformClassProbaForecaster(BaseClassProbaForecaster):
    """Returns uniform class probabilities. Minimal class-proba Tier 1."""

    _tags = {"ignores_exogenous": True, "stateful": True}

    def __init__(self):
        super().__init__(
            feature_transformer=None,
            target_transformer=None,
            target_as_feature=None,
        )

    def __sklearn_tags__(self) -> Tags:
        tags = super().__sklearn_tags__()
        assert tags.forecaster_tags is not None
        tags.forecaster_tags.requires_exogenous = False
        return tags

    @property
    def _observation_horizon(self):
        return 1

    def _fit(self, y_t, X_t, forecasting_horizon):
        # Extract classes from training data
        self.classes_ = {}
        self.n_classes_ = {}
        self.label_to_code_ = {}
        if isinstance(y_t, dict):
            for _group, df in y_t.items():
                for col in [c for c in df.columns if c != "time"]:
                    base_col = col.split("__")[-1] if "__" in col else col
                    if base_col not in self.classes_:
                        unique_vals = sorted(df[col].unique().to_list())
                        self.classes_[base_col] = [str(v) for v in unique_vals]
                        self.n_classes_[base_col] = len(unique_vals)
                        self.label_to_code_[base_col] = {str(v): i for i, v in enumerate(unique_vals)}
        else:
            for col in [c for c in y_t.columns if c != "time"]:
                unique_vals = sorted(y_t[col].unique().to_list())
                self.classes_[col] = [str(v) for v in unique_vals]
                self.n_classes_[col] = len(unique_vals)
                self.label_to_code_[col] = {str(v): i for i, v in enumerate(unique_vals)}

    def _predict_class_proba_one(self, groups, **params):
        h = self.fit_forecasting_horizon_

        if self.groups_ is None:
            data: dict[str, list[float]] = {}
            for target_col, class_labels in self.classes_.items():
                n = len(class_labels)
                prob = 1.0 / n
                for label in class_labels:
                    data[f"{target_col}_proba_{label}"] = [prob] * h
            y_pred = pl.DataFrame(data)
        else:
            parts = []
            for group in groups:
                data = {}
                for target_col, class_labels in self.classes_.items():
                    n = len(class_labels)
                    prob = 1.0 / n
                    for label in class_labels:
                        data[f"{group}__{target_col}_proba_{label}"] = [prob] * h
                parts.append(pl.DataFrame(data))
            y_pred = pl.concat(parts, how="horizontal")

        return self._add_time_columns(y_pred)


# Fixtures


@pytest.fixture
def daily_series():
    rng = np.random.default_rng(42)
    n = 100
    return pl.DataFrame({
        "time": pl.datetime_range(
            pl.lit("2020-01-01").str.to_datetime(),
            pl.lit("2020-01-01").str.to_datetime() + pl.duration(days=n - 1),
            interval="1d",
            eager=True,
        ),
        "value": rng.normal(10, 1, n).tolist(),
    })


@pytest.fixture
def categorical_series():
    """Categorical time series with 3 classes for class-proba testing."""
    n = 100
    rng = np.random.default_rng(42)
    classes = ["low", "mid", "high"]
    return pl.DataFrame({
        "time": pl.datetime_range(
            pl.lit("2020-01-01").str.to_datetime(),
            pl.lit("2020-01-01").str.to_datetime() + pl.duration(days=n - 1),
            interval="1d",
            eager=True,
        ),
        "category": [classes[i] for i in rng.integers(0, 3, n)],
    })


@pytest.mark.integration
class TestLastValueForecaster:
    """Verify minimal Tier 1 point forecaster extension."""

    def test_systematic_checks(self, daily_series):
        y_train, y_test = daily_series[:80], daily_series[80:]
        forecaster = _LastValueForecaster()
        forecaster.fit(y_train, forecasting_horizon=len(y_test))

        run_checks(
            forecaster,
            _yield_yohou_forecaster_checks(forecaster, y_train, None, y_test, None),
        )

    def test_fit_sets_attributes(self, daily_series):
        f = _LastValueForecaster()
        f.fit(daily_series[:80], forecasting_horizon=5)
        assert f.fit_forecasting_horizon_ == 5
        check_is_fitted(f)

    def test_predict_output_shape(self, daily_series):
        f = _LastValueForecaster()
        f.fit(daily_series[:80], forecasting_horizon=10)
        y_pred = f.predict()
        # vintage_time + time + value columns
        assert "vintage_time" in y_pred.columns
        assert "time" in y_pred.columns
        assert len(y_pred) == 10


@pytest.mark.integration
class TestWindowMeanForecaster:
    """Verify Tier 1 point forecaster with params, constraints, panel.

    Subsumes all tests from TestTier1FitHook.
    """

    def test_systematic_checks(self, daily_series):
        y_train, y_test = daily_series[:80], daily_series[80:]
        forecaster = _WindowMeanForecaster(window=7)
        forecaster.fit(y_train, forecasting_horizon=len(y_test))

        run_checks(
            forecaster,
            _yield_yohou_forecaster_checks(forecaster, y_train, None, y_test, None),
        )

    def test_fit_sets_attributes(self, daily_series):
        m = _WindowMeanForecaster(window=7)
        m.fit(daily_series[:80], forecasting_horizon=5)
        assert hasattr(m, "mean_")
        assert hasattr(m, "fit_forecasting_horizon_")
        assert m.fit_forecasting_horizon_ == 5

    def test_observation_horizon_from_property(self):
        m = _WindowMeanForecaster(window=10)
        assert m.observation_horizon == 10

    def test_observation_horizon_changes_with_param(self):
        m = _WindowMeanForecaster(window=3)
        assert m.observation_horizon == 3
        m.set_params(window=20)
        assert m.observation_horizon == 20

    def test_fit_and_check_is_fitted(self, daily_series):
        m = _WindowMeanForecaster(window=5)
        m.fit(daily_series[:80], forecasting_horizon=3)
        check_is_fitted(m)

    def test_clone_preserves_params(self):
        m = _WindowMeanForecaster(window=12)
        m2 = clone(m)
        assert m2.window == 12
        assert m2.observation_horizon == 12

    def test_systematic_checks_panel(self, y_X_factory):
        y, _ = y_X_factory(length=100, n_targets=1, n_features=0, panel=True, n_groups=2)
        y_train, y_test = y[:80], y[80:]
        forecaster = _WindowMeanForecaster(window=5, panel_strategy="global")
        forecaster.fit(y_train, forecasting_horizon=len(y_test))

        run_checks(
            forecaster,
            _yield_yohou_forecaster_checks(forecaster, y_train, None, y_test, None),
        )


@pytest.mark.integration
class TestConstantIntervalForecaster:
    """Verify Tier 1 interval forecaster extension."""

    def test_systematic_checks(self, daily_series):
        y_train, y_test = daily_series[:80], daily_series[80:]
        forecaster = _ConstantIntervalForecaster(half_width=2.0)
        forecaster.fit(y_train, forecasting_horizon=len(y_test), coverage_rates=[0.9])

        run_checks(
            forecaster,
            _yield_yohou_forecaster_checks(forecaster, y_train, None, y_test, None),
            expected_failures={"check_fit_predict_without_exogenous"},
        )

    def test_bounds_columns_present(self, daily_series):
        f = _ConstantIntervalForecaster(half_width=2.0)
        f.fit(daily_series[:80], forecasting_horizon=5, coverage_rates=[0.9])
        y_pred = f.predict_interval()
        assert any("_lower_" in c for c in y_pred.columns)
        assert any("_upper_" in c for c in y_pred.columns)

    def test_bounds_order(self, daily_series):
        f = _ConstantIntervalForecaster(half_width=2.0)
        f.fit(daily_series[:80], forecasting_horizon=5, coverage_rates=[0.9])
        y_pred = f.predict_interval()
        lower_col = [c for c in y_pred.columns if "_lower_" in c][0]
        upper_col = [c for c in y_pred.columns if "_upper_" in c][0]
        assert (y_pred[lower_col] <= y_pred[upper_col]).all()


@pytest.mark.integration
class TestUniformClassProbaForecaster:
    """Verify Tier 1 class-proba forecaster extension."""

    def test_systematic_checks(self, categorical_series):
        y_train, y_test = categorical_series[:80], categorical_series[80:]
        forecaster = _UniformClassProbaForecaster()
        forecaster.fit(y_train, forecasting_horizon=len(y_test))

        run_checks(
            forecaster,
            _yield_yohou_forecaster_checks(forecaster, y_train, None, y_test, None),
            expected_failures={
                "check_observe_extends_observations",
                "check_rewind_replaces_observations",
            },
        )

    def test_probabilities_sum_to_one(self, categorical_series):
        f = _UniformClassProbaForecaster()
        f.fit(categorical_series[:80], forecasting_horizon=5)
        y_pred = f.predict_class_proba()
        proba_cols = [c for c in y_pred.columns if "_proba_" in c]
        row_sums = y_pred.select(proba_cols).sum_horizontal()
        assert all(abs(s - 1.0) < 1e-10 for s in row_sums.to_list())

    def test_classes_attribute_set(self, categorical_series):
        f = _UniformClassProbaForecaster()
        f.fit(categorical_series[:80], forecasting_horizon=3)
        assert hasattr(f, "classes_")
        assert "category" in f.classes_
        assert len(f.classes_["category"]) == 3


@pytest.mark.integration
class TestCustomForecasterInSearch:
    """Verify custom forecaster works inside GridSearchCV."""

    def test_window_mean_in_grid_search(self, daily_series):
        scorer = MeanAbsoluteError()
        cv = SlidingWindowSplitter(n_splits=2, test_size=10)
        search = GridSearchCV(
            _WindowMeanForecaster(),
            param_grid={"window": [3, 7]},
            scoring=scorer,
            cv=cv,
        )
        search.fit(daily_series, forecasting_horizon=5)

        assert hasattr(search, "best_params_")
        assert search.best_params_["window"] in (3, 7)
        assert "mean_test_score" in search.cv_results_
        assert len(search.cv_results_["params"]) == 2
