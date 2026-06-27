"""Targeted tests for patch coverage gaps in PR #34.

Covers uncovered branches in:
- base/forecaster.py: _tags merging into input_tags/target_tags/top-level
- base/forecaster.py: observation_horizon pre-fit and with dict transformers
- model_selection/split.py: _tags merging into input_tags/top-level
- interval/base.py: BaseIntervalForecaster concrete fit()
- class_proba/base.py: BaseClassProbaForecaster concrete fit()
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import polars as pl

from yohou.interval.base import BaseIntervalForecaster
from yohou.point.base import BasePointForecaster
from yohou.utils.tags import Tags


class _TagMergingForecaster(BasePointForecaster):
    """Forecaster with _tags keys targeting input_tags and top-level tags."""

    _tags = {
        "stateful": True,
        "allow_nan": True,
        "non_deterministic": True,
    }

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
        return 0

    def _fit(self, y_t, X_t, forecasting_horizon):
        pass

    def _predict_one(self, groups, **params):
        cols = [c for c in self.local_y_schema_ if c != "time"]
        h = self.fit_forecasting_horizon_
        data = {c: [0.0] * h for c in cols}
        return pl.DataFrame(data)


class TestForecasterTagMerging:
    """__sklearn_tags__ merges _tags into input_tags and top-level Tags."""

    def test_tags_merged_into_input_tags(self):
        f = _TagMergingForecaster()
        tags = f.__sklearn_tags__()
        assert tags.input_tags is not None
        assert tags.input_tags.allow_nan is True

    def test_tags_merged_into_top_level(self):
        f = _TagMergingForecaster()
        tags = f.__sklearn_tags__()
        assert tags.non_deterministic is True

    def test_tags_merged_into_forecaster_tags(self):
        f = _TagMergingForecaster()
        tags = f.__sklearn_tags__()
        assert tags.forecaster_tags is not None
        assert tags.forecaster_tags.stateful is True

    def test_tags_merged_into_target_tags(self):
        """A _tags key only present on target_tags goes through that branch."""

        class _TargetTagForecaster(BasePointForecaster):
            _tags = {"required": True}

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
                return 0

            def _fit(self, y_t, X_t, forecasting_horizon):
                pass

            def _predict_one(self, groups, **params):
                cols = [c for c in self.local_y_schema_ if c != "time"]
                h = self.fit_forecasting_horizon_
                return pl.DataFrame({c: [0.0] * h for c in cols})

        f = _TargetTagForecaster()
        tags = f.__sklearn_tags__()
        assert tags.target_tags is not None
        assert tags.target_tags.required is True


class TestObservationHorizonPreFit:
    """observation_horizon is accessible before fit() is called."""

    def test_pre_fit_returns_self_observation_horizon(self):
        f = _TagMergingForecaster()
        assert f.observation_horizon == 0

    def test_observation_horizon_setter(self):
        """The base class _observation_horizon property supports assignment."""

        class _SetterForecaster(BasePointForecaster):
            _tags = {"stateful": True}

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

            def _fit(self, y_t, X_t, forecasting_horizon):
                self._observation_horizon = 5

            def _predict_one(self, groups, **params):
                cols = [c for c in self.local_y_schema_ if c != "time"]
                h = self.fit_forecasting_horizon_
                return pl.DataFrame({c: [0.0] * h for c in cols})

        f = _SetterForecaster()
        assert f._observation_horizon == 0
        f._observation_horizon = 5
        assert f._observation_horizon == 5
        assert f.observation_horizon == 5


class TestSplitterTagMerging:
    """ExpandingWindowSplitter _tags merge splitter_type via _tags dict."""

    def test_expanding_splitter_type_from_tags(self):
        from yohou.model_selection.split import ExpandingWindowSplitter

        s = ExpandingWindowSplitter(n_splits=3)
        tags = s.__sklearn_tags__()
        assert tags.splitter_tags is not None
        assert tags.splitter_tags.splitter_type == "expanding"

    def test_custom_splitter_input_tag_merging(self):
        from yohou.model_selection.split import BaseSplitter

        class _CustomSplitter(BaseSplitter):
            _tags = {"allow_nan": True, "non_deterministic": True}

            def split(self, y, X_actual=None):
                yield np.arange(5), np.arange(5, 10)

            def _iter_test_indices(self, y, X_actual=None):
                yield np.arange(5, 10)

            def get_n_splits(self, y=None, X_actual=None):
                return 1

        s = _CustomSplitter()
        tags = s.__sklearn_tags__()
        assert tags.input_tags is not None
        assert tags.input_tags.allow_nan is True
        assert tags.non_deterministic is True


class _MinimalIntervalForecaster(BaseIntervalForecaster):
    """Minimal interval forecaster exercising the concrete fit() path."""

    _tags = {"ignores_exogenous": True}

    def __init__(self):
        super().__init__(
            feature_transformer=None,
            target_as_feature=None,
        )

    def __sklearn_tags__(self) -> Tags:
        tags = super().__sklearn_tags__()
        assert tags.forecaster_tags is not None
        tags.forecaster_tags.requires_exogenous = False
        return tags

    @property
    def _observation_horizon(self):
        return 0

    def _fit(self, y_t, X_t, forecasting_horizon):
        cols = [c for c in self.local_y_schema_ if c != "time"]
        self.last_values_ = dict.fromkeys(cols, 0.0)

    def _predict_one(self, groups, coverage_rates=None, **params):
        if coverage_rates is None:
            coverage_rates = self.fit_coverage_rates_
        h = self.fit_forecasting_horizon_
        cols = [c for c in self.local_y_schema_ if c != "time"]
        data: dict[str, list[float]] = {}
        for rate in coverage_rates:
            for c in cols:
                data[f"{c}_lower_{rate}"] = [0.0] * h
                data[f"{c}_upper_{rate}"] = [1.0] * h
        return pl.DataFrame(data)


class TestIntervalForecasterConcreteFit:
    """BaseIntervalForecaster.fit() calls _pre_fit and _fit."""

    def test_fit_sets_coverage_rates(self):
        f = _MinimalIntervalForecaster()
        y = pl.DataFrame({
            "time": pl.datetime_range(
                start=datetime(2020, 1, 1),
                end=datetime(2020, 1, 1) + timedelta(days=29),
                interval="1d",
                eager=True,
            ),
            "value": np.random.default_rng(42).standard_normal(30),
        })
        f.fit(y, forecasting_horizon=3, coverage_rates=[0.9])
        assert f.fit_coverage_rates_ == [0.9]
        assert f.fit_forecasting_horizon_ == 3

    def test_fit_default_coverage_rates(self):
        f = _MinimalIntervalForecaster()
        y = pl.DataFrame({
            "time": pl.datetime_range(
                start=datetime(2020, 1, 1),
                end=datetime(2020, 1, 1) + timedelta(days=29),
                interval="1d",
                eager=True,
            ),
            "value": np.random.default_rng(42).standard_normal(30),
        })
        f.fit(y, forecasting_horizon=1)
        assert f.fit_coverage_rates_ == [0.95]

    def test_predict_interval_after_fit(self):
        f = _MinimalIntervalForecaster()
        y = pl.DataFrame({
            "time": pl.datetime_range(
                start=datetime(2020, 1, 1),
                end=datetime(2020, 1, 1) + timedelta(days=29),
                interval="1d",
                eager=True,
            ),
            "value": np.random.default_rng(42).standard_normal(30),
        })
        f.fit(y, forecasting_horizon=2, coverage_rates=[0.9])
        assert hasattr(f, "last_values_")
        assert hasattr(f, "fit_forecasting_horizon_")
        assert hasattr(f, "interval_")


class TestClassProbaForecasterConcreteFit:
    """BaseClassProbaForecaster.fit() calls _pre_fit and _fit."""

    def test_fit_calls_pre_fit_and_fit(self):
        from sklearn.linear_model import LogisticRegression

        from yohou.class_proba.reduction import ClassProbaReductionForecaster

        rng = np.random.default_rng(42)
        n = 50
        y = pl.DataFrame({
            "time": pl.datetime_range(
                start=datetime(2020, 1, 1),
                end=datetime(2020, 1, 1) + timedelta(days=n - 1),
                interval="1d",
                eager=True,
            ),
            "category": rng.choice(["A", "B", "C"], size=n).tolist(),
        })

        f = ClassProbaReductionForecaster(estimator=LogisticRegression())
        f.fit(y, forecasting_horizon=1)
        assert hasattr(f, "fit_forecasting_horizon_")
        result = f.predict_class_proba()
        assert "time" in result.columns


class TestValidateFitParams:
    """_validate_fit_params raises when forecasting_horizon < 1."""

    def test_forecasting_horizon_zero_raises(self):
        import pytest

        f = _TagMergingForecaster()
        with pytest.raises(ValueError, match="forecasting_horizon must be >= 1"):
            f._validate_fit_params(0)

    def test_forecasting_horizon_negative_raises(self):
        import pytest

        f = _TagMergingForecaster()
        with pytest.raises(ValueError, match="forecasting_horizon must be >= 1"):
            f._validate_fit_params(-1)

    def test_forecasting_horizon_valid_returns_value(self):
        f = _TagMergingForecaster()
        assert f._validate_fit_params(5) == 5


class TestSlidingWindowSplitterOverflow:
    """SlidingWindowSplitter raises when train + test exceeds data."""

    def test_train_plus_test_exceeds_n_samples(self):
        import pytest

        from yohou.model_selection.split import SlidingWindowSplitter

        cv = SlidingWindowSplitter(n_splits=2, train_size=8, test_size=5)
        y = pl.DataFrame({
            "time": pl.datetime_range(
                start=datetime(2020, 1, 1),
                end=datetime(2020, 1, 10),
                interval="1d",
                eager=True,
            ),
            "value": list(range(10)),
        })
        # The guard in _iter_test_indices is normally unreachable because
        # split() resolves and validates train_size_ first.  Set the fitted
        # train_size_ directly to bypass that and cover the defensive check.
        cv.train_size_ = 8  # skip upstream validation
        with pytest.raises(ValueError, match="greater than n_samples"):
            list(cv._iter_test_indices(y))
