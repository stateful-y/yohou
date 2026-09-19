"""Reduction forecasters recognise, validate and align step columns from the actual transformer."""

from datetime import timedelta

import polars as pl
import pytest
from sklearn.linear_model import LinearRegression

from yohou.base import BaseActualTransformer
from yohou.compose import FeaturePipeline, FeatureUnion
from yohou.point import PointReductionForecaster
from yohou.preprocessing import LagTransformer
from yohou.utils._compat import _check_feature_names_in


class _StepProbe(BaseActualTransformer):
    """Step-output probe: ``{col}_{name}_step_h`` holds ``col`` shifted by ``H - h``.

    Causal (every shift is backwards) and requires the forecasting horizon as fit
    metadata, like a real step-output transformer.
    """

    __metadata_request__fit = {"forecasting_horizon": True}
    _tags = {"stateful": True, "batch_invariant": True, "produces_step_columns": True}

    def __init__(self, name: str = "probe"):
        self.name = name

    @property
    def observation_horizon(self) -> int:
        """Deepest shift taken."""
        return self.forecasting_horizon_ - 1

    def fit(self, X, y=None, **params):
        """Fit, reading the horizon from fit metadata."""
        self.forecasting_horizon_ = params["forecasting_horizon"]
        return super().fit(X, y)

    def _transform(self, X):
        horizon = self.forecasting_horizon_
        columns = [c for c in X.columns if c != "time"]
        out = X.select([
            pl.col("time"),
            *[
                pl.col(c).shift(horizon - h).alias(f"{c}_{self.name}_step_{h}")
                for c in columns
                for h in range(1, horizon + 1)
            ],
        ])
        return out[self._observation_horizon :]

    def get_feature_names_out(self, input_features=None):
        """Names of the step columns."""
        features = _check_feature_names_in(self, input_features)
        return [f"{c}_{self.name}_step_{h}" for c in features for h in range(1, self.forecasting_horizon_ + 1)]


class _Rename(BaseActualTransformer):
    """Stateless transformer that appends a fixed suffix to every column name."""

    def __init__(self, suffix: str = "_ramp_step_2"):
        self.suffix = suffix

    @property
    def observation_horizon(self) -> int:
        """Stateless."""
        return 0

    def _transform(self, X):
        return X.rename({c: f"{c}{self.suffix}" for c in X.columns if c != "time"})

    def get_feature_names_out(self, input_features=None):
        """Suffixed names."""
        return [f"{c}{self.suffix}" for c in _check_feature_names_in(self, input_features)]


def _forecaster(actual_transformer, **kwargs):
    params = {"reduction_strategy": "direct", "step_feature_alignment": "matched", **kwargs}
    return PointReductionForecaster(estimator=LinearRegression(), actual_transformer=actual_transformer, **params)


def _union():
    return FeatureUnion([("lag", LagTransformer(lag=1)), ("seasonal", _StepProbe())])


class TestRecording:
    """Step-output columns are recorded apart from derived step columns."""

    def test_standard(self, y_X_factory):
        """Every probe output is recorded; the derived-step record stays empty."""
        y, X = y_X_factory(length=80, n_targets=1, n_features=1)
        forecaster = _forecaster(_union()).fit(y, X, forecasting_horizon=4)

        expected = {f"seasonal_{c}_probe_step_{h}" for c in ["y_0", "X_0"] for h in range(1, 5)}
        assert forecaster._actual_step_column_names_ == expected
        assert forecaster._actual_step_column_local_names_ == expected
        assert forecaster._step_column_names_ == set()

    @pytest.mark.parametrize("panel_strategy", ["global", "multivariate"])
    def test_panel(self, y_X_panel_factory, panel_strategy):
        """Both spellings are recorded under both panel strategies."""
        y, X = y_X_panel_factory(n_groups=2, length=80, n_targets=1, n_features=1)
        forecaster = _forecaster(_union(), panel_strategy=panel_strategy).fit(y, X, forecasting_horizon=3)

        names = forecaster._actual_step_column_names_
        local = forecaster._actual_step_column_local_names_
        assert names and local
        assert all(n.rsplit("_step_", 1)[1] in {"1", "2", "3"} for n in names)
        if panel_strategy == "global":
            assert {n.split("__", 1)[1] for n in names} == local
            assert {n.split("__", 1)[0] for n in names} == set(forecaster.groups_)
        else:
            assert names == local

    def test_untagged_transformer_records_nothing(self, y_X_factory):
        """Without the tag, a ``_step_<n>`` suffix is an ordinary feature reaching every model."""
        y, X = y_X_factory(length=80, n_targets=1, n_features=1)
        forecaster = _forecaster(FeatureUnion([("lag", LagTransformer(lag=1)), ("rename", _Rename())]))
        forecaster.fit(y, X, forecasting_horizon=3)

        assert forecaster._actual_step_column_names_ == set()
        for step in range(1, 4):
            assert "rename_y_0_ramp_step_2" in forecaster._filter_step_features(_design_row(forecaster), step).columns


def _design_row(forecaster):
    """The observed feature row without its time column."""
    return forecaster._X_t_observed.drop("time")


class TestValidation:
    """Misfiled or renamed step columns fail at fit."""

    def test_stray_incomplete_block(self, y_X_factory):
        """An untagged ``*_step_2`` inside a tagged union forms an incomplete block."""
        y, X = y_X_factory(length=80, n_targets=1, n_features=1)
        union = FeatureUnion([("seasonal", _StepProbe()), ("custom", _Rename())])
        with pytest.raises(ValueError, match=r"_ramp_step_<n>' cover steps \[2\]"):
            _forecaster(union).fit(y, X, forecasting_horizon=4)

    @pytest.mark.parametrize("panel_strategy", ["global", "multivariate"])
    def test_stray_incomplete_block_panel(self, y_X_panel_factory, panel_strategy):
        """The same incomplete block fails under both panel strategies."""
        y, X = y_X_panel_factory(n_groups=2, length=80, n_targets=1, n_features=1)
        union = FeatureUnion([("seasonal", _StepProbe()), ("custom", _Rename())])
        with pytest.raises(ValueError, match=r"_ramp_step_<n>' cover steps \[2\]"):
            _forecaster(union, panel_strategy=panel_strategy).fit(y, X, forecasting_horizon=4)

    def test_renamed_by_later_pipeline_step(self, y_X_factory):
        """A LagTransformer after the probe renames every step column away."""
        y, X = y_X_factory(length=80, n_targets=1, n_features=1)
        pipeline = FeaturePipeline([("seasonal", _StepProbe()), ("lag", LagTransformer(lag=1))])
        with pytest.raises(ValueError, match="none of its .* output columns ends in '_step_<n>'"):
            _forecaster(pipeline).fit(y, X, forecasting_horizon=4)

    def test_collision_with_derived_step_column(self, y_X_factory):
        """A step-output name equal to a step column derived from X_future is rejected."""
        y, X = y_X_factory(length=80, n_targets=1, n_features=1)
        horizon = 3
        times = pl.datetime_range(y["time"][0], y["time"][-1] + timedelta(days=horizon), interval="1d", eager=True)
        X_future = pl.DataFrame({"time": times, "X_0_probe": pl.Series(range(len(times)), dtype=pl.Float64)})
        forecaster = _forecaster(_StepProbe())
        with pytest.raises(ValueError, match="X_0_probe_step_1"):
            forecaster.fit(y, X, forecasting_horizon=horizon, X_future=X_future)


class TestFiltering:
    """Per-step models see only their own step-output columns."""

    def test_matched_and_cumulative(self, y_X_factory):
        """Matched keeps one step's columns, cumulative keeps steps up to h, others pass through."""
        y, X = y_X_factory(length=80, n_targets=1, n_features=1)
        matched = _forecaster(_union()).fit(y, X, forecasting_horizon=4)
        row = _design_row(matched)

        kept = matched._filter_step_features(row, 3).columns
        assert set(kept) == {
            "lag_y_0_lag_1",
            "lag_X_0_lag_1",
            "seasonal_y_0_probe_step_3",
            "seasonal_X_0_probe_step_3",
        }

        cumulative = _forecaster(_union(), step_feature_alignment="cumulative").fit(y, X, forecasting_horizon=4)
        kept = cumulative._filter_step_features(_design_row(cumulative), 2).columns
        assert {c for c in kept if "_step_" in c} == {
            f"seasonal_{c}_probe_step_{h}" for c in ["y_0", "X_0"] for h in (1, 2)
        }

    def test_predict_observe_predict(self, y_X_factory):
        """With only step-output columns, the forecaster predicts, observes and predicts again."""
        y, X = y_X_factory(length=90, n_targets=1, n_features=1)
        forecaster = _forecaster(_union()).fit(y[:80], X[:80], forecasting_horizon=4)
        first = forecaster.predict(forecasting_horizon=4)
        forecaster.observe(y[80:85], X[80:85])
        second = forecaster.predict(forecasting_horizon=4)
        assert first.height == second.height == 4
        assert second["time"][0] > first["time"][0]


def test_unfiltered_columns_warn(y_X_factory):
    """Multi-output gives every model all step columns, and fit says so."""
    y, X = y_X_factory(length=80, n_targets=1, n_features=1)
    forecaster = PointReductionForecaster(
        estimator=LinearRegression(), actual_transformer=_union(), reduction_strategy="multi-output"
    )
    with pytest.warns(UserWarning, match=r"produces 8 step column\(s\)"):
        forecaster.fit(y, X, forecasting_horizon=4)


def test_matched_direct_does_not_warn(y_X_factory, recwarn):
    """The aligned configuration emits no unfiltered-columns warning."""
    y, X = y_X_factory(length=80, n_targets=1, n_features=1)
    _forecaster(_union()).fit(y, X, forecasting_horizon=4)
    assert not [w for w in recwarn if "step column(s)" in str(w.message)]


def test_recursive_predict_rejected(y_X_factory):
    """Predicting past the fit horizon raises, since step columns cover only that horizon."""
    y, X = y_X_factory(length=80, n_targets=1, n_features=1)
    forecaster = _forecaster(_union()).fit(y, X, forecasting_horizon=4)
    with pytest.raises(ValueError, match="step-output features cover only the fit horizon"):
        forecaster.predict(forecasting_horizon=8)
