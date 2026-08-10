"""Tests for the step-kind transformer base and its slot guards."""

from datetime import datetime, timedelta

import polars as pl
import pytest
from sklearn.linear_model import Ridge

from yohou.base import BaseStepTransformer
from yohou.base.step_transformer import _is_step_indexed, _step_index
from yohou.base.utils import (
    _is_step_kind,
    _kind_of,
    _require_actual_transformer,
    _require_forecast_transformer,
    _require_step_transformer,
)
from yohou.compose import ColumnTransformer, FeaturePipeline, FeatureUnion, PerVintageActualTransformer
from yohou.compose.utils import check_homogeneous_kinds, common_kind
from yohou.point import PointReductionForecaster
from yohou.preprocessing import FunctionTransformer, StandardScaler, StepAggregator


@pytest.fixture
def step_frame() -> pl.DataFrame:
    """A step frame with two base columns over a horizon of three."""
    times = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(5)]
    return pl.DataFrame({
        "time": times,
        "temp_step_1": [1.0, 2.0, 3.0, 4.0, 5.0],
        "temp_step_2": [2.0, 3.0, 4.0, 5.0, 6.0],
        "temp_step_3": [9.0, 1.0, 4.0, 2.0, 8.0],
        "rain_step_1": [0.0, 1.0, 0.0, 1.0, 0.0],
        "rain_step_2": [1.0, 0.0, 1.0, 0.0, 1.0],
        "rain_step_3": [0.0, 0.0, 1.0, 1.0, 0.0],
    })


class _Identity(BaseStepTransformer):
    """Minimal step transformer used to exercise the base contract."""

    def _transform(self, X: pl.DataFrame) -> pl.DataFrame:
        return X

    def get_feature_names_out(self, input_features: list[str] | None = None) -> list[str]:
        return list(self.feature_names_in_)


class TestStepIndexPartition:
    """The single site that parses a horizon index out of a column name."""

    @pytest.mark.parametrize(
        ("name", "indexed", "index"),
        [
            ("temp_step_1", True, 1),
            ("temp_step_10", True, 10),
            ("temp_step_mean", False, None),
            ("wx_step_pc1", False, None),
            ("temp_step_n_covered", False, None),
            # Anchoring matters: a base column that already contains "_step_"
            # must resolve on its real trailing index, not the embedded one.
            ("foo_step_3_step_1", True, 1),
        ],
    )
    def test_partition(self, name, indexed, index):
        """Step-indexed names are separated from horizon-agnostic ones."""
        assert _is_step_indexed(name) is indexed
        assert _step_index(name) == index


class TestBaseContract:
    """Fit/transform behaviour shared by every step transformer."""

    def test_kind_tag_readable_before_fit(self):
        """The kind tag is static, so a slot can reject a wrong-kind object early."""
        assert _Identity().__sklearn_tags__().transformer_tags.kind == "step"
        assert _is_step_kind(_Identity())
        assert _kind_of(_Identity()) == "step"

    def test_fit_records_schema(self, step_frame):
        """fit records the non-index feature schema."""
        t = _Identity().fit(step_frame)
        assert t.feature_names_in_ == [c for c in step_frame.columns if c != "time"]
        assert t.n_features_in_ == 6

    def test_transform_preserves_time(self, step_frame):
        """transform keeps the index column and row order."""
        out = _Identity().fit_transform(step_frame)
        assert out["time"].to_list() == step_frame["time"].to_list()

    def test_missing_time_raises(self, step_frame):
        """A frame without the index column is rejected."""
        t = _Identity().fit(step_frame)
        with pytest.raises(ValueError, match="must contain a 'time' column"):
            t.transform(step_frame.drop("time"))

    def test_non_datetime_time_raises(self, step_frame):
        """A 'time' column that is not a date or datetime is rejected."""
        bad = step_frame.with_columns(time=pl.Series(range(len(step_frame))))
        with pytest.raises(ValueError, match="must be Date or Datetime"):
            _Identity().fit(bad)

    def test_schema_change_after_fit_raises(self, step_frame):
        """Feature columns differing from fit are rejected at transform."""
        t = _Identity().fit(step_frame)
        with pytest.raises(ValueError, match="do not match those seen during fit"):
            t.transform(step_frame.drop("rain_step_3"))

    def test_min_steps_readable_before_fit(self):
        """min_steps is parameter-derived, so it does not require a fit.

        The forecaster asserts the horizon against it *before* fitting the slot;
        requiring a fit would only delay the clearer error.
        """
        assert _Identity().min_steps == 1


class TestMemoryApiRefused:
    """The memory API is unavailable on step-kind objects, leaves and compositions."""

    @pytest.mark.parametrize("method", ["observe", "rewind", "observe_transform", "rewind_transform"])
    def test_leaf_has_no_memory_api(self, step_frame, method):
        """A leaf step transformer does not carry the memory API at all.

        `BaseStepTransformer` subclasses `_BaseTransformer` directly, as
        `BaseForecastTransformer` does, so these methods are simply absent rather
        than present-and-guarded. The tag guard exists for compositions, which are
        structurally `BaseActualTransformer` subclasses reporting ``kind="step"``.
        """
        t = _Identity().fit(step_frame)
        assert not hasattr(t, method)

    @pytest.mark.parametrize("method", ["observe", "rewind", "observe_transform", "rewind_transform"])
    def test_composition_refuses(self, step_frame, method):
        """A step-kind composition refuses them too.

        The composers override the paired ``*_transform`` methods separately from
        ``observe``/``rewind``, a gap this library already had to close once for
        forecast-kind transformers.
        """
        union = FeatureUnion([("a", StepAggregator())])
        with pytest.raises(ValueError, match="step-kind"):
            getattr(union, method)(step_frame)

    def test_actual_kind_unaffected(self):
        """The guard is a no-op for actual-kind transformers."""
        times = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(5)]
        X = pl.DataFrame({"time": times, "a": [1.0, 2.0, 3.0, 4.0, 5.0]})
        t = StandardScaler().fit(X)
        t.observe(X)  # must not raise


class TestSlotGuards:
    """Each slot admits exactly one kind."""

    def test_step_in_actual_slot_refused(self):
        """A step transformer in an actual slot names the right destination."""
        with pytest.raises(ValueError, match="step_transformer slot"):
            _require_actual_transformer(StepAggregator(), "actual_transformer")

    def test_actual_in_step_slot_refused(self):
        """An actual transformer in the step slot is refused."""
        with pytest.raises(ValueError, match="must be a step-kind transformer"):
            _require_step_transformer(StandardScaler(), "step_transformer")

    def test_forecast_in_step_slot_refused(self):
        """A forecast transformer in the step slot is refused."""
        with pytest.raises(ValueError, match="must be a step-kind transformer"):
            _require_step_transformer(PerVintageActualTransformer(FunctionTransformer()), "step_transformer")

    def test_step_in_forecast_slot_refused(self):
        """A step transformer in the forecast slot is refused."""
        with pytest.raises(ValueError, match="must be a forecast-kind transformer"):
            _require_forecast_transformer(StepAggregator(), "forecast_transformer")

    def test_step_transformer_accepted(self):
        """A step transformer passes its own slot guard."""
        _require_step_transformer(StepAggregator(), "step_transformer")

    def test_none_accepted(self):
        """An unset slot is always fine."""
        _require_step_transformer(None, "step_transformer")


class TestCompositionKind:
    """Compositions inherit the step kind from homogeneous children."""

    @pytest.mark.parametrize(
        "make",
        [
            lambda: FeatureUnion([("a", StepAggregator())]),
            lambda: FeaturePipeline([("a", StepAggregator())]),
            lambda: ColumnTransformer([("a", StepAggregator(), ["temp_step_1"])]),
        ],
        ids=["union", "pipeline", "column_transformer"],
    )
    def test_inherits_step_kind(self, make):
        """A composition of step children reports kind='step' and passes the slot guard."""
        composition = make()
        assert composition.__sklearn_tags__().transformer_tags.kind == "step"
        _require_step_transformer(composition, "step_transformer")

    def test_common_kind_of_step_children(self):
        """common_kind returns 'step' for homogeneous step children."""
        assert common_kind([("a", StepAggregator()), ("b", StepAggregator())]) == "step"

    def test_mixing_kinds_refused(self):
        """A mixed composition names every kind present, not a fixed pair."""
        named = [("s", StepAggregator()), ("a", StandardScaler())]
        with pytest.raises(ValueError, match="cannot mix"):
            check_homogeneous_kinds(named, "FeatureUnion")

    def test_mixing_message_lists_members(self):
        """The message enumerates the offending members per kind."""
        named = [("s", StepAggregator()), ("a", StandardScaler())]
        with pytest.raises(ValueError) as excinfo:
            check_homogeneous_kinds(named, "FeatureUnion")
        message = str(excinfo.value)
        assert "actual=['a']" in message
        assert "step=['s']" in message


class TestForecasterSlot:
    """The slot on a forecaster: parameter, tag, and horizon guard."""

    @pytest.fixture
    def training_data(self):
        """A short series with a known-future feature."""
        n, horizon = 40, 4
        times = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(n)]
        future = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(n + horizon)]
        y = pl.DataFrame({"time": times, "y": [float(i % 7) + i * 0.1 for i in range(n)]})
        X_future = pl.DataFrame({"time": future, "temp": [float((i * 3) % 11) for i in range(n + horizon)]})
        return y, X_future, horizon

    def test_slot_is_a_parameter(self):
        """Every reduction family exposes the slot in get_params."""
        assert "step_transformer" in PointReductionForecaster(estimator=Ridge()).get_params()

    def test_tag_follows_the_parameter(self):
        """uses_step_transformer tracks whether the slot is set."""
        unset = PointReductionForecaster(estimator=Ridge())
        set_ = PointReductionForecaster(estimator=Ridge(), step_transformer=StepAggregator())
        assert not unset.__sklearn_tags__().forecaster_tags.uses_step_transformer
        assert set_.__sklearn_tags__().forecaster_tags.uses_step_transformer

    def test_wrong_kind_refused_at_fit(self, training_data):
        """A misconfigured slot is rejected at fit, not silently accepted."""
        y, X_future, horizon = training_data
        forecaster = PointReductionForecaster(estimator=Ridge(), step_transformer=StandardScaler())
        with pytest.raises(ValueError, match="must be a step-kind transformer"):
            forecaster.fit(y, forecasting_horizon=horizon, X_future=X_future)

    def test_wrong_kind_refused_without_exogenous(self, training_data):
        """The guard runs even on a fit carrying no X_future or X_forecast.

        A wrong-kind slot is a static misconfiguration, so it must not lurk until
        exogenous data happens to appear.
        """
        y, _, horizon = training_data
        forecaster = PointReductionForecaster(estimator=Ridge(), step_transformer=StandardScaler())
        with pytest.raises(ValueError, match="must be a step-kind transformer"):
            forecaster.fit(y, forecasting_horizon=horizon)

    def test_horizon_below_min_steps_refused(self, training_data):
        """The horizon guard names the horizon and the minimum, before the inner fit."""
        from sklearn.decomposition import PCA

        from yohou.preprocessing import StepColumnReducer

        y, X_future, _ = training_data
        forecaster = PointReductionForecaster(
            estimator=Ridge(),
            step_transformer=StepColumnReducer(reducer=PCA(n_components=4)),
        )
        with pytest.raises(ValueError, match=r"forecasting_horizon \(2\) is below the minimum block width"):
            forecaster.fit(y, forecasting_horizon=2, X_future=X_future)
