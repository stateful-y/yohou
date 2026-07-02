"""Contract tests for BaseTransformer lifecycle methods.

Verifies fit, transform, observe, rewind, observe_transform,
rewind_transform, tags, and fitted attributes using concrete test
transformer implementations.
"""

import sys
from pathlib import Path

import polars as pl
import pytest
from sklearn.utils.validation import check_is_fitted

from yohou.preprocessing import LagTransformer

sys.path.insert(0, str(Path(__file__).parent.parent))
from conftest import InvertibleTransformer, SimpleTransformer, StatelessTransformer


class TestBaseTransformerFit:
    """Tests for fit lifecycle and fitted attributes."""

    def test_fit_returns_self(self, time_series_factory):
        """Fit returns the transformer instance."""
        X = time_series_factory(length=50)
        t = SimpleTransformer(observation_horizon=2)
        result = t.fit(X)
        assert result is t

    # feature_names_in_ and n_features_in_ presence and values are covered by
    # check_fit_sets_attributes in the systematic suite (tests/test_common.py).

    def test_fit_sets_schema(self, time_series_factory):
        """Fit records the non-time input schema in X_schema_."""
        X = time_series_factory(length=50, n_components=2)
        t = SimpleTransformer(observation_horizon=0)
        t.fit(X)
        # X_schema_ (the time-excluded feature schema) is not covered by
        # check_fit_sets_attributes.
        assert t.X_schema_ == X.drop("time").schema

    def test_fit_sets_interval(self, time_series_factory):
        """Fit sets interval_ attribute from time series."""
        X = time_series_factory(length=50)
        t = SimpleTransformer(observation_horizon=0)
        t.fit(X)
        assert hasattr(t, "interval_")
        assert t.interval_ == "1s"

    def test_fit_sets_observed_time(self, time_series_factory):
        """Fit sets observed_time_ to last timestamp."""
        X = time_series_factory(length=50)
        t = SimpleTransformer(observation_horizon=0)
        t.fit(X)
        assert hasattr(t, "observed_time_")
        assert t.observed_time_ == X["time"][-1]

    def test_fit_sets_default_observation_horizon(self, time_series_factory):
        """Fit sets observation_horizon=0 if not set by subclass."""
        X = time_series_factory(length=50)
        t = StatelessTransformer()
        t.fit(X)
        assert t.observation_horizon == 0

    def test_is_fitted_after_fit(self, time_series_factory):
        """Transformer passes check_is_fitted after fit."""
        X = time_series_factory(length=50)
        t = SimpleTransformer(observation_horizon=0)
        t.fit(X)
        check_is_fitted(t)


class TestBaseTransformerObservationHorizon:
    """Tests for observation_horizon property."""

    def test_horizon_after_fit(self, time_series_factory):
        """Observation horizon reflects the configured value after fit."""
        X = time_series_factory(length=50)
        t = SimpleTransformer(observation_horizon=5)
        t.fit(X)
        assert t.observation_horizon == 5

    @pytest.mark.parametrize("horizon", [0, 3, 7])
    def test_fit_stores_horizon_rows_in_memory(self, time_series_factory, horizon):
        """After fit, _X_observed holds exactly observation_horizon rows.

        Covers the stateless (horizon == 0, empty frame) and stateful
        (horizon > 0) cases for the post-fit memory buffer.
        """
        X = time_series_factory(length=50)
        t = SimpleTransformer(observation_horizon=horizon)
        t.fit(X)
        assert hasattr(t, "_X_observed")
        assert len(t._X_observed) == horizon

    def test_insufficient_data_error_names_values(self, time_series_factory):
        """The insufficient-memory error reports the horizon and row count.

        The message includes both ``observation_horizon`` and ``len(X)`` so the
        failure is diagnosable.
        """
        X = time_series_factory(length=3)
        t = SimpleTransformer(observation_horizon=10)
        with pytest.raises(ValueError, match=r"observation_horizon=10 but X has 3 rows"):
            t.fit(X)


class TestBaseTransformerObserve:
    """Tests for observe() method."""

    def test_observe_returns_self(self, time_series_factory):
        """Observe returns the transformer instance."""
        X_all = time_series_factory(length=60)
        t = SimpleTransformer(observation_horizon=3)
        t.fit(X_all[:50])
        result = t.observe(X_all[50:])
        assert result is t

    def test_observe_updates_memory(self, time_series_factory):
        """Observe updates _X_observed with new data window."""
        X_all = time_series_factory(length=60)
        t = SimpleTransformer(observation_horizon=3)
        t.fit(X_all[:50])
        old_last_time = t._X_observed["time"][-1]
        t.observe(X_all[50:])
        new_last_time = t._X_observed["time"][-1]
        assert new_last_time > old_last_time

    def test_observe_maintains_horizon_size(self, time_series_factory):
        """After observe, memory window remains observation_horizon rows."""
        X_all = time_series_factory(length=60)
        t = SimpleTransformer(observation_horizon=3)
        t.fit(X_all[:50])
        t.observe(X_all[50:])
        assert len(t._X_observed) == 3

    def test_observe_updates_observed_time(self, time_series_factory):
        """Observe updates observed_time_ to last timestamp of new data."""
        X_all = time_series_factory(length=60)
        t = SimpleTransformer(observation_horizon=3)
        t.fit(X_all[:50])
        t.observe(X_all[50:])
        assert t.observed_time_ == X_all["time"][-1]

    # observe()/rewind()/transform()/observe_transform() raising NotFittedError
    # when unfitted is covered by check_transformer_methods_call_check_is_fitted
    # in the systematic suite (tests/test_common.py).


class TestBaseTransformerRewind:
    """Tests for rewind() method."""

    def test_rewind_returns_self(self, time_series_factory):
        """Rewind returns the transformer instance."""
        X = time_series_factory(length=50)
        t = SimpleTransformer(observation_horizon=3)
        t.fit(X)
        result = t.rewind(X)
        assert result is t

    def test_rewind_resets_memory(self, time_series_factory):
        """Rewind replaces memory window from given data."""
        X = time_series_factory(length=50)
        t = SimpleTransformer(observation_horizon=3)
        t.fit(X)
        t.rewind(X[:10])
        assert t._X_observed["time"][-1] == X["time"][9]


class TestBaseTransformerObserveTransform:
    """Tests for observe_transform() method."""

    def test_observe_transform_returns_dataframe(self, time_series_factory):
        """observe_transform returns a polars DataFrame."""
        X_all = time_series_factory(length=60)
        t = SimpleTransformer(observation_horizon=0, add_constant=1.0)
        t.fit(X_all[:50])
        result = t.observe_transform(X_all[50:])
        assert isinstance(result, pl.DataFrame)

    def test_observe_transform_preserves_length(self, time_series_factory):
        """observe_transform returns same number of rows as input."""
        X_all = time_series_factory(length=60)
        t = SimpleTransformer(observation_horizon=0, add_constant=1.0)
        t.fit(X_all[:50])
        result = t.observe_transform(X_all[50:])
        assert len(result) == 10

    def test_observe_transform_updates_memory(self, time_series_factory):
        """observe_transform updates internal state."""
        X_all = time_series_factory(length=60)
        t = SimpleTransformer(observation_horizon=3, add_constant=0.0)
        t.fit(X_all[:50])
        t.observe_transform(X_all[50:])
        assert t.observed_time_ == X_all["time"][-1]

    def test_observe_transform_preserves_time(self, time_series_factory):
        """observe_transform output has time column."""
        X_all = time_series_factory(length=60)
        t = SimpleTransformer(observation_horizon=0, add_constant=1.0)
        t.fit(X_all[:50])
        result = t.observe_transform(X_all[50:])
        assert "time" in result.columns

    def test_observe_transform_uses_pre_observation_memory(self, time_series_factory):
        """observe_transform transforms from memory BEFORE the observation.

        The documented contract is ``transform(concat(_X_observed, X))[-len(X):]``
        so the first rows of ``X`` are computed against the prior memory rather
        than against ``X`` alone. A stateful lag transformer makes this visible:
        without the prior memory the leading lag rows would be dropped.
        """
        X_all = time_series_factory(length=60)
        t = LagTransformer([1, 2])
        t.fit(X_all[:50])

        new_data = X_all[50:]
        expected = t.transform(pl.concat([t._X_observed, new_data]))[-len(new_data) :]

        result = t.observe_transform(new_data)

        assert result.equals(expected)
        # The first output row aligns with the first new row (no warmup drop),
        # proving the prior memory was used rather than transform(new_data).
        assert result["time"][0] == new_data["time"][0]


class TestBaseTransformerRewindTransform:
    """Tests for rewind_transform() method."""

    def test_rewind_transform_returns_dataframe(self, time_series_factory):
        """rewind_transform returns a polars DataFrame."""
        X = time_series_factory(length=50)
        t = SimpleTransformer(observation_horizon=0, add_constant=1.0)
        t.fit(X)
        result = t.rewind_transform(X)
        assert isinstance(result, pl.DataFrame)

    def test_rewind_transform_rewinds_state(self, time_series_factory):
        """rewind_transform resets the observation window."""
        X_all = time_series_factory(length=60)
        t = SimpleTransformer(observation_horizon=3)
        t.fit(X_all[:50])
        t.observe(X_all[50:])
        # After observe, memory includes new data
        assert t.observed_time_ == X_all["time"][-1]
        # Rewind to earlier data
        t.rewind_transform(X_all[:20])
        assert t.observed_time_ == X_all["time"][19]

    def test_rewind_transform_preserves_time(self, time_series_factory):
        """rewind_transform output has time column."""
        X = time_series_factory(length=50)
        t = SimpleTransformer(observation_horizon=0, add_constant=1.0)
        t.fit(X)
        result = t.rewind_transform(X)
        assert "time" in result.columns


class TestBaseTransformerFitTransform:
    """Tests for fit_transform() method."""

    def test_fit_transform_returns_dataframe(self, time_series_factory):
        """fit_transform returns a polars DataFrame."""
        X = time_series_factory(length=50)
        t = SimpleTransformer(observation_horizon=0, add_constant=1.0)
        result = t.fit_transform(X)
        assert isinstance(result, pl.DataFrame)

    def test_fit_transform_preserves_time(self, time_series_factory):
        """fit_transform output has time column."""
        X = time_series_factory(length=50)
        t = SimpleTransformer(observation_horizon=0, add_constant=1.0)
        result = t.fit_transform(X)
        assert "time" in result.columns

    def test_fit_transform_makes_fitted(self, time_series_factory):
        """fit_transform leaves transformer in fitted state."""
        X = time_series_factory(length=50)
        t = SimpleTransformer(observation_horizon=0)
        t.fit_transform(X)
        check_is_fitted(t)


class TestBaseTransformerTags:
    """Tests for __sklearn_tags__()."""

    def test_estimator_type_is_transformer(self):
        """Tags report estimator_type as transformer."""
        t = SimpleTransformer(observation_horizon=0)
        tags = t.__sklearn_tags__()
        assert tags.estimator_type == "transformer"

    def test_invertible_tag_when_method_present(self):
        """Tags report invertible=True when inverse_transform exists."""
        t = InvertibleTransformer(observation_horizon=0)
        tags = t.__sklearn_tags__()
        assert tags.transformer_tags.invertible is True

    def test_not_invertible_by_default(self):
        """Tags report invertible=False by default."""
        t = SimpleTransformer(observation_horizon=0)
        tags = t.__sklearn_tags__()
        assert tags.transformer_tags.invertible is False

    def test_invertible_round_trip(self, time_series_factory):
        """inverse_transform(X_p=...) reverses transform via the base-class signature."""
        from polars.testing import assert_frame_equal

        X = time_series_factory(length=50)
        t = InvertibleTransformer(observation_horizon=0, offset=7.0)
        t.fit(X)
        X_t = t.transform(X)
        X_back = t.inverse_transform(X_t, X_p=None)
        assert_frame_equal(X_back, X)


class TestBaseTransformerUpdateXObserved:
    """Tests for the internal _update_X_observed API."""

    def test_insufficient_data_raises(self, time_series_factory):
        """A direct _update_X_observed call reports horizon and row count.

        The post-fit memory length for both stateless and stateful horizons is
        covered by test_fit_stores_horizon_rows_in_memory above; this test pins
        the error path on the internal API directly.
        """
        X = time_series_factory(length=50)
        t = SimpleTransformer(observation_horizon=10)
        t.fit(X)
        with pytest.raises(ValueError, match=r"observation_horizon=10 but X has 5 rows"):
            t._update_X_observed(X[:5])
