"""Tests for yohou.testing.transformer check functions."""

from yohou.preprocessing.window import LagTransformer
from yohou.stationarity.transformers import LogTransformer
from yohou.testing.transformer import (
    check_feature_names_out_match,
    check_fit_idempotent,
    check_fit_sets_attributes,
    check_fit_transform_equivalence,
    check_insufficient_data_raises,
    check_inverse_transform_identity,
    check_inverse_transform_round_trip,
    check_memory_bounded,
    check_observation_horizon_after_fit,
    check_observation_horizon_not_fitted,
    check_observe_concatenates_memory,
    check_observe_transform_equivalence,
    check_panel_data_support,
    check_rewind_transform_behavior,
    check_rewind_updates_memory,
    check_tags_accessible_before_fit,
    check_tags_match_capabilities,
    check_tags_static_after_fit,
    check_transform_output_structure,
    check_transformer_preserve_dtypes,
    check_transformers_unfitted_stateless,
)


class TestTransformerFitChecks:
    """Tests for transformer fit-related check functions."""

    def test_fit_sets_attributes(self, y_X_factory):
        """Test check_fit_sets_attributes passes for valid transformer."""
        y, X = y_X_factory(length=50, n_targets=1, n_features=2, seed=42)
        transformer = LagTransformer(lag=3)

        # Should not raise
        check_fit_sets_attributes(transformer, X[:40], y[:40])

    def test_fit_sets_attributes_no_y(self, y_X_factory):
        """Test check validates transformer that doesn't use y."""
        y, X = y_X_factory(length=50, n_targets=1, n_features=2, seed=42)
        transformer = LagTransformer(lag=3)

        # Should not raise - y is optional
        check_fit_sets_attributes(transformer, X[:40], y=None)

    def test_fit_idempotent(self, y_X_factory):
        """Test check_fit_idempotent passes for valid transformer."""
        y, X = y_X_factory(length=50, n_targets=1, n_features=2, seed=42)
        transformer = LagTransformer(lag=3)
        transformer.fit(X[:40], y[:40])

        # Should not raise
        check_fit_idempotent(transformer, X[:40], y[:40])

    def test_fit_transform_equivalence(self, y_X_factory):
        """Test check_fit_transform_equivalence passes for valid transformer."""
        y, X = y_X_factory(length=50, n_targets=1, n_features=2, seed=42)
        transformer = LagTransformer(lag=3)

        # Should not raise
        check_fit_transform_equivalence(transformer, X[:40], y[:40])


class TestTransformerObservationHorizonChecks:
    """Tests for transformer observation horizon check functions."""

    def test_not_fitted(self, y_X_factory):
        """Test check_observation_horizon_not_fitted passes for unfitted transformer."""
        y, X = y_X_factory(length=50, n_targets=1, n_features=2, seed=42)
        transformer = LagTransformer(lag=3)

        # Should not raise - unfitted transformer correctly raises NotFittedError
        check_observation_horizon_not_fitted(transformer, X)

    def test_after_fit(self, y_X_factory):
        """Test check_observation_horizon_after_fit passes for valid transformer."""
        y, X = y_X_factory(length=50, n_targets=1, n_features=2, seed=42)
        transformer = LagTransformer(lag=3)
        transformer.fit(X[:40], y[:40])

        # Should not raise
        check_observation_horizon_after_fit(transformer, X[:40], y[:40])

    def test_insufficient_data_raises(self, y_X_factory):
        """Test check_insufficient_data_raises passes for valid transformer."""
        y, X = y_X_factory(length=50, n_targets=1, n_features=2, seed=42)
        transformer = LagTransformer(lag=10)

        # Pass data long enough to fit (lag=10 needs >10 rows)
        check_insufficient_data_raises(transformer, X[:30], y[:30])


class TestTransformerMemoryChecks:
    """Tests for transformer memory and state check functions."""

    def test_rewind_updates_memory(self, y_X_factory):
        """Test check_rewind_updates_memory passes for valid transformer."""
        y, X = y_X_factory(length=50, n_targets=1, n_features=2, seed=42)
        transformer = LagTransformer(lag=3)
        transformer.fit(X[:30], y[:30])

        # Should not raise
        check_rewind_updates_memory(transformer, X[20:40], y[20:40])

    def test_observe_concatenates_memory(self, y_X_factory):
        """Test check_observe_concatenates_memory passes for valid transformer."""
        y, X = y_X_factory(length=50, n_targets=1, n_features=2, seed=42)
        transformer = LagTransformer(lag=3)
        transformer.fit(X[:30], y[:30])

        # Should not raise
        check_observe_concatenates_memory(transformer, X[30:40], y[30:40])

    def test_observe_transform_equivalence(self, y_X_factory):
        """Test check_observe_transform_equivalence passes for valid transformer."""
        y, X = y_X_factory(length=50, n_targets=1, n_features=2, seed=42)
        transformer = LagTransformer(lag=3)
        transformer.fit(X[:30], y[:30])

        # Should not raise
        check_observe_transform_equivalence(transformer, X[30:40], y[30:40])

    def test_rewind_transform_behavior(self, y_X_factory):
        """Test check_rewind_transform_behavior passes for valid transformer."""
        y, X = y_X_factory(length=100, n_targets=1, n_features=2, seed=42)
        transformer = LagTransformer(lag=5)

        # Should not raise
        check_rewind_transform_behavior(transformer, X, y)

    def test_memory_bounded(self, y_X_factory):
        """Test check_memory_bounded passes for valid transformer."""
        y, X = y_X_factory(length=50, n_targets=1, n_features=2, seed=42, panel=False)
        transformer = LagTransformer(lag=3)
        transformer.fit(X[:20], y[:20])

        # Should not raise - check validates memory stays bounded
        check_memory_bounded(transformer, X[:20], X[20:40], y[:20])

    def test_unfitted_stateless(self, y_X_factory):
        """Test check_transformers_unfitted_stateless passes for valid transformer."""
        y, X = y_X_factory(length=50, n_targets=1, n_features=2, seed=42)
        transformer = LagTransformer(lag=3)

        # Should not raise
        check_transformers_unfitted_stateless(transformer, X)


class TestTransformerOutputChecks:
    """Tests for transformer output check functions."""

    def test_transform_output_structure(self, y_X_factory):
        """Test check_transform_output_structure passes for valid transformer."""
        y, X = y_X_factory(length=50, n_targets=1, n_features=2, seed=42)
        transformer = LagTransformer(lag=3)
        transformer.fit(X[:40], y[:40])

        # Should not raise
        check_transform_output_structure(transformer, X[:40], y[:40])

    def test_feature_names_out_match(self, y_X_factory):
        """Test check_feature_names_out_match passes for valid transformer."""
        y, X = y_X_factory(length=50, n_targets=1, n_features=2, seed=42)
        transformer = LagTransformer(lag=3)
        transformer.fit(X[:40], y[:40])

        # Should not raise
        check_feature_names_out_match(transformer, X[:40], y[:40])

    def test_preserve_dtypes(self, y_X_factory):
        """Test check_transformer_preserve_dtypes passes for valid transformer."""
        y, X = y_X_factory(length=50, n_targets=1, n_features=2, seed=42)
        transformer = LagTransformer(lag=3)
        transformer.fit(X[:40], y[:40])

        # Should not raise
        check_transformer_preserve_dtypes(transformer, X[:40], y[:40])

    def test_panel_data_support(self, y_X_factory):
        """Test check_panel_data_support passes for valid transformer."""
        y, X = y_X_factory(length=50, n_targets=1, n_features=2, seed=42, panel=True)
        transformer = LagTransformer(lag=3)
        transformer.fit(X[:40], y[:40])

        # Should not raise
        check_panel_data_support(transformer, X[:40], y[:40])


class TestTransformerInverseChecks:
    """Tests for transformer inverse transform check functions."""

    def test_inverse_transform_identity(self, y_X_factory):
        """Test check_inverse_transform_identity passes for invertible transformer."""
        y, X = y_X_factory(length=50, n_targets=1, n_features=2, seed=42)
        transformer = LogTransformer()
        transformer.fit(X[:40], y[:40])

        # Should not raise
        check_inverse_transform_identity(transformer, X[:40], y[:40])

    def test_inverse_transform_identity_skips_non_invertible(self, y_X_factory):
        """Test check skips transformers without inverse_transform."""
        y, X = y_X_factory(length=50, n_targets=1, n_features=2, seed=42)
        transformer = LagTransformer(lag=3)
        transformer.fit(X[:40], y[:40])

        # Should not raise - check skips non-invertible transformers
        check_inverse_transform_identity(transformer, X[:40], y[:40])

    def test_inverse_transform_round_trip(self, y_X_factory):
        """Test check_inverse_transform_round_trip passes for invertible transformer."""
        y, X = y_X_factory(length=50, n_targets=1, n_features=2, seed=42)
        transformer = LogTransformer()
        transformer.fit(X[:40], y[:40])

        # Should not raise
        check_inverse_transform_round_trip(transformer, X[:40], y[:40])


class TestTransformerTagChecks:
    """Tests for transformer tag check functions."""

    def test_tags_accessible_before_fit(self, y_X_factory):
        """Test check_tags_accessible_before_fit passes for valid transformer."""
        y, X = y_X_factory(length=50, n_targets=1, n_features=2, seed=42)
        transformer = LagTransformer(lag=3)

        # Should not raise
        check_tags_accessible_before_fit(transformer, X)

    def test_tags_static_after_fit(self, y_X_factory):
        """Test check_tags_static_after_fit passes for valid transformer."""
        y, X = y_X_factory(length=50, n_targets=1, n_features=2, seed=42)
        transformer = LagTransformer(lag=3)
        transformer.fit(X[:40], y[:40])

        # Should not raise
        check_tags_static_after_fit(transformer, X[:40], y[:40])

    def test_tags_match_capabilities(self, y_X_factory):
        """Test check_tags_match_capabilities passes for valid transformer."""
        y, X = y_X_factory(length=50, n_targets=1, n_features=2, seed=42)
        transformer = LagTransformer(lag=3)
        transformer.fit(X[:40], y[:40])

        # Should not raise
        check_tags_match_capabilities(transformer, X[:40])
