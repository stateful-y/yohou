"""Tests for yohou.testing.transformer check functions."""

from yohou.preprocessing.stationarization import LogTransform
from yohou.preprocessing.window import LagTransformer
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
    check_panel_data_support,
    check_reset_updates_memory,
    check_tags_accessible_before_fit,
    check_tags_match_capabilities,
    check_tags_static_after_fit,
    check_transform_output_structure,
    check_transformer_preserve_dtypes,
    check_transformers_unfitted_stateless,
    check_update_concatenates_memory,
    check_update_transform_equivalence,
)


def test_check_fit_sets_attributes(y_X_factory):
    """Test check_fit_sets_attributes passes for valid transformer."""
    y, X = y_X_factory(length=50, n_targets=1, n_features=2, seed=42)
    transformer = LagTransformer(lag=3)

    # Should not raise
    check_fit_sets_attributes(transformer, X[:40], y[:40])


def test_check_fit_sets_attributes_no_y(y_X_factory):
    """Test check validates transformer that doesn't use y."""
    y, X = y_X_factory(length=50, n_targets=1, n_features=2, seed=42)
    transformer = LagTransformer(lag=3)

    # Should not raise - y is optional
    check_fit_sets_attributes(transformer, X[:40], y=None)


def test_check_observation_horizon_not_fitted(y_X_factory):
    """Test check_observation_horizon_not_fitted passes for unfitted transformer."""
    y, X = y_X_factory(length=50, n_targets=1, n_features=2, seed=42)
    transformer = LagTransformer(lag=3)

    # Should not raise - unfitted transformer correctly raises NotFittedError
    check_observation_horizon_not_fitted(transformer, X)


def test_check_observation_horizon_after_fit(y_X_factory):
    """Test check_observation_horizon_after_fit passes for valid transformer."""
    y, X = y_X_factory(length=50, n_targets=1, n_features=2, seed=42)
    transformer = LagTransformer(lag=3)
    transformer.fit(X[:40], y[:40])

    # Should not raise
    check_observation_horizon_after_fit(transformer, X[:40], y[:40])


def test_check_reset_updates_memory(y_X_factory):
    """Test check_reset_updates_memory passes for valid transformer."""
    y, X = y_X_factory(length=50, n_targets=1, n_features=2, seed=42)
    transformer = LagTransformer(lag=3)
    transformer.fit(X[:30], y[:30])

    # Should not raise
    check_reset_updates_memory(transformer, X[20:40], y[20:40])


def test_check_update_concatenates_memory(y_X_factory):
    """Test check_update_concatenates_memory passes for valid transformer."""
    y, X = y_X_factory(length=50, n_targets=1, n_features=2, seed=42)
    transformer = LagTransformer(lag=3)
    transformer.fit(X[:30], y[:30])

    # Should not raise
    check_update_concatenates_memory(transformer, X[30:40], y[30:40])


def test_check_update_transform_equivalence(y_X_factory):
    """Test check_update_transform_equivalence passes for valid transformer."""
    y, X = y_X_factory(length=50, n_targets=1, n_features=2, seed=42)
    transformer = LagTransformer(lag=3)
    transformer.fit(X[:30], y[:30])

    # Should not raise
    check_update_transform_equivalence(transformer, X[30:40], y[30:40])


def test_check_insufficient_data_raises(y_X_factory):
    """Test check_insufficient_data_raises passes for valid transformer.

    The check function fits the transformer with the provided data,
    then tests transform() with insufficient data (< observation_horizon).
    """
    y, X = y_X_factory(length=50, n_targets=1, n_features=2, seed=42)
    transformer = LagTransformer(lag=10)

    # Pass data long enough to fit (lag=10 needs >10 rows)
    # Check will then test transform with insufficient data
    check_insufficient_data_raises(transformer, X[:30], y[:30])


def test_check_transform_output_structure(y_X_factory):
    """Test check_transform_output_structure passes for valid transformer."""
    y, X = y_X_factory(length=50, n_targets=1, n_features=2, seed=42)
    transformer = LagTransformer(lag=3)
    transformer.fit(X[:40], y[:40])

    # Should not raise
    check_transform_output_structure(transformer, X[:40], y[:40])


def test_check_feature_names_out_match(y_X_factory):
    """Test check_feature_names_out_match passes for valid transformer."""
    y, X = y_X_factory(length=50, n_targets=1, n_features=2, seed=42)
    transformer = LagTransformer(lag=3)
    transformer.fit(X[:40], y[:40])

    # Should not raise
    check_feature_names_out_match(transformer, X[:40], y[:40])


def test_check_inverse_transform_identity(y_X_factory):
    """Test check_inverse_transform_identity passes for invertible transformer."""
    y, X = y_X_factory(length=50, n_targets=1, n_features=2, seed=42)
    transformer = LogTransform()
    transformer.fit(X[:40], y[:40])

    # Should not raise
    check_inverse_transform_identity(transformer, X[:40], y[:40])


def test_check_inverse_transform_identity_skips_non_invertible(y_X_factory):
    """Test check skips transformers without inverse_transform."""
    y, X = y_X_factory(length=50, n_targets=1, n_features=2, seed=42)
    transformer = LagTransformer(lag=3)
    transformer.fit(X[:40], y[:40])

    # Should not raise - check skips non-invertible transformers
    check_inverse_transform_identity(transformer, X[:40], y[:40])


def test_check_panel_data_support(y_X_factory):
    """Test check_panel_data_support passes for valid transformer."""
    y, X = y_X_factory(length=50, n_targets=1, n_features=2, seed=42, panel=True)
    transformer = LagTransformer(lag=3)
    transformer.fit(X[:40], y[:40])

    # Should not raise
    check_panel_data_support(transformer, X[:40], y[:40])


def test_check_transformers_unfitted_stateless(y_X_factory):
    """Test check_transformers_unfitted_stateless passes for valid transformer."""
    y, X = y_X_factory(length=50, n_targets=1, n_features=2, seed=42)
    transformer = LagTransformer(lag=3)

    # Should not raise
    check_transformers_unfitted_stateless(transformer, X)


def test_check_transformer_preserve_dtypes(y_X_factory):
    """Test check_transformer_preserve_dtypes passes for valid transformer."""
    y, X = y_X_factory(length=50, n_targets=1, n_features=2, seed=42)
    transformer = LagTransformer(lag=3)
    transformer.fit(X[:40], y[:40])

    # Should not raise
    check_transformer_preserve_dtypes(transformer, X[:40], y[:40])


def test_check_fit_idempotent(y_X_factory):
    """Test check_fit_idempotent passes for valid transformer."""
    y, X = y_X_factory(length=50, n_targets=1, n_features=2, seed=42)
    transformer = LagTransformer(lag=3)
    transformer.fit(X[:40], y[:40])

    # Should not raise
    check_fit_idempotent(transformer, X[:40], y[:40])


def test_check_inverse_transform_round_trip(y_X_factory):
    """Test check_inverse_transform_round_trip passes for invertible transformer."""
    y, X = y_X_factory(length=50, n_targets=1, n_features=2, seed=42)
    transformer = LogTransform()
    transformer.fit(X[:40], y[:40])

    # Should not raise
    check_inverse_transform_round_trip(transformer, X[:40], y[:40])


def test_check_fit_transform_equivalence(y_X_factory):
    """Test check_fit_transform_equivalence passes for valid transformer."""
    y, X = y_X_factory(length=50, n_targets=1, n_features=2, seed=42)
    transformer = LagTransformer(lag=3)

    # Should not raise
    check_fit_transform_equivalence(transformer, X[:40], y[:40])


def test_check_memory_bounded(y_X_factory):
    """Test check_memory_bounded passes for valid transformer."""
    y, X = y_X_factory(length=50, n_targets=1, n_features=2, seed=42, panel=False)
    transformer = LagTransformer(lag=3)
    transformer.fit(X[:20], y[:20])

    # Should not raise - check validates memory stays bounded
    # X_train=X[:20], X_test=X[20:40], y=y[:20]
    check_memory_bounded(transformer, X[:20], X[20:40], y[:20])


def test_check_tags_accessible_before_fit(y_X_factory):
    """Test check_tags_accessible_before_fit passes for valid transformer."""
    y, X = y_X_factory(length=50, n_targets=1, n_features=2, seed=42)
    transformer = LagTransformer(lag=3)

    # Should not raise
    check_tags_accessible_before_fit(transformer, X)


def test_check_tags_static_after_fit(y_X_factory):
    """Test check_tags_static_after_fit passes for valid transformer."""
    y, X = y_X_factory(length=50, n_targets=1, n_features=2, seed=42)
    transformer = LagTransformer(lag=3)
    transformer.fit(X[:40], y[:40])

    # Should not raise
    check_tags_static_after_fit(transformer, X[:40], y[:40])


def test_check_tags_match_capabilities(y_X_factory):
    """Test check_tags_match_capabilities passes for valid transformer."""
    y, X = y_X_factory(length=50, n_targets=1, n_features=2, seed=42)
    transformer = LagTransformer(lag=3)
    transformer.fit(X[:40], y[:40])

    # Should not raise
    check_tags_match_capabilities(transformer, X[:40])
