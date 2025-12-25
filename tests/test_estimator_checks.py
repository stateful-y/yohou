"""sklearn compatibility testing with hybrid approach.

Tests that yohou transformers are compatible with sklearn patterns while
also validating time series-specific requirements.
"""

import polars as pl
import pytest
from sklearn.base import clone

# ============================================================================
# TIME SERIES-SPECIFIC CHECKS
# ============================================================================


def check_time_column_required(estimator, time_series_factory):
    """Check that transformer requires 'time' column.

    Yohou transformers should raise an error when given data
    without a 'time' column.
    """
    X = time_series_factory(length=20)
    # Remove time column
    X_no_time = X.select(pl.all().exclude("time"))

    with pytest.raises((ValueError, KeyError, Exception)):
        clone(estimator).fit(X_no_time)


def check_observation_horizon_property(estimator, time_series_factory):
    """Check observation_horizon property exists after fit."""
    X = time_series_factory(length=20)
    estimator_clone = clone(estimator)
    estimator_clone.fit(X)

    assert hasattr(estimator_clone, "observation_horizon"), (
        "Fitted transformer must have observation_horizon property"
    )

    horizon = estimator_clone.observation_horizon
    assert isinstance(horizon, int), f"observation_horizon must be int, got {type(horizon)}"
    assert horizon >= 0, f"observation_horizon must be non-negative, got {horizon}"


def check_update_reset_contract(estimator, time_series_factory):
    """Check update() and reset() methods exist and work."""
    X = time_series_factory(length=20)
    X_new = time_series_factory(length=10, seed=99)

    estimator_clone = clone(estimator)
    estimator_clone.fit(X)

    # update should work
    assert hasattr(estimator_clone, "update"), "Transformer must have update() method"
    estimator_clone.update(X_new)

    # reset should work
    assert hasattr(estimator_clone, "reset"), "Transformer must have reset() method"
    estimator_clone.reset(X_new)


def check_polars_dataframe_io(estimator, time_series_factory):
    """Check transformer accepts and returns polars DataFrames."""
    X = time_series_factory(length=20)
    estimator_clone = clone(estimator)

    # Input should be polars DataFrame
    assert isinstance(X, pl.DataFrame), "Test data should be polars DataFrame"

    estimator_clone.fit(X)
    X_trans = estimator_clone.transform(X)

    # Output should also be polars DataFrame
    assert isinstance(X_trans, pl.DataFrame), (
        f"transform() must return polars DataFrame, got {type(X_trans)}"
    )

    # Should have time column
    assert "time" in X_trans.columns, "transform() output must contain 'time' column"


# ============================================================================
# PARAMETRIZED TESTS FOR ALL TRANSFORMERS
# ============================================================================


@pytest.mark.parametrize(
    "transformer_name",
    ["SeasonalDifferencing", "SeasonalLogDifferencing", "LagTransformer"],
)
def test_yohou_specific_checks(
    transformer_name,
    transformer_registry,
    time_series_factory,
):
    """Run time series-specific checks for all transformers."""
    config = transformer_registry[transformer_name]
    estimator = config["transformer"]

    # Run all time series-specific checks
    check_time_column_required(estimator, time_series_factory)
    check_observation_horizon_property(estimator, time_series_factory)
    check_update_reset_contract(estimator, time_series_factory)
    check_polars_dataframe_io(estimator, time_series_factory)


def test_all_transformers_clonable(transformer_registry):
    """Test all transformers can be cloned via sklearn.base.clone."""
    for name, config in transformer_registry.items():
        transformer = config["transformer"]

        # Clone should work
        transformer_clone = clone(transformer)

        # Should be different objects
        assert transformer_clone is not transformer, f"{name}: clone should create new object"

        # Parameters should match
        assert transformer.get_params() == transformer_clone.get_params(), (
            f"{name}: clone parameters mismatch"
        )


def test_all_transformers_fit_transform(transformer_registry, time_series_factory):
    """Test all transformers have working fit_transform."""
    X = time_series_factory(length=50)

    for name, config in transformer_registry.items():
        transformer = clone(config["transformer"])
        tags = config.get("tags", {})

        # Make data positive for log transforms
        if tags.get("requires_positive_X", False):
            X_pos = X.select([pl.col("time"), (pl.all().exclude("time") + 100.0)])
            X_test = X_pos
        else:
            X_test = X

        # fit_transform should work
        X_trans = transformer.fit_transform(X_test)

        # Output should be DataFrame with time column
        assert isinstance(X_trans, pl.DataFrame), f"{name}: fit_transform should return DataFrame"
        assert "time" in X_trans.columns, f"{name}: fit_transform output should have 'time' column"


def test_all_transformers_get_feature_names_out(transformer_registry, time_series_factory):
    """Test all transformers implement get_feature_names_out."""
    X = time_series_factory(length=50)

    for name, config in transformer_registry.items():
        transformer = clone(config["transformer"])
        tags = config.get("tags", {})

        # Make data positive for log transforms
        if tags.get("requires_positive_X", False):
            X_test = X.select([pl.col("time"), (pl.all().exclude("time") + 100.0)])
        else:
            X_test = X

        transformer.fit(X_test)

        # get_feature_names_out should work
        feature_names = transformer.get_feature_names_out()

        assert feature_names is not None, f"{name}: get_feature_names_out should not return None"
        assert len(feature_names) > 0, f"{name}: get_feature_names_out should return non-empty list"


def test_transformers_with_edge_cases(
    transformer_registry,
    edge_case_datasets_factory,
):
    """Test all transformers handle edge cases appropriately."""
    for transformer_name, config in transformer_registry.items():
        transformer = clone(config["transformer"])
        edge_cases = edge_case_datasets_factory(observation_horizon=5)

        # Empty data should raise
        with pytest.raises((ValueError, Exception)):
            transformer.fit(edge_cases["empty"])


# ============================================================================
# SKLEARN COMPATIBILITY NOTES
# ============================================================================

# Note: We don't use sklearn's parametrize_with_checks here because:
# 1. Yohou transformers use polars DataFrames, not numpy arrays
# 2. Time series-specific requirements (time column, observation_horizon)
# 3. update/reset methods are not part of sklearn's API
#
# Known sklearn check incompatibilities:
# - check_transformer_data_not_an_array: We require polars DataFrames
# - check_methods_subset_invariance: Time series order matters
# - check_fit2d_1sample: Time series need minimum length > observation_horizon
# - check_fit2d_1feature: We require 'time' column + at least one feature
#
# These incompatibilities are by design and reflect time series requirements.
