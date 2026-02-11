"""sklearn compatibility testing with hybrid approach.

Tests that yohou transformers are compatible with sklearn patterns while
also validating time series-specific requirements.
"""

import polars as pl
import pytest
from sklearn.base import clone

# TIME SERIES-SPECIFIC CHECKS


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

    assert hasattr(estimator_clone, "observation_horizon"), "Fitted transformer must have observation_horizon property"

    horizon = estimator_clone.observation_horizon
    assert isinstance(horizon, int), f"observation_horizon must be int, got {type(horizon)}"
    assert horizon >= 0, f"observation_horizon must be non-negative, got {horizon}"


def check_observe_rewind_contract(estimator, time_series_train_test_factory):
    """Check observe() and rewind() methods exist and work."""
    X, X_new = time_series_train_test_factory(train_length=20, test_length=10)

    estimator_clone = clone(estimator)
    estimator_clone.fit(X)

    # observe should work
    assert hasattr(estimator_clone, "observe"), "Transformer must have observe() method"
    estimator_clone.observe(X_new)

    # rewind should work
    assert hasattr(estimator_clone, "rewind"), "Transformer must have rewind() method"
    estimator_clone.rewind(X_new)


def check_polars_dataframe_io(estimator, time_series_factory):
    """Check transformer accepts and returns polars DataFrames."""
    X = time_series_factory(length=20)
    estimator_clone = clone(estimator)

    # Input should be polars DataFrame
    assert isinstance(X, pl.DataFrame), "Test data should be polars DataFrame"

    estimator_clone.fit(X)
    X_trans = estimator_clone.transform(X)

    # Output should also be polars DataFrame
    assert isinstance(X_trans, pl.DataFrame), f"transform() must return polars DataFrame, got {type(X_trans)}"

    # Should have time column
    assert "time" in X_trans.columns, "transform() output must contain 'time' column"


class TestYohouSpecificChecks:
    """Time series-specific checks for yohou transformers."""

    @pytest.mark.parametrize(
        "transformer_name",
        ["SeasonalDifferencing", "SeasonalLogDifferencing", "LagTransformer"],
    )
    def test_yohou_specific_checks(
        self,
        transformer_name,
        transformer_registry,
        time_series_factory,
        time_series_train_test_factory,
    ):
        """Run time series-specific checks for all transformers."""
        config = transformer_registry[transformer_name]
        estimator = config["transformer"]

        # Run all time series-specific checks
        check_time_column_required(estimator, time_series_factory)
        check_observation_horizon_property(estimator, time_series_factory)
        check_observe_rewind_contract(estimator, time_series_train_test_factory)
        check_polars_dataframe_io(estimator, time_series_factory)


class TestTransformerSklearnCompat:
    """Tests for sklearn compatibility of all registered transformers."""

    def test_all_transformers_clonable(self, transformer_registry):
        """Test all transformers can be cloned via sklearn.base.clone."""
        for name, config in transformer_registry.items():
            transformer = config["transformer"]

            # Clone should work
            transformer_clone = clone(transformer)

            # Should be different objects
            assert transformer_clone is not transformer, f"{name}: clone should create new object"

            # Parameters should match
            assert transformer.get_params() == transformer_clone.get_params(), f"{name}: clone parameters mismatch"

    def test_all_transformers_fit_transform(self, transformer_registry, time_series_factory):
        """Test all transformers have working fit_transform."""
        X = time_series_factory(length=50)

        for name, config in transformer_registry.items():
            transformer = clone(config["transformer"])
            config.get("tags", {})

            # Adjust data for transformers with min_value constraints
            sklearn_tags = transformer.__sklearn_tags__()
            if sklearn_tags.input_tags and sklearn_tags.input_tags.min_value is not None:
                # Make data satisfy min_value constraint by adding offset
                offset = max(0.0, sklearn_tags.input_tags.min_value + 1.0)
                X_test = X.select([pl.col("time"), (pl.all().exclude("time") + offset)])
            else:
                X_test = X

            # fit_transform should work
            X_trans = transformer.fit_transform(X_test)

            # Output should be DataFrame with time column
            assert isinstance(X_trans, pl.DataFrame), f"{name}: fit_transform should return DataFrame"
            assert "time" in X_trans.columns, f"{name}: fit_transform output should have 'time' column"

    def test_all_transformers_get_feature_names_out(self, transformer_registry, time_series_factory):
        """Test all transformers implement get_feature_names_out."""
        X = time_series_factory(length=50)

        for name, config in transformer_registry.items():
            transformer = clone(config["transformer"])
            config.get("tags", {})

            # Adjust data for transformers with min_value constraints
            sklearn_tags = transformer.__sklearn_tags__()
            if sklearn_tags.input_tags and sklearn_tags.input_tags.min_value is not None:
                # Make data satisfy min_value constraint by adding offset
                offset = max(0.0, sklearn_tags.input_tags.min_value + 1.0)
                X_test = X.select([pl.col("time"), (pl.all().exclude("time") + offset)])
            else:
                X_test = X

            transformer.fit(X_test)

            # get_feature_names_out should work
            feature_names = transformer.get_feature_names_out()

            assert feature_names is not None, f"{name}: get_feature_names_out should not return None"
            assert len(feature_names) > 0, f"{name}: get_feature_names_out should return non-empty list"

    def test_transformers_with_edge_cases(
        self,
        transformer_registry,
        edge_case_datasets_factory,
    ):
        """Test all transformers handle edge cases appropriately."""
        for _transformer_name, config in transformer_registry.items():
            transformer = clone(config["transformer"])
            edge_cases = edge_case_datasets_factory(observation_horizon=5)

            # Empty data should raise
            with pytest.raises((ValueError, Exception)):
                transformer.fit(edge_cases["empty"])
