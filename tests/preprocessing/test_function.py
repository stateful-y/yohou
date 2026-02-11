"""Tests for function-based transformers.

Tests FunctionTransformer using both the check generator pattern
and transformer-specific tests.
"""

from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal
from sklearn.base import clone

from conftest import run_checks
from yohou.preprocessing.function import FunctionTransformer
from yohou.testing import _yield_yohou_transformer_checks


def create_positive_data(length: int = 50, seed: int = 42) -> pl.DataFrame:
    """Create positive-valued data for log transform testing.

    Parameters
    ----------
    length : int
        Number of samples.
    seed : int
        Random seed.

    Returns
    -------
    pl.DataFrame
        DataFrame with time column and positive-valued columns.

    """
    np.random.seed(seed)
    time = [datetime(2021, 1, 1) + timedelta(days=i) for i in range(length)]
    return pl.DataFrame({
        "time": time,
        "value_a": np.abs(np.random.randn(length)) + 1.0,
        "value_b": np.abs(np.random.randn(length)) + 1.0,
    })


class TestFunctionTransformerSystematic:
    """Systematic check generator tests for FunctionTransformer."""

    @pytest.mark.parametrize(
        "transformer,expected_failures",
        [
            # Identity transform (no func)
            (FunctionTransformer(), []),
            # Log/Exp transform - inverse should work
            (FunctionTransformer(func=np.log, inverse_func=np.exp), []),
            # Single func without inverse - will fail inverse check
            (
                FunctionTransformer(func=np.square),
                [
                    "check_inverse_transform_identity",
                    "check_inverse_transform_round_trip",
                    "check_inverse_observe_transform_identity",
                ],
            ),
        ],
        ids=["identity", "log_exp", "square_no_inverse"],
    )
    def test_systematic_checks(
        self,
        transformer,
        expected_failures,
        time_series_train_test_factory,
    ):
        """Run all applicable checks for FunctionTransformer."""
        # Generate continuous train/test data with positive values
        X_train, X_test = time_series_train_test_factory(
            train_length=60,
            test_length=30,
        )
        # Ensure positive values for log transform compatibility
        X_train = X_train.select([
            pl.col("time"),
            (pl.all().exclude("time").abs() + 1.0),
        ])
        X_test = X_test.select([
            pl.col("time"),
            (pl.all().exclude("time").abs() + 1.0),
        ])

        # Fit transformer
        transformer_fitted = clone(transformer)
        transformer_fitted.fit(X_train)

        # Run all checks from generator
        run_checks(
            transformer_fitted,
            _yield_yohou_transformer_checks(transformer_fitted, X_train, None, X_test),
            expected_failures=set(expected_failures),
        )


class TestFunctionTransformerBasic:
    """Basic functionality tests for FunctionTransformer."""

    def test_identity_transform(self):
        """Test identity transformation (func=None)."""
        X = create_positive_data()
        transformer = FunctionTransformer()
        transformer.fit(X)

        X_t = transformer.transform(X)

        # Should be identical
        assert_frame_equal(X, X_t)

    def test_log_transform(self):
        """Test log transformation."""
        X = create_positive_data()
        transformer = FunctionTransformer(func=np.log, inverse_func=np.exp, check_inverse=False)
        transformer.fit(X)

        X_t = transformer.transform(X)

        # Time should be preserved
        assert_frame_equal(X_t.select("time"), X.select("time"))

        # Values should be log-transformed
        for col in X.columns:
            if col == "time":
                continue
            np.testing.assert_allclose(
                X_t[col].to_numpy(),
                np.log(X[col].to_numpy()),
                rtol=1e-10,
            )

    def test_inverse_transform(self):
        """Test inverse transformation round-trip."""
        X = create_positive_data()
        transformer = FunctionTransformer(func=np.log, inverse_func=np.exp, check_inverse=False)
        transformer.fit(X)

        X_t = transformer.transform(X)
        X_inv = transformer.inverse_transform(X_t)

        # Should recover original values
        for col in X.columns:
            if col == "time":
                continue
            np.testing.assert_allclose(
                X_inv[col].to_numpy(),
                X[col].to_numpy(),
                rtol=1e-5,
            )

    def test_inverse_transform_identity(self):
        """Test identity inverse transform."""
        X = create_positive_data()
        transformer = FunctionTransformer()  # No funcs = identity
        transformer.fit(X)

        X_inv = transformer.inverse_transform(X)

        assert_frame_equal(X, X_inv)

    def test_check_inverse_warning(self):
        """Test that check_inverse raises warning for non-invertible funcs."""
        X = create_positive_data()

        # Define non-inverse functions
        def bad_inverse(x, **kwargs):
            return x * 2  # Wrong inverse

        transformer = FunctionTransformer(func=np.log, inverse_func=bad_inverse, check_inverse=True)

        with pytest.warns(UserWarning, match="not strictly inverse"):
            transformer.fit(X)

    def test_check_inverse_disabled(self):
        """Test that check_inverse=False skips the check."""
        X = create_positive_data()

        def bad_inverse(x, **kwargs):
            return x * 2

        transformer = FunctionTransformer(func=np.log, inverse_func=bad_inverse, check_inverse=False)

        # Should not raise warning
        transformer.fit(X)  # No warning


class TestFunctionTransformerKwargs:
    """Test keyword arguments passing."""

    def test_kw_args(self):
        """Test that kw_args are passed to func."""
        X = create_positive_data()

        def power_func(df, power=2):
            # Apply power to each column
            return pl.DataFrame({col: df[col].to_numpy() ** power for col in df.columns})

        transformer = FunctionTransformer(func=power_func, kw_args={"power": 3})
        transformer.fit(X)

        X_t = transformer.transform(X)

        # Values should be cubed
        for col in X.columns:
            if col == "time":
                continue
            np.testing.assert_allclose(
                X_t[col].to_numpy(),
                X[col].to_numpy() ** 3,
                rtol=1e-10,
            )

    def test_inv_kw_args(self):
        """Test that inv_kw_args are passed to inverse_func."""
        X = create_positive_data()

        def power_func(df, power=2):
            return pl.DataFrame({col: df[col].to_numpy() ** power for col in df.columns})

        def root_func(df, power=2):
            return pl.DataFrame({col: df[col].to_numpy() ** (1 / power) for col in df.columns})

        transformer = FunctionTransformer(
            func=power_func,
            inverse_func=root_func,
            kw_args={"power": 3},
            inv_kw_args={"power": 3},
            check_inverse=False,
        )
        transformer.fit(X)

        X_t = transformer.transform(X)
        X_inv = transformer.inverse_transform(X_t)

        for col in X.columns:
            if col == "time":
                continue
            np.testing.assert_allclose(
                X_inv[col].to_numpy(),
                X[col].to_numpy(),
                rtol=1e-5,
            )


class TestFunctionTransformerFeatureNames:
    """Test feature name handling."""

    def test_feature_names_one_to_one(self):
        """Test one-to-one feature names."""
        X = create_positive_data()
        transformer = FunctionTransformer(feature_names_out="one-to-one")
        transformer.fit(X)

        names = transformer.get_feature_names_out()

        assert list(names) == ["value_a", "value_b"]

    def test_feature_names_callable(self):
        """Test callable feature names."""
        X = create_positive_data()

        def custom_names(transformer, input_features):
            return [f"transformed_{f}" for f in input_features]

        transformer = FunctionTransformer(feature_names_out=custom_names)
        transformer.fit(X)

        names = transformer.get_feature_names_out()

        assert list(names) == ["transformed_value_a", "transformed_value_b"]

    def test_feature_names_none(self):
        """Test default feature names (None)."""
        X = create_positive_data()
        transformer = FunctionTransformer()
        transformer.fit(X)

        names = transformer.get_feature_names_out()

        # Should return input features by default
        assert list(names) == ["value_a", "value_b"]


class TestFunctionTransformerNumpyReturn:
    """Test handling of numpy array returns from func."""

    def test_numpy_return_2d(self):
        """Test func returning numpy 2D array."""
        X = create_positive_data()

        def numpy_func(df):
            return df.to_numpy() * 2

        transformer = FunctionTransformer(func=numpy_func)
        transformer.fit(X)

        X_t = transformer.transform(X)

        assert "time" in X_t.columns
        assert len(X_t) == len(X)

    def test_numpy_return_1d(self):
        """Test func returning numpy 1D array (expanded to 2D)."""
        # Single column
        time = [datetime(2021, 1, 1) + timedelta(days=i) for i in range(50)]
        X = pl.DataFrame({
            "time": time,
            "value": np.random.randn(50).tolist(),
        })

        def numpy_func(df):
            # Return 1D array
            return df.to_numpy().flatten() * 2

        transformer = FunctionTransformer(func=numpy_func)
        transformer.fit(X)

        X_t = transformer.transform(X)

        assert "time" in X_t.columns
        assert len(X_t) == len(X)


class TestFunctionTransformerSkleanCompatibility:
    """Test sklearn compatibility."""

    def test_clone(self):
        """Test cloning preserves parameters."""
        transformer = FunctionTransformer(func=np.log, inverse_func=np.exp, check_inverse=False)
        cloned = clone(transformer)

        assert cloned.func is transformer.func
        assert cloned.inverse_func is transformer.inverse_func
        assert cloned.check_inverse == transformer.check_inverse

    def test_get_params(self):
        """Test get_params returns all parameters."""
        transformer = FunctionTransformer(func=np.log, inverse_func=np.exp, check_inverse=True)
        params = transformer.get_params()

        assert params["func"] is np.log
        assert params["inverse_func"] is np.exp
        assert params["check_inverse"] is True

    def test_set_params(self):
        """Test set_params modifies parameters."""
        transformer = FunctionTransformer()
        transformer.set_params(func=np.log, check_inverse=False)

        assert transformer.func is np.log
        assert transformer.check_inverse is False


class TestFunctionTransformerStatelessness:
    """Test that FunctionTransformer is stateless."""

    def test_observation_horizon_zero(self):
        """Test that observation_horizon is 0 (stateless)."""
        X = create_positive_data()
        transformer = FunctionTransformer(func=np.log)
        transformer.fit(X)

        assert transformer.observation_horizon == 0

    def test_no_memory_stored(self):
        """Test that _X_observed is empty (stateless)."""
        X = create_positive_data()
        transformer = FunctionTransformer(func=np.log)
        transformer.fit(X)

        # Stateless transformers should have empty _X_observed
        assert len(transformer._X_observed) == 0
