"""Tests for signal processing transformers.

Tests NumericalIntegrator and NumericalDifferentiator using both
the check generator pattern and transformer-specific tests.
"""

from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest
from sklearn.base import clone

from conftest import run_checks
from yohou.preprocessing.signal import (
    NumericalDifferentiator,
    NumericalIntegrator,
)
from yohou.testing import _yield_yohou_transformer_checks

# Generate high-frequency data for signal processing tests
# Use 1ms intervals to simulate high-frequency signals
LENGTH = 500


def create_signal_data(length: int = 500, seed: int = 42) -> pl.DataFrame:
    """Create high-frequency signal data for testing.

    Parameters
    ----------
    length : int
        Number of samples.
    seed : int
        Random seed.

    Returns
    -------
    pl.DataFrame
        DataFrame with time column and signal columns.

    """
    np.random.seed(seed)
    time = [datetime(2021, 1, 1) + timedelta(milliseconds=i) for i in range(length)]
    return pl.DataFrame({
        "time": time,
        "signal_a": np.sin(np.linspace(0, 4 * np.pi, length)) + np.random.randn(length) * 0.1,
        "signal_b": np.cos(np.linspace(0, 4 * np.pi, length)) + np.random.randn(length) * 0.1,
    })


@pytest.fixture(scope="module")
def signal_data() -> pl.DataFrame:
    """Module-scoped fixture providing signal test data."""
    return create_signal_data(length=LENGTH)


class TestIntegrateTransformer:
    """Tests for NumericalIntegrator."""

    @pytest.mark.parametrize(
        "transformer,expected_failures",
        [
            (
                NumericalIntegrator(method="cumulative_trapezoid"),
                [
                    "check_inverse_transform_identity",
                    "check_inverse_transform_round_trip",
                    "check_inverse_observe_transform_identity",
                ],
            ),
            (
                NumericalIntegrator(method="cumulative_simpson"),
                [
                    "check_inverse_transform_identity",
                    "check_inverse_transform_round_trip",
                    "check_inverse_observe_transform_identity",
                ],
            ),
        ],
        ids=["Integrate_trapezoid", "Integrate_simpson"],
    )
    def test_systematic_checks(
        self,
        transformer,
        expected_failures,
        time_series_train_test_factory,
    ):
        """Run all applicable checks for NumericalIntegrator."""
        min_horizon = 10
        X_train, X_test = time_series_train_test_factory(
            train_length=min_horizon + 100,
            test_length=min_horizon + 50,
        )

        transformer_fitted = clone(transformer)
        transformer_fitted.fit(X_train)

        run_checks(
            transformer_fitted,
            _yield_yohou_transformer_checks(transformer_fitted, X_train, None, X_test),
            expected_failures=set(expected_failures),
        )

    def test_integrate_basic(self, signal_data):
        """Test basic integration functionality."""
        transformer = NumericalIntegrator()
        transformer.fit(signal_data)

        X_t = transformer.transform(signal_data)

        # Should have time column
        assert "time" in X_t.columns
        # Should have transformed column names
        assert "signal_a_integrated" in X_t.columns
        assert "signal_b_integrated" in X_t.columns
        # Should have same length
        assert len(X_t) == len(signal_data)

    def test_integrate_detects_sampling_interval(self, signal_data):
        """Test that integration detects correct sampling interval."""
        transformer = NumericalIntegrator()
        transformer.fit(signal_data)

        # 1ms interval = 0.001 seconds
        assert abs(transformer.sampling_interval_ - 0.001) < 1e-9

    def test_integrate_cumulative_trapezoid(self):
        """Test cumulative trapezoid integration."""
        # Create simple linear signal where integral is known
        time = [datetime(2021, 1, 1) + timedelta(seconds=i) for i in range(10)]
        X_linear = pl.DataFrame({
            "time": time,
            "value": [1.0] * 10,  # Constant 1, integral should be t
        })

        transformer = NumericalIntegrator(method="cumulative_trapezoid")
        transformer.fit(X_linear)
        X_t = transformer.transform(X_linear)

        # First value should be 0 (initial=0)
        assert X_t["value_integrated"][0] == 0.0
        # Last value should be approximately 9 (integral of constant 1 from 0 to 9)
        assert abs(X_t["value_integrated"][-1] - 9.0) < 0.1

    def test_integrate_cumulative_simpson(self):
        """Test cumulative simpson integration."""
        time = [datetime(2021, 1, 1) + timedelta(seconds=i) for i in range(11)]
        X_linear = pl.DataFrame({
            "time": time,
            "value": [1.0] * 11,
        })

        transformer = NumericalIntegrator(method="cumulative_simpson")
        transformer.fit(X_linear)
        X_t = transformer.transform(X_linear)

        # Should have time column and integrated values
        assert "time" in X_t.columns
        assert len(X_t) == len(X_linear)

    def test_integrate_streaming_state(self):
        """Test that integration state is preserved between transform calls."""
        # Create data with 1000Hz sampling rate (1ms interval)
        time = [datetime(2021, 1, 1) + timedelta(milliseconds=i) for i in range(200)]
        X_signal = pl.DataFrame({
            "time": time,
            "value": np.ones(200),  # Constant 1.0, easy to verify integral
        })

        transformer = NumericalIntegrator()
        transformer.fit(X_signal[:50])

        # Transform first chunk
        X_t1 = transformer.transform(X_signal[:100])
        # Transform second chunk - should continue from first
        X_t2 = transformer.transform(X_signal[100:])

        # Integral of constant 1.0 over dt=0.001s grows linearly: integral(t) = t
        # At end of first chunk (t=0.1s), integral should be ~0.099 (trapezoid)
        # At end of second chunk (t=0.2s), integral should be ~0.199
        last_val_chunk1 = X_t1["value_integrated"][-1]
        first_val_chunk2 = X_t2["value_integrated"][0]
        last_val_chunk2 = X_t2["value_integrated"][-1]

        # Second chunk should start higher than first chunk ended (state preserved)
        assert first_val_chunk2 > last_val_chunk1
        # Final value should be approximately 2x the first chunk's end
        assert last_val_chunk2 > last_val_chunk1 * 1.5

    def test_integrate_rewind_state(self):
        """Test rewind() clears accumulated offset."""
        time = [datetime(2021, 1, 1) + timedelta(milliseconds=i) for i in range(100)]
        X_signal = pl.DataFrame({
            "time": time,
            "value": np.ones(100),
        })

        transformer = NumericalIntegrator()
        transformer.fit(X_signal)

        # Transform first chunk
        X_t1 = transformer.transform(X_signal)

        # Rewind state and transform again
        transformer.rewind(X_signal)
        X_t2 = transformer.transform(X_signal)

        # Both should produce same result after rewind
        np.testing.assert_array_almost_equal(X_t1["value_integrated"].to_numpy(), X_t2["value_integrated"].to_numpy())

    def test_integrate_inverse_transform(self, signal_data):
        """Test inverse transform (differentiation)."""
        transformer = NumericalIntegrator()
        transformer.fit(signal_data)

        X_t = transformer.transform(signal_data)
        X_inv = transformer.inverse_transform(X_t)

        # Should have original column names
        assert "signal_a" in X_inv.columns
        assert "signal_b" in X_inv.columns
        # Should have time column
        assert "time" in X_inv.columns
        # Should have same length
        assert len(X_inv) == len(X_t)

    def test_integrate_feature_names(self, signal_data):
        """Test get_feature_names_out."""
        transformer = NumericalIntegrator()
        transformer.fit(signal_data)

        feature_names = transformer.get_feature_names_out()
        assert feature_names == ["signal_a_integrated", "signal_b_integrated"]

    def test_clone(self):
        """Test that NumericalIntegrator can be cloned."""
        transformer = NumericalIntegrator(method="cumulative_simpson")
        cloned = clone(transformer)
        assert cloned.method == transformer.method

    def test_invalid_method(self, signal_data):
        """Test that invalid method raises ValueError."""
        with pytest.raises(ValueError, match="method"):
            NumericalIntegrator(method="invalid_method").fit(signal_data)

    def test_single_column(self):
        """Test integration with a single column."""
        time = [datetime(2021, 1, 1) + timedelta(milliseconds=i) for i in range(100)]
        X_single = pl.DataFrame({"time": time, "value": np.random.randn(100)})
        transformer = NumericalIntegrator()
        transformer.fit(X_single)
        X_t = transformer.transform(X_single)
        assert X_t.columns == ["time", "value_integrated"]

    def test_requires_two_samples(self):
        """Test that single-row input raises ValueError."""
        time = [datetime(2021, 1, 1)]
        X_single = pl.DataFrame({"time": time, "value": [1.0]})
        transformer = NumericalIntegrator()
        with pytest.raises(ValueError, match="2 time points"):
            transformer.fit(X_single)


class TestDifferentiateTransformer:
    """Tests for NumericalDifferentiator."""

    @pytest.mark.parametrize(
        "transformer,expected_failures",
        [
            (
                NumericalDifferentiator(order=1),
                [
                    "check_inverse_transform_identity",
                    "check_inverse_transform_round_trip",
                    "check_inverse_observe_transform_identity",
                ],
            ),
            (
                NumericalDifferentiator(order=2),
                [
                    "check_inverse_transform_identity",
                    "check_inverse_transform_round_trip",
                    "check_inverse_observe_transform_identity",
                ],
            ),
        ],
        ids=["Differentiate_order1", "Differentiate_order2"],
    )
    def test_systematic_checks(
        self,
        transformer,
        expected_failures,
        time_series_train_test_factory,
    ):
        """Run all applicable checks for NumericalDifferentiator."""
        min_horizon = 10
        X_train, X_test = time_series_train_test_factory(
            train_length=min_horizon + 100,
            test_length=min_horizon + 50,
        )

        transformer_fitted = clone(transformer)
        transformer_fitted.fit(X_train)

        run_checks(
            transformer_fitted,
            _yield_yohou_transformer_checks(transformer_fitted, X_train, None, X_test),
            expected_failures=set(expected_failures),
        )


class TestDifferentiateTransformerBehavior:
    """Tests for NumericalDifferentiator."""

    def test_differentiate_basic(self, signal_data):
        """Test basic differentiation functionality."""
        transformer = NumericalDifferentiator()
        transformer.fit(signal_data)

        X_t = transformer.transform(signal_data)

        # Should have time column
        assert "time" in X_t.columns
        # Should have transformed column names
        assert "signal_a_differentiated" in X_t.columns
        assert "signal_b_differentiated" in X_t.columns
        # Should have same length
        assert len(X_t) == len(signal_data)

    def test_differentiate_detects_sampling_interval(self, signal_data):
        """Test that differentiation detects correct sampling interval."""
        transformer = NumericalDifferentiator()
        transformer.fit(signal_data)

        # 1ms interval = 0.001 seconds
        assert abs(transformer.sampling_interval_ - 0.001) < 1e-9

    def test_differentiate_constant(self):
        """Test differentiation of constant signal gives zero."""
        time = [datetime(2021, 1, 1) + timedelta(seconds=i) for i in range(10)]
        X_const = pl.DataFrame({
            "time": time,
            "value": [5.0] * 10,
        })

        transformer = NumericalDifferentiator()
        transformer.fit(X_const)
        X_t = transformer.transform(X_const)

        # Derivative of constant should be 0
        diff_values = X_t["value_differentiated"].to_numpy()
        assert np.allclose(diff_values, 0.0)

    def test_differentiate_linear(self):
        """Test differentiation of linear signal gives constant."""
        time = [datetime(2021, 1, 1) + timedelta(seconds=i) for i in range(10)]
        X_linear = pl.DataFrame({
            "time": time,
            "value": [float(i) for i in range(10)],  # y = t
        })

        transformer = NumericalDifferentiator()
        transformer.fit(X_linear)
        X_t = transformer.transform(X_linear)

        # Derivative of t should be 1
        diff_values = X_t["value_differentiated"].to_numpy()
        assert np.allclose(diff_values, 1.0)

    def test_differentiate_edge_order_1(self, signal_data):
        """Test differentiation with order=1."""
        transformer = NumericalDifferentiator(order=1)
        transformer.fit(signal_data)
        X_t = transformer.transform(signal_data)

        # Should complete without error
        assert len(X_t) == len(signal_data)

    def test_differentiate_edge_order_2(self, signal_data):
        """Test differentiation with order=2."""
        transformer = NumericalDifferentiator(order=2)
        transformer.fit(signal_data)
        X_t = transformer.transform(signal_data)

        # Should complete without error
        assert len(X_t) == len(signal_data)

    def test_differentiate_inverse_transform(self, signal_data):
        """Test inverse transform (integration)."""
        transformer = NumericalDifferentiator()
        transformer.fit(signal_data)

        X_t = transformer.transform(signal_data)
        X_inv = transformer.inverse_transform(X_t)

        # Should have original column names
        assert "signal_a" in X_inv.columns
        assert "signal_b" in X_inv.columns
        # Should have time column
        assert "time" in X_inv.columns
        # Should have same length
        assert len(X_inv) == len(X_t)

    def test_differentiate_feature_names(self, signal_data):
        """Test get_feature_names_out."""
        transformer = NumericalDifferentiator()
        transformer.fit(signal_data)

        feature_names = transformer.get_feature_names_out()
        assert feature_names == ["signal_a_differentiated", "signal_b_differentiated"]

    def test_clone(self):
        """Test that NumericalDifferentiator can be cloned."""
        transformer = NumericalDifferentiator(order=2)
        cloned = clone(transformer)

        assert cloned.order == transformer.order

    def test_invalid_edge_order(self, signal_data):
        """Test that invalid order raises error."""
        with pytest.raises(ValueError, match="order"):
            NumericalDifferentiator(order=0).fit(signal_data)

        with pytest.raises(ValueError, match="order"):
            NumericalDifferentiator(order=3).fit(signal_data)

    def test_single_column(self):
        """Test differentiation with single column."""
        time = [datetime(2021, 1, 1) + timedelta(milliseconds=i) for i in range(100)]
        X_single = pl.DataFrame({
            "time": time,
            "value": np.random.randn(100),
        })

        transformer = NumericalDifferentiator()
        transformer.fit(X_single)
        X_t = transformer.transform(X_single)

        assert X_t.columns == ["time", "value_differentiated"]

    def test_requires_two_samples(self):
        """Test that differentiation requires at least 2 samples."""
        time = [datetime(2021, 1, 1)]
        X_single = pl.DataFrame({
            "time": time,
            "value": [1.0],
        })

        transformer = NumericalDifferentiator()
        with pytest.raises(ValueError, match="2 time points"):
            transformer.fit(X_single)


class TestSignalRoundTrip:
    """Round-trip tests for signal processing transformers."""

    def test_integrate_differentiate(self):
        """Test that differentiate(integrate(x)) approx x."""
        time = [datetime(2021, 1, 1) + timedelta(milliseconds=i) for i in range(100)]
        X_signal = pl.DataFrame({
            "time": time,
            "value": np.sin(np.linspace(0, 2 * np.pi, 100)),
        })

        integrate = NumericalIntegrator()
        differentiate = NumericalDifferentiator()

        integrate.fit(X_signal)
        differentiate.fit(X_signal)

        X_integrated = integrate.transform(X_signal)
        X_roundtrip = integrate.inverse_transform(X_integrated)

        original = X_signal["value"].to_numpy()[10:-10]
        recovered = X_roundtrip["value"].to_numpy()[10:-10]

        correlation = np.corrcoef(original, recovered)[0, 1]
        assert correlation > 0.9

    def test_differentiate_integrate(self):
        """Test that integrate(differentiate(x)) approx x (up to constant)."""
        time = [datetime(2021, 1, 1) + timedelta(milliseconds=i) for i in range(100)]
        X_signal = pl.DataFrame({
            "time": time,
            "value": np.sin(np.linspace(0, 2 * np.pi, 100)),
        })

        differentiate = NumericalDifferentiator()

        differentiate.fit(X_signal)

        X_differentiated = differentiate.transform(X_signal)
        X_roundtrip = differentiate.inverse_transform(X_differentiated)

        original = X_signal["value"].to_numpy()
        recovered = X_roundtrip["value"].to_numpy()

        original_centered = original - np.mean(original)
        recovered_centered = recovered - np.mean(recovered)

        correlation = np.corrcoef(original_centered[10:-10], recovered_centered[10:-10])[0, 1]
        assert correlation > 0.9
