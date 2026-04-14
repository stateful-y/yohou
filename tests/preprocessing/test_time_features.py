"""Tests for Fourier and time index feature transformers.

Tests FourierFeatureTransformer and TimeIndexTransformer using both
the systematic check generator pattern and transformer-specific tests.
"""

from datetime import datetime

import numpy as np
import polars as pl
import pytest
from sklearn.base import clone

from conftest import run_checks
from yohou.preprocessing.time_features import FourierFeatureTransformer, TimeIndexTransformer
from yohou.testing import _yield_yohou_transformer_checks


class TestFourierFeatureTransformerSystematic:
    """Systematic check generator tests for FourierFeatureTransformer."""

    @pytest.mark.parametrize(
        "transformer,expected_failures",
        [
            (FourierFeatureTransformer(seasonality=7.0), []),
            (FourierFeatureTransformer(seasonality=7.0, harmonics=[1, 2, 3]), []),
        ],
        ids=["default_harmonic", "multi_harmonic"],
    )
    def test_systematic_checks(
        self,
        transformer,
        expected_failures,
        time_series_train_test_factory,
    ):
        """Run all applicable checks for FourierFeatureTransformer."""
        X_train, X_test = time_series_train_test_factory(
            train_length=60,
            test_length=30,
        )

        transformer_fitted = clone(transformer)
        transformer_fitted.fit(X_train)

        run_checks(
            transformer_fitted,
            _yield_yohou_transformer_checks(transformer_fitted, X_train, None, X_test),
            expected_failures=set(expected_failures),
        )


class TestFourierFeatureTransformerBasic:
    """Basic functionality tests for FourierFeatureTransformer."""

    def test_correct_column_count(self):
        """Test output has 2 * len(harmonics) new columns."""
        time = pl.datetime_range(start=datetime(2020, 1, 1), end=datetime(2020, 1, 15), interval="1d", eager=True)
        X = pl.DataFrame({"time": time, "value": range(len(time))})

        transformer = FourierFeatureTransformer(seasonality=7.0, harmonics=[1, 2, 3])
        transformer.fit(X)
        X_t = transformer.transform(X)

        fourier_cols = [c for c in X_t.columns if c.startswith("fourier_")]
        assert len(fourier_cols) == 6  # 3 harmonics * 2 (sin + cos)

    def test_values_in_range(self):
        """Test sin/cos values are in [-1, 1]."""
        time = pl.datetime_range(start=datetime(2020, 1, 1), end=datetime(2020, 2, 1), interval="1d", eager=True)
        X = pl.DataFrame({"time": time, "value": range(len(time))})

        transformer = FourierFeatureTransformer(seasonality=7.0, harmonics=[1, 2])
        transformer.fit(X)
        X_t = transformer.transform(X)

        for col in X_t.columns:
            if col.startswith("fourier_"):
                values = X_t[col].to_numpy()
                assert np.all(values >= -1.0 - 1e-10)
                assert np.all(values <= 1.0 + 1e-10)

    def test_column_names(self):
        """Test column naming convention."""
        time = pl.datetime_range(start=datetime(2020, 1, 1), end=datetime(2020, 1, 15), interval="1d", eager=True)
        X = pl.DataFrame({"time": time, "value": range(len(time))})

        transformer = FourierFeatureTransformer(seasonality=7.0, harmonics=[1, 2])
        transformer.fit(X)
        X_t = transformer.transform(X)

        assert "fourier_7.0_sin_1" in X_t.columns
        assert "fourier_7.0_cos_1" in X_t.columns
        assert "fourier_7.0_sin_2" in X_t.columns
        assert "fourier_7.0_cos_2" in X_t.columns

    def test_preserves_time_and_original(self):
        """Test time column preserved and original features dropped."""
        time = pl.datetime_range(start=datetime(2020, 1, 1), end=datetime(2020, 1, 15), interval="1d", eager=True)
        X = pl.DataFrame({"time": time, "value": range(len(time))})

        transformer = FourierFeatureTransformer(seasonality=7.0)
        transformer.fit(X)
        X_t = transformer.transform(X)

        assert "time" in X_t.columns
        assert "value" not in X_t.columns


class TestFourierFeatureTransformerHarmonics:
    """Tests for harmonic parameter behavior."""

    def test_default_single_harmonic(self):
        """Test default harmonics=[1] produces 2 columns."""
        time = pl.datetime_range(start=datetime(2020, 1, 1), end=datetime(2020, 1, 15), interval="1d", eager=True)
        X = pl.DataFrame({"time": time, "value": range(len(time))})

        transformer = FourierFeatureTransformer(seasonality=7.0)
        transformer.fit(X)
        X_t = transformer.transform(X)

        fourier_cols = [c for c in X_t.columns if c.startswith("fourier_")]
        assert len(fourier_cols) == 2

    def test_empty_harmonics_raises(self):
        """Test empty harmonics list raises ValueError."""
        time = pl.datetime_range(start=datetime(2020, 1, 1), end=datetime(2020, 1, 15), interval="1d", eager=True)
        X = pl.DataFrame({"time": time, "value": range(len(time))})

        transformer = FourierFeatureTransformer(seasonality=7.0, harmonics=[])
        with pytest.raises(ValueError, match="cannot be empty"):
            transformer.fit(X)

    def test_negative_harmonic_raises(self):
        """Test negative harmonic raises ValueError."""
        time = pl.datetime_range(start=datetime(2020, 1, 1), end=datetime(2020, 1, 15), interval="1d", eager=True)
        X = pl.DataFrame({"time": time, "value": range(len(time))})

        transformer = FourierFeatureTransformer(seasonality=7.0, harmonics=[-1])
        with pytest.raises(ValueError, match="positive integers"):
            transformer.fit(X)

    def test_nyquist_limit_raises(self):
        """Test harmonic exceeding Nyquist limit raises ValueError."""
        time = pl.datetime_range(start=datetime(2020, 1, 1), end=datetime(2020, 1, 15), interval="1d", eager=True)
        X = pl.DataFrame({"time": time, "value": range(len(time))})

        transformer = FourierFeatureTransformer(seasonality=7.0, harmonics=[4])
        with pytest.raises(ValueError, match="Nyquist"):
            transformer.fit(X)


class TestFourierFeatureTransformerNonIntegerSeasonality:
    """Tests for non-integer seasonality periods."""

    def test_non_integer_period(self):
        """Test non-integer seasonality works correctly."""
        time = pl.datetime_range(start=datetime(2020, 1, 1), end=datetime(2021, 1, 1), interval="1d", eager=True)
        X = pl.DataFrame({"time": time, "value": range(len(time))})

        transformer = FourierFeatureTransformer(seasonality=365.25, harmonics=[1])
        transformer.fit(X)
        X_t = transformer.transform(X)

        assert "fourier_365.25_sin_1" in X_t.columns
        assert "fourier_365.25_cos_1" in X_t.columns
        assert len(X_t) == len(X)


class TestFourierFeatureTransformerContinuity:
    """Tests for boundary continuity of Fourier features."""

    def test_periodicity(self):
        """Test that values repeat approximately after one full period."""
        time = pl.datetime_range(start=datetime(2020, 1, 1), end=datetime(2020, 1, 22), interval="1d", eager=True)
        X = pl.DataFrame({"time": time, "value": range(len(time))})

        transformer = FourierFeatureTransformer(seasonality=7.0, harmonics=[1])
        transformer.fit(X)
        X_t = transformer.transform(X)

        sin_vals = X_t["fourier_7.0_sin_1"].to_numpy()
        np.testing.assert_allclose(sin_vals[0], sin_vals[7], atol=1e-10)
        np.testing.assert_allclose(sin_vals[0], sin_vals[14], atol=1e-10)


class TestFourierFeatureTransformerEdgeCases:
    """Edge case tests for FourierFeatureTransformer."""

    def test_single_row(self):
        """Test transformer works with two-row DataFrame (minimum for interval detection)."""
        X = pl.DataFrame({
            "time": [datetime(2020, 1, 1), datetime(2020, 1, 2)],
            "value": [1.0, 2.0],
        })

        transformer = FourierFeatureTransformer(seasonality=7.0)
        transformer.fit(X)
        X_t = transformer.transform(X)

        assert len(X_t) == 2
        assert "fourier_7.0_sin_1" in X_t.columns

    def test_get_feature_names_out(self):
        """Test get_feature_names_out returns correct names."""
        time = pl.datetime_range(start=datetime(2020, 1, 1), end=datetime(2020, 1, 15), interval="1d", eager=True)
        X = pl.DataFrame({"time": time, "value": range(len(time))})

        transformer = FourierFeatureTransformer(seasonality=7.0, harmonics=[1, 2])
        transformer.fit(X)

        names = transformer.get_feature_names_out()
        assert names == [
            "fourier_7.0_sin_1",
            "fourier_7.0_cos_1",
            "fourier_7.0_sin_2",
            "fourier_7.0_cos_2",
        ]

    def test_column_name_conflict_raises(self):
        """Test that conflicting column names raise ValueError."""
        time = pl.datetime_range(start=datetime(2020, 1, 1), end=datetime(2020, 1, 15), interval="1d", eager=True)
        X = pl.DataFrame({"time": time, "fourier_7.0_sin_1": range(len(time))})

        transformer = FourierFeatureTransformer(seasonality=7.0, harmonics=[1])
        with pytest.raises(ValueError, match="conflict"):
            transformer.fit(X)


class TestTimeIndexTransformerSystematic:
    """Systematic check generator tests for TimeIndexTransformer."""

    @pytest.mark.parametrize(
        "transformer,expected_failures",
        [
            (TimeIndexTransformer(), []),
            (TimeIndexTransformer(normalize=True), []),
            (TimeIndexTransformer(degree=3), []),
            (TimeIndexTransformer(normalize=True, degree=2), []),
        ],
        ids=["default", "normalized", "polynomial", "normalized_polynomial"],
    )
    def test_systematic_checks(
        self,
        transformer,
        expected_failures,
        time_series_train_test_factory,
    ):
        """Run all applicable checks for TimeIndexTransformer."""
        X_train, X_test = time_series_train_test_factory(
            train_length=60,
            test_length=30,
        )

        transformer_fitted = clone(transformer)
        transformer_fitted.fit(X_train)

        run_checks(
            transformer_fitted,
            _yield_yohou_transformer_checks(transformer_fitted, X_train, None, X_test),
            expected_failures=set(expected_failures),
        )


class TestTimeIndexTransformerBasic:
    """Basic functionality tests for TimeIndexTransformer."""

    def test_sequential_integer_index(self):
        """Test produces sequential integer index starting at 0."""
        time = pl.datetime_range(start=datetime(2020, 1, 1), end=datetime(2020, 1, 11), interval="1d", eager=True)
        X = pl.DataFrame({"time": time, "value": range(len(time))})

        transformer = TimeIndexTransformer()
        transformer.fit(X)
        X_t = transformer.transform(X)

        indices = X_t["time_index"].to_list()
        assert indices == list(range(len(time)))

    def test_preserves_time_and_original(self):
        """Test time column preserved and original features dropped."""
        time = pl.datetime_range(start=datetime(2020, 1, 1), end=datetime(2020, 1, 11), interval="1d", eager=True)
        X = pl.DataFrame({"time": time, "value": range(len(time))})

        transformer = TimeIndexTransformer()
        transformer.fit(X)
        X_t = transformer.transform(X)

        assert "time" in X_t.columns
        assert "value" not in X_t.columns
        assert "time_index" in X_t.columns

    def test_integer_dtype_default(self):
        """Test default output dtype is Int64."""
        time = pl.datetime_range(start=datetime(2020, 1, 1), end=datetime(2020, 1, 11), interval="1d", eager=True)
        X = pl.DataFrame({"time": time, "value": range(len(time))})

        transformer = TimeIndexTransformer()
        transformer.fit(X)
        X_t = transformer.transform(X)

        assert X_t["time_index"].dtype == pl.Int64


class TestTimeIndexTransformerNormalized:
    """Tests for normalized index output."""

    def test_normalized_range(self):
        """Test normalized values span [0, 1] for fit data."""
        time = pl.datetime_range(start=datetime(2020, 1, 1), end=datetime(2020, 1, 11), interval="1d", eager=True)
        X = pl.DataFrame({"time": time, "value": range(len(time))})

        transformer = TimeIndexTransformer(normalize=True)
        transformer.fit(X)
        X_t = transformer.transform(X)

        values = X_t["time_index"].to_numpy()
        np.testing.assert_allclose(values[0], 0.0)
        np.testing.assert_allclose(values[-1], 1.0)

    def test_normalized_exceeds_range_on_new_data(self):
        """Test normalized values can exceed [0, 1] on future data."""
        time_train = pl.datetime_range(start=datetime(2020, 1, 1), end=datetime(2020, 1, 11), interval="1d", eager=True)
        time_test = pl.datetime_range(start=datetime(2020, 1, 11), end=datetime(2020, 1, 21), interval="1d", eager=True)
        X_train = pl.DataFrame({"time": time_train, "value": range(len(time_train))})
        X_test = pl.DataFrame({"time": time_test, "value": range(len(time_test))})

        transformer = TimeIndexTransformer(normalize=True)
        transformer.fit(X_train)
        X_t = transformer.transform(X_test)

        values = X_t["time_index"].to_numpy()
        assert values[-1] > 1.0

    def test_float_dtype_when_normalized(self):
        """Test normalized output dtype is Float64."""
        time = pl.datetime_range(start=datetime(2020, 1, 1), end=datetime(2020, 1, 11), interval="1d", eager=True)
        X = pl.DataFrame({"time": time, "value": range(len(time))})

        transformer = TimeIndexTransformer(normalize=True)
        transformer.fit(X)
        X_t = transformer.transform(X)

        assert X_t["time_index"].dtype == pl.Float64


class TestTimeIndexTransformerPolynomial:
    """Tests for polynomial degree output."""

    def test_degree_2_produces_2_columns(self):
        """Test degree=2 produces time_index and time_index_2."""
        time = pl.datetime_range(start=datetime(2020, 1, 1), end=datetime(2020, 1, 11), interval="1d", eager=True)
        X = pl.DataFrame({"time": time, "value": range(len(time))})

        transformer = TimeIndexTransformer(degree=2)
        transformer.fit(X)
        X_t = transformer.transform(X)

        assert "time_index" in X_t.columns
        assert "time_index_2" in X_t.columns

    def test_degree_3_produces_3_columns(self):
        """Test degree=3 produces 3 index columns."""
        time = pl.datetime_range(start=datetime(2020, 1, 1), end=datetime(2020, 1, 11), interval="1d", eager=True)
        X = pl.DataFrame({"time": time, "value": range(len(time))})

        transformer = TimeIndexTransformer(degree=3)
        transformer.fit(X)
        X_t = transformer.transform(X)

        assert "time_index" in X_t.columns
        assert "time_index_2" in X_t.columns
        assert "time_index_3" in X_t.columns

    def test_polynomial_values_correct(self):
        """Test polynomial column values are correct powers."""
        time = pl.datetime_range(start=datetime(2020, 1, 1), end=datetime(2020, 1, 6), interval="1d", eager=True)
        X = pl.DataFrame({"time": time, "value": range(len(time))})

        transformer = TimeIndexTransformer(degree=3)
        transformer.fit(X)
        X_t = transformer.transform(X)

        t = X_t["time_index"].to_numpy().astype(np.float64)
        t2 = X_t["time_index_2"].to_numpy()
        t3 = X_t["time_index_3"].to_numpy()

        np.testing.assert_allclose(t2, t**2)
        np.testing.assert_allclose(t3, t**3)

    def test_polynomial_float_dtype(self):
        """Test polynomial columns are Float64."""
        time = pl.datetime_range(start=datetime(2020, 1, 1), end=datetime(2020, 1, 11), interval="1d", eager=True)
        X = pl.DataFrame({"time": time, "value": range(len(time))})

        transformer = TimeIndexTransformer(degree=2)
        transformer.fit(X)
        X_t = transformer.transform(X)

        assert X_t["time_index_2"].dtype == pl.Float64


class TestTimeIndexTransformerEdgeCases:
    """Edge case tests for TimeIndexTransformer."""

    def test_single_row(self):
        """Test transformer works with two-row DataFrame (minimum for interval detection)."""
        X = pl.DataFrame({
            "time": [datetime(2020, 1, 1), datetime(2020, 1, 2)],
            "value": [1.0, 2.0],
        })

        transformer = TimeIndexTransformer()
        transformer.fit(X)
        X_t = transformer.transform(X)

        assert X_t["time_index"][0] == 0

    def test_single_row_normalized(self):
        """Test normalized two-row produces 0 and 1."""
        X = pl.DataFrame({
            "time": [datetime(2020, 1, 1), datetime(2020, 1, 2)],
            "value": [1.0, 2.0],
        })

        transformer = TimeIndexTransformer(normalize=True)
        transformer.fit(X)
        X_t = transformer.transform(X)

        assert X_t["time_index"][0] == 0.0
        assert X_t["time_index"][1] == 1.0

    def test_get_feature_names_out(self):
        """Test get_feature_names_out returns correct names."""
        time = pl.datetime_range(start=datetime(2020, 1, 1), end=datetime(2020, 1, 11), interval="1d", eager=True)
        X = pl.DataFrame({"time": time, "value": range(len(time))})

        transformer = TimeIndexTransformer(degree=3)
        transformer.fit(X)

        names = transformer.get_feature_names_out()
        assert names == ["time_index", "time_index_2", "time_index_3"]

    def test_column_name_conflict_raises(self):
        """Test that conflicting column names raise ValueError."""
        time = pl.datetime_range(start=datetime(2020, 1, 1), end=datetime(2020, 1, 11), interval="1d", eager=True)
        X = pl.DataFrame({"time": time, "time_index": range(len(time))})

        transformer = TimeIndexTransformer()
        with pytest.raises(ValueError, match="conflict"):
            transformer.fit(X)
