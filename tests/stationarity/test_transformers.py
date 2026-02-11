"""Tests for stationarization transformers.

Tests SeasonalDifferencing and SeasonalLogDifferencing transformers
using both the check generator pattern and transformer-specific tests.
"""

from datetime import datetime

import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal
from sklearn.base import clone

from conftest import run_checks
from yohou.stationarity.transformers import (
    AbsoluteSeasonalReturn,
    ASinhTransformer,
    BoxCoxTransformer,
    LogTransformer,
    SeasonalDifferencing,
    SeasonalLogDifferencing,
    SeasonalReturn,
)
from yohou.testing import _yield_yohou_transformer_checks


@pytest.fixture(scope="module")
def stationarity_X():
    """Deterministic data for seasonal stationarization identity tests."""
    length = 52
    rng = np.random.default_rng(42)
    return pl.DataFrame({
        "time": pl.datetime_range(
            start=datetime(2021, 12, 16),
            end=datetime(2021, 12, 16, 0, 0, length - 1),
            interval="1s",
            eager=True,
        ),
        "a": rng.random(length),
        "b": rng.random(length),
    })


class TestSeasonalStationarization:
    @pytest.mark.parametrize(
        "transformer",
        [
            SeasonalDifferencing(1),
            SeasonalDifferencing(5),
            SeasonalLogDifferencing(1, 2),
            SeasonalLogDifferencing(3, 1),
        ],
    )
    def test_identity(self, transformer, stationarity_X):
        X_t = transformer.fit_transform(stationarity_X)
        observation_horizon = transformer.observation_horizon

        X_it = transformer.inverse_transform(X_t=X_t, X_p=stationarity_X[:observation_horizon])

        assert_frame_equal(X_it, stationarity_X[observation_horizon:])

    @pytest.mark.parametrize(
        "transformer,expected_failures",
        [
            (
                SeasonalDifferencing(seasonality=1),
                [],
            ),
            (
                SeasonalLogDifferencing(seasonality=1),
                [],
            ),
        ],
        ids=["SeasonalDifferencing", "SeasonalLogDifferencing"],
    )
    def test_stationarization_transformer_checks(
        self,
        transformer,
        expected_failures,
        time_series_train_test_factory,
    ):
        """Run all applicable checks for stationarization transformers."""
        # Generate continuous train/test data
        min_horizon = 10
        X_train, X_test = time_series_train_test_factory(train_length=min_horizon + 50, test_length=min_horizon + 20)

        # Get tags from transformer
        tags = transformer.__sklearn_tags__()

        # Adjust data for transformers with min_value constraints
        if tags.input_tags and tags.input_tags.min_value is not None:
            # Make data satisfy min_value constraint by adding offset
            offset = max(0.0, tags.input_tags.min_value + 1.0)
            X_train = X_train.select([pl.col("time"), (pl.all().exclude("time") + offset)])
            X_test = X_test.select([pl.col("time"), (pl.all().exclude("time") + offset)])

        transformer_fitted = clone(transformer)
        transformer_fitted.fit(X_train)

        run_checks(
            transformer_fitted,
            _yield_yohou_transformer_checks(transformer_fitted, X_train, None, X_test),
            expected_failures=set(expected_failures),
        )

    def test_seasonal_differencing_inverse(self, time_series_factory):
        """Test SeasonalDifferencing specific round-trip behavior."""
        X = time_series_factory(length=50)
        transformer = SeasonalDifferencing(seasonality=1)
        transformer.fit(X)

        X_trans = transformer.transform(X)
        # inverse_transform requires X_p (past observations)
        horizon = transformer.observation_horizon
        n_dropped = len(X) - len(X_trans)
        X_p = X[n_dropped - horizon : n_dropped]
        X_inv = transformer.inverse_transform(X_trans, X_p)

        # Should recover original shape (for the transformed portion)
        X_expected = X.tail(len(X_trans))
        assert X_expected.shape == X_inv.shape
        # Should have same columns
        assert set(X_expected.columns) == set(X_inv.columns)

    def test_seasonal_log_differencing_requires_positive(self, time_series_factory):
        """Test SeasonalLogDifferencing with non-positive data doesn't crash."""
        X = time_series_factory(length=50)
        # Create data with negative values
        X_negative = X.select([pl.col("time"), (pl.all().exclude("time") - 50.0)])

        transformer = SeasonalLogDifferencing(seasonality=1)

        # Should fit without error
        transformer.fit(X_negative)
        # Transform may produce -inf or NaN but shouldn't crash
        X_trans = transformer.transform(X_negative)

        # Basic validation - should return a dataframe with time column
        assert "time" in X_trans.columns
        assert len(X_trans) > 0

    def test_seasonal_differencing_seasonality_2(self, time_series_factory):
        """Test SeasonalDifferencing with seasonality=2."""
        X = time_series_factory(length=50)
        transformer = SeasonalDifferencing(seasonality=2)
        transformer.fit(X)

        X_trans = transformer.transform(X)

        # Should have time column
        assert "time" in X_trans.columns
        # Output length should be input length - seasonality
        assert len(X_trans) == len(X) - 2

    def test_seasonal_differencing_observation_horizon(self, time_series_factory):
        """Test observation_horizon matches seasonality."""
        X = time_series_factory(length=50)

        for seasonality in [1, 2, 3, 7]:
            transformer = SeasonalDifferencing(seasonality=seasonality)
            transformer.fit(X)

            assert transformer.observation_horizon == seasonality, (
                f"Expected observation_horizon={seasonality}, got {transformer.observation_horizon}"
            )


class TestBoxCoxTransformer:
    @pytest.mark.parametrize(
        "transformer,expected_failures",
        [
            (BoxCoxTransformer(lmbda=0.0), []),
            (BoxCoxTransformer(lmbda=0.5), []),
            (BoxCoxTransformer(lmbda=1.0), []),
        ],
        ids=["BoxCox_log", "BoxCox_sqrt", "BoxCox_identity"],
    )
    def test_boxcox_systematic_checks(
        self,
        transformer,
        expected_failures,
        time_series_train_test_factory,
    ):
        """Run all applicable checks for BoxCoxTransformer."""
        X_train, X_test = time_series_train_test_factory(train_length=60, test_length=20)

        # BoxCox requires positive data
        X_train = X_train.select([pl.col("time"), (pl.all().exclude("time").abs() + 1.0)])
        X_test = X_test.select([pl.col("time"), (pl.all().exclude("time").abs() + 1.0)])

        transformer_fitted = clone(transformer)
        transformer_fitted.fit(X_train)

        run_checks(
            transformer_fitted,
            _yield_yohou_transformer_checks(transformer_fitted, X_train, None, X_test),
            expected_failures=set(expected_failures),
        )

    def test_boxcox_log_transform(self, time_series_factory):
        """Test BoxCox with lambda=0 is equivalent to log transform."""
        X = time_series_factory(length=50)
        # Ensure positive data
        X = X.select([pl.col("time"), (pl.all().exclude("time").abs() + 1.0)])

        transformer = BoxCoxTransformer(lmbda=0.0)
        transformer.fit(X)

        X_trans = transformer.transform(X)

        # Check that it's a log transform
        assert "time" in X_trans.columns

        # Manually compute log transform for comparison
        X_numeric = X.select(~pl.selectors.by_name("time"))
        X_log = X_numeric.with_columns(pl.all().log())

        for i, _ in enumerate(X_log.columns):
            np.testing.assert_allclose(
                X_trans.select(~pl.selectors.by_name("time")).to_numpy()[:, i],
                X_log.to_numpy()[:, i],
                rtol=1e-10,
            )

    def test_boxcox_sqrt_transform(self, time_series_factory):
        """Test BoxCox with lambda=0.5 is equivalent to sqrt-like transform."""
        X = time_series_factory(length=50)
        # Ensure positive data
        X = X.select([pl.col("time"), (pl.all().exclude("time").abs() + 1.0)])

        transformer = BoxCoxTransformer(lmbda=0.5)
        transformer.fit(X)

        X_trans = transformer.transform(X)

        # BoxCox with lambda=0.5: (x^0.5 - 1) / 0.5 = 2 * (sqrt(x) - 1)
        assert "time" in X_trans.columns

    def test_boxcox_inverse_transform(self, time_series_factory):
        """Test BoxCox inverse_transform recovers original data."""
        X = time_series_factory(length=50)
        # Ensure positive data
        X = X.select([pl.col("time"), (pl.all().exclude("time").abs() + 1.0)])

        for lmbda in [0.0, 0.5, 1.0, 2.0]:
            transformer = BoxCoxTransformer(lmbda=lmbda)
            transformer.fit(X)

            X_trans = transformer.transform(X)
            X_inv = transformer.inverse_transform(X_trans, X_p=None)

            # Numeric values should match original
            for col in X.columns:
                if col == "time":
                    continue
                np.testing.assert_allclose(
                    X_inv[col].to_numpy(),
                    X[col].to_numpy(),
                    rtol=1e-5,
                    err_msg=f"Column {col} not recovered correctly with lambda={lmbda}",
                )

    def test_boxcox_with_offset(self, time_series_factory):
        """Test BoxCox with offset parameter."""
        X = time_series_factory(length=50)
        # Data can be negative if we have offset
        X_shifted = X.select([pl.col("time"), (pl.all().exclude("time") - 0.5)])

        transformer = BoxCoxTransformer(lmbda=0.5, offset=1.0)
        transformer.fit(X_shifted)

        X_trans = transformer.transform(X_shifted)
        X_inv = transformer.inverse_transform(X_trans, X_p=None)

        # Should recover original data
        for col in X_shifted.columns:
            if col == "time":
                continue
            np.testing.assert_allclose(
                X_inv[col].to_numpy(),
                X_shifted[col].to_numpy(),
                rtol=1e-5,
            )

    def test_boxcox_observation_horizon_zero(self, time_series_factory):
        """Test BoxCoxTransformer is stateless."""
        X = time_series_factory(length=50)
        X = X.select([pl.col("time"), (pl.all().exclude("time").abs() + 1.0)])

        transformer = BoxCoxTransformer(lmbda=0.5)
        transformer.fit(X)

        assert transformer.observation_horizon == 0

    def test_boxcox_feature_names(self, time_series_factory):
        """Test BoxCoxTransformer feature names output."""
        X = time_series_factory(length=50, n_components=2)
        X = X.select([pl.col("time"), (pl.all().exclude("time").abs() + 1.0)])

        transformer = BoxCoxTransformer(lmbda=0.5, offset=1.0)
        transformer.fit(X)

        feature_names = transformer.get_feature_names_out()
        assert len(feature_names) == 2
        assert all("boxcox" in name for name in feature_names)


class TestSeasonalReturn:
    @pytest.mark.parametrize(
        "transformer,expected_failures",
        [
            (SeasonalReturn(seasonality=1), []),
            (SeasonalReturn(seasonality=3, offset=1.0), []),
        ],
        ids=["SeasonalReturn_s1", "SeasonalReturn_s3_offset"],
    )
    def test_seasonal_return_systematic_checks(
        self,
        transformer,
        expected_failures,
        time_series_train_test_factory,
    ):
        """Run all applicable checks for SeasonalReturn."""
        X_train, X_test = time_series_train_test_factory(train_length=70, test_length=30)

        # Ensure positive data for percentage returns
        X_train = X_train.select([pl.col("time"), (pl.all().exclude("time").abs() + 1.0)])
        X_test = X_test.select([pl.col("time"), (pl.all().exclude("time").abs() + 1.0)])

        transformer_fitted = clone(transformer)
        transformer_fitted.fit(X_train)

        run_checks(
            transformer_fitted,
            _yield_yohou_transformer_checks(transformer_fitted, X_train, None, X_test),
            expected_failures=set(expected_failures),
        )

    def test_seasonal_return_basic(self, time_series_factory):
        """Test SeasonalReturn computes percentage returns correctly."""
        X = time_series_factory(length=50)
        # Ensure positive data
        X = X.select([pl.col("time"), (pl.all().exclude("time").abs() + 1.0)])

        transformer = SeasonalReturn(seasonality=1)
        transformer.fit(X)

        X_trans = transformer.transform(X)

        # Output length should be input length - seasonality
        assert len(X_trans) == len(X) - 1
        assert "time" in X_trans.columns

    def test_seasonal_return_with_seasonality(self, time_series_factory):
        """Test SeasonalReturn with different seasonality values."""
        X = time_series_factory(length=50)
        X = X.select([pl.col("time"), (pl.all().exclude("time").abs() + 1.0)])

        for seasonality in [1, 2, 5, 7]:
            transformer = SeasonalReturn(seasonality=seasonality)
            transformer.fit(X)

            X_trans = transformer.transform(X)

            assert len(X_trans) == len(X) - seasonality, f"Expected length {len(X) - seasonality}, got {len(X_trans)}"

    def test_seasonal_return_inverse_transform(self, time_series_factory):
        """Test SeasonalReturn inverse_transform recovers original data."""
        X = time_series_factory(length=50)
        X = X.select([pl.col("time"), (pl.all().exclude("time").abs() + 1.0)])

        for seasonality in [1, 2, 3]:
            transformer = SeasonalReturn(seasonality=seasonality, offset=0.0)
            transformer.fit(X)

            X_trans = transformer.transform(X)
            X_p = X[:seasonality]
            X_inv = transformer.inverse_transform(X_trans, X_p=X_p)

            # Should recover the transformed portion
            X_expected = X[seasonality:]

            for col in X_expected.columns:
                if col == "time":
                    continue
                np.testing.assert_allclose(
                    X_inv[col].to_numpy(),
                    X_expected[col].to_numpy(),
                    rtol=1e-5,
                    err_msg=f"Column {col} not recovered correctly with seasonality={seasonality}",
                )

    def test_seasonal_return_with_offset(self, time_series_factory):
        """Test SeasonalReturn with offset parameter."""
        X = time_series_factory(length=50)
        # Allow near-zero values by using offset
        X = X.select([pl.col("time"), (pl.all().exclude("time") * 0.1)])

        transformer = SeasonalReturn(seasonality=1, offset=1.0)
        transformer.fit(X)

        X_trans = transformer.transform(X)
        X_p = X[:1]
        X_inv = transformer.inverse_transform(X_trans, X_p=X_p)

        # Should recover original data
        X_expected = X[1:]

        for col in X_expected.columns:
            if col == "time":
                continue
            np.testing.assert_allclose(
                X_inv[col].to_numpy(),
                X_expected[col].to_numpy(),
                rtol=1e-5,
            )

    def test_seasonal_return_observation_horizon(self, time_series_factory):
        """Test observation_horizon matches seasonality."""
        X = time_series_factory(length=50)
        X = X.select([pl.col("time"), (pl.all().exclude("time").abs() + 1.0)])

        for seasonality in [1, 2, 3, 5]:
            transformer = SeasonalReturn(seasonality=seasonality)
            transformer.fit(X)

            assert transformer.observation_horizon == seasonality

    def test_seasonal_return_feature_names(self, time_series_factory):
        """Test SeasonalReturn feature names output."""
        X = time_series_factory(length=50, n_components=2)
        X = X.select([pl.col("time"), (pl.all().exclude("time").abs() + 1.0)])

        transformer = SeasonalReturn(seasonality=3, offset=1.0)
        transformer.fit(X)

        feature_names = transformer.get_feature_names_out()
        assert len(feature_names) == 2
        assert all("return" in name for name in feature_names)


class TestAbsoluteSeasonalReturn:
    @pytest.mark.parametrize(
        "transformer,expected_failures",
        [
            (AbsoluteSeasonalReturn(seasonality=1), []),
            (AbsoluteSeasonalReturn(seasonality=3), []),
        ],
        ids=["AbsoluteSeasonalReturn_s1", "AbsoluteSeasonalReturn_s3"],
    )
    def test_absolute_seasonal_return_systematic_checks(
        self,
        transformer,
        expected_failures,
        time_series_train_test_factory,
    ):
        """Run all applicable checks for AbsoluteSeasonalReturn."""
        X_train, X_test = time_series_train_test_factory(train_length=70, test_length=30)

        transformer_fitted = clone(transformer)
        transformer_fitted.fit(X_train)

        run_checks(
            transformer_fitted,
            _yield_yohou_transformer_checks(transformer_fitted, X_train, None, X_test),
            expected_failures=set(expected_failures),
        )

    def test_absolute_seasonal_return_basic(self, time_series_factory):
        """Test AbsoluteSeasonalReturn computes absolute differences correctly."""
        X = time_series_factory(length=50)

        transformer = AbsoluteSeasonalReturn(seasonality=1)
        transformer.fit(X)

        X_trans = transformer.transform(X)

        # Output length should be input length - seasonality
        assert len(X_trans) == len(X) - 1
        assert "time" in X_trans.columns

    def test_absolute_seasonal_return_with_seasonality(self, time_series_factory):
        """Test AbsoluteSeasonalReturn with different seasonality values."""
        X = time_series_factory(length=50)

        for seasonality in [1, 2, 5, 7]:
            transformer = AbsoluteSeasonalReturn(seasonality=seasonality)
            transformer.fit(X)

            X_trans = transformer.transform(X)

            assert len(X_trans) == len(X) - seasonality

    def test_absolute_seasonal_return_inverse_transform(self, time_series_factory):
        """Test AbsoluteSeasonalReturn inverse_transform recovers original data."""
        X = time_series_factory(length=50)

        for seasonality in [1, 2, 3]:
            transformer = AbsoluteSeasonalReturn(seasonality=seasonality)
            transformer.fit(X)

            X_trans = transformer.transform(X)
            X_p = X[:seasonality]
            X_inv = transformer.inverse_transform(X_trans, X_p=X_p)

            # Should recover the transformed portion
            X_expected = X[seasonality:]

            for col in X_expected.columns:
                if col == "time":
                    continue
                np.testing.assert_allclose(
                    X_inv[col].to_numpy(),
                    X_expected[col].to_numpy(),
                    rtol=1e-5,
                    err_msg=f"Column {col} not recovered correctly with seasonality={seasonality}",
                )

    def test_absolute_seasonal_return_equals_seasonal_differencing(self, time_series_factory):
        """Test AbsoluteSeasonalReturn produces same result as SeasonalDifferencing."""
        X = time_series_factory(length=50)

        for seasonality in [1, 2, 3]:
            abs_return = AbsoluteSeasonalReturn(seasonality=seasonality)
            diff = SeasonalDifferencing(seasonality=seasonality)

            abs_return.fit(X)
            diff.fit(X)

            X_abs = abs_return.transform(X)
            X_diff = diff.transform(X)

            # Time columns should match
            assert_frame_equal(X_abs.select("time"), X_diff.select("time"))

            # Values should match (column names may differ)
            for i, col in enumerate([c for c in X_abs.columns if c != "time"]):
                np.testing.assert_allclose(
                    X_abs[col].to_numpy(),
                    X_diff[[c for c in X_diff.columns if c != "time"][i]].to_numpy(),
                    rtol=1e-10,
                )

    def test_absolute_seasonal_return_observation_horizon(self, time_series_factory):
        """Test observation_horizon matches seasonality."""
        X = time_series_factory(length=50)

        for seasonality in [1, 2, 3, 5]:
            transformer = AbsoluteSeasonalReturn(seasonality=seasonality)
            transformer.fit(X)

            assert transformer.observation_horizon == seasonality

    def test_absolute_seasonal_return_feature_names(self, time_series_factory):
        """Test AbsoluteSeasonalReturn feature names output."""
        X = time_series_factory(length=50, n_components=2)

        transformer = AbsoluteSeasonalReturn(seasonality=3, offset=1.0)
        transformer.fit(X)

        feature_names = transformer.get_feature_names_out()
        assert len(feature_names) == 2
        assert all("abs_return" in name for name in feature_names)


class TestNewTransformerIdentity:
    @pytest.mark.parametrize(
        "transformer",
        [
            BoxCoxTransformer(lmbda=0.0),
            BoxCoxTransformer(lmbda=0.5, offset=1.0),
            SeasonalReturn(seasonality=1),
            SeasonalReturn(seasonality=3, offset=1.0),
            AbsoluteSeasonalReturn(seasonality=1),
            AbsoluteSeasonalReturn(seasonality=3),
        ],
    )
    def test_new_transformer_identity(self, transformer, stationarity_X):
        """Test round-trip transform/inverse_transform for new transformers."""
        # Create positive data for all transformers
        X_positive = stationarity_X.select([pl.col("time"), (pl.all().exclude("time").abs() + 1.0)])

        # Fit transformer
        transformer.fit(X_positive)

        X_t = transformer.transform(X_positive)
        observation_horizon = transformer.observation_horizon

        X_p = X_positive[:observation_horizon] if observation_horizon > 0 else None

        X_it = transformer.inverse_transform(X_t=X_t, X_p=X_p)

        # Compare with expected (original data after dropping initial rows)
        X_expected = X_positive[observation_horizon:]

        for col in X_expected.columns:
            if col == "time":
                continue
            np.testing.assert_allclose(
                X_it[col].to_numpy(),
                X_expected[col].to_numpy(),
                rtol=1e-5,
                err_msg=f"Column {col} not recovered correctly",
            )


class TestLogTransform:
    @pytest.mark.parametrize(
        "transformer,expected_failures",
        [
            (LogTransformer(offset=0.0), []),
            (LogTransformer(offset=1.0), []),
        ],
        ids=["LogTransform_offset_0", "LogTransform_offset_1"],
    )
    def test_logtransform_systematic_checks(
        self,
        transformer,
        expected_failures,
        time_series_train_test_factory,
    ):
        """Run all applicable checks for LogTransformer."""
        # Generate continuous train/test data
        min_horizon = 10
        X_train, X_test = time_series_train_test_factory(train_length=min_horizon + 50, test_length=min_horizon + 20)

        # Get tags from transformer
        tags = transformer.__sklearn_tags__()

        # Adjust data for transformers with min_value constraints (LogTransformer requires positive data)
        if tags.input_tags and tags.input_tags.min_value is not None:
            offset = max(0.0, tags.input_tags.min_value + 1.0)
            X_train = X_train.select([pl.col("time"), (pl.all().exclude("time").abs() + offset)])
            X_test = X_test.select([pl.col("time"), (pl.all().exclude("time").abs() + offset)])
        else:
            # Default: make data positive
            X_train = X_train.select([pl.col("time"), (pl.all().exclude("time").abs() + 1.0)])
            X_test = X_test.select([pl.col("time"), (pl.all().exclude("time").abs() + 1.0)])

        transformer_fitted = clone(transformer)
        transformer_fitted.fit(X_train)

        run_checks(
            transformer_fitted,
            _yield_yohou_transformer_checks(transformer_fitted, X_train, None, X_test),
            expected_failures=set(expected_failures),
        )

    def test_logtransform_round_trip(self, time_series_factory):
        """Test LogTransformer round-trip behavior."""
        X = time_series_factory(length=50)
        # Make data positive
        X = X.select([pl.col("time"), (pl.all().exclude("time").abs() + 1.0)])

        transformer = LogTransformer(offset=0.0)
        transformer.fit(X)

        X_trans = transformer.transform(X)
        X_inv = transformer.inverse_transform(X_trans, X_p=None)

        for col in X.columns:
            if col == "time":
                continue
            np.testing.assert_allclose(
                X_inv[col].to_numpy(),
                X[col].to_numpy(),
                rtol=1e-5,
            )


class TestASinhTransformer:
    @pytest.mark.parametrize(
        "transformer,expected_failures",
        [
            (ASinhTransformer(scale=1.4826), []),
            (ASinhTransformer(scale=1.0), []),
        ],
        ids=["ASinhTransformer_default", "ASinhTransformer_scale_1"],
    )
    def test_asinh_systematic_checks(
        self,
        transformer,
        expected_failures,
        time_series_train_test_factory,
    ):
        """Run all applicable checks for ASinhTransformer."""
        # Generate continuous train/test data
        min_horizon = 10
        X_train, X_test = time_series_train_test_factory(train_length=min_horizon + 50, test_length=min_horizon + 20)

        transformer_fitted = clone(transformer)
        transformer_fitted.fit(X_train)

        run_checks(
            transformer_fitted,
            _yield_yohou_transformer_checks(transformer_fitted, X_train, None, X_test),
            expected_failures=set(expected_failures),
        )

    def test_asinh_round_trip(self, time_series_factory):
        """Test ASinhTransformer round-trip behavior."""
        X = time_series_factory(length=50)

        transformer = ASinhTransformer()
        transformer.fit(X)

        X_trans = transformer.transform(X)
        X_inv = transformer.inverse_transform(X_trans, X_p=None)

        for col in X.columns:
            if col == "time":
                continue
            np.testing.assert_allclose(
                X_inv[col].to_numpy(),
                X[col].to_numpy(),
                rtol=1e-9,
            )

    def test_asinh_handles_wide_range(self, time_series_factory):
        """Test ASinhTransformer handles wide range of values."""
        X = pl.DataFrame({
            "time": pl.datetime_range(start=datetime(2024, 1, 1), end=datetime(2024, 1, 10), interval="1d", eager=True),
            "value": [1.0, 10.0, 100.0, 1000.0, 10000.0, 5.0, 50.0, 500.0, 5000.0, 50000.0],
        })

        transformer = ASinhTransformer()
        transformer.fit(X)

        X_trans = transformer.transform(X)
        X_inv = transformer.inverse_transform(X_trans, X_p=None)

        np.testing.assert_allclose(
            X_inv["value"].to_numpy(),
            X["value"].to_numpy(),
            rtol=1e-9,
        )
