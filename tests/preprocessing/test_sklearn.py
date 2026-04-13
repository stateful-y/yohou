"""Tests for sklearn transformer and scaler wrappers.

Tests scalers (StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler, SklearnScaler)
and transformers (Normalizer, PolynomialFeatures, PowerTransformer, QuantileTransformer,
SplineTransformer, SklearnTransformer) using both the check generator pattern and
component-specific tests.
"""

import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal
from sklearn.base import clone
from sklearn.preprocessing import Normalizer as SklearnNormalizer
from sklearn.preprocessing import StandardScaler as SklearnStandardScaler

from conftest import run_checks
from yohou.preprocessing import (
    MaxAbsScaler,
    MinMaxScaler,
    Normalizer,
    PolynomialFeatures,
    PowerTransformer,
    QuantileTransformer,
    RobustScaler,
    SklearnScaler,
    SklearnTransformer,
    SplineTransformer,
    StandardScaler,
)
from yohou.testing import _yield_yohou_transformer_checks


class TestSystematicChecks:
    """Comprehensive check generator tests for scalers and transformers."""

    @pytest.mark.parametrize(
        "transformer,expected_failures",
        [
            (StandardScaler(), []),
            (MinMaxScaler(), []),
            (RobustScaler(), []),
            (MaxAbsScaler(), []),
        ],
        ids=["StandardScaler", "MinMaxScaler", "RobustScaler", "MaxAbsScaler"],
    )
    def test_scaler_systematic_checks(
        self,
        transformer,
        expected_failures,
        time_series_train_test_factory,
    ):
        """Run all applicable checks for scaler transformers."""
        # Generate continuous train/test data
        X_train, X_test = time_series_train_test_factory(train_length=60, test_length=20)

        # Fit transformer
        transformer_fitted = clone(transformer)
        transformer_fitted.fit(X_train)

        # Run all checks from generator
        run_checks(
            transformer_fitted,
            _yield_yohou_transformer_checks(transformer_fitted, X_train, None, X_test),
            expected_failures=set(expected_failures),
        )

    @pytest.mark.parametrize(
        "transformer,expected_failures",
        [
            (Normalizer(), []),
            (PolynomialFeatures(degree=2, include_bias=False), []),
            (PowerTransformer(method="yeo-johnson"), []),
            (QuantileTransformer(n_quantiles=10), ["check_inverse_observe_transform_identity"]),
            (SplineTransformer(n_knots=4, degree=3), []),
        ],
        ids=["Normalizer", "PolynomialFeatures", "PowerTransformer", "QuantileTransformer", "SplineTransformer"],
    )
    def test_transformer_systematic_checks(
        self,
        transformer,
        expected_failures,
        time_series_train_test_factory,
    ):
        """Run all applicable checks for sklearn transformers."""
        # Generate continuous train/test data
        X_train, X_test = time_series_train_test_factory(train_length=60, test_length=20)

        # Fit transformer
        transformer_fitted = clone(transformer)
        transformer_fitted.fit(X_train)

        # Run all checks from generator
        run_checks(
            transformer_fitted,
            _yield_yohou_transformer_checks(transformer_fitted, X_train, None, X_test),
            expected_failures=set(expected_failures),
        )


class TestStandardScaler:
    """StandardScaler specific tests."""

    def test_standard_scaler_fit_transform(self, time_series_factory):
        """Test StandardScaler fit and transform produce zero mean, unit variance."""
        X = time_series_factory(length=100, n_components=2)
        scaler = StandardScaler()
        scaler.fit(X)

        X_scaled = scaler.transform(X)

        # Time column should be preserved
        assert "time" in X_scaled.columns
        assert_frame_equal(X_scaled.select(pl.col("time")), X.select(pl.col("time")))

        # Numeric columns should have ~0 mean and ~1 std
        for col in X_scaled.columns:
            if col == "time":
                continue
            mean = X_scaled[col].mean()
            std = X_scaled[col].std()
            assert abs(mean) < 1e-10, f"Column {col} mean should be ~0, got {mean}"
            assert abs(std - 1.0) < 0.1, f"Column {col} std should be ~1, got {std}"

    def test_standard_scaler_inverse_transform(self, time_series_factory):
        """Test StandardScaler inverse_transform recovers original data."""
        X = time_series_factory(length=50, n_components=2)
        scaler = StandardScaler()
        scaler.fit(X)

        X_scaled = scaler.transform(X)
        X_recovered = scaler.inverse_transform(X_scaled)

        # Time column preserved
        assert "time" in X_recovered.columns

        # Numeric values should match original (within tolerance)
        for col in X.columns:
            if col == "time":
                continue
            np.testing.assert_allclose(
                X_recovered[col].to_numpy(),
                X[col].to_numpy(),
                rtol=1e-5,
                err_msg=f"Column {col} not recovered correctly",
            )

    def test_standard_scaler_fitted_attributes(self, time_series_factory):
        """Test StandardScaler exposes sklearn attributes after fit."""
        X = time_series_factory(length=50, n_components=3)
        scaler = StandardScaler()
        scaler.fit(X)

        # Check that sklearn attributes are accessible
        assert hasattr(scaler, "scale_")
        assert hasattr(scaler, "mean_")
        assert hasattr(scaler, "var_")

        # Attributes should have correct shape
        n_features = len([c for c in X.columns if c != "time"])
        assert len(scaler.scale_) == n_features
        assert len(scaler.mean_) == n_features
        assert len(scaler.var_) == n_features

    def test_standard_scaler_with_mean_false(self, time_series_factory):
        """Test StandardScaler with with_mean=False."""
        X = time_series_factory(length=50, n_components=2)
        scaler = StandardScaler(with_mean=False)
        scaler.fit(X)

        X_scaled = scaler.transform(X)

        # Mean should not be zero (only scaled, not centered)
        for col in X_scaled.columns:
            if col == "time":
                continue
            mean = X_scaled[col].mean()
            # Mean won't be exactly 0 since we're not centering
            assert mean != 0.0, f"Column {col} mean should not be 0 when with_mean=False"

    def test_standard_scaler_with_std_false(self, time_series_factory):
        """Test StandardScaler with with_std=False."""
        X = time_series_factory(length=50, n_components=2)
        scaler = StandardScaler(with_std=False)
        scaler.fit(X)

        X_scaled = scaler.transform(X)

        # Mean should be ~0, but std should match original ratio
        for col in X_scaled.columns:
            if col == "time":
                continue
            mean = X_scaled[col].mean()
            assert abs(mean) < 1e-10, f"Column {col} mean should be ~0"


class TestMinMaxScaler:
    """MinMaxScaler specific tests."""

    def test_minmax_scaler_default_range(self, time_series_factory):
        """Test MinMaxScaler scales to [0, 1] by default."""
        X = time_series_factory(length=50, n_components=2)
        scaler = MinMaxScaler()
        scaler.fit(X)

        X_scaled = scaler.transform(X)

        # Time column preserved
        assert "time" in X_scaled.columns

        # All values should be in [0, 1]
        for col in X_scaled.columns:
            if col == "time":
                continue
            assert X_scaled[col].min() >= 0.0, f"Column {col} has min < 0"
            assert X_scaled[col].max() <= 1.0, f"Column {col} has max > 1"
            # Should actually span the full range
            assert abs(X_scaled[col].min() - 0.0) < 1e-10
            assert abs(X_scaled[col].max() - 1.0) < 1e-10

    def test_minmax_scaler_custom_range(self, time_series_factory):
        """Test MinMaxScaler with custom feature_range."""
        X = time_series_factory(length=50, n_components=2)
        scaler = MinMaxScaler(feature_range=(-1, 1))
        scaler.fit(X)

        X_scaled = scaler.transform(X)

        # All values should be in [-1, 1]
        for col in X_scaled.columns:
            if col == "time":
                continue
            assert X_scaled[col].min() >= -1.0, f"Column {col} has min < -1"
            assert X_scaled[col].max() <= 1.0, f"Column {col} has max > 1"

    def test_minmax_scaler_fitted_attributes(self, time_series_factory):
        """Test MinMaxScaler exposes sklearn attributes after fit."""
        X = time_series_factory(length=50, n_components=3)
        scaler = MinMaxScaler()
        scaler.fit(X)

        # Check that sklearn attributes are accessible
        assert hasattr(scaler, "min_")
        assert hasattr(scaler, "scale_")
        assert hasattr(scaler, "data_min_")
        assert hasattr(scaler, "data_max_")
        assert hasattr(scaler, "data_range_")

        n_features = len([c for c in X.columns if c != "time"])
        assert len(scaler.data_min_) == n_features

    def test_minmax_scaler_inverse_transform(self, time_series_factory):
        """Test MinMaxScaler inverse_transform recovers original data."""
        X = time_series_factory(length=50, n_components=2)
        scaler = MinMaxScaler()
        scaler.fit(X)

        X_scaled = scaler.transform(X)
        X_recovered = scaler.inverse_transform(X_scaled)

        for col in X.columns:
            if col == "time":
                continue
            np.testing.assert_allclose(
                X_recovered[col].to_numpy(),
                X[col].to_numpy(),
                rtol=1e-5,
            )


class TestRobustScaler:
    """RobustScaler specific tests."""

    def test_robust_scaler_with_outliers(self, time_series_factory):
        """Test RobustScaler handles outliers better than StandardScaler."""
        X = time_series_factory(length=50, n_components=1)
        # Add outlier
        X_with_outlier = X.with_columns([
            pl
            .when(pl.col("time").dt.ordinal_day() == 1)
            .then(pl.lit(1000.0))  # Outlier
            .otherwise(pl.exclude("time"))
            .alias(X.columns[1])
        ])

        scaler = RobustScaler()
        scaler.fit(X_with_outlier)
        X_scaled = scaler.transform(X_with_outlier)

        # Time column preserved
        assert "time" in X_scaled.columns

        # Should still produce reasonable values (outlier won't dominate)
        assert X_scaled[X_scaled.columns[1]].median() is not None

    def test_robust_scaler_fitted_attributes(self, time_series_factory):
        """Test RobustScaler exposes sklearn attributes after fit."""
        X = time_series_factory(length=50, n_components=2)
        scaler = RobustScaler()
        scaler.fit(X)

        assert hasattr(scaler, "center_")
        assert hasattr(scaler, "scale_")

        n_features = len([c for c in X.columns if c != "time"])
        assert len(scaler.center_) == n_features
        assert len(scaler.scale_) == n_features

    def test_robust_scaler_without_centering(self, time_series_factory):
        """Test RobustScaler with with_centering=False."""
        X = time_series_factory(length=50, n_components=2)
        scaler = RobustScaler(with_centering=False)
        scaler.fit(X)

        X_scaled = scaler.transform(X)
        assert "time" in X_scaled.columns

        # Without centering, median won't be 0
        for col in X_scaled.columns:
            if col != "time":
                # Just verify it works, don't check specific values
                assert X_scaled[col].is_not_null().all()

    def test_robust_scaler_custom_quantile_range(self, time_series_factory):
        """Test RobustScaler with custom quantile_range."""
        X = time_series_factory(length=50, n_components=2)
        scaler = RobustScaler(quantile_range=(10.0, 90.0))
        scaler.fit(X)

        X_scaled = scaler.transform(X)
        assert "time" in X_scaled.columns
        assert len(X_scaled) == len(X)


class TestMaxAbsScaler:
    """MaxAbsScaler specific tests."""

    def test_maxabs_scaler_range(self, time_series_factory):
        """Test MaxAbsScaler scales to [-1, 1] based on max absolute value."""
        X = time_series_factory(length=50, n_components=2)
        # Shift to have both positive and negative values
        X_shifted = X.select([
            pl.col("time"),
            (pl.exclude("time") - 0.5).name.keep(),  # Center around 0
        ])

        scaler = MaxAbsScaler()
        scaler.fit(X_shifted)
        X_scaled = scaler.transform(X_shifted)

        # Time column preserved
        assert "time" in X_scaled.columns

        # Max absolute value should be 1
        for col in X_scaled.columns:
            if col == "time":
                continue
            max_abs = max(abs(X_scaled[col].max()), abs(X_scaled[col].min()))
            assert abs(max_abs - 1.0) < 1e-10, f"Column {col} max abs should be 1, got {max_abs}"

    def test_maxabs_scaler_fitted_attributes(self, time_series_factory):
        """Test MaxAbsScaler exposes sklearn attributes after fit."""
        X = time_series_factory(length=50, n_components=2)
        scaler = MaxAbsScaler()
        scaler.fit(X)

        assert hasattr(scaler, "scale_")
        assert hasattr(scaler, "max_abs_")

        n_features = len([c for c in X.columns if c != "time"])
        assert len(scaler.scale_) == n_features
        assert len(scaler.max_abs_) == n_features

    def test_maxabs_scaler_inverse_transform(self, time_series_factory):
        """Test MaxAbsScaler inverse_transform recovers original data."""
        X = time_series_factory(length=50, n_components=2)
        scaler = MaxAbsScaler()
        scaler.fit(X)

        X_scaled = scaler.transform(X)
        X_recovered = scaler.inverse_transform(X_scaled)

        for col in X.columns:
            if col == "time":
                continue
            np.testing.assert_allclose(
                X_recovered[col].to_numpy(),
                X[col].to_numpy(),
                rtol=1e-5,
            )


class TestSklearnScaler:
    """SklearnScaler base class tests."""

    def test_sklearn_scaler_with_explicit_scaler(self, time_series_factory):
        """Test SklearnScaler with explicitly provided scaler class."""
        X = time_series_factory(length=50, n_components=2)

        scaler = SklearnScaler(scaler=SklearnStandardScaler, with_mean=True)
        scaler.fit(X)

        X_scaled = scaler.transform(X)

        # Should work like StandardScaler
        assert "time" in X_scaled.columns
        for col in X_scaled.columns:
            if col == "time":
                continue
            mean = X_scaled[col].mean()
            assert abs(mean) < 1e-10

    def test_sklearn_scaler_requires_scaler_argument(self):
        """Test SklearnScaler requires scaler argument when no default."""
        with pytest.raises(TypeError, match="missing required keyword argument"):
            SklearnScaler()

    def test_sklearn_scaler_get_feature_names_out(self, time_series_factory):
        """Test get_feature_names_out returns correct feature names."""
        X = time_series_factory(length=50, n_components=3)
        scaler = StandardScaler()
        scaler.fit(X)

        feature_names = scaler.get_feature_names_out()
        expected_names = [c for c in X.columns if c != "time"]

        assert feature_names == expected_names


class TestScalerPanelData:
    """Scalers with panel data tests."""

    def test_standard_scaler_panel_data(self, panel_time_series_factory):
        """Test StandardScaler with panel data (prefixed columns)."""
        X = panel_time_series_factory(length=50, n_series=2, n_groups=2)

        scaler = StandardScaler()
        scaler.fit(X)
        X_scaled = scaler.transform(X)

        # Time column preserved
        assert "time" in X_scaled.columns

        # All other columns should be present
        assert len(X_scaled.columns) == len(X.columns)

        # Scaled values should have ~0 mean
        for col in X_scaled.columns:
            if col == "time":
                continue
            mean = X_scaled[col].mean()
            assert abs(mean) < 1e-10, f"Column {col} mean should be ~0"

    def test_minmax_scaler_panel_data(self, panel_time_series_factory):
        """Test MinMaxScaler with panel data."""
        X = panel_time_series_factory(length=50, n_series=2, n_groups=2)

        scaler = MinMaxScaler()
        scaler.fit(X)
        X_scaled = scaler.transform(X)

        # All columns should be in [0, 1]
        for col in X_scaled.columns:
            if col == "time":
                continue
            assert X_scaled[col].min() >= 0.0
            assert X_scaled[col].max() <= 1.0


class TestScalerObservationHorizon:
    """Scaler observation horizon tests."""

    def test_scaler_observation_horizon_is_zero(self, time_series_factory):
        """Test that scalers are stateless (observation_horizon=0)."""
        X = time_series_factory(length=50)

        for scaler_cls in [StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler]:
            scaler = scaler_cls()
            scaler.fit(X)
            assert scaler.observation_horizon == 0, f"{scaler_cls.__name__} should be stateless"

    def test_scaler_no_memory_stored(self, time_series_factory):
        """Test that scalers don't store observations (stateless)."""
        X = time_series_factory(length=50)
        scaler = StandardScaler()
        scaler.fit(X)

        # Stateless transformers should not have _X_observed
        # Or if they do, it should be None or empty
        if hasattr(scaler, "_X_observed"):
            assert scaler._X_observed is None or len(scaler._X_observed) == 0


class TestScalerCloneAndParams:
    """Scaler cloning and serialization tests."""

    def test_scaler_clone(self, time_series_factory):
        """Test that scalers can be cloned."""
        X = time_series_factory(length=50)

        for scaler_cls in [StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler]:
            scaler = scaler_cls()
            scaler.fit(X)

            cloned = clone(scaler)

            # Cloned should not be fitted
            assert not hasattr(cloned, "instance_") or cloned.instance_ is None

    def test_scaler_get_set_params(self, time_series_factory):
        """Test get_params and set_params work correctly."""
        scaler = StandardScaler(with_mean=False, with_std=True)

        params = scaler.get_params()
        assert params["with_mean"] is False
        assert params["with_std"] is True

        scaler.set_params(with_mean=True)
        assert scaler.get_params()["with_mean"] is True


class TestInverseTransformAvailability:
    """Inverse transform availability tests."""

    def test_transformers_without_inverse_transform(self):
        """Test that certain transformers do NOT have inverse_transform."""
        # These transformers should NOT have inverse_transform
        transformers_without_inverse = [
            Normalizer(),
            PolynomialFeatures(),
            SplineTransformer(),
        ]

        for transformer in transformers_without_inverse:
            assert not hasattr(transformer, "inverse_transform"), (
                f"{transformer.__class__.__name__} should not have inverse_transform"
            )

    def test_transformers_with_inverse_transform(self):
        """Test that certain transformers DO have inverse_transform."""
        # These transformers should have inverse_transform
        transformers_with_inverse = [
            PowerTransformer(),
            QuantileTransformer(),
            StandardScaler(),
            MinMaxScaler(),
            RobustScaler(),
            MaxAbsScaler(),
        ]

        for transformer in transformers_with_inverse:
            assert hasattr(transformer, "inverse_transform"), (
                f"{transformer.__class__.__name__} should have inverse_transform"
            )


class TestNormalizer:
    """Normalizer specific tests."""

    def test_normalizer_l2_norm(self, time_series_factory):
        """Test Normalizer with L2 norm produces unit norm rows."""
        X = time_series_factory(length=20, n_components=3)
        normalizer = Normalizer(norm="l2")
        normalizer.fit(X)

        X_norm = normalizer.transform(X)

        # Time column preserved
        assert "time" in X_norm.columns

        # Each row (excluding time) should have L2 norm ~1
        numeric_cols = [c for c in X_norm.columns if c != "time"]
        for i in range(len(X_norm)):
            row_values = [X_norm[col][i] for col in numeric_cols]
            l2_norm = np.sqrt(sum(v**2 for v in row_values))
            assert abs(l2_norm - 1.0) < 1e-10, f"Row {i} L2 norm should be 1, got {l2_norm}"

    def test_normalizer_l1_norm(self, time_series_factory):
        """Test Normalizer with L1 norm."""
        X = time_series_factory(length=20, n_components=2)
        normalizer = Normalizer(norm="l1")
        normalizer.fit(X)

        X_norm = normalizer.transform(X)

        # Each row (excluding time) should have L1 norm ~1
        numeric_cols = [c for c in X_norm.columns if c != "time"]
        for i in range(len(X_norm)):
            row_values = [abs(X_norm[col][i]) for col in numeric_cols]
            l1_norm = sum(row_values)
            assert abs(l1_norm - 1.0) < 1e-10, f"Row {i} L1 norm should be 1, got {l1_norm}"

    def test_normalizer_max_norm(self, time_series_factory):
        """Test Normalizer with max norm."""
        X = time_series_factory(length=20, n_components=2)
        normalizer = Normalizer(norm="max")
        normalizer.fit(X)

        X_norm = normalizer.transform(X)

        # Each row (excluding time) should have max absolute value ~1
        numeric_cols = [c for c in X_norm.columns if c != "time"]
        for i in range(len(X_norm)):
            row_values = [abs(X_norm[col][i]) for col in numeric_cols]
            max_abs = max(row_values)
            assert abs(max_abs - 1.0) < 1e-10, f"Row {i} max abs should be 1, got {max_abs}"

    def test_normalizer_no_inverse_transform(self, time_series_factory):
        """Test that Normalizer does not have inverse_transform method."""
        X = time_series_factory(length=20)
        normalizer = Normalizer()
        normalizer.fit(X)

        assert not hasattr(normalizer, "inverse_transform")


class TestPolynomialFeatures:
    """PolynomialFeatures specific tests."""

    def test_polynomial_features_degree_2(self, time_series_factory):
        """Test PolynomialFeatures generates correct number of features."""
        X = time_series_factory(length=20, n_components=2)
        poly = PolynomialFeatures(degree=2, include_bias=False)
        poly.fit(X)

        X_poly = poly.transform(X)

        # Time column preserved
        assert "time" in X_poly.columns

        # For 2 features, degree=2, no bias: original (2) + degree 2 combinations (3) = 5
        # a, b, a^2, ab, b^2
        numeric_cols = [c for c in X_poly.columns if c != "time"]
        assert len(numeric_cols) == 5

    def test_polynomial_features_interaction_only(self, time_series_factory):
        """Test PolynomialFeatures with interaction_only=True."""
        X = time_series_factory(length=20, n_components=2)
        poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
        poly.fit(X)

        X_poly = poly.transform(X)

        # For 2 features, degree=2, interaction_only, no bias: a, b, ab = 3
        numeric_cols = [c for c in X_poly.columns if c != "time"]
        assert len(numeric_cols) == 3

    def test_polynomial_features_with_bias(self, time_series_factory):
        """Test PolynomialFeatures with bias term."""
        X = time_series_factory(length=20, n_components=2)
        poly = PolynomialFeatures(degree=2, include_bias=True)
        poly.fit(X)

        X_poly = poly.transform(X)

        # With bias: 1 + 2 + 3 = 6 (1, a, b, a^2, ab, b^2)
        numeric_cols = [c for c in X_poly.columns if c != "time"]
        assert len(numeric_cols) == 6

    def test_polynomial_features_fitted_attributes(self, time_series_factory):
        """Test PolynomialFeatures exposes sklearn attributes after fit."""
        X = time_series_factory(length=20, n_components=2)
        poly = PolynomialFeatures(degree=2, include_bias=False)
        poly.fit(X)

        # Check that sklearn attributes are accessible
        assert hasattr(poly, "n_output_features_")
        assert hasattr(poly, "powers_")

        assert poly.n_output_features_ == 5  # 2 features, degree=2, no bias

    def test_polynomial_features_no_inverse_transform(self, time_series_factory):
        """Test that PolynomialFeatures does not have inverse_transform method."""
        X = time_series_factory(length=20)
        poly = PolynomialFeatures()
        poly.fit(X)

        assert not hasattr(poly, "inverse_transform")


class TestPowerTransformer:
    """PowerTransformer specific tests."""

    def test_power_transformer_yeo_johnson(self, time_series_factory):
        """Test PowerTransformer with Yeo-Johnson method."""
        X = time_series_factory(length=50, n_components=2)
        pt = PowerTransformer(method="yeo-johnson")
        pt.fit(X)

        X_transformed = pt.transform(X)

        # Time column preserved
        assert "time" in X_transformed.columns

        # Transformed data should have ~0 mean and ~1 std (standardize=True by default)
        for col in X_transformed.columns:
            if col == "time":
                continue
            mean = X_transformed[col].mean()
            std = X_transformed[col].std()
            assert abs(mean) < 1e-10, f"Column {col} mean should be ~0"
            assert abs(std - 1.0) < 0.1, f"Column {col} std should be ~1"

    def test_power_transformer_box_cox(self, time_series_factory):
        """Test PowerTransformer with Box-Cox method (positive data only)."""
        X = time_series_factory(length=50, n_components=2)
        # Ensure positive values for Box-Cox
        X_positive = X.with_columns([(pl.exclude("time").abs() + 1.0).name.keep()])

        pt = PowerTransformer(method="box-cox")
        pt.fit(X_positive)

        X_transformed = pt.transform(X_positive)
        assert "time" in X_transformed.columns

    def test_power_transformer_inverse_transform(self, time_series_factory):
        """Test PowerTransformer inverse_transform recovers original data."""
        X = time_series_factory(length=50, n_components=2)
        pt = PowerTransformer(method="yeo-johnson")
        pt.fit(X)

        X_transformed = pt.transform(X)
        X_recovered = pt.inverse_transform(X_transformed)

        # Numeric values should match original (within tolerance)
        for col in X.columns:
            if col == "time":
                continue
            np.testing.assert_allclose(
                X_recovered[col].to_numpy(),
                X[col].to_numpy(),
                rtol=1e-5,
                atol=1e-10,  # Allow small absolute tolerance for values near zero
                err_msg=f"Column {col} not recovered correctly",
            )

    def test_power_transformer_fitted_attributes(self, time_series_factory):
        """Test PowerTransformer exposes sklearn attributes after fit."""
        X = time_series_factory(length=50, n_components=3)
        pt = PowerTransformer(method="yeo-johnson")
        pt.fit(X)

        assert hasattr(pt, "lambdas_")

        n_features = len([c for c in X.columns if c != "time"])
        assert len(pt.lambdas_) == n_features

    def test_power_transformer_no_standardize(self, time_series_factory):
        """Test PowerTransformer with standardize=False."""
        X = time_series_factory(length=50, n_components=2)
        pt = PowerTransformer(method="yeo-johnson", standardize=False)
        pt.fit(X)

        X_transformed = pt.transform(X)
        assert "time" in X_transformed.columns

        # Without standardize, mean won't necessarily be 0
        # Just verify it works
        for col in X_transformed.columns:
            if col != "time":
                assert X_transformed[col].is_not_null().all()


class TestQuantileTransformer:
    """QuantileTransformer specific tests."""

    def test_quantile_transformer_uniform(self, time_series_factory):
        """Test QuantileTransformer with uniform distribution."""
        X = time_series_factory(length=100, n_components=2)
        qt = QuantileTransformer(n_quantiles=50, output_distribution="uniform", random_state=42)
        qt.fit(X)

        X_transformed = qt.transform(X)

        # Time column preserved
        assert "time" in X_transformed.columns

        # Transformed values should be in [0, 1] for uniform distribution
        for col in X_transformed.columns:
            if col == "time":
                continue
            assert X_transformed[col].min() >= 0.0, f"Column {col} has min < 0"
            assert X_transformed[col].max() <= 1.0, f"Column {col} has max > 1"

    def test_quantile_transformer_normal(self, time_series_factory):
        """Test QuantileTransformer with normal distribution."""
        X = time_series_factory(length=100, n_components=2)
        qt = QuantileTransformer(n_quantiles=50, output_distribution="normal", random_state=42)
        qt.fit(X)

        X_transformed = qt.transform(X)

        # Transformed data should be roughly centered around 0
        for col in X_transformed.columns:
            if col == "time":
                continue
            mean = X_transformed[col].mean()
            # Mean should be close to 0 for normal distribution
            assert abs(mean) < 0.5, f"Column {col} mean should be ~0, got {mean}"

    def test_quantile_transformer_inverse_transform(self, time_series_factory):
        """Test QuantileTransformer inverse_transform recovers original data."""
        X = time_series_factory(length=100, n_components=2)
        qt = QuantileTransformer(n_quantiles=50, random_state=42)
        qt.fit(X)

        X_transformed = qt.transform(X)
        X_recovered = qt.inverse_transform(X_transformed)

        # Numeric values should match original (within tolerance)
        for col in X.columns:
            if col == "time":
                continue
            np.testing.assert_allclose(
                X_recovered[col].to_numpy(),
                X[col].to_numpy(),
                rtol=1e-3,  # Quantile transform is approximate
                err_msg=f"Column {col} not recovered correctly",
            )

    def test_quantile_transformer_fitted_attributes(self, time_series_factory):
        """Test QuantileTransformer exposes sklearn attributes after fit."""
        X = time_series_factory(length=100, n_components=2)
        qt = QuantileTransformer(n_quantiles=50, random_state=42)
        qt.fit(X)

        assert hasattr(qt, "n_quantiles_")
        assert hasattr(qt, "quantiles_")
        assert hasattr(qt, "references_")

        # n_quantiles_ should be <= requested n_quantiles (capped by n_samples)
        assert qt.n_quantiles_ <= 50

    def test_quantile_transformer_handles_outliers(self, time_series_factory):
        """Test QuantileTransformer reduces impact of outliers."""
        X = time_series_factory(length=50, n_components=1)
        # Add extreme outlier
        X_with_outlier = X.with_columns([
            pl
            .when(pl.col("time").dt.ordinal_day() == 1)
            .then(pl.lit(10000.0))
            .otherwise(pl.exclude("time"))
            .alias(X.columns[1])
        ])

        qt = QuantileTransformer(n_quantiles=10, random_state=42)
        qt.fit(X_with_outlier)
        X_transformed = qt.transform(X_with_outlier)

        # Outlier should be mapped to max quantile (1.0 for uniform)
        assert X_transformed[X_transformed.columns[1]].max() <= 1.0


class TestSplineTransformer:
    """SplineTransformer specific tests."""

    def test_spline_transformer_basic(self, time_series_factory):
        """Test SplineTransformer generates spline basis features."""
        X = time_series_factory(length=50, n_components=2)
        spline = SplineTransformer(n_knots=5, degree=3)
        spline.fit(X)

        X_spline = spline.transform(X)

        # Time column preserved
        assert "time" in X_spline.columns

        # Should generate more features than input
        assert len(X_spline.columns) > len(X.columns)

    def test_spline_transformer_feature_count(self, time_series_factory):
        """Test SplineTransformer generates correct number of features."""
        X = time_series_factory(length=50, n_components=1)
        # For 1 feature: n_splines = n_knots + degree - 1 = 5 + 3 - 1 = 7
        # With bias: 7, without bias would be 6
        spline = SplineTransformer(n_knots=5, degree=3, include_bias=True)
        spline.fit(X)

        X_spline = spline.transform(X)

        # 7 spline basis features + time column
        numeric_cols = [c for c in X_spline.columns if c != "time"]
        assert len(numeric_cols) == 7

    def test_spline_transformer_different_degree(self, time_series_factory):
        """Test SplineTransformer with different polynomial degrees."""
        X = time_series_factory(length=50, n_components=1)

        for degree in [1, 2, 3]:
            spline = SplineTransformer(n_knots=5, degree=degree)
            spline.fit(X)
            X_spline = spline.transform(X)
            assert "time" in X_spline.columns

    def test_spline_transformer_fitted_attributes(self, time_series_factory):
        """Test SplineTransformer exposes sklearn attributes after fit."""
        X = time_series_factory(length=50, n_components=2)
        spline = SplineTransformer(n_knots=5, degree=3)
        spline.fit(X)

        assert hasattr(spline, "bsplines_")
        assert hasattr(spline, "n_features_out_")

        n_features = len([c for c in X.columns if c != "time"])
        assert len(spline.bsplines_) == n_features

    def test_spline_transformer_no_inverse_transform(self, time_series_factory):
        """Test that SplineTransformer does not have inverse_transform method."""
        X = time_series_factory(length=50)
        spline = SplineTransformer()
        spline.fit(X)

        assert not hasattr(spline, "inverse_transform")

    def test_spline_transformer_quantile_knots(self, time_series_factory):
        """Test SplineTransformer with quantile knot positions."""
        X = time_series_factory(length=50, n_components=1)
        spline = SplineTransformer(n_knots=5, degree=3, knots="quantile")
        spline.fit(X)

        X_spline = spline.transform(X)
        assert "time" in X_spline.columns


class TestSklearnTransformer:
    """SklearnTransformer base class tests."""

    def test_sklearn_transformer_with_explicit_transformer(self, time_series_factory):
        """Test SklearnTransformer with explicitly provided transformer class."""
        X = time_series_factory(length=50, n_components=2)

        transformer = SklearnTransformer(transformer=SklearnNormalizer, norm="l2")
        transformer.fit(X)

        X_transformed = transformer.transform(X)

        # Should work like Normalizer
        assert "time" in X_transformed.columns

    def test_sklearn_transformer_requires_transformer_argument(self):
        """Test SklearnTransformer requires transformer argument when no default."""
        with pytest.raises(TypeError, match="missing required keyword argument"):
            SklearnTransformer()

    def test_sklearn_transformer_get_feature_names_out(self, time_series_factory):
        """Test get_feature_names_out returns correct feature names."""
        X = time_series_factory(length=50, n_components=3)
        transformer = Normalizer()
        transformer.fit(X)

        feature_names = transformer.get_feature_names_out()
        expected_names = [c for c in X.columns if c != "time"]

        assert feature_names == expected_names


class TestTransformerPanelData:
    """Transformers with panel data tests."""

    def test_normalizer_panel_data(self, panel_time_series_factory):
        """Test Normalizer with panel data (prefixed columns)."""
        X = panel_time_series_factory(length=50, n_series=2, n_groups=2)

        normalizer = Normalizer(norm="l2")
        normalizer.fit(X)
        X_norm = normalizer.transform(X)

        # Time column preserved
        assert "time" in X_norm.columns

        # All columns should be present
        assert len(X_norm.columns) == len(X.columns)

    def test_polynomial_features_panel_data(self, panel_time_series_factory):
        """Test PolynomialFeatures with panel data."""
        X = panel_time_series_factory(length=50, n_series=2, n_groups=2)

        poly = PolynomialFeatures(degree=2, include_bias=False)
        poly.fit(X)
        X_poly = poly.transform(X)

        # Time column preserved
        assert "time" in X_poly.columns

        # Should have more columns than input
        assert len(X_poly.columns) > len(X.columns)

    def test_power_transformer_panel_data(self, panel_time_series_factory):
        """Test PowerTransformer with panel data."""
        X = panel_time_series_factory(length=50, n_series=2, n_groups=2)

        pt = PowerTransformer(method="yeo-johnson")
        pt.fit(X)
        X_transformed = pt.transform(X)

        # Time column preserved
        assert "time" in X_transformed.columns

        # Same number of columns
        assert len(X_transformed.columns) == len(X.columns)

    def test_quantile_transformer_panel_data(self, panel_time_series_factory):
        """Test QuantileTransformer with panel data."""
        X = panel_time_series_factory(length=100, n_series=2, n_groups=2)

        qt = QuantileTransformer(n_quantiles=50, random_state=42)
        qt.fit(X)
        X_transformed = qt.transform(X)

        # All values should be in [0, 1]
        for col in X_transformed.columns:
            if col == "time":
                continue
            assert X_transformed[col].min() >= 0.0
            assert X_transformed[col].max() <= 1.0

    def test_spline_transformer_panel_data(self, panel_time_series_factory):
        """Test SplineTransformer with panel data."""
        X = panel_time_series_factory(length=50, n_series=2, n_groups=2)

        spline = SplineTransformer(n_knots=5, degree=3)
        spline.fit(X)
        X_spline = spline.transform(X)

        # Time column preserved
        assert "time" in X_spline.columns

        # Should have more columns
        assert len(X_spline.columns) > len(X.columns)


class TestTransformerObservationHorizon:
    """Transformer observation horizon tests."""

    def test_transformer_observation_horizon_is_zero(self, time_series_factory):
        """Test that transformers are stateless (observation_horizon=0)."""
        X = time_series_factory(length=50)

        transformer_classes = [
            Normalizer,
            PolynomialFeatures,
            PowerTransformer,
            QuantileTransformer,
            SplineTransformer,
        ]

        for cls in transformer_classes:
            transformer = cls()
            transformer.fit(X)
            assert transformer.observation_horizon == 0, f"{cls.__name__} should be stateless"


class TestTransformerCloneAndParams:
    """Transformer cloning and serialization tests."""

    def test_transformer_clone(self, time_series_factory):
        """Test that transformers can be cloned."""
        X = time_series_factory(length=50)

        transformer_instances = [
            Normalizer(),
            PolynomialFeatures(degree=2, include_bias=False),
            PowerTransformer(method="yeo-johnson"),
            QuantileTransformer(n_quantiles=10),
            SplineTransformer(n_knots=4),
        ]

        for transformer in transformer_instances:
            transformer.fit(X)
            cloned = clone(transformer)

            # Cloned should not be fitted
            assert not hasattr(cloned, "instance_") or cloned.instance_ is None

    def test_transformer_get_set_params(self):
        """Test get_params and set_params work correctly for transformers."""
        normalizer = Normalizer(norm="l1")
        params = normalizer.get_params()
        assert params["norm"] == "l1"

        normalizer.set_params(norm="l2")
        assert normalizer.get_params()["norm"] == "l2"

        poly = PolynomialFeatures(degree=3, interaction_only=True)
        params = poly.get_params()
        assert params["degree"] == 3
        assert params["interaction_only"] is True

        poly.set_params(degree=2, interaction_only=False)
        assert poly.get_params()["degree"] == 2
        assert poly.get_params()["interaction_only"] is False
