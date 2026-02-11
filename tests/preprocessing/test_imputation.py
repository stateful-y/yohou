"""Tests for imputation transformers.

Tests SimpleImputer, TransformedSpaceKNNImputer, SimpleTimeImputer,
and SeasonalImputer basic functionality.
"""

from conftest import run_checks
from yohou.preprocessing import (
    LagTransformer,
    SeasonalImputer,
    SimpleImputer,
    SimpleTimeImputer,
    TransformedSpaceKNNImputer,
)
from yohou.testing import _yield_yohou_transformer_checks

# Inverse transform checks to skip for sklearn imputers.
# sklearn's SimpleImputer has inverse_transform but it only works
# when add_indicator=True. Like sklearn's GridSearchCV, we expose the method
# if it exists and let sklearn raise an error if conditions aren't met.
_IMPUTER_SKIP_CHECKS = {
    "check_inverse_transform_identity",
    "check_inverse_transform_round_trip",
    "check_inverse_observe_transform_identity",
}


class TestSimpleImputer:
    """Test SimpleImputer functionality."""

    def test_systematic_checks(self, time_series_train_test_factory):
        """Run all systematic checks for SimpleImputer."""
        transformer = SimpleImputer(strategy="mean")

        X_train, X_test = time_series_train_test_factory(train_length=100, test_length=50)

        transformer.fit(X_train)

        run_checks(
            transformer,
            _yield_yohou_transformer_checks(transformer, X_train, None, X_test),
            expected_failures=_IMPUTER_SKIP_CHECKS,
        )

    def test_mean_fills_nulls(self, time_series_with_nulls_factory):
        """Test SimpleImputer mean strategy fills all nulls."""
        X = time_series_with_nulls_factory(length=50, n_components=2, null_fraction=0.1)
        imputer = SimpleImputer(strategy="mean")
        imputer.fit(X)
        X_imputed = imputer.transform(X)

        # Check no nulls remain
        assert X_imputed.null_count().sum_horizontal().item() == 0

    def test_median_fills_nulls(self, time_series_with_nulls_factory):
        """Test SimpleImputer median strategy fills all nulls."""
        X = time_series_with_nulls_factory(length=50, n_components=2, null_fraction=0.1)
        imputer = SimpleImputer(strategy="median")
        imputer.fit(X)
        X_imputed = imputer.transform(X)

        # Check no nulls remain
        assert X_imputed.null_count().sum_horizontal().item() == 0

    def test_preserves_time_column(self, time_series_with_nulls_factory):
        """Test SimpleImputer preserves time column."""
        X = time_series_with_nulls_factory(length=50, n_components=2)
        imputer = SimpleImputer(strategy="mean")
        imputer.fit(X)
        X_imputed = imputer.transform(X)

        assert "time" in X_imputed.columns
        assert len(X_imputed) == len(X)


class TestSeasonalImputer:
    """Test SeasonalImputer functionality."""

    def test_systematic_checks(self, time_series_train_test_factory):
        """Run all systematic checks for SeasonalImputer."""
        transformer = SeasonalImputer(period=7, fill_method="seasonal_mean")

        X_train, X_test = time_series_train_test_factory(train_length=100, test_length=50)

        transformer.fit(X_train)

        run_checks(
            transformer,
            _yield_yohou_transformer_checks(transformer, X_train, None, X_test),
        )

    def test_seasonal_mean_fills_nulls(self, time_series_with_nulls_factory):
        """Test SeasonalImputer seasonal_mean fills nulls."""
        X = time_series_with_nulls_factory(length=70, n_components=1, null_fraction=0.1)
        imputer = SeasonalImputer(period=7, fill_method="seasonal_mean")
        imputer.fit(X)
        X_imputed = imputer.transform(X)

        # Check no nulls remain
        assert X_imputed.null_count().sum_horizontal().item() == 0

    def test_preserves_time_column(self, time_series_with_nulls_factory):
        """Test SeasonalImputer preserves time column."""
        X = time_series_with_nulls_factory(length=70, n_components=1)
        imputer = SeasonalImputer(period=7, fill_method="seasonal_mean")
        imputer.fit(X)
        X_imputed = imputer.transform(X)

        assert "time" in X_imputed.columns
        assert len(X_imputed) == len(X)


class TestSimpleTimeImputer:
    """Test SimpleTimeImputer functionality."""

    def test_systematic_checks(self, time_series_train_test_factory):
        """Run all systematic checks for SimpleTimeImputer."""
        transformer = SimpleTimeImputer(method="linear")

        X_train, X_test = time_series_train_test_factory(train_length=100, test_length=50)

        transformer.fit(X_train)

        run_checks(
            transformer,
            _yield_yohou_transformer_checks(transformer, X_train, None, X_test),
        )


class TestTransformedSpaceKNNImputer:
    """Test TransformedSpaceKNNImputer functionality."""

    # With a transformer the output schema changes (e.g., lag features), so
    # checks that compare output feature names against input names will fail.
    # Also not invertible, so skip inverse checks.
    _SKIP_CHECKS = {
        *_IMPUTER_SKIP_CHECKS,
    }

    def test_systematic_checks_no_transformer(self, time_series_train_test_factory):
        """Run systematic checks for TransformedSpaceKNNImputer without transformer."""
        transformer = TransformedSpaceKNNImputer(n_neighbors=3)

        X_train, X_test = time_series_train_test_factory(train_length=100, test_length=50)

        transformer.fit(X_train)

        run_checks(
            transformer,
            _yield_yohou_transformer_checks(transformer, X_train, None, X_test),
            expected_failures=self._SKIP_CHECKS,
        )

    def test_systematic_checks_with_transformer(self, time_series_train_test_factory):
        """Run systematic checks with a LagTransformer."""
        transformer = TransformedSpaceKNNImputer(
            n_neighbors=3,
            transformer=LagTransformer(lag=3),
        )

        X_train, X_test = time_series_train_test_factory(train_length=100, test_length=50)

        transformer.fit(X_train)

        run_checks(
            transformer,
            _yield_yohou_transformer_checks(transformer, X_train, None, X_test),
            expected_failures=self._SKIP_CHECKS,
        )

    def test_no_transformer_fills_nulls(self, time_series_with_nulls_factory):
        """Test that without a transformer it behaves like KNNImputer."""
        X = time_series_with_nulls_factory(length=50, n_components=2, null_fraction=0.05)
        imputer = TransformedSpaceKNNImputer(n_neighbors=3)
        imputer.fit(X)
        X_imputed = imputer.transform(X)

        assert X_imputed.null_count().sum_horizontal().item() == 0
        assert X_imputed.columns == X.columns

    def test_with_lag_transformer(self, time_series_factory):
        """Test KNN imputation in lag feature space."""
        X = time_series_factory(length=50, n_components=1, seed=42)

        imputer = TransformedSpaceKNNImputer(
            n_neighbors=3,
            transformer=LagTransformer(lag=3),
        )
        imputer.fit(X)
        X_imputed = imputer.transform(X)

        # Output should have lag columns, not original columns
        assert "time" in X_imputed.columns
        assert any("lag" in c for c in X_imputed.columns)
        # No nulls in output
        assert X_imputed.null_count().sum_horizontal().item() == 0

    def test_observation_horizon_inherited(self, time_series_factory):
        """Test observation_horizon is inherited from inner transformer."""
        X = time_series_factory(length=50, n_components=1, seed=42)

        # No transformer => obs_horizon = 0
        imputer_plain = TransformedSpaceKNNImputer(n_neighbors=3)
        imputer_plain.fit(X)
        assert imputer_plain.observation_horizon == 0

        # LagTransformer(lag=5) => obs_horizon = 5
        imputer_lag = TransformedSpaceKNNImputer(
            n_neighbors=3,
            transformer=LagTransformer(lag=5),
        )
        imputer_lag.fit(X)
        assert imputer_lag.observation_horizon == 5

    def test_preserves_time_column(self, time_series_factory):
        """Test time column is preserved in output."""
        X = time_series_factory(length=30, n_components=2, seed=42)

        imputer = TransformedSpaceKNNImputer(
            n_neighbors=3,
            transformer=LagTransformer(lag=2),
        )
        imputer.fit(X)
        X_imputed = imputer.transform(X)

        assert "time" in X_imputed.columns

    def test_degeneracy_matches_knn_imputer(self, time_series_with_nulls_factory):
        """Test that transformer=None produces same result as sklearn KNNImputer."""
        from sklearn.impute import KNNImputer as sklearn_KNNImputer

        X = time_series_with_nulls_factory(length=50, n_components=2, null_fraction=0.05, seed=42)

        # Fit sklearn KNNImputer directly on numeric columns
        import polars.selectors as cs

        X_numeric = X.select(~cs.by_name("time"))
        plain = sklearn_KNNImputer(n_neighbors=3)
        plain.fit(X_numeric.to_numpy())
        X_plain_np = plain.transform(X_numeric.to_numpy())

        transformed = TransformedSpaceKNNImputer(n_neighbors=3)
        transformed.fit(X)
        X_transformed = transformed.transform(X)

        # Results should be identical (same underlying sklearn KNNImputer)
        import numpy as np

        for i, col in enumerate(X_numeric.columns):
            np.testing.assert_array_almost_equal(
                X_plain_np[:, i],
                X_transformed[col].to_numpy(),
            )
