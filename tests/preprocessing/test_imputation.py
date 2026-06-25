"""Tests for imputation transformers.

Tests SimpleImputer, TransformedSpaceKNNImputer, SimpleTimeImputer,
and SeasonalImputer basic functionality.
"""

from datetime import datetime

import numpy as np
import polars as pl
import pytest

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

    def test_empty_season_bucket_yields_null_not_nan(self):
        """An empty season bucket leaves a polars null, never a float NaN.

        Regression for the 2026-06-15 QA finding: when a season bucket had no
        observations at fit time, ``_transform`` wrote ``np.nan`` over the
        polars null, silently violating the data contract.
        """
        time = pl.datetime_range(
            start=datetime(2020, 1, 1),
            end=datetime(2020, 1, 10),
            interval="1d",
            eager=True,
        )
        # period=2: even rows are season 0, odd rows season 1. Season 0 has no
        # observations at fit, so its positions cannot be imputed.
        values = [None, 1.0, None, 2.0, None, 3.0, None, 4.0, None, 5.0]
        X = pl.DataFrame({"time": time, "value": values})

        imputer = SeasonalImputer(period=2, fill_method="seasonal_mean")
        imputer.fit(X)
        X_imputed = imputer.transform(X)

        # The empty-bucket positions remain null, and no float NaN is emitted.
        assert X_imputed["value"].is_nan().sum() == 0
        assert X_imputed["value"].null_count() == 5

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


class TestSimpleImputerStatistics:
    """Tests for SimpleImputer.statistics_ property."""

    def test_statistics_after_fit(self, time_series_with_nulls_factory):
        """statistics_ returns underlying imputer statistics."""
        from yohou.preprocessing.imputation import SimpleImputer

        X = time_series_with_nulls_factory(length=50, n_components=2, null_fraction=0.1, seed=42)
        imputer = SimpleImputer(strategy="mean")
        imputer.fit(X)

        stats = imputer.statistics_
        assert stats is not None
        assert len(stats) > 0


class TestTimeSeriesInterpolatorMethods:
    """Tests for various interpolation methods in TimeSeriesInterpolator."""

    @pytest.fixture
    def df_with_nulls(self):
        """DataFrame with null values for interpolation testing."""
        from datetime import datetime, timedelta

        time = pl.datetime_range(
            start=datetime(2020, 1, 1),
            end=datetime(2020, 1, 1) + timedelta(days=19),
            interval="1d",
            eager=True,
        )
        values = [
            1.0,
            None,
            3.0,
            None,
            5.0,
            6.0,
            None,
            8.0,
            9.0,
            None,
            11.0,
            12.0,
            None,
            14.0,
            15.0,
            None,
            17.0,
            18.0,
            19.0,
            20.0,
        ]
        return pl.DataFrame({"time": time, "val": values})

    def test_forward_fill(self, df_with_nulls):
        """method='forward' fills nulls forward."""
        from yohou.preprocessing.imputation import SimpleTimeImputer

        imputer = SimpleTimeImputer(method="forward")
        imputer.fit(df_with_nulls)
        result = imputer.transform(df_with_nulls)
        assert result["val"].null_count() == 0
        assert result["val"][1] == 1.0

    def test_backward_fill(self, df_with_nulls):
        """method='backward' fills nulls backward."""
        from yohou.preprocessing.imputation import SimpleTimeImputer

        imputer = SimpleTimeImputer(method="backward")
        imputer.fit(df_with_nulls)
        result = imputer.transform(df_with_nulls)
        assert result["val"].null_count() == 0
        assert result["val"][1] == 3.0

    def test_nearest_fill(self, df_with_nulls):
        """method='nearest' fills from both directions."""
        from yohou.preprocessing.imputation import SimpleTimeImputer

        imputer = SimpleTimeImputer(method="nearest")
        imputer.fit(df_with_nulls)
        result = imputer.transform(df_with_nulls)
        assert result["val"].null_count() == 0

    def test_nearest_selects_by_temporal_distance(self):
        """method='nearest' splits an interior gap toward the closer anchor."""
        from datetime import datetime

        from yohou.preprocessing.imputation import SimpleTimeImputer

        X = pl.DataFrame({
            "time": [datetime(2020, 1, i) for i in range(1, 6)],
            "val": [1.0, None, None, None, 5.0],
        })
        result = SimpleTimeImputer(method="nearest").fit(X).transform(X)
        # Must not collapse to an unconditional forward-fill.
        assert result["val"].to_list() != [1.0, 1.0, 1.0, 1.0, 5.0]
        # Points nearer the leading anchor take 1.0; nearer the trailing take 5.0.
        assert result["val"].to_list() == [1.0, 1.0, 1.0, 5.0, 5.0]

    def test_fill_both(self, df_with_nulls):
        """method='fill_both' fills forward then backward."""
        from yohou.preprocessing.imputation import SimpleTimeImputer

        imputer = SimpleTimeImputer(method="fill_both")
        imputer.fit(df_with_nulls)
        result = imputer.transform(df_with_nulls)
        assert result["val"].null_count() == 0


class TestSeasonalImputerMedian:
    """Tests for SeasonalImputer with seasonal_median strategy."""

    def test_seasonal_median(self, time_series_with_nulls_factory):
        """seasonal_median uses median instead of mean for imputation."""
        X = time_series_with_nulls_factory(length=70, n_components=1, null_fraction=0.1, seed=42)
        imputer = SeasonalImputer(period=7, fill_method="seasonal_median")
        imputer.fit(X)
        X_imputed = imputer.transform(X)
        assert X_imputed.null_count().sum_horizontal().item() == 0


class TestSeasonalImputerIrregularInterval:
    """SeasonalImputer on calendar (month/year) intervals with no fixed step."""

    def test_monthly_interval_uses_row_position_fallback(self):
        """Calendar intervals have no fixed second-step; season index falls back to row position."""
        time = pl.datetime_range(start=datetime(2020, 1, 1), end=datetime(2021, 12, 1), interval="1mo", eager=True)
        values = [float(i) for i in range(len(time))]
        values[3] = None  # introduce a gap
        X = pl.DataFrame({"time": time, "val": values})

        imputer = SeasonalImputer(period=12, fill_method="seasonal_mean")
        imputer.fit(X)
        assert imputer._step_seconds_ is None

        result = imputer.transform(X)
        # The gap aligns with a season that has another observed value, so it is filled.
        assert result["val"].null_count() == 0

    def test_empty_seasons_remain_nan(self):
        """A season with no observed data stays NaN (covers the missing-bucket branch)."""
        # Only 3 monthly points but period=12, so seasons 3..11 have no rows at all.
        time = pl.datetime_range(start=datetime(2020, 1, 1), end=datetime(2020, 3, 1), interval="1mo", eager=True)
        X = pl.DataFrame({"time": time, "val": [1.0, None, 3.0]})

        imputer = SeasonalImputer(period=12, fill_method="seasonal_mean")
        imputer.fit(X)

        # Season index 1 (row 1) has only a null, so its seasonal value is NaN.
        seasonal = imputer.seasonal_values_["val"]
        assert np.isnan(seasonal[1])
