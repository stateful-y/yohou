"""Tests for imputation transformers.

Tests SimpleImputer, TransformedSpaceKNNImputer, SimpleTimeImputer,
SeasonalImputer and SegmentImputer basic functionality.
"""

from datetime import date, datetime, timedelta

import numpy as np
import polars as pl
import pytest

from conftest import run_checks
from yohou.preprocessing import (
    LagTransformer,
    SeasonalImputer,
    SegmentImputer,
    SimpleImputer,
    SimpleTimeImputer,
    TransformedSpaceKNNImputer,
)
from yohou.segmenting import CalendarSegmenter
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

    @pytest.mark.parametrize("strategy", ["mean", "median"])
    def test_fills_nulls_with_strategy_statistic(self, strategy, time_series_with_nulls_factory):
        """SimpleImputer fills every missing value with the column-wise mean/median.

        Asserts both missing-value elimination and the strategy-specific fill
        value, so a regression that imputed the wrong statistic (or zeros) would
        be caught. The fixture marks missing values with float NaN.
        """
        X = time_series_with_nulls_factory(length=50, n_components=2, null_fraction=0.1)
        imputer = SimpleImputer(strategy=strategy)
        imputer.fit(X)
        X_imputed = imputer.transform(X)

        # No missing values remain after imputation.
        assert X_imputed.null_count().sum_horizontal().item() == 0
        for col in (c for c in X_imputed.columns if c != "time"):
            assert X_imputed[col].is_nan().sum() == 0

        # Imputed positions equal the per-column statistic over observed values.
        for col in (c for c in X.columns if c != "time"):
            missing_mask = X[col].is_nan()
            assert missing_mask.sum() > 0, "fixture should produce at least one missing value"
            observed = X[col].to_numpy()
            expected = np.nanmean(observed) if strategy == "mean" else np.nanmedian(observed)
            filled = X_imputed.filter(missing_mask)[col].to_numpy()
            np.testing.assert_allclose(filled, expected, rtol=1e-6)


class TestSeasonalImputer:
    """Test SeasonalImputer functionality."""

    def test_systematic_checks(self, time_series_train_test_factory):
        """Run all systematic checks for SeasonalImputer."""
        transformer = SeasonalImputer(seasonality=7, fill_method="seasonal_mean")

        X_train, X_test = time_series_train_test_factory(train_length=100, test_length=50)

        transformer.fit(X_train)

        run_checks(
            transformer,
            _yield_yohou_transformer_checks(transformer, X_train, None, X_test),
        )

    def test_seasonal_mean_fills_nulls(self, time_series_with_nulls_factory):
        """Test SeasonalImputer seasonal_mean fills nulls."""
        X = time_series_with_nulls_factory(length=70, n_components=1, null_fraction=0.1)
        imputer = SeasonalImputer(seasonality=7, fill_method="seasonal_mean")
        imputer.fit(X)
        X_imputed = imputer.transform(X)

        # Check no nulls remain
        assert X_imputed.null_count().sum_horizontal().item() == 0

    def test_empty_season_bucket_yields_null_not_nan(self):
        """An empty season bucket leaves a polars null, never a float NaN.

        When a season bucket has no observations at fit time, ``_transform``
        must leave the polars null in place rather than writing ``np.nan``
        over it, which would silently violate the data contract.
        """
        time = pl.datetime_range(
            start=datetime(2020, 1, 1),
            end=datetime(2020, 1, 10),
            interval="1d",
            eager=True,
        )
        # seasonality=2: even rows are season 0, odd rows season 1. Season 0 has no
        # observations at fit, so its positions cannot be imputed.
        values = [None, 1.0, None, 2.0, None, 3.0, None, 4.0, None, 5.0]
        X = pl.DataFrame({"time": time, "value": values})

        imputer = SeasonalImputer(seasonality=2, fill_method="seasonal_mean")
        imputer.fit(X)
        X_imputed = imputer.transform(X)

        # The empty-bucket positions remain null, and no float NaN is emitted.
        assert X_imputed["value"].is_nan().sum() == 0
        assert X_imputed["value"].null_count() == 5


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
        # statistics_ must equal the per-column means of the observed values, in
        # input column order. The fixture marks missing values with float NaN, so
        # use nanmean (sklearn ignores NaN when fitting). A regression returning
        # zeros or medians would be caught here.
        expected = [np.nanmean(X[col].to_numpy()) for col in X.columns if col != "time"]
        np.testing.assert_allclose(np.asarray(stats, dtype=float), expected, rtol=1e-6)


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
        imputer = SeasonalImputer(seasonality=7, fill_method="seasonal_median")
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

        imputer = SeasonalImputer(seasonality=12, fill_method="seasonal_mean")
        imputer.fit(X)
        assert imputer._step_seconds_ is None

        result = imputer.transform(X)
        # The gap aligns with a season that has another observed value, so it is filled.
        assert result["val"].null_count() == 0

    def test_empty_seasons_remain_null_in_output(self):
        """A season with no observed data leaves a polars null in the output.

        Contract-based version of the missing-bucket branch: row 1 falls in a
        season whose only observation is itself null, so it cannot be imputed.
        The transform output must keep that position as a polars null and must
        not emit a float NaN.
        """
        # Only 3 monthly points but seasonality=12, so seasons 3..11 have no rows at all.
        time = pl.datetime_range(start=datetime(2020, 1, 1), end=datetime(2020, 3, 1), interval="1mo", eager=True)
        X = pl.DataFrame({"time": time, "val": [1.0, None, 3.0]})

        imputer = SeasonalImputer(seasonality=12, fill_method="seasonal_mean")
        imputer.fit(X)
        X_imputed = imputer.transform(X)

        # The un-imputable row stays null; the observed rows pass through unchanged.
        assert X_imputed["val"].to_list() == [1.0, None, 3.0]
        assert X_imputed["val"].is_nan().sum() == 0


def _hourly(start: datetime, values: list, tz: str | None = None) -> pl.DataFrame:
    """Hourly frame with one ``value`` column, optionally localised to ``tz``."""
    times = pl.Series("time", [start + timedelta(hours=h) for h in range(len(values))])
    if tz is not None:
        times = times.dt.replace_time_zone(tz)
    return pl.DataFrame({"time": times, "value": pl.Series(values, dtype=pl.Float64)})


class TestSegmentImputer:
    """Test SegmentImputer functionality."""

    def test_systematic_checks(self):
        """Run all systematic checks on a local-hour imputer with tz-aware data."""
        rng = np.random.default_rng(0)
        X = _hourly(datetime(2024, 3, 25), list(rng.normal(size=24 * 10)), tz="UTC")
        X_train, X_test = X[: 24 * 7], X[24 * 7 :]
        transformer = SegmentImputer(CalendarSegmenter("hour", time_zone="Europe/Paris")).fit(X_train)
        run_checks(transformer, _yield_yohou_transformer_checks(transformer, X_train, None, X_test))

    def test_missing_hour_filled_from_same_hour(self):
        """The null 03:00 value is filled with the mean of the other 03:00 values."""
        values = [float(h) for _ in range(3) for h in range(24)]
        values[3], values[27], values[51] = 10.0, 12.0, None
        X = _hourly(datetime(2024, 1, 1), values)
        out = SegmentImputer(CalendarSegmenter("hour")).fit_transform(X)
        expected = list(values)
        expected[51] = 11.0
        assert out["value"].to_list() == expected

    def test_local_time_across_daylight_saving(self):
        """18:00 Paris is 17:00 UTC in March and 16:00 UTC in April; the fill follows local time."""
        fit_values = [1.0 if h % 24 == 17 else 0.0 for h in range(24 * 30)]
        X_fit = _hourly(datetime(2024, 3, 1), fit_values, tz="UTC")
        imputer = SegmentImputer(CalendarSegmenter("hour", time_zone="Europe/Paris")).fit(X_fit)
        april = [None if h in (16, 17) else 0.0 for h in range(24)]
        out = imputer.transform(_hourly(datetime(2024, 4, 1), april, tz="UTC"))
        assert out["value"][16] == 1.0
        assert out["value"][17] == 0.0

    def test_median_fill(self):
        """With fill_method='median' the 03:00 values 1, 2 and 100 fill with 2."""
        values = [0.0] * 96
        values[3], values[27], values[51], values[75] = 1.0, 2.0, 100.0, None
        out = SegmentImputer(CalendarSegmenter("hour"), fill_method="median").fit_transform(
            _hourly(datetime(2024, 1, 1), values)
        )
        assert out["value"][75] == 2.0

    def test_integer_column(self):
        """An integer column is returned as Float64 with the mean filled in."""
        X = pl.DataFrame({
            "time": pl.datetime_range(datetime(2024, 1, 1), datetime(2024, 1, 3), interval="1d", eager=True),
            "count": pl.Series([10, 11, None], dtype=pl.Int64),
        })
        out = SegmentImputer().fit_transform(X)
        assert out["count"].dtype == pl.Float64
        assert out["count"].to_list() == [10.0, 11.0, 10.5]

    def test_invalid_fill_method(self):
        """fill_method='mode' is rejected at fit."""
        with pytest.raises(ValueError, match="fill_method"):
            SegmentImputer(fill_method="mode").fit(_hourly(datetime(2024, 1, 1), [1.0, 2.0]))

    def test_row_outside_every_segment(self):
        """A null at 03:00, outside business hours, takes the pooled statistic."""
        values = [float(h) for _ in range(2) for h in range(24)]
        values[27] = None
        imputer = SegmentImputer(CalendarSegmenter("hour", values=list(range(8, 19))))
        out = imputer.fit_transform(_hourly(datetime(2024, 1, 1), values))
        pooled = np.mean([v for v in values if v is not None])
        assert out["value"][27] == pytest.approx(pooled)
        assert imputer.fill_values_["value"] == pytest.approx(pooled)

    def test_label_unseen_at_fit(self):
        """A null in June, unseen when fitting on January to March, takes the pooled statistic."""
        winter = pl.DataFrame({
            "time": pl.date_range(date(2024, 1, 1), date(2024, 3, 31), eager=True),
        }).with_columns(pl.col("time").dt.month().cast(pl.Float64).alias("value"))
        imputer = SegmentImputer(CalendarSegmenter("month")).fit(winter)
        june = pl.DataFrame({
            "time": pl.date_range(date(2024, 6, 1), date(2024, 6, 3), eager=True),
            "value": [None, 5.0, 5.0],
        })
        out = imputer.transform(june)
        assert out["value"][0] == pytest.approx(winter["value"].mean())

    def test_no_segmenter(self):
        """Without a segmenter every missing value takes the column mean of the fit data."""
        values = [1.0, None, 3.0, float("nan"), 5.0]
        out = SegmentImputer().fit_transform(_hourly(datetime(2024, 1, 1), values))
        assert out["value"].to_list() == [1.0, 3.0, 3.0, 3.0, 5.0]

    def test_unfillable_value_is_null_not_nan(self):
        """A column with no finite value at fit keeps its gaps as null, never NaN."""
        X = _hourly(datetime(2024, 1, 1), [None, float("nan"), None])
        out = SegmentImputer(CalendarSegmenter("hour")).fit_transform(X)
        assert out["value"].null_count() == 3
        assert out["value"].is_nan().sum() == 0

    def test_infinite_values_neither_used_nor_filled(self):
        """An infinite value is excluded from the statistic and left in place."""
        X = _hourly(datetime(2024, 1, 1), [float("inf"), 1.0, None, 3.0])
        out = SegmentImputer().fit_transform(X)
        assert out["value"].to_list() == [float("inf"), 1.0, 2.0, 3.0]

    def test_columns_get_their_own_statistics(self):
        """Each column is filled from its own per-label statistic, recorded in segment_fill_values_."""
        hours = [float(h % 24) for h in range(48)]
        a, b = list(hours), [10.0 * h for h in hours]
        a[27], b[28] = None, None
        X = pl.DataFrame({
            "time": [datetime(2024, 1, 1) + timedelta(hours=h) for h in range(48)],
            "a__value": a,
            "b__value": b,
        })
        imputer = SegmentImputer(CalendarSegmenter("hour")).fit(X)
        out = imputer.transform(X)
        assert (out["a__value"][27], out["b__value"][28]) == (3.0, 40.0)
        assert imputer.segment_fill_values_["a__value"][3] == 3.0
        assert imputer.segment_fill_values_["b__value"][4] == 40.0

    def test_every_column_returned_as_float64(self):
        """Value columns become Float64 whether or not they had gaps."""
        X = pl.DataFrame({
            "time": [datetime(2024, 1, 1) + timedelta(hours=h) for h in range(3)],
            "full": pl.Series([1, 2, 3], dtype=pl.Int64),
            "gappy": pl.Series([1, None, 3], dtype=pl.Int64),
        })
        out = SegmentImputer().fit_transform(X)
        assert out.schema["full"] == pl.Float64
        assert out.schema["gappy"] == pl.Float64

    def test_chunked_transform(self):
        """Transforming in two consecutive chunks equals transforming whole."""
        values = [float(h % 24) if h % 5 else None for h in range(24 * 4)]
        X = _hourly(datetime(2024, 1, 1), values, tz="UTC")
        imputer = SegmentImputer(CalendarSegmenter("hour", time_zone="Europe/Paris")).fit(X)
        whole = imputer.transform(X)
        chunks = pl.concat([imputer.transform(X[:37]), imputer.transform(X[37:])])
        assert chunks.equals(whole)
        assert imputer.__sklearn_tags__().transformer_tags.batch_invariant
