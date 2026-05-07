"""Tests for validate_data functionality."""

from datetime import datetime

import polars as pl
import pytest
from polars.exceptions import ColumnNotFoundError

from yohou.interval import IntervalReductionForecaster
from yohou.metrics import MeanAbsoluteError
from yohou.model_selection import ExpandingWindowSplitter
from yohou.point import SeasonalNaive
from yohou.preprocessing import LagTransformer
from yohou.utils.validate_data import validate_scorer_data


@pytest.fixture
def sample_time_series():
    """Create a sample time series for testing."""
    time = pl.datetime_range(
        start=datetime(2020, 1, 1),
        end=datetime(2020, 1, 20),
        interval="1d",
        eager=True,
    )
    return pl.DataFrame({"time": time, "value": range(20)})


@pytest.fixture
def sample_forecaster_data():
    """Create sample forecaster data (target + features)."""
    time = pl.datetime_range(
        start=datetime(2020, 1, 1),
        end=datetime(2020, 1, 20),
        interval="1d",
        eager=True,
    )
    y = pl.DataFrame({"time": time, "target": range(20)})
    X = pl.DataFrame({"time": time, "feature": range(20, 40)})
    return y, X


@pytest.fixture
def sample_transformer_data():
    """Create sample transformer data."""
    time = pl.datetime_range(
        start=datetime(2020, 1, 1),
        end=datetime(2020, 1, 20),
        interval="1d",
        eager=True,
    )
    X = pl.DataFrame({"time": time, "feature1": range(20), "feature2": range(20, 40)})
    return X


@pytest.fixture
def sample_scorer_data():
    """Create sample scorer data (truth and predictions)."""
    time = pl.datetime_range(
        start=datetime(2020, 1, 1),
        end=datetime(2020, 1, 10),
        interval="1d",
        eager=True,
    )
    y_true = pl.DataFrame({"time": time, "target": range(10)})
    y_pred = pl.DataFrame({
        "vintage_time": time[:10],
        "time": time[:10],
        "target": range(10),  # Use int to match y_true type
    })
    return y_true, y_pred


@pytest.fixture
def sample_splitter_data():
    """Create sample splitter data."""
    time = pl.datetime_range(
        start=datetime(2020, 1, 1),
        end=datetime(2020, 1, 30),
        interval="1d",
        eager=True,
    )
    y = pl.DataFrame({"time": time, "target": range(30)})
    X = pl.DataFrame({"time": time, "feature": range(30, 60)})
    return y, X


class TestValidateForecasterData:
    """Tests for validate_data in forecaster context."""

    def test_validate_data_fit_context(self, sample_forecaster_data):
        """Test validate_data in fit context (reset=True)."""
        y, X = sample_forecaster_data
        forecaster = SeasonalNaive(seasonality=2)

        # Fit should call validate_data with reset=True
        forecaster.fit(y, X, forecasting_horizon=2)

        # Check that fitted attributes were set
        assert hasattr(forecaster, "interval_")
        assert hasattr(forecaster, "n_features_in_")
        assert hasattr(forecaster, "local_y_schema_")
        # SeasonalNaive uses target_as_feature=None, so features are X only
        assert forecaster.n_features_in_ == 1

    def test_validate_data_observe_context(self, sample_forecaster_data):
        """Test validate_data in observe context (reset=False)."""
        y, X = sample_forecaster_data
        forecaster = SeasonalNaive(seasonality=2)
        forecaster.fit(y[:15], X[:15], forecasting_horizon=2)

        # Observe should call validate_data with reset=False (schema check only, no interval validation yet)
        forecaster.observe(y[15:], X[15:])

        # Should not change n_features_in_
        # SeasonalNaive uses target_as_feature=None, so features are X only
        assert forecaster.n_features_in_ == 1

    def test_validate_data_schema_consistency(self, sample_forecaster_data):
        """Test that schema consistency is enforced."""
        y, X = sample_forecaster_data
        forecaster = SeasonalNaive(seasonality=2)
        forecaster.fit(y[:15], forecasting_horizon=2)

        # Wrong column name should fail
        y_wrong = y[15:].with_columns(pl.col("target").alias("wrong_col")).drop("target")

        with pytest.raises(ColumnNotFoundError):
            forecaster.observe(y_wrong)

    def test_validate_data_interval_consistency_fit(self, sample_forecaster_data):
        """Test that interval consistency is checked during fit."""
        y, X = sample_forecaster_data

        # Create data with inconsistent intervals
        time_wrong = [
            datetime(2020, 1, 1),
            datetime(2020, 1, 2),
            datetime(2020, 1, 4),  # Skip day 3 - inconsistent interval
            datetime(2020, 1, 5),
        ]
        y_wrong_interval = pl.DataFrame({"time": time_wrong, "target": [10, 11, 12, 13]})

        forecaster = SeasonalNaive(seasonality=2)
        with pytest.raises(ValueError, match="Time series has inconsistent intervals"):
            forecaster.fit(y_wrong_interval, forecasting_horizon=2)

    def test_validate_data_panel_fit(self):
        """Test validate_data with panel data in fit context."""
        time = pl.datetime_range(
            start=datetime(2020, 1, 1),
            end=datetime(2020, 1, 10),
            interval="1d",
            eager=True,
        )
        y = pl.DataFrame({
            "time": time,
            "sales__store_1": range(10),
            "sales__store_2": range(10, 20),
        })

        forecaster = SeasonalNaive(seasonality=2)
        forecaster.fit(y, forecasting_horizon=2)

        # Should detect panel structure
        assert forecaster.groups_ == ["sales"]
        assert "store_1" in forecaster.local_y_schema_
        assert "store_2" in forecaster.local_y_schema_

    def test_validate_data_panel_observe(self):
        """Test validate_data with panel data in observe context."""
        time = pl.datetime_range(
            start=datetime(2020, 1, 1),
            end=datetime(2020, 1, 15),
            interval="1d",
            eager=True,
        )
        y = pl.DataFrame({
            "time": time,
            "sales__store_1": range(15),
            "sales__store_2": range(15, 30),
        })

        forecaster = SeasonalNaive(seasonality=2)
        forecaster.fit(y[:10], forecasting_horizon=2)

        # Observe with panel data
        forecaster.observe(y[10:])

        # Should maintain panel structure
        assert forecaster.groups_ == ["sales"]

    def test_validate_data_with_none_X(self):
        """Test validate_data works when X is None."""
        time = pl.datetime_range(
            start=datetime(2020, 1, 1),
            end=datetime(2020, 1, 10),
            interval="1d",
            eager=True,
        )
        y = pl.DataFrame({"time": time, "target": range(10)})

        forecaster = SeasonalNaive(seasonality=2)
        forecaster.fit(y, X_actual=None, forecasting_horizon=2)

        # Observe with X_actual=None should work
        forecaster.observe(y[-2:], X_actual=None)

        assert forecaster._y_observed.shape[0] == 2

    def test_validate_pre_fit_raises_when_no_features(self):
        """Test that fit raises ValueError when target_as_feature=None, X=None, and ignores_exogenous=False."""
        from sklearn.linear_model import QuantileRegressor
        from sklearn.multioutput import MultiOutputRegressor

        time = pl.datetime_range(
            start=datetime(2020, 1, 1),
            end=datetime(2020, 1, 20),
            interval="1d",
            eager=True,
        )
        y = pl.DataFrame({"time": time, "target": range(20)})

        forecaster = IntervalReductionForecaster(
            estimator=MultiOutputRegressor(QuantileRegressor()),
            target_as_feature=None,
        )

        with pytest.raises(ValueError, match="target_as_feature=None requires X_actual to be provided"):
            forecaster.fit(y, X_actual=None, forecasting_horizon=3)

    def test_validate_data_preserves_column_order(self):
        """Test that validate_data ensures time column is handled correctly."""
        time = pl.datetime_range(
            start=datetime(2020, 1, 1),
            end=datetime(2020, 1, 10),
            interval="1d",
            eager=True,
        )
        # Create with columns in expected order
        y = pl.DataFrame({
            "time": time,
            "target": range(10),
        })

        forecaster = SeasonalNaive(seasonality=2)
        forecaster.fit(y, forecasting_horizon=2)

        # Verify that forecaster stores observations correctly
        assert "time" in forecaster._y_observed.columns
        assert "target" in forecaster._y_observed.columns
        assert forecaster._y_observed.shape[0] == 2  # observation_horizon = seasonality = 2


class TestValidateTransformerData:
    """Tests for validate_transformer_data."""

    def test_validate_transformer_fit_context(self, sample_transformer_data):
        """Test validate_transformer_data in fit context (reset=True)."""
        X = sample_transformer_data
        transformer = LagTransformer(lag=[1, 2])

        # Fit should call validate_transformer_data with reset=True
        transformer.fit(X)

        # Check that fitted attributes were set
        assert hasattr(transformer, "interval_")
        assert hasattr(transformer, "n_features_in_")
        assert transformer.n_features_in_ == 2
        assert transformer.observation_horizon == 2

    def test_validate_transformer_transform_context(self, sample_transformer_data):
        """Test validate_transformer_data in transform context (reset=False)."""
        X = sample_transformer_data
        transformer = LagTransformer(lag=[1, 2])
        transformer.fit(X[:15])

        # Transform should call validate_transformer_data with reset=False
        X_transformed = transformer.transform(X[15:])

        # Should return DataFrame with lag features
        assert isinstance(X_transformed, pl.DataFrame)
        assert "time" in X_transformed.columns
        assert "feature1_lag_1" in X_transformed.columns
        assert "feature2_lag_2" in X_transformed.columns

    def test_validate_transformer_schema_consistency(self, sample_transformer_data):
        """Test that transformer schema consistency is enforced."""
        X = sample_transformer_data
        transformer = LagTransformer(lag=[1, 2])
        transformer.fit(X[:15])

        # Wrong column name should fail
        X_wrong = X[15:].with_columns(pl.col("feature1").alias("wrong_col")).drop("feature1")

        with pytest.raises(ColumnNotFoundError):
            transformer.transform(X_wrong)

    def test_validate_transformer_interval_consistency_fit(self, sample_transformer_data):
        """Test that transformer checks interval consistency during fit."""
        # Create data with inconsistent intervals
        time_wrong = [
            datetime(2020, 1, 1),
            datetime(2020, 1, 2),
            datetime(2020, 1, 4),  # Skip day 3 - inconsistent interval
            datetime(2020, 1, 5),
        ]
        X_wrong_interval = pl.DataFrame({
            "time": time_wrong,
            "feature1": [10, 11, 12, 13],
            "feature2": [20, 21, 22, 23],
        })

        transformer = LagTransformer(lag=1)
        with pytest.raises(ValueError, match="Time series has inconsistent intervals"):
            transformer.fit(X_wrong_interval)

    def test_validate_transformer_panel_fit(self):
        """Test validate_transformer_data with panel data in fit context."""
        time = pl.datetime_range(
            start=datetime(2020, 1, 1),
            end=datetime(2020, 1, 10),
            interval="1d",
            eager=True,
        )
        X = pl.DataFrame({
            "time": time,
            "temp__sensor_1": range(10),
            "temp__sensor_2": range(10, 20),
        })

        transformer = LagTransformer(lag=1)
        transformer.fit(X)

        # Should detect panel structure (check internal state)
        assert hasattr(transformer, "interval_")

    def test_validate_transformer_observe_extends_memory(self, sample_transformer_data):
        """Test that transformer observe extends observation memory."""
        X = sample_transformer_data
        transformer = LagTransformer(lag=[1, 2])
        transformer.fit(X[:15])

        transformer._X_observed.shape[0]

        # Observe with new data
        transformer.observe(X[15:17])

        # Memory should still be bounded by observation_horizon
        assert transformer._X_observed.shape[0] == transformer.observation_horizon

    def test_validate_transformer_rewind_memory(self, sample_transformer_data):
        """Test that transformer rewind maintains observation horizon."""
        X = sample_transformer_data
        transformer = LagTransformer(lag=[1, 2])
        transformer.fit(X[:10])

        # Rewind with new data
        transformer.rewind(X[10:15])

        # Memory should be rewound to observation_horizon rows
        assert transformer._X_observed.shape[0] == transformer.observation_horizon


class TestValidateScorerData:
    """Tests for validate_scorer_data."""

    def test_validate_scorer_basic(self, sample_scorer_data):
        """Test validate_scorer_data basic functionality."""
        y_true, y_pred = sample_scorer_data
        scorer = MeanAbsoluteError()

        # Score should call validate_scorer_data
        scorer.fit(y_true)
        score = scorer.score(y_true, y_pred)

        # Should return a scalar or DataFrame depending on aggregation
        assert isinstance(score, float | int | pl.DataFrame)

    def test_validate_scorer_missing_column(self, sample_scorer_data):
        """Test that scorer validation checks column presence."""
        y_true, y_pred = sample_scorer_data
        scorer = MeanAbsoluteError()

        # Remove target column from predictions
        y_pred_wrong = y_pred.drop("target")

        with pytest.raises(ValueError, match="missing in `y_pred`"):
            scorer.fit(y_true)
            scorer.score(y_true, y_pred_wrong)

    def test_validate_scorer_type_mismatch(self, sample_scorer_data):
        """Test that scorer validation checks column types."""
        y_true, y_pred = sample_scorer_data
        scorer = MeanAbsoluteError()

        # Change type of target column
        y_pred_wrong = y_pred.with_columns(pl.col("target").cast(pl.Utf8))

        with pytest.raises(ValueError, match="type mismatch"):
            scorer.fit(y_true)
            scorer.score(y_true, y_pred_wrong)

    def test_validate_scorer_panel_consistency(self):
        """Test that scorer validation checks panel structure consistency."""
        time = pl.datetime_range(
            start=datetime(2020, 1, 1),
            end=datetime(2020, 1, 10),
            interval="1d",
            eager=True,
        )
        y_true = pl.DataFrame({
            "time": time,
            "sales__store_1": range(10),
            "sales__store_2": range(10, 20),
        })
        # Predictions missing one store
        y_pred = pl.DataFrame({
            "vintage_time": time,
            "time": time,
            "sales__store_1": range(10),
            # Missing sales__store_2
        })

        scorer = MeanAbsoluteError()
        with pytest.raises(ValueError, match="missing in `y_pred`"):
            scorer.fit(y_true)
            scorer.score(y_true, y_pred)

    def test_validate_scorer_none_inputs(self):
        """Test that scorer validation rejects None inputs."""
        time = pl.datetime_range(
            start=datetime(2020, 1, 1),
            end=datetime(2020, 1, 10),
            interval="1d",
            eager=True,
        )
        y_valid = pl.DataFrame({"time": time, "target": range(10)})
        scorer = MeanAbsoluteError()

        with pytest.raises(ValueError, match="Cannot be None"):
            scorer.fit(None)
            scorer.score(None, y_valid)

        with pytest.raises(ValueError, match="cannot be None"):
            scorer.fit(y_valid)
            scorer.score(y_valid, None)

    def test_point_scorer_strips_interval_columns(self):
        """Point scorer validation strips _lower_/_upper_ columns from y_pred."""
        time = pl.datetime_range(
            start=datetime(2020, 1, 1),
            end=datetime(2020, 1, 10),
            interval="1d",
            eager=True,
        )
        y_true = pl.DataFrame({
            "time": time,
            "target": list(range(10)),
        })
        # y_pred with extra interval columns (as in mixed multimetric scenarios)
        y_pred = pl.DataFrame({
            "vintage_time": time,
            "time": time,
            "target": [float(x) for x in range(10)],
            "target_lower_0.9": [float(x - 1) for x in range(10)],
            "target_upper_0.9": [float(x + 1) for x in range(10)],
        })

        scorer = MeanAbsoluteError()
        scorer.fit(y_true)
        # Should not raise despite extra _lower_/_upper_ columns
        score = scorer.score(y_true, y_pred)
        assert isinstance(score, float | int)

    def test_point_scorer_no_extra_columns_unchanged(self):
        """Point scorer with matching columns does not strip anything."""
        time = pl.datetime_range(
            start=datetime(2020, 1, 1),
            end=datetime(2020, 1, 10),
            interval="1d",
            eager=True,
        )
        y_true = pl.DataFrame({
            "time": time,
            "target": list(range(10)),
        })
        y_pred = pl.DataFrame({
            "vintage_time": time,
            "time": time,
            "target": [float(x) for x in range(10)],
        })

        scorer = MeanAbsoluteError()
        scorer.fit(y_true)
        score = scorer.score(y_true, y_pred)
        assert isinstance(score, float | int)

    def test_validate_scorer_y_true_none_at_score_time_raises(self):
        """validate_scorer_data raises ValueError when y_true is None in score mode."""
        scorer = MeanAbsoluteError()
        y_pred = pl.DataFrame({
            "time": pl.datetime_range(
                start=datetime(2020, 1, 1),
                end=datetime(2020, 1, 3),
                interval="1d",
                eager=True,
            ),
            "val": [1.0, 2.0, 3.0],
        })
        with pytest.raises(ValueError, match="cannot be None"):
            validate_scorer_data(scorer, y_true=None, y_pred=y_pred, reset=False)


class TestValidateSplitterData:
    """Tests for validate_splitter_data."""

    def test_validate_splitter_fit_context(self, sample_splitter_data):
        """Test validate_splitter_data in fit context."""
        y, X = sample_splitter_data
        splitter = ExpandingWindowSplitter(n_splits=3, test_size=3)

        # Split should call validate_splitter_data
        splits = list(splitter.split(y, X))

        # Should return valid train/test splits
        assert len(splits) > 0
        for train_idx, test_idx in splits:
            assert len(train_idx) >= 10
            assert len(test_idx) == 3

    def test_validate_splitter_interval_consistency_fit(self, sample_splitter_data):
        """Test that splitter validates interval consistency."""
        # Create data with inconsistent intervals (need at least 3 rows)
        time_wrong = [
            datetime(2020, 1, 1),
            datetime(2020, 1, 2),
            datetime(2020, 1, 4),  # Skip day 3
            datetime(2020, 1, 5),
            datetime(2020, 1, 6),
        ]
        y_wrong = pl.DataFrame({"time": time_wrong, "target": [1, 2, 3, 4, 5]})

        splitter = ExpandingWindowSplitter(n_splits=2, test_size=1)

        with pytest.raises(ValueError, match="inconsistent intervals"):
            list(splitter.split(y_wrong))

    def test_validate_splitter_panel_structure(self):
        """Test that splitter handles panel data structure."""
        time = pl.datetime_range(
            start=datetime(2020, 1, 1),
            end=datetime(2020, 1, 20),
            interval="1d",
            eager=True,
        )
        y = pl.DataFrame({
            "time": time,
            "sales__store_1": range(20),
            "sales__store_2": range(20, 40),
        })

        splitter = ExpandingWindowSplitter(n_splits=2, test_size=3)

        # Should detect panel structure during split
        splits = list(splitter.split(y))
        assert len(splits) > 0

        # Splitter should handle panel data (validated internally)
        assert hasattr(splitter, "interval_")

    def test_validate_splitter_panel_mismatch(self):
        """Test that splitter validates panel consistency between y and X."""
        time = pl.datetime_range(
            start=datetime(2020, 1, 1),
            end=datetime(2020, 1, 20),
            interval="1d",
            eager=True,
        )
        y = pl.DataFrame({
            "time": time,
            "sales__store_1": range(20),
            "sales__store_2": range(20, 40),
        })
        X = pl.DataFrame({
            "time": time,
            "temp__sensor_1": range(40, 60),  # Different panel group
        })

        splitter = ExpandingWindowSplitter(n_splits=2, test_size=3)

        with pytest.raises(ValueError, match="Panel groups mismatch"):
            list(splitter.split(y, X))

    def test_validate_splitter_y_none(self):
        """Test that splitter validation rejects None y."""
        splitter = ExpandingWindowSplitter(n_splits=2, test_size=3)

        with pytest.raises(
            (ValueError, AttributeError, TypeError),
            match="(cannot be None|'NoneType' object has no attribute|has no len)",
        ):
            list(splitter.split(None))

    def test_validate_splitter_sets_interval(self, sample_splitter_data):
        """Test that splitter validation sets interval attribute."""
        y, X = sample_splitter_data
        splitter = ExpandingWindowSplitter(n_splits=3, test_size=5)

        # Before split, interval_ should not exist
        assert not hasattr(splitter, "interval_")

        # After split, interval_ should be set
        list(splitter.split(y, X))
        assert hasattr(splitter, "interval_")
        assert splitter.interval_ is not None

    def test_validate_splitter_panel_internal_consistency(self):
        """Test that splitter validates internal panel structure consistency."""
        time = pl.datetime_range(
            start=datetime(2020, 1, 1),
            end=datetime(2020, 1, 20),
            interval="1d",
            eager=True,
        )
        # Create panel data with mismatched local columns
        y = pl.DataFrame({
            "time": time,
            "sales__store_1": range(20),
            "sales__store_2": range(20, 40),
            "revenue__store_1": range(40, 60),
            # Missing revenue__store_2 - internal inconsistency
        })

        splitter = ExpandingWindowSplitter(n_splits=2, test_size=3)

        with pytest.raises(ValueError, match="Panel structure mismatch"):
            list(splitter.split(y))


class TestValidateTimeWeight:
    """Tests for validate_time_weight validation function."""

    @pytest.fixture
    def y_global(self):
        """Return a simple global time series DataFrame."""
        return pl.DataFrame({
            "time": [datetime(2024, 1, i) for i in range(1, 11)],
            "y": list(range(10)),
        })

    def test_none_is_valid(self, y_global):
        """None time_weight should pass without error."""
        from yohou.utils.validate_data import validate_time_weight

        validate_time_weight(None, y_global)

    def test_callable_is_valid(self, y_global):
        """A callable time_weight should pass without error."""
        from yohou.utils.validate_data import validate_time_weight

        validate_time_weight(lambda t: t, y_global)

    def test_invalid_type_raises(self, y_global):
        """Non-callable, non-DataFrame, non-None should raise."""
        from yohou.utils.validate_data import validate_time_weight

        with pytest.raises(ValueError, match="must be callable"):
            validate_time_weight("invalid", y_global)

    def test_dataframe_missing_time_column(self):
        """DataFrame without 'time' column should raise."""
        from yohou.utils.validate_data import validate_time_weight

        y = pl.DataFrame({"time": [datetime(2024, 1, 1)], "y": [1.0]})
        tw = pl.DataFrame({"weight": [1.0]})

        with pytest.raises(ValueError, match="must have 'time' column"):
            validate_time_weight(tw, y)

    def test_dataframe_no_weight_columns(self):
        """DataFrame with only 'time' column should raise."""
        from yohou.utils.validate_data import validate_time_weight

        y = pl.DataFrame({"time": [datetime(2024, 1, 1)], "y": [1.0]})
        tw = pl.DataFrame({"time": [datetime(2024, 1, 1)]})

        with pytest.raises(ValueError, match="at least one weight column"):
            validate_time_weight(tw, y)

    def test_global_missing_weight_column(self):
        """Global data without 'weight' column should raise."""
        from yohou.utils.validate_data import validate_time_weight

        y = pl.DataFrame({"time": [datetime(2024, 1, 1)], "y": [1.0]})
        tw = pl.DataFrame({
            "time": [datetime(2024, 1, 1)],
            "wrong_name": [1.0],
        })

        with pytest.raises(ValueError, match="must have 'weight' column"):
            validate_time_weight(tw, y)

    def test_global_valid_dataframe(self, y_global):
        """Valid global DataFrame time_weight should pass."""
        from yohou.utils.validate_data import validate_time_weight

        tw = pl.DataFrame({
            "time": [datetime(2024, 1, i) for i in range(1, 11)],
            "weight": [float(i) for i in range(1, 11)],
        })
        validate_time_weight(tw, y_global)

    def test_nan_in_weight_raises(self, y_global):
        """NaN values in weight column should raise."""
        from yohou.utils.validate_data import validate_time_weight

        tw = pl.DataFrame({
            "time": [datetime(2024, 1, i) for i in range(1, 4)],
            "weight": [1.0, None, 1.0],
        })

        with pytest.raises(ValueError, match="contains NaN"):
            validate_time_weight(tw, y_global)

    def test_negative_weight_raises(self, y_global):
        """Negative values in weight column should raise."""
        from yohou.utils.validate_data import validate_time_weight

        tw = pl.DataFrame({
            "time": [datetime(2024, 1, i) for i in range(1, 4)],
            "weight": [1.0, -1.0, 1.0],
        })

        with pytest.raises(ValueError, match="contains negative"):
            validate_time_weight(tw, y_global)

    def test_infinite_weight_raises(self, y_global):
        """Infinite values in weight column should raise."""
        from yohou.utils.validate_data import validate_time_weight

        tw = pl.DataFrame({
            "time": [datetime(2024, 1, i) for i in range(1, 4)],
            "weight": [1.0, float("inf"), 1.0],
        })

        with pytest.raises(ValueError, match="contains infinite"):
            validate_time_weight(tw, y_global)

    def test_zero_sum_weight_raises(self, y_global):
        """All-zero weights should raise."""
        from yohou.utils.validate_data import validate_time_weight

        tw = pl.DataFrame({
            "time": [datetime(2024, 1, i) for i in range(1, 4)],
            "weight": [0.0, 0.0, 0.0],
        })

        with pytest.raises(ValueError, match="sums to zero"):
            validate_time_weight(tw, y_global)

    def test_panel_with_group_columns(self, y_global):
        """Panel data with group-specific weight columns should pass."""
        from yohou.utils.validate_data import validate_time_weight

        tw = pl.DataFrame({
            "time": [datetime(2024, 1, i) for i in range(1, 4)],
            "store_1_weight": [1.0, 2.0, 3.0],
            "store_2_weight": [3.0, 2.0, 1.0],
        })
        validate_time_weight(tw, y_global, groups=["store_1", "store_2"])

    def test_panel_with_global_weight(self, y_global):
        """Panel data with global 'weight' column should pass."""
        from yohou.utils.validate_data import validate_time_weight

        tw = pl.DataFrame({
            "time": [datetime(2024, 1, i) for i in range(1, 4)],
            "weight": [1.0, 2.0, 3.0],
        })
        validate_time_weight(tw, y_global, groups=["store_1"])

    def test_panel_missing_both_group_and_global(self, y_global):
        """Panel data without group or global weight columns should raise."""
        from yohou.utils.validate_data import validate_time_weight

        tw = pl.DataFrame({
            "time": [datetime(2024, 1, i) for i in range(1, 4)],
            "something_else": [1.0, 2.0, 3.0],
        })

        with pytest.raises(ValueError, match="must have either"):
            validate_time_weight(tw, y_global, groups=["store_1"])


class TestValidateScorerDataInverse:
    """Tests for validate_scorer_data with inverse=True."""

    def test_inverse_y_pred_none_raises(self):
        """inverse=True with y_pred=None raises ValueError."""
        from yohou.utils.validate_data import validate_scorer_data

        scorer = MeanAbsoluteError()
        scorer.fit(
            pl.DataFrame({
                "time": [datetime(2024, 1, i) for i in range(1, 4)],
                "value": [1.0, 2.0, 3.0],
            })
        )

        with pytest.raises(ValueError):
            validate_scorer_data(scorer, y_pred=None, inverse=True)

    def test_inverse_scores_none_raises(self):
        """inverse=True with scores=None raises ValueError."""
        from yohou.utils.validate_data import validate_scorer_data

        times = [datetime(2024, 1, i) for i in range(1, 4)]
        scorer = MeanAbsoluteError()
        scorer.fit(pl.DataFrame({"time": times, "value": [1.0, 2.0, 3.0]}))

        y_pred = pl.DataFrame({"time": times, "value": [1.1, 2.1, 3.1]})
        with pytest.raises(ValueError):
            validate_scorer_data(scorer, y_pred=y_pred, scores=None, inverse=True)

    def test_inverse_valid_data_passes(self):
        """inverse=True with matching y_pred and scores passes."""
        from yohou.utils.validate_data import validate_scorer_data

        times = [datetime(2024, 1, i) for i in range(1, 4)]
        scorer = MeanAbsoluteError()
        scorer.fit(pl.DataFrame({"time": times, "value": [1.0, 2.0, 3.0]}))

        y_pred = pl.DataFrame({"time": times, "value": [1.1, 2.1, 3.1]})
        scores = pl.DataFrame({"time": times, "value": [0.1, 0.1, 0.1]})
        result = validate_scorer_data(scorer, y_pred=y_pred, scores=scores, inverse=True)
        assert result is not None


class TestValidateTransformerDataInverse:
    """Tests for validate_transformer_data with inverse=True."""

    def test_inverse_no_X_t_no_X_raises(self):
        """inverse=True without X_t or X raises ValueError."""
        from yohou.utils.validate_data import validate_transformer_data

        t = LagTransformer(lag=1)
        times = [datetime(2024, 1, i) for i in range(1, 11)]
        X = pl.DataFrame({"time": times, "val": range(10)})
        t.fit(X)

        with pytest.raises(ValueError):
            validate_transformer_data(t, X=None, reset=False, inverse=True, X_t=None)

    def test_inverse_X_t_used_as_primary(self):
        """inverse=True with X_t uses it as the primary data."""
        from yohou.utils.validate_data import validate_transformer_data

        t = LagTransformer(lag=1)
        times = [datetime(2024, 1, i) for i in range(1, 11)]
        X = pl.DataFrame({"time": times, "val": list(range(10))})
        t.fit(X)

        X_t = pl.DataFrame({
            "time": times[1:],
            "val": list(range(1, 10)),
        })
        X_p = X.head(1)
        result = validate_transformer_data(
            t,
            X=X_t,
            reset=False,
            inverse=True,
            X_t=X_t,
            X_p=X_p,
            observation_horizon=1,
        )
        assert result is not None


class TestValidateScorerDataEdgeCases:
    """Edge case tests for validate_scorer_data."""

    def test_y_true_none_raises(self):
        """y_true=None raises ValueError."""
        from yohou.utils.validate_data import validate_scorer_data

        times = [datetime(2024, 1, i) for i in range(1, 4)]
        scorer = MeanAbsoluteError()
        scorer.fit(pl.DataFrame({"time": times, "value": [1.0, 2.0, 3.0]}))

        y_pred = pl.DataFrame({"time": times, "value": [1.1, 2.1, 3.1]})
        with pytest.raises(ValueError):
            validate_scorer_data(scorer, y_true=None, y_pred=y_pred)

    def test_y_pred_none_raises(self):
        """y_pred=None raises ValueError."""
        from yohou.utils.validate_data import validate_scorer_data

        times = [datetime(2024, 1, i) for i in range(1, 4)]
        scorer = MeanAbsoluteError()
        y_true = pl.DataFrame({"time": times, "value": [1.0, 2.0, 3.0]})
        scorer.fit(y_true)

        with pytest.raises(ValueError):
            validate_scorer_data(scorer, y_true=y_true, y_pred=None)


class TestValidateScorerDataPanelEdgeCases:
    """Tests for scorer data validation edge cases."""

    def test_panel_group_mismatch_raises(self):
        """Mismatched panel groups between y_true and y_pred raises ValueError."""
        from yohou.utils.validate_data import validate_scorer_data

        times = [datetime(2024, 1, i) for i in range(1, 4)]
        scorer = MeanAbsoluteError()
        y_true = pl.DataFrame({
            "time": times,
            "g1__val": [1.0, 2.0, 3.0],
            "g2__val": [4.0, 5.0, 6.0],
        })
        y_pred = pl.DataFrame({
            "time": times,
            "g1__val": [1.1, 2.1, 3.1],
            "g3__val": [4.1, 5.1, 6.1],
        })
        scorer.fit(y_true)
        with pytest.raises(ValueError):
            validate_scorer_data(scorer, y_true=y_true, y_pred=y_pred)

    def test_interval_missing_lower_upper_raises(self):
        """Interval scorer missing lower/upper columns raises ValueError."""
        from yohou.metrics import EmpiricalCoverage
        from yohou.utils.validate_data import validate_scorer_data

        times = [datetime(2024, 1, i) for i in range(1, 4)]
        y_true = pl.DataFrame({"time": times, "val": [1.0, 2.0, 3.0]})
        y_pred = pl.DataFrame({
            "vintage_time": [datetime(2023, 12, 31)] * 3,
            "time": times,
            "val": [1.1, 2.1, 3.1],
        })
        scorer = EmpiricalCoverage()
        scorer.fit(y_true)
        with pytest.raises(ValueError):
            validate_scorer_data(scorer, y_true=y_true, y_pred=y_pred)


class TestValidateTransformerDataEdgeCases:
    """Tests for transformer data validation edge cases."""

    def test_fit_with_none_raises(self):
        """Fitting transformer with X=None raises ValueError."""
        from yohou.utils.validate_data import validate_transformer_data

        transformer = LagTransformer(lag=[1])
        with pytest.raises(ValueError):
            validate_transformer_data(transformer, X=None, reset=True)

    def test_transform_with_none_raises(self):
        """Transforming with X=None raises ValueError."""
        from yohou.utils.validate_data import validate_transformer_data

        times = [datetime(2024, 1, i) for i in range(1, 11)]
        X = pl.DataFrame({"time": times, "val": range(10)})
        transformer = LagTransformer(lag=[1])
        transformer.fit(X)

        with pytest.raises(ValueError):
            validate_transformer_data(transformer, X=None, reset=False)


class TestValidateTimeWeightEdgeCases:
    """Tests for validate_time_weight edge cases."""

    def test_invalid_type_raises(self):
        """Non-callable non-DataFrame time_weight raises ValueError."""
        from yohou.utils.validate_data import validate_time_weight

        y = pl.DataFrame({
            "time": [datetime(2024, 1, 1), datetime(2024, 1, 2)],
            "val": [1.0, 2.0],
        })
        with pytest.raises(ValueError, match="must be callable"):
            validate_time_weight(42, y)

    def test_null_weights_raises(self):
        """Weight column with null values raises ValueError."""
        from yohou.utils.validate_data import validate_time_weight

        y = pl.DataFrame({
            "time": [datetime(2024, 1, 1), datetime(2024, 1, 2), datetime(2024, 1, 3)],
            "val": [1.0, 2.0, 3.0],
        })
        tw = pl.DataFrame({
            "time": [datetime(2024, 1, 1), datetime(2024, 1, 2), datetime(2024, 1, 3)],
            "weight": [1.0, None, 1.0],
        })
        with pytest.raises(ValueError, match="contains NaN"):
            validate_time_weight(tw, y)


class TestValidateScorerDataResetNone:
    """Tests for validate_scorer_data reset=True with y_true=None."""

    def test_reset_y_true_none_raises(self):
        """reset=True with y_true=None raises ValueError about y_train."""
        from yohou.utils.validate_data import validate_scorer_data

        scorer = MeanAbsoluteError()
        with pytest.raises(ValueError, match="y_train"):
            validate_scorer_data(scorer, y_true=None, reset=True)


class TestValidateScorerDataInverseObservedTime:
    """Tests for inverse scorer path with vintage_time in y_pred."""

    def test_inverse_vintage_time_dropped(self):
        """vintage_time column in y_pred is dropped during inverse scoring."""
        from yohou.utils.validate_data import validate_scorer_data

        times = [datetime(2024, 1, i) for i in range(1, 4)]
        scorer = MeanAbsoluteError()
        scorer.fit(pl.DataFrame({"time": times, "value": [1.0, 2.0, 3.0]}))

        y_pred = pl.DataFrame({
            "vintage_time": [datetime(2023, 12, 31)] * 3,
            "time": times,
            "value": [1.1, 2.1, 3.1],
        })
        scores = pl.DataFrame({"time": times, "value": [0.1, 0.1, 0.1]})
        result_pred, result_scores, context = validate_scorer_data(
            scorer,
            y_pred=y_pred,
            scores=scores,
            inverse=True,
        )
        assert "vintage_time" not in result_pred.columns
        assert "time" not in result_pred.columns
        assert len(context.time_values) == 3

    def test_inverse_column_mismatch_raises(self):
        """Mismatched columns between y_pred and scores raises ValueError."""
        from yohou.utils.validate_data import validate_scorer_data

        times = [datetime(2024, 1, i) for i in range(1, 4)]
        scorer = MeanAbsoluteError()
        scorer.fit(pl.DataFrame({"time": times, "value": [1.0, 2.0, 3.0]}))

        y_pred = pl.DataFrame({"time": times, "value": [1.1, 2.1, 3.1]})
        scores = pl.DataFrame({"time": times, "other_col": [0.1, 0.1, 0.1]})
        with pytest.raises(ValueError, match="Column mismatch"):
            validate_scorer_data(scorer, y_pred=y_pred, scores=scores, inverse=True)


class TestValidateScorerDataPointStripExtra:
    """Tests for point scorer stripping extra interval columns."""

    def test_extra_interval_columns_stripped(self):
        """Point scorer strips _lower_/_upper_ columns from y_pred."""
        from yohou.utils.validate_data import validate_scorer_data

        times = [datetime(2024, 1, i) for i in range(1, 4)]
        scorer = MeanAbsoluteError()
        y_true = pl.DataFrame({"time": times, "val": [1.0, 2.0, 3.0]})
        scorer.fit(y_true)

        y_pred = pl.DataFrame({
            "vintage_time": [datetime(2023, 12, 31)] * 3,
            "time": times,
            "val": [1.1, 2.1, 3.1],
            "val_lower_0.9": [0.1, 0.1, 0.1],
            "val_upper_0.9": [2.1, 3.1, 4.1],
        })
        _, result_pred, _ = validate_scorer_data(scorer, y_true=y_true, y_pred=y_pred)
        assert "val_lower_0.9" not in result_pred.columns
        assert "val_upper_0.9" not in result_pred.columns
        assert "val" in result_pred.columns


class TestValidateSplitterDataBothNonNone:
    """Tests for validate_splitter_data with both y and X provided."""

    def test_both_y_and_x_validated(self):
        """Both y and X_actual are validated and panel groups checked."""
        from yohou.utils.validate_data import validate_splitter_data

        times = [datetime(2024, 1, i) for i in range(1, 11)]
        y = pl.DataFrame({"time": times, "target": range(10)})
        X_actual = pl.DataFrame({"time": times, "feature": range(10, 20)})
        splitter = ExpandingWindowSplitter(n_splits=3, test_size=2)
        result_y, result_X_actual = validate_splitter_data(splitter, y, X_actual)
        assert result_y is not None
        assert result_X_actual is not None


class TestValidateTransformerInversePaths:
    """Tests for inverse transform validation paths."""

    def test_inverse_stateful_no_xp_raises(self):
        """Stateful inverse transform without X_p raises ValueError."""
        from yohou.utils.validate_data import validate_transformer_data

        t = LagTransformer(lag=3)
        times = [datetime(2024, 1, i) for i in range(1, 11)]
        X = pl.DataFrame({"time": times, "val": list(range(10))})
        t.fit(X)

        X_t = pl.DataFrame({"time": times[3:], "val": list(range(3, 10))})
        with pytest.raises(ValueError, match="X_p cannot be None"):
            validate_transformer_data(
                t,
                X=X_t,
                reset=False,
                inverse=True,
                X_t=X_t,
                X_p=None,
                stateful=True,
            )

    def test_inverse_observation_horizon_no_xp_raises(self):
        """Inverse transform with observation_horizon > 0 and no X_p raises ValueError."""
        from yohou.utils.validate_data import validate_transformer_data

        t = LagTransformer(lag=3)
        times = [datetime(2024, 1, i) for i in range(1, 11)]
        X = pl.DataFrame({"time": times, "val": list(range(10))})
        t.fit(X)

        X_t = pl.DataFrame({"time": times[3:], "val": list(range(3, 10))})
        with pytest.raises(ValueError, match="X_p cannot be None"):
            validate_transformer_data(
                t,
                X=X_t,
                reset=False,
                inverse=True,
                X_t=X_t,
                X_p=None,
                observation_horizon=3,
            )

    def test_inverse_with_xp_and_xt_passes(self):
        """Inverse transform with valid X_t and X_p passes."""
        from yohou.utils.validate_data import validate_transformer_data

        t = LagTransformer(lag=3)
        times = [datetime(2024, 1, i) for i in range(1, 21)]
        X = pl.DataFrame({"time": times, "val": list(range(20))})
        t.fit(X)

        X_t = pl.DataFrame({"time": times[3:], "val": list(range(3, 20))})
        X_p = pl.DataFrame({"time": times[:5], "val": list(range(5))})
        result = validate_transformer_data(
            t,
            X=X_t,
            reset=False,
            inverse=True,
            X_t=X_t,
            X_p=X_p,
            observation_horizon=3,
        )
        assert result is not None


class TestValidateScorerDataClassProba:
    """Tests for validate_score_data with class_proba prediction type."""

    def test_class_proba_missing_proba_columns_raises(self):
        """Missing probability columns for a target raises ValueError."""
        from yohou.metrics.class_proba import LogLoss

        time = pl.datetime_range(
            start=datetime(2020, 1, 1),
            end=datetime(2020, 1, 5),
            interval="1d",
            eager=True,
        )
        y_true = pl.DataFrame({"time": time, "weather": ["sun", "rain", "sun", "rain", "sun"]})
        # y_pred has no weather_proba_* columns
        y_pred = pl.DataFrame({"time": time, "other_proba_a": [0.5] * 5, "other_proba_b": [0.5] * 5})

        scorer = LogLoss()
        scorer.fit(y_true)
        with pytest.raises(ValueError, match="No probability columns found for target"):
            scorer.score(y_true, y_pred)

    def test_class_proba_non_numeric_proba_columns_raises(self):
        """Non-numeric probability columns raise ValueError."""
        from yohou.metrics.class_proba import LogLoss

        time = pl.datetime_range(
            start=datetime(2020, 1, 1),
            end=datetime(2020, 1, 5),
            interval="1d",
            eager=True,
        )
        y_true = pl.DataFrame({"time": time, "weather": ["sun", "rain", "sun", "rain", "sun"]})
        y_pred = pl.DataFrame({
            "time": time,
            "weather_proba_sun": ["high", "low", "high", "low", "high"],
            "weather_proba_rain": ["low", "high", "low", "high", "low"],
        })

        scorer = LogLoss()
        scorer.fit(y_true)
        with pytest.raises(ValueError, match="must be numeric"):
            scorer.score(y_true, y_pred)


class TestValidatePlottingDataCategorical:
    """Tests for validate_plotting_data with include_categorical parameter."""

    def test_default_excludes_categorical(self):
        """By default, categorical columns are excluded."""
        from yohou.utils.validate_data import validate_plotting_data

        time = pl.datetime_range(
            start=datetime(2020, 1, 1),
            end=datetime(2020, 1, 3),
            interval="1d",
            eager=True,
        )
        df = pl.DataFrame({"time": time, "y": [1.0, 2.0, 3.0], "cat": ["a", "b", "c"]})
        result = validate_plotting_data(df, exclude=["time"])
        assert result == ["y"]

    def test_include_categorical_true(self):
        """include_categorical=True returns categorical columns alongside numeric."""
        from yohou.utils.validate_data import validate_plotting_data

        time = pl.datetime_range(
            start=datetime(2020, 1, 1),
            end=datetime(2020, 1, 3),
            interval="1d",
            eager=True,
        )
        df = pl.DataFrame({"time": time, "y": [1.0, 2.0, 3.0], "cat": ["a", "b", "c"]})
        result = validate_plotting_data(df, exclude=["time"], include_categorical=True)
        assert "y" in result
        assert "cat" in result

    def test_include_categorical_no_duplicates(self):
        """Numeric columns are not duplicated when include_categorical=True."""
        from yohou.utils.validate_data import validate_plotting_data

        time = pl.datetime_range(
            start=datetime(2020, 1, 1),
            end=datetime(2020, 1, 3),
            interval="1d",
            eager=True,
        )
        df = pl.DataFrame({"time": time, "y": [1.0, 2.0, 3.0]})
        result = validate_plotting_data(df, exclude=["time"], include_categorical=True)
        assert result == ["y"]

    def test_include_categorical_only_categorical(self):
        """Works when DataFrame has only categorical columns (besides time)."""
        from yohou.utils.validate_data import validate_plotting_data

        time = pl.datetime_range(
            start=datetime(2020, 1, 1),
            end=datetime(2020, 1, 3),
            interval="1d",
            eager=True,
        )
        df = pl.DataFrame({"time": time, "cat1": ["a", "b", "c"], "cat2": ["x", "y", "z"]})
        result = validate_plotting_data(df, exclude=["time"], include_categorical=True)
        assert result == ["cat1", "cat2"]

    def test_include_categorical_respects_exclude(self):
        """Excluded columns are excluded even when include_categorical=True."""
        from yohou.utils.validate_data import validate_plotting_data

        time = pl.datetime_range(
            start=datetime(2020, 1, 1),
            end=datetime(2020, 1, 3),
            interval="1d",
            eager=True,
        )
        df = pl.DataFrame({"time": time, "y": [1.0, 2.0, 3.0], "cat": ["a", "b", "c"]})
        result = validate_plotting_data(df, exclude=["time", "cat"], include_categorical=True)
        assert result == ["y"]

    def test_include_categorical_with_explicit_columns(self):
        """When columns are explicitly specified, include_categorical is irrelevant."""
        from yohou.utils.validate_data import validate_plotting_data

        time = pl.datetime_range(
            start=datetime(2020, 1, 1),
            end=datetime(2020, 1, 3),
            interval="1d",
            eager=True,
        )
        df = pl.DataFrame({"time": time, "y": [1.0, 2.0, 3.0], "cat": ["a", "b", "c"]})
        result = validate_plotting_data(df, columns=["cat"])
        assert result == ["cat"]


class TestComputeForecastingStep:
    """Tests for _compute_forecasting_step helper."""

    def test_fixed_daily_interval(self):
        """Daily interval computes correct integer steps."""
        from yohou.utils.validate_data import _compute_forecasting_step

        observed = pl.Series([datetime(2020, 1, 1)] * 3)
        time = pl.Series([
            datetime(2020, 1, 2),
            datetime(2020, 1, 3),
            datetime(2020, 1, 4),
        ])
        result = _compute_forecasting_step(time, observed, "1d")
        assert result.to_list() == [1, 2, 3]

    def test_fixed_hourly_interval(self):
        """Hourly interval computes correct integer steps."""
        from yohou.utils.validate_data import _compute_forecasting_step

        observed = pl.Series([datetime(2020, 1, 1)] * 3)
        time = pl.Series([
            datetime(2020, 1, 1, 1),
            datetime(2020, 1, 1, 2),
            datetime(2020, 1, 1, 3),
        ])
        result = _compute_forecasting_step(time, observed, "1h")
        assert result.to_list() == [1, 2, 3]

    def test_monthly_interval(self):
        """Monthly interval uses calendar-unit arithmetic."""
        from yohou.utils.validate_data import _compute_forecasting_step

        observed = pl.Series([datetime(2020, 1, 1)] * 3)
        time = pl.Series([
            datetime(2020, 2, 1),
            datetime(2020, 3, 1),
            datetime(2020, 4, 1),
        ])
        result = _compute_forecasting_step(time, observed, "1mo")
        assert result.to_list() == [1, 2, 3]

    def test_quarterly_interval(self):
        """Quarterly (3mo) interval uses calendar-unit arithmetic."""
        from yohou.utils.validate_data import _compute_forecasting_step

        observed = pl.Series([datetime(2020, 1, 1)] * 2)
        time = pl.Series([
            datetime(2020, 4, 1),
            datetime(2020, 7, 1),
        ])
        result = _compute_forecasting_step(time, observed, "1q")
        assert result.to_list() == [1, 2]

    def test_yearly_interval(self):
        """Yearly interval uses calendar-unit arithmetic."""
        from yohou.utils.validate_data import _compute_forecasting_step

        observed = pl.Series([datetime(2020, 1, 1)] * 2)
        time = pl.Series([
            datetime(2021, 1, 1),
            datetime(2022, 1, 1),
        ])
        result = _compute_forecasting_step(time, observed, "1y")
        assert result.to_list() == [1, 2]

    def test_weekly_interval(self):
        """Weekly (fixed) interval computes correct steps."""
        from yohou.utils.validate_data import _compute_forecasting_step

        observed = pl.Series([datetime(2020, 1, 1)] * 2)
        time = pl.Series([
            datetime(2020, 1, 8),
            datetime(2020, 1, 15),
        ])
        result = _compute_forecasting_step(time, observed, "1w")
        assert result.to_list() == [1, 2]


class TestResolveColumnsGroupFilter:
    """Tests for validate_plotting_data with groups + columns filter."""

    def test_groups_with_columns_filter(self):
        """Groups + columns filter returns matching panel columns."""
        from yohou.utils.validate_data import validate_plotting_data

        df = pl.DataFrame({
            "time": [1, 2, 3],
            "sales__east": [10.0, 20.0, 30.0],
            "sales__west": [15.0, 25.0, 35.0],
            "returns__east": [1.0, 2.0, 3.0],
        })
        result = validate_plotting_data(df, groups=["sales"], columns=["east"])
        assert result == ["sales__east"]

    def test_groups_with_columns_no_match_raises(self):
        """Groups + columns with no match raises ValueError."""
        from yohou.utils.validate_data import validate_plotting_data

        df = pl.DataFrame({
            "time": [1, 2, 3],
            "sales__east": [10.0, 20.0, 30.0],
        })
        with pytest.raises(ValueError, match="No panel columns found"):
            validate_plotting_data(df, groups=["sales"], columns=["nonexistent"])


class TestTruncatePartialVintage:
    """Tests for _truncate_partial_vintage."""

    def test_partial_vintage_truncated(self):
        """Last vintage with shorter gap is removed."""
        from yohou.utils.validate_data import _truncate_partial_vintage

        # y_true must have the same rows as y_pred (aligned/expanded)
        rows_true = []
        rows_pred = []
        for ot in [datetime(2020, 1, 1), datetime(2020, 1, 8), datetime(2020, 1, 15), datetime(2020, 1, 19)]:
            for i in range(1, 4):
                rows_true.append({"time": datetime(2020, 1, i), "value": float(i)})
                rows_pred.append({
                    "vintage_time": ot,
                    "time": datetime(2020, 1, i),
                    "value": float(i) + 0.5,
                })
        y_true = pl.DataFrame(rows_true)
        y_pred = pl.DataFrame(rows_pred)
        y_true_out, y_pred_out = _truncate_partial_vintage(y_true, y_pred)
        # Last vintage (Jan 19) should be removed
        remaining = y_pred_out["vintage_time"].unique().sort()
        assert len(remaining) == 3
        assert datetime(2020, 1, 19) not in remaining.to_list()
        assert len(y_true_out) == 9  # 3 vintages × 3 time points


class TestValidateXFutureStructure:
    """Tests for _validate_X_future_structure."""

    def test_missing_time_column_raises(self):
        """X_future without 'time' column raises ValueError."""
        from yohou.utils.validate_data import _validate_X_future_structure

        df = pl.DataFrame({"value": [1.0, 2.0]})
        with pytest.raises(ValueError, match="time"):
            _validate_X_future_structure(df)

    def test_wrong_time_dtype_raises(self):
        """X_future with non-date 'time' column raises ValueError."""
        from yohou.utils.validate_data import _validate_X_future_structure

        df = pl.DataFrame({"time": [1, 2], "value": [1.0, 2.0]})
        with pytest.raises(ValueError, match="dtype"):
            _validate_X_future_structure(df)

    def test_valid_passes(self):
        """X_future with valid structure passes."""
        from yohou.utils.validate_data import _validate_X_future_structure

        time = pl.datetime_range(datetime(2020, 1, 1), datetime(2020, 1, 5), interval="1d", eager=True)
        df = pl.DataFrame({"time": time, "value": [1.0] * len(time)})
        _validate_X_future_structure(df)


class TestValidateXForecastStructure:
    """Tests for _validate_X_forecast_structure."""

    def test_missing_vintage_time_raises(self):
        """X_forecast without 'vintage_time' column raises ValueError."""
        from yohou.utils.validate_data import _validate_X_forecast_structure

        time = pl.datetime_range(datetime(2020, 1, 1), datetime(2020, 1, 5), interval="1d", eager=True)
        df = pl.DataFrame({"time": time, "value": [1.0] * len(time)})
        with pytest.raises(ValueError, match="vintage_time"):
            _validate_X_forecast_structure(df)

    def test_missing_time_raises(self):
        """X_forecast without 'time' column raises ValueError."""
        from yohou.utils.validate_data import _validate_X_forecast_structure

        time = pl.datetime_range(datetime(2020, 1, 1), datetime(2020, 1, 5), interval="1d", eager=True)
        df = pl.DataFrame({"vintage_time": time, "value": [1.0] * len(time)})
        with pytest.raises(ValueError, match="time"):
            _validate_X_forecast_structure(df)

    def test_wrong_vintage_time_dtype_raises(self):
        """X_forecast with non-date 'vintage_time' column raises ValueError."""
        from yohou.utils.validate_data import _validate_X_forecast_structure

        time = pl.datetime_range(datetime(2020, 1, 1), datetime(2020, 1, 5), interval="1d", eager=True)
        df = pl.DataFrame({"vintage_time": [1, 2, 3, 4, 5], "time": time, "value": [1.0] * len(time)})
        with pytest.raises(ValueError, match="dtype"):
            _validate_X_forecast_structure(df)


class TestValidateStepSourceSchema:
    """Tests for _validate_step_source_schema."""

    def test_schema_mismatch_raises(self):
        """Mismatched schema raises ValueError."""
        from yohou.utils.validate_data import _validate_step_source_schema

        time = pl.datetime_range(datetime(2020, 1, 1), datetime(2020, 1, 5), interval="1d", eager=True)
        df = pl.DataFrame({"time": time, "value": [1.0] * len(time)})
        expected = {"wrong_col": pl.Float64}
        with pytest.raises(ValueError, match="schema mismatch"):
            _validate_step_source_schema(df, expected, "X_future", exclude_cols={"time"})

    def test_matching_schema_passes(self):
        """Matching schema passes validation."""
        from yohou.utils.validate_data import _validate_step_source_schema

        time = pl.datetime_range(datetime(2020, 1, 1), datetime(2020, 1, 5), interval="1d", eager=True)
        df = pl.DataFrame({"time": time, "value": [1.0] * len(time)})
        expected = {"value": pl.Float64}
        _validate_step_source_schema(df, expected, "X_future", exclude_cols={"time"})
