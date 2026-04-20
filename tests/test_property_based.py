"""Property-based tests using Hypothesis strategies.

Verifies structural invariants that must hold for *any* valid input,
complementing the deterministic tests elsewhere in the suite.
"""

from __future__ import annotations

import pytest
from hypothesis import given, settings

from strategies import st_forecasting_horizon, st_time_series, st_y_X


class TestTransformerInvariants:
    """Property-based checks for transformer contracts."""

    @given(df=st_time_series(min_length=10, max_length=100))
    @settings(max_examples=30, deadline=5000)
    def test_stateless_transform_preserves_time_column(self, df):
        """Stateless transformers always preserve the ``time`` column."""
        from yohou.preprocessing import FunctionTransformer

        t = FunctionTransformer(func=lambda x: x * 2, inverse_func=lambda x: x / 2)
        result = t.fit(df).transform(df)

        assert "time" in result.columns
        assert result["time"].equals(df["time"])

    @given(df=st_time_series(min_length=10, max_length=100))
    @settings(max_examples=30, deadline=5000)
    def test_stateless_transform_preserves_row_count(self, df):
        """Stateless transformers never change the number of rows."""
        from yohou.preprocessing import FunctionTransformer

        t = FunctionTransformer(func=lambda x: x + 1, inverse_func=lambda x: x - 1)
        result = t.fit(df).transform(df)

        assert len(result) == len(df)

    @given(df=st_time_series(min_length=10, max_length=100))
    @settings(max_examples=30, deadline=5000)
    def test_identity_transform_roundtrip(self, df):
        """Identity FunctionTransformer is a no-op roundtrip."""
        from yohou.preprocessing import FunctionTransformer

        t = FunctionTransformer()
        out = t.fit(df).transform(df)

        assert out.equals(df)

    @given(df=st_time_series(min_length=10, max_length=100, min_columns=1))
    @settings(max_examples=30, deadline=5000)
    def test_standard_scaler_column_count_invariant(self, df):
        """StandardScaler preserves column count."""
        from yohou.preprocessing import StandardScaler

        scaler = StandardScaler()
        result = scaler.fit(df).transform(df)

        assert result.columns == df.columns

    @given(df=st_time_series(min_length=10, max_length=100, min_columns=1))
    @settings(max_examples=30, deadline=5000)
    def test_minmax_scaler_column_count_invariant(self, df):
        """MinMaxScaler preserves column count."""
        from yohou.preprocessing import MinMaxScaler

        scaler = MinMaxScaler()
        result = scaler.fit(df).transform(df)

        assert result.columns == df.columns


class TestForecasterInvariants:
    """Property-based checks for forecaster contracts."""

    @given(
        data=st_y_X(min_length=20, max_length=100, max_targets=1, min_features=0, max_features=0),
        horizon=st_forecasting_horizon(min_value=1, max_value=10),
    )
    @settings(max_examples=20, deadline=10000)
    def test_naive_predict_length_matches_horizon(self, data, horizon):
        """NaiveForecaster predictions always have exactly ``horizon`` rows."""
        from yohou.point import SeasonalNaive

        y, _ = data
        forecaster = SeasonalNaive(seasonality=1)
        forecaster.fit(y, forecasting_horizon=horizon)
        pred = forecaster.predict()

        # Predictions have vintage_time and time columns
        assert len(pred) == horizon

    @given(
        data=st_y_X(min_length=20, max_length=100, max_targets=1, min_features=0, max_features=0),
        horizon=st_forecasting_horizon(min_value=1, max_value=10),
    )
    @settings(max_examples=20, deadline=10000)
    def test_predict_has_time_and_vintage_time(self, data, horizon):
        """All forecasters produce predictions with both time columns."""
        from yohou.point import SeasonalNaive

        y, _ = data
        forecaster = SeasonalNaive(seasonality=1)
        forecaster.fit(y, forecasting_horizon=horizon)
        pred = forecaster.predict()

        assert "time" in pred.columns
        assert "vintage_time" in pred.columns

    @given(
        data=st_y_X(min_length=30, max_length=100, max_targets=1, min_features=0, max_features=0),
    )
    @settings(max_examples=15, deadline=10000)
    def test_predict_time_after_training(self, data):
        """Prediction times must be strictly after the last training time."""
        from yohou.point import SeasonalNaive

        y, _ = data
        forecaster = SeasonalNaive(seasonality=1)
        forecaster.fit(y, forecasting_horizon=3)
        pred = forecaster.predict()

        last_train_time = y["time"].max()
        first_pred_time = pred["time"].min()
        assert first_pred_time > last_train_time


class TestScorerInvariants:
    """Property-based checks for scorer contracts."""

    @given(df=st_time_series(min_length=5, max_length=50, min_columns=1, max_columns=3))
    @settings(max_examples=30, deadline=5000)
    def test_mae_self_score_is_zero(self, df):
        """MAE of a DataFrame against itself is always 0."""
        from yohou.metrics import MeanAbsoluteError

        scorer = MeanAbsoluteError()
        scorer.fit(df)
        score = scorer.score(df, df)
        assert score == pytest.approx(0.0, abs=1e-10)

    @given(df=st_time_series(min_length=5, max_length=50, min_columns=1, max_columns=3))
    @settings(max_examples=30, deadline=5000)
    def test_mse_self_score_is_zero(self, df):
        """MSE of a DataFrame against itself is always 0."""
        from yohou.metrics import MeanSquaredError

        scorer = MeanSquaredError()
        scorer.fit(df)
        score = scorer.score(df, df)
        assert score == pytest.approx(0.0, abs=1e-10)

    @given(df=st_time_series(min_length=5, max_length=50, min_columns=1, max_columns=3))
    @settings(max_examples=30, deadline=5000)
    def test_mae_is_non_negative(self, df):
        """MAE is always non-negative for any prediction."""
        import numpy as np
        import polars as pl
        import polars.selectors as cs

        from yohou.metrics import MeanAbsoluteError

        rng = np.random.default_rng(42)
        # Create a shifted version as prediction
        numeric_cols = df.select(~cs.by_name("time")).columns
        pred_data = {"time": df["time"]}
        for col in numeric_cols:
            pred_data[col] = df[col].to_numpy() + rng.standard_normal(len(df))
        pred = pl.DataFrame(pred_data)

        scorer = MeanAbsoluteError()
        scorer.fit(df)
        score = scorer.score(df, pred)
        assert score >= 0.0
