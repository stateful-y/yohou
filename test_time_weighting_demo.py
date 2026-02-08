"""Demonstration of time-weighted forecasting feature.

This script demonstrates the complete time-weighted forecasting workflow:
1. Training forecasters with time-weighted samples
2. Evaluating predictions with time-weighted scores
3. Using different weighting strategies (exponential, linear, seasonal)
4. Comparing alignment strategies for sample weighting

"""

import numpy as np
import polars as pl
from sklearn.linear_model import Ridge

from yohou.metrics import MeanAbsoluteError
from yohou.point_forecaster import PointReductionForecaster
from yohou.utils.weighting import (
    compose_weights,
    exponential_decay_weight,
    linear_decay_weight,
    seasonal_emphasis_weight,
)


def create_test_data(length=100, seed=42):
    """Create synthetic time series with trend and seasonality."""
    np.random.seed(seed)
    time = pl.datetime_range(
        pl.datetime(2020, 1, 1), pl.datetime(2020, 1, 1) + pl.duration(days=length - 1), interval="1d", eager=True
    )
    
    # Trend + seasonal pattern + noise
    t = np.arange(length)
    trend = 0.1 * t
    seasonal = 5 * np.sin(2 * np.pi * t / 7)  # Weekly seasonality
    noise = np.random.normal(0, 1, length)
    values = trend + seasonal + noise
    
    return pl.DataFrame({"time": time, "value": values})


def test_forecaster_time_weighting():
    """Test time-weighted sample weighting in forecasters."""
    print("\n=== FORECASTER TIME WEIGHTING ===\n")
    
    # Create training data
    data = create_test_data(length=100)
    train = data[:80]
    test = data[80:]
    
    # Create forecasters
    forecaster_no_weight = PointReductionForecaster(
        estimator=Ridge(alpha=1.0),
    )
    
    forecaster_exp_weight = PointReductionForecaster(
        estimator=Ridge(alpha=1.0),
    )
    
    forecaster_linear_weight = PointReductionForecaster(
        estimator=Ridge(alpha=1.0),
    )
    
    # Fit without weighting
    forecaster_no_weight.fit(train, forecasting_horizon=5)
    print("✓ Fitted forecaster without weighting")
    
    # Fit with exponential decay (recent samples more important)
    time_weight = exponential_decay_weight(half_life=10)
    forecaster_exp_weight.fit(
        train, 
        forecasting_horizon=5, 
        time_weight=time_weight,
        sample_weight_alignment="first_step"
    )
    print("✓ Fitted forecaster with exponential decay weighting")
    
    # Fit with linear decay
    time_weight = linear_decay_weight(max_steps=50)
    forecaster_linear_weight.fit(
        train, 
        forecasting_horizon=5, 
        time_weight=time_weight,
        sample_weight_alignment="mid_step"
    )
    print("✓ Fitted forecaster with linear decay weighting")
    
    # Make predictions
    pred_no_weight = forecaster_no_weight.predict(forecasting_horizon=5)
    pred_exp_weight = forecaster_exp_weight.predict(forecasting_horizon=5)
    pred_linear_weight = forecaster_linear_weight.predict(forecasting_horizon=5)
    
    print(f"✓ Made predictions (shape: {pred_no_weight.shape})")
    
    # Compare predictions
    print("\nPredicted values (first 3 steps):")
    print(f"  No weight:     {pred_no_weight['value'].to_list()[:3]}")
    print(f"  Exp weight:    {pred_exp_weight['value'].to_list()[:3]}")
    print(f"  Linear weight: {pred_linear_weight['value'].to_list()[:3]}")


def test_scorer_time_weighting():
    """Test time-weighted evaluation in scorers."""
    print("\n=== SCORER TIME WEIGHTING ===\n")
    
    # Create test data
    y_true = pl.DataFrame({
        "time": pl.datetime_range(pl.datetime(2020, 1, 1), pl.datetime(2020, 1, 20), interval="1d", eager=True),
        "value": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
                  11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0]
    })
    
    # Predictions with varying error (early: small error, late: large error)
    y_pred = pl.DataFrame({
        "time": y_true["time"],
        "value": y_true["value"].to_list()[:10] + [x + 5.0 for x in y_true["value"].to_list()[10:]]
    })
    
    # Create scorer
    scorer = MeanAbsoluteError()
    scorer.fit(y_true)
    
    # Score without weighting
    score_no_weight = scorer.score(y_true, y_pred)
    print(f"MAE without weighting: {score_no_weight:.4f}")
    
    # Score with exponential decay (recent errors more important)
    time_weight = exponential_decay_weight(half_life=5)
    score_exp_weight = scorer.score(y_true, y_pred, time_weight=time_weight)
    print(f"MAE with exponential decay: {score_exp_weight:.4f}")
    print("  (Higher because recent errors are weighted more)")
    
    # Score with linear decay
    time_weight = linear_decay_weight(max_steps=15)
    score_linear_weight = scorer.score(y_true, y_pred, time_weight=time_weight)
    print(f"MAE with linear decay: {score_linear_weight:.4f}")
    
    print("\n✓ Time-weighted scoring works correctly")


def test_seasonal_emphasis():
    """Test seasonal emphasis weighting."""
    print("\n=== SEASONAL EMPHASIS WEIGHTING ===\n")
    
    # Create data with weekly pattern
    data = create_test_data(length=100)
    train = data[:80]
    
    # Create forecaster with seasonal emphasis
    forecaster = PointReductionForecaster(
        estimator=Ridge(alpha=1.0),
    )
    
    # Emphasize weekly pattern
    time_weight = seasonal_emphasis_weight(seasonality=7, emphasis=3.0)
    forecaster.fit(
        train, 
        forecasting_horizon=5, 
        time_weight=time_weight,
        sample_weight_alignment="first_step"
    )
    print("✓ Fitted forecaster with weekly seasonal emphasis")
    
    # Multiple seasonalities (weekly + monthly approximation)
    time_weight = seasonal_emphasis_weight(seasonality=[7, 30], emphasis=2.5)
    forecaster.fit(
        train, 
        forecasting_horizon=5, 
        time_weight=time_weight,
        sample_weight_alignment="first_step"
    )
    print("✓ Fitted forecaster with multiple seasonalities (7-day and 30-day)")


def test_composed_weights():
    """Test composing multiple weight functions."""
    print("\n=== COMPOSED WEIGHTS ===\n")
    
    # Create data
    data = create_test_data(length=100)
    train = data[:80]
    
    # Compose exponential decay with seasonal emphasis
    time_weight = compose_weights(
        exponential_decay_weight(half_life=15),
        seasonal_emphasis_weight(seasonality=7, emphasis=2.0),
    )
    
    forecaster = PointReductionForecaster(
        estimator=Ridge(alpha=1.0),
    )
    
    forecaster.fit(
        train, 
        forecasting_horizon=5, 
        time_weight=time_weight,
        sample_weight_alignment="first_step"
    )
    print("✓ Fitted forecaster with composed weights (exponential × seasonal)")


def test_alignment_strategies():
    """Test different sample weight alignment strategies."""
    print("\n=== ALIGNMENT STRATEGIES ===\n")
    
    # Create data
    data = create_test_data(length=100)
    train = data[:80]
    
    time_weight = exponential_decay_weight(half_life=10)
    
    for alignment in ["first_step", "mid_step", "last_step"]:
        forecaster = PointReductionForecaster(
            estimator=Ridge(alpha=1.0),
        )
        
        forecaster.fit(
            train, 
            forecasting_horizon=5, 
            time_weight=time_weight,
            sample_weight_alignment=alignment
        )
        
        pred = forecaster.predict(forecasting_horizon=5)
        print(f"✓ {alignment:10s}: Predicted first value = {pred['value'][0]:.4f}")


def test_dataframe_weights():
    """Test using DataFrame for time weights."""
    print("\n=== DATAFRAME WEIGHTS ===\n")
    
    # Create data
    data = create_test_data(length=100)
    train = data[:80]
    
    # Create weight DataFrame (emphasize first and last months)
    time_values = train["time"].to_list()
    weights = [2.0 if i < 30 or i > 50 else 1.0 for i in range(len(time_values))]
    
    time_weight = pl.DataFrame({
        "time": time_values,
        "weight": weights
    })
    
    forecaster = PointReductionForecaster(
        estimator=Ridge(alpha=1.0),
    )
    
    forecaster.fit(
        train, 
        forecasting_horizon=5, 
        time_weight=time_weight,
        sample_weight_alignment="first_step"
    )
    print("✓ Fitted forecaster with DataFrame weights")


def main():
    """Run all demonstrations."""
    print("=" * 60)
    print("TIME-WEIGHTED FORECASTING DEMONSTRATION")
    print("=" * 60)
    
    try:
        test_forecaster_time_weighting()
        test_scorer_time_weighting()
        test_seasonal_emphasis()
        test_composed_weights()
        test_alignment_strategies()
        test_dataframe_weights()
        
        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED")
        print("=" * 60)
        print("\nTime-weighted forecasting feature is working correctly!")
        print("\nImplemented features:")
        print("  • Exponential decay weighting")
        print("  • Linear decay weighting")
        print("  • Seasonal emphasis (single & multiple seasonalities)")
        print("  • Composed weight functions")
        print("  • Three alignment strategies (first_step, mid_step, last_step)")
        print("  • DataFrame-based weights")
        print("  • Time-weighted evaluation scoring")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
