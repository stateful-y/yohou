# Metrics

### Point Metrics ([View](/examples/metrics/point_metrics/) | [Open in marimo](/examples/metrics/point_metrics/edit/))

**MAE, RMSE, MAPE, and All Point Scorers**

Evaluate point forecasts with all 8 scorers: MAE, MSE, RMSE, MedianAE, MAPE, sMAPE, RMSSE, and MASE. Covers aggregation modes and time-weighted scoring for emphasizing recent performance.

### Interval Metrics ([View](/examples/metrics/interval_metrics/) | [Open in marimo](/examples/metrics/interval_metrics/edit/))

**Coverage, Winkler Score, and Interval Evaluation**

Evaluate prediction intervals with `EmpiricalCoverage`, `IntervalScore`, `MeanIntervalWidth`, `PinballLoss`, and `CalibrationError`. Essential for assessing whether your intervals are well-calibrated.

### Conformity Scorers ([View](/examples/metrics/conformity_scorers/) | [Open in marimo](/examples/metrics/conformity_scorers/edit/))

**Conformity Scorers for Conformal Prediction**

Understand `Residual`, `AbsoluteResidual`, `GammaResidual`, and `AbsoluteGammaResidual`. These are the components that determine how conformal intervals adapt to your data's error structure.

### Aggregation Modes ([View](/examples/metrics/aggregation_modes/) | [Open in marimo](/examples/metrics/aggregation_modes/edit/))

**Timewise, Componentwise, Groupwise, and More**

Explore all aggregation modes for scorers on panel data: timewise, componentwise, groupwise, coveragewise, and all. Understanding aggregation is critical for interpreting forecast quality across dimensions.

### Time-Weighted Scoring ([View](/examples/metrics/time_weighted_scoring/) | [Open in marimo](/examples/metrics/time_weighted_scoring/edit/))

**Weighting Functions for Temporal Emphasis**

Apply `exponential_decay_weight`, `linear_decay_weight`, `seasonal_emphasis_weight`, and `compose_weights` to emphasize recent or seasonal performance in your evaluation metrics.
