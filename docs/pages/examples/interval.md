# Interval Forecasting

### Conformal Prediction Intervals ([View](/examples/interval/conformal_forecasting/) | [Open in marimo](/examples/interval/conformal_forecasting/edit/))

**Distribution-Free Prediction Intervals**

Build prediction intervals with coverage guarantees using `SplitConformalForecaster`. This example shows how conformal prediction provides valid intervals without distributional assumptions. You only need a point forecaster and a calibration set.

### Interval Reduction Forecasting ([View](/examples/interval/interval_reduction/) | [Open in marimo](/examples/interval/interval_reduction/edit/))

**Native Quantile-Based Intervals**

Generate prediction intervals directly from quantile regression with `IntervalReductionForecaster`. Compares the quantile regression approach with the conformal approach, showing the trade-offs between parametric flexibility and distribution-free guarantees.

### CatBoost MultiQuantile ([View](/examples/interval/catboost_multiquantile/) | [Open in marimo](/examples/interval/catboost_multiquantile/edit/))

**Interval Forecasting with CatBoost's Native MultiQuantile Loss**

Fit all quantile levels simultaneously in a single CatBoost model using the native `MultiQuantile` loss function. More efficient than fitting separate models per quantile.

### Conformal Conformity Scorers ([View](/examples/interval/conformal_conformity_scorers/) | [Open in marimo](/examples/interval/conformal_conformity_scorers/edit/))

**How Conformity Scorers Shape Prediction Intervals**

Explore how different conformity scorers (`Residual`, `AbsoluteResidual`, `GammaResidual`, and more) affect interval calibration and width. Understanding scorer choice is key to getting useful conformal intervals.

### Distance Similarity ([View](/examples/interval/distance_similarity/) | [Open in marimo](/examples/interval/distance_similarity/edit/))

**Adaptive Conformal Intervals via Similarity Weighting**

Use `DistanceSimilarity` to up-weight calibration residuals from observations similar to the prediction target. Produces locally-adaptive intervals that narrow in familiar regimes and widen in unfamiliar ones.

### Panel Prediction Intervals ([View](/examples/interval/panel_intervals/) | [Open in marimo](/examples/interval/panel_intervals/edit/))

**Conformal and Quantile Intervals on Panel Data**

Apply both conformal and quantile regression intervals to panel data with per-group calibration and coverage analysis. Shows how interval forecasting scales to grouped time series.
