# Residual Diagnostics

After fitting a forecaster, the natural question is: did the model capture all the predictable structure? Residual analysis answers this by examining what the model left behind. If the residuals look like random noise, the model has done its job. If they show patterns, there is information the model failed to exploit, and the forecast could be improved.

## What Are Residuals?

In the context of forecasting, residuals are the differences between the model's fitted (in-sample) values and the actual observations. For a one-step-ahead model they are simply `e(t) = y(t) - y_hat(t)` where `y_hat(t)` is the fitted value at time `t`. Innovation residuals are the specific case where each residual is the one-step forecast error given all data up to `t-1`.

Residuals differ from forecast errors in an important way: residuals use the training data (where the model was fit), while forecast errors use held-out test data. Residual diagnostics check the model's internal consistency: whether it has extracted all available information. Forecast errors on test data measure actual predictive performance and are covered in [Forecast Accuracy](forecast-accuracy.md).

## Properties of Good Residuals

A well-specified forecasting model produces residuals that satisfy four properties, roughly ordered from most to least important:

**1. Uncorrelated.** Good residuals have no significant autocorrelation at any lag. If residuals at lag 7 are correlated, the model has missed a weekly pattern, and adding a lag-7 feature or seasonal component would improve it. This is the most critical property because correlated residuals mean the forecast is leaving predictable information unused.

**2. Zero mean.** The residuals should be centered around zero on average. A non-zero mean indicates systematic bias: the model consistently over-predicts or under-predicts. This is usually the easiest problem to fix (add a bias correction or adjust the intercept).

**3. Constant variance (homoscedasticity).** The spread of residuals should be roughly the same across all time periods. If residuals are small during calm periods and large during volatile ones, variance-stabilizing transforms like [`BoxCoxTransformer`](/pages/api/generated/yohou.stationarity.transformers.BoxCoxTransformer/) or [`ASinhTransformer`](/pages/api/generated/yohou.stationarity.transformers.ASinhTransformer/) can help.

**4. Normal distribution.** Normally distributed residuals are desirable for constructing parametric prediction intervals, but this property is less critical for point forecasting. Yohou's conformal prediction approach (see [Interval Forecasting](interval-forecasting.md)) does not require normality at all, which is one of its key advantages.

If the first two properties hold, the model's point forecasts are unbiased and efficient. If all four hold, life is good.

## Visual Diagnostics

Yohou's plotting module provides tools for each diagnostic check:

[`plot_residuals`](/pages/api/generated/yohou.plotting.evaluation.plot_residuals/) shows residuals over time. Look for:

- **Non-zero mean**: A horizontal band that is noticeably above or below zero.
- **Changing variance**: Residuals that fan out (growing variance over time) or compress (shrinking variance).
- **Trends in residuals**: A slow drift upward or downward, indicating the trend was not fully removed.
- **Outliers**: Isolated spikes that may warrant investigation or outlier treatment.

[`plot_autocorrelation`](/pages/api/generated/yohou.plotting.diagnostics.plot_autocorrelation/) applied to residuals reveals remaining temporal structure:

- **Significant spikes at specific lags**: The model missed patterns at those lags. If the spike is at a seasonal lag (7, 12, 52, etc.), a seasonal component or differencing transform would help.
- **Slowly decaying ACF**: The residuals still contain trend-like structure. More aggressive differencing or a stronger trend component is needed.
- **All values within bounds**: The residuals are consistent with white noise. The model has captured the available structure.

For a formal significance test beyond visual inspection, the Ljung-Box test checks whether the set of autocorrelations up to a given lag is jointly significantly different from zero. While yohou does not implement this test directly, it is available in statsmodels (`statsmodels.stats.diagnostic.acorr_ljungbox`) and can be applied to residual DataFrames extracted from the forecaster.

## Residuals and Conformal Prediction

Yohou's conformal prediction framework connects directly to residual analysis. Conformity scores like [`Residual`](/pages/api/generated/yohou.metrics.conformity.Residual/), [`AbsoluteResidual`](/pages/api/generated/yohou.metrics.conformity.AbsoluteResidual/), and [`GammaResidual`](/pages/api/generated/yohou.metrics.conformity.GammaResidual/) are essentially standardized residuals computed on a calibration set. The empirical distribution of these scores determines the prediction interval width. If the residuals are well-behaved (uncorrelated, constant variance), the conformal intervals will be well-calibrated. If the residuals exhibit structure (e.g., heteroscedasticity), the [`GammaResidual`](/pages/api/generated/yohou.metrics.conformity.GammaResidual/) scorer adapts interval width to the prediction magnitude, producing narrower intervals where the model is more precise and wider intervals where it is less certain.

This connection means that residual diagnostics are not just about point forecast quality: they directly inform the choice of conformity scorer for interval prediction.

## What to Do When Diagnostics Fail

| Diagnostic finding | Likely cause | Suggested action |
|-------------------|-------------|-----------------|
| Autocorrelation at seasonal lags | Missing seasonal component | Add [`SeasonalDifferencing`](/pages/api/generated/yohou.stationarity.transformers.SeasonalDifferencing/) or a seasonality forecaster in a [`DecompositionPipeline`](/pages/api/generated/yohou.compose.decomposition_pipeline.DecompositionPipeline/) |
| Autocorrelation at low lags (1-3) | Insufficient lag features | Add more lags via [`LagTransformer`](/pages/api/generated/yohou.preprocessing.window.LagTransformer/) |
| Non-zero mean | Systematic bias | Check trend removal; consider adding a bias term |
| Growing variance | Heteroscedasticity | Apply [`BoxCoxTransformer`](/pages/api/generated/yohou.stationarity.transformers.BoxCoxTransformer/) or [`ASinhTransformer`](/pages/api/generated/yohou.stationarity.transformers.ASinhTransformer/) as target transformer |
| Isolated large residuals | Outliers | Investigate data quality; consider [`OutlierThresholdHandler`](/pages/api/generated/yohou.preprocessing.outlier.OutlierThresholdHandler/) |

## References

- Hyndman, R.J. & Athanasopoulos, G. (2021). [Forecasting: principles and practice](https://otexts.com/fpp3/), 3rd edition, OTexts. Chapters 5.3 and 5.4.
- Ljung, G.M. & Box, G.E.P. (1978). "On a measure of lack of fit in time series models." Biometrika, 65(2), 297-303.

## Connections

Residual diagnostics complement [Forecast Accuracy](forecast-accuracy.md) metrics:
accuracy tells you how well the model predicts, diagnostics tell you whether there is
room to improve. The conformal prediction framework in
[Interval Forecasting](interval-forecasting.md) uses residual-based conformity scores
to construct prediction intervals. The [Stationarity](stationarity.md) page covers
the transforms used to address non-stationary residual patterns.

Practical examples: [Evaluation](/examples/evaluation/) and
[Forecasting Visualization](/examples/forecasting_visualization/) demonstrate
residual plots and ACF analysis on fitted forecasters.

!!! note
    Residual analysis applies to numeric (point and interval) forecasts. For
    categorical forecasts, evaluate using calibration curves and classification
    metrics (LogLoss, BrierScore) instead. See
    [Class-Probability Forecasting](class-probability-forecasting.md).
