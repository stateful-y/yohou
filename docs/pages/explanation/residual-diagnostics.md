# Residual Diagnostics

After fitting a forecaster, the natural question is: did the model capture all the
predictable structure? Residuals, the differences between fitted (in-sample) values
and actual observations, answer this by examining what the model left behind:

$$e_t = y_t - \hat{y}_t$$

If residuals look like random noise, the model has done its job. If they show patterns,
there is information the model failed to exploit, and the forecast can be improved.

Residuals differ from forecast errors in an important way. Residuals use the training
data (where the model was fit), while forecast errors use held-out test data. Residual
diagnostics check the model's internal consistency: whether it has extracted all
available information from history. Forecast errors measure actual predictive
performance and are covered in [Forecast Accuracy](forecast-accuracy.md).

## Properties of Good Residuals

A well-specified forecasting model produces residuals that satisfy four properties,
roughly ordered from most to least important:

**1. Uncorrelated.** Good residuals have no significant autocorrelation at any lag. If
residuals at lag 7 are correlated, the model has missed a weekly pattern, and adding a
lag-7 feature or seasonal component would improve it. This is the most critical property
because correlated residuals mean the forecast is leaving predictable information unused.

**2. Zero mean.** The residuals should be centered around zero on average. A non-zero
mean indicates systematic bias: the model consistently over-predicts or under-predicts.
This is usually the easiest problem to fix (add a bias correction or adjust the
intercept).

**3. Constant variance (homoscedasticity).** The spread of residuals should be roughly
the same across all time periods. If residuals are small during calm periods and large
during volatile ones, variance-stabilizing transforms can help:
[`BoxCoxTransformer`](/pages/api/generated/yohou.stationarity.transformers.BoxCoxTransformer/)
works well for power-law scaling but requires strictly positive data, while
[`ASinhTransformer`](/pages/api/generated/yohou.stationarity.transformers.ASinhTransformer/)
handles negative values and outliers more robustly via the inverse hyperbolic sine.
See [Stationarity](stationarity.md) for details on these transforms.

**4. Normal distribution.** Normally distributed residuals are desirable for
constructing parametric prediction intervals, but this property is less critical for
point forecasting. Yohou's conformal prediction approach (see
[Interval Forecasting](interval-forecasting.md)) does not require normality at all,
which is one of its key advantages.

If the first two properties hold, the model's point forecasts are unbiased and
efficient. If all four hold, the model is well-specified and prediction intervals based
on distributional assumptions will be reliable.

## Visual Diagnostics

Yohou's plotting module provides tools for each diagnostic check.

### Four-Panel Residual Diagnostics

[`plot_residuals`](/pages/api/generated/yohou.plotting.evaluation.plot_residuals/)
produces a 4-panel diagnostic layout when called with a single target column. The four
panels map directly to the four properties above:

1. **Residuals Over Time** checks for bias and non-stationarity. A horizontal band
   noticeably above or below zero signals systematic bias, and checking whether trend
   removal was adequate is the natural first step. A fan-out pattern (variance growing
   over time) signals heteroscedasticity, pointing toward a variance-stabilizing
   transform. Isolated spikes that stand well outside the normal range suggest outliers
   worth investigating for data quality issues or genuine exceptional events.

2. **Residuals vs Fitted Values** checks for heteroscedasticity and nonlinear
   misspecification. If the scatter shows a funnel shape (wider spread at higher fitted
   values), the model's errors scale with the prediction magnitude, and a
   variance-stabilizing transform or a multiplicative model may be more appropriate. A
   curved pattern suggests the model missed a nonlinear relationship.

3. **Histogram of Residuals** checks the distributional shape. A roughly bell-shaped
   histogram centered at zero is consistent with normality. Heavy tails or strong skew
   suggest that parametric intervals based on normality assumptions would be
   miscalibrated, though conformal intervals remain valid.

4. **Q-Q Plot** compares the empirical residual quantiles against theoretical normal
   quantiles. Points falling along the 45-degree reference line confirm normality.
   Systematic deviations at the tails (S-curves or J-curves) reveal heavy tails or skew
   that the histogram may obscure.

When multiple columns are resolved (through `columns` or `groups`),
[`plot_residuals`](/pages/api/generated/yohou.plotting.evaluation.plot_residuals/)
produces a faceted layout showing residuals over time for each column, which is useful
for comparing residual behavior across components in multivariate or panel data.

### Autocorrelation Analysis

[`plot_autocorrelation`](/pages/api/generated/yohou.plotting.diagnostics.plot_autocorrelation/)
applied to residuals reveals remaining temporal structure. Significant spikes at seasonal
lags (7 for weekly, 12 for monthly, 52 for annual-in-weekly data) mean the model missed
a periodic pattern, and a seasonal component or differencing step would remove it.
Significant spikes at low lags (1 through 3) mean the model's lag feature set was
insufficient: more recent observations carry information the model did not see. A slowly
decaying ACF rather than sharp spikes indicates trend-like structure still in the
residuals, suggesting more aggressive differencing or a stronger trend component. When
all autocorrelation values fall within the significance bounds, the residuals are
consistent with white noise and the model has extracted what was predictable.

[`plot_partial_autocorrelation`](/pages/api/generated/yohou.plotting.diagnostics.plot_partial_autocorrelation/)
complements the ACF by showing correlation between the series and each lag after
removing the effect of intermediate lags. This is useful for identifying the
autoregressive order: a PACF that cuts off sharply after lag $p$ suggests that $p$ lag
features would capture the remaining linear dependence. The distinction matters because
a significant ACF at lag 2 could be a direct effect or merely a consequence of strong
lag-1 correlation; the PACF separates these cases.

### Additional Diagnostic Plots

[`plot_lag_scatter`](/pages/api/generated/yohou.plotting.diagnostics.plot_lag_scatter/)
creates scatter plots of $y_t$ vs $y_{t-k}$ for chosen lags $k$. Applied to residuals,
these reveal nonlinear dependence that the ACF (a linear measure) would miss. A curve
or cluster pattern in the lag scatter indicates that a more flexible model or additional
nonlinear features could improve predictions.

### Ljung-Box Test

For formal significance testing beyond visual inspection, the Ljung-Box test checks
whether the autocorrelations up to a given lag are jointly significantly different from
zero. Yohou does not implement this test directly, but it is available via
`statsmodels.stats.diagnostic.acorr_ljungbox` and can be applied to residual DataFrames.

## Extracting Residuals

To compute residuals from a fitted forecaster, subtract its in-sample predictions from
the actuals. [`plot_residuals`](/pages/api/generated/yohou.plotting.evaluation.plot_residuals/)
computes this internally from `y_pred` and `y_truth`, so no manual subtraction is needed
when using the diagnostic plots.

For
[`DecompositionPipeline`](/pages/api/generated/yohou.compose.decomposition_pipeline.DecompositionPipeline/),
setting `store_residuals=True` at construction stores residuals after each component
forecaster in the `residuals_` attribute (a `dict[str, pl.DataFrame]` keyed by
component name). This lets you diagnose each stage of the decomposition independently,
checking whether the trend model left seasonal structure and whether the seasonal model
left autocorrelation.

## Residuals and Conformal Prediction

Yohou's conformal prediction framework connects directly to residual analysis.
Conformity scorers compute standardized residuals on a calibration set, and the
empirical distribution of these scores determines prediction interval width. Four
point-forecast conformity scorers are available:

| Scorer | Formula | Intervals | When to use |
|---|---|---|---|
| [`Residual`](/pages/api/generated/yohou.metrics.conformity.Residual/) | $s = y - \hat{y}$ | Asymmetric | Errors are systematically skewed |
| [`AbsoluteResidual`](/pages/api/generated/yohou.metrics.conformity.AbsoluteResidual/) | $s = \lvert y - \hat{y} \rvert$ | Symmetric | Errors have roughly constant variance |
| [`GammaResidual`](/pages/api/generated/yohou.metrics.conformity.GammaResidual/) | $s = \frac{y - \hat{y}}{\hat{y} + \epsilon}$ | Asymmetric, adaptive | Variance scales with prediction magnitude |
| [`AbsoluteGammaResidual`](/pages/api/generated/yohou.metrics.conformity.AbsoluteGammaResidual/) | $s = \left\lvert \frac{y - \hat{y}}{\hat{y} + \epsilon} \right\rvert$ | Symmetric, adaptive | Symmetric errors that scale with magnitude |

If the residuals are well-behaved (uncorrelated, constant variance), the conformal
intervals will be well-calibrated. If the residuals exhibit heteroscedasticity, the
gamma scorers adapt interval width to the prediction magnitude, producing narrower
intervals where the model is more precise and wider intervals where it is less certain.

Improving residual quality directly narrows conformal intervals. A model with large,
variable residuals produces wide intervals because the conformity score distribution is
spread out. A model with small, stable residuals produces tight intervals because the
distribution is concentrated. This makes residual diagnostics a tool for interval
quality, not just point forecast quality: if the residual ACF shows significant
autocorrelation, the conformity scores on the calibration set may not be exchangeable,
which can degrade coverage guarantees.

The diagnostic plots guide the choice of conformity scorer directly. A flat
residual-over-time plot with constant spread points to `AbsoluteResidual`. A fan-shaped
residuals-vs-fitted scatter (wider spread at higher predicted values) points to
`GammaResidual` or `AbsoluteGammaResidual`.

## Panel Data

All diagnostic plotting functions support panel data through the `groups` and
`facet_by` parameters. `facet_by="group"` creates one subplot per group, while
`facet_by="member"` creates one per member. This lets you check whether residual
properties hold across all entities in a panel, or whether certain groups exhibit
patterns (such as higher variance or remaining seasonality) that others do not.

## References

- Hyndman, R.J. & Athanasopoulos, G. (2021). [Forecasting: principles and practice](https://otexts.com/fpp3/), 3rd edition, OTexts. Chapters 5.3 and 5.4.
- Ljung, G.M. & Box, G.E.P. (1978). "On a measure of lack of fit in time series models." Biometrika, 65(2), 297-303.

## Connections

Residual diagnostics complement [Forecast Accuracy](forecast-accuracy.md) metrics:
accuracy tells you how well the model predicts, diagnostics tell you whether there is
room to improve. The conformal prediction framework in
[Interval Forecasting](interval-forecasting.md) uses residual-based conformity scores
to construct prediction intervals. The [Stationarity](stationarity.md) page covers
the transforms used to address non-stationary residual patterns. The
[Visualization](visualization.md) page provides an overview of all available plotting
functions.

For practical recipes, see
[How to Evaluate Forecast Accuracy](../how-to/evaluate-forecast-accuracy.md) and
[How to Visualize Forecasts](../how-to/visualize-forecasts.md).

Interactive examples: [Evaluation](/examples/evaluation/) and
[Forecasting Visualization](/examples/forecasting_visualization/) demonstrate
residual plots and ACF analysis on fitted forecasters.
[Conformity Scorers](/examples/conformity_scorers/) compares different conformity
scorers on the same dataset.

!!! note
    Residual analysis applies to numeric (point and interval) forecasts. For
    categorical forecasts, evaluate using calibration curves and classification
    metrics (LogLoss, BrierScore) instead. See
    [Class-Probability Forecasting](class-probability-forecasting.md).
