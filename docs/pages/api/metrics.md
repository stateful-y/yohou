# yohou.metrics

Scorers for evaluating point forecasts, prediction intervals, and conformal prediction calibration.

**User guide**: See the [Model Selection](../user-guide/model-selection.md) section for further details.

## Base Classes

::: yohou.metrics.BasePointScorer
    options:
      show_root_heading: true
      show_source: true
      members_order: source

::: yohou.metrics.BaseIntervalScorer
    options:
      show_root_heading: true
      show_source: true
      members_order: source

::: yohou.metrics.BaseConformityScorer
    options:
      show_root_heading: true
      show_source: true
      members_order: source

## Point Scorers

::: yohou.metrics.MeanAbsoluteError
    options:
      show_root_heading: true
      show_source: true
      members_order: source

::: yohou.metrics.MeanSquaredError
    options:
      show_root_heading: true
      show_source: true
      members_order: source

::: yohou.metrics.RootMeanSquaredError
    options:
      show_root_heading: true
      show_source: true
      members_order: source

::: yohou.metrics.MeanAbsolutePercentageError
    options:
      show_root_heading: true
      show_source: true
      members_order: source

::: yohou.metrics.SymmetricMeanAbsolutePercentageError
    options:
      show_root_heading: true
      show_source: true
      members_order: source

::: yohou.metrics.MedianAbsoluteError
    options:
      show_root_heading: true
      show_source: true
      members_order: source

::: yohou.metrics.MeanAbsoluteScaledError
    options:
      show_root_heading: true
      show_source: true
      members_order: source

::: yohou.metrics.RootMeanSquaredScaledError
    options:
      show_root_heading: true
      show_source: true
      members_order: source

## Conformity Scorers

::: yohou.metrics.Residual
    options:
      show_root_heading: true
      show_source: true
      members_order: source

::: yohou.metrics.AbsoluteResidual
    options:
      show_root_heading: true
      show_source: true
      members_order: source

::: yohou.metrics.GammaResidual
    options:
      show_root_heading: true
      show_source: true
      members_order: source

::: yohou.metrics.AbsoluteGammaResidual
    options:
      show_root_heading: true
      show_source: true
      members_order: source

::: yohou.metrics.QuantileResidual
    options:
      show_root_heading: true
      show_source: true
      members_order: source

::: yohou.metrics.AbsoluteQuantileResidual
    options:
      show_root_heading: true
      show_source: true
      members_order: source

## Interval Scorers

::: yohou.metrics.IntervalScore
    options:
      show_root_heading: true
      show_source: true
      members_order: source

::: yohou.metrics.CalibrationError
    options:
      show_root_heading: true
      show_source: true
      members_order: source

::: yohou.metrics.EmpiricalCoverage
    options:
      show_root_heading: true
      show_source: true
      members_order: source

::: yohou.metrics.MeanIntervalWidth
    options:
      show_root_heading: true
      show_source: true
      members_order: source

::: yohou.metrics.PinballLoss
    options:
      show_root_heading: true
      show_source: true
      members_order: source
