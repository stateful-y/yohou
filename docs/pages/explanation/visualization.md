# Visualization

Yohou includes a Plotly-based plotting module that mirrors the stages of a
forecasting project. Rather than a bag of unrelated chart types, the module is
organized around what you are trying to understand at each phase of the
analysis. All functions accept polars DataFrames directly, return interactive
`plotly.graph_objects.Figure` objects, and handle panel data automatically.


## The Forecasting Visualization Workflow

A time series project moves through distinct analytical phases, and different
visualizations serve each one. Understanding this progression helps you pick the
right plot at the right moment.

### Exploring the Data

Before building any model, you need to understand the raw data: its shape,
patterns, and quality. Functions like `plot_time_series` and
`plot_rolling_statistics` give a first look at trends and volatility.
`plot_missing_data` reveals gaps that might need imputation.
`plot_distribution` and `plot_boxplot` expose the value landscape. These
exploration plots take a single DataFrame and require minimal configuration,
making them the natural starting point.

### Understanding the Structure

Once you have a general sense of the data, diagnostics go deeper into its
temporal structure. ACF and PACF plots (`plot_autocorrelation`,
`plot_partial_autocorrelation`) reveal lag dependencies and help inform model
order selection. Seasonality plots (`plot_seasonality`, `plot_subseasonality`)
decompose a series by period to show recurring patterns. `plot_lag_scatter`
exposes nonlinear relationships that correlation plots might miss.
Frequency-domain tools (`plot_spectrum`, `plot_phase`) complement the time-domain
view for signals with complex periodic structure.

This stage is about building intuition. The patterns you observe here directly
inform choices about stationarity transforms, lag features, and seasonal
components. See [Time Series Patterns](time-series-patterns.md) for how to
translate visual observations into modeling decisions.

### Visualizing Forecasts

After fitting a model, the central question is "how do the predictions look?"
`plot_forecast` overlays predicted values against actuals with optional
historical context. When you pass predictions as a `dict[str, pl.DataFrame]`,
multiple models appear side by side for comparison. Prediction intervals render
automatically when the forecast includes interval columns.

`plot_decomposition` displays the individual components of a decomposed series
(trend, seasonality, residual) as separate subplots, helping you assess whether
the decomposition captured the right structure. `plot_time_weight` visualizes how
the weighting function emphasized different time periods during training.

### Evaluating Model Quality

After generating predictions, you need to assess whether they are trustworthy.
`plot_residuals` produces a four-panel diagnostic (residuals over time, versus
fitted values, histogram, Q-Q plot) that tests whether residuals behave like
white noise. `plot_calibration` checks whether prediction intervals achieve
their nominal coverage rates. See [Residual Diagnostics](residual-diagnostics.md)
for how to interpret these plots.

Scoring visualizations reveal where and when the model struggles.
`plot_score_per_step` shows how error varies across forecast horizon steps,
while `plot_score_per_vintage` tracks accuracy over successive forecast origins
to detect degradation over time. `plot_score_time_series` and
`plot_score_distribution` display per-timestep patterns and their spread.
`plot_score_heatmap` combines two dimensions (e.g. step and vintage) into a
single view, making it easy to spot systematic weaknesses.
For a final model selection decision, `plot_score_summary` compares aggregate
scores across models and metrics in a grouped bar chart, and `plot_group_scores`
breaks down performance by panel group to ensure no entity is left behind.

### Cross-Validation and Search

`plot_splits` visualizes the train/test partitions from a temporal CV splitter,
confirming that temporal ordering is preserved and the folds look sensible.
`plot_cv_results_scatter` displays hyperparameter search results, connecting
parameter choices to performance.


## Panel Data

Most plotting functions accept a `groups` parameter for working with
panel (grouped) time series. When panel columns are detected, the module creates
faceted subplots (one per group) with consistent colors across groups. The
`facet_n_cols` parameter controls the grid layout. Passing a specific list of
group names filters which panels appear, which is useful for large panel datasets
where plotting everything at once would be overwhelming.


## Plotly and Styling

Every function returns a `plotly.graph_objects.Figure`, so plots are interactive
by default (zoom, pan, hover, legend toggling). A shared `apply_default_layout`
utility standardizes appearance across all plots. Yohou ships with a 12-color
palette (`palette_yohou()`) that cycles when more series exist than colors.

Functions that compare predictions against actuals share a consistent parameter
convention (`y_train`, `y_test`, `y_pred`, `n_history`), making the interface
predictable as you move from visualizing forecasts to evaluating their quality.
Exploration and diagnostic functions use `df` as input and `columns` to select
which value columns to plot.

---

**See also**: [Core Concepts](core-concepts.md) for the time column contract and
panel data conventions. [Forecasting](forecasting.md) covers the prediction
workflow that produces the data these plotting functions visualize.
[API Reference: yohou.plotting](/pages/api/plotting/) for the full function
listing with parameters and examples.
