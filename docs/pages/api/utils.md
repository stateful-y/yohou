---
template: api-submodule.html
---

# yohou.utils

Utility functions for data manipulation, validation, and tabularization.

### Classes

| Name | Description |
|------|-------------|
| [`ForecasterTags`](generated/yohou.utils.ForecasterTags.md) | Tags specific to time series forecasters. |
| [`InputTags`](generated/yohou.utils.InputTags.md) | Tags describing input requirements. |
| [`SplitterTags`](generated/yohou.utils.SplitterTags.md) | Tags specific to cross-validation splitters. |
| [`Tags`](generated/yohou.utils.Tags.md) | Metadata tags for yohou estimators. |
| [`TargetTags`](generated/yohou.utils.TargetTags.md) | Tags describing target (y) requirements. |
| [`TransformerTags`](generated/yohou.utils.TransformerTags.md) | Tags specific to time series transformers. |

### Functions

| Name | Description |
|------|-------------|
| [`all_displays`](generated/yohou.utils.all_displays.md) | Get a list of all displays from `yohou`. |
| [`all_estimators`](generated/yohou.utils.all_estimators.md) | Get a list of all estimators from `yohou`. |
| [`all_functions`](generated/yohou.utils.all_functions.md) | Get a list of all functions from `yohou`. |
| [`dict_to_panel`](generated/yohou.utils.dict_to_panel.md) | Convert a dict of group DataFrames to a single DataFrame with prefixed columns. |
| [`get_group_df`](generated/yohou.utils.get_group_df.md) | Extract and rename columns for a specific panel group. |
| [`inspect_panel`](generated/yohou.utils.inspect_panel.md) | Inspect DataFrame columns to distinguish global and local (panel) data. |
| [`panel_aware_prefix`](generated/yohou.utils.panel_aware_prefix.md) | Add a prefix to a column name while preserving the panel group prefix. |
| [`panel_aware_rename`](generated/yohou.utils.panel_aware_rename.md) | Apply a rename function to a column name while preserving the panel group prefix. |
| [`panel_aware_suffix`](generated/yohou.utils.panel_aware_suffix.md) | Add a suffix to a column name while preserving the panel group prefix. |
| [`select_panel_columns`](generated/yohou.utils.select_panel_columns.md) | Select panel group columns and optionally global columns of a DataFrame. |
| [`window_forecasts`](generated/yohou.utils.window_forecasts.md) | Window forecast data into step-indexed columns using as-of vintage selection. |
| [`window_futures`](generated/yohou.utils.window_futures.md) | Window known-future features into step-indexed columns. |
| [`cast`](generated/yohou.utils.cast.md) | Cast columns according to schema with integer rounding. |
| [`get_categorical_columns`](generated/yohou.utils.get_categorical_columns.md) | Get list of categorical column names from a DataFrame. |
| [`get_numeric_columns`](generated/yohou.utils.get_numeric_columns.md) | Get list of numeric column names from a DataFrame. |
| [`tabularize`](generated/yohou.utils.tabularize.md) | Convert time series to tabular format using lags. |
| [`validate_forecaster_data`](generated/yohou.utils.validate_forecaster_data.md) | Validate and prepare input data for forecasters. |
| [`validate_plotting_data`](generated/yohou.utils.validate_plotting_data.md) | Validate a DataFrame for plotting and resolve columns. |
| [`validate_plotting_params`](generated/yohou.utils.validate_plotting_params.md) | Validate common plotting function parameters. |
| [`validate_scorer_data`](generated/yohou.utils.validate_scorer_data.md) | Validate and prepare scorer input data. |
| [`validate_splitter_data`](generated/yohou.utils.validate_splitter_data.md) | Validate and prepare input data for time series splitters. |
| [`validate_transformer_data`](generated/yohou.utils.validate_transformer_data.md) | Validate data for transformers. |
| [`add_interval`](generated/yohou.utils.add_interval.md) | Add n intervals to a datetime (handles variable-length intervals). |
| [`check_continuity`](generated/yohou.utils.check_continuity.md) | Validate temporal continuity between consecutive DataFrames. |
| [`check_forecasting_horizon_positive`](generated/yohou.utils.check_forecasting_horizon_positive.md) | Validate forecasting horizon is positive. |
| [`check_groups`](generated/yohou.utils.check_groups.md) | Validate and normalize panel group names for forecaster operations. |
| [`check_inputs`](generated/yohou.utils.check_inputs.md) | Validate that target and feature DataFrames have consistent time intervals. |
| [`check_interval_consistency`](generated/yohou.utils.check_interval_consistency.md) | Validate that a time series has uniform time spacing. |
| [`check_panel_groups_match`](generated/yohou.utils.check_panel_groups_match.md) | Validate that y and X_actual have compatible panel group structures. |
| [`check_panel_internal_consistency`](generated/yohou.utils.check_panel_internal_consistency.md) | Validate that all panel groups in a DataFrame have the same local column structure. |
| [`check_schema`](generated/yohou.utils.check_schema.md) | Validate DataFrame schema and return with proper column ordering. |
| [`check_scorer_column_selection`](generated/yohou.utils.check_scorer_column_selection.md) | Subselect columns based on scorer configuration. |
| [`check_sufficient_rows`](generated/yohou.utils.check_sufficient_rows.md) | Validate DataFrame has sufficient rows for operation. |
| [`check_time_column`](generated/yohou.utils.check_time_column.md) | Validate that time column exists, has proper dtype, no nulls, and is sorted. |
| [`check_X_actual_required`](generated/yohou.utils.check_X_actual_required.md) | Validate X_actual is provided when required for recursive prediction. |
| [`interval_to_timedelta`](generated/yohou.utils.interval_to_timedelta.md) | Convert fixed interval to timedelta, or None for variable intervals. |
| [`parse_interval`](generated/yohou.utils.parse_interval.md) | Parse interval string into (multiplier, unit). |
| [`validate_column_names`](generated/yohou.utils.validate_column_names.md) | Validate that __ separator is used only for panel data group names. |
| [`validate_search_data`](generated/yohou.utils.validate_search_data.md) | Validate input data for hyperparameter search (GridSearchCV, RandomizedSearchCV). |
