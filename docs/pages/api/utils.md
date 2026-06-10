---
template: api-submodule.html
---

# yohou.utils

Validation, panel data, weighting, tags, discovery, and other utility functions.

### Discovery

| Name | Description |
| --- | --- |
| [`all_estimators`](generated/yohou.utils.discovery.all_estimators.md) | Get a list of all estimators from `yohou`. |
| [`all_displays`](generated/yohou.utils.discovery.all_displays.md) | Get a list of all displays from `yohou`. |
| [`all_functions`](generated/yohou.utils.discovery.all_functions.md) | Get a list of all functions from `yohou`. |

### Panel data

| Name | Description |
| --- | --- |
| [`inspect_panel`](generated/yohou.utils.panel.inspect_panel.md) | Inspect DataFrame columns to distinguish global and local (panel) data. |
| [`get_group_df`](generated/yohou.utils.panel.get_group_df.md) | Extract and rename columns for a specific panel group. |
| [`dict_to_panel`](generated/yohou.utils.panel.dict_to_panel.md) | Convert a dict of group DataFrames to a single DataFrame with prefixed columns. |
| [`select_panel_columns`](generated/yohou.utils.panel.select_panel_columns.md) | Select panel group columns and optionally global columns of a DataFrame. |
| [`panel_aware_rename`](generated/yohou.utils.panel.panel_aware_rename.md) | Apply a rename function to a column name while preserving the panel group prefix. |
| [`panel_aware_prefix`](generated/yohou.utils.panel.panel_aware_prefix.md) | Add a prefix to a column name while preserving the panel group prefix. |
| [`panel_aware_suffix`](generated/yohou.utils.panel.panel_aware_suffix.md) | Add a suffix to a column name while preserving the panel group prefix. |
| [`check_groups`](generated/yohou.utils.validation.check_groups.md) | Validate and normalize panel group names for forecaster operations. |
| [`check_groups_exist`](generated/yohou.utils.validation.check_groups_exist.md) | Validate all requested panel groups exist in fitted forecaster. |
| [`check_panel_groups_match`](generated/yohou.utils.validation.check_panel_groups_match.md) | Validate that y and X have matching panel group structures. |
| [`check_panel_internal_consistency`](generated/yohou.utils.validation.check_panel_internal_consistency.md) | Validate that all panel groups in a DataFrame have the same local column structure. |

### Data validation

| Name | Description |
| --- | --- |
| [`validate_forecaster_data`](generated/yohou.utils.validate_data.validate_forecaster_data.md) | Validate data for forecasters. |
| [`validate_transformer_data`](generated/yohou.utils.validate_data.validate_transformer_data.md) | Validate data for transformers. |
| [`validate_scorer_data`](generated/yohou.utils.validate_data.validate_scorer_data.md) | Validate and prepare scorer input data. |
| [`validate_splitter_data`](generated/yohou.utils.validate_data.validate_splitter_data.md) | Validate data for splitters. |
| [`validate_plotting_data`](generated/yohou.utils.validate_data.validate_plotting_data.md) | Validate a DataFrame for plotting and resolve columns. |
| [`validate_plotting_params`](generated/yohou.utils.validate_data.validate_plotting_params.md) | Validate common plotting function parameters. |
| [`validate_search_data`](generated/yohou.utils.validation.validate_search_data.md) | Validate input data for hyperparameter search (GridSearchCV, RandomizedSearchCV). |
| [`validate_time_weight`](generated/yohou.utils.validate_data.validate_time_weight.md) | Validate time_weight parameter for forecasters and scorers. |
| [`validate_column_names`](generated/yohou.utils.validation.validate_column_names.md) | Validate that \_\_ separator is used only for panel data group names. |

### Time series validation

| Name | Description |
| --- | --- |
| [`check_time_column`](generated/yohou.utils.validation.check_time_column.md) | Validate that time column exists, has proper dtype, no nulls, and is sorted. |
| [`check_interval_consistency`](generated/yohou.utils.validation.check_interval_consistency.md) | Validate that a time series has uniform time spacing. |
| [`check_continuity`](generated/yohou.utils.validation.check_continuity.md) | Validate temporal continuity between consecutive DataFrames. |
| [`check_sufficient_rows`](generated/yohou.utils.validation.check_sufficient_rows.md) | Validate DataFrame has sufficient rows for operation. |
| [`check_inputs`](generated/yohou.utils.validation.check_inputs.md) | Validate that target and feature DataFrames have consistent time intervals. |
| [`check_schema`](generated/yohou.utils.validation.check_schema.md) | Validate DataFrame schema and return with proper column ordering. |
| [`check_X_actual_required`](generated/yohou.utils.validation.check_X_actual_required.md) | Validate X_actual is provided when required for recursive prediction. |
| [`check_forecasting_horizon_positive`](generated/yohou.utils.validation.check_forecasting_horizon_positive.md) | Validate forecasting horizon is positive. |
| [`check_scorer_column_selection`](generated/yohou.utils.validation.check_scorer_column_selection.md) | Subselect columns based on scorer configuration. |

### Weighting

| Name | Description |
| --- | --- |
| [`BaseWeighter`](generated/yohou.utils.weighting.BaseWeighter.md) | Base class for time-axis weighter estimators. |
| [`ExponentialDecayWeighter`](generated/yohou.utils.weighting.ExponentialDecayWeighter.md) | Exponential decay weights giving more weight to recent keys. |
| [`LinearDecayWeighter`](generated/yohou.utils.weighting.LinearDecayWeighter.md) | Linear decay weights giving more weight to recent keys. |
| [`SeasonalEmphasisWeighter`](generated/yohou.utils.weighting.SeasonalEmphasisWeighter.md) | Weights emphasizing keys in phase with the most recent seasonal position. |
| [`LookupWeighter`](generated/yohou.utils.weighting.LookupWeighter.md) | Explicit per-key weights from a mapping. |
| [`TableWeighter`](generated/yohou.utils.weighting.TableWeighter.md) | DataFrame-driven weights resolved by joining on a key column. |
| [`ProductWeighter`](generated/yohou.utils.weighting.ProductWeighter.md) | Compose weighters by element-wise multiplication. |
| [`combine_weight_vectors`](generated/yohou.utils.weighting.combine_weight_vectors.md) | Combine weight vectors multiplicatively and normalize. |
| [`normalize_weights`](generated/yohou.utils.weighting.normalize_weights.md) | Normalize weights so they sum to the number of elements. |
| [`resolve_weighter_to_array`](generated/yohou.utils.weighting.resolve_weighter_to_array.md) | Resolve a weighter to a validated numpy weight array aligned to the key series. |
| [`validate_weight_array`](generated/yohou.utils.weighting.validate_weight_array.md) | Validate a resolved weight array for NaN, negatives, infinities, and all-zero. |

### Time intervals

| Name | Description |
| --- | --- |
| [`add_interval`](generated/yohou.utils.validation.add_interval.md) | Add n intervals to a datetime (handles variable-length intervals). |
| [`interval_to_timedelta`](generated/yohou.utils.validation.interval_to_timedelta.md) | Convert fixed interval to timedelta, or None for variable intervals. |
| [`parse_interval`](generated/yohou.utils.validation.parse_interval.md) | Parse interval string into (multiplier, unit). |

### Polars helpers

| Name | Description |
| --- | --- |
| [`cast`](generated/yohou.utils.polars.cast.md) | Cast columns according to schema with integer rounding. |
| [`get_numeric_columns`](generated/yohou.utils.polars.get_numeric_columns.md) | Get list of numeric column names from a DataFrame. |
| [`tabularize`](generated/yohou.utils.tabularization.tabularize.md) | Convert time series to tabular format using lags. |
