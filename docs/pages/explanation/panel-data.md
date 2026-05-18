# Panel Data

Many forecasting problems involve multiple related time series recorded in parallel:
sales across dozens of stores, power consumption across hundreds of meters, sensor
readings across an entire factory floor. These collections are called *panel data*
(also known as longitudinal data or grouped time series), and working with them
effectively requires a data model that makes group identity explicit without forcing
the user into an entirely different API. Yohou's answer is a flat DataFrame with a
naming convention: the same polars DataFrame that carries a single time series can
carry a thousand of them, and the same forecaster `fit` call handles all groups at once.

## The Naming Convention

Yohou encodes panel structure through column names that contain a double underscore
(`__`). A column named `{group}__{variable}` is parsed as belonging to panel group
`group` with local variable name `variable`. The text before the first `__` identifies
the group; everything after it is the variable name within that group.

```python
y = pl.DataFrame({
    "time":              dates,
    "store_1__sales":    [100, 110, ...],
    "store_2__sales":    [150, 160, ...],
    "store_1__returns":  [5, 3, ...],
    "store_2__returns":  [8, 6, ...],
})
```

This represents two groups (`store_1`, `store_2`), each with two variables (`sales`,
`returns`). Columns without a `__` (other than `"time"`) are *global* columns that
belong to the dataset as a whole rather than to any specific group. The
[`inspect_panel`](/pages/api/generated/yohou.utils.panel.inspect_panel/) utility
parses this structure, returning a list of global column names and a dictionary
mapping each group prefix to its full column names.

### Validation rules

Yohou enforces two constraints on panel structure at fit time:

1. **Consistent variables across groups.** Every group must have the same set of
   variable suffixes. If `store_1` has columns `store_1__sales` and
   `store_1__returns`, then `store_2` must also have `store_2__sales` and
   `store_2__returns`. Mismatched suffixes raise a `ValueError`.

2. **No conflicts between global and panel names.** A global column cannot share
   its name with the unprefixed variable name of a panel column. For example,
   having both a global `sales` column and a panel column `store_1__sales` (whose
   unprefixed name is also `sales`) is ambiguous and rejected.

### Panel utility functions

The [`yohou.utils.panel`](/pages/api/utils/#panel) module provides helper functions
for working with panel DataFrames:

- [`inspect_panel`](/pages/api/generated/yohou.utils.panel.inspect_panel/) parses a DataFrame into global columns and a group dictionary.
- [`get_group_df`](/pages/api/generated/yohou.utils.panel.get_group_df/) extracts the columns for a single group with prefixes removed, plus any global columns.
- [`select_panel_columns`](/pages/api/generated/yohou.utils.panel.select_panel_columns/) filters a DataFrame to keep only columns belonging to specified groups.
- [`dict_to_panel`](/pages/api/generated/yohou.utils.panel.dict_to_panel/) converts a dictionary of per-group DataFrames back into a single panel DataFrame with prefixed columns.
- [`panel_aware_rename`](/pages/api/generated/yohou.utils.panel.panel_aware_rename/), [`panel_aware_prefix`](/pages/api/generated/yohou.utils.panel.panel_aware_prefix/), and [`panel_aware_suffix`](/pages/api/generated/yohou.utils.panel.panel_aware_suffix/) apply renaming operations while preserving the group prefix structure.

These utilities are mostly used internally by forecasters and transformers, but they
are also available for custom preprocessing or analysis pipelines.

## Panel Strategies

When a forecaster detects panel columns, it must decide how to use the group
structure. The `panel_strategy` parameter on forecasters accepts two values
(`"global"` and `"multivariate"`), while a third approach (local) is achieved
through a separate composition class.

### Global (`panel_strategy="global"`, the default)

The global strategy treats panel groups as separate entities that share a single
model. Each group gets its own transformer state and observation buffers, so lag
features and rolling statistics are computed independently per group. The forecaster
strips the group prefix before passing data to transformers, so a
[`LagTransformer`](/pages/api/generated/yohou.preprocessing.window.LagTransformer/)
operating on `store_1` sees a plain `sales` column, not `store_1__sales`. After
transformation, rows from all groups are stacked into a single training dataset with
a consistent feature schema, and one estimator (for example, one gradient boosting
model) learns from all of them at once.

This is the *pooled model* approach: the estimator generalizes across groups,
learning patterns that appear in `store_1` and `store_2` simultaneously. A new group
with only a few observations benefits from patterns the model learned from groups
with richer histories. The trade-off is that genuinely idiosyncratic groups, whose
patterns differ sharply from the rest of the panel, can be averaged away.

The global strategy suits most panel forecasting problems where groups share an
underlying dynamic (all stores respond to the same promotions, all meters are in
the same climate zone). It is the natural starting point.

### Multivariate (`panel_strategy="multivariate"`)

The multivariate strategy bypasses panel detection entirely. Columns with `__` are
treated as ordinary wide-format columns, and the full DataFrame (all groups, all
variables, all time steps) is fed to one transformer and one model as a single
multivariate time series.

This allows the model to see cross-group relationships directly. If `store_1__sales`
and `store_2__sales` move together during promotions, a multivariate feature
transformer can compute their correlation or ratio as an explicit feature. The global
strategy computes features per group independently and has no mechanism to express
"how does store_1 compare to store_2 right now?"

The cost is scale: the feature matrix grows with the number of groups and variables
combined, and the single model must learn a high-dimensional mapping. For large
panels this can be impractical, and the cross-group signal benefit is often marginal
compared to the global strategy.

Multivariate is most compelling for small panels (fewer than ~20 groups) where
cross-series correlations are known to be strong, such as closely related product
families or geographically adjacent weather stations.

### Local (via [`LocalPanelForecaster`](/pages/api/generated/yohou.compose.local_panel_forecaster.LocalPanelForecaster/))

The local approach is not a value of `panel_strategy`. Instead, it is achieved by
wrapping any forecaster in
[`LocalPanelForecaster`](/pages/api/generated/yohou.compose.local_panel_forecaster.LocalPanelForecaster/),
which fits a separate, independent forecaster instance for each group.
Group A's model is trained only on group A's data; group B's model only on group B's.
The groups never interact.

This maximises per-group specialization: a model can learn that `store_1` has
completely different seasonal patterns from `store_2` without those signals being
blended. The cost is data volume. Groups with short histories can produce poorly
estimated models, and any pattern that appears in multiple groups but is
underrepresented in a single group's history will be missed.

Local strategies make sense when groups genuinely have independent dynamics, when the
dataset is large enough that each group has abundant history, or when group identity
carries strong business meaning (for example, fundamentally different product
categories that should never share model parameters).

For the implementation perspective on
[`LocalPanelForecaster`](/pages/api/generated/yohou.compose.local_panel_forecaster.LocalPanelForecaster/),
see [Forecaster Composition](forecaster-composition.md).

### Choosing a strategy

| Strategy       | Groups share a model? | Cross-group features? | Scales to many groups? |
|----------------|-----------------------|-----------------------|------------------------|
| Global         | Yes (pooled)          | No                    | Yes                    |
| Multivariate   | Yes (single wide)     | Yes                   | No (~20 groups max)    |
| Local          | No (independent)      | No                    | Yes                    |

Start with global. Switch to local when groups have genuinely distinct dynamics and
enough individual history. Use multivariate only for small panels where inter-group
correlations are known to be informative.

## Panel-Aware Behavior in Forecasters

Forecasters detect panel structure automatically at fit time. When `panel_strategy`
is `"global"`, the forecaster calls
[`inspect_panel`](/pages/api/generated/yohou.utils.panel.inspect_panel/) on the
input, validates that all groups have the same variable suffixes, creates per-group
transformer instances, and builds a combined training dataset by stacking the
per-group tabularized rows. The fitted model then sees rows from all groups with a
consistent feature schema.

### The `groups` parameter

The `observe`, `predict`, and `rewind` methods accept a `groups` parameter
(`list[str] | None`) that allows operating on a subset of panel groups. This is
useful when new data arrives for some entities but not others: a retailer may
receive daily sales for most stores but weekly reconciled data for a handful, and
yohou can update and predict each subset on its own schedule without touching the
others. When `groups` is `None` (the default), all groups are included. The same
parameter is available on
[`LocalPanelForecaster`](/pages/api/generated/yohou.compose.local_panel_forecaster.LocalPanelForecaster/),
including on `observe_predict` and `observe_predict_interval`.

### How transformers interact with panel data

Transformers do not need explicit panel-awareness code. When `panel_strategy="global"`,
the forecaster handles the panel layer: it strips the group prefix, passes
unprefixed data to the transformer, collects the output, and re-adds the prefix.
Each group gets its own transformer clone with independent state. A
[`LagTransformer`](/pages/api/generated/yohou.preprocessing.window.LagTransformer/)
applied to panel data computes lags within each group, never bleeding one group's
history into another's lag features. Observation horizons and memory buffers are
also tracked per group.

When `panel_strategy="multivariate"`, the full wide DataFrame (prefixes included) is
passed to a single transformer instance. Transformers that create features from all
input columns will naturally produce cross-group features.

## Panel Scoring

Scorers aggregate errors across multiple dimensions. The `aggregation_method`
parameter controls which dimensions are collapsed before returning the result. The
five dimensions are:

- `"stepwise"`: collapse across forecasting steps (lead times). The result retains
  the vintage time and component axes.
- `"vintagewise"`: collapse across vintage times (observation origins). The result
  retains the step and component axes.
- `"componentwise"`: collapse across components (target columns). For panel data,
  this means collapsing individual variable names within and across groups.
- `"groupwise"`: collapse across panel groups, producing one score per group
  averaged over time and horizon. This is the most direct way to identify which
  entities the model handles poorly.
- `"coveragewise"`: collapse across coverage rates (interval scorers only).

The special value `"all"` collapses every dimension and produces a single scalar,
suitable for hyperparameter search where a single score is needed to rank
configurations.

Multiple dimensions can be combined: `aggregation_method=["stepwise", "vintagewise"]`
collapses both time axes but preserves the component and group structure.

### Filtering and weighting groups

Scorers accept a `groups` parameter that can be a list of group names to include
or a dictionary mapping group names to weights:

```python
scorer = MeanAbsoluteError(aggregation_method="all", groups=["store_1", "store_2"])
scorer = MeanAbsoluteError(aggregation_method="all", groups={"store_1": 0.7, "store_2": 0.3})
```

This is useful when certain entities are more important than others, or when
evaluation should focus on a specific subset of the panel.

See [Forecast Accuracy](forecast-accuracy.md) for the full discussion of aggregation
modes and their relationship to model selection.

## Cross-Validation with Panel Data

All built-in splitters
([`ExpandingWindowSplitter`](/pages/api/generated/yohou.model_selection.split.ExpandingWindowSplitter/),
[`SlidingWindowSplitter`](/pages/api/generated/yohou.model_selection.split.SlidingWindowSplitter/))
support panel data natively. They operate on row indices of the full DataFrame,
keeping all groups together in each train/test split. This means a given time step
is either in the training set or the test set for all groups simultaneously, which
is the correct behavior for time series cross-validation (avoiding leaking future
information from one group into the training window of another).

## Connections

Panel data builds on the time column contract and the three data shapes (univariate,
multivariate, panel) described in [Core Concepts](core-concepts.md). Composite
forecasters such as
[`LocalPanelForecaster`](/pages/api/generated/yohou.compose.local_panel_forecaster.LocalPanelForecaster/)
are covered in [Forecaster Composition](forecaster-composition.md), which explains
how `observe` and `rewind` propagate through nested components.
[Forecast Accuracy](forecast-accuracy.md) covers the full set of aggregation modes
and their relationship to model selection.

For practical recipes, see [How to Work with Panel Data](../how-to/panel-data.md).
The panel utility API is documented in the
[yohou.utils.panel reference](/pages/api/utils/#panel).
