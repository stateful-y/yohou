# Tags

Tags are class-level dictionaries (`_tags`) that describe component capabilities. For an explanation of tag resolution, MRO merging, and dynamic tags, see [Extending Yohou](../explanation/extending-yohou.md#the-tag-system).

## Top-Level Tags

These apply to all estimator types.

| Tag | Type | Default | Description |
|-----|------|---------|-------------|
| `estimator_type` | `str` or `None` | `None` | One of `"transformer"`, `"forecaster"`, `"scorer"`, `"similarity"`, `"splitter"` |
| `requires_fit` | `bool` | `True` | Whether the estimator needs to be fitted before use |
| `non_deterministic` | `bool` | `False` | Whether the estimator produces non-deterministic results |

## Input Tags (`input_tags`)

| Tag | Type | Default | Description |
|-----|------|---------|-------------|
| `requires_time_column` | `bool` | `True` | Whether a `"time"` column is required in input DataFrames |
| `pairwise` | `bool` | `False` | Whether the estimator expects pairwise inputs (e.g., distance matrices) |
| `allow_nan` | `bool` | `False` | Whether NaN values are allowed in inputs |
| `min_value` | `float` or `None` | `None` | Minimum value constraint for inputs (exclusive) |

## Target Tags (`target_tags`)

| Tag | Type | Default | Description |
|-----|------|---------|-------------|
| `required` | `bool` | `False` | Whether `fit()` requires the `y` parameter |
| `min_value` | `float` or `None` | `None` | Minimum value constraint for target values (exclusive) |

## Transformer Tags (`transformer_tags`)

| Tag | Type | Default | Description |
|-----|------|---------|-------------|
| `stateful` | `bool` | `False` | Whether the transformer maintains an observation window |
| `invertible` | `bool` | `False` | Whether `inverse_transform` is available |
| `preserves_dtype` | `bool` | `False` | Whether the transformer preserves input data types |

## Forecaster Tags (`forecaster_tags`)

| Tag | Type | Default | Description |
|-----|------|---------|-------------|
| `forecaster_type` | `frozenset` or `None` | `None` | Capabilities: `{"point"}`, `{"interval"}`, `{"class_proba"}`, or combinations |
| `stateful` | `bool` | `False` | Whether the forecaster uses an observation horizon mechanism |
| `uses_reduction` | `bool` | `False` | Whether the forecaster converts to tabular regression |
| `uses_target_transformer` | `bool` | `False` | Whether a target transformer is applied before fitting |
| `uses_feature_transformer` | `bool` | `False` | Whether a feature transformer is applied before fitting |
| `supports_panel_data` | `bool` | `True` | Whether the forecaster handles panel data |
| `supports_time_weight` | `bool` | `False` | Whether time weighting is supported during training |
| `supports_vintage_weight` | `bool` | `False` | Whether vintage weighting is supported |
| `requires_exogenous` | `bool` | `True` | Whether `X_actual` must be provided at `fit()` time |
| `tracks_observations` | `bool` | `True` | Whether observation tracking follows the standard pattern |

## Scorer Tags (`scorer_tags`)

| Tag | Type | Default | Description |
|-----|------|---------|-------------|
| `prediction_type` | `str` or `None` | `None` | One of `"point"`, `"interval"`, `"class_proba"` |
| `lower_is_better` | `bool` | `True` | Whether lower scores indicate better performance |
| `requires_calibration` | `bool` | `False` | Whether calibration data is needed from `fit()` |
| `symmetric` | `bool` | `False` | Whether conformity scores are symmetric |
| `multiplicative` | `bool` | `False` | Whether residuals are normalized by prediction magnitude |

## Splitter Tags (`splitter_tags`)

| Tag | Type | Default | Description |
|-----|------|---------|-------------|
| `splitter_type` | `str` or `None` | `None` | One of `"expanding"`, `"sliding"`, `"gap"` |
| `supports_panel_data` | `bool` | `False` | Whether the splitter handles panel data |
| `produces_non_overlapping_tests` | `bool` | `True` | Whether test sets across splits are guaranteed non-overlapping |
| `stateful` | `bool` | `False` | Whether the splitter maintains state across `split()` calls |

## Similarity Tags (`similarity_tags`)

| Tag | Type | Default | Description |
|-----|------|---------|-------------|
| `symmetric` | `bool` | `True` | Whether `similarity(A, B) == similarity(B, A)` |
| `requires_predictions` | `bool` | `True` | Whether predictions (`y_pred`) are needed in addition to actuals |
| `produces_weights` | `bool` | `True` | Whether the measure produces weights for conformity scores |
