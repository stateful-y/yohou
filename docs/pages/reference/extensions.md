# Extensions

Extension packages add forecasters, metrics, and integrations to Yohou. This page lists all official and community extensions, and documents the base classes available for building custom components.

## Official Extensions

| Name | Install | Description |
|------|---------|-------------|
| yohou-optuna | `uv add yohou-optuna` | Hyperparameter optimization via [Optuna](https://optuna.org/). Provides [`OptunaSearchCV`](https://github.com/stateful-y/yohou/tree/main/packages/yohou-optuna) as a drop-in replacement for [`GridSearchCV`](/pages/api/generated/yohou.model_selection.search.GridSearchCV/) and [`RandomizedSearchCV`](/pages/api/generated/yohou.model_selection.search.RandomizedSearchCV/). ([source](https://github.com/stateful-y/yohou/tree/main/packages/yohou-optuna)) |
| yohou-nixtla | `uv add yohou-nixtla` | Integration with [Nixtla](https://nixtla.io/) forecasting libraries (statsforecast, mlforecast, neuralforecast). Wraps Nixtla models as Yohou forecasters. ([source](https://github.com/stateful-y/yohou/tree/main/packages/yohou-nixtla)) |

## Community Extensions

No community extensions are listed yet. Community extensions can be submitted via a [GitHub issue](https://github.com/stateful-y/yohou/issues/new).

## Extension Points

All custom components inherit from one of the base classes below. Each base class provides the estimator interface (`fit`, `predict`, `score`, etc.) and requires subclasses to implement specific abstract methods.

For step-by-step implementation guides, see [Create a Point Forecaster](../how-to/create-a-point-forecaster.md), [Create an Interval Forecaster](../how-to/create-an-interval-forecaster.md), [Create a Transformer](../how-to/create-a-transformer.md), and [Create a Custom Scorer](../how-to/create-a-scorer.md). For an explanation of how tags, MRO merging, and dynamic configuration work, see [Extending Yohou](../explanation/extending-yohou.md).

### Forecasters

| Base Class | Import | Abstract Methods |
|-----------|--------|-----------------|
| [`BasePointForecaster`](/pages/api/generated/yohou.point.base.BasePointForecaster/) | `yohou.point` | `fit()`, `_predict_one()` |
| [`BaseIntervalForecaster`](/pages/api/generated/yohou.interval.base.BaseIntervalForecaster/) | `yohou.interval` | `fit()`, `_predict_interval_one()` |
| [`BaseClassProbaForecaster`](/pages/api/generated/yohou.class_proba.base.BaseClassProbaForecaster/) | `yohou.class_proba` | `fit()`, `_predict_class_proba_one()` |

### Scorers

| Base Class | Import | Abstract Methods |
|-----------|--------|-----------------|
| [`BasePointScorer`](/pages/api/generated/yohou.metrics.base.BasePointScorer/) | `yohou.metrics` | `score()`, `_compute_raw_errors()` |
| [`BaseIntervalScorer`](/pages/api/generated/yohou.metrics.base.BaseIntervalScorer/) | `yohou.metrics` | `score()`, `_compute_raw_scores()` |
| [`BaseClassProbaScorer`](/pages/api/generated/yohou.metrics.base.BaseClassProbaScorer/) | `yohou.metrics` | `score()`, `_compute_raw_errors()` |

### Transformers

| Base Class | Import | Abstract Methods |
|-----------|--------|-----------------|
| [`BaseTransformer`](/pages/api/generated/yohou.base.transformer.BaseTransformer/) | `yohou.base` | `_transform()`, `get_feature_names_out()` |

Optional overrides: `_fit()` (default no-op), `_inverse_transform()` (required only for invertible transformers).

### Splitters

| Base Class | Import | Abstract Methods |
|-----------|--------|-----------------|
| [`BaseSplitter`](/pages/api/generated/yohou.model_selection.split.BaseSplitter/) | `yohou.model_selection` | `split()`, `_iter_test_indices()`, `get_n_splits()` |

### Search Strategies

| Base Class | Import | Abstract Methods |
|-----------|--------|-----------------|
| [`BaseSearchCV`](/pages/api/generated/yohou.model_selection.search.BaseSearchCV/) | `yohou.model_selection.search` | `_run_search()` |

Built-in implementations: [`GridSearchCV`](/pages/api/generated/yohou.model_selection.search.GridSearchCV/), [`RandomizedSearchCV`](/pages/api/generated/yohou.model_selection.search.RandomizedSearchCV/). Extend [`BaseSearchCV`](/pages/api/generated/yohou.model_selection.search.BaseSearchCV/) only for custom search strategies (e.g., Bayesian optimization).

## See Also

- [Tags](tags.md): tag system for declaring component capabilities
- [Data Catalog](data-catalog.md): bundled datasets for testing and examples
