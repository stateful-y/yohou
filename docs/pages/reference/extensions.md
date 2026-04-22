# Extensions

Like Scikit-Learn, Yohou is designed to be extended. Whether it is to develop new models or integrate with other libraries, extensions allow you to enhance Yohou's capabilities. Below are the current official and community extensions.

## Official extensions

### Yohou-Optuna

Hyperparameter optimization via [Optuna](https://optuna.org/).

- **Package**: `Yohou-Optuna`
- **Install**: `uv add yohou-optuna`
- **Source**: [`stateful-y/yohou-optuna`](https://github.com/stateful-y/yohou/tree/main/packages/yohou-optuna)

## Yohou-Nixtla

Integration with [Nixtla](https://nixtla.io/)'s forecasting libraries.

- **Package**: `Yohou-Nixtla`
- **Install**: `uv add yohou-nixtla`
- **Source**: [`stateful-y/yohou-nixtla`](https://github.com/stateful-y/yohou/tree/main/packages/yohou-nixtla)

## Community extensions

There are no community extensions at this time. Want to add yours to this list? You can open an [issue](https://github.com/stateful-y/yohou/issues/new).

## Built-in Extension Points

Yohou provides abstract base classes for creating custom components that integrate
seamlessly with the rest of the framework:

- **Forecasters**: `BasePointForecaster`, `BaseIntervalForecaster`,
  `BaseClassProbaForecaster` for custom forecasting algorithms
  (import from `yohou.base`)
- **Metrics**: `BasePointScorer`, `BaseIntervalScorer`, `BaseClassProbaScorer`
  for custom evaluation metrics
  (import from `yohou.metrics`)
- **Transformers**: `BaseTransformer` for custom preprocessing steps
  (import from `yohou.preprocessing`)
- **Splitters**: `BaseSplitter` for custom cross-validation strategies
  (import from `yohou.model_selection`)
- **Ensembles**: `VotingPointForecaster`, `VotingIntervalForecaster`,
  `VotingClassProbaForecaster` for composition patterns
  (import from `yohou.ensemble`)

See [How to Create Custom Estimators](../how-to/custom-estimators.md) for
step-by-step instructions.
