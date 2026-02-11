# Integrations

!!! note "Under Development"
    This section is under active development. Check back soon for detailed integration guides.

Yohou provides workspace packages that integrate with third-party forecasting ecosystems.

## Yohou-Optuna

Hyperparameter optimization via [Optuna](https://optuna.org/).

- **Package**: `yohou-optuna`
- **Install**: `uv add yohou-optuna`
- **Source**: [`packages/yohou-optuna/`](https://github.com/stateful-y/yohou/tree/main/packages/yohou-optuna)

## Yohou-Nixtla

Integration with [Nixtla](https://nixtla.io/)'s forecasting libraries.

- **Package**: `yohou-nixtla`
- **Install**: `uv add yohou-nixtla`
- **Source**: [`packages/yohou-nixtla/`](https://github.com/stateful-y/yohou/tree/main/packages/yohou-nixtla)
- **Libraries**: statsforecast, mlforecast, neuralforecast

## Related Projects

| Project | Description |
|---------|-------------|
| [scikit-learn](https://scikit-learn.org/) | Machine learning library (core dependency) |
| [polars](https://pola.rs/) | Fast DataFrame library (core dependency) |
| [marimo](https://marimo.io/) | Reactive notebook system (used for examples) |
