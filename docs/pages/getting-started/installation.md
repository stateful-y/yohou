# Installation

## Install Yohou

=== "uv (recommended)"

    ```bash
    # Add to an existing project
    uv add yohou

    # Or install globally
    uv tool install yohou
    ```

=== "pip"

    ```bash
    pip install yohou
    ```

=== "conda"

    ```bash
    conda install -c conda-forge yohou
    ```

=== "mamba"

    ```bash
    mamba install -c conda-forge yohou
    ```

## Verify Installation

```python
import yohou
print(yohou.__version__)
```

## Development Setup

To contribute to Yohou or install from source:

=== "just"

    ```bash
    git clone https://github.com/stateful-y/yohou.git
    cd yohou
    just install  # Installs dev dependencies + pre-commit hooks
    ```

=== "uv"

    ```bash
    git clone https://github.com/stateful-y/yohou.git
    cd yohou
    uv sync --group dev
    uv run pre-commit install
    ```

## Optional Packages

### Yohou-Optuna

Hyperparameter tuning with [Optuna](https://optuna.org/). Provides `OptunaSearchCV` for Bayesian optimization of forecaster hyperparameters as an alternative to `GridSearchCV` and `RandomizedSearchCV`.

=== "uv"

    ```bash
    uv add yohou-optuna
    ```

=== "pip"

    ```bash
    pip install yohou-optuna
    ```

See [Extensions](../extensions/index.md) for usage details.

### Yohou-Nixtla

Statistical, ML, and neural forecasting with [Nixtla](https://nixtla.io/). Wraps `statsforecast`, `mlforecast`, and `neuralforecast` models into Yohou's sklearn-compatible API.

=== "uv"

    ```bash
    uv add yohou-nixtla
    ```

=== "pip"

    ```bash
    pip install yohou-nixtla
    ```

See [Extensions](../extensions/index.md) for usage details.

## Python Version Support

Yohou supports Python 3.11, 3.12, 3.13, and 3.14.
