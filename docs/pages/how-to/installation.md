# How to Install Yohou

This guide shows you how to install Yohou and its optional extras.

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

## Plotting Extra

For Plotly-based visualization, install the plotting optional extra:

=== "uv"

    ```bash
    uv add yohou[plotting]
    ```

=== "pip"

    ```bash
    pip install yohou[plotting]
    ```

Core Yohou works without plotly installed. The plotting extra adds `plotly` and
`plotly-resampler` for interactive charts.

!!! note
    `plotly-resampler` depends on `tsdownsample`, which does not yet ship
    Python 3.14 wheels. On Python 3.14, install without resampler support or
    disable it at runtime:

    ```python
    from yohou.plotting import set_config
    set_config(resampler=False)
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

[`yohou-optuna`](https://pypi.org/project/yohou-optuna/) adds `OptunaSearchCV` for Bayesian hyperparameter optimization. [`yohou-nixtla`](https://pypi.org/project/yohou-nixtla/) wraps the Nixtla `statsforecast`, `mlforecast`, and `neuralforecast` model families into Yohou's API. See [Extensions](../reference/extensions.md) for installation and usage details.

## Python Version Support

Yohou supports Python 3.11, 3.12, 3.13, and 3.14.

## See Also

With Yohou installed, head to [Getting Started](../tutorials/getting-started.md) to load a dataset and fit your first forecaster.
