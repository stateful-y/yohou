# Installation

## Using uv (Recommended)

```bash
# Add to an existing project
uv add yohou

# Or install globally
uv tool install yohou
```

## Using pip

```bash
pip install yohou
```

## Verify Installation

```python
import yohou
print(yohou.__version__)
```

## Development Setup

To contribute to Yohou or install from source:

```bash
git clone https://github.com/stateful-y/yohou.git
cd yohou
just install  # Installs dev dependencies + pre-commit hooks
```

Or manually:

```bash
uv sync --group dev
uv run pre-commit install
```

## Optional Packages

!!! info "Under Development"
    Detailed installation and usage guides for optional integrations are coming soon.

### yohou-optuna

Hyperparameter tuning with [Optuna](https://optuna.org/):

```bash
uv add yohou-optuna
```

### yohou-nixtla

Statistical, ML, and neural forecasting with [Nixtla](https://nixtla.io/):

```bash
uv add yohou-nixtla
```

## Python Version Support

Yohou supports Python 3.11, 3.12, 3.13, and 3.14.
