<p align="center">
  <picture>
    <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/gtauzin/yohou/main/docs/images/logo_light.png">
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/gtauzin/yohou/main/docs/images/logo_dark.png">
    <img src="https://raw.githubusercontent.com/gtauzin/yohou/main/docs/images/logo_light.png" alt="yohou">
  </picture>
</p>

[![Python Version](https://img.shields.io/pypi/pyversions/yohou)](https://pypi.org/project/yohou/)
[![License](https://img.shields.io/github/license/gtauzin/yohou)](https://github.com/gtauzin/yohou/blob/main/LICENSE)
[![PyPI Version](https://img.shields.io/pypi/v/yohou)](https://pypi.org/project/yohou/)

## What is yohou?

**yohou** is a Python framework for time series forecasting that seamlessly combines the power of [scikit-learn](https://scikit-learn.org) with the performance of [polars](https://pola.rs/). Built on scikit-learn's API, yohou extends it with time series-specific operations while maintaining full compatibility with the sklearn ecosystem.

## What are the features of yohou?

- **Scikit-learn compatible:** Built on the latest scikit-learn API, allowing the use of any tabular regressor for forecasting with full metadata routing and feature name support.
- **Extended API:** Adds `update`, `reset`, and `update_predict` methods to sklearn's standard `fit`/`transform`/`predict` interface for incremental learning: fit/forecast_point/forecast_interval/roll/revert/roll_forecast_point/roll_forecast_interval
- **Multi-DataFrame support:** Works with polars, pandas, and more via [Narwhals](https://narwhals-dev.github.io/narwhals/).
- **Multiple reduction strategies:** Supports Recursive, Direct, Multi-output, and DirRec forecasting approaches.
- **Panel data support:** Enables both local and global forecasting for panel (hierarchical) time series data.
- **Flexible forecasting horizons:** Train on one horizon and predict on another by applying the model recursively.
- **Exogenous features:** Distinguishes between ex-ante (known in advance) and ex-post (observed after) features.
- **Point and interval forecasting:** Native support for both point predictions and prediction intervals.
- **Conformal prediction:** Statistical guarantees for prediction intervals via conformal prediction methods.
- **Hierarchical forecasting:** Reconciliation methods for coherent hierarchical forecasts.
- **Ensembling**
- **Divine intervention**
- **Hyperparameter optimization:** Optuna-based cross-validation and nested cross-validation for time series.
- **Time series metrics:** Specialized scoring functions designed for temporal data.
- **Visualization tools:** Plotly-based interactive visualizations for EDA, model tuning, evaluation, uncertainty quantification, and comparison.

**Any feature missing on this list?** Search our [issue tracker](https://github.com/gtauzin/yohou/issues) to see if someone has already requested it and add a comment to it explaining your use-case. Otherwise, please open a new issue describing the requested feature and possible use-case scenario. We prioritize our roadmap based on user feedback, so we would love to hear from you.

## How to install yohou?

Install yohou and its dependencies using `pip`:

```bash
pip install yohou
```

or using `uv`:

```bash
uv pip install yohou
```

or using `conda` (coming soon):

```bash
conda install -c conda-forge yohou
```

or alternatively, add `yohou` to your `requirements.txt` or `pyproject.toml` file.

## How to get started with yohou?

Here's a quick example to get you started:

```python
import polars as pl
from yohou.point_forecaster import PointReductionForecaster
from sklearn.linear_model import Ridge

# Prepare your time series data - should include a "time" column with datetime values
y = pl.DataFrame({
    "time": pl.datetime_range(start="2020-01-01", end="2020-12-31", interval="1d"),
    "sales": [...]  # Your time series data
})

# Create and fit a forecaster
forecaster = PointReductionForecaster(
    estimator=Ridge(),
    forecasting_horizon=7
)
forecaster.fit(y=y, forecasting_horizon=7)

# Make predictions
y_pred = forecaster.predict(forecasting_horizon=7)

# Update with new observations and predict again
y_new = pl.DataFrame({...})  # New observations
y_pred = forecaster.update_predict(y=y_new, forecasting_horizon=7)
```

## Where can I learn more?

Full documentation is available at [https://yohou.readthedocs.io/](https://yohou.readthedocs.io/) (coming soon).

For questions and discussions, please open a [discussion](https://github.com/gtauzin/yohou/discussions).

## Can I contribute?

We welcome contributions, feedback, and questions:

- **Report issues or request features:** [GitHub Issues](https://github.com/gtauzin/yohou/issues)
- **Join the discussion:** [GitHub Discussions](https://github.com/gtauzin/yohou/discussions)
- **Contributing Guide:** [CONTRIBUTING.md](https://github.com/gtauzin/yohou/blob/main/CONTRIBUTING.md)

## License

This project is licensed under the terms of the [BSD License](https://github.com/gtauzin/yohou/blob/main/LICENSE).

## Acknowledgements

This project is developed and maintained by [Guillaume Tauzin](https://github.com/gtauzin).
