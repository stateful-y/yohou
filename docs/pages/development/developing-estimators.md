# Developing Custom Estimators

!!! note "Under Development"
    This guide is under active development. Check back soon for comprehensive instructions on building custom forecasters, transformers, scorers, and splitters.

## Overview

Yohou's estimator API extends Scikit-Learn's conventions with time series-specific operations. This guide covers how to implement custom components that integrate seamlessly with Yohou's ecosystem.

## Creating a Custom Forecaster

Coming soon.

**API Reference**: See [yohou.base](../api/base.md) for base classes and [yohou.testing](../api/testing.md) for systematic checks.

## Creating a Custom Transformer

Coming soon.

**API Reference**: See [yohou.base.BaseTransformer](../api/base.md) and [yohou.testing](../api/testing.md) for systematic checks.

## Creating a Custom Scorer

Coming soon.

**API Reference**: See [yohou.metrics](../api/metrics.md) for base classes and [yohou.testing](../api/testing.md) for systematic checks.

## Creating a Custom Splitter

Coming soon.

**API Reference**: See [yohou.model_selection](../api/model_selection.md) for base classes and [yohou.testing](../api/testing.md) for systematic checks.

## Testing Your Estimator

Yohou provides systematic check generators that validate API conformance automatically:

```python
from yohou.testing import _yield_yohou_forecaster_checks

for check_name, check_func, check_kwargs in _yield_yohou_forecaster_checks(
    forecaster_fitted, y_train, X_train, y_test, X_test,
    tags={"forecaster_type": frozenset({"point"}), "uses_reduction": False}
):
    check_func(forecaster_fitted, **check_kwargs)
```

See the [Testing API Reference](../api/testing.md) for the full list of available checks.
