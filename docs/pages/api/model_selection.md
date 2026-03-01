---
template: api-submodule.html
---

# yohou.model_selection

Time series cross-validation splitters and hyperparameter search.

### Search estimators

| Name | Description |
| --- | --- |
| [`BaseSearchCV`](generated/yohou.model_selection.search.BaseSearchCV.md) | Abstract base class for hyperparameter search with time series cross-validation. |
| [`GridSearchCV`](generated/yohou.model_selection.search.GridSearchCV.md) | Exhaustive search over specified parameter values for a forecaster. |
| [`RandomizedSearchCV`](generated/yohou.model_selection.search.RandomizedSearchCV.md) | Randomized search on hyperparameters. |

### Splitters

| Name | Description |
| --- | --- |
| [`BaseSplitter`](generated/yohou.model_selection.split.BaseSplitter.md) | Base class for yohou time series cross-validation splitters. |
| [`ExpandingWindowSplitter`](generated/yohou.model_selection.split.ExpandingWindowSplitter.md) | Expanding window time series cross-validation splitter. |
| [`SlidingWindowSplitter`](generated/yohou.model_selection.split.SlidingWindowSplitter.md) | Sliding window time series cross-validation splitter. |

### Utilities

| Name | Description |
| --- | --- |
| [`check_cv`](generated/yohou.model_selection.split.check_cv.md) | Input checker utility for building a cross-validator. |
