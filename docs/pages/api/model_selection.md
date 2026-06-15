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

### Cross-validation

| Name | Description |
| --- | --- |
| [`cross_validate`](generated/yohou.model_selection.validation.cross_validate.md) | Evaluate a forecaster across cross-validation splits, returning multiple scores. |
| [`cross_val_score`](generated/yohou.model_selection.validation.cross_val_score.md) | Evaluate a forecaster across cross-validation splits with a single scorer. |
| [`cross_val_predict`](generated/yohou.model_selection.validation.cross_val_predict.md) | Generate cross-validated predictions for each split. |

### Utilities

| Name | Description |
| --- | --- |
| [`train_test_split`](generated/yohou.model_selection.split.train_test_split.md) | Split a time series into contiguous train and test sets. |
| [`check_cv_alignment`](generated/yohou.model_selection.split.check_cv_alignment.md) | Validate that a cross-validator's splits align with the data. |
