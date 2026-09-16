---
template: api-submodule.html
---

# yohou.model_selection

Model selection tools including cross-validation and hyperparameter search.

### Classes

| Name | Description |
|------|-------------|
| [`BaseEarlyStoppingAdapter`](generated/yohou.model_selection.BaseEarlyStoppingAdapter.md) | Base class for translating early stopping between a search and a boosting library. |
| [`CatBoostAdapter`](generated/yohou.model_selection.CatBoostAdapter.md) | Early-stopping adapter for CatBoost regressors and classifiers. |
| [`LightGBMAdapter`](generated/yohou.model_selection.LightGBMAdapter.md) | Early-stopping adapter for LightGBM's scikit-learn estimators. |
| [`XGBoostAdapter`](generated/yohou.model_selection.XGBoostAdapter.md) | Early-stopping adapter for XGBoost's scikit-learn estimators. |
| [`BaseSearchCV`](generated/yohou.model_selection.BaseSearchCV.md) | Abstract base class for hyperparameter search with cross-validation. |
| [`GridSearchCV`](generated/yohou.model_selection.GridSearchCV.md) | Exhaustive search over specified parameter values for a forecaster. |
| [`RandomizedSearchCV`](generated/yohou.model_selection.RandomizedSearchCV.md) | Randomized search on hyperparameters. |
| [`BaseSplitter`](generated/yohou.model_selection.BaseSplitter.md) | Base class for yohou time series cross-validation splitters. |
| [`ExpandingWindowSplitter`](generated/yohou.model_selection.ExpandingWindowSplitter.md) | Expanding window time series cross-validation splitter. |
| [`SlidingWindowSplitter`](generated/yohou.model_selection.SlidingWindowSplitter.md) | Sliding window time series cross-validation splitter. |

### Functions

| Name | Description |
|------|-------------|
| [`check_cv`](generated/yohou.model_selection.check_cv.md) | Input checker utility for building a cross-validator. |
| [`check_cv_alignment`](generated/yohou.model_selection.check_cv_alignment.md) | Inspect how a CV splitter's test windows align with a forecasting horizon. |
| [`train_test_split`](generated/yohou.model_selection.train_test_split.md) | Split time series data into temporal train and test sets. |
| [`cross_val_predict`](generated/yohou.model_selection.cross_val_predict.md) | Generate cross-validated predictions for each fold. |
| [`cross_val_score`](generated/yohou.model_selection.cross_val_score.md) | Evaluate a forecaster by cross-validation and return test scores. |
| [`cross_validate`](generated/yohou.model_selection.cross_validate.md) | Evaluate a forecaster by cross-validation and return test scores and timings. |
