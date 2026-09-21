"""Model selection tools including cross-validation and hyperparameter search."""

from .early_stopping import (
    BaseEarlyStoppingAdapter,
    CatBoostEarlyStoppingAdapter,
    HistGradientBoostingEarlyStoppingAdapter,
    LightGBMEarlyStoppingAdapter,
    XGBoostEarlyStoppingAdapter,
)
from .search import (
    BaseSearchCV,
    GridSearchCV,
    RandomizedSearchCV,
)
from .split import (
    BaseSplitter,
    ExpandingWindowSplitter,
    SlidingWindowSplitter,
    check_cv,
    check_cv_alignment,
    train_test_split,
)
from .validation import (
    cross_val_predict,
    cross_val_score,
    cross_validate,
)

__all__ = [
    "BaseEarlyStoppingAdapter",
    "BaseSearchCV",
    "BaseSplitter",
    "CatBoostEarlyStoppingAdapter",
    "ExpandingWindowSplitter",
    "GridSearchCV",
    "HistGradientBoostingEarlyStoppingAdapter",
    "LightGBMEarlyStoppingAdapter",
    "RandomizedSearchCV",
    "SlidingWindowSplitter",
    "XGBoostEarlyStoppingAdapter",
    "check_cv",
    "check_cv_alignment",
    "cross_val_predict",
    "cross_val_score",
    "cross_validate",
    "train_test_split",
]
