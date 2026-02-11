---
description: "Guide for extending hyperparameter search (GridSearchCV, RandomizedSearchCV) in Yohou. Less common than other components — most users will use built-in classes."
---

# Creating New Hyperparameter Search Classes

**Note**: This is an advanced use case. Most users should use the built-in `GridSearchCV` and `RandomizedSearchCV` classes, which are fully compatible with yohou forecasters.

---

## When to Extend Search Classes

Consider extending only if:
- You need custom sampling strategies (e.g., Bayesian optimization, genetic algorithms)
- You want to add time series-specific early stopping or pruning
- You're implementing ensemble-based hyperparameter tuning

**For most use cases**, use built-in classes with custom scorers or splitters instead.

---

## Minimal Search Template (Custom Sampler)

Yohou's search classes extend sklearn's `BaseSearchCV`. If you need a custom search strategy, follow this pattern:

```python
"""Module docstring."""

from sklearn.model_selection._search import BaseSearchCV
from sklearn.base import clone
import numpy as np

from yohou.model_selection import ExpandingWindowSplitter
from yohou.metrics import MeanAbsoluteError


class MyCustomSearch(BaseSearchCV):
    """Custom hyperparameter search for time series forecasters.

    Parameters
    ----------
    forecaster : BaseForecaster
        Yohou forecaster to tune.
    param_space : dict
        Parameter space (custom format for your sampler).
    scoring : BaseScorer
        Yohou scorer for evaluation.
    cv : BaseSplitter, default=None
        Time series cross-validator. If None, uses ExpandingWindowSplitter(n_splits=3).
    n_jobs : int, default=None
        Number of parallel jobs.

    Examples
    --------
    >>> from yohou.point_forecaster import PointReductionForecaster
    >>> from yohou.metrics import MeanAbsoluteError
    >>> search = MyCustomSearch(
    ...     forecaster=PointReductionForecaster(),
    ...     param_space={"estimator__alpha": [0.1, 1.0, 10.0]},
    ...     scoring=MeanAbsoluteError(),
    ... )
    """

    def __init__(
        self,
        forecaster,
        param_space,
        scoring,
        cv=None,
        n_jobs=None,
    ):
        self.forecaster = forecaster
        self.param_space = param_space
        self.scoring = scoring
        self.cv = cv
        self.n_jobs = n_jobs

    def fit(self, y, X=None, forecasting_horizon=1, **params):
        """Run hyperparameter search.

        Parameters
        ----------
        y : pl.DataFrame
            Target time series.
        X : pl.DataFrame, optional
            Exogenous features.
        forecasting_horizon : int, default=1
            Forecasting horizon.
        **params : dict
            Metadata routing.

        Returns
        -------
        self
            Fitted search object.

        """
        # Use default CV if not provided
        if self.cv is None:
            cv = ExpandingWindowSplitter(n_splits=3)
        else:
            cv = self.cv

        # Generate candidate parameters (custom logic)
        candidates = self._sample_candidates(self.param_space)

        # Evaluate each candidate via cross-validation
        results = []
        for params_dict in candidates:
            forecaster = clone(self.forecaster)
            forecaster.set_params(**params_dict)

            # Cross-validate
            scores = []
            for train_idx, test_idx in cv.split(y, X):
                y_train, y_test = y[train_idx], y[test_idx]
                X_train = X[train_idx] if X is not None else None
                X_test = X[test_idx] if X is not None else None

                forecaster.fit(y_train, X_train, forecasting_horizon=forecasting_horizon)
                y_pred = forecaster.predict(forecasting_horizon=forecasting_horizon, X=X_test)

                score = self.scoring.score(y_test, y_pred)
                scores.append(score)

            mean_score = np.mean(scores)
            results.append({"params": params_dict, "score": mean_score, "scores": scores})

        # Select best parameters
        best_idx = self._select_best(results)
        self.best_params_ = results[best_idx]["params"]
        self.best_score_ = results[best_idx]["score"]
        self.cv_results_ = results

        # Refit on full data
        self.best_forecaster_ = clone(self.forecaster)
        self.best_forecaster_.set_params(**self.best_params_)
        self.best_forecaster_.fit(y, X, forecasting_horizon=forecasting_horizon)

        return self

    def _sample_candidates(self, param_space):
        """Generate candidate parameter sets (custom logic).

        Parameters
        ----------
        param_space : dict
            Parameter space definition.

        Returns
        -------
        list of dict
            Candidate parameter dictionaries.

        """
        # Example: Random sampling
        candidates = []
        for _ in range(10):  # Sample 10 candidates
            candidate = {}
            for param_name, param_range in param_space.items():
                candidate[param_name] = np.random.choice(param_range)
            candidates.append(candidate)
        return candidates

    def _select_best(self, results):
        """Select best candidate based on scoring.

        Parameters
        ----------
        results : list of dict
            Results from cross-validation.

        Returns
        -------
        int
            Index of best candidate.

        """
        # Lower is better for most yohou metrics
        if self.scoring.lower_is_better:
            return int(np.argmin([r["score"] for r in results]))
        else:
            return int(np.argmax([r["score"] for r in results]))

    def predict(self, forecasting_horizon=None, X=None, **params):
        """Generate forecasts using best forecaster.

        Parameters
        ----------
        forecasting_horizon : int, optional
            Forecasting horizon. If None, uses value from fit().
        X : pl.DataFrame, optional
            Exogenous features.
        **params : dict
            Metadata routing.

        Returns
        -------
        pl.DataFrame
            Predictions.

        """
        return self.best_forecaster_.predict(
            forecasting_horizon=forecasting_horizon, X=X, **params
        )
```

---

## Built-In Search Classes

**Use these instead of custom implementations** unless you have specific requirements:

### GridSearchCV

Exhaustive search over parameter grid:

```python
from yohou.model_selection import GridSearchCV
from yohou.point_forecaster import PointReductionForecaster
from yohou.metrics import MeanAbsoluteError

search = GridSearchCV(
    forecaster=PointReductionForecaster(),
    param_grid={
        "estimator__alpha": [0.1, 1.0, 10.0],
        "estimator__fit_intercept": [True, False],
    },
    scoring=MeanAbsoluteError(),
    cv=ExpandingWindowSplitter(n_splits=5),
)
search.fit(y, X, forecasting_horizon=3)
print(search.best_params_)
```

### RandomizedSearchCV

Random sampling from parameter distributions:

```python
from yohou.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint

search = RandomizedSearchCV(
    forecaster=PointReductionForecaster(),
    param_distributions={
        "estimator__alpha": uniform(0.01, 10.0),
        "estimator__max_iter": randint(100, 1000),
    },
    n_iter=20,  # Number of random samples
    scoring=MeanAbsoluteError(),
)
search.fit(y, X, forecasting_horizon=3)
```

---

## Common Pitfalls

- **Not using time series CV**: Always use `ExpandingWindowSplitter` or `SlidingWindowSplitter`, NOT `KFold`
- **Refitting on wrong data**: Refit best model on full training data, not last CV fold
- **Ignoring scorer direction**: Check `scoring.lower_is_better` to determine best score
- **Metadata routing issues**: Pass `**params` through to forecaster methods
- **Parallel execution complexity**: Time series CV is sequential by nature — parallelism limited to trials

---

## Testing Patterns

Yohou provides systematic check functions for search:

```python
from yohou.testing import _yield_yohou_search_checks

for check_name, check_func, check_kwargs in _yield_yohou_search_checks(
    search_fitted, y_train, X_train, y_test, X_test
):
    check_func(search_fitted, **check_kwargs)
```

**See**: [Search Testing Infrastructure](.github/prompts/search-testing-infrastructure.prompt.md) for all 19 checks.

---

## Real-World Examples to Study

**Built-in search classes**:
- `src/yohou/model_selection/search.py`:
  - `GridSearchCV` - Exhaustive grid search
  - `RandomizedSearchCV` - Random sampling

**Testing**:
- `tests/model_selection/test_search.py` - Search tests
- `src/yohou/testing/search.py` - Check functions (19 checks)

**Usage examples**:
- `examples/scorer_tutorial.py` - Hyperparameter tuning with scorers
- Test files in `tests/model_selection/` - Various search scenarios

---

## Recommendation

**For 99% of use cases**: Use `GridSearchCV` or `RandomizedSearchCV` with:
- Custom `scoring` (any yohou metric)
- Custom `cv` (ExpandingWindowSplitter, SlidingWindowSplitter, or custom splitter)
- Custom `param_grid`/`param_distributions`

**Only extend if**: You're implementing novel search strategies (e.g., Bayesian optimization, SMAC, Optuna integration).
