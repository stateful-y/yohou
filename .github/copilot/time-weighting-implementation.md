# Plan: Add Time-Based Weighting to Scorers and Reduction Forecasters

**TL;DR**: Implement `time_weight` parameter in both scorers (for evaluation weighting) and reduction forecasters (for training sample weighting). Scorers use `time_weight` to weight prediction errors during aggregation. Reduction forecasters convert `time_weight` to `sample_weight` arrays for sklearn estimators during fit. Both leverage existing metadata routing infrastructure.

---

## Part 1: Scorer Time Weighting (Evaluation-Time)

### Comprehensive Weighting Approaches in Forecasting

#### 1. **Explicit Weight Series** (DataFrame)
- User provides `pl.DataFrame` with `"time"` and `"weight"` columns
- Direct control, easy validation
- Use case: Custom business logic (weight specific events/promotions)

#### 2. **Callable/Function-Based**
- **Exponential Decay**: Recent predictions weighted more (e.g., `λ^t` decay)
- **Linear Decay**: Uniform decrease over time
- **Step Functions**: Binary emphasis (weight recent N periods at 1.0, rest at 0.1)
- **Forecast Horizon Weighting**: Near-term predictions > far-term (common in supply chain)

#### 3. **Automatic Seasonal Weighting**
- **Pattern Detection**: Analyze historical errors to identify high-error periods
- **Calendar-Based**: Weight business days vs weekends, holidays, end-of-month
- **Seasonal Phases**: Emphasize/de-emphasize specific cycles (e.g., Q4 in retail)
- **Fourier-Informed**: Use harmonic components to weight seasonal peaks/troughs

#### 4. **Adaptive/Learning-Based**
- **Error-Driven**: Weight timesteps inversely proportional to past error variance
- **Residual-Based**: Higher weights where model previously underperformed
- **Cross-Validation History**: Use CV fold scores to determine timestep importance

#### 5. **Hybrid Approaches**
- Combine multiple strategies: `w_final = w_recency × w_seasonal × w_business_rule`
- Compositional pattern: `compose_weights([decay_fn, seasonal_fn])`

### Implementation Steps

#### 1. Modify Base Scorer Implementation

**File**: `src/yohou/metrics/base.py`

**Changes**:
- Add `time_weight` parameter to `score()` method signature (line ~195-210)
- Update `_aggregate_scores()` to apply normalized weights during final aggregation
  - Scalar aggregation: Line ~379 (both timewise and componentwise)
  - Componentwise aggregation: Line ~387 (per-timestep scores)
- Weight normalization: Ensure weights sum to 1.0 before application
- Support both DataFrame and callable formats

**Example signature**:
```python
def score(
    self,
    y_truth: pl.DataFrame,
    y_pred: pl.DataFrame,
    time_weight: pl.DataFrame | Callable[[pl.Series], pl.Series] | None = None,
    **params
) -> float | pl.DataFrame:
    """Compute weighted score.

    Parameters
    ----------
    time_weight : pl.DataFrame, Callable, or None
        - pl.DataFrame: Must have "time" and "weight" columns
        - Callable: Function f(time: pl.Series) -> pl.Series of weights
        - None: Equal weighting (default)
    """
```

**Weight processing logic**:
```python
def _process_time_weights(
    self,
    time_weight: pl.DataFrame | Callable | None
) -> np.ndarray:
    """Convert time_weight to normalized array aligned with self._time_values_."""
    if time_weight is None:
        return np.ones(len(self._time_values_)) / len(self._time_values_)

    if callable(time_weight):
        weight_series = time_weight(pl.Series("time", self._time_values_))
    else:  # DataFrame
        # Join on time column, fill missing with 1.0
        weight_df = pl.DataFrame({"time": self._time_values_})
        weight_df = weight_df.join(time_weight, on="time", how="left")
        weight_series = weight_df["weight"].fill_null(1.0)

    weights = weight_series.to_numpy()

    # Normalize to sum to 1
    if not np.isclose(weights.sum(), 0.0):
        weights = weights / weights.sum()
    else:
        weights = np.ones_like(weights) / len(weights)

    return weights
```

#### 2. Create Weighting Utilities Module

**New file**: `src/yohou/utils/weighting.py`

(Full implementation provided in original plan - includes exponential_decay_weight, linear_decay_weight, seasonal_emphasis_weight, forecast_horizon_weight, step_weight, compose_weights)

#### 3. Update Model Selection Infrastructure

**File**: `src/yohou/model_selection/split.py`

**Changes**:
- Add optional `time_weight_fn` parameter to `ExpandingWindowSplitter` and `SlidingWindowSplitter`
- Generate time-aligned weights for test splits
- Return weights alongside indices (optional `return_time_weights=True`)

#### 4-5. Testing and Documentation

(Full testing and documentation sections from original plan)

---

## Part 2: Reduction Forecaster Time Weighting (Training-Time)

### Overview

**Motivation**: Weight training samples by time to give more importance to recent observations or specific temporal patterns during model fitting.

**Key Difference from Scorer Weighting**:
- **Scorer weighting**: Applied during evaluation (prediction error aggregation)
- **Forecaster weighting**: Applied during training (sklearn estimator `fit()` receives `sample_weight`)

**Current Status**:
- ✅ `time_weight` parameter already declared in `PointReductionForecaster.fit()` (line 113 in `src/yohou/point_forecaster/reduction.py`)
- ✅ `_estimator_fit_one()` accepts `time_weight` parameter (line 1380 in `src/yohou/base.py`)
- ❌ Conversion to `sample_weight` array **NOT implemented** (noted in docstring: "Converted to sample_weight during tabularization")

### Implementation Approach

#### 1. Understanding Tabularization

**Current Flow** (`BaseReductionForecaster._get_tabularized_dataset`):

```python
# Input time series (before tabularization)
y_t = pl.DataFrame({
    "time": [t1, t2, t3, t4, t5, t6, t7, t8],
    "value": [10, 20, 30, 40, 50, 60, 70, 80]
})

# After tabularization with forecasting_horizon=3
# Creates lag features: lag_1, lag_2, lag_3 (for 3-step-ahead prediction)
# Drops last 3 rows (no targets available) and first max(lags) rows (no features)

X_tab (features):
  Row 0: lag features from t1-t3 → predicts [t2, t3, t4]
  Row 1: lag features from t2-t4 → predicts [t3, t4, t5]
  Row 2: lag features from t3-t5 → predicts [t4, t5, t6]
  ... (5 rows total)

y_tab (targets):
  Row 0: [value@t2, value@t3, value@t4]  # step_1, step_2, step_3
  Row 1: [value@t3, value@t4, value@t5]
  Row 2: [value@t4, value@t5, value@t6]
```

**Critical Insight**: After tabularization, rows no longer correspond 1:1 with original time series rows. Each tabularized row represents a **training sample** (feature-target pair) centered at a specific time.

#### 2. time_weight to sample_weight Conversion Logic

**Requirements**:
1. Convert `time_weight` (aligned with original `y_t` times) to `sample_weight` (aligned with tabularized rows)
2. Handle both global and panel data cases
3. Support DataFrame and callable formats
4. Normalize weights for sklearn estimators

**Proposed Implementation** (modify `_estimator_fit_one`):

```python
def _estimator_fit_one(
    self,
    y_t: pl.DataFrame,
    X_t: pl.DataFrame,
    forecasting_horizon: StrictInt,
    time_weight: Callable | pl.DataFrame | None = None,
    estimator_params: dict[str, Any] | None = None,
    estimator_fit_params: dict[str, Any] | None = None,
) -> BaseEstimator:
    """Fit an sklearn estimator on tabularized time series data."""
    estimator = clone(self.estimator).set_params(**(estimator_params or {}))

    if self.panel_group_names_ is None:
        # Global time series
        X_tab, y_tab = self._get_tabularized_dataset(y_t, X_t, forecasting_horizon)

        # Convert time_weight to sample_weight
        sample_weight = self._process_time_weight_to_sample_weight(
            time_weight, y_t, forecasting_horizon
        )
    else:
        # Panel data: stack all series
        X_tab_list, y_tab_list, sample_weight_list = [], [], []

        for panel_group_name in self.panel_group_names_:
            y_t_local = y_t[panel_group_name]
            X_t_local = X_t[panel_group_name]
            y_columns = [c for c in y_t_local.columns if c != "time"]

            X_tab_local, y_tab_local = self._get_tabularized_dataset(
                y_t_local, X_t_local, forecasting_horizon, y_columns=y_columns
            )

            # Per-group time_weight (if dict) or global
            group_time_weight = time_weight.get(panel_group_name) if isinstance(time_weight, dict) else time_weight
            sample_weight_local = self._process_time_weight_to_sample_weight(
                group_time_weight, y_t_local, forecasting_horizon
            )

            X_tab_list.append(X_tab_local)
            y_tab_list.append(y_tab_local)
            sample_weight_list.append(sample_weight_local)

        X_tab = np.vstack(X_tab_list)
        y_tab = np.vstack(y_tab_list)
        sample_weight = np.concatenate(sample_weight_list) if sample_weight_list[0] is not None else None

    # Fit with sample_weight if provided
    fit_params = estimator_fit_params or {}
    if sample_weight is not None:
        fit_params["sample_weight"] = sample_weight

    estimator.fit(X_tab, y_tab, **fit_params)

    return estimator
```

#### 3. Core Conversion Helper Method

**New method in `BaseReductionForecaster`**:

```python
def _process_time_weight_to_sample_weight(
    self,
    time_weight: Callable | pl.DataFrame | None,
    y_t: pl.DataFrame,
    forecasting_horizon: int
) -> np.ndarray | None:
    """Convert time_weight to sample_weight array aligned with tabularized data.

    Parameters
    ----------
    time_weight : Callable, pl.DataFrame, or None
        Time-based weighting function or DataFrame.
    y_t : pl.DataFrame
        Transformed target time series (before tabularization).
    forecasting_horizon : int
        Number of forecast steps.

    Returns
    -------
    np.ndarray or None
        Sample weights aligned with tabularized rows (X_tab, y_tab).
        None if time_weight is None.

    Notes
    -----
    Mapping strategy:
    - Each tabularized row corresponds to a specific "center time" in the original series
    - For forecasting_horizon=H, row i uses features from time[i:i+H] and predicts time[i+1:i+H+1]
    - Center time is defined as time[i + H // 2] (midpoint of prediction window)
    - Weight for row i = time_weight at center time

    Example:
    - y_t.time = [t1, t2, t3, t4, t5, t6, t7, t8], forecasting_horizon=3
    - After tabularization: 5 rows (indices 0-4)
    - Row 0 predicts [t2, t3, t4] → center_time = t3 → weight from time_weight@t3
    - Row 1 predicts [t3, t4, t5] → center_time = t4 → weight from time_weight@t4
    - ...
    """
    if time_weight is None:
        return None

    # Extract time column from y_t
    time_series = y_t["time"]

    # Process time_weight into weights aligned with y_t
    if callable(time_weight):
        weight_series = time_weight(time_series)
    else:  # DataFrame
        # Join on time column, fill missing with 1.0
        weight_df = pl.DataFrame({"time": time_series})
        weight_df = weight_df.join(time_weight, on="time", how="left")
        weight_series = weight_df["weight"].fill_null(1.0)

    # Convert to numpy array aligned with y_t times
    weights_y_t = weight_series.to_numpy()

    # Determine number of tabularized rows
    # Tabularization drops first max(lags) rows and last forecasting_horizon rows
    max_lag = forecasting_horizon  # In _get_tabularized_dataset, lags = range(1 + forecasting_horizon)
    n_tab_rows = len(y_t) - max_lag - forecasting_horizon

    if n_tab_rows <= 0:
        return None  # Not enough data for tabularization

    # Map weights from y_t to tabularized rows
    # Strategy: Use weight at "target center time" for each row
    # Row i corresponds to times y_t[max_lag + i : max_lag + i + forecasting_horizon + 1]
    # Center of prediction window: max_lag + i + forecasting_horizon // 2
    sample_weight = np.zeros(n_tab_rows)
    for i in range(n_tab_rows):
        # Center index in original y_t for this tabularized row
        center_idx = max_lag + i + forecasting_horizon // 2
        center_idx = min(center_idx, len(weights_y_t) - 1)  # Clamp to valid range
        sample_weight[i] = weights_y_t[center_idx]

    # Normalize to sum to 1 (sklearn convention for sample_weight)
    if not np.isclose(sample_weight.sum(), 0.0):
        sample_weight = sample_weight / sample_weight.sum() * n_tab_rows  # Preserve total "sample count"

    return sample_weight
```

#### 4. Panel Data Considerations

**Options for Panel Time Weighting**:

**A. Global weighting** (single `time_weight` for all groups):
```python
forecaster.fit(
    y_panel,
    X_panel,
    forecasting_horizon=10,
    time_weight=exponential_decay_weight(half_life=7)  # Applied to all groups
)
```

**B. Per-group weighting** (dict of `time_weight` per group):
```python
forecaster.fit(
    y_panel,
    X_panel,
    forecasting_horizon=10,
    time_weight={
        "store_1": exponential_decay_weight(half_life=7),   # Volatile, weight recent
        "store_2": None,                                     # Stable, no weighting
        "store_3": seasonal_emphasis_weight(...)             # Seasonal patterns
    }
)
```

**Recommendation**: Support **both** (check `isinstance(time_weight, dict)` in `_estimator_fit_one`).

#### 5. Interval Forecaster Extension

**File**: `src/yohou/interval_forecaster/reduction.py`

Same implementation pattern as `PointReductionForecaster`:

```python
class IntervalReductionForecaster(BaseReductionForecaster, BaseIntervalForecaster):
    def fit(
        self,
        y: pl.DataFrame,
        X: pl.DataFrame | None = None,
        forecasting_horizon: StrictInt = 1,
        time_weight: Callable | pl.DataFrame | None = None,  # ← Already declared
        **params,
    ) -> "IntervalReductionForecaster":
        """Fit interval forecaster with time-based sample weighting."""
        # Same logic as PointReductionForecaster
        # time_weight flows to _estimator_fit_one → sample_weight
        ...
```

**Note**: `IntervalReductionForecaster` uses quantile regressors internally (e.g., `QuantileRegressor`), which also support `sample_weight`.

#### 6. Testing Strategy

**Unit Tests** (`tests/point_forecaster/test_reduction.py`, `tests/interval_forecaster/test_reduction.py`):

```python
def test_reduction_forecaster_time_weight_dataframe(y_X_factory):
    """Test time_weight as DataFrame during fit."""
    y, X = y_X_factory(length=100, n_targets=1, seed=42)

    # Create recency weights
    time_weight = pl.DataFrame({
        "time": y["time"],
        "weight": np.linspace(0.5, 2.0, len(y))  # Recent data weighted 4x more
    })

    forecaster = PointReductionForecaster()
    forecaster.fit(y[:80], X[:80], forecasting_horizon=10, time_weight=time_weight[:80])

    # Verify fitting completed (actual weight effect requires inspecting estimator internals)
    assert hasattr(forecaster, "estimator_")

    # Predictions should work
    y_pred = forecaster.predict(forecasting_horizon=10)
    assert len(y_pred) == 10


def test_reduction_forecaster_time_weight_callable(y_X_factory):
    """Test time_weight as callable during fit."""
    from yohou.utils.weighting import exponential_decay_weight

    y, X = y_X_factory(length=100, n_targets=1, seed=42)

    weight_fn = exponential_decay_weight(half_life=10)

    forecaster = PointReductionForecaster()
    forecaster.fit(y[:80], X[:80], forecasting_horizon=10, time_weight=weight_fn)

    assert hasattr(forecaster, "estimator_")
    y_pred = forecaster.predict(forecasting_horizon=10)
    assert len(y_pred) == 10


def test_reduction_forecaster_time_weight_panel_global(y_X_factory):
    """Test global time_weight for panel data."""
    # Create panel data
    y = y_X_factory.panel(length=100, n_targets=2, n_groups=3, seed=42)

    weight_fn = exponential_decay_weight(half_life=7)

    forecaster = PointReductionForecaster()
    forecaster.fit(y[:80], forecasting_horizon=10, time_weight=weight_fn)

    assert hasattr(forecaster, "estimator_")
    y_pred = forecaster.predict(forecasting_horizon=10)
    assert len(y_pred) == 10


def test_reduction_forecaster_time_weight_panel_per_group(y_X_factory):
    """Test per-group time_weight for panel data."""
    y = y_X_factory.panel(length=100, n_targets=1, n_groups=2, seed=42)

    # Different weighting strategies per group
    time_weight = {
        "group_0": exponential_decay_weight(half_life=5),
        "group_1": None  # No weighting for group_1
    }

    forecaster = PointReductionForecaster()
    forecaster.fit(y[:80], forecasting_horizon=10, time_weight=time_weight)

    assert hasattr(forecaster, "estimator_")
    y_pred = forecaster.predict(forecasting_horizon=10)
    assert len(y_pred) == 10


def test_reduction_forecaster_time_weight_affects_fit():
    """Verify time_weight actually affects model training (not just ignored)."""
    # Create synthetic data where recent trend differs from historical
    time = pl.datetime_range(datetime(2020, 1, 1), datetime(2020, 4, 10), interval="1d", eager=True)

    # Historical trend: upward (days 1-90)
    # Recent trend: downward (days 91-100)
    values = np.concatenate([
        np.linspace(10, 100, 90),  # Historical upward
        np.linspace(100, 50, 10)   # Recent downward
    ])

    y = pl.DataFrame({"time": time, "value": values})

    # Model 1: No weighting (learns historical upward trend)
    forecaster_unweighted = PointReductionForecaster()
    forecaster_unweighted.fit(y, forecasting_horizon=5)
    y_pred_unweighted = forecaster_unweighted.predict(forecasting_horizon=5)

    # Model 2: Strong recency weighting (learns recent downward trend)
    weight_fn = exponential_decay_weight(half_life=3)
    forecaster_weighted = PointReductionForecaster()
    forecaster_weighted.fit(y, forecasting_horizon=5, time_weight=weight_fn)
    y_pred_weighted = forecaster_weighted.predict(forecasting_horizon=5)

    # Weighted model should predict lower values (downward trend)
    assert y_pred_weighted["value"].mean() < y_pred_unweighted["value"].mean()
```

**Integration Tests** (`tests/model_selection/test_search.py`):

```python
def test_searchcv_with_time_weight_forecaster(y_X_factory):
    """Test time_weight propagates through RandomizedSearchCV to forecaster.fit()."""
    from scipy.stats import uniform
    from yohou.model_selection import RandomizedSearchCV
    from yohou.utils.weighting import exponential_decay_weight

    y, X = y_X_factory(length=100, n_targets=1, seed=42)

    search = RandomizedSearchCV(
        forecaster=PointReductionForecaster(),
        param_distributions={"estimator__alpha": uniform(0.1, 1.0)},
        scoring=MeanAbsoluteError(),
        n_trials=5,
        cv=SlidingWindowSplitter(window_size=50, horizon=10, step=10)
    )

    weight_fn = exponential_decay_weight(half_life=7)

    # time_weight should route to forecaster.fit() during CV
    search.fit(y, X, forecasting_horizon=10, time_weight=weight_fn)

    assert hasattr(search, "best_forecaster_")
```

#### 7. Documentation Updates

**Docstring for `_process_time_weight_to_sample_weight`**: (Already included above)

**Example in `PointReductionForecaster` docstring**:

```python
class PointReductionForecaster(BaseReductionForecaster, BasePointForecaster):
    """Point forecaster using sklearn estimators on tabularized time series.

    Examples
    --------
    >>> # Example 1: Fit with recency bias
    >>> from yohou.utils.weighting import exponential_decay_weight
    >>> forecaster = PointReductionForecaster()
    >>> weight_fn = exponential_decay_weight(half_life=7)
    >>> forecaster.fit(y, X, forecasting_horizon=5, time_weight=weight_fn)
    >>> y_pred = forecaster.predict(forecasting_horizon=5)

    >>> # Example 2: Custom weight DataFrame
    >>> time_weight = pl.DataFrame({
    ...     "time": y["time"],
    ...     "weight": [1.0] * 80 + [2.0] * 20  # Weight last 20 days more
    ... })
    >>> forecaster.fit(y, X, forecasting_horizon=5, time_weight=time_weight)
    """
```

**Update architecture guide** (`.github/copilot/sklearn-metadata-routing-implementation.md`):

Remove "not yet implemented" note and replace with:

```markdown
### time_weight: Training-Time Sample Weighting

**Status**: ✅ Fully implemented in `BaseReductionForecaster._estimator_fit_one()`.

**Usage**:
```python
# Exponential decay: recent observations weighted more
from yohou.utils.weighting import exponential_decay_weight
forecaster = PointReductionForecaster()
forecaster.fit(y, X, forecasting_horizon=10, time_weight=exponential_decay_weight(half_life=7))
```

**Conversion Logic**:
1. `time_weight` (callable or DataFrame) aligned with original time series rows
2. Converted to `sample_weight` array aligned with tabularized training samples
3. Each tabularized row weighted by `time_weight` at its "center time"
4. Passed to sklearn estimator's `fit(X_tab, y_tab, sample_weight=...)`

**Panel Data**:
- Global: Single `time_weight` applies to all groups
- Per-group: `dict[str, time_weight]` for group-specific weighting
```

---

## Part 3: Unified Design Considerations

### 1. Shared Weighting Utilities

**Critical**: Both scorers and forecasters use the **same** weighting utilities (`yohou.utils.weighting`).

**Why `utils.weighting`?**
- True shared infrastructure used by both scorers and forecasters
- Not tied to metrics specifically - also used for training sample weighting
- Consistent with other shared utilities in `yohou.utils` (tabularization, validation, etc.)
- Clear separation of concerns: utilities are foundational building blocks

**Import pattern**:
```python
# In scorers (src/yohou/metrics/base.py)
from yohou.utils.weighting import exponential_decay_weight

# In forecasters (src/yohou/base.py)
from yohou.utils.weighting import exponential_decay_weight

# User-facing imports
from yohou.utils.weighting import exponential_decay_weight, compose_weights
```

### 2. Callable Signature Consistency

**Requirement**: All weight functions must return `pl.Series` (not numpy arrays) for consistency.

**Standard signature**:
```python
def weight_function(time: pl.Series) -> pl.Series:
    """
    Parameters
    ----------
    time : pl.Series
        Datetime series representing timestamps.

    Returns
    -------
    pl.Series
        Weight values (non-negative, will be normalized).
    """
```

**Enforcement**: Document in `weighting.py` module docstring and each function docstring.

### 3. Weight Normalization Semantics

**Scorers**: Normalize to sum to 1 (weights are probabilities)
```python
weights = weights / weights.sum()  # Sum = 1.0
```

**Forecasters**: Normalize to preserve "sample count" (sklearn convention)
```python
weights = weights / weights.sum() * n_samples  # Sum = n_samples
```

**Why different?**
- **Scorers**: Weighted average of errors (probabilistic interpretation)
- **Forecasters**: `sample_weight` in sklearn represents "effective sample count" (sum > 1 is valid)

**Implementation**: Each processor (`_process_time_weights` in scorers, `_process_time_weight_to_sample_weight` in forecasters) applies appropriate normalization.

### 4. Metadata Routing Consistency

**Both scorers and forecasters**:
- Declare `time_weight` as **explicit parameter** (not in `**params`)
- Reason: API discoverability, special handling required

**GridSearchCV/RandomizedSearchCV routing**:
- `time_weight` passed in `fit()` → routed to forecaster's `fit()` (training-time)
- `time_weight` passed in `score()` → routed to scorer's `score()` (evaluation-time)
- **Different `time_weight` values can be used for training vs evaluation**

**Example**:
```python
search = RandomizedSearchCV(
    forecaster=PointReductionForecaster(),
    param_distributions={"estimator__alpha": uniform(0.01, 1.0)},
    scoring=MeanAbsoluteError(),
    n_iter=10,
    ...
)

# Train with exponential decay, evaluate with uniform weighting
search.fit(
    y, X,
    forecasting_horizon=10,
    time_weight=exponential_decay_weight(half_life=7)  # For forecaster.fit()
)

# Later, score with different weighting
search.score(
    y_test, X_test,
    time_weight=forecast_horizon_weight(decay_rate=0.9)  # For scorer.score()
)
```

### 5. Validation and Error Handling

**Common validation for both**:
- Check `time_weight` type (DataFrame, callable, or None)
- If DataFrame: Validate `"time"` and `"weight"` columns exist
- If callable: Verify signature matches `(pl.Series) -> pl.Series`
- Weights must be non-negative
- Handle missing times gracefully (fill with 1.0)

**Forecaster-specific validation**:
- Check `time_weight` aligns with `y_t` time range
- Verify sklearn estimator supports `sample_weight` (most do, but not all)
- Raise informative error if estimator doesn't support weighting:
  ```python
  if sample_weight is not None:
      # Check if estimator supports sample_weight
      if not _check_sample_weight_support(estimator):
          raise ValueError(
              f"Estimator {type(estimator).__name__} does not support sample_weight. "
              "Cannot use time_weight parameter."
          )
  ```

**Helper function**:
```python
def _check_sample_weight_support(estimator: BaseEstimator) -> bool:
    """Check if estimator's fit() accepts sample_weight parameter."""
    import inspect
    fit_signature = inspect.signature(estimator.fit)
    return "sample_weight" in fit_signature.parameters
```

### 6. Performance Considerations

**Scorer weighting**: Minimal overhead (simple array multiplication in aggregation)

**Forecaster weighting**:
- One-time conversion cost during `fit()`
- Negligible compared to model training time
- **Optimization**: Cache callable results if same `time_weight` used multiple times (not needed initially)

### 7. Backward Compatibility

**Both implementations are backward compatible**:
- Default `time_weight=None` → existing behavior (no weighting)
- No breaking changes to existing APIs
- Tests for unweighted behavior still pass

---

## Implementation Priority

### Phase 1: Core Scorer Weighting (Part 1)
1. Create `src/yohou/utils/weighting.py` with utility functions
2. Modify `BaseScorer.score()` to accept `time_weight`
3. Implement `_process_time_weights()` helper
4. Update `_aggregate_scores()` to apply weights

### Phase 2: Core Forecaster Weighting (Part 2)
1. Implement `_process_time_weight_to_sample_weight()` in `BaseReductionForecaster`
2. Modify `_estimator_fit_one()` to pass `sample_weight` to estimator
3. Add estimator `sample_weight` support validation

### Phase 3: Testing
1. Unit tests for scorer weighting (DataFrame/callable/None formats)
2. Unit tests for forecaster weighting (single series + panel data)
3. Integration tests for GridSearchCV/RandomizedSearchCV routing (both fit and score)
4. Tests for weighting utility functions
5. Effect verification tests (weighted vs unweighted predictions differ)

### Phase 4: Documentation
1. Update `BaseScorer.score()` docstrings
2. Update `PointReductionForecaster.fit()` docstrings
3. Update `.github/copilot/sklearn-metadata-routing-implementation.md`
4. Create example notebook (`examples/time_weighted_forecasting.py`)
5. Add section to user guide

### Phase 5: Optional Enhancements
1. Splitter integration (`time_weight_fn` parameter)
2. Panel data per-group weighting (dict support)
3. Automatic weight generation presets (`time_weight="auto_seasonal"`)
4. Performance optimization (caching for expensive callables)

---

## Open Questions for Discussion

### Critical Decisions

1. **Module placement for weighting utilities**: `metrics.weighting` (current) vs `yohou.weighting` (top-level)?
   - **Recommendation**: `metrics.weighting` (scorers are primary users, forecasters borrow it)

2. **Panel data per-group weighting**: Support `dict[str, time_weight]` for forecasters?
   - **Recommendation**: Yes (check `isinstance(time_weight, dict)` in `_estimator_fit_one`)
   - Scorers: Global only initially (panel group weights already exist via `panel_group_weight` parameter)

3. **Estimator compatibility checking**: Raise error if sklearn estimator doesn't support `sample_weight`?
   - **Recommendation**: Yes (fail fast with clear message)

4. **Center time definition**: Use midpoint of prediction window for sample weight mapping?
   - **Recommendation**: Yes (simple, intuitive)
   - **Alternative**: Use last feature time (most recent observation) - could bias towards recent data

5. **Metadata routing declaration**: Should forecasters/scorers declare `set_fit_request(time_weight=True)`?
   - **Recommendation**: No (explicit parameter is sufficient, aligns with current pattern)

### Future Enhancements

6. **Automatic seasonal weighting**: Support `time_weight="auto_seasonal"` preset?
   - **Recommendation**: Not initially (keep explicit control, add later if requested)

7. **Cross-validation splitter integration**: Auto-generate `time_weight` per fold?
   - **Recommendation**: Phase 5 (optional, not core requirement)

8. **Different weights for training vs evaluation**: Document this as a feature or discourage?
   - **Recommendation**: Document as advanced feature (useful for different objectives)

---

## API Summary

### Scorer Weighting (Evaluation-Time)

```python
from yohou.metrics import MeanAbsoluteError
from yohou.utils.weighting import exponential_decay_weight

scorer = MeanAbsoluteError()

# Option 1: DataFrame weights
weights = pl.DataFrame({"time": times, "weight": [1.0, 0.8, 0.6, 0.4, 0.2]})
score = scorer.score(y_truth, y_pred, time_weight=weights)

# Option 2: Callable weights
weight_fn = exponential_decay_weight(half_life=7)
score = scorer.score(y_truth, y_pred, time_weight=weight_fn)

# Option 3: No weighting (default)
score = scorer.score(y_truth, y_pred)
```

### Forecaster Weighting (Training-Time)

```python
from yohou.point_forecaster import PointReductionForecaster
from yohou.utils.weighting import exponential_decay_weight, seasonal_emphasis_weight, compose_weights

forecaster = PointReductionForecaster()

# Option 1: Exponential decay (recency bias)
weight_fn = exponential_decay_weight(half_life=10)
forecaster.fit(y, X, forecasting_horizon=5, time_weight=weight_fn)

# Option 2: Seasonal emphasis
weight_fn = seasonal_emphasis_weight(seasonality=12, emphasis_phases=[11], emphasis_factor=2.0)
forecaster.fit(y, X, forecasting_horizon=5, time_weight=weight_fn)

# Option 3: Hybrid (recency + seasonal)
weight_fn = compose_weights([
    exponential_decay_weight(half_life=14),
    seasonal_emphasis_weight(seasonality=12, emphasis_phases=[11], emphasis_factor=2.0)
])
forecaster.fit(y, X, forecasting_horizon=5, time_weight=weight_fn)

# Option 4: Panel data per-group weighting
time_weight = {
    "store_1": exponential_decay_weight(half_life=5),
    "store_2": None,  # No weighting
    "store_3": seasonal_emphasis_weight(seasonality=7, emphasis_phases=[5, 6], emphasis_factor=1.5)
}
forecaster.fit(y_panel, forecasting_horizon=5, time_weight=time_weight)
```

### Unified Workflow (RandomizedSearchCV)

```python
from scipy.stats import uniform
from yohou.model_selection import RandomizedSearchCV

search = RandomizedSearchCV(
    forecaster=PointReductionForecaster(),
    scoring=MeanAbsoluteError(),
    ...
)

# Train with recency bias
search.fit(
    y, X,
    forecasting_horizon=10,
    time_weight=exponential_decay_weight(half_life=7)  # For forecaster training
)

# Evaluate with forecast horizon weighting
search.score(
    y_test, X_test,
    time_weight=forecast_horizon_weight(decay_rate=0.9)  # For scorer evaluation
)
```

---

## References

- Current scorer implementation: [src/yohou/metrics/base.py](src/yohou/metrics/base.py)
- Current forecaster implementation: [src/yohou/base.py](src/yohou/base.py) (lines 1374-1475)
- Metadata routing guide: `.github/copilot/sklearn-metadata-routing-implementation.md`
- Tabularization utility: [src/yohou/utils/tabularization.py](src/yohou/utils/tabularization.py)
- Weighting utilities: [src/yohou/utils/weighting.py](src/yohou/utils/weighting.py) (to be created)
- GridSearchCV/RandomizedSearchCV implementation: [src/yohou/model_selection/search.py](src/yohou/model_selection/search.py)
- Cross-validation: [src/yohou/model_selection/split.py](src/yohou/model_selection/split.py)

---

## Implementation Notes

**Critical Implementation Detail for Forecasters**:

The `_process_time_weight_to_sample_weight()` method must carefully handle the time alignment issue. The current proposal uses a "center time" strategy, but alternatives include:

1. **Center time** (recommended): `center_idx = max_lag + i + forecasting_horizon // 2`
   - Pro: Balanced, intuitive
   - Con: Doesn't emphasize most recent observation

2. **Last feature time**: `last_feature_idx = max_lag + i + forecasting_horizon - 1`
   - Pro: Emphasizes most recent observation used for prediction
   - Con: May double-weight recent periods

3. **First target time**: `first_target_idx = max_lag + i + 1`
   - Pro: Emphasizes nearest prediction target
   - Con: Deemphasizes feature recency

**Recommendation**: Implement center time, document clearly, and provide example showing the mapping. Consider making the strategy configurable in future (via parameter like `sample_weight_alignment="center"`).
