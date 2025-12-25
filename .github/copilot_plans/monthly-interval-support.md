# Plan: Support Monthly and Variable-Length Intervals in Yohou

## Problem Statement

Yohou currently validates time series intervals using exact `timedelta` matching, which fails for monthly data because months have variable lengths (28-31 days). The validation in `check_interval_consistency()` expects uniform intervals, but monthly data inherently has 4 different interval lengths.

**Error observed:**
```
ValueError: Time series has inconsistent intervals. Expected uniform interval of 31 days, 0:00:00,
but found 4 different intervals: [timedelta(days=28), timedelta(days=29), timedelta(days=30), timedelta(days=31)]
```

## Current Implementation Analysis

### `src/yohou/utils/validation.py`

**`check_interval_consistency(df: pl.DataFrame) -> timedelta`**
- Uses `pl.col("time").diff()` to calculate intervals
- Expects all intervals to be identical
- Returns a single `timedelta` object
- Fails for monthly, quarterly, yearly frequencies

**Issues:**
1. `timedelta` cannot represent "1 month" abstractly (only concrete days)
2. Monthly intervals vary: 28-31 days (leap years, month lengths)
3. No way to store/retrieve "1mo" semantic interval
4. All forecasters and transformers assume fixed `timedelta` intervals

## Proposed Solution: Interval String Abstraction

### Core Design

Introduce an **interval representation** that uses strings for all frequencies:

```python
from datetime import timedelta

# Interval is always a string
Interval = str  # e.g., "1d", "1h", "1mo", "1q", "1y"

# Supported string formats (aligned with polars)
# Fixed intervals:
# - "1h", "2h", "12h" → hours
# - "1d", "7d" → days
# - "1w", "2w" → weeks
#
# Variable-length intervals:
# - "1mo", "2mo", "3mo", "6mo", "12mo" → months (variable 28-31 days)
# - "1q" → quarters (3 months, ~89-92 days)
# - "1y", "2y" → years (365-366 days with leap years)
```

### Implementation Steps

#### 1. Update `check_interval_consistency()` in `src/yohou/utils/validation.py`

**New logic:**
- Calculate unique deltas between consecutive timestamps
- Try to infer a regular pattern from the deltas
- **Always return a string** representation of the interval:
  - Fixed intervals: "1d", "7d", "1h", "1w"
  - Variable intervals: "1mo", "2mo", "3mo", "1q", "1y"
- If deltas are uniform timedelta, convert to string format
- If inconsistent, detect if intervals follow a semantic pattern:
  - Calculate month differences between consecutive dates
  - All intervals 28-31 days with consistent month differences → infer "Nmo" (1mo, 2mo, 3mo, etc.)
  - All intervals ~89-92 days with 3-month pattern → infer "1q"
  - All intervals 365-366 days → infer "1y" or "2y"
  - Check day-of-month consistency (allow month-end adjustments)
- Return `str` type for all cases

**Signature change:**
```python
def check_interval_consistency(df: pl.DataFrame) -> Interval:
    """Check time series has consistent intervals and return interval.

    Returns
    -------
    str
        String representation of the interval.
        Examples: "1d", "1h", "1w", "1mo", "3mo", "1q", "1y"
    """
```

**Implementation approach** (inspired by pandas `infer_freq`):
```python
def check_interval_consistency(df: pl.DataFrame) -> str:
    """Infer time series interval with support for variable-length periods.

    Always returns a string representation of the interval.
    Uses a robust frequency inference algorithm inspired by pandas.infer_freq:
    1. Calculate unique deltas
    2. Check for common patterns (hourly, daily, weekly, monthly, etc.)
    3. Convert uniform timedeltas to string format
    4. Infer month/quarter/year frequencies by analyzing delta distributions
    5. Support multi-period intervals (e.g., "2mo", "3mo", "6mo")
    """
    time_series = df["time"].to_list()

    if len(time_series) < 2:
        raise ValueError("Need at least 2 time points to infer interval")

    # Calculate deltas
    deltas = [time_series[i+1] - time_series[i] for i in range(len(time_series) - 1)]
    unique_deltas = sorted(set(deltas))

    # Fast path: exact timedelta match - convert to string
    if len(unique_deltas) == 1:
        return _timedelta_to_string(unique_deltas[0])

    # Check if deltas are all similar (within small tolerance for rounding)
    delta_days = [d.days for d in unique_deltas]
    min_delta, max_delta = min(delta_days), max(delta_days)

    # Sub-day intervals with small variation (e.g., hourly with DST)
    if max_delta == 0:
        # All deltas are sub-day
        delta_seconds = [d.total_seconds() for d in unique_deltas]
        if max(delta_seconds) - min(delta_seconds) <= 3600:  # ±1 hour tolerance
            median_seconds = sorted(delta_seconds)[len(delta_seconds)//2]
            return _timedelta_to_string(timedelta(seconds=median_seconds))

    # Infer based on delta distribution
    freq = _infer_freq_from_deltas(time_series, unique_deltas)
    if freq is not None:
        return freq

    # Could not infer - raise detailed error
    raise ValueError(
        f"Time series has inconsistent intervals. "
        f"Found {len(unique_deltas)} different intervals: {unique_deltas}. "
        f"Cannot infer a regular frequency pattern."
    )


def _timedelta_to_string(td: timedelta) -> str:
    """Convert a timedelta to string interval format.

    Examples:
        timedelta(hours=1) → "1h"
        timedelta(days=1) → "1d"
        timedelta(days=7) → "1w"
        timedelta(days=14) → "2w"

    # Weekly pattern: 7-day intervals
    if len(unique_deltas) == 1 and delta_days[0] % 7 == 0:
        weeks = delta_days[0] // 7
        return f"{weeks}w"

    # Daily pattern: uniform day intervals
    if len(unique_deltas) == 1:
        return _timedelta_to_string(unique_deltas[0])

    # Monthly patterns: 28-31 day range
    elif total_seconds % 86400 == 0:
        days = int(total_seconds // 86400)
        return f"{days}d"
    elif total_seconds % 3600 == 0:
        hours = int(total_seconds // 3600)
        return f"{hours}h"
    elif total_seconds % 60 == 0:
        minutes = int(total_seconds // 60)
        return f"{minutes}min"
    else:
        seconds = int(total_seconds)
        return f"{seconds}s"


def _infer_freq_from_deltas(time_series: list[datetime], unique_deltas: list[timedelta]) -> str | None:
    """Infer frequency from delta distribution.

    Inspired by pandas._FrequencyInferer logic:
    - Analyze month differences for monthly/quarterly/yearly patterns
    - Check day-of-month consistency
    - Support multi-period intervals (2mo, 3mo, 6mo, etc.)
    """
    delta_days = [d.days for d in unique_deltas]
    min_delta, max_delta = min(delta_days), max(delta_days)

    # Weekly pattern: 7-day intervals
    if len(unique_deltas) == 1 and delta_days[0] % 7 == 0:
        weeks = delta_days[0] // 7
        return f"{weeks}w" if weeks > 1 else timedelta(days=7)

    # Monthly patterns: 28-31 day range
    if 28 <= min_delta <= 31 and 28 <= max_delta <= 31:
        month_freq = _infer_monthly_freq(time_series)
        if month_freq:
            return month_freq

    # Quarterly patterns: ~89-92 day range (3 months)
    if 89 <= min_delta <= 92 and 89 <= max_delta <= 92:
        return _infer_quarterly_freq(time_series)

    # Semi-annual: ~181-184 day range (6 months)
    if 181 <= min_delta <= 184 and 181 <= max_delta <= 184:
        return "6mo"

    # Yearly patterns: 365-366 day range
    if 365 <= min_delta <= 366 and 365 <= max_delta <= 366:
        return "1y"

    # Multi-month patterns (2, 4, 6, 12 months)
    # Check if deltas are roughly N months
    for n_months in [2, 3, 4, 6, 12]:
        expected_min = n_months * 28
        expected_max = n_months * 31
        if expected_min <= min_delta <= expected_max and expected_min <= max_delta <= expected_max:
            if _verify_n_month_pattern(time_series, n_months):
                return f"{n_months}mo"

    return None


def _infer_monthly_freq(time_series: list[datetime]) -> str | None:
    """Infer monthly frequency (1mo, 2mo, 3mo, etc.) by checking month differences."""
    # Calculate month differences
    month_diffs = []
    for i in range(len(time_series) - 1):
        d1, d2 = time_series[i], time_series[i + 1]
        month_diff = (d2.year - d1.year) * 12 + (d2.month - d1.month)
        month_diffs.append(month_diff)

    unique_month_diffs = set(month_diffs)

    # All month differences should be the same for regular monthly data
    if len(unique_month_diffs) == 1:
        n_months = month_diffs[0]
        # Verify day-of-month consistency
        if _check_day_of_month_consistency(time_series):
            return f"{n_months}mo" if n_months > 1 else "1mo"

    return None


def _infer_quarterly_freq(time_series: list[datetime]) -> str | None:
    """Infer quarterly frequency by checking 3-month patterns."""
    month_diffs = []
    for i in range(len(time_series) - 1):
        d1, d2 = time_series[i], time_series[i + 1]
        month_diff = (d2.year - d1.year) * 12 + (d2.month - d1.month)
        month_diffs.append(month_diff)

    # Quarterly should be exactly 3 months
    if all(md == 3 for md in month_diffs) and _check_day_of_month_consistency(time_series):
        return "1q"

    return None


def _verify_n_month_pattern(time_series: list[datetime], n_months: int) -> bool:
    """Verify that time series follows N-month intervals."""
    for i in range(len(time_series) - 1):
        d1, d2 = time_series[i], time_series[i + 1]
        month_diff = (d2.year - d1.year) * 12 + (d2.month - d1.month)
        if month_diff != n_months:
            return False
    return _check_day_of_month_consistency(time_series)


def _check_day_of_month_consistency(time_series: list[datetime]) -> bool:
    """Check if day-of-month is consistent (allowing for month-end edge cases).

    Examples:
    - [Jan 31, Feb 28, Mar 31] → True (month-end adjusted)
    - [Jan 15, Feb 15, Mar 15] → True (consistent day)
    - [Jan 10, Feb 15, Mar 20] → False (inconsistent)
    """
    days = [d.day for d in time_series]
    unique_days = set(days)

    # All same day-of-month
    if len(unique_days) == 1:
        return True

    # Check if variations are due to month-end adjustments
    # E.g., 31 → 28/29 → 31 is acceptable for Jan 31 + 1mo
    import calendar

    for i in range(len(time_series) - 1):
        d1, d2 = time_series[i], time_series[i + 1]
        days_in_d2_month = calendar.monthrange(d2.year, d2.month)[1]

        # Check if d1.day would overflow into next month
        target_day = d1.day
        actual_day = d2.day

        # If target day exceeds days in month, should be capped at month end
        if target_day > days_in_d2_month:
            if actual_day != days_in_d2_month:
                return False
        elif target_day != actual_day:
            return False

    return True
**Store interval as `str` type:**
```python
class BaseForecaster:
    interval_: str  # Changed from timedelta to str

    def _pre_fit(self, y, X_post, X_ante, forecasting_horizon):
        self.interval_ = check_inputs(y, X_post, X_ante)  # Now returns str
        # ... rest of method
```

**No changes needed in most forecasting logic** - interval is stored and used via helper functions.
        # ... rest of method
```

**No changes needed in most forecasting logic** - interval is stored but rarely used directly in computations.

#### 3. Add Interval Utilities in `src/yohou/utils/validation.py`

**Helper functions:**
```python
def parse_interval(interval: str) -> tuple[int, str]:
    """Parse interval string into (multiplier, unit).

    Examples:
        "1d" → (1, "d")
        "3mo" → (3, "mo")
        "2w" → (2, "w")
    """
    import re
    match = re.match(r'(\d+)(mo|q|y|w|d|h|min|s)', interval)
    if not match:
        raise ValueError(f"Invalid interval format: {interval}")
    return int(match.group(1)), match.group(2)


def is_fixed_interval(interval: str) -> bool:
    """Check if interval is fixed-length (timedelta-based)."""
    _, unit = parse_interval(interval)
    return unit in {"d", "h", "min", "s", "w"}


def is_variable_interval(interval: str) -> bool:
    """Check if interval is variable-length (monthly, yearly)."""
    _, unit = parse_interval(interval)
    """Add n intervals to a datetime (handles variable-length intervals).

    Supports multi-period intervals like "2mo", "3mo", "6mo", etc.
    """
    multiplier, unit = parse_interval(interval)
    total_units = multiplier * n

    if unit == "d":
        return dt + timedelta(days=total_units)
    elif unit == "h":
        return dt + timedelta(hours=total_units)
    elif unit == "min":
        return dt + timedelta(minutes=total_units)
    elif unit == "s":
        return dt + timedelta(seconds=total_units)
    elif unit == "w":
        return dt + timedelta(weeks=total_units)
    elif unit == "mo":
    if unit == "d":
        return timedelta(days=multiplier)
    elif unit == "h":
        return timedelta(hours=multiplier)
    elif unit == "min":
        return timedelta(minutes=multiplier)
    elif unit == "s":
    elif unit == "q":
        # Quarters are 3 months
        return add_interval(dt, "3mo", n)
    elif unit == "y":
        # Add years (handles leap years)
        return dt.replace(year=dt.year + total_units)
    else:
        raise ValueError(f"Unsupported interval unit: {unit}")
        return dt + (interval * n)

    # Parse string interval (e.g., "2mo" → (2, "mo"))
    import re
    match = re.match(r'(\d+)(mo|q|y|w|d|h)', interval)
    if not match:
        raise ValueError(f"Invalid interval format: {interval}")

    multiplier = int(match.group(1))
    unit = match.group(2)
    total_units = multiplier * n

    if unit == "mo":
        # Add months handling year rollover
        month = dt.month - 1 + total_units
        year = dt.year + month // 12
        month = month % 12 + 1
        # Handle day overflow (e.g., Jan 31 + 1mo = Feb 28/29)
        day = min(dt.day, _days_in_month(year, month))
        return dt.replace(year=year, month=month, day=day)
    elif unit == "q":
        # Quarters are 3 months
        return add_interval(dt, "1mo", total_units * 3)
    elif unit == "y":
        # Add years (handles leap years)
        return dt.replace(year=dt.year + total_units)
    elif unit == "w":
        return dt + timedelta(weeks=total_units)
    elif unit == "d":
        return dt + timedelta(days=total_units)
    elif unit == "h":
        return dt + timedelta(hours=total_units)
    else:
        raise ValueError(f"Unsupported interval unit: {unit}")

def _days_in_month(year: int, month: int) -> int:
    """Get number of days in a month."""
    import calendar
    return calendar.monthrange(year, month)[1]
```

#### 4. Update Prediction Time Generation

**In forecasters that generate future timestamps** (e.g., `_add_time_columns()`), use `add_interval()`:

```python
def _add_time_columns(self, y_pred: pl.DataFrame) -> pl.DataFrame:
    """Add time columns to predictions."""
    last_time = self._y_observed["time"].max()

    # Generate future times using interval-aware helper
    future_times = [
        add_interval(last_time, self.interval_, i + 1)
        for i in range(len(y_pred))
    ]

    return y_pred.with_columns([
        pl.Series("time", future_times),
        pl.lit(last_time).alias("observed_time"),
        pl.Series("predicted_time", future_times)
    ])
```

#### 5. Update Tests

**Add test cases for monthly/quarterly/yearly data:**
```python
# tests/utils/test_validation.py

def test_check_interval_consistency_monthly():
    """Test monthly interval detection."""
    df = pl.DataFrame({
        "time": pl.date_range(
            start=date(2020, 1, 1),
            end=date(2020, 12, 1),
            interval="1mo",
            eager=True
        ),
        "value": range(12)
    })
    interval = check_interval_consistency(df)
    assert interval == "1mo"

def test_check_interval_consistency_quarterly():
    """Test quarterly interval detection."""
    # Generate quarterly dates
    dates = [date(2020, 1, 1), date(2020, 4, 1), date(2020, 7, 1), date(2020, 10, 1)]
    df = pl.DataFrame({"time": dates, "value": range(4)})
    interval = check_interval_consistency(df)
    assert interval == "1q"
```

**Update existing forecaster tests** to work with both fixed and variable intervals.

#### 6. Documentation Updates

**Update docstrings to mention interval types:**
- `BaseForecaster.interval_`: Document it's always a `str` like "1d", "1mo", "1q"
- `check_interval_consistency()`: Explain string return format and monthly/quarterly/yearly detection
- Add tutorial section on handling different frequencies and string format

### Backward Compatibility

**Breaking changes:**
- `interval_` attribute changes from `timedelta` to `Union[timedelta, str]`
- Code that directly uses `interval_` for arithmetic will need updates

### Backward Compatibility

**Breaking changes:**
- `interval_` attribute changes from `timedelta` to `str`
- All interval values are now strings ("1d", "1h", "1mo", etc.)
- Code that directly uses `interval_` for arithmetic will need updates

**Migration path:**
1. Use `interval_to_timedelta()` to convert to timedelta if needed (only works for fixed intervals)
2. Use `add_interval()` helper for all interval arithmetic (works for all interval types)
3. Use `parse_interval()` to get multiplier and unit for custom logic
**Deprecation strategy:**
- Phase 1: Change `interval_` to `str`, add conversion utilities, update all internal usage
- Phase 2: Update documentation and examples to use string intervals
- Phase 3: Add warnings if users try to use interval_ as timedelta directly (detect via type checking in custom code)
- **Serialization-friendly**: Easy to save/load models (JSON, pickle)
- **Human-readable**: Clear display in logs and error messages
- **Extensible**: Easy to add new interval types without type changes
## Alternative Approaches Considered

### 1. **Polars Period Type**
- Use `pl.Duration` or custom period handling
- **Rejected**: Polars doesn't have first-class period support for variable-length intervals

### 2. **Pandas Period Index**
- Convert to pandas for interval handling
- **Rejected**: Adds pandas dependency, breaks polars-first design

### 3. **Relaxed Validation**
- Allow ±1 day tolerance for monthly data
- **Rejected**: Doesn't handle quarterly/yearly, brittle heuristic

### 4. **User-Specified Interval**
- Require users to pass `interval="1mo"` parameter
- **Rejected**: Breaks auto-detection UX, extra boilerplate

### Phase 1: Core Support (High Priority)
- [ ] Update `check_interval_consistency()` to always return `str`
- [ ] Add `_timedelta_to_string()` converter
- [ ] Change `interval_` type to `str` in all classes
- [ ] Add `parse_interval()`, `interval_to_timedelta()`, `is_fixed_interval()`, `is_variable_interval()` utilities
- [ ] Update `add_interval()` to work with string intervals
- [ ] Add monthly pattern detection with `_infer_monthly_freq()`
- [ ] Fix air passengers tutorial
- [ ] Basic tests for monthly data and timedelta conversion
- [ ] Fix air passengers tutorial
- [ ] Basic tests for monthly data

### Phase 2: Extended Support (Medium Priority)
- [ ] Add quarterly and yearly detection
- [ ] Update all forecasters to use `add_interval()`
- [ ] Comprehensive test coverage
- [ ] Documentation updates

### Phase 3: Polish (Low Priority)
- [ ] Performance optimization for large datasets
- [ ] Support custom intervals (e.g., "2mo", "6mo")
- [ ] Interval conversion utilities
- [ ] Edge case handling (DST, leap seconds)

## Testing Strategy

**Use `@pytest.mark.parametrize` with random start dates for comprehensive coverage:**

**tests/utils/test_validation_intervals.py:**

```python
import pytest
from datetime import datetime, timedelta
import polars as pl
import random
from yohou.utils.validation import check_interval_consistency
from yohou.utils.polars import add_interval, parse_interval, interval_to_timedelta, _timedelta_to_string


def random_start_date(seed=None):
    """Generate random datetime between 1950 and 2030."""
    if seed is not None:
        random.seed(seed)
    year = random.randint(1950, 2030)
    month = random.randint(1, 12)
    day = random.randint(1, 28)  # Safe day for all months
    hour = random.randint(0, 23)
    minute = random.randint(0, 59)
    return datetime(year, month, day, hour, minute)


@pytest.mark.parametrize("interval_str,n_periods,seed", [
    # Fixed intervals - daily
    ("1d", 100, 42), ("1d", 365, 123), ("7d", 52, 456), ("14d", 26, 789),

    # Fixed intervals - sub-daily
    ("1h", 168, 111), ("6h", 240, 222), ("12h", 120, 333),

    # Fixed intervals - weekly
    ("1w", 52, 444), ("2w", 26, 555),

    # Variable intervals - monthly
    ("1mo", 120, 666), ("2mo", 60, 777), ("3mo", 40, 888), ("6mo", 20, 999),

    # Variable intervals - quarterly and yearly
    ("1q", 40, 1111), ("1y", 10, 2222), ("2y", 10, 3333),
])
def test_check_interval_consistency_parametrized(interval_str, n_periods, seed):
    """Test check_interval_consistency with wide range of intervals and random start dates."""
    start = random_start_date(seed)
    time_series = [add_interval(start, interval_str, i) for i in range(n_periods)]
    df = pl.DataFrame({"time": time_series})

    detected_interval = check_interval_consistency(df)
    assert detected_interval == interval_str


@pytest.mark.parametrize("start_day,interval_str,n_periods,seed", [
    # Month-end edge cases - start on day 31
    (31, "1mo", 12, 4444), (31, "2mo", 6, 5555), (31, "3mo", 4, 6666),

    # Month-end edge cases - start on day 30
    (30, "1mo", 12, 7777), (30, "2mo", 6, 8888),

    # Month-end edge cases - start on day 29 (leap year handling)
    (29, "1mo", 24, 9999), (29, "2mo", 12, 10000),
])
def test_check_interval_consistency_month_end(start_day, interval_str, n_periods, seed):
    """Test monthly intervals with month-end edge cases (Jan 31 → Feb 28/29 → Mar 31)."""
    random.seed(seed)
    year = random.randint(2000, 2030)

    # Pick a month that has the required start_day
    if start_day == 31:
        month = random.choice([1, 3, 5, 7, 8, 10, 12])
    elif start_day == 30:
        month = random.choice([1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    else:
        month = random.randint(1, 12)

    start = datetime(year, month, start_day)
    time_series = [add_interval(start, interval_str, i) for i in range(n_periods)]
    df = pl.DataFrame({"time": time_series})

    detected_interval = check_interval_consistency(df)
    assert detected_interval == interval_str


@pytest.mark.parametrize("td,expected_str", [
    (timedelta(days=1), "1d"), (timedelta(days=7), "1w"), (timedelta(days=14), "2w"),
    (timedelta(days=30), "30d"), (timedelta(hours=1), "1h"), (timedelta(hours=6), "6h"),
    (timedelta(hours=24), "1d"), (timedelta(minutes=30), "30min"), (timedelta(seconds=60), "1min"),
])
def test_timedelta_to_string(td, expected_str):
    """Test conversion from timedelta to string format."""
    assert _timedelta_to_string(td) == expected_str


@pytest.mark.parametrize("interval_str,expected_multiplier,expected_unit", [
    ("1d", 1, "d"), ("7d", 7, "d"), ("1w", 1, "w"), ("2w", 2, "w"),
    ("1h", 1, "h"), ("12h", 12, "h"), ("1mo", 1, "mo"), ("3mo", 3, "mo"),
    ("6mo", 6, "mo"), ("1q", 1, "q"), ("1y", 1, "y"), ("2y", 2, "y"),
])
def test_parse_interval(interval_str, expected_multiplier, expected_unit):
    """Test parsing interval strings into (multiplier, unit) tuples."""
    multiplier, unit = parse_interval(interval_str)
    assert multiplier == expected_multiplier and unit == expected_unit


@pytest.mark.parametrize("interval_str,expected_td", [
    ("1d", timedelta(days=1)), ("7d", timedelta(days=7)), ("1w", timedelta(weeks=1)),
    ("2w", timedelta(weeks=2)), ("1h", timedelta(hours=1)), ("12h", timedelta(hours=12)),
    ("30min", timedelta(minutes=30)), ("60s", timedelta(seconds=60)),
    ("1mo", None), ("1q", None), ("1y", None),  # Variable intervals
])
def test_interval_to_timedelta(interval_str, expected_td):
    """Test conversion from fixed interval strings back to timedelta."""
    assert interval_to_timedelta(interval_str) == expected_td


@pytest.mark.parametrize("start_seed,interval_str,n,expected_months_added", [
    (42, "1mo", 1, 1), (42, "1mo", 12, 12), (123, "2mo", 6, 12),
    (456, "3mo", 4, 12), (789, "6mo", 2, 12), (111, "1q", 4, 12),
    (222, "1y", 2, 24),
])
def test_add_interval_monthly(start_seed, interval_str, n, expected_months_added):
    """Test add_interval for monthly patterns with random start dates."""
    start = random_start_date(start_seed)
    result = add_interval(start, interval_str, n)

    months_diff = (result.year - start.year) * 12 + (result.month - start.month)
    assert months_diff == expected_months_added


@pytest.mark.parametrize("start_seed,interval_str,n_periods", [
    (333, "1d", 100), (444, "1w", 52), (555, "1mo", 24),
    (666, "3mo", 12), (777, "1q", 8), (888, "1y", 5),
])
def test_add_interval_roundtrip(start_seed, interval_str, n_periods):
    """Test that repeatedly adding intervals creates consistent time series."""
    start = random_start_date(start_seed)
    time_series = [add_interval(start, interval_str, i) for i in range(n_periods)]
    df = pl.DataFrame({"time": time_series})

    detected = check_interval_consistency(df)
    assert detected == interval_str
```

**Integration tests:**
- Air passengers tutorial end-to-end
- Forecasting with monthly data (fit + predict)
- Pipeline with monthly transformers
- Cross-validation with monthly splits

**Regression tests:**
- Existing daily/hourly tests still pass
- No performance degradation for fixed intervals

## Success Criteria

1. ✅ Air passengers tutorial runs without errors
2. ✅ Monthly, quarterly, yearly data supported
3. ✅ No breaking changes to existing daily/hourly usage
4. ✅ Clear error messages for truly inconsistent intervals
5. ✅ Documentation includes frequency handling guidance
6. ✅ Test coverage >95% for interval utilities

## Open Questions

1. **Custom intervals**: Support "2mo", "6mo", "18mo"? → **Yes**, implemented with regex parsing in `add_interval()` and inference in `_infer_monthly_freq()`
2. **Mixed frequencies**: Allow dataset with both daily and monthly? → No, require uniform
3. **Business days**: Handle "1bd" (skip weekends)? → Out of scope, future feature
4. **Week start**: Handle week-start variations (Mon vs Sun)? → Use ISO 8601 (Monday)
5. **Timezone handling**: How to handle DST transitions? → Store UTC, document TZ behavior

## Resources

- Polars duration docs: https://docs.pola.rs/user-guide/expressions/temporal/
- Python `relativedelta`: https://dateutil.readthedocs.io/en/stable/relativedelta.html
- Pandas period handling: https://pandas.pydata.org/docs/user_guide/timeseries.html#time-span-representation
