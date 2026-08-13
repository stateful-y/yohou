# Control Yohou's Diagnostic Output

Yohou reports two kinds of thing, through two different channels, and you control
them separately.

- **Warnings** mean *this may be wrong and you should look*. They go through
  Python's `warnings` machinery.
- **Log records** mean *here is what I did*. They go through the standard
  `logging` module, under the `yohou` logger.

Nothing is printed directly to standard output. If you see output from Yohou that
you cannot turn off through either mechanism, that is a bug worth reporting.

## Turn the log output on

The library attaches only a `NullHandler` and **sets no level anywhere**, which is
the standard convention for a library: Yohou emits, your application decides where
records go and at what level. Until you configure something, you see nothing.

```python
import logging

logging.basicConfig(level=logging.INFO)
```

That is enough to see Yohou's records alongside the rest of your application's.

## Target one subsystem

Loggers follow the module path, so you can raise or lower one part of the library
without touching the others:

```python
logging.getLogger("yohou").setLevel(logging.WARNING)          # quiet overall
logging.getLogger("yohou.compose").setLevel(logging.DEBUG)    # except composition
```

`yohou.compose`, `yohou.model_selection`, `yohou.plotting` and the rest are all
separately controllable. Setting a level on `yohou` affects every child that has
not set its own.

## Turn a warning off

Warnings are ordinary Python warnings, so the usual filters apply:

```python
import warnings

warnings.filterwarnings("ignore", message="X_forecast covers")
```

Prefer filtering by category where one exists, since message text is not a stable
interface:

```python
from yohou import ForecastCoverageWarning

warnings.filterwarnings("ignore", category=ForecastCoverageWarning)
```

## Read the coverage warning's detail

`ForecastCoverageWarning` carries the per-column breakdown the check computed, not
only the worst number, because which channel is starved is what tells you what to
do:

```python
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    forecaster.observe(y=frame)

for entry in caught:
    if isinstance(entry.message, ForecastCoverageWarning):
        print(entry.message.coverage)             # {'temp': 0, 'load': 48}
        print(entry.message.forecasting_horizon)  # 48
```

A value of `0` means every step feature derived from that column is null, so a
model relying on it is predicting without it. A value between 0 and the horizon
means the later steps are null.

It subclasses `UserWarning`, so code that already catches `UserWarning` keeps
working.

## Diagnostics that used to be warnings

Two diagnostics moved from the warning channel to the log channel, because both
state what the library did rather than something you may need to act on. If you
were catching either, catch the log record instead.

| what | where it goes now |
| --- | --- |
| `PerVintageActualTransformer` dropped N vintages with too few rows | `yohou.compose.per_vintage`, at INFO |
| `Interpolated N NaN value(s) before decomposition` | `yohou.plotting.forecasting`, at INFO |

The first says "this is expected for the truncated tail of a forecast frame" in
its own text, which is the clearest sign it was never a warning. Everything else
that warns today still warns: a condition you may need to act on stays in the
channel you cannot easily ignore.

## Estimator progress output

`ColumnTransformer` and `FeatureUnion` accept `verbose`, matching the sklearn
classes they mirror. It defaults to `False`, which means silent:

```python
ColumnTransformer(transformers=[...], verbose=True).fit(frame)
# [ColumnTransformer] (step 1 of 4) Processing lags, total=0.1s
```

This output goes to standard output rather than through logging, because it
mirrors sklearn's behaviour exactly. If you want it captured, set `verbose=False`
and raise the level on the `yohou` logger instead.

!!! note "Changed behaviour"

    Before this was fixed, both classes printed on every fit regardless of
    `verbose`. If you were relying on seeing those lines, set `verbose=True`.
