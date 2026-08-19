---
name: create-yohou-conformal-adapter
description: "Step-by-step guide for implementing new conformal adapters in Yohou, the estimator family that SplitConformalForecaster uses for adaptive conformal inference (online miscoverage-level adjustment). Covers the stateful BaseConformalAdapter lifecycle (fit/observe/predict/rewind), the (coverage_rate, tail) keying, the symmetric-vs-asymmetric level split, epsilon clipping, _parameter_constraints, and wiring into _yield_yohou_conformal_adapter_checks. Use when creating, extending, or testing any conformal adapter class."
---

# Creating New Conformal Adapters

A conformal adapter maintains a time-varying effective miscoverage level and
updates it online from realized coverage, so `SplitConformalForecaster`
restores its target coverage under drift. It is an optional add-on: with
`adapter=None` the forecaster uses the static calibrated level. An adapter is a
stateful lifecycle estimator that mirrors the forecaster lifecycle: `fit`,
`observe`, `predict`, `rewind`.

The adapter is the level-recursion state machine only. It never sees the
calibration scores or similarity weights; the forecaster holds those, computes
the per-row miscoverage indicators, and feeds them in. This separation is what
lets the adapter compose with `similarity` (the similarity sets the weights,
the adapter sets the level).

## Quick Decision Tree

- **Vanilla single-rate online adjustment** -> use the shipped
  `AdaptiveConformalInference`; do not write a new class.
- **A genuinely new level-adaptation rule** (e.g. DtACI expert aggregation,
  AgACI) -> extend `BaseConformalAdapter` in `src/yohou/interval/adapter.py`.

Composition first, subclassing only for a novel rule.

## The keying model (read this first)

The forecaster clones **one adapter per horizon step** into a per-step dict
`adapters_` (mirroring `similarities_`). Each clone owns one step. Inside its
step the adapter is keyed by `(coverage_rate, tail)`:

- **coverage rate**: one effective level per tracked rate, seeded at fit from
  `1 - coverage_rate`. A rate depends on the requested interval, unlike a
  similarity weight, so the rate dimension lives *inside* the adapter.
- **tail**: for a **symmetric** conformity scorer, one level per rate (a float);
  for an **asymmetric** scorer, two levels per rate (a `(lower, upper)` tuple),
  each targeting `alpha/2`.

`predict()` returns `{coverage_rate: level}` where `level` is a float
(symmetric) or a `(lower, upper)` tuple (asymmetric). The forecaster reads the
scorer's `symmetric` tag and passes it to `fit`.

## The lifecycle contract

- `fit(coverage_rates, *, symmetric)` -> seed one level per key; initialize a
  bounded per-key history stack for rewind. Set fitted attributes ending in `_`.
- `observe(errors)` -> `errors` is a **list of per-row dicts**; each dict maps a
  coverage rate to its miscoverage signal (a float for symmetric, a
  `(lower, upper)` tuple for asymmetric). Apply one update and push one history
  entry **per row**, so rewind is row-exact. A row a step could not score
  arrives as the zero-update sentinel (err equal to the target), which the
  forecaster supplies; your update must be a no-op for it.
- `predict()` -> current levels; call `check_is_fitted` first.
- `rewind(n_rows)` -> pop `n_rows` history entries per key, never below the seed.

## Read First (ground truth)

- `src/yohou/interval/base.py` -> `BaseConformalAdapter`: the abstract methods
  and the `estimator_type="conformal_adapter"` tags.
- `src/yohou/interval/adapter.py` -> `AdaptiveConformalInference`: the full
  Gibbs-Candes update, `alpha_pooling`, `epsilon` clipping, per-tail logic.

    Note on `alpha_pooling`: it is declared on `BaseConformalAdapter`, but your
    subclass must still accept it in its own `__init__` and forward it with
    `super().__init__(alpha_pooling=alpha_pooling)`. Estimator parameter
    discovery reads the most derived constructor only, so omitting it drops the
    setting from `get_params`, makes `adapter__alpha_pooling` unaddressable in a
    search, and lets `clone` silently reset a configured `"shared"` to the
    default. Nothing raises if you get this wrong.
- `src/yohou/interval/split_conformal.py` -> `_observe_adapter`,
  `_rewind_adapter`, `_adapter_step_errors`, `_pool_adapter_errors`,
  `_adapter_inverse_score`, and the `predict_interval` adapter branch: how the
  forecaster drives the adapter and computes the indicators.
- `tests/test_common.py` (`_conformal_adapter_instances`,
  `TestConformalAdapterCommon`) and `tests/interval/test_adapter.py` for the
  fixture and behavioral shapes.

## Templates

- [Adapter class template](./adapter_template.py): `MyAdapter(BaseConformalAdapter)`
  with the full `fit`/`observe`/`predict`/`rewind` lifecycle and one tunable param.
- [Test file template](./test_adapter_template.py): behavioral tests plus the
  `_yield_yohou_conformal_adapter_checks` systematic-checks wiring.

## Wiring checklist

1. Class in `src/yohou/interval/adapter.py` extending `BaseConformalAdapter`,
   with `_parameter_constraints` and a runnable docstring example.
2. Export from `src/yohou/interval/__init__.py` (`__all__` and the import).
3. Register a working instance in `tests/test_common.py`
   `_conformal_adapter_instances()` and add the class name to `_SKIP_COMMON` so
   the generic sweep defers to `TestConformalAdapterCommon`.
4. Behavioral and integration tests in `tests/interval/test_adapter.py`.
5. If the adapter needs a new tag, extend `ConformalAdapterTags` in
   `src/yohou/utils/tags.py`.

## Invariants the systematic checks enforce

- `predict` returns one level per tracked rate, each in `[0, 1]` (per tail).
- `observe` then `rewind` of the same rows restores the levels exactly.
- A miscovered observation lowers the level; a covered one raises it.
- A positive `epsilon` keeps the level within `[epsilon, 1 - epsilon]`.
- `predict`, `observe`, and `rewind` raise `NotFittedError` before `fit`.

Run `just test-fast` and `just lint` before opening a PR.
