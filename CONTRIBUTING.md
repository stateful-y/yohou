# Contributing

Thank you for your interest in contributing! All contributions (bug reports, fixes, documentation improvements, and feature suggestions) are welcome.

Please review our [Code of Conduct](CODE_OF_CONDUCT.md) before participating.

## Quick Setup

```bash
git clone <repo-url>
cd <project-dir>
uv sync --group dev
uv run prek install -f
just test-fast
```

## Full Guidelines

For the complete contributing guide, including test strategy, code quality standards, commit conventions, and CI/CD details, see:

**[Full Contributing Guide](docs/pages/how-to/contributing.md)**

## Naming Estimators

Transformer class names follow one rule, so a new name need not be debated:

- A name that says what the object **is**, a doer, carries no suffix: `StandardScaler`,
  `SeasonalImputer`, `Downsampler`, `StepAggregator`.
- A name that says what the object **produces**, a thing, carries `Transformer`:
  `LagTransformer`, `RollingStatisticsTransformer`, `FourierFeatureTransformer`. On its own,
  `Lag` would read like a value in a configuration, not a step that is fitted.

Two prefixes mark families that work with step columns (`{base}_step_1..H`, one column per
forecast step):

- `Step*` transformers **consume** step columns that already exist (step-kind, in the
  `step_transformer` slot).
- `Horizon*` transformers **produce** step columns from history (actual-kind, tagged
  `produces_step_columns`, in the `actual_transformer` slot).

Seasonal estimators name their season length `seasonality`.

## Reporting Issues

Found a bug? Have a suggestion? [Open an issue](../../issues/new/choose) and include:

- Python and uv versions
- Steps to reproduce
- Expected vs. actual behavior
