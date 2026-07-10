# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).


## [Unreleased]

### Features
- Add `BaseForecastTransformer`, a base class for transformers over `X_forecast`
  frames (two time axes: `vintage_time` and `time`), and `PerVintageActualTransformer`,
  which lifts any stateless `BaseActualTransformer` onto the vintage axis by applying
  it independently to each vintage.
- Add a `kind` tag (`"actual"` | `"forecast"`) to `TransformerTags`. The composition
  estimators (`FeatureUnion`, `FeaturePipeline`, `ColumnTransformer`) now operate on
  either kind, derive their kind from their children, and reject mixed-kind
  compositions. `ColumnTransformer` treats `vintage_time` as a protected index column.

### Breaking
- **Renamed `BaseTransformer` to `BaseActualTransformer`.** The transformer base is
  split into a private `_BaseTransformer` (shared scaffolding), `BaseActualTransformer`
  (single-axis, with the observe/rewind memory API), and `BaseForecastTransformer`
  (forecast frames, stateless). `BaseTransformer` is removed; update imports and
  subclasses to `BaseActualTransformer`.

## [0.1.0-alpha.10] - 2026-07-02

This **minor release** includes 22 commits.


### Features
- Replace exact vintage matching with as-of selection  ([#74](https://github.com/stateful-y/yohou/pull/74)) by @gtauzin
- Make y_test optional in plot_forecast and fix weighted_quantile normalization  ([#76](https://github.com/stateful-y/yohou/pull/76)) by @gtauzin
- Add cross_validate, cross_val_score, cross_val_predict  ([#68](https://github.com/stateful-y/yohou/pull/68)) by @gtauzin
- Make time-axis weighting estimator-based and tunable  ([#81](https://github.com/stateful-y/yohou/pull/81)) by @gtauzin
- Make CompositeSimilarity a real composition; rename TemporalSimilarity  ([#82](https://github.com/stateful-y/yohou/pull/82)) by @gtauzin
- Add weighter, similarity, and composition check harnesses  ([#83](https://github.com/stateful-y/yohou/pull/83)) by @gtauzin
- Add quantile calibration mode with square [0,1] axes  ([#89](https://github.com/stateful-y/yohou/pull/89)) by @gtauzin
- Add plot_nested_splits for nested CV visualization  ([#90](https://github.com/stateful-y/yohou/pull/90)) by @gtauzin
- Add configurable line_shape for all line plots  ([#92](https://github.com/stateful-y/yohou/pull/92)) by @gtauzin

### Bug Fixes
- Remap vintage_time in _predict_with_step_override  ([#67](https://github.com/stateful-y/yohou/pull/67)) by @gtauzin
- Clip forecast step columns to forecasting horizon window  ([#69](https://github.com/stateful-y/yohou/pull/69)) by @gtauzin
- Override observe_predict_interval in SplitConformalForecaster  ([#73](https://github.com/stateful-y/yohou/pull/73)) by @gtauzin
- Handle NaN features at predict time when nan_handling='drop'  ([#77](https://github.com/stateful-y/yohou/pull/77)) by @gtauzin
- Handle Pipeline steps that lack set_fit_request for sample_weight  ([#78](https://github.com/stateful-y/yohou/pull/78)) by @gtauzin
- Support scikit-learn 1.9.0 callback context and empty FeatureUnion by @gtauzin
- Route forecasted features through X_forecast  ([#80](https://github.com/stateful-y/yohou/pull/80)) by @gtauzin
- Correct CalibrationError to empirical-coverage deviation  ([#84](https://github.com/stateful-y/yohou/pull/84)) by @gtauzin
- Remediate issues across codebase  ([#85](https://github.com/stateful-y/yohou/pull/85)) by @gtauzin
- Remediate correctness findings  ([#86](https://github.com/stateful-y/yohou/pull/86)) by @gtauzin
- Remediate remaining code quality issues  ([#87](https://github.com/stateful-y/yohou/pull/87)) by @gtauzin
- Remediate more QA findings  ([#91](https://github.com/stateful-y/yohou/pull/91)) by @gtauzin

### Refactoring
- Remove pivot_forecasts (no longer used internally)  ([#75](https://github.com/stateful-y/yohou/pull/75)) by @gtauzin

### Contributors

Thanks to all contributors for this release:
- @gtauzin

## [0.1.0-alpha.9] - 2026-05-19

This **minor release** includes 2 commits.


### Bug Fixes
- Use Polars binary serialization for WASM loading  ([#65](https://github.com/stateful-y/yohou/pull/65)) by @gtauzin

### Miscellaneous Tasks
- Add workflow to regenerate dataset parquets  ([#64](https://github.com/stateful-y/yohou/pull/64)) by @gtauzin

### Contributors

Thanks to all contributors for this release:
- @gtauzin

## [0.1.0-alpha.8] - 2026-05-19

This **minor release** includes 3 commits.


### Features
- Add WASM/Pyodide support for dataset fetchers  ([#62](https://github.com/stateful-y/yohou/pull/62)) by @gtauzin
- Add `nan_handling` parameter to reduction forecasters  ([#60](https://github.com/stateful-y/yohou/pull/60)) by @gtauzin

### Bug Fixes
- Extend See Also numpydoc patch to handle MkDocs link syntax  ([#61](https://github.com/stateful-y/yohou/pull/61)) by @gtauzin

### Contributors

Thanks to all contributors for this release:
- @gtauzin

## [0.1.0-alpha.7] - 2026-05-18

This **minor release** includes 1 commit.


### Documentation
- Restructure documentation with Diataxis framework and update template  ([#34](https://github.com/stateful-y/yohou/pull/34)) by @gtauzin

### Contributors

Thanks to all contributors for this release:
- @gtauzin

## [0.1.0-alpha.6] - 2026-05-08

This **minor release** includes 9 commits.


### Features
- Allow coverage_rate=0 for point forecasts  ([#50](https://github.com/stateful-y/yohou/pull/50)) by @gtauzin
- Support global-only exogenous features with panel data  ([#52](https://github.com/stateful-y/yohou/pull/52)) by @gtauzin
- Add similarity extensions and API hardening  ([#53](https://github.com/stateful-y/yohou/pull/53)) by @gtauzin
- Add 9 forecasting metrics and 2 base scorer classes  ([#54](https://github.com/stateful-y/yohou/pull/54)) by @gtauzin
- Add X_actual, X_future, X_forecast API with step-indexed columns  ([#56](https://github.com/stateful-y/yohou/pull/56)) by @gtauzin

### Bug Fixes
- Detect LightGBM quantile alpha parameter  ([#48](https://github.com/stateful-y/yohou/pull/48)) by @gtauzin
- Pass Polars DataFrames to estimators instead of numpy arrays  ([#49](https://github.com/stateful-y/yohou/pull/49)) by @gtauzin
- Sync observed_time_ in SplitConformalForecaster observe and rewind  ([#51](https://github.com/stateful-y/yohou/pull/51)) by @gtauzin

### Refactoring
- Remove gap parameter from splitters  ([#47](https://github.com/stateful-y/yohou/pull/47)) by @gtauzin

### Contributors

Thanks to all contributors for this release:
- @gtauzin

## [0.1.0-alpha.5] - 2026-04-22

This **minor release** includes 2 commits.

### Refactoring
- Add `_fit()`, `_transform()`, and `_inverse_transform()` hooks to `BaseTransformer`, eliminating boilerplate validation and `check_is_fitted` calls from every subclass ([#45](https://github.com/stateful-y/yohou/pull/45)) by @gtauzin
- Auto-merge `_parameter_constraints` across the MRO via `__init_subclass__`, removing manual `**BaseTransformer._parameter_constraints` spreads ([#45](https://github.com/stateful-y/yohou/pull/45)) by @gtauzin
- Replace `__sklearn_tags__` overrides with declarative `_tags` dicts, merged automatically by the base class ([#45](https://github.com/stateful-y/yohou/pull/45)) by @gtauzin
- Deduplicate transformer subclasses in `preprocessing/` and `stationarity/` (net 870 lines removed) ([#45](https://github.com/stateful-y/yohou/pull/45)) by @gtauzin
- Add LaTeX equations and literature references to `BoxCoxTransformer`, `LogTransformer`, and `SeasonalDifferencing` docstrings ([#45](https://github.com/stateful-y/yohou/pull/45)) by @gtauzin

### Testing
- Add `test_compat` nox session for pinned-dependency compatibility testing ([#45](https://github.com/stateful-y/yohou/pull/45)) by @gtauzin

### Contributors

Thanks to all contributors for this release:
- @gtauzin

## [0.1.0-alpha.4] - 2026-04-22

This **minor release** includes 13 commits.


### Features
- Enable interval forecaster support in SearchCV  ([#25](https://github.com/stateful-y/yohou/pull/25)) by @gtauzin
- Add direct and dir-rec reduction strategies with example notebooks  ([#27](https://github.com/stateful-y/yohou/pull/27)) by @gtauzin
- Add categorical class-probability forecasting  ([#30](https://github.com/stateful-y/yohou/pull/30)) by @gtauzin
- Add calendar, holiday, Fourier, and time index transformers  ([#39](https://github.com/stateful-y/yohou/pull/39)) by @gtauzin
- Add VotingForecaster and VotingClassProbaForecaster  ([#40](https://github.com/stateful-y/yohou/pull/40)) by @gtauzin
- Add vintage weighting to reduction forecasters  ([#42](https://github.com/stateful-y/yohou/pull/42)) by @gtauzin
- Add MeanSeasonalNaive and MeanLagTransformer  ([#43](https://github.com/stateful-y/yohou/pull/43)) by @gtauzin

### Bug Fixes
- Skip plotting extras in test_docstrings on Python 3.14  ([#38](https://github.com/stateful-y/yohou/pull/38)) by @gtauzin

### Refactoring
- Centralize private sklearn imports into utils/_compat.py  ([#26](https://github.com/stateful-y/yohou/pull/26)) by @gtauzin
- Codebase quality overhaul  ([#28](https://github.com/stateful-y/yohou/pull/28)) by @gtauzin
- Standardize arguments, panel layout, and plotting defaults  ([#33](https://github.com/stateful-y/yohou/pull/33)) by @gtauzin
- Unify scorer API with fit(forecaster) and scoring dimensions  ([#41](https://github.com/stateful-y/yohou/pull/41)) by @gtauzin

### Contributors

Thanks to all contributors for this release:
- @gtauzin

## [0.1.0-alpha.3] - 2026-03-01

This **minor release** includes 5 commits.


### Features
- Replace bundled datasets with remote fetchers and migrate examples to PEP 723  ([#17](https://github.com/stateful-y/yohou/pull/17)) by @gtauzin
- Add `fetch_kdd_cup()` dataset (KDD Cup 2018 air quality, 270 hourly series) ([#18](https://github.com/stateful-y/yohou/pull/18)) by @gtauzin
- Add panel-aware naming utilities (`panel_aware_rename`, `panel_aware_prefix`, `panel_aware_suffix`) ([#18](https://github.com/stateful-y/yohou/pull/18)) by @gtauzin
- Add `"groupwise"` aggregation mode for scorers ([#18](https://github.com/stateful-y/yohou/pull/18)) by @gtauzin
- Add STL mode in `plot_components` via `components` list and `stl_kwargs` ([#18](https://github.com/stateful-y/yohou/pull/18)) by @gtauzin
- Add auto-detection of panel data in plotting functions ([#18](https://github.com/stateful-y/yohou/pull/18)) by @gtauzin
- Add multivariate panel faceting with per-member colours across all panel-enabled plots ([#18](https://github.com/stateful-y/yohou/pull/18)) by @gtauzin
- Add explicit `__init__` on all sklearn wrapper classes ([#18](https://github.com/stateful-y/yohou/pull/18)) by @gtauzin
- Add `check_panel_group_preservation()` transformer check ([#18](https://github.com/stateful-y/yohou/pull/18)) by @gtauzin
- Add new examples: `kdd_cup.py`, `nixtla_forecasters.py`, `nixtla_panel.py` ([#18](https://github.com/stateful-y/yohou/pull/18)) by @gtauzin

### Bug Fixes
- Fix `ColumnTransformer` sample count check (relax to `output_samples > n_samples`) ([#18](https://github.com/stateful-y/yohou/pull/18)) by @gtauzin
- Fix `_detect_multiquantile_loss()` to search with `deep=True` for nested parameters ([#18](https://github.com/stateful-y/yohou/pull/18)) by @gtauzin

### Refactoring
- Rename `inspect_locality()` to `inspect_panel()` (alias retained) ([#18](https://github.com/stateful-y/yohou/pull/18)) by @gtauzin
- Change `FeatureUnion`/`ColumnTransformer` prefix separator from `__` to `_` to avoid panel separator collisions ([#18](https://github.com/stateful-y/yohou/pull/18)) by @gtauzin
- Rewrite `_hstack()` to join on `"time"` column instead of index-based slicing ([#18](https://github.com/stateful-y/yohou/pull/18)) by @gtauzin
- Change `FourierSeasonalityForecaster` default `harmonics` from `[1, 2, 3]` to `[1]` ([#18](https://github.com/stateful-y/yohou/pull/18)) by @gtauzin
- Rename `plot_residual_time_series` to `plot_residuals`; use z-scored residuals in Q-Q plot ([#18](https://github.com/stateful-y/yohou/pull/18)) by @gtauzin
- Simplify `plot_rolling_statistics` (remove `fill_between`/`band_opacity`) ([#18](https://github.com/stateful-y/yohou/pull/18)) by @gtauzin
- Adjust `palette_yohou()` colour ordering ([#18](https://github.com/stateful-y/yohou/pull/18)) by @gtauzin
- Rename internal `reset` variables to `rewind` for consistency ([#18](https://github.com/stateful-y/yohou/pull/18)) by @gtauzin
- Remove `plot_stl_components` (merged into `plot_components`) ([#18](https://github.com/stateful-y/yohou/pull/18)) by @gtauzin
- Move `validate_plotting_params` from `plotting/_utils.py` to `utils/validate_data.py` ([#18](https://github.com/stateful-y/yohou/pull/18)) by @gtauzin

### Documentation
- Add extensions page and auto-generated API reference ([#18](https://github.com/stateful-y/yohou/pull/18)) by @gtauzin
- Add MathJax support and `mkdocs-autorefs` plugin ([#18](https://github.com/stateful-y/yohou/pull/18)) by @gtauzin
- Add gallery CSS for example notebooks ([#18](https://github.com/stateful-y/yohou/pull/18)) by @gtauzin
- Replace Sphinx-style cross-references with plain backtick style in docstrings ([#18](https://github.com/stateful-y/yohou/pull/18)) by @gtauzin
- Add comprehensive `See Also` sections across base classes and utilities ([#18](https://github.com/stateful-y/yohou/pull/18)) by @gtauzin

### Testing
- Add property-based tests (Hypothesis) with strategies module ([#18](https://github.com/stateful-y/yohou/pull/18)) by @gtauzin
- Add serialization, thread-safety, and contract test suites ([#18](https://github.com/stateful-y/yohou/pull/18)) by @gtauzin
- Add feature pipeline/union, scorer aggregation, search CV, and signal plotting tests ([#18](https://github.com/stateful-y/yohou/pull/18)) by @gtauzin
- Refactor integration tests (~14 files, ~3400 lines) ([#18](https://github.com/stateful-y/yohou/pull/18)) by @gtauzin

### Miscellaneous Tasks
- Add `hypothesis` as test dependency ([#18](https://github.com/stateful-y/yohou/pull/18)) by @gtauzin
- Add `rumdl` markdown linter to pre-commit and nox ([#18](https://github.com/stateful-y/yohou/pull/18)) by @gtauzin
- Add Justfile commands: `build-fast`, `serve-fast`, `link` ([#18](https://github.com/stateful-y/yohou/pull/18)) by @gtauzin
- Cleanup docs hooks, conformity docstrings, pre-commit config, and logos  ([#18](https://github.com/stateful-y/yohou/pull/18)) by @gtauzin
- Update copier template to v0.15.0  ([#19](https://github.com/stateful-y/yohou/pull/19)) by @gtauzin

### Contributors

Thanks to all contributors for this release:
- @gtauzin

## [0.1.0-alpha.2] - 2026-02-23

This **minor release** includes 2 commits.


### Features
- Polish forecasting plots and overhaul quickstart  ([#14](https://github.com/stateful-y/yohou/pull/14)) by @gtauzin

### Miscellaneous Tasks
- Apply copier template update (fdb4552)  ([#15](https://github.com/stateful-y/yohou/pull/15)) by @gtauzin

### Contributors

Thanks to all contributors for this release:
- @gtauzin

## [0.1.0-alpha.1] - 2026-02-20

This **minor release** includes 1 commit.

- Initial commit

### Contributors

Thanks to all contributors for this release:
- @gtauzin

## [Unreleased]

### Added
- Initial project setup
