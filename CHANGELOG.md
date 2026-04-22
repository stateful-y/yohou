# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).


## [0.1.0-alpha.4] - 2026-04-22

This **minor release** includes 13 commits.


### Features
- Enable interval forecaster support in SearchCV  ([#25](https://github.com/stateful-y/yohou/pull/25)) by @gtauzin
- Add direct and dir-rec reduction strategies with example notebooks  ([#27](https://github.com/stateful-y/yohou/pull/27)) by @gtauzin
- Add categorical class-probability forecasting  ([#30](https://github.com/stateful-y/yohou/pull/30)) by @gtauzin
- Add calendar, holiday, Fourier, and time index transformers  ([#39](https://github.com/stateful-y/yohou/pull/39)) by @gtauzin
- Add VotingForecaster and VotingClassProbaForecaster  ([#40](https://github.com/stateful-y/yohou/pull/40)) by @gtauzin
- Add vintage and step weighting to reduction forecasters  ([#42](https://github.com/stateful-y/yohou/pull/42)) by @gtauzin
- Add MeanSeasonalNaive and MeanLagTransformer  ([#43](https://github.com/stateful-y/yohou/pull/43)) by @gtauzin

### Bug Fixes
- Skip plotting extras in test_docstrings on Python 3.14  ([#38](https://github.com/stateful-y/yohou/pull/38)) by @gtauzin

### Refactoring
- Centralize private sklearn imports into utils/_compat.py  ([#26](https://github.com/stateful-y/yohou/pull/26)) by @gtauzin
- Codebase quality overhaul  ([#28](https://github.com/stateful-y/yohou/pull/28)) by @gtauzin
- Standardize arguments, panel layout, and plotting defaults  ([#33](https://github.com/stateful-y/yohou/pull/33)) by @gtauzin
- Unify scorer API with fit(forecaster) and scoring dimensions  ([#41](https://github.com/stateful-y/yohou/pull/41)) by @gtauzin

### Miscellaneous Tasks
- Delete file committed by mistake by @gtauzin

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
