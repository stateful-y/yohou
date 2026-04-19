# Yohou vs Skforecast: Capability & Documentation Assessment

## A. Capability Comparison (18 Dimensions)

---

### 1. Core Forecasting Strategies

**Yohou**: Three reduction strategies in `BaseReductionForecaster`: `"multi-output"` (default, single model predicts all H steps), `"direct"` (H independent models, parallelizable via `n_jobs`), `"dir-rec"` (direct-recursive hybrid). All strategies can also be applied recursively for horizons beyond the fitted forecasting_horizon. Exposed via `PointReductionForecaster` and `IntervalReductionForecaster`. `SeasonalNaive` as baseline. Decomposition-based forecasting via `DecompositionPipeline` (trend + seasonality + residual as separate forecasters).

**Skforecast**: Strategy encoded in class choice: `ForecasterRecursive` (recursive, single model), `ForecasterDirect` (one model per step, `steps` fixed at init). `ForecasterEquivalentDate` (offset-based seasonal baseline). `ForecasterRecursiveClassifier` (classification-based). `ForecasterStats` wraps statistical models. `ForecasterRnn` for deep learning.

**Delta**: Yohou consolidates all ML strategies into one class (`PointReductionForecaster`) with a `reduction_strategy` param - cleaner API. Skforecast uses separate classes per strategy. Yohou is missing: (a) `ForecasterEquivalentDate` concept (simple offset-based baseline), (b) classification-based forecasting (`ForecasterRecursiveClassifier`). The dir-rec hybrid exists in both. Multi-output is unique to yohou's approach.

**Yohou angle**: Single-class strategy selection is more composable - same forecaster works in pipelines regardless of strategy. Multi-output strategy has no skforecast equivalent.

---

### 2. DataFrame Library

**Yohou**: Polars end-to-end. Mandatory `"time"` column (datetime type). No pandas dependency in core. Column-based panel convention (`group__column`).

**Skforecast**: Pandas exclusively. Requires `DatetimeIndex` with freq set on Series/DataFrame. Panel data via wide DataFrame, dict of Series, or long format conversion.

**Delta**: Fundamentally different data layer. Yohou gains polars performance (lazy evaluation, zero-copy, Rust backend), loses pandas ecosystem compatibility. Skforecast gains instant compatibility with the huge pandas-based ML ecosystem.

**Yohou angle**: Polars-first is a genuine performance and correctness advantage (no SettingWithCopyWarning, no index alignment bugs, deterministic behavior).

---

### 3. sklearn Integration Depth

**Yohou**: Deep integration - all estimators are `BaseEstimator` subclasses. Full metadata routing (`set_*_request()`, composite methods registered at import). `_parameter_constraints` for automatic validation. Tags system (`ForecasterTags`, `TransformerTags`, etc.). `MetaEstimatorMixin` for search. Clone/deep-copy compatible.

**Skforecast**: Estimators accept sklearn regressors as constructor params but have a custom `ForecasterBase` (not `BaseEstimator`). No metadata routing. Parameters validated internally. `get_params()` / `set_params()` supported but not via sklearn's machinery.

**Delta**: Yohou is architecturally closer to sklearn - metadata routing, `_parameter_constraints`, tag system. Skforecast is sklearn-compatible (accepts sklearn estimators) but not sklearn-native (doesn't inherit sklearn base classes). This means yohou estimators work natively in sklearn's ecosystem (Pipeline, clone, etc.) while skforecast's only work in their own ecosystem.

**Yohou angle**: First-class sklearn citizen. Metadata routing is a genuine differentiator for complex pipelines.

---

### 4. Preprocessing / Transformers

**Yohou**: 24+ transformers covering: imputation (4: SimpleImputer, SimpleTimeImputer, SeasonalImputer, TransformedSpaceKNNImputer), outlier handling (2: OutlierThresholdHandler, OutlierPercentileHandler), resampling (2: Downsampler, Upsampler), windowing (4: LagTransformer, RollingStatisticsTransformer, SlidingWindowFunctionTransformer, ExponentialMovingAverage), signal (3: NumericalFilter, NumericalDifferentiator, NumericalIntegrator), function (1: FunctionTransformer), stationarity (7: BoxCoxTransformer, LogTransformer, SeasonalDifferencing, SeasonalLogDifferencing, SeasonalReturn, AbsoluteSeasonalReturn, ASinhTransformer), scaling (9 sklearn wrappers: StandardScaler, MinMaxScaler, etc.). All are stateful with `observe`/`rewind` memory management.

**Skforecast**: 5 preprocessing classes: `RollingFeatures` (9 rolling stats with numba JIT), `TimeSeriesDifferentiator`, `DateTimeFeatureTransformer`, `QuantileBinner`, `ConformalIntervalCalibrator`. Scaling is via `transformer_y`/`transformer_exog` constructor params accepting any sklearn transformer.

**Delta**: Yohou has dramatically more built-in transformers (24+ vs 5). Skforecast relies on external libraries (feature_engine, sklearn) passed as params. Yohou provides: imputation (skforecast has none), outlier handling (skforecast has none), resampling (skforecast has none), signal processing (skforecast has none). Skforecast has `DateTimeFeatureTransformer` (calendar features) which yohou doesn't have as a dedicated class (though `FunctionTransformer` can replicate it). `QuantileBinner` and `ConformalIntervalCalibrator` are unique to skforecast.

**Yohou angle**: Rich, composable transformer ecosystem with stateful memory. Every transformer supports observe/rewind streaming.

---

### 5. Rolling / Window Features

**Yohou**: `LagTransformer` (configurable lag indices), `RollingStatisticsTransformer` (mean, std, min, max, quantiles, customizable window), `SlidingWindowFunctionTransformer` (arbitrary callable on window), `ExponentialMovingAverage` (configurable alpha/span). All support panel data and inverse transforms.

**Skforecast**: `RollingFeatures` with 9 built-in stats (mean, std, min, max, sum, median, ratio_min_max, coef_variation, ewm). Numba-JIT acceleration. Multiple `RollingFeatures` objects combinable. Passed as `window_features` constructor param.

**Delta**: Both cover similar ground. Skforecast's numba JIT gives performance edge for large datasets. Yohou's `SlidingWindowFunctionTransformer` is more flexible (arbitrary callables). Yohou's transformers are standalone and composable in pipelines; skforecast's are tightly coupled to forecaster constructors.

**Yohou angle**: Standalone composable transformers. Arbitrary function support via `SlidingWindowFunctionTransformer`.

---

### 6. Stationarity / Decomposition

**Yohou**: `DecompositionPipeline` with explicit forecaster steps for each component. Trend: `PolynomialTrendForecaster` (degree parameter). Seasonality: `PatternSeasonalityForecaster` (average season), `FourierSeasonalityForecaster` (Fourier basis, tunable n_pairs). Transform-based: `SeasonalDifferencing`, `SeasonalLogDifferencing`, `SeasonalReturn`, `AbsoluteSeasonalReturn`, `BoxCoxTransformer`, `LogTransformer`, `ASinhTransformer`. The decomposition pipeline can be nested within other pipelines and supports observe/rewind.

**Skforecast**: `differentiation` integer parameter on forecasters (auto inverse-transform on predictions). `TimeSeriesDifferentiator` class for manual use. No explicit decomposition pipeline or trend/seasonality estimators.

**Delta**: Yohou has a significantly richer decomposition system. The `DecompositionPipeline` with explicit trend/seasonality/residual forecasters has no skforecast equivalent. Skforecast's `differentiation` param is simpler but limited to integer differencing. Yohou's Fourier seasonality, pattern seasonality, polynomial trend are all forecasters that participate in the pipeline lifecycle (fit, predict, observe).

**Yohou angle**: Decomposition is a first-class architectural pattern, not just a parameter.

---

### 7. Metrics / Scoring

**Yohou**: 21 scorer classes. Point (8): MAE, MSE, RMSE, RMSSE, MAPE, SMAPE, MASE, MdAE. Interval (5): EmpiricalCoverage, MeanIntervalWidth, IntervalScore, PinballLoss, CalibrationError. Conformity (6): Residual, AbsoluteResidual, GammaResidual, AbsoluteGammaResidual, QuantileResidual, AbsoluteQuantileResidual. All support `time_weight` via metadata routing. 6 aggregation modes (stepwise, vintagewise, componentwise, groupwise, coveragewise, all). Panel-aware scoring.

**Skforecast**: Functions (not classes): MASE, RMSSE, SMAPE, `create_mean_pinball_loss`, `add_y_train_argument`. Distribution: `crps_from_predictions`, `crps_from_quantiles`. Coverage: `calculate_coverage`.

**Delta**: Yohou has far more metrics (21 classes vs ~8 functions). Yohou's class-based design enables configuration (aggregation mode, time weighting) and sklearn integration. Skforecast has CRPS (Continuous Ranked Probability Score) which yohou lacks. Yohou's conformity scorers are unique (for conformal prediction calibration). Yohou's time weighting in metrics has no skforecast equivalent.

**Yohou angle**: Class-based metrics with aggregation modes, time weighting, and panel awareness. Conformity scorers for conformal prediction.

---

### 8. Model Selection / CV

**Yohou**: 2 splitters (`ExpandingWindowSplitter`, `SlidingWindowSplitter`) with `gap` parameter. `GridSearchCV`, `RandomizedSearchCV` extending `BaseSearchCV` (which extends `BaseForecaster`). `OptunaSearchCV` via yohou-optuna extension. Search classes support the full forecaster API (predict, observe, rewind after fit). Multi-metric scoring.

**Skforecast**: `TimeSeriesFold` (very flexible: `fold_stride`, `gap`, `skip_folds`, `refit` as int or bool, `fixed_train_size`, `allow_incomplete_fold`, `differentiation`). `OneStepAheadFold` for fast tuning approximation. 9 search functions: 3 strategies (grid/random/bayesian) x 3 scopes (single/multi-series/stats). Return: `return_best=True` auto-updates forecaster. `output_file` for incremental saving. Multi-series search with `aggregate_metric`.

**Delta**: Skforecast's `TimeSeriesFold` is more feature-rich than yohou's splitters (fold_stride, skip_folds, refit-as-int, allow_incomplete_fold). `OneStepAheadFold` for fast screening has no yohou equivalent. Skforecast's function-based search (9 functions) vs yohou's class-based search is a design choice. Yohou's search classes being forecasters themselves (can predict, observe after search) is unique.

**Yohou angle**: Search result is a forecaster (can predict/observe), not just a results dict. Clean class hierarchy.

---

### 9. Interval / Probabilistic Forecasting

**Yohou**: `IntervalReductionForecaster` (quantile regression via reduction), `SplitConformalForecaster` (wraps any point forecaster, configurable conformity scorers, similarity weighting via `DistanceSimilarity`). `coverage_rates` parameter on predict_interval. 5 interval metrics + 6 conformity scorers.

**Skforecast**: 3 methods - bootstrapping (residual resampling, `store_in_sample_residuals=True`, binned residuals via `KBinsDiscretizer`), conformal (distribution-free), parametric (ForecasterStats only, `alpha` param). `predict_interval()`, `predict_quantiles()` (arbitrary quantiles), `predict_dist()` (scipy distribution fitting). Out-of-sample residuals via `set_out_sample_residuals()`. `ConformalIntervalCalibrator` for post-hoc calibration.

**Delta**: Skforecast has broader interval methods: bootstrapping (not in yohou), parametric (not in yohou), `predict_quantiles` (not in yohou), `predict_dist` (not in yohou), binned residuals for heteroscedastic data (not in yohou). Yohou has richer conformal prediction (6 conformity scorers, similarity weighting) and quantile regression via reduction. Yohou's interval forecasting participates in observe/rewind streaming; skforecast's doesn't.

**Yohou angle**: Adaptive conformal prediction with similarity weighting. Streaming interval updates via observe/rewind.

---

### 10. Panel / Multi-Series Data

**Yohou**: Native via `__` column prefix convention. `inspect_panel()`, `get_group_df()`, `select_panel_columns()` utilities. `groups` parameter on all forecaster/transformer methods. `LocalPanelForecaster` for per-group models. `ColumnForecaster` for per-column forecasters. `panel_strategy` on `BaseForecaster`. All transformers and metrics are panel-aware. Panel support is architectural (every method, every class).

**Skforecast**: `ForecasterRecursiveMultiSeries` (global model, encoding options: ordinal/ordinal_category/onehot/None). Per-series transformers, weights, differentiation. Wide/dict/long input formats with reshaping utilities. `ForecasterDirectMultiVariate` (all series as features for one target, per-series lags). `series_weights` for relative importance.

**Delta**: Different design philosophies. Yohou: panel is a property of the data (column naming), every class handles it. Skforecast: dedicated multi-series forecaster classes with their own configs. Skforecast has encoding options (ordinal, onehot) for series identity - yohou doesn't model series identity as a feature. Skforecast's per-series lags (ForecasterDirectMultiVariate) has no yohou equivalent. Yohou's universal panel support across all classes (transformers, metrics, splitters) is more pervasive.

**Yohou angle**: Panel is architectural, not a separate class. Universal support across the entire library.

---

### 11. Composition / Pipelines

**Yohou**: 7 composition classes: `DecompositionPipeline` (sequential decomposition), `FeaturePipeline` (sequential transforms), `FeatureUnion` (parallel transforms), `ColumnTransformer` (per-column transforms), `ColumnForecaster` (per-column forecasters), `ForecastedFeatureForecaster` (chain target+feature forecasters), `LocalPanelForecaster` (per-group models). All support observe/rewind. Nestable.

**Skforecast**: No pipeline abstraction. Composition via constructor params: `transformer_y`, `transformer_exog`, `window_features`, `differentiation`. Each forecaster independently manages its transforms.

**Delta**: Yohou has an explicit, rich composition system. Skforecast embeds composition in constructor params (simpler but less flexible). `DecompositionPipeline` and `ForecastedFeatureForecaster` have no skforecast equivalents. Yohou pipelines participate in the full lifecycle (fit, predict, observe, rewind).

**Yohou angle**: Composition is first-class with dedicated classes. Full lifecycle support including streaming.

---

### 12. Statistical Models

**Yohou**: None in core. Via yohou-nixtla extension: `AutoARIMAForecaster`, `ARIMAForecaster`, `AutoThetaForecaster`, `ThetaForecaster`, `AutoETSForecaster`, `AutoCESForecaster`, `HoltWintersForecaster`, `CrostonForecaster`, `NaiveForecaster`, `SeasonalNaiveForecaster`. These wrap statsforecast models with yohou's polars/sklearn API. 10 statistical model classes total.

**Skforecast**: `ForecasterStats` wrapping `Arima`, `Sarimax`, `Ets`, `Arar` (4 model wrappers). Auto variants via `order=None` (AutoARIMA) and `model='ZZZ'` (AutoETS). Accepts list of models for simultaneous fitting. Only `Sarimax` supports exogenous. Dedicated `backtesting_stats`/`grid_search_stats` functions.

**Delta**: Yohou-nixtla has more model variety (10 vs 4), including Theta, CES, HoltWinters, Croston. Skforecast has SARIMAX (exogenous support) which yohou-nixtla doesn't expose separately. Skforecast can fit multiple statistical models simultaneously. Neither has statistical models in core - both require an import.

**Yohou angle**: Wider model variety via nixtla. All models inherit yohou's full API (observe/rewind, panel data, metadata routing).

---

### 13. Deep Learning

**Yohou**: None in core. Via yohou-nixtla extension: `NBEATSForecaster`, `NHITSForecaster`, `PatchTSTForecaster`, `TimesNetForecaster`, `MLPForecaster` (5 classes wrapping neuralforecast).

**Skforecast**: `ForecasterRnn` with `create_and_compile_model` helper. LSTM/GRU/RNN layers via Keras. Stacked layers, exogenous architecture. Conformal intervals only.

**Delta**: Yohou-nixtla has more modern architectures (N-BEATS, N-HiTS, PatchTST, TimesNet) vs skforecast's classic RNNs (LSTM/GRU). Skforecast allows custom Keras model construction; yohou-nixtla uses pre-built architectures from neuralforecast. Neither is in core.

**Yohou angle**: State-of-the-art neural architectures via nixtla (transformer-based, not just RNN).

---

### 14. Streaming / Online Learning

**Yohou**: Core architectural feature. `observe(y, X, groups)` adds data without refitting. `rewind(y, X, groups)` truncates memory to `observation_horizon`. `observe_predict(y, X)` and `observe_predict_interval(y, X, coverage_rates)` are atomic combined operations. All transformers have `observe`/`rewind`/`observe_transform`/`rewind_transform`. Memory management is bounded. Panel-selective observation (`groups` param).

**Skforecast**: No equivalent. `refit=int` in backtesting controls re-training frequency. `last_window` on some forecasters stores recent data. No incremental update API.

**Delta**: This is yohou's most distinctive capability. Skforecast requires full refit for model updates. Yohou's observe/rewind pattern enables: rolling production forecasts without refitting, bounded memory consumption, selective panel group updates, atomic observe+predict operations.

**Yohou angle**: Unique architectural differentiator. No equivalent in skforecast or most forecasting libraries.

---

### 15. Explainability

**Yohou**: Not present. No feature importance extraction. No SHAP integration.

**Skforecast**: `get_feature_importances()` method on Recursive, Direct, RecursiveMultiSeries, DirectMultiVariate forecasters. SHAP value integration documented in user guides.

**Delta**: Skforecast has built-in feature importance and SHAP. Yohou has nothing. For reduction forecasters, feature importances could be extracted from the underlying sklearn estimator manually, but no convenience method exists.

---

### 16. Drift Detection

**Yohou**: Not present. Mentioned briefly in "Practical Issues" explanation page.

**Skforecast**: `RangeDriftDetector` (lightweight out-of-range check, fast for real-time). `PopulationDriftDetector` (KS/Chi-Square/Jensen-Shannon statistical tests, configurable chunk_size and threshold). Pipeline integration pattern documented.

**Delta**: Skforecast has two dedicated drift detectors. Yohou has none.

---

### 17. Deployment / Production

**Yohou**: `observe_predict` workflow documented in one example notebook. No save/load guide. No deployment architecture docs. The observe/rewind pattern naturally supports production use (bounded memory, incremental updates).

**Skforecast**: Save/load forecaster guide. Production deployment documentation.

**Delta**: Skforecast has dedicated deployment documentation. Yohou's observe/rewind pattern is arguably better for production (streaming without refitting) but lacks documentation around serialization, deployment patterns, and monitoring.

---

### 18. Feature Selection

**Yohou**: Not present as a dedicated module.

**Skforecast**: `select_features` and `select_features_multiseries` functions. Wraps sklearn selectors (RFECV, SelectFromModel). Returns selected lags, window features, and exog. `select_only` param, `force_inclusion`, `subsample` for efficiency.

**Delta**: Skforecast provides time-series-aware feature selection. Yohou has nothing dedicated, though sklearn's feature selection can be applied manually to tabularized data.

---

## B. Documentation Comparison

| Dimension | Skforecast Docs | Yohou Docs | Gap |
|---|---|---|---|
| **1. Forecasting Strategies** | Dedicated guide per forecaster type + "Choosing a Forecaster" decision support + recursive vs direct comparison table | Tutorial (Your First Forecast), Explanation (Forecasting), 2 example notebooks | Missing: decision support for choosing strategy, explicit direct vs multi-output comparison |
| **2. DataFrame Library** | Implicit (pandas everywhere) | Explanation (Core Concepts covers polars contract) | Covered |
| **3. sklearn Integration** | Implicit | Explanation (Core Concepts, Advanced Topics on metadata routing) | Could document metadata routing patterns more explicitly |
| **4. Preprocessing** | Feature Engineering guide (calendar, cyclical, rolling, scaling) | Explanation (Preprocessing), 8 example notebooks | Missing: calendar/datetime feature generation guide (DateTimeFeatureTransformer equivalent) |
| **5. Rolling/Window** | RollingFeatures in Feature Engineering guide | Preprocessing explanation + window_transformers notebook | Covered |
| **6. Stationarity** | Brief (differentiation param only) | Explanation (Stationarity), 4 notebooks | Yohou is better documented here |
| **7. Metrics** | API ref only | How-to (Evaluate Accuracy), Explanation (Forecast Accuracy), 5 notebooks | Covered well |
| **8. Model Selection** | Hyperparameter Optimization guide (comprehensive) | Explanation (Model Selection), 6 notebooks | Missing: OneStepAheadFold concept, fast screening strategies |
| **9. Intervals** | Prediction Intervals guide (3 methods, method comparison, binned residuals) | Explanation (Interval Forecasting), 5 notebooks | Missing: bootstrapping intervals guide, method comparison table, binned residuals concept |
| **10. Panel Data** | Forecasting Multiple Series guide (encoding, per-series config, reshaping) | How-to (Panel Data), 6+ panel notebooks | Missing: encoding/series identity concept, per-series configuration patterns |
| **11. Composition** | None (implicit via constructor params) | 7 composition notebooks | Yohou is better documented here |
| **12. Statistical Models** | Statistical Models guide (Auto-ARIMA, Auto-ETS, SARIMAX) | Nixtla notebook only | Missing: dedicated statistical models guide |
| **13. Deep Learning** | Deep Learning Forecasting guide (architecture, custom models) | Nixtla notebook only | Missing: dedicated DL guide |
| **14. Streaming** | None | observe_predict_workflow notebook | Missing: dedicated streaming/online learning explanation page |
| **15. Explainability** | Feature importance + SHAP in user guides | None | Missing entirely |
| **16. Drift Detection** | Dedicated guide (2 detectors, integration pattern) | Brief mention in Practical Issues | Missing entirely |
| **17. Deployment** | Save/load + production deployment guide | Brief mention in Advanced Topics | Missing: production guide, serialization guide |
| **18. Feature Selection** | Feature Selection guide (RFECV, multi-series, subsample) | None | Missing entirely |

---

## C. Yohou Unique Strengths (Not Found in Skforecast)

1. **Polars-first**: Zero-copy operations, lazy evaluation, Rust backend. No pandas SettingWithCopyWarning, no index alignment bugs.

2. **observe/rewind streaming**: Core architectural pattern for production forecasting without refitting. Bounded memory. Panel-selective updates. No equivalent in skforecast.

3. **sklearn metadata routing**: Native `time_weight` propagation, `groups` routing. First-class sklearn citizen with `_parameter_constraints`, tags system.

4. **Stateful transformers with memory**: Every transformer supports observe/rewind. Memory is bounded to `observation_horizon`. Enables streaming preprocessing.

5. **DecompositionPipeline**: Trend + seasonality + residual as composable forecasters. Each component is a first-class forecaster participating in the full lifecycle.

6. **ForecastedFeatureForecaster**: Chain target forecaster with feature forecasters. Target predictions feed as exogenous to feature forecasters.

7. **Rich composition system**: 7 composition classes (vs skforecast's 0). FeaturePipeline, FeatureUnion, ColumnTransformer, ColumnForecaster, LocalPanelForecaster - all with observe/rewind.

8. **Signal processing transformers**: NumericalFilter, NumericalDifferentiator, NumericalIntegrator. Engineering time series domain.

9. **Rich imputation**: 4 imputers including SeasonalImputer and TransformedSpaceKNNImputer. Skforecast has none.

10. **Outlier handling**: OutlierThresholdHandler, OutlierPercentileHandler. Skforecast has none.

11. **Resampling transformers**: Downsampler, Upsampler for frequency conversion. Skforecast has none.

12. **Conformity scorer system**: 6 conformity scorers for calibrating conformal prediction (Residual, AbsoluteResidual, GammaResidual, etc.).

13. **Time-weighted metrics**: `time_weight` support in all scorers via metadata routing. Configure how recent vs older errors contribute.

14. **Multi-output reduction strategy**: Single model predicts all H steps simultaneously. Not available in skforecast.

15. **Dir-rec hybrid strategy**: Direct-recursive hybrid available as a strategy option. Skforecast has recursive and direct only.

16. **31 plotly visualization functions** (vs skforecast's 7): Comprehensive exploration, diagnostics, forecasting, evaluation, model selection, and signal processing plots.

17. **Testing infrastructure**: 5 check generators yielding 91+ systematic checks. Reusable across all estimator types.

18. **Panel-as-architecture**: Panel data support is pervasive across every class, not limited to dedicated multi-series forecasters.

---

## D. yohou-skforecast Extension Mapping

Candidates for a `yohou-skforecast` workspace package that wraps skforecast implementations:

### Forecasters

| Skforecast Class | yohou-skforecast Class | Mapping |
|---|---|---|
| `ForecasterEquivalentDate` | `EquivalentDateForecaster` | Extends `BasePointForecaster`. Offset-based seasonal baseline. Params: `offset` (timedelta or str). Simple implementation - may be better as a yohou-core naive forecaster. |
| `ForecasterRnn` | `RnnForecaster` | Extends `BasePointForecaster`, wraps `ForecasterRnn`. Polars I/O adapter. Params: `model`, `steps`, `lags`, `transformer_series`. Manage pandas/polars conversion at boundaries. |
| `ForecasterRecursiveClassifier` | `ClassificationForecaster` | Extends `BasePointForecaster`. Wraps skforecast's classifier with polars I/O. Requires predict_proba. Niche use case. |

### Utilities

| Skforecast Class | yohou-skforecast Class | Mapping |
|---|---|---|
| `RangeDriftDetector` | `RangeDriftDetector` | New base class `BaseDriftDetector` in yohou-skforecast (or yohou core). Wraps skforecast's implementation with polars I/O. Methods: `fit(y, X)`, `detect(y, X) -> DriftReport`. |
| `PopulationDriftDetector` | `PopulationDriftDetector` | Same pattern. Wraps skforecast's statistical tests (KS, Chi-Square, Jensen-Shannon). Polars I/O. |
| `ConformalIntervalCalibrator` | `IntervalCalibrator` | Extends `BaseTransformer`. Post-hoc interval calibration. May be better in yohou core as it complements `SplitConformalForecaster`. |
| `QuantileBinner` | `QuantileBinner` | Extends `BaseTransformer`. Bin continuous target for binned residual intervals. Wraps skforecast's implementation. |
| `DateTimeFeatureTransformer` | `DateTimeFeatureTransformer` | Extends `BaseTransformer`. Extracts calendar features (hour, day, month, weekday, etc.). Could also be a yohou-core transformer. |

### Functions

| Skforecast Function | yohou-skforecast | Mapping |
|---|---|---|
| `select_features` | `select_features(forecaster, X, y)` | Function that wraps skforecast's feature selection with polars-to-pandas conversion. Returns selected feature names. |
| `backtesting_*` functions | Not needed | Yohou's CV + search already covers this use case natively. |

### Not Mapped (use yohou-nixtla instead)

- `ForecasterStats` (Arima, Sarimax, Ets, Arar) - already covered by yohou-nixtla's statsforecast integration with more models.

---

## E. Categorized Action List

### Implement in yohou core

| # | Item | Rationale | Scope |
|---|---|---|---|
| 1 | **EquivalentDateForecaster** (offset baseline) | Universal baseline forecaster. Simple: predict = value from equivalent past date. More principled than just `SeasonalNaive`. | New class in `point/naive.py` |
| 2 | **DateTimeFeatureTransformer** | Calendar feature extraction (hour, day_of_week, month, etc.) is a universal need. FunctionTransformer can do it but a dedicated class is cleaner. | New class in `preprocessing/` |
| 3 | **get_feature_importances()** on reduction forecasters | Trivial to implement - delegate to underlying sklearn estimator's `feature_importances_` or `coef_`. Universal expectation. | Method on `BaseReductionForecaster` |
| 4 | **predict_quantiles()** on interval forecasters | Return arbitrary quantiles instead of just lower/upper bounds. Natural extension of `predict_interval`. | Method on `BaseIntervalForecaster` |
| 5 | **OneStepAheadFold** equivalent | Fast hyperparameter screening without recursive multi-step prediction. Significant speedup for tuning. | New splitter in `model_selection/split.py` |
| 6 | **CRPS metric** (Continuous Ranked Probability Score) | Standard probabilistic forecast evaluation metric. Missing from yohou's otherwise comprehensive metrics. | New class in `metrics/interval.py` |
| 7 | **Bootstrapping interval method** | Residual bootstrapping as an alternative to conformal/quantile regression for prediction intervals. Standard approach. | New forecaster or method on existing interval forecasters |
| 8 | **IntervalCalibrator** transformer | Post-hoc interval calibration (like skforecast's `ConformalIntervalCalibrator`). Complements `SplitConformalForecaster`. | New class in `interval/` or `preprocessing/` |
| 9 | **Binned residuals** for heteroscedastic intervals | Group residuals by predicted value level for better-calibrated intervals. Works with bootstrapping. | Enhancement to interval forecasters |
| 10 | **Out-of-sample residuals management** | `set_out_sample_residuals()` / `get_out_sample_residuals()` for interval methods that use residuals. | Methods on interval forecasters |

### Wrap in yohou-skforecast

| # | Item | Rationale |
|---|---|---|
| 1 | **RangeDriftDetector** | Lightweight real-time drift check. Useful for production monitoring. Thin polars wrapper. |
| 2 | **PopulationDriftDetector** | Statistical drift tests (KS, Chi-Square, Jensen-Shannon). Batch monitoring. Thin polars wrapper. |
| 3 | **ForecasterRnn** (LSTM/GRU wrapper) | For users wanting classic RNN architectures. Complements yohou-nixtla's modern architectures. |
| 4 | **select_features / select_features_multiseries** | Feature selection with time-series awareness. Wraps skforecast's implementation with polars I/O. |

### Add to docs

| # | Item | Details |
|---|---|---|
| 1 | **Choosing a Forecasting Strategy** (explanation page) | Decision support: when to use multi-output vs direct vs dir-rec. Comparison table. Currently users must guess. |
| 2 | **Production deployment guide** (how-to) | Serialization, observe/rewind in production, monitoring loop, memory management. The observe pattern is yohou's killer feature but poorly documented for production. |
| 3 | **Streaming / Online Learning** (explanation page) | Dedicated page explaining observe/rewind architecture, bounded memory, when to refit vs observe. Currently implicit in examples. |
| 4 | **Statistical models with yohou-nixtla** (how-to) | Dedicated guide for ARIMA, ETS, Theta via nixtla. Currently only example notebooks. |
| 5 | **Deep learning with yohou-nixtla** (how-to) | Dedicated guide for N-BEATS, PatchTST, etc. Currently only example notebooks. |
| 6 | **Interval method comparison** (explanation page) | When to use conformal vs quantile regression vs (future) bootstrapping. Coverage vs width tradeoffs. |
| 7 | **Feature importance / Explainability** (how-to) | How to extract feature importances from reduction forecasters. SHAP integration guide. |
| 8 | **Troubleshooting common errors** | Expand existing troubleshooting page with time-series-specific issues (frequency, exog coverage, panel naming). |
| 9 | **Calendar / datetime features** (how-to) | How to create calendar features until a dedicated DateTimeFeatureTransformer exists. |
| 10 | **Multi-series encoding strategies** | Document yohou's panel approach vs encoding series identity as a feature. When each is appropriate. |

### Skip (deliberate scope)

| # | Item | Reason |
|---|---|---|
| 1 | **ForecasterRecursiveClassifier** | Niche use case. Classification-based forecasting doesn't align with yohou's regression-first approach. Can be added later if demand emerges. |
| 2 | **9 backtesting functions** | Yohou's CV + search architecture already handles this cleanly. Function-based backtesting conflicts with class-based design. |
| 3 | **pandas compatibility layer** | Would undermine polars-first philosophy. Users needing pandas can convert at boundaries. |
| 4 | **Per-forecaster-type search functions** | Yohou's unified `GridSearchCV`/`RandomizedSearchCV` work for all forecaster types. Having separate functions per type is a skforecast limitation, not a feature. |
| 5 | **Dark theme / theme system for plots** | Yohou uses plotly (interactive, theme-agnostic). Users can customize plotly themes directly. |
| 6 | **backtesting_gif_creator** | Nice-to-have but not core functionality. |
| 7 | **Multiple simultaneous model fitting** (ForecasterStats pattern) | Yohou's approach is explicit - fit each forecaster separately. Implicit multi-model fitting adds complexity without clear benefit. |
| 8 | **Long/wide/dict reshaping utilities** | Yohou's `__` convention handles panel data without format conversion. These utilities solve a problem yohou doesn't have. |
| 9 | **QuantileBinner** | Niche preprocessing. Can be achieved with polars `cut()` or sklearn's `KBinsDiscretizer`. |
| 10 | **freeze_params for statistical models** | Specific to ForecasterStats auto-search pattern. yohou-nixtla handles auto model selection differently. |

---

## F. What NOT to Adopt from Skforecast

1. **Pandas-first data layer**: Skforecast requires DatetimeIndex with freq. Yohou's polars + explicit time column is architecturally superior (no implicit state in index, no freq inference issues).

2. **Function-based search/backtesting**: 9 separate functions (grid_search_forecaster, grid_search_forecaster_multiseries, grid_search_stats, bayesian_search_forecaster, ...) vs yohou's unified class hierarchy. The function explosion is a maintenance burden skforecast carries.

3. **Per-class strategy splitting**: ForecasterRecursive vs ForecasterDirect as separate classes. Yohou's `reduction_strategy` parameter is cleaner and more composable.

4. **ForecasterBase (custom base class)**: Skforecast doesn't inherit sklearn's BaseEstimator. This means skforecast forecasters can't participate in sklearn's ecosystem (Pipeline, clone, GridSearchCV). Yohou's full sklearn integration is a design advantage.

5. **Constructor-embedded composition**: `transformer_y`, `transformer_exog`, `window_features`, `differentiation` as constructor params. This conflates model definition with preprocessing and makes complex pipelines hard to express. Yohou's explicit composition classes are more flexible.

6. **store_in_sample_residuals flag**: Requiring users to set `store_in_sample_residuals=True` at fit time to enable prediction intervals is error-prone. Yohou should automatically store what's needed.

7. **Series encoding as forecaster param**: `encoding='ordinal'/'onehot'` on ForecasterRecursiveMultiSeries mixes data encoding with model definition. Encoding should be a preprocessing step.

8. **Frequency requirement on index**: Skforecast requires `.freq` on DatetimeIndex. Yohou infers frequency from data via `check_interval_consistency`, which is more robust.

9. **Separate stats ecosystem**: Skforecast has entirely separate backtesting and search functions for statistical models (`backtesting_stats`, `grid_search_stats`). Yohou's unified API (all forecasters share the same search/CV interface) is a clear advantage.
