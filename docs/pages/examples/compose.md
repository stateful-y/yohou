# Composition

### Pipeline Composition ([View](/examples/compose/pipeline_composition/) | [Editable](/examples/compose/pipeline_composition/edit/))

**Nested Pipeline Patterns**

Master Yohou's composition building blocks: `FeaturePipeline` (sequential transforms, horizon = sum), `FeatureUnion` (parallel transforms), `DecompositionPipeline` with features, and multi-level nesting. The foundation for building complex forecasting systems.

### Column Transformer ([View](/examples/compose/column_transformer/) | [Editable](/examples/compose/column_transformer/edit/))

**Column-Wise Feature Transformation**

Apply different transformers to different columns with `ColumnTransformer`. Includes panel-aware column selection for mixed-type DataFrames.

### Feature Union ([View](/examples/compose/feature_union/) | [Editable](/examples/compose/feature_union/edit/))

**Parallel Feature Engineering with FeatureUnion**

Combine multiple transformers in parallel using `FeatureUnion`. The resulting `observation_horizon` is the maximum across all children. This example explains why and demonstrates the pattern.

### Decomposition Variations ([View](/examples/compose/decomposition_variations/) | [Editable](/examples/compose/decomposition_variations/edit/))

**DecompositionPipeline Configurations**

Explore `DecompositionPipeline` variants: 2-component and 3-component decomposition, `target_transformer` integration, and panel decomposition. Shows how to customize decomposition to match your data's structure.

### ForecastedFeatureForecaster (Advanced) ([View](/examples/compose/forecasted_feature_advanced/) | [Editable](/examples/compose/forecasted_feature_advanced/edit/))

**Advanced Feature Forecasting Strategies**

Deep dive into `ForecastedFeatureForecaster` strategies (`actual`, `predicted`, `rewind`), `split_ratio` tuning, and multivariate feature chains. For when simple feature forecasting isn't enough.

### LocalPanelForecaster ([View](/examples/compose/local_panel_forecaster/) | [Editable](/examples/compose/local_panel_forecaster/edit/))

**Independent Per-Group Forecasting**

Fit independent forecasters per panel group with `LocalPanelForecaster`. Uses `sklearn.base.clone()` for isolation, supports parallel execution, and allows selective group training.

### Panel Pipelines ([View](/examples/compose/panel_pipelines/) | [Editable](/examples/compose/panel_pipelines/edit/))

**All Composition Patterns with Panel Data**

Comprehensive showcase of every composition meta-estimator on panel data: `ColumnForecaster`, `DecompositionPipeline`, `FeaturePipeline`, `FeatureUnion`, and `ColumnTransformer`.
