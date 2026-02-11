## Pre-publishing
- Check scorers by a making tutorial
- Check metadata routing

- Add target_forecaster to ColumnForecaster? What was the plan without it?
- Implement SklearnTransformer

- Check docstring consistency
- Check inline TODOs
- Make sure pre-commit is happy on all files

- Rename yohou class functions

## Post-publishing

- Make sure we can predict and score in transformed space as well as that we can access predictions
- Implement NestedCV (making sure we avoid retraining on splits already handled in a previous outer cv)
- Only if we have/want stateful scorers? Make sure scorer define an obsevred_time and ensure continuity between fit and score (and inverse_score?)
- Check skore for CV reporting
- Make scorers fit method take forecasting_horizon for raw score to include the forecasting step it corresponds to. Allows scorers to have a forecasting_steps class parameter to subselect the forecasting steps to score on.
- Can we use skore/skrub to visualize CV results?
- Refactor reducers to enable direct and multi-output strategies
- Implement new conformal prediction approaches
- Implement ensembling
- Implement divine intervention
- Create PanelForecaster that acts like ColumnForecaster but per panel group in parallel
- Ensure feature naming works accross transformers and meta transformers
- Implement visualizations (get inspiration from pytimetk)
- Implement variety of tutorials
- Move to Narwhals
- Implement hierarchical forecasting
- Implement signal processing transformers
- Improve set up of documentation
- Create sklearn-inspired user guide
- Recreate repository and release
