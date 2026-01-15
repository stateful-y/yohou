## Pre-publishing
- Enable time_weight for forecaster and scorer
- Implement an interval scorer
- Add scorer to testing infrastructure
- Add splitter to testing infrastructure
- Simplify SearchCV, make sure it follows BaseSearchCV as closely as possible
- Make sure we can predict and score in transformed space as well as that we can access predictions
- Implement variety of splitters
- Implement variety of scorers
- Implement NestedCV (making sure we avoid retraining on splits already handled in a previous outer cv)
- Implement SkLearnTransformer
- Implement SkTimeTransformer
- Implement SkTimeForecaster
- Check hyperparameter validation and docstring consistency
- Check inline
- Make sure pre-commit is happy on all files
- Rename yohou class functions
- Decide on LICENSE

## Post-publishing
- Can we use skore/skrub to visualize CV results?
- Refactor estimator_check into a separate testing module
- Refactor reducers to enable direct and multi-output strategies
- Implement new conformal prediction approaches
- Implement ensembling
- Implement divine intervention
- Ensure feature naming works accross transformers and meta transformers
- Implement visualizations (get inspiration from pytimetk)
- Implement variety of tutorials
- Move to Narwhals
- Implement hierarchical forecasting
- Implement signal processing transformers
- Improve set up of documentation
- Create sklearn-inspired user guide
- Recreate repository and release

## Prompts
### validation
