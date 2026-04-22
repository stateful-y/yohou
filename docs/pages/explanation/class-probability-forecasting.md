# Class-Probability Forecasting

Time series forecasting conventionally targets numeric values, but many real-world
problems involve categorical outcomes: air quality levels (Good, Moderate, Unhealthy),
demand categories (Low, Normal, High), or equipment states (Running, Idle, Fault).
Class-probability forecasting extends the familiar fit-predict workflow to these
settings, producing calibrated probability distributions over categories at each
future timestep rather than point estimates.

Yohou models this through [`BaseClassProbaForecaster`](/pages/api/generated/yohou.class_proba.base.BaseClassProbaForecaster/) and its concrete implementation
[`ClassProbaReductionForecaster`](/pages/api/generated/yohou.class_proba.reduction.ClassProbaReductionForecaster/), which convert the forecasting problem into tabular
supervised classification. The result is a per-timestep probability simplex: a vector
of non-negative values summing to one, representing the model's belief about the
likelihood of each class.

## From Numeric to Categorical Prediction

Point forecasters produce a single numeric value per timestep. Interval forecasters
produce bounds that capture uncertainty in that value. Class-probability forecasters
occupy a different dimension: they produce a probability distribution across
discrete categories.

The three prediction types form a natural hierarchy:

- **Point predictions**: $\hat{y}_t \in \mathbb{R}$ - single value per timestep
- **Interval predictions**: $[\hat{y}_t^L, \hat{y}_t^U]$ - bounds with coverage guarantees
- **Class-probability predictions**: $\hat{p}_t \in \Delta^{K-1}$ - probability simplex over $K$ classes

where $\Delta^{K-1} = \{p \in \mathbb{R}^K : p_k \geq 0, \sum_{k=1}^K p_k = 1\}$ is the probability simplex.

The key difference is that class-probability forecasters work with categorical targets
(string or integer labels) rather than continuous numeric values. The internal machinery
encodes these categories to integers for model training and decodes predictions back
to the original labels.

## The Reduction Strategy

[`ClassProbaReductionForecaster`](/pages/api/generated/yohou.class_proba.reduction.ClassProbaReductionForecaster/) follows the same reduction pattern as
[`PointReductionForecaster`](/pages/api/generated/yohou.point.reduction.PointReductionForecaster/): it transforms the sequential forecasting problem into a
standard tabular supervised learning problem. The insight is that any sklearn
classifier can be repurposed as a categorical time series forecaster by constructing
appropriate features from the historical sequence.

Internally, `ClassProbaReductionForecaster` encodes categorical targets to integers,
tabularizes the time series into feature rows, fits a standard sklearn classifier,
and maps `predict_proba()` outputs back to the original class labels. This means
any classifier that implements `predict_proba()` can serve as the backbone:
`LogisticRegression` (the default), `RandomForestClassifier`,
`GradientBoostingClassifier`, or even neural network wrappers.

The appeal of this approach is leverage: the rich ecosystem of sklearn classifiers
becomes immediately available for time series classification without writing custom
sequential models. The trade-off is that the lag-based feature construction imposes
a fixed-window view of history, which may miss long-range dependencies that
recurrent or attention-based architectures capture natively.

### Multi-step strategies

Two strategies handle multi-step-ahead forecasting:

**Multi-output** (`reduction_strategy="multi-output"`): A single model predicts all
$H$ horizon steps simultaneously. This is faster but cannot model dependencies
between forecast steps.

**Direct** (`reduction_strategy="direct"`): $H$ independent models, one per horizon
step. Each model specializes in predicting a specific lead time. This is slower
(especially with large $H$) but allows each model to adapt to the characteristics
of its specific forecast distance.

The dir-rec strategy available for point and interval forecasters is not supported
here because recursive probability chaining introduces calibration complications
that outweigh the benefits.

## Predictions: Hard Labels vs. Soft Probabilities

A fitted class-probability forecaster offers two prediction methods:

**`predict_class_proba()`** returns a DataFrame with probability columns for each
class and target. For a target column `"weather"` with classes
`["sunny", "rainy", "cloudy"]`, the output contains columns
`weather_proba_sunny`, `weather_proba_rainy`, and `weather_proba_cloudy`.
Each row's probabilities sum to 1.

**`predict()`** returns hard class labels by taking the argmax of the probability
distribution: $\hat{y}_t = \arg\max_k \hat{p}_{t,k}$. This is a convenience that
discards calibration information.

Prefer `predict_class_proba()` when downstream decisions depend on confidence
levels. A weather routing system might treat a 51% chance of rain very differently
from a 95% chance, even though both produce the same hard label.

```python
# Soft probabilities - preserves uncertainty
y_proba = forecaster.predict_class_proba()
# DataFrame with weather_proba_sunny, weather_proba_rainy, weather_proba_cloudy

# Hard labels - argmax only
y_pred = forecaster.predict()
# DataFrame with "weather" column: ["sunny", "rainy", ...]
```

## Scoring and Evaluation

Evaluating class-probability forecasts requires metrics that account for the full
probability distribution, not just the argmax label. Yohou provides three
class-probability scorers: [`Accuracy`](/pages/api/generated/yohou.metrics.class_proba.Accuracy/),
[`LogLoss`](/pages/api/generated/yohou.metrics.class_proba.LogLoss/), and
[`BrierScore`](/pages/api/generated/yohou.metrics.class_proba.BrierScore/).

The important distinction is between accuracy and the other two. Accuracy only
looks at the argmax label and ignores confidence entirely. A model that predicts
51% sunny and one that predicts 99% sunny receive the same accuracy score if
sunny is correct. Log loss and Brier score are both *proper scoring rules*,
meaning they are uniquely minimized when the predicted probabilities match the
true class frequencies. This makes them the right choice for model selection
whenever the probability estimates themselves matter to downstream decisions.

Between the two proper scoring rules, log loss penalizes confident wrong
predictions more harshly (predicting 0.01 for the true class is catastrophic),
while Brier score is more forgiving of near-misses. Log loss is generally
preferred for decision-support systems where overconfidence is dangerous.
Brier score works well when you want a bounded, interpretable measure of
calibration quality.

See [Forecast Accuracy](forecast-accuracy.md) for the mathematical definitions
and a broader discussion of proper scoring rules.

## Calibration

A forecaster is well-calibrated if, across all timesteps where it predicts class $k$
with probability $p$, the class $k$ actually occurs roughly $p$ fraction of the time.
Calibration is separate from discrimination (the ability to distinguish between
classes). A model can have excellent discrimination (it ranks likely outcomes higher)
but poor calibration (its stated probabilities are systematically off).

Calibration matters because consumers of probability forecasts take the numbers
at face value. A logistics planner who sees 80% probability of high demand and
allocates resources accordingly is implicitly trusting that "80%" means roughly
4 out of 5 similar situations result in high demand. If the model is overconfident
and the true rate is closer to 50%, those resource decisions are systematically
wrong.

You can assess calibration visually using [`plot_calibration()`](/pages/api/generated/yohou.plotting.evaluation.plot_calibration/), which plots predicted
probabilities against observed frequencies. A perfectly calibrated model follows the
diagonal. Deviations above the diagonal indicate underconfidence (the model says 60%
but the true rate is 80%); deviations below indicate overconfidence.

The reduction approach tends to inherit the calibration properties of its backbone
classifier. Some classifiers like `GradientBoostingClassifier` produce well-calibrated
probabilities by default, while others like `SVC` or `RandomForestClassifier` may
benefit from post-hoc calibration (e.g., sklearn's `CalibratedClassifierCV`).

## Connections

Class-probability forecasting interacts with several other parts of Yohou:

- **Preprocessing**: Use [`CalendarFeatureTransformer`](/pages/api/generated/yohou.preprocessing.CalendarFeatureTransformer/), [`FourierFeatureTransformer`](/pages/api/generated/yohou.preprocessing.FourierFeatureTransformer/),
  or [`HolidayFeatureTransformer`](/pages/api/generated/yohou.preprocessing.HolidayFeatureTransformer/) as the `feature_transformer` to supply exogenous
  features derived from the time column
- **Ensemble**: [`VotingClassProbaForecaster`](/pages/api/generated/yohou.ensemble.voting_class_proba.VotingClassProbaForecaster/) combines multiple class-probability
  forecasters using soft voting (averaged probabilities) or hard voting (majority
  class)
- **Model selection**: [`GridSearchCV`](/pages/api/generated/yohou.model_selection.search.GridSearchCV/) and [`RandomizedSearchCV`](/pages/api/generated/yohou.model_selection.search.RandomizedSearchCV/) accept class-proba
  scorers (e.g., [`LogLoss`](/pages/api/generated/yohou.metrics.class_proba.LogLoss/)) as the `scoring` parameter
- **Panel data**: Class-probability forecasters support panel data natively through
  the `panel_strategy` parameter

### Related pages

- [Ensemble Forecasting](ensemble-forecasting.md): combining class-probability forecasters
- [Forecast Accuracy](forecast-accuracy.md): metric theory including proper scoring rules
- [How to Forecast Categorical Time Series](../how-to/classification-forecasting.md): step-by-step guide
- [API Reference: yohou.class_proba](../api/class_proba.md)
- [Class-Probability Examples](../examples/class_proba.md)
