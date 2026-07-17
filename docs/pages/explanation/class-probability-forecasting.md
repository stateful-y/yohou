# Class-Probability Forecasting

Time series forecasting conventionally targets numeric values, but many real-world
problems involve categorical outcomes: air quality levels (Good, Moderate, Unhealthy),
demand categories (Low, Normal, High), or equipment states (Running, Idle, Fault).
Class-probability forecasting extends Yohou's fit-observe-predict workflow to these
settings, producing a probability distribution over categories at each future
timestep rather than a single numeric value. The result is a per-timestep
*probability simplex*: a vector of non-negative values summing to one, representing
the model's belief about the likelihood of each class.

## Prediction Types

Yohou supports three kinds of predictions, each suited to a different target type:

- **Point predictions**: $\hat{y}_t \in \mathbb{R}$, a single numeric value per timestep.
- **Interval predictions**: $[\hat{y}_t^L, \hat{y}_t^U]$, bounds on a numeric value with coverage guarantees.
- **Class-probability predictions**: $\hat{p}_t \in \Delta^{K-1}$, a probability simplex over $K$ categorical classes.

where $\Delta^{K-1} = \{p \in \mathbb{R}^K : p_k \geq 0, \sum_{k=1}^K p_k = 1\}$.

Point and interval forecasters operate on continuous numeric targets.
Class-probability forecasters operate on categorical targets (string or integer
labels). Internally, the categories are label-encoded to integers for model
training and decoded back to the original labels at prediction time. The encoding
sorts classes alphabetically, so for classes `["sunny", "rainy", "cloudy"]` the
internal mapping is `{"cloudy": 0, "rainy": 1, "sunny": 2}`. This mapping is
stored in the `label_to_code_` fitted attribute and remains fixed for the
lifetime of the forecaster.

## The Reduction Approach

[`ClassProbaReductionForecaster`](/pages/api/generated/yohou.class_proba.ClassProbaReductionForecaster/) applies the same reduction pattern described in
[Reduction Forecasting](reduction-forecasting.md): it tabularizes a time series into
feature rows and trains a scikit-learn estimator on the result. The difference is that
the estimator is a *classifier* (any scikit-learn classifier implementing
`predict_proba()`), and the target is categorical rather than numeric.

The pipeline at fit time prepends two steps to the standard tabularization:

1. **Class discovery**: unique class labels are extracted from the target column(s) and sorted alphabetically.
2. **Label encoding**: categorical targets are converted to integer codes (e.g., `{"cloudy": 0, "rainy": 1, "sunny": 2}`).

The encoded series is then tabularized and the classifier is trained, following the
same mechanics as [`PointReductionForecaster`](/pages/api/generated/yohou.point.PointReductionForecaster/).

At prediction time, the classifier's `predict_proba()` output is mapped back to
columns named after the original class labels.

All reduction concepts (target and feature transformers, `target_as_feature`,
`step_feature_alignment`, sample weighting) work the same way as for point
forecasters. See [Reduction Forecasting](reduction-forecasting.md) for the full
treatment.

### Multi-step strategies

[`ClassProbaReductionForecaster`](/pages/api/generated/yohou.class_proba.ClassProbaReductionForecaster/) supports the `"multi-output"` and `"direct"`
reduction strategies. The `"dir-rec"` strategy available for point and interval
forecasters is not supported. Recursive probability chaining requires feeding
predicted class labels back as features, and errors in early steps compound
through the chain. For numeric targets this is manageable, but for categorical
targets a single misclassified step can shift the entire downstream feature
distribution. The `observe()` method provides a manual alternative for
step-by-step forecasting when needed.

## Predictions: Hard Labels vs. Soft Probabilities

A fitted class-probability forecaster offers two prediction methods:

**`predict_class_proba()`** returns a DataFrame with probability columns for each
class and target. For a target column `"weather"` with classes
`["cloudy", "rainy", "sunny"]`, the output contains columns
`weather_proba_cloudy`, `weather_proba_rainy`, and `weather_proba_sunny`.
Each row's probabilities sum to 1.

**`predict()`** returns hard class labels by taking the argmax of the probability
distribution: $\hat{y}_t = \arg\max_k \hat{p}_{t,k}$. This discards
calibration information and returns a single class label per timestep.

Prefer `predict_class_proba()` when downstream decisions depend on confidence
levels. A weather routing system might treat a 51% chance of rain very differently
from a 95% chance, even though both produce the same hard label.

```python
# Soft probabilities: preserves uncertainty
y_proba = forecaster.predict_class_proba()
# Columns: time, vintage_time, weather_proba_cloudy, weather_proba_rainy, weather_proba_sunny

# Hard labels: argmax only
y_pred = forecaster.predict()
# Columns: time, vintage_time, weather
```

### Multi-target outputs

When the training data contains multiple categorical columns, probability columns
are produced for each target independently:

```text
weather_proba_cloudy, weather_proba_rainy, weather_proba_sunny,
mood_proba_happy, mood_proba_sad
```

### Panel data outputs

For panel data, group prefixes are prepended with the `__` separator:

```text
location_1__weather_proba_cloudy, location_1__weather_proba_rainy,
location_2__weather_proba_cloudy, location_2__weather_proba_rainy
```

All panel groups sharing a base target name (e.g., `weather`) must have the
same set of classes.

## Streaming with Observe and Rewind

Like other Yohou forecasters, class-probability forecasters support streaming
predictions through `observe()` and `rewind()`. After fitting, call `observe()`
to feed new ground-truth observations into the forecaster's buffer, then call
`predict_class_proba()` or `predict()` to generate updated predictions that
incorporate the new data. Call `rewind()` to roll back observations.

This is particularly useful when you need step-by-step forecasting with
intermediate decisions: predict one step, observe the outcome, then predict
the next step with the updated history.

## Scoring and Evaluation

Yohou provides three families of scorers for class-probability forecasts:

**Proper scoring rules** operate directly on predicted probability distributions.
[`LogLoss`](/pages/api/generated/yohou.metrics.LogLoss/),
[`BrierScore`](/pages/api/generated/yohou.metrics.BrierScore/), and
[`RankedProbabilityScore`](/pages/api/generated/yohou.metrics.RankedProbabilityScore/)
are all uniquely minimized when the predicted probabilities match the true class
frequencies. This property makes them the most reliable choice for model
selection. Among these:

- [`LogLoss`](/pages/api/generated/yohou.metrics.LogLoss/) penalizes confident wrong predictions most harshly (predicting 0.01 for the true class is catastrophic).
- [`BrierScore`](/pages/api/generated/yohou.metrics.BrierScore/) measures the mean squared difference between predicted probabilities and one-hot encoded true labels, making it more forgiving of near-misses.
- [`RankedProbabilityScore`](/pages/api/generated/yohou.metrics.RankedProbabilityScore/) compares cumulative distributions and respects ordinal class ordering. An optional `class_order` parameter specifies the ordering explicitly.

**Hard-label scorers** convert probabilities to class labels via argmax, then
compute standard classification metrics.
[`Accuracy`](/pages/api/generated/yohou.metrics.Accuracy/),
[`Precision`](/pages/api/generated/yohou.metrics.Precision/),
[`Recall`](/pages/api/generated/yohou.metrics.Recall/), and
[`FBetaScore`](/pages/api/generated/yohou.metrics.FBetaScore/)
all discard confidence information. `Precision`, `Recall`, and `FBetaScore`
support `average` modes (`"macro"`, `"micro"`, `"weighted"`) for multiclass
targets.

**Ranking scorers** evaluate how well predicted probabilities separate classes
across decision thresholds.
[`ROCAuC`](/pages/api/generated/yohou.metrics.ROCAuC/) and
[`PRAuC`](/pages/api/generated/yohou.metrics.PRAuC/) measure
discrimination ability (whether the model assigns higher probabilities to
correct classes) without requiring well-calibrated probability values. Both use
a one-vs-rest strategy for multiclass problems.

For model selection, prefer proper scoring rules when calibration matters. Use
hard-label scorers when only the final class assignment matters. Use ranking
scorers when you care about the model's ability to distinguish between classes
regardless of calibration. See [Forecast Accuracy](forecast-accuracy.md) for the
mathematical definitions and a broader discussion of proper scoring rules.

## Calibration

A forecaster is well-calibrated if, across all timesteps where it predicts class
$k$ with probability $p$, the class $k$ actually occurs roughly $p$ fraction of
the time. Calibration is distinct from discrimination (the ability to rank likely
outcomes higher). A model can discriminate well while producing systematically
overconfident or underconfident probabilities.

Calibration matters because consumers of probability forecasts take the numbers
at face value. A logistics planner who sees 80% probability of high demand
allocates resources accordingly. If the model is overconfident and the true rate
is closer to 50%, those resource decisions are systematically wrong.

[`plot_calibration()`](/pages/api/generated/yohou.plotting.plot_calibration/) produces reliability diagrams that plot predicted
probabilities against observed frequencies. A perfectly calibrated model follows
the diagonal. Deviations above the diagonal indicate underconfidence (predicted
60%, observed 80%); deviations below indicate overconfidence.

The reduction approach inherits the calibration properties of its backbone
classifier. Some classifiers like [`GradientBoostingClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html) produce
well-calibrated probabilities by default, while others like
[`RandomForestClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html) may benefit from post-hoc calibration
(e.g., scikit-learn's [`CalibratedClassifierCV`](https://scikit-learn.org/stable/modules/generated/sklearn.calibration.CalibratedClassifierCV.html)).

## Panel Data

Class-probability forecasters support panel data natively through the
`panel_strategy` parameter. With `"global"` (the default), a single estimator is
trained across all groups. Classes are discovered per base target name (after
stripping group prefixes), so `group_0__weather` and `group_1__weather` share
the same class set. Scorers support per-group filtering and weighting through
the `groups` parameter and `"groupwise"` aggregation.

## Ensembles

[`VotingClassProbaForecaster`](/pages/api/generated/yohou.ensemble.VotingClassProbaForecaster/) combines multiple class-probability forecasters using
two methods. **Soft voting** (the default) averages class probabilities across
base forecasters, optionally with custom `weights`. It preserves calibration
better than hard voting because it operates on the full probability simplex.
**Hard voting** lets each base forecaster vote for its argmax class, and the
majority wins. All base forecasters must discover the same classes for a given
target. See [Ensemble Forecasting](ensemble-forecasting.md) for the general
theory.

## Connections

- **Preprocessing**: transformers such as [`CalendarFeatureTransformer`](/pages/api/generated/yohou.preprocessing.CalendarFeatureTransformer/), [`FourierFeatureTransformer`](/pages/api/generated/yohou.preprocessing.FourierFeatureTransformer/), and [`HolidayFeatureTransformer`](/pages/api/generated/yohou.preprocessing.HolidayFeatureTransformer/) supply exogenous features derived from the time column.
- **Hyperparameter search**: [`GridSearchCV`](/pages/api/generated/yohou.model_selection.GridSearchCV/) and [`RandomizedSearchCV`](/pages/api/generated/yohou.model_selection.RandomizedSearchCV/) accept class-proba scorers such as [`LogLoss`](/pages/api/generated/yohou.metrics.LogLoss/) as the `scoring` parameter.
- **Theory**: [Forecast Accuracy](forecast-accuracy.md) covers metric theory including proper scoring rules.
- **Tutorial**: [Class-Probability Forecasting](../tutorials/class-proba-forecasting.md) walks through a complete classification workflow.
- **Practice**: [How to Forecast with Class Probabilities](../how-to/class-probability-forecasting.md) provides step-by-step recipes.
- **API**: the full reference is at [yohou.class_proba](../api/class_proba.md).
- **Examples**: interactive notebooks are available in the [Class-Probability Examples](../examples/index.md#forecasting-models).
