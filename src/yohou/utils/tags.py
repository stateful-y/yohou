"""Tags system for yohou estimators.

This module provides a tag system for estimator capabilities and requirements,
following the same spirit as sklearn.utils.Tags but tailored for time series
forecasting with yohou.
"""

from dataclasses import dataclass
from typing import Literal

__all__ = [
    "CLASS_PROBA",
    "ForecasterTags",
    "INTERVAL",
    "InputTags",
    "POINT",
    "POINT_INTERVAL",
    "ScorerTags",
    "SimilarityTags",
    "SplitterTags",
    "Tags",
    "TargetTags",
    "TransformerTags",
]

# Forecaster type constants
POINT: frozenset[str] = frozenset({"point"})
INTERVAL: frozenset[str] = frozenset({"interval"})
CLASS_PROBA: frozenset[str] = frozenset({"class_proba"})
POINT_INTERVAL: frozenset[str] = frozenset({"point", "interval"})


@dataclass
class InputTags:
    """Tags describing input requirements.

    Parameters
    ----------
    requires_time_column : bool, default=True
        Whether the estimator requires a "time" column in input DataFrames.
        All yohou estimators require this for temporal ordering.
    pairwise : bool, default=False
        Whether the estimator expects pairwise inputs (e.g., distance matrices).
    allow_nan : bool, default=False
        Whether NaN values are allowed in inputs.
    min_value : float or None, default=None
        Minimum value constraint for inputs. If not None, inputs must be
        greater than this value (exclusive). For example, log transforms
        require min_value=0.0 (inputs must be > 0).

    """

    requires_time_column: bool = True
    pairwise: bool = False
    allow_nan: bool = False
    min_value: float | None = None


@dataclass
class TargetTags:
    """Tags describing target (y) requirements.

    Parameters
    ----------
    required : bool, default=False
        Whether fit() requires the y parameter. False for unsupervised transformers.
    min_value : float or None, default=None
        Minimum value constraint for target. If not None, target values must be
        greater than this value (exclusive).

    """

    required: bool = False
    min_value: float | None = None


@dataclass
class TransformerTags:
    """Tags specific to time series transformers.

    Parameters
    ----------
    stateful : bool, default=False
        Whether the transformer maintains state across observations.
        True if the transformer uses an observation horizon mechanism.
    invertible : bool, default=False
        Whether the transformer has an inverse_transform method that can
        revert transformations.
    preserves_dtype : bool, default=False
        Whether the transformer preserves input data types.

    """

    stateful: bool = False
    invertible: bool = False
    preserves_dtype: bool = False


@dataclass
class ForecasterTags:
    """Tags specific to time series forecasters.

    Parameters
    ----------
    forecaster_type : frozenset of str or None, default=None
        Set of forecaster capabilities. Valid elements are ``"point"``,
        ``"interval"``, and ``"class_proba"``. Use the module-level
        constants ``POINT``, ``INTERVAL``, ``CLASS_PROBA``, and
        ``POINT_INTERVAL`` for convenience. ``None`` means not
        determined or not applicable.
    stateful : bool, default=False
        Whether the forecaster maintains state across observations.
        True if the forecaster uses an observation horizon mechanism.
    uses_reduction : bool, default=False
        Whether the forecaster uses reduction to supervised learning
        (converts time series forecasting to tabular regression).
    uses_target_transformer : bool, default=False
        Whether the forecaster uses a target transformer to transform
        the target time series before fitting.
    uses_feature_transformer : bool, default=False
        Whether the forecaster uses a feature transformer to transform
        the feature time series before fitting.
    supports_panel_data : bool, default=True
        Whether the forecaster can handle panel data (multiple time series
        with prefixed column names).
    supports_time_weight : bool, default=False
        Whether the forecaster supports time-based weighting of observations.
    supports_vintage_weight : bool, default=False
        Whether the forecaster supports vintage-based weighting of observations.
    ignores_exogenous : bool, default=False
        Whether the forecaster ignores exogenous features ``X``.
        When ``True``, the forecaster does not use exogenous features
        and ``X=None`` is always valid at fit time.  When ``False``
        (default), ``X=None`` is only valid when ``target_as_feature``
        is not ``None`` (so the target itself provides features).
    tracks_observations : bool, default=True
        Whether the forecaster tracks observations in the standard way
        (update extends _y_observed, reset replaces it). Set to False for
        meta-forecasters like DecompositionPipeline that delegate observation tracking
        to child forecasters with custom residual-based logic.

    """

    forecaster_type: frozenset[str] | None = None
    stateful: bool = False
    uses_reduction: bool = False
    uses_target_transformer: bool = False
    uses_feature_transformer: bool = False
    supports_panel_data: bool = True
    supports_time_weight: bool = False
    supports_vintage_weight: bool = False
    ignores_exogenous: bool = False
    tracks_observations: bool = True

    _VALID_FORECASTER_TYPE_ELEMENTS: frozenset[str] = frozenset({"point", "interval", "class_proba"})

    def __post_init__(self) -> None:
        """Validate forecaster_type contains only valid elements."""
        if self.forecaster_type is not None:
            if not isinstance(self.forecaster_type, frozenset):
                raise TypeError(
                    f"forecaster_type must be a frozenset or None, got {type(self.forecaster_type).__name__}"
                )
            invalid = self.forecaster_type - self._VALID_FORECASTER_TYPE_ELEMENTS
            if invalid:
                raise ValueError(
                    f"forecaster_type contains invalid elements {invalid}. "
                    f"Valid elements are {self._VALID_FORECASTER_TYPE_ELEMENTS}"
                )


@dataclass
class ScorerTags:
    """Tags specific to forecasting metrics/scorers.

    Parameters
    ----------
    prediction_type : {"point", "interval", "class_proba"} or None, default=None
        Type of prediction this scorer evaluates:
        - "point": Evaluates point forecasts
        - "interval": Evaluates prediction intervals
        - "class_proba": Evaluates class-probability forecasts
        - None: Not determined or not applicable
    lower_is_better : bool, default=True
        Whether lower scores indicate better performance (e.g., MeanAbsoluteError, MSE).
        False for metrics where higher is better (e.g., R²).
    requires_calibration : bool, default=False
        Whether the scorer requires calibration data from fit().
    symmetric : bool, default=False
        Whether the scorer produces symmetric (absolute) conformity
        scores.  Symmetric scorers yield equal width intervals on both
        sides of the point prediction.  Asymmetric scorers (the default)
        allow different widths above and below.
    multiplicative : bool, default=False
        Whether the scorer normalises residuals by the prediction
        magnitude (e.g. gamma residuals).  When ``True``, weighted
        interval reconstruction scales quantiles by
        ``(prediction + epsilon)``.

    """

    prediction_type: Literal["point", "interval", "class_proba"] | None = None
    lower_is_better: bool = True
    requires_calibration: bool = False
    symmetric: bool = False
    multiplicative: bool = False


@dataclass
class SimilarityTags:
    """Tags specific to similarity measures for interval forecasting.

    Parameters
    ----------
    symmetric : bool, default=True
        Whether the similarity measure is symmetric:
        similarity(A, B) == similarity(B, A).
    requires_predictions : bool, default=True
        Whether the similarity measure requires predictions (y_pred)
        in addition to actuals (y) during fit.
    produces_weights : bool, default=True
        Whether the similarity measure produces weights for conformity scores.

    """

    symmetric: bool = True
    requires_predictions: bool = True
    produces_weights: bool = True


@dataclass
class SplitterTags:
    """Tags specific to cross-validation splitters.

    Parameters
    ----------
    splitter_type : {"expanding", "sliding", "gap"} or None, default=None
        Type of cross-validation splitter:
        - "expanding": Expanding window (train set grows over time)
        - "sliding": Sliding window (fixed train set size)
        - "gap": Wraps another splitter to add gap between train and test
        - None: Not determined or not applicable
    supports_panel_data : bool, default=False
        Whether the splitter can handle panel data (multiple time series
        with prefixed column names).
    produces_non_overlapping_tests : bool, default=True
        Whether test sets from different splits are guaranteed not to overlap.
        True for expanding/sliding windows, may be False for custom splitters.
    stateful : bool, default=False
        Whether the splitter maintains state across split() calls.

    """

    splitter_type: Literal["expanding", "sliding", "gap"] | None = None
    supports_panel_data: bool = False
    produces_non_overlapping_tests: bool = True
    stateful: bool = False


@dataclass
class Tags:
    """Metadata tags for yohou estimators.

    This dataclass holds metadata about estimator capabilities and requirements,
    following the same pattern as sklearn.utils.Tags but tailored for time series
    forecasting.

    Parameters
    ----------
    estimator_type : {"transformer", "forecaster", "scorer", "similarity", "splitter"} or None, default=None
        Type of estimator. Determines which specialized tags are relevant.
    requires_fit : bool, default=True
        Whether the estimator needs to be fitted before use.
    non_deterministic : bool, default=False
        Whether the estimator produces non-deterministic results (e.g., uses
        random sampling).
    input_tags : InputTags or None, default=None
        Tags describing input requirements. Automatically initialized if None
    target_tags : TargetTags or None, default=None
        Tags describing target requirements. Automatically initialized if None.
    transformer_tags : TransformerTags or None, default=None
        Tags specific to transformers. Only relevant when estimator_type="transformer".
        Automatically initialized if None and estimator_type="transformer".
    forecaster_tags : ForecasterTags or None, default=None
        Tags specific to forecasters. Only relevant when estimator_type="forecaster".
        Automatically initialized if None and estimator_type="forecaster".
    scorer_tags : ScorerTags or None, default=None
        Tags specific to scorers/metrics. Only relevant when estimator_type="scorer".
        Automatically initialized if None and estimator_type="scorer".
    similarity_tags : SimilarityTags or None, default=None
        Tags specific to similarity measures. Only relevant when estimator_type="similarity".
        Automatically initialized if None and estimator_type="similarity".
    splitter_tags : SplitterTags or None, default=None
        Tags specific to cross-validation splitters. Only relevant when estimator_type="splitter".
        Automatically initialized if None and estimator_type="splitter".

    """

    estimator_type: Literal["transformer", "forecaster", "scorer", "similarity", "splitter"] | None = None
    requires_fit: bool = True
    non_deterministic: bool = False
    input_tags: InputTags | None = None
    target_tags: TargetTags | None = None
    transformer_tags: TransformerTags | None = None
    splitter_tags: SplitterTags | None = None
    forecaster_tags: ForecasterTags | None = None
    scorer_tags: ScorerTags | None = None
    similarity_tags: SimilarityTags | None = None

    def __post_init__(self):
        """Initialize nested tags after dataclass initialization."""
        # Always initialize input_tags and target_tags
        if self.input_tags is None:
            self.input_tags = InputTags()
        if self.target_tags is None:
            self.target_tags = TargetTags()

        # Initialize type-specific tags based on estimator_type
        if self.estimator_type == "transformer" and self.transformer_tags is None:
            self.transformer_tags = TransformerTags()
        elif self.estimator_type == "forecaster" and self.forecaster_tags is None:
            self.forecaster_tags = ForecasterTags()
        elif self.estimator_type == "scorer" and self.scorer_tags is None:
            self.scorer_tags = ScorerTags()
        elif self.estimator_type == "similarity" and self.similarity_tags is None:
            self.similarity_tags = SimilarityTags()
        elif self.estimator_type == "splitter" and self.splitter_tags is None:
            self.splitter_tags = SplitterTags()
