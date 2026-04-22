"""Preprocessing transformers for stationarization and feature engineering."""

from .calendar import CalendarFeatureTransformer, HolidayFeatureTransformer
from .function import FunctionTransformer
from .imputation import (
    SeasonalImputer,
    SimpleImputer,
    SimpleTimeImputer,
    TransformedSpaceKNNImputer,
)
from .outlier import (
    OutlierPercentileHandler,
    OutlierThresholdHandler,
)
from .resampling import (
    Downsampler,
    Upsampler,
)
from .signal import (
    NumericalDifferentiator,
    NumericalFilter,
    NumericalIntegrator,
)
from .sklearn_base import (
    SklearnScaler,
    SklearnTransformer,
)
from .sklearn_wrappers import (
    MaxAbsScaler,
    MinMaxScaler,
    Normalizer,
    PolynomialFeatures,
    PowerTransformer,
    QuantileTransformer,
    RobustScaler,
    SplineTransformer,
    StandardScaler,
)
from .time_features import FourierFeatureTransformer, TimeIndexTransformer
from .window import (
    ExponentialMovingAverage,
    LagTransformer,
    MeanLagTransformer,
    RollingStatisticsTransformer,
    SlidingWindowFunctionTransformer,
)

__all__ = [
    # Function transformers
    "FunctionTransformer",
    "SlidingWindowFunctionTransformer",
    "RollingStatisticsTransformer",
    "ExponentialMovingAverage",
    # Signal processing
    "NumericalIntegrator",
    "NumericalDifferentiator",
    "NumericalFilter",
    # Resampling
    "Downsampler",
    "Upsampler",
    # Windowing
    "LagTransformer",
    "MeanLagTransformer",
    # Sklearn scalers
    "SklearnScaler",
    "StandardScaler",
    "MinMaxScaler",
    "RobustScaler",
    "MaxAbsScaler",
    # Sklearn transformers
    "SklearnTransformer",
    "Normalizer",
    "PolynomialFeatures",
    "PowerTransformer",
    "QuantileTransformer",
    "SplineTransformer",
    # Imputation
    "SimpleImputer",
    "TransformedSpaceKNNImputer",
    "SimpleTimeImputer",
    "SeasonalImputer",
    # Outlier handling
    "OutlierThresholdHandler",
    "OutlierPercentileHandler",
    # Calendar and time features
    "CalendarFeatureTransformer",
    "HolidayFeatureTransformer",
    "FourierFeatureTransformer",
    "TimeIndexTransformer",
]
