"""Preprocessing transformers for stationarization and feature engineering."""

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
from .window import (
    ExponentialMovingAverage,
    LagTransformer,
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
]
