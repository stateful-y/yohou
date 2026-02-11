"""Check functions for yohou reduction forecasters.

This module provides validation functions specific to reduction forecasters
(BaseReductionForecaster implementations).
"""


def check_estimator_parameter(forecaster) -> None:
    """Check estimator parameter is sklearn BaseEstimator.

    Parameters
    ----------
    forecaster : BaseReductionForecaster
        Reduction forecaster instance

    Raises
    ------
    AssertionError
        If estimator is not a sklearn BaseEstimator

    """
    from sklearn.base import BaseEstimator

    assert hasattr(forecaster, "estimator"), "Reduction forecaster must have 'estimator' parameter"

    estimator = forecaster.estimator
    assert isinstance(estimator, BaseEstimator), f"estimator should be sklearn BaseEstimator, got {type(estimator)}"


def check_reduction_strategy(forecaster) -> None:
    """Check reduction_strategy parameter is valid.

    Parameters
    ----------
    forecaster : BaseReductionForecaster
        Reduction forecaster instance

    Raises
    ------
    AssertionError
        If reduction_strategy is invalid

    """
    if not hasattr(forecaster, "reduction_strategy"):
        # Not all reduction forecasters expose this parameter
        return

    strategy = forecaster.reduction_strategy
    valid_strategies = ["direct", "multi-output"]

    assert strategy in valid_strategies, f"reduction_strategy should be in {valid_strategies}, got '{strategy}'"
