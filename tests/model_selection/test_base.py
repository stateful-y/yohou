import optuna
import pytest
from sklearn.utils._param_validation import InvalidParameterError

from yohou.model_selection.base import Sampler, Storage


@pytest.mark.parametrize(
    "wrapper, estimator_class",
    [
        (Sampler, optuna.samplers.GridSampler),
        (Storage, optuna.storages.RDBStorage),
    ],
)
def test_validate_estimator_params(wrapper, estimator_class):
    estimator = wrapper(estimator_class)
    estimator._validate_params()

    with pytest.raises(InvalidParameterError):
        estimator = wrapper(optuna.pruners.MedianPruner)
        estimator._validate_params()
