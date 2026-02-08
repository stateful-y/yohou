import pytest
from sklearn.utils._param_validation import InvalidParameterError

# Skip all tests in this module if optuna is not installed
pytest.importorskip("optuna")

import optuna
from yohou.model_selection.optuna import Sampler, Storage


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
