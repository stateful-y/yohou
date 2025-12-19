import optuna

from yohou.base import BaseWrapper


class Sampler(BaseWrapper):
    _estimator_name = "sampler"
    _estimator_base_class = optuna.samplers.BaseSampler

    def __init__(self, sampler: type = optuna.samplers.TPESampler, **params: dict[str, object]) -> None:
        BaseWrapper.__init__(self, estimator_class=sampler, **params)


class Storage(BaseWrapper):
    _estimator_name = "storage"
    _estimator_base_class = optuna.storages.BaseStorage

    def __init__(self, storage: type = optuna.storages.RDBStorage, **params: dict[str, object]) -> None:
        BaseWrapper.__init__(self, estimator_class=storage, **params)
