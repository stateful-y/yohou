import abc

import numpy as np
import polars as pl
from sklearn.base import BaseEstimator


class BaseScorer(BaseEstimator, metaclass=abc.ABCMeta):  # type: ignore[misc]
    @property
    def prediction_type(self) -> str:
        return str(self._prediction_type)

    def _validate_inputs(
        self, y_truth: pl.DataFrame, y_pred: pl.DataFrame
    ) -> tuple[pl.DataFrame, pl.DataFrame]:
        y_truth = y_truth.join(
            y_pred.rename({"predicted_time": "time"})["time"],
            on="time",
        )

        y_pred = y_pred.filter(pl.col("predicted_time").is_in(y_truth["time"].implode()))

        y_truth = y_truth.drop("time")
        y_pred = y_pred.drop("observed_time", "predicted_time")

        return y_truth, y_pred

    def fit(self, y_train: pl.DataFrame) -> "BaseScorer":
        return self

    @abc.abstractmethod
    def score(self, y_truth: pl.DataFrame, y_pred: pl.DataFrame) -> float:
        raise NotImplementedError()

    def __call__(self, y_truth: pl.DataFrame, y_pred: pl.DataFrame) -> float:
        return self.score(y_truth, y_pred)


class BasePointScorer(BaseScorer, metaclass=abc.ABCMeta):
    _prediction_type = "point"


class BaseIntervalScorer(BaseScorer, metaclass=abc.ABCMeta):
    _prediction_type = "point"


class BaseConformityScorer(BaseScorer, metaclass=abc.ABCMeta):
    @staticmethod
    def _compute_assymetric_quantiles(
        conformity_scores: pl.DataFrame, coverage_rate: float
    ) -> tuple[pl.DataFrame, pl.DataFrame]:
        lower_quantile = np.quantile(conformity_scores, coverage_rate / 2.0, method="lower")

        upper_quantile = np.quantile(conformity_scores, 1 - coverage_rate / 2.0, method="upper")

        return lower_quantile, upper_quantile

    @staticmethod
    def _compute_symetric_quantiles(
        conformity_scores: pl.DataFrame, coverage_rate: float
    ) -> pl.DataFrame:
        quantile = np.quantile(conformity_scores, 1 - coverage_rate, method="lower")

        return quantile

    @staticmethod
    def _format_y_pred_interval(
        lower_bound: pl.DataFrame, upper_bound: pl.DataFrame, coverage_rate: float
    ) -> pl.DataFrame:
        lower_bound.columns = [f"{col}_lower_{coverage_rate}" for col in lower_bound.columns]
        upper_bound.columns = [f"{col}_upper_{coverage_rate}" for col in upper_bound.columns]

        y_pred_interval = pl.concat([lower_bound, upper_bound], how="horizontal")

        return y_pred_interval

    @abc.abstractmethod
    def inverse_score(
        self, y_pred: pl.DataFrame, conformity_scores: pl.DataFrame, coverage_rate: float
    ) -> pl.DataFrame:
        raise NotImplementedError()
