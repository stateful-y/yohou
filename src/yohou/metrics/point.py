import numpy as np
import polars as pl

from .base import BasePointScorer


class MAE(BasePointScorer):
    def score(self, y_truth: pl.DataFrame, y_pred: pl.DataFrame) -> float:
        y_truth, y_pred = self._validate_inputs(y_truth, y_pred)

        score = (y_truth - y_pred).select(pl.all().abs().mean())

        score_value: float = float(np.mean(score.rows()))

        return score_value


class MSE(BasePointScorer):
    def score(self, y_truth: pl.DataFrame, y_pred: pl.DataFrame) -> float:
        y_truth, y_pred = self._validate_inputs(y_truth, y_pred)

        score = (y_truth - y_pred).select(pl.all().pow(2).mean())

        score_value: float = float(np.mean(score.rows()))

        return score_value
