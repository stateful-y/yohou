import polars as pl

from .base import BaseConformityScorer


class Residual(BaseConformityScorer):
    _prediction_type = "point"

    def score(self, y_truth: pl.DataFrame, y_pred: pl.DataFrame) -> pl.DataFrame:
        self._validate_inputs(y_truth, y_pred)

        scores = y_truth - y_pred

        return scores

    def inverse_score(
        self, y_pred: pl.DataFrame, conformity_scores: pl.DataFrame, coverage_rate: float
    ) -> pl.DataFrame:
        lower_quantile, upper_quantile = self._compute_assymetric_quantile(
            conformity_scores, coverage_rate
        )
        lower_bound, upper_bound = y_pred + lower_quantile, y_pred + upper_quantile

        y_pred_interval = self._format_y_pred_interval(lower_bound, upper_bound, coverage_rate)

        return y_pred_interval


class AbsoluteResidual(Residual):
    def score(self, y_truth: pl.DataFrame, y_pred: pl.DataFrame) -> pl.DataFrame:
        self._validate_inputs(y_truth, y_pred)

        scores = (y_truth - y_pred).abs()

        return scores

    def inverse_score(
        self, y_pred: pl.DataFrame, conformity_scores: pl.DataFrame, coverage_rate: float
    ) -> pl.DataFrame:
        quantile = self._compute_symetric_quantile(conformity_scores, coverage_rate)
        lower_bound, upper_bound = y_pred - quantile, y_pred + quantile

        y_pred_interval = self._format_y_pred_interval(lower_bound, upper_bound, coverage_rate)

        return y_pred_interval


class GammaResidual(BaseConformityScorer):
    _prediction_type = "point"

    def __init__(self, epsilon: float = 1e-8) -> None:
        BaseConformityScorer.__init__(self)

        self.epsilon = epsilon

    def score(self, y_truth: pl.DataFrame, y_pred: pl.DataFrame) -> pl.DataFrame:
        self._validate_inputs(y_truth, y_pred)

        scores = (y_truth - y_pred) / (y_pred + self.epsilon)

        return scores


class AbsoluteGammaResidual(GammaResidual):
    def score(self, y_truth: pl.DataFrame, y_pred: pl.DataFrame) -> pl.DataFrame:
        scores = GammaResidual.score(self, y_truth, y_pred).abs()

        return scores


class QuantileResidual(BaseConformityScorer):
    _prediction_type = "interval"

    def score(self, y_truth: pl.DataFrame, y_pred: pl.DataFrame) -> pl.DataFrame:
        pass


class AbsoluteQuantileResidual(BaseConformityScorer):
    _prediction_type = "interval"

    def score(self, y_truth: pl.DataFrame, y_pred: pl.DataFrame) -> pl.DataFrame:
        pass
