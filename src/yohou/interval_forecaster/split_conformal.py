"""Implementation of conformal forecasters."""

from typing import List, Literal

import polars as pl
from pydantic import StrictFloat, StrictInt
from sklearn.base import clone
from sklearn.model_selection import train_test_split

from yohou.metrics import BaseConformityScorer, Residual
from yohou.point_forecaster import BasePointForecaster, SeasonalNaive

from .base import BaseIntervalForecaster, BaseSimilarity


class SplitConformalForecaster(BaseIntervalForecaster):
    """Split conformal forecaster implementation.

    Parameters
    ----------
    coverage_rates: list of floats, default=[0.05]
        List of miscoverage levels to generate intervals for.

    """

    @property
    def prediction_types(self) -> set[str]:
        """Get prediction types produced by this forecaster.

        Returns
        -------
        set of str
            {"point", "interval"} - produces both point predictions and intervals.

        """
        return {"point", "interval"}

    def __init__(
        self,
        point_forecaster: BasePointForecaster = SeasonalNaive(),
        calibration_size: StrictInt = 100,
        coverage_rates: List[StrictFloat] = [0.05],
        conformity_scorer: BaseConformityScorer = Residual(),
        similarity: BaseSimilarity | None = None,
        update_strategy: Literal["average", "constant"] = "average",
    ):
        BaseIntervalForecaster.__init__(
            self,
            coverage_rates=coverage_rates,
            update_strategy=update_strategy,
        )

        self.point_forecaster = point_forecaster
        self.conformity_scorer = conformity_scorer
        self.similarity = similarity
        self.calibration_size = calibration_size

    def fit(
        self,
        y: pl.DataFrame,
        X: pl.DataFrame | None = None,
        forecasting_horizon: StrictInt = 1,
    ) -> "SplitConformalForecaster":
        """Fits the forecaster and returns it.

        Parameters
        ----------
        y : pl.DataFrame
            Target time series.

        X : pl.DataFrame or None, default=None
            Exogenous feature time series.

        forecasting_horizon : int >= 1, default=1
            Horizon to forecast.

        Returns
        -------
        self

        """
        y_train, y_calib, X_train, X_calib = train_test_split(
            y, X, test_size=self.calibration_size, shuffle=False
        )

        self.point_forecaster_ = clone(self.point_forecaster).fit(
            y=y_train,
            X=X_train,
            forecasting_horizon=forecasting_horizon,
        )

        # Use None to delegate to fit_forecasting_horizon_
        y_pred_calib = self.point_forecaster_.update_predict(
            y=y_calib,
            X=X_calib,
            forecasting_horizon=None,
            stride=None,
            predict_transformed=False,
        )

        y_pred_calib = y_pred_calib.drop("observed_time")

        conformity_scorers = {}
        conformity_scores = pl.DataFrame()
        for step in range(1, 1 + forecasting_horizon):
            y_pred_calib_step = y_pred_calib[step::forecasting_horizon]
            y_truth_step = y.filter(pl.col("time") == y["time"])

            conformity_scorer_step = clone(self.conformity_scorer).fit()
            conformity_scores_step = conformity_scorer_step.score(
                y_truth=y_truth_step, y_pred=y_pred_calib_step
            )

            conformity_scores_step = conformity_scores_step.with_columns(step=1 + step)
            conformity_scores = pl.concat([conformity_scores, conformity_scores_step])

            conformity_scorers[f"step_{step}"] = conformity_scorer_step

        self.conformity_scorers_ = conformity_scorers
        self.conformity_scores_ = conformity_scores

        similarities = {}
        weights = pl.DataFrame()
        if self.similarity is not None:
            for step in range(1, 1 + forecasting_horizon):
                y_pred_calib_step = y_pred_calib[step::forecasting_horizon]
                y_truth_step = y.filter(pl.col("time") == y["time"])

                similarity_step = clone(self.similarity)
                similarity_step.fit()

                weights_step = similarity_step.predict()

                weights_step = weights_step.with_columns(step=1 + step)
                weights = pl.concat([weights, weights_step])

                similarities[f"step_{step}"] = similarity_step

            self.similarities_ = similarities
            self.weights_ = weights

        return self

    def predict(
        self,
        X: pl.DataFrame | None = None,
        forecasting_horizon: StrictInt = 1,
        panel_group: str | None = None,
        predict_transformed: bool = False,
    ) -> pl.DataFrame:
        """Predicts the model forecasting horizon from the observation horizon.

        Parameters
        ----------
        X : pl.DataFrame or None, default=None
            Exogenous feature time series.

        forecasting_horizon : int >= 1, default=1
            Horizon to forecast.

        panel_group : str or None, default=None
            For panel data (local_group_names_ is not None):
            - If None: predict for all groups (default behavior)
            - If str: predict only for the specified group (cross-learning)
            For global data: parameter is ignored.

        predict_transformed : bool, default=False
            If ``True``, the predictions are returned in the transformed space.

        Returns
        -------
        pl.DataFrame
            Predicted interval time series.
        """
        y_pred = self.point_forecaster_.predict(X=X).drop("observed_time")

        y_pred_intervals = pl.DataFrame()
        for step in range(1, 1 + forecasting_horizon):
            y_pred_step = y_pred[[step]]
            conformity_scorer_step = self.conformity_scorers_[f"step_{step}"]
            conformity_scores_step = self.conformity_scores_.filter(pl.col("step") == step)

            y_pred_intervals_step = pl.DataFrame()
            for coverage_rate in self.coverage_rates:
                y_pred_interval_rate_step = conformity_scorer_step.inverse_score(
                    y_pred=y_pred_step,
                    conformity_scores=conformity_scores_step,
                    coverage_rate=coverage_rate,
                )

                y_pred_intervals_step = pl.concat(
                    [y_pred_intervals_step, y_pred_interval_rate_step],
                    how="horizontal",
                )

            y_pred_intervals = pl.concat([y_pred_intervals, y_pred_intervals_step])

        return y_pred_intervals
