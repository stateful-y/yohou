"""Implementation of conformal forecasters."""

from typing import List, Literal

import polars as pl
from pydantic import StrictFloat, StrictInt
from sklearn.base import clone
from sklearn.model_selection import train_test_split

from yohou.base import Tags
from yohou.metrics import BaseConformityScorer, Residual
from yohou.point_forecaster import BasePointForecaster, SeasonalNaive
from yohou.utils import validate_data

from .base import BaseIntervalForecaster, BaseSimilarity


class SplitConformalForecaster(BaseIntervalForecaster):
    """Split conformal forecaster implementation.

    Parameters
    ----------
    point_forecaster : BasePointForecaster, default=SeasonalNaive()
        Point forecaster used to generate point predictions.
    calibration_size : int >= 1, default=100
        Number of observations to use for calibration.
    conformity_scorer : BaseConformityScorer, default=Residual()
        Scorer used to compute conformity scores.
    similarity : BaseSimilarity or None, default=None
        Similarity measure to weight conformity scores.

    """

    def __sklearn_tags__(self) -> Tags:
        """Get estimator tags.

        Returns
        -------
        Tags
            Estimator tags with forecaster_type set to "both" since this
            forecaster produces both point predictions and intervals.

        """
        tags = super().__sklearn_tags__()
        # SplitConformal wraps a point forecaster and adds intervals
        tags.forecaster_tags.forecaster_type = "both"
        return tags

    def __init__(
        self,
        point_forecaster: BasePointForecaster = SeasonalNaive(),
        calibration_size: StrictInt = 100,
        conformity_scorer: BaseConformityScorer = Residual(),
        similarity: BaseSimilarity | None = None,
    ):
        BaseIntervalForecaster.__init__(self)

        self.point_forecaster = point_forecaster
        self.conformity_scorer = conformity_scorer
        self.similarity = similarity
        self.calibration_size = calibration_size

    def fit(
        self,
        y: pl.DataFrame,
        X: pl.DataFrame | None = None,
        forecasting_horizon: StrictInt = 1,
        coverage_rates: List[StrictFloat] | None = None,
        **params,
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
        coverage_rates : list of float or None, default=None
            Coverage rates for the prediction intervals. If None, uses ``[0.95]``
        **params : dict
            Metadata to route to nested estimators.

        Returns
        -------
        self

        """
        forecasting_horizon, self.fit_coverage_rates_ = self._validate_fit_params(
            forecasting_horizon, coverage_rates
        )

        # TODO: No _pre_fit call?
        self.fit_forecasting_horizon_ = forecasting_horizon

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
        forecasting_horizon: StrictInt | None = None,
        panel_group_names: list[str] | None = None,
        predict_transformed: bool = False,
        **params,
    ) -> pl.DataFrame:
        """Predicts the model forecasting horizon from the observation horizon.

        Parameters
        ----------
        X : pl.DataFrame or None, default=None
            Exogenous feature time series.
        forecasting_horizon : int >= 1 or None, default=None
            Horizon to forecast. If None, uses ``fit_forecasting_horizon_``.
        panel_group_names : list of str or None, default=None
            Group prefixes for panel data:
            - If None: predict for all groups
            - If list of str: predict only for the specified panel groups
            Parameter is ignored if the forecaster was not fitted on panel data.
        predict_transformed : bool, default=False
            If ``True``, the predictions are returned in the transformed space.
        **params : dict
            Metadata to route to nested estimators.

        Returns
        -------
        pl.DataFrame
            Predicted time series.

        """
        _, X, panel_group_names = validate_data(
            self,
            y=None,
            X=X,
            reset=False,
            panel_group_names=panel_group_names,
            check_continuity=False,
        )

        return self.point_forecaster_.predict(
            X=X,
            forecasting_horizon=forecasting_horizon,
            panel_group_names=panel_group_names,
            predict_transformed=predict_transformed,
        )

    def predict_interval(
        self,
        X: pl.DataFrame | None = None,
        forecasting_horizon: StrictInt | None = None,
        coverage_rates: list[float] | None = None,
        strategy: Literal["mean", "median", "point"] | None = None,
        panel_group_names: list[str] | None = None,
        **params,
    ) -> pl.DataFrame:
        """Predicts an interval according to coverage rates.

        Parameters
        ----------
        X : pl.DataFrame or None, default=None
            Exogenous feature time series.
        forecasting_horizon : int >= 1 or None, default=None
            Horizon to forecast. If None, uses ``fit_forecasting_horizon_``.
        coverage_rates : list of floats or None, default=None
            Coverage rates for the prediction intervals. If None, uses ``fit_coverage_rates_``.
        strategy : {"mean", "median", "point"} or None, default=None
            Strategy for updating with new point observations:
            - "mean": use the mean of the interval bounds as point observation
            - "median": use the median of the interval bounds as point observation
            - "point": use the point forecast directly (if available)
            If None, defaults to "mean".
        panel_group_names : list of str or None, default=None
            Group prefixes for panel data:
            - If None: predict for all groups
            - If list of str: predict only for the specified panel groups
            Parameter is ignored if the forecaster was not fitted on panel data.
        **params : dict
            Metadata to route to nested estimators.

        Returns
        -------
        pl.DataFrame
            Predicted time series with interval bounds.

        """
        _, X, panel_group_names = validate_data(
            self,
            y=None,
            X=X,
            reset=False,
            panel_group_names=panel_group_names,
            check_continuity=False,
        )

        forecasting_horizon, coverage_rates = self._validate_predict_params(
            forecasting_horizon, coverage_rates
        )

        y_pred = self.point_forecaster_.predict(X=X).drop("observed_time")

        y_pred_intervals = pl.DataFrame()
        for step in range(1, 1 + forecasting_horizon):
            y_pred_step = y_pred[[step]]
            conformity_scorer_step = self.conformity_scorers_[f"step_{step}"]
            conformity_scores_step = self.conformity_scores_.filter(pl.col("step") == step)

            y_pred_intervals_step = pl.DataFrame()
            for coverage_rate in coverage_rates:
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
