"""Implementation of conformal forecasters."""

import numbers
from typing import Literal

import numpy as np
import polars as pl
from pydantic import StrictFloat, StrictInt
from sklearn.base import clone
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_is_fitted

from yohou.metrics import AbsoluteResidual, BaseConformityScorer, Residual
from yohou.point import BasePointForecaster, SeasonalNaive
from yohou.utils import POINT_INTERVAL, Tags, validate_forecaster_data
from yohou.utils._compat import Interval, _fit_context

from .base import BaseIntervalForecaster, BaseSimilarity
from .utils import weighted_quantile

__all__ = ["SplitConformalForecaster"]


class SplitConformalForecaster(BaseIntervalForecaster):
    """Split conformal forecaster implementation.

    Wraps a point forecaster and calibrates prediction intervals using
    split conformal prediction.  A held-out calibration set is used to
    compute conformity scores whose quantiles define the interval width.

    Parameters
    ----------
    point_forecaster : BasePointForecaster, default=SeasonalNaive()
        Point forecaster used to generate point predictions.
    calibration_size : int >= 1, default=100
        Number of observations to use for calibration.
    conformity_scorer : BaseConformityScorer, default=Residual()
        Scorer used to compute conformity scores.
    similarity : BaseSimilarity or None, default=None
        Similarity measure used to weight conformity scores.
    panel_strategy : {"global", "multivariate"}, default="global"
        How to handle panel data. See `BaseForecaster` for details.

    Attributes
    ----------
    fit_coverage_rates_ : list of float
        Coverage rates used during fit.

    Notes
    -----
    The data is split into a training portion and a calibration portion
    of size ``calibration_size``.  The point forecaster is fit on the
    training portion, then conformity scores are computed on the
    calibration portion.  At prediction time, interval bounds are
    derived from the empirical quantiles of these scores.

    See Also
    --------
    `BaseSimilarity` : Similarity weighting for adaptive intervals.
    `Residual` : Default conformity scorer.
    `IntervalReductionForecaster` : Alternative interval forecaster.

    """

    _parameter_constraints: dict = {
        **BaseIntervalForecaster._parameter_constraints,
        "point_forecaster": [BasePointForecaster],
        "calibration_size": [Interval(numbers.Integral, 1, None, closed="left")],
        "conformity_scorer": [BaseConformityScorer],
        "similarity": [BaseSimilarity, None],
    }

    def __sklearn_tags__(self) -> Tags:
        """Get estimator tags.

        Returns
        -------
        Tags
            Estimator tags with forecaster_type set to POINT_INTERVAL since this
            forecaster produces both point predictions and intervals.

        """
        tags = super().__sklearn_tags__()
        assert tags.forecaster_tags is not None
        # SplitConformal wraps a point forecaster and adds intervals
        tags.forecaster_tags.forecaster_type = POINT_INTERVAL
        return tags

    def __init__(
        self,
        point_forecaster: BasePointForecaster = SeasonalNaive(),
        calibration_size: StrictInt = 100,
        conformity_scorer: BaseConformityScorer = Residual(),
        similarity: BaseSimilarity | None = None,
        panel_strategy: Literal["global", "multivariate"] = "global",
    ):
        BaseIntervalForecaster.__init__(self, panel_strategy=panel_strategy)

        self.point_forecaster = point_forecaster
        self.conformity_scorer = conformity_scorer
        self.similarity = similarity
        self.calibration_size = calibration_size

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(
        self,
        y: pl.DataFrame,
        X: pl.DataFrame | None = None,
        forecasting_horizon: StrictInt = 1,
        coverage_rates: list[StrictFloat] | None = None,
        **params,
    ) -> "SplitConformalForecaster":
        """Fit the forecaster to historical data.

        Trains the wrapped point forecaster, calibrates conformity scores
        on a held-out calibration set, and optionally fits similarity
        weights.

        Parameters
        ----------
        y : pl.DataFrame
            Target time series with a ``"time"`` column (datetime) and one
            or more numeric value columns.
        X : pl.DataFrame or None, default=None
            Exogenous features with a ``"time"`` column matching ``y``.
            If ``None``, no exogenous features are used.
        forecasting_horizon : int, default=1
            Number of time steps to forecast into the future.
        coverage_rates : list of float or None, default=None
            Coverage levels for prediction intervals (e.g., ``[0.9, 0.95]``
            for 90 % and 95 % intervals).  If ``None``, defaults to
            ``[0.95]``.
        **params : dict
            Metadata to route to nested estimators.

        Returns
        -------
        self
            The fitted forecaster instance.

        """
        # Validate data and set interval
        y, X, _ = validate_forecaster_data(self, y, X, reset=True)

        # _pre_fit: set schemas/panel attributes, fit transformers
        # (target_transformer=None, feature_transformer=None → no-ops),
        # and populate observation buffers (observed_time_, _y_observed,
        # _X_t_observed).  Called on the full y before the
        # train/calibration split so base-class state reflects the
        # complete training history.
        self._pre_fit(y, X, forecasting_horizon)

        # Validate interval-specific parameters (coverage rates)
        _, self.fit_coverage_rates_ = self._validate_fit_params(self.fit_forecasting_horizon_, coverage_rates)

        # Handle splitting with optional X
        if X is None:
            y_train, y_calib = train_test_split(y, test_size=self.calibration_size, shuffle=False)
            X_train, X_calib = None, None
        else:
            y_train, y_calib, X_train, X_calib = train_test_split(y, X, test_size=self.calibration_size, shuffle=False)

        self.point_forecaster_ = clone(self.point_forecaster).fit(
            y=y_train,
            X=X_train,
            forecasting_horizon=forecasting_horizon,
        )

        # TODO: Reconsider
        # stride=1: each row of y_calib produces one prediction window of length
        # forecasting_horizon.  This yields calibration_size - step + 1 conformity
        # scores for each horizon step k, instead of the ~2-3 scores that result
        # from stride=forecasting_horizon.  More calibration scores per step gives
        # quantiles that are stable and well-separated.
        y_pred_calib = self.point_forecaster_.observe_predict(
            y=y_calib,
            X=X_calib,
            forecasting_horizon=None,
            stride=1,
            predict_transformed=False,
        )

        conformity_scorers = {}
        conformity_scores = pl.DataFrame()
        similarities = {}
        weights_list: list[pl.DataFrame] = []

        for step in range(1, 1 + forecasting_horizon):
            y_pred_calib_step = y_pred_calib[step - 1 :: forecasting_horizon]
            y_truth_step = y_calib

            conformity_scorer_step = clone(self.conformity_scorer).fit(y_calib, forecaster=self.point_forecaster_)
            conformity_scores_step = conformity_scorer_step.score(y_truth_step, y_pred_calib_step)

            conformity_scores_step = conformity_scores_step.with_columns(step=step)
            conformity_scores = pl.concat([conformity_scores, conformity_scores_step])

            conformity_scorers[f"step_{step}"] = conformity_scorer_step

            # Fit similarity on the same scored subset to ensure length alignment
            if self.similarity is not None:
                scored_times_df = conformity_scores_step.drop("step").select("time")
                y_pred_for_sim = y_pred_calib_step.drop("vintage_time", strict=False).join(
                    scored_times_df, on="time", how="semi"
                )

                similarity_step = clone(self.similarity)
                similarity_step.fit(y=y_calib, y_pred=y_pred_for_sim)

                weights_array = similarity_step.predict(y_pred=y_pred_for_sim)
                weight_col_names = [f"w_{i}" for i in range(weights_array.shape[1])]
                weights_step = pl.DataFrame(weights_array, schema=weight_col_names)
                weights_step = weights_step.with_columns(step=pl.lit(step))
                weights_list.append(weights_step)

                similarities[f"step_{step}"] = similarity_step

        self.conformity_scorers_ = conformity_scorers
        self.conformity_scores_ = conformity_scores

        if self.similarity is not None:
            self.similarities_ = similarities
            self.weights_ = pl.concat(weights_list, how="diagonal")

        return self

    def observe(
        self,
        y: pl.DataFrame,
        X: pl.DataFrame | None = None,
        groups: list[str] | None = None,
    ) -> "SplitConformalForecaster":
        """Observe new data and update the wrapped point forecaster.

        Delegates to the wrapped point forecaster's ``observe()`` method
        to update its observation buffers without refitting.

        Parameters
        ----------
        y : pl.DataFrame
            Target time series with a ``"time"`` column (datetime) and one
            or more numeric value columns.
        X : pl.DataFrame or None, default=None
            Exogenous features with a ``"time"`` column matching ``y``.
            If ``None``, no exogenous features are used.
        groups : list of str or None, default=None
            Panel group prefixes to operate on.  If ``None``, all groups
            are used.  Ignored when the forecaster was not fitted on panel
            data.

        Returns
        -------
        self
            The forecaster with updated observation buffers.

        """
        check_is_fitted(
            self,
            ["point_forecaster_", "local_y_schema_", "local_X_schema_", "shared_X_schema_", "groups_"],
        )

        y, X, groups = validate_forecaster_data(self, y, X, reset=False, groups=groups)

        self.point_forecaster_.observe(y=y, X=X, groups=groups)
        return self

    def rewind(
        self,
        y: pl.DataFrame,
        X: pl.DataFrame | None = None,
        groups: list[str] | None = None,
    ) -> "SplitConformalForecaster":
        """Rewind the wrapped point forecaster's observation buffers.

        Delegates to the wrapped point forecaster's ``rewind()`` method.

        Parameters
        ----------
        y : pl.DataFrame
            Target time series with a ``"time"`` column (datetime) and one
            or more numeric value columns.
        X : pl.DataFrame or None, default=None
            Exogenous features with a ``"time"`` column matching ``y``.
            If ``None``, no exogenous features are used.
        groups : list of str or None, default=None
            Panel group prefixes to operate on.  If ``None``, all groups
            are used.  Ignored when the forecaster was not fitted on panel
            data.

        Returns
        -------
        self
            The forecaster with rewound observation buffers.

        """
        check_is_fitted(
            self,
            ["point_forecaster_", "local_y_schema_", "local_X_schema_", "shared_X_schema_", "groups_"],
        )

        y, X, groups = validate_forecaster_data(self, y, X, reset=False, groups=groups)

        self.point_forecaster_.rewind(y=y, X=X, groups=groups)
        return self

    def predict(
        self,
        X: pl.DataFrame | None = None,
        forecasting_horizon: StrictInt | None = None,
        groups: list[str] | None = None,
        predict_transformed: bool = False,
        **params,
    ) -> pl.DataFrame:
        """Generate point forecasts.

        Delegates to the wrapped point forecaster.

        Parameters
        ----------
        X : pl.DataFrame or None, default=None
            Exogenous features with a ``"time"`` column matching ``y``.
            If ``None``, no exogenous features are used.
        forecasting_horizon : int or None, default=None
            Number of time steps to forecast into the future.  If ``None``,
            uses the horizon specified at fit time.
        groups : list of str or None, default=None
            Panel group prefixes to operate on.  If ``None``, all groups
            are used.  Ignored when the forecaster was not fitted on panel
            data.
        predict_transformed : bool, default=False
            If ``True``, return predictions in the transformed space without
            applying inverse target transformation.
        **params : dict
            Metadata to route to nested estimators.

        Returns
        -------
        pl.DataFrame
            Point predictions with ``"vintage_time"``, ``"time"``, and one
            column per target variable.

        """
        check_is_fitted(
            self,
            ["local_y_schema_", "local_X_schema_", "shared_X_schema_", "groups_"],
        )

        _, X, groups = validate_forecaster_data(
            self,
            y=None,
            X=X,
            reset=False,
            groups=groups,
        )

        return self.point_forecaster_.predict(
            X=X,
            forecasting_horizon=forecasting_horizon,
            groups=groups,
            predict_transformed=predict_transformed,
        )

    def observe_predict(
        self,
        y: pl.DataFrame,
        X: pl.DataFrame | None = None,
        forecasting_horizon: StrictInt | None = None,
        groups: list[str] | None = None,
        stride: StrictInt | None = None,
        predict_transformed: bool = False,
        **params,
    ) -> pl.DataFrame:
        """Alternate recursive observe and predict.

        Equivalent to calling ``observe(y, X)`` then ``predict(X)``.
        Returns point predictions.

        Parameters
        ----------
        y : pl.DataFrame
            Target time series with a ``"time"`` column (datetime) and one
            or more numeric value columns.
        X : pl.DataFrame or None, default=None
            Exogenous features with a ``"time"`` column matching ``y``.
            If ``None``, no exogenous features are used.
        forecasting_horizon : int or None, default=None
            Number of time steps to forecast into the future.  If ``None``,
            uses the horizon specified at fit time.
        groups : list of str or None, default=None
            Panel group prefixes to operate on.  If ``None``, all groups
            are used.  Ignored when the forecaster was not fitted on panel
            data.
        stride : int or None, default=None
            Step size for rolling update-predict.  If ``None``, defaults to
            ``forecasting_horizon``.
        predict_transformed : bool, default=False
            If ``True``, return predictions in the transformed space without
            applying inverse target transformation.
        **params : dict
            Metadata to route to nested estimators.

        Returns
        -------
        pl.DataFrame
            Point predictions with ``"vintage_time"``, ``"time"``, and one
            column per target variable.

        Raises
        ------
        sklearn.exceptions.NotFittedError
            If the forecaster has not been fitted yet.
        ValueError
            If ``y`` / ``X`` have invalid structure or ``groups``
            contains names not seen during fit.

        """
        check_is_fitted(
            self,
            ["local_y_schema_", "local_X_schema_", "shared_X_schema_", "groups_"],
        )

        y, X, groups = validate_forecaster_data(
            self,
            y=y,
            X=X,
            reset=False,
            groups=groups,
        )

        forecasting_horizon, _ = self._validate_predict_params(forecasting_horizon)
        if stride is None:
            stride = self.fit_forecasting_horizon_

        # Initial prediction before observing new data
        y_pred_i = self.predict(
            X=X,
            forecasting_horizon=forecasting_horizon,
            groups=groups,
            predict_transformed=predict_transformed,
            **params,
        )

        y_pred = y_pred_i
        for i in range(0, len(y), stride):
            y_slice = y[i : i + stride]

            X_slice = None
            if X is not None:
                X_slice = X.join(y_slice.select("time"), on="time", how="semi")

            self.observe(y=y_slice, X=X_slice, groups=groups)

            X_future = None
            if X is not None:
                last_time = y_slice["time"][-1]
                X_future = X.filter(pl.col("time") > last_time)

            y_pred_i = self.predict(
                X=X_future,
                forecasting_horizon=forecasting_horizon,
                groups=groups,
                predict_transformed=predict_transformed,
                **params,
            )

            y_pred = pl.concat([y_pred, y_pred_i])

        return y_pred

    def _weighted_inverse_score(
        self,
        y_pred_step: pl.DataFrame,
        conformity_scores_step: pl.DataFrame,
        coverage_rate: float,
        weights: np.ndarray,
        conformity_scorer_step: BaseConformityScorer,
    ) -> pl.DataFrame:
        """Compute prediction intervals using similarity-weighted quantiles.

        Parameters
        ----------
        y_pred_step : pl.DataFrame
            Point predictions for one step (single row) with ``"time"``
            and value columns.
        conformity_scores_step : pl.DataFrame
            Conformity scores for this step with ``"time"`` and value
            columns.
        coverage_rate : float
            Target coverage probability.
        weights : np.ndarray
            Similarity weights of shape ``(n_calibration,)``.
        conformity_scorer_step : BaseConformityScorer
            Fitted conformity scorer (used for type detection and
            formatting).

        Returns
        -------
        pl.DataFrame
            Interval columns (no ``"time"`` column).

        """
        value_cols = [c for c in conformity_scores_step.columns if c != "time"]
        scores_no_time = conformity_scores_step.drop("time", strict=False)
        y_pred_values = y_pred_step.drop("time")

        if isinstance(conformity_scorer_step, AbsoluteResidual):
            # Symmetric: ±quantile
            lower_data: dict[str, list[float]] = {}
            upper_data: dict[str, list[float]] = {}
            for col in value_cols:
                scores_col = scores_no_time[col].to_numpy().astype(np.float64)
                q = weighted_quantile(scores_col, coverage_rate, weights)
                pred_val = float(y_pred_values[col][0])
                lower_data[col] = [pred_val - q]
                upper_data[col] = [pred_val + q]

            lower_bound = pl.DataFrame(lower_data)
            upper_bound = pl.DataFrame(upper_data)
            return conformity_scorer_step._format_y_pred_interval(lower_bound, upper_bound, coverage_rate)

        if isinstance(conformity_scorer_step, Residual):
            # Asymmetric: add lower/upper quantiles directly
            lower_data = {}
            upper_data = {}
            for col in value_cols:
                scores_col = scores_no_time[col].to_numpy().astype(np.float64)
                alpha = 1.0 - coverage_rate
                lower_q = weighted_quantile(scores_col, alpha / 2.0, weights)
                upper_q = weighted_quantile(scores_col, 1.0 - alpha / 2.0, weights)
                pred_val = float(y_pred_values[col][0])
                lower_data[col] = [pred_val + lower_q]
                upper_data[col] = [pred_val + upper_q]

            lower_bound = pl.DataFrame(lower_data)
            upper_bound = pl.DataFrame(upper_data)
            return conformity_scorer_step._format_y_pred_interval(lower_bound, upper_bound, coverage_rate)

        # Unsupported scorer type: fall back to unweighted inverse_score
        return conformity_scorer_step.inverse_score(
            y_pred=y_pred_step,
            conformity_scores=conformity_scores_step,
            coverage_rate=coverage_rate,
        ).drop("time")

    def predict_interval(
        self,
        X: pl.DataFrame | None = None,
        forecasting_horizon: StrictInt | None = None,
        coverage_rates: list[float] | None = None,
        strategy: Literal["mean", "median", "point"] | None = None,
        groups: list[str] | None = None,
        **params,
    ) -> pl.DataFrame:
        """Generate interval forecasts.

        Uses calibrated conformity scores to construct prediction intervals
        around point forecasts.

        Parameters
        ----------
        X : pl.DataFrame or None, default=None
            Exogenous features with a ``"time"`` column matching ``y``.
            If ``None``, no exogenous features are used.
        forecasting_horizon : int or None, default=None
            Number of time steps to forecast into the future.  If ``None``,
            uses the horizon specified at fit time.
        coverage_rates : list of float or None, default=None
            Coverage levels for prediction intervals (e.g., ``[0.9, 0.95]``
            for 90 % and 95 % intervals).  If ``None``, defaults to the rates
            used at fit time.
        strategy : {"mean", "median", "point"} or None, default=None
            Strategy for deriving point predictions from prediction intervals
            during recursive multi-step forecasting.  If ``None``, defaults
            to ``"mean"``.
        groups : list of str or None, default=None
            Panel group prefixes to operate on.  If ``None``, all groups
            are used.  Ignored when the forecaster was not fitted on panel
            data.
        **params : dict
            Metadata to route to nested estimators.

        Returns
        -------
        pl.DataFrame
            Interval predictions with ``"vintage_time"``, ``"time"``, and
            lower/upper bound columns for each target at each coverage rate.

        """
        check_is_fitted(
            self,
            ["local_y_schema_", "local_X_schema_", "shared_X_schema_", "groups_"],
        )

        _, X, groups = validate_forecaster_data(
            self,
            y=None,
            X=X,
            reset=False,
            groups=groups,
        )

        forecasting_horizon, coverage_rates = self._validate_predict_params(forecasting_horizon, coverage_rates)

        y_pred_full = self.point_forecaster_.predict(X=X)
        has_vintage_time = "vintage_time" in y_pred_full.columns
        if has_vintage_time:
            vintage_time_col = y_pred_full["vintage_time"]
            y_pred = y_pred_full.drop("vintage_time")
        else:
            y_pred = y_pred_full

        # Extract time for later reconstruction
        y_pred_time = y_pred.select("time")
        y_pred_values = y_pred.drop("time")

        y_pred_intervals = pl.DataFrame()
        for step in range(1, 1 + forecasting_horizon):
            # Get step predictions
            y_pred_step_values = y_pred_values.slice(step - 1, 1)
            y_pred_step_time = y_pred_time.slice(step - 1, 1)

            # Combine time and values for inverse_score (conformity scorers need time)
            y_pred_step = y_pred_step_time.hstack(y_pred_step_values)

            conformity_scorer_step = self.conformity_scorers_[f"step_{step}"]
            conformity_scores_step = self.conformity_scores_.filter(pl.col("step") == step).drop("step")

            y_pred_intervals_step = pl.DataFrame()
            for coverage_rate in coverage_rates:
                if self.similarity is not None and hasattr(self, "similarities_"):
                    similarity_step = self.similarities_[f"step_{step}"]
                    weights_array = similarity_step.predict(y_pred=y_pred_step)
                    step_weights = weights_array[0].astype(np.float64)

                    y_pred_interval_rate_step = self._weighted_inverse_score(
                        y_pred_step=y_pred_step,
                        conformity_scores_step=conformity_scores_step,
                        coverage_rate=coverage_rate,
                        weights=step_weights,
                        conformity_scorer_step=conformity_scorer_step,
                    )
                else:
                    y_pred_interval_rate_step = conformity_scorer_step.inverse_score(
                        y_pred=y_pred_step,
                        conformity_scores=conformity_scores_step,
                        coverage_rate=coverage_rate,
                    ).drop("time")

                y_pred_intervals_step = pl.concat(
                    [y_pred_intervals_step, y_pred_interval_rate_step],
                    how="horizontal",
                )

            # Add time column once at the front
            y_pred_intervals_step = pl.concat([y_pred_step_time, y_pred_intervals_step], how="horizontal")

            y_pred_intervals = pl.concat([y_pred_intervals, y_pred_intervals_step])

        if has_vintage_time:
            return y_pred_intervals.insert_column(0, vintage_time_col.head(len(y_pred_intervals)))
        return y_pred_intervals
