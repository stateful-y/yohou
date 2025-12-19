"""Implementation of reduction-based interval forecasters."""

from typing import List, Literal, Optional

import polars as pl
import polars.selectors as cs
from pydantic import StrictFloat, StrictInt
from sklearn.base import RegressorMixin
from sklearn.linear_model import QuantileRegressor
from sklearn.multioutput import MultiOutputRegressor

from yohou.base import BaseReductionForecaster
from yohou.utils import concat_struct, neg_struct

from .base import BaseIntervalForecaster


class IntervalReductionForecaster(BaseReductionForecaster, BaseIntervalForecaster):
    def __init__(
        self,
        estimator: RegressorMixin = MultiOutputRegressor(QuantileRegressor()),
        coverage_rates: List[StrictFloat] = [0.5],
        update_strategy: Literal["average", "constant"] = "average",
    ):
        BaseReductionForecaster.__init__(
            self,
            feature_transformer=None,
            target_transformer=None,
            estimator=estimator,
        )

        BaseIntervalForecaster.__init__(
            self,
            coverage_rates=coverage_rates,
            update_strategy=update_strategy,
        )

    def fit(
        self,
        y: pl.DataFrame,
        X_ante: Optional[pl.DataFrame] = None,
        X_post: Optional[pl.DataFrame] = None,
        forecasting_horizon: StrictInt = 1,
    ) -> "IntervalReductionForecaster":
        """Fits the forecaster and returns it.

        Parameters
        ----------
        y : pl.DataFrame
            Target time series.

        X_ante : pl.DataFrame or None, default=None
            Ex-ante feature time series.

        X_post : pl.DataFrame or None, default=None
            Ex-post feature time series.

        forecasting_horizon : int > 1, default=1
            Horizon to forecast.

        Returns
        -------
        self

        """
        y_t, X_t = BaseIntervalForecaster._pre_fit(
            self,
            y=y,
            X_ante=X_ante,
            X_post=X_post,
            forecasting_horizon=forecasting_horizon,
        )

        # TODO: Do that to y instead of y_t?
        time = y_t.select(cs.by_name("time"))
        y_t = concat_struct(
            [
                neg_struct(y_t.select(~cs.by_name("time")), prefix="negative_"),
                y_t.select(~cs.by_name("time")),
            ],
            how="horizontal",
        )
        y_t = pl.concat([time, y_t], how="horizontal")

        # y_t = pl.concat(
        #     [
        #         time,
        #         y_t.select(~cs.by_name("time")).with_columns((-pl.all()).prefix("negative_"))
        #         .drop(y_t.columns),
        #         y_t.select(~cs.by_name("time")),
        #     ],
        #     how="horizontal",
        # )
        y_pred_local_columns = [
            f"negative_{col}" for col in self.local_y_names_
        ] + self.local_y_names_

        estimator_param_names = list(self.estimator.get_params(deep=True))
        quantile_param_names = [
            param_name
            for param_name in estimator_param_names
            if param_name.split("__")[-1] == "quantile"
        ]

        if len(quantile_param_names) > 1:
            raise ValueError()

        quantile_param_name = quantile_param_names[0]

        estimators = {}
        # TODO: Support CatBoost multiquantile
        for coverage_rate in self.coverage_rates:
            estimator_params = {
                quantile_param_name: 1.0 - coverage_rate / 2.0,
            }
            estimator_rate = self._estimator_fit_one(
                y_t,
                X_t,
                forecasting_horizon,
                y_pred_local_columns=y_pred_local_columns,
                **estimator_params,
            )
            estimators[f"coverage_rate_{coverage_rate}"] = estimator_rate

        self.estimator_ = estimators
        return self

    def _predict_one(
        self,
    ) -> pl.DataFrame:
        """Predicts the model forecasting horizon from the
        observation horizon.

        Returns
        -------
        pl.DataFrame
            Predicted time series.

        """
        y_pred = pl.DataFrame()
        for coverage_rate in self.coverage_rates:
            estimator_rate = self.estimator_[f"coverage_rate_{coverage_rate}"]

            y_pred_lower_columns = [f"{col}_lower_{coverage_rate}" for col in self.local_y_names_]
            y_pred_upper_columns = [f"{col}_upper_{coverage_rate}" for col in self.local_y_names_]

            y_pred_rate = self._estimator_predict_one(
                estimator_rate,
                y_pred_local_columns=y_pred_lower_columns + y_pred_upper_columns,
            )

            y_pred_rate = neg_struct(y_pred_rate, local_col_names=y_pred_lower_columns)

            if y_pred.shape[1] == 0:
                # First iteration, use the entire dataframe
                y_pred = y_pred_rate
            else:
                # For subsequent iterations, merge columns
                if self.local_group_names_ is not None:
                    # For struct columns, process each group separately
                    struct_updates = {}
                    for group_name in self.local_group_names_:
                        # Unnest the struct in both dataframes
                        y_pred_group = y_pred[[group_name]].unnest(group_name)
                        y_pred_rate_group = y_pred_rate[[group_name]].unnest(group_name)

                        # Find new columns in this iteration
                        new_cols = [
                            col
                            for col in y_pred_rate_group.columns
                            if col not in y_pred_group.columns
                        ]
                        if new_cols:
                            # Merge the new columns
                            y_pred_group_merged = pl.concat(
                                [y_pred_group, y_pred_rate_group.select(new_cols)], how="horizontal"
                            )
                            # Store the merged struct
                            struct_updates[group_name] = y_pred_group_merged.to_struct(group_name)

                    # Apply all struct updates
                    for group_name, struct_col in struct_updates.items():
                        y_pred = y_pred.with_columns(**{group_name: struct_col})
                else:
                    # For non-struct columns, only add columns that don't exist yet
                    new_cols = [col for col in y_pred_rate.columns if col not in y_pred.columns]
                    if new_cols:
                        y_pred = pl.concat([y_pred, y_pred_rate.select(new_cols)], how="horizontal")

        y_pred = self._add_time_columns(y_pred)

        return y_pred
