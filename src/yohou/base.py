import abc
import inspect
from copy import deepcopy
from typing import Optional

import numpy as np
import polars as pl
import polars.selectors as cs
from pydantic import StrictInt
from sklearn.base import (
    BaseEstimator,
    RegressorMixin,
    TransformerMixin,
    clone,
)
from sklearn.linear_model import LinearRegression
from sklearn.utils._param_validation import InvalidParameterError
from sklearn.utils.validation import check_is_fitted

from yohou.utils import check_inputs, concat_struct, inspect_locality, tabularize

__all__ = ["BaseTransformer", "BaseForecaster", "BaseWrapper"]


REQUIRED_PARAM_VALUE = "__REQUIRED__"


class BaseTransformer(BaseEstimator, TransformerMixin, metaclass=abc.ABCMeta):  # type: ignore[misc]
    """Base class for time series transfomers."""

    @property
    def memory_size(self) -> int:
        return getattr(self, "_memory_size", 1)

    @abc.abstractmethod
    def fit(self, X: pl.DataFrame, y: Optional[pl.DataFrame] = None) -> "BaseTransformer":
        """Fits the transformer and returns it.

        Parameters
        ----------
        X : pl.DataFrame
            Input time series.

        y : pl.DataFrame or None, default=None
            Target time series. Ignored and only present for
            API consistency.

        Returns
        -------
        self

        """
        self.reset(X)

        self.feature_names_in_ = X.select(~cs.by_name("time")).columns
        self.n_features_in_ = len(self.feature_names_in_)

        return self

    def reset(
        self,
        X: pl.DataFrame,
    ) -> "BaseTransformer":
        """Resets the transformer by resetting the observation horizon.

        Parameters
        ----------
        X : pl.DataFrame
            Input time series.

        Returns
        -------
        self

        """
        if len(X) < self.memory_size:
            raise ValueError("Not enough input data to set the transformer memory.")

        self._X_observed = X[-self.memory_size :]

        return self

    def update(self, X: pl.DataFrame) -> "BaseTransformer":
        """Updates the transformer and returns it.

        Parameters
        ----------
        X : pl.DataFrame
            Input time series.

        Returns
        -------
        self

        """
        self.reset(pl.concat([self._X_observed, X]))

        return self

    @abc.abstractmethod
    def transform(self, X: pl.DataFrame) -> pl.DataFrame:
        """Transforms the input time series.

        Parameters
        ----------
        X : pl.DataFrame
            Feature time series.

        Returns
        -------
        pl.DataFrame
            Transformed time series.

        """
        raise NotImplementedError()

    def update_transform(self, X: pl.DataFrame) -> pl.DataFrame:
        """Transforms the input, updates the transformer and returns
        the transformed input.

        Parameters
        ----------
        X : pl.DataFrame
            Input time series.

        Returns
        -------
        self

        """
        X_t = self.transform(X)
        self.update(X)

        return X_t

    @abc.abstractmethod
    def get_feature_names_out(self, input_features: list[str] | None = None) -> list[str]:
        """Get output feature names for transformation.

        Parameters
        ----------
        input_features : array-like of str or None, default=None
            Input features.

        Returns
        -------
        feature_names_out : ndarray of str objects
            Transformed feature names.
        """
        raise NotImplementedError()


class BaseForecaster(BaseEstimator, metaclass=abc.ABCMeta):  # type: ignore[misc]
    """Base class for forecasters.

    Parameters
    ----------
    target_transformer : instance of `BaseTransformer` or None, default=None
        Transformer used to transform the target time series into the new target.

    feature_transformer : instance of `BaseTransformer` or None, default=None
        Transformer used to transform the target time series into features.
        .
    """

    def __init__(
        self,
        target_transformer: Optional[BaseTransformer] = None,
        feature_transformer: Optional[BaseTransformer] = None,
    ) -> None:
        self.target_transformer = target_transformer
        self.feature_transformer = feature_transformer

    @property
    def prediction_type(self) -> str:
        return str(self._prediction_type)

    def _set_local_groups(
        self, y: pl.DataFrame, X_ante: pl.DataFrame | None, X_post: pl.DataFrame | None
    ) -> None:
        y_global_names, y_local_groups = inspect_locality(y)

        local_group_names, local_y_names = None, y_global_names
        if len(y_local_groups):
            local_group_names = list(y_local_groups.keys())
            local_y_names = y_local_groups[local_group_names[0]]

            if len(y_global_names):
                raise ValueError("`y` contains both local and global columns.")

            y_unique_elements = np.unique(np.array(list(y_local_groups.values())), axis=0)
            if y_unique_elements.shape[0] > 1:
                raise ValueError("The local groups in `y` do not have the same column names.")

        local_X_ante_names = []
        if X_ante is not None:
            X_ante_global_names, X_ante_local_groups = inspect_locality(X_ante)
            local_X_ante_names = X_ante_global_names

            if len(X_ante_local_groups):
                if local_group_names:
                    local_X_ante_names += X_ante_local_groups[local_group_names[0]]
                if list(X_ante_local_groups.keys()) != list(y_local_groups.keys()):
                    raise ValueError(
                        "`X_ante` and `y` do not have the same local group column names."
                    )

                X_ante_unique_elements = np.unique(
                    np.array(list(X_ante_local_groups.values())), axis=0
                )
                if X_ante_unique_elements.shape[0] > 1:
                    raise ValueError(
                        "The local groups in `X_ante` do not have the same column names."
                    )

        local_X_post_names = []
        if X_post is not None:
            X_post_global_names, X_post_local_groups = inspect_locality(X_post)
            local_X_post_names = X_post_global_names

            if len(X_post_local_groups):
                if local_group_names:
                    local_X_post_names += X_ante_local_groups[local_group_names[0]]
                if list(X_post_local_groups.keys()) != list(y_local_groups.keys()):
                    raise ValueError(
                        "`X_post` and `y` do not have the same local group column names."
                    )

                X_post_unique_elements = np.unique(
                    np.array(list(X_post_local_groups.values())), axis=0
                )
                if X_post_unique_elements.shape[0] > 1:
                    raise ValueError(
                        "The local groups in `X_post` do not have the same column names."
                    )

        self.local_group_names_ = local_group_names
        self.local_y_names_ = local_y_names
        self.local_X_names_ = local_X_ante_names + local_X_post_names

    def _fit_transform_one(
        self, y: pl.DataFrame, X: pl.DataFrame | None
    ) -> tuple[pl.DataFrame, pl.DataFrame, BaseTransformer | None, BaseTransformer | None]:
        y_t = y
        target_transformer, target_memory_size = None, 0
        if self.target_transformer is not None:
            target_transformer = clone(self.target_transformer)
            y_t = target_transformer.fit_transform(y)
            target_memory_size = target_transformer.memory_size

        X_t = y_t
        if X is not None:
            X_t = pl.concat(
                [y_t, X.select(~cs.by_name("time"))[target_memory_size:]],
                how="horizontal",
            )

        feature_transformer = None
        if self.feature_transformer is not None:
            feature_transformer = clone(self.feature_transformer)
            X_t = feature_transformer.fit_transform(X_t)
            if target_transformer is not None:
                feature_memory_size = target_transformer.memory_size
                y_t = y_t[feature_memory_size:]

        return y_t, X_t, target_transformer, feature_transformer

    def _update_one(
        self,
        y: pl.DataFrame,
        X: pl.DataFrame | None,
        target_transformer: BaseTransformer | None,
        feature_transformer: BaseTransformer | None,
    ) -> pl.DataFrame:
        y_t_new = y
        if self.target_transformer is not None and target_transformer is not None:
            y_t_new = target_transformer.update_transform(y)

        X_t = y_t_new
        if X is not None:
            X_t = concat_struct(
                [X_t, X.select(~cs.by_name("time"))],
                how="horizontal",
            )

        X_t_new = X_t
        if self.feature_transformer is not None and feature_transformer is not None:
            X_t_new = feature_transformer.update_transform(X_t)

        return X_t_new

    # TODO: Enforce column order
    def _preprocess_inputs(
        self,
        y: pl.DataFrame,
        X_ante: Optional[pl.DataFrame] = None,
        X_post: Optional[pl.DataFrame] = None,
    ) -> tuple[pl.DataFrame, pl.DataFrame | None]:
        time = y.select(cs.by_name("time"))

        X = None
        self.X_ante_columns = None
        if X_ante is not None:
            X_ante = X_ante.select(~cs.by_name("time"))
            X = X_ante

        self.X_post_columns_ = None
        if X_post is not None:
            if X_ante is None:
                X = pl.DataFrame()

            X_post = X_post.filter(pl.col("time") == y["time"]).select(~cs.by_name("time"))
            X = pl.concat([X, X_post], how="horizontal")

        if X is not None:
            X = pl.concat([time, X], how="horizontal")

        return y, X

    def _pre_fit(
        self,
        y: pl.DataFrame,
        X_ante: Optional[pl.DataFrame] = None,
        X_post: Optional[pl.DataFrame] = None,
        forecasting_horizon: StrictInt = 1,
    ) -> tuple[pl.DataFrame, pl.DataFrame]:
        self.interval_ = check_inputs(y, X_ante, X_post)
        self._set_local_groups(y, X_ante, X_post)

        if forecasting_horizon < 1:
            raise ValueError(
                f"`forecasting_horizon` should be a positive int. It is: {forecasting_horizon}"
            )

        self.fit_forecasting_horizon_ = forecasting_horizon

        y, X = self._preprocess_inputs(y, X_ante, X_post)

        y_t, X_t = self._fit_transform_inputs(y, X)

        # TODO: Do we need to keep all in memory?
        self._X_t_observed = X_t
        self._y_observed = y.filter(pl.col("time") == X_t["time"])
        if X_ante is not None:
            self._X_ante_observed = X_ante.filter(pl.col("time") == X_t["time"])

        return y_t, X_t

    def _fit_transform_inputs(
        self, y: pl.DataFrame, X: pl.DataFrame | None
    ) -> tuple[pl.DataFrame, pl.DataFrame]:
        y = y.select(["time"] + self.local_y_names_)
        if X is not None:
            X = X.select(["time"] + self.local_X_names_)

        y_t, X_t, self.target_transformer_, self.feature_transformer_ = self._fit_transform_one(
            y, X
        )

        self.local_y_t_names_ = [col for col in y_t.columns if col != "time"]
        self.local_X_t_names_ = [col for col in X_t.columns if col != "time"]

        return y_t, X_t

    def _update_inputs(self, y: pl.DataFrame, X: pl.DataFrame | None) -> pl.DataFrame:
        y = y.select(["time"] + self.local_y_names_)
        if X is not None:
            X = X.select(["time"] + self.local_X_names_)

        X_t_new = self._update_one(y, X, self.target_transformer_, self.feature_transformer_)

        return X_t_new

    @abc.abstractmethod
    def fit(
        self,
        y: pl.DataFrame,
        X_ante: Optional[pl.DataFrame] = None,
        X_post: Optional[pl.DataFrame] = None,
        forecasting_horizon: StrictInt = 1,
    ) -> "BaseForecaster":
        """Fits the forecaster and returns it.

        Parameters
        ----------
        y : pl.DataFrame
            Target time series.

        X_ante : pl.DataFrame or None, default=None
            Ex-ante feature time series.

        X_post : pl.DataFrame or None, default=None
            Ex-post feature time series.

        forecasting_horizon : int >= 1, default=1
            Horizon to forecast.

        Returns
        -------
        self

        """
        raise NotImplementedError()

    @abc.abstractmethod
    def reset(
        self,
        y: pl.DataFrame,
        X_ante: Optional[pl.DataFrame],
        X_post: Optional[pl.DataFrame],
    ) -> "BaseForecaster":
        """Resets the forecaster by resetting the observation horizon.

        Parameters
        ----------
        y : pl.DataFrame
            Target time series.

        X_ante : pl.DataFrame or None
            Ex-ante feature time series.

        X_post : pl.DataFrame or None
            Ex-post feature time series.

        Returns
        -------
        self

        """
        # TODO: Write a reset function and use it to predict the train set
        raise NotImplementedError()

    def update(
        self,
        y: pl.DataFrame,
        X_ante: Optional[pl.DataFrame],
        X_post: Optional[pl.DataFrame],
    ) -> "BaseForecaster":
        """Updates the forecaster with more recent data and
        returns it.

        Parameters
        ----------
        y : pl.DataFrame
            Target time series.

        X_ante : pl.DataFrame or None
            Ex-ante feature time series.

        X_post : pl.DataFrame or None
            Ex-post feature time series.


        Returns
        -------
        self

        """
        check_is_fitted(self, "fit_forecasting_horizon_")
        y, X = self._preprocess_inputs(y, X_ante, X_post)

        X_t_new = self._update_inputs(y, X)

        self._y_observed = pl.concat([self._y_observed, y])
        self._X_t_observed = pl.concat([self._X_t_observed, X_t_new])
        self._X_ante_observed = pl.concat([self._X_ante_observed, X_ante])

        return self

    def _add_time_columns(self, y_pred: pl.DataFrame) -> pl.DataFrame:
        time = self._y_observed[-self.fit_forecasting_horizon_ :][["time"]]
        time = (
            time.with_columns(observed_time=self._y_observed["time"][-1])
            .with_columns(
                predicted_time=pl.col("time").dt.offset_by(
                    f"{int(self.fit_forecasting_horizon_ * self.interval_.total_seconds())}s"
                )
            )
            .select(~cs.by_name("time"))
        )

        y_pred = pl.concat([time, y_pred], how="horizontal")

        return y_pred

    def _predict_one(self) -> pl.DataFrame:
        raise NotImplementedError()

    def predict(
        self,
        X_post: Optional[pl.DataFrame] = None,
        forecasting_horizon: StrictInt = 1,
        predict_transformed: bool = True,
    ) -> pl.DataFrame:
        """Predicts the model forecasting horizon from the
        observation horizon.

        Parameters
        ----------
        X_post : pl.DataFrame
            Ex-post feature time series.

        forecasting_horizon : int > 1
            Horizon to forecast.

        predict_transformed : bool, default=True
            If ``True``, the predictions are returned in the transformed
            space.

        Returns
        -------
        pl.DataFrame
            Predicted time series.

        """
        check_is_fitted(self, "fit_forecasting_horizon_")
        forecaster = deepcopy(self)

        y_pred = pl.DataFrame()
        for step in range(1, forecasting_horizon + 1, self.fit_forecasting_horizon_):
            y_pred_step = forecaster._predict_one()

            y_pred_step_inv = y_pred_step
            if self.target_transformer is not None and forecaster.target_transformer_ is not None:
                y_pred_step_inv = forecaster.target_transformer_.inverse_transform(
                    X_t=y_pred_step,
                    X_p=forecaster._y_observed,
                )

                if not predict_transformed:
                    y_pred_step = y_pred_step_inv

            y_pred = pl.concat([y_pred, y_pred_step])

            if step + self.fit_forecasting_horizon_ <= forecasting_horizon:
                time = y_pred_step.select(cs.by_name("predicted_time")).rename(
                    {"predicted_time": "time"}
                )
                X_ante_old = forecaster._X_ante_observed[[-1]].select(~cs.by_name("time"))
                X_ante = pl.concat([X_ante_old] * len(time))
                X_ante = pl.concat([time, X_ante], how="horizontal")

                y = y_pred_step_inv.rename({"predicted_time": "time"}).select(
                    ~cs.by_name("observed_time")
                )
                forecaster.update(y, X_ante, X_post)

        y_pred = y_pred.with_columns(observed_time=y_pred["observed_time"][0])

        if forecasting_horizon % self.fit_forecasting_horizon_:
            end = (
                self.fit_forecasting_horizon_ - forecasting_horizon % self.fit_forecasting_horizon_
            )
            y_pred = y_pred[:-end]

        return y_pred

    def update_predict(
        self,
        y: pl.DataFrame,
        X_ante: Optional[pl.DataFrame] = None,
        X_post: Optional[pl.DataFrame] = None,
        forecasting_horizon: StrictInt = 1,
        stride: StrictInt = 1,
        predict_transformed: bool = False,
    ) -> pl.DataFrame:
        """Alternate `recursive_predict` and `update`.

        Parameters
        ----------
        y : pl.DataFrame
            Target time series for updates.

        X_ante : pl.DataFrame or None, default=None
            Ex-ante feature time series for updates.

        X_post : pl.DataFrame or None, default=None
            Ex-post feature time series.

        forecasting_horizon : int >= 1, default=1
            Horizon to forecast recursively.

        stride : int >= 1, default=1
            Number of new observations to use for each update.

        predict_transformed : bool, default=False
            If ``True``, the predictions are returned in the transformed
            space.

        Returns
        -------
        pl.DataFrame
            Predicted time series.

        """
        y_pred_i = self.predict(X_post, forecasting_horizon=forecasting_horizon)

        y_pred = y_pred_i
        for i in range(0, len(y), stride):
            X_ante_slice = None
            if X_ante is not None:
                X_ante_slice = X_ante[i : i + stride]

            self.update(y=y[i : i + stride], X_ante=X_ante_slice, X_post=X_post)

            y_pred_i = self.predict(
                X_post=X_post,
                forecasting_horizon=forecasting_horizon,
                predict_transformed=predict_transformed,
            )

            y_pred = pl.concat([y_pred, y_pred_i])

        return y_pred


class BaseReductionForecaster(BaseForecaster, metaclass=abc.ABCMeta):
    def __init__(
        self,
        estimator: RegressorMixin = LinearRegression(),
        target_transformer: Optional[BaseTransformer] = None,
        feature_transformer: Optional[BaseTransformer] = None,
    ):
        BaseForecaster.__init__(
            self,
            target_transformer=target_transformer,
            feature_transformer=feature_transformer,
        )

        self.estimator = estimator

    def _fit_transform_inputs(
        self,
        y: pl.DataFrame,
        X: pl.DataFrame,
    ) -> tuple[pl.DataFrame, pl.DataFrame]:
        # TODO: Use name for reoder in local case too
        if self.local_group_names_ is None:
            return BaseForecaster._fit_transform_inputs(self, y, X)

        y_t_dict, X_t_dict = {}, {}
        target_transformer_dict, feature_transformer_dict = {}, {}
        for i, local_group_name in enumerate(self.local_group_names_):
            y_local = y[
                [
                    col
                    for col, dtype in y.schema.items()
                    if dtype != pl.Struct or col == local_group_name
                ]
            ].unnest(local_group_name)
            y_local = y_local.select(["time"] + self.local_y_names_)

            X_local = X[
                [
                    col
                    for col, dtype in X.schema.items()
                    if dtype != pl.Struct or col == local_group_name
                ]
            ].unnest(local_group_name)
            X_local = X_local.select(["time"] + self.local_X_names_)

            (
                y_t_local,
                X_t_local,
                target_transformer_local,
                feature_transformer_local,
            ) = self._fit_transform_one(y_local, X_local)

            if i == 0:
                self.local_y_t_names_ = y_t_local.select(~cs.by_name("time")).columns
                self.local_X_t_names_ = X_t_local.select(~cs.by_name("time")).columns

            y_t_dict[local_group_name] = y_t_local.select(~cs.by_name("time")).select(
                self.local_y_t_names_
            )
            X_t_dict[local_group_name] = X_t_local.select(~cs.by_name("time")).select(
                self.local_X_t_names_
            )
            target_transformer_dict[local_group_name] = target_transformer_local
            feature_transformer_dict[local_group_name] = feature_transformer_local

        time = y_t_local.select(cs.by_name("time"))
        y_t = pl.concat([time, pl.DataFrame(y_t_dict)], how="horizontal")
        X_t = pl.concat([time, pl.DataFrame(X_t_dict)], how="horizontal")

        return y_t, X_t

    def _update_inputs(self, y: pl.DataFrame, X: pl.DataFrame | None) -> pl.DataFrame:
        if self.local_group_names_ is None:
            return BaseForecaster._update_inputs(self, y, X)

        X_t_new_dict = {}
        for local_group_name in self.local_group_names_:
            y_local = y[
                [
                    col
                    for col, dtype in y.schema.items()
                    if dtype != pl.Struct or col == local_group_name
                ]
            ].unnest(local_group_name)
            y_local = y_local.select(["time"] + self.local_y_names_)

            X_local = None
            if X is not None:
                X_local = X[
                    [
                        col
                        for col, dtype in X.schema.items()
                        if dtype != pl.Struct or col == local_group_name
                    ]
                ].unnest(local_group_name)
                X_local = X_local.select(["time"] + self.local_X_names_)

            local_target_transformer = None
            if self.target_transformer is not None and isinstance(self.target_transformer_, dict):
                local_target_transformer = self.target_transformer_[local_group_name]

            local_feature_transformer = None
            if self.feature_transformer is not None and isinstance(self.feature_transformer_, dict):
                local_feature_transformer = self.feature_transformer_[local_group_name]

            X_t_new_dict[local_group_name] = self._update_one(
                y_local,
                X_local,
                local_target_transformer,
                local_feature_transformer,
            ).select(~cs.by_name("time"))

        time = y.select(cs.by_name("time"))
        X_t_new = pl.DataFrame(X_t_new_dict)
        X_t_new = pl.concat([time, X_t_new], how="horizontal")

        return X_t_new

    def _get_tabularized_dataset(
        self,
        y_t: pl.DataFrame,
        X_t: pl.DataFrame,
        forecasting_horizon: int,
        y_pred_local_columns: list[str] | None,
    ) -> tuple[
        np.ndarray[tuple[int, int], np.dtype[np.float64]],
        np.ndarray[tuple[int, int], np.dtype[np.float64]],
    ]:
        if y_pred_local_columns is None:
            y_pred_local_columns = self.local_y_names_

        X_tab = X_t.select(~cs.by_name("time"))[:-forecasting_horizon]
        y_tab = tabularize(
            y_t.select(~cs.by_name("time")),
            lags=list(range(1 + forecasting_horizon)),
        ).rename(
            {
                f"{col}_lag_{lag}": f"{col}_step_{forecasting_horizon - lag}"
                for lag in range(1 + forecasting_horizon)
                for col in y_pred_local_columns
            }
        )[
            [
                f"{col}_step_{step}"
                for step in range(1, 1 + forecasting_horizon)
                for col in y_pred_local_columns
            ]
        ]

        return X_tab.to_numpy(), y_tab.to_numpy()

    def _estimator_fit_one(
        self,
        y_t: pl.DataFrame,
        X_t: pl.DataFrame,
        forecasting_horizon: StrictInt,
        y_pred_local_columns: list[str] | None = None,
        **estimator_params: object,
    ) -> BaseEstimator:
        if y_pred_local_columns is None:
            y_pred_local_columns = self.local_y_names_

        estimator = clone(self.estimator).set_params(**estimator_params)

        if self.local_group_names_ is None:
            X_tab, y_tab = self._get_tabularized_dataset(
                y_t,
                X_t,
                forecasting_horizon,
                y_pred_local_columns,
            )

        else:
            X_tab_list, y_tab_list = [], []
            for local_group_name in self.local_group_names_:
                y_t_local = y_t[
                    [
                        col
                        for col, dtype in y_t.schema.items()
                        if dtype != pl.Struct or col == local_group_name
                    ]
                ].unnest(local_group_name)
                y_t_local = y_t_local.select(["time"] + y_pred_local_columns)

                X_t_local = X_t[
                    [
                        col
                        for col, dtype in X_t.schema.items()
                        if dtype != pl.Struct or col == local_group_name
                    ]
                ].unnest(local_group_name)
                X_t_local = X_t_local.select(["time"] + self.local_X_t_names_)

                X_tab_local, y_tab_local = self._get_tabularized_dataset(
                    y_t_local,
                    X_t_local,
                    forecasting_horizon,
                    y_pred_local_columns,
                )

                X_tab_list.append(X_tab_local)
                y_tab_list.append(y_tab_local)

            X_tab = np.vstack(X_tab_list)
            y_tab = np.vstack(y_tab_list)

        estimator.fit(X_tab, y_tab)

        return estimator

    def _estimator_predict_one(
        self,
        estimator: RegressorMixin,
        y_pred_local_columns: list[str] | None = None,
    ) -> pl.DataFrame:
        # TODO: Adapt for cross-learning
        if y_pred_local_columns is None:
            y_pred_local_columns = self.local_y_names_

        X_t = self._X_t_observed[[-1]].select(~cs.by_name("time"))

        if self.local_group_names_ is None:
            X_tab = X_t.select(self.local_X_t_names_).to_numpy()
            y_tab_pred = estimator.predict(X_tab)
            y_pred = pl.DataFrame(
                y_tab_pred.reshape(self.fit_forecasting_horizon_, len(y_pred_local_columns)),
                schema=y_pred_local_columns,
            )

        else:
            y_pred_dict = {}
            for local_group_name in self.local_group_names_:
                X_tab = X_t[
                    [
                        col
                        for col, dtype in X_t.schema.items()
                        if dtype != pl.Struct or col == local_group_name
                    ]
                ].unnest(local_group_name)
                X_tab = X_tab.select(self.local_X_t_names_).to_numpy()

                y_tab_pred = estimator.predict(X_tab)
                y_pred_dict[local_group_name] = pl.DataFrame(
                    y_tab_pred.reshape(self.fit_forecasting_horizon_, len(y_pred_local_columns)),
                    schema=y_pred_local_columns,
                )

            y_pred = pl.DataFrame(y_pred_dict)

        return y_pred


class BaseWrapper(BaseEstimator, metaclass=abc.ABCMeta):  # type: ignore[misc]
    """Base class for wrapping classes into scikit-learn
    estimators.

    This class is meant to wrap any class into a scikit-learn
    estimator equipped with a `get_params` and a `set_params`
    method.

    Parameters
    ----------
    estimator_class : class
        Class to be wrapped.

    **params
        Parameters to the constructor of the class to be wrapped.
    """

    _required_parameters = ["estimator_class"]
    _estimator_name: Optional[str] = None
    _estimator_base_class = None

    def __init__(self, estimator_class: type, **params: object) -> None:
        self.estimator_class = estimator_class
        self.params = self._validate_estimator_params(dict(params))

    @property
    def estimator_name(self) -> str:
        if not isinstance(self._estimator_name, str):
            raise ValueError("Class should define a static `_estimator_name`.")

        return self._estimator_name

    @property
    def estimator_base_class(self) -> type:
        if self._estimator_base_class is None:
            raise ValueError("Class should define a static `_estimator_base_class`.")

        return self._estimator_base_class

    def _validate_estimator_params(self, params: dict[str, object]) -> dict[str, object]:
        """
        Validate estimator parameters.

        Check the estimator parameter names and set the omitted ones
        to their default value as per the ``estimator_class``
        constructor.

        Parameters
        ----------
        params : dict
            Dictionary of estimator parameters.

        Returns
        -------
        dict
            Validated dictionary of estimator parameters.
        """
        # Get constructor via type to avoid instance __init__ access issue
        constructor = self.estimator_class.__init__  # type: ignore[misc]
        constructor_signature = inspect.signature(constructor)
        valid_class_params = {
            key: val.default
            for key, val in constructor_signature.parameters.items()
            if key != "self"
        }

        validated_params = {}
        for param_name, param_val in params.items():
            if param_name not in valid_class_params:
                raise ValueError(
                    f"{param_name} is not a valid parameter for class {self.estimator_class}."
                )

            validated_params[param_name] = param_val

        for param_name, param_val in valid_class_params.items():
            if param_name not in params:
                if param_val is inspect._empty:
                    param_val = REQUIRED_PARAM_VALUE

                validated_params[param_name] = param_val

        return validated_params

    def _validate_params(self) -> None:
        """Validate types and values of constructor parameters

        The expected type and values must be defined in the `_parameter_constraints`
        class attribute, which is a dictionary `param_name: list of constraints`. See
        the docstring of `validate_parameter_constraints` for a description of the
        accepted constraints.
        """
        if not issubclass(self.estimator_class, self.estimator_base_class):
            caller_name = self.__class__.__name__

            raise InvalidParameterError(
                f"The {self.estimator_name!r} parameter of {caller_name} must be "
                f" a sub class of {self.estimator_base_class}. Got "
                f"{self.estimator_class!r} instead."
            )

    def instantiate(self) -> "BaseWrapper":
        self._validate_params()

        for param_name, param_value in self.params.items():
            if param_value == REQUIRED_PARAM_VALUE:
                raise ValueError(
                    f"Class `{self.estimator_class.__name__}` requires parameter {param_name}."
                )

        # Create instance by calling the class constructor
        self.instance_ = self.estimator_class(**self.params)

        return self

    def get_params(self, deep: bool = True) -> dict[str, object]:
        """
        Get parameters for this estimator.

        Parameters
        ----------
        deep : bool, default=True
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.

        Returns
        -------
        params : dict
            Parameter names mapped to their values.
        """
        params: dict[str, object] = {self.estimator_name: self.estimator_class}

        params.update(self.params)

        return params

    def set_params(self, **params: object) -> "BaseWrapper":
        """Set the parameters of this estimator.

        The method works on simple estimators as well as on nested objects
        (such as :class:`~sklearn.pipeline.Pipeline`). The latter have
        parameters of the form ``<component>__<parameter>`` so that it's
        possible to update each component of a nested object.

        Parameters
        ----------
        **params : dict
            Estimator parameters.

        Returns
        -------
        self : estimator instance
            Estimator instance.
        """
        if self.estimator_name in params:
            estimator_class_value = params.pop(self.estimator_name)
            if isinstance(estimator_class_value, type):
                self.estimator_class = estimator_class_value

        self.params = self._validate_estimator_params(params)

        return self
