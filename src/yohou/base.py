"""Core base classes for transformers, forecasters, and wrappers."""

import abc
import inspect
from copy import deepcopy
from typing import Literal

import numpy as np
import polars as pl
import polars.selectors as cs
from pydantic import StrictInt
from sklearn.base import (
    BaseEstimator,
    TransformerMixin,
    clone,
)
from sklearn.linear_model import LinearRegression
from sklearn.utils._param_validation import InvalidParameterError
from sklearn.utils.validation import check_is_fitted

from yohou.utils import add_interval, check_inputs, concat_struct, inspect_locality, tabularize

PredictionType = Literal["point", "interval"]

__all__ = ["BaseTransformer", "BaseForecaster", "BaseWrapper", "PredictionType"]


REQUIRED_PARAM_VALUE = "__REQUIRED__"


class BaseTransformer(BaseEstimator, TransformerMixin, metaclass=abc.ABCMeta):
    """Base class for time series transformers."""

    @property
    def observation_horizon(self) -> int:
        """Get the number of time steps needed for stateful operations.

        The observation horizon defines how many recent observations the forecaster
        needs to maintain in its memory.

        Returns
        -------
        int
            Number of time steps to retain.

        Raises
        ------
        NotFittedError
            If the transformer has not been fitted yet.

        """
        check_is_fitted(self, "_observation_horizon")
        return self._observation_horizon

    def fit(self, X: pl.DataFrame, y: pl.DataFrame | None = None) -> "BaseTransformer":
        """Fits the transformer and returns it.

        Parameters
        ----------
        X : pl.DataFrame
            Input time series.

        y : pl.DataFrame or None, default=None
            Target time series. Ignored and only present for API consistency.

        Returns
        -------
        self

        """
        self.reset(X)

        self.feature_names_in_ = X.select(~cs.by_name("time")).columns
        self.n_features_in_ = len(self.feature_names_in_)

        return self

    def reset(self, X: pl.DataFrame) -> "BaseTransformer":
        """Resets the transformer by resetting the observation horizon.

        Parameters
        ----------
        X : pl.DataFrame
            Input time series.

        Returns
        -------
        self

        """
        if len(X) < self.observation_horizon:
            raise ValueError("Not enough input data to set the transformer memory.")

        if self.observation_horizon > 0:
            self._X_observed = X[-self.observation_horizon :]
        else:
            self._X_observed = X[:0]

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

    def inverse_transform(self, X_t: pl.DataFrame, X_p: pl.DataFrame | None) -> pl.DataFrame:
        """Inverts the input transformed time series.

        Parameters
        ----------
        X_t : pl.DataFrame
            Transformed time series.

        X_p : pl.DataFrame or None
            Untransformed time series corresponding to at least `observation_horizon` immediately
            previous time stamps. Can be None if `observation_horizon == 0`.

        Returns
        -------
        pl.DataFrame
            Inverted transformed time series.
        """
        raise NotImplementedError("This transformer is not invertible.")

    def update_transform(self, X: pl.DataFrame) -> pl.DataFrame:
        """Transforms the input, updates the transformer and returns
        the transformed input.

        Parameters
        ----------
        X : pl.DataFrame
            Input time series.

        Returns
        -------
        pl.DataFrame
            Transformed input time series.

        """
        if self.observation_horizon > 0:
            X_full = pl.concat([self._X_observed, X])
            X_t = self.transform(X_full)
            X_t = X_t[-len(X) :]
        else:
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


class BaseForecaster(BaseEstimator, metaclass=abc.ABCMeta):
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
        target_transformer: BaseTransformer | None = None,
        feature_transformer: BaseTransformer | None = None,
    ) -> None:
        self.target_transformer = target_transformer
        self.feature_transformer = feature_transformer

    @property
    @abc.abstractmethod
    def prediction_types(self) -> set[PredictionType]:
        """Get the types of predictions this forecaster produces.

        Returns
        -------
        set of {"point", "interval"}
            Set of prediction types produced by this forecaster.
            Point forecasters return {"point"}, interval forecasters return {"interval"},
            and forecasters producing both return {"point", "interval"}.

        """
        raise NotImplementedError()

    def _set_local_groups(
        self, y: pl.DataFrame, X_post: pl.DataFrame | None, X_ante: pl.DataFrame | None
    ) -> None:
        """Detect and validate panel data structure across target and features.

        Inspects whether the data contains global (single time series) or local
        (panel/struct columns with multiple time series) and ensures consistency
        across y, X_post, and X_ante. Sets instance attributes for downstream use.

        Parameters
        ----------
        y : pl.DataFrame
            Target time series, may contain struct columns for panel data.

        X_post : pl.DataFrame or None
            Ex-ante features, may contain struct columns.

        X_ante : pl.DataFrame or None
            Ex-post features, may contain struct columns.

        Returns
        -------
        None
            Sets the following attributes:
            - `local_group_names_` : list of str or None
                Names of struct columns representing local groups (e.g., ["stores"])
            - `local_y_names_` : list of str
                Names of target columns within each local group
            - `local_X_names_` : list of str
                Names of feature columns (global + local combined)

        Raises
        ------
        ValueError
            If y contains both global and local columns (ambiguous structure),
            or if struct column field names don't match across local groups,
            or if X_post/X_ante local groups don't match y's structure.

        Notes
        -----
        Panel data example:
            y has struct column "sales" with fields ["store_1", "store_2"]
            → local_group_names_ = ["sales"]
            → local_y_names_ = ["store_1", "store_2"]

        Global data example:
            y has regular columns ["sales"]
            → local_group_names_ = None
            → local_y_names_ = ["sales"]

        See Also
        --------
        :func:`yohou.utils.polars.inspect_locality` : Detects struct columns

        """
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

        local_X_post_names = []
        if X_post is not None:
            X_post_global_names, X_post_local_groups = inspect_locality(X_post)
            local_X_post_names = X_post_global_names

            if len(X_post_local_groups):
                if local_group_names:
                    local_X_post_names += X_post_local_groups[local_group_names[0]]
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

        local_X_ante_names = []
        if X_ante is not None:
            X_ante_global_names, X_ante_local_groups = inspect_locality(X_ante)
            local_X_ante_names = X_ante_global_names

            if len(X_ante_local_groups):
                if local_group_names:
                    local_X_ante_names += X_post_local_groups[local_group_names[0]]
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

        self.local_group_names_ = local_group_names
        self.local_y_names_ = local_y_names
        self.local_X_names_ = local_X_post_names + local_X_ante_names

    def _fit_transform_one(
        self, y: pl.DataFrame, X: pl.DataFrame | None
    ) -> tuple[pl.DataFrame, pl.DataFrame, BaseTransformer | None, BaseTransformer | None]:
        """Fit and apply target and feature transformers to a single time series.

        Orchestrates the transformation pipeline: target transformer first (if any),
        then feature transformer (if any). Handles observation horizon alignment to
        ensure transformed data matches temporally.

        Parameters
        ----------
        y : pl.DataFrame
            Target time series with "time" column.

        X : pl.DataFrame or None
            Combined features (X_post + X_ante) with "time" column, or None.

        Returns
        -------
        y_t : pl.DataFrame
            Transformed target time series.

        X_t : pl.DataFrame
            Transformed feature matrix (includes transformed y if no separate X provided).

        target_transformer : BaseTransformer or None
            Fitted target transformer instance, or None if not used.

        feature_transformer : BaseTransformer or None
            Fitted feature transformer instance, or None if not used.

        Notes
        -----
        Transformation order matters:
        1. Apply target_transformer to y → y_t
        2. Concatenate y_t with X (aligned by observation horizon)
        3. Apply feature_transformer to combined → X_t
        4. Trim y_t if feature transformer has its own observation horizon

        This ensures features can include lagged versions of the transformed target.

        See Also
        --------
        :class:`BaseTransformer` : Base class for transformers

        """
        y_t = y
        target_transformer, target_observation_horizon = None, 0
        if self.target_transformer is not None:
            target_transformer = clone(self.target_transformer)
            y_t = target_transformer.fit_transform(y)
            target_observation_horizon = target_transformer.observation_horizon

        X_t = y_t
        if X is not None:
            X_t = pl.concat(
                [y_t, X.select(~cs.by_name("time"))[target_observation_horizon:]],
                how="horizontal",
            )

        feature_transformer = None
        if self.feature_transformer is not None:
            feature_transformer = clone(self.feature_transformer)
            X_t = feature_transformer.fit_transform(X_t)
            feature_observation_horizon = feature_transformer.observation_horizon
            y_t = y_t[feature_observation_horizon:]

        return y_t, X_t, target_transformer, feature_transformer

    def _update_one(
        self,
        y: pl.DataFrame,
        X: pl.DataFrame | None,
        target_transformer: BaseTransformer | None,
        feature_transformer: BaseTransformer | None,
    ) -> pl.DataFrame:
        """Update transformers with new observations.

        Parameters
        ----------
        y : pl.DataFrame
            New target observations.

        X : pl.DataFrame or None
            New features.

        target_transformer : BaseTransformer or None
            Target transformer to update.

        feature_transformer : BaseTransformer or None
            Feature transformer to update.

        Returns
        -------
        pl.DataFrame
            Transformed new observations.

        """
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
        X_post: pl.DataFrame | None = None,
        X_ante: pl.DataFrame | None = None,
    ) -> tuple[pl.DataFrame, pl.DataFrame | None]:
        """Combine target and exogenous features into standard format.

        Merges X_post (ex-ante features) and X_ante (ex-post features) into a single
        feature DataFrame X, handling None cases. Ex-post features are filtered to
        match the target's time index.

        Parameters
        ----------
        y : pl.DataFrame
            Target time series with "time" column.

        X_post : pl.DataFrame or None, default=None
            Ex-ante features (known in advance) with "time" column.

        X_ante : pl.DataFrame or None, default=None
            Ex-post features (observed after) with "time" column.

        Returns
        -------
        y : pl.DataFrame
            Target unchanged (passed through).

        X : pl.DataFrame or None
            Combined features with "time" column, or None if no features provided.
            Structure: [time, X_post columns..., X_ante columns...]

        Notes
        -----
        Ex-ante vs Ex-post:
        - Ex-ante: Known before forecasting (holidays, planned promotions)
        - Ex-post: Only known at prediction time (actual weather, traffic)

        X_ante is filtered to match y's time index since it's only available
        for observed periods.

        """
        time = y.select(cs.by_name("time"))

        X = None
        self.X_post_columns = None
        if X_post is not None:
            X_post = X_post.select(~cs.by_name("time"))
            X = X_post

        self.X_ante_columns_ = None
        if X_ante is not None:
            if X_post is None:
                X = pl.DataFrame()

            # Join X_ante with y to align on time
            X_ante = (
                y.select("time").join(X_ante, on="time", how="inner").select(~cs.by_name("time"))
            )
            if X is not None:
                X = pl.concat([X, X_ante], how="horizontal")
            else:
                X = X_ante

        if X is not None:
            X = pl.concat([time, X], how="horizontal")

        return y, X

    def _pre_fit(
        self,
        y: pl.DataFrame,
        X_post: pl.DataFrame | None = None,
        X_ante: pl.DataFrame | None = None,
        forecasting_horizon: StrictInt = 1,
    ) -> tuple[pl.DataFrame, pl.DataFrame]:
        """Preprocess and transform inputs before fitting.

        Parameters
        ----------
        y : pl.DataFrame
            Target time series.

        X_post : pl.DataFrame or None, default=None
            Ex-ante features.

        X_ante : pl.DataFrame or None, default=None
            Ex-post features.

        forecasting_horizon : int, default=1
            Number of steps ahead to forecast.

        Returns
        -------
        y_t : pl.DataFrame
            Transformed target.

        X_t : pl.DataFrame
            Transformed features.

        """
        self.interval_ = check_inputs(y, X_post, X_ante)
        self._set_local_groups(y, X_post, X_ante)

        if forecasting_horizon < 1:
            raise ValueError(
                f"`forecasting_horizon` should be a positive int. It is: {forecasting_horizon}"
            )

        self.fit_forecasting_horizon_ = forecasting_horizon

        y, X = self._preprocess_inputs(y, X_post, X_ante)

        y_t, X_t = self._fit_transform_inputs(y, X)

        # TODO: Do we need to keep all in memory?
        self._X_t_observed = X_t

        if self.local_group_names_ is None:
            # Store untransformed data for inverse_transform
            if self.target_transformer_ is not None and hasattr(
                self.target_transformer_, "observation_horizon"
            ):
                target_obs_horizon = self.target_transformer_.observation_horizon
                if target_obs_horizon > 0:
                    self._y_observed = y[-target_obs_horizon:]
                else:
                    # Stateless transformer: filter by transformed times
                    self._y_observed = y.filter(pl.col("time").is_in(X_t["time"].to_list()))
            else:
                # No transformer: filter by transformed times
                self._y_observed = y.filter(pl.col("time").is_in(X_t["time"].to_list()))

            self._X_post_observed = None
            if X_post is not None:
                self._X_post_observed = X_post.filter(pl.col("time").is_in(X_t["time"].to_list()))

            return y_t, X_t

        # TODO Check it is correct
        # Panel data case: handle observation storage similarly to global case
        # but account for potential observation horizon from transformers
        if self.target_transformer_ is not None and isinstance(self.target_transformer_, dict):
            # Get observation horizon from first local transformer
            first_group = self.local_group_names_[0]
            first_transformer = self.target_transformer_[first_group]
            if hasattr(first_transformer, "observation_horizon"):
                target_obs_horizon = first_transformer.observation_horizon
                if target_obs_horizon > 0:
                    self._y_observed = y[-target_obs_horizon:]
                else:
                    # Stateless transformer: filter by transformed times
                    self._y_observed = y.filter(pl.col("time").is_in(X_t["time"].to_list()))
            else:
                self._y_observed = y.filter(pl.col("time").is_in(X_t["time"].to_list()))
        else:
            # No transformer: filter by transformed times
            self._y_observed = y.filter(pl.col("time").is_in(X_t["time"].to_list()))

        self._X_post_observed = None
        if X_post is not None:
            self._X_post_observed = X_post.filter(pl.col("time").is_in(X_t["time"].to_list()))

        return y_t, X_t

    def _fit_transform_inputs(
        self, y: pl.DataFrame, X: pl.DataFrame | None
    ) -> tuple[pl.DataFrame, pl.DataFrame]:
        """Fit the transformers and transform inputs.

        Parameters
        ----------
        y : pl.DataFrame
            Target time series.

        X : pl.DataFrame or None
            Feature time series.

        Returns
        -------
        y_t : pl.DataFrame
            Transformed target.

        X_t : pl.DataFrame
            Transformed features.

        """
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
        """Update transformers with new inputs.

        Parameters
        ----------
        y : pl.DataFrame
            New target observations.

        X : pl.DataFrame or None
            New features.

        Returns
        -------
        pl.DataFrame
            Transformed new observations.

        """
        y = y.select(["time"] + self.local_y_names_)
        if X is not None:
            X = X.select(["time"] + self.local_X_names_)

        X_t_new = self._update_one(y, X, self.target_transformer_, self.feature_transformer_)

        return X_t_new

    @abc.abstractmethod
    def fit(
        self,
        y: pl.DataFrame,
        X_post: pl.DataFrame | None = None,
        X_ante: pl.DataFrame | None = None,
        forecasting_horizon: StrictInt = 1,
    ) -> "BaseForecaster":
        """Fits the forecaster and returns it.

        Parameters
        ----------
        y : pl.DataFrame
            Target time series.

        X_post : pl.DataFrame or None, default=None
            Ex-ante feature time series.

        X_ante : pl.DataFrame or None, default=None
            Ex-post feature time series.

        forecasting_horizon : int >= 1, default=1
            Horizon to forecast.

        Returns
        -------
        self

        """
        raise NotImplementedError()

    # TODO: Use this to predict the train set
    def reset(
        self,
        y: pl.DataFrame,
        X_post: pl.DataFrame | None = None,
        X_ante: pl.DataFrame | None = None,
    ) -> "BaseForecaster":
        """Resets the forecaster by resetting the observation horizon.

        Parameters
        ----------
        y : pl.DataFrame
            Target time series.

        X_post : pl.DataFrame or None
            Ex-ante feature time series.

        X_ante : pl.DataFrame or None
            Ex-post feature time series.

        Returns
        -------
        self

        """
        self._y_observed = y
        self._X_post_observed = X_post

        return self

    def update(
        self,
        y: pl.DataFrame,
        X_post: pl.DataFrame | None = None,
        X_ante: pl.DataFrame | None = None,
    ) -> "BaseForecaster":
        """Updates the forecaster with more recent data and
        returns it.

        Parameters
        ----------
        y : pl.DataFrame
            Target time series.

        X_post : pl.DataFrame or None
            Ex-ante feature time series.

        X_ante : pl.DataFrame or None
            Ex-post feature time series.


        Returns
        -------
        self

        """
        check_is_fitted(self, "fit_forecasting_horizon_")
        y, X = self._preprocess_inputs(y, X_post, X_ante)

        X_t_new = self._update_inputs(y, X)

        self._y_observed = pl.concat([self._y_observed, y])
        self._X_t_observed = pl.concat([self._X_t_observed, X_t_new])
        if X_post is not None:
            if self._X_post_observed is not None:
                self._X_post_observed = pl.concat([self._X_post_observed, X_post])
            else:
                self._X_post_observed = X_post

        return self

    def _add_time_columns(self, y_pred: pl.DataFrame) -> pl.DataFrame:
        """Add time metadata columns to predictions.

        Parameters
        ----------
        y_pred : pl.DataFrame
            Predictions without time columns.

        Returns
        -------
        pl.DataFrame
            Predictions with observed_time and time columns.

        """
        time = self._y_observed[-self.fit_forecasting_horizon_ :][["time"]]

        # Use add_interval to handle both fixed and variable intervals
        observed_time = self._y_observed["time"][-1]
        predicted_times = [
            add_interval(t, self.interval_, self.fit_forecasting_horizon_)
            for t in time["time"].to_list()
        ]

        time = pl.DataFrame(
            {"observed_time": [observed_time] * len(predicted_times), "time": predicted_times}
        )

        y_pred = pl.concat([time, y_pred], how="horizontal")

        return y_pred

    def _predict_one(self) -> pl.DataFrame:
        """Generate one-step or multi-step prediction (must be implemented by subclasses).

        Returns
        -------
        pl.DataFrame
            Predicted values for the forecasting horizon.

        """
        raise NotImplementedError()

    @staticmethod
    def _predict(
        forecaster: "BaseForecaster",
        cross_learning_group: str | None = None,
    ) -> tuple[pl.DataFrame, pl.DataFrame]:
        """Generate one-step or multi-step prediction.

        Parameters
        ----------
        forecaster : BaseForecaster
            Fitted forecaster to use for prediction.

        cross_learning_group : str or None, default=None
            Group to forecast in case of cross-learning.

        Returns
        -------
        y_pred_step : pl.DataFrame
            Predicted time series in transformed space.

        y_pred_step_inv : pl.DataFrame
            Inverse transformed predicted time series (original scale).

        """
        y_pred_step = forecaster._predict_one()

        y_pred_step_inv = y_pred_step
        if forecaster.target_transformer is not None:
            if cross_learning_group is None:
                # Remove "observed_time" before inverse_transform (transformers don't handle it)
                observed_time = y_pred_step.select(cs.by_name("observed_time"))
                y_pred_step_no_obs = y_pred_step.select(~cs.by_name("observed_time"))

                y_pred_step_inv = forecaster.target_transformer_.inverse_transform(
                    X_t=y_pred_step_no_obs,
                    X_p=forecaster._y_observed,
                )

                # Add "observed_time" back
                y_pred_step_inv = pl.concat([observed_time, y_pred_step_inv], how="horizontal")
            else:
                y_pred_step_inv_dict = {}

                transformer = forecaster.target_transformer_[cross_learning_group]

                # Extract the group's data
                y_pred_step_group = y_pred_step.select(
                    cs.by_name("observed_time")
                    | cs.by_name("time")
                    | cs.by_name(cross_learning_group)
                ).unnest(cross_learning_group)

                # Remove "observed_time" before inverse_transform (transformers don't handle it)
                observed_time = y_pred_step_group.select(cs.by_name("observed_time"))
                y_pred_step_group_no_obs = y_pred_step_group.select(
                    ~cs.by_name("observed_time")
                )

                # Inverse transform
                y_pred_step_group_inv = transformer.inverse_transform(
                    X_t=y_pred_step_group_no_obs,
                    X_p=forecaster._y_observed.select(
                        cs.by_name("time") | cs.by_name(cross_learning_group)
                    ).unnest(cross_learning_group),
                )

                # Add "observed_time" back
                y_pred_step_group_inv = pl.concat(
                    [observed_time, y_pred_step_group_inv], how="horizontal"
                )

                # Store in dict
                y_pred_step_inv_dict[cross_learning_group] = y_pred_step_group_inv.select(
                    ~cs.by_name("observed_time") & ~cs.by_name("time")
                )

                # Reconstruct the struct
                y_pred_step_inv = concat_struct(y_pred_step_inv_dict, how="horizontal")

                # Add time columns back
                time_cols = y_pred_step.select(
                    cs.by_name("observed_time") | cs.by_name("time")
                )
                y_pred_step_inv = pl.concat([time_cols, y_pred_step_inv], how="horizontal")

        return y_pred_step, y_pred_step_inv

    @abc.abstractmethod
    def predict(
        self,
        X_ante: pl.DataFrame | None = None,
        forecasting_horizon: StrictInt = 1,
        cross_learning_group: str | None = None,
        predict_transformed: bool = True,
    ) -> pl.DataFrame:
        """Predicts the model forecasting horizon from the observation horizon.

        Parameters
        ----------
        X_ante : pl.DataFrame or None, default=None
            Ex-post feature time series.

        forecasting_horizon : int >= 1, default=1
            Horizon to forecast.

        cross_learning_group : str or None, default=None
            For panel data (local_group_names_ is not None):
            - If None: predict for all groups (default behavior)
            - If str: predict only for the specified group (cross-learning)
            For global data: parameter is ignored.

        predict_transformed : bool, default=True
            If ``True``, the predictions are returned in the transformed space.

        Returns
        -------
        pl.DataFrame
            Predicted time series.

        """
        raise NotImplementedError()

    def update_predict(
        self,
        y: pl.DataFrame,
        X_post: pl.DataFrame | None = None,
        X_ante: pl.DataFrame | None = None,
        forecasting_horizon: StrictInt = 1,
        stride: StrictInt = 1,
        predict_transformed: bool = False,
    ) -> pl.DataFrame:
        """Alternate `recursive_predict` and `update`.

        Parameters
        ----------
        y : pl.DataFrame
            Target time series for updates.

        X_post : pl.DataFrame or None, default=None
            Ex-ante feature time series for updates.

        X_ante : pl.DataFrame or None, default=None
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
        y_pred_i = self.predict(X_ante, forecasting_horizon=forecasting_horizon)

        y_pred = y_pred_i
        for i in range(0, len(y), stride):
            X_post_slice = None
            if X_post is not None:
                X_post_slice = X_post[i : i + stride]

            self.update(y=y[i : i + stride], X_post=X_post_slice, X_ante=X_ante)

            y_pred_i = self.predict(
                X_ante=X_ante,
                forecasting_horizon=forecasting_horizon,
                predict_transformed=predict_transformed,
            )

            y_pred = pl.concat([y_pred, y_pred_i])

        return y_pred


class BaseReductionForecaster(BaseForecaster, metaclass=abc.ABCMeta):
    """Base class for forecasters using reduction to supervised learning.

    Converts the time series forecasting task to a tabular one.

    Parameters
    ----------
    estimator : instance of `BaseEstimator`, default=LinearRegression()
        Estimator used to fit the tabularized data.

    reduction_strategy : {"direct", "multi-output"}, default="multi-output"
        Reduction strategy to use.

    target_transformer : instance of `BaseTransformer` or None, default=None
        Transformer used to transform the target time series into the new target.

    feature_transformer : instance of `BaseTransformer` or None, default=None
        Transformer used to transform the target time series into features.

    Notes
    -----
    Reduction strategies:
    - Direct: Separate model for each horizon step; predicts directly from inputs.
    - Multi-output: Single model predicts all horizon steps simultaneously.

    All models can be applied recursively for multi-step forecasting by specifying
    the forecasting horizon during prediction.

    See Also
    --------
    :class:`yohou.point_forecaster.PointReductionForecaster` : Point forecaster using
    reduction
    :class:`yohou.interval_forecaster.IntervalReductionForecaster` : Interval forecaster
    using reduction
    """

    def __init__(
        self,
        estimator: BaseEstimator = LinearRegression(),
        reduction_strategy: Literal["direct", "multi-output"] = "multi-output",
        target_transformer: BaseTransformer | None = None,
        feature_transformer: BaseTransformer | None = None,
    ):
        BaseForecaster.__init__(
            self,
            target_transformer=target_transformer,
            feature_transformer=feature_transformer,
        )

        self.estimator = estimator
        self.reduction_strategy = reduction_strategy

    def _fit_transform_inputs(
        self,
        y: pl.DataFrame,
        X: pl.DataFrame | None,
    ) -> tuple[pl.DataFrame, pl.DataFrame]:
        """Fit and transform inputs.

        Parameters
        ----------
        y : pl.DataFrame
            Target time series (may contain struct columns for panel data).

        X : pl.DataFrame or None
            Feature time series.

        Returns
        -------
        y_t : pl.DataFrame
            Transformed target.

        X_t : pl.DataFrame
            Transformed features.

        """
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

        self.target_transformer_ = target_transformer_dict
        self.feature_transformer_ = feature_transformer_dict

        time = y_t_local.select(cs.by_name("time"))
        y_t = pl.concat([time, pl.DataFrame(y_t_dict)], how="horizontal")
        X_t = pl.concat([time, pl.DataFrame(X_t_dict)], how="horizontal")

        return (
            y_t,
            X_t,
        )

    def _update_inputs(self, y: pl.DataFrame, X: pl.DataFrame | None) -> pl.DataFrame:
        """Update forecaster and transformers with new observations.

        Parameters
        ----------
        y : pl.DataFrame
            New target observations.

        X : pl.DataFrame or None
            New features.

        Returns
        -------
        pl.DataFrame
            Transformed new observations.

        """
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
        """Convert transformed time series to tabular supervised learning format.

        Creates feature matrix (X_tab) and target matrix (y_tab) suitable for training
        sklearn regressors. Target columns are lagged and renamed to indicate forecast
        steps (lag_1 → step_1 for 1-step-ahead prediction, etc.).

        Parameters
        ----------
        y_t : pl.DataFrame
            Transformed target time series.

        X_t : pl.DataFrame
            Transformed feature matrix (may include lagged y_t).

        forecasting_horizon : int
            Number of steps to forecast (determines how many lag features needed).

        y_pred_local_columns : list of str or None
            Target column names to predict. If None, uses all local_y_names_.

        Returns
        -------
        X_tab : np.ndarray of shape (n_samples, n_features)
            Feature matrix for supervised learning. Excludes "time" column and
            truncates last forecasting_horizon rows (no targets available).

        y_tab : np.ndarray of shape (n_samples, forecasting_horizon * n_targets)
            Target matrix with columns for each (target, step) combination.
            Columns follow pattern: {target}_step_{1}, {target}_step_{2}, ...

        Notes
        -----
        Lag-to-step renaming convention:
        - Input: y with lag_0, lag_1, lag_2, ..., lag_H features
        - For forecasting_horizon=3:
            - lag_1 → step_1 (1-step-ahead target)
            - lag_2 → step_2 (2-step-ahead target)
            - lag_3 → step_3 (3-step-ahead target)
            - lag_0 is the most recent observation (not a target)

        This convention makes it clear that we're predicting future values, not
        explaining historical ones.

        See Also
        --------
        :func:`yohou.utils.tabularization.tabularize` : Creates lagged features

        """
        if y_pred_local_columns is None:
            y_pred_local_columns = self.local_y_names_  # TODO: Should y_t column names

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
        """Fit an sklearn estimator on tabularized time series data.

        Converts time series to supervised learning format and trains the estimator.
        Handles both global (single series) and local (panel data) cases, stacking
        panel data vertically for training a single global model.

        Parameters
        ----------
        y_t : pl.DataFrame
            Transformed target time series.

        X_t : pl.DataFrame
            Transformed feature matrix.

        forecasting_horizon : int
            Number of steps to forecast.

        y_pred_local_columns : list of str or None, default=None
            Target columns to predict. If None, predicts all targets.

        **estimator_params : object
            Additional parameters to pass to the estimator's set_params method.

        Returns
        -------
        BaseEstimator
            Fitted sklearn regressor.

        Notes
        -----
        For panel data (local_group_names_ is not None):
        - Unnests each struct column
        - Tabularizes each local time series separately
        - Stacks all series vertically (X_tab = vstack of all local X_tabs)
        - Trains single model across all series (global model)

        For global data:
        - Directly tabularizes and fits

        This enables "global forecasting" where patterns learned from multiple
        series can benefit predictions for all series.

        See Also
        --------
        :meth:`_get_tabularized_dataset` : Creates supervised learning matrices
        :meth:`_estimator_predict_one` : Uses fitted model for prediction

        """
        # TODO: Is y_pred_local_columns the right name?
        # How should it be handled in _get_tabularized_dataset?
        if y_pred_local_columns is None:
            y_pred_local_columns = self.local_y_t_names_  # TODO: Was self.local_y_names_

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
        estimator: BaseEstimator,
        y_pred_local_columns: list[str] | None = None,
    ) -> pl.DataFrame:
        """Generate predictions using fitted estimator on tabularized data.

        Parameters
        ----------
        estimator : BaseEstimator
            Fitted scikit-learn estimator.

        y_pred_local_columns : list of str or None, default=None
            Column names for predictions.

        Returns
        -------
        pl.DataFrame
            Predictions for the forecasting horizon.

        """
        # TODO: Adapt for cross-learning
        if y_pred_local_columns is None:
            y_pred_local_columns = self.local_y_names_

        X_t = self._X_t_observed[[-1]].select(~cs.by_name("time"))

        if self.local_group_names_ is None:
            X_tab = X_t.select(self.local_X_t_names_).to_numpy()
            y_tab_pred = estimator.predict(X_tab)  # type: ignore[attr-defined]
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

                y_tab_pred = estimator.predict(X_tab)  # type: ignore[attr-defined]
                y_pred_local = pl.DataFrame(
                    y_tab_pred.reshape(self.fit_forecasting_horizon_, len(y_pred_local_columns)),
                    schema=y_pred_local_columns,
                )

                y_pred_dict[local_group_name] = y_pred_local

            y_pred = pl.DataFrame(y_pred_dict)

        return y_pred


class BaseWrapper(BaseEstimator, metaclass=abc.ABCMeta):
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
    _estimator_name: str | None = None
    _estimator_base_class = None

    def __init__(self, estimator_class: type, **params: object) -> None:
        self.estimator_class = estimator_class
        self.params = self._validate_estimator_params(dict(params))

    @property
    def estimator_name(self) -> str:
        """Get the name of the wrapped estimator type.

        Returns
        -------
        str
            The estimator name.

        """
        if not isinstance(self._estimator_name, str):
            raise ValueError("Class should define a static `_estimator_name`.")

        return self._estimator_name

    @property
    def estimator_base_class(self) -> type:
        """Get the required base class for the wrapped estimator.

        Returns
        -------
        type
            The base class.

        """
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
        constructor = self.estimator_class.__init__
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
        """Validate parameters and create an instance.

        Returns
        -------
        self

        """
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
