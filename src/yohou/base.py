"""Core base classes for transformers, forecasters, and wrappers."""

import abc
import inspect
from typing import Any, Callable, Literal

import numpy as np
import polars as pl
import polars.selectors as cs
from pydantic import StrictInt
from sklearn.base import (
    BaseEstimator,
    TransformerMixin,
    _fit_context,
    clone,
)
from sklearn.linear_model import LinearRegression
from sklearn.utils._param_validation import InvalidParameterError
from sklearn.utils.metadata_routing import MetadataRouter, MethodMapping
from sklearn.utils.validation import check_is_fitted

from yohou.utils import (
    add_interval,
    cast,
    check_inputs,
    check_schema,
    get_group_df,
    inspect_locality,
    tabularize,
)

PredictionType = Literal["point", "interval"]

__all__ = ["BaseTransformer", "BaseForecaster", "BaseWrapper", "PredictionType"]


REQUIRED_PARAM_VALUE = "__REQUIRED__"


class BaseTransformer(BaseEstimator, TransformerMixin, metaclass=abc.ABCMeta):
    """Base class for time series transformers."""

    _parameter_constraints: dict = {}

    @property
    def observation_horizon(self) -> int:
        """Get the number of time steps needed for stateful operations.

        The observation horizon defines how many recent observations the transformer
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

    def _update_X_observed(self, X: pl.DataFrame) -> None:
        """Update stored observed data for stateful transformations.

        Parameters
        ----------
        X : pl.DataFrame
            Feature time series.

        """
        if self.observation_horizon > 0:
            if self.observation_horizon > len(X):
                raise ValueError("Not enough input data to set the transformer memory.")

            self._X_observed = X[-self.observation_horizon :]
            self.observed_time_ = X["time"][-1]
        else:
            self._X_observed = X[:0]
            # For stateless transformers, only update observed_time_ if X is non-empty
            if len(X) > 0:
                self.observed_time_ = X["time"][-1]

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X: pl.DataFrame, y: pl.DataFrame | None = None, **params) -> "BaseTransformer":
        """Fits the transformer and returns it.

        Parameters
        ----------
        X : pl.DataFrame
            Input time series.

        y : pl.DataFrame or None, default=None
            Target time series. Ignored and only present for API consistency.

        **params : dict
            Metadata to route to nested estimators.

        Returns
        -------
        self

        """
        # Router transformers would call process_routing() in their fit function

        self._update_X_observed(X)

        self.feature_names_in_ = X.select(~cs.by_name("time")).columns
        self.n_features_in_ = len(self.feature_names_in_)

        # Store input schema for dtype preservation
        self.input_schema_ = dict(X.select(~cs.by_name("time")).schema)

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
        check_is_fitted(self, ["n_features_in_"])

        self._update_X_observed(X)

        return self

    def update(self, X: pl.DataFrame) -> "BaseTransformer":
        """Updates the transformer and returns it.

        This method extends the internal memory buffer with new observations,
        then calls reset() to maintain the fixed observation horizon window.

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
    def transform(self, X: pl.DataFrame, **params) -> pl.DataFrame:
        """Transforms the input time series.

        Parameters
        ----------
        X : pl.DataFrame
            Feature time series.

        **params : dict
            Metadata to route to nested estimators.

        Returns
        -------
        pl.DataFrame
            Transformed time series.

        """
        raise NotImplementedError()

    def inverse_transform(
        self, X_t: pl.DataFrame, X_p: pl.DataFrame | None, **params
    ) -> pl.DataFrame:
        """Inverts the input transformed time series.

        Parameters
        ----------
        X_t : pl.DataFrame
            Transformed time series.

        X_p : pl.DataFrame or None
            Untransformed time series corresponding to at least `observation_horizon` immediately
            previous time stamps. Can be None if `observation_horizon == 0`.

        **params : dict
            Metadata to route to nested estimators.

        Returns
        -------
        pl.DataFrame
            Inverted transformed time series.
        """
        raise NotImplementedError("This transformer is not invertible.")

    def update_transform(self, X: pl.DataFrame, **params) -> pl.DataFrame:
        """Transforms the input, updates the transformer and returns
        the transformed input.

        Parameters
        ----------
        X : pl.DataFrame
            Input time series.

        **params : dict
            Metadata to route to `transform()`.

        Returns
        -------
        pl.DataFrame
            Transformed input time series.

        """
        check_is_fitted(self, ["n_features_in_"])

        # Route all params to transform only (update is memory management)
        if self.observation_horizon > 0:
            X_full = pl.concat([self._X_observed, X])
            X_t = self.transform(X_full, **params)
            X_t = X_t[-len(X) :]
        else:
            X_t = self.transform(X, **params)

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
        Transformer used to transform the `input_features` time series into features.
    input_features: "X" | "y_t|X" | "y|X", default="y_t|X"
        Defines how the feature or the input to the ``feature_transformer``
         if passed is built.
    """

    _parameter_constraints: dict = {
        "target_transformer": [BaseTransformer, None],
        "feature_transformer": [BaseTransformer, None],
        "input_features": [str],
    }

    def __init__(
        self,
        feature_transformer: BaseTransformer | None = None,
        target_transformer: BaseTransformer | None = None,
        input_features: Literal["X", "y_t|X", "y|X"] = "y_t|X",
    ):
        self.feature_transformer = feature_transformer
        self.target_transformer = target_transformer
        self.input_features = input_features

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
        check_is_fitted(self, ["target_transformer_"])

        target_observation_horizon = 0
        if self.target_transformer is not None:
            if isinstance(self.target_transformer_, dict):
                # In panel data, all local transformers share the same horizon
                first_transformer = next(iter(self.target_transformer_.values()))
                target_observation_horizon = first_transformer.observation_horizon
            else:
                target_observation_horizon = self.target_transformer_.observation_horizon

        # TODO: Handle feature transformer observation horizon?
        feature_observation_horizon = 0
        # if "y" in self.input_features:
        #     feature_observation_horizon = 1

        self_observation_horizon = getattr(self, "_observation_horizon", 0)
        return max(
            self_observation_horizon, target_observation_horizon, feature_observation_horizon
        )

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

    def _set_input_attributes(self, y: pl.DataFrame, X: pl.DataFrame | None) -> None:
        """Detect and validate panel data structure across target and features.

        Inspects whether the data contains global (single time series) or local
        (panel columns with multiple time series) and ensures consistency
        across y and X. Sets instance attributes for downstream use.

            Sets the following attributes:
            - `panel_group_names_` : list of str or None
                Group prefixes for panel data (e.g., ["sales", "inventory"])
            - `local_y_schema_` : dict of str to pl.DataType
                Schema (column names → dtypes) for target columns
            - `local_X_schema_` : dict of str to pl.DataType
                Schema (column names → dtypes) for feature columns
            - `global_X_schema_` : dict of str to pl.DataType or None
                Schema (column names → dtypes) for global feature columns found in X
                alongside local groups.

        Parameters
        ----------
        y : pl.DataFrame
            Target time series.
        X : pl.DataFrame or None
            Feature time series.

        Raises
        ------
        ValueError
            If y contains both global and local columns (ambiguous structure),
            or if group column suffixes don't match across groups,
            or if X local groups don't match y's structure.

        Notes
        -----
        Panel data example:
            y has columns ["sales__store_1", "sales__store_2"] (both Int64)
            → panel_group_names_ = ["sales"]
            → local_y_schema_ = {"store_1": pl.Int64, "store_2": pl.Int64}
            (Note: schema has unprefixed column names)

        Global data example:
            y has regular column "sales" (Int64)
            → panel_group_names_ = None
            → local_y_schema_ = {"sales": pl.Int64}

        See Also
        --------
        :func:`yohou.utils.panel.inspect_locality` : Detects group columns

        """
        y_global_names, y_panel_groups = inspect_locality(y)
        if X is not None:
            X_global_names, X_panel_groups = inspect_locality(X)

            if len(X_panel_groups):
                if list(X_panel_groups.keys()) != list(y_panel_groups.keys()):
                    raise ValueError("`X` and `y` do not have the same local group names.")

        self.panel_group_names_ = list(y_panel_groups.keys()) or None

        # Non-panel data
        if self.panel_group_names_ is None:
            self.local_y_schema_ = dict(y.select(~cs.by_name("time")).schema)
            self.global_X_schema_ = None

            self.local_X_schema_ = None
            if X is not None:
                self.local_X_schema_ = dict(X.select(~cs.by_name("time")).schema)

        # Panel data
        else:
            # Extract suffixes from first group to validate consistency
            first_group_cols = y_panel_groups[self.panel_group_names_[0]]
            first_group_suffixes = [col.split("__", 1)[1] for col in first_group_cols]

            if len(y_global_names):
                raise ValueError("`y` contains both local and global columns.")

            # Validate all groups have the same suffixes
            for group_name in self.panel_group_names_[1:]:
                group_cols = y_panel_groups[group_name]
                group_suffixes = [col.split("__", 1)[1] for col in group_cols]
                if sorted(group_suffixes) != sorted(first_group_suffixes):
                    raise ValueError(
                        f"The local groups in `y` do not have the same column suffixes. "
                        f"Group '{self.panel_group_names_[0]}': {sorted(first_group_suffixes)}, "
                        f"Group '{group_name}': {sorted(group_suffixes)}"
                    )

            # Extract y schema
            local_y = y.select(first_group_cols).rename(
                {col: col.split("__", 1)[1] for col in first_group_cols}
            )
            self.local_y_schema_ = dict(local_y.schema)

            self.local_X_schema_ = None
            self.global_X_schema_ = None
            if X is not None:
                # Validate X groups have same suffixes
                first_X_group_cols = X_panel_groups[self.panel_group_names_[0]]
                first_X_suffixes = [col.split("__", 1)[1] for col in first_X_group_cols]

                for group_name in self.panel_group_names_[1:]:
                    group_cols = X_panel_groups[group_name]
                    group_suffixes = [col.split("__", 1)[1] for col in group_cols]
                    if sorted(group_suffixes) != sorted(first_X_suffixes):
                        raise ValueError(
                            f"The local groups in `X` do not have the same column suffixes. "
                            f"Group '{self.panel_group_names_[0]}': {sorted(first_X_suffixes)}, "
                            f"Group '{group_name}': {sorted(group_suffixes)}"
                        )

                # Extract X schema
                self.global_X_schema_ = dict(X.select(X_global_names).schema)
                local_X = X.select(first_X_group_cols).rename(
                    {col: col.split("__", 1)[1] for col in first_X_group_cols}
                )
                self.local_X_schema_ = dict(local_X.schema)

    def _set_transformed_attributes(
        self,
        y_t: pl.DataFrame | dict[str, pl.DataFrame],
        X_t: pl.DataFrame | dict[str, pl.DataFrame] | None,
    ) -> None:
        """Set attributes for transformed data schemas.

        This method stores the schemas of transformed data (y_t and X_t) after
        target_transformer and feature_transformer have been applied. These schemas
        are used by reduction forecasters for tabularization.

        Parameters
        ----------
        y_t : pl.DataFrame or dict[str, pl.DataFrame]
            Transformed target time series. For panel data, this is a dict mapping
            group names to DataFrames.
        X_t : pl.DataFrame, dict[str, pl.DataFrame], or None
            Transformed feature time series. For panel data, this is a dict mapping
            group names to DataFrames.

        Notes
        -----
        Sets the following attributes:
        - `local_y_t_schema_` : dict of str to pl.DataType
            Schema for transformed target columns (unprefixed names)
        - `local_X_t_schema_` : dict of str to pl.DataType or None
            Schema for transformed feature columns (unprefixed names, None if X_t is None)

        For panel data, takes the schema from the first group since all groups
        have the same structure after transformers are applied.

        See Also
        --------
        _set_input_attributes : Sets schemas for input (untransformed) data
        """
        # Non-panel data
        if self.panel_group_names_ is None:
            # Global data (single DataFrame)
            self.local_y_t_schema_ = dict(y_t.select(~cs.by_name("time")).schema)

            # Store transformed feature schema (if X_t exists)
            if X_t is not None:
                self.local_X_t_schema_ = dict(X_t.select(~cs.by_name("time")).schema)
            else:
                self.local_X_t_schema_ = None

        # Panel data
        else:
            # Get schema from first group (all groups have same structure)
            first_group_name = next(iter(y_t))
            y_t_df = y_t[first_group_name]
            self.local_y_t_schema_ = dict(y_t_df.select(~cs.by_name("time")).schema)

            self.local_X_t_schema_ = None
            if X_t is not None:
                if isinstance(X_t, dict):
                    X_t_first_group = X_t[first_group_name]
                    if X_t_first_group is not None:
                        self.local_X_t_schema_ = dict(
                            X_t_first_group.select(~cs.by_name("time")).schema
                        )

    def _update_y_X_t_observed(
        self, y: pl.DataFrame, X_t: pl.DataFrame | None, panel_group_names: list[str]
    ) -> None:
        """Update stored observed data for inverse transforms.

        Parameters
        ----------
        y : pl.DataFrame
            Target time series.
        X_t : pl.DataFrame or None
            Transformed feature time series.
        panel_group_names : list of str
            Group prefixes for panel data. Updateonly for the specified panel groups
            Parameter is ignored if the forecaster was not fitted on panel data.

        """

        if self.panel_group_names_ is None:
            # Non-panel data
            self.observed_time_ = y["time"][-1]

            self._X_t_observed = None
            if X_t is not None:
                self._X_t_observed = X_t[[-1]]

            # Store untransformed data for inverse_transform
            y_observed = None
            if self.observation_horizon > 0:
                if self.observation_horizon > len(y):
                    raise ValueError("Not enough data to set observed y.")

                y_observed = y[-self.observation_horizon :]

            # TODO: Alignment
            if X_t is not None and y_observed is not None:
                y_observed = y_observed.filter(pl.col("time").is_in(X_t["time"].to_list()))

            self._y_observed = y_observed

        else:
            # Panel data
            self.observed_time_ = {}

            X_t_observed = None
            if X_t is not None:
                X_t_observed = {}

            y_observed = {}
            for panel_group_name in panel_group_names:
                # Extract group columns for y and store last observation_horizon rows
                y_group = y.get(panel_group_name)

                # TODO: Move into a check funtion?
                if self.observation_horizon > len(y_group):
                    raise ValueError(
                        f"Not enough data to set observed y for group {panel_group_name}."
                    )

                self.observed_time_[panel_group_name] = y_group["time"][-1]
                y_observed[panel_group_name] = y_group[-self.observation_horizon :]

                # Store X_t_observed for this group
                X_t_observed[panel_group_name] = None
                if X_t is not None:
                    # X_t is a dict: extract group columns for X_t
                    X_t_group = X_t.get(panel_group_name)
                    if X_t_group is not None:
                        X_t_observed[panel_group_name] = X_t_group[[-1]]
                    else:
                        X_t_observed[panel_group_name] = None
                else:
                    # X_t is a single DataFrame (not dict)
                    X_t_observed[panel_group_name] = X_t[[-1]]

            self._y_observed = y_observed
            self._X_t_observed = X_t_observed

    def _build_feature_input(
        self,
        y: pl.DataFrame,
        y_t: pl.DataFrame,
        X: pl.DataFrame | None,
    ) -> pl.DataFrame | None:
        """Build feature input based on input_features parameter.

        Constructs the input to the feature_transformer by combining original y,
        transformed y_t, and exogenous features X according to the input_features
        configuration.

        Parameters
        ----------
        y : pl.DataFrame
            Original target time series (untransformed).
        y_t : pl.DataFrame
            Transformed target time series.
        X : pl.DataFrame or None
            Exogenous feature time series.

        Returns
        -------
        pl.DataFrame or None
            Feature input for feature_transformer.

        Notes
        -----
        The input_features parameter controls what features are available:
        - "y_t|X": Transformed target + exogenous features (default)
        - "y|X": Original target + exogenous features
        - "X": Only exogenous features (no target)

        For "y|X", the original y is aligned with y_t by taking rows from
        target_observation_horizon onwards to match the transformed data.

        """
        if self.input_features == "y_t|X":
            # Default: use transformed target
            X_feat_in = y_t
            if X is not None:
                X_feat_in = pl.concat(
                    [y_t, X.select(~cs.by_name("time"))],
                    how="horizontal",
                )
        elif self.input_features == "y|X":
            # Use original target (aligned with transformed data)
            X_feat_in = y
            if X is not None:
                X_feat_in = pl.concat(
                    [y, X.select(~cs.by_name("time"))],
                    how="horizontal",
                )
        elif self.input_features == "X":
            # Only exogenous features
            if X is None:
                if self.feature_transformer is not None:
                    # TODO: Raise error in fit
                    raise ValueError("input_features='X' requires X to be provided, but X is None.")
                else:
                    X_feat_in = None

            X_feat_in = X
        else:
            raise ValueError(
                f"Invalid input_features='{self.input_features}'. "
                "Must be one of: 'y_t|X', 'y|X', 'X'."
            )

        return X_feat_in

    def _fit_transform_transformers(
        self, y: pl.DataFrame, X: pl.DataFrame | None
    ) -> tuple[pl.DataFrame, pl.DataFrame | None, BaseTransformer | None, BaseTransformer | None]:
        """Fit and apply target and feature transformers to a single time series.

        Orchestrates the transformation pipeline: target transformer first (if any),
        then feature transformer (if any). Handles observation horizon alignment to
        ensure transformed data matches temporally.

        Parameters
        ----------
        y : pl.DataFrame
            Target time series with "time" column.
        X : pl.DataFrame or None
            Feature time series with "time" column.

        Returns
        -------
        y_t : pl.DataFrame or None
            Transformed target time series.
        X_t : pl.DataFrame or None
            Transformed feature matrix (includes transformed y if no separate X provided).

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
        target_transformer = None
        if self.target_transformer is not None:
            target_transformer = clone(self.target_transformer)
            y_t = target_transformer.fit_transform(y)

        X_feat_in = self._build_feature_input(y, y_t, X)

        X_t = X_feat_in
        feature_transformer = None
        if self.feature_transformer is not None and X_feat_in is not None:
            feature_transformer = clone(self.feature_transformer)
            X_t = feature_transformer.fit_transform(X_feat_in)
            feature_observation_horizon = feature_transformer.observation_horizon
            # TODO: Alignment
            y_t = y_t[feature_observation_horizon:]

        return y_t, X_t, target_transformer, feature_transformer

    def _update_transformers(
        self,
        y: pl.DataFrame,
        X: pl.DataFrame | None,
        target_transformer: BaseTransformer | None,
        feature_transformer: BaseTransformer | None,
    ) -> pl.DataFrame | None:
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
        pl.DataFrame or None
            Transformed new observations.

        """
        y_t = y
        if target_transformer is not None:
            y_t = target_transformer.update_transform(y)

        X_feat_in = self._build_feature_input(y, y_t, X)

        X_t = X_feat_in
        if feature_transformer is not None and X_feat_in is not None:
            X_t = feature_transformer.update_transform(X_feat_in)

        return X_t

    def _reset_transformers(
        self,
        y: pl.DataFrame,
        X: pl.DataFrame | None,
        target_transformer: BaseTransformer | None,
        feature_transformer: BaseTransformer | None,
    ) -> pl.DataFrame | None:
        """Reset transformers.

        Parameters
        ----------
        y : pl.DataFrame
            New target observations.
        X : pl.DataFrame or None
            New features.
        target_transformer : BaseTransformer or None
            Target transformer to reset.
        feature_transformer : BaseTransformer or None
            Feature transformer to reset.

        Returns
        -------
        pl.DataFrame or None
            Transformed new observations.

        """
        y_t = y
        if target_transformer is not None:
            target_transformer.reset(X=y[: -self.observation_horizon])
            y_t = target_transformer.update_transform(y[-self.observation_horizon :])

        X_feat_in = self._build_feature_input(y, y_t, X)

        X_t = X_feat_in
        if feature_transformer is not None and X_feat_in is not None:
            feature_observation_horizon = feature_transformer.observation_horizon
            feature_transformer.reset(X=X_feat_in[-feature_observation_horizon - 1 : -1])
            X_t = feature_transformer.update_transform(X_feat_in[[-1]])

        return X_t

    def _pre_fit(
        self,
        y: pl.DataFrame,
        X: pl.DataFrame | None = None,
        forecasting_horizon: StrictInt = 1,
    ) -> tuple[
        pl.DataFrame | dict[str, pl.DataFrame] | None, pl.DataFrame | dict[str, pl.DataFrame] | None
    ]:
        """Preprocess and transform inputs before fitting.

        Parameters
        ----------
        y : pl.DataFrame
            Target time series.
        X : pl.DataFrame or None, default=None
            Features time series.
        forecasting_horizon : int, default=1
            Number of steps ahead to forecast.

        Returns
        -------
        y_t : pl.DataFrame or None
            Transformed target.
        X_t : pl.DataFrame or None
            Transformed features.

        """
        self.interval_ = check_inputs(y, X)
        self._set_input_attributes(y, X)

        if forecasting_horizon < 1:
            raise ValueError(
                f"`forecasting_horizon` should be a positive int. It is: {forecasting_horizon}"
            )

        self.fit_forecasting_horizon_ = forecasting_horizon

        y_t, X_t = self._fit_transform_inputs(y, X)

        self._set_transformed_attributes(y_t, X_t)

        if self.panel_group_names_ is not None:
            y = {
                group: y.select(
                    ["time"] + [f"{group}__{col}" for col in self.local_y_schema_.keys()]
                ).rename({f"{group}__{col}": col for col in self.local_y_schema_.keys()})
                for group in self.panel_group_names_
            }

        self._update_y_X_t_observed(y, X_t, self.panel_group_names_ or [])

        return y_t, X_t

    def _fit_transform_inputs(
        self, y: pl.DataFrame, X: pl.DataFrame | None
    ) -> tuple[
        pl.DataFrame | dict[str, pl.DataFrame] | None, pl.DataFrame | dict[str, pl.DataFrame] | None
    ]:
        """Fit the transformers and transform inputs.

        Parameters
        ----------
        y : pl.DataFrame
            Target time series.
        X : pl.DataFrame or None
            Feature time series.

        Returns
        -------
        y_t : pl.DataFrame or dict[str, pl.DataFrame] or None
            Transformed target.
        X_t : pl.DataFrame or dict[str, pl.DataFrame] or None
            Transformed features.

        """
        # Non-panel data
        if self.panel_group_names_ is None:
            # Global data: schemas contain actual column names
            y = y.select(["time"] + list(self.local_y_schema_.keys()))

            if X is not None:
                X = X.select(["time"] + list(self.local_X_schema_.keys()))

            y_t, X_t, target_transformer, feature_transformer = self._fit_transform_transformers(
                y, X
            )

        # Panel data
        else:
            y_t, X_t = {}, {}
            target_transformer, feature_transformer = {}, {}

            for group_name in self.panel_group_names_:
                # Extract group data using get_group_df
                y_local = get_group_df(df=y, group_name=group_name, schema=self.local_y_schema_)

                X_local = None
                if X is not None:
                    # Build schema for X (local + global columns)
                    X_schema = dict(self.local_X_schema_)  # Start with local columns
                    if self.global_X_schema_:
                        X_schema.update(self.global_X_schema_)  # Add global columns
                    X_local = get_group_df(df=X, group_name=group_name, schema=X_schema)

                (
                    y_t[group_name],
                    X_t[group_name],
                    target_transformer_local,
                    feature_transformer_local,
                ) = self._fit_transform_transformers(y_local, X_local)

                target_transformer[group_name] = target_transformer_local
                feature_transformer[group_name] = feature_transformer_local

        self.target_transformer_ = target_transformer
        self.feature_transformer_ = feature_transformer

        return y_t, X_t

    @abc.abstractmethod
    @_fit_context(prefer_skip_nested_validation=True)
    def fit(
        self,
        y: pl.DataFrame,
        X: pl.DataFrame | None = None,
        forecasting_horizon: StrictInt = 1,
        **params,
    ) -> "BaseForecaster":
        """Fits the forecaster and returns it.

        Parameters
        ----------
        y : pl.DataFrame
            Target time series.
        X : pl.DataFrame or None, default=None
            Feature time series.
        forecasting_horizon : int >= 1, default=1
            Horizon to forecast.
        **params : dict
            Metadata to route to nested estimators.

        Returns
        -------
        self

        """
        raise NotImplementedError()

    def reset(
        self,
        y: pl.DataFrame,
        X: pl.DataFrame | None = None,
        panel_group_names: list[str] | None = None,
    ) -> "BaseForecaster":
        """Resets the forecaster by resetting the observation horizon.

        Parameters
        ----------
        y : pl.DataFrame
            Target time series.
        X : pl.DataFrame or None
            Feature time series.
        panel_group_names : list of str or None, default=None
            Group prefixes for panel data:
            - If None: predict for all groups
            - If list of str: predict only for the specified panel groups
            Parameter is ignored if the forecaster was not fitted on panel data.

        Returns
        -------
        self

        """
        check_is_fitted(self, "fit_forecasting_horizon_")

        if panel_group_names is None:
            panel_group_names = self.panel_group_names_
        else:
            # Validate specified panel groups
            if self.panel_group_names_ is None:
                raise ValueError(
                    "The forecaster was fitted on global data, but `panel_group_names` "
                    "were provided for reset."
                )
            for panel_group in panel_group_names:
                if panel_group not in self.panel_group_names_:
                    raise ValueError(f"Panel group '{panel_group}' not found in fitted forecaster.")

        # TODO: Turn this into a check_inputs call
        if self.observation_horizon > 0:
            # Select columns based on schema
            if self.panel_group_names_ is None:
                # Global data: schemas contain actual column names
                y = y.select(["time"] + list(self.local_y_schema_.keys()))
                if X is not None:
                    X = X.select(["time"] + list(self.local_X_schema_.keys()))
            else:
                # Panel data: schemas contain unprefixed names, need to reconstruct prefixed columns
                y_cols = ["time"]
                for group_name in self.panel_group_names_:
                    y_cols.extend([f"{group_name}__{col}" for col in self.local_y_schema_.keys()])
                y = y.select(y_cols)

                if X is not None:
                    X_cols = ["time"]
                    for group_name in self.panel_group_names_:
                        X_cols.extend(
                            [f"{group_name}__{col}" for col in self.local_X_schema_.keys()]
                        )
                    X_cols.extend(list(self.global_X_schema_.keys()))  # Add global columns
                    X = X.select(X_cols)

        else:  # TODO: This should only be useful for trend/seasonality forecasters - Use tags?
            # If there is no observation horizon, only check for time column presence
            if "time" not in y.columns:
                raise ValueError("y must contain 'time' column.")
            if X is not None and "time" not in X.columns:
                raise ValueError("X must contain 'time' column.")

        # Non-panel data
        if self.panel_group_names_ is None:
            # Global data: use _reset_transformers
            X_t = self._reset_transformers(
                y, X, self.target_transformer_, self.feature_transformer_
            )

        # Panel data
        else:
            X_t = {}

            for panel_group_name in panel_group_names:
                # Extract group data using get_group_df
                y_local = get_group_df(
                    df=y, group_name=panel_group_name, schema=self.local_y_schema_
                )

                X_local = None
                if X is not None:
                    # Build schema for X (local + global columns)
                    X_schema = dict(self.local_X_schema_)  # Start with local columns
                    if self.global_X_schema_:
                        X_schema.update(self.global_X_schema_)  # Add global columns
                    X_local = get_group_df(df=X, group_name=panel_group_name, schema=X_schema)

                local_target_transformer = None
                if self.target_transformer is not None and isinstance(
                    self.target_transformer_, dict
                ):
                    local_target_transformer = self.target_transformer_[panel_group_name]

                local_feature_transformer = None
                if self.feature_transformer is not None and isinstance(
                    self.feature_transformer_, dict
                ):
                    local_feature_transformer = self.feature_transformer_[panel_group_name]

                X_t_local = self._reset_transformers(
                    y_local,
                    X_local,
                    local_target_transformer,
                    local_feature_transformer,
                )

                # Store transformed X with unprefixed columns for this group
                X_t[panel_group_name] = X_t_local

            y = {
                group: y.select(
                    ["time"] + [f"{group}__{col}" for col in self.local_y_schema_.keys()]
                ).rename({f"{group}__{col}": col for col in self.local_y_schema_.keys()})
                for group in self.panel_group_names_
            }

        self._update_y_X_t_observed(y, X_t, panel_group_names)

        return self

    def update(
        self,
        y: pl.DataFrame,
        X: pl.DataFrame | None = None,
        panel_group_names: list[str] | None = None,
    ) -> "BaseForecaster":
        """Updates the forecaster with more recent data and returns it.

        Parameters
        ----------
        y : pl.DataFrame
            Target time series.
        X : pl.DataFrame or None
            Feature time series.
        panel_group_names : list of str or None, default=None
            Group prefixes for panel data:
            - If None: predict for all groups (default behavior)
            - If list of str: predict only for the specified panel groups
            Parameter is ignored if the forecaster was not fitted on panel data.

        Returns
        -------
        self

        """
        check_is_fitted(self, "fit_forecasting_horizon_")

        if panel_group_names is None:
            panel_group_names = self.panel_group_names_
        else:
            # Validate specified panel groups
            if self.panel_group_names_ is None:
                raise ValueError(
                    "The forecaster was fitted on global data, but `panel_group_names` "
                    "were provided for update."
                )
            for panel_group in panel_group_names:
                if panel_group not in self.panel_group_names_:
                    raise ValueError(f"Panel group '{panel_group}' not found in fitted forecaster.")

        # Validate schema and enforce column order
        if self.panel_group_names_ is None:
            # Non-panel data
            y = check_schema(y, self.local_y_schema_)

            if self._y_observed is not None:
                y = pl.concat([self._y_observed, y], how="vertical")

            if X is not None:
                X = check_schema(X, self.local_X_schema_)

        else:
            # Panel data
            y = check_schema(y, self.local_y_schema_, panel_group_names=panel_group_names)

            # Validate and prepare X if needed
            if X is not None:
                # Validate local X columns (with panel prefixes)
                if self.local_X_schema_:
                    X_local = check_schema(
                        X, self.local_X_schema_, panel_group_names=self.panel_group_names_
                    )

                # Validate global X columns (no prefixes)
                if self.global_X_schema_:
                    X_global = check_schema(X, self.global_X_schema_)

                # Reconstruct X_selected with both local and global columns
                if self.local_X_schema_ and self.global_X_schema_:
                    X = pl.concat([X_local, X_global.select(~cs.by_name("time"))], how="horizontal")
                elif self.local_X_schema_:
                    X = X_local
                elif self.global_X_schema_:
                    X = X_global

        # Non-panel data
        if self.panel_group_names_ is None:
            # Global data: use BaseForecaster._update_transformers
            y_updated = y
            X_t_updated = self._update_transformers(
                y, X, self.target_transformer_, self.feature_transformer_
            )

        # Panel data
        else:
            y_updated, X_t_updated = {}, {}
            for panel_group_name in panel_group_names:
                # Extract group data for new observations only
                y_local = get_group_df(
                    df=y, group_name=panel_group_name, schema=self.local_y_schema_
                )

                X_local = None
                if X is not None:
                    # Build schema for X (local + global columns)
                    X_schema = dict(self.local_X_schema_)  # Start with local columns
                    if self.global_X_schema_:
                        X_schema.update(self.global_X_schema_)  # Add global columns
                    X_local = get_group_df(df=X, group_name=panel_group_name, schema=X_schema)

                local_target_transformer = None
                if self.target_transformer is not None and isinstance(
                    self.target_transformer_, dict
                ):
                    local_target_transformer = self.target_transformer_[panel_group_name]

                local_feature_transformer = None
                if self.feature_transformer is not None and isinstance(
                    self.feature_transformer_, dict
                ):
                    local_feature_transformer = self.feature_transformer_[panel_group_name]

                # Update transformers with new data only
                X_t_updated[panel_group_name] = self._update_transformers(
                    y_local,
                    X_local,
                    local_target_transformer,
                    local_feature_transformer,
                )

                # For y_updated, concatenate stored observations with new observations
                # This ensures we have enough data to satisfy observation_horizon
                if self._y_observed is not None and panel_group_name in self._y_observed:
                    y_full = pl.concat(
                        [self._y_observed[panel_group_name], y_local], how="vertical"
                    )
                else:
                    y_full = y_local
                y_updated[panel_group_name] = y_full

        self._update_y_X_t_observed(y_updated, X_t_updated, panel_group_names)

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
        # Panel data case: observed_time_ is a dict
        if self.panel_group_names_ is not None:
            # For panel data, we need to create time columns for each group
            # All groups share the same time progression
            # Use the first group's observed_time as reference
            first_group_name = list(self.observed_time_.keys())[0]
            observed_time_value = self.observed_time_[first_group_name]

            predicted_times = [
                add_interval(observed_time_value, self.interval_, n=n)
                for n in range(1, len(y_pred) + 1)
            ]

            time = pl.DataFrame(
                {"observed_time": [observed_time_value] * len(y_pred), "time": predicted_times}
            )
        # Global data case: observed_time_ is a datetime
        else:
            # Use add_interval to handle both fixed and variable intervals
            predicted_times = [
                add_interval(self.observed_time_, self.interval_, n=n)
                for n in range(1, len(y_pred) + 1)
            ]

            time = pl.DataFrame(
                {"observed_time": [self.observed_time_] * len(y_pred), "time": predicted_times}
            )

        y_pred = pl.concat([time, y_pred], how="horizontal")

        return y_pred

    # TODO: Route parameters?
    def _predict_one(
        self,
        panel_group_names: list[str],
        **params,
    ) -> pl.DataFrame:
        """Predicts `_fit_forecasting_horizon` steps from the observation horizon.

        Parameters
        ----------
        panel_group_names : list of str
            Panel group names to predict for.
        **params : dict
            Additional parameters for prediction.

        Returns
        -------
        pl.DataFrame
            Predicted time series.

        """
        raise NotImplementedError(
            f"The forecaster of type {type(self)} does not implement_predict_one."
        )

    @staticmethod
    def _predict(
        forecaster: "BaseForecaster",
        panel_group_names: list[str],
        **predict_one_params,
    ) -> tuple[pl.DataFrame, pl.DataFrame]:
        """Generate one-step or multi-step prediction.

        Parameters
        ----------
        forecaster : BaseForecaster
            Fitted forecaster to use for prediction.
        panel_group_names : list of str or None, default=None
            Group prefixes for panel data:
            - If None: predict for all groups
            - If list of str: predict only for the specified panel groups
            Parameter is ignored if the forecaster was not fitted on panel data.
        **predict_one_params : dict
            Params to the _predict_one method.

        Returns
        -------
        y_pred_step : pl.DataFrame
            Predicted time series in transformed space.
        y_pred_step_inv : pl.DataFrame
            Inverse transformed predicted time series (original scale).

        """
        # TODO: Should we remove the panel group logic from _predict_one?
        y_pred_step = forecaster._predict_one(
            panel_group_names=panel_group_names, **predict_one_params
        )

        if forecaster.target_transformer is None:
            if panel_group_names is None:
                # Non-panel data

                y_pred_step = cast(y_pred_step, forecaster.local_y_schema_)

            else:
                # Panel data
                y_pred_step = cast(
                    y_pred_step,
                    {
                        f"{panel_group_name}__{col}": dtype
                        for panel_group_name in panel_group_names
                        for col, dtype in forecaster.local_y_schema_.items()
                    },
                )

            y_pred_step_inv = y_pred_step

        else:
            if panel_group_names is None:
                # Non-panel data

                # Remove "observed_time" before inverse_transform (transformers don't handle it)
                observed_time = y_pred_step.select(cs.by_name("observed_time"))
                y_pred_step_no_obs = y_pred_step.select(~cs.by_name("observed_time"))

                y_pred_step_inv = forecaster.target_transformer_.inverse_transform(
                    X_t=y_pred_step_no_obs,
                    X_p=forecaster._y_observed,
                )

                # Cast to restore original dtypes
                y_pred_step_inv_cast = cast(
                    y_pred_step_inv.select(~cs.by_name("time")), forecaster.local_y_schema_
                )

                # Reconstruct with time column
                y_pred_step_inv = pl.concat(
                    [y_pred_step_inv.select(cs.by_name("time")), y_pred_step_inv_cast],
                    how="horizontal",
                )

                # Add "observed_time" back
                y_pred_step_inv = pl.concat([observed_time, y_pred_step_inv], how="horizontal")

            else:
                # Panel data
                y_pred_step_inv_dict = {}

                for panel_group_name in panel_group_names:
                    transformer = forecaster.target_transformer_[panel_group_name]

                    # Extract the group's data
                    group_cols = [
                        c for c in y_pred_step.columns if c.startswith(f"{panel_group_name}__")
                    ]
                    y_pred_step_group = y_pred_step.select(
                        cs.by_name("observed_time") | cs.by_name("time") | cs.by_name(group_cols)
                    )

                    # Remove "observed_time" before inverse_transform as transformers don't handle it
                    observed_time = y_pred_step_group.select(cs.by_name("observed_time"))
                    y_pred_step_group_no_obs = y_pred_step_group.select(
                        ~cs.by_name("observed_time")
                    )

                    # Inverse transform
                    y_observed_local = forecaster._y_observed[panel_group_name]
                    y_pred_step_group_inv = transformer.inverse_transform(
                        X_t=y_pred_step_group_no_obs,
                        X_p=y_observed_local,
                    )

                    # Cast to restore original dtypes
                    # For panel data, need to create prefixed schema for casting
                    local_y_schema = {
                        f"{panel_group_name}__{col}": dtype
                        for col, dtype in forecaster.local_y_schema_.items()
                    }
                    y_pred_step_group_inv_cast = cast(
                        y_pred_step_group_inv.select(~cs.by_name("time")), local_y_schema
                    )

                    # Reconstruct with time column
                    y_pred_step_group_inv = pl.concat(
                        [
                            y_pred_step_group_inv.select(cs.by_name("time")),
                            y_pred_step_group_inv_cast,
                        ],
                        how="horizontal",
                    )

                    # Add "observed_time" back
                    y_pred_step_group_inv = pl.concat(
                        [observed_time, y_pred_step_group_inv], how="horizontal"
                    )

                    # Store in dict (without time columns)
                    y_pred_step_inv_dict[panel_group_name] = y_pred_step_group_inv.select(
                        ~cs.by_name("observed_time") & ~cs.by_name("time")
                    )

                times = y_pred_step.select(cs.by_name("observed_time") | cs.by_name("time"))
                y_pred_inv_cols = pl.concat(list(y_pred_step_inv_dict.values()), how="horizontal")

                y_pred_step_inv = pl.concat([times, y_pred_inv_cols], how="horizontal")

        return y_pred_step, y_pred_step_inv

    def get_metadata_routing(self) -> MetadataRouter:
        """Get metadata routing for this forecaster.

        BaseForecaster is both a consumer AND a router:
        - Consumer: Can accept metadata like forecasting_horizon
        - Router: Forwards metadata to target_transformer and feature_transformer

        Subclasses with additional nested estimators should call super() and
        add their own child routing.

        Returns
        -------
        router : MetadataRouter
            Router that forwards metadata to transformers.
        """
        router = MetadataRouter(owner=self)
        router.add_self_request(self)

        # Route to target_transformer if present
        # This allows target_transformer to receive metadata if it requests it
        if hasattr(self, "target_transformer") and self.target_transformer is not None:
            router.add(
                target_transformer=self.target_transformer,
                method_mapping=MethodMapping()
                .add(caller="fit", callee="fit")
                .add(caller="fit", callee="transform"),
            )

        # Route to feature_transformer if present
        if hasattr(self, "feature_transformer") and self.feature_transformer is not None:
            router.add(
                feature_transformer=self.feature_transformer,
                method_mapping=MethodMapping()
                .add(caller="fit", callee="fit")
                .add(caller="fit", callee="transform"),
            )

        return router


class BaseReductionForecaster(BaseForecaster, metaclass=abc.ABCMeta):
    """Base class for forecasters using reduction to supervised learning.

    Converts the time series forecasting task to a tabular one.

    Parameters
    ----------
    estimator : instance of `BaseEstimator`, default=LinearRegression()
        Estimator used to fit the tabularized data.
    reduction_strategy : {"direct", "multi-output"}, default="multi-output"
        Reduction strategy to use.
    input_features : {"y_t|X", "y|X", "X"}, default="y_t|X"
        Specifies which features to use for training the estimator.
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
        input_features: Literal["y_t|X", "y|X", "X"] = "y_t|X",
        target_transformer: BaseTransformer | None = None,
        feature_transformer: BaseTransformer | None = None,
    ):
        BaseForecaster.__init__(
            self,
            input_features=input_features,
            target_transformer=target_transformer,
            feature_transformer=feature_transformer,
        )

        self.estimator = estimator
        self.reduction_strategy = reduction_strategy

    def _get_tabularized_dataset(
        self,
        y_t: pl.DataFrame,
        X_t: pl.DataFrame,
        forecasting_horizon: int,
        y_columns: list[str] | None = None,
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
        # Use provided y_columns or fall back to all columns from local_y_t_schema_
        if y_columns is None:
            y_columns = list(self.local_y_t_schema_.keys())

        X_tab = X_t.select(~cs.by_name("time"))[:-forecasting_horizon]
        y_tab = tabularize(
            y_t.select(~cs.by_name("time")),
            lags=list(range(1 + forecasting_horizon)),
        ).rename(
            {
                f"{col}_lag_{lag}": f"{col}_step_{forecasting_horizon - lag}"
                for lag in range(1 + forecasting_horizon)
                for col in y_columns
            }
        )[[f"{col}_step_{step}" for step in range(1, 1 + forecasting_horizon) for col in y_columns]]

        return X_tab.to_numpy(), y_tab.to_numpy()

    def _estimator_fit_one(
        self,
        y_t: pl.DataFrame,
        X_t: pl.DataFrame,
        forecasting_horizon: StrictInt,
        time_weight: Callable | pl.DataFrame | None = None,
        estimator_params: dict[str, Any] | None = None,
        estimator_fit_params: dict[str, Any] | None = None,
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
        time_weight : callable or pl.DataFrame or None, default=None
            Time weighting function or DataFrame to weight samples.
            Converted to sample_weight during tabularization.
        estimator_params : dict
            Additional parameters to pass to the estimator's set_params method.
        estimator_fit_params : dict
            Additional parameters to pass to the estimator's fit method.

        Returns
        -------
        BaseEstimator
            Fitted sklearn regressor.

        Notes
        -----
        For panel data (panel_group_names_ is not None):
        - Unnests each panel column
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
        estimator = clone(self.estimator).set_params(**(estimator_params or {}))

        if self.panel_group_names_ is None:
            # Global time series
            X_tab, y_tab = self._get_tabularized_dataset(
                y_t,
                X_t,
                forecasting_horizon,
            )

        else:
            # Panel data: stack all series
            # y_t and X_t are dicts mapping group_name to DataFrames
            X_tab_list, y_tab_list = [], []
            for panel_group_name in self.panel_group_names_:
                # Get DataFrames for this group (already have unprefixed columns)
                y_t_local = y_t[panel_group_name]
                X_t_local = X_t[panel_group_name]

                # Get column names (excluding "time") for tabularization
                y_columns = [c for c in y_t_local.columns if c != "time"]

                # Pass the group's DataFrame to tabularize
                X_tab_local, y_tab_local = self._get_tabularized_dataset(
                    y_t_local,
                    X_t_local,
                    forecasting_horizon,
                    y_columns=y_columns,
                )

                X_tab_list.append(X_tab_local)
                y_tab_list.append(y_tab_local)

            X_tab = np.vstack(X_tab_list)
            y_tab = np.vstack(y_tab_list)

        estimator.fit(X_tab, y_tab, **(estimator_fit_params or {}))

        return estimator

    # TODO: Refactor so that complexity due to panel data be in _predict_one
    # instead
    def _estimator_predict_one(
        self,
        estimator: BaseEstimator,
        panel_group_names: list[str],
    ) -> pl.DataFrame:
        """Generate predictions using fitted estimator on tabularized data.

        Parameters
        ----------
        estimator : BaseEstimator
            Fitted scikit-learn estimator.
        panel_group_names : list of str
            Panel group names to predict for.

        Returns
        -------
        pl.DataFrame
            Predictions for the forecasting horizon.

        """
        # Non-panel data
        if self.panel_group_names_ is None:
            # Global data: _X_t_observed is a DataFrame
            X_t = self._X_t_observed[[-1]].select(~cs.by_name("time"))
            X_tab = X_t.select(list(self.local_X_t_schema_.keys())).to_numpy()
            y_tab_pred = estimator.predict(X_tab)  # type: ignore[attr-defined]
            y_pred = pl.DataFrame(
                y_tab_pred.reshape(
                    self.fit_forecasting_horizon_, len(list(self.local_y_t_schema_.keys()))
                ),
                schema=list(self.local_y_t_schema_.keys()),
            )
            # Cast to preserve dtypes from transformed target schema
            y_pred = cast(y_pred, self.local_y_t_schema_)

        # Panel data
        else:
            # Panel data: _X_t_observed is a dict (one DataFrame per group)
            # Each DataFrame has unprefixed columns (after get_group_df applied)
            y_pred_dict = {}
            for panel_group_name in panel_group_names:
                # Get X_t for this group (already unprefixed)
                X_t_group = self._X_t_observed[panel_group_name][[-1]].select(~cs.by_name("time"))

                # Use transformed schema to get feature order
                X_tab = X_t_group.select(list(self.local_X_t_schema_.keys())).to_numpy()

                # Get y columns from transformed schema (unprefixed)
                group_y_cols = list(self.local_y_t_schema_.keys())

                y_tab_pred = estimator.predict(X_tab)  # type: ignore[attr-defined]
                y_pred_local = pl.DataFrame(
                    y_tab_pred.reshape(self.fit_forecasting_horizon_, len(group_y_cols)),
                    schema=group_y_cols,
                )
                # Cast to preserve dtypes from transformed target schema
                y_pred_local = cast(y_pred_local, self.local_y_t_schema_)

                # Re-prefix column names for concatenation (e.g., "a" → "x__a")
                y_pred_local = y_pred_local.rename(
                    {col: f"{panel_group_name}__{col}" for col in group_y_cols}
                )

                y_pred_dict[panel_group_name] = y_pred_local

            y_pred = pl.concat(list(y_pred_dict.values()), how="horizontal")

        return y_pred

    def get_metadata_routing(self) -> MetadataRouter:
        """Get metadata routing including wrapped estimator.

        BaseReductionForecaster is a router because it wraps a sklearn estimator.
        It needs to forward metadata (like time_weight) from the forecaster's
        fit() method to the wrapped estimator's fit() method.

        Returns
        -------
        router : MetadataRouter
            Router that forwards to transformers (from parent) and wrapped estimator.
        """
        # Get parent routing (for target_transformer, feature_transformer)
        router = super().get_metadata_routing()

        # Add wrapped sklearn estimator routing
        if hasattr(self, "estimator") and self.estimator is not None:
            router.add(
                estimator=self.estimator,
                method_mapping=MethodMapping().add(caller="fit", callee="fit"),
            )

        return router


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
        (such as :class:`~sklearn.pipeline.FeaturePipeline`). The latter have
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
