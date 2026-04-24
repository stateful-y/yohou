"""Panel (multi-series) forecaster mixin for type-safe operations."""

from datetime import datetime, timedelta
from typing import TYPE_CHECKING

import polars as pl
import polars.selectors as cs

from yohou.base.utils import (
    _fit_transform_transformers_one,
    _observe_transformers_one,
    _rewind_transformers_one,
)
from yohou.utils import add_interval, get_group_df, inspect_panel

if TYPE_CHECKING:
    from yohou.base.transformer import BaseTransformer

__all__ = ["BasePanelForecaster"]


class BasePanelForecaster:
    """Mixin providing panel (dict of DataFrames) forecaster operations.

    This mixin provides methods with narrow return types for panel data
    (dict[str, pl.DataFrame]). Child classes that need type narrowing can
    explicitly call these methods via `BasePanelForecaster._pre_fit_panel(self, ...)`.

    See Also
    --------
    `BaseForecaster` : Main forecaster base combining standard and panel operations.
    `BaseStandardForecaster` : Standard (single DataFrame) forecaster mixin.
    `BaseReductionForecaster` : Reduction-based forecaster using sklearn regressors.

    """

    # Type hints for attributes set by BaseForecaster
    target_transformer: "BaseTransformer | None"
    feature_transformer: "BaseTransformer | None"
    target_as_feature: str | None
    groups_: list[str]
    local_y_schema_: dict[str, pl.DataType]
    local_X_schema_: dict[str, pl.DataType] | None
    shared_X_schema_: dict[str, pl.DataType] | None
    observation_horizon: int
    observed_time_: dict[str, datetime]
    interval_: timedelta | str

    def _set_input_attributes_panel(
        self,
        y: pl.DataFrame,
        X: pl.DataFrame | None,
        y_panel_groups: dict[str, list[str]],
        X_panel_groups: dict[str, list[str]] | None,
    ) -> None:
        """Set input attributes for panel data.

        Parameters
        ----------
        y : pl.DataFrame
            Target time series with panel columns.
        X : pl.DataFrame or None
            Feature time series with panel columns.
        y_panel_groups : dict[str, list[str]]
            Panel groups from y (group_name -> column_names).
        X_panel_groups : dict[str, list[str]] or None
            Panel groups from X.

        """
        self.groups_ = list(y_panel_groups.keys())

        # Extract suffixes from first group to validate consistency
        first_group_cols = y_panel_groups[self.groups_[0]]
        first_group_suffixes = [col.split("__", 1)[1] for col in first_group_cols]

        # Validate all groups have the same suffixes
        for group_name in self.groups_[1:]:
            group_cols = y_panel_groups[group_name]
            group_suffixes = [col.split("__", 1)[1] for col in group_cols]
            if sorted(group_suffixes) != sorted(first_group_suffixes):
                raise ValueError(
                    f"The local groups in `y` do not have the same column suffixes. "
                    f"Group '{self.groups_[0]}': {sorted(first_group_suffixes)}, "
                    f"Group '{group_name}': {sorted(group_suffixes)}"
                )

        # Extract y schema from first group
        local_y = y.select(first_group_cols).rename({col: col.split("__", 1)[1] for col in first_group_cols})
        self.local_y_schema_ = dict(local_y.schema)

        self.local_X_schema_ = None
        self.shared_X_schema_ = None
        if X is not None and X_panel_groups is not None:
            X_shared_names, _ = inspect_panel(X)

            if X_panel_groups:
                # X has panel columns: validate suffixes match across groups
                first_X_group_cols = X_panel_groups[self.groups_[0]]
                first_X_suffixes = [col.split("__", 1)[1] for col in first_X_group_cols]

                for group_name in self.groups_[1:]:
                    group_cols = X_panel_groups[group_name]
                    group_suffixes = [col.split("__", 1)[1] for col in group_cols]
                    if sorted(group_suffixes) != sorted(first_X_suffixes):
                        raise ValueError(
                            f"The local groups in `X` do not have the same column suffixes. "
                            f"Group '{self.groups_[0]}': {sorted(first_X_suffixes)}, "
                            f"Group '{group_name}': {sorted(group_suffixes)}"
                        )

                # Extract X schema (local + shared)
                self.shared_X_schema_ = dict(X.select(X_shared_names).schema)
                local_X = X.select(first_X_group_cols).rename({
                    col: col.split("__", 1)[1] for col in first_X_group_cols
                })
                self.local_X_schema_ = dict(local_X.schema)
            else:
                # Global-only X: all non-time columns are shared across groups
                self.shared_X_schema_ = dict(X.select(X_shared_names).schema)
                self.local_X_schema_ = {}

    def _fit_transform_inputs_panel(
        self, y: pl.DataFrame, X: pl.DataFrame | None
    ) -> tuple[dict[str, pl.DataFrame], dict[str, pl.DataFrame] | None]:
        """Fit transformers and transform inputs for panel data.

        Parameters
        ----------
        y : pl.DataFrame
            Target time series with panel columns.
        X : pl.DataFrame or None
            Feature time series with panel columns.

        Returns
        -------
        y_t : dict[str, pl.DataFrame]
            Transformed target per group.
        X_t : dict[str, pl.DataFrame] or None
            Transformed features per group.

        """
        y_t: dict[str, pl.DataFrame] = {}
        X_t: dict[str, pl.DataFrame | None] = {}
        target_transformer: dict[str, BaseTransformer | None] = {}
        feature_transformer: dict[str, BaseTransformer | None] = {}

        for group_name in self.groups_:
            # Extract group data using get_group_df
            y_local = get_group_df(df=y, group_name=group_name, schema=self.local_y_schema_)

            X_local = None
            if X is not None and self.local_X_schema_ is not None:
                # Build schema for X (local + shared columns)
                X_schema = dict(self.local_X_schema_)
                if self.shared_X_schema_:
                    X_schema.update(self.shared_X_schema_)
                X_local = get_group_df(df=X, group_name=group_name, schema=X_schema)

            (
                y_t_local,
                X_t_local,
                target_transformer_local,
                feature_transformer_local,
            ) = _fit_transform_transformers_one(
                y=y_local,
                X=X_local,
                target_transformer=self.target_transformer,
                feature_transformer=self.feature_transformer,
                target_as_feature=self.target_as_feature,
            )

            y_t[group_name] = y_t_local
            X_t[group_name] = X_t_local
            target_transformer[group_name] = target_transformer_local
            feature_transformer[group_name] = feature_transformer_local

        self.target_transformer_ = target_transformer
        self.feature_transformer_ = feature_transformer

        # Filter out None values from X_t if all are None
        X_t_result: dict[str, pl.DataFrame] | None = None
        if any(v is not None for v in X_t.values()):
            X_t_result = {k: v for k, v in X_t.items() if v is not None}

        return y_t, X_t_result

    def _set_transformed_attributes_panel(
        self,
        y_t: dict[str, pl.DataFrame],
        X_t: dict[str, pl.DataFrame] | None,
    ) -> None:
        """Set transformed data attributes for panel data.

        Parameters
        ----------
        y_t : dict[str, pl.DataFrame]
            Transformed target per group.
        X_t : dict[str, pl.DataFrame] or None
            Transformed features per group.

        """
        # Get schema from first group (all groups have same structure)
        first_group_name = next(iter(y_t))
        y_t_df = y_t[first_group_name]
        self.local_y_t_schema_ = dict(y_t_df.select(~cs.by_name("time")).schema)

        self.local_X_t_schema_ = None
        if X_t is not None:
            X_t_first_group = X_t[first_group_name]
            if X_t_first_group is not None:
                self.local_X_t_schema_ = dict(X_t_first_group.select(~cs.by_name("time")).schema)

        # Store n_features_in_ and feature_names_in_ for sklearn compatibility
        if self.local_X_t_schema_:
            self.n_features_in_ = len(self.local_X_t_schema_)
            self.feature_names_in_ = list(self.local_X_t_schema_.keys())
        else:
            self.n_features_in_ = 0
            self.feature_names_in_ = []

    def _update_y_X_t_observed_panel(
        self,
        y: pl.DataFrame,
        X_t: dict[str, pl.DataFrame] | None,
        groups: list[str],
    ) -> None:
        """Update stored observed data for panel data.

        Extracts per-group target DataFrames from ``y`` and delegates
        to ``_update_y_X_t_observed_from_dicts``.

        Parameters
        ----------
        y : pl.DataFrame
            Target time series with panel columns (original format).
        X_t : dict[str, pl.DataFrame] or None
            Transformed features per group.
        groups : list[str]
            Panel group names to update.

        """
        y_dict: dict[str, pl.DataFrame] = {}
        for panel_group_name in groups:
            y_dict[panel_group_name] = get_group_df(df=y, group_name=panel_group_name, schema=self.local_y_schema_)
        self._update_y_X_t_observed_from_dicts(y_dict, X_t, groups)

    def _pre_fit_panel(
        self,
        y: pl.DataFrame,
        X: pl.DataFrame | None,
        forecasting_horizon: int,
        y_panel_groups: dict[str, list[str]],
        X_panel_groups: dict[str, list[str]] | None,
    ) -> tuple[dict[str, pl.DataFrame], dict[str, pl.DataFrame] | None]:
        """Preprocessing and transform for panel data (narrow types).

        Parameters
        ----------
        y : pl.DataFrame
            Target time series with panel columns (already validated).
        X : pl.DataFrame or None
            Feature time series with panel columns (already validated).
        forecasting_horizon : int
            Number of steps ahead to forecast.
        y_panel_groups : dict[str, list[str]]
            Panel groups from y (group_name -> column_names).
        X_panel_groups : dict[str, list[str]] or None
            Panel groups from X.

        Returns
        -------
        y_t : dict[str, pl.DataFrame]
            Transformed target per group.
        X_t : dict[str, pl.DataFrame] or None
            Transformed features per group.

        """
        self._set_input_attributes_panel(y, X, y_panel_groups, X_panel_groups)
        y_t, X_t = self._fit_transform_inputs_panel(y, X)
        self._set_transformed_attributes_panel(y_t, X_t)
        self._update_y_X_t_observed_panel(y, X_t, self.groups_)

        return y_t, X_t

    def _rewind_panel(
        self,
        y: pl.DataFrame,
        X: pl.DataFrame | None,
        groups: list[str],
    ) -> "BasePanelForecaster":
        """Reset state for panel data.

        Parameters
        ----------
        y : pl.DataFrame
            Target time series with panel columns.
        X : pl.DataFrame or None
            Feature time series with panel columns.
        groups : list[str]
            Panel group names to reset.

        Returns
        -------
        self

        """
        X_t: dict[str, pl.DataFrame | None] = {}

        for panel_group_name in groups:
            # Extract group data using get_group_df
            y_local = get_group_df(df=y, group_name=panel_group_name, schema=self.local_y_schema_)

            X_local = None
            if X is not None and self.local_X_schema_ is not None:
                # Build schema for X (local + shared columns)
                X_schema = dict(self.local_X_schema_)
                if self.shared_X_schema_:
                    X_schema.update(self.shared_X_schema_)
                X_local = get_group_df(df=X, group_name=panel_group_name, schema=X_schema)

            local_target_transformer = None
            if self.target_transformer is not None and isinstance(self.target_transformer_, dict):
                local_target_transformer = self.target_transformer_[panel_group_name]

            local_feature_transformer = None
            if self.feature_transformer is not None and isinstance(self.feature_transformer_, dict):
                local_feature_transformer = self.feature_transformer_[panel_group_name]

            X_t_local = _rewind_transformers_one(
                y_local,
                X_local,
                local_target_transformer,
                local_feature_transformer,
                self.observation_horizon,
                self.target_as_feature,
            )

            X_t[panel_group_name] = X_t_local

        # Filter out None values
        X_t_filtered: dict[str, pl.DataFrame] | None = None
        if any(v is not None for v in X_t.values()):
            X_t_filtered = {k: v for k, v in X_t.items() if v is not None}

        self._update_y_X_t_observed_panel(y, X_t_filtered, groups)
        return self

    def _observe_panel(
        self,
        y: pl.DataFrame,
        X: pl.DataFrame | None,
        groups: list[str],
    ) -> "BasePanelForecaster":
        """Update state with new observations for panel data.

        Parameters
        ----------
        y : pl.DataFrame
            New target observations with panel columns.
        X : pl.DataFrame or None
            New feature observations with panel columns.
        groups : list[str]
            Panel group names to update.

        Returns
        -------
        self

        """
        X_t_updated: dict[str, pl.DataFrame | None] = {}
        y_updated: dict[str, pl.DataFrame] = {}

        for panel_group_name in groups:
            # Extract group data for new observations only
            y_local = get_group_df(df=y, group_name=panel_group_name, schema=self.local_y_schema_)

            X_local = None
            if X is not None and self.local_X_schema_ is not None:
                # Build schema for X (local + shared columns)
                X_schema = dict(self.local_X_schema_)
                if self.shared_X_schema_:
                    X_schema.update(self.shared_X_schema_)
                X_local = get_group_df(df=X, group_name=panel_group_name, schema=X_schema)

            local_target_transformer = None
            if self.target_transformer is not None and isinstance(self.target_transformer_, dict):
                local_target_transformer = self.target_transformer_[panel_group_name]

            local_feature_transformer = None
            if self.feature_transformer is not None and isinstance(self.feature_transformer_, dict):
                local_feature_transformer = self.feature_transformer_[panel_group_name]

            # Update transformers with new data only
            X_t_updated[panel_group_name] = _observe_transformers_one(
                y_local,
                X_local,
                local_target_transformer,
                local_feature_transformer,
                self.target_as_feature,
            )

            # For y_updated, concatenate stored observations with new observations
            y_full = y_local
            if self._y_observed is not None and panel_group_name in self._y_observed:
                assert isinstance(self._y_observed, dict)
                y_stored = self._y_observed[panel_group_name]
                if y_stored is not None:
                    y_full = pl.concat([y_stored, y_local], how="vertical")
            y_updated[panel_group_name] = y_full

        # Filter out None values from X_t_updated
        X_t_filtered: dict[str, pl.DataFrame] | None = None
        if any(v is not None for v in X_t_updated.values()):
            X_t_filtered = {k: v for k, v in X_t_updated.items() if v is not None}

        # Update observed state - pass the concatenated y_updated dict
        # The panel mixin's _update_y_X_t_observed_panel expects original format DataFrame
        # but we have pre-extracted dicts here. We need to use the dict-based update.
        self._update_y_X_t_observed_from_dicts(y_updated, X_t_filtered, groups)

        return self

    def _update_y_X_t_observed_from_dicts(
        self,
        y: dict[str, pl.DataFrame],
        X_t: dict[str, pl.DataFrame] | None,
        groups: list[str],
    ) -> None:
        """Update stored observed data from pre-split dicts.

        This is an alternative to _update_y_X_t_observed_panel that accepts
        pre-split dictionaries instead of the original DataFrame format.

        Parameters
        ----------
        y : dict[str, pl.DataFrame]
            Target time series per group (already extracted).
        X_t : dict[str, pl.DataFrame] or None
            Transformed features per group.
        groups : list[str]
            Panel group names to update.

        """
        self.observed_time_ = {}

        X_t_observed: dict[str, pl.DataFrame | None] | None = None
        if X_t is not None:
            X_t_observed = {}

        y_observed: dict[str, pl.DataFrame | None] = {}
        for panel_group_name in groups:
            y_group = y[panel_group_name]

            if self.observation_horizon > len(y_group):
                raise ValueError(f"Not enough data to set observed y for group {panel_group_name}.")

            self.observed_time_[panel_group_name] = y_group["time"][-1]
            y_observed[panel_group_name] = (
                y_group[-self.observation_horizon :] if self.observation_horizon > 0 else None
            )

            # Store X_t_observed for this group
            if X_t_observed is not None and X_t is not None:
                X_t_group = X_t.get(panel_group_name)
                X_t_observed[panel_group_name] = X_t_group[[-1]] if X_t_group is not None else None

        self._y_observed = y_observed
        self._X_t_observed = X_t_observed

    def _add_time_columns_panel(self, y_pred: pl.DataFrame) -> pl.DataFrame:
        """Add time metadata columns to predictions for panel data.

        Parameters
        ----------
        y_pred : pl.DataFrame
            Predictions without time columns.

        Returns
        -------
        pl.DataFrame
            Predictions with vintage_time and time columns.

        """
        # For panel data, all groups share the same time progression
        # Use the first group's observed_time_ as reference
        first_group_name = list(self.observed_time_.keys())[0]
        observed_time_value = self.observed_time_[first_group_name]

        predicted_times = [add_interval(observed_time_value, self.interval_, n=n) for n in range(1, len(y_pred) + 1)]

        time = pl.DataFrame({"vintage_time": [observed_time_value] * len(y_pred), "time": predicted_times})

        return pl.concat([time, y_pred], how="horizontal")
