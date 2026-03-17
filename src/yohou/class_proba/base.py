"""Base class for class-probability forecasters."""

import abc

import polars as pl
from pydantic import StrictInt
from sklearn.utils.validation import check_is_fitted

from yohou.base import BaseForecaster
from yohou.utils import Tags, select_panel_columns, validate_forecaster_data

__all__ = ["BaseClassProbaForecaster"]


class BaseClassProbaForecaster(BaseForecaster, metaclass=abc.ABCMeta):
    """Base class for class-probability forecasters.

    Class-probability forecasters produce per-class probability distributions
    for categorical time series at each forecast step. The primary output
    method is ``predict_class_proba``; ``predict`` returns the argmax class.

    Parameters
    ----------
    target_transformer : instance of `BaseTransformer` or None, default=None
        Transformer used to transform the target time series into the new target.
    feature_transformer : instance of `BaseTransformer` or None, default=None
        Transformer used to transform the target time series into features.
    target_as_feature : {"transformed", "raw"} or None, default="transformed"
        Controls whether the target is included as a feature.
        ``"transformed"`` includes the transformed target, ``"raw"``
        includes the raw target, and ``None`` uses only exogenous features.
    panel_strategy : {"global", "multivariate"}, default="global"
        How to handle panel data. See `BaseForecaster` for details.

    Notes
    -----
    Subclasses must implement ``_predict_class_proba_one`` to produce
    probability forecasts for a single forecast step. The ``forecaster_type``
    tag is set to ``"class_proba"``.

    See Also
    --------
    `ClassProbaReductionForecaster` : ML-based class-probability forecaster.
    `BasePointForecaster` : Base class for point forecasters.

    """

    classes_: dict[str, list[str]]
    n_classes_: dict[str, int]
    label_to_code_: dict[str, dict[str, float]]

    def __sklearn_tags__(self) -> Tags:
        """Get estimator tags.

        Returns
        -------
        Tags
            Estimator tags with yohou-specific attributes.

        """
        tags = super().__sklearn_tags__()
        assert tags.forecaster_tags is not None
        tags.forecaster_tags.forecaster_type = "class_proba"
        return tags

    def fit(
        self,
        y: pl.DataFrame,
        X: pl.DataFrame | None = None,
        forecasting_horizon: StrictInt = 1,
        **params,
    ) -> "BaseClassProbaForecaster":
        """Fit the forecaster to historical data.

        Parameters
        ----------
        y : pl.DataFrame
            Target time series with a ``"time"`` column (datetime) and one
            or more categorical value columns.
        X : pl.DataFrame or None, default=None
            Exogenous features with a ``"time"`` column matching ``y``.
            If ``None``, no exogenous features are used.
        forecasting_horizon : int, default=1
            Number of time steps to forecast into the future.
        **params : dict
            Metadata to route to nested estimators.

        Returns
        -------
        self
            The fitted forecaster instance.

        Raises
        ------
        ValueError
            If ``forecasting_horizon`` < 1, or if ``y`` / ``X`` have invalid
            structure (e.g., missing ``"time"`` column).

        """
        forecasting_horizon = self._validate_fit_params(forecasting_horizon)

        BaseForecaster._pre_fit(
            self,
            y=y,
            X=X,
            forecasting_horizon=forecasting_horizon,
        )

        return self

    def _validate_fit_params(self, forecasting_horizon: StrictInt) -> StrictInt:
        """Validate fit parameters.

        Parameters
        ----------
        forecasting_horizon : int
            Forecasting horizon to validate.

        Returns
        -------
        int
            Validated forecasting horizon.

        Raises
        ------
        ValueError
            If forecasting_horizon < 1.

        """
        if forecasting_horizon < 1:
            raise ValueError(f"forecasting_horizon must be >= 1, got {forecasting_horizon}")
        return forecasting_horizon

    def _validate_predict_params(self, forecasting_horizon: StrictInt | None) -> StrictInt:
        """Validate and return predict parameters.

        Parameters
        ----------
        forecasting_horizon : int or None
            Forecasting horizon to validate. If None, uses fit_forecasting_horizon_.

        Returns
        -------
        int
            Validated forecasting horizon.

        Raises
        ------
        ValueError
            If forecasting_horizon < 1.

        """
        if forecasting_horizon is None:
            forecasting_horizon = self.fit_forecasting_horizon_
        return self._validate_fit_params(forecasting_horizon)

    @abc.abstractmethod
    def _predict_class_proba_one(
        self,
        panel_group_names: list[str],
        **params,
    ) -> pl.DataFrame:
        """Produce probability forecasts for one fit-horizon block.

        Must be implemented by subclasses. Returns a DataFrame where
        each target column is expanded into ``n_classes`` columns named
        ``{target}_proba_{class_label}``.

        Parameters
        ----------
        panel_group_names : list of str
            Panel group names to predict for.
        **params : dict
            Metadata to route to nested estimators.

        Returns
        -------
        pl.DataFrame
            Probability predictions with ``"observed_time"``, ``"time"``,
            and columns ``{target}_proba_{class_label}`` for each class.

        """

    def predict_class_proba(
        self,
        X: pl.DataFrame | None = None,
        forecasting_horizon: StrictInt | None = None,
        panel_group_names: list[str] | None = None,
        **params,
    ) -> pl.DataFrame:
        """Generate class-probability forecasts.

        Parameters
        ----------
        X : pl.DataFrame or None, default=None
            Exogenous features with a ``"time"`` column matching ``y``.
            If ``None``, no exogenous features are used.
        forecasting_horizon : int or None, default=None
            Number of time steps to forecast into the future. If ``None``,
            uses the horizon specified at fit time.
        panel_group_names : list of str or None, default=None
            Panel group prefixes to operate on. If ``None``, all groups
            are used. Ignored when the forecaster was not fitted on panel
            data.
        **params : dict
            Metadata to route to nested estimators.

        Returns
        -------
        pl.DataFrame
            Probability predictions with ``"observed_time"``, ``"time"``,
            and columns ``{target}_proba_{class_label}`` for each class.

        Raises
        ------
        sklearn.exceptions.NotFittedError
            If the forecaster has not been fitted yet.
        ValueError
            If ``X`` has invalid structure or ``panel_group_names`` contains
            names not seen during fit.

        """
        check_is_fitted(
            self,
            ["local_y_schema_", "local_X_schema_", "shared_X_schema_", "panel_group_names_"],
        )

        _, X, panel_group_names = validate_forecaster_data(
            self,
            y=None,
            X=X,
            reset=False,
            panel_group_names=panel_group_names,
        )

        forecasting_horizon = self._validate_predict_params(forecasting_horizon)

        if panel_group_names is not None and X is not None:
            X = select_panel_columns(
                X,
                panel_group_names,
                include_global=True,
            )

        def step_fn(forecaster, groups):
            """Produce one class-probability prediction block."""
            y_pred_step = forecaster._predict_class_proba_one(
                panel_group_names=groups,
                **params,
            )
            return y_pred_step, y_pred_step

        def derive_observation_fn(forecaster, y_pred_step, X, step):
            """Derive observation via argmax and re-encoding."""
            y_obs = self._argmax_from_proba(y_pred_step)
            y_obs = self._encode_observation(y_obs)

            X_slice = None
            if X is not None:
                start_idx = step
                end_idx = start_idx + self.fit_forecasting_horizon_
                X_slice = X[start_idx:end_idx]

                if len(X_slice) != len(y_obs):
                    raise ValueError(
                        f"Missing X for future steps. Needed {len(y_obs)} rows, "
                        f"but X slice has {len(X_slice)} rows."
                    )

            return y_obs, X_slice

        return self._recursive_predict(
            X=X,
            forecasting_horizon=forecasting_horizon,
            panel_group_names=panel_group_names,
            step_fn=step_fn,
            derive_observation_fn=derive_observation_fn,
        )

    def predict(
        self,
        X: pl.DataFrame | None = None,
        forecasting_horizon: StrictInt | None = None,
        panel_group_names: list[str] | None = None,
        **params,
    ) -> pl.DataFrame:
        """Generate argmax class forecasts from class probabilities.

        Convenience method that calls ``predict_class_proba`` and returns
        the most-likely class for each time step and target column.

        Parameters
        ----------
        X : pl.DataFrame or None, default=None
            Exogenous features with a ``"time"`` column matching ``y``.
            If ``None``, no exogenous features are used.
        forecasting_horizon : int or None, default=None
            Number of time steps to forecast into the future. If ``None``,
            uses the horizon specified at fit time.
        panel_group_names : list of str or None, default=None
            Panel group prefixes to operate on. If ``None``, all groups
            are used. Ignored when the forecaster was not fitted on panel
            data.
        **params : dict
            Metadata to route to nested estimators.

        Returns
        -------
        pl.DataFrame
            Point predictions with ``"observed_time"``, ``"time"``, and one
            column per target variable containing the most-likely class.

        Raises
        ------
        sklearn.exceptions.NotFittedError
            If the forecaster has not been fitted yet.

        """
        y_proba = self.predict_class_proba(
            X=X,
            forecasting_horizon=forecasting_horizon,
            panel_group_names=panel_group_names,
            **params,
        )
        return self._argmax_from_proba(y_proba)

    def _argmax_from_proba(self, y_proba: pl.DataFrame) -> pl.DataFrame:
        """Convert probability DataFrame to argmax class DataFrame.

        Takes the probability output (columns named ``{target}_proba_{class}``)
        and returns the class with highest probability for each target
        and time step.

        Parameters
        ----------
        y_proba : pl.DataFrame
            Probability predictions from ``predict_class_proba``.

        Returns
        -------
        pl.DataFrame
            DataFrame with ``"time"``, and one column per original target
            containing the class label with highest probability.

        """
        check_is_fitted(self, ["classes_"])

        time_cols = [c for c in ("observed_time", "time") if c in y_proba.columns]
        result = y_proba.select(time_cols)

        for target_col, class_labels in self.classes_.items():
            proba_cols = [f"{target_col}_proba_{label}" for label in class_labels]
            # For each row, find the index of the max probability column
            # then map that index to the class label.
            argmax_series = y_proba.select(pl.concat_list(proba_cols).list.arg_max().alias("_idx"))["_idx"]
            label_series = pl.Series(values=class_labels)
            result = result.with_columns(
                argmax_series.map_elements(
                    lambda idx, _labels=label_series: _labels[idx],
                    return_dtype=pl.String,
                ).alias(target_col),
            )

        return result

    def _encode_observation(self, y_obs: pl.DataFrame) -> pl.DataFrame:
        """Encode argmax string labels back to float codes for observation.

        Used during recursive prediction to convert argmax class labels
        back to the integer-coded format expected by ``observe()``.

        Parameters
        ----------
        y_obs : pl.DataFrame
            Observation with string class labels.

        Returns
        -------
        pl.DataFrame
            Observation with float-coded class labels matching the fit schema.

        """
        check_is_fitted(self, ["label_to_code_"])

        exprs = []
        for col in y_obs.columns:
            if col in ("observed_time", "time"):
                continue
            mapping = self.label_to_code_[col]
            exprs.append(pl.col(col).cast(pl.String).replace_strict(mapping, return_dtype=pl.Float64).alias(col))
        return y_obs.with_columns(exprs)

    def _encode_y_input(self, y: pl.DataFrame) -> pl.DataFrame:
        """Encode user-facing categorical y to float codes for internal use.

        Handles both panel (``{group}__{target}``) and non-panel column
        names by looking up the base target name in ``label_to_code_``.

        Parameters
        ----------
        y : pl.DataFrame
            Target data with string or already-encoded columns.

        Returns
        -------
        pl.DataFrame
            Target data with float-coded columns matching ``local_y_schema_``.

        """
        check_is_fitted(self, ["label_to_code_"])

        exprs = []
        for col in y.columns:
            if col == "time":
                continue
            # Skip columns that are already numeric (already encoded)
            if y[col].dtype.is_numeric():
                continue
            # For panel columns like "group_0__weather", extract "weather"
            base_col = col.split("__")[-1] if "__" in col else col
            if base_col in self.label_to_code_:
                mapping = self.label_to_code_[base_col]
                exprs.append(
                    pl.col(col).replace_strict(mapping, return_dtype=pl.Float64).alias(col)
                )
        if exprs:
            return y.with_columns(exprs)
        return y

    def observe(
        self,
        y: pl.DataFrame,
        X: pl.DataFrame | None = None,
        panel_group_names: list[str] | None = None,
    ) -> "BaseClassProbaForecaster":
        """Observe new data, encoding categorical targets before validation.

        Overrides ``BaseForecaster.observe`` to encode string target columns
        to float codes before schema validation.

        Parameters
        ----------
        y : pl.DataFrame
            Target time series with a ``"time"`` column (datetime) and one
            or more categorical value columns.
        X : pl.DataFrame or None, default=None
            Exogenous features with a ``"time"`` column matching ``y``.
            If ``None``, no exogenous features are used.
        panel_group_names : list of str or None, default=None
            Panel group prefixes to operate on. If ``None``, all groups
            are used.

        Returns
        -------
        self
            The forecaster with updated observation buffers.

        """
        y = self._encode_y_input(y)
        return super().observe(y, X, panel_group_names=panel_group_names)

    def rewind(
        self,
        y: pl.DataFrame,
        X: pl.DataFrame | None = None,
        panel_group_names: list[str] | None = None,
    ) -> "BaseClassProbaForecaster":
        """Rewind memory, encoding categorical targets before validation.

        Overrides ``BaseForecaster.rewind`` to encode string target columns
        to float codes before schema validation.

        Parameters
        ----------
        y : pl.DataFrame
            Target time series with a ``"time"`` column (datetime) and one
            or more categorical value columns.
        X : pl.DataFrame or None, default=None
            Exogenous features with a ``"time"`` column matching ``y``.
            If ``None``, no exogenous features are used.
        panel_group_names : list of str or None, default=None
            Panel group prefixes to operate on. If ``None``, all groups
            are used.

        Returns
        -------
        self
            The forecaster with rewound observation buffers.

        """
        y = self._encode_y_input(y)
        return super().rewind(y, X, panel_group_names=panel_group_names)

    def observe_predict_class_proba(
        self,
        y: pl.DataFrame,
        X: pl.DataFrame | None = None,
        forecasting_horizon: StrictInt | None = None,
        panel_group_names: list[str] | None = None,
        stride: StrictInt | None = None,
        **params,
    ) -> pl.DataFrame:
        """Alternate recursive predict_class_proba and observe.

        Equivalent to calling ``observe(y, X)`` then
        ``predict_class_proba(X)``. Returns probability predictions.

        Parameters
        ----------
        y : pl.DataFrame
            Target time series with a ``"time"`` column (datetime) and one
            or more categorical value columns.
        X : pl.DataFrame or None, default=None
            Exogenous features with a ``"time"`` column matching ``y``.
            If ``None``, no exogenous features are used.
        forecasting_horizon : int or None, default=None
            Number of time steps to forecast into the future. If ``None``,
            uses the horizon specified at fit time.
        panel_group_names : list of str or None, default=None
            Panel group prefixes to operate on. If ``None``, all groups
            are used. Ignored when the forecaster was not fitted on panel
            data.
        stride : int or None, default=None
            Step size for rolling update-predict. If ``None``, defaults to
            ``forecasting_horizon``.
        **params : dict
            Metadata to route to nested estimators.

        Returns
        -------
        pl.DataFrame
            Probability predictions with ``"observed_time"``, ``"time"``,
            and columns ``{target}_proba_{class_label}`` for each class.

        Raises
        ------
        sklearn.exceptions.NotFittedError
            If the forecaster has not been fitted yet.
        ValueError
            If ``y`` / ``X`` have invalid structure or ``panel_group_names``
            contains names not seen during fit.

        """
        check_is_fitted(
            self,
            ["local_y_schema_", "local_X_schema_", "shared_X_schema_", "panel_group_names_"],
        )

        y = self._encode_y_input(y)

        y, X, panel_group_names = validate_forecaster_data(
            self,
            y=y,
            X=X,
            reset=False,
            panel_group_names=panel_group_names,
        )

        forecasting_horizon = self._validate_predict_params(forecasting_horizon)
        if stride is None:
            stride = self.fit_forecasting_horizon_

        return self._observe_predict_loop(
            predict_fn=self.predict_class_proba,
            y=y,
            X=X,
            panel_group_names=panel_group_names,
            stride=stride,
            forecasting_horizon=forecasting_horizon,
            **params,
        )

    def observe_predict(
        self,
        y: pl.DataFrame,
        X: pl.DataFrame | None = None,
        forecasting_horizon: StrictInt | None = None,
        panel_group_names: list[str] | None = None,
        stride: StrictInt | None = None,
        **params,
    ) -> pl.DataFrame:
        """Alternate recursive predict and observe.

        Equivalent to calling ``observe(y, X)`` then ``predict(X)``.
        Returns argmax class predictions.

        Parameters
        ----------
        y : pl.DataFrame
            Target time series with a ``"time"`` column (datetime) and one
            or more categorical value columns.
        X : pl.DataFrame or None, default=None
            Exogenous features with a ``"time"`` column matching ``y``.
            If ``None``, no exogenous features are used.
        forecasting_horizon : int or None, default=None
            Number of time steps to forecast into the future. If ``None``,
            uses the horizon specified at fit time.
        panel_group_names : list of str or None, default=None
            Panel group prefixes to operate on. If ``None``, all groups
            are used. Ignored when the forecaster was not fitted on panel
            data.
        stride : int or None, default=None
            Step size for rolling update-predict. If ``None``, defaults to
            ``forecasting_horizon``.
        **params : dict
            Metadata to route to nested estimators.

        Returns
        -------
        pl.DataFrame
            Point predictions with ``"observed_time"``, ``"time"``, and one
            column per target variable containing the most-likely class.

        Raises
        ------
        sklearn.exceptions.NotFittedError
            If the forecaster has not been fitted yet.
        ValueError
            If ``y`` / ``X`` have invalid structure or ``panel_group_names``
            contains names not seen during fit.

        """
        check_is_fitted(
            self,
            ["local_y_schema_", "local_X_schema_", "shared_X_schema_", "panel_group_names_"],
        )

        y = self._encode_y_input(y)

        y, X, panel_group_names = validate_forecaster_data(
            self,
            y=y,
            X=X,
            reset=False,
            panel_group_names=panel_group_names,
        )

        forecasting_horizon = self._validate_predict_params(forecasting_horizon)
        if stride is None:
            stride = self.fit_forecasting_horizon_

        return self._observe_predict_loop(
            predict_fn=self.predict,
            y=y,
            X=X,
            panel_group_names=panel_group_names,
            stride=stride,
            forecasting_horizon=forecasting_horizon,
            **params,
        )
