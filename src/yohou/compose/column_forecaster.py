"""Composition classes for composing forecasters."""

from collections import Counter
from numbers import Integral
from typing import Any, Literal

import polars as pl
import polars.selectors as cs
from sklearn.base import clone
from sklearn.utils import Bunch
from sklearn.utils.metadata_routing import (
    MethodMapping,
    process_routing,
)
from sklearn.utils.metaestimators import available_if
from sklearn.utils.parallel import Parallel, delayed
from sklearn.utils.validation import check_is_fitted

from yohou.base import BaseForecaster
from yohou.utils import Tags
from yohou.utils._compat import StrOptions, _BaseComposition, _fit_context, _raise_for_params

__all__ = ["ColumnForecaster"]


def _column_forecaster_has(method_name: str):
    """Check if all nested forecasters have a given method.

    This function is used with ``sklearn.utils.metaestimators.available_if``
    to conditionally expose prediction methods based on whether all nested
    forecasters support them.

    Parameters
    ----------
    method_name : str
        Name of the method to check for (e.g., ``"predict"``,
        ``"predict_interval"``).

    Returns
    -------
    callable
        A function that accepts a ColumnForecaster and returns True if
        all nested forecasters have the specified method.

    """

    def check(self):
        """Check if all forecasters have the required method."""
        if hasattr(self, "forecasters_"):
            forecasters = [f for _, f, _ in self.forecasters_]
            if self.remainder_forecaster_ is not None:
                forecasters.append(self.remainder_forecaster_)
        else:
            forecasters = [f for _, f, _ in self.forecasters]
            if isinstance(self.remainder, BaseForecaster):
                forecasters.append(self.remainder)
        return all(hasattr(f, method_name) for f in forecasters)

    return check


def _fit_one_forecaster(
    forecaster: BaseForecaster,
    name: str,
    y: pl.DataFrame,
    X_actual: pl.DataFrame | None,
    forecasting_horizon: int,
    params: Any,
    X_future: pl.DataFrame | None = None,
    X_forecast: pl.DataFrame | None = None,
) -> tuple[str, BaseForecaster]:
    """Fit a single forecaster.

    Parameters
    ----------
    forecaster : BaseForecaster
        Forecaster to fit (will be cloned).
    name : str
        Name of the forecaster.
    y : pl.DataFrame
        Target time series.
    X_actual : pl.DataFrame, optional
        Exogenous features.
    forecasting_horizon : int
        Forecasting horizon.
    params : Bunch
        Routed parameters from process_routing.
    X_future : pl.DataFrame or None, default=None
        Known future features.
    X_forecast : pl.DataFrame or None, default=None
        External forecasts.

    Returns
    -------
    tuple[str, BaseForecaster]
        Name and fitted forecaster.

    """
    forecaster_clone = clone(forecaster)
    forecaster_clone.fit(
        y, X_actual, forecasting_horizon=forecasting_horizon, X_future=X_future, X_forecast=X_forecast, **params.fit
    )
    return name, forecaster_clone


class ColumnForecaster(BaseForecaster, _BaseComposition):
    """Applies different forecasters to different column subsets.

    This estimator allows different columns or column subsets of the target
    time series to be forecasted separately using different forecasters.
    The predictions from each forecaster are concatenated horizontally
    to form the final prediction.

    This is the forecasting equivalent of sklearn's ColumnTransformer.

    Parameters
    ----------
    forecasters : list of (name, forecaster, columns) tuples
        List specifying which forecaster applies to which columns.

        name : str
            Unique identifier for the forecaster. Used for accessing via
            ``named_forecasters_`` and for ``get_params``/``set_params``.
        forecaster : BaseForecaster
            Forecaster instance to apply to the specified columns.
        columns : str or list of str
            Column name(s) this forecaster will predict.

    remainder : {"drop", "passthrough"} or BaseForecaster, default="drop"
        How to handle columns not assigned to any forecaster:

        - ``"drop"``: Unassigned columns are excluded from predictions.
        - ``"passthrough"``: Accepted as an alias of ``"drop"``; unassigned
          columns are excluded from predictions. Note this differs from
          sklearn's ``ColumnTransformer``, where ``"passthrough"`` carries
          columns through unchanged: a forecaster cannot pass observed
          columns through as future predictions, so both strings drop them.
        - estimator: A ``BaseForecaster`` instance used to forecast the
          unassigned columns.

    n_jobs : int or None, default=None
        Number of jobs to run in parallel for fitting forecasters.
        ``None`` means 1 unless in a ``joblib.parallel_backend`` context.
        ``-1`` means using all processors.

    verbose_feature_names_out : bool, default=False
        If True, prefix output column names with the forecaster name
        (e.g., 'sales_forecaster__sales'). If False, keep original column names.
        This default differs from `ColumnTransformer` (which defaults to
        ``verbose_feature_names_out=True``): the forecaster side preserves
        original column names by default, while the transformer side prefixes
        by default for feature clarity. Columns from the remainder forecaster
        are never prefixed, even when this is ``True``.
    panel_strategy : {"global", "multivariate"}, default="global"
        How to handle panel data. See `BaseForecaster` for details.

    Attributes
    ----------
    forecasters_ : list of (name, fitted_forecaster, columns) tuples
        The fitted forecasters with their column assignments.

    column_map_ : dict of str to list of str
        Mapping from each forecaster name to the list of columns it was
        assigned at fit time.

    named_forecasters_ : Bunch
        Access any fitted forecaster by name.
        ``forecaster.named_forecasters_['sales']`` returns the forecaster
        named 'sales' (a forecaster name from the ``forecasters`` argument,
        not a panel-data group).

    remainder_forecaster_ : BaseForecaster or None
        The fitted remainder forecaster, or ``None`` if ``remainder`` is a
        string or there were no unassigned columns.

    remainder_cols_ : list of str
        Column names handled by the remainder forecaster.

    Examples
    --------
    >>> import polars as pl
    >>> from datetime import datetime
    >>> from yohou.compose import ColumnForecaster
    >>> from yohou.point import SeasonalNaive
    >>>
    >>> y = pl.DataFrame({
    ...     "time": pl.datetime_range(
    ...         start=datetime(2022, 1, 1), end=datetime(2022, 1, 10), interval="1d", eager=True
    ...     ),
    ...     "sales": range(10),
    ...     "inventory": range(10),
    ...     "price": range(10),
    ... })
    >>>
    >>> # Apply different forecasters to different columns
    >>> forecaster = ColumnForecaster([
    ...     ("sales_model", SeasonalNaive(seasonality=1), ["sales", "inventory"]),
    ...     ("price_model", SeasonalNaive(seasonality=1), "price"),
    ... ])
    >>>
    >>> forecaster = forecaster.fit(y, forecasting_horizon=3)
    >>> y_pred = forecaster.predict(forecasting_horizon=3)
    >>>
    >>> # Access fitted forecasters by name
    >>> sales_forecaster = forecaster.named_forecasters_["sales_model"]

    See Also
    --------
    - [`ColumnTransformer`][yohou.compose.column_transformer.ColumnTransformer] : Column-wise transformer composition.
    - [`DecompositionPipeline`][yohou.compose.decomposition_pipeline.DecompositionPipeline] : Sequential residual-based forecaster composition.

    Notes
    -----
    - Each column must appear in exactly one forecaster's column list
      (no overlapping columns allowed)
    - Columns not assigned to any forecaster are handled according to
      ``remainder``: dropped, or forecasted by an estimator
    - Forecasters are fitted in parallel when ``n_jobs`` is not ``None`` or
      ``1`` (e.g., ``-1`` uses all CPUs)
    - Predictions are concatenated in the order: forecasters, then remainder
    - All forecasters (including the remainder) receive the same exogenous
      inputs: ``X_actual``, ``X_future``, and ``X_forecast``

    """

    _parameter_constraints: dict = {
        **BaseForecaster._parameter_constraints,
        "forecasters": [list],
        "remainder": [StrOptions({"drop", "passthrough"}), BaseForecaster],
        "n_jobs": [Integral, None],
        "verbose_feature_names_out": ["boolean"],
    }

    def __init__(
        self,
        forecasters: list[tuple[str, BaseForecaster, str | list[str]]],
        *,
        remainder: str | BaseForecaster = "drop",
        n_jobs: int | None = None,
        verbose_feature_names_out: bool = False,
        panel_strategy: Literal["global", "multivariate"] = "global",
    ):
        super().__init__(target_transformer=None, actual_transformer=None, panel_strategy=panel_strategy)
        self.forecasters = forecasters
        self.remainder = remainder
        self.n_jobs = n_jobs
        self.verbose_feature_names_out = verbose_feature_names_out

    @property
    def _forecasters(self) -> list[tuple[str, BaseForecaster]]:
        """Adapter for _BaseComposition._get_params (expects 2-tuples).

        Returns
        -------
        list of (str, BaseForecaster)
            Forecaster tuples with columns stripped.

        """
        try:
            return [(name, forecaster) for name, forecaster, _ in self.forecasters]  # ty: ignore[invalid-assignment]
        except (TypeError, ValueError):
            return self.forecasters  # ty: ignore[invalid-return-type]

    @_forecasters.setter
    def _forecasters(self, value: list[tuple[str, BaseForecaster]]) -> None:
        """Merge new 2-tuples back with original columns.

        Parameters
        ----------
        value : list of (str, BaseForecaster)
            New forecaster 2-tuples from _set_params.

        """
        try:
            self.forecasters = [
                (name, forecaster, col)
                for ((name, forecaster), (_, _, col)) in zip(value, self.forecasters, strict=True)  # ty: ignore[invalid-assignment]
            ]
        except (TypeError, ValueError):
            self.forecasters = value

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Get parameters for this estimator.

        Parameters
        ----------
        deep : bool, default=True
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.

        Returns
        -------
        dict
            Parameter names mapped to their values.

        """
        return self._get_params("_forecasters", deep=deep)

    def set_params(self, **params: Any) -> "ColumnForecaster":
        """Set the parameters of this estimator.

        Valid parameter keys can be listed with ``get_params()``.

        Parameters
        ----------
        **params : dict
            Estimator parameters.

        Returns
        -------
        self
            Estimator instance.

        """
        self._set_params("_forecasters", **params)
        return self

    @property
    def named_forecasters_(self) -> Bunch:
        """Access fitted forecasters by name.

        Returns
        -------
        Bunch
            Dictionary-like object with forecaster names as keys and
            fitted forecaster objects as values.

        """
        check_is_fitted(self, ["forecasters_"])
        return Bunch(**{name: forecaster for name, forecaster, _ in self.forecasters_})

    def _validate_column_assignments(self, all_columns: list[str]) -> tuple[dict[str, list[str]], list[str]]:
        """Validate column assignments and return column mapping.

        Parameters
        ----------
        all_columns : list of str
            All non-time columns in y.

        Returns
        -------
        column_map : dict
            Mapping from forecaster name to list of columns.
        remainder_cols : list of str
            Columns not assigned to any forecaster.

        Raises
        ------
        ValueError
            If names are duplicated, contain ``"__"``, collide with a
            constructor argument, if column assignments overlap, or if
            assigned columns don't exist.

        """
        # Validate unique names (kept first so the duplicate message stays stable)
        names = [name for name, _, _ in self.forecasters]  # ty: ignore[invalid-assignment]
        name_counts = Counter(names)
        duplicates = [name for name, count in name_counts.items() if count > 1]
        if duplicates:
            raise ValueError(f"Duplicate forecaster names: {duplicates}")

        # Reuse the _BaseComposition checks for the remaining naming rules: reject
        # names containing "__" (which would corrupt nested-parameter routing) and
        # names that collide with constructor arguments. Names are already unique
        # here, so its uniqueness check is a no-op.
        self._validate_names(names)

        # Build column map and track assigned columns
        column_map: dict[str, list[str]] = {}
        assigned_columns: list[str] = []

        for name, _, columns in self.forecasters:  # ty: ignore[invalid-assignment]
            cols = [columns] if isinstance(columns, str) else list(columns)

            # Check columns exist
            missing = set(cols) - set(all_columns)
            if missing:
                raise ValueError(f"Forecaster '{name}' references non-existent columns: {missing}")

            # Check for overlaps
            overlaps = set(cols) & set(assigned_columns)
            if overlaps:
                raise ValueError(f"Column(s) {overlaps} assigned to multiple forecasters")

            column_map[name] = cols
            assigned_columns.extend(cols)

        # Remainder columns
        remainder_cols = [col for col in all_columns if col not in assigned_columns]

        return column_map, remainder_cols

    def __sklearn_tags__(self) -> Tags:
        """Get estimator tags.

        Returns
        -------
        Tags
            Estimator tags with yohou-specific attributes.

        """
        tags = super().__sklearn_tags__()
        assert tags.forecaster_tags is not None

        # Meta-forecaster: delegates observation tracking to children
        tags.forecaster_tags.tracks_observations = False

        # Collect all forecasters (unfitted or fitted)
        forecasters_to_check: list[BaseForecaster] = []
        if hasattr(self, "forecasters_"):
            forecasters_to_check = [f for _, f, _ in self.forecasters_]
            if self.remainder_forecaster_ is not None:
                forecasters_to_check.append(self.remainder_forecaster_)
        else:
            forecasters_to_check = [f for _, f, _ in self.forecasters]
            if isinstance(self.remainder, BaseForecaster):
                forecasters_to_check.append(self.remainder)

        if forecasters_to_check:
            # Stateful if any forecaster is stateful
            tags.forecaster_tags.stateful = any(
                getattr(f.__sklearn_tags__().forecaster_tags, "stateful", False) for f in forecasters_to_check
            )

            # Determine forecaster_type from nested forecasters' tags
            all_types: frozenset[str] = frozenset()
            for f in forecasters_to_check:
                f_tags = f.__sklearn_tags__()
                if f_tags.forecaster_tags and f_tags.forecaster_tags.forecaster_type:
                    all_types = all_types | f_tags.forecaster_tags.forecaster_type

            # Aggregate types
            if all_types:
                tags.forecaster_tags.forecaster_type = all_types

            # Aggregate other tags
            tags.forecaster_tags.uses_reduction = any(
                getattr(f.__sklearn_tags__().forecaster_tags, "uses_reduction", False) for f in forecasters_to_check
            )
            tags.forecaster_tags.uses_target_transformer = any(
                getattr(f.__sklearn_tags__().forecaster_tags, "uses_target_transformer", False)
                for f in forecasters_to_check
            )
            tags.forecaster_tags.uses_actual_transformer = any(
                getattr(f.__sklearn_tags__().forecaster_tags, "uses_actual_transformer", False)
                for f in forecasters_to_check
            )
            tags.forecaster_tags.uses_forecast_transformer = any(
                getattr(f.__sklearn_tags__().forecaster_tags, "uses_forecast_transformer", False)
                for f in forecasters_to_check
            )
            tags.forecaster_tags.uses_step_transformer = any(
                getattr(f.__sklearn_tags__().forecaster_tags, "uses_step_transformer", False)
                for f in forecasters_to_check
            )
            tags.forecaster_tags.supports_panel_data = all(
                getattr(f.__sklearn_tags__().forecaster_tags, "supports_panel_data", True) for f in forecasters_to_check
            )

        return tags

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(
        self,
        y: pl.DataFrame,
        X_actual: pl.DataFrame | None = None,
        forecasting_horizon: int = 1,
        X_future: pl.DataFrame | None = None,
        X_forecast: pl.DataFrame | None = None,
        **params,
    ) -> "ColumnForecaster":
        """Fit all forecasters on their respective column subsets.

        Parameters
        ----------
        y : pl.DataFrame
            Target time series with "time" column.
        X_actual : pl.DataFrame or None, default=None
            Actual feature observations with a ``"time"`` column aligned
            with ``y``. Forwarded to each child forecaster.
        forecasting_horizon : int, default=1
            Number of steps ahead to forecast.
        X_future : pl.DataFrame or None, default=None
            Known future features with a ``"time"`` column. Deterministic
            values available for past and future dates. Bypasses the
            actual transformer.
        X_forecast : pl.DataFrame or None, default=None
            External forecasts with ``"vintage_time"`` and ``"time"``
            columns. Bypasses the actual transformer.
        **params : dict
            Metadata to route to nested estimators.

        Returns
        -------
        self

        """
        # Validate params before routing
        _raise_for_params(params, self, "fit")

        # Route metadata to nested forecasters
        routed_params = process_routing(self, "fit", **params)

        # Store fit parameters
        self.fit_forecasting_horizon_ = forecasting_horizon
        self._y_observed = y
        self._X_observed = X_actual

        # Get all non-time columns
        all_columns = [col for col in y.columns if col != "time"]

        # Validate column assignments
        column_map, remainder_cols = self._validate_column_assignments(all_columns)
        self.column_map_ = column_map
        self.remainder_cols_ = remainder_cols

        # Fit forecasters in parallel
        results = Parallel(n_jobs=self.n_jobs)(
            delayed(_fit_one_forecaster)(
                forecaster,
                name,
                y.select(["time"] + cols),
                X_actual,
                forecasting_horizon,
                routed_params.get(name, Bunch(fit={})),
                X_future=X_future,
                X_forecast=X_forecast,
            )
            for name, forecaster, cols in self.forecasters  # ty: ignore[invalid-assignment]
            for cols in [column_map[name]]  # Normalize columns
        )

        # Store fitted forecasters with columns
        self.forecasters_ = [(name, fitted_forecaster, column_map[name]) for name, fitted_forecaster in results]

        # Fit remainder forecaster if remainder is an estimator and there are remainder columns
        if remainder_cols and isinstance(self.remainder, BaseForecaster):
            y_remainder = y.select(["time"] + remainder_cols)
            self.remainder_forecaster_ = clone(self.remainder)
            remainder_params = routed_params.get("remainder", Bunch(fit={}))
            self.remainder_forecaster_.fit(
                y_remainder,
                X_actual,
                forecasting_horizon=forecasting_horizon,
                X_future=X_future,
                X_forecast=X_forecast,
                **remainder_params.fit,
            )
        else:
            self.remainder_forecaster_ = None

        # Set attributes from first forecaster
        first_name, first_forecaster, _ = self.forecasters_[0]
        self.interval_ = first_forecaster.interval_
        self.groups_ = first_forecaster.groups_

        # Combine schemas from all forecasters
        self.local_y_schema_ = {}
        for _name, forecaster, _ in self.forecasters_:
            if hasattr(forecaster, "local_y_schema_"):
                self.local_y_schema_.update(forecaster.local_y_schema_)
        if self.remainder_forecaster_ is not None and hasattr(self.remainder_forecaster_, "local_y_schema_"):
            self.local_y_schema_.update(self.remainder_forecaster_.local_y_schema_)

        # X_actual schemas are merged across all forecasters; since every
        # forecaster receives the same X_actual these are consistent, but
        # merging avoids discarding schema seen by later forecasters.
        all_forecasters = [f for _, f, _ in self.forecasters_]
        if self.remainder_forecaster_ is not None:
            all_forecasters.append(self.remainder_forecaster_)
        self.local_X_actual_schema_ = None
        self.shared_X_actual_schema_ = None
        for forecaster in all_forecasters:
            local_schema = getattr(forecaster, "local_X_actual_schema_", None)
            if local_schema is not None:
                self.local_X_actual_schema_ = {**(self.local_X_actual_schema_ or {}), **local_schema}
            shared_schema = getattr(forecaster, "shared_X_actual_schema_", None)
            if shared_schema is not None:
                self.shared_X_actual_schema_ = {**(self.shared_X_actual_schema_ or {}), **shared_schema}

        # Set transformed schema attributes (no transformation for meta-forecaster)
        self.local_y_t_schema_ = self.local_y_schema_
        self.local_X_t_schema_ = self.local_X_actual_schema_
        self._X_t_observed = X_actual

        return self

    def _collect_prediction(
        self,
        name: str | None,
        y_pred: pl.DataFrame,
        predictions: list[pl.DataFrame],
        time_columns: pl.DataFrame | None,
        time_col_names: list[str],
    ) -> tuple[pl.DataFrame, list[str]]:
        """Split time columns from features and append features to ``predictions``.

        On the first prediction, the present time columns (``"vintage_time"``
        and/or ``"time"``) are detected and captured. Feature columns are
        prefixed with ``name`` when ``verbose_feature_names_out`` is True,
        mirroring `ColumnTransformer`'s ``{name}__{col}`` naming.

        Parameters
        ----------
        name : str or None
            Forecaster name used as the prefix, or ``None`` for the remainder
            forecaster (which is never prefixed).
        y_pred : pl.DataFrame
            Prediction from a single child forecaster.
        predictions : list of pl.DataFrame
            Accumulator of feature-only predictions, appended to in place.
        time_columns : pl.DataFrame or None
            Time columns captured from the first prediction, or ``None`` if no
            prediction has been seen yet.
        time_col_names : list of str
            Names of the captured time columns.

        Returns
        -------
        tuple of (pl.DataFrame, list of str)
            The (possibly newly captured) ``time_columns`` and ``time_col_names``.

        """
        if time_columns is None:
            time_col_names = [c for c in ["vintage_time", "time"] if c in y_pred.columns]
            time_columns = y_pred.select(time_col_names)

        features = y_pred.select(~cs.by_name(*time_col_names))
        if self.verbose_feature_names_out and name is not None:
            features = features.rename({col: f"{name}__{col}" for col in features.columns})
        predictions.append(features)

        return time_columns, time_col_names

    @available_if(_column_forecaster_has("predict"))
    def predict(
        self,
        forecasting_horizon: int | None = None,
        groups: list[str] | None = None,
        predict_transformed: bool = False,
        X_future: pl.DataFrame | None = None,
        X_forecast: pl.DataFrame | None = None,
        **params,
    ) -> pl.DataFrame:
        """Predict using all forecasters and concatenate results.

        Parameters
        ----------
        forecasting_horizon : int, optional
            Forecasting horizon. If None, uses horizon from fit.
        groups : list of str or None, default=None
            Group prefixes for panel data.
        predict_transformed : bool, default=False
            Return transformed predictions.
        X_future : pl.DataFrame or None, default=None
            Known future features override. Re-derives step columns
            without mutating forecaster state.
        X_forecast : pl.DataFrame or None, default=None
            External forecast override with ``"vintage_time"`` and
            ``"time"`` columns. Re-derives step columns without mutating
            forecaster state.
        **params : dict
            Metadata to route to nested estimators.

        Returns
        -------
        pl.DataFrame
            Concatenated predictions from all forecasters.

        """
        check_is_fitted(self, ["forecasters_", "column_map_", "remainder_cols_"])

        # Validate params before routing
        _raise_for_params(params, self, "predict")

        # Route metadata to nested forecasters
        routed_params = process_routing(self, "predict", **params)

        predictions: list[pl.DataFrame] = []
        time_columns: pl.DataFrame | None = None
        time_col_names: list[str] = []

        # Predict with each forecaster
        for name, forecaster, _cols in self.forecasters_:
            forecaster_params = routed_params.get(name, Bunch(predict={}))
            y_pred = forecaster.predict(
                forecasting_horizon=forecasting_horizon,
                groups=groups,
                predict_transformed=predict_transformed,
                X_future=X_future,
                X_forecast=X_forecast,
                **forecaster_params.predict,
            )

            time_columns, time_col_names = self._collect_prediction(
                name, y_pred, predictions, time_columns, time_col_names
            )

        # Predict with remainder forecaster if present
        if self.remainder_forecaster_ is not None and self.remainder_cols_:
            remainder_params = routed_params.get("remainder", Bunch(predict={}))
            y_pred_remainder = self.remainder_forecaster_.predict(
                forecasting_horizon=forecasting_horizon,
                groups=groups,
                predict_transformed=predict_transformed,
                X_future=X_future,
                X_forecast=X_forecast,
                **remainder_params.predict,
            )
            time_columns, time_col_names = self._collect_prediction(
                None, y_pred_remainder, predictions, time_columns, time_col_names
            )

        if time_columns is None:  # pragma: no cover - defensive: a fitted ColumnForecaster always produces output
            raise ValueError("ColumnForecaster has no fitted forecasters to predict from.")
        result = pl.concat([time_columns] + predictions, how="horizontal")

        return result

    def observe(
        self,
        y: pl.DataFrame,
        X_actual: pl.DataFrame | None = None,
        groups: list[str] | None = None,
        X_future: pl.DataFrame | None = None,
        X_forecast: pl.DataFrame | None = None,
    ) -> "ColumnForecaster":
        """Observe all forecasters with new observations.

        Parameters
        ----------
        y : pl.DataFrame
            New target data with "time" column.
        X_actual : pl.DataFrame or None, default=None
            New actual feature observations with a ``"time"`` column
            aligned with ``y``. Forwarded to each child forecaster.
        groups : list of str or None, default=None
            Group prefixes for panel data.
        X_future : pl.DataFrame or None, default=None
            Known future features with a ``"time"`` column.
        X_forecast : pl.DataFrame or None, default=None
            External forecasts with ``"vintage_time"`` and ``"time"``
            columns.

        Returns
        -------
        self

        """
        check_is_fitted(self, ["forecasters_", "column_map_", "remainder_cols_"])

        # Observe each forecaster with its column subset
        for _name, forecaster, cols in self.forecasters_:
            y_subset = y.select(["time"] + cols)
            forecaster.observe(y_subset, X_actual, groups, X_future=X_future, X_forecast=X_forecast)

        # Observe remainder forecaster if present
        if self.remainder_forecaster_ is not None and self.remainder_cols_:
            y_remainder = y.select(["time"] + self.remainder_cols_)
            self.remainder_forecaster_.observe(y_remainder, X_actual, groups, X_future=X_future, X_forecast=X_forecast)

        # Bookkeeping only: this meta-forecaster delegates bounded buffering
        # to its child forecasters, so _y_observed/_X_observed are not used for
        # prediction. Hold only the latest batch (as rewind does) to keep memory
        # bounded on long-running streams instead of accumulating every batch.
        self._y_observed = y
        self._X_observed = X_actual

        return self

    def rewind(
        self,
        y: pl.DataFrame,
        X_actual: pl.DataFrame | None = None,
        groups: list[str] | None = None,
        X_future: pl.DataFrame | None = None,
        X_forecast: pl.DataFrame | None = None,
    ) -> "ColumnForecaster":
        """Rewind all forecasters to new observation window.

        Parameters
        ----------
        y : pl.DataFrame
            New target data with "time" column.
        X_actual : pl.DataFrame or None, default=None
            Actual feature observations to restore the observation
            state to. Must align with ``y``.
        groups : list of str or None, default=None
            Group prefixes for panel data.
        X_future : pl.DataFrame or None, default=None
            Known future features with a ``"time"`` column.
        X_forecast : pl.DataFrame or None, default=None
            External forecasts with ``"vintage_time"`` and ``"time"``
            columns.

        Returns
        -------
        self

        """
        check_is_fitted(self, ["forecasters_", "column_map_", "remainder_cols_"])

        # Rewind each forecaster with its column subset
        for _name, forecaster, cols in self.forecasters_:
            y_subset = y.select(["time"] + cols)
            forecaster.rewind(y_subset, X_actual, groups, X_future=X_future, X_forecast=X_forecast)

        # Rewind remainder forecaster if present
        if self.remainder_forecaster_ is not None and self.remainder_cols_:
            y_remainder = y.select(["time"] + self.remainder_cols_)
            self.remainder_forecaster_.rewind(y_remainder, X_actual, groups, X_future=X_future, X_forecast=X_forecast)

        self._y_observed = y
        self._X_observed = X_actual

        return self

    @available_if(_column_forecaster_has("predict"))
    def observe_predict(
        self,
        y: pl.DataFrame,
        X_actual: pl.DataFrame | None = None,
        forecasting_horizon: int | None = None,
        groups: list[str] | None = None,
        stride: int | None = None,
        predict_transformed: bool = False,
        X_future: pl.DataFrame | None = None,
        X_forecast: pl.DataFrame | None = None,
        **params,
    ) -> pl.DataFrame:
        """Alternate recursive predict and observe.

        Parameters
        ----------
        y : pl.DataFrame
            Target time series for updates.
        X_actual : pl.DataFrame or None, default=None
            Actual feature observations with a ``"time"`` column aligned
            with ``y``. Sliced and observed incrementally at each step
            of the rolling loop.
        forecasting_horizon : int or None, default=None
            Horizon to forecast. If None, uses ``fit_forecasting_horizon_``.
        groups : list of str or None, default=None
            Group prefixes for panel data.
        stride : int or None, default=None
            Number of observations per update step. If None, uses
            ``fit_forecasting_horizon_``.
        predict_transformed : bool, default=False
            If ``True``, return predictions in transformed space.
        X_future : pl.DataFrame or None, default=None
            Known future features.
        X_forecast : pl.DataFrame or None, default=None
            External forecasts.
        **params : dict
            Metadata to route to nested estimators.

        Returns
        -------
        pl.DataFrame
            Predicted time series.

        """
        check_is_fitted(self, ["forecasters_", "column_map_", "remainder_cols_"])

        fh = forecasting_horizon if forecasting_horizon is not None else self.fit_forecasting_horizon_
        if stride is None:
            stride = self.fit_forecasting_horizon_

        return self._observe_predict_loop(
            predict_fn=self.predict,
            y=y,
            X_actual=X_actual,
            X_future=X_future,
            X_forecast=X_forecast,
            groups=groups,
            stride=stride,
            observe_fn=self.observe,
            forecasting_horizon=fh,
            predict_transformed=predict_transformed,
            **params,
        )

    @available_if(_column_forecaster_has("predict_interval"))
    def predict_interval(
        self,
        forecasting_horizon: int | None = None,
        coverage_rates: list[float] | None = None,
        groups: list[str] | None = None,
        predict_transformed: bool = False,
        X_future: pl.DataFrame | None = None,
        X_forecast: pl.DataFrame | None = None,
        **params,
    ) -> pl.DataFrame:
        """Predict intervals using all forecasters and concatenate results.

        Parameters
        ----------
        forecasting_horizon : int, optional
            Forecasting horizon. If None, uses horizon from fit.
        coverage_rates : list of float, optional
            Coverage rates for prediction intervals.
        groups : list of str or None, default=None
            Group prefixes for panel data.
        predict_transformed : bool, default=False
            Accepted for signature symmetry with ``predict`` and not
            forwarded: interval children emit original-scale bounds by
            design and their ``predict_interval`` takes no such flag.
        X_future : pl.DataFrame or None, default=None
            Known future features override. Re-derives step columns
            without mutating forecaster state.
        X_forecast : pl.DataFrame or None, default=None
            External forecast override with ``"vintage_time"`` and
            ``"time"`` columns. Re-derives step columns without mutating
            forecaster state.
        **params : dict
            Metadata to route to nested estimators.

        Returns
        -------
        pl.DataFrame
            Concatenated interval predictions from all forecasters.

        """
        check_is_fitted(self, ["forecasters_", "column_map_", "remainder_cols_"])

        # Validate params before routing
        _raise_for_params(params, self, "predict_interval")

        # Route metadata to nested forecasters
        routed_params = process_routing(self, "predict_interval", **params)

        predictions: list[pl.DataFrame] = []
        time_columns: pl.DataFrame | None = None

        # Determine which time columns are present (predict_interval may not have vintage_time)
        time_col_names: list[str] = []

        # Predict with each forecaster
        for name, forecaster, _cols in self.forecasters_:
            forecaster_params = routed_params.get(name, Bunch(predict_interval={}))
            # predict_transformed is not forwarded: no interval child accepts
            # it on predict_interval (bounds are emitted on the original scale
            # by design), so the old forwarding landed in **params and was
            # silently swallowed. Kept out rather than surfaced as an
            # unrequested-metadata error now that children route their params.
            y_pred = forecaster.predict_interval(
                forecasting_horizon=forecasting_horizon,
                coverage_rates=coverage_rates,
                groups=groups,
                X_future=X_future,
                X_forecast=X_forecast,
                **forecaster_params.predict_interval,
            )

            time_columns, time_col_names = self._collect_prediction(
                name, y_pred, predictions, time_columns, time_col_names
            )

        # Predict with remainder forecaster if present
        if self.remainder_forecaster_ is not None and self.remainder_cols_:
            remainder_params = routed_params.get("remainder", Bunch(predict_interval={}))
            y_pred_remainder = self.remainder_forecaster_.predict_interval(
                forecasting_horizon=forecasting_horizon,
                coverage_rates=coverage_rates,
                groups=groups,
                X_future=X_future,
                X_forecast=X_forecast,
                **remainder_params.predict_interval,
            )
            time_columns, time_col_names = self._collect_prediction(
                None, y_pred_remainder, predictions, time_columns, time_col_names
            )

        if time_columns is None:  # pragma: no cover - defensive: a fitted ColumnForecaster always produces output
            raise ValueError("ColumnForecaster has no fitted forecasters to predict from.")
        result = pl.concat([time_columns] + predictions, how="horizontal")

        return result

    @available_if(_column_forecaster_has("predict_interval"))
    def observe_predict_interval(
        self,
        y: pl.DataFrame,
        X_actual: pl.DataFrame | None = None,
        forecasting_horizon: int | None = None,
        coverage_rates: list[float] | None = None,
        groups: list[str] | None = None,
        stride: int | None = None,
        predict_transformed: bool = False,
        X_future: pl.DataFrame | None = None,
        X_forecast: pl.DataFrame | None = None,
        **params,
    ) -> pl.DataFrame:
        """Alternate recursive predict_interval and observe.

        Parameters
        ----------
        y : pl.DataFrame
            Target time series for updates.
        X_actual : pl.DataFrame or None, default=None
            Actual feature observations with a ``"time"`` column aligned
            with ``y``. Sliced and observed incrementally at each step
            of the rolling loop.
        forecasting_horizon : int or None, default=None
            Horizon to forecast. If None, uses ``fit_forecasting_horizon_``.
        coverage_rates : list of float or None, default=None
            Coverage rates for prediction intervals.
        groups : list of str or None, default=None
            Group prefixes for panel data.
        stride : int or None, default=None
            Number of observations per update step. If None, uses
            ``fit_forecasting_horizon_``.
        predict_transformed : bool, default=False
            If ``True``, return predictions in transformed space.
        X_future : pl.DataFrame or None, default=None
            Known future features.
        X_forecast : pl.DataFrame or None, default=None
            External forecasts.
        **params : dict
            Metadata to route to nested estimators.

        Returns
        -------
        pl.DataFrame
            Predicted interval time series.

        """
        check_is_fitted(self, ["forecasters_", "column_map_", "remainder_cols_"])

        fh = forecasting_horizon if forecasting_horizon is not None else self.fit_forecasting_horizon_
        if stride is None:
            stride = self.fit_forecasting_horizon_

        return self._observe_predict_loop(
            predict_fn=self.predict_interval,
            y=y,
            X_actual=X_actual,
            X_future=X_future,
            X_forecast=X_forecast,
            groups=groups,
            stride=stride,
            observe_fn=self.observe,
            forecasting_horizon=fh,
            coverage_rates=coverage_rates,
            predict_transformed=predict_transformed,
            **params,
        )

    @available_if(_column_forecaster_has("predict_class_proba"))
    def predict_class_proba(
        self,
        forecasting_horizon: int | None = None,
        groups: list[str] | None = None,
        predict_transformed: bool = False,
        X_future: pl.DataFrame | None = None,
        X_forecast: pl.DataFrame | None = None,
        **params,
    ) -> pl.DataFrame:
        """Predict class probabilities using all forecasters and concatenate results.

        Parameters
        ----------
        forecasting_horizon : int, optional
            Forecasting horizon. If None, uses horizon from fit.
        groups : list of str or None, default=None
            Group prefixes for panel data.
        predict_transformed : bool, default=False
            Return transformed predictions. Forwarded to each child
            forecaster for parity with ``predict``/``predict_interval``.
        X_future : pl.DataFrame or None, default=None
            Known future features override. Re-derives step columns
            without mutating forecaster state.
        X_forecast : pl.DataFrame or None, default=None
            External forecast override with ``"vintage_time"`` and
            ``"time"`` columns. Re-derives step columns without mutating
            forecaster state.
        **params : dict
            Metadata to route to nested estimators.

        Returns
        -------
        pl.DataFrame
            Concatenated class-probability predictions from all forecasters.

        """
        check_is_fitted(self, ["forecasters_", "column_map_", "remainder_cols_"])

        _raise_for_params(params, self, "predict_class_proba")
        routed_params = process_routing(self, "predict_class_proba", **params)

        predictions: list[pl.DataFrame] = []
        time_columns: pl.DataFrame | None = None
        time_col_names: list[str] = []

        for name, forecaster, _cols in self.forecasters_:
            forecaster_params = routed_params.get(name, Bunch(predict_class_proba={}))
            y_pred = forecaster.predict_class_proba(
                forecasting_horizon=forecasting_horizon,
                groups=groups,
                predict_transformed=predict_transformed,
                X_future=X_future,
                X_forecast=X_forecast,
                **forecaster_params.predict_class_proba,
            )

            time_columns, time_col_names = self._collect_prediction(
                name, y_pred, predictions, time_columns, time_col_names
            )

        if self.remainder_forecaster_ is not None and self.remainder_cols_:
            remainder_params = routed_params.get("remainder", Bunch(predict_class_proba={}))
            y_pred_remainder = self.remainder_forecaster_.predict_class_proba(
                forecasting_horizon=forecasting_horizon,
                groups=groups,
                predict_transformed=predict_transformed,
                X_future=X_future,
                X_forecast=X_forecast,
                **remainder_params.predict_class_proba,
            )
            time_columns, time_col_names = self._collect_prediction(
                None, y_pred_remainder, predictions, time_columns, time_col_names
            )

        if time_columns is None:  # pragma: no cover - defensive: a fitted ColumnForecaster always produces output
            raise ValueError("ColumnForecaster has no fitted forecasters to predict from.")
        return pl.concat([time_columns] + predictions, how="horizontal")

    @available_if(_column_forecaster_has("predict_class_proba"))
    def observe_predict_class_proba(
        self,
        y: pl.DataFrame,
        X_actual: pl.DataFrame | None = None,
        forecasting_horizon: int | None = None,
        groups: list[str] | None = None,
        stride: int | None = None,
        predict_transformed: bool = False,
        X_future: pl.DataFrame | None = None,
        X_forecast: pl.DataFrame | None = None,
        **params,
    ) -> pl.DataFrame:
        """Alternate recursive predict_class_proba and observe.

        Parameters
        ----------
        y : pl.DataFrame
            Target time series for updates.
        X_actual : pl.DataFrame or None, default=None
            Actual feature observations with a ``"time"`` column aligned
            with ``y``. Sliced and observed incrementally at each step
            of the rolling loop.
        forecasting_horizon : int or None, default=None
            Horizon to forecast. If None, uses ``fit_forecasting_horizon_``.
        groups : list of str or None, default=None
            Group prefixes for panel data.
        stride : int or None, default=None
            Number of observations per update step. If None, uses
            ``fit_forecasting_horizon_``.
        predict_transformed : bool, default=False
            If ``True``, return predictions in transformed space. Forwarded
            to each child for parity with ``observe_predict``/
            ``observe_predict_interval``.
        X_future : pl.DataFrame or None, default=None
            Known future features.
        X_forecast : pl.DataFrame or None, default=None
            External forecasts.
        **params : dict
            Metadata to route to nested estimators.

        Returns
        -------
        pl.DataFrame
            Predicted class-probability time series.

        """
        check_is_fitted(self, ["forecasters_", "column_map_", "remainder_cols_"])

        fh = forecasting_horizon if forecasting_horizon is not None else self.fit_forecasting_horizon_
        if stride is None:
            stride = self.fit_forecasting_horizon_

        return self._observe_predict_loop(
            predict_fn=self.predict_class_proba,
            y=y,
            X_actual=X_actual,
            X_future=X_future,
            X_forecast=X_forecast,
            groups=groups,
            stride=stride,
            observe_fn=self.observe,
            forecasting_horizon=fh,
            predict_transformed=predict_transformed,
            **params,
        )

    def get_metadata_routing(self):
        """Get metadata routing for this estimator.

        Returns
        -------
        MetadataRouter
            Metadata routing configuration.

        """
        # Build on the inherited router rather than from scratch: the parent
        # carries this class's own $self_request and the transformer children
        # (target/actual/forecast), which a from-scratch MetadataRouter drops,
        # making the class's own request keys invisible when nested.
        router = super().get_metadata_routing()

        # Create method mapping for forecasters
        method_mapping = (
            MethodMapping()
            .add(caller="fit", callee="fit")
            .add(caller="predict", callee="predict")
            .add(caller="predict_interval", callee="predict_interval")
            .add(caller="predict_class_proba", callee="predict_class_proba")
            .add(caller="observe_predict", callee="observe_predict")
            .add(caller="observe_predict_interval", callee="observe_predict_interval")
            .add(caller="observe_predict_class_proba", callee="observe_predict_class_proba")
        )

        # Add routing for each named forecaster
        for name, forecaster, _ in self.forecasters:  # ty: ignore[invalid-assignment]
            router.add(**{name: forecaster}, method_mapping=method_mapping)

        # Add routing for remainder forecaster (only if it's a forecaster)
        if isinstance(self.remainder, BaseForecaster):
            router.add(remainder=self.remainder, method_mapping=method_mapping)

        return router
