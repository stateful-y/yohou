"""Base mixin for ensemble forecasters."""

from __future__ import annotations

import warnings
from collections import Counter
from typing import Any

import numpy as np
import polars as pl
from sklearn.base import clone
from sklearn.utils import Bunch
from sklearn.utils.parallel import Parallel, delayed
from sklearn.utils.validation import check_is_fitted

from yohou.base import BaseForecaster

__all__: list[str] = []


def _fit_one_forecaster(
    forecaster: BaseForecaster,
    name: str,
    y: pl.DataFrame,
    X_actual: pl.DataFrame | None,
    forecasting_horizon: int,
    fit_params: dict[str, Any],
    extra_fit_kwargs: dict[str, Any] | None = None,
    X_future: pl.DataFrame | None = None,
    X_forecast: pl.DataFrame | None = None,
) -> tuple[str, BaseForecaster | None, str | None]:
    """Clone and fit a single base forecaster.

    Parameters
    ----------
    forecaster : BaseForecaster
        Forecaster to clone and fit.
    name : str
        Name of the forecaster.
    y : pl.DataFrame
        Target time series.
    X_actual : pl.DataFrame or None
        Exogenous features.
    forecasting_horizon : int
        Forecasting horizon.
    fit_params : dict
        Routed fit parameters for this forecaster.
    extra_fit_kwargs : dict or None
        Additional keyword arguments to pass to the child's ``fit``
        method (e.g. ``coverage_rates``).
    X_future : pl.DataFrame or None, default=None
        Known future features.
    X_forecast : pl.DataFrame or None, default=None
        External forecasts.

    Returns
    -------
    tuple of (str, BaseForecaster or None, str or None)
        Name, fitted forecaster (or None on failure), and error message
        (or None on success).

    """
    try:
        forecaster_clone = clone(forecaster)
        all_params = {**fit_params}
        if extra_fit_kwargs:
            all_params.update(extra_fit_kwargs)
        forecaster_clone.fit(
            y,
            X_actual,
            forecasting_horizon=forecasting_horizon,
            X_future=X_future,
            X_forecast=X_forecast,
            **all_params,
        )
        return name, forecaster_clone, None
    except Exception as exc:  # noqa: BLE001
        return name, None, f"{type(exc).__name__}: {exc}"


def _ensemble_has(method_name: str):
    """Check if all surviving base forecasters have a given method.

    Parameters
    ----------
    method_name : str
        Name of the method to check for.

    Returns
    -------
    callable
        A check function for ``available_if`` decorator.

    """

    def check(self):
        """Check if all fitted forecasters have the required method."""
        if hasattr(self, "forecasters_"):
            return all(hasattr(f, method_name) for _, f in self.forecasters_)
        return all(hasattr(f, method_name) for _, f in self.forecasters)

    return check


class _BaseEnsembleForecaster:
    """Mixin providing shared logic for ensemble forecasters.

    Provides parallel fitting, observe/rewind delegation, named access,
    and ``_BaseComposition``-compatible property adapters for ensemble
    forecasters that combine predictions from multiple base forecasters.

    """

    # Stubs for attributes provided by concrete subclasses.
    forecasters_: list[tuple[str, BaseForecaster]]
    weights: list[float] | None

    # Family the ensemble's children must belong to. Concrete voting
    # subclasses tighten this to their own base (e.g. BaseIntervalForecaster)
    # so a wrong-family child fails at fit() rather than deep inside predict.
    _required_forecaster_type: type = BaseForecaster

    def _validate_forecasters_list(self) -> None:
        """Validate the forecasters parameter.

        Raises
        ------
        ValueError
            If names are not unique, tuples are malformed, names are not
            strings, names contain ``"__"``, forecasters are not
            ``BaseForecaster`` instances, or a forecaster does not belong to
            the ensemble's required family (``_required_forecaster_type``).

        """
        names = []
        for item in self.forecasters:
            if not isinstance(item, tuple) or len(item) != 2:
                raise ValueError(f"Each entry in `forecasters` must be a (name, forecaster) tuple, got {item!r}")
            name, forecaster = item
            if not isinstance(name, str):
                raise ValueError(f"Forecaster name must be a string, got {type(name).__name__}")
            if not isinstance(forecaster, BaseForecaster):
                raise ValueError(
                    f"Forecaster '{name}' must be a BaseForecaster instance, got {type(forecaster).__name__}"
                )
            if not isinstance(forecaster, self._required_forecaster_type):
                raise ValueError(
                    f"Forecaster '{name}' must be a {self._required_forecaster_type.__name__} instance "
                    f"for this ensemble, got {type(forecaster).__name__}."
                )
            if "__" in name:
                raise ValueError(
                    f"Forecaster name '{name}' must not contain '__': the double underscore is "
                    f"reserved for nested-parameter routing (e.g. set_params('{name}__param'))."
                )
            names.append(name)

        counts = Counter(names)
        duplicates = [n for n, c in counts.items() if c > 1]
        if duplicates:
            raise ValueError(f"Duplicate forecaster names: {duplicates}")

    def _fit_forecasters_parallel(
        self,
        y: pl.DataFrame,
        X_actual: pl.DataFrame | None,
        forecasting_horizon: int,
        routed_params: Any,
        n_jobs: int | None,
        extra_fit_kwargs: dict[str, Any] | None = None,
        X_future: pl.DataFrame | None = None,
        X_forecast: pl.DataFrame | None = None,
    ) -> list[tuple[str, BaseForecaster]]:
        """Clone and fit all base forecasters in parallel.

        Failed forecasters are skipped with a warning. Raises if all fail.

        Parameters
        ----------
        y : pl.DataFrame
            Target time series.
        X_actual : pl.DataFrame or None
            Exogenous features.
        forecasting_horizon : int
            Forecasting horizon.
        routed_params : Bunch
            Routed metadata parameters.
        n_jobs : int or None
            Number of parallel jobs.
        extra_fit_kwargs : dict or None
            Additional keyword arguments to pass to each child's ``fit``
            method (e.g. ``coverage_rates``).
        X_future : pl.DataFrame or None, default=None
            Known future features.
        X_forecast : pl.DataFrame or None, default=None
            External forecasts.

        Returns
        -------
        list of (str, BaseForecaster)
            Successfully fitted (name, forecaster) pairs.

        Raises
        ------
        RuntimeError
            If all base forecasters fail during fitting.

        """
        results = Parallel(n_jobs=n_jobs)(
            delayed(_fit_one_forecaster)(
                forecaster,
                name,
                y,
                X_actual,
                forecasting_horizon,
                getattr(routed_params.get(name, Bunch(fit={})), "fit", {}),
                extra_fit_kwargs,
                X_future,
                X_forecast,
            )
            for name, forecaster in self.forecasters
        )

        fitted: list[tuple[str, BaseForecaster]] = []
        for name, fitted_forecaster, error_msg in results:
            if fitted_forecaster is None:
                warnings.warn(
                    f"Forecaster '{name}' failed during fit and will be skipped: {error_msg}",
                    UserWarning,
                    stacklevel=2,
                )
            else:
                fitted.append((name, fitted_forecaster))

        if not fitted:
            raise RuntimeError("All base forecasters failed during fit. Cannot create ensemble.")

        return fitted

    @property
    def named_forecasters_(self) -> Bunch:
        """Access fitted forecasters by name.

        Returns
        -------
        Bunch
            Dictionary-like object with forecaster names as keys.

        """
        check_is_fitted(self, ["forecasters_"])
        return Bunch(**dict(self.forecasters_))

    @property
    def _forecasters(self) -> list[tuple[str, BaseForecaster]]:
        """Adapter for _BaseComposition._get_params.

        Returns
        -------
        list of (str, BaseForecaster)
            Named forecaster tuples.

        """
        return list(self.forecasters)

    @_forecasters.setter
    def _forecasters(self, value: list[tuple[str, BaseForecaster]]) -> None:
        """Set forecasters from _BaseComposition._set_params.

        Parameters
        ----------
        value : list of (str, BaseForecaster)
            New forecaster tuples.

        """
        self.forecasters = value

    def _validate_schemas_match(self) -> None:
        """Verify all surviving fitted forecasters predict the same columns.

        Raises
        ------
        ValueError
            If target column names or dtypes differ across forecasters.

        """
        schemas = {}
        for name, forecaster in self.forecasters_:
            schema = dict(forecaster.local_y_schema_)
            schemas[name] = schema

        reference_name, reference_schema = next(iter(schemas.items()))
        for name, schema in schemas.items():
            if set(schema.keys()) != set(reference_schema.keys()):
                raise ValueError(
                    f"Forecaster '{name}' predicts columns {set(schema.keys())} "
                    f"but '{reference_name}' predicts {set(reference_schema.keys())}. "
                    f"All base forecasters must predict the same target columns."
                )
            if schema != reference_schema:
                mismatched = {
                    col: (schema[col], reference_schema[col]) for col in schema if schema[col] != reference_schema[col]
                }
                raise ValueError(
                    f"Forecaster '{name}' predicts column dtypes that differ from "
                    f"'{reference_name}': {mismatched} (column: (this, reference)). "
                    f"All base forecasters must predict the same target dtypes."
                )

    def _validate_transformed_schemas_match(self) -> None:
        """Verify children agree on the transformed-space target schema.

        Only relevant for ``predict_transformed=True``: the children's
        transformed-space predictions are combined directly, so they must
        share ``local_y_t_schema_``. Children that legitimately differ here
        (e.g. different target transformers) are fine for the default
        inverse-transformed path, so this is checked lazily rather than at fit.

        Raises
        ------
        ValueError
            If any surviving forecaster's ``local_y_t_schema_`` differs.

        """
        t_schemas = {}
        for name, forecaster in self.forecasters_:
            t_schema = getattr(forecaster, "local_y_t_schema_", None)
            if t_schema is not None:
                t_schemas[name] = dict(t_schema)
        if not t_schemas:
            return
        ref_name, ref_schema = next(iter(t_schemas.items()))
        for name, t_schema in t_schemas.items():
            if t_schema != ref_schema:
                raise ValueError(
                    f"predict_transformed=True requires all base forecasters to share the "
                    f"transformed-space target schema, but '{name}' has {t_schema} while "
                    f"'{ref_name}' has {ref_schema}. Use predict_transformed=False, or give the "
                    f"forecasters matching target transformers."
                )

    def _derive_fitted_attributes(
        self,
        first_forecaster: BaseForecaster,
        forecasting_horizon: int,
        y: pl.DataFrame,
        X_actual: pl.DataFrame | None,
    ) -> None:
        """Copy fitted attributes from the first surviving child forecaster.

        Parameters
        ----------
        first_forecaster : BaseForecaster
            First surviving fitted child forecaster.
        forecasting_horizon : int
            Forecasting horizon used during fitting.
        y : pl.DataFrame
            Training target data.
        X_actual : pl.DataFrame or None
            Training exogenous data.

        """
        self.fit_forecasting_horizon_ = forecasting_horizon
        self.interval_ = first_forecaster.interval_
        self.groups_ = first_forecaster.groups_
        self.local_y_schema_ = dict(first_forecaster.local_y_schema_)
        self.local_X_actual_schema_ = getattr(first_forecaster, "local_X_actual_schema_", None)
        self.shared_X_actual_schema_ = getattr(first_forecaster, "shared_X_actual_schema_", None)
        self.local_y_t_schema_ = getattr(first_forecaster, "local_y_t_schema_", self.local_y_schema_)
        # The transformed feature schema/buffer must reflect the post-pipeline
        # space, which only the child knows; the raw X_actual is not a valid
        # stand-in when a child wraps a feature transformer. Mirror the child's
        # transformed state so the contract attributes are accurate. The
        # ensemble itself never reads these on any predict path (predict,
        # observe, and rewind all delegate to the children), so they exist
        # solely to satisfy the BaseForecaster fitted-attribute contract.
        self.local_X_t_schema_ = getattr(first_forecaster, "local_X_t_schema_", self.local_X_actual_schema_)
        self._y_observed = y
        self._X_observed = X_actual
        self._X_t_observed = getattr(first_forecaster, "_X_t_observed", X_actual)

    def _compute_effective_weights(self) -> None:
        """Compute effective weights for surviving forecasters.

        Maps original weights to only the forecasters that survived
        fitting. Sets ``self.weights_`` to the filtered weights list
        or ``None``.

        """
        if self.weights is not None:
            fitted_names = {name for name, _ in self.forecasters_}
            # self.forecasters and self.weights always have equal length here
            # (guaranteed by pre-fit validation); strict=True asserts that.
            self.weights_ = [
                w for (name, _), w in zip(self.forecasters, self.weights, strict=True) if name in fitted_names
            ]
        else:
            self.weights_ = None

    @staticmethod
    def _aggregate_values(
        predictions: list[pl.DataFrame],
        target_cols: list[str],
        method: str,
        weights: list[float] | None,
    ) -> pl.DataFrame:
        """Aggregate target columns from multiple predictions.

        Parameters
        ----------
        predictions : list of pl.DataFrame
            Predictions from each base forecaster.
        target_cols : list of str
            Column names to aggregate.
        method : str
            Aggregation method: ``"mean"`` or ``"median"``.
        weights : list of float or None
            Per-forecaster weights for weighted averaging.

        Returns
        -------
        pl.DataFrame
            Aggregated predictions with time columns.

        """
        time_cols = [c for c in ("vintage_time", "time") if c in predictions[0].columns]
        time_df = predictions[0].select(time_cols)

        agg_exprs = []
        for col in target_cols:
            values = np.column_stack([pred[col].to_numpy() for pred in predictions])

            if method == "mean":
                if weights is not None:
                    aggregated = np.average(values, axis=1, weights=weights)
                else:
                    aggregated = np.mean(values, axis=1)
            else:
                aggregated = np.median(values, axis=1)

            agg_exprs.append(pl.Series(name=col, values=aggregated))

        return time_df.with_columns(agg_exprs)

    @staticmethod
    def _aggregate_interval_values(
        predictions: list[pl.DataFrame],
        interval_cols: list[str],
        strategy: str,
        weights: list[float] | None,
    ) -> pl.DataFrame:
        """Aggregate interval columns using the configured strategy.

        Parameters
        ----------
        predictions : list of pl.DataFrame
            Interval predictions from each base forecaster.
        interval_cols : list of str
            Interval column names (lower/upper bounds).
        strategy : str
            Aggregation strategy: ``"mean"``, ``"median"``, or
            ``"envelope"``.
        weights : list of float or None
            Per-forecaster weights for weighted averaging. Only used when
            ``strategy="mean"``; silently ignored for ``"median"`` and
            ``"envelope"``.

        Returns
        -------
        pl.DataFrame
            Aggregated interval predictions.

        Notes
        -----
        For the ``"envelope"`` strategy, columns whose names contain neither
        ``"_lower_"`` nor ``"_upper_"`` fall back to mean aggregation.

        """
        time_cols = [c for c in ("vintage_time", "time") if c in predictions[0].columns]
        time_df = predictions[0].select(time_cols)

        agg_exprs = []
        for col in interval_cols:
            values = np.column_stack([pred[col].to_numpy() for pred in predictions])

            if strategy == "envelope":
                if "_lower_" in col:
                    aggregated = np.min(values, axis=1)
                elif "_upper_" in col:
                    aggregated = np.max(values, axis=1)
                else:
                    aggregated = np.mean(values, axis=1)
            elif strategy == "median":
                aggregated = np.median(values, axis=1)
            elif strategy == "mean":
                if weights is not None:
                    aggregated = np.average(values, axis=1, weights=weights)
                else:
                    aggregated = np.mean(values, axis=1)
            else:
                raise ValueError(f"Unknown interval aggregation strategy {strategy!r}")

            agg_exprs.append(pl.Series(name=col, values=aggregated))

        return time_df.with_columns(agg_exprs)

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Get parameters for this estimator.

        Parameters
        ----------
        deep : bool, default=True
            If True, returns parameters for contained sub-estimators.

        Returns
        -------
        dict
            Parameter names mapped to their values.

        """
        return self._get_params("_forecasters", deep=deep)  # ty: ignore[unresolved-attribute]

    def set_params(self, **params: Any):
        """Set the parameters of this estimator.

        Parameters
        ----------
        **params : dict
            Estimator parameters.

        Returns
        -------
        self

        """
        self._set_params("_forecasters", **params)  # ty: ignore[unresolved-attribute]
        return self

    def _sk_visual_block_(self):
        """Return a visual block for the HTML repr."""
        from sklearn.utils._repr_html.estimator import _VisualBlock  # noqa: PLC0415

        names, estimators = zip(*self.forecasters, strict=False)
        return _VisualBlock("parallel", estimators, names=names)

    def observe(
        self,
        y: pl.DataFrame,
        X_actual: pl.DataFrame | None = None,
        groups: list[str] | None = None,
        X_future: pl.DataFrame | None = None,
        X_forecast: pl.DataFrame | None = None,
        **params,
    ):
        """Observe new data on all surviving base forecasters.

        Parameters
        ----------
        y : pl.DataFrame
            New target observations.
        X_actual : pl.DataFrame or None, default=None
            New actual feature observations with a ``"time"`` column
            aligned with ``y``. Forwarded to each child forecaster.
        groups : list of str or None, default=None
            Panel group prefixes to observe.
        X_future : pl.DataFrame or None, default=None
            Known future features.
        X_forecast : pl.DataFrame or None, default=None
            External forecasts.
        **params : dict
            Metadata routing parameters.

        Returns
        -------
        self

        """
        check_is_fitted(self, ["forecasters_"])
        for _name, forecaster in self.forecasters_:
            forecaster.observe(
                y=y, X_actual=X_actual, groups=groups, X_future=X_future, X_forecast=X_forecast, **params
            )
        return self

    def rewind(
        self,
        y: pl.DataFrame,
        X_actual: pl.DataFrame | None = None,
        groups: list[str] | None = None,
        X_future: pl.DataFrame | None = None,
        X_forecast: pl.DataFrame | None = None,
        **params,
    ):
        """Rewind all surviving base forecasters.

        Parameters
        ----------
        y : pl.DataFrame
            Target data to rewind to.
        X_actual : pl.DataFrame or None, default=None
            Actual feature observations to restore the observation
            state to. Must align with ``y``.
        groups : list of str or None, default=None
            Panel group prefixes to rewind.
        X_future : pl.DataFrame or None, default=None
            Known future features.
        X_forecast : pl.DataFrame or None, default=None
            External forecasts.
        **params : dict
            Metadata routing parameters.

        Returns
        -------
        self

        """
        check_is_fitted(self, ["forecasters_"])
        for _name, forecaster in self.forecasters_:
            forecaster.rewind(y=y, X_actual=X_actual, groups=groups, X_future=X_future, X_forecast=X_forecast, **params)
        return self
