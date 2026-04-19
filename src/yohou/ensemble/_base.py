"""Base mixin for ensemble forecasters."""

from __future__ import annotations

import warnings
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
    X: pl.DataFrame | None,
    forecasting_horizon: int,
    fit_params: dict[str, Any],
    extra_fit_kwargs: dict[str, Any] | None = None,
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
    X : pl.DataFrame or None
        Exogenous features.
    forecasting_horizon : int
        Forecasting horizon.
    fit_params : dict
        Routed fit parameters for this forecaster.
    extra_fit_kwargs : dict or None
        Additional keyword arguments to pass to the child's ``fit``
        method (e.g. ``coverage_rates``).

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
        forecaster_clone.fit(y, X, forecasting_horizon=forecasting_horizon, **all_params)
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

    def _validate_forecasters_list(self) -> None:
        """Validate the forecasters parameter.

        Raises
        ------
        ValueError
            If names are not unique or tuples are malformed.

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
            names.append(name)

        duplicates = [n for n in set(names) if names.count(n) > 1]
        if duplicates:
            raise ValueError(f"Duplicate forecaster names: {duplicates}")

    def _fit_forecasters_parallel(
        self,
        y: pl.DataFrame,
        X: pl.DataFrame | None,
        forecasting_horizon: int,
        routed_params: Any,
        n_jobs: int | None,
        extra_fit_kwargs: dict[str, Any] | None = None,
    ) -> list[tuple[str, BaseForecaster]]:
        """Clone and fit all base forecasters in parallel.

        Failed forecasters are skipped with a warning. Raises if all fail.

        Parameters
        ----------
        y : pl.DataFrame
            Target time series.
        X : pl.DataFrame or None
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
                X,
                forecasting_horizon,
                getattr(routed_params.get(name, Bunch(fit={})), "fit", {}),
                extra_fit_kwargs,
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
            If target column schemas differ across forecasters.

        """
        schemas = {}
        for name, forecaster in self.forecasters_:
            schema = dict(forecaster.local_y_schema_)
            schemas[name] = schema

        reference_name, reference_schema = next(iter(schemas.items()))
        for name, schema in schemas.items():
            if schema != reference_schema:
                raise ValueError(
                    f"Forecaster '{name}' predicts columns {set(schema.keys())} "
                    f"but '{reference_name}' predicts {set(reference_schema.keys())}. "
                    f"All base forecasters must predict the same target columns."
                )

    def _derive_fitted_attributes(
        self,
        first_forecaster: BaseForecaster,
        forecasting_horizon: int,
        y: pl.DataFrame,
        X: pl.DataFrame | None,
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
        X : pl.DataFrame or None
            Training exogenous data.

        """
        self.fit_forecasting_horizon_ = forecasting_horizon
        self.interval_ = first_forecaster.interval_
        self.groups_ = first_forecaster.groups_
        self.local_y_schema_ = dict(first_forecaster.local_y_schema_)
        self.local_X_schema_ = getattr(first_forecaster, "local_X_schema_", None)
        self.shared_X_schema_ = getattr(first_forecaster, "shared_X_schema_", None)
        self.local_y_t_schema_ = self.local_y_schema_
        self.local_X_t_schema_ = self.local_X_schema_
        self._y_observed = y
        self._X_observed = X
        self._X_t_observed = X

    def _compute_effective_weights(self) -> None:
        """Compute effective weights for surviving forecasters.

        Maps original weights to only the forecasters that survived
        fitting. Sets ``self.weights_`` to the filtered weights list
        or ``None``.

        """
        if self.weights is not None:
            fitted_names = {name for name, _ in self.forecasters_}
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
        time_cols = [c for c in ("observed_time", "time") if c in predictions[0].columns]
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
            Per-forecaster weights for weighted averaging.

        Returns
        -------
        pl.DataFrame
            Aggregated interval predictions.

        """
        time_cols = [c for c in ("observed_time", "time") if c in predictions[0].columns]
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
            elif weights is not None:
                aggregated = np.average(values, axis=1, weights=weights)
            else:
                aggregated = np.mean(values, axis=1)

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
        X: pl.DataFrame | None = None,
        groups: list[str] | None = None,
        **params,
    ):
        """Observe new data on all surviving base forecasters.

        Parameters
        ----------
        y : pl.DataFrame
            New target observations.
        X : pl.DataFrame or None, default=None
            New exogenous observations.
        groups : list of str or None, default=None
            Panel group prefixes to observe.
        **params : dict
            Metadata routing parameters.

        Returns
        -------
        self

        """
        check_is_fitted(self, ["forecasters_"])
        for _name, forecaster in self.forecasters_:
            forecaster.observe(y=y, X=X, groups=groups, **params)
        return self

    def rewind(
        self,
        y: pl.DataFrame,
        X: pl.DataFrame | None = None,
        groups: list[str] | None = None,
        **params,
    ):
        """Rewind all surviving base forecasters.

        Parameters
        ----------
        y : pl.DataFrame
            Target data to rewind to.
        X : pl.DataFrame or None, default=None
            Exogenous data to rewind to.
        groups : list of str or None, default=None
            Panel group prefixes to rewind.
        **params : dict
            Metadata routing parameters.

        Returns
        -------
        self

        """
        check_is_fitted(self, ["forecasters_"])
        for _name, forecaster in self.forecasters_:
            forecaster.rewind(y=y, X=X, groups=groups, **params)
        return self
