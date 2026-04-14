"""Base mixin for ensemble forecasters."""

from __future__ import annotations

import warnings
from typing import Any

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

    Returns
    -------
    tuple of (str, BaseForecaster or None, str or None)
        Name, fitted forecaster (or None on failure), and error message
        (or None on success).

    """
    try:
        forecaster_clone = clone(forecaster)
        forecaster_clone.fit(y, X, forecasting_horizon=forecasting_horizon, **fit_params)
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

    def _observe_forecasters(
        self,
        y: pl.DataFrame,
        X: pl.DataFrame | None,
        **params,
    ) -> None:
        """Delegate observe to all successfully fitted forecasters.

        Parameters
        ----------
        y : pl.DataFrame
            New target observations.
        X : pl.DataFrame or None
            New exogenous observations.
        **params : dict
            Metadata routing parameters.

        """
        check_is_fitted(self, ["forecasters_"])
        for _name, forecaster in self.forecasters_:  # ty: ignore[unresolved-attribute]
            forecaster.observe(y=y, X=X, **params)

    def _rewind_forecasters(
        self,
        y: pl.DataFrame,
        X: pl.DataFrame | None,
        **params,
    ) -> None:
        """Delegate rewind to all successfully fitted forecasters.

        Parameters
        ----------
        y : pl.DataFrame
            Target data to rewind to.
        X : pl.DataFrame or None
            Exogenous data to rewind to.
        **params : dict
            Metadata routing parameters.

        """
        check_is_fitted(self, ["forecasters_"])
        for _name, forecaster in self.forecasters_:  # ty: ignore[unresolved-attribute]
            forecaster.rewind(y=y, X=X, **params)

    @property
    def named_forecasters_(self) -> Bunch:
        """Access fitted forecasters by name.

        Returns
        -------
        Bunch
            Dictionary-like object with forecaster names as keys.

        """
        check_is_fitted(self, ["forecasters_"])
        return Bunch(**dict(self.forecasters_))  # ty: ignore[unresolved-attribute]

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
