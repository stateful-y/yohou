"""Composition classes for forecaster transformer pipelines.

This module provides composition-based forecaster wrappers that apply
transformers to targets or features before/after forecasting.
"""

import polars as pl
import polars.selectors as cs
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.utils.metadata_routing import _raise_for_params, process_routing
from sklearn.utils.metaestimators import _BaseComposition

from yohou.base import BaseForecaster


# TODO: Add target and feature transformers?
class ColumnForecaster(BaseForecaster, _BaseComposition):
    """Applies forecasters to columns of the target time series.

    This estimator allows different forecasters to be used for different
    subsets of the target columns. It is useful when different time series
    require different modeling strategies (e.g., different seasonality,
    different distributions).

    Parameters
    ----------
    forecasters : list of (str, forecaster, columns) tuples
        List of (name, forecaster, column(s)) tuples specifying the
        forecaster objects to be applied to subsets of the data.

        name : str
            Name of the forecaster.
        forecaster : estimator
            Forecaster object.
        columns : str or list of str
            Column name(s) to be forecasted by this forecaster.

    n_jobs : int, default=None
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors.

    verbose : bool, default=False
        If True, the time elapsed while fitting each forecaster will be
        printed as it is completed.

    Attributes
    ----------
    forecasters_ : list
        The list of fitted forecasters.

    Examples
    --------
    >>> import polars as pl
    >>> from datetime import datetime
    >>> from yohou.forecaster.composition import ColumnForecaster
    >>> from yohou.point_forecaster import PointReductionForecaster, SeasonalNaive
    >>> from sklearn.linear_model import Ridge
    >>>
    >>> y = pl.DataFrame({
    ...     "time": pl.datetime_range(
    ...         start=datetime(2022, 1, 1),
    ...         end=datetime(2022, 1, 10),
    ...         interval="1d",
    ...         eager=True
    ...     ),
    ...     "sales": range(10),
    ...     "inventory": range(10)
    ... })
    >>>
    >>> forecaster = ColumnForecaster([
    ...     ("sales_model", PointReductionForecaster(estimator=Ridge()), ["sales"]),
    ...     ("inventory_model", SeasonalNaive(seasonality=7), ["inventory"])
    ... ])
    >>>
    >>> _ = forecaster.fit(y, forecasting_horizon=3)
    >>> y_pred = forecaster.predict(forecasting_horizon=3)
    """

    def __init__(
        self,
        forecasters: list[tuple[str, BaseForecaster, str | list[str]]],
        n_jobs: int | None = None,
        verbose: bool = False,
    ):
        self.forecasters = forecasters
        self.n_jobs = n_jobs
        self.verbose = verbose

    def fit(
        self,
        y: pl.DataFrame,
        X: pl.DataFrame | None = None,
        forecasting_horizon: int = 1,
        **params,
    ) -> "ColumnForecaster":
        """Fit all forecasters.

        Parameters
        ----------
        y : pl.DataFrame
            Target time series.
        X : pl.DataFrame, optional
            Exogenous features.
        forecasting_horizon : int, default=1
            Forecasting horizon.
        **params : dict
            Metadata to route to nested estimators.

        Returns
        -------
        self
        """
        # Validate params before routing
        _raise_for_params(params, self, "fit")

        # Validate inputs
        self._validate_names([name for name, _, _ in self.forecasters])

        # Route metadata to nested forecasters
        routed_params = process_routing(self, "fit", **params)

        # Store fit parameters
        self.fit_forecasting_horizon_ = forecasting_horizon
        self.interval_ = None  # Will be set from first forecaster
        self._y_observed = y
        self._X_observed = X

        # Parallel fit
        self.forecasters_ = Parallel(n_jobs=self.n_jobs)(
            delayed(_fit_one_forecaster)(
                forecaster=clone(forecaster),
                y=y.select(["time"] + (cols if isinstance(cols, list) else [cols])),
                X=X,
                forecasting_horizon=forecasting_horizon,
                message_clsname="ColumnForecaster",
                message=self._log_message(name, idx, len(self.forecasters)),
                params=routed_params[name],
            )
            for idx, (name, forecaster, cols) in enumerate(self.forecasters)
        )

        # Set attributes from first fitted forecaster
        if self.forecasters_:
            self.interval_ = self.forecasters_[0].interval_
            self.panel_group_names_ = self.forecasters_[0].panel_group_names_
            # Combine local_y_names from all forecasters?
            # For now, just assume they are compatible or handle separately.

        return self

    def predict(
        self,
        X: pl.DataFrame | None = None,
        forecasting_horizon: int | None = None,
        panel_group_names: list[str] | None = None,
        predict_transformed: bool = False,
        **params,
    ) -> pl.DataFrame:
        """Predict using all forecasters and concatenate results.

        Parameters
        ----------
        X : pl.DataFrame, optional
            Exogenous features.
        forecasting_horizon : int, optional
            Forecasting horizon.
        panel_group_names : list of str or None, default=None
            Group prefixes for panel data:
            - If None: predict for all groups
            - If list of str: predict only for the specified panel groups
            Parameter is ignored if the forecaster was not fitted on panel data.
        predict_transformed : bool, default=False
            Return transformed predictions.
        **params : dict
            Metadata to route to nested estimators.

        Returns
        -------
        pl.DataFrame
            Concatenated predictions.
        """
        # Validate params before routing
        _raise_for_params(params, self, "predict")

        # Route metadata to nested forecasters
        routed_params = process_routing(self, "predict", **params)

        # Parallel predict
        predictions = Parallel(n_jobs=self.n_jobs)(
            delayed(_predict_one_forecaster)(
                forecaster=forecaster,
                X=X,
                forecasting_horizon=forecasting_horizon,
                panel_group_names=panel_group_names,
                predict_transformed=predict_transformed,
                params=routed_params[name],
            )
            for forecaster, (name, _, _) in zip(self.forecasters_, self.forecasters)
        )

        if not predictions:
            return pl.DataFrame()

        # Concatenate predictions horizontally
        # All predictions should have "observed_time" and "time" columns
        # We need to merge them on these keys

        # Start with time columns from first prediction
        result = predictions[0].select(["observed_time", "time"])

        for pred in predictions:
            # Drop time columns and horizontally concat
            pred_cols = pred.select(~cs.by_name("observed_time", "time"))
            result = pl.concat([result, pred_cols], how="horizontal")

        return result

    def update(
        self,
        y: pl.DataFrame,
        X: pl.DataFrame | None = None,
        panel_group_names: list[str] | None = None,
    ) -> "ColumnForecaster":
        """Update all forecasters.

        Parameters
        ----------
        y : pl.DataFrame
            New target data.
        X : pl.DataFrame, optional
            New exogenous features.
        panel_group_names : list of str or None, default=None
            Group prefixes for panel data:
            - If None: update all groups
            - If list of str: update only for the specified panel groups
            Parameter is ignored if the forecaster was not fitted on panel data.

        Returns
        -------
        self
        """
        Parallel(n_jobs=self.n_jobs)(
            delayed(forecaster.update)(
                y=y.select(["time"] + (cols if isinstance(cols, list) else [cols])),
                X=X,
                panel_group_names=panel_group_names,
            )
            # TODO: What's this?
            for forecaster, (_, _, cols) in zip(self.forecasters_, self.forecasters)
        )

        self._y_observed = pl.concat([self._y_observed, y])
        if X is not None:
            self._X_observed = (
                pl.concat([self._X_observed, X]) if self._X_observed is not None else X
            )

        return self

    def reset(
        self,
        y: pl.DataFrame,
        X: pl.DataFrame | None = None,
        panel_group_names: list[str] | None = None,
    ) -> "ColumnForecaster":
        """Reset all forecasters.

        Parameters
        ----------
        y : pl.DataFrame
            New target data.
        X : pl.DataFrame, optional
            New exogenous features.
        panel_group_names : list of str or None, default=None
            Group prefixes for panel data:
            - If None: reset all groups
            - If list of str: reset only for the specified panel groups
            Parameter is ignored if the forecaster was not fitted on panel data.

        Returns
        -------
        self
        """
        Parallel(n_jobs=self.n_jobs)(
            delayed(forecaster.reset)(
                y=y.select(["time"] + (cols if isinstance(cols, list) else [cols])),
                X=X,
                panel_group_names=panel_group_names,
            )
            # TODO: What's this?
            for forecaster, (_, _, cols) in zip(self.forecasters_, self.forecasters)
        )

        self._y_observed = y
        self._X_observed = X

        return self

    @property
    def prediction_types(self) -> set[str]:
        """Return union of prediction types from all forecasters.

        Returns
        -------
        set[str]
            Set of prediction types.
        """
        types = set()
        if hasattr(self, "forecasters_"):
            for f in self.forecasters_:
                types.update(f.prediction_types)
        return types

    def _log_message(self, name: str, idx: int, total: int) -> str | None:
        """Generate progress message for verbose logging.

        Parameters
        ----------
        name : str
            Name of the forecaster step.
        idx : int
            Current step index (0-based).
        total : int
            Total number of steps.

        Returns
        -------
        str or None
            Formatted message string if verbose is True, None otherwise.

        """
        if not self.verbose:
            return None
        return f"(step {idx + 1} of {total}) Processing {name}"

    def get_metadata_routing(self):
        """Get metadata routing for this estimator.

        Returns
        -------
        MetadataRouter
            Metadata routing configuration.

        """
        from sklearn.utils.metadata_routing import MetadataRouter, MethodMapping

        router = MetadataRouter(owner=self)

        # Add routing for each forecaster
        # Note: update and reset are NOT routed (memory management operations)
        for name, forecaster, _ in self.forecasters:
            router.add(
                **{name: forecaster},
                method_mapping=MethodMapping()
                .add(caller="fit", callee="fit")
                .add(caller="predict", callee="predict")
                .add(caller="predict_interval", callee="predict_interval")
                .add(caller="update_predict", callee="update_predict")
                .add(caller="update_predict_interval", callee="update_predict_interval"),
            )

        return router


def _fit_one_forecaster(
    forecaster: BaseForecaster,
    y: pl.DataFrame,
    X: pl.DataFrame | None,
    forecasting_horizon: int,
    message_clsname: str,
    message: str | None,
    params,
) -> BaseForecaster:
    """Fit a single forecaster.

    Parameters
    ----------
    forecaster : BaseForecaster
        Forecaster to fit.
    y : pl.DataFrame
        Target time series.
    X : pl.DataFrame, optional
        Exogenous features.
    forecasting_horizon : int
        Forecasting horizon.
    message_clsname : str
        Class name for logging.
    message : str, optional
        Progress message.
    params : Bunch
        Routed parameters from process_routing.

    Returns
    -------
    BaseForecaster
        Fitted forecaster.

    """
    if message:
        print(f"[{message_clsname}] {message}")

    forecaster.fit(y, X, forecasting_horizon=forecasting_horizon, **params.fit)
    return forecaster


def _predict_one_forecaster(
    forecaster: BaseForecaster,
    X: pl.DataFrame | None,
    forecasting_horizon: int | None,
    panel_group_names: list[str] | None,
    predict_transformed: bool,
    params,  # Bunch object from process_routing
) -> pl.DataFrame:
    """Predict with a single forecaster.

    Parameters
    ----------
    forecaster : BaseForecaster
        Fitted forecaster.
    X : pl.DataFrame, optional
        Exogenous features.
    forecasting_horizon : int, optional
        Forecasting horizon.
    panel_group_names : list of str or None
        Group prefixes for panel data.
    predict_transformed : bool
        Return transformed predictions.
    params : Bunch
        Routed parameters from process_routing.

    Returns
    -------
    pl.DataFrame
        Predictions.

    """
    return forecaster.predict(
        X=X,
        forecasting_horizon=forecasting_horizon,
        panel_group_names=panel_group_names,
        predict_transformed=predict_transformed,
        **params.predict,
    )
