"""VotingClassProbaForecaster for combining class-probability forecasters."""

from __future__ import annotations

from numbers import Integral
from typing import Literal

import numpy as np
import polars as pl
from pydantic import StrictInt
from sklearn.utils import Bunch
from sklearn.utils.metadata_routing import (
    MetadataRouter,
    MethodMapping,
    process_routing,
)
from sklearn.utils.validation import check_is_fitted

from yohou.class_proba import BaseClassProbaForecaster
from yohou.utils import CLASS_PROBA, Tags
from yohou.utils._compat import StrOptions, _BaseComposition, _fit_context, _raise_for_params

from ._base import _BaseEnsembleForecaster

__all__ = ["VotingClassProbaForecaster"]


def _majority_vote(predictions: list[pl.DataFrame], target_col: str) -> list:
    """Compute the per-row majority vote with lexicographic tie-break.

    For each row, counts the votes contributed by every forecaster's
    ``target_col`` and returns the most frequent label; ties are broken by
    choosing the lexicographically smallest label. This is the vectorised
    equivalent of a per-row ``Counter`` loop and preserves the same tie-break.

    Parameters
    ----------
    predictions : list of pl.DataFrame
        Per-forecaster prediction frames, each containing ``target_col``.
    target_col : str
        Name of the column carrying each forecaster's predicted label.

    Returns
    -------
    list
        One winning label per row, in row order.

    """
    n_rows = len(predictions[0])
    votes = pl.DataFrame({
        str(forecaster_idx): pred[target_col] for forecaster_idx, pred in enumerate(predictions)
    }).with_row_index("__row__")

    tallies = (
        votes
        .unpivot(index="__row__", value_name="__label__")
        .group_by("__row__", "__label__")
        .len("__count__")
        .sort(["__row__", "__count__", "__label__"], descending=[False, True, False])
        .group_by("__row__", maintain_order=True)
        .first()
        .sort("__row__")
    )

    winners_by_row = dict(zip(tallies["__row__"].to_list(), tallies["__label__"].to_list(), strict=True))
    return [winners_by_row[row_idx] for row_idx in range(n_rows)]


class VotingClassProbaForecaster(_BaseEnsembleForecaster, BaseClassProbaForecaster, _BaseComposition):
    """Combines class-probability forecasters via voting.

    Aggregates predictions from multiple ``BaseClassProbaForecaster``
    instances using soft (probability averaging) or hard (majority vote)
    strategies.

    If a base forecaster fails during ``fit``, it is silently skipped
    with a warning.  The ensemble raises only when all base forecasters
    fail.

    Parameters
    ----------
    forecasters : list of (name, forecaster) tuples
        Named base class-probability forecasters to combine. Each entry
        is a ``(name, forecaster)`` tuple where *name* is a unique string
        and *forecaster* is a `BaseClassProbaForecaster` instance.
    method : {"soft", "hard"}, default="soft"
        Aggregation strategy:

        - ``"soft"``: weighted average of class probabilities.
        - ``"hard"``: majority vote of argmax predictions. Ties are
          broken deterministically by choosing the lexicographically first
          tied class (via Python ``sorted``).
    weights : list of float or None, default=None
        Per-forecaster weights. Raw values are passed to
        ``numpy.average`` which normalizes internally. Only used with
        ``method="soft"``. Silently ignored with ``method="hard"``, though
        ``weights_`` is still populated after fit regardless of ``method``.
    n_jobs : int or None, default=None
        Number of parallel jobs for fitting base forecasters.
        ``None`` means 1 unless in a ``joblib.parallel_backend`` context.
        ``-1`` means using all processors.

    Attributes
    ----------
    forecasters_ : list of (str, BaseClassProbaForecaster)
        Successfully fitted base forecasters.
    classes_ : dict of str to list of str
        Mapping from target column to sorted class labels.
    n_classes_ : dict of str to int
        Number of classes per target column.
    label_to_code_ : dict of str to dict of str to int
        Mapping from target column to label-to-code dict.
    weights_ : list of float or None
        Effective per-forecaster weights after removing forecasters that
        failed to fit. ``None`` when no weights were supplied.

    Examples
    --------
    >>> import polars as pl
    >>> from datetime import datetime
    >>> from yohou.ensemble import VotingClassProbaForecaster
    >>> from yohou.class_proba import ClassProbaReductionForecaster
    >>> from sklearn.tree import DecisionTreeClassifier
    >>>
    >>> time = pl.datetime_range(
    ...     start=datetime(2022, 1, 1), end=datetime(2022, 4, 10), interval="1d", eager=True
    ... )
    >>> categories = ["sunny", "rainy", "cloudy"]
    >>> y = pl.DataFrame({
    ...     "time": time,
    ...     "weather": [categories[i % 3] for i in range(len(time))],
    ... })
    >>>
    >>> forecaster = VotingClassProbaForecaster(
    ...     forecasters=[
    ...         (
    ...             "dt_1",
    ...             ClassProbaReductionForecaster(
    ...                 estimator=DecisionTreeClassifier(random_state=42),
    ...                 reduction_strategy="direct",
    ...             ),
    ...         ),
    ...         (
    ...             "dt_2",
    ...             ClassProbaReductionForecaster(
    ...                 estimator=DecisionTreeClassifier(random_state=123),
    ...                 reduction_strategy="direct",
    ...             ),
    ...         ),
    ...     ],
    ...     method="soft",
    ... )
    >>> forecaster.fit(y, forecasting_horizon=3)  # doctest: +ELLIPSIS
    VotingClassProbaForecaster(...)
    >>> y_pred = forecaster.predict(forecasting_horizon=3)
    >>> len(y_pred)
    3

    See Also
    --------
    - [`VotingPointForecaster`][yohou.ensemble.voting_point.VotingPointForecaster] : Ensemble for point forecasters.
    - [`VotingIntervalForecaster`][yohou.ensemble.voting_interval.VotingIntervalForecaster] : Ensemble for interval forecasters.
    - [`BaseClassProbaForecaster`][yohou.class_proba.base.BaseClassProbaForecaster] : Base class for class-probability forecasters.

    Notes
    -----
    - All base forecasters must discover the same classes at fit time.
      A ``ValueError`` is raised if class sets differ.
    - Weights are only used with ``method="soft"``; they are silently
      ignored with ``method="hard"``.

    """

    _parameter_constraints: dict = {
        "forecasters": [list],
        "method": [StrOptions({"soft", "hard"})],
        "weights": [list, None],
        "n_jobs": [Integral, None],
    }

    def __init__(
        self,
        forecasters: list[tuple[str, BaseClassProbaForecaster]],
        *,
        method: Literal["soft", "hard"] = "soft",
        weights: list[float] | None = None,
        n_jobs: int | None = None,
    ):
        super().__init__()
        self.forecasters = forecasters
        self.method = method
        self.weights = weights
        self.n_jobs = n_jobs

    def __sklearn_tags__(self) -> Tags:
        """Get estimator tags.

        Returns
        -------
        Tags
            Estimator tags with yohou-specific attributes.

        """
        tags = super().__sklearn_tags__()
        assert tags.forecaster_tags is not None

        tags.forecaster_tags.forecaster_type = CLASS_PROBA
        tags.forecaster_tags.tracks_observations = False
        tags.forecaster_tags.supports_panel_data = True

        forecasters_to_check = (
            [f for _, f in self.forecasters_] if hasattr(self, "forecasters_") else [f for _, f in self.forecasters]
        )

        if forecasters_to_check:
            tags.forecaster_tags.stateful = any(
                getattr(f.__sklearn_tags__().forecaster_tags, "stateful", False) for f in forecasters_to_check
            )

        return tags

    def _validate_classes_consistent(self) -> None:
        """Check that all surviving forecasters discovered the same classes.

        Raises
        ------
        ValueError
            If class sets differ across base forecasters.

        """
        reference_name, reference_forecaster = self.forecasters_[0]
        reference_classes = reference_forecaster.classes_  # ty: ignore[unresolved-attribute]

        for name, forecaster in self.forecasters_[1:]:
            if forecaster.classes_ != reference_classes:  # ty: ignore[unresolved-attribute]
                raise ValueError(
                    f"Forecaster '{name}' discovered classes {forecaster.classes_} "  # ty: ignore[unresolved-attribute]
                    f"but '{reference_name}' discovered {reference_classes}. "
                    f"All base forecasters must discover the same classes."
                )

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(
        self,
        y: pl.DataFrame,
        X_actual: pl.DataFrame | None = None,
        forecasting_horizon: StrictInt = 1,
        X_future: pl.DataFrame | None = None,
        X_forecast: pl.DataFrame | None = None,
        **params,
    ) -> VotingClassProbaForecaster:
        """Fit all base class-probability forecasters.

        Parameters
        ----------
        y : pl.DataFrame
            Target time series with ``"time"`` column and categorical
            value columns.
        X_actual : pl.DataFrame or None, default=None
            Actual feature observations with a ``"time"`` column aligned
            with ``y``. Forwarded to each child forecaster.
        forecasting_horizon : int, default=1
            Number of steps ahead to forecast.
        X_future : pl.DataFrame or None, default=None
            Known future features with ``"time"`` column.
        X_forecast : pl.DataFrame or None, default=None
            External forecasts with ``"vintage_time"`` and ``"time"`` columns.
        **params : dict
            Metadata routing parameters.

        Returns
        -------
        self
            Fitted ensemble.

        Raises
        ------
        ValueError
            If ``weights`` length does not match the number of
            forecasters, or if base forecasters discover different
            classes.
        RuntimeError
            If all base forecasters fail during fitting.

        """
        _raise_for_params(params, self, "fit")
        routed_params = process_routing(self, "fit", **params)

        if forecasting_horizon < 1:
            raise ValueError(f"forecasting_horizon must be >= 1, got {forecasting_horizon}")

        self._validate_forecasters_list()

        if self.weights is not None and len(self.weights) != len(self.forecasters):
            raise ValueError(
                f"Number of weights ({len(self.weights)}) must match number of forecasters ({len(self.forecasters)})"
            )

        self.forecasters_ = self._fit_forecasters_parallel(
            y=y,
            X_actual=X_actual,
            forecasting_horizon=forecasting_horizon,
            routed_params=routed_params,
            n_jobs=self.n_jobs,
            X_future=X_future,
            X_forecast=X_forecast,
        )

        self._validate_classes_consistent()

        # Derive fitted attributes from first surviving forecaster
        _first_name, first_forecaster = self.forecasters_[0]
        self._derive_fitted_attributes(first_forecaster, forecasting_horizon, y, X_actual)

        self.classes_ = dict(first_forecaster.classes_)  # ty: ignore[unresolved-attribute]
        self.n_classes_ = dict(first_forecaster.n_classes_)  # ty: ignore[unresolved-attribute]
        self.label_to_code_ = dict(first_forecaster.label_to_code_)  # ty: ignore[unresolved-attribute]

        # Compute effective weights for surviving forecasters
        self._compute_effective_weights()

        return self

    @staticmethod
    def _routed_for(routed_params: Bunch, name: str, callee: str) -> dict:
        """Per-forecaster routed metadata for one callee, or empty if none."""
        return getattr(routed_params.get(name, Bunch(**{callee: {}})), callee, {})

    def _predict_class_proba_one(
        self,
        groups: list[str],
        **params,
    ) -> pl.DataFrame:
        """Produce aggregated probability forecasts for one fit-horizon block.

        Single-block predict hook required by the abstract base class. It
        delegates to the same soft/hard voting helpers used by
        ``predict_class_proba`` so the voting logic lives in one place.

        Parameters
        ----------
        groups : list of str
            Panel group names to predict for.
        **params : dict
            Metadata routing parameters.

        Returns
        -------
        pl.DataFrame
            Aggregated probability predictions.

        """
        _raise_for_params(params, self, "predict_class_proba")
        routed_params = process_routing(self, "predict_class_proba", **params)
        if self.method == "soft":
            return self._soft_vote_predict_class_proba(groups=groups, routed_params=routed_params)
        return self._hard_vote_predict_class_proba(groups=groups, routed_params=routed_params)

    def predict_class_proba(  # ty: ignore[invalid-method-override]
        self,
        forecasting_horizon: StrictInt | None = None,
        groups: list[str] | None = None,
        X_future: pl.DataFrame | None = None,
        X_forecast: pl.DataFrame | None = None,
        **params,
    ) -> pl.DataFrame:
        """Generate aggregated class-probability forecasts.

        Parameters
        ----------
        forecasting_horizon : int or None, default=None
            Number of steps ahead. If ``None``, uses value from ``fit``.
        groups : list of str or None, default=None
            Panel group prefixes to predict.
        X_future : pl.DataFrame or None, default=None
            Known future features override. Re-derives step columns
            without mutating forecaster state.
        X_forecast : pl.DataFrame or None, default=None
            External forecast override with ``"vintage_time"`` and
            ``"time"`` columns. Re-derives step columns without mutating
            forecaster state.
        **params : dict
            Metadata routing parameters.

        Returns
        -------
        pl.DataFrame
            Probability predictions with ``"vintage_time"``,
            ``"time"``, and ``{target}_proba_{class}`` columns.

        """
        check_is_fitted(self, ["forecasters_", "classes_"])
        _raise_for_params(params, self, "predict_class_proba")
        routed_params = process_routing(self, "predict_class_proba", **params)

        if self.method == "soft":
            return self._soft_vote_predict_class_proba(
                forecasting_horizon=forecasting_horizon,
                groups=groups,
                X_future=X_future,
                X_forecast=X_forecast,
                routed_params=routed_params,
            )
        return self._hard_vote_predict_class_proba(
            forecasting_horizon=forecasting_horizon,
            groups=groups,
            X_future=X_future,
            X_forecast=X_forecast,
            routed_params=routed_params,
        )

    def _soft_vote_predict_class_proba(
        self,
        forecasting_horizon: StrictInt | None = None,
        groups: list[str] | None = None,
        X_future: pl.DataFrame | None = None,
        X_forecast: pl.DataFrame | None = None,
        *,
        routed_params: Bunch,
    ) -> pl.DataFrame:
        """Soft vote: weighted average of class probabilities.

        Parameters
        ----------
        forecasting_horizon : int or None, default=None
            Forecasting horizon.
        groups : list of str or None, default=None
            Panel group prefixes.
        X_future : pl.DataFrame or None, default=None
            Known future features override.
        X_forecast : pl.DataFrame or None, default=None
            External forecast override.
        routed_params : Bunch
            Pre-routed per-forecaster metadata for ``predict_class_proba``.

        Returns
        -------
        pl.DataFrame
            Averaged probability predictions.

        """
        predictions = []
        for name, forecaster in self.forecasters_:
            y_proba = forecaster.predict_class_proba(  # ty: ignore[unresolved-attribute]
                forecasting_horizon=forecasting_horizon,
                groups=groups,
                X_future=X_future,
                X_forecast=X_forecast,
                **self._routed_for(routed_params, name, "predict_class_proba"),
            )
            predictions.append(y_proba)

        time_df = predictions[0].select(["vintage_time", "time"])
        proba_cols = [c for c in predictions[0].columns if c not in ("vintage_time", "time")]

        agg_exprs = []
        for col in proba_cols:
            values = np.column_stack([pred[col].to_numpy() for pred in predictions])

            if self.weights_ is not None:
                aggregated = np.average(values, axis=1, weights=self.weights_)
            else:
                aggregated = np.mean(values, axis=1)

            agg_exprs.append(pl.Series(name=col, values=aggregated))

        return time_df.with_columns(agg_exprs)

    def _hard_vote_predict_class_proba(
        self,
        forecasting_horizon: StrictInt | None = None,
        groups: list[str] | None = None,
        X_future: pl.DataFrame | None = None,
        X_forecast: pl.DataFrame | None = None,
        *,
        routed_params: Bunch,
    ) -> pl.DataFrame:
        """Hard vote: majority vote converted to one-hot probabilities.

        Parameters
        ----------
        forecasting_horizon : int or None, default=None
            Forecasting horizon.
        groups : list of str or None, default=None
            Panel group prefixes.
        X_future : pl.DataFrame or None, default=None
            Known future features override.
        X_forecast : pl.DataFrame or None, default=None
            External forecast override.
        routed_params : Bunch
            Pre-routed per-forecaster metadata for ``predict_class_proba``.

        Returns
        -------
        pl.DataFrame
            One-hot probability predictions from majority vote.

        """
        predictions = []
        for name, forecaster in self.forecasters_:
            y_pred = forecaster.predict(  # ty: ignore[unresolved-attribute]
                forecasting_horizon=forecasting_horizon,
                groups=groups,
                X_future=X_future,
                X_forecast=X_forecast,
                **self._routed_for(routed_params, name, "predict"),
            )
            predictions.append(y_pred)

        time_df = predictions[0].select(["vintage_time", "time"])
        target_cols = [c for c in predictions[0].columns if c not in ("vintage_time", "time")]

        result = time_df.clone()
        for target_col in target_cols:
            class_labels = self.classes_[target_col]

            winners = _majority_vote(predictions, target_col)
            winners_series = pl.Series(values=winners)

            for label in class_labels:
                col_name = f"{target_col}_proba_{label}"
                result = result.with_columns(winners_series.eq(label).cast(pl.Float64).alias(col_name))

        return result

    def predict(  # ty: ignore[invalid-method-override]
        self,
        forecasting_horizon: StrictInt | None = None,
        groups: list[str] | None = None,
        X_future: pl.DataFrame | None = None,
        X_forecast: pl.DataFrame | None = None,
        **params,
    ) -> pl.DataFrame:
        """Generate argmax class predictions from the ensemble.

        Parameters
        ----------
        forecasting_horizon : int or None, default=None
            Number of steps ahead. If ``None``, uses value from ``fit``.
        groups : list of str or None, default=None
            Panel group prefixes.
        X_future : pl.DataFrame or None, default=None
            Known future features override. Re-derives step columns
            without mutating forecaster state.
        X_forecast : pl.DataFrame or None, default=None
            External forecast override with ``"vintage_time"`` and
            ``"time"`` columns. Re-derives step columns without mutating
            forecaster state.
        **params : dict
            Metadata routing parameters.

        Returns
        -------
        pl.DataFrame
            Predictions with ``"vintage_time"``, ``"time"``, and one
            column per target with the most likely class label.

        """
        check_is_fitted(self, ["forecasters_", "classes_"])
        _raise_for_params(params, self, "predict")
        routed_params = process_routing(self, "predict", **params)

        if self.method == "hard":
            return self._hard_vote_predict(
                forecasting_horizon=forecasting_horizon,
                groups=groups,
                X_future=X_future,
                X_forecast=X_forecast,
                routed_params=routed_params,
            )

        y_proba = self._soft_vote_predict_class_proba(
            forecasting_horizon=forecasting_horizon,
            groups=groups,
            X_future=X_future,
            X_forecast=X_forecast,
            routed_params=routed_params,
        )
        return self._ensemble_argmax_from_proba(y_proba)

    def _ensemble_argmax_from_proba(self, y_proba: pl.DataFrame) -> pl.DataFrame:
        """Convert probability DataFrame to argmax class DataFrame.

        Panel-aware version that handles both panel-prefixed and plain
        proba column names.

        Parameters
        ----------
        y_proba : pl.DataFrame
            Probability predictions.

        Returns
        -------
        pl.DataFrame
            DataFrame with argmax class labels.

        """
        time_cols = [c for c in ("vintage_time", "time") if c in y_proba.columns]
        result = y_proba.select(time_cols)

        groups = self.groups_ or [None]

        for group in groups:
            for target_col, class_labels in self.classes_.items():
                if group is not None:
                    proba_cols = [f"{group}__{target_col}_proba_{label}" for label in class_labels]
                    out_col = f"{group}__{target_col}"
                else:
                    proba_cols = [f"{target_col}_proba_{label}" for label in class_labels]
                    out_col = target_col

                argmax_series = y_proba.select(pl.concat_list(proba_cols).list.arg_max().cast(pl.UInt32).alias("_idx"))[
                    "_idx"
                ]
                label_series = pl.Series(values=class_labels)
                result = result.with_columns(label_series.gather(argmax_series).alias(out_col))

        return result

    def _hard_vote_predict(
        self,
        forecasting_horizon: StrictInt | None = None,
        groups: list[str] | None = None,
        X_future: pl.DataFrame | None = None,
        X_forecast: pl.DataFrame | None = None,
        *,
        routed_params: Bunch,
    ) -> pl.DataFrame:
        """Hard vote: majority vote of argmax predictions.

        Parameters
        ----------
        forecasting_horizon : int or None, default=None
            Forecasting horizon.
        groups : list of str or None, default=None
            Panel group prefixes.
        X_future : pl.DataFrame or None, default=None
            Known future features override.
        X_forecast : pl.DataFrame or None, default=None
            External forecast override.
        **params : dict
            Routing parameters.

        Returns
        -------
        pl.DataFrame
            Majority vote predictions.

        """
        predictions = []
        for name, forecaster in self.forecasters_:
            y_pred = forecaster.predict(  # ty: ignore[unresolved-attribute]
                forecasting_horizon=forecasting_horizon,
                groups=groups,
                X_future=X_future,
                X_forecast=X_forecast,
                **self._routed_for(routed_params, name, "predict"),
            )
            predictions.append(y_pred)

        time_df = predictions[0].select(["vintage_time", "time"])
        target_cols = [c for c in predictions[0].columns if c not in ("vintage_time", "time")]

        result = time_df.clone()
        for target_col in target_cols:
            winners = _majority_vote(predictions, target_col)
            result = result.with_columns(pl.Series(name=target_col, values=winners))

        return result

    def get_metadata_routing(self) -> MetadataRouter:
        """Get metadata routing configuration.

        Returns
        -------
        MetadataRouter
            Router with mappings for all base forecasters.

        """
        router = MetadataRouter(owner=self.__class__.__name__)

        for name, forecaster in self.forecasters:
            router.add(
                **{name: forecaster},
                method_mapping=MethodMapping()
                .add(caller="fit", callee="fit")
                .add(caller="predict", callee="predict")
                .add(caller="predict", callee="predict_class_proba")
                .add(caller="predict_class_proba", callee="predict_class_proba")
                .add(caller="predict_class_proba", callee="predict"),
            )

        return router
