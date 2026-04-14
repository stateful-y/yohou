"""VotingClassProbaForecaster for combining class-probability forecasters."""

from __future__ import annotations

from collections import Counter
from numbers import Integral
from typing import Any, Literal

import numpy as np
import polars as pl
from pydantic import StrictInt
from sklearn.utils.metadata_routing import (
    MetadataRouter,
    MethodMapping,
    process_routing,
)
from sklearn.utils.validation import check_is_fitted

from yohou.class_proba import BaseClassProbaForecaster
from yohou.utils import Tags
from yohou.utils._compat import StrOptions, _BaseComposition, _fit_context, _raise_for_params

from ._base import _BaseEnsembleForecaster

__all__ = ["VotingClassProbaForecaster"]


class VotingClassProbaForecaster(BaseClassProbaForecaster, _BaseEnsembleForecaster, _BaseComposition):
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
    voting : {"soft", "hard"}, default="soft"
        Voting strategy:

        - ``"soft"``: weighted average of class probabilities.
        - ``"hard"``: majority vote of argmax predictions. Ties are
          broken deterministically by choosing the first class in sorted
          order (via ``numpy.argmax``), matching sklearn convention.
    weights : list of float or None, default=None
        Per-forecaster weights. Raw values are passed to
        ``numpy.average`` which normalizes internally. Only used with
        ``voting="soft"``. Silently ignored with ``voting="hard"``.
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
    label_to_code_ : dict of str to dict of str to float
        Mapping from target column to label-to-code dict.

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
    ...     voting="soft",
    ... )
    >>> forecaster.fit(y, forecasting_horizon=3)  # doctest: +ELLIPSIS
    VotingClassProbaForecaster(...)
    >>> y_pred = forecaster.predict(forecasting_horizon=3)
    >>> len(y_pred)
    3

    See Also
    --------
    `VotingForecaster` : Ensemble for point and interval forecasters.
    `BaseClassProbaForecaster` : Base class for class-probability forecasters.

    Notes
    -----
    - All base forecasters must discover the same classes at fit time.
      A ``ValueError`` is raised if class sets differ.
    - Weights are only used with ``voting="soft"``; they are silently
      ignored with ``voting="hard"``.

    """

    _parameter_constraints: dict = {
        "forecasters": [list],
        "voting": [StrOptions({"soft", "hard"})],
        "weights": [list, None],
        "n_jobs": [Integral, None],
    }

    def __init__(
        self,
        forecasters: list[tuple[str, BaseClassProbaForecaster]],
        *,
        voting: Literal["soft", "hard"] = "soft",
        weights: list[float] | None = None,
        n_jobs: int | None = None,
    ):
        super().__init__()
        self.forecasters = forecasters
        self.voting = voting
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

        tags.forecaster_tags.forecaster_type = "class_proba"
        tags.forecaster_tags.tracks_observations = False
        tags.forecaster_tags.supports_panel_data = True

        return tags

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
        return self._get_params("_forecasters", deep=deep)

    def set_params(self, **params: Any) -> VotingClassProbaForecaster:
        """Set the parameters of this estimator.

        Parameters
        ----------
        **params : dict
            Estimator parameters.

        Returns
        -------
        self

        """
        self._set_params("_forecasters", **params)
        return self

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
        X: pl.DataFrame | None = None,
        forecasting_horizon: StrictInt = 1,
        **params,
    ) -> VotingClassProbaForecaster:
        """Fit all base class-probability forecasters.

        Parameters
        ----------
        y : pl.DataFrame
            Target time series with ``"time"`` column and categorical
            value columns.
        X : pl.DataFrame or None, default=None
            Exogenous features with ``"time"`` column.
        forecasting_horizon : int, default=1
            Number of steps ahead to forecast.
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
            X=X,
            forecasting_horizon=forecasting_horizon,
            routed_params=routed_params,
            n_jobs=self.n_jobs,
        )

        self._validate_classes_consistent()

        # Derive fitted attributes from first surviving forecaster
        _first_name, first_forecaster = self.forecasters_[0]
        self.fit_forecasting_horizon_ = forecasting_horizon
        self.interval_ = first_forecaster.interval_
        self.panel_group_names_ = first_forecaster.panel_group_names_
        self.local_y_schema_ = dict(first_forecaster.local_y_schema_)
        self.local_X_schema_ = getattr(first_forecaster, "local_X_schema_", None)
        self.shared_X_schema_ = getattr(first_forecaster, "shared_X_schema_", None)
        self.local_y_t_schema_ = self.local_y_schema_
        self.local_X_t_schema_ = self.local_X_schema_
        self._y_observed = y
        self._X_observed = X
        self._X_t_observed = X

        self.classes_ = dict(first_forecaster.classes_)  # ty: ignore[unresolved-attribute]
        self.n_classes_ = dict(first_forecaster.n_classes_)  # ty: ignore[unresolved-attribute]
        self.label_to_code_ = dict(first_forecaster.label_to_code_)  # ty: ignore[unresolved-attribute]

        # Compute effective weights for surviving forecasters
        if self.weights is not None:
            fitted_names = {name for name, _ in self.forecasters_}
            self.weights_ = [
                w for (name, _), w in zip(self.forecasters, self.weights, strict=True) if name in fitted_names
            ]
        else:
            self.weights_ = None

        return self

    def _predict_class_proba_one(
        self,
        panel_group_names: list[str],
        **params,
    ) -> pl.DataFrame:
        """Produce aggregated probability forecasts.

        Parameters
        ----------
        panel_group_names : list of str
            Panel group names to predict for.
        **params : dict
            Metadata routing parameters.

        Returns
        -------
        pl.DataFrame
            Aggregated probability predictions.

        """
        if self.voting == "soft":
            return self._soft_vote_proba(panel_group_names=panel_group_names, **params)
        return self._hard_vote_proba(panel_group_names=panel_group_names, **params)

    def _soft_vote_proba(
        self,
        panel_group_names: list[str],
        **params,
    ) -> pl.DataFrame:
        """Compute weighted average of class probabilities.

        Parameters
        ----------
        panel_group_names : list of str
            Panel group names to predict for.
        **params : dict
            Metadata routing parameters.

        Returns
        -------
        pl.DataFrame
            Averaged probability predictions.

        """
        predictions = []
        for _name, forecaster in self.forecasters_:
            y_proba = forecaster.predict_class_proba(  # ty: ignore[unresolved-attribute]
                panel_group_names=panel_group_names,
                **params,
            )
            predictions.append(y_proba)

        time_df = predictions[0].select(["observed_time", "time"])
        proba_cols = [c for c in predictions[0].columns if c not in ("observed_time", "time")]

        agg_exprs = []
        for col in proba_cols:
            values = np.column_stack([pred[col].to_numpy() for pred in predictions])

            if self.weights_ is not None:
                aggregated = np.average(values, axis=1, weights=self.weights_)
            else:
                aggregated = np.mean(values, axis=1)

            agg_exprs.append(pl.Series(name=col, values=aggregated))

        return time_df.with_columns(agg_exprs)

    def _hard_vote_proba(
        self,
        panel_group_names: list[str],
        **params,
    ) -> pl.DataFrame:
        """Compute majority vote and convert to one-hot probabilities.

        Parameters
        ----------
        panel_group_names : list of str
            Panel group names to predict for.
        **params : dict
            Metadata routing parameters.

        Returns
        -------
        pl.DataFrame
            One-hot probability predictions from majority vote.

        """
        predictions = []
        for _name, forecaster in self.forecasters_:
            y_pred = forecaster.predict(  # ty: ignore[unresolved-attribute]
                panel_group_names=panel_group_names,
                **params,
            )
            predictions.append(y_pred)

        time_df = predictions[0].select(["observed_time", "time"])
        target_cols = [c for c in predictions[0].columns if c not in ("observed_time", "time")]

        result = time_df.clone()
        for target_col in target_cols:
            class_labels = self.classes_[target_col]
            n_rows = len(predictions[0])

            # Count votes for each row
            winners = []
            for row_idx in range(n_rows):
                votes = [pred[target_col][row_idx] for pred in predictions]
                vote_counts = Counter(votes)
                # Most common; ties broken by Counter ordering (insertion order)
                # then by sorted class order for determinism
                max_count = max(vote_counts.values())
                candidates = sorted(label for label, count in vote_counts.items() if count == max_count)
                winners.append(candidates[0])

            # Build one-hot probability columns
            for label in class_labels:
                col_name = f"{target_col}_proba_{label}"
                proba_values = [1.0 if w == label else 0.0 for w in winners]
                result = result.with_columns(pl.Series(name=col_name, values=proba_values))

        return result

    def predict_class_proba(
        self,
        X: pl.DataFrame | None = None,
        forecasting_horizon: StrictInt | None = None,
        panel_group_names: list[str] | None = None,
        **params,
    ) -> pl.DataFrame:
        """Generate aggregated class-probability forecasts.

        Parameters
        ----------
        X : pl.DataFrame or None, default=None
            Exogenous features.
        forecasting_horizon : int or None, default=None
            Number of steps ahead. If ``None``, uses value from ``fit``.
        panel_group_names : list of str or None, default=None
            Panel group prefixes to predict.
        **params : dict
            Metadata routing parameters.

        Returns
        -------
        pl.DataFrame
            Probability predictions with ``"observed_time"``,
            ``"time"``, and ``{target}_proba_{class}`` columns.

        """
        check_is_fitted(self, ["forecasters_", "classes_"])

        if self.voting == "soft":
            return self._soft_vote_predict_class_proba(
                X=X,
                forecasting_horizon=forecasting_horizon,
                panel_group_names=panel_group_names,
                **params,
            )
        return self._hard_vote_predict_class_proba(
            X=X,
            forecasting_horizon=forecasting_horizon,
            panel_group_names=panel_group_names,
            **params,
        )

    def _soft_vote_predict_class_proba(
        self,
        X: pl.DataFrame | None = None,
        forecasting_horizon: StrictInt | None = None,
        panel_group_names: list[str] | None = None,
        **params,
    ) -> pl.DataFrame:
        """Soft vote: weighted average of class probabilities.

        Parameters
        ----------
        X : pl.DataFrame or None, default=None
            Exogenous features.
        forecasting_horizon : int or None, default=None
            Forecasting horizon.
        panel_group_names : list of str or None, default=None
            Panel group prefixes.
        **params : dict
            Routing parameters.

        Returns
        -------
        pl.DataFrame
            Averaged probability predictions.

        """
        predictions = []
        for _name, forecaster in self.forecasters_:
            y_proba = forecaster.predict_class_proba(  # ty: ignore[unresolved-attribute]
                X=X,
                forecasting_horizon=forecasting_horizon,
                panel_group_names=panel_group_names,
                **params,
            )
            predictions.append(y_proba)

        time_df = predictions[0].select(["observed_time", "time"])
        proba_cols = [c for c in predictions[0].columns if c not in ("observed_time", "time")]

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
        X: pl.DataFrame | None = None,
        forecasting_horizon: StrictInt | None = None,
        panel_group_names: list[str] | None = None,
        **params,
    ) -> pl.DataFrame:
        """Hard vote: majority vote converted to one-hot probabilities.

        Parameters
        ----------
        X : pl.DataFrame or None, default=None
            Exogenous features.
        forecasting_horizon : int or None, default=None
            Forecasting horizon.
        panel_group_names : list of str or None, default=None
            Panel group prefixes.
        **params : dict
            Routing parameters.

        Returns
        -------
        pl.DataFrame
            One-hot probability predictions from majority vote.

        """
        predictions = []
        for _name, forecaster in self.forecasters_:
            y_pred = forecaster.predict(  # ty: ignore[unresolved-attribute]
                X=X,
                forecasting_horizon=forecasting_horizon,
                panel_group_names=panel_group_names,
                **params,
            )
            predictions.append(y_pred)

        time_df = predictions[0].select(["observed_time", "time"])
        target_cols = [c for c in predictions[0].columns if c not in ("observed_time", "time")]

        result = time_df.clone()
        for target_col in target_cols:
            class_labels = self.classes_[target_col]
            n_rows = len(predictions[0])

            winners = []
            for row_idx in range(n_rows):
                votes = [pred[target_col][row_idx] for pred in predictions]
                vote_counts = Counter(votes)
                max_count = max(vote_counts.values())
                candidates = sorted(label for label, count in vote_counts.items() if count == max_count)
                winners.append(candidates[0])

            for label in class_labels:
                col_name = f"{target_col}_proba_{label}"
                proba_values = [1.0 if w == label else 0.0 for w in winners]
                result = result.with_columns(pl.Series(name=col_name, values=proba_values))

        return result

    def predict(
        self,
        X: pl.DataFrame | None = None,
        forecasting_horizon: StrictInt | None = None,
        panel_group_names: list[str] | None = None,
        **params,
    ) -> pl.DataFrame:
        """Generate argmax class predictions from the ensemble.

        Parameters
        ----------
        X : pl.DataFrame or None, default=None
            Exogenous features.
        forecasting_horizon : int or None, default=None
            Number of steps ahead. If ``None``, uses value from ``fit``.
        panel_group_names : list of str or None, default=None
            Panel group prefixes.
        **params : dict
            Metadata routing parameters.

        Returns
        -------
        pl.DataFrame
            Predictions with ``"observed_time"``, ``"time"``, and one
            column per target with the most likely class label.

        """
        check_is_fitted(self, ["forecasters_", "classes_"])

        if self.voting == "hard":
            return self._hard_vote_predict(
                X=X,
                forecasting_horizon=forecasting_horizon,
                panel_group_names=panel_group_names,
                **params,
            )

        y_proba = self.predict_class_proba(
            X=X,
            forecasting_horizon=forecasting_horizon,
            panel_group_names=panel_group_names,
            **params,
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
        time_cols = [c for c in ("observed_time", "time") if c in y_proba.columns]
        result = y_proba.select(time_cols)

        groups = self.panel_group_names_ or [None]

        for group in groups:
            for target_col, class_labels in self.classes_.items():
                if group is not None:
                    proba_cols = [f"{group}__{target_col}_proba_{label}" for label in class_labels]
                    out_col = f"{group}__{target_col}"
                else:
                    proba_cols = [f"{target_col}_proba_{label}" for label in class_labels]
                    out_col = target_col

                argmax_series = y_proba.select(pl.concat_list(proba_cols).list.arg_max().alias("_idx"))["_idx"]
                label_series = pl.Series(values=class_labels)
                result = result.with_columns(
                    argmax_series.map_elements(
                        lambda idx, _labels=label_series: _labels[idx],
                        return_dtype=pl.String,
                    ).alias(out_col),
                )

        return result

    def _hard_vote_predict(
        self,
        X: pl.DataFrame | None = None,
        forecasting_horizon: StrictInt | None = None,
        panel_group_names: list[str] | None = None,
        **params,
    ) -> pl.DataFrame:
        """Hard vote: majority vote of argmax predictions.

        Parameters
        ----------
        X : pl.DataFrame or None, default=None
            Exogenous features.
        forecasting_horizon : int or None, default=None
            Forecasting horizon.
        panel_group_names : list of str or None, default=None
            Panel group prefixes.
        **params : dict
            Routing parameters.

        Returns
        -------
        pl.DataFrame
            Majority vote predictions.

        """
        predictions = []
        for _name, forecaster in self.forecasters_:
            y_pred = forecaster.predict(  # ty: ignore[unresolved-attribute]
                X=X,
                forecasting_horizon=forecasting_horizon,
                panel_group_names=panel_group_names,
                **params,
            )
            predictions.append(y_pred)

        time_df = predictions[0].select(["observed_time", "time"])
        target_cols = [c for c in predictions[0].columns if c not in ("observed_time", "time")]

        result = time_df.clone()
        for target_col in target_cols:
            n_rows = len(predictions[0])
            winners = []
            for row_idx in range(n_rows):
                votes = [pred[target_col][row_idx] for pred in predictions]
                vote_counts = Counter(votes)
                max_count = max(vote_counts.values())
                candidates = sorted(label for label, count in vote_counts.items() if count == max_count)
                winners.append(candidates[0])

            result = result.with_columns(pl.Series(name=target_col, values=winners))

        return result

    def observe(
        self,
        y: pl.DataFrame,
        X: pl.DataFrame | None = None,
        panel_group_names: list[str] | None = None,
        **params,
    ) -> VotingClassProbaForecaster:
        """Observe new data on all surviving base forecasters.

        Parameters
        ----------
        y : pl.DataFrame
            New target observations.
        X : pl.DataFrame or None, default=None
            New exogenous observations.
        panel_group_names : list of str or None, default=None
            Panel group prefixes.
        **params : dict
            Metadata routing parameters.

        Returns
        -------
        self

        """
        check_is_fitted(self, ["forecasters_"])
        for _name, forecaster in self.forecasters_:
            forecaster.observe(y=y, X=X, panel_group_names=panel_group_names, **params)
        return self

    def rewind(
        self,
        y: pl.DataFrame,
        X: pl.DataFrame | None = None,
        panel_group_names: list[str] | None = None,
        **params,
    ) -> VotingClassProbaForecaster:
        """Rewind all surviving base forecasters.

        Parameters
        ----------
        y : pl.DataFrame
            Target data to rewind to.
        X : pl.DataFrame or None, default=None
            Exogenous data to rewind to.
        panel_group_names : list of str or None, default=None
            Panel group prefixes.
        **params : dict
            Metadata routing parameters.

        Returns
        -------
        self

        """
        check_is_fitted(self, ["forecasters_"])
        for _name, forecaster in self.forecasters_:
            forecaster.rewind(y=y, X=X, panel_group_names=panel_group_names, **params)
        return self

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
                .add(caller="predict_class_proba", callee="predict_class_proba"),
            )

        return router
