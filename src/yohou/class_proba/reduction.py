"""Implementation of reduction-based class-probability forecasters."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal, cast

import numpy as np
import polars as pl
from pydantic import StrictInt
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression

from yohou.base import BaseReductionForecaster, BaseTransformer
from yohou.utils._compat import HasMethods, StrOptions, _fit_context

from .base import BaseClassProbaForecaster

__all__ = ["ClassProbaReductionForecaster"]


class ClassProbaReductionForecaster(BaseReductionForecaster, BaseClassProbaForecaster):
    """Class-probability forecaster using sklearn classifiers on tabularized time series.

    Converts categorical time series forecasting to a tabular classification task.
    The target is encoded to integer codes before tabularization; predictions use
    ``predict_proba`` to return per-class probability distributions.

    Parameters
    ----------
    estimator : BaseEstimator, default=LogisticRegression()
        Classifier used to fit the tabularized data. Must implement
        ``fit``, ``predict``, and ``predict_proba``.
    reduction_strategy : {"direct", "multi-output"}, default="multi-output"
        Strategy for multi-step forecasting.
    target_transformer : BaseTransformer or None, default=None
        Transformer for target preprocessing.
    feature_transformer : BaseTransformer or None, default=None
        Transformer for feature engineering (typically LagTransformer).
    target_as_feature : {"transformed", "raw"} or None, default="transformed"
        Whether to include the target variable as a feature for reduction.
        If ``"transformed"``, the transformed target is used. If ``"raw"``,
        the raw target is used. If ``None``, the target is not included as
        a feature.
    panel_strategy : {"global", "multivariate"}, default="global"
        How to handle panel data. See `BaseForecaster` for details.
    n_jobs : int or None, default=None
        Number of jobs to run in parallel for the ``"direct"`` strategy.
        ``None`` means 1. ``-1`` means using all processors.

    Attributes
    ----------
    classes_ : dict[str, list[str]]
        Mapping from target column name to its class labels, discovered at
        fit time from the unique values in each target column.
    label_to_code_ : dict[str, dict[str, int]]
        Mapping from target column name to a dict mapping class labels to
        integer codes.
    estimator_ : BaseEstimator or list[BaseEstimator]
        Fitted sklearn classifier(s).

    Examples
    --------
    >>> import polars as pl
    >>> from datetime import datetime
    >>> from yohou.class_proba import ClassProbaReductionForecaster
    >>>
    >>> df = pl.DataFrame({
    ...     "time": pl.datetime_range(
    ...         start=datetime(2021, 1, 1),
    ...         end=datetime(2021, 1, 10),
    ...         interval="1d",
    ...         eager=True,
    ...     ),
    ...     "weather": ["sun", "sun", "rain", "rain", "cloud", "sun", "rain", "cloud", "sun", "rain"],
    ... })
    >>>
    >>> train = df[:8]
    >>> forecaster = ClassProbaReductionForecaster()
    >>> _ = forecaster.fit(y=train, forecasting_horizon=1)
    >>>
    >>> y_proba = forecaster.predict_class_proba(forecasting_horizon=1)
    >>> len(y_proba)
    1

    Notes
    -----
    The target columns are label-encoded to integer codes before
    tabularization. The encoding is stored in ``classes_`` and
    ``label_to_code_`` so that ``predict_class_proba`` can map the
    classifier's probability output back to the original class labels.

    See Also
    --------
    `BaseClassProbaForecaster` : Base class for class-probability forecasters.
    `PointReductionForecaster` : ML-based point forecaster.
    `BaseReductionForecaster` : Base class for reduction forecasters.

    """

    _parameter_constraints: dict = {
        **BaseReductionForecaster._parameter_constraints,
        **BaseClassProbaForecaster._parameter_constraints,
        "estimator": [HasMethods(["fit", "predict", "predict_proba"])],
        "reduction_strategy": [StrOptions({"direct", "multi-output"})],
    }

    _supports_panel = True

    def __init__(
        self,
        estimator: BaseEstimator = LogisticRegression(),
        reduction_strategy: Literal["direct", "multi-output"] = "multi-output",
        target_transformer: BaseTransformer | None = None,
        feature_transformer: BaseTransformer | None = None,
        target_as_feature: Literal["transformed", "raw"] | None = "transformed",
        n_jobs: int | None = None,
        panel_strategy: Literal["global", "multivariate"] = "global",
    ) -> None:
        BaseReductionForecaster.__init__(
            self,
            estimator=estimator,
            reduction_strategy=reduction_strategy,
            target_as_feature=target_as_feature,
            n_jobs=n_jobs,
            panel_strategy=panel_strategy,
        )

        BaseClassProbaForecaster.__init__(
            self,
            target_transformer=target_transformer,
            feature_transformer=feature_transformer,
            target_as_feature=target_as_feature,
            panel_strategy=panel_strategy,
        )

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(
        self,
        y: pl.DataFrame,
        X: pl.DataFrame | None = None,
        forecasting_horizon: StrictInt = 1,
        time_weight: Callable | pl.DataFrame | None = None,
        sample_weight_alignment: str = "first_step",
        **params,
    ) -> ClassProbaReductionForecaster:
        """Fit the forecaster to historical data.

        Encodes categorical targets to integer codes, tabularizes the time
        series, and fits the wrapped sklearn classifier.

        Parameters
        ----------
        y : pl.DataFrame
            Target time series with a ``"time"`` column (datetime) and one
            or more categorical (String, Categorical, or Enum) value columns.
        X : pl.DataFrame or None, default=None
            Exogenous features with a ``"time"`` column matching ``y``.
            If ``None``, no exogenous features are used.
        forecasting_horizon : int, default=1
            Number of time steps to forecast into the future.
        time_weight : callable, pl.DataFrame, or None, default=None
            Per-timestep weights for fitting.
        sample_weight_alignment : str, default="first_step"
            Strategy for converting ``time_weight`` to sklearn
            ``sample_weight`` across forecast horizons.
        **params : dict
            Metadata to route to nested estimators.

        Returns
        -------
        self
            The fitted forecaster instance.

        """
        forecasting_horizon = self._validate_fit_params(forecasting_horizon)

        # Discover classes from y before _pre_fit (which may transform y)
        self.classes_: dict[str, list[str]] = {}
        self.label_to_code_: dict[str, dict[str, int]] = {}
        for col in y.columns:
            if col == "time":
                continue
            unique_vals = sorted(y[col].drop_nulls().unique().cast(pl.String).to_list())
            self.classes_[col] = unique_vals
            self.label_to_code_[col] = {label: i for i, label in enumerate(unique_vals)}

        # Encode target columns to integer codes for tabularization
        y_encoded = self._encode_target(y)

        y_t, X_t = self._pre_fit(
            y=y_encoded,
            X=X,
            forecasting_horizon=forecasting_horizon,
        )

        self.estimator_ = self._estimator_fit_one(
            y_t,
            X_t,
            forecasting_horizon,
            time_weight=time_weight,
            sample_weight_alignment=sample_weight_alignment,
            estimator_fit_params=params,
        )

        return self

    def _encode_target(self, y: pl.DataFrame) -> pl.DataFrame:
        """Encode categorical target columns to integer codes.

        Parameters
        ----------
        y : pl.DataFrame
            Target data with categorical columns.

        Returns
        -------
        pl.DataFrame
            Target data with categorical columns replaced by integer codes.

        """
        exprs = []
        for col in y.columns:
            if col == "time":
                continue
            mapping = self.label_to_code_[col]
            # Cast to String first to handle Categorical/Enum/String uniformly,
            # then replace labels with integer codes.
            exprs.append(pl.col(col).cast(pl.String).replace_strict(mapping, return_dtype=pl.Float64).alias(col))
        return y.with_columns(exprs)

    def _predict_class_proba_one(
        self,
        panel_group_names: list[str],
        **params,
    ) -> pl.DataFrame:
        """Produce probability forecasts for one fit-horizon block.

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
        y_proba = self._estimator_predict_proba_one(
            self.estimator_,
            panel_group_names=panel_group_names,
        )
        y_proba = self._add_time_columns(y_proba)
        return y_proba

    def _estimator_predict_proba_one(
        self,
        estimator: BaseEstimator | list[BaseEstimator],
        panel_group_names: list[str],
    ) -> pl.DataFrame:
        """Dispatch estimator probability prediction to the strategy-specific method.

        Parameters
        ----------
        estimator : BaseEstimator or list[BaseEstimator]
            Fitted estimator(s).
        panel_group_names : list of str
            Panel group names to predict for.

        Returns
        -------
        pl.DataFrame
            Probability predictions.

        """
        if self.reduction_strategy == "direct":
            assert isinstance(estimator, list)
            return self._estimator_predict_proba_direct(
                cast(list[BaseEstimator], estimator), panel_group_names
            )
        assert isinstance(estimator, BaseEstimator)
        return self._estimator_predict_proba_multi_output(estimator, panel_group_names)

    def _estimator_predict_proba_multi_output(
        self,
        estimator: BaseEstimator,
        panel_group_names: list[str],
    ) -> pl.DataFrame:
        """Generate probability predictions using a fitted multi-output estimator.

        Parameters
        ----------
        estimator : BaseEstimator
            Fitted sklearn classifier.
        panel_group_names : list of str
            Panel group names to predict for.

        Returns
        -------
        pl.DataFrame
            Probability predictions.

        """
        if self.panel_group_names_ is None:
            X_tab = self._get_predict_features()
            return self._predict_proba_and_reshape(estimator, X_tab)

        y_pred_dict: dict[str, pl.DataFrame] = {}
        for panel_group_name in panel_group_names:
            X_tab = self._get_predict_features(panel_group_name)
            y_pred_dict[panel_group_name] = self._predict_proba_and_reshape(estimator, X_tab, panel_group_name)
        return pl.concat(list(y_pred_dict.values()), how="horizontal")

    def _estimator_predict_proba_direct(
        self,
        estimators: list[BaseEstimator],
        panel_group_names: list[str],
    ) -> pl.DataFrame:
        """Generate probability predictions using H independent direct estimators.

        Parameters
        ----------
        estimators : list[BaseEstimator]
            H fitted estimators, one per horizon step.
        panel_group_names : list of str
            Panel group names to predict for.

        Returns
        -------
        pl.DataFrame
            Probability predictions.

        """
        if self.panel_group_names_ is None:
            X_tab = self._get_predict_features()
            frames = []
            for estimator in estimators:
                frames.append(self._predict_proba_and_reshape_single_step(estimator, X_tab))
            return pl.concat(frames)

        y_pred_dict: dict[str, list[pl.DataFrame]] = {g: [] for g in panel_group_names}
        for panel_group_name in panel_group_names:
            X_tab = self._get_predict_features(panel_group_name)
            for estimator in estimators:
                y_pred_dict[panel_group_name].append(
                    self._predict_proba_and_reshape_single_step(estimator, X_tab, panel_group_name)
                )
        return pl.concat(
            [pl.concat(v) for v in y_pred_dict.values()],
            how="horizontal",
        )

    def _predict_proba_and_reshape(
        self,
        estimator: BaseEstimator,
        X_tab: np.ndarray,
        panel_group_name: str | None = None,
    ) -> pl.DataFrame:
        """Call predict_proba and reshape to probability DataFrame.

        For multi-output, the estimator predicts all H steps at once.
        Each step has n_targets columns; each target column's integer
        prediction maps to n_classes probability columns.

        Parameters
        ----------
        estimator : BaseEstimator
            Fitted classifier.
        X_tab : np.ndarray
            Feature array of shape ``(1, n_features)``.
        panel_group_name : str or None
            Panel group prefix for column naming.

        Returns
        -------
        pl.DataFrame
            Probability DataFrame with ``{target}_proba_{class}`` columns.

        """
        assert self.local_y_t_schema_ is not None
        y_cols = list(self.local_y_t_schema_.keys())
        fh = self.fit_forecasting_horizon_
        n_targets = len(y_cols)

        # For multi-output with H*n_targets outputs, sklearn wraps in
        # MultiOutputClassifier or similar. We handle both cases.
        proba = estimator.predict_proba(X_tab)  # type: ignore[attr-defined]

        # Build result row by row (one row per forecast step)
        result_data: dict[str, list[Any]] = {}
        for target_col in y_cols:
            for label in self.classes_[target_col]:
                col_name = f"{target_col}_proba_{label}"
                if panel_group_name is not None:
                    col_name = f"{panel_group_name}__{col_name}"
                result_data[col_name] = []

        if isinstance(proba, list):
            # MultiOutputClassifier returns list of arrays, one per output
            # Each array has shape (1, n_classes_for_that_output)
            # Outputs are ordered: target1_step1, target2_step1, ..., target1_step2, ...
            for step in range(fh):
                for t_idx, target_col in enumerate(y_cols):
                    output_idx = step * n_targets + t_idx
                    step_proba = proba[output_idx][0]  # shape (n_classes,)
                    classes_for_target = self.classes_[target_col]
                    for c_idx, label in enumerate(classes_for_target):
                        col_name = f"{target_col}_proba_{label}"
                        if panel_group_name is not None:
                            col_name = f"{panel_group_name}__{col_name}"
                        if c_idx < len(step_proba):
                            result_data[col_name].append(float(step_proba[c_idx]))
                        else:
                            result_data[col_name].append(0.0)
        else:
            # Single-output classifier or single-target: proba shape (1, n_classes)
            # For multi-step, we need H rows from a single prediction. If
            # multi-output is used, the model predicts H*n_targets columns
            # and predict_proba returns a single array.
            # Fall back: treat as single-step single-target
            assert n_targets == 1
            target_col = y_cols[0]
            classes_for_target = self.classes_[target_col]

            if proba.ndim == 2 and proba.shape[0] == 1:
                # Single row prediction, map to fh=1
                step_proba = proba[0]
                for c_idx, label in enumerate(classes_for_target):
                    col_name = f"{target_col}_proba_{label}"
                    if panel_group_name is not None:
                        col_name = f"{panel_group_name}__{col_name}"
                    result_data[col_name].append(float(step_proba[c_idx]) if c_idx < len(step_proba) else 0.0)
                # If fh > 1, replicate (the recursive loop in predict_class_proba handles stepping)
                # This branch should only be reached with fh=1 in multi-output mode.

        return pl.DataFrame(result_data)

    def _predict_proba_and_reshape_single_step(
        self,
        estimator: BaseEstimator,
        X_tab: np.ndarray,
        panel_group_name: str | None = None,
    ) -> pl.DataFrame:
        """Call predict_proba for a single-step direct estimator.

        Parameters
        ----------
        estimator : BaseEstimator
            Fitted single-step classifier.
        X_tab : np.ndarray
            Feature array of shape ``(1, n_features)``.
        panel_group_name : str or None
            Panel group prefix for column naming.

        Returns
        -------
        pl.DataFrame
            Single-row probability DataFrame.

        """
        assert self.local_y_t_schema_ is not None
        y_cols = list(self.local_y_t_schema_.keys())

        proba = estimator.predict_proba(X_tab)  # type: ignore[attr-defined]

        result_data: dict[str, list[float]] = {}

        if isinstance(proba, list):
            # Multiple targets
            for t_idx, target_col in enumerate(y_cols):
                step_proba = proba[t_idx][0]
                for c_idx, label in enumerate(self.classes_[target_col]):
                    col_name = f"{target_col}_proba_{label}"
                    if panel_group_name is not None:
                        col_name = f"{panel_group_name}__{col_name}"
                    result_data[col_name] = [float(step_proba[c_idx]) if c_idx < len(step_proba) else 0.0]
        else:
            # Single target
            assert len(y_cols) == 1
            target_col = y_cols[0]
            step_proba = proba[0]
            for c_idx, label in enumerate(self.classes_[target_col]):
                col_name = f"{target_col}_proba_{label}"
                if panel_group_name is not None:
                    col_name = f"{panel_group_name}__{col_name}"
                result_data[col_name] = [float(step_proba[c_idx]) if c_idx < len(step_proba) else 0.0]

        return pl.DataFrame(result_data)
