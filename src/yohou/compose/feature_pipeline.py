"""FeaturePipeline and FeatureUnion utilities for chaining transformers."""

from collections.abc import Iterator
from copy import deepcopy
from typing import Any, cast

import polars as pl
from joblib import Memory
from sklearn.pipeline import Pipeline as sklearn_Pipeline
from sklearn.utils import (
    Bunch,
)
from sklearn.utils.metadata_routing import (
    MetadataRouter,
    MethodMapping,
    process_routing,
)
from sklearn.utils.metaestimators import available_if
from sklearn.utils.validation import (
    check_is_fitted,
)

from yohou.base import BaseActualTransformer
from yohou.base.transformer import _BaseTransformer
from yohou.base.utils import _require_actual_memory_api
from yohou.compose.column_transformer import ColumnTransformer
from yohou.compose.feature_union import FeatureUnion
from yohou.compose.utils import check_homogeneous_kinds, common_kind, index_columns
from yohou.utils import Tags
from yohou.utils._compat import (
    HasMethods,
    Hidden,
    _BaseComposition,
    _fit_context,
    _print_elapsed_time,
    _raise_for_params,
)
from yohou.utils.validate_data import validate_transformer_data

from .utils import _observe_transform_one, _rewind_transform_one

__all__ = [
    "ColumnTransformer",
    "FeaturePipeline",
    "FeatureUnion",
]


# TODO: Could transform_input (sklearn's Pipeline feature for routing metadata
# through pipeline steps) make sense for forecasting? E.g., routing y vs X to
# different steps, or passing validation sets through the pipeline.
class FeaturePipeline(BaseActualTransformer, _BaseComposition):
    """
    A sequence of time series transformers.

    `FeaturePipeline` allows you to sequentially apply a list of time series
    transformers to preprocess the data.

    Steps of the pipeline must be instances of `BaseActualTransformer`. Non-last
    steps must also implement `transform`. The pipeline dispatches
    `observe_transform()` internally; steps do not need to expose a bare
    `observe` method.

    The purpose of the pipeline is to assemble several steps that can be
    cross-validated together while setting different parameters. For this, it
    enables setting parameters of the various steps using their names and the
    parameter name separated by a `'__'`, as in the example below. A step's
    estimator may be replaced entirely by setting the parameter with its name
    to another estimator, or a transformer removed by setting it to
    `'passthrough'` or `None`.

    Parameters
    ----------
    steps : list of tuples
        List of (name of step, estimator) tuples that are to be chained in
        sequential order. To be compatible with the scikit-learn API, all steps
        must define `fit`. All non-last steps must also define `transform`. See
        [Combining Estimators](https://scikit-learn.org/stable/modules/compose.html) for more details.

    memory : str or object with the joblib.Memory interface, default=None
        Used to cache the fitted transformers of the pipeline. The last step
        will never be cached, even if it is a transformer. By default, no
        caching is performed. If a string is given, it is the path to the
        caching directory. Enabling caching triggers a clone of the transformers
        before fitting. Therefore, the transformer instance given to the
        pipeline cannot be inspected directly. Use the attribute ``named_steps``
        or ``steps`` to inspect estimators within the pipeline. Caching the
        transformers is advantageous when fitting is time consuming.

    verbose : bool, default=False
        If True, the time elapsed while fitting each step will be printed as it
        is completed.

    Attributes
    ----------
    named_steps : `Bunch`
        Dictionary-like object, with the following attributes.
        Read-only attribute to access any step parameter by user given name.
        Keys are step names and values are steps parameters.

    n_features_in_ : int
        Number of features seen during ``fit``. Only defined if the
        underlying first estimator in `steps` exposes such an attribute
        when fit.

    feature_names_in_ : list[str]
        List of feature names seen during ``fit`` of the first step. Only
        defined if the underlying estimator exposes such an attribute when fit.

    See Also
    --------
    - [`sklearn.pipeline.Pipeline`][sklearn.pipeline.Pipeline] : Underlying scikit-learn pipeline class.
    - [`BaseActualTransformer`][yohou.base.transformer.BaseActualTransformer] : Base class for time series transformers.
    - [`FeatureUnion`][yohou.compose.feature_union.FeatureUnion] : Parallel transformer combination.
    - [`ColumnTransformer`][yohou.compose.column_transformer.ColumnTransformer] : Apply transformers to specific columns.

    Notes
    -----
    All input data must include a `time` column with datetime values. The `time`
    column is preserved through all transformations.

    The `observation_horizon` property accumulates across all steps, returning
    the sum of all transformer observation horizons. This indicates the total
    amount of historical data required by the pipeline. The sum is correct
    because steps run sequentially, each consuming the previous step's output.
    This contrasts with `FeatureUnion`, whose `observation_horizon` takes the
    maximum across its parallel transformers.

    Supports time series-specific `observe()` method for incremental learning,
    allowing the pipeline to incorporate new observations without full retraining.

    The final step can be a forecaster, enabling end-to-end forecasting pipelines
    that transform features and generate predictions.

    Examples
    --------
    >>> import polars as pl
    >>> from datetime import datetime, timedelta
    >>> from yohou.compose import FeaturePipeline
    >>> from yohou.stationarity import SeasonalDifferencing
    >>> from yohou.preprocessing import LagTransformer
    >>>
    >>> # Create sample weekly time series data (52 weeks)
    >>> time = pl.datetime_range(
    ...     start=datetime(2023, 1, 1),
    ...     end=datetime(2023, 1, 1) + timedelta(weeks=51),
    ...     interval="1w",
    ...     eager=True,
    ... )
    >>> data = pl.DataFrame({"time": time, "sales": range(1, 53)})
    >>>
    >>> # Example 1: Create a sequential preprocessing pipeline
    >>> pipe = FeaturePipeline([
    ...     ("deseason", SeasonalDifferencing(seasonality=4)),
    ...     ("lags", LagTransformer(lag=[1, 2, 3])),
    ... ])
    >>>
    >>> # Example 2: Access individual steps by name
    >>> pipe.named_steps["deseason"]  # doctest: +ELLIPSIS
    SeasonalDifferencing(...)
    >>>
    >>> # Example 3: Access individual steps by position
    >>> pipe[0]  # doctest: +ELLIPSIS
    SeasonalDifferencing(...)

    """

    # BaseEstimator interface
    _required_parameters = ["steps"]

    # sklearn's Pipeline reads ``transform_input`` in ``_get_metadata_for_step``.
    # Yohou does not expose it as a constructor parameter (it is not a tunable
    # estimator param), so it is a class attribute fixed to ``None``: this keeps
    # sklearn's short-circuit working without leaking into ``get_params``.
    transform_input = None

    _parameter_constraints: dict[str, Any] = {
        "steps": [list, Hidden(tuple)],
        "memory": [None, str, HasMethods(["cache"])],
        "verbose": ["boolean"],
    }

    def __init__(
        self,
        steps: list[tuple[str, Any]],
        *,
        memory: None | Memory | str = None,
        verbose: bool = False,
    ) -> None:
        self.steps = steps
        self.memory = memory
        self.verbose = verbose

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Get parameters for this estimator.

        Parameters
        ----------
        deep : bool, default=True
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.

        Returns
        -------
        params : dict[str, Any]
            Parameter names mapped to their values.

        """
        return _BaseComposition._get_params(self, attr="steps", deep=deep)

    def set_params(self, **params: Any) -> "FeaturePipeline":
        """Set the parameters of this estimator.

        Parameters
        ----------
        **params : dict
            Estimator parameters.

        Returns
        -------
        self : FeaturePipeline
            FeaturePipeline instance.

        """
        _BaseComposition._set_params(self, attr="steps", **params)
        return self

    def _iter(
        self,
        with_final: bool = True,
        filter_passthrough: bool = True,
    ) -> Iterator[tuple[int, str, Any]]:
        """Generate (idx, name, trans) tuples from self.steps.

        Parameters
        ----------
        with_final : bool, default=True
            Include the final estimator.

        filter_passthrough : bool, default=True
            Filter out 'passthrough' steps.

        Yields
        ------
        idx : int
            Step index.
        name : str
            Step name.
        trans : Any
            Step transformer.

        """
        # Cast self to sklearn_Pipeline for parent method call.
        # FeaturePipeline extends sklearn_Pipeline, so this is safe.
        return sklearn_Pipeline._iter(
            cast(sklearn_Pipeline, self),
            with_final=with_final,
            filter_passthrough=filter_passthrough,
        )

    def __len__(self) -> int:
        """Return the length of the FeaturePipeline.

        Returns
        -------
        length : int
            Number of steps in the pipeline.

        """
        return len(self.steps)

    def __getitem__(self, ind: int | str | slice) -> Any:
        """Return a sub-pipeline or a single estimator in the pipeline.

        Parameters
        ----------
        ind : int, str, or slice
            Index, name, or slice of the step to retrieve.

        Returns
        -------
        estimator : Any
            The estimator or sub-pipeline.

        """
        if isinstance(ind, slice):
            if ind.step is not None:
                raise ValueError("FeaturePipeline slicing only supports a step of 1")
            return self.__class__(steps=self.steps[ind], memory=self.memory, verbose=self.verbose)
        elif isinstance(ind, int):
            _, est = self.steps[ind]
            return est
        else:
            # String case - get by name
            return self.named_steps[ind]

    def _fit(self, X: pl.DataFrame, y: pl.DataFrame | None, routed_params: Any, **kwargs: Any) -> pl.DataFrame:  # ty: ignore[invalid-method-override]
        """Fit the pipeline.

        Parameters
        ----------
        X : pl.DataFrame
            Training data.

        y : pl.DataFrame | None
            Training targets.

        routed_params : Any
            Routed parameters.

        **kwargs : Any
            Additional keyword arguments forwarded to the parent ``_fit``
            (e.g. ``callback_ctx`` in scikit-learn >= 1.9).

        Returns
        -------
        X_t : pl.DataFrame
            Transformed data.

        """
        if "callback_ctx" not in kwargs:
            # sklearn >= 1.9 requires callback_ctx in Pipeline._fit
            _init = getattr(sklearn_Pipeline, "_init_callback_context", None)
            if _init is not None:
                kwargs["callback_ctx"] = _init(self, max_subtasks=len(self.steps))
        return sklearn_Pipeline._fit(self, X, y, routed_params, **kwargs)  # ty: ignore[invalid-argument-type]

    @property
    def named_steps(self) -> Bunch:
        """Access the steps by name.

        Returns
        -------
        named_steps : Bunch
            Dictionary-like object with step names as keys.

        """
        return sklearn_Pipeline.named_steps.fget(self)  # ty: ignore[invalid-argument-type]

    @property
    def _final_estimator(self) -> Any:
        """Get the final estimator.

        Returns
        -------
        estimator : Any
            The final estimator in the pipeline.

        """
        return sklearn_Pipeline._final_estimator.fget(self)  # ty: ignore[invalid-argument-type]

    def _log_message(self, step_idx: int) -> str:
        """Get log message for a step.

        Parameters
        ----------
        step_idx : int
            Index of the step.

        Returns
        -------
        message : str
            Log message.

        """
        return sklearn_Pipeline._log_message(self, step_idx)  # ty: ignore[invalid-argument-type]

    def _check_method_params(self, method: str, props: dict[str, Any]) -> Any:
        """Check and route method parameters.

        Parameters
        ----------
        method : str
            Method name.

        props : dict[str, Any]
            Properties to check.

        Returns
        -------
        routed_params : Any
            Routed parameters.

        """
        # Validate params before routing (sklearn pattern)
        _raise_for_params(props, self, method)
        return process_routing(self, method, **props)

    def get_feature_names_out(self, input_features: list[str] | None = None) -> Any:
        """Get output feature names for transformation.

        Parameters
        ----------
        input_features : list[str] | None, default=None
            Input feature names.

        Returns
        -------
        feature_names_out : Any
            Output feature names.

        """
        return super().get_feature_names_out(input_features)

    @property
    def n_features_in_(self) -> int:
        """Number of features seen during fit.

        Returns
        -------
        n_features_in_ : int
            Number of input features.

        """
        return sklearn_Pipeline.n_features_in_.fget(self)  # ty: ignore[invalid-argument-type]

    @property
    def feature_names_in_(self) -> Any:
        """Names of features seen during fit.

        Returns
        -------
        feature_names_in_ : Any
            Names of input features.

        """
        return sklearn_Pipeline.feature_names_in_.fget(self)  # ty: ignore[invalid-argument-type]

    def __sklearn_is_fitted__(self) -> bool:
        """Check if the pipeline is fitted.

        Returns
        -------
        is_fitted : bool
            True if the pipeline is fitted.

        """
        return sklearn_Pipeline.__sklearn_is_fitted__(self)  # ty: ignore[invalid-argument-type]

    def _sk_visual_block_(self) -> Any:
        """Get visual block representation.

        Returns
        -------
        visual_block : Any
            Visual block representation.

        """
        # Cast self to sklearn_Pipeline for parent method call.
        # FeaturePipeline extends sklearn_Pipeline, so this is safe.
        return sklearn_Pipeline._sk_visual_block_(cast(sklearn_Pipeline, self))

    def _get_metadata_for_step(self, **params: Any) -> Any:
        """Get metadata for a specific step.

        Parameters
        ----------
        **params : dict
            Arguments passed from sklearn's _fit method.

        Returns
        -------
        metadata : Any
            Metadata for the step.

        """
        # Cast self to sklearn_Pipeline for parent method call.
        # FeaturePipeline extends sklearn_Pipeline, so this is safe.
        return sklearn_Pipeline._get_metadata_for_step(cast(sklearn_Pipeline, self), **params)

    def __sklearn_tags__(self) -> Tags:
        """Get estimator tags.

        Returns
        -------
        Tags
            Estimator tags with yohou-specific attributes.

        """
        tags = super().__sklearn_tags__()

        # Aggregate tags from steps (static capability check)
        if hasattr(self, "steps") and self.steps is not None:
            transformers = [t for _, t in self.steps if t != "passthrough" and t is not None]
            if transformers:
                assert tags.transformer_tags is not None
                assert tags.input_tags is not None
                # Stateful if any step is stateful
                tags.transformer_tags.stateful = any(
                    t.__sklearn_tags__().transformer_tags.stateful for t in transformers
                )

                # Invertible if all steps are invertible
                tags.transformer_tags.invertible = all(
                    t.__sklearn_tags__().transformer_tags.invertible for t in transformers
                )

                # Tolerates an irregular time grid only if every step does. Inside the
                # ``if transformers:`` guard so an empty pipeline keeps the base default
                # rather than ``all([]) == True``.
                tags.transformer_tags.accepts_irregular_grid = all(
                    t.__sklearn_tags__().transformer_tags.accepts_irregular_grid for t in transformers
                )

                # min_value is the one of the first transformer
                tags.input_tags.min_value = transformers[0].__sklearn_tags__().input_tags.min_value

                # A pipeline inherits its kind from its (homogeneous) steps.
                tags.transformer_tags.kind = common_kind([(name, t) for name, t in self.steps])

        return tags

    @property
    def min_vintage_rows(self) -> int:
        """Get the smallest vintage length that yields non-empty output.

        Steps run in sequence, each consuming rows from what the previous one left,
        so the requirements accumulate rather than taking a maximum. This parallels
        ``observation_horizon``, which sums across steps here while
        ``FeatureUnion`` takes the max of its parallel branches.

        The value is a safe upper bound rather than a tight one. A step reporting a
        minimum of ``m`` consumes at most ``m - 1`` rows, so
        ``sum(m_i - 1) + 1`` never understates the requirement, and it is exact for
        a single step. A chain of stateless steps is the loose case: each reports
        ``2`` while consuming nothing, so the sum grows where the true minimum stays
        at ``2``. Over-reporting fails a fit that would have worked, which is
        recoverable; under-reporting admits a configuration that silently empties
        every vintage, which is the failure the minimum exists to prevent.

        Only forecast-kind children report a minimum. On an actual-kind composition
        this reports ``1`` and is never consulted.

        Returns
        -------
        int
            Smallest vintage length yielding non-empty output, at least ``1``.

        Raises
        ------
        NotFittedError
            If the composition has not been fitted yet.

        """
        check_is_fitted(self)
        consumed = 0
        for _, t in self.steps:
            if t != "passthrough" and t is not None and hasattr(t, "min_vintage_rows"):
                consumed += t.min_vintage_rows - 1
        return consumed + 1 if consumed else 1

    @property
    def observation_horizon(self) -> int:
        """Get cumulative observation horizon across all steps.

        Returns
        -------
        int
            Total observation horizon needed.

        Raises
        ------
        NotFittedError
            If the pipeline has not been fitted yet.

        """
        check_is_fitted(self)

        observation_horizon = 0
        for _, t in self.steps:
            if t != "passthrough" and t is not None and hasattr(t, "observation_horizon"):
                observation_horizon += t.observation_horizon

        return observation_horizon

    def _update_X_observed(self, X: pl.DataFrame) -> None:
        """Update pipeline-level observation tracking.

        Unlike individual transformers, the pipeline does not store
        ``observation_horizon`` rows of raw input.  Each step manages
        its own memory buffer.  We only keep the last row for temporal
        continuity checks in `observe` and record
        ``observed_time_``.

        Parameters
        ----------
        X : pl.DataFrame
            Raw pipeline input.

        """
        # Store one row for continuity checks, not the full observation_horizon
        self._X_observed = X[-1:] if len(X) > 0 else X[:0]
        if len(X) > 0:
            self.observed_time_ = X["time"][-1]

    def rewind(self, X: pl.DataFrame) -> "FeaturePipeline":
        """Rewind the pipeline state to the provided data.

        Propagates ``rewind_transform()`` through each step in order.
        Each step transforms its input (dropping warmup rows) and rewinds
        its own memory, then passes the transformed data to the next step.

        Parameters
        ----------
        X : pl.DataFrame
            Input time series.

        Returns
        -------
        self

        Raises
        ------
        NotFittedError
            If the pipeline has not been fitted yet.

        """
        _require_actual_memory_api(self, "rewind")
        check_is_fitted(self)

        # Propagate rewind_transform through each step:
        # step.rewind_transform(Xt) = step.transform(Xt) + step.rewind(Xt)
        Xt = X
        for _, _, transform in self._iter():
            Xt = transform.rewind_transform(Xt)

        # Update pipeline-level tracking
        self._update_X_observed(X)

        return self

    def observe(self, X: pl.DataFrame) -> "FeaturePipeline":
        """Observe new data and update step memory.

        Propagates ``observe_transform()`` through each step in order.
        Each step uses its own memory to provide context, transforms the
        data, updates its observation buffer, then passes the result to
        the next step.

        Parameters
        ----------
        X : pl.DataFrame
            New observations to incorporate.

        Returns
        -------
        self

        Raises
        ------
        NotFittedError
            If the pipeline has not been fitted yet.
        ValueError
            If ``X`` is not temporally continuous with previous
            observations.

        """
        _require_actual_memory_api(self, "observe")
        check_is_fitted(self)
        # Schema + continuity validation at pipeline level
        X = validate_transformer_data(self, X=X, reset=False, check_continuity=True)

        # Propagate observe_transform through each step:
        # step.observe_transform(Xt) = concat(_X_observed, Xt) -> transform -> observe
        Xt = X
        for _, _, transform in self._iter():
            Xt = transform.observe_transform(Xt)

        # Update pipeline-level tracking
        self._update_X_observed(X)

        return self

    def _validate_steps(self) -> None:
        """Validate that all steps are BaseActualTransformer instances.

        Raises
        ------
        TypeError
            If any step is not a BaseActualTransformer or 'passthrough'.

        """
        names, transformers = zip(*self.steps, strict=False)

        # validate names
        self._validate_names(names)

        for t in transformers:
            if t is None or t == "passthrough":
                continue
            if not isinstance(t, _BaseTransformer):
                raise TypeError(
                    "All steps should be instances of `BaseActualTransformer` or "
                    "`BaseForecastTransformer` or be the string 'passthrough' "
                    f"'{t}' (type {type(t)}) doesn't"
                )

        # A pipeline must be homogeneous in kind (all actual or all forecast).
        check_homogeneous_kinds([(name, t) for name, t in self.steps], "FeaturePipeline")

    @_fit_context(
        # estimators in FeaturePipeline.steps are not validated yet
        prefer_skip_nested_validation=False
    )
    def fit(self, X: pl.DataFrame, y: pl.DataFrame | None = None, **params: Any) -> "FeaturePipeline":
        """Fit the model.

        Fit all the transformers one after the other and sequentially transform the
        data. Finally, fit the transformed data using the final estimator.

        Parameters
        ----------
        X : pl.DataFrame
            Training data. Must fulfill input requirements of first step of the
            pipeline.

        y : pl.DataFrame or None, default=None
            Training targets. Must fulfill label requirements for all steps of
            the pipeline.

        **params : dict of str -> object
            - If `enable_metadata_routing=False` (default):

                Parameters passed to the ``fit`` method of each step, where
                each parameter name is prefixed such that parameter ``p`` for step
                ``s`` has key ``s__p``.

            - If `enable_metadata_routing=True`:

                Parameters requested and accepted by steps. Each step must have
                requested certain metadata for these parameters to be forwarded to
                them.

        Returns
        -------
        self : FeaturePipeline
            FeaturePipeline with fitted steps.
        """
        routed_params = self._check_method_params(method="fit", props=params)
        X_t = self._fit(X, y, routed_params, raw_params=params)
        with _print_elapsed_time("FeaturePipeline", self._log_message(len(self.steps) - 1)):
            if self._final_estimator != "passthrough":
                last_step_params = routed_params[self.steps[-1][0]]
                self._final_estimator.fit(X_t, y, **last_step_params["fit"])

        # Set pipeline-level attributes for rewind/observe validation.
        # Individual steps are already fitted with their own attributes.
        self.X_schema_ = dict(X.select(pl.exclude(index_columns(X))).schema)
        self._update_X_observed(X)

        return self

    @_fit_context(
        # estimators in FeaturePipeline.steps are not validated yet
        prefer_skip_nested_validation=False
    )
    def fit_transform(self, X: pl.DataFrame, y: pl.DataFrame | None = None, **params: Any) -> pl.DataFrame:
        """Fit the model and transform with the final estimator.

        Fit all the transformers one after the other and sequentially transform
        the data. Only valid if the final estimator either implements
        `fit_transform` or `fit` and `transform`.

        Parameters
        ----------
        X : pl.DataFrame
            Training data. Must fulfill input requirements of first step of the
            pipeline.

        y : pl.DataFrame or None, default=None
            Training targets. Must fulfill label requirements for all steps of
            the pipeline.

        **params : dict of str -> object
            - If `enable_metadata_routing=False` (default):

                Parameters passed to the ``fit`` method of each step, where
                each parameter name is prefixed such that parameter ``p`` for step
                ``s`` has key ``s__p``.

            - If `enable_metadata_routing=True`:

                Parameters requested and accepted by steps. Each step must have
                requested certain metadata for these parameters to be forwarded to
                them.

        Returns
        -------
        X_t : pl.DataFrame
            Transformed data.
        """
        routed_params = self._check_method_params(method="fit_transform", props=params)
        X_t = self._fit(X, y, routed_params, raw_params=params)

        last_step = self._final_estimator
        with _print_elapsed_time("FeaturePipeline", self._log_message(len(self.steps) - 1)):
            if last_step == "passthrough":
                return X_t

            last_step_params = routed_params[self.steps[-1][0]]
            result = last_step.fit_transform(X_t, y, **last_step_params["fit_transform"])
            return result

    def transform(self, X: pl.DataFrame, **params: Any) -> pl.DataFrame:
        """Transform the data, and apply `transform` with the final estimator.

        Call `transform` of each transformer in the pipeline. The transformed
        data are finally passed to the final estimator that calls
        `transform` method. Only valid if the final estimator
        implements `transform`.

        This also works where final estimator is `None` in which case all prior
        transformations are applied.

        Parameters
        ----------
        X : pl.DataFrame
            Data to transform. Must fulfill input requirements of first step
            of the pipeline.

        **params : dict of str -> object
            Parameters requested and accepted by steps. Each step must have
            requested certain metadata for these parameters to be forwarded to
            them.

        Returns
        -------
        X_t : pl.DataFrame
            Transformed data.
        """
        _raise_for_params(params, self, "transform")

        # not branching here since params is only available if
        # enable_metadata_routing=True
        routed_params = process_routing(self, "transform", **params)
        X_t = X
        for _, name, transform in self._iter():
            X_t = transform.transform(X_t, **routed_params[name].transform)
        return X_t

    def observe_transform(self, X: pl.DataFrame, **params: Any) -> pl.DataFrame:
        """Observe and transform the data through the pipeline.

        This method atomically observes each transformer with new data and
        transforms it in sequence. Each step transforms using its own
        pre-observe memory and then updates its state before passing the result
        to the next step. The pipeline does not take a global snapshot; state is
        updated step-by-step. This is more efficient and correct than calling
        observe() then transform() separately.

        Parameters
        ----------
        X : pl.DataFrame
            New data to observe and transform. Must fulfill input requirements
            of first step of the pipeline.

        **params : dict of str -> object
            Parameters routed to the `transform` methods of the steps. Each step must
            have requested certain metadata via `set_transform_request()` for these
            parameters to be forwarded to them.

        Returns
        -------
        X_t : pl.DataFrame
            Transformed data corresponding to the new input rows.

        """
        _require_actual_memory_api(self, "observe_transform")
        _raise_for_params(params, self, "observe_transform")

        routed_params = process_routing(self, "observe_transform", **params)

        # Transform sequentially through all steps using their observe_transform
        # Each transformer handles its own memory management internally
        X_t = X
        for _, name, transform in self._iter():
            X_t = _observe_transform_one(transform, X_t, None, None, routed_params[name])

        return X_t

    def rewind_transform(self, X: pl.DataFrame, **params: Any) -> pl.DataFrame:
        """Rewind and transform the data through the pipeline.

        This method applies rewind_transform semantics to each transformer in sequence:
        transforms from scratch without using pre-existing memory, discards warmup rows,
        and rewinds the internal state with the input data.

        Parameters
        ----------
        X : pl.DataFrame
            Data to transform and use for rewinding state. Must fulfill input requirements
            of first step of the pipeline.

        **params : dict of str -> object
            Parameters routed to the `rewind_transform` methods of the steps. Each step must
            have requested certain metadata via `set_rewind_transform_request()` for these
            parameters to be forwarded to them.

        Returns
        -------
        X_t : pl.DataFrame
            Transformed data with warmup rows discarded.

        """
        _require_actual_memory_api(self, "rewind_transform")
        _raise_for_params(params, self, "rewind_transform")

        routed_params = process_routing(self, "rewind_transform", **params)

        # Transform sequentially through all steps using their rewind_transform
        # Each transformer handles its own rewind and warmup discard internally
        X_t = X
        for _, name, transform in self._iter():
            X_t = _rewind_transform_one(transform, X_t, None, None, routed_params[name])

        return X_t

    def _can_inverse_transform(self) -> bool:
        """Check if all steps support `inverse_transform`.

        Returns
        -------
        bool
            True if all steps are invertible.

        """
        return all(
            t.__sklearn_tags__().transformer_tags.invertible
            for _, _, t in self._iter()
            if t.__sklearn_tags__().transformer_tags is not None
        )

    @available_if(_can_inverse_transform)
    def inverse_transform(self, X_t: pl.DataFrame, X_p: pl.DataFrame, **params: Any) -> pl.DataFrame:
        """Apply `inverse_transform` for each step in a reverse order.

        All estimators in the pipeline must support `inverse_transform`.

        Parameters
        ----------
        X_t : pl.DataFrame
            Transformed data to inverse-transform. Must fulfill input
            requirements of the last step's ``inverse_transform`` method.

        X_p : pl.DataFrame
            Untransformed data corresponding to at least ``observation_horizon``
            immediately previous time stamps. Used by stateful steps to
            reconstruct original-space values during inverse transformation.
            When ``observation_horizon == 0``, this is unused but still required.

        **params : dict of str -> object
            Parameters requested and accepted by steps. Each step must have
            requested certain metadata for these parameters to be forwarded to
            them.

        Returns
        -------
        pl.DataFrame
            Inverse transformed data in the original feature space.
        """
        _raise_for_params(params, self, "inverse_transform")

        # we don't have to branch here, since params is only non-empty if
        # enable_metadata_routing=True.
        routed_params = process_routing(self, "inverse_transform", **params)
        reverse_iter = reversed(list(self._iter()))

        if self.observation_horizon:
            # Build X_p_iter_list by transforming X_p through each step
            # The key insight: for the first transformer's inverse, we need
            # X_p[sum_of_other_observation_horizons : observation_horizon]
            # not X_p[:first_observation_horizon]
            steps_list = list(self._iter())

            X_p_iter = X_p
            X_p_iter_list = []
            for _idx, (_, _, transform) in enumerate(steps_list[:-1]):
                # NOTE: No transform metadata is routed here. This is an internal
                # forward-pass to compute what X_p looks like in each step's
                # transformed space, not a user-facing transform call. If a
                # transformer ever requires routed metadata for this internal
                # transform, a dedicated routing mapping would be needed.
                X_p_iter = deepcopy(transform).rewind_transform(X_p_iter)
                X_p_iter_list.append(X_p_iter)

            # The forward pass builds X_p_iter_list least-to-most transformed:
            # [A(X_p), B(A(X_p)), ...]. reverse_iter walks steps last-to-first
            # ([..., B, A]), so each step's inverse must receive X_p expressed in
            # its own input space (the output of the step before it). Reverse the
            # forward-pass list so the most-transformed context pairs with the
            # last step.
            X_p_iter_list.reverse()

            # For the first transformer's inverse, we need the slice of X_p
            # that comes after all the memory used by other transformers
            first_transform = steps_list[0][2]
            offset = sum(t.observation_horizon for _, _, t in steps_list[1:])
            X_p_iter_list.append(X_p[offset : offset + first_transform.observation_horizon])

            X = X_t
            for (_, name, transform), X_p_iter in zip(reverse_iter, X_p_iter_list, strict=False):
                X = transform.inverse_transform(X_t=X, X_p=X_p_iter, **routed_params[name].inverse_transform)

        else:
            X = X_t
            for _, name, transform in reverse_iter:
                X = transform.inverse_transform(X_t=X, X_p=X_p, **routed_params[name].inverse_transform)

        return X

    def get_metadata_routing(self) -> MetadataRouter:
        """Get metadata routing of this object.

        Please check [Metadata Routing User Guide](https://scikit-learn.org/stable/metadata_routing.html) on how the routing
        mechanism works.

        Returns
        -------
        routing : MetadataRouter
            A `MetadataRouter` encapsulating
            routing information.
        """
        router = MetadataRouter(owner=self.__class__.__name__)

        # first we add all steps except the last one
        for _, name, trans in self._iter(with_final=False, filter_passthrough=True):
            method_mapping = MethodMapping()
            # fit, fit_predict, and fit_transform call fit_transform if it
            # exists, or else fit and transform
            if hasattr(trans, "fit_transform"):
                (
                    method_mapping
                    .add(caller="fit", callee="fit_transform")
                    .add(caller="fit_transform", callee="fit_transform")
                    .add(caller="fit_predict", callee="fit_transform")
                )
            else:
                (
                    method_mapping
                    .add(caller="fit", callee="fit")
                    .add(caller="fit", callee="transform")
                    .add(caller="fit_transform", callee="fit")
                    .add(caller="fit_transform", callee="transform")
                    .add(caller="fit_predict", callee="fit")
                    .add(caller="fit_predict", callee="transform")
                )

            (
                method_mapping
                .add(caller="predict", callee="transform")
                .add(caller="predict_proba", callee="transform")
                .add(caller="decision_function", callee="transform")
                .add(caller="predict_log_proba", callee="transform")
                .add(caller="transform", callee="transform")
                .add(caller="observe_transform", callee="observe_transform")
                .add(caller="rewind_transform", callee="rewind_transform")
                .add(caller="inverse_transform", callee="inverse_transform")
                .add(caller="score", callee="transform")
            )

            router.add(method_mapping=method_mapping, **{name: trans})

        final_name, final_est = self.steps[-1]
        if final_est is None or final_est == "passthrough":
            return router

        # then we add the last step
        method_mapping = MethodMapping()
        if hasattr(final_est, "fit_transform"):
            method_mapping.add(caller="fit_transform", callee="fit_transform")
        else:
            method_mapping.add(caller="fit", callee="fit").add(caller="fit", callee="transform")
        (
            method_mapping
            .add(caller="fit", callee="fit")
            .add(caller="transform", callee="transform")
            .add(caller="observe_transform", callee="observe_transform")
            .add(caller="rewind_transform", callee="rewind_transform")
            .add(caller="inverse_transform", callee="inverse_transform")
            .add(caller="score", callee="score")
        )

        router.add(method_mapping=method_mapping, **{final_name: final_est})
        return router
