"""FeaturePipeline and FeatureUnion utilities for chaining transformers."""

from collections import Counter
from collections.abc import Callable, Iterator
from copy import deepcopy
from itertools import chain
from numbers import Integral
from typing import Any

import polars as pl
import polars.selectors as cs
from joblib import Memory
from sklearn.base import _fit_context, clone
from sklearn.compose._column_transformer import (
    ColumnTransformer as sklearn_ColumnTransformer,
)
from sklearn.compose._column_transformer import (
    _check_X,
)
from sklearn.pipeline import (
    FeatureUnion as sklearn_FeatureUnion,
)
from sklearn.pipeline import (
    Pipeline as sklearn_Pipeline,
)
from sklearn.pipeline import (
    _fit_one,
    _fit_transform_one,
    _transform_one,
)
from sklearn.preprocessing import FunctionTransformer
from sklearn.utils import (
    Bunch,
    _safe_indexing,
)
from sklearn.utils._dataframe import is_pandas_df
from sklearn.utils._param_validation import HasMethods, Hidden, StrOptions
from sklearn.utils._set_output import (
    _get_output_config,
)
from sklearn.utils._user_interface import _print_elapsed_time
from sklearn.utils.metadata_routing import (
    MetadataRouter,
    MethodMapping,
    _raise_for_params,
    process_routing,
)
from sklearn.utils.metaestimators import _BaseComposition, available_if
from sklearn.utils.parallel import Parallel, delayed
from sklearn.utils.validation import (
    _check_feature_names,
    _check_n_features,
    _get_feature_names,
    _num_samples,
    check_is_fitted,
)

from yohou.base import BaseTransformer, Tags

__all__ = ["FeaturePipeline", "FeatureUnion", "ColumnTransformer"]


class FeaturePipeline(BaseTransformer, _BaseComposition):
    """
        A sequence of time series transformers.

        `FeaturePipeline` allows you to sequentially apply a list of time series
        transformers to preprocess the data.

        Steps of the pipeline must be 'transforms', that is, they must implement
        `fit`, `transform` and `update` methods.

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
            :ref:`Combining Estimators <combining_estimators>` for more details.
    model_selection/_search
        transform_input : list of str, default=None
            The names of the :term:`metadata` parameters that should be transformed by the
            pipeline before passing it to the step consuming it.

            This enables transforming some input arguments to ``fit`` (other than ``X``)
            to be transformed by the steps of the pipeline up to the step which requires
            them. Requirement is defined via :ref:`metadata routing <metadata_routing>`.
            For instance, this can be used to pass a validation set through the pipeline.

            You can only set this if metadata routing is enabled, which you
            can enable using ``sklearn.set_config(enable_metadata_routing=True)``.

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
        named_steps : :class:`~sklearn.utils.Bunch`
            Dictionary-like object, with the following attributes.
            Read-only attribute to access any step parameter by user given name.
            Keys are step names and values are steps parameters.

        n_features_in_ : int
            Number of features seen during :term:`fit`. Only defined if the
            underlying first estimator in `steps` exposes such an attribute
            when fit.

        feature_names_in_ : ndarray of shape (`n_features_in_`,)
            Names of features seen during :term:`fit`. Only defined if the
            underlying estimator exposes such an attribute when fit.

        See Also
        --------
        sklearn.pipeline.FeaturePipeline : Underlying scikit-learn pipeline class.
        BaseTransformer : Base class for time series transformers.
        FeatureUnion : Parallel transformer combination.
        ColumnTransformer : Apply transformers to specific columns.

        Notes
        -----
        All input data must include a `time` column with datetime values. The `time`
        column is preserved through all transformations.

        The `observation_horizon` property accumulates across all steps, returning
        the sum of all transformer observation horizons. This indicates the total
        amount of historical data required by the pipeline.

        Supports time series-specific `update()` method for incremental learning,
        allowing the pipeline to incorporate new observations without full retraining.

        The final step can be a forecaster, enabling end-to-end forecasting pipelines
        that transform features and generate predictions.

        Examples
        --------
        >>> import polars as pl
        >>> from datetime import datetime, timedelta
        >>> from yohou.pipeline import FeaturePipeline
        >>> from yohou.preprocessing import SeasonalDifferencing
        >>> from yohou.preprocessing.window import LagTransformer
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

    _parameter_constraints: dict[str, Any] = {
        "steps": [list, Hidden(tuple)],
        "transform_input": [list, None],
        "memory": [None, str, HasMethods(["cache"])],
        "verbose": ["boolean"],
    }

    def __init__(
        self,
        steps: list[tuple[str, Any]],
        *,
        # TODO: Can we have a transform_input for forecasting?
        transform_input: list[str] | None = None,
        memory: None | Memory | str = None,
        verbose: bool = False,
    ) -> None:
        self.steps = steps
        self.transform_input = transform_input
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
        return _BaseComposition._get_params(self, attr="steps", deep=deep)  # type: ignore[return-value]

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
        return sklearn_Pipeline._iter(  # type: ignore[return-value]
            self,
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

    def _fit(self, X: pl.DataFrame, y: pl.DataFrame | None, routed_params: Any) -> pl.DataFrame:
        """Fit the pipeline.

        Parameters
        ----------
        X : pl.DataFrame
            Training data.

        y : pl.DataFrame | None
            Training targets.

        routed_params : Any
            Routed parameters.

        Returns
        -------
        X_t : pl.DataFrame
            Transformed data.

        """
        return sklearn_Pipeline._fit(self, X, y, routed_params)  # type: ignore[return-value]

    @property
    def named_steps(self) -> Bunch:
        """Access the steps by name.

        Returns
        -------
        named_steps : Bunch
            Dictionary-like object with step names as keys.

        """
        return sklearn_Pipeline.named_steps.fget(self)  # type: ignore[attr-defined]

    @property
    def _final_estimator(self) -> Any:
        """Get the final estimator.

        Returns
        -------
        estimator : Any
            The final estimator in the pipeline.

        """
        return sklearn_Pipeline._final_estimator.fget(self)  # type: ignore[attr-defined]

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
        return sklearn_Pipeline._log_message(self, step_idx)  # type: ignore[return-value]

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
        return sklearn_Pipeline.n_features_in_.fget(self)  # type: ignore[attr-defined]

    @property
    def feature_names_in_(self) -> Any:
        """Names of features seen during fit.

        Returns
        -------
        feature_names_in_ : Any
            Names of input features.

        """
        return sklearn_Pipeline.feature_names_in_.fget(self)  # type: ignore[attr-defined]

    def __sklearn_is_fitted__(self) -> bool:
        """Check if the pipeline is fitted.

        Returns
        -------
        is_fitted : bool
            True if the pipeline is fitted.

        """
        return sklearn_Pipeline.__sklearn_is_fitted__(self)  # type: ignore[return-value]

    def _sk_visual_block_(self) -> Any:
        """Get visual block representation.

        Returns
        -------
        visual_block : Any
            Visual block representation.

        """
        # Delegate to sklearn's implementation
        # Access the method from sklearn_Pipeline and call it as unbound
        return sklearn_Pipeline._sk_visual_block_(self)  # type: ignore[arg-type]

    def _get_metadata_for_step(self, **kwargs: Any) -> Any:
        """Get metadata for a specific step.

        Parameters
        ----------
        **kwargs : dict
            Arguments passed from sklearn's _fit method.

        Returns
        -------
        metadata : Any
            Metadata for the step.

        """
        return sklearn_Pipeline._get_metadata_for_step(self, **kwargs)

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
                # Stateful if any step is stateful
                tags.transformer_tags.stateful = any(
                    t.__sklearn_tags__().transformer_tags.stateful for t in transformers
                )

                # Invertible if all steps are invertible
                tags.transformer_tags.invertible = all(
                    t.__sklearn_tags__().transformer_tags.invertible for t in transformers
                )
                tags.transformer_tags.invertible = all(
                    hasattr(t, "inverse_transform") and callable(t.inverse_transform) for t in transformers
                )

                # min_value is the one of the first transformer
                tags.input_tags.min_value = transformers[0].__sklearn_tags__().input_tags.min_value

        return tags

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

    def reset(self, X: pl.DataFrame) -> "FeaturePipeline":
        """Resets the pipeline.

        Parameters
        ----------
        X : pl.DataFrame
            Input time series.

        Returns
        -------
        self

        """
        # TODO: We don't want to store _X_observed in the pipeline
        BaseTransformer.reset(self, X)

        Xt = X
        for _, _, transform in self._iter():
            if hasattr(transform, "reset"):
                transform.reset(Xt)

            if hasattr(transform, "transform"):
                Xt = transform.transform(Xt)

        return self

    def _validate_steps(self) -> None:
        """Validate that all steps are BaseTransformer instances.

        Raises
        ------
        TypeError
            If any step is not a BaseTransformer or 'passthrough'.

        """
        names, transformers = zip(*self.steps, strict=False)

        # validate names
        self._validate_names(names)

        for t in transformers:
            if t is None or t == "passthrough":
                continue
            if not isinstance(t, BaseTransformer):
                raise TypeError(
                    "All steps should be instances of `BaseTransformer` "
                    "or be the string 'passthrough' "
                    "'%s' (type %s) doesn't" % (t, type(t))
                )

    @_fit_context(  # type: ignore[untyped-decorator]
        # estimators in FeaturePipeline.steps are not validated yet
        prefer_skip_nested_validation=False
    )
    def fit(self, X: pl.DataFrame, y: pl.DataFrame | None = None, **params: Any) -> "FeaturePipeline":
        """Fit the model.

        Fit all the transformers one after the other and sequentially transform the
        data. Finally, fit the transformed data using the final estimator.

        Parameters
        ----------
        X : iterable
            Training data. Must fulfill input requirements of first step of the
            pipeline.

        y : iterable, default=None
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
        self : object
            FeaturePipeline with fitted steps.
        """
        routed_params = self._check_method_params(method="fit", props=params)
        X_t = self._fit(X, y, routed_params)
        with _print_elapsed_time("FeaturePipeline", self._log_message(len(self.steps) - 1)):
            if self._final_estimator != "passthrough":
                last_step_params = routed_params[self.steps[-1][0]]
                self._final_estimator.fit(X_t, y, **last_step_params["fit"])

        return self

    @_fit_context(  # type: ignore[untyped-decorator]
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
        X : iterable
            Training data. Must fulfill input requirements of first step of the
            pipeline.

        y : iterable, default=None
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
        X_t : ndarray of shape (n_samples, n_transformed_features)
            Transformed samples.
        """
        routed_params = self._check_method_params(method="fit_transform", props=params)
        X_t = self._fit(X, y, routed_params)

        last_step = self._final_estimator
        with _print_elapsed_time("FeaturePipeline", self._log_message(len(self.steps) - 1)):
            if last_step == "passthrough":
                return X_t  # type: ignore[return-value]

            last_step_params = routed_params[self.steps[-1][0]]
            result = last_step.fit_transform(X_t, y, **last_step_params["fit_transform"])
            return result  # type: ignore[return-value]

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
        X : iterable
            Data to transform. Must fulfill input requirements of first step
            of the pipeline.

        **params : dict of str -> object
            Parameters requested and accepted by steps. Each step must have
            requested certain metadata for these parameters to be forwarded to
            them.

        Returns
        -------
        X_t : ndarray of shape (n_samples, n_transformed_features)
            Transformed data.
        """
        _raise_for_params(params, self, "transform")

        # not branching here since params is only available if
        # enable_metadata_routing=True
        routed_params = process_routing(self, "transform", **params)
        X_t = X
        for _, name, transform in self._iter():
            X_t = transform.transform(X_t, **routed_params[name].transform)
        return X_t  # type: ignore[return-value]

    def update_transform(self, X: pl.DataFrame, **params: Any) -> pl.DataFrame:
        """Update and transform the data through the pipeline.

        This method atomically updates each transformer with new data and
        transforms it in sequence. The transformation uses the pre-update state,
        then updates the memory. This is more efficient and correct than calling
        update() then transform() separately.

        Parameters
        ----------
        X : pl.DataFrame
            New data to update with and transform. Must fulfill input requirements
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
        _raise_for_params(params, self, "update_transform")

        routed_params = process_routing(self, "update_transform", **params)

        # Transform sequentially through all steps using their update_transform
        # Each transformer handles its own memory management internally
        X_t = X
        for _, name, transform in self._iter():
            X_t = _update_transform_one(transform, X_t, None, None, routed_params[name])

        return X_t

    def _can_inverse_transform(self) -> bool:
        """Check if all steps support `inverse_transform`.

        Returns
        -------
        bool
            True if all steps have `inverse_transform` method.

        """
        return all(hasattr(t, "inverse_transform") for _, _, t in self._iter())

    @available_if(_can_inverse_transform)  # type: ignore[untyped-decorator]
    def inverse_transform(self, X_t: pl.DataFrame, X_p: pl.DataFrame, **params: Any) -> pl.DataFrame:
        """Apply `inverse_transform` for each step in a reverse order.

        All estimators in the pipeline must support `inverse_transform`.

        Parameters
        ----------
        X_t : array-like of shape (n_samples, n_transformed_features)
            Data samples, where ``n_samples`` is the number of samples and
            ``n_features`` is the number of features. Must fulfill
            input requirements of last step of pipeline's
            ``inverse_transform`` method.

        **params : dict of str -> object
            Parameters requested and accepted by steps. Each step must have
            requested certain metadata for these parameters to be forwarded to
            them.

        Returns
        -------
        X_t : ndarray of shape (n_samples, n_features)
            Inverse transformed data, that is, data in the original feature
            space.
        """
        _raise_for_params(params, self, "inverse_transform")

        # we don't have to branch here, since params is only non-empty if
        # enable_metadata_routing=True.
        routed_params = process_routing(self, "inverse_transform", **params)
        reverse_iter = reversed(list(self._iter()))

        if self.observation_horizon:
            # TODO: Routes params
            # routed_params_previous = process_routing(self, "fit_transform", **params)

            # Build X_p_iter_list by transforming X_p through each step
            # The key insight: for the first transformer's inverse, we need
            # X_p[sum_of_other_observation_horizons : observation_horizon]
            # not X_p[:first_observation_horizon]
            steps_list = list(self._iter())

            X_p_iter = X_p
            X_p_iter_list = []
            for idx, (_, _, transform) in enumerate(steps_list[:-1]):
                # Transform X_p_iter through this step using the fitted transform
                X_p_iter = (
                    deepcopy(transform)
                    .reset(X_p_iter[: transform.observation_horizon])
                    .transform(
                        X_p_iter[transform.observation_horizon :],  # **routed_params_previous
                    )
                )
                X_p_iter_list.append(X_p_iter)

            # For the first transformer's inverse, we need the slice of X_p
            # that comes after all the memory used by other transformers
            first_transform = steps_list[0][2]
            offset = sum(t.observation_horizon for _, _, t in steps_list[1:])
            X_p_iter_list.append(X_p[offset : offset + first_transform.observation_horizon])

            # NOTE: Do NOT reverse! X_p_iter_list is built as:
            # [X_p for last step's inverse, ..., X_p for first step's inverse]
            # which matches reverse_iter: [last step, ..., first step]
            # X_p_iter_list.reverse()

            X = X_t
            for (_, _, transform), X_p_iter in zip(reverse_iter, X_p_iter_list, strict=False):
                X = transform.inverse_transform(X_t=X, X_p=X_p_iter)

        else:
            X = X_t
            for _, name, transform in reverse_iter:
                X = transform.inverse_transform(X_t, **routed_params[name].inverse_transform)

        return X  # type: ignore[return-value]  # type: ignore[return-value]

    def get_metadata_routing(self) -> MetadataRouter:
        """Get metadata routing of this object.

        Please check :ref:`User Guide <metadata_routing>` on how the routing
        mechanism works.

        Returns
        -------
        routing : MetadataRouter
            A :class:`~sklearn.utils.metadata_routing.MetadataRouter` encapsulating
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
                    method_mapping.add(caller="fit", callee="fit_transform")
                    .add(caller="fit_transform", callee="fit_transform")
                    .add(caller="fit_predict", callee="fit_transform")
                )
            else:
                (
                    method_mapping.add(caller="fit", callee="fit")
                    .add(caller="fit", callee="transform")
                    .add(caller="fit_transform", callee="fit")
                    .add(caller="fit_transform", callee="transform")
                    .add(caller="fit_predict", callee="fit")
                    .add(caller="fit_predict", callee="transform")
                )

            (
                method_mapping.add(caller="predict", callee="transform")
                .add(caller="predict", callee="transform")
                .add(caller="predict_proba", callee="transform")
                .add(caller="decision_function", callee="transform")
                .add(caller="predict_log_proba", callee="transform")
                .add(caller="transform", callee="transform")
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
            method_mapping.add(caller="fit", callee="fit")
            .add(caller="transform", callee="transform")
            .add(caller="inverse_transform", callee="inverse_transform")
            .add(caller="score", callee="score")
        )

        router.add(method_mapping=method_mapping, **{final_name: final_est})
        return router


def _hstack(Xs: list[pl.DataFrame], column_names: list[list[str]], observation_horizons: list[int]) -> pl.DataFrame:
    """Stack transformed features horizontally, aligning observation horizons.

    Parameters
    ----------
    Xs : list of pl.DataFrame
        List of transformed DataFrames.

    column_names : list of list of str
        Column names for each DataFrame.

    observation_horizons : list of int
        Observation horizon for each transformer.

    Returns
    -------
    pl.DataFrame
        Horizontally concatenated features.

    """
    ref_observation_horizon = max(observation_horizons)
    time = Xs[0].select(cs.by_name("time"))[ref_observation_horizon - observation_horizons[0] :]  # type: ignore[attr-defined]

    # Rename columns before concat to avoid duplicates
    Xs_renamed = []
    col_idx = 0
    for X, observation_horizon, cols in zip(Xs, observation_horizons, column_names, strict=False):
        X_no_time = X.select(~cs.by_name("time"))[ref_observation_horizon - observation_horizon :]  # type: ignore[attr-defined]
        # Create rename mapping for this transformer's columns
        rename_map = dict(zip(X_no_time.columns, cols, strict=False))
        X_renamed = X_no_time.rename(rename_map)
        Xs_renamed.append(X_renamed)
        col_idx += len(cols)

    Xs_concat = pl.concat(Xs_renamed, how="horizontal")
    result = pl.concat([time, Xs_concat], how="horizontal")

    return result


class FeatureUnion(BaseTransformer, _BaseComposition):
    """Concatenates results of multiple transformer objects.

    This estimator applies a list of transformer objects in parallel to the
    input data, then concatenates the results. This is useful to combine
    several feature extraction mechanisms into a single transformer.

    Parameters of the transformers may be set using its name and the parameter
    name separated by a '__'. A transformer may be replaced entirely by
    setting the parameter with its name to another transformer, removed by
    setting to 'drop' or disabled by setting to 'passthrough' (features are
    passed without transformation).

    Parameters
    ----------
    transformer_list : list of (str, transformer) tuples
        List of transformer objects to be applied to the data. The first
        half of each tuple is the name of the transformer. The transformer can
        be 'drop' for it to be ignored or can be 'passthrough' for features to
        be passed unchanged.

    n_jobs : int, default=None
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    transformer_weights : dict, default=None
        Multiplicative weights for features per transformer.
        Keys are transformer names, values the weights.
        Raises ValueError if key not present in ``transformer_list``.

    verbose : bool, default=False
        If True, the time elapsed while fitting each transformer will be
        printed as it is completed.

    verbose_feature_names_out : bool, default=True
        If True, :meth:`get_feature_names_out` will prefix all feature names
        with the name of the transformer that generated that feature.
        If False, :meth:`get_feature_names_out` will not prefix any feature
        names and will error if feature names are not unique.

    Attributes
    ----------
    named_transformers : :class:`~sklearn.utils.Bunch`
        Dictionary-like object, with the following attributes.
        Read-only attribute to access any transformer parameter by user
        given name. Keys are transformer names and values are
        transformer parameters.

    n_features_in_ : int
        Number of features seen during :term:`fit`. Only defined if the
        underlying first transformer in `transformer_list` exposes such an
        attribute when fit.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when
        `X` has feature names that are all strings.

    See Also
    --------
    sklearn.pipeline.FeatureUnion : Underlying scikit-learn feature union class.
    FeaturePipeline : Sequential transformer chaining.
    BaseTransformer : Base class for transformers.
    preprocessing.window.LagTransformer : Common transformer for lag features.

    Notes
    -----
    Transformers run in parallel when `n_jobs` is set to a value other than 1.
    This can significantly improve performance for computationally expensive transformers.

    Results are concatenated horizontally with automatic time alignment. The
    internal `_hstack()` function handles transformers with different observation
    horizons by aligning their outputs to the maximum observation horizon.

    The `observation_horizon` property returns the MAXIMUM across all transformers
    (not the sum). This is because all transformers operate on the same input data,
    and the union needs enough history to satisfy the most demanding transformer.

    Useful for multi-scale feature engineering, such as combining short-term and
    long-term lag features, or mixing different preprocessing approaches in parallel.

    All transformers must accept the same input time series with a `time` column.

    Examples
    --------
    >>> import polars as pl
    >>> from datetime import datetime, timedelta
    >>> from yohou.pipeline import FeatureUnion
    >>> from yohou.preprocessing.window import LagTransformer
    >>>
    >>> # Create sample weekly time series data (52 weeks)
    >>> time = pl.datetime_range(
    ...     start=datetime(2023, 1, 1),
    ...     end=datetime(2023, 1, 1) + timedelta(weeks=51),
    ...     interval="1w",
    ...     eager=True,
    ... )
    >>> data = pl.DataFrame({"time": time, "demand": range(1, 53)})
    >>>
    >>> # Example 1: Combine short-term and long-term lags for multi-scale features
    >>> union = FeatureUnion([
    ...     ("short_lags", LagTransformer(lag=[1, 2, 3])),
    ...     ("long_lags", LagTransformer(lag=[7, 14, 21])),
    ... ])
    >>>
    >>> # Example 2: Access transformers by name
    >>> union.named_transformers["short_lags"]  # doctest: +ELLIPSIS
    LagTransformer(...)
    >>>
    >>> # Example 3: Access transformers by position
    >>> union[0]  # doctest: +ELLIPSIS
    LagTransformer(...)

    """

    _required_parameters = ["transformer_list"]

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
        return _BaseComposition._get_params(self, attr="transformer_list", deep=deep)  # type: ignore[return-value]

    def set_params(self, **params: Any) -> "FeatureUnion":
        """Set the parameters of this estimator.

        Parameters
        ----------
        **params : dict
            Estimator parameters.

        Returns
        -------
        self : FeatureUnion
            FeatureUnion instance.

        """
        _BaseComposition._set_params(self, attr="transformer_list", **params)
        return self

    def _iter(self) -> Iterator[tuple[str, Any, float]]:
        """Generate (name, trans, weight) tuples excluding None and 'drop' transformers.

        Yields
        ------
        name : str
            Transformer name.
        trans : Any
            Transformer instance.
        weight : float
            Transformer weight.

        """
        return sklearn_FeatureUnion._iter(self)  # type: ignore[arg-type,return-value]

    def __getitem__(self, ind: int | str | slice) -> Any:
        """Return a sub-union or a single transformer.

        Parameters
        ----------
        ind : int, str, or slice
            Index, name, or slice of the transformer to retrieve.

        Returns
        -------
        transformer : Any
            The transformer or sub-union.

        """
        if isinstance(ind, slice):
            if ind.step is not None:
                raise ValueError("FeatureUnion slicing only supports a step of 1")
            return self.__class__(
                transformer_list=self.transformer_list[ind],
                n_jobs=self.n_jobs,
                transformer_weights=self.transformer_weights,
                verbose=self.verbose,
            )
        elif isinstance(ind, int):
            _, est = self.transformer_list[ind]
            return est
        else:
            # String case - get by name
            return self.named_transformers[ind]

    @property
    def named_transformers(self) -> Bunch:
        """Access the transformers by name.

        Returns
        -------
        named_transformers : Bunch
            Dictionary-like object with transformer names as keys.

        """
        return Bunch(**{name: trans for name, trans in self.transformer_list})

    def _log_message(self, name: str, idx: int, total: int) -> str:
        """Get log message for a transformer.

        Parameters
        ----------
        name : str
            Transformer name.
        idx : int
            Current index.
        total : int
            Total number of transformers.

        Returns
        -------
        message : str
            Log message.

        """
        return f"(step {idx} of {total}) Processing {name}"

    def _parallel_func(self, X: pl.DataFrame, y: pl.DataFrame | None, func: Any, routed_params: Any) -> Any:
        """Run func in parallel on X and y.

        Parameters
        ----------
        X : pl.DataFrame
            Input data.
        y : pl.DataFrame | None
            Target data.
        func : Any
            Function to apply.
        routed_params : Any
            Routed parameters.

        Returns
        -------
        results : Any
            Results from parallel execution.

        """
        return sklearn_FeatureUnion._parallel_func(self, X, y, func, routed_params)  # type: ignore[arg-type]

    def _update_transformer_list(self, transformers: Any) -> None:
        """Update transformer_list with fitted transformers.

        Parameters
        ----------
        transformers : Any
            Fitted transformers.

        """
        transformers_iter = iter(transformers)
        self.transformer_list[:] = [
            (name, next(transformers_iter) if old is not None else None) for name, old in self.transformer_list
        ]

    def get_feature_names_out(self, input_features: list[str] | None = None) -> Any:
        """Get output feature names.

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
        # Delegate to first transformer
        for _, trans in self.transformer_list:
            if hasattr(trans, "n_features_in_"):
                return trans.n_features_in_
        raise AttributeError("n_features_in_ not available")

    @property
    def feature_names_in_(self) -> Any:
        """Names of features seen during fit.

        Returns
        -------
        feature_names_in_ : Any
            Names of input features.

        """
        for _, trans in self.transformer_list:
            if hasattr(trans, "feature_names_in_"):
                return trans.feature_names_in_
        raise AttributeError("feature_names_in_ not available")

    def _add_prefix_for_feature_names_out(self, feature_names_out: list[list[str]]) -> list[str]:
        """Add prefixes to feature names.

        Parameters
        ----------
        feature_names_out : list[list[str]]
            Feature names from each transformer.

        Returns
        -------
        prefixed_names : list[str]
            Feature names with prefixes.

        """
        return sklearn_FeatureUnion._add_prefix_for_feature_names_out(self, feature_names_out)  # type: ignore[arg-type]

    def __sklearn_tags__(self) -> Tags:
        """Get estimator tags.

        Returns
        -------
        Tags
            Estimator tags with yohou-specific attributes.

        """
        tags = super().__sklearn_tags__()

        # Aggregate tags from transformers (static capability check)
        if hasattr(self, "transformer_list") and self.transformer_list is not None:
            transformers = [t for _, t in self.transformer_list if t not in ("drop", "passthrough") and t is not None]
            if transformers:
                # Stateful if any transformer is stateful
                tags.transformer_tags.stateful = any(
                    t.__sklearn_tags__().transformer_tags.stateful for t in transformers
                )

                # Not invertible unless there is only one transformer and it is invertible
                tags.transformer_tags.invertible = (
                    len(transformers) == 1 and transformers[0].__sklearn_tags__().transformer_tags.invertible
                )

                # Aggregate min_value: take the maximum (most restrictive)
                # All transformers receive the same input, so we need to satisfy all constraints
                min_values = [t.__sklearn_tags__().input_tags.min_value for t in transformers]
                non_none_min_values = [v for v in min_values if v is not None]
                tags.input_tags.min_value = max(non_none_min_values) if non_none_min_values else None

        return tags

    def __sklearn_is_fitted__(self) -> bool:
        """Check if fitted.

        Returns
        -------
        is_fitted : bool
            True if the union is fitted.

        """
        return sklearn_FeatureUnion.__sklearn_is_fitted__(self)  # type: ignore[return-value]

    def _sk_visual_block_(self) -> Any:
        """Get visual block representation.

        Returns
        -------
        visual_block : Any
            Visual block representation.

        """
        return sklearn_FeatureUnion._sk_visual_block_(self)  # type: ignore[arg-type]

    def _get_observation_horizons(self) -> list[int]:
        """Get observation horizons from all transformers.

        Returns
        -------
        observation_horizons : list[int]
            List of observation horizons from each transformer.

        """
        observation_horizons = []
        for _, t, _ in self._iter():
            observation_horizon = 0
            if t != "passthrough" and t is not None and hasattr(t, "observation_horizon"):
                observation_horizon = t.observation_horizon

            observation_horizons.append(observation_horizon)

        return observation_horizons

    @property
    def observation_horizon(self) -> int:
        """Maximum observation horizon across all transformers.

        Returns
        -------
        int
            Maximum observation horizon needed.

        Raises
        ------
        NotFittedError
            If the feature union has not been fitted yet.

        """
        check_is_fitted(self)

        observation_horizons = self._get_observation_horizons()
        observation_horizon = max(observation_horizons)

        return observation_horizon

    def __init__(
        self,
        transformer_list: list[tuple[str, Any]],
        *,
        n_jobs: int | None = None,
        transformer_weights: dict[str, float] | None = None,
        verbose: bool = False,
        verbose_feature_names_out: bool = True,
    ) -> None:
        self.transformer_list = transformer_list
        self.n_jobs = n_jobs
        self.transformer_weights = transformer_weights
        self.verbose = verbose
        self.verbose_feature_names_out = verbose_feature_names_out

    def _validate_transformers(self) -> None:
        """Validate all transformers are BaseTransformer instances.

        Raises
        ------
        TypeError
            If any transformer is invalid.

        """
        names, transformers = zip(*self.transformer_list, strict=False)

        # validate names
        self._validate_names(names)

        # validate estimators
        for t in transformers:
            if t in ("drop", "passthrough"):
                continue
            if not (hasattr(t, "fit") or hasattr(t, "fit_transform")) or not hasattr(t, "transform"):
                raise TypeError(
                    "All estimators should implement fit and transform. '%s' (type %s) doesn't" % (t, type(t))
                )

    def _validate_transformer_weights(self) -> None:
        """Validate transformer weights dictionary.

        Raises
        ------
        ValueError
            If weight keys don't match transformer names.

        """
        if not self.transformer_weights:
            return

        transformer_names = set(name for name, _ in self.transformer_list)
        for name in self.transformer_weights:
            if name not in transformer_names:
                raise ValueError(
                    f'Attempting to weight transformer "{name}", but it is not present in transformer_list.'
                )

    def fit(self, X: pl.DataFrame, y: pl.DataFrame | None = None, **fit_params: Any) -> "FeatureUnion":
        """Fit all transformers using X.

        Parameters
        ----------
        X : iterable or array-like, depending on transformers
            Input data, used to fit transformers.

        y : array-like of shape (n_samples, n_outputs), default=None
            Targets for supervised learning.

        **fit_params : dict, default=None
            - If `enable_metadata_routing=False` (default):
              Parameters directly passed to the `fit` methods of the
              sub-transformers.

            - If `enable_metadata_routing=True`:
              Parameters safely routed to the `fit` methods of the
              sub-transformers. See :ref:`Metadata Routing User Guide
              <metadata_routing>` for more details.

        Returns
        -------
        self : object
            FeatureUnion class instance.
        """
        _raise_for_params(fit_params, self, "fit")
        routed_params = process_routing(self, "fit", **fit_params)
        transformers = self._parallel_func(X, y, _fit_one, routed_params)

        if not transformers:
            # All transformers are None
            return self

        self._update_transformer_list(transformers)
        return self

    def fit_transform(self, X: pl.DataFrame, y: pl.DataFrame | None = None, **params: object) -> object:
        """Fit all transformers, transform the data and concatenate results.

        Parameters
        ----------
        X : iterable or array-like, depending on transformers
            Input data to be transformed.

        y : array-like of shape (n_samples, n_outputs), default=None
            Targets for supervised learning.

        **params : dict, default=None
            - If `enable_metadata_routing=False` (default):
              Parameters directly passed to the `fit` methods of the
              sub-transformers.

            - If `enable_metadata_routing=True`:
              Parameters safely routed to the `fit` methods of the
              sub-transformers. See :ref:`Metadata Routing User Guide
              <metadata_routing>` for more details.

        Returns
        -------
        X_t : array-like or sparse matrix of \
                shape (n_samples, sum_n_components)
            The `hstack` of results of transformers. `sum_n_components` is the
            sum of `n_components` (output dimension) over transformers.
        """
        routed_params = process_routing(self, "fit_transform", **params)
        results = self._parallel_func(X, y, _fit_transform_one, routed_params)
        if not results:
            # All transformers are None
            time = X.select(cs.by_name("time"))
            return time

        Xs, transformers = zip(*results, strict=False)
        self._update_transformer_list(transformers)

        # Extract actual column names from each DataFrame (excluding time)
        column_names = [[col for col in X_t.columns if col != "time"] for X_t in Xs]

        result = _hstack(
            list(Xs),
            # column_names=self.get_feature_names_out(),
            column_names=column_names,
            observation_horizons=self._get_observation_horizons(),
        )
        return result  # type: ignore[return-value]

    def transform(self, X: pl.DataFrame, **params: Any) -> pl.DataFrame:
        """Transform X separately by each transformer, concatenate results.

        Parameters
        ----------
        X : iterable or array-like, depending on transformers
            Input data to be transformed.

        **params : dict, default=None

            Parameters routed to the `transform` method of the sub-transformers via the
            metadata routing API. See :ref:`Metadata Routing User Guide
            <metadata_routing>` for more details.

        Returns
        -------
        X_t : array-like or sparse matrix of shape (n_samples, sum_n_components)
            The `hstack` of results of transformers. `sum_n_components` is the
            sum of `n_components` (output dimension) over transformers.
        """
        _raise_for_params(params, self, "transform")
        routed_params = process_routing(self, "transform", **params)

        Xs = Parallel(n_jobs=self.n_jobs)(
            delayed(_transform_one)(trans, X, None, weight, routed_params[name]) for name, trans, weight in self._iter()
        )
        if not Xs:
            # All transformers are None
            time = X.select(cs.by_name("time"))
            return time

        # Extract actual column names from each DataFrame (excluding time)
        column_names = [[col for col in X_t.columns if col != "time"] for X_t in Xs]

        result = _hstack(
            Xs,
            # column_names=self.get_feature_names_out(),
            column_names=column_names,
            observation_horizons=self._get_observation_horizons(),
        )
        return result  # type: ignore[return-value]

    def update_transform(self, X: pl.DataFrame, **params: Any) -> pl.DataFrame:
        """Update and transform X in parallel for each transformer, concatenate results.

        This method atomically updates each transformer with new data and
        transforms it in parallel. The transformation uses the pre-update state,
        then updates the memory. This is more efficient and correct than calling
        update() then transform() separately.

        Parameters
        ----------
        X : pl.DataFrame
            New data to update with and transform.

        **params : dict, default=None
            Parameters routed to the `transform` methods of the sub-transformers
            via the metadata routing API. See :ref:`Metadata Routing User Guide
            <metadata_routing>` for more details.

        Returns
        -------
        X_t : pl.DataFrame
            Horizontally stacked results of transformers, aligned by observation horizons.

        """
        _raise_for_params(params, self, "update_transform")
        routed_params = process_routing(self, "update_transform", **params)

        # Parallel execution of update_transform on all transformers
        Xs = Parallel(n_jobs=self.n_jobs)(
            delayed(_update_transform_one)(trans, X, None, weight, routed_params[name])
            for name, trans, weight in self._iter()
        )

        if not Xs:
            # All transformers are None
            time = X.select(cs.by_name("time"))
            return time

        # Extract actual column names from each DataFrame (excluding time)
        column_names = [[col for col in X_t.columns if col != "time"] for X_t in Xs]

        result = _hstack(
            Xs,
            column_names=column_names,
            observation_horizons=self._get_observation_horizons(),
        )

        return result

    def get_metadata_routing(self) -> MetadataRouter:
        """Get metadata routing of this object.

        Please check :ref:`User Guide <metadata_routing>` on how the routing
        mechanism works.

        Returns
        -------
        routing : MetadataRouter
            A :class:`~sklearn.utils.metadata_routing.MetadataRouter` encapsulating
            routing information.
        """
        router = MetadataRouter(owner=self)

        for name, transformer in self.transformer_list:
            router.add(
                **{name: transformer},
                method_mapping=MethodMapping()
                .add(caller="fit", callee="fit")
                .add(caller="fit_transform", callee="fit_transform")
                .add(caller="fit_transform", callee="fit")
                .add(caller="fit_transform", callee="transform")
                .add(caller="transform", callee="transform"),
            )

        return router


_ERR_MSG_1DCOLUMN = (
    "1D data passed to a transformer that expects 2D data. "
    "Try to specify the column selection as a list of one "
    "item instead of a scalar."
)


class ColumnTransformer(BaseTransformer, _BaseComposition):
    """Applies transformers to columns of a polars DataFrame.

    This estimator allows different columns or column subsets of the input
    to be transformed separately and the features generated by each transformer
    will be concatenated to form a single feature space.
    This is useful for heterogeneous or columnar data, to combine several
    feature extraction mechanisms or transformations into a single transformer.

    Parameters
    ----------
    transformers : list of tuples
        List of (name, transformer, columns) tuples specifying the
        transformer objects to be applied to subsets of the data.

        name : str
            Like in FeaturePipeline and FeatureUnion, this allows the transformer and
            its parameters to be set using ``set_params`` and searched in grid
            search.
        transformer : {'drop', 'passthrough'} or estimator
            Estimator must support :term:`fit` and :term:`transform`.
            Special-cased strings 'drop' and 'passthrough' are accepted as
            well, to indicate to drop the columns or to pass them through
            untransformed, respectively.
        columns :  str, array-like of str, int, array-like of int, \
                array-like of bool, slice or callable
            Indexes the data on its second axis. Integers are interpreted as
            positional columns, while strings can reference DataFrame columns
            by name.  A scalar string or int should be used where
            ``transformer`` expects X to be a 1d array-like (vector),
            otherwise a 2d array will be passed to the transformer.
            A callable is passed the input data `X` and can return any of the
            above. To select multiple columns by name or dtype, you can use
            :obj:`make_column_selector`.

    remainder : {'drop', 'passthrough'} or estimator, default='drop'
        By default, only the specified columns in `transformers` are
        transformed and combined in the output, and the non-specified
        columns are dropped. (default of ``'drop'``).
        By specifying ``remainder='passthrough'``, all remaining columns that
        were not specified in `transformers`, but present in the data passed
        to `fit` will be automatically passed through. This subset of columns
        is concatenated with the output of the transformers. For dataframes,
        extra columns not seen during `fit` will be excluded from the output
        of `transform`.
        By setting ``remainder`` to be an estimator, the remaining
        non-specified columns will use the ``remainder`` estimator. The
        estimator must support :term:`fit` and :term:`transform`.
        Note that using this feature requires that the DataFrame columns
        input at :term:`fit` and :term:`transform` have identical order.

    n_jobs : int, default=None
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    transformer_weights : dict, default=None
        Multiplicative weights for features per transformer. The output of the
        transformer is multiplied by these weights. Keys are transformer names,
        values the weights.

    verbose : bool, default=False
        If True, the time elapsed while fitting each transformer will be
        printed as it is completed.

    verbose_feature_names_out : bool, default=True
        If True, :meth:`ColumnTransformer.get_feature_names_out` will prefix
        all feature names with the name of the transformer that generated that
        feature.
        If False, :meth:`ColumnTransformer.get_feature_names_out` will not
        prefix any feature names and will error if feature names are not
        unique.

    Attributes
    ----------
    transformers_ : list
        The collection of fitted transformers as tuples of (name,
        fitted_transformer, column). `fitted_transformer` can be an estimator,
        or `'drop'`; `'passthrough'` is replaced with an equivalent
        :class:`~sklearn.preprocessing.FunctionTransformer`. In case there were
        no columns selected, this will be the unfitted transformer. If there
        are remaining columns, the final element is a tuple of the form:
        ('remainder', transformer, remaining_columns) corresponding to the
        ``remainder`` parameter. If there are remaining columns, then
        ``len(transformers_)==len(transformers)+1``, otherwise
        ``len(transformers_)==len(transformers)``.

    named_transformers_ : :class:`~sklearn.utils.Bunch`
        Read-only attribute to access any transformer by given name.
        Keys are transformer names and values are the fitted transformer
        objects.

    output_indices_ : dict
        A dictionary from each transformer name to a slice, where the slice
        corresponds to indices in the transformed output. This is useful to
        inspect which transformer is responsible for which transformed
        feature(s).

    n_features_in_ : int
        Number of features seen during :term:`fit`. Only defined if the
        underlying transformers expose such an attribute when fit.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

    See Also
    --------
    sklearn.compose.ColumnTransformer : Underlying scikit-learn column transformer.
    FeaturePipeline : Sequential transformation.
    BaseTransformer : Base transformer interface.
    preprocessing.stationarization.SeasonalDifferencing : Common column-wise transformer.

    Notes
    -----
    The order of the columns in the transformed feature matrix follows the
    order of how the columns are specified in the `transformers` list.
    Columns of the original feature matrix that are not specified are
    dropped from the resulting transformed feature matrix, unless specified
    in the `passthrough` keyword. Those columns specified with `passthrough`
    are added at the right to the output of the transformers.

    Apply heterogeneous preprocessing to different columns, useful when different
    time series have different characteristics (e.g., different seasonal patterns).

    Column selection by name (string) works seamlessly with polars DataFrames,
    allowing intuitive column-specific transformations.

    Time alignment across columns with different observation horizons is handled
    automatically by the internal `_hstack()` function, ensuring all transformed
    columns are properly aligned in time.

    Setting `remainder='passthrough'` (default is 'drop') preserves untransformed
    columns in the output, useful for keeping auxiliary columns that don't require
    transformation.

    The `verbose_feature_names_out` parameter (default=True) prefixes output column
    names with transformer names (e.g., 'deseason__sales') to prevent name collisions
    when multiple transformers produce columns with the same names.

    The `observation_horizon` property returns the MAXIMUM across all column
    transformers, as the transformer needs enough history to satisfy the most
    demanding column-specific transformation.

    All columns must share the same `time` index. The `time` column is automatically
    handled and preserved in the output.

    Examples
    --------
    >>> import polars as pl
    >>> from datetime import datetime, timedelta
    >>> from yohou.pipeline import ColumnTransformer
    >>> from yohou.preprocessing import SeasonalDifferencing, SeasonalLogDifferencing
    >>>
    >>> # Create sample weekly time series data with multiple columns (52 weeks)
    >>> time = pl.datetime_range(
    ...     start=datetime(2023, 1, 1),
    ...     end=datetime(2023, 1, 1) + timedelta(weeks=51),
    ...     interval="1w",
    ...     eager=True
    ... )
    >>> data = pl.DataFrame({
    ...     "time": time,
    ...     "sales": range(1, 53),
    ...     "temperature": range(10, 62)
    ... })
    >>>
    >>> # Example 1: Apply different seasonal differencing to different columns
    >>> ct = ColumnTransformer([
    ...     ('sales_diff', SeasonalDifferencing(seasonality=4), 'sales'),
    ...     ('temp_diff', SeasonalDifferencing(seasonality=7), 'temperature')
    ... ])
    >>>
    >>> # Example 2: Use remainder='passthrough' to keep auxiliary columns
    >>> ct_passthrough = ColumnTransformer(
    ...     [('sales_diff', SeasonalDifferencing(seasonality=4), 'sales')],
    ...     remainder='passthrough'
    ... )
    >>>
    >>> # Example 3: Disable verbose_feature_names_out for cleaner names
    >>> ct_clean = ColumnTransformer(
    ...     [('diff', SeasonalDifferencing(seasonality=4), 'sales')],
    ...     verbose_feature_names_out=False
    ... )
    """

    _parameter_constraints: dict[str, Any] = {
        "transformers": [list, Hidden(tuple)],
        "remainder": [
            StrOptions({"drop", "passthrough"}),
            HasMethods(["fit", "transform"]),
            HasMethods(["fit_transform", "transform"]),
        ],
        "n_jobs": [Integral, None],
        "transformer_weights": [dict, None],
        "verbose": ["verbose"],
        "verbose_feature_names_out": ["boolean"],
    }

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
        return _BaseComposition._get_params(self, attr="transformers", deep=deep)  # type: ignore[return-value]

    def set_params(self, **params: Any) -> "ColumnTransformer":
        """Set the parameters of this estimator.

        Parameters
        ----------
        **params : dict
            Estimator parameters.

        Returns
        -------
        self : ColumnTransformer
            ColumnTransformer instance.

        """
        _BaseComposition._set_params(self, attr="transformers", **params)
        return self

    @property
    def _transformers(self) -> list[tuple[str, Any, Any]]:
        """List of (name, fitted_transformer, column) tuples.

        Returns
        -------
        transformers : list[tuple[str, Any, Any]]
            The fitted transformers.

        """
        return sklearn_ColumnTransformer._transformers.fget(self)  # type: ignore[attr-defined]

    def _iter(
        self,
        fitted: bool = False,
        column_as_labels: bool = False,
        skip_drop: bool = False,
        skip_empty_columns: bool = True,
    ) -> Iterator[tuple[str, Any, Any, Any]]:
        """Generate (name, trans, column, weight) tuples.

        Parameters
        ----------
        fitted : bool, default=False
            Whether to iterate over fitted transformers.
        column_as_labels : bool, default=False
            Whether to return columns as labels.
        skip_drop : bool, default=False
            Whether to skip 'drop' transformers.
        skip_empty_columns : bool, default=True
            Whether to skip transformers with empty columns.

        Yields
        ------
        name : str
            Transformer name.
        trans : Any
            Transformer instance.
        column : Any
            Column specification.
        weight : Any
            Transformer weight.

        """
        return sklearn_ColumnTransformer._iter(
            self,  # type: ignore[arg-type]
            fitted=fitted,
            column_as_labels=column_as_labels,
            skip_drop=skip_drop,
            skip_empty_columns=skip_empty_columns,
        )

    def __getitem__(self, ind: int | str | slice) -> Any:
        """Return a sub-transformer or a single transformer.

        Parameters
        ----------
        ind : int, str, or slice
            Index, name, or slice of the transformer to retrieve.

        Returns
        -------
        transformer : Any
            The transformer or sub-transformer.

        """
        if isinstance(ind, slice):
            if ind.step is not None:
                raise ValueError("ColumnTransformer slicing only supports a step of 1")
            return self.__class__(
                transformers=self.transformers[ind],
                remainder=self.remainder,
                n_jobs=self.n_jobs,
                transformer_weights=self.transformer_weights,
                verbose=self.verbose,
            )
        elif isinstance(ind, int):
            name, trans, _ = self.transformers[ind]
            # If fitted, use named_transformers_, otherwise return from transformers
            if hasattr(self, "named_transformers_"):
                return self.named_transformers_[name]  # type: ignore[attr-defined]
            return trans
        else:
            # String case - get by name
            if hasattr(self, "named_transformers_"):
                return self.named_transformers_[ind]  # type: ignore[attr-defined]
            # Not fitted yet, search in transformers list
            for name, trans, _ in self.transformers:
                if name == ind:
                    return trans
            raise KeyError(f"Transformer {ind} not found")

    def _log_message(self, name: str, idx: int, total: int) -> str:
        """Get log message for a transformer.

        Parameters
        ----------
        name : str
            Transformer name.
        idx : int
            Current index.
        total : int
            Total number of transformers.

        Returns
        -------
        message : str
            Log message.

        """
        return f"(step {idx} of {total}) Processing {name}"

    def _update_fitted_transformers(self, transformers: Any) -> None:
        """Update fitted transformers.

        Parameters
        ----------
        transformers : Any
            Fitted transformers.


        """
        # Directly use sklearn's implementation - it's tightly coupled with internal state
        sklearn_ColumnTransformer._update_fitted_transformers(self, transformers)  # type: ignore[arg-type]

    def _get_feature_name_out_for_transformer(self, name: str, trans: Any, feature_names_in: Any) -> Any:
        """Get feature names for a transformer.

        Parameters
        ----------
        name : str
            Transformer name.
        trans : Any
            Transformer instance.
        feature_names_in : Any
            Input feature names.

        Returns
        -------
        feature_names_out : Any
            Output feature names.

        """
        return sklearn_ColumnTransformer._get_feature_name_out_for_transformer(
            self,
            name,
            trans,
            feature_names_in,  # type: ignore[arg-type]
        )

    def get_feature_names_out(self, input_features: list[str] | None = None) -> Any:
        """Get output feature names.

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

    def _get_remainder_cols(self, indices: Any) -> Any:
        """Get remainder columns.

        Parameters
        ----------
        indices : Any
            Column indices.

        Returns
        -------
        remainder_cols : Any
            Remainder columns.

        """
        # Directly use sklearn's implementation - it calls _get_remainder_cols_dtype internally
        return sklearn_ColumnTransformer._get_remainder_cols(self, indices)  # type: ignore[arg-type]

    def _get_remainder_cols_dtype(self) -> Any:
        """Get dtype of remainder columns.

        Returns
        -------
        dtype : Any
            Data type of remainder columns.

        """
        return sklearn_ColumnTransformer._get_remainder_cols_dtype(self)  # type: ignore[arg-type]

    def _add_prefix_for_feature_names_out(self, feature_names_out: Any) -> Any:
        """Add prefixes to feature names.

        Parameters
        ----------
        feature_names_out : Any
            Feature names from transformers.

        Returns
        -------
        prefixed_names : Any
            Feature names with prefixes.

        """
        return sklearn_ColumnTransformer._add_prefix_for_feature_names_out(
            self,
            feature_names_out,  # type: ignore[arg-type]
        )

    def _sk_visual_block_(self) -> Any:
        """Get visual block representation.

        Returns
        -------
        visual_block : Any
            Visual block representation.

        """
        return sklearn_ColumnTransformer._sk_visual_block_(self)  # type: ignore[arg-type]

    def _validate_remainder(self, X: Any) -> None:
        """Validate remainder parameter.

        Parameters
        ----------
        X : Any
            Input data.

        """
        # Let sklearn handle validation completely
        sklearn_ColumnTransformer._validate_remainder(self, X)  # type: ignore[arg-type]

    def _validate_column_callables(self, X: Any) -> None:
        """Validate column callables.

        Parameters
        ----------
        X : Any
            Input data.

        """
        # Let sklearn handle validation
        sklearn_ColumnTransformer._validate_column_callables(self, X)  # type: ignore[arg-type]

    def _record_output_indices(self, Xs: Any) -> None:
        """Record output indices for each transformer.

        Parameters
        ----------
        Xs : Any
            Transformed outputs.

        """
        # Let sklearn handle recording
        sklearn_ColumnTransformer._record_output_indices(self, Xs)  # type: ignore[arg-type]

    def __init__(
        self,
        transformers: list[tuple[str, Any, Any]],
        *,
        remainder: str | Any = "drop",
        n_jobs: int | None = None,
        transformer_weights: dict[str, float] | None = None,
        verbose: bool = False,
        verbose_feature_names_out: bool = True,
    ) -> None:
        self.transformers = transformers
        self.remainder = remainder
        self.n_jobs = n_jobs
        self.transformer_weights = transformer_weights
        self.verbose = verbose
        self.verbose_feature_names_out = verbose_feature_names_out

    def _get_observation_horizons(self) -> list[int]:
        """Get observation horizons from all fitted transformers.

        Returns
        -------
        observation_horizons : list[int]
            List of observation horizons from each transformer.

        """
        observation_horizons = []
        for _, t, _, _ in self._iter(
            fitted=True,
            column_as_labels=True,
            skip_drop=False,
            skip_empty_columns=False,
        ):
            observation_horizon = 0
            if t not in ("drop", "passthrough") and t is not None and hasattr(t, "observation_horizon"):
                observation_horizon = t.observation_horizon

            observation_horizons.append(observation_horizon)

        return observation_horizons

    @property
    def observation_horizon(self) -> int:
        """Maximum observation horizon across all transformers.

        Returns
        -------
        int
            Maximum observation horizon needed.

        Raises
        ------
        NotFittedError
            If the column transformer has not been fitted yet.

        """
        check_is_fitted(self)

        observation_horizons = self._get_observation_horizons()
        observation_horizon = max(observation_horizons)

        return observation_horizon

    def _validate_transformers(self) -> None:
        """Validate names of transformers and the transformers themselves.

        This checks whether given transformers have the required methods, i.e.
        `fit` or `fit_transform` and `transform` implemented.
        """
        if not self.transformers:
            return

        names, transformers, _ = zip(*self.transformers, strict=False)

        # validate names
        self._validate_names(names)

        # validate estimators
        for t in transformers:
            if t == "passthrough":
                continue
            if not isinstance(t, BaseTransformer):
                # Used to validate the transformers in the `transformers` list
                raise TypeError(
                    "All estimators should be instances of `BaseTransformer` "
                    "or be the string 'passthrough' "
                    "'%s' (type %s) doesn't" % (t, type(t))
                )

    def _call_func_on_transformers(
        self,
        X: pl.DataFrame,
        y: pl.DataFrame | None,
        func: Callable,
        column_as_labels: bool,
        routed_params: dict[str, dict[str, dict[str, Any]]],
    ) -> list[pl.DataFrame]:
        """
        Private function to fit and/or transform on demand.

        Parameters
        ----------
        X : {array-like, dataframe} of shape (n_samples, n_features)
            The data to be used in fit and/or transform.

        y : array-like of shape (n_samples,)
            Targets.

        func : callable
            Function to call, which can be _fit_transform_one or
            _transform_one.

        column_as_labels : bool
            Used to iterate through transformers. If True, columns are returned
            as strings. If False, columns are returned as they were given by
            the user. Can be True only if the ``ColumnTransformer`` is already
            fitted.

        routed_params : dict
            The routed parameters as the output from ``process_routing``.

        Returns
        -------
        Return value (transformers and/or transformed X data) depends
        on the passed function.
        """
        time = X.select(cs.by_name("time"))
        X = X.select(~cs.by_name("time"))

        if func is _fit_transform_one:
            fitted = False
        else:  # func is _transform_one
            fitted = True

        def safe_indexing(X: pl.DataFrame, columns: object, axis: int) -> object:
            """Safe indexing helper for polars DataFrames."""
            Xi = _safe_indexing(X, columns, axis=axis)

            if isinstance(Xi, pl.Series):
                Xi = Xi.to_frame()

            return Xi

        transformers = list(
            self._iter(
                fitted=fitted,
                column_as_labels=column_as_labels,
                skip_drop=True,
                skip_empty_columns=True,
            )
        )
        try:
            jobs = []
            for idx, (name, trans, column, weight) in enumerate(transformers, start=1):
                if func is _fit_transform_one:
                    if trans == "passthrough":
                        output_config = _get_output_config("transform", self)
                        trans = FunctionTransformer(
                            check_inverse=False,
                            feature_names_out="one-to-one",
                        ).set_output(transform=output_config["dense"])

                    extra_args = dict(
                        message_clsname="ColumnTransformer",
                        message=self._log_message(name, idx, len(transformers)),
                    )
                else:  # func is _transform_one
                    extra_args = {}
                jobs.append(
                    delayed(func)(
                        transformer=clone(trans) if not fitted else trans,
                        X=pl.concat(
                            [time, safe_indexing(X, column, axis=1)],  # type: ignore[list-item]
                            how="horizontal",
                        ),
                        y=y,
                        weight=weight,
                        **extra_args,
                        params=routed_params[name],
                    )
                )

            return Parallel(n_jobs=self.n_jobs)(jobs)  # type: ignore[no-any-return]

        except ValueError as e:
            if "Expected 2D array, got 1D array instead" in str(e):
                raise ValueError(_ERR_MSG_1DCOLUMN) from e
            else:
                raise

    def fit(self, X: pl.DataFrame, y: pl.DataFrame | None = None, **params: Any) -> "ColumnTransformer":
        """Fit all transformers using X.

        Parameters
        ----------
        X : {array-like, dataframe} of shape (n_samples, n_features)
            Input data, of which specified subsets are used to fit the
            transformers.

        y : array-like of shape (n_samples,...), default=None
            Targets for supervised learning.

        **params : dict, default=None
            Parameters to be passed to the underlying transformers' ``fit`` and
            ``transform`` methods.

            You can only pass this if metadata routing is enabled, which you
            can enable using ``sklearn.set_config(enable_metadata_routing=True)``.

        Returns
        -------
        self : ColumnTransformer
            This estimator.
        """
        _raise_for_params(params, self, "fit")
        # we use fit_transform to make sure to set sparse_output_ (for which we
        # need the transformed data) to have consistent output type in predict
        self.fit_transform(X, y=y, **params)
        return self

    @_fit_context(  # type: ignore[untyped-decorator]
        # estimators in ColumnTransformer.transformers are not validated yet
        prefer_skip_nested_validation=False
    )
    def fit_transform(self, X: pl.DataFrame, y: pl.DataFrame | None = None, **params: Any) -> pl.DataFrame:
        """Fit all transformers, transform the data and concatenate results.

        Parameters
        ----------
        X : {array-like, dataframe} of shape (n_samples, n_features)
            Input data, of which specified subsets are used to fit the
            transformers.

        y : array-like of shape (n_samples,), default=None
            Targets for supervised learning.

        **params : dict, default=None
            Parameters to be passed to the underlying transformers' ``fit`` and
            ``transform`` methods.

            You can only pass this if metadata routing is enabled, which you
            can enable using ``sklearn.set_config(enable_metadata_routing=True)``.

        Returns
        -------
        X_t : {array-like, sparse matrix} of \
                shape (n_samples, sum_n_components)
            Horizontally stacked results of transformers. sum_n_components is the
            sum of n_components (output dimension) over transformers. If
            any result is a sparse matrix, everything will be converted to
            sparse matrices.
        """
        _raise_for_params(params, self, "fit_transform")
        _check_feature_names(self, X, reset=True)

        X = _check_X(X)
        # set n_features_in_ attribute
        _check_n_features(self, X, reset=True)
        self._validate_transformers()
        n_samples = _num_samples(X)

        self._validate_column_callables(X)
        self._validate_remainder(X)

        routed_params = process_routing(self, "fit_transform", **params)

        result = self._call_func_on_transformers(
            X,
            y,
            _fit_transform_one,
            column_as_labels=False,
            routed_params=routed_params,
        )

        if not result:
            self._update_fitted_transformers([])
            # All transformers are None
            time = X.select(cs.by_name("time"))
            return time

        Xs, transformers = zip(*result, strict=False)

        self.sparse_output_ = False

        self._update_fitted_transformers(transformers)
        self._record_output_indices(Xs)

        result = self._hstack(list(Xs), n_samples=n_samples)
        return result  # type: ignore[return-value]

    def transform(self, X: pl.DataFrame, **params: Any) -> pl.DataFrame:
        """Transform X separately by each transformer, concatenate results.

        Parameters
        ----------
        X : {array-like, dataframe} of shape (n_samples, n_features)
            The data to be transformed by subset.

        **params : dict, default=None
            Parameters to be passed to the underlying transformers' ``transform``
            method.

            You can only pass this if metadata routing is enabled, which you
            can enable using ``sklearn.set_config(enable_metadata_routing=True)``.

        Returns
        -------
        X_t : {array-like, sparse matrix} of \
                shape (n_samples, sum_n_components)
            Horizontally stacked results of transformers. sum_n_components is the
            sum of n_components (output dimension) over transformers. If
            any result is a sparse matrix, everything will be converted to
            sparse matrices.
        """
        _raise_for_params(params, self, "transform")
        check_is_fitted(self)
        X = _check_X(X)

        # If ColumnTransformer is fit using a dataframe, and now a dataframe is
        # passed to be transformed, we select columns by name instead. This
        # enables the user to pass X at transform time with extra columns which
        # were not present in fit time, and the order of the columns doesn't
        # matter.
        fit_dataframe_and_transform_dataframe = hasattr(self, "feature_names_in_") and (
            is_pandas_df(X) or hasattr(X, "__dataframe__")
        )

        n_samples = _num_samples(X)
        column_names = _get_feature_names(X)

        if fit_dataframe_and_transform_dataframe:
            named_transformers = self.named_transformers_  # type: ignore[attr-defined]
            # check that all names seen in fit are in transform, unless
            # they were dropped
            non_dropped_indices = [
                ind
                for name, ind in self._transformer_to_input_indices.items()  # type: ignore[attr-defined]
                if name in named_transformers and named_transformers[name] != "drop"
            ]

            all_indices = set(chain(*non_dropped_indices))
            all_names = set(self.feature_names_in_[ind] for ind in all_indices)

            diff = all_names - set(column_names)
            if diff:
                raise ValueError(f"columns are missing: {diff}")
        else:
            # ndarray was used for fitting or transforming, thus we only
            # check that n_features_in_ is consistent
            self._check_n_features(X, reset=False)  # type: ignore[attr-defined]

        routed_params = process_routing(self, "transform", **params)

        Xs = self._call_func_on_transformers(
            X,
            None,
            _transform_one,
            column_as_labels=fit_dataframe_and_transform_dataframe,
            routed_params=routed_params,
        )

        if not Xs:
            # All transformers are None
            time = X.select(cs.by_name("time"))
            return time

        result = self._hstack(list(Xs), n_samples=n_samples)
        return result  # type: ignore[return-value]

    def update_transform(self, X: pl.DataFrame, **params: Any) -> pl.DataFrame:
        """Update and transform X by each transformer, concatenate results.

        This method atomically updates each column transformer with new data and
        transforms it. The transformation uses the pre-update state, then updates
        the memory. This is more efficient and correct than calling update() then
        transform() separately.

        Parameters
        ----------
        X : pl.DataFrame
            New data to update with and transform.

        **params : dict, default=None
            Parameters routed to the `transform` methods of the transformers.

            You can only pass this if metadata routing is enabled, which you
            can enable using ``sklearn.set_config(enable_metadata_routing=True)``.

        Returns
        -------
        X_t : pl.DataFrame
            Horizontally stacked results of transformers.

        """
        _raise_for_params(params, self, "update_transform")
        check_is_fitted(self)
        X = _check_X(X)

        n_samples = _num_samples(X)

        routed_params = process_routing(self, "update_transform", **params)

        Xs = self._call_func_on_transformers(
            X,
            None,
            _update_transform_one,
            column_as_labels=False,
            routed_params=routed_params,
        )

        if not Xs:
            # All transformers are None
            time = X.select(cs.by_name("time"))
            return time

        result = self._hstack(list(Xs), n_samples=n_samples)

        return result

    def _hstack(self, Xs: list[pl.DataFrame], *, n_samples: int) -> pl.DataFrame:
        """Stacks Xs horizontally.

        This allows subclasses to control the stacking behavior, while reusing
        everything else from ColumnTransformer.

        Parameters
        ----------
        Xs : list of {array-like, sparse matrix, dataframe}
            The container to concatenate.

        n_samples : int
            The number of samples in the input data to checking the transformation
            consistency.
        """
        # rename before stacking as it avoids to error on temporary duplicated
        # columns
        transformer_names = [
            t[0]
            for t in self._iter(
                fitted=True,
                column_as_labels=False,
                skip_drop=True,
                skip_empty_columns=True,
            )
        ]
        feature_names_outs = [[col for col in X.columns if col != "time"] for X in Xs if X.shape[1] != 1]
        if self.verbose_feature_names_out:
            # `_add_prefix_for_feature_names_out` takes care about raising
            # an error if there are duplicated columns.
            feature_names_outs = self._add_prefix_for_feature_names_out(
                list(zip(transformer_names, feature_names_outs, strict=False))
            )
        else:
            # check for duplicated columns and raise if any
            feature_names_outs = list(chain.from_iterable(feature_names_outs))
            feature_names_count = Counter(feature_names_outs)
            if any(count > 1 for count in feature_names_count.values()):
                duplicated_feature_names = sorted(name for name, count in feature_names_count.items() if count > 1)
                err_msg = (
                    "Duplicated feature names found before concatenating the"
                    " outputs of the transformers:"
                    f" {duplicated_feature_names}.\n"
                )
                for transformer_name, X in zip(transformer_names, Xs, strict=False):
                    if X.shape[1] == 1:
                        continue
                    dup_cols_in_transformer = sorted(set(X.columns).intersection(duplicated_feature_names))
                    if dup_cols_in_transformer:
                        err_msg += (
                            f"Transformer {transformer_name} has conflicting "
                            f"columns names: {dup_cols_in_transformer}.\n"
                        )
                raise ValueError(
                    err_msg + "Either make sure that the transformers named above "
                    "do not generate columns with conflicting names or set "
                    "verbose_feature_names_out=True to automatically "
                    "prefix to the output feature names with the name "
                    "of the transformer to prevent any conflicting "
                    "names."
                )

        output = _hstack(
            Xs,
            column_names=feature_names_outs,
            observation_horizons=self._get_observation_horizons(),
        )
        output_samples = output.shape[0]
        if output_samples != n_samples - self.observation_horizon:
            raise ValueError(
                "Concatenating DataFrames from the transformer's output lead to an inconsistent number of samples."
            )

        return output

    def get_metadata_routing(self) -> MetadataRouter:
        """Get metadata routing of this object.

        Please check :ref:`User Guide <metadata_routing>` on how the routing
        mechanism works.

        Returns
        -------
        routing : MetadataRouter
            A :class:`~sklearn.utils.metadata_routing.MetadataRouter` encapsulating
            routing information.
        """
        router = MetadataRouter(owner=self)
        # Here we don't care about which columns are used for which
        # transformers, and whether or not a transformer is used at all, which
        # might happen if no columns are selected for that transformer. We
        # request all metadata requested by all transformers.
        transformers = chain(self.transformers, [("remainder", self.remainder, None)])
        for name, step, _ in transformers:
            method_mapping = MethodMapping()
            if hasattr(step, "fit_transform"):
                (
                    method_mapping.add(caller="fit", callee="fit_transform").add(
                        caller="fit_transform", callee="fit_transform"
                    )
                )
            else:
                (
                    method_mapping.add(caller="fit", callee="fit")
                    .add(caller="fit", callee="transform")
                    .add(caller="fit_transform", callee="fit")
                    .add(caller="fit_transform", callee="transform")
                )
            method_mapping.add(caller="transform", callee="transform")
            router.add(method_mapping=method_mapping, **{name: step})

        return router


def _update_transform_one(
    transformer: Any, X: pl.DataFrame, y: None, weight: float | None, routed_params: Any
) -> pl.DataFrame:
    """Update and transform data using a single transformer.

    Parameters
    ----------
    transformer : estimator
        The transformer to update and transform with.
    X : pl.DataFrame
        Input data to update and transform.
    y : None
        Not used, present for API consistency.
    weight : float | None
        Weight to apply to transformed output.
    routed_params : Any
        Routed parameters for the transformer.

    Returns
    -------
    pl.DataFrame
        Transformed data.

    """
    X_transformed = transformer.update_transform(X, **routed_params.get("update_transform", {}))
    if weight is None:
        return X_transformed
    return X_transformed * weight
