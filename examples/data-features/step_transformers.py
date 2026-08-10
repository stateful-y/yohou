# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "numpy",
#     "scikit-learn",
#     "yohou[plotting]",
# ]
# ///

import marimo

__generated_with = "0.23.1"
__gallery__ = {
    "title": "How to Reduce Forecast Step Features",
    "description": "Shrink the step columns a forecaster derives from X_future and X_forecast using the step_transformer slot, with StepAggregator, lifted scikit-learn reducers, and composition for keeping raw steps alongside summaries.",
    "category": "how-to",
    "section": "data-features",
    "companion": "/pages/how-to/reduce-step-features/",
    "api_references": [
        "StepAggregator",
        "StepColumnReducer",
        "StepFrameReducer",
        "FeatureUnion",
        "PointReductionForecaster",
        "make_exogenous_regression",
    ],
}
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # How to Reduce Forecast Step Features

        A forecaster turns each exogenous column into a block of step columns, one
        per horizon step. Over a long horizon that is a lot of columns for a
        little information. This notebook shrinks those blocks with the
        `step_transformer` slot, using
        [`StepAggregator`](/pages/api/generated/yohou.preprocessing.step.StepAggregator/)
        and the two scikit-learn wrappers
        [`StepColumnReducer`](/pages/api/generated/yohou.preprocessing.step.StepColumnReducer/)
        and
        [`StepFrameReducer`](/pages/api/generated/yohou.preprocessing.step.StepFrameReducer/).

        **Prerequisites:** Familiarity with the three exogenous types
        ([Exogenous Features](/pages/tutorials/exogenous-features/)). For why the
        step frame is its own transformer kind, see
        [Transformer Kinds](/pages/explanation/transformer-kinds/).
        """
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"## 1. One Column Becomes H Columns")


@app.cell
def _():
    from sklearn.linear_model import Ridge

    from yohou.datasets import make_exogenous_regression
    from yohou.point import PointReductionForecaster

    H = 12
    data = make_exogenous_regression(n_samples=200, forecasting_horizon=H)
    X_actual, X_future = data.X_actual, data.X_future
    # Trim the last H observations: their forward window runs past the end of
    # X_future, so they would be partially covered. Section 6 covers that case
    # deliberately; here it would only be noise.
    y = data.y[:-H]

    plain = PointReductionForecaster(estimator=Ridge(), nan_handling="drop")
    plain.fit(y, X_actual, forecasting_horizon=H, X_future=X_future)

    print("X_future columns:", [c for c in X_future.columns if c != "time"])
    print("derived step columns:", len(plain._step_column_names_))
    print(sorted(plain._step_column_names_)[:4], "...")

    return H, PointReductionForecaster, Ridge, X_actual, X_future, y


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Each known-future column expands to `{base}_step_1` through
        `{base}_step_H`, holding that column's values over the horizon ahead of
        each observation. That block is the only place in the pipeline where "the
        H values ahead of now" exist as one aligned row, which is what makes a
        transformation along the horizon expressible.

        ## 2. Summarise Each Block
        """
    )


@app.cell
def _(H, PointReductionForecaster, Ridge, X_actual, X_future, y):
    from yohou.preprocessing import StepAggregator

    summarised = PointReductionForecaster(
        estimator=Ridge(),
        nan_handling="drop",
        step_transformer=StepAggregator(aggregations=("min", "max", "mean")),
    )
    summarised.fit(y, X_actual, forecasting_horizon=H, X_future=X_future)

    print("step columns now:", sorted(summarised._step_column_names_))

    return StepAggregator, summarised


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Three columns per variable instead of twelve. The output names carry no
        horizon index, which matters for the `"direct"` reduction strategy: a
        summary describes the whole block, so it reaches every per-step estimator
        rather than being filtered by `step_feature_alignment`.

        The available aggregations are `min`, `max`, `mean`, `std`, and `sum`.
        The set is closed on purpose, so that there is one way to express an
        unusual reduction rather than two (see section 5).

        ## 3. Keep the Raw Steps as Well
        """
    )


@app.cell
def _(H, PointReductionForecaster, Ridge, StepAggregator, X_actual, X_future, y):
    from yohou.compose import FeatureUnion

    both = PointReductionForecaster(
        estimator=Ridge(),
        nan_handling="drop",
        step_transformer=FeatureUnion([
            ("raw", "passthrough"),
            ("agg", StepAggregator(aggregations=("mean",))),
        ]),
    )
    both.fit(y, X_actual, forecasting_horizon=H, X_future=X_future)

    names = sorted(both._step_column_names_)
    print("summaries:", [n for n in names if n.endswith("_step_mean")])
    print("raw steps kept:", len([n for n in names if not n.endswith("_step_mean")]))

    return (FeatureUnion,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The slot replaces the block rather than adding to it, so keeping both is
        a composition. There is no `keep_step_columns` flag, because a
        `FeatureUnion` already expresses it.

        ## 4. Learn a Reduction Instead of Fixing One
        """
    )


@app.cell
def _(H, PointReductionForecaster, Ridge, X_actual, X_future, y):
    from sklearn.decomposition import PCA

    from yohou.preprocessing import StepColumnReducer, StepFrameReducer

    per_column = PointReductionForecaster(
        estimator=Ridge(),
        nan_handling="drop",
        step_transformer=StepColumnReducer(reducer=PCA(n_components=2)),
    )
    per_column.fit(y, X_actual, forecasting_horizon=H, X_future=X_future)

    whole_frame = PointReductionForecaster(
        estimator=Ridge(),
        nan_handling="drop",
        step_transformer=StepFrameReducer(reducer=PCA(n_components=3), prefix="wx"),
    )
    whole_frame.fit(y, X_actual, forecasting_horizon=H, X_future=X_future)

    print("per column :", sorted(per_column._step_column_names_))
    print("whole frame:", sorted(whole_frame._step_column_names_))

    return PCA, StepColumnReducer


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The two wrappers differ only in what they hand to the inner estimator.
        `StepColumnReducer` fits one clone per variable, so each variable's
        horizon profile is compressed on its own terms and the output still says
        which variable it came from. `StepFrameReducer` fits one clone over every
        step column at once, so it can exploit correlation between variables, at
        the cost of that provenance; its `prefix` names the resulting block.

        Both require an inner estimator whose output width is fixed before fit.
        A `PCA(n_components=0.95)` is refused, because under panel data each group
        is fitted separately and groups producing different widths would not share
        a column schema.

        ## 5. Anything Else Goes Through a FunctionTransformer
        """
    )


@app.cell
def _(StepColumnReducer):
    import numpy as np
    import polars as pl
    from sklearn.preprocessing import FunctionTransformer as SklearnFunctionTransformer

    from datetime import datetime, timedelta

    block = pl.DataFrame({
        "time": [datetime(2024, 1, 1) + timedelta(days=i) for i in range(5)],
        "temp_step_1": [1.0, 2.0, 3.0, 4.0, 5.0],
        "temp_step_2": [2.0, 4.0, 6.0, 8.0, 10.0],
        "temp_step_3": [9.0, 1.0, 4.0, 2.0, 8.0],
    })

    p90 = SklearnFunctionTransformer(lambda a: np.percentile(a, 90, axis=1, keepdims=True))
    print(StepColumnReducer(reducer=p90).fit_transform(block))

    return block, pl


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        A 90th percentile is not in `StepAggregator`'s vocabulary, and does not
        need to be: wrapping a `FunctionTransformer` covers it. Keeping one
        extension mechanism rather than two is why the vocabulary stays closed.

        ## 6. Partial Coverage
        """
    )


@app.cell
def _(StepAggregator, block, pl):
    partial = block.with_columns(
        pl.when(pl.arange(0, 5) < 2).then(None).otherwise(pl.col("temp_step_3")).alias("temp_step_3")
    )

    ignored = StepAggregator(aggregations=("mean",), emit_coverage=True).fit_transform(partial)
    propagated = StepAggregator(aggregations=("mean",), null_policy="propagate").fit_transform(partial)

    print("ignore   :", ignored["temp_step_mean"].to_list())
    print("covered  :", ignored["temp_step_n_covered"].to_list())
    print("propagate:", propagated["temp_step_mean"].to_list())

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Forecast data does not always reach the full horizon, and the missing
        steps arrive as nulls. `"ignore"` summarises whatever is present, which
        means a mean over two steps and a mean over three share a column name and
        the model cannot tell them apart. `emit_coverage=True` adds the count so
        it can; it is off by default because under full coverage that column is
        constant.

        The wrappers have no null policy of their own, because the decision
        belongs to the inner estimator. If the inner estimator cannot cope, the
        fit raises naming the variable and its coverage, and suggests composing a
        `SimpleImputer` into the inner estimator, which works:
        """
    )


@app.cell
def _(PCA, StepColumnReducer, block, pl):
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline

    partial_block = block.with_columns(
        pl.when(pl.arange(0, 5) < 2).then(None).otherwise(pl.col("temp_step_3")).alias("temp_step_3")
    )

    try:
        StepColumnReducer(reducer=PCA(n_components=2)).fit(partial_block)
    except ValueError as exc:
        print("refused:", str(exc)[:150], "...")

    imputing = StepColumnReducer(reducer=Pipeline([("i", SimpleImputer()), ("p", PCA(n_components=2))]))
    print("with imputer:", imputing.fit_transform(partial_block).columns)

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Summary

        - The `step_transformer` slot reduces the step columns derived from
          `X_future` and `X_forecast`, before they join the design matrix.
        - `StepAggregator` covers arithmetic summaries over a closed vocabulary;
          the two wrappers lift any scikit-learn transformer onto the step axis.
        - Choose `StepColumnReducer` to keep per-variable provenance, and
          `StepFrameReducer` to exploit correlation across variables.
        - Composition, not flags: `FeatureUnion` keeps raw steps alongside
          summaries, `ColumnTransformer` routes different variables to different
          reductions, and a `FunctionTransformer` covers anything unusual.
        - Partial coverage is a supported state. `StepAggregator` has a
          `null_policy` and an optional coverage column; the wrappers defer to
          the inner estimator and tell you what to compose when it cannot cope.

        **Next:** [How to Reduce Forecast Step Features](/pages/how-to/reduce-step-features/)
        for the reference version of this material, and
        [Transformer Kinds](/pages/explanation/transformer-kinds/) for why the
        step frame needs a kind of its own.
        """
    )


if __name__ == "__main__":
    app.run()
