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
    "title": "How to Transform Features on the Forecast Channel",
    "description": "Lift stateless transformers onto the vintage axis with PerVintageActualTransformer, compose them with FeatureUnion, and feed the result to a forecaster's X_forecast channel.",
    "category": "how-to",
    "section": "data-features",
    "companion": "/pages/how-to/transform-forecast-features/",
    "api_references": [
        "PerVintageActualTransformer",
        "FunctionTransformer",
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
        # How to Transform Features on the Forecast Channel

        This notebook derives features from an `X_forecast` frame using
        [`PerVintageActualTransformer`](/pages/api/generated/yohou.compose.per_vintage.PerVintageActualTransformer/),
        composes several of them with
        [`FeatureUnion`](/pages/api/generated/yohou.compose.feature_union.FeatureUnion/),
        and passes the result to a forecaster's `X_forecast` channel.

        **Prerequisites:** Familiarity with the three exogenous types
        ([Exogenous Features](/pages/tutorials/exogenous-features/)) and
        forecast vintages ([Work with Forecast Vintages](/pages/how-to/forecast-vintages/)).
        For why the two transformer kinds exist, see
        [Transformer Kinds](/pages/explanation/transformer-kinds/).
        """
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"## 1. An X_forecast Frame Has Two Time Axes")


@app.cell
def _():
    import polars as pl

    from yohou.datasets import make_exogenous_regression

    H = 6
    data = make_exogenous_regression(n_samples=120, forecasting_horizon=H)
    y, X_actual, X_forecast = data.y, data.X_actual, data.X_forecast

    print("X_forecast columns:", X_forecast.columns)
    print("vintages:", X_forecast["vintage_time"].n_unique())
    print(X_forecast.head(4))

    return H, X_actual, X_forecast, pl, y


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Each row is identified by `vintage_time` (when the forecast was issued)
        *and* `time` (what it forecasts). Several vintages say something about the
        same `time`, so `time` alone does not identify a row. That is why an
        ordinary single-axis transformer cannot consume this frame.

        ## 2. Lift a Stateless Transformer onto the Vintage Axis
        """
    )


@app.cell
def _(X_forecast, pl):
    from yohou.compose import PerVintageActualTransformer
    from yohou.preprocessing import FunctionTransformer

    anomaly = PerVintageActualTransformer(
        FunctionTransformer(
            func=lambda df: df.select(
                (pl.col("wx_temp") - pl.col("wx_temp").mean()).alias("wx_anomaly")
            ),
            feature_names_out=lambda self, names: ["wx_anomaly"],
        )
    )

    X_forecast_t = anomaly.fit_transform(X_forecast)

    print("out columns:", X_forecast_t.columns)
    print(X_forecast_t.head(4))

    return FunctionTransformer, PerVintageActualTransformer, X_forecast_t, anomaly


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The `vintage_time` and `time` index columns survive, and each vintage's
        anomaly is computed against that vintage's own mean. Nothing is borrowed
        from a neighbouring vintage.

        ## 3. Each Vintage Is Transformed Independently
        """
    )


@app.cell
def _(X_forecast_t, pl):
    # Each vintage's anomalies are centred on that vintage alone, so they sum to
    # roughly zero within every vintage rather than only across the frame.
    per_vintage = (
        X_forecast_t.group_by("vintage_time")
        .agg(pl.col("wx_anomaly").mean().alias("mean_anomaly"))
        .sort("vintage_time")
    )

    print(per_vintage.head(4))
    print("max |mean| across vintages:", per_vintage["mean_anomaly"].abs().max())

    return (per_vintage,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 4. Fit a Transformer per Vintage

        The anomaly above computed its mean inline with a `FunctionTransformer`.
        The wrapper's real value is lifting a transformer that *fits* from data: a
        `SimpleImputer` fills each vintage's missing values from that vintage's own
        mean. It is leakage-free because a vintage is fully known at its
        `vintage_time`. (To normalize against the *training* distribution instead,
        put the scaler in the estimator pipeline, not here.)
        """
    )


@app.cell
def _(PerVintageActualTransformer, X_forecast, pl):
    from yohou.preprocessing import SimpleImputer

    # Knock a gap into the second row of every vintage, then impute per vintage.
    X_gappy = X_forecast.with_columns(
        pl.when(pl.int_range(pl.len()).over("vintage_time") == 1)
        .then(None)
        .otherwise(pl.col("wx_temp"))
        .alias("wx_temp")
    )
    imputed = PerVintageActualTransformer(SimpleImputer(strategy="mean")).fit_transform(X_gappy)

    print("gaps introduced:", X_gappy["wx_temp"].null_count())
    print("gaps remaining after per-vintage impute:", imputed["wx_temp"].null_count())
    print(imputed.head(4))
    return SimpleImputer, X_gappy, imputed


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 5. Compose Forecast Transformers

        A [`FeatureUnion`](/pages/api/generated/yohou.compose.feature_union.FeatureUnion/)
        of forecast-kind branches is itself forecast-kind, and it aligns its branches
        on `vintage_time` and `time` together rather than on `time` alone.
        """
    )


@app.cell
def _(FunctionTransformer, PerVintageActualTransformer, X_forecast, anomaly, pl):
    from yohou.compose import FeatureUnion

    squared = PerVintageActualTransformer(
        FunctionTransformer(
            func=lambda df: df.select((pl.col("wx_temp") ** 2).alias("wx_sq")),
            feature_names_out=lambda self, names: ["wx_sq"],
        )
    )

    features = FeatureUnion(
        transformer_list=[("anom", anomaly), ("sq", squared)],
    )
    X_forecast_u = features.fit_transform(X_forecast)

    print("union kind:", features.__sklearn_tags__().transformer_tags.kind)
    print("union columns:", X_forecast_u.columns)

    return FeatureUnion, X_forecast_u, features, squared


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 6. Two Constraints the Framework Enforces

        Each branch must be **stateless**, since each is lifted independently, and a
        composition must be **homogeneous in kind**. Both are errors rather than
        silent wrong answers.
        """
    )


@app.cell
def _(FeatureUnion, FunctionTransformer, PerVintageActualTransformer, X_forecast, anomaly, pl):
    from yohou.preprocessing import LagTransformer

    # A stateful inner transformer cannot be lifted: the vintage axis is
    # discontinuous, so there is no contiguous history for its buffer to hold.
    try:
        PerVintageActualTransformer(
            FunctionTransformer(
                func=lambda df: df.select(pl.col("wx_temp").diff().alias("ramp")),
                feature_names_out=lambda self, names: ["ramp"],
            )
        ).fit(X_forecast)
    except ValueError as err:
        print("stateless rule:", err)

    # Mixing kinds in one container has no correct alignment, so it is rejected.
    try:
        FeatureUnion(
            transformer_list=[("anom", anomaly), ("lag", LagTransformer([1]))],
        ).fit(X_forecast)
    except ValueError as err:
        print("\nhomogeneity rule:", err)

    return (LagTransformer,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 7. Feed the Result to a Forecaster

        Derived forecast features enter through the `X_forecast` channel. They cannot
        be passed as `feature_transformer`, which processes single-axis `X_actual` data
        and accepts actual-kind transformers only.
        """
    )


@app.cell
def _(H, X_actual, X_forecast_t, y):
    from sklearn.ensemble import HistGradientBoostingRegressor

    from yohou.model_selection import train_test_split
    from yohou.point import PointReductionForecaster

    y_train, _y_test, X_actual_train, _X_actual_test = train_test_split(
        y, X_actual, test_size=20
    )

    forecaster = PointReductionForecaster(
        estimator=HistGradientBoostingRegressor(max_iter=30, max_depth=3, random_state=0),
        reduction_strategy="direct",
    )
    forecaster.fit(
        y=y_train,
        X_actual=X_actual_train,
        forecasting_horizon=H,
        X_forecast=X_forecast_t,
    )
    y_pred = forecaster.predict(forecasting_horizon=H, X_forecast=X_forecast_t)

    print(y_pred.head())

    return (
        HistGradientBoostingRegressor,
        PointReductionForecaster,
        forecaster,
        train_test_split,
        y_pred,
        y_train,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Next Steps

        - [Transform Forecast Features](/pages/how-to/transform-forecast-features/) for the full guide
        - [Transformer Kinds](/pages/explanation/transformer-kinds/) for why the vintage axis rules out statefulness
        - [Feature Pipelines](/pages/explanation/feature-pipelines/) for how composition derives its kind
        """
    )
    return


if __name__ == "__main__":
    app.run()
