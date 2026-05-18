# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "yohou[plotting]",
# ]
# ///

import marimo

__generated_with = "0.20.2"
__gallery__ = {
    "title": "How to Handle Missing Data",
    "description": "Compare SimpleTimeImputer, SeasonalImputer, SimpleImputer, and TransformedSpaceKNNImputer on synthetic block and scattered gaps in monthly tourism data.",
    "category": "how-to",
    "section": "data-features",
    "companion": "/pages/how-to/handle-missing-data/",
    "api_references": ["SeasonalImputer", "SimpleImputer", "SimpleTimeImputer", "TransformedSpaceKNNImputer"],
}
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # How to Handle Missing Data

    Missing values are common in real-world time series. Yohou provides
    temporal-aware imputation methods alongside sklearn wrappers.

    This notebook shows how to compare SimpleTimeImputer, SeasonalImputer, SimpleImputer, and TransformedSpaceKNNImputer on synthetic block and scattered gaps in monthly tourism data.
    """)


@app.cell(hide_code=True)
def _():
    import plotly.graph_objects as go
    import polars as pl

    from yohou.datasets import fetch_tourism_monthly
    from yohou.plotting import plot_time_series
    from yohou.preprocessing import SeasonalImputer, SimpleImputer, SimpleTimeImputer, TransformedSpaceKNNImputer

    return (
        SeasonalImputer,
        SimpleImputer,
        SimpleTimeImputer,
        TransformedSpaceKNNImputer,
        fetch_tourism_monthly,
        go,
        pl,
        plot_time_series,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Prepare the Data: Synthetic Gaps in Monthly Tourism

    We load the monthly tourism dataset with `fetch_tourism_monthly()`, pick a
    single series, and punch holes in it: a contiguous block of 5 missing months
    plus 4 scattered single-month gaps. This mix of gap patterns lets us compare
    how each imputer handles short isolated gaps versus longer contiguous ones.
    """)


@app.cell
def _(fetch_tourism_monthly, mo, pl):
    tourism = (
        fetch_tourism_monthly().frame.select("time", "T1__tourists").drop_nulls().rename({"T1__tourists": "tourists"})
    )

    # Create gaps: indices 30-34 (block), and scattered singles
    _gap_indices = list(range(30, 35)) + [50, 70, 90, 110]
    _mask = pl.Series([i in _gap_indices for i in range(len(tourism))])
    tourism_missing = tourism.with_columns(
        pl.when(_mask).then(None).otherwise(pl.col("tourists")).alias("tourists"),
    )
    _n_missing = tourism_missing.null_count()["tourists"][0]

    mo.md(
        f"**Original**: {len(tourism)} months, no gaps\n\n"
        f"**With gaps**: {_n_missing} missing values "
        f"(block of 5 + 4 scattered)"
    )
    return tourism, tourism_missing


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    [`plot_time_series`](/pages/api/generated/yohou.plotting.exploration.plot_time_series/) renders one or more numeric columns against the `"time"`
    axis. Gaps in the data appear as breaks in the line, making missing regions
    easy to spot visually.
    """)


@app.cell
def _(plot_time_series, tourism_missing):
    plot_time_series(tourism_missing, title="Monthly Tourism: With Synthetic Gaps")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. SimpleTimeImputer: Linear Interpolation

    [`SimpleTimeImputer`](/pages/api/generated/yohou.preprocessing.imputation.SimpleTimeImputer/) fills missing values using time-aware strategies.
    With `method="linear"`, it draws a straight line between the last known
    value before a gap and the first known value after it, then places the
    missing points along that line. This works well for short, smooth gaps.

    After imputation we use [`plot_time_series`](/pages/api/generated/yohou.plotting.exploration.plot_time_series/) to confirm the gaps are filled.
    """)


@app.cell
def _(SimpleTimeImputer, plot_time_series, tourism_missing):
    imp_linear = SimpleTimeImputer(method="linear")
    imp_linear.fit(tourism_missing)
    filled_linear = imp_linear.transform(tourism_missing)
    _combined = filled_linear.rename({"tourists": "imputed"}).join(
        tourism_missing.rename({"tourists": "with gaps"}),
        on="time",
    )
    plot_time_series(_combined, title="Linear Interpolation")
    return (filled_linear,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. SimpleTimeImputer: Forward Fill

    Forward fill (`method="forward"`) carries the last observed value forward
    through each gap. It preserves the level before the gap but introduces a
    flat segment, so it is most appropriate when gradual drift between
    observations is unlikely.
    """)


@app.cell
def _(SimpleTimeImputer, plot_time_series, tourism_missing):
    imp_forward = SimpleTimeImputer(method="forward")
    imp_forward.fit(tourism_missing)
    filled_forward = imp_forward.transform(tourism_missing)
    _combined = filled_forward.rename({"tourists": "imputed"}).join(
        tourism_missing.rename({"tourists": "with gaps"}),
        on="time",
    )
    plot_time_series(_combined, title="Forward Fill")
    return (filled_forward,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. SimpleTimeImputer with `limit`

    The `limit` parameter caps the maximum number of consecutive NaN
    values that are filled. Useful when you don't want to interpolate
    across long gaps. Here we set `limit=2`, so the block of 5
    consecutive missing values is only partially filled while shorter
    gaps are handled completely.
    """)


@app.cell
def _(SimpleTimeImputer, plot_time_series, tourism_missing):
    imp_limited = SimpleTimeImputer(method="forward", limit=2)
    imp_limited.fit(tourism_missing)
    filled_limited = imp_limited.transform(tourism_missing)
    _combined = filled_limited.rename({"tourists": "imputed (limit=2)"}).join(
        tourism_missing.rename({"tourists": "with gaps"}),
        on="time",
    )
    plot_time_series(_combined, title="Linear Interpolation with limit=2")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. SeasonalImputer

    [`SeasonalImputer`](/pages/api/generated/yohou.preprocessing.imputation.SeasonalImputer/) fills missing values by looking at the same position in
    previous seasonal cycles. Setting `period=12` on monthly data means a
    missing January is filled with the average of all other Januaries in the
    series. The `fill_method` parameter controls the aggregation:
    `"seasonal_mean"` uses the mean, while `"seasonal_median"` uses the median.
    This approach is effective when the series has strong, repeating seasonal
    patterns.
    """)


@app.cell
def _(SeasonalImputer, plot_time_series, tourism_missing):
    imp_seasonal = SeasonalImputer(period=12, fill_method="seasonal_mean")
    imp_seasonal.fit(tourism_missing)
    filled_seasonal = imp_seasonal.transform(tourism_missing)
    _combined = filled_seasonal.rename({"tourists": "imputed"}).join(
        tourism_missing.rename({"tourists": "with gaps"}),
        on="time",
    )
    plot_time_series(_combined, title="Seasonal Imputer (period=12, seasonal_mean)")
    return (filled_seasonal,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. Sklearn-Wrapped Imputers

    Yohou wraps several Scikit-Learn imputers so they work seamlessly with
    time-indexed Polars DataFrames.

    [`SimpleImputer`](/pages/api/generated/yohou.preprocessing.imputation.SimpleImputer/) replaces every missing value with a single summary
    statistic of the column (controlled by the `strategy` parameter:
    `"mean"`, `"median"`, or `"constant"`). It ignores temporal order
    entirely, which makes it a useful baseline.
    """)


@app.cell
def _(SimpleImputer, plot_time_series, tourism_missing):
    imp_mean = SimpleImputer(strategy="mean")
    imp_mean.fit(tourism_missing)
    filled_mean = imp_mean.transform(tourism_missing)
    _combined = filled_mean.rename({"tourists": "imputed"}).join(
        tourism_missing.rename({"tourists": "with gaps"}),
        on="time",
    )
    plot_time_series(_combined, title="SimpleImputer (strategy='mean')")
    return (filled_mean,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    [`TransformedSpaceKNNImputer`](/pages/api/generated/yohou.preprocessing.imputation.TransformedSpaceKNNImputer/) estimates each missing value from its
    `n_neighbors` nearest complete observations in feature space. This
    produces locally adaptive fills that can follow nonlinear patterns
    better than a global mean or median.
    """)


@app.cell
def _(TransformedSpaceKNNImputer, plot_time_series, tourism_missing):
    imp_knn = TransformedSpaceKNNImputer(n_neighbors=5)
    imp_knn.fit(tourism_missing)
    filled_knn = imp_knn.transform(tourism_missing)
    _combined = filled_knn.rename({"tourists": "imputed"}).join(
        tourism_missing.rename({"tourists": "with gaps"}),
        on="time",
    )
    plot_time_series(_combined, title="TransformedSpaceKNNImputer (n_neighbors=5)")
    return (filled_knn,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7. Compare Methods

    Because we created the gaps ourselves, we can measure each method's
    quality both visually and numerically.

    First we overlay all five imputed series on a single interactive chart.
    Click legend entries to toggle individual methods on and off, making it
    easy to compare how each one fills the same gaps.
    """)


@app.cell
def _(
    filled_forward,
    filled_knn,
    filled_linear,
    filled_mean,
    filled_seasonal,
    plot_time_series,
    tourism_missing,
):
    _all_methods = (
        tourism_missing
        .rename({"tourists": "with gaps"})
        .join(filled_linear.rename({"tourists": "Linear"}), on="time")
        .join(filled_forward.rename({"tourists": "Forward"}), on="time")
        .join(filled_seasonal.rename({"tourists": "Seasonal (12)"}), on="time")
        .join(filled_mean.rename({"tourists": "Mean"}), on="time")
        .join(filled_knn.rename({"tourists": "KNN (5)"}), on="time")
    )
    plot_time_series(_all_methods, title="All Imputation Methods")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Next we compute the Mean Absolute Error (MAE) between the imputed
    values and the original (pre-gap) values at the nine gap positions.
    A grouped bar chart makes the ranking immediately visible.
    """)


@app.cell
def _(
    filled_forward,
    filled_knn,
    filled_linear,
    filled_mean,
    filled_seasonal,
    go,
    tourism,
):
    _gap_indices = list(range(30, 35)) + [50, 70, 90, 110]
    _true_vals = tourism["tourists"].gather(_gap_indices)

    _methods = {
        "Linear": filled_linear,
        "Forward": filled_forward,
        "Seasonal (12)": filled_seasonal,
        "Mean": filled_mean,
        "KNN (5)": filled_knn,
    }

    _scores = {}
    for _name, _filled in _methods.items():
        _filled_vals = _filled["tourists"].gather(_gap_indices)
        _scores[_name] = round(float((_filled_vals - _true_vals).abs().mean()), 2)

    _sorted = sorted(_scores.items(), key=lambda x: x[1])
    _names = [s[0] for s in _sorted]
    _values = [s[1] for s in _sorted]

    fig = go.Figure(
        go.Bar(
            x=_names,
            y=_values,
            text=[f"{v:.2f}" for v in _values],
            textposition="outside",
        )
    )
    fig.update_layout(title="Imputation MAE at Gap Positions", yaxis_title="MAE")
    fig




@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Alternative: Skip NaN Rows with `nan_handling`

        Instead of imputing, you can let the reduction forecaster drop any
        tabularized training row that contains NaN. This is useful when gaps
        are sparse and you want to avoid imputation artifacts.

        Tree-based estimators handle NaN natively (`nan_handling="pass"`),
        while linear models need the rows removed (`nan_handling="drop"`).
        """
    )
    return


@app.cell
def _(tourism_missing):
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.linear_model import Ridge

    from yohou.point import PointReductionForecaster

    # Tree-based: pass NaN through (default behavior)
    tree_forecaster = PointReductionForecaster(
        estimator=HistGradientBoostingRegressor(),
        nan_handling="pass",
    )
    tree_forecaster.fit(y=tourism_missing, forecasting_horizon=3)
    tree_pred = tree_forecaster.predict(forecasting_horizon=3)

    # Linear: drop NaN rows before fitting
    linear_forecaster = PointReductionForecaster(
        estimator=Ridge(),
        nan_handling="drop",
    )
    linear_forecaster.fit(y=tourism_missing, forecasting_horizon=3)
    linear_pred = linear_forecaster.predict(forecasting_horizon=3)

    return linear_pred, tree_pred


@app.cell
def _(linear_pred, tree_pred):
    import polars as pl

    comparison = pl.DataFrame({
        "time": tree_pred["time"],
        "tree_prediction": tree_pred["tourists"],
        "linear_prediction": linear_pred["tourists"],
    })
    comparison
    return (comparison,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Next Steps

        - [Handle Missing Data](/pages/how-to/handle-missing-data/) for the full guide
        - [Clean and Resample Time Series](/pages/how-to/clean-and-resample/) for related techniques
        """
    )
    return

if __name__ == "__main__":
    app.run()
