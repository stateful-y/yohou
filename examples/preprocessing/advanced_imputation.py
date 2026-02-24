# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "plotly",
#     "scikit-learn",
#     "yohou",
# ]
# ///
"""Advanced Imputation.

Demonstrates SimpleTimeImputer, SeasonalImputer, and sklearn-wrapped
imputers (SimpleImputer, TransformedSpaceKNNImputer) on time series with synthetic gaps.
"""

import marimo

__generated_with = "0.19.11"
app = marimo.App(width="medium")

@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Advanced Imputation

    Missing values are common in real-world time series. Yohou provides
    temporal-aware imputation methods alongside sklearn wrappers.

    ## What You'll Learn

    - `SimpleTimeImputer`: linear, forward, backward, nearest, fill_both
    - `SeasonalImputer`: seasonal mean/median fill
    - `SimpleImputer` / `TransformedSpaceKNNImputer` (sklearn wrappers)
    - Comparing methods on synthetic gaps
    """)
    return

@app.cell(hide_code=True)
def _():
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
        pl,
        plot_time_series,
    )

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Create Data with Synthetic Gaps
    """)
    return

@app.cell
def _(fetch_tourism_monthly, mo, pl):
    air = fetch_tourism_monthly().frame.select("time", "T1__tourists").drop_nulls().rename({"T1__tourists": "Passengers"})

    # Create gaps: indices 30-34 (block), and scattered singles
    _gap_indices = list(range(30, 35)) + [50, 70, 90, 110]
    _mask = pl.Series([i in _gap_indices for i in range(len(air))])
    air_missing = air.with_columns(
        pl.when(_mask).then(None).otherwise(pl.col("Passengers")).alias("Passengers"),
    )
    _n_missing = air_missing.null_count()["Passengers"][0]

    mo.md(
        f"**Original**: {len(air)} months, no gaps\n\n"
        f"**With gaps**: {_n_missing} missing values "
        f"(block of 5 + 4 scattered)"
    )
    return air, air_missing

@app.cell
def _(air_missing, plot_time_series):
    plot_time_series(air_missing, title="Monthly Tourism: With Synthetic Gaps")
    return

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. SimpleTimeImputer: Linear Interpolation
    """)
    return

@app.cell
def _(SimpleTimeImputer, air_missing, plot_time_series):
    imp_linear = SimpleTimeImputer(method="linear")
    imp_linear.fit(air_missing)
    filled_linear = imp_linear.transform(air_missing)
    plot_time_series(filled_linear, title="Linear Interpolation")
    return filled_linear, imp_linear

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. SimpleTimeImputer: Forward Fill
    """)
    return

@app.cell
def _(SimpleTimeImputer, air_missing, plot_time_series):
    imp_forward = SimpleTimeImputer(method="forward")
    imp_forward.fit(air_missing)
    filled_forward = imp_forward.transform(air_missing)
    plot_time_series(filled_forward, title="Forward Fill")
    return filled_forward, imp_forward

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. SimpleTimeImputer with `limit`

    The `limit` parameter caps the maximum number of consecutive NaN
    values that are filled. Useful when you don't want to interpolate
    across long gaps.
    """)
    return

@app.cell
def _(SimpleTimeImputer, air_missing, mo):
    _imp_limited = SimpleTimeImputer(method="linear", limit=2)
    _imp_limited.fit(air_missing)
    _filled_limited = _imp_limited.transform(air_missing)
    _remaining_nulls = _filled_limited.null_count()["Passengers"][0]

    mo.md(
        f"**limit=2**: {_remaining_nulls} nulls remain\n\n"
        "The block gap (5 consecutive) is only partially filled."
    )
    return

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. SeasonalImputer

    Fill missing values using seasonal patterns (e.g., same month from
    other years).
    """)
    return

@app.cell
def _(SeasonalImputer, air_missing, plot_time_series):
    imp_seasonal = SeasonalImputer(period=12, fill_method="seasonal_mean")
    imp_seasonal.fit(air_missing)
    filled_seasonal = imp_seasonal.transform(air_missing)
    plot_time_series(filled_seasonal, title="Seasonal Imputer (period=12, seasonal_mean)")
    return filled_seasonal, imp_seasonal

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. Sklearn-Wrapped Imputers

    `SimpleImputer` (mean/median fill) and `TransformedSpaceKNNImputer` (neighbour-based)
    are wrapped for yohou time series compatibility.
    """)
    return

@app.cell
def _(SimpleImputer, air_missing, plot_time_series):
    imp_mean = SimpleImputer(strategy="mean")
    imp_mean.fit(air_missing)
    filled_mean = imp_mean.transform(air_missing)
    plot_time_series(filled_mean, title="SimpleImputer (strategy='mean')")
    return filled_mean, imp_mean

@app.cell
def _(TransformedSpaceKNNImputer, air_missing, plot_time_series):
    imp_knn = TransformedSpaceKNNImputer(n_neighbors=5)
    imp_knn.fit(air_missing)
    filled_knn = imp_knn.transform(air_missing)
    plot_time_series(filled_knn, title="TransformedSpaceKNNImputer (n_neighbors=5)")
    return filled_knn, imp_knn

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7. Compare Methods

    Measure imputation error against the known true values at gap
    positions.
    """)
    return

@app.cell
def _(air, filled_forward, filled_knn, filled_linear, filled_mean, filled_seasonal, mo, pl):
    _gap_indices = list(range(30, 35)) + [50, 70, 90, 110]
    _true_vals = air["Passengers"].gather(_gap_indices)

    _methods = {
        "Linear": filled_linear,
        "Forward": filled_forward,
        "Seasonal (12)": filled_seasonal,
        "Mean": filled_mean,
        "KNN (5)": filled_knn,
    }

    _rows = []
    for _name, _filled in _methods.items():
        _filled_vals = _filled["Passengers"].gather(_gap_indices)
        _mae = ((_filled_vals - _true_vals).abs()).mean()
        _rows.append({"Method": _name, "MAE at Gaps": round(float(_mae), 2)})

    mo.ui.table(pl.DataFrame(_rows))
    return

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Key Takeaways

    - **`SimpleTimeImputer`**: Best for temporal data (linear, forward, backward, nearest)
    - **`SeasonalImputer`**: Uses seasonal patterns when periodicity is known
    - **`limit`**: Prevents filling across long gaps
    - **`SimpleImputer`/`TransformedSpaceKNNImputer`**: sklearn wrappers for non-temporal strategies
    - **Linear interpolation** typically best for short gaps; seasonal imputation for long gaps with known periodicity

    ## Next Steps

    - **Data cleaning**: See `examples/preprocessing/data_cleaning.py`
    - **Resampling**: See `examples/preprocessing/resampling_advanced.py`
    - **Sklearn wrappers**: See `examples/preprocessing/sklearn_wrappers.py`
    """)
    return

if __name__ == "__main__":
    app.run()
