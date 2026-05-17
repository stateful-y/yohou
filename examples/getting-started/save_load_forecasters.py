# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "yohou",
# ]
# ///

import marimo

__generated_with = "0.23.1"
__gallery__ = {
    "title": "How to Save and Load Forecasters",
    "description": "Serialize fitted forecasters with joblib and pickle, reload them in a fresh session, and produce predictions without retraining.",
    "category": "how-to",
    "companion": "/pages/how-to/save-load-forecasters/",
    "section": "getting-started",
    "api_references": ["PointReductionForecaster", "SeasonalNaive"],
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
        # How to Save and Load Forecasters

        This notebook shows how to serialize a fitted forecaster to disk
        and reload it in a new session to produce predictions without retraining.

        **Prerequisites:** Familiarity with fit/predict
        ([Getting Started](/pages/tutorials/getting-started/)).
        """
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"## 1. Fit a Forecaster")


@app.cell
def _():
    import polars as pl
    from sklearn.linear_model import Ridge

    from yohou.datasets import fetch_sunspot
    from yohou.model_selection import train_test_split
    from yohou.point import PointReductionForecaster
    from yohou.preprocessing import LagTransformer

    bunch = fetch_sunspot()
    y = bunch.frame.group_by_dynamic("time", every="1mo").agg(
        pl.col("sunspot_number").mean()
    )

    forecasting_horizon = 12
    y_train, y_test = train_test_split(y, test_size=forecasting_horizon)

    forecaster = PointReductionForecaster(
        estimator=Ridge(),
        feature_transformer=LagTransformer(lag=[1, 6, 12]),
    )
    forecaster.fit(y_train, forecasting_horizon=forecasting_horizon)
    y_pred = forecaster.predict(forecasting_horizon=forecasting_horizon)
    y_pred.head()

    return bunch, forecaster, forecasting_horizon, pl, y, y_pred, y_test, y_train


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"## 2. Save with joblib")


@app.cell
def _(forecaster):
    import tempfile
    from pathlib import Path

    import joblib

    tmp_dir = Path(tempfile.mkdtemp())
    joblib_path = tmp_dir / "forecaster.joblib"

    joblib.dump(forecaster, joblib_path)
    print(f"Saved to {joblib_path} ({joblib_path.stat().st_size:,} bytes)")

    return joblib, joblib_path, tmp_dir


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"## 3. Load and Predict with joblib")


@app.cell
def _(forecasting_horizon, joblib, joblib_path, y_pred):
    loaded = joblib.load(joblib_path)
    y_pred_loaded = loaded.predict(forecasting_horizon=forecasting_horizon)

    # Verify predictions match the original
    assert y_pred.equals(y_pred_loaded), "Predictions should be identical"
    print("Loaded forecaster produces identical predictions.")
    y_pred_loaded.head()

    return loaded, y_pred_loaded


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"## 4. Save with pickle")


@app.cell
def _(forecaster, tmp_dir):
    import pickle

    pickle_path = tmp_dir / "forecaster.pkl"

    with open(pickle_path, "wb") as _f:
        pickle.dump(forecaster, _f)

    print(f"Saved to {pickle_path} ({pickle_path.stat().st_size:,} bytes)")

    return pickle, pickle_path


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"## 5. Load and Predict with pickle")


@app.cell
def _(forecasting_horizon, pickle, pickle_path, y_pred):
    with open(pickle_path, "rb") as _f:
        loaded_pkl = pickle.load(_f)  # noqa: S301

    y_pred_pkl = loaded_pkl.predict(forecasting_horizon=forecasting_horizon)
    assert y_pred.equals(y_pred_pkl), "Predictions should be identical"
    print("Pickle-loaded forecaster produces identical predictions.")
    y_pred_pkl.head()

    return loaded_pkl, y_pred_pkl


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Security Warning

        !!! warning "Never load untrusted pickle files"

        Pickle files can execute arbitrary code when loaded. Only load files you
        created yourself or received from a trusted source. For sharing models
        across trust boundaries, consider exporting predictions instead of
        serialized objects.
        """
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Next Steps

        - [How to Save and Load Forecasters](/pages/how-to/save-load-forecasters/) for the full guide
        - [Forecasting Workflow](/pages/tutorials/forecasting-workflow/) for the training and evaluation pipeline that produces a fitted forecaster
        - [Core Concepts](/pages/explanation/core-concepts/) for the fit/observe/predict lifecycle
        """
    )


if __name__ == "__main__":
    app.run()
