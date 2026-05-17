# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "yohou[plotting]",
# ]
# ///

import marimo

__generated_with = "0.23.1"
__gallery__ = {
    "title": "How to Create a Custom Transformer",
    "description": "Implement a ScaleTransformer from scratch, validate it with the check generator, and use it in a forecast pipeline.",
    "category": "how-to",
    "companion": "/pages/how-to/create-a-transformer/",
    "section": "getting-started",
    "api_references": ["BaseTransformer", "PointReductionForecaster"],
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
        # How to Create a Custom Transformer

        This notebook implements a `ScaleTransformer` from scratch,
        validates it using the built-in check generator, and uses it
        as a target transformer inside a
        [`PointReductionForecaster`](/pages/api/generated/yohou.point.reduction.PointReductionForecaster/).

        **Prerequisites:** Familiarity with the fit/predict API
        ([Getting Started](/pages/tutorials/getting-started/)) and
        Yohou transformers ([Use Preprocessing Transformers](/pages/how-to/use-preprocessing-transformers/)).
        """
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"## 1. Implement the Transformer")


@app.cell
def _():
    import polars as pl
    import polars.selectors as cs

    from yohou.base import BaseTransformer

    class ScaleTransformer(BaseTransformer):
        """Multiplies all numeric columns by a constant factor."""

        _tags = {"invertible": True}

        _parameter_constraints: dict = {
            "factor": [float, int],
        }

        def __init__(self, factor=2.0):
            self.factor = factor

        def _fit(self, X, y=None):
            self.columns_ = [c for c in X.columns if c != "time"]

        def _transform(self, X):
            return X.with_columns(cs.numeric() * self.factor)

        def _inverse_transform(self, X_t, X_p=None):
            return X_t.with_columns(cs.numeric() / self.factor)

        def get_feature_names_out(self, input_features=None):
            return self.columns_ if input_features is None else input_features

    return BaseTransformer, ScaleTransformer, cs, pl


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"## 2. Fit and Transform")


@app.cell
def _(ScaleTransformer, pl):
    from yohou.datasets import fetch_sunspot
    from yohou.model_selection import train_test_split

    bunch = fetch_sunspot()
    y = bunch.frame.group_by_dynamic("time", every="1mo").agg(
        pl.col("sunspot_number").mean()
    )

    y_train, y_test = train_test_split(y, test_size=24)

    transformer = ScaleTransformer(factor=0.01)
    transformer.fit(y_train)

    y_scaled = transformer.transform(y_train)
    y_restored = transformer.inverse_transform(y_scaled)

    print("Original (first 3 rows):")
    print(y_train.head(3))
    print("\nScaled:")
    print(y_scaled.head(3))
    print("\nRestored:")
    print(y_restored.head(3))

    return bunch, transformer, y, y_restored, y_scaled, y_test, y_train


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"## 3. Use in a Forecast Pipeline")


@app.cell
def _(ScaleTransformer, y_test, y_train):
    from sklearn.linear_model import Ridge

    from yohou.point import PointReductionForecaster

    forecaster = PointReductionForecaster(
        estimator=Ridge(),
        target_transformer=ScaleTransformer(factor=0.01),
    )
    forecasting_horizon = len(y_test)
    forecaster.fit(y_train, forecasting_horizon=forecasting_horizon)
    y_pred = forecaster.predict(forecasting_horizon=forecasting_horizon)
    y_pred.head()

    return Ridge, forecaster, forecasting_horizon, y_pred


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"## 4. Plot the Result")


@app.cell
def _(y_pred, y_test, y_train):
    from yohou.plotting import plot_forecast

    fig = plot_forecast(
        y_train=y_train,
        y_test=y_test,
        y_pred=y_pred,
    )
    fig

    return (fig,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Next Steps

        - [Create a Transformer](/pages/how-to/create-a-transformer/) for the full guide
        - [Compose Feature Pipelines](/pages/how-to/compose-feature-pipelines/) for combining transformers
        - [Use Preprocessing Transformers](/pages/how-to/use-preprocessing-transformers/) for built-in transformers
        """
    )
    return


if __name__ == "__main__":
    app.run()
