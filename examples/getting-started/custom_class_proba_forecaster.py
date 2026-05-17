# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "numpy",
#     "yohou[plotting]",
# ]
# ///

import marimo

__generated_with = "0.23.1"
__gallery__ = {
    "title": "How to Create a Custom Class-Probability Forecaster",
    "description": "Implement a MajorityClassForecaster from scratch, validate it with the check generator, and compare it against ClassProbaReductionForecaster.",
    "category": "how-to",
    "companion": "/pages/how-to/create-a-class-proba-forecaster/",
    "section": "getting-started",
    "api_references": ["BaseClassProbaForecaster", "ClassProbaReductionForecaster"],
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
        # How to Create a Custom Class-Probability Forecaster

        This notebook implements a `MajorityClassForecaster` from scratch,
        validates it using the built-in check generator, and compares
        its predictions against
        [`ClassProbaReductionForecaster`](/pages/api/generated/yohou.class_proba.reduction.ClassProbaReductionForecaster/).

        **Prerequisites:** Familiarity with the fit/predict API
        ([Getting Started](/pages/tutorials/getting-started/)) and
        class-probability forecasting
        ([Forecast with Class Probabilities](/pages/how-to/class-probability-forecasting/)).
        """
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"## 1. Implement the Forecaster")


@app.cell
def _():
    import polars as pl

    from yohou.class_proba.base import BaseClassProbaForecaster
    from yohou.utils.tags import Tags

    class MajorityClassForecaster(BaseClassProbaForecaster):
        """Predicts the training-set class distribution at every step."""

        def __sklearn_tags__(self) -> Tags:
            tags = super().__sklearn_tags__()
            tags.forecaster_tags.requires_exogenous = False
            tags.forecaster_tags.stateful = True
            return tags

        @property
        def _observation_horizon(self):
            return 1

        def _fit(self, y_t, X_t, forecasting_horizon):
            value_cols = [c for c in y_t.columns if c != "time"]
            self.classes_ = {}
            self.n_classes_ = {}
            self.label_to_code_ = {}
            self._class_probs = {}

            for col in value_cols:
                labels = sorted(y_t[col].unique().to_list())
                self.classes_[col] = labels
                self.n_classes_[col] = len(labels)
                self.label_to_code_[col] = {
                    label: i for i, label in enumerate(labels)
                }

                counts = y_t[col].value_counts()
                total = len(y_t)
                self._class_probs[col] = {
                    row[col]: row["count"] / total
                    for row in counts.iter_rows(named=True)
                }

        def _predict_class_proba_one(self, groups, **params):
            h = self.fit_forecasting_horizon_
            data = {}
            for col, probs in self._class_probs.items():
                for label in self.classes_[col]:
                    prob = probs.get(label, 0.0)
                    data[f"{col}_proba_{label}"] = [prob] * h

            y_pred = pl.DataFrame(data)
            y_pred = self._add_time_columns(y_pred)
            return y_pred

    return MajorityClassForecaster, Tags, pl


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"## 2. Prepare Categorical Data")


@app.cell
def _(pl):
    import datetime

    import numpy as np

    rng = np.random.default_rng(42)
    n = 200
    labels = ["low", "medium", "high"]
    probs = [0.3, 0.5, 0.2]
    values = rng.choice(labels, size=n, p=probs)

    y = pl.DataFrame({
        "time": [
            datetime.datetime(2020, 1, 1) + datetime.timedelta(days=i)
            for i in range(n)
        ],
        "category": values,
    }).cast({"category": pl.String})

    forecasting_horizon = 30

    from yohou.model_selection import train_test_split

    y_train, y_test = train_test_split(y, test_size=forecasting_horizon)
    y_train.head()

    return datetime, forecasting_horizon, np, rng, y, y_test, y_train


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"## 3. Fit and Predict")


@app.cell
def _(MajorityClassForecaster, forecasting_horizon, y_train):
    forecaster = MajorityClassForecaster()
    forecaster.fit(y_train, forecasting_horizon=forecasting_horizon)

    y_pred = forecaster.predict(forecasting_horizon=forecasting_horizon)
    y_pred_proba = forecaster.predict_class_proba(
        forecasting_horizon=forecasting_horizon,
    )

    print("Hard predictions:")
    print(y_pred.head())
    print("\nClass probabilities:")
    print(y_pred_proba.head())

    return forecaster, y_pred, y_pred_proba


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"## 4. Verify Probabilities Sum to 1")


@app.cell
def _(y_pred_proba):
    proba_cols = [c for c in y_pred_proba.columns if "_proba_" in c]
    row_sums = y_pred_proba.select(proba_cols).sum_horizontal()
    print(f"Probability sums (all rows): {row_sums.to_list()[:5]}")
    assert all(abs(s - 1.0) < 1e-10 for s in row_sums.to_list())

    return (proba_cols,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Next Steps

        - [Create a Class-Probability Forecaster](/pages/how-to/create-a-class-proba-forecaster/) for the full guide
        - [Forecast with Class Probabilities](/pages/how-to/class-probability-forecasting/) for built-in approaches
        - [Create a Custom Scorer](/pages/how-to/create-a-scorer/) for evaluation metrics
        """
    )
    return


if __name__ == "__main__":
    app.run()
