from datetime import datetime

import polars as pl
import polars.testing
import pytest
from sklearn.model_selection import train_test_split

from yohou.interval_forecaster import IntervalReductionForecaster

length = 22

time = pl.DataFrame(
    {
        "time": pl.datetime_range(
            start=datetime(2021, 12, 16),
            end=datetime(2021, 12, 16, 0, 0, length - 1),
            interval="1s",
            eager=True,
        ),
    }
)
y = pl.DataFrame(
    {
        "a": range(length),
        "b": range(10, length + 10),
    },
    schema={
        "a": pl.Float64,
        "b": pl.Float64,
    },
)
y = pl.concat([time, y], how="horizontal")

X_ante = pl.DataFrame(
    {
        "c": range(length),
        "d": range(10, length + 10),
        "e": range(20, length + 20),
    },
    schema={
        "c": pl.Float64,
        "d": pl.Float64,
        "e": pl.Float64,
    },
)
X_ante = pl.concat([time, X_ante], how="horizontal")

y_train, y_test, X_ante_train, X_ante_test = train_test_split(
    y, X_ante, test_size=0.2, shuffle=False
)


@pytest.mark.parametrize(
    "fit_forecasting_horizon, predict_forecasting_horizon, expected_a",
    [
        (1, 5, [17, 17.4, 17.56, 17.624, 17.6496]),
        (3, 5, [17, 18, 19, 18.2, 19.2]),
        (3, 2, [17, 18]),
    ],
)
def test_predict(fit_forecasting_horizon, predict_forecasting_horizon, expected_a):
    coverage_rates = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    forecaster = IntervalReductionForecaster(coverage_rates=coverage_rates)

    forecaster.fit(y=y_train, X_ante=X_ante_train, forecasting_horizon=fit_forecasting_horizon)

    y_pred = forecaster.predict(
        forecasting_horizon=predict_forecasting_horizon,
        X_post=None,
        predict_transformed=False,
    )

    for col in y_train.columns:
        if col == "time":
            continue
        for coverage_rate in coverage_rates:
            assert all(
                y_pred[f"{col}_upper_{coverage_rate}"] + 1e-14
                >= y_pred[f"{col}_lower_{coverage_rate}"]
            )


@pytest.mark.parametrize(
    "fit_forecasting_horizon, predict_forecasting_horizon, stride, expected_a",
    [
        (1, 5, 1, [17, 17.4, 17.56, 17.624, 17.6496]),
        (3, 5, 2, [17, 18, 19, 18.2, 19.2]),
        (3, 2, 1, [17, 18]),
    ],
)
def test_update_predict(fit_forecasting_horizon, predict_forecasting_horizon, stride, expected_a):
    coverage_rates = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    forecaster = IntervalReductionForecaster(coverage_rates=coverage_rates)

    forecaster.fit(y=y_train, X_ante=X_ante_train, forecasting_horizon=fit_forecasting_horizon)

    y_pred = forecaster.update_predict(
        y=y_test,
        X_ante=X_ante_test,
        forecasting_horizon=predict_forecasting_horizon,
        stride=stride,
    )

    for col in y_train.columns:
        if col == "time":
            continue
        for coverage_rate in coverage_rates:
            assert all(
                y_pred[f"{col}_upper_{coverage_rate}"] + 1e-14
                >= y_pred[f"{col}_lower_{coverage_rate}"]
            )


y_struct = pl.DataFrame(
    {
        "x": pl.DataFrame(
            {
                "a": range(length),
                "b": range(10, length + 10),
            }
        ),
        "y": pl.DataFrame(
            {
                "a": range(10, length + 10),
                "b": range(20, length + 20),
            }
        ),
    },
    schema={
        "x": pl.Struct({"a": pl.Float64, "b": pl.Float64}),
        "y": pl.Struct({"a": pl.Float64, "b": pl.Float64}),
    },
)
y_struct = pl.concat([time, y_struct], how="horizontal")

X_ante_struct = pl.DataFrame(
    {
        "x": pl.DataFrame(
            {
                "c": range(length),
            }
        ),
        "y": pl.DataFrame(
            {
                "c": range(10, length + 10),
            }
        ),
        "d": range(10, length + 10),
        "e": range(20, length + 20),
    },
    schema={
        "x": pl.Struct({"c": pl.Float64}),
        "y": pl.Struct({"c": pl.Float64}),
        "d": pl.Float64,
        "e": pl.Float64,
    },
)
X_ante_struct = pl.concat([time, X_ante_struct], how="horizontal")

y_train_struct, y_test_struct, X_ante_train_struct, X_ante_test_struct = train_test_split(
    y_struct, X_ante_struct, test_size=0.2, shuffle=False
)


@pytest.mark.parametrize(
    "fit_forecasting_horizon, predict_forecasting_horizon, stride, expected_a",
    [
        (1, 5, 1, [17, 17.4, 17.56, 17.624, 17.6496]),
        (3, 5, 2, [17, 18, 19, 18.2, 19.2]),
        (3, 2, 1, [17, 18]),
    ],
)
def test_update_predict_global(
    fit_forecasting_horizon, predict_forecasting_horizon, stride, expected_a
):
    coverage_rates = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    forecaster = IntervalReductionForecaster(coverage_rates=coverage_rates)

    forecaster.fit(
        y=y_train_struct,
        X_ante=X_ante_train_struct,
        forecasting_horizon=fit_forecasting_horizon,
    )

    y_pred = forecaster.update_predict(
        y=y_test_struct,
        X_ante=X_ante_test_struct,
        forecasting_horizon=predict_forecasting_horizon,
        stride=stride,
    )

    # Check that upper bounds >= lower bounds for struct columns
    # Each struct column contains fields like a_lower_0.1, a_upper_0.1, etc.
    for group_col in y_train_struct.columns:
        if group_col == "time":
            continue
        # Unnest just this struct column
        y_pred_group = y_pred[[group_col]].unnest(group_col)

        # Check all fields within this struct
        for field_col in ["a", "b"]:  # Fields within the structs
            for coverage_rate in coverage_rates:
                assert all(
                    y_pred_group[f"{field_col}_upper_{coverage_rate}"] + 1e-13
                    >= y_pred_group[f"{field_col}_lower_{coverage_rate}"]
                )
