from datetime import datetime

import numpy as np
import polars as pl
import pytest
from sklearn.model_selection import train_test_split

from yohou.point_forecaster import PointReductionForecaster

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
    forecaster = PointReductionForecaster()

    forecaster.fit(y=y_train, X_ante=X_ante_train, forecasting_horizon=fit_forecasting_horizon)

    y_pred = forecaster.predict(forecasting_horizon=predict_forecasting_horizon)

    expected_y_pred = pl.DataFrame(
        {
            "observed_time": [y_train["time"][-1]] * predict_forecasting_horizon,
            "predicted_time": pl.datetime_range(
                start=datetime(2021, 12, 16, 0, 0, len(y_train)),
                end=datetime(2021, 12, 16, 0, 0, len(y_train) + predict_forecasting_horizon - 1),
                interval="1s",
                eager=True,
            ),
            "a": expected_a,
            "b": np.array(expected_a) + 10,
        },
        schema={
            "observed_time": pl.Datetime(time_unit="us", time_zone=None),
            "predicted_time": pl.Datetime(time_unit="us", time_zone=None),
            "a": pl.Float64,
            "b": pl.Float64,
        },
    )
    pl.testing.assert_frame_equal(y_pred, expected_y_pred)


@pytest.mark.parametrize(
    "fit_forecasting_horizon, predict_forecasting_horizon, stride, expected_a",
    [
        (1, 5, 1, [22, 22.4, 22.56, 22.624, 22.6496]),
        (3, 5, 2, [22, 23, 24, 23.2, 24.2]),
        (3, 2, 1, [22, 23]),
    ],
)
def test_update_predict(fit_forecasting_horizon, predict_forecasting_horizon, stride, expected_a):
    forecaster = PointReductionForecaster()

    forecaster.fit(y=y_train, X_ante=X_ante_train, forecasting_horizon=fit_forecasting_horizon)

    y_pred = forecaster.update_predict(
        y=y_test,
        X_ante=X_ante_test,
        forecasting_horizon=predict_forecasting_horizon,
        stride=stride,
    )

    # Get the last set of predictions (after all updates)
    y_pred = y_pred.tail(predict_forecasting_horizon)

    expected_y_pred = pl.DataFrame(
        {
            "observed_time": [y_test["time"][-1]] * predict_forecasting_horizon,
            "predicted_time": pl.datetime_range(
                start=datetime(2021, 12, 16, 0, 0, len(y_train) + len(y_test)),
                end=datetime(
                    2021, 12, 16, 0, 0, len(y_train) + len(y_test) + predict_forecasting_horizon - 1
                ),
                interval="1s",
                eager=True,
            ),
            "a": expected_a,
            "b": np.array(expected_a) + 10,
        },
        schema={
            "observed_time": pl.Datetime(time_unit="us", time_zone=None),
            "predicted_time": pl.Datetime(time_unit="us", time_zone=None),
            "a": pl.Float64,
            "b": pl.Float64,
        },
    )
    pl.testing.assert_frame_equal(y_pred, expected_y_pred)


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
        # Note: struct columns produce different predictions than flat columns
        # due to how the reduction forecaster handles grouped features
        (1, 5, 1, [22.0, 22.666667, 23.111111, 23.407407, 23.604938]),
        (3, 5, 2, [22.0, 23.0, 24.0, 24.0, 25.0]),
        (3, 2, 1, [22.0, 23.0]),
    ],
)
def test_update_predict_global(
    fit_forecasting_horizon, predict_forecasting_horizon, stride, expected_a
):
    forecaster = PointReductionForecaster()

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

    # Get the last set of predictions (after all updates)
    y_pred = y_pred.tail(predict_forecasting_horizon)

    expected_y_pred = pl.DataFrame(
        {
            "observed_time": [y_test_struct["time"][-1]] * predict_forecasting_horizon,
            "predicted_time": pl.datetime_range(
                start=datetime(2021, 12, 16, 0, 0, len(y_train_struct) + len(y_test_struct)),
                end=datetime(
                    2021,
                    12,
                    16,
                    0,
                    0,
                    len(y_train_struct) + len(y_test_struct) + predict_forecasting_horizon - 1,
                ),
                interval="1s",
                eager=True,
            ),
            "x": pl.DataFrame(
                {
                    "a": expected_a,
                    "b": np.array(expected_a) + 10,
                },
                schema={"a": pl.Float64, "b": pl.Float64},
            ),
            "y": pl.DataFrame(
                {
                    "a": np.array(expected_a) + 10,
                    "b": np.array(expected_a) + 20,
                },
                schema={"a": pl.Float64, "b": pl.Float64},
            ),
        },
        schema={
            "observed_time": pl.Datetime(time_unit="us", time_zone=None),
            "predicted_time": pl.Datetime(time_unit="us", time_zone=None),
            "x": pl.Struct({"a": pl.Float64, "b": pl.Float64}),
            "y": pl.Struct({"a": pl.Float64, "b": pl.Float64}),
        },
    )
    pl.testing.assert_frame_equal(y_pred, expected_y_pred)
