"""Test doubles for exercising the ``holdout_size`` tag through composites.

No yohou point forecaster holds rows back, and a split-conformal forecaster
cannot sit where a point forecaster is required, so the composite tests use
``_HoldoutStubForecaster``: a real seasonal-naive forecaster that declares a
fixed held-back count.
"""

import numbers

import polars as pl

from yohou.base import BaseActualTransformer
from yohou.point import SeasonalNaive
from yohou.utils import Tags
from yohou.utils._compat import Interval


class _HoldoutStubForecaster(SeasonalNaive):
    """Seasonal-naive forecaster declaring a fixed held-back count.

    Parameters
    ----------
    holdout : int, default=0
        The ``holdout_size`` its forecaster tags declare.
    seasonality : int, default=1
        Passed to ``SeasonalNaive``.
    panel_strategy : {"global", "multivariate"}, default="global"
        Passed to ``SeasonalNaive``.

    """

    _parameter_constraints: dict = {
        **SeasonalNaive._parameter_constraints,
        "holdout": [Interval(numbers.Integral, 0, None, closed="left")],
    }

    def __init__(self, holdout: int = 0, seasonality: int = 1, panel_strategy: str = "global"):
        super().__init__(seasonality=seasonality, panel_strategy=panel_strategy)
        self.holdout = holdout

    def __sklearn_tags__(self) -> Tags:
        """Declare ``holdout`` as the held-back count.

        Returns
        -------
        Tags
            ``SeasonalNaive``'s tags with ``holdout_size`` set to ``holdout``.

        """
        tags = super().__sklearn_tags__()
        tags.forecaster_tags.holdout_size = self.holdout
        return tags


class _SelectColumn(BaseActualTransformer):
    """Stateless extractor selecting one column and renaming it, for ``CombiningForecaster`` terms."""

    def __init__(self, column: str, out_name: str):
        self.column = column
        self.out_name = out_name
        self._observation_horizon = 0

    @property
    def observation_horizon(self) -> int:
        """Return the (zero) observation horizon of this stateless extractor."""
        return 0

    def fit(self, X: pl.DataFrame, y: pl.DataFrame | None = None) -> "_SelectColumn":
        """Fit against the input frame, setting the transformer schema."""
        BaseActualTransformer.fit(self, X, y)
        return self

    def transform(self, X: pl.DataFrame) -> pl.DataFrame:
        """Return ``time`` plus the selected column renamed to ``out_name``."""
        return X.select([pl.col("time"), pl.col(self.column).alias(self.out_name)])

    def get_feature_names_out(self, input_features: list[str] | None = None) -> list[str]:
        """Return the single output feature name."""
        return [self.out_name]
