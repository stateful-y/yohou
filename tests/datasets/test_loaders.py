"""Tests for bundled dataset loaders."""

import polars as pl
import pytest

from yohou.datasets.loaders import (
    load_air_passengers,
    load_australian_tourism,
    load_ett_m1,
    load_store_sales,
    load_sunspots,
    load_vic_electricity,
    load_walmart_sales,
)

ALL_LOADERS = [
    load_air_passengers,
    load_sunspots,
    load_australian_tourism,
    load_vic_electricity,
    load_store_sales,
    load_walmart_sales,
    load_ett_m1,
]


class TestLoaderCommon:
    """Common tests for all dataset loaders."""

    @pytest.mark.parametrize("loader", ALL_LOADERS, ids=lambda fn: fn.__name__)
    def test_loader_returns_dataframe(self, loader):
        """Each loader should return a polars DataFrame."""
        df = loader()
        assert isinstance(df, pl.DataFrame)

    @pytest.mark.parametrize("loader", ALL_LOADERS, ids=lambda fn: fn.__name__)
    def test_loader_has_time_column(self, loader):
        """Each loader should produce a DataFrame with a 'time' column."""
        df = loader()
        assert "time" in df.columns

    @pytest.mark.parametrize("loader", ALL_LOADERS, ids=lambda fn: fn.__name__)
    def test_loader_non_empty(self, loader):
        """Each loader should produce a non-empty DataFrame."""
        df = loader()
        assert len(df) > 0

    @pytest.mark.parametrize("loader", ALL_LOADERS, ids=lambda fn: fn.__name__)
    def test_loader_has_non_time_columns(self, loader):
        """Each loader should have at least one non-time column."""
        df = loader()
        non_time_cols = [c for c in df.columns if c != "time"]
        assert len(non_time_cols) >= 1


class TestLoaderSchemas:
    """Tests for specific dataset schemas."""

    def test_air_passengers_schema(self):
        """Air passengers dataset should have expected columns."""
        df = load_air_passengers()
        assert "Passengers" in df.columns
        assert df.schema["Passengers"] == pl.Int64

    def test_sunspots_schema(self):
        """Sunspots dataset should have expected columns."""
        df = load_sunspots()
        assert "Sunspots" in df.columns
        assert df.schema["Sunspots"] == pl.Float64

    def test_vic_electricity_schema(self):
        """Victoria electricity dataset should have expected columns."""
        df = load_vic_electricity()
        assert "Demand" in df.columns
        assert "Temperature" in df.columns

    def test_store_sales_schema(self):
        """Store sales dataset should have panel columns with __ separator."""
        df = load_store_sales()
        assert "time" in df.columns
        assert len(df.columns) == 10  # time + 9 panel columns
        non_time = [c for c in df.columns if c != "time"]
        assert all("__sales" in c for c in non_time)

    def test_walmart_sales_schema(self):
        """Walmart sales dataset should have panel columns with __ separator."""
        df = load_walmart_sales()
        assert "time" in df.columns
        assert len(df.columns) == 7  # time + 6 panel columns
        non_time = [c for c in df.columns if c != "time"]
        assert all("__" in c for c in non_time)

    def test_australian_tourism_schema(self):
        """Australian tourism dataset should have panel columns with __ separator."""
        df = load_australian_tourism()
        assert "time" in df.columns
        assert len(df.columns) == 9  # time + 8 state columns
        non_time = [c for c in df.columns if c != "time"]
        assert all("__trips" in c for c in non_time)

    def test_ett_m1_schema(self):
        """ETTm1 dataset should have expected multivariate columns."""
        df = load_ett_m1()
        assert "time" in df.columns
        assert "OT" in df.columns
        assert len(df.columns) == 8  # time + 7 features
