"""Tests for WASM dataset loading path."""

from __future__ import annotations

import io
from datetime import datetime
from unittest.mock import patch

import polars as pl
import pytest
from sklearn.utils import Bunch

from yohou.datasets._fetchers import (
    _CDN_BASE_URL,
    _fetch_classification_wasm,
    _fetch_dataset_wasm,
    _is_wasm,
    fetch_air_quality_classification,
    fetch_demand_classification,
)
from yohou.datasets._registry import SUNSPOT


class TestIsWasm:
    def test_returns_false_in_native_python(self):
        assert _is_wasm() is False

    def test_returns_true_when_emscripten(self):
        with patch("yohou.datasets._fetchers.sys") as mock_sys:
            mock_sys.platform = "emscripten"
            from yohou.datasets import _fetchers

            assert _fetchers._is_wasm() is True


class TestFetchDatasetWasmRouting:
    def test_wasm_path_called_when_emscripten(self):
        with (
            patch("yohou.datasets._fetchers._is_wasm", return_value=True),
            patch("yohou.datasets._fetchers._fetch_dataset_wasm") as mock_wasm,
        ):
            mock_wasm.return_value = Bunch(frame=pl.DataFrame())
            from yohou.datasets._fetchers import _fetch_dataset
            from yohou.datasets._registry import SUNSPOT

            _fetch_dataset(SUNSPOT, "sunspot", value_column_name="sunspot_number")
            mock_wasm.assert_called_once_with("sunspot", SUNSPOT, n_series=None)

    def test_wasm_url_is_correct(self):
        expected_url = f"{_CDN_BASE_URL}/sunspot.bin"
        assert "cdn.jsdelivr.net" in expected_url
        assert "stateful-y/yohou@datasets" in expected_url
        assert expected_url.endswith("/sunspot.bin")


class TestFetchDatasetWasmBunch:
    @pytest.fixture()
    def sample_parquet_bytes(self):
        df = pl.DataFrame(
            {
                "time": [
                    datetime(2020, 1, 1),
                    datetime(2020, 2, 1),
                    datetime(2020, 3, 1),
                ],
                "sunspot_number": [10.0, 20.0, 30.0],
            },
            schema={"time": pl.Datetime, "sunspot_number": pl.Float64},
        )
        return df.serialize(format="binary")

    def test_returns_bunch_with_correct_keys(self, sample_parquet_bytes):
        mock_response = io.BytesIO(sample_parquet_bytes)
        mock_response.read = lambda: sample_parquet_bytes

        with patch("yohou.datasets._fetchers.urlopen", return_value=mock_response):
            result = _fetch_dataset_wasm("sunspot", SUNSPOT)

        assert isinstance(result, Bunch)
        assert "frame" in result
        assert "feature_names" in result
        assert "DESCR" in result
        assert "frequency" in result
        assert "n_series" in result
        assert "filename" in result

    def test_frame_is_polars_dataframe(self, sample_parquet_bytes):
        mock_response = io.BytesIO(sample_parquet_bytes)
        mock_response.read = lambda: sample_parquet_bytes

        with patch("yohou.datasets._fetchers.urlopen", return_value=mock_response):
            result = _fetch_dataset_wasm("sunspot", SUNSPOT)

        assert isinstance(result.frame, pl.DataFrame)
        assert "time" in result.frame.columns
        assert "sunspot_number" in result.frame.columns

    def test_feature_names_excludes_time(self, sample_parquet_bytes):
        mock_response = io.BytesIO(sample_parquet_bytes)
        mock_response.read = lambda: sample_parquet_bytes

        with patch("yohou.datasets._fetchers.urlopen", return_value=mock_response):
            result = _fetch_dataset_wasm("sunspot", SUNSPOT)

        assert result.feature_names == ["sunspot_number"]
        assert result.n_series == 1

    def test_metadata_from_registry(self, sample_parquet_bytes):
        mock_response = io.BytesIO(sample_parquet_bytes)
        mock_response.read = lambda: sample_parquet_bytes

        with patch("yohou.datasets._fetchers.urlopen", return_value=mock_response):
            result = _fetch_dataset_wasm("sunspot", SUNSPOT)

        assert SUNSPOT.descr == result.DESCR
        assert result.frequency == SUNSPOT.frequency

    def test_n_series_slices_value_columns(self):
        df = pl.DataFrame({
            "time": [datetime(2020, 1, 1), datetime(2020, 2, 1)],
            "a": [1.0, 2.0],
            "b": [3.0, 4.0],
            "c": [5.0, 6.0],
        })
        payload = df.serialize(format="binary")
        mock_response = io.BytesIO(payload)
        mock_response.read = lambda: payload

        with patch("yohou.datasets._fetchers.urlopen", return_value=mock_response):
            result = _fetch_dataset_wasm("sunspot", SUNSPOT, n_series=2)

        assert result.frame.columns == ["time", "a", "b"]
        assert result.feature_names == ["a", "b"]
        assert result.n_series == 2


class TestFetchClassificationWasm:
    @pytest.fixture()
    def classification_parquet_bytes(self):
        y_df = pl.DataFrame(
            {
                "time": [
                    datetime(2020, 1, 1),
                    datetime(2020, 2, 1),
                    datetime(2020, 3, 1),
                ],
                "air_quality": ["good", "moderate", "unhealthy"],
            },
            schema={"time": pl.Datetime, "air_quality": pl.String},
        )
        x_df = pl.DataFrame(
            {
                "time": [
                    datetime(2020, 1, 1),
                    datetime(2020, 2, 1),
                    datetime(2020, 3, 1),
                ],
                "pm10": [10.0, 20.0, 30.0],
                "no2": [5.0, 10.0, 15.0],
            },
            schema={"time": pl.Datetime, "pm10": pl.Float64, "no2": pl.Float64},
        )
        return y_df.serialize(format="binary"), x_df.serialize(format="binary")

    def test_returns_bunch_with_classification_keys(self, classification_parquet_bytes):
        y_bytes, x_bytes = classification_parquet_bytes

        def mock_urlopen(url):
            data = y_bytes if "_y.bin" in url else x_bytes
            response = io.BytesIO(data)
            return response

        with patch("yohou.datasets._fetchers.urlopen", side_effect=mock_urlopen):
            result = _fetch_classification_wasm("air_quality_classification")

        assert isinstance(result, Bunch)
        assert "y" in result
        assert "X_actual" in result
        assert "feature_names" in result
        assert "target_names" in result
        assert "classes" in result
        assert "DESCR" in result

    def test_classes_are_sorted(self, classification_parquet_bytes):
        y_bytes, x_bytes = classification_parquet_bytes

        def mock_urlopen(url):
            data = y_bytes if "_y.bin" in url else x_bytes
            response = io.BytesIO(data)
            return response

        with patch("yohou.datasets._fetchers.urlopen", side_effect=mock_urlopen):
            result = _fetch_classification_wasm("air_quality_classification")

        assert result.classes == sorted(result.classes)
        assert result.target_names == ["air_quality"]
        assert result.feature_names == ["pm10", "no2"]


class TestClassificationWasmRouting:
    def test_fetch_air_quality_routes_to_wasm(self):
        mock_bunch = Bunch(y=pl.DataFrame(), X_actual=pl.DataFrame())
        with (
            patch("yohou.datasets._fetchers._is_wasm", return_value=True),
            patch(
                "yohou.datasets._fetchers._fetch_classification_wasm",
                return_value=mock_bunch,
            ) as mock_cls_wasm,
        ):
            result = fetch_air_quality_classification()
            mock_cls_wasm.assert_called_once_with("air_quality_classification")
            assert result is mock_bunch

    def test_fetch_demand_classification_routes_to_wasm(self):
        mock_bunch = Bunch(y=pl.DataFrame(), X_actual=pl.DataFrame())
        with (
            patch("yohou.datasets._fetchers._is_wasm", return_value=True),
            patch(
                "yohou.datasets._fetchers._fetch_classification_wasm",
                return_value=mock_bunch,
            ) as mock_cls_wasm,
        ):
            result = fetch_demand_classification()
            mock_cls_wasm.assert_called_once_with("demand_classification")
            assert result is mock_bunch
