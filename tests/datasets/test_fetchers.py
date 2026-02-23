"""Tests for dataset fetcher functions."""

from __future__ import annotations

import os
import zipfile

import polars as pl
import pytest
from sklearn.utils import Bunch

from yohou.datasets._fetchers import (
    clear_data_home,
    fetch_dominick,
    fetch_electricity_demand,
    fetch_hospital,
    fetch_pedestrian_counts,
    fetch_sunspot,
    fetch_tourism_monthly,
    fetch_tourism_quarterly,
    get_data_home,
)

ALL_FETCHERS = [
    fetch_tourism_monthly,
    fetch_sunspot,
    fetch_tourism_quarterly,
    fetch_electricity_demand,
    fetch_dominick,
    fetch_pedestrian_counts,
    fetch_hospital,
]


def _make_fake_tsf_zip(directory: str, zip_name: str, tsf_name: str) -> None:
    """Create a minimal ZIP containing a fake TSF file for mocked tests."""
    tsf_content = (
        "@attribute series_name string\n"
        "@attribute start_timestamp date\n"
        "@frequency monthly\n"
        "@horizon 12\n"
        "@missing false\n"
        "@equallength true\n"
        "@data\n"
        "T1:2000-01-01 00-00-00:10.0,20.0,30.0\n"
        "T2:2000-01-01 00-00-00:40.0,50.0,60.0\n"
    )
    zip_path = os.path.join(directory, zip_name)
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr(tsf_name, tsf_content)


class TestGetDataHome:
    """Tests for get_data_home."""

    def test_default_path(self, monkeypatch):
        """Default path is ~/yohou_data when env var is not set."""
        monkeypatch.delenv("YOHOU_DATA", raising=False)
        result = get_data_home()
        expected = os.path.join(os.path.expanduser("~"), "yohou_data")
        assert result == expected

    def test_custom_path(self, tmp_path):
        """Custom path is used when provided."""
        result = get_data_home(tmp_path)
        assert result == str(tmp_path)

    def test_env_variable(self, monkeypatch, tmp_path):
        """YOHOU_DATA environment variable is respected."""
        custom = str(tmp_path / "custom_data")
        monkeypatch.setenv("YOHOU_DATA", custom)
        result = get_data_home()
        assert result == custom
        assert os.path.isdir(custom)

    def test_creates_directory(self, tmp_path):
        """Directory is created if it doesn't exist."""
        new_dir = tmp_path / "new_data_home"
        assert not new_dir.exists()
        result = get_data_home(new_dir)
        assert result == str(new_dir)
        assert new_dir.is_dir()


class TestClearDataHome:
    """Tests for clear_data_home."""

    def test_removes_directory(self, tmp_path):
        """clear_data_home removes the data directory."""
        data_home = tmp_path / "data"
        data_home.mkdir()
        (data_home / "file.txt").write_text("test")
        assert data_home.exists()
        clear_data_home(data_home)
        assert not data_home.exists()


class TestFetchCommon:
    """Common tests for all fetch_* functions using mocked downloads."""

    @pytest.fixture(autouse=True)
    def _mock_fetch_file(self, tmp_path, monkeypatch):
        """Mock sklearn.datasets.fetch_file to create a local fake ZIP."""

        def _fake_fetch_file(
            url,
            folder=None,
            local_filename=None,
            sha256=None,
            n_retries=3,
            delay=1,
        ):
            # Determine TSF filename from the URL
            import re

            m = re.search(r"/files/(.+?)\.zip", url)
            zip_stem = m.group(1) if m else "dataset"
            tsf_name = f"{zip_stem}.tsf"
            _make_fake_tsf_zip(folder, local_filename, tsf_name)
            return os.path.join(folder, local_filename)

        monkeypatch.setattr("yohou.datasets._fetchers.fetch_file", _fake_fetch_file)

    @pytest.mark.parametrize("fetcher", ALL_FETCHERS, ids=lambda fn: fn.__name__)
    def test_returns_bunch(self, fetcher, tmp_path):
        """Each fetcher returns a Bunch object."""
        bunch = fetcher(data_home=tmp_path)
        assert isinstance(bunch, Bunch)

    @pytest.mark.parametrize("fetcher", ALL_FETCHERS, ids=lambda fn: fn.__name__)
    def test_bunch_has_required_keys(self, fetcher, tmp_path):
        """Each Bunch has the required attributes."""
        bunch = fetcher(data_home=tmp_path)
        assert hasattr(bunch, "frame")
        assert hasattr(bunch, "DESCR")
        assert hasattr(bunch, "feature_names")
        assert hasattr(bunch, "frequency")
        assert hasattr(bunch, "n_series")
        assert hasattr(bunch, "filename")

    @pytest.mark.parametrize("fetcher", ALL_FETCHERS, ids=lambda fn: fn.__name__)
    def test_frame_is_polars_dataframe(self, fetcher, tmp_path):
        """The frame attribute is a non-empty polars DataFrame with a time column."""
        bunch = fetcher(data_home=tmp_path)
        assert isinstance(bunch.frame, pl.DataFrame)
        assert "time" in bunch.frame.columns
        assert len(bunch.frame) > 0

    @pytest.mark.parametrize("fetcher", ALL_FETCHERS, ids=lambda fn: fn.__name__)
    def test_frame_has_non_time_columns(self, fetcher, tmp_path):
        """The frame has at least one non-time column."""
        bunch = fetcher(data_home=tmp_path)
        non_time = [c for c in bunch.frame.columns if c != "time"]
        assert len(non_time) >= 1

    @pytest.mark.parametrize("fetcher", ALL_FETCHERS, ids=lambda fn: fn.__name__)
    def test_feature_names_matches_columns(self, fetcher, tmp_path):
        """feature_names lists non-time columns."""
        bunch = fetcher(data_home=tmp_path)
        expected = [c for c in bunch.frame.columns if c != "time"]
        assert bunch.feature_names == expected

    @pytest.mark.parametrize("fetcher", ALL_FETCHERS, ids=lambda fn: fn.__name__)
    def test_caching_reuses_parquet(self, fetcher, tmp_path):
        """Second call loads from cached parquet, producing identical data."""
        bunch1 = fetcher(data_home=tmp_path)
        bunch2 = fetcher(data_home=tmp_path)
        assert bunch1.frame.equals(bunch2.frame)

    @pytest.mark.parametrize("fetcher", ALL_FETCHERS, ids=lambda fn: fn.__name__)
    def test_download_if_missing_false_raises(self, fetcher, tmp_path):
        """download_if_missing=False raises OSError when no cache exists."""
        empty_dir = tmp_path / "empty"
        with pytest.raises(OSError, match="Data not found"):
            fetcher(data_home=empty_dir, download_if_missing=False)

    @pytest.mark.parametrize("fetcher", ALL_FETCHERS, ids=lambda fn: fn.__name__)
    def test_download_if_missing_false_uses_cache(self, fetcher, tmp_path):
        """download_if_missing=False works when cache exists."""
        bunch1 = fetcher(data_home=tmp_path)
        bunch2 = fetcher(data_home=tmp_path, download_if_missing=False)
        assert bunch1.frame.equals(bunch2.frame)


class TestFetchIntegration:
    """Integration tests that download from Zenodo (requires network)."""

    @pytest.mark.slow
    @pytest.mark.integration
    @pytest.mark.parametrize(
        "fetcher",
        [
            fetch_tourism_monthly,
            fetch_sunspot,
            fetch_tourism_quarterly,
            fetch_electricity_demand,
            fetch_pedestrian_counts,
            fetch_hospital,
        ],
        ids=lambda fn: fn.__name__,
    )
    def test_real_download(self, fetcher, tmp_path):
        """Real download from Zenodo produces valid Bunch."""
        bunch = fetcher(data_home=tmp_path)
        assert isinstance(bunch.frame, pl.DataFrame)
        assert "time" in bunch.frame.columns
        assert len(bunch.frame) > 0
        assert bunch.frame.schema["time"] == pl.Datetime("us")

    @pytest.mark.slow
    @pytest.mark.integration
    def test_real_download_dominick(self, tmp_path):
        """Real download of the large Dominick dataset."""
        bunch = fetch_dominick(data_home=tmp_path)
        assert isinstance(bunch.frame, pl.DataFrame)
        assert "time" in bunch.frame.columns
        assert bunch.n_series == 115704
