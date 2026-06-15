"""Tests for dataset fetcher functions."""

from __future__ import annotations

import os
import zipfile
from datetime import datetime

import polars as pl
import pytest
from sklearn.utils import Bunch

from yohou.datasets._fetchers import (
    _restructure_kdd_cup_columns,
    clear_data_home,
    fetch_air_quality_classification,
    fetch_demand_classification,
    fetch_dominick,
    fetch_electricity_demand,
    fetch_hospital,
    fetch_kdd_cup,
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
    fetch_kdd_cup,
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

    def test_n_series_limits_columns(self, tmp_path):
        """n_series limits the number of series parsed and cached."""
        bunch = fetch_tourism_monthly(data_home=tmp_path, n_series=1)
        non_time = [c for c in bunch.frame.columns if c != "time"]
        assert len(non_time) == 1
        assert bunch.n_series == 1

    def test_n_series_none_loads_all(self, tmp_path):
        """n_series=None loads all series (2 in the mock TSF)."""
        bunch = fetch_tourism_monthly(data_home=tmp_path, n_series=None)
        non_time = [c for c in bunch.frame.columns if c != "time"]
        assert len(non_time) == 2
        assert bunch.n_series == 2

    def test_n_series_separate_cache(self, tmp_path):
        """Different n_series values produce separate cache files."""
        bunch_all = fetch_tourism_monthly(data_home=tmp_path, n_series=None)
        bunch_sub = fetch_tourism_monthly(data_home=tmp_path, n_series=1)
        assert bunch_all.filename != bunch_sub.filename
        assert len(bunch_all.feature_names) == 2
        assert len(bunch_sub.feature_names) == 1


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
        """Real download of the large Dominick dataset (default subset)."""
        bunch = fetch_dominick(data_home=tmp_path)
        assert isinstance(bunch.frame, pl.DataFrame)
        assert "time" in bunch.frame.columns
        assert bunch.n_series == 50

    @pytest.mark.slow
    @pytest.mark.integration
    def test_real_download_kdd_cup(self, tmp_path):
        """Real download of the KDD Cup 2018 dataset (default subset)."""
        bunch = fetch_kdd_cup(data_home=tmp_path)
        assert isinstance(bunch.frame, pl.DataFrame)
        assert "time" in bunch.frame.columns
        assert bunch.n_series == 30  # default n_groups=5 × 6 measurements
        for col in bunch.feature_names:
            assert "__" in col
            assert not col.endswith("__value")


class TestRestructureKddCup:
    """Tests for the _restructure_kdd_cup_columns helper."""

    def test_renames_measurement_columns(self):
        """Columns matching the KDD Cup pattern are restructured."""
        frame = pl.DataFrame({
            "time": [1, 2],
            "beijing_dongsi_aq_pm2.5__value": [10.0, 20.0],
            "beijing_dongsi_aq_co__value": [1.0, 2.0],
        }).cast({"time": pl.Datetime})
        result = _restructure_kdd_cup_columns(frame)
        assert "beijing_dongsi_aq__pm2.5" in result.columns
        assert "beijing_dongsi_aq__co" in result.columns
        assert "beijing_dongsi_aq_pm2.5__value" not in result.columns

    def test_preserves_time_column(self):
        """The time column is never renamed."""
        frame = pl.DataFrame({
            "time": [1, 2],
            "beijing_dongsi_aq_o3__value": [3.0, 4.0],
        }).cast({"time": pl.Datetime})
        result = _restructure_kdd_cup_columns(frame)
        assert "time" in result.columns

    def test_no_op_on_non_matching_columns(self):
        """Non-matching columns pass through unchanged."""
        frame = pl.DataFrame({
            "time": [1, 2],
            "T1__value": [10.0, 20.0],
            "T2__value": [30.0, 40.0],
        }).cast({"time": pl.Datetime})
        result = _restructure_kdd_cup_columns(frame)
        assert result.columns == ["time", "T1__value", "T2__value"]


@pytest.fixture(scope="module")
def air_quality_data():
    """Fetch the air quality classification dataset once per module."""
    return fetch_air_quality_classification()


@pytest.fixture(scope="module")
def demand_data():
    """Fetch the demand classification dataset once per module."""
    return fetch_demand_classification()


class TestFetchAirQualityClassification:
    """Tests for fetch_air_quality_classification."""

    def test_returns_bunch(self, air_quality_data):
        """Result is a sklearn Bunch."""
        assert isinstance(air_quality_data, Bunch)

    def test_bunch_keys(self, air_quality_data):
        """Bunch contains required keys."""
        for key in ("y", "X_actual", "feature_names", "target_names", "classes", "DESCR"):
            assert key in air_quality_data

    def test_y_schema(self, air_quality_data):
        """y has time (datetime) and air_quality (string) columns."""
        y = air_quality_data.y
        assert "time" in y.columns
        assert "air_quality" in y.columns
        assert y["time"].dtype.is_temporal()
        assert y["air_quality"].dtype == pl.Utf8

    def test_x_schema(self, air_quality_data):
        """X_actual has time (datetime) and 5 numeric feature columns."""
        X_actual = air_quality_data.X_actual
        assert "time" in X_actual.columns
        assert len(X_actual.columns) == 6  # time + 5 features
        for col in X_actual.columns:
            if col != "time":
                assert X_actual[col].dtype.is_numeric()

    def test_y_x_alignment(self, air_quality_data):
        """y and X_actual have the same length and time column."""
        y, X_actual = air_quality_data.y, air_quality_data.X_actual
        assert len(y) == len(X_actual)
        assert y["time"].equals(X_actual["time"])

    def test_classes(self, air_quality_data):
        """Classes match WHO air quality categories."""
        classes = sorted(air_quality_data.classes)
        assert classes == ["good", "hazardous", "moderate", "unhealthy"]

    def test_all_labels_in_classes(self, air_quality_data):
        """All y labels are among the declared classes."""
        labels = set(air_quality_data.y["air_quality"].to_list())
        assert labels <= set(air_quality_data.classes)

    def test_target_names(self, air_quality_data):
        """target_names is ["air_quality"]."""
        assert air_quality_data.target_names == ["air_quality"]

    def test_feature_names(self, air_quality_data):
        """feature_names matches X_actual columns minus time."""
        expected = [c for c in air_quality_data.X_actual.columns if c != "time"]
        assert air_quality_data.feature_names == expected

    def test_no_nulls_in_y(self, air_quality_data):
        """y has no null values."""
        assert air_quality_data.y.null_count().row(0) == (0, 0)

    def test_descr_is_string(self, air_quality_data):
        """DESCR is a non-empty string."""
        assert isinstance(air_quality_data.DESCR, str)
        assert len(air_quality_data.DESCR) > 0


class TestFetchDemandClassification:
    """Tests for fetch_demand_classification."""

    def test_returns_bunch(self, demand_data):
        """Result is a sklearn Bunch."""
        assert isinstance(demand_data, Bunch)

    def test_bunch_keys(self, demand_data):
        """Bunch contains required keys."""
        for key in ("y", "X_actual", "feature_names", "target_names", "classes", "DESCR"):
            assert key in demand_data

    def test_y_schema(self, demand_data):
        """y has time (datetime) and demand_level (string) columns."""
        y = demand_data.y
        assert "time" in y.columns
        assert "demand_level" in y.columns
        assert y["time"].dtype.is_temporal()
        assert y["demand_level"].dtype == pl.Utf8

    def test_x_schema(self, demand_data):
        """X_actual has time (datetime) and 4 numeric feature columns."""
        X_actual = demand_data.X_actual
        assert "time" in X_actual.columns
        assert len(X_actual.columns) == 5  # time + 4 features
        for col in X_actual.columns:
            if col != "time":
                assert X_actual[col].dtype.is_numeric()

    def test_y_x_alignment(self, demand_data):
        """y and X_actual have the same length and time column."""
        y, X_actual = demand_data.y, demand_data.X_actual
        assert len(y) == len(X_actual)
        assert y["time"].equals(X_actual["time"])

    def test_classes(self, demand_data):
        """Classes are low, medium, high."""
        classes = sorted(demand_data.classes)
        assert classes == ["high", "low", "medium"]

    def test_all_labels_in_classes(self, demand_data):
        """All y labels are among the declared classes."""
        labels = set(demand_data.y["demand_level"].to_list())
        assert labels <= set(demand_data.classes)

    def test_target_names(self, demand_data):
        """target_names is ["demand_level"]."""
        assert demand_data.target_names == ["demand_level"]

    def test_feature_names(self, demand_data):
        """feature_names matches X_actual columns minus time."""
        expected = [c for c in demand_data.X_actual.columns if c != "time"]
        assert demand_data.feature_names == expected

    def test_feature_columns_are_panel(self, demand_data):
        """Feature columns use __ separator convention."""
        for col in demand_data.feature_names:
            assert "__" in col

    def test_no_nulls_in_y(self, demand_data):
        """y has no null values."""
        assert demand_data.y.null_count().row(0) == (0, 0)

    def test_descr_is_string(self, demand_data):
        """DESCR is a non-empty string."""
        assert isinstance(demand_data.DESCR, str)
        assert len(demand_data.DESCR) > 0


class TestClassificationMissingColumns:
    """The classification fetchers raise clear errors when expected columns are absent."""

    def test_air_quality_missing_pm25_raises(self, monkeypatch):
        """A KDD Cup frame without the pm2.5 target column raises ValueError."""
        time = pl.datetime_range(start=datetime(2020, 1, 1), end=datetime(2020, 1, 3), interval="1h", eager=True)
        frame = pl.DataFrame({"time": time, "beijing__pm10": [1.0] * len(time)})

        monkeypatch.setattr(
            "yohou.datasets._fetchers.fetch_kdd_cup",
            lambda **kwargs: Bunch(frame=frame),
        )

        with pytest.raises(ValueError, match="pm2.5"):
            fetch_air_quality_classification()

    def test_demand_missing_columns_raises(self, monkeypatch):
        """An electricity demand frame missing required columns raises ValueError."""
        time = pl.datetime_range(start=datetime(2020, 1, 1), end=datetime(2020, 1, 3), interval="1h", eager=True)
        frame = pl.DataFrame({"time": time, "nsw__demand": [1.0] * len(time)})

        monkeypatch.setattr(
            "yohou.datasets._fetchers.fetch_electricity_demand",
            lambda **kwargs: Bunch(frame=frame),
        )

        with pytest.raises(ValueError, match="electricity demand"):
            fetch_demand_classification()
