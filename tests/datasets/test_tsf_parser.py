"""Tests for TSF parser."""

from datetime import datetime
from io import BytesIO

import polars as pl
import pytest

from yohou.datasets._tsf_parser import (
    TSF_FREQUENCY_MAP,
    _parse_tsf,
    _parse_tsf_timestamp,
)


class TestParseTsfTimestamp:
    """Tests for TSF timestamp parsing."""

    def test_basic_timestamp(self):
        """Parse a standard TSF timestamp."""
        result = _parse_tsf_timestamp("1979-01-01 00-00-00")
        assert result == datetime(1979, 1, 1, 0, 0, 0)

    def test_timestamp_with_time(self):
        """Parse a TSF timestamp with non-zero time."""
        result = _parse_tsf_timestamp("2009-05-01 12-30-45")
        assert result == datetime(2009, 5, 1, 12, 30, 45)

    def test_timestamp_with_whitespace(self):
        """Parse a TSF timestamp with surrounding whitespace."""
        result = _parse_tsf_timestamp("  2000-01-01 00-00-00  ")
        assert result == datetime(2000, 1, 1, 0, 0, 0)


class TestFrequencyMap:
    """Tests for TSF frequency mapping."""

    @pytest.mark.parametrize(
        ("tsf_freq", "polars_freq"),
        [
            ("daily", "1d"),
            ("weekly", "1w"),
            ("monthly", "1mo"),
            ("quarterly", "3mo"),
            ("yearly", "1y"),
            ("hourly", "1h"),
            ("half_hourly", "30m"),
        ],
    )
    def test_known_frequencies(self, tsf_freq, polars_freq):
        """Known TSF frequencies map correctly."""
        assert TSF_FREQUENCY_MAP[tsf_freq] == polars_freq


class TestParseTsf:
    """Tests for the main _parse_tsf function."""

    def _make_tsf(self, content: str) -> BytesIO:
        """Create a BytesIO TSF source from text content."""
        return BytesIO(content.encode("utf-8"))

    def test_univariate_with_timestamp(self):
        """Parse a single series with timestamp."""
        tsf = self._make_tsf(
            "@attribute series_name string\n"
            "@attribute start_timestamp date\n"
            "@frequency monthly\n"
            "@horizon 12\n"
            "@missing false\n"
            "@data\n"
            "T1:2000-01-01 00-00-00:10.0,20.0,30.0\n"
        )
        df, meta = _parse_tsf(tsf, value_column_name="val")

        assert isinstance(df, pl.DataFrame)
        assert "time" in df.columns
        assert "val" in df.columns
        assert len(df) == 3
        assert meta["frequency"] == "1mo"
        assert meta["horizon"] == 12
        assert meta["n_series"] == 1
        assert df["val"].to_list() == [10.0, 20.0, 30.0]

    def test_panel_same_start(self):
        """Parse multiple series with the same start timestamp."""
        tsf = self._make_tsf(
            "@attribute series_name string\n"
            "@attribute start_timestamp date\n"
            "@frequency monthly\n"
            "@missing false\n"
            "@data\n"
            "S1:2000-01-01 00-00-00:1.0,2.0,3.0\n"
            "S2:2000-01-01 00-00-00:4.0,5.0,6.0\n"
        )
        df, meta = _parse_tsf(tsf, value_column_name="sales")

        assert df.shape == (3, 3)
        assert df.columns == ["time", "S1__sales", "S2__sales"]
        assert meta["n_series"] == 2

    def test_panel_different_starts(self):
        """Parse multiple series with different start timestamps."""
        tsf = self._make_tsf(
            "@attribute series_name string\n"
            "@attribute start_timestamp date\n"
            "@frequency monthly\n"
            "@missing false\n"
            "@data\n"
            "S1:2000-01-01 00-00-00:1.0,2.0\n"
            "S2:2000-02-01 00-00-00:3.0,4.0\n"
        )
        df, meta = _parse_tsf(tsf, value_column_name="val")

        assert "time" in df.columns
        assert len(df) == 3  # Jan, Feb, Mar
        assert meta["n_series"] == 2

    def test_panel_unequal_length(self):
        """Parse panel with series of different lengths."""
        tsf = self._make_tsf(
            "@attribute series_name string\n"
            "@attribute start_timestamp date\n"
            "@frequency monthly\n"
            "@missing false\n"
            "@data\n"
            "S1:2000-01-01 00-00-00:1.0,2.0,3.0\n"
            "S2:2000-01-01 00-00-00:4.0,5.0\n"
        )
        df, meta = _parse_tsf(tsf, value_column_name="val")

        assert df.shape == (3, 3)
        assert df["S2__val"].to_list()[2] is None

    def test_n_series_1_truncation_keeps_panel_naming(self):
        """Truncating a multi-series file to n_series=1 keeps group__column naming.

        Regression for the 2026-06-21 QA finding: ``n_series=1`` took a
        single-series fast-path that named the column bare (``sales``)
        rather than the panel form (``S1__sales``), silently diverging
        from the ``n_series>=2`` output schema and breaking the panel
        data contract.
        """
        tsf = self._make_tsf(
            "@attribute series_name string\n"
            "@attribute start_timestamp date\n"
            "@frequency monthly\n"
            "@missing false\n"
            "@data\n"
            "S1:2000-01-01 00-00-00:1.0,2.0,3.0\n"
            "S2:2000-01-01 00-00-00:4.0,5.0,6.0\n"
        )
        df, meta = _parse_tsf(tsf, value_column_name="sales", n_series=1)

        assert meta["n_series"] == 1
        assert df.columns == ["time", "S1__sales"]

    def test_genuinely_univariate_file_keeps_bare_name(self):
        """A file declaring a single series keeps the bare value column name.

        The bare-name path remains correct for genuinely univariate
        datasets (e.g. sunspot), where no panel structure exists.
        """
        tsf = self._make_tsf(
            "@attribute series_name string\n"
            "@attribute start_timestamp date\n"
            "@frequency monthly\n"
            "@missing false\n"
            "@data\n"
            "T1:2000-01-01 00-00-00:10.0,20.0,30.0\n"
        )
        df, meta = _parse_tsf(tsf, value_column_name="val")

        assert meta["n_series"] == 1
        assert df.columns == ["time", "val"]

    def test_no_timestamp(self):
        """Parse dataset without start_timestamp attribute."""
        tsf = self._make_tsf(
            "@attribute series_name string\n@frequency weekly\n@missing false\n@data\nT1:10.0,20.0,30.0\nT2:40.0,50.0\n"
        )
        df, meta = _parse_tsf(tsf, value_column_name="profit")

        assert "time" in df.columns
        assert len(df) == 3
        assert "T1__profit" in df.columns
        assert "T2__profit" in df.columns
        assert meta["frequency"] == "1w"

    def test_extra_attributes(self):
        """Parse dataset with extra attributes (e.g., state)."""
        tsf = self._make_tsf(
            "@attribute series_name string\n"
            "@attribute state string\n"
            "@attribute start_timestamp date\n"
            "@frequency half_hourly\n"
            "@missing false\n"
            "@data\n"
            "T1:NSW:2002-01-01 00-00-00:100.0,200.0\n"
            "T2:VIC:2002-01-01 00-00-00:300.0,400.0\n"
        )
        df, meta = _parse_tsf(tsf, value_column_name="demand")

        assert "nsw__demand" in df.columns
        assert "vic__demand" in df.columns
        assert meta["frequency"] == "30m"

    def test_missing_values_question_mark(self):
        """Parse series with '?' missing value markers."""
        tsf = self._make_tsf(
            "@attribute series_name string\n"
            "@attribute start_timestamp date\n"
            "@frequency daily\n"
            "@missing true\n"
            "@data\n"
            "T1:2000-01-01 00-00-00:1.0,?,3.0\n"
        )
        df, meta = _parse_tsf(tsf, value_column_name="val")

        assert df["val"][1] is None
        assert meta["missing"]

    def test_comments_and_blank_lines_skipped(self):
        """Comments and blank lines in the header are ignored."""
        tsf = self._make_tsf(
            "# This is a comment\n"
            "\n"
            "@attribute series_name string\n"
            "@attribute start_timestamp date\n"
            "# Another comment\n"
            "@frequency daily\n"
            "@data\n"
            "T1:2000-01-01 00-00-00:1.0,2.0\n"
        )
        df, meta = _parse_tsf(tsf, value_column_name="val")

        assert len(df) == 2
        assert meta["n_series"] == 1

    def test_metadata_fields(self):
        """All metadata fields are correctly extracted."""
        tsf = self._make_tsf(
            "@relation TestRelation\n"
            "@attribute series_name string\n"
            "@attribute start_timestamp date\n"
            "@frequency monthly\n"
            "@horizon 24\n"
            "@missing false\n"
            "@equallength true\n"
            "@data\n"
            "T1:2000-01-01 00-00-00:1.0\n"
        )
        _, meta = _parse_tsf(tsf, value_column_name="val")

        assert meta["relation"] == "TestRelation"
        assert meta["horizon"] == 24
        assert meta["missing"] is False
        assert meta["equallength"] is True

    def test_str_filepath_source(self, tmp_path):
        """A str file path is parsed identically to a BytesIO source."""
        content = (
            "@attribute series_name string\n"
            "@attribute start_timestamp date\n"
            "@frequency monthly\n"
            "@data\n"
            "T1:2000-01-01 00-00-00:10.0,20.0,30.0\n"
        )
        path = tmp_path / "sample.tsf"
        path.write_text(content)

        df_path, meta_path = _parse_tsf(str(path), value_column_name="val")
        df_buf, meta_buf = _parse_tsf(self._make_tsf(content), value_column_name="val")

        assert df_path.equals(df_buf)
        assert meta_path == meta_buf
        assert df_path["val"].to_list() == [10.0, 20.0, 30.0]

    def test_empty_data_section_returns_empty_frame(self):
        """A @data section with no data rows yields an empty Datetime frame."""
        tsf = self._make_tsf(
            "@attribute series_name string\n@attribute start_timestamp date\n@frequency monthly\n@data\n"
        )
        df, meta = _parse_tsf(tsf, value_column_name="val")

        assert df.shape == (0, 1)
        assert df.columns == ["time"]
        assert df.schema["time"] == pl.Datetime
        assert meta["n_series"] == 0

    def test_invalid_polars_frequency_string_raises(self):
        """A frequency that is unmapped and not <int><unit> raises ValueError."""
        tsf = self._make_tsf(
            "@attribute series_name string\n"
            "@attribute start_timestamp date\n"
            "@frequency special\n"
            "@data\n"
            "T1:2000-01-01 00-00-00:1.0,2.0,3.0\n"
        )
        with pytest.raises(ValueError, match="Cannot parse polars frequency string"):
            _parse_tsf(tsf, value_column_name="val")

    def test_no_data_section_raises(self):
        """Missing @data section raises ValueError."""
        tsf = self._make_tsf("@attribute series_name string\n@frequency daily\n")
        with pytest.raises(ValueError, match="@data"):
            _parse_tsf(tsf)

    def test_missing_frequency_raises(self):
        """Missing @frequency directive raises an explicit ValueError."""
        tsf = self._make_tsf("@attribute series_name string\n@data\nT1:1.0,2.0,3.0\n")
        with pytest.raises(ValueError, match="missing a required @frequency"):
            _parse_tsf(tsf, value_column_name="val")

    def test_malformed_timestamp_raises_with_context(self):
        """A malformed start_timestamp raises an error naming the series."""
        tsf = self._make_tsf(
            "@attribute series_name string\n"
            "@attribute start_timestamp date\n"
            "@frequency monthly\n"
            "@data\n"
            "T1:not-a-timestamp:1.0,2.0,3.0\n"
        )
        with pytest.raises(ValueError, match="start_timestamp.*'T1'"):
            _parse_tsf(tsf, value_column_name="val")

    def test_time_column_is_datetime(self):
        """The time column should always be Datetime type."""
        tsf = self._make_tsf(
            "@attribute series_name string\n"
            "@attribute start_timestamp date\n"
            "@frequency monthly\n"
            "@data\n"
            "T1:2000-01-01 00-00-00:1.0,2.0,3.0\n"
        )
        df, _ = _parse_tsf(tsf, value_column_name="val")

        assert df.schema["time"] == pl.Datetime("us")
