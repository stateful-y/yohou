"""Lightweight parser for Monash Time Series Forecasting (.tsf) files."""

from __future__ import annotations

import re
from datetime import datetime
from typing import IO

import polars as pl

TSF_FREQUENCY_MAP: dict[str, str] = {
    "daily": "1d",
    "weekly": "1w",
    "monthly": "1mo",
    "yearly": "1y",
    "quarterly": "3mo",
    "hourly": "1h",
    "half_hourly": "30m",
    "minutely": "1m",
    "secondly": "1s",
    "10_minutes": "10m",
    "15_minutes": "15m",
    "30_minutes": "30m",
}


def _parse_tsf_timestamp(s: str) -> datetime:
    """Parse a TSF timestamp string to a datetime.

    TSF uses the format ``YYYY-MM-DD HH-MM-SS`` (dashes in time part).
    """
    return datetime.strptime(s.strip(), "%Y-%m-%d %H-%M-%S")


def parse_tsf(
    source: str | IO[bytes],
    *,
    value_column_name: str = "value",
) -> tuple[pl.DataFrame, dict]:
    """Parse a Monash ``.tsf`` file into a wide polars DataFrame.

    Parameters
    ----------
    source : str or file-like
        Path to a ``.tsf`` file, or an open file-like object (binary mode).
    value_column_name : str
        Name for the value column(s). For panel data, column names become
        ``"{series_name}__{value_column_name}"``.

    Returns
    -------
    tuple of (pl.DataFrame, dict)
        A tuple of ``(dataframe, metadata)`` where *dataframe* has a
        ``"time"`` column (Datetime) and one column per series, and
        *metadata* is a dict with keys ``"frequency"``, ``"horizon"``,
        ``"missing"``, ``"equallength"``, ``"relation"``, ``"n_series"``.

    """
    lines = _read_lines(source)
    attributes, header_meta, data_start_idx = _parse_header(lines)

    series_list = _parse_data_lines(lines[data_start_idx:], attributes)

    polars_freq = TSF_FREQUENCY_MAP.get(header_meta["frequency_raw"], header_meta["frequency_raw"])

    has_timestamp = any(name == "start_timestamp" for name, _ in attributes)

    frame = _build_dataframe(
        series_list,
        polars_freq=polars_freq,
        has_timestamp=has_timestamp,
        value_column_name=value_column_name,
    )

    metadata = {
        "frequency": polars_freq,
        "horizon": header_meta["horizon"],
        "missing": header_meta["missing"],
        "equallength": header_meta["equallength"],
        "relation": header_meta["relation"],
        "n_series": len(series_list),
    }

    return frame, metadata


def _read_lines(source: str | IO[bytes]) -> list[str]:
    """Read all lines from a file path or file-like object."""
    if isinstance(source, str):
        with open(source, encoding="utf-8", errors="replace") as f:
            return f.read().splitlines()

    content = source.read()
    if isinstance(content, bytes):
        content = content.decode("utf-8", errors="replace")
    return content.splitlines()


def _parse_header(
    lines: list[str],
) -> tuple[list[tuple[str, str]], dict, int]:
    """Parse TSF header lines.

    Returns ``(attributes, header_meta, data_start_index)``.
    """
    attributes: list[tuple[str, str]] = []
    header_meta: dict = {
        "frequency_raw": "",
        "horizon": None,
        "missing": False,
        "equallength": False,
        "relation": "",
    }

    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("@attribute"):
            parts = stripped.split()
            attr_name = parts[1]
            attr_type = parts[2] if len(parts) > 2 else "string"
            attributes.append((attr_name, attr_type))
        elif stripped.startswith("@frequency"):
            header_meta["frequency_raw"] = stripped.split()[1]
        elif stripped.startswith("@horizon"):
            header_meta["horizon"] = int(stripped.split()[1])
        elif stripped.startswith("@missing"):
            header_meta["missing"] = stripped.split()[1].lower() == "true"
        elif stripped.startswith("@equallength"):
            header_meta["equallength"] = stripped.split()[1].lower() == "true"
        elif stripped.startswith("@relation"):
            header_meta["relation"] = " ".join(stripped.split()[1:])
        elif stripped.startswith("@data"):
            return attributes, header_meta, i + 1

    msg = "TSF file does not contain an @data section"
    raise ValueError(msg)


def _parse_data_lines(
    data_lines: list[str],
    attributes: list[tuple[str, str]],
) -> list[dict]:
    """Parse data lines into a list of series dicts.

    Each dict has keys: ``name``, ``extra_attrs``, ``start_timestamp``,
    ``values``.
    """
    n_attrs = len(attributes)
    series_list: list[dict] = []

    for line in data_lines:
        stripped = line.strip()
        if not stripped:
            continue

        parts = stripped.split(":")
        attr_values = parts[:n_attrs]
        values_str = ":".join(parts[n_attrs:])

        series_name = attr_values[0].strip()
        start_timestamp = None
        extra_attrs: dict[str, str] = {}

        for j, (attr_name, _attr_type) in enumerate(attributes):
            if attr_name == "start_timestamp":
                start_timestamp = _parse_tsf_timestamp(attr_values[j])
            elif attr_name != "series_name":
                extra_attrs[attr_name] = attr_values[j].strip()

        values = [None if v.strip() == "?" else float(v) for v in values_str.split(",")]

        series_list.append({
            "name": series_name,
            "extra_attrs": extra_attrs,
            "start_timestamp": start_timestamp,
            "values": values,
        })

    return series_list


def _build_dataframe(
    series_list: list[dict],
    *,
    polars_freq: str,
    has_timestamp: bool,
    value_column_name: str,
) -> pl.DataFrame:
    """Convert parsed series into a wide polars DataFrame with a time column."""
    if not series_list:
        return pl.DataFrame({"time": pl.Series([], dtype=pl.Datetime)})

    n_series = len(series_list)

    if n_series == 1:
        return _build_single_series(
            series_list[0],
            polars_freq=polars_freq,
            has_timestamp=has_timestamp,
            value_column_name=value_column_name,
        )

    return _build_panel(
        series_list,
        polars_freq=polars_freq,
        has_timestamp=has_timestamp,
        value_column_name=value_column_name,
    )


def _generate_time_index(
    start: datetime | None,
    n: int,
    polars_freq: str,
) -> pl.Series:
    """Generate a datetime time index of length *n*."""
    if start is None:
        start = datetime(2000, 1, 1)

    m = re.match(r"(\d+)(\w+)", polars_freq)
    if m is None:
        msg = f"Cannot parse polars frequency string: {polars_freq!r}"
        raise ValueError(msg)

    amount = int(m.group(1))
    unit = m.group(2)
    end_offset = f"{amount * n}{unit}"

    end = pl.Series("_", [start]).dt.offset_by(end_offset).item()
    series = pl.datetime_range(start, end, interval=polars_freq, eager=True)
    return series[:n].alias("time")


def _build_single_series(
    series: dict,
    *,
    polars_freq: str,
    has_timestamp: bool,
    value_column_name: str,
) -> pl.DataFrame:
    """Build a DataFrame for a single univariate series."""
    values = series["values"]
    n = len(values)
    start = series["start_timestamp"] if has_timestamp else None

    time_col = _generate_time_index(start, n, polars_freq)
    value_col = pl.Series(value_column_name, values, dtype=pl.Float64)

    return pl.DataFrame([time_col, value_col])


def _build_panel(
    series_list: list[dict],
    *,
    polars_freq: str,
    has_timestamp: bool,
    value_column_name: str,
) -> pl.DataFrame:
    """Build a wide panel DataFrame from multiple series."""
    if has_timestamp and _all_same_start(series_list):
        return _build_panel_same_start(
            series_list,
            polars_freq=polars_freq,
            value_column_name=value_column_name,
        )

    if has_timestamp:
        return _build_panel_different_starts(
            series_list,
            polars_freq=polars_freq,
            value_column_name=value_column_name,
        )

    return _build_panel_no_timestamp(
        series_list,
        polars_freq=polars_freq,
        value_column_name=value_column_name,
    )


def _all_same_start(series_list: list[dict]) -> bool:
    """Check if all series share the same start timestamp."""
    starts = {s["start_timestamp"] for s in series_list}
    return len(starts) == 1


def _build_panel_same_start(
    series_list: list[dict],
    *,
    polars_freq: str,
    value_column_name: str,
) -> pl.DataFrame:
    """Build panel DataFrame when all series share the same start timestamp."""
    max_len = max(len(s["values"]) for s in series_list)
    start = series_list[0]["start_timestamp"]
    time_col = _generate_time_index(start, max_len, polars_freq)

    columns: dict[str, pl.Series] = {"time": time_col}
    for s in series_list:
        col_name = _panel_column_name(s, value_column_name)
        vals = s["values"]
        if len(vals) < max_len:
            vals = vals + [None] * (max_len - len(vals))
        columns[col_name] = pl.Series(col_name, vals, dtype=pl.Float64)

    return pl.DataFrame(columns)


def _build_panel_different_starts(
    series_list: list[dict],
    *,
    polars_freq: str,
    value_column_name: str,
) -> pl.DataFrame:
    """Build panel DataFrame when series have different start timestamps."""
    all_times: set[datetime] = set()
    series_time_map: list[tuple[str, dict[datetime, float | None]]] = []

    for s in series_list:
        col_name = _panel_column_name(s, value_column_name)
        start = s["start_timestamp"]
        n = len(s["values"])
        times = _generate_time_index(start, n, polars_freq).to_list()
        time_val_map = dict(zip(times, s["values"], strict=True))
        all_times.update(times)
        series_time_map.append((col_name, time_val_map))

    sorted_times = sorted(all_times)
    time_col = pl.Series("time", sorted_times, dtype=pl.Datetime("us"))

    columns: dict[str, pl.Series] = {"time": time_col}
    for col_name, time_val_map in series_time_map:
        vals = [time_val_map.get(t) for t in sorted_times]
        columns[col_name] = pl.Series(col_name, vals, dtype=pl.Float64)

    return pl.DataFrame(columns)


def _build_panel_no_timestamp(
    series_list: list[dict],
    *,
    polars_freq: str,
    value_column_name: str,
) -> pl.DataFrame:
    """Build panel DataFrame when no timestamps are available."""
    max_len = max(len(s["values"]) for s in series_list)
    start = datetime(2000, 1, 1)
    time_col = _generate_time_index(start, max_len, polars_freq)

    columns: dict[str, pl.Series] = {"time": time_col}
    for s in series_list:
        col_name = _panel_column_name(s, value_column_name)
        vals = s["values"]
        if len(vals) < max_len:
            vals = vals + [None] * (max_len - len(vals))
        columns[col_name] = pl.Series(col_name, vals, dtype=pl.Float64)

    return pl.DataFrame(columns)


def _panel_column_name(series: dict, value_column_name: str) -> str:
    """Build a panel column name using yohou's ``__`` separator convention."""
    name = series["name"]
    extra = series.get("extra_attrs", {})
    if extra:
        name = "_".join(extra.values()).lower().replace(" ", "_")
    return f"{name}__{value_column_name}"
