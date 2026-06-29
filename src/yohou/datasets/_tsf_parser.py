"""Lightweight parser for Monash Time Series Forecasting (.tsf) files."""

from __future__ import annotations

import re
from collections.abc import Iterator
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
}


def _parse_tsf_timestamp(s: str) -> datetime:
    """Parse a TSF timestamp string to a datetime.

    TSF uses the format ``YYYY-MM-DD HH-MM-SS`` (dashes in time part).
    """
    return datetime.strptime(s.strip(), "%Y-%m-%d %H-%M-%S")


def _parse_tsf(
    source: str | IO[bytes],
    *,
    value_column_name: str = "value",
    n_series: int | None = None,
) -> tuple[pl.DataFrame, dict]:
    """Parse a Monash ``.tsf`` file into a wide polars DataFrame.

    Parameters
    ----------
    source : str or file-like
        Path to a ``.tsf`` file, or an open file-like object (binary mode).
    value_column_name : str
        Name for the value column(s). For a single-series file the value
        column is named exactly ``value_column_name`` (no ``__`` separator and
        no series-name prefix). For panel data (multiple series) the column
        prefix depends on the series attributes: when a series has no extra
        attributes beyond ``series_name`` and ``start_timestamp``, columns
        become ``"{series_name}__{value_column_name}"``; when other extra
        attributes are present, their values (joined with underscores,
        lowercased, spaces replaced by underscores) are used as the prefix
        instead, giving ``"{joined_attributes}__{value_column_name}"``.
    n_series : int or None
        Maximum number of series to parse. ``None`` parses all series.
        Use this to limit memory consumption for large datasets.

    Returns
    -------
    tuple of (pl.DataFrame, dict)
        A tuple of ``(dataframe, metadata)`` where *dataframe* has a
        ``"time"`` column (Datetime) and one column per series, and
        *metadata* is a dict with keys ``"frequency"``, ``"horizon"``,
        ``"missing"``, ``"equallength"``, ``"relation"``, ``"n_series"``.

    Raises
    ------
    ValueError
        In any of the following cases:

        - the file contains no ``@data`` section;
        - the file is missing a required ``@frequency`` directive;
        - a ``start_timestamp`` attribute value cannot be parsed as a
          datetime string;
        - the frequency string cannot be parsed into a polars duration
          (raised indirectly via :func:`_generate_time_index`).

    """
    line_iter = _iter_text_lines(source)
    attributes, header_meta = _parse_header(line_iter)

    if not header_meta["frequency_raw"]:
        msg = "TSF file is missing a required @frequency directive"
        raise ValueError(msg)

    series_list = _parse_data_lines(line_iter, attributes, n_series=n_series)

    polars_freq = TSF_FREQUENCY_MAP.get(header_meta["frequency_raw"], header_meta["frequency_raw"])

    has_timestamp = "start_timestamp" in attributes

    frame = _build_dataframe(
        series_list,
        polars_freq=polars_freq,
        has_timestamp=has_timestamp,
        value_column_name=value_column_name,
        n_series=n_series,
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


def _iter_text_lines(source: str | IO[bytes]) -> Iterator[str]:
    """Yield decoded text lines from a file path or file-like object.

    Unlike reading all lines into a list, this streams one line at a
    time so that large files are never fully loaded into memory.
    """
    if isinstance(source, str):
        with open(source, encoding="utf-8", errors="replace") as f:
            yield from (line.rstrip("\n\r") for line in f)
    else:
        for raw_line in source:
            if isinstance(raw_line, bytes):
                yield raw_line.decode("utf-8", errors="replace").rstrip("\n\r")
            else:
                yield raw_line.rstrip("\n\r")


def _parse_header(
    line_iter: Iterator[str],
) -> tuple[list[str], dict]:
    """Parse TSF header lines from a line iterator.

    Advances the iterator past the ``@data`` marker and returns
    ``(attributes, header_meta)``, where *attributes* is the list of
    attribute names in declaration order.
    """
    attributes: list[str] = []
    header_meta: dict = {
        "frequency_raw": "",
        "horizon": None,
        "missing": False,
        "equallength": False,
        "relation": "",
    }

    for line in line_iter:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("@attribute"):
            attr_name = stripped.split()[1]
            attributes.append(attr_name)
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
            return attributes, header_meta

    msg = "TSF file does not contain an @data section"
    raise ValueError(msg)


def _parse_data_lines(
    line_iter: Iterator[str],
    attributes: list[str],
    *,
    n_series: int | None = None,
) -> list[dict]:
    """Parse data lines from a line iterator into a list of series dicts.

    Each dict has keys: ``name``, ``extra_attrs``, ``start_timestamp``,
    ``values``.
    """
    n_attrs = len(attributes)
    series_list: list[dict] = []

    for line_number, line in enumerate(line_iter, start=1):
        stripped = line.strip()
        if not stripped:
            continue

        parts = stripped.split(":")
        attr_values = parts[:n_attrs]
        values_str = ":".join(parts[n_attrs:])

        series_name = attr_values[0].strip()
        start_timestamp = None
        extra_attrs: dict[str, str] = {}

        for j, attr_name in enumerate(attributes):
            if attr_name == "start_timestamp":
                try:
                    start_timestamp = _parse_tsf_timestamp(attr_values[j])
                except ValueError as exc:
                    msg = (
                        f"Cannot parse start_timestamp {attr_values[j]!r} for "
                        f"series {series_name!r} (data line {line_number})"
                    )
                    raise ValueError(msg) from exc
            elif attr_name != "series_name":
                extra_attrs[attr_name] = attr_values[j].strip()

        values = [None if v.strip() == "?" else float(v) for v in values_str.split(",")]

        series_list.append({
            "name": series_name,
            "extra_attrs": extra_attrs,
            "start_timestamp": start_timestamp,
            "values": values,
        })

        if n_series is not None and len(series_list) >= n_series:
            break

    return series_list


def _build_dataframe(
    series_list: list[dict],
    *,
    polars_freq: str,
    has_timestamp: bool,
    value_column_name: str,
    n_series: int | None = None,
) -> pl.DataFrame:
    """Convert parsed series into a wide polars DataFrame with a time column.

    The bare value-column name (no ``__`` separator) is used only for a
    genuinely univariate file, i.e. when ``n_series`` was not requested and
    the file itself declares a single series. When ``n_series`` truncation
    leaves a single series, panel naming (``{group}__{value_column_name}``)
    is preserved so the output schema matches the ``n_series >= 2`` case.
    """
    if not series_list:
        return pl.DataFrame({"time": pl.Series([], dtype=pl.Datetime)})

    if len(series_list) == 1 and n_series is None:
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
    """Generate a datetime time index of length *n*.

    If *start* is None, defaults to ``datetime(2000, 1, 1)``.
    """
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
    start: datetime | None = None,
) -> pl.DataFrame:
    """Build panel DataFrame when all series share the same start timestamp.

    When *start* is None, the shared start is read from the first series'
    ``start_timestamp``. Callers without timestamps pass an explicit *start*.
    """
    max_len = max(len(s["values"]) for s in series_list)
    if start is None:
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
    per_series: list[pl.DataFrame] = []
    for s in series_list:
        col_name = _panel_column_name(s, value_column_name)
        start = s["start_timestamp"]
        n = len(s["values"])
        time_col = _generate_time_index(start, n, polars_freq).cast(pl.Datetime("us"))
        per_series.append(
            pl.DataFrame({"time": time_col, col_name: pl.Series(col_name, s["values"], dtype=pl.Float64)})
        )

    return pl.concat(per_series, how="align_full").sort("time")


def _build_panel_no_timestamp(
    series_list: list[dict],
    *,
    polars_freq: str,
    value_column_name: str,
) -> pl.DataFrame:
    """Build panel DataFrame when no timestamps are available.

    Identical to the same-start case but with a synthetic shared start of
    ``datetime(2000, 1, 1)``.
    """
    return _build_panel_same_start(
        series_list,
        polars_freq=polars_freq,
        value_column_name=value_column_name,
        start=datetime(2000, 1, 1),
    )


def _panel_column_name(series: dict, value_column_name: str) -> str:
    """Build a panel column name using yohou's ``__`` separator convention.

    If the series has extra attributes beyond ``series_name`` and
    ``start_timestamp``, their values are joined with underscores
    (lowercased, spaces replaced by underscores) and used as the column
    prefix instead of the series name. Otherwise the series name is used.
    """
    name = series["name"]
    extra = series.get("extra_attrs", {})
    if extra:
        name = "_".join(extra.values()).lower().replace(" ", "_")
    return f"{name}__{value_column_name}"
