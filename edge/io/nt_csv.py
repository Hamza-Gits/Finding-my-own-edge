"""NinjaTrader (and generic) CSV ingestion — the RELIABLE primary data path.

NinjaTrader 8 'Historical Data > Export' writes semicolon-delimited rows:

    yyyyMMdd HHmmss;open;high;low;close;volume
    20260612 060100;29845.75;29850.00;29845.00;29848.25;123

This reader also accepts the common comma-delimited variants (e.g. Dukascopy-
style `UTC,Open,High,Low,Close,Volume` with `dd.mm.yyyy HH:MM:SS.fff UTC`
timestamps) so any exported CSV the user has on disk can be ingested.

The exported timestamps' timezone is whatever the NinjaTrader instance was set
to display. Pass `source_tz` explicitly (default 'UTC') and we localize then
convert to the exchange tz downstream.
"""
from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd

_OHLCV = ["open", "high", "low", "close", "volume"]


def _sniff_delimiter(sample: str) -> str:
    for d in (";", ",", "\t"):
        if d in sample:
            return d
    return ","


def _looks_like_header(first_field: str) -> bool:
    f = first_field.strip().lower()
    return any(k in f for k in ("date", "time", "utc", "open", "timestamp"))


def read_nt_csv(
    path: str | Path,
    source_tz: str = "UTC",
    exchange_tz: str | None = None,
) -> pd.DataFrame:
    """Read a NinjaTrader/generic OHLCV CSV into the canonical schema.

    Parameters
    ----------
    path : file to read.
    source_tz : timezone the file's timestamps are expressed in.
    exchange_tz : if given, convert the index to this tz (e.g. 'America/Chicago').

    Returns a tz-aware, ascending DataFrame [open, high, low, close, volume].
    """
    path = Path(path)
    with open(path, "r", encoding="utf-8-sig", newline="") as fh:
        sample = fh.readline()
    delim = _sniff_delimiter(sample)
    has_header = _looks_like_header(sample.split(delim)[0])

    rows: list[tuple] = []
    with open(path, "r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.reader(fh, delimiter=delim)
        if has_header:
            next(reader, None)
        for rec in reader:
            if not rec or len(rec) < 5:
                continue
            ts = _parse_timestamp(rec[0])
            o, h, l, c = (float(rec[i]) for i in range(1, 5))
            v = float(rec[5]) if len(rec) > 5 and rec[5] != "" else 0.0
            rows.append((ts, o, h, l, c, v))

    if not rows:
        raise ValueError(f"{path}: no data rows parsed")

    df = pd.DataFrame(rows, columns=["ts", *_OHLCV]).set_index("ts").sort_index()
    df = df[~df.index.duplicated(keep="last")]
    df.index = pd.DatetimeIndex(df.index).tz_localize(source_tz)
    if exchange_tz:
        df.index = df.index.tz_convert(exchange_tz)
    df.index.name = "ts"
    return df


def _parse_timestamp(raw: str) -> pd.Timestamp:
    """Parse the several timestamp shapes seen in exports, tz-naive."""
    s = raw.strip().replace(" UTC", "").strip()
    # NinjaTrader native:  'yyyyMMdd HHmmss'  or  'yyyyMMdd HHmmssfff'
    if len(s) >= 15 and s[8] == " " and s[:8].isdigit() and s[9:15].isdigit():
        date, t = s[:8], s[9:]
        micro = int(t[6:].ljust(6, "0")[:6]) if len(t) > 6 else 0
        return pd.Timestamp(
            year=int(date[:4]), month=int(date[4:6]), day=int(date[6:8]),
            hour=int(t[:2]), minute=int(t[2:4]), second=int(t[4:6]), microsecond=micro,
        )
    # Fallback: let pandas infer (handles 'dd.mm.yyyy HH:MM:SS.fff', ISO, etc.)
    return pd.Timestamp(pd.to_datetime(s, dayfirst="." in s))
