"""NinjaTrader 8 .ncd binary reader.

Reverse-engineered file layout (verified empirically on this machine's data):

    Header (28 bytes, all little-endian):
        int32   version       (observed value = 1)
        double  tick_size      (e.g. 0.25 for MNQ/MES)
        double  base_price     (anchor: first record's reference price)
        int64   base_time      (.NET DateTime ticks = 100 ns since 0001-01-01)

    DAY files  (db/day/<CONTRACT>/<year>.Last.ncd):
        UNCOMPRESSED fixed 48-byte records from offset 28:
            int64   time   (.NET ticks)
            double  open, high, low, close
            int64   volume
        -> Fully decoded and VERIFIED (OHLC invariants hold; values sane).
           Used as a ground-truth ORACLE for validating the minute decoder.

    MINUTE / TICK files (db/minute, db/tick, db/cache):
        PROPRIETARY COMPRESSED stream from offset 28 (~6-7 bytes/bar, a
        delta+varint scheme). NOT safely decodable bit-exact without a
        ground-truth CSV. `read_minute_ncd` therefore raises until the format
        is locked by `validate_minute_decoder` against a NinjaTrader CSV export.

Timezone: db .ncd timestamps are UTC (db minute base time 00:01:00 vs the same
bar rendered 18:01:00 ET in db/cache). Callers convert UTC -> exchange tz via
edge.data.sessions. (To be CONFIRMED against the user's CSV export.)
"""
from __future__ import annotations

import struct
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# .NET DateTime epoch (0001-01-01) in 100-ns ticks.
_NET_EPOCH = datetime(1, 1, 1, tzinfo=timezone.utc)
_HEADER_FMT = "<idqq"  # NOTE: base_price is a double; we read it separately below.
_HEADER_SIZE = 28
_DAY_RECORD_SIZE = 48


def _net_ticks_to_utc(ticks: int) -> datetime:
    """Convert .NET DateTime ticks (100 ns since 0001-01-01) to a UTC datetime."""
    return _NET_EPOCH + timedelta(microseconds=ticks // 10)


def read_ncd_header(data: bytes) -> dict:
    """Parse the common 28-byte .ncd header."""
    if len(data) < _HEADER_SIZE:
        raise ValueError("file too small to contain an .ncd header")
    version = struct.unpack_from("<i", data, 0)[0]
    tick_size = struct.unpack_from("<d", data, 4)[0]
    base_price = struct.unpack_from("<d", data, 12)[0]
    base_time = struct.unpack_from("<q", data, 20)[0]
    return {
        "version": version,
        "tick_size": tick_size,
        "base_price": base_price,
        "base_time_ticks": base_time,
        "base_time_utc": _net_ticks_to_utc(base_time),
    }


def read_day_ncd(path: str | Path) -> pd.DataFrame:
    """Decode a DAY .ncd file into a canonical OHLCV frame (UTC index).

    Returns a DataFrame indexed by tz-aware UTC timestamps with columns
    open, high, low, close, volume. VERIFIED layout — safe to trust.
    """
    data = Path(path).read_bytes()
    header = read_ncd_header(data)
    rows = []
    off = _HEADER_SIZE
    n = len(data)
    while off + _DAY_RECORD_SIZE <= n:
        t = struct.unpack_from("<q", data, off)[0]
        o, h, l, c = struct.unpack_from("<dddd", data, off + 8)
        v = struct.unpack_from("<q", data, off + 40)[0]
        rows.append((_net_ticks_to_utc(t), o, h, l, c, float(v)))
        off += _DAY_RECORD_SIZE
    if off != n:
        # Day files have observed to end exactly on a record boundary; a non-zero
        # remainder signals a format mismatch we must not silently ignore.
        raise ValueError(
            f"day file {path}: {n - off} trailing bytes — record layout mismatch"
        )
    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
    df = df.set_index("ts").sort_index()
    df.index = df.index.tz_convert("UTC")
    _validate_ohlc(df, source=str(path))
    return df


def read_day_contract(day_dir: str | Path) -> pd.DataFrame:
    """Concatenate all yearly DAY files in a contract folder (e.g. 'MNQ 09-26')."""
    files = sorted(Path(day_dir).glob("*.ncd"))
    if not files:
        raise FileNotFoundError(f"no .ncd day files in {day_dir}")
    frames = [read_day_ncd(f) for f in files]
    return pd.concat(frames).sort_index()


class MinuteDecoderNotValidated(NotImplementedError):
    """Raised when minute/tick .ncd decode is requested before bit-exact validation."""


def read_minute_ncd(path: str | Path) -> pd.DataFrame:
    """Decode a MINUTE .ncd file.

    The per-bar payload is a proprietary compressed delta+varint stream. Until
    `validate_minute_decoder` confirms a candidate decoder bit-exact against a
    NinjaTrader CSV export, we refuse to emit numbers we cannot vouch for —
    shipping a guessed decoder would silently corrupt the research foundation,
    violating the prime directive. Use `nt_csv` to ingest minute data meanwhile.
    """
    raise MinuteDecoderNotValidated(
        "Minute .ncd decoding is not yet validated. Export the data to CSV from "
        "NinjaTrader (Control Center > Tools > Historical Data > Export) and "
        "ingest via edge.io.nt_csv, or provide one CSV day so the binary decoder "
        "can be locked bit-exact. See README 'Getting your data in'."
    )


def _validate_ohlc(df: pd.DataFrame, source: str = "", tol: float = 1e-6) -> None:
    """Assert OHLC bar invariants; raise on violation (fail loud, never silent)."""
    bad_high = (df["high"] < df[["open", "close"]].max(axis=1) - tol)
    bad_low = (df["low"] > df[["open", "close"]].min(axis=1) + tol)
    bad_hl = (df["high"] < df["low"] - tol)
    n_bad = int(bad_high.sum() + bad_low.sum() + bad_hl.sum())
    if n_bad:
        raise ValueError(f"{source}: {n_bad} OHLC-invariant violations")
    if (df["volume"] < 0).any():
        raise ValueError(f"{source}: negative volume")
