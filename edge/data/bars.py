"""Bar aggregation: 1-min -> daily-RTH and 5-min-RTH frames.

Discovery candidates trade at their natural decision frequency, not at 1-min:
  * DAILY-RTH bars (~1 per trading day) for momentum / mean-reversion / gap /
    day-of-week — small enough that a 1,000-permutation MCPT is trivial.
  * 5-min-RTH bars for the opening-range breakout, which needs intraday shape.

RTH = [08:30, 15:00) Central. The daily 'date' is the Central calendar date of
the session. Overnight (ETH) is excluded from these frames but available for an
explicit gap feature (prior RTH close -> today RTH open).
"""
from __future__ import annotations

import pandas as pd

from .sessions import tag_sessions

_AGG = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}


def _rth(df: pd.DataFrame) -> pd.DataFrame:
    if "session" not in df.columns:
        df = tag_sessions(df)
    return df[df["session"] == "RTH"]


def daily_rth_bars(df_1min: pd.DataFrame) -> pd.DataFrame:
    """One OHLCV bar per trading day from RTH 1-min bars (index = session date)."""
    rth = _rth(df_1min)
    day = rth.index.normalize()
    g = rth.groupby(day)
    out = pd.DataFrame({
        "open": g["open"].first(), "high": g["high"].max(),
        "low": g["low"].min(), "close": g["close"].last(),
        "volume": g["volume"].sum(),
    })
    out.index.name = "ts"
    return out.sort_index()


def five_min_rth_bars(df_1min: pd.DataFrame) -> pd.DataFrame:
    """5-minute RTH bars (label = bar close time), empty bins dropped."""
    rth = _rth(df_1min)[["open", "high", "low", "close", "volume"]]
    out = rth.resample("5min", label="right", closed="right").agg(_AGG)
    out = out.dropna(subset=["open", "close"])
    # keep only bars that fall in RTH wall-time (resample can emit boundary bins)
    out = tag_sessions(out)
    out = out[out["session"] == "RTH"].drop(columns="session")
    out.index.name = "ts"
    return out.sort_index()


def daily_gap_frame(df_1min: pd.DataFrame) -> pd.DataFrame:
    """Daily RTH bars plus the overnight gap = RTH open / prior RTH close - 1."""
    d = daily_rth_bars(df_1min)
    d["prev_close"] = d["close"].shift(1)
    d["gap"] = d["open"] / d["prev_close"] - 1.0
    return d.dropna(subset=["prev_close"])
