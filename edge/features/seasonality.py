"""Intraday seasonality signals — opening-range breakout and calendar effects.

These are SIMPLE, low-parameter rules by design (the spec prefers stable plateaus
over complex models). Each returns a target-position Series in {-1,0,+1} aligned
to the input frame, so they plug straight into the backtest/gate machinery.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _day_bounds(index: pd.DatetimeIndex) -> np.ndarray:
    """Start offsets of each calendar day in a sorted intraday index (+ final n)."""
    day = index.normalize().asi8
    change = np.flatnonzero(np.diff(day) != 0) + 1
    return np.concatenate([[0], change, [len(index)]])


def opening_range_breakout(df: pd.DataFrame, or_bars: int = 1,
                           direction: str = "both") -> pd.Series:
    """First breakout of the day's opening range, held to the session close.

    The opening range is the high/low of the first `or_bars` bars of each RTH
    day. The first bar to CLOSE beyond it sets the position (long above, short
    below) for the remainder of that day; flat overnight. `direction` restricts
    to 'long', 'short', or 'both'. One trade per day -> low turnover.
    """
    n = len(df)
    pos = np.zeros(n)
    h = df["high"].to_numpy(); l = df["low"].to_numpy(); c = df["close"].to_numpy()
    bounds = _day_bounds(pd.DatetimeIndex(df.index))
    allow_long = direction in ("both", "long")
    allow_short = direction in ("both", "short")

    for k in range(len(bounds) - 1):
        s, e = bounds[k], bounds[k + 1]
        if e - s <= or_bars:
            continue
        or_high = h[s:s + or_bars].max()
        or_low = l[s:s + or_bars].min()
        cc = c[s + or_bars:e]
        up = np.flatnonzero(cc > or_high)
        dn = np.flatnonzero(cc < or_low)
        i_up = up[0] if up.size else np.inf
        i_dn = dn[0] if dn.size else np.inf
        if i_up < i_dn and allow_long:
            pos[s + or_bars + int(i_up):e] = 1.0
        elif i_dn < i_up and allow_short:
            pos[s + or_bars + int(i_dn):e] = -1.0
    return pd.Series(pos, index=df.index)


def day_of_week_long(df: pd.DataFrame, dow: int = 0) -> pd.Series:
    """Hold long only on a given weekday (0=Mon .. 4=Fri). A calendar-effect probe."""
    mask = (pd.DatetimeIndex(df.index).dayofweek == dow).astype(float)
    return pd.Series(mask, index=df.index)


def gap_follow(df: pd.DataFrame, threshold: float = 0.0) -> pd.Series:
    """Go WITH the overnight gap. Computes gap = open/prev_close - 1 from OHLC, so
    it is valid on permuted frames too (the permutation only carries OHLC)."""
    o = df["open"].to_numpy()
    prev_c = np.roll(df["close"].to_numpy(), 1)
    g = o / prev_c - 1.0
    g[0] = 0.0
    pos = np.where(g > threshold, 1.0, np.where(g < -threshold, -1.0, 0.0))
    return pd.Series(pos, index=df.index)


def gap_fade(df: pd.DataFrame, threshold: float = 0.0) -> pd.Series:
    """Fade the overnight gap (mean-reversion of the open back toward prior close)."""
    return -gap_follow(df, threshold)
