"""Compression-gated opening-range breakout (research hypothesis H1).

Mechanism: daily volatility is autocorrelated and mean-reverting, so a QUIET
overnight (small ETH range relative to the 20-day ATR of the RTH range) signals a
coiled balance with resting liquidity at the edges. We ARM the opening-range
breakout only on such compressed days; direction is chosen intraday by whichever
side breaks first, so the rule is symmetric long/short and cannot be bull-market
drift. Flat overnight.

The compression series is computed ONCE on real data and held fixed; the intraday
breakout is what the permutation test attacks. Positions are forced flat on each
day's last bar so the bar-level backtest never leaks the overnight gap.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..data.bars import daily_rth_bars
from ..data.sessions import tag_sessions
from .seasonality import _day_bounds


def daily_compression(df_1min: pd.DataFrame, atr_window: int = 20) -> pd.Series:
    """Per session-date compression ratio = overnight_range / ATR20(RTH range).

    Overnight bars (ETH from the prior RTH close to today's open) are mapped to
    the session date they lead into (so Friday-evening + Sunday Globex map to
    Monday). ATR20 is the trailing 20-day mean of the RTH daily range, shifted one
    day (no lookahead).
    """
    if "session" not in df_1min.columns:
        df_1min = tag_sessions(df_1min)

    rth_daily = daily_rth_bars(df_1min)
    rng = (rth_daily["high"] - rth_daily["low"])
    atr20 = rng.rolling(atr_window).mean().shift(1)          # uses D-20..D-1 only
    rth_norm = rth_daily.index                               # tz-aware session midnights
    rth_int = rth_norm.asi8

    eth = df_1min[df_1min["session"] == "ETH"]
    idx = pd.DatetimeIndex(eth.index)
    bd = idx.normalize().asi8
    from datetime import time as _t
    is_evening = np.array([x >= _t(15, 0) for x in idx.time])   # after RTH close
    # session date: evening bars -> next RTH date strictly after; early/pre-open
    # bars -> first RTH date on/after their own calendar date.
    sess_pos = np.where(is_evening,
                        np.searchsorted(rth_int, bd, side="right"),
                        np.searchsorted(rth_int, bd, side="left"))
    valid = sess_pos < len(rth_int)
    eth_df = pd.DataFrame({"high": eth["high"].to_numpy()[valid],
                           "low": eth["low"].to_numpy()[valid],
                           "pos": sess_pos[valid]})
    g = eth_df.groupby("pos")
    on_range = g["high"].max() - g["low"].min()
    on_range.index = rth_norm[on_range.index.to_numpy()]    # map back to tz-aware dates

    comp = (on_range / atr20.reindex(on_range.index)).dropna()
    comp.name = "compression"
    return comp


def trailing_rank(series: pd.Series, min_periods: int = 60) -> pd.Series:
    """Lookahead-free percentile rank: rank[i] = fraction of PRIOR values <= value[i].

    Turns the raw compression ratio into a 'bottom q' arming signal — arm if
    rank <= q means 'this day is in the quietest q fraction of days seen so far',
    which is the tercile/quantile the hypothesis actually intends (and keeps the
    armed-day count adequate, unlike an absolute ratio cutoff).
    """
    v = series.to_numpy(dtype=float)
    out = np.full(v.size, np.nan)
    for i in range(min_periods, v.size):
        prior = v[:i]
        out[i] = float(np.mean(prior <= v[i]))
    return pd.Series(out, index=series.index, name="comp_rank")


def make_compression_orb(compression: pd.Series, tick_size: float):
    """Build a position signal_fn(df5, **params) bound to a fixed compression map.

    The returned function works on any frame sharing the original index (incl. a
    bar-permutation of it), so it slots straight into the MCPT/gate machinery.
    """
    comp_map = {int(k.normalize().value): float(v) for k, v in compression.items()}
    # Cache the per-frame day structure: MCPT calls the signal ~thousands of times
    # on frames that share ONE index (permutations preserve it), so we compute the
    # day bounds/keys once per distinct frame shape instead of every call.
    _cache: dict = {}

    def _day_structure(df):
        idx = pd.DatetimeIndex(df.index)
        key = (len(idx), int(idx.asi8[0]), int(idx.asi8[-1]))
        hit = _cache.get(key)
        if hit is None:
            hit = (_day_bounds(idx), idx.normalize().asi8)
            _cache[key] = hit
        return hit

    def signal(df: pd.DataFrame, or_bars: int = 3, buffer_ticks: float = 1.0,
               compression_q: float = 0.3, stop_mult: float = 1.0,
               direction: str = "both") -> pd.Series:
        n = len(df)
        pos = np.zeros(n)
        h = df["high"].to_numpy(); l = df["low"].to_numpy(); c = df["close"].to_numpy()
        bounds, day_keys = _day_structure(df)
        buf = buffer_ticks * tick_size
        allow_long = direction in ("both", "long")
        allow_short = direction in ("both", "short")

        for k in range(len(bounds) - 1):
            s, e = bounds[k], bounds[k + 1]
            if e - s <= or_bars + 1:
                continue
            # armed iff compression rank <= q; NaN/missing -> not armed (skip).
            if not (comp_map.get(int(day_keys[s]), np.inf) <= compression_q):
                continue
            or_high = h[s:s + or_bars].max()
            or_low = l[s:s + or_bars].min()
            width = or_high - or_low
            if width <= 0:
                continue
            last = e - 1                                   # force flat on last bar
            cc = c[s + or_bars:last]
            up = np.flatnonzero(cc > or_high + buf)
            dn = np.flatnonzero(cc < or_low - buf)
            i_up = up[0] if up.size else np.inf
            i_dn = dn[0] if dn.size else np.inf
            if i_up < i_dn and allow_long:
                ent = s + or_bars + int(i_up)
                stop = or_high - stop_mult * width
                after = c[ent:last]
                hit = np.flatnonzero(after <= stop)
                exit_at = ent + int(hit[0]) if hit.size else last
                pos[ent:exit_at] = 1.0
            elif i_dn < i_up and allow_short:
                ent = s + or_bars + int(i_dn)
                stop = or_low + stop_mult * width
                after = c[ent:last]
                hit = np.flatnonzero(after >= stop)
                exit_at = ent + int(hit[0]) if hit.size else last
                pos[ent:exit_at] = -1.0
        return pd.Series(pos, index=df.index)

    return signal
