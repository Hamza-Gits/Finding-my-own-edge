"""Continuous-contract construction from quarterly futures via volume rolls.

Quarterly index futures (MNQ/MES: H,M,U,Z = Mar,Jun,Sep,Dec) each trade for a
few months then expire. To get one long series we must stitch contracts, and the
HONEST way matters:

  * Roll on VOLUME CROSSOVER (the spec's rule): switch to the deferred contract
    the first day its volume exceeds the front month's. This tracks where the
    liquidity actually is, instead of a fixed calendar offset.
  * Keep BOTH series:
      - `unadj`  : raw stitched prices. Has visible jumps at each roll. Use this
                   for anything that reasons about ABSOLUTE price levels.
      - `adj`    : back-adjusted (Panama, additive). Roll gaps removed so returns
                   and point-ranges are continuous across rolls. Use this for
                   indicators, returns, ORB ranges, backtests.
    Additive (not ratio) adjustment preserves POINT distances — important because
    micro futures P&L is linear in points, and an ORB range is a point range.

A back-adjusted series is NOT a tradeable price (its past levels are synthetic);
that is exactly why we retain the unadjusted series alongside it. Every roll is
logged to a schedule so the stitching is fully auditable.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

# Quarterly delivery months for the equity-index complex.
_QUARTERLY = {3, 6, 9, 12}
_LABEL_RE = re.compile(r"^([A-Z0-9]+)\s+(\d{2})-(\d{2})$")


@dataclass(frozen=True)
class Contract:
    symbol: str          # e.g. "MNQ"
    month: int           # delivery month 1..12
    year: int            # full year, e.g. 2026
    label: str           # original "MNQ 09-26"

    @property
    def expiry(self) -> pd.Timestamp:
        """Approximate expiry = 3rd Friday of the delivery month (Central date)."""
        return third_friday(self.year, self.month)

    def __lt__(self, other: "Contract") -> bool:
        return (self.year, self.month) < (other.year, other.month)


def parse_contract(label: str) -> Contract:
    """'MNQ 09-26' -> Contract(symbol='MNQ', month=9, year=2026)."""
    m = _LABEL_RE.match(label.strip())
    if not m:
        raise ValueError(f"unrecognized contract label: {label!r}")
    sym, mm, yy = m.group(1), int(m.group(2)), int(m.group(3))
    year = 2000 + yy
    return Contract(symbol=sym, month=mm, year=year, label=label.strip())


def third_friday(year: int, month: int) -> pd.Timestamp:
    """3rd Friday of a month (CME quarterly futures expiry proxy)."""
    first = pd.Timestamp(year=year, month=month, day=1)
    # weekday(): Mon=0..Fri=4. Days until first Friday:
    offset = (4 - first.dayofweek) % 7
    return first + pd.Timedelta(days=offset + 14)


def _daily_volume(df: pd.DataFrame) -> pd.Series:
    """Total volume per calendar day (index = naive date)."""
    by_day = df["volume"].groupby(df.index.normalize().tz_localize(None)).sum()
    by_day.index = pd.DatetimeIndex(by_day.index)
    return by_day


def _roll_gap(old: pd.DataFrame, new: pd.DataFrame, roll_date: pd.Timestamp) -> float:
    """Additive price gap (new - old) at the roll, from overlapping bars.

    Uses the median close difference over timestamps both contracts share within
    +/-1 day of the roll. Falls back to last-old/first-new close if no overlap.
    """
    lo = roll_date - pd.Timedelta(days=1)
    hi = roll_date + pd.Timedelta(days=1)
    o_naive = old.index.tz_localize(None)   # roll_date is naive; compare on naive
    n_naive = new.index.tz_localize(None)
    o = old.loc[(o_naive >= lo) & (o_naive <= hi), "close"]
    n = new.loc[(n_naive >= lo) & (n_naive <= hi), "close"]
    common = o.index.intersection(n.index)  # both tz-aware Central
    if len(common) >= 5:
        return float((n.loc[common] - o.loc[common]).median())
    # Fallback: compare nearest available closes across the boundary.
    old_before = old.loc[o_naive < roll_date, "close"]
    new_after = new.loc[n_naive >= roll_date, "close"]
    if len(old_before) and len(new_after):
        return float(new_after.iloc[0] - old_before.iloc[-1])
    return 0.0


@dataclass
class Continuous:
    adj: pd.DataFrame          # back-adjusted OHLCV (use for returns/indicators)
    unadj: pd.DataFrame        # raw stitched OHLCV (use for absolute-price logic)
    schedule: pd.DataFrame     # one row per roll: from,to,roll_date,gap,active_start


def build_continuous(frames: dict[str, pd.DataFrame]) -> Continuous:
    """Stitch per-contract OHLCV frames into continuous adj + unadj series.

    `frames` maps contract label -> tz-aware OHLCV DataFrame (exchange tz).
    Roll = first date the next contract out-volumes the current one, capped at
    the current contract's expiry. Anti-whipsaw: rolls are monotone forward.
    """
    contracts = sorted((parse_contract(lbl) for lbl in frames), key=lambda c: (c.year, c.month))
    if not contracts:
        raise ValueError("no contracts to stitch")
    if len(contracts) == 1:
        df = frames[contracts[0].label].sort_index()
        sched = pd.DataFrame(columns=["from", "to", "roll_date", "gap", "active_start"])
        return Continuous(adj=df.copy(), unadj=df.copy(), schedule=sched)

    dvol = {c.label: _daily_volume(frames[c.label]) for c in contracts}

    # --- determine roll date for each adjacent pair -----------------------
    active_start = {contracts[0].label: frames[contracts[0].label].index.min().normalize().tz_localize(None)}
    roll_rows = []
    for cur, nxt in zip(contracts, contracts[1:]):
        v_cur, v_nxt = dvol[cur.label], dvol[nxt.label]
        days = v_cur.index.union(v_nxt.index).sort_values()
        start = active_start[cur.label]
        cross = None
        for d in days:
            if d <= start:
                continue
            if d > cur.expiry + pd.Timedelta(days=1):
                break
            vc = float(v_cur.get(d, 0.0))
            vn = float(v_nxt.get(d, 0.0))
            if vn > vc and vn > 0:
                cross = d
                break
        if cross is None:                       # no crossover -> roll at expiry/last day
            cap = min(cur.expiry, v_cur.index.max() if len(v_cur) else cur.expiry)
            cross = cap
        gap = _roll_gap(frames[cur.label], frames[nxt.label], cross)
        roll_rows.append({"from": cur.label, "to": nxt.label, "roll_date": cross, "gap": gap})
        active_start[nxt.label] = cross

    schedule = pd.DataFrame(roll_rows)

    # --- assemble unadjusted by slicing each contract's active window -----
    bounds = {}  # label -> (start, end) naive dates; end exclusive (None for last)
    labels = [c.label for c in contracts]
    starts = [active_start[lbl] for lbl in labels]
    for i, lbl in enumerate(labels):
        end = starts[i + 1] if i + 1 < len(labels) else None
        bounds[lbl] = (starts[i], end)

    pieces = []
    for lbl in labels:
        df = frames[lbl]
        naive = df.index.tz_localize(None)
        s, e = bounds[lbl]
        mask = naive >= s
        if e is not None:
            mask &= naive < e
        pieces.append(df.loc[mask])
    unadj = pd.concat(pieces).sort_index()
    unadj = unadj[~unadj.index.duplicated(keep="last")]

    # --- back-adjust (additive Panama): offset[c_i] = sum of gaps at rolls j>=i
    gaps = schedule["gap"].to_numpy() if len(schedule) else np.array([])
    offset = {labels[-1]: 0.0}
    cum = 0.0
    for i in range(len(labels) - 2, -1, -1):
        cum += float(gaps[i])
        offset[labels[i]] = cum

    adj_pieces = []
    for lbl in labels:
        df = frames[lbl]
        naive = df.index.tz_localize(None)
        s, e = bounds[lbl]
        mask = naive >= s
        if e is not None:
            mask &= naive < e
        piece = df.loc[mask].copy()
        off = offset[lbl]
        for col in ("open", "high", "low", "close"):
            piece[col] = piece[col] + off
        adj_pieces.append(piece)
    adj = pd.concat(adj_pieces).sort_index()
    adj = adj[~adj.index.duplicated(keep="last")]

    schedule = schedule.assign(active_start=[active_start[l] for l in schedule["to"]])
    return Continuous(adj=adj, unadj=unadj, schedule=schedule)
