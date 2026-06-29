"""Timezone alignment and RTH/ETH session tagging for CME index futures.

All analysis happens in the exchange timezone (America/Chicago = CME Central),
which `zoneinfo` handles across CST/CDT DST transitions automatically. Raw
NinjaTrader db .ncd timestamps are UTC; CSV exports may be in any tz (pass it in).
"""
from __future__ import annotations

from datetime import time

import pandas as pd

EXCHANGE_TZ = "America/Chicago"
# Regular Trading Hours for the equity-index complex, Central time.
RTH_START = time(8, 30)
RTH_END = time(15, 0)


def to_exchange_tz(df: pd.DataFrame, source_tz: str | None = None,
                   exchange_tz: str = EXCHANGE_TZ) -> pd.DataFrame:
    """Return `df` with its index expressed in the exchange timezone.

    If the index is tz-naive, it is first localized to `source_tz` (required).
    DST is handled by the tz database; ambiguous/nonexistent wall-times (only an
    issue for naive localization) are shifted forward rather than dropped.
    """
    out = df.copy()
    idx = pd.DatetimeIndex(out.index)
    if idx.tz is None:
        if source_tz is None:
            raise ValueError("tz-naive index requires source_tz to localize")
        idx = idx.tz_localize(source_tz, ambiguous="NaT", nonexistent="shift_forward")
        # Drop any ambiguous-time rows that could not be localized.
        out = out[~idx.isna()]
        idx = idx[~idx.isna()]
    out.index = idx.tz_convert(exchange_tz)
    out.index.name = "ts"
    return out.sort_index()


def tag_sessions(df: pd.DataFrame, rth_start: time = RTH_START,
                 rth_end: time = RTH_END) -> pd.DataFrame:
    """Add a 'session' column ∈ {'RTH','ETH'} based on exchange-local wall time.

    Requires a tz-aware index already in the exchange tz. RTH is [08:30, 15:00)
    on weekdays; everything else (including weekend Globex) is ETH.
    """
    idx = pd.DatetimeIndex(df.index)
    if idx.tz is None:
        raise ValueError("tag_sessions requires a tz-aware (exchange-tz) index")
    t = idx.time
    is_weekday = idx.dayofweek < 5
    in_rth = pd.Series(
        [(rth_start <= x < rth_end) for x in t], index=df.index
    ) & pd.Series(is_weekday, index=df.index)
    out = df.copy()
    out["session"] = pd.Categorical(
        ["RTH" if v else "ETH" for v in in_rth], categories=["RTH", "ETH"]
    )
    return out


def rth_only(df: pd.DataFrame) -> pd.DataFrame:
    """Convenience: subset to RTH bars (tags first if needed)."""
    if "session" not in df.columns:
        df = tag_sessions(df)
    return df[df["session"] == "RTH"]
