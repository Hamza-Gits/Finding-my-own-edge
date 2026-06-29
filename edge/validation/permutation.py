"""Bar-permutation generator (Timothy Masters' method).

Algorithm
---------
Work in log space. Decompose each bar (from `start_index+1` onward) into:
    gap   = log(open_t)  - log(close_{t-1})    # overnight / inter-bar gap
    r_h   = log(high_t)  - log(open_t)          # open -> high
    r_l   = log(low_t)   - log(open_t)          # open -> low
    r_c   = log(close_t) - log(open_t)          # open -> close
The intrabar triple (r_h, r_l, r_c) is permuted **as a unit** (each bar's
internal geometry is preserved, so reconstructed OHLC always satisfy
high >= max(open,close) and low <= min(open,close)); the **gaps are permuted
separately**. The first `start_index+1` bars are kept as an anchor. The synthetic
path is rebuilt by cumulatively summing the shuffled log components and
exponentiating.

This preserves the marginal distribution of every relative move (it is a
reshuffle) — and hence volatility, skew and kurtosis of those moves — while
DESTROYING trend, serial correlation, regime cycles and any repeating structure.
Re-running the full strategy (including its optimization) on many permutations
yields p = fraction of permutations whose metric >= the real metric: the
probability a worthless strategy + the same search could match the real result.

KNOWN FLAWS (state plainly; cross-check with block bootstrap, never rely alone):
  * Permutation destroys volatility clustering and long memory, so it is
    ANTI-CONSERVATIVE for strategies that genuinely exploit vol persistence and
    conservative for others.
  * It assumes non-overlapping trades.
  * Strongly asymmetric raw-return distributions can distort it (centering the
    raw returns removes long/short bias — Masters).
For multi-market spreads, pass several series to `permute_ohlc_multi` so the
SAME permutation is applied to all, preserving cross-sectional co-movement.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

_OHLC = ["open", "high", "low", "close"]


def _decompose(log_df: pd.DataFrame, start: int):
    o = log_df["open"].to_numpy()
    h = log_df["high"].to_numpy()
    l = log_df["low"].to_numpy()
    c = log_df["close"].to_numpy()
    prev_c = np.roll(c, 1)
    gap = (o - prev_c)[start + 1:]
    r_h = (h - o)[start + 1:]
    r_l = (l - o)[start + 1:]
    r_c = (c - o)[start + 1:]
    return gap, r_h, r_l, r_c


def _reconstruct(log_df: pd.DataFrame, start: int, gap, r_h, r_l, r_c) -> pd.DataFrame:
    """Rebuild a log-OHLC frame from shuffled components — fully vectorized.

    Each reconstructed close moves by (gap + r_c), so the close path is just a
    cumulative sum of those increments off the anchor close; the open of bar i is
    the previous close plus that bar's gap, and high/low hang off the open. This
    is identical to the obvious per-bar loop but ~100x faster (the MCPT calls it
    thousands of times), which is what makes 1,000-permutation tests on real
    intraday data tractable.
    """
    n = len(log_df)
    out = np.empty((n, 4))
    o0 = log_df["open"].to_numpy()
    h0 = log_df["high"].to_numpy()
    l0 = log_df["low"].to_numpy()
    c0 = log_df["close"].to_numpy()
    # Keep anchor bars (0..start) verbatim.
    out[: start + 1, 0] = o0[: start + 1]
    out[: start + 1, 1] = h0[: start + 1]
    out[: start + 1, 2] = l0[: start + 1]
    out[: start + 1, 3] = c0[: start + 1]

    anchor_close = c0[start]
    cc = gap + r_c                                   # per-bar log close-to-close change
    close_seg = anchor_close + np.cumsum(cc)         # reconstructed closes
    prev_close = np.empty_like(close_seg)
    prev_close[0] = anchor_close
    prev_close[1:] = close_seg[:-1]
    open_seg = prev_close + gap
    out[start + 1:, 0] = open_seg
    out[start + 1:, 1] = open_seg + r_h
    out[start + 1:, 2] = open_seg + r_l
    out[start + 1:, 3] = close_seg

    return pd.DataFrame(np.exp(out), columns=_OHLC, index=log_df.index)


def permute_ohlc(df: pd.DataFrame, start_index: int = 0,
                 rng: np.random.Generator | None = None) -> pd.DataFrame:
    """Return one bar-permuted synthetic OHLC frame (volume carried unchanged)."""
    rng = rng or np.random.default_rng()
    log_df = np.log(df[_OHLC])
    gap, r_h, r_l, r_c = _decompose(log_df, start_index)
    m = gap.size
    p_intra = rng.permutation(m)
    p_gap = rng.permutation(m)
    out = _reconstruct(log_df, start_index, gap[p_gap], r_h[p_intra],
                       r_l[p_intra], r_c[p_intra])
    if "volume" in df.columns:
        out["volume"] = df["volume"].to_numpy()
    return out


def permute_ohlc_multi(dfs: list[pd.DataFrame], start_index: int = 0,
                       rng: np.random.Generator | None = None) -> list[pd.DataFrame]:
    """Permute several aligned series with a SHARED permutation.

    Cross-sectional structure (e.g. ES vs NQ co-movement) is preserved because
    every series is reshuffled identically; only the time ordering is destroyed.
    Use this when testing spread / relative-strength edges.
    """
    rng = rng or np.random.default_rng()
    logs = [np.log(d[_OHLC]) for d in dfs]
    m = len(dfs[0]) - start_index - 1
    p_intra = rng.permutation(m)
    p_gap = rng.permutation(m)
    out = []
    for d, log_df in zip(dfs, logs):
        gap, r_h, r_l, r_c = _decompose(log_df, start_index)
        rec = _reconstruct(log_df, start_index, gap[p_gap], r_h[p_intra],
                           r_l[p_intra], r_c[p_intra])
        if "volume" in d.columns:
            rec["volume"] = d["volume"].to_numpy()
        out.append(rec)
    return out
