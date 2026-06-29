"""Synthetic OHLCV generators with controllable structure.

Used for (a) unit tests that need deterministic data, and (b) the gate
power/size validation the plan requires:
  * `random_walk_ohlc`   -> a WORTHLESS series (no edge): a correct gate must
                            NOT flag an edge here (controls false-positive rate).
  * `trend_ohlc`         -> a series with a PLANTED edge (autocorrelated returns):
                            a correct gate SHOULD detect it (confirms power).
All generators are seeded for reproducibility.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _ohlc_from_close(
    ts: pd.DatetimeIndex, close: np.ndarray, rng: np.random.Generator,
    intrabar: float, vol_base: float,
) -> pd.DataFrame:
    close = np.asarray(close, dtype=float)
    open_ = np.empty_like(close)
    open_[0] = close[0]
    open_[1:] = close[:-1]
    wig = np.abs(rng.normal(0.0, intrabar, size=(close.size, 2)))
    high = np.maximum(open_, close) + wig[:, 0]
    low = np.minimum(open_, close) - wig[:, 1]
    volume = rng.poisson(vol_base, size=close.size).astype(float) + 1.0
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=ts,
    )


def _index(n: int, start: str, freq: str, tz: str) -> pd.DatetimeIndex:
    idx = pd.date_range(start=start, periods=n, freq=freq, tz=tz)
    idx.name = "ts"
    return idx


def random_walk_ohlc(
    n: int = 5000, *, seed: int = 7, start_price: float = 20000.0,
    sigma: float = 2.0, freq: str = "1min", start: str = "2022-01-03 08:30",
    tz: str = "America/Chicago", intrabar: float = 1.0, vol_base: float = 100.0,
) -> pd.DataFrame:
    """A driftless Gaussian random walk: NO exploitable structure."""
    rng = np.random.default_rng(seed)
    steps = rng.normal(0.0, sigma, size=n)
    close = start_price + np.cumsum(steps)
    return _ohlc_from_close(_index(n, start, freq, tz), close, rng, intrabar, vol_base)


def trend_ohlc(
    n: int = 5000, *, seed: int = 7, start_price: float = 20000.0,
    sigma: float = 2.0, phi: float = 0.30, freq: str = "1min",
    start: str = "2022-01-03 08:30", tz: str = "America/Chicago",
    intrabar: float = 1.0, vol_base: float = 100.0,
) -> pd.DataFrame:
    """AR(1) returns with positive autocorrelation `phi` => a PLANTED momentum edge.

    r_t = phi * r_{t-1} + eps_t. A momentum rule has genuine expectancy here, so a
    correctly-built permutation gate should reject the worthless-strategy null.
    """
    rng = np.random.default_rng(seed)
    eps = rng.normal(0.0, sigma, size=n)
    r = np.empty(n)
    r[0] = eps[0]
    for i in range(1, n):
        r[i] = phi * r[i - 1] + eps[i]
    close = start_price + np.cumsum(r)
    return _ohlc_from_close(_index(n, start, freq, tz), close, rng, intrabar, vol_base)
