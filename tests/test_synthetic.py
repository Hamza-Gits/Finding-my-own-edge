"""Tests for synthetic generators: OHLC invariants + intended autocorrelation."""
import numpy as np

from edge.io import synthetic


def test_random_walk_ohlc_invariants():
    df = synthetic.random_walk_ohlc(n=2000, seed=1)
    assert (df["high"] >= df[["open", "close"]].max(axis=1)).all()
    assert (df["low"] <= df[["open", "close"]].min(axis=1)).all()
    assert (df["volume"] > 0).all()
    assert df.index.tz is not None


def test_random_walk_returns_have_low_autocorr():
    df = synthetic.random_walk_ohlc(n=20000, seed=2)
    r = np.diff(np.log(df["close"].to_numpy()))
    ac1 = np.corrcoef(r[:-1], r[1:])[0, 1]
    assert abs(ac1) < 0.05  # essentially uncorrelated


def test_trend_ohlc_returns_have_positive_autocorr():
    df = synthetic.trend_ohlc(n=20000, seed=2, phi=0.30)
    r = np.diff(np.log(df["close"].to_numpy()))
    ac1 = np.corrcoef(r[:-1], r[1:])[0, 1]
    assert ac1 > 0.15  # planted momentum is detectable
