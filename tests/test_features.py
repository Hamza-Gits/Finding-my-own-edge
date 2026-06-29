"""Unit tests for bar aggregation and seasonality signals.

These guard the discovery inputs: a silently-wrong daily bar or an ORB that peeks
at the future would corrupt every downstream gate.
"""
import numpy as np
import pandas as pd
import pytest

from edge.io import synthetic
from edge.data.sessions import tag_sessions
from edge.data.bars import daily_rth_bars, five_min_rth_bars
from edge.features.seasonality import opening_range_breakout, gap_follow, gap_fade


@pytest.fixture
def intraday():
    # 10 trading days of 1-min bars in Central tz, OHLC-valid.
    df = synthetic.random_walk_ohlc(n=10 * 24 * 60, seed=1,
                                    start="2022-03-01", freq="1min", tz="America/Chicago")
    return tag_sessions(df)


def test_daily_rth_bars_one_per_day_and_valid(intraday):
    d = daily_rth_bars(intraday)
    # one bar per calendar day that has RTH data
    rth = intraday[intraday["session"] == "RTH"]
    assert len(d) == rth.index.normalize().nunique()
    assert (d["high"] >= d[["open", "close"]].max(axis=1) - 1e-9).all()
    assert (d["low"] <= d[["open", "close"]].min(axis=1) + 1e-9).all()
    assert (d["volume"] > 0).all()


def test_five_min_rth_bars_are_rth_only(intraday):
    f5 = five_min_rth_bars(intraday)
    t = pd.DatetimeIndex(f5.index).time
    from datetime import time as _t
    assert all(_t(8, 30) <= x <= _t(15, 0) for x in t)
    assert (f5["high"] >= f5[["open", "close"]].max(axis=1) - 1e-9).all()


def test_orb_no_lookahead_and_one_trade_per_day(intraday):
    f5 = five_min_rth_bars(intraday)
    pos = opening_range_breakout(f5, or_bars=3, direction="both")
    assert set(np.unique(pos.to_numpy())) <= {-1.0, 0.0, 1.0}
    # During the opening-range window itself the strategy must be FLAT (no peeking).
    day = pd.DatetimeIndex(f5.index).normalize()
    for d in np.unique(day.asi8)[:5]:
        sl = np.flatnonzero(day.asi8 == d)
        assert (pos.to_numpy()[sl[:3]] == 0).all()       # first 3 bars flat
        # within a day the nonzero position never flips sign (first breakout holds)
        nz = pos.to_numpy()[sl][pos.to_numpy()[sl] != 0]
        if nz.size:
            assert (nz == nz[0]).all()


def test_orb_long_only_is_nonnegative(intraday):
    f5 = five_min_rth_bars(intraday)
    pos = opening_range_breakout(f5, or_bars=1, direction="long")
    assert (pos.to_numpy() >= 0).all()


def test_gap_follow_fade_are_opposite_and_from_ohlc():
    d = synthetic.random_walk_ohlc(n=300, seed=2, freq="1D", tz="America/Chicago")
    f = gap_follow(d, threshold=0.0)
    g = gap_fade(d, threshold=0.0)
    assert (f.to_numpy() == -g.to_numpy()).all()
    # gap sign matches open vs prior close
    o = d["open"].to_numpy(); pc = np.roll(d["close"].to_numpy(), 1)
    expected = np.sign(o / pc - 1.0); expected[0] = 0.0
    assert (np.sign(f.to_numpy()) == expected).all()
