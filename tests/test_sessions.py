"""Tests for timezone alignment (DST) and RTH/ETH tagging."""
import pandas as pd

from edge.data import sessions


def _utc_frame(times_utc, price=20000.0):
    idx = pd.DatetimeIndex(times_utc).tz_localize("UTC")
    return pd.DataFrame(
        {"open": price, "high": price + 1, "low": price - 1, "close": price,
         "volume": 1.0},
        index=idx,
    )


def test_utc_to_central_dst_summer_and_winter():
    # In summer Central is UTC-5 (CDT); in winter UTC-6 (CST).
    df = _utc_frame(["2022-07-01 18:30", "2022-01-03 18:30"])
    out = sessions.to_exchange_tz(df)
    times = {t.date().isoformat(): t for t in out.index}
    assert times["2022-07-01"].hour == 13   # 18:30 UTC - 5h = 13:30 CDT
    assert times["2022-01-03"].hour == 12    # 18:30 UTC - 6h = 12:30 CST


def test_naive_localization_requires_source_tz():
    df = pd.DataFrame(
        {"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0},
        index=pd.DatetimeIndex(["2022-07-01 09:00"]),
    )
    try:
        sessions.to_exchange_tz(df)
    except ValueError as e:
        assert "source_tz" in str(e)
    else:
        raise AssertionError("expected ValueError for tz-naive index")


def test_rth_eth_tagging():
    # 14:00 UTC = 09:00 CDT (RTH) in summer; 02:00 UTC = 21:00 prev day CST-ish (ETH).
    df = _utc_frame(["2022-07-01 14:00", "2022-07-01 02:00", "2022-07-02 15:00"])
    out = sessions.tag_sessions(sessions.to_exchange_tz(df))
    s = out["session"].astype(str).tolist()
    # 14:00 UTC -> 09:00 CDT weekday -> RTH
    assert "RTH" in s
    # 02:00 UTC -> 21:00 prev-day CDT -> ETH
    assert "ETH" in s
