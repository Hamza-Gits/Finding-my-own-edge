"""Per-year two-sided leg gate: power (kills single-year concentration) and size
(passes a genuinely two-sided rule). This gate replaces the too-weak pooled
min(pnl_long, pnl_short) > 0 honesty check.
"""
import numpy as np
import pandas as pd

from edge.validation.leg_gate import per_year_leg_gate


def _index(years, per_year=120):
    """A DatetimeIndex with `per_year` bars in each calendar year."""
    stamps = []
    for y in years:
        stamps += list(pd.date_range(f"{y}-03-01", periods=per_year, freq="D"))
    return pd.DatetimeIndex(stamps)


def test_balanced_two_sided_passes():
    years = list(range(2019, 2027))
    idx = _index(years)
    rng = np.random.default_rng(0)
    # both legs reliably positive every year (signal >> per-year noise), no dominance
    r_long = pd.Series(rng.normal(0.0010, 0.002, len(idx)), index=idx)
    r_short = pd.Series(rng.normal(0.0010, 0.002, len(idx)), index=idx)
    r_both = r_long + r_short
    res = per_year_leg_gate(idx, r_long, r_short, r_both)
    assert res.passed, res.reason
    assert res.long_pos_years >= res.min_pos_years
    assert res.short_pos_years >= res.min_pos_years


def test_single_year_concentration_fails():
    years = list(range(2019, 2027))
    idx = _index(years)
    yr = idx.year.to_numpy()
    # long leg fine; short leg only makes money in 2022 (the drift year)
    r_long = pd.Series(0.0010, index=idx)
    r_short = pd.Series(-0.0003, index=idx)        # negative most years
    r_short[yr == 2022] = 0.02                      # huge in 2022 only
    r_both = r_long + r_short
    res = per_year_leg_gate(idx, r_long, r_short, r_both)
    assert not res.passed
    assert res.short_peak_year == 2022
    assert res.short_peak_share > 0.40
    # dropping the concentrated year should be the most fragile removal
    assert res.most_fragile_year == 2022
