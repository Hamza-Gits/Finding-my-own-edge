"""Per-year two-sided leg gate — the honest replacement for pooled min(pnl)>0.

A symmetric long/short rule is supposed to prove it is NOT smuggled drift by showing
BOTH legs contribute. The naive check, `min(pnl_long, pnl_short) > 0` on POOLED P&L, is
too weak: a leg can be net-positive overall while booking nearly all of it in a single
trend year (e.g. the 2022 bear inflating a short leg). That is exactly the drift the
pipeline exists to reject, and the pooled check waves it through.

This gate decomposes each leg's P&L BY YEAR and requires:
  (1) each leg net-positive in at least `min_pos_frac` of the years, AND
  (2) no single year contributes more than `max_year_share` of a leg's total.

It also reports the leave-one-year-out net Sharpe of the combined rule (drift-robustness:
if dropping one year guts the Sharpe, the "edge" lived in that one regime). Apply this
BEFORE the expensive full-gate stack — it is cheap and kills 2022-concentration cleanly.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..discovery.backtest import sharpe

# 5-min RTH bars: 78 bars/day * 252 trading days.
_ANN_5MIN = float(np.sqrt(252 * 78))


@dataclass
class LegGateResult:
    passed: bool
    reason: str
    years: list[int]
    long_by_year: dict[int, float]
    short_by_year: dict[int, float]
    both_by_year: dict[int, float]
    long_pos_years: int
    short_pos_years: int
    long_peak_share: float
    short_peak_share: float
    long_peak_year: int
    short_peak_year: int
    full_sharpe: float
    loo_sharpe: dict[int, float]          # year -> Sharpe of combined rule with that year removed
    most_fragile_year: int
    most_fragile_drop: float
    min_pos_years: int = field(default=0)
    max_year_share: float = field(default=0.40)


def _leg_stats(by_year: dict[int, float]) -> tuple[int, int, float, float]:
    """(peak_year, peak_value, peak_share_of_total, n_positive_years) for one leg."""
    total = sum(by_year.values())
    peak_year = max(by_year, key=lambda k: by_year[k])
    peak_val = by_year[peak_year]
    share = peak_val / total if total != 0 else float("nan")
    n_pos = sum(1 for v in by_year.values() if v > 0)
    return peak_year, peak_val, share, n_pos


def per_year_leg_gate(index: pd.DatetimeIndex,
                      r_long: pd.Series, r_short: pd.Series, r_both: pd.Series,
                      *, min_pos_frac: float = 0.75, max_year_share: float = 0.40,
                      ann: float = _ANN_5MIN) -> LegGateResult:
    """Decompose long/short leg P&L by calendar year and apply the two-sided gate.

    `r_long`, `r_short`, `r_both` are per-bar NET return series (already cost-adjusted)
    for the long-only, short-only, and combined positions on the SAME frozen params.
    `index` is their shared DatetimeIndex (tz-aware ok).
    """
    yr = pd.DatetimeIndex(index).year.to_numpy()
    years = [int(y) for y in np.unique(yr)]

    long_by_year = {int(y): float(r_long[yr == y].sum()) for y in years}
    short_by_year = {int(y): float(r_short[yr == y].sum()) for y in years}
    both_by_year = {int(y): float(r_both[yr == y].sum()) for y in years}

    lpy, _, lshare, lpos = _leg_stats(long_by_year)
    spy, _, sshare, spos = _leg_stats(short_by_year)

    full_sharpe = sharpe(r_both) * ann
    loo = {int(y): float(sharpe(r_both[yr != y]) * ann) for y in years}
    # "fragile" = the year whose REMOVAL drops the Sharpe most (edge concentrated there)
    drops = {y: full_sharpe - s for y, s in loo.items()}
    fragile_year = max(drops, key=lambda k: drops[k])

    min_pos_years = int(np.ceil(min_pos_frac * len(years)))
    passed = (lpos >= min_pos_years and spos >= min_pos_years
              and lshare <= max_year_share and sshare <= max_year_share)
    if passed:
        reason = (f"two-sided across years: long+ in {lpos}/{len(years)}, short+ in {spos}/{len(years)}; "
                  f"peak-year shares long {lshare:.0%} / short {sshare:.0%}")
    else:
        bits = []
        if lpos < min_pos_years:
            bits.append(f"long positive in only {lpos}/{len(years)} years (<{min_pos_years})")
        if spos < min_pos_years:
            bits.append(f"short positive in only {spos}/{len(years)} years (<{min_pos_years})")
        if lshare > max_year_share:
            bits.append(f"long {lshare:.0%} concentrated in {lpy} (>{max_year_share:.0%})")
        if sshare > max_year_share:
            bits.append(f"short {sshare:.0%} concentrated in {spy} (>{max_year_share:.0%})")
        reason = "; ".join(bits)

    return LegGateResult(
        passed=passed, reason=reason, years=years,
        long_by_year=long_by_year, short_by_year=short_by_year, both_by_year=both_by_year,
        long_pos_years=lpos, short_pos_years=spos,
        long_peak_share=lshare, short_peak_share=sshare,
        long_peak_year=lpy, short_peak_year=spy,
        full_sharpe=full_sharpe, loo_sharpe=loo,
        most_fragile_year=fragile_year, most_fragile_drop=drops[fragile_year],
        min_pos_years=min_pos_years, max_year_share=max_year_share,
    )


def format_leg_gate(res: LegGateResult) -> str:
    """Human-readable block for console + markdown report."""
    lines = [f"{'year':>6} | {'long':>10} {'short':>10} | {'both':>10}", "-" * 46]
    for y in res.years:
        lines.append(f"{y:>6} | {res.long_by_year[y]:>+10.4f} {res.short_by_year[y]:>+10.4f} | "
                     f"{res.both_by_year[y]:>+10.4f}")
    lines.append("-" * 46)
    lines.append(f"{'TOTAL':>6} | {sum(res.long_by_year.values()):>+10.4f} "
                 f"{sum(res.short_by_year.values()):>+10.4f} | {sum(res.both_by_year.values()):>+10.4f}")
    lines.append("")
    lines.append(f"long+ {res.long_pos_years}/{len(res.years)} yrs (peak {res.long_peak_year} "
                 f"{res.long_peak_share:.0%}); short+ {res.short_pos_years}/{len(res.years)} yrs "
                 f"(peak {res.short_peak_year} {res.short_peak_share:.0%})")
    lines.append(f"full net Sharpe {res.full_sharpe:+.2f}; most fragile = drop {res.most_fragile_year} "
                 f"-> Sharpe falls {res.most_fragile_drop:+.2f}")
    lines.append(f"PER-YEAR LEG GATE: {'PASS' if res.passed else 'FAIL'} ({res.reason})")
    return "\n".join(lines)
