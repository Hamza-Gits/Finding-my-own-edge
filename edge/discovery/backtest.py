"""Bar-level backtest and objective functions.

Objectives are computed on BAR-LEVEL strategy returns (not trade-level): bar
returns are far more numerous and statistically stable, give the permutation
machinery the granular pairing it needs, and avoid small-sample trade noise
(Masters). A position decided using information up to bar t's close earns bar
t+1's return; position changes pay a per-turn cost (so all results are net).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def strategy_returns(df: pd.DataFrame, positions: pd.Series,
                     cost_per_turn: float = 0.0) -> pd.Series:
    """Net per-bar log returns of holding `positions` (in contracts, signed).

    positions[t] is the target position chosen at bar t's close; it is held over
    bar t+1, so we shift by one (no look-ahead). Turnover |Δposition| pays
    `cost_per_turn` (a per-side cost as a fraction of notional).
    """
    logret = np.log(df["close"]).diff()
    pos = positions.shift(1).fillna(0.0)
    gross = pos * logret
    turn = positions.diff().abs().fillna(positions.abs())
    net = gross - turn * cost_per_turn
    return net.fillna(0.0)


def profit_factor(returns: pd.Series | np.ndarray) -> float:
    """Sum of gains / |sum of losses|. Robust to all-positive (returns large finite)."""
    r = np.asarray(returns, dtype=float)
    gains = r[r > 0].sum()
    losses = -r[r < 0].sum()
    if losses <= 0:
        return float("inf") if gains > 0 else 0.0
    return float(gains / losses)


def sharpe(returns: pd.Series | np.ndarray, ddof: int = 1) -> float:
    """Non-annualized Sharpe of the per-bar returns."""
    r = np.asarray(returns, dtype=float)
    sd = r.std(ddof=ddof)
    return 0.0 if sd == 0 else float(r.mean() / sd)


def bounded_profit_factor(returns, cap: float = 1e6) -> float:
    """Profit factor capped so optimization never chases an infinite spike."""
    pf = profit_factor(returns)
    return min(pf, cap)
