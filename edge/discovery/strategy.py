"""Strategy abstraction + grid optimizer.

A Strategy maps an OHLC frame and parameters to a per-bar target position series.
Strategies are deliberately SIMPLE and LOW-PARAMETER (the spec prefers stable
parameter plateaus over complex models). The optimizer scans a parameter grid;
crucially, the WHOLE optimization is re-run inside each permutation so the MCPT
p-value measures data-mining bias, not just one fitted curve.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from .backtest import strategy_returns, bounded_profit_factor

# A signal function: (df, **params) -> position Series in {-1,0,+1} (or sized).
SignalFn = Callable[..., pd.Series]


@dataclass
class Strategy:
    name: str
    signal_fn: SignalFn
    param_grid: dict[str, list]
    cost_per_turn: float = 0.0
    objective: Callable[[pd.Series], float] = bounded_profit_factor

    def positions(self, df: pd.DataFrame, **params) -> pd.Series:
        return self.signal_fn(df, **params)

    def returns(self, df: pd.DataFrame, **params) -> pd.Series:
        return strategy_returns(df, self.positions(df, **params), self.cost_per_turn)

    def evaluate(self, df: pd.DataFrame, **params) -> float:
        return self.objective(self.returns(df, **params))

    def grid(self) -> list[dict]:
        keys = list(self.param_grid)
        return [dict(zip(keys, combo))
                for combo in itertools.product(*(self.param_grid[k] for k in keys))]


@dataclass
class OptResult:
    params: dict
    metric: float
    all_metrics: dict = field(default_factory=dict)  # param-tuple -> metric


def optimize(strategy: Strategy, df: pd.DataFrame,
             record: "TrialRegistry | None" = None) -> OptResult:
    """Exhaustive grid search; returns the best parameter set and its metric.

    If a TrialRegistry is supplied, EVERY evaluated configuration is logged (the
    honest trial count N that later deflates the Sharpe).
    """
    best = None
    all_metrics = {}
    for params in strategy.grid():
        m = strategy.evaluate(df, **params)
        all_metrics[tuple(sorted(params.items()))] = m
        if record is not None:
            record.log(strategy.name, params, m)
        if best is None or m > best.metric:
            best = OptResult(params=params, metric=m)
    best.all_metrics = all_metrics
    return best


# --- Example simple strategies (used for gate validation and as candidates) ---

def time_series_momentum(df: pd.DataFrame, lookback: int = 20) -> pd.Series:
    """Long if price rose over `lookback` bars, short if it fell (sign of momentum)."""
    mom = df["close"].diff(lookback)
    return pd.Series(np.sign(mom.to_numpy()), index=df.index).fillna(0.0)


def mean_reversion(df: pd.DataFrame, lookback: int = 20, z: float = 1.0) -> pd.Series:
    """Fade deviations from a rolling mean beyond z standard deviations."""
    ma = df["close"].rolling(lookback).mean()
    sd = df["close"].rolling(lookback).std()
    dev = (df["close"] - ma) / sd
    pos = pd.Series(0.0, index=df.index)
    pos[dev > z] = -1.0
    pos[dev < -z] = 1.0
    return pos.fillna(0.0)
