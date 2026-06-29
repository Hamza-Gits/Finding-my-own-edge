"""Stationary (block) bootstrap — a cross-check on the permutation gate.

The bar-permutation MCPT destroys volatility clustering and long memory, so it
can mis-size for edges that live in those features. The stationary bootstrap
(Politis & Romano 1994) resamples BLOCKS of random geometric length, preserving
short-range serial dependence, and gives a confidence interval / p-value for the
metric under resampling. We require survivors to agree across BOTH tests — if the
permutation says 'edge' but the block bootstrap's CI straddles zero, we distrust it.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

from ..discovery.backtest import sharpe


def stationary_bootstrap_indices(n: int, mean_block: float,
                                 rng: np.random.Generator) -> np.ndarray:
    """Indices for one stationary-bootstrap resample (geometric block lengths)."""
    p = 1.0 / max(mean_block, 1.0)
    idx = np.empty(n, dtype=int)
    idx[0] = rng.integers(n)
    restarts = rng.random(n) < p
    steps = rng.integers(n, size=n)
    for t in range(1, n):
        idx[t] = steps[t] if restarts[t] else (idx[t - 1] + 1) % n
    return idx


def stationary_bootstrap(returns: np.ndarray, metric_fn: Callable = sharpe,
                         n_boot: int = 1000, mean_block: float = 20.0,
                         seed: int = 7) -> np.ndarray:
    """Distribution of `metric_fn` over `n_boot` stationary-bootstrap resamples."""
    r = np.asarray(returns, dtype=float)
    rng = np.random.default_rng(seed)
    out = np.empty(n_boot)
    for b in range(n_boot):
        out[b] = metric_fn(r[stationary_bootstrap_indices(r.size, mean_block, rng)])
    return out


def bootstrap_report(returns: np.ndarray, metric_fn: Callable = sharpe,
                     n_boot: int = 1000, mean_block: float = 20.0,
                     seed: int = 7, alpha: float = 0.05) -> dict:
    """Real metric, bootstrap CI, and P(metric <= 0) under the resampling null."""
    r = np.asarray(returns, dtype=float)
    real = metric_fn(r)
    boots = stationary_bootstrap(r, metric_fn, n_boot, mean_block, seed)
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"real_metric": float(real), "ci_low": float(lo), "ci_high": float(hi),
            "frac_le_zero": float((boots <= 0).mean()), "n_boot": n_boot,
            "mean_block": mean_block, "boots": boots,
            "ci_excludes_zero": bool(lo > 0 or hi < 0)}
