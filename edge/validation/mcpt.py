"""In-sample Monte Carlo Permutation Test (Masters).

Re-run the FULL strategy INCLUDING its optimization on each of >=1,000 bar
permutations; the p-value is the fraction of permutations whose optimized metric
matches or beats the real optimized metric:

    p = (1 + #{perm_metric >= real_metric}) / (1 + n_perm)

This estimates the probability that a worthless strategy combined with the SAME
search could have produced the observed performance — i.e. it prices in
data-mining bias. HARD GATE: reject the candidate unless p < 0.01.

Null hypothesis (Masters): the strategy is worthless — its position decisions
are unrelated to subsequent returns. Unlike a mean-zero bootstrap, this is not
fooled by a merely long-biased model in a drifting market.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from .permutation import permute_ohlc

# An optimizer: df -> (best_metric, best_params)
OptimizeFn = Callable[[pd.DataFrame], tuple[float, dict]]


@dataclass
class MCPTResult:
    real_metric: float
    p_value: float
    n_perm: int
    perm_metrics: np.ndarray
    real_params: dict

    def passed(self, threshold: float = 0.01) -> bool:
        return self.p_value < threshold


def insample_mcpt(df: pd.DataFrame, optimize_fn: OptimizeFn, n_perm: int = 1000,
                  start_index: int = 0, seed: int = 7,
                  progress: bool = False) -> MCPTResult:
    """Run the in-sample permutation test. `optimize_fn` MUST re-optimize."""
    real_metric, real_params = optimize_fn(df)
    rng = np.random.default_rng(seed)
    perm_metrics = np.empty(n_perm, dtype=float)
    it = range(n_perm)
    if progress:
        try:
            from tqdm import trange
            it = trange(n_perm, desc="in-sample MCPT")
        except ImportError:
            pass
    count_ge = 0
    for i in it:
        perm_df = permute_ohlc(df, start_index=start_index, rng=rng)
        m, _ = optimize_fn(perm_df)
        perm_metrics[i] = m
        if m >= real_metric:
            count_ge += 1
    p = (1 + count_ge) / (1 + n_perm)
    return MCPTResult(real_metric=real_metric, p_value=p, n_perm=n_perm,
                      perm_metrics=perm_metrics, real_params=real_params)
