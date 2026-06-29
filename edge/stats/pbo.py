"""Probability of Backtest Overfitting (PBO) via CSCV.

Bailey, Borwein, López de Prado & Zhu (2017), 'The Probability of Backtest
Overfitting'. Given a matrix of per-bar returns for N strategy variants (every
parameter set you tried is a column), CSCV measures how often the variant that
looks best IN-SAMPLE fails to stay above median OUT-OF-SAMPLE.

Procedure:
  1. Split the T observations into S disjoint, equal time blocks (S even).
  2. For every way to choose S/2 blocks as IS (the rest OOS):
       - rank the N variants by IS performance; take the IS-best, n*.
       - find n*'s relative rank omega in (0,1) among the N variants OOS.
       - logit lambda = ln(omega / (1 - omega)).
  3. PBO = fraction of splits with lambda <= 0  (IS-best lands in the bottom
     half OOS). HARD GATE: PBO < 0.20.

A high PBO means your "best" parameters are an artefact of the search, not a real
edge — the single most important guard against fooling yourself with a grid.
"""
from __future__ import annotations

from itertools import combinations
from typing import Callable

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from ..discovery.backtest import sharpe


def pbo_cscv(returns_matrix: pd.DataFrame | np.ndarray, n_splits: int = 16,
             metric_fn: Callable[[np.ndarray], float] = sharpe) -> dict:
    """Compute PBO. `returns_matrix` is (T observations x N variants)."""
    M = np.asarray(returns_matrix, dtype=float)
    T, N = M.shape
    if N < 2:
        raise ValueError("PBO needs >= 2 strategy variants (columns)")
    if n_splits % 2 != 0:
        n_splits -= 1
    blocks = np.array_split(np.arange(T), n_splits)

    logits = []
    n_star_oos_better = 0
    for is_blocks in combinations(range(n_splits), n_splits // 2):
        is_set = set(is_blocks)
        is_idx = np.concatenate([blocks[b] for b in is_blocks])
        oos_idx = np.concatenate([blocks[b] for b in range(n_splits) if b not in is_set])

        is_perf = np.array([metric_fn(M[is_idx, j]) for j in range(N)])
        oos_perf = np.array([metric_fn(M[oos_idx, j]) for j in range(N)])
        n_star = int(np.argmax(is_perf))                # IS-best variant
        rank = rankdata(oos_perf)[n_star]               # 1..N (1 = worst)
        omega = rank / (N + 1.0)                         # relative rank in (0,1)
        omega = min(max(omega, 1e-6), 1 - 1e-6)
        logits.append(float(np.log(omega / (1 - omega))))
        if oos_perf[n_star] > np.median(oos_perf):
            n_star_oos_better += 1

    logits = np.array(logits)
    n_comb = len(logits)
    pbo = float(np.mean(logits <= 0.0))
    return {"pbo": pbo, "logits": logits, "n_combinations": n_comb,
            "median_logit": float(np.median(logits)),
            "frac_is_best_beats_oos_median": n_star_oos_better / n_comb}
