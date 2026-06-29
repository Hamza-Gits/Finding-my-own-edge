"""Purged k-fold + embargo and Combinatorial Purged Cross-Validation (CPCV).

López de Prado's fix for the leakage that ordinary k-fold inflicts on serially
dependent financial data: when a test block sits next to a training block, the
serial correlation (and any label that spans the boundary) leaks information.

  * PURGE   — drop training observations adjacent to the test block.
  * EMBARGO — additionally drop a small fraction of bars AFTER each test block,
              because predictability decays gradually rather than instantly.

CPCV forms many train/test partitions by leaving out k of N groups in every
combination, producing a DISTRIBUTION of OOS performance (not a single number) —
far harder to fool than one lucky split.
"""
from __future__ import annotations

from itertools import combinations
from typing import Callable

import numpy as np
import pandas as pd

from ..discovery.backtest import strategy_returns, sharpe


def purged_kfold(n_obs: int, n_folds: int = 6, embargo_pct: float = 0.01
                 ) -> list[tuple[np.ndarray, np.ndarray]]:
    """Contiguous test folds with a purge+embargo gap removed from training."""
    bounds = np.linspace(0, n_obs, n_folds + 1).astype(int)
    emb = int(n_obs * embargo_pct)
    splits = []
    for i in range(n_folds):
        t0, t1 = int(bounds[i]), int(bounds[i + 1])
        test = np.arange(t0, t1)
        train_mask = np.ones(n_obs, dtype=bool)
        train_mask[max(0, t0 - emb):min(n_obs, t1 + emb)] = False
        splits.append((np.where(train_mask)[0], test))
    return splits


def cpcv_splits(n_obs: int, n_groups: int = 6, k: int = 2,
                embargo_pct: float = 0.01
                ) -> list[tuple[np.ndarray, np.ndarray, tuple]]:
    """All C(n_groups, k) leave-k-groups-out partitions with purge+embargo."""
    groups = np.array_split(np.arange(n_obs), n_groups)
    emb = int(n_obs * embargo_pct)
    splits = []
    for test_groups in combinations(range(n_groups), k):
        test_idx = np.concatenate([groups[g] for g in test_groups])
        train_mask = np.ones(n_obs, dtype=bool)
        for g in test_groups:
            gi = groups[g]
            train_mask[max(0, gi[0] - emb):min(n_obs, gi[-1] + 1 + emb)] = False
        splits.append((np.where(train_mask)[0], np.sort(test_idx), test_groups))
    return splits


def cpcv_oos_distribution(df: pd.DataFrame, optimize_fn: Callable,
                          positions_fn: Callable, cost_per_turn: float = 0.0,
                          n_groups: int = 6, k: int = 2, embargo_pct: float = 0.01,
                          metric_fn=sharpe) -> dict:
    """OOS metric distribution across all CPCV partitions.

    For each partition: optimize on the (purged) training rows, evaluate the
    chosen params on the held-out rows. Returns the array of OOS metrics plus
    summary stats — the spread is the honest read on parameter stability.
    """
    splits = cpcv_splits(len(df), n_groups, k, embargo_pct)
    oos_metrics = []
    for train_idx, test_idx, _ in splits:
        train = df.iloc[train_idx]
        _, params = optimize_fn(train)
        test = df.iloc[test_idx]
        rets = strategy_returns(test, positions_fn(test, **params), cost_per_turn)
        oos_metrics.append(metric_fn(rets))
    arr = np.array(oos_metrics, dtype=float)
    return {"oos_metrics": arr, "median": float(np.median(arr)),
            "mean": float(arr.mean()), "std": float(arr.std(ddof=1) if arr.size > 1 else 0.0),
            "frac_positive": float((arr > 0).mean()), "n_partitions": len(arr)}
