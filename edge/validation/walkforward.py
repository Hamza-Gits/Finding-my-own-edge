"""Walk-forward evaluation and walk-forward MCPT (the OOS-honest gate).

In-sample MCPT prices in data-mining bias, but it still optimizes and tests on
the SAME bars. Walk-forward goes further: re-optimize on a training window, then
trade the next (untouched) block with the FIXED parameters, and stitch all the
out-of-sample blocks into one OOS curve. Nothing in the OOS curve ever saw its
own parameters chosen.

Walk-forward MCPT then asks: could this OOS curve arise from a worthless system?
Following Masters, we keep the FIRST training window real and permute everything
after it, then re-run the entire walk-forward per permutation (each later fold
re-optimizes on the now-permuted past). p = fraction of permutations whose OOS
metric >= the real OOS metric. HARD GATE: p < 0.05 (>=1y OOS) / p < 0.01 (>=2y).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from ..discovery.backtest import strategy_returns, sharpe
from .permutation import permute_ohlc

# optimizer: train_df -> (best_metric, best_params)
OptimizeFn = Callable[[pd.DataFrame], tuple[float, dict]]
# positions: (df, **params) -> target-position Series aligned to df
PositionsFn = Callable[..., pd.Series]


@dataclass
class WFResult:
    oos_returns: pd.Series        # stitched out-of-sample per-bar returns
    oos_metric: float
    fold_params: list[dict]
    fold_metrics: list[float]     # OOS metric per fold
    n_folds: int
    first_train_end: int          # index where the first OOS block begins


def walk_forward(df: pd.DataFrame, optimize_fn: OptimizeFn,
                 positions_fn: PositionsFn, cost_per_turn: float = 0.0,
                 n_folds: int = 5, min_train_frac: float = 0.4,
                 anchored: bool = True, metric_fn=sharpe) -> WFResult:
    """Rolling/anchored walk-forward. Each test block is traded with params fit
    ONLY on prior data; indicator warm-up uses the train tail (no leakage)."""
    n = len(df)
    test_start = int(n * min_train_frac)
    edges = np.linspace(test_start, n, n_folds + 1).astype(int)
    train_len = test_start  # for rolling mode, keep window ~ initial train size

    oos_parts, fold_params, fold_metrics = [], [], []
    for i in range(n_folds):
        ts, te = int(edges[i]), int(edges[i + 1])
        if ts >= te:
            continue
        train = df.iloc[:ts] if anchored else df.iloc[max(0, ts - train_len):ts]
        _, params = optimize_fn(train)
        ctx = df.iloc[:te]                       # warm-up + test context
        pos = positions_fn(ctx, **params)
        rets = strategy_returns(ctx, pos, cost_per_turn).iloc[ts:te]
        oos_parts.append(rets)
        fold_params.append(params)
        fold_metrics.append(metric_fn(rets))

    oos = pd.concat(oos_parts) if oos_parts else pd.Series(dtype=float)
    return WFResult(oos_returns=oos, oos_metric=metric_fn(oos),
                    fold_params=fold_params, fold_metrics=fold_metrics,
                    n_folds=len(oos_parts), first_train_end=test_start)


@dataclass
class WFMCPTResult:
    real_oos_metric: float
    p_value: float
    n_perm: int
    perm_metrics: np.ndarray

    def passed(self, threshold: float = 0.05) -> bool:
        return self.p_value < threshold


def walk_forward_mcpt(df: pd.DataFrame, optimize_fn: OptimizeFn,
                      positions_fn: PositionsFn, cost_per_turn: float = 0.0,
                      n_folds: int = 5, min_train_frac: float = 0.4,
                      n_perm: int = 200, seed: int = 7, metric_fn=sharpe,
                      progress: bool = False) -> WFMCPTResult:
    """Permute only the post-first-train bars; re-run the full walk-forward."""
    real = walk_forward(df, optimize_fn, positions_fn, cost_per_turn,
                        n_folds=n_folds, min_train_frac=min_train_frac,
                        metric_fn=metric_fn)
    real_metric = real.oos_metric
    keep = real.first_train_end            # bars [0, keep) stay real
    rng = np.random.default_rng(seed)

    it = range(n_perm)
    if progress:
        try:
            from tqdm import trange
            it = trange(n_perm, desc="WF-MCPT")
        except ImportError:
            pass

    perm_metrics = np.empty(n_perm)
    count_ge = 0
    for i in it:
        pdf = permute_ohlc(df, start_index=max(0, keep - 1), rng=rng)
        wf = walk_forward(pdf, optimize_fn, positions_fn, cost_per_turn,
                          n_folds=n_folds, min_train_frac=min_train_frac,
                          metric_fn=metric_fn)
        perm_metrics[i] = wf.oos_metric
        if wf.oos_metric >= real_metric:
            count_ge += 1
    p = (1 + count_ge) / (1 + n_perm)
    return WFMCPTResult(real_oos_metric=real_metric, p_value=p,
                        n_perm=n_perm, perm_metrics=perm_metrics)
