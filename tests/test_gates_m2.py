"""Validate the M2 gate machinery (walk-forward MCPT, CPCV, PBO, bootstrap).

Same philosophy as the in-sample gate tests: prove POWER (flags a planted edge)
and SIZE (does not crown noise) before trusting these on real data.
"""
import numpy as np
import pytest

from edge.io import synthetic
from edge.discovery.strategy import Strategy, time_series_momentum
from edge.discovery.backtest import sharpe, strategy_returns
from edge.validation.walkforward import walk_forward, walk_forward_mcpt
from edge.stats.cv import purged_kfold, cpcv_splits, cpcv_oos_distribution
from edge.stats.pbo import pbo_cscv
from edge.validation.bootstrap import bootstrap_report

GRID = [2, 3, 5, 8, 13, 21, 34]


def _mom_opt():
    strat = Strategy("tsmom", time_series_momentum, {"lookback": GRID},
                     cost_per_turn=0.0, objective=sharpe)

    def opt(df):
        best_m, best_p = -np.inf, None
        for p in strat.grid():
            m = strat.evaluate(df, **p)
            if m > best_m:
                best_m, best_p = m, p
        return best_m, best_p
    return opt


# ---------- walk-forward + WF-MCPT ----------

def test_walk_forward_runs_and_stitches_oos():
    df = synthetic.trend_ohlc(n=4000, seed=1, phi=0.35)
    wf = walk_forward(df, _mom_opt(), time_series_momentum, n_folds=4)
    assert wf.n_folds == 4
    assert len(wf.oos_returns) > 0
    assert len(wf.fold_params) == 4


def test_wf_mcpt_has_power_on_planted_edge():
    df = synthetic.trend_ohlc(n=4000, seed=2, phi=0.4)
    res = walk_forward_mcpt(df, _mom_opt(), time_series_momentum,
                            n_folds=4, n_perm=100, seed=7)
    assert res.real_oos_metric > 0
    assert res.p_value < 0.05            # detects the real OOS edge


def test_wf_mcpt_correct_size_on_noise():
    # Hard gate must not fire across several random-walk seeds.
    ps = []
    for sd in range(4):
        df = synthetic.random_walk_ohlc(n=3000, seed=200 + sd)
        ps.append(walk_forward_mcpt(df, _mom_opt(), time_series_momentum,
                                    n_folds=4, n_perm=80, seed=7).p_value)
    ps = np.array(ps)
    assert (ps < 0.01).sum() == 0        # no false survivor at the strict threshold


# ---------- purged CV / CPCV ----------

def test_purged_kfold_train_test_disjoint_with_embargo():
    splits = purged_kfold(1000, n_folds=5, embargo_pct=0.02)
    assert len(splits) == 5
    for train, test in splits:
        assert set(train).isdisjoint(set(test))     # purge holds
        # embargo: no training index within the embargo gap around the test block
        gap = int(1000 * 0.02)
        assert not any((test.min() - gap) <= ti < test.min() for ti in train)


def test_cpcv_partitions_count_and_coverage():
    splits = cpcv_splits(1200, n_groups=6, k=2, embargo_pct=0.01)
    assert len(splits) == 15                          # C(6,2)
    for train, test, groups in splits:
        assert set(train).isdisjoint(set(test))


def test_cpcv_oos_distribution_positive_for_edge():
    df = synthetic.trend_ohlc(n=6000, seed=3, phi=0.4)
    d = cpcv_oos_distribution(df, _mom_opt(), time_series_momentum,
                              n_groups=6, k=2)
    assert d["n_partitions"] == 15
    assert d["frac_positive"] > 0.6                   # edge holds across most splits


# ---------- PBO via CSCV ----------

def _returns_matrix(df):
    cols = {lb: strategy_returns(df, time_series_momentum(df, lookback=lb))
            for lb in GRID}
    import pandas as pd
    return pd.DataFrame(cols)


def test_pbo_low_for_real_edge_high_for_noise():
    edge = _returns_matrix(synthetic.trend_ohlc(n=8000, seed=4, phi=0.4))
    noise = _returns_matrix(synthetic.random_walk_ohlc(n=8000, seed=4))
    pbo_edge = pbo_cscv(edge, n_splits=10)["pbo"]
    pbo_noise = pbo_cscv(noise, n_splits=10)["pbo"]
    assert pbo_edge < 0.25                            # not overfit -> passes gate
    assert pbo_noise > pbo_edge                       # noise is more overfit-prone


# ---------- stationary bootstrap ----------

def test_bootstrap_ci_excludes_zero_for_edge_includes_for_noise():
    edge_r = strategy_returns(synthetic.trend_ohlc(n=6000, seed=5, phi=0.4),
                              time_series_momentum(
                                  synthetic.trend_ohlc(n=6000, seed=5, phi=0.4), 5))
    noise_r = strategy_returns(synthetic.random_walk_ohlc(n=6000, seed=5),
                               time_series_momentum(
                                   synthetic.random_walk_ohlc(n=6000, seed=5), 5))
    edge = bootstrap_report(edge_r.to_numpy(), n_boot=500, mean_block=20, seed=7)
    noise = bootstrap_report(noise_r.to_numpy(), n_boot=500, mean_block=20, seed=7)
    # The semantic gate: does the resampled Sharpe CI exclude zero?
    assert edge["frac_le_zero"] < 0.10                # edge Sharpe robustly > 0
    assert edge["ci_excludes_zero"]                   # edge CI is entirely above 0
    assert not noise["ci_excludes_zero"]              # noise CI straddles 0
    assert noise["frac_le_zero"] > edge["frac_le_zero"]   # noise far less robust
