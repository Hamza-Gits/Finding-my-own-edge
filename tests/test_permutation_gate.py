"""Permutation engine invariants + MCPT gate power/size validation.

These are the load-bearing correctness tests for the whole pipeline: a gate that
cannot tell a real edge from noise is worse than useless. We verify:
  * the bar-permutation preserves the marginal return distribution but destroys
    serial correlation (structure), and keeps OHLC invariants intact;
  * the in-sample MCPT has POWER  (flags a PLANTED momentum edge: p small);
  * the in-sample MCPT has correct SIZE (does NOT flag a worthless random walk).
"""
import numpy as np
import pytest

from edge.io import synthetic
from edge.validation.permutation import permute_ohlc
from edge.validation.mcpt import insample_mcpt
from edge.discovery.strategy import Strategy, time_series_momentum
from edge.discovery.backtest import sharpe


def _autocorr1(x):
    x = np.asarray(x, float)
    return np.corrcoef(x[:-1], x[1:])[0, 1]


def test_permutation_preserves_distribution_destroys_autocorr():
    df = synthetic.trend_ohlc(n=6000, seed=3, phi=0.35)
    rng = np.random.default_rng(0)
    perm = permute_ohlc(df, rng=rng)
    r0 = np.log(df["close"]).diff().dropna().to_numpy()
    rp = np.log(perm["close"]).diff().dropna().to_numpy()
    # Marginal moments approximately preserved (close-to-close = gap + open-close,
    # permuted with independent shuffles, so ~preserved not exact).
    assert rp.std() == pytest.approx(r0.std(), rel=0.15)
    # Real series has momentum; permuted should have ~zero autocorrelation.
    assert _autocorr1(r0) > 0.15
    assert abs(_autocorr1(rp)) < 0.06
    # OHLC invariants survive reconstruction.
    assert (perm["high"] >= perm[["open", "close"]].max(axis=1) - 1e-9).all()
    assert (perm["low"] <= perm[["open", "close"]].min(axis=1) + 1e-9).all()


def _momentum_optimizer():
    strat = Strategy("tsmom", time_series_momentum,
                     {"lookback": [5, 10, 20, 40, 80]}, cost_per_turn=0.0,
                     objective=sharpe)

    def opt(df):
        best_m, best_p = -np.inf, None
        for p in strat.grid():
            m = strat.evaluate(df, **p)
            if m > best_m:
                best_m, best_p = m, p
        return best_m, best_p
    return opt


def test_mcpt_has_power_on_planted_edge():
    df = synthetic.trend_ohlc(n=5000, seed=5, phi=0.35)
    res = insample_mcpt(df, _momentum_optimizer(), n_perm=200, seed=7)
    assert res.p_value < 0.05            # gate detects the real edge


def test_mcpt_has_correct_size_on_random_walk():
    # Under the worthless-strategy null the p-value is ~uniform, so ANY single
    # draw can land near 0.05 by chance (~1-in-20). The properties that matter:
    #   (1) the HARD gate (p<0.01) must not fire on noise -> no false survivors;
    #   (2) p-values are not SYSTEMATICALLY significant (mean well above 0.05).
    ps = []
    for sd in range(5):
        df = synthetic.random_walk_ohlc(n=2500, seed=100 + sd)
        ps.append(insample_mcpt(df, _momentum_optimizer(), n_perm=120, seed=7).p_value)
    ps = np.array(ps)
    assert (ps < 0.01).sum() == 0        # no false survivor at the real threshold
    assert ps.mean() > 0.30              # not systematically significant
