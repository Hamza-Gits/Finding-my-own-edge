"""Demonstrate the validation gate end-to-end on synthetic data.

Runs the SAME machinery that real edges will face, against two controls:
  * a series with a PLANTED momentum edge  -> the gate should say GO
  * a worthless random walk                -> the gate should say NO-GO
This proves the gate has correct power and size before any real data arrives.

    .venv/Scripts/python.exe scripts/demo_gate.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from edge.io import synthetic
from edge.validation.mcpt import insample_mcpt
from edge.discovery.strategy import Strategy, time_series_momentum
from edge.discovery.backtest import sharpe
from edge.discovery.trials import TrialRegistry
from edge.stats import dsr

N_PERM = 300
GRID = {"lookback": [5, 10, 20, 40, 80]}


def run_case(name: str, df) -> None:
    reg = TrialRegistry()
    strat = Strategy("tsmom", time_series_momentum, GRID, cost_per_turn=0.0,
                     objective=sharpe)

    def optimize(d):
        best_m, best_p = -np.inf, None
        for p in strat.grid():
            r = strat.returns(d, **p)
            m = sharpe(r)
            reg.log("tsmom", p, m, sharpe=m)   # log EVERY trial -> N for DSR
            if m > best_m:
                best_m, best_p = m, p
        return best_m, best_p

    res = insample_mcpt(df, optimize, n_perm=N_PERM, seed=7)
    best_ret = strat.returns(df, **res.real_params)
    g3, g4 = dsr._moments(best_ret.to_numpy())
    d = dsr.deflated_sharpe_ratio(sharpe(best_ret), len(best_ret), g3, g4,
                                  var_sr_trials=reg.var_sharpe(),
                                  n_trials=reg.n)
    mcpt_ok = res.passed(0.01)
    dsr_ok = d["dsr"] > 0.95
    verdict = "GO" if (mcpt_ok and dsr_ok) else "NO-GO"

    print(f"\n=== {name} ===")
    print(f"  best params         : {res.real_params}")
    print(f"  bar-level Sharpe     : {sharpe(best_ret):+.4f}  (non-annualized)")
    print(f"  in-sample MCPT p     : {res.p_value:.4f}   (gate: <0.01 -> {mcpt_ok})")
    print(f"  trials logged (N)    : {reg.n}")
    print(f"  Deflated Sharpe (DSR): {d['dsr']:.4f}   (gate: >0.95 -> {dsr_ok})")
    print(f"  VERDICT              : {verdict}")


def main():
    print("Validation-gate demonstration (synthetic controls)")
    print(f"momentum lookback grid={GRID['lookback']}, permutations={N_PERM}")
    run_case("PLANTED EDGE (AR(1) momentum, phi=0.35)",
             synthetic.trend_ohlc(n=5000, seed=11, phi=0.35))
    run_case("WORTHLESS NOISE (random walk)",
             synthetic.random_walk_ohlc(n=5000, seed=11))
    print("\nExpected: planted edge -> GO ; worthless noise -> NO-GO")


if __name__ == "__main__":
    main()
