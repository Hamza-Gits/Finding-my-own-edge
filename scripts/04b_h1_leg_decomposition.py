"""Per-year long/short leg decomposition for H1 (the falsification test).

The H1 'two-sided' honesty check used POOLED leg P&L: min(pnl_long, pnl_short)>0.
The audit flagged this as too weak — a leg can be net-positive overall while booking
nearly all of it in a single trend year (2022 bear), which is exactly the smuggled-drift
failure mode the pipeline exists to reject. This script decomposes each leg's P&L BY YEAR
on the FROZEN headline best_p, net of 2-tick costs, via the reusable per-year leg gate
(`edge.validation.leg_gate`) that should gate every symmetric candidate BEFORE the
expensive full-gate stack.

    .venv/Scripts/python.exe scripts/04b_h1_leg_decomposition.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from edge.config import load_config
from edge.costs.cost_model import FuturesCosts
from edge.data.bars import five_min_rth_bars
from edge.discovery.backtest import strategy_returns
from edge.features.conditional_orb import daily_compression, make_compression_orb, trailing_rank
from edge.validation.leg_gate import per_year_leg_gate, format_leg_gate

BEST_P = {"or_bars": 3, "buffer_ticks": 4, "compression_q": 0.33, "stop_mult": 0.5}


def main():
    cfg = load_config()
    proc = cfg.path("processed")
    adj = pd.read_parquet(proc / "MNQ_1min_adj.parquet")
    unadj = pd.read_parquet(proc / "MNQ_1min_unadj.parquet")
    inst = cfg.instrument("MNQ")
    tick = inst["tick_size"]

    F5 = five_min_rth_bars(adj)
    comp = trailing_rank(daily_compression(adj), min_periods=60)
    sig = make_compression_orb(comp, tick)

    costs = FuturesCosts.from_config(cfg, "MNQ")
    notional = float(unadj["close"].median()) * inst["multiplier"]
    cpt2 = costs.round_trip_cost(slippage_ticks=2) / (2 * notional)

    r_long = strategy_returns(F5, sig(F5, **{**BEST_P, "direction": "long"}), cpt2)
    r_short = strategy_returns(F5, sig(F5, **{**BEST_P, "direction": "short"}), cpt2)
    r_both = strategy_returns(F5, sig(F5, **{**BEST_P, "direction": "both"}), cpt2)

    res = per_year_leg_gate(F5.index, r_long, r_short, r_both)

    print("=" * 60)
    print("H1 per-year leg decomposition (frozen best_p, NET of 2-tick)")
    print(f"best_p = {BEST_P}")
    print("=" * 60)
    print(format_leg_gate(res))
    print("=" * 60)
    print("Pooled min(pnl_long,pnl_short)>0 said 'two-sided OK'; the per-year")
    print(f"gate says {'PASS' if res.passed else 'FAIL'} - this is the honest two-sided test.")


if __name__ == "__main__":
    main()
