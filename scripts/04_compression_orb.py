"""Run hypothesis H1 (compression-gated ORB) through the full gate stack.

A focused, single-candidate run: the compression series is computed once on real
MNQ, the breakout is evaluated on 5-min RTH bars, and the candidate faces the same
honest gates as everything else (IS-MCPT, WF-MCPT, DSR, PBO, bootstrap, cost
sensitivity). Adds the hypothesis's own honesty check: long-leg vs short-leg P&L
(a real compression edge should be roughly two-sided, not secret drift).

    .venv/Scripts/python.exe scripts/04_compression_orb.py quick
    .venv/Scripts/python.exe scripts/04_compression_orb.py
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from edge.config import load_config
from edge.costs.cost_model import FuturesCosts
from edge.data.bars import five_min_rth_bars
from edge.discovery.backtest import strategy_returns, sharpe, profit_factor
from edge.discovery.strategy import Strategy
from edge.discovery.trials import TrialRegistry
from edge.features.conditional_orb import daily_compression, make_compression_orb, trailing_rank
from edge.validation.mcpt import insample_mcpt
from edge.validation.walkforward import walk_forward_mcpt
from edge.stats import dsr
from edge.stats.pbo import pbo_cscv
from edge.validation.bootstrap import bootstrap_report

QUICK = len(sys.argv) > 1 and sys.argv[1].lower() == "quick"
IS_PERM = 80 if QUICK else 500
WF_PERM = 40 if QUICK else 150
BOOT = 300 if QUICK else 1000

# compression_q is now a trailing-PERCENTILE level: arm the quietest q fraction of
# days (bottom tercile etc.), which keeps the armed-day sample adequate.
GRID = {"or_bars": [3], "buffer_ticks": [1, 2, 4],
        "compression_q": [0.2, 0.33, 0.4], "stop_mult": [0.5, 1.0]}

cfg = load_config()
G = cfg["gates"]


def grid_search(strat, df, registry=None):
    best_m, best_p = -np.inf, None
    for p in strat.grid():
        m = strat.evaluate(df, **p)
        if registry is not None:
            registry.log(strat.name, p, m, sharpe=m)
        if m > best_m:
            best_m, best_p = m, p
    return best_m, best_p


def main():
    print(f"{'='*78}\nH1: COMPRESSION-GATED ORB  ({'QUICK' if QUICK else 'FULL'})\n{'='*78}")
    proc = cfg.path("processed")
    adj = pd.read_parquet(proc / "MNQ_1min_adj.parquet")
    unadj = pd.read_parquet(proc / "MNQ_1min_unadj.parquet")
    inst = cfg.instrument("MNQ")
    tick = inst["tick_size"]

    F5 = five_min_rth_bars(adj)
    comp_raw = daily_compression(adj)
    comp = trailing_rank(comp_raw, min_periods=60)          # arm bottom-q fraction
    armed_frac = {q: float((comp <= q).mean()) for q in GRID["compression_q"]}
    print(f"5-min RTH bars : {len(F5):,}")
    print(f"compression days: {len(comp):,}  armed fraction "
          + "  ".join(f"q={q}:{armed_frac[q]:.0%}" for q in GRID["compression_q"]))

    costs = FuturesCosts.from_config(cfg, "MNQ")
    px = float(unadj["close"].median())
    notional = px * inst["multiplier"]
    cpt1 = costs.round_trip_cost(slippage_ticks=1) / (2 * notional)
    cpt2 = costs.round_trip_cost(slippage_ticks=2) / (2 * notional)

    sig = make_compression_orb(comp, tick)
    strat = Strategy("compression_orb", sig, GRID, cost_per_turn=cpt1, objective=sharpe)

    registry = TrialRegistry()
    best_m, best_p = grid_search(strat, F5, registry)
    N, var_sr = registry.n, registry.var_sharpe()

    pos = strat.positions(F5, **best_p)
    r1 = strategy_returns(F5, pos, cpt1)
    r2 = strategy_returns(F5, pos, cpt2)
    trades = int((pos.diff().abs() > 0).sum())
    ann = np.sqrt(252 * 78)
    print(f"best params    : {best_p}")
    print(f"bar Sharpe     : {sharpe(r1):+.4f}  (ann {sharpe(r1)*ann:+.2f}) | net2 {sharpe(r2)*ann:+.2f}")
    print(f"trades (pos chg): {trades:,}   N={N}  Var[SR]={var_sr:.2e}")

    # honesty check: long-leg vs short-leg P&L on the best params
    long_only = sig(F5, **{**best_p, "direction": "long"})
    short_only = sig(F5, **{**best_p, "direction": "short"})
    pnl_long = strategy_returns(F5, long_only, cpt1).sum()
    pnl_short = strategy_returns(F5, short_only, cpt1).sum()
    print(f"leg P&L (log)  : long {pnl_long:+.4f}  short {pnl_short:+.4f}  "
          f"(two-sided? {'yes' if min(pnl_long, pnl_short) > 0 else 'NO -> drift risk'})")

    # --- gates ---
    def opt(d):
        return grid_search(strat, d, None)
    mc = insample_mcpt(F5, opt, n_perm=IS_PERM, seed=cfg.seed, progress=True)
    print(f"\nIS-MCPT p      : {mc.p_value:.4f}  (gate < {G['insample_mcpt_p_max']}; {IS_PERM} perms)")

    res = {"best_p": best_p, "N": N, "trades": trades, "armed_frac": armed_frac,
           "sharpe_ann1": sharpe(r1) * ann, "sharpe_ann2": sharpe(r2) * ann,
           "pf": profit_factor(r1.to_numpy()), "is_mcpt_p": mc.p_value,
           "pnl_long": pnl_long, "pnl_short": pnl_short, "cpt1": cpt1, "cpt2": cpt2}

    enough_trades = trades >= G["min_trades"]
    passed_is = mc.p_value < G["insample_mcpt_p_max"] and sharpe(r2) > 0 and enough_trades

    if passed_is:
        wf = walk_forward_mcpt(F5, opt, sig, cost_per_turn=cpt1, n_folds=5,
                               n_perm=WF_PERM, seed=cfg.seed)
        d = dsr.deflated_sharpe_from_returns(r1.to_numpy(), var_sr, N)
        rm = pd.DataFrame({str(p): strategy_returns(F5, strat.positions(F5, **p), cpt1)
                           for p in strat.grid()})
        pbo = pbo_cscv(rm, n_splits=10 if QUICK else 14)["pbo"]
        boot = bootstrap_report(r1.to_numpy(), n_boot=BOOT, mean_block=40, seed=cfg.seed)
        res.update({"wf_mcpt_p": wf.p_value, "dsr": d["dsr"], "pbo": pbo,
                    "boot_ci_excl0": boot["ci_excludes_zero"]})
        checks = [
            (mc.p_value < G["insample_mcpt_p_max"], f"IS-MCPT p={mc.p_value:.3f}"),
            (wf.p_value < G["wf_mcpt_p_max_2yr"], f"WF-MCPT p={wf.p_value:.3f}"),
            (d["dsr"] > G["dsr_min"], f"DSR={d['dsr']:.3f}"),
            (pbo < G["pbo_max"], f"PBO={pbo:.3f}"),
            (sharpe(r2) > 0, "net-2tk Sharpe>0"),
            (boot["ci_excludes_zero"], "bootstrap CI excludes 0"),
            (min(pnl_long, pnl_short) > 0, "two-sided (long & short both +)"),
        ]
        failing = [m for ok, m in checks if not ok]
        res["verdict"] = "GO" if not failing else "NO-GO"
        res["reason"] = "ALL GATES PASSED" if not failing else "; ".join(failing)
    else:
        res["verdict"] = "NO-GO"
        res["reason"] = ("trades < %d (only %d)" % (G["min_trades"], trades) if not enough_trades
                         else "net Sharpe <= 0 after 2-tick costs" if sharpe(r2) <= 0
                         else f"in-sample MCPT p={mc.p_value:.3f} >= {G['insample_mcpt_p_max']}")
    print(f"VERDICT        : {res['verdict']}  ({res['reason']})")

    # equity plot
    fig, ax = plt.subplots(figsize=(9, 3.2))
    ax.plot(r1.cumsum().index, r1.cumsum().values, lw=0.8)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_title(f"compression_orb {best_p} — cumulative net log-return (1-tick)")
    fig.tight_layout(); fig.savefig(cfg.path("reports") / "04_eq_compression_orb.png", dpi=110)
    plt.close(fig)

    write_report(res)
    print(f"report -> {cfg.path('reports') / '04_compression_orb_report.md'}")


def write_report(r):
    L = ["# H1 Compression-Gated ORB — gate report\n",
         "_Generated by `scripts/04_compression_orb.py`._\n",
         f"**Verdict: {r['verdict']} — {r['reason']}**\n",
         "## Numbers\n",
         f"- best params: `{r['best_p']}`  (honest N = {r['N']})",
         f"- armed-day fraction: " + ", ".join(f"q={q}: {f:.0%}" for q, f in r['armed_frac'].items()),
         f"- trades (position changes): **{r['trades']:,}** (min-trades gate {load_config()['gates']['min_trades']})",
         f"- annualized Sharpe: **{r['sharpe_ann1']:.2f}** (1-tick) / **{r['sharpe_ann2']:.2f}** (2-tick); PF {r['pf']:.2f}",
         f"- leg P&L (log, 1-tick): long {r['pnl_long']:+.4f} / short {r['pnl_short']:+.4f} "
         f"→ {'two-sided ✔' if min(r['pnl_long'], r['pnl_short']) > 0 else 'one-sided — drift risk ✘'}",
         f"- in-sample MCPT p = **{r['is_mcpt_p']:.4f}**"]
    if "wf_mcpt_p" in r:
        L.append(f"- walk-forward MCPT p = **{r['wf_mcpt_p']:.4f}**; DSR = **{r['dsr']:.4f}**; "
                 f"PBO = **{r['pbo']:.4f}**; bootstrap CI excludes 0: {r['boot_ci_excl0']}")
    L += ["\n## Interpretation\n",
          "The compression gate arms the breakout only after a quiet overnight; direction is "
          "chosen intraday so the rule is symmetric. The long/short leg split is the key honesty "
          "check — a one-sided result would mean we are back to capturing drift, not a real "
          "compression-expansion edge.\n",
          "Equity curve: `reports/04_eq_compression_orb.png`."]
    (load_config().path("reports") / "04_compression_orb_report.md").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
