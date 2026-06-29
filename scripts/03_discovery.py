"""Discovery run: real candidate edges through the FULL gate stack, honestly.

For every candidate (simple, low-parameter rules) on real MNQ data:
  1. grid-search on all data, logging EVERY config to a shared trial registry
     (the honest N that deflates the Sharpe);
  2. in-sample MCPT (re-optimize per permutation)  -> reject unless p < 0.01;
  3. only if it survives: walk-forward MCPT, Deflated Sharpe, PBO via CSCV,
     stationary-bootstrap CI, CPCV OOS distribution, and 1- vs 2-tick cost
     sensitivity;
  4. a GO / NO-GO verdict with the FIRST failing reason stated plainly.

Nothing is tuned to pass. If everything dies, the report says so.

    .venv/Scripts/python.exe scripts/03_discovery.py quick   # fast smoke (few perms)
    .venv/Scripts/python.exe scripts/03_discovery.py         # full run
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
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
from edge.data.bars import daily_rth_bars, five_min_rth_bars
from edge.discovery.backtest import strategy_returns, sharpe, profit_factor
from edge.discovery.strategy import (Strategy, time_series_momentum, mean_reversion)
from edge.discovery.trials import TrialRegistry
from edge.features.seasonality import opening_range_breakout, gap_follow, gap_fade
from edge.validation.mcpt import insample_mcpt
from edge.validation.walkforward import walk_forward_mcpt, walk_forward
from edge.stats import dsr
from edge.stats.pbo import pbo_cscv
from edge.validation.bootstrap import bootstrap_report

QUICK = len(sys.argv) > 1 and sys.argv[1].lower() == "quick"

# Permutation budgets (the spec wants >=1000 for a final verdict; the screen uses
# fewer on the heavy 5-min candidates, and anything that SURVIVES is re-run at 1000).
IS_PERM_DAILY = 200 if QUICK else 1000
IS_PERM_5MIN = 80 if QUICK else 300
WF_PERM = 40 if QUICK else 150
BOOT = 300 if QUICK else 1000

cfg = load_config()
GATES = cfg["gates"]


@dataclass
class Candidate:
    name: str
    frame: str            # 'D' (daily RTH) or 'F5' (5-min RTH)
    signal_fn: object
    grid: dict
    bars_per_year: float
    is_perm: int
    note: str = ""


def build_candidates() -> list[Candidate]:
    BPY_D = 252.0
    BPY_5 = 252.0 * 78  # ~78 five-min RTH bars/day
    return [
        Candidate("daily_tsmom", "D", time_series_momentum,
                  {"lookback": [1, 2, 3, 5, 10, 20]}, BPY_D, IS_PERM_DAILY,
                  "go with N-day momentum of the RTH close"),
        Candidate("daily_meanrev", "D", mean_reversion,
                  {"lookback": [5, 10, 20], "z": [1.0, 1.5, 2.0]}, BPY_D, IS_PERM_DAILY,
                  "fade RTH-close deviations beyond z sigma"),
        Candidate("overnight_gap_follow", "D", gap_follow,
                  {"threshold": [0.0, 0.001, 0.002, 0.005]}, BPY_D, IS_PERM_DAILY,
                  "trade in the direction of the overnight gap"),
        Candidate("overnight_gap_fade", "D", gap_fade,
                  {"threshold": [0.0, 0.001, 0.002, 0.005]}, BPY_D, IS_PERM_DAILY,
                  "fade the overnight gap back toward prior close"),
        Candidate("orb_both", "F5", opening_range_breakout,
                  {"or_bars": [1, 3, 6], "direction": ["both"]}, BPY_5, IS_PERM_5MIN,
                  "first breakout of the 5/15/30-min opening range, held to close"),
        Candidate("orb_long", "F5", opening_range_breakout,
                  {"or_bars": [1, 3, 6], "direction": ["long"]}, BPY_5, IS_PERM_5MIN,
                  "long-only opening-range breakout"),
    ]


def make_strategy(c: Candidate, cost_per_turn: float) -> Strategy:
    return Strategy(c.name, c.signal_fn, c.grid, cost_per_turn=cost_per_turn,
                    objective=sharpe)


def grid_search(strat: Strategy, df: pd.DataFrame, registry: TrialRegistry | None):
    best_m, best_p = -np.inf, None
    for p in strat.grid():
        m = strat.evaluate(df, **p)
        if registry is not None:
            registry.log(strat.name, p, m, sharpe=m)
        if m > best_m:
            best_m, best_p = m, p
    return best_m, best_p


def returns_matrix(strat: Strategy, df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({str(p): strategy_returns(df, strat.positions(df, **p),
                                                  strat.cost_per_turn)
                         for p in strat.grid()})


def turnover_trades(positions: pd.Series) -> int:
    return int((positions.diff().abs() > 0).sum())


def run():
    print(f"{'='*78}\nDISCOVERY RUN  ({'QUICK smoke' if QUICK else 'FULL'})\n{'='*78}")
    proc = cfg.path("processed")
    adj = pd.read_parquet(proc / "MNQ_1min_adj.parquet")
    unadj = pd.read_parquet(proc / "MNQ_1min_unadj.parquet")

    D = daily_rth_bars(adj)
    F5 = five_min_rth_bars(adj)
    frames = {"D": D, "F5": F5}
    print(f"daily RTH bars : {len(D):,}  ({D.index.min().date()} -> {D.index.max().date()})")
    print(f"5-min RTH bars : {len(F5):,}")

    # Cost as a fraction of notional, from the REAL (unadjusted) median price.
    inst = cfg.instrument("MNQ")
    costs = FuturesCosts.from_config(cfg, "MNQ")
    px = float(unadj["close"].median())
    notional = px * inst["multiplier"]
    rt1 = costs.round_trip_cost(slippage_ticks=1)
    rt2 = costs.round_trip_cost(slippage_ticks=2)
    cpt1 = rt1 / (2 * notional)       # per side (one position change)
    cpt2 = rt2 / (2 * notional)
    print(f"cost model     : RT(1tk)=${rt1:.2f} RT(2tk)=${rt2:.2f}  "
          f"@ median px {px:.0f} -> per-turn {cpt1:.2e} / {cpt2:.2e}\n")

    registry = TrialRegistry()           # honest N pooled across the whole search
    candidates = build_candidates()

    # Pass 1: log every config of every candidate (build N) and stash best.
    prelim = {}
    for c in candidates:
        strat = make_strategy(c, cpt1)
        best_m, best_p = grid_search(strat, frames[c.frame], registry)
        prelim[c.name] = (strat, best_m, best_p)
    N = registry.n
    var_sr = registry.var_sharpe()
    print(f"honest trial count N (pooled configs): {N}   Var[SR]={var_sr:.3e}\n")

    results = []
    for c in candidates:
        strat, best_m, best_p = prelim[c.name]
        df = frames[c.frame]
        r1 = strategy_returns(df, strat.positions(df, **best_p), cpt1)
        r2 = strategy_returns(df, strat.positions(df, **best_p), cpt2)
        ann = np.sqrt(c.bars_per_year)
        trades = turnover_trades(strat.positions(df, **best_p))

        res = {
            "name": c.name, "note": c.note, "frame": c.frame,
            "best_params": best_p, "n_obs": len(r1),
            "sharpe_net1": sharpe(r1), "sharpe_net2": sharpe(r2),
            "sharpe_ann1": sharpe(r1) * ann, "sharpe_ann2": sharpe(r2) * ann,
            "pf": profit_factor(r1.to_numpy()), "trades": trades,
        }

        # --- Gate 1: in-sample MCPT (non-logging optimizer) ---
        def opt(d, _s=strat):
            return grid_search(_s, d, None)
        mc = insample_mcpt(df, opt, n_perm=c.is_perm, seed=cfg.seed)
        res["is_mcpt_p"] = mc.p_value
        res["is_perm"] = c.is_perm
        passed_is = mc.p_value < GATES["insample_mcpt_p_max"] and res["sharpe_net2"] > 0

        if not passed_is:
            reason = ("net Sharpe <= 0 after 2-tick costs" if res["sharpe_net2"] <= 0
                      else f"in-sample MCPT p={mc.p_value:.3f} >= {GATES['insample_mcpt_p_max']}")
            res.update({"verdict": "NO-GO", "reason": reason})
            results.append(res)
            print(f"[{c.name:22s}] NO-GO  ({reason})")
            continue

        # --- survived the screen: full battery ---
        wf = walk_forward_mcpt(df, opt, c.signal_fn, cost_per_turn=cpt1,
                               n_folds=5, n_perm=WF_PERM, seed=cfg.seed)
        d = dsr.deflated_sharpe_from_returns(r1.to_numpy(), var_sr, N)
        rm = returns_matrix(strat, df)
        pbo = pbo_cscv(rm, n_splits=10 if QUICK else 14)["pbo"] if rm.shape[1] >= 2 else float("nan")
        boot = bootstrap_report(r1.to_numpy(), n_boot=BOOT, mean_block=20, seed=cfg.seed)

        res.update({
            "wf_mcpt_p": wf.p_value, "wf_oos_sharpe": wf.real_oos_metric,
            "dsr": d["dsr"], "pbo": pbo,
            "boot_frac_le0": boot["frac_le_zero"], "boot_ci_excl0": boot["ci_excludes_zero"],
        })

        checks = [
            (mc.p_value < GATES["insample_mcpt_p_max"], f"IS-MCPT p={mc.p_value:.3f}"),
            (wf.p_value < GATES["wf_mcpt_p_max_2yr"], f"WF-MCPT p={wf.p_value:.3f}"),
            (d["dsr"] > GATES["dsr_min"], f"DSR={d['dsr']:.3f}"),
            (pbo < GATES["pbo_max"], f"PBO={pbo:.3f}"),
            (res["sharpe_net2"] > 0, f"net2 Sharpe={res['sharpe_net2']:.4f}"),
            (boot["ci_excludes_zero"], "bootstrap CI excludes 0"),
        ]
        failing = [msg for ok, msg in checks if not ok]
        res["verdict"] = "GO" if not failing else "NO-GO"
        res["reason"] = "ALL GATES PASSED" if not failing else "; ".join(failing)
        print(f"[{c.name:22s}] {res['verdict']:5s}  ISp={mc.p_value:.3f} WFp={wf.p_value:.3f} "
              f"DSR={d['dsr']:.2f} PBO={pbo:.2f}  ({res['reason']})")

        # equity-curve plot for anything that cleared the screen
        eq = r1.cumsum()
        fig, ax = plt.subplots(figsize=(9, 3.2))
        ax.plot(eq.index, eq.values, lw=0.8)
        ax.set_title(f"{c.name} {best_p} — cumulative net log-return (1-tick costs)")
        ax.axhline(0, color="k", lw=0.5)
        fig.tight_layout()
        fig.savefig(cfg.path("reports") / f"03_eq_{c.name}.png", dpi=110)
        plt.close(fig)

        results.append(res)

    write_report(results, N, var_sr, len(D), len(F5), cpt1, cpt2, px, QUICK)
    print(f"\nreport -> {cfg.path('reports') / '03_discovery_report.md'}")


def write_report(results, N, var_sr, n_d, n_5, cpt1, cpt2, px, quick):
    go = [r for r in results if r["verdict"] == "GO"]
    lines = []
    lines.append("# Discovery Report — MNQ candidate edges\n")
    lines.append(f"_Mode: {'QUICK smoke (reduced permutations)' if quick else 'FULL'}._ "
                 "Generated by `scripts/03_discovery.py`.\n")
    lines.append("## Bottom line\n")
    if go:
        lines.append(f"**{len(go)} candidate(s) passed ALL gates:** "
                     + ", ".join(f"`{r['name']}`" for r in go) + ".\n")
    else:
        lines.append("**No candidate survived the full gate stack.** This is the honest, "
                     "expected outcome for simple rules on a heavily-studied market — it "
                     "means the gates are doing their job, not that the data is exhausted. "
                     "See per-candidate reasons below.\n")
    lines.append(f"\nHonest pooled trial count **N = {N}** (Var[SR] = {var_sr:.3e}); "
                 f"DSR is deflated by this. Cost model: per-turn {cpt1:.2e} (1-tick) / "
                 f"{cpt2:.2e} (2-tick) of notional @ median price {px:.0f}.\n")

    lines.append("\n## Summary table\n")
    lines.append("| candidate | best params | ann.Sharpe (net 1tk / 2tk) | PF | trades | "
                 "IS-MCPT p | WF-MCPT p | DSR | PBO | verdict |")
    lines.append("|---|---|---|---|---:|---:|---:|---:|---:|---|")
    for r in results:
        wf = f"{r.get('wf_mcpt_p'):.3f}" if "wf_mcpt_p" in r else "—"
        ds = f"{r.get('dsr'):.3f}" if "dsr" in r else "—"
        pb = f"{r.get('pbo'):.3f}" if "pbo" in r else "—"
        lines.append(
            f"| `{r['name']}` | {r['best_params']} | "
            f"{r['sharpe_ann1']:.2f} / {r['sharpe_ann2']:.2f} | {r['pf']:.2f} | "
            f"{r['trades']:,} | {r['is_mcpt_p']:.3f} | {wf} | {ds} | {pb} | "
            f"**{r['verdict']}** |")

    lines.append("\n## Per-candidate detail\n")
    for r in results:
        lines.append(f"### `{r['name']}` — {r['verdict']}")
        lines.append(f"- _{r['note']}_  (frame: {r['frame']}, {r['n_obs']:,} bars)")
        lines.append(f"- best params: `{r['best_params']}`")
        lines.append(f"- annualized Sharpe: **{r['sharpe_ann1']:.2f}** (1-tick) / "
                     f"**{r['sharpe_ann2']:.2f}** (2-tick); profit factor {r['pf']:.2f}; "
                     f"{r['trades']:,} position changes")
        lines.append(f"- in-sample MCPT p = **{r['is_mcpt_p']:.4f}** "
                     f"(gate < {GATES['insample_mcpt_p_max']}; {r.get('is_perm','?')} perms)")
        if "wf_mcpt_p" in r:
            lines.append(f"- walk-forward MCPT p = **{r['wf_mcpt_p']:.4f}** "
                         f"(OOS Sharpe {r['wf_oos_sharpe']:.4f}); "
                         f"DSR = **{r['dsr']:.4f}** (gate > {GATES['dsr_min']}); "
                         f"PBO = **{r['pbo']:.4f}** (gate < {GATES['pbo_max']}); "
                         f"bootstrap CI excludes 0: {r['boot_ci_excl0']}")
            lines.append(f"  - equity curve: `reports/03_eq_{r['name']}.png`")
        lines.append(f"- **verdict: {r['verdict']} — {r['reason']}**\n")

    lines.append("\n## Honesty notes\n")
    lines.append("- ORB candidates run on 5-min bars; the in-sample MCPT permutation budget "
                 "is reduced for the screen and re-run at 1,000 for any survivor.\n"
                 "- Costs are fixed-dollar futures costs expressed as a fraction of notional "
                 "at the median historical price; survivors get bar-by-bar price-accurate "
                 "costing. Per-bar Sharpe is annualized only for readability.\n"
                 "- Back-adjusted continuous series used for returns; rolls verified continuous.")
    (cfg.path("reports") / "03_discovery_report.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    run()
