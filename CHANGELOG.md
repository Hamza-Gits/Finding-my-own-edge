# Changelog

Running log of the edge-discovery pipeline. Newest first. Every entry is a real,
verifiable change — no aspirational claims.

## 2026-06-30

### H1 Compression-Gated ORB — fully gated, honest NO-GO (+ new per-year leg gate)
- Ran H1 (compression-gated opening-range breakout) through the **full** gate stack on real MNQ
  (`scripts/04_compression_orb.py`, 500 IS perms). Mechanism: arm the breakout only after a quiet
  overnight; direction chosen intraday so it is structurally symmetric (not bull beta).
- **Passed** IS-MCPT (p=0.0040, only 1/500 perms beat it), 1,350 trades, ann Sharpe 1.16/1.09
  (1-/2-tick). **Failed** WF-MCPT (p=0.192) and PBO (0.44) → classic overfit signature: great
  in-sample, parameters do not transfer out-of-sample. **Honest NO-GO.**
- **Adversarial code audit (16-agent workflow):** 5 independent skeptics traced the arming and
  intraday logic on the real ns-resolution data and found **no lookahead/leak** — so the 1.1 Sharpe
  is real, and the OOS failure is genuine overfitting, not a bug. (One non-blocking robustness note:
  the `comp_map` key would mismatch under a future pandas-3 µs-resolution index — harden later.)
- **New honesty gate:** `edge/validation/leg_gate.py` — per-year long/short leg decomposition,
  replacing the too-weak pooled `min(pnl_long, pnl_short) > 0`. On H1 it **FAILS** where the pooled
  check passed: the short leg books **71% of its P&L in 2022** alone and is negative in 3/8 years;
  dropping 2022 collapses net Sharpe 1.09 → 0.79. `scripts/04b_h1_leg_decomposition.py` reproduces it.
- This gate is now applied **before** the expensive full-gate stack for every symmetric candidate.
- Next: **H4 — overnight-inventory gap-fade** (sign-symmetric, abundant sample, flat overnight),
  with mandatory roll-gap handling and the per-year leg gate. See `reports/04_hypotheses.md`.

## 2026-06-29

### M3 first discovery run — 6 candidates, honest NO-GO (gates working)
- Feature/candidate layer: `edge/data/bars.py` (daily-RTH + 5-min-RTH aggregation),
  `edge/features/seasonality.py` (opening-range breakout, overnight-gap follow/fade,
  day-of-week). `scripts/03_discovery.py` runs each candidate through the FULL gate
  stack with a shared honest trial registry; emits `reports/03_discovery_report.md`
  + equity plots.
- Candidates tested on real MNQ (2019→2026): daily momentum, daily mean-reversion,
  overnight-gap follow, overnight-gap fade, **5-min opening-range breakout (both &
  long-only)** — the latter being the user's headline "5 Min ORB".
- **Result: every candidate is NO-GO.** Attractive-looking annualized Sharpes
  (0.4–0.9) and PFs up to 1.50 all collapse under the in-sample MCPT (p ≈ 0.23–0.74,
  none below the 0.01 gate). The standout lesson: ORB-long shows ~0.87 annualized
  Sharpe but p≈0.28 — that "edge" is just 2019–2026 bull-market drift captured by a
  long-biased rule (beta, not predictive structure), and the permutation test sees
  through it exactly as Masters intended. `overnight_gap_follow` dies on cost
  fragility (net Sharpe ≤ 0 after 2-tick slippage).
- This is the prime directive in action: the pipeline HONESTLY kills plausible-looking
  fakes. No threshold was loosened. 5 new feature tests (39 total passing).

### Real data ingested (M1 data spine complete on futures)
- Staged **48 raw NinjaTrader contract exports** (235 MB) into `data/raw/` (gitignored;
  provenance in `data/MANIFEST.csv`): MNQ ×30 (2019→2026), MES ×17, ES ×1.
- **Confirmed source timezone = UTC empirically** (not assumed): localizing under UTC puts
  the daily Globex reopen at 17:00 CT and the RTH volume peak at 08:30 CT across winter,
  summer, and multiple years. Other candidate zones (London/NY/Chicago) all misalign.
- Built `edge/data/rolls.py` — continuous-contract stitching via **volume-crossover rolls**,
  emitting **both** a back-adjusted (Panama, additive) series and a raw unadjusted series,
  plus an auditable roll schedule.
- `scripts/01_ingest.py` → continuous, session-tagged 1-min parquet for MNQ/MES/ES:
  - **MNQ: 2,503,028 bars, 2019-05-05 → 2026-06-22, 29 rolls**
  - **MES: 1,430,354 bars, 2020-06-07 → 2026-06-22, 16 rolls**
- **Roll continuity verified**: adj series moves ~0.0005 at roll boundaries (normal bar) vs
  up to 1.3% unadjusted — back-adjustment is removing the synthetic jumps, not creating them.
- Data assessment written to `reports/01_data_coverage.md` (answers "will this work": yes for
  futures-first; ES/NQ intermarket and tick microstructure flagged as honest gaps).
- Repo initialized and connected to GitHub (`Hamza-Gits/Finding-my-own-edge`).

### M2 gate machinery complete
- `edge/validation/walkforward.py` — anchored/rolling walk-forward (warm-up via train
  tail, no leakage) + **walk-forward MCPT** (keep first train real, permute the rest,
  re-run the whole WF per permutation; p-gate 0.05/0.01).
- `edge/stats/cv.py` — purged k-fold + embargo and **CPCV** (C(N,k) partitions → an OOS
  metric *distribution*, not one lucky split).
- `edge/stats/pbo.py` — **PBO via CSCV** (Bailey et al.); fraction of splits where the
  IS-best variant lands in the bottom half OOS. Gate < 0.20.
- `edge/validation/bootstrap.py` — **stationary block bootstrap** cross-check (preserves
  serial dependence the permutation destroys); survivors must agree across both.
- **Vectorized the bar-permutation reconstruction** (was a Python loop): 50k-bar permute
  now 36 ms; the WF-MCPT test that took 11 min now runs in seconds. This is what makes
  1,000-permutation tests on real intraday data tractable.
- New gate self-tests pass: all four show power on a planted edge and correct size on
  noise. **34 tests passing.**

### Previously (M0 + M2 core, pre-changelog)
- Environment + package scaffold (`edge/`), pinned quant stack (pandas 2.2.3 to dodge a
  Windows tz `date_range` segfault), central `config/config.yaml`.
- NinjaTrader `.ncd` **day** reader cracked & validated; minute/tick binary deliberately
  NOT guessed (refuses with a clear error) — CSV export is the trusted path instead.
- Validation gate machinery (core): bar-permutation engine, **in-sample MCPT** (Masters),
  **Deflated Sharpe / PSR / MinTRL** (reproduces the Bailey–López de Prado worked example
  bit-for-bit), trial registry, cost model, sessions.
- Gate self-validation: detects a planted AR(1) edge (GO) and kills a data-mined random
  walk (NO-GO). 26 tests passing.

## Pending (next)
- Finish M2 gates: walk-forward MCPT, block bootstrap, purged CPCV + embargo, PBO via CSCV.
- M3: feature library + discovery loop on real MNQ (ORB, momentum, mean-reversion,
  overnight-gap, time-of-day, MES–MNQ spread) through the full gate stack.
- M4: low-correlation portfolio + 50K-Flex eval simulator (both floor mechanics).
