# Changelog

Running log of the edge-discovery pipeline. Newest first. Every entry is a real,
verifiable change — no aspirational claims.

## 2026-06-29

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
