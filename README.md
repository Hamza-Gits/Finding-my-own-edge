# Finding a New Edge — intraday edge-discovery & validation pipeline

A disciplined, **intellectually honest** pipeline that discovers candidate
intraday edges on MNQ/MES and puts each through hard statistical gates (Masters
permutation tests, walk-forward MCPT, Deflated Sharpe, PBO, purged CPCV) so that
only edges with **real, cost-survivable expectancy** are ever proposed.
**If nothing survives, it says so.** The prime directive is honest measurement,
not hitting a target number.

- Full research spec: `compass_artifact_*.md`
- Running progress log: [`CHANGELOG.md`](CHANGELOG.md)
- Data assessment: [`reports/01_data_coverage.md`](reports/01_data_coverage.md)

---

## Status (2026-06-29)

| Stage | State |
|---|---|
| M0 Environment + scaffold | ✅ done |
| **M1 Data spine (MNQ/MES 1-min)** | ✅ **done — real data loaded** |
| M2 Gate machinery | 🟡 core proven (MCPT + DSR); walk-forward / PBO / CPCV in progress |
| M3 Feature library + discovery | ⬜ next |
| M4 Portfolio + 50K-Flex eval sim | ⬜ |
| M5 FX add-on + reporting | ⬜ |

**Data loaded:** MNQ 2.5M bars (2019→2026, ~7.1y), MES 1.43M bars (~6y), 1-min,
UTC→CME-Central, back-adjusted continuous + unadjusted, roll continuity verified.

```
.venv/Scripts/python.exe -m pytest -q          # test suite (26 passing)
.venv/Scripts/python.exe scripts/01_ingest.py  # rebuild continuous bars from data/raw/
.venv/Scripts/python.exe scripts/demo_gate.py  # watch the gate pass an edge / kill noise
```

---

## How the data got here

Raw history was exported from NinjaTrader 8 as per-contract CSVs
(`SYM MM-YY.Last.txt`, `yyyyMMdd HHmmss;O;H;L;C;V`, UTC) and staged in `data/raw/`
(gitignored — licensed vendor data; full provenance in `data/MANIFEST.csv`).
`scripts/01_ingest.py` localizes UTC→America/Chicago, validates OHLC invariants,
stitches a volume-roll continuous series (back-adjusted + unadjusted), tags
RTH/ETH sessions, and writes parquet to `data/processed/`.

### Optional future exports (not blocking)
- `NQ ##-##` + `ES ##-##` minute (full history) → enables true ES–NQ intermarket edges.
- `MNQ` tick (recent ~1y) → enables genuine microstructure features (VPIN, Kyle's λ).
- MT5 FX export → Phase 5.

---

## The gates (why a survivor is believable)

Every candidate must pass ALL of these, with thresholds fixed in `config/config.yaml`
and **never loosened** to manufacture a survivor:

1. **In-sample MCPT** (Masters) — re-runs the *full optimization* on ≥1,000 bar
   permutations; reject unless p < 0.01. Prices in data-mining bias.
2. **Walk-forward MCPT** — permute only OOS data; reject unless p < 0.05 (1y) / 0.01 (2y+).
3. **Deflated Sharpe Ratio** — deflated by the honest trial count N; require DSR > 0.95.
4. **PBO via CSCV** — probability of backtest overfitting < 0.20.
5. **Cost sensitivity** — must survive 1- and 2-tick slippage.

The machinery is self-validated: it flags a planted AR(1) edge (GO) and kills a
data-mined random walk (NO-GO). See `scripts/demo_gate.py`.

---

## Layout

```
config/config.yaml   all paths, specs, costs, eval rules, gate thresholds, seeds
edge/                package: io, data, costs, features, validation, stats, portfolio, report
scripts/             01_ingest -> (discovery) -> gates -> portfolio_eval
tests/               pytest suite (data layer, costs, sessions, gates)
data/raw/            raw NinjaTrader exports (gitignored; see data/MANIFEST.csv)
data/processed/      continuous back-adjusted/unadjusted parquet (gitignored, reproducible)
reports/             run artifacts: coverage, ranked tables, equity plots, GO/NO-GO verdicts
```
