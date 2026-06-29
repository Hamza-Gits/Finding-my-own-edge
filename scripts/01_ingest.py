"""Ingest raw per-contract NinjaTrader exports -> continuous, session-tagged bars.

Pipeline (per instrument family found in data/raw/):
  1. read each 'SYM MM-YY.Last.txt'  (UTC timestamps -> America/Chicago)
  2. validate OHLC invariants, fail loud on corruption
  3. record provenance to data/MANIFEST.csv (contract, span, n_bars, sha256)
  4. stitch a continuous front-month series (volume-crossover roll):
        <SYM>_1min_adj.parquet    (back-adjusted; returns/indicators/backtests)
        <SYM>_1min_unadj.parquet  (raw stitched; absolute-price logic)
        <SYM>_roll_schedule.csv   (auditable roll log)
  5. print a coverage report.

    .venv/Scripts/python.exe scripts/01_ingest.py            # all instruments
    .venv/Scripts/python.exe scripts/01_ingest.py MNQ MES    # subset
"""
from __future__ import annotations

import hashlib
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from edge.config import load_config
from edge.io.nt_csv import read_nt_csv
from edge.data.rolls import build_continuous, parse_contract
from edge.data.sessions import tag_sessions

_OHLC = ["open", "high", "low", "close"]


def _sha256(path: Path, _buf: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(_buf), b""):
            h.update(chunk)
    return h.hexdigest()


def _validate_ohlc(df: pd.DataFrame, label: str) -> None:
    hi = df[["open", "close"]].max(axis=1)
    lo = df[["open", "close"]].min(axis=1)
    bad_hi = (df["high"] < hi - 1e-9).sum()
    bad_lo = (df["low"] > lo + 1e-9).sum()
    bad_v = (df["volume"] < 0).sum()
    if bad_hi or bad_lo or bad_v:
        raise ValueError(
            f"{label}: OHLC invariants violated "
            f"(high<max {bad_hi}, low>min {bad_lo}, vol<0 {bad_v})"
        )


def _group_files(raw_dir: Path) -> dict[str, list[Path]]:
    groups: dict[str, list[Path]] = defaultdict(list)
    for f in sorted(raw_dir.glob("*.txt")):
        sym = f.name.split(" ")[0]
        groups[sym].append(f)
    return groups


def main(argv: list[str]) -> None:
    cfg = load_config()
    cfg.ensure_dirs()
    raw_dir = cfg.path("raw")
    proc_dir = cfg.path("processed")
    src_tz = cfg["timezone"]["source_tz"]
    exch_tz = cfg["timezone"]["exchange_tz"]

    groups = _group_files(raw_dir)
    wanted = [s.upper() for s in argv] or sorted(groups)
    print(f"raw dir : {raw_dir}")
    print(f"tz      : {src_tz} -> {exch_tz}")
    print(f"symbols : {', '.join(wanted)}\n")

    manifest_rows = []
    for sym in wanted:
        files = groups.get(sym, [])
        if not files:
            print(f"[{sym}] no files, skipping")
            continue
        frames: dict[str, pd.DataFrame] = {}
        for f in files:
            label = f.name.replace(".Last.txt", "")
            df = read_nt_csv(f, source_tz=src_tz, exchange_tz=exch_tz)
            _validate_ohlc(df, label)
            frames[label] = df
            manifest_rows.append({
                "symbol": sym,
                "contract": label,
                "expiry_proxy": parse_contract(label).expiry.date(),
                "first_bar": df.index.min(),
                "last_bar": df.index.max(),
                "n_bars": len(df),
                "file_bytes": f.stat().st_size,
                "sha256": _sha256(f),
            })

        cont = build_continuous(frames)
        adj = tag_sessions(cont.adj)
        unadj = cont.unadj

        adj.to_parquet(proc_dir / f"{sym}_1min_adj.parquet")
        unadj.to_parquet(proc_dir / f"{sym}_1min_unadj.parquet")
        cont.schedule.to_csv(proc_dir / f"{sym}_roll_schedule.csv", index=False)

        rth = (adj["session"] == "RTH").sum()
        span_days = (adj.index.max() - adj.index.min()).days
        print(f"[{sym}] {len(frames):2d} contracts | {len(adj):>9,} bars "
              f"| {adj.index.min().date()} -> {adj.index.max().date()} ({span_days}d) "
              f"| RTH {rth/len(adj):4.0%} | {len(cont.schedule)} rolls")

    if manifest_rows:
        man = pd.DataFrame(manifest_rows).sort_values(["symbol", "expiry_proxy"])
        man_path = ROOT / "data" / "MANIFEST.csv"
        man.to_csv(man_path, index=False)
        print(f"\nmanifest -> {man_path}  ({len(man)} contracts)")
        print(f"processed parquet -> {proc_dir}")


if __name__ == "__main__":
    main(sys.argv[1:])
