"""Data availability + ingest smoke-check.

Run this any time to see what data the pipeline can currently see:
  * NinjaTrader DAY .ncd coverage (decoded reliably; used as a validation oracle)
  * Any CSVs you've exported into data/nt_csv/  (the primary research feed)
It validates each CSV it finds (parses, checks OHLC invariants, prints coverage).

Usage:
    .venv/Scripts/python.exe scripts/00_check_data.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from edge.config import load_config
from edge.data import sessions
from edge.io import nt_csv
from edge.io import ncd_reader


def _hr():
    print("-" * 72)


def check_ncd_day(cfg):
    db = cfg.path("ninjatrader_db")
    day = db / "day"
    print(f"[NinjaTrader DAY .ncd]  {day}")
    if not day.exists():
        print("  (not found)")
        return
    contracts = sorted(p.name for p in day.iterdir() if p.is_dir())
    roots = sorted({c.split()[0] for c in contracts})
    print(f"  {len(contracts)} contract folders across symbols: {', '.join(roots)}")
    for sym in ("MNQ", "MES"):
        cs = [c for c in contracts if c.startswith(sym + " ")]
        if not cs:
            continue
        try:
            df = ncd_reader.read_day_contract(day / cs[-1])
            print(f"  {sym}: latest contract {cs[-1]} -> {len(df)} daily bars "
                  f"[{df.index.min().date()} .. {df.index.max().date()}]")
        except Exception as e:  # pragma: no cover - diagnostic only
            print(f"  {sym}: decode error: {e}")


def check_csv(cfg):
    d = cfg.path("nt_csv_export")
    print(f"[Exported CSVs]  {d}")
    files = sorted([*d.glob("*.csv"), *d.glob("*.txt")])
    if not files:
        print("  (none yet — see README 'Getting your data in')")
        return
    src_tz = cfg["timezone"]["source_tz"]
    exch = cfg["timezone"]["exchange_tz"]
    for f in files:
        try:
            df = nt_csv.read_nt_csv(f, source_tz=src_tz, exchange_tz=exch)
            df = sessions.tag_sessions(df)
            ncd_reader._validate_ohlc(df, source=f.name)
            span = f"{df.index.min()} .. {df.index.max()}"
            rth = int((df['session'] == 'RTH').sum())
            print(f"  OK  {f.name}: {len(df):,} bars  [{span}]  RTH={rth:,}")
        except Exception as e:
            print(f"  FAIL {f.name}: {e}")


def main():
    cfg = load_config()
    cfg.ensure_dirs()
    _hr(); check_ncd_day(cfg)
    _hr(); check_csv(cfg)
    _hr()


if __name__ == "__main__":
    main()
