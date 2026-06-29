"""Tests for the NinjaTrader .ncd DAY reader (the verified, uncompressed format).

These run against the real on-disk data when present; skipped otherwise so the
suite stays green on machines without the NinjaTrader store.
"""
import struct
from pathlib import Path

import pytest

from edge.io import ncd_reader

DAY_DIR = Path(r"C:/Users/hamza/Documents/NinjaTrader 8/db/day/MNQ 09-26")
HAVE_DATA = DAY_DIR.exists() and any(DAY_DIR.glob("*.ncd"))


@pytest.mark.skipif(not HAVE_DATA, reason="NinjaTrader day data not present")
def test_day_header_and_first_record():
    f = sorted(DAY_DIR.glob("*.ncd"))[0]
    header = ncd_reader.read_ncd_header(f.read_bytes())
    assert header["version"] == 1
    assert header["tick_size"] == 0.25
    # base_price anchors the first record's open.
    assert header["base_price"] == pytest.approx(29740.25, abs=1e-6)


@pytest.mark.skipif(not HAVE_DATA, reason="NinjaTrader day data not present")
def test_day_decode_known_values():
    df = ncd_reader.read_day_contract(DAY_DIR)
    # Known-good decoded values for 2026-06-12 (verified by hand from the bytes).
    row = df[df.index.date == __import__("datetime").date(2026, 6, 12)].iloc[0]
    assert row["open"] == pytest.approx(29740.25)
    assert row["high"] == pytest.approx(30053.00)
    assert row["low"] == pytest.approx(29518.25)
    assert row["close"] == pytest.approx(29954.75)
    assert row["volume"] == pytest.approx(519731)


@pytest.mark.skipif(not HAVE_DATA, reason="NinjaTrader day data not present")
def test_day_ohlc_invariants_hold_everywhere():
    df = ncd_reader.read_day_contract(DAY_DIR)
    assert (df["high"] >= df[["open", "close"]].max(axis=1) - 1e-6).all()
    assert (df["low"] <= df[["open", "close"]].min(axis=1) + 1e-6).all()
    assert (df["volume"] > 0).all()
    assert df.index.is_monotonic_increasing


def test_minute_decode_refuses_until_validated(tmp_path):
    # A correct, fake 28-byte header + junk payload; must refuse, not guess.
    blob = struct.pack("<i", 1) + struct.pack("<d", 0.25) + struct.pack("<d", 100.0)
    blob += struct.pack("<q", 639164736600000000) + b"\x21\x51\x00\x18"
    p = tmp_path / "x.Last.ncd"
    p.write_bytes(blob)
    with pytest.raises(ncd_reader.MinuteDecoderNotValidated):
        ncd_reader.read_minute_ncd(p)
