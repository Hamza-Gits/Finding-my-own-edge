"""Configuration loader. Single source of truth = config/config.yaml.

Resolves relative pipeline paths against the project root and exposes a small
typed accessor so modules never hard-code constants.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "config.yaml"

# Pipeline-owned paths are made absolute relative to PROJECT_ROOT; external data
# sources (NinjaTrader/MT5) are absolute already.
_PIPELINE_PATH_KEYS = {"nt_csv_export", "raw", "interim", "processed", "reports"}


@dataclass
class Config:
    raw: dict[str, Any]

    def __getitem__(self, key: str) -> Any:
        return self.raw[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.raw.get(key, default)

    # convenience accessors -------------------------------------------------
    @property
    def seed(self) -> int:
        return int(self.raw["seed"])

    def path(self, key: str) -> Path:
        p = Path(self.raw["paths"][key])
        if key in _PIPELINE_PATH_KEYS and not p.is_absolute():
            p = PROJECT_ROOT / p
        return p

    def instrument(self, symbol: str) -> dict[str, Any]:
        return self.raw["instruments"][symbol]

    def ensure_dirs(self) -> None:
        for key in _PIPELINE_PATH_KEYS:
            self.path(key).mkdir(parents=True, exist_ok=True)


def load_config(path: str | os.PathLike | None = None) -> Config:
    path = Path(path) if path else DEFAULT_CONFIG
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    return Config(raw=raw)
