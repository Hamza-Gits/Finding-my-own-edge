"""Trial registry — logs EVERY candidate configuration evaluated.

The honest trial count N (and the dispersion of trial Sharpes) is what deflates
the Sharpe ratio. Under-counting trials inflates DSR and is a primary route to
self-deception, so every grid point, every strategy variant, every seasonality
bucket tested must be logged here. The registry can persist to JSONL so N
survives across sessions of searching.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class Trial:
    strategy: str
    params: dict
    metric: float
    sharpe: float | None = None


@dataclass
class TrialRegistry:
    path: Path | None = None
    trials: list[Trial] = field(default_factory=list)

    def log(self, strategy: str, params: dict, metric: float,
            sharpe: float | None = None) -> None:
        t = Trial(strategy=strategy, params=dict(params), metric=metric, sharpe=sharpe)
        self.trials.append(t)
        if self.path is not None:
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps({"strategy": t.strategy, "params": t.params,
                                     "metric": t.metric, "sharpe": t.sharpe}) + "\n")

    @property
    def n(self) -> int:
        return len(self.trials)

    def sharpes(self) -> np.ndarray:
        return np.array([t.sharpe for t in self.trials if t.sharpe is not None],
                        dtype=float)

    def var_sharpe(self) -> float:
        """Var[SR] across trials — the deflation input for DSR. 0 if < 2 trials."""
        s = self.sharpes()
        return float(s.var(ddof=1)) if s.size >= 2 else 0.0

    @classmethod
    def load(cls, path: str | Path) -> "TrialRegistry":
        path = Path(path)
        reg = cls(path=path)
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    d = json.loads(line)
                    reg.trials.append(Trial(d["strategy"], d["params"],
                                            d["metric"], d.get("sharpe")))
        return reg
