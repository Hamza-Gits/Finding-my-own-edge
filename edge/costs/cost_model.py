"""Transaction-cost model for futures and FX.

Costs are the single biggest killer of intraday edges, so they are modeled
explicitly and every backtest is reported NET of them. Futures cost per
round-trip per contract = commission + slippage (in ticks -> dollars). The
slippage is stress-tested at 2 ticks per the spec.

Dollar P&L for a futures position:
    pnl_$ = points_moved * multiplier * contracts  -  round_trip_cost * contracts
where round_trip_cost = commission_rt + 2 * slippage_ticks * tick_value
(one tick of slippage on entry AND on exit => factor 2).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FuturesCosts:
    symbol: str
    multiplier: float
    tick_size: float
    tick_value: float
    commission_rt: float = 0.50      # $ round-trip per micro
    slippage_ticks: float = 1.0      # per side

    def round_trip_cost(self, slippage_ticks: float | None = None) -> float:
        """Total $ cost to open+close ONE contract (commission + entry+exit slippage)."""
        s = self.slippage_ticks if slippage_ticks is None else slippage_ticks
        return self.commission_rt + 2.0 * s * self.tick_value

    def trade_pnl(self, entry: float, exit: float, side: int, contracts: int = 1,
                  slippage_ticks: float | None = None) -> float:
        """Net $ P&L of a round-trip trade. side=+1 long, -1 short."""
        gross = (exit - entry) * side * self.multiplier * contracts
        return gross - self.round_trip_cost(slippage_ticks) * contracts

    @classmethod
    def from_config(cls, cfg, symbol: str) -> "FuturesCosts":
        spec = cfg.instrument(symbol)
        fc = cfg["costs"]["futures"]
        return cls(
            symbol=symbol,
            multiplier=float(spec["multiplier"]),
            tick_size=float(spec["tick_size"]),
            tick_value=float(spec["tick_value"]),
            commission_rt=float(fc["commission_rt_per_micro"]),
            slippage_ticks=float(fc["slippage_ticks"]),
        )


def per_bar_cost_in_return(costs: FuturesCosts, price: float, contracts: int = 1,
                           slippage_ticks: float | None = None) -> float:
    """Round-trip cost expressed as a fraction of notional (for bar-level returns).

    notional = price * multiplier * contracts. Useful when objective functions
    operate on returns rather than dollars.
    """
    notional = price * costs.multiplier * contracts
    if notional <= 0:
        raise ValueError("non-positive notional")
    return costs.round_trip_cost(slippage_ticks) * contracts / notional
