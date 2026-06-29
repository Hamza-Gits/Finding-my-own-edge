"""Tests for the cost model arithmetic."""
import pytest

from edge.config import load_config
from edge.costs.cost_model import FuturesCosts, per_bar_cost_in_return


def test_mnq_round_trip_cost_default():
    # MNQ tick_value $0.50; default 1 tick/side slippage + $0.50 commission RT.
    c = FuturesCosts("MNQ", multiplier=2.0, tick_size=0.25, tick_value=0.50)
    # commission 0.50 + 2 sides * 1 tick * 0.50 = 0.50 + 1.00 = 1.50
    assert c.round_trip_cost() == pytest.approx(1.50)
    # stress at 2 ticks/side: 0.50 + 2*2*0.50 = 2.50
    assert c.round_trip_cost(slippage_ticks=2) == pytest.approx(2.50)


def test_mes_round_trip_cost_default():
    c = FuturesCosts("MES", multiplier=5.0, tick_size=0.25, tick_value=1.25)
    # 0.50 + 2*1*1.25 = 3.00
    assert c.round_trip_cost() == pytest.approx(3.00)


def test_trade_pnl_long_net_of_costs():
    c = FuturesCosts("MNQ", multiplier=2.0, tick_size=0.25, tick_value=0.50)
    # +4 points on MNQ = 4*2 = $8 gross; minus $1.50 RT = $6.50 net.
    assert c.trade_pnl(entry=20000, exit=20004, side=+1) == pytest.approx(6.50)
    # short losing the same move
    assert c.trade_pnl(entry=20000, exit=20004, side=-1) == pytest.approx(-9.50)


def test_four_tick_scalp_haircut():
    # A 4-tick MNQ scalp grosses 4*0.50 = $2.00; 1-tick+comm cost = $1.50 -> keeps $0.50.
    c = FuturesCosts("MNQ", multiplier=2.0, tick_size=0.25, tick_value=0.50)
    gross = 4 * c.tick_value
    net = gross - c.round_trip_cost()
    assert net == pytest.approx(0.50)
    assert net / gross == pytest.approx(0.25)  # 75% haircut


def test_from_config_roundtrip():
    cfg = load_config()
    c = FuturesCosts.from_config(cfg, "MES")
    assert c.tick_value == pytest.approx(1.25)
    assert c.round_trip_cost() == pytest.approx(3.00)


def test_per_bar_cost_in_return_positive():
    c = FuturesCosts("MNQ", multiplier=2.0, tick_size=0.25, tick_value=0.50)
    r = per_bar_cost_in_return(c, price=20000.0)
    assert 0 < r < 1e-3
