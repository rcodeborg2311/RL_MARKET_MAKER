"""Tests for fill simulator."""
import pytest
from mmenv.simulator import FillSimulator


@pytest.fixture
def sim() -> FillSimulator:
    return FillSimulator()


def test_no_fill_when_bid_below_all_trades(sim: FillSimulator):
    """No fill when agent bid is below all sell trade prices."""
    trades = [
        {"price": 105.0, "qty": 1.0, "side": "sell"},
        {"price": 106.0, "qty": 0.5, "side": "sell"},
    ]
    bid_f, ask_f = sim.simulate_fills(100.0, 1.0, 110.0, 1.0, trades)
    assert bid_f == pytest.approx(0.0)


def test_full_fill_bid_above_aggressive_sell(sim: FillSimulator):
    """Full fill when agent bid >= aggressive sell price."""
    trades = [{"price": 100.0, "qty": 2.0, "side": "sell"}]
    bid_f, _ = sim.simulate_fills(101.0, 1.0, 105.0, 1.0, trades)
    assert bid_f == pytest.approx(1.0)  # capped at agent_bid_qty


def test_partial_fill_when_trade_qty_less_than_agent(sim: FillSimulator):
    """Partial fill when trade_qty < agent_qty."""
    trades = [{"price": 99.0, "qty": 0.3, "side": "sell"}]
    bid_f, _ = sim.simulate_fills(100.0, 1.0, 105.0, 1.0, trades)
    assert bid_f == pytest.approx(0.3)


def test_ask_fill_aggressive_buy(sim: FillSimulator):
    """Ask fill when aggressive buy trade >= ask price."""
    trades = [{"price": 106.0, "qty": 1.0, "side": "buy"}]
    _, ask_f = sim.simulate_fills(100.0, 1.0, 105.0, 1.0, trades)
    assert ask_f == pytest.approx(1.0)


def test_no_ask_fill_when_trade_below_ask(sim: FillSimulator):
    """No ask fill when buy trade price < ask price."""
    trades = [{"price": 104.0, "qty": 1.0, "side": "buy"}]
    _, ask_f = sim.simulate_fills(100.0, 1.0, 105.0, 1.0, trades)
    assert ask_f == pytest.approx(0.0)
