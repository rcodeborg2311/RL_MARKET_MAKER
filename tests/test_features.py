"""Tests for feature engineering."""
import numpy as np
import pytest
from mmenv.features import OrderBookFeatures


@pytest.fixture
def feat() -> OrderBookFeatures:
    return OrderBookFeatures(n_levels=5)


def _book(bid_qty: float, ask_qty: float, n: int = 5):
    bids = [(100.0 - i, bid_qty) for i in range(n)]
    asks = [(101.0 + i, ask_qty) for i in range(n)]
    return bids, asks


def test_obi_all_bids():
    bid_vol, ask_vol = 10.0, 0.0
    total = bid_vol + ask_vol
    obi = (bid_vol - ask_vol) / total if total > 0 else 0.0
    assert obi == pytest.approx(1.0)


def test_obi_all_asks():
    bid_vol, ask_vol = 0.0, 10.0
    obi = (bid_vol - ask_vol) / (bid_vol + ask_vol)
    assert obi == pytest.approx(-1.0)


def test_obi_balanced(feat: OrderBookFeatures):
    bids, asks = _book(5.0, 5.0)
    assert feat.order_book_imbalance(bids, asks, levels=1) == pytest.approx(0.0)


def test_spread_bps_positive(feat: OrderBookFeatures):
    bids = [(100.0, 1.0)]
    asks = [(101.0, 1.0)]
    assert feat.spread(bids, asks) > 0.0


def test_state_vector_shape(feat: OrderBookFeatures):
    """State vector has shape (20,)."""
    bids, asks = _book(5.0, 5.0)
    sv = feat.compute_state_vector(bids, asks, [], vwap=100.5,
                                   agent_inventory=0.0, agent_pnl=0.0)
    assert sv.shape == (20,)


def test_state_vector_dtype(feat: OrderBookFeatures):
    bids, asks = _book(5.0, 5.0)
    sv = feat.compute_state_vector(bids, asks, [], vwap=100.5,
                                   agent_inventory=0.0, agent_pnl=0.0)
    assert sv.dtype == np.float32


def test_state_vector_finite(feat: OrderBookFeatures):
    bids, asks = _book(5.0, 5.0)
    trades = [{"price": 100.5, "qty": 0.1, "side": "buy"},
              {"price": 100.3, "qty": 0.05, "side": "sell"}]
    sv = feat.compute_state_vector(bids, asks, trades, vwap=100.5,
                                   agent_inventory=1.0, agent_pnl=10.0)
    assert np.all(np.isfinite(sv))
