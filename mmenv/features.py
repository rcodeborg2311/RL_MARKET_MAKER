"""
Professional market microstructure feature engineering.
20-dimensional state vector used by the RL agent.
"""
import numpy as np
from collections import deque
from typing import List, Tuple


STATE_DIM = 20


class OrderBookFeatures:
    """
    Computes 20 microstructure features from L2 order book snapshots.

    State vector layout:
      [0]  OBI top-1                  order book imbalance at best level
      [1]  OBI all-N                  depth-weighted imbalance
      [2]  spread_bps                 best ask - best bid in bps
      [3]  wmid_norm                  (WMID/mid - 1), micro-price signal
      [4]  vwap_dev_bps               deviation of mid from rolling VWAP
      [5]  trade_imbalance            signed flow from aggressive trades
      [6-11] qty levels 1-3 bid/ask  normalized depth profile
      [12] agent_inventory            ∈ [-1, 1]
      [13] agent_pnl_norm             normalized running PnL
      [14] realized_vol               rolling 20-step log-return std
      [15] price_momentum             5-step mid log-return, scaled
      [16] spread_z                   (spread - μ_spread) / σ_spread
      [17] kyle_lambda                |Δmid| / traded_volume, price impact proxy
      [18] bid_depth_slope            how depth drops off across bid levels
      [19] ask_depth_slope            how depth drops off across ask levels
    """

    STATE_DIM = 20

    def __init__(self, n_levels: int = 5) -> None:
        self.n_levels = n_levels
        self._qty_history: deque = deque(maxlen=100)
        self._mid_history: deque = deque(maxlen=50)
        self._spread_history: deque = deque(maxlen=100)
        self._vol_history: deque = deque(maxlen=20)
        # For Kyle lambda: (|delta_mid|, total_volume) pairs
        self._impact_history: deque = deque(maxlen=20)
        self._prev_mid: float = 0.0

    def mid_price(self, bids: List[Tuple], asks: List[Tuple]) -> float:
        return (float(bids[0][0]) + float(asks[0][0])) / 2.0

    def spread(self, bids: List[Tuple], asks: List[Tuple]) -> float:
        """Spread in basis points."""
        b, a = float(bids[0][0]), float(asks[0][0])
        mid = (b + a) / 2.0
        if mid == 0:
            return 0.0
        return (a - b) / mid * 10_000.0

    def order_book_imbalance(self, bids: List[Tuple], asks: List[Tuple], levels: int = 1) -> float:
        bv = sum(float(bids[i][1]) for i in range(min(levels, len(bids))))
        av = sum(float(asks[i][1]) for i in range(min(levels, len(asks))))
        tot = bv + av
        return (bv - av) / tot if tot > 0 else 0.0

    def weighted_mid_price(self, bids: List[Tuple], asks: List[Tuple]) -> float:
        bp, bq = float(bids[0][0]), float(bids[0][1])
        ap, aq = float(asks[0][0]), float(asks[0][1])
        tot = bq + aq
        if tot == 0:
            return (bp + ap) / 2.0
        return (ap * bq + bp * aq) / tot

    def vwap_deviation(self, bids: List[Tuple], asks: List[Tuple], vwap: float) -> float:
        mid = self.mid_price(bids, asks)
        if vwap == 0:
            return 0.0
        return (mid - vwap) / vwap * 10_000.0

    def depth_imbalance(self, bids: List[Tuple], asks: List[Tuple]) -> float:
        return self.order_book_imbalance(bids, asks, levels=self.n_levels)

    def trade_imbalance(self, recent_trades: list, window: int = 50) -> float:
        trades = recent_trades[-window:]
        bv = sum(float(t.get("qty", t.get("size", 0))) for t in trades if t.get("side") == "buy")
        sv = sum(float(t.get("qty", t.get("size", 0))) for t in trades if t.get("side") == "sell")
        tot = bv + sv
        return (bv - sv) / tot if tot > 0 else 0.0

    def _depth_slope(self, levels: List[Tuple]) -> float:
        """
        Normalized slope of depth across levels.
        Positive = depth increasing away from best (typical healthy book).
        Negative = depth thinning (potential one-sided pressure).
        Returned as slope of linear fit / mean_qty, clipped to [-3, 3].
        """
        if len(levels) < 2:
            return 0.0
        qtys = np.array([float(lvl[1]) for lvl in levels[:self.n_levels]], dtype=np.float32)
        if qtys.mean() == 0:
            return 0.0
        xs = np.arange(len(qtys), dtype=np.float32)
        slope = float(np.polyfit(xs, qtys, 1)[0])
        return float(np.clip(slope / (qtys.mean() + 1e-8), -3.0, 3.0))

    def compute_state_vector(
        self,
        bids: List[Tuple],
        asks: List[Tuple],
        recent_trades: list,
        vwap: float,
        agent_inventory: float,
        agent_pnl: float,
        max_inventory: float = 10.0,
        capital: float = 10_000.0,
    ) -> np.ndarray:
        """Returns 20-dim float32 state vector."""
        # ── qty normalization ─────────────────────────────────────────────────
        raw_qtys = [float(bids[i][1]) for i in range(min(3, len(bids)))]
        raw_qtys += [float(asks[i][1]) for i in range(min(3, len(asks)))]
        self._qty_history.append(raw_qtys)
        all_qtys = [q for snap in self._qty_history for q in snap]
        qty_mean = float(np.mean(all_qtys)) if all_qtys else 1.0
        if qty_mean == 0:
            qty_mean = 1.0

        mid = self.mid_price(bids, asks)
        sp_bps = self.spread(bids, asks)
        wmid = self.weighted_mid_price(bids, asks)
        wmid_norm = (wmid / mid) - 1.0 if mid != 0 else 0.0

        # ── rolling history ───────────────────────────────────────────────────
        self._mid_history.append(mid)
        self._spread_history.append(sp_bps)

        # realized vol (20-step log-return std)
        if len(self._mid_history) >= 2:
            mids = np.array(list(self._mid_history), dtype=np.float64)
            log_rets = np.diff(np.log(mids + 1e-10))
            realized_vol = float(np.std(log_rets[-20:])) * 100.0  # scale to ~O(1)
        else:
            realized_vol = 0.001

        # 5-step price momentum
        if len(self._mid_history) >= 6:
            mids_arr = np.array(list(self._mid_history), dtype=np.float64)
            momentum = float(np.log((mids_arr[-1] + 1e-10) / (mids_arr[-6] + 1e-10))) * 1000.0
        else:
            momentum = 0.0

        # spread z-score
        if len(self._spread_history) >= 10:
            spreads = np.array(list(self._spread_history), dtype=np.float64)
            sp_z = (sp_bps - spreads.mean()) / (spreads.std() + 1e-8)
        else:
            sp_z = 0.0

        # Kyle lambda: |Δmid| / sum(trade_volume) since last tick
        delta_mid = abs(mid - self._prev_mid) if self._prev_mid > 0 else 0.0
        trade_vol = sum(float(t.get("qty", t.get("size", 0))) for t in recent_trades[-10:])
        if trade_vol > 0:
            raw_lambda = delta_mid / trade_vol
            self._impact_history.append(raw_lambda)
        if self._impact_history:
            lambda_norm = float(np.mean(list(self._impact_history))) * 1000.0
        else:
            lambda_norm = 0.0
        self._prev_mid = mid

        # depth slopes
        padded_bids = list(bids) + [(0.0, 0.0)] * max(0, 5 - len(bids))
        padded_asks = list(asks) + [(0.0, 0.0)] * max(0, 5 - len(asks))
        bid_slope = self._depth_slope(padded_bids)
        ask_slope = self._depth_slope(padded_asks)

        features = np.array([
            self.order_book_imbalance(bids, asks, levels=1),            # [0]
            self.depth_imbalance(bids, asks),                            # [1]
            sp_bps,                                                       # [2]
            wmid_norm,                                                    # [3]
            self.vwap_deviation(bids, asks, vwap) if vwap > 0 else 0.0, # [4]
            self.trade_imbalance(recent_trades),                          # [5]
            float(padded_bids[0][1]) / qty_mean,                         # [6]
            float(padded_asks[0][1]) / qty_mean,                         # [7]
            float(padded_bids[1][1]) / qty_mean,                         # [8]
            float(padded_asks[1][1]) / qty_mean,                         # [9]
            float(padded_bids[2][1]) / qty_mean,                         # [10]
            float(padded_asks[2][1]) / qty_mean,                         # [11]
            float(np.clip(agent_inventory / max_inventory, -1.0, 1.0)),  # [12]
            float(agent_pnl / (capital * 0.01)),                         # [13]
            float(np.clip(realized_vol, 0.0, 10.0)),                     # [14]
            float(np.clip(momentum, -5.0, 5.0)),                         # [15]
            float(np.clip(sp_z, -5.0, 5.0)),                             # [16]
            float(np.clip(lambda_norm, 0.0, 10.0)),                      # [17]
            float(np.clip(bid_slope, -3.0, 3.0)),                        # [18]
            float(np.clip(ask_slope, -3.0, 3.0)),                        # [19]
        ], dtype=np.float32)

        return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
