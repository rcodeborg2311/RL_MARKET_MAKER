"""Custom Gymnasium environment for market making."""
import dataclasses
from typing import Dict, List, Optional, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from mmenv.features import OrderBookFeatures
from mmenv.simulator import FillSimulator

STATE_DIM = 20


@dataclasses.dataclass
class EnvConfig:
    gamma: float = 0.1            # inventory risk aversion
    max_inventory: float = 10.0
    tick_size: float = 0.01
    lot_size: float = 0.001
    max_steps: int = 1000
    capital: float = 10_000.0
    vol_window: int = 20          # steps for sigma estimate
    reward_scale: float = 100.0   # scale rewards to O(1)


class MarketMakingEnv(gym.Env):
    """
    Gymnasium market-making environment.

    Action: Box(2,) — bid/ask offsets in ticks from mid.
      action[0] ∈ [-5, 0]:  bid offset
      action[1] ∈ [0, +5]:  ask offset

    Observation: Box(20,) — professional microstructure state.

    Reward (Avellaneda-Stoikov + mark-to-market):
      r_t = spread_pnl_t - gamma * inventory_t^2 * sigma^2
    where spread_pnl_t = mark-to-market gain from fills this step.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        book_snapshots: List[Dict],
        config: Optional[EnvConfig] = None,
        render_mode: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.book_snapshots = book_snapshots
        self.config = config or EnvConfig()
        self.render_mode = render_mode

        self.features = OrderBookFeatures(n_levels=5)
        self.simulator = FillSimulator()

        self.action_space = spaces.Box(
            low=np.array([-5.0, -5.0], dtype=np.float32),
            high=np.array([5.0, 5.0], dtype=np.float32),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(STATE_DIM,), dtype=np.float32
        )

        self._step_idx: int = 0
        self._inventory: float = 0.0
        self._pnl: float = 0.0
        self._mid_history: List[float] = []
        self._vwap: float = 0.0
        self._vwap_num: float = 0.0   # Σ(price × qty) for proper VWAP
        self._vwap_den: float = 0.0   # Σ(qty)
        self._fill_history: List[bool] = []

    def _get_snapshot(self, idx: int) -> Dict:
        return self.book_snapshots[idx % len(self.book_snapshots)]

    def _compute_sigma(self) -> float:
        if len(self._mid_history) < 2:
            return 0.001
        prices = np.array(self._mid_history[-self.config.vol_window:])
        if len(prices) < 2:
            return 0.001
        return float(np.std(np.diff(np.log(prices + 1e-10)))) + 1e-8

    def _update_vwap(self, trades: list) -> None:
        """Accumulate VWAP from executed trades this step."""
        for t in trades:
            p = float(t.get("price", 0))
            q = float(t.get("qty", t.get("size", 0)))
            if p > 0 and q > 0:
                self._vwap_num += p * q
                self._vwap_den += q
        self._vwap = (self._vwap_num / self._vwap_den) if self._vwap_den > 0 else 0.0

    def _get_state(self, snapshot: Dict) -> np.ndarray:
        bids, asks = snapshot["bids"], snapshot["asks"]
        return self.features.compute_state_vector(
            bids=bids, asks=asks,
            recent_trades=snapshot.get("trades", []),
            vwap=self._vwap,
            agent_inventory=self._inventory,
            agent_pnl=self._pnl,
            max_inventory=self.config.max_inventory,
            capital=self.config.capital,
        )

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        self._step_idx = 0
        self._inventory = 0.0
        self._pnl = 0.0
        self._mid_history = []
        self._vwap = 0.0
        self._vwap_num = 0.0
        self._vwap_den = 0.0
        self._fill_history = []
        self.features = OrderBookFeatures(n_levels=5)

        snap = self._get_snapshot(0)
        self._mid_history.append(self.features.mid_price(snap["bids"], snap["asks"]))
        return self._get_state(snap), {}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        snap = self._get_snapshot(self._step_idx)
        bids, asks = snap["bids"], snap["asks"]
        trades = snap.get("trades", [])

        mid = self.features.mid_price(bids, asks)
        self._mid_history.append(mid)

        # Decode action
        bid_offset = float(np.clip(action[0], -5.0, 0.0))
        ask_offset = float(np.clip(action[1], 0.0, 5.0))
        bid_price = mid + bid_offset * self.config.tick_size
        ask_price = mid + ask_offset * self.config.tick_size
        if ask_price <= bid_price:
            ask_price = bid_price + self.config.tick_size

        # Update VWAP from market trades at this step
        self._update_vwap(trades)

        # Fills — use trades from the *current* snapshot only (no look-ahead)
        bid_filled, ask_filled = self.simulator.simulate_fills(
            bid_price, self.config.lot_size,
            ask_price, self.config.lot_size,
            trades,
        )

        # Update state
        self._inventory += bid_filled - ask_filled
        spread_pnl = ask_filled * (ask_price - mid) + bid_filled * (mid - bid_price)
        self._pnl += spread_pnl

        self._fill_history.append((bid_filled > 0) or (ask_filled > 0))

        # Reward: scaled AS reward + small fill bonus to incentivise tighter quoting
        sigma = self._compute_sigma()
        inv_penalty = self.config.gamma * (self._inventory ** 2) * (sigma ** 2)
        fill_bonus = 0.0001 * (bid_filled + ask_filled)   # tiny per-lot bonus
        reward = float((spread_pnl - inv_penalty + fill_bonus) * self.config.reward_scale)

        self._step_idx += 1
        terminated = bool(abs(self._inventory) > self.config.max_inventory)
        truncated = bool(self._step_idx >= self.config.max_steps)

        next_state = self._get_state(self._get_snapshot(self._step_idx))

        recent = self._fill_history[-100:]
        fill_rate = sum(recent) / len(recent) if recent else 0.0

        return next_state, reward, terminated, truncated, {
            "fill_rate": fill_rate,
            "inventory": self._inventory,
            "pnl": self._pnl,
            "bid_price": bid_price,
            "ask_price": ask_price,
            "mid_price": mid,
            "bid_filled": bid_filled,
            "ask_filled": ask_filled,
            "reward": reward,
            "sigma": sigma,
        }
