"""
LOBSTER file parser and synthetic L2 data generator.
"""
import os
from typing import Dict, List, Optional

import numpy as np


class LOBSTERParser:
    """
    Parses LOBSTER format files or generates synthetic GBM-driven L2 data.

    Synthetic data:
      - Mid price: geometric Brownian motion (mu=0, sigma=0.001)
      - Bid/ask symmetrically placed around mid across 5 levels
      - Quantities drawn from LogNormal(0, 0.5)
      - Trades: Poisson(2) per step, randomly buy or sell near best bid/ask
    """

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed

    def parse_or_generate(
        self,
        filepath: Optional[str] = None,
        n_steps: int = 500_000,
    ) -> List[Dict]:
        """
        Returns list of snapshots:
          {'bids': [(price, qty), ...], 'asks': [(price, qty), ...], 'trades': [...]}
        """
        if filepath and os.path.exists(filepath):
            try:
                return self._parse_lobster(filepath)
            except Exception as exc:
                print(f"[LOBSTERParser] Parse failed ({exc}), falling back to synthetic.")
        return self._generate_synthetic(n_steps)

    def _parse_lobster(self, filepath: str) -> List[Dict]:
        import pandas as pd

        directory = os.path.dirname(filepath) or "."
        files = os.listdir(directory)
        msg_files = [f for f in files if "message" in f.lower() and f.endswith(".csv")]
        ob_files = [f for f in files if "orderbook" in f.lower() and f.endswith(".csv")]

        if not msg_files or not ob_files:
            raise FileNotFoundError("LOBSTER message/orderbook files not found.")

        msg_path = os.path.join(directory, sorted(msg_files)[0])
        ob_path = os.path.join(directory, sorted(ob_files)[0])

        messages = pd.read_csv(
            msg_path, header=None,
            names=["time", "type", "order_id", "size", "price", "direction"]
        )
        orderbook = pd.read_csv(ob_path, header=None)
        n_levels = orderbook.shape[1] // 4

        col_names = []
        for i in range(1, n_levels + 1):
            col_names.extend([f"ask_price_{i}", f"ask_qty_{i}", f"bid_price_{i}", f"bid_qty_{i}"])
        orderbook.columns = col_names[: orderbook.shape[1]]

        snapshots = []
        for idx in range(len(orderbook)):
            row = orderbook.iloc[idx]
            bids, asks = [], []
            for i in range(1, min(n_levels + 1, 6)):
                try:
                    bp = row[f"bid_price_{i}"] / 10_000.0
                    bq = row[f"bid_qty_{i}"]
                    ap = row[f"ask_price_{i}"] / 10_000.0
                    aq = row[f"ask_qty_{i}"]
                    if bp > 0 and bq > 0:
                        bids.append((bp, bq))
                    if ap > 0 and aq > 0:
                        asks.append((ap, aq))
                except KeyError:
                    break

            trades = []
            if idx < len(messages):
                msg = messages.iloc[idx]
                if int(msg["type"]) in [4, 5]:
                    side = "buy" if int(msg["direction"]) == 1 else "sell"
                    trades.append({
                        "price": msg["price"] / 10_000.0,
                        "qty": msg["size"],
                        "side": side,
                    })

            if bids and asks:
                snapshots.append({"bids": bids, "asks": asks, "trades": trades})

        return snapshots

    def _generate_synthetic(self, n_steps: int) -> List[Dict]:
        """Generate synthetic L2 snapshots via GBM."""
        rng = np.random.default_rng(self.seed)

        S0 = 50_000.0
        sigma = 0.001
        dt = 1.0
        tick = 0.01
        half_spread = 5 * tick

        log_ret = rng.normal(0.0, sigma * np.sqrt(dt), n_steps)
        mid_prices = S0 * np.exp(np.cumsum(log_ret))

        snapshots = []
        for i in range(n_steps):
            mid = float(mid_prices[i])
            bids, asks = [], []
            for lvl in range(5):
                bp = round(mid - half_spread - lvl * tick * 2, 2)
                ap = round(mid + half_spread + lvl * tick * 2, 2)
                bq = float(max(0.001, rng.lognormal(0, 0.5)))
                aq = float(max(0.001, rng.lognormal(0, 0.5)))
                bids.append((bp, round(bq, 4)))
                asks.append((ap, round(aq, 4)))

            n_trades = int(rng.poisson(2))
            trades = []
            for _ in range(n_trades):
                side = "buy" if rng.random() > 0.5 else "sell"
                if side == "buy":
                    tp = asks[0][0] + float(rng.uniform(0, tick))
                else:
                    tp = bids[0][0] - float(rng.uniform(0, tick))
                tq = float(max(0.001, rng.lognormal(-1, 0.5)))
                trades.append({"price": round(tp, 2), "qty": round(tq, 4), "side": side})

            snapshots.append({"bids": bids, "asks": asks, "trades": trades})

        return snapshots
