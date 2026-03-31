"""
Coinbase Exchange REST API — public order book + trades for BTC-USD.
No API key required. Works on Streamlit Cloud, Render, anywhere.

Endpoints used (no auth):
  GET https://api.exchange.coinbase.com/products/BTC-USD/book?level=2
  GET https://api.exchange.coinbase.com/products/BTC-USD/trades?limit=100
"""
import time
from typing import Dict, List, Tuple

try:
    import requests
    _REQUESTS_OK = True
except ImportError:
    _REQUESTS_OK = False

BASE = "https://api.exchange.coinbase.com"
PRODUCT = "BTC-USD"


class CoinbaseRESTFeed:
    """
    Polls Coinbase Exchange REST API for L2 order book and recent trades.
    Call get_snapshot() each tick — takes ~80-150ms per call.

    Caches the last response so callers that call faster than the poll
    interval get the most recent data without hammering the API.
    """

    POLL_INTERVAL = 0.5   # seconds between actual HTTP calls

    def __init__(self) -> None:
        if not _REQUESTS_OK:
            raise ImportError("pip install requests")
        self._bids: List[Tuple[float, float]] = []
        self._asks: List[Tuple[float, float]] = []
        self._trades: List[Dict]              = []
        self._last_poll: float                = 0.0
        self._last_trade_ids: set             = set()
        self._connected: bool                 = False
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": "rl-market-maker/1.0"})

    def _fetch(self) -> None:
        """Fetch book + trades from Coinbase Exchange REST API."""
        now = time.time()
        if now - self._last_poll < self.POLL_INTERVAL:
            return   # use cached data

        try:
            # ── Level-2 order book (aggregated) ──────────────────────────────
            resp = self._session.get(
                f"{BASE}/products/{PRODUCT}/book",
                params={"level": "2"},
                timeout=1.5,
            )
            resp.raise_for_status()
            book = resp.json()

            self._bids = [
                (float(b[0]), float(b[1]))
                for b in (book.get("bids") or [])[:10]
            ]
            self._asks = [
                (float(a[0]), float(a[1]))
                for a in (book.get("asks") or [])[:10]
            ]

            # ── Recent trades ─────────────────────────────────────────────────
            resp2 = self._session.get(
                f"{BASE}/products/{PRODUCT}/trades",
                params={"limit": "100"},
                timeout=1.5,
            )
            resp2.raise_for_status()
            raw = resp2.json()

            # Only keep trades we haven't seen before (deduplicate by trade_id)
            new_trades = []
            for t in raw:
                tid = t.get("trade_id")
                if tid not in self._last_trade_ids:
                    new_trades.append({
                        "price": float(t.get("price", 0)),
                        "qty":   float(t.get("size",  0)),
                        # Coinbase REST: side = aggressor side
                        # "buy" = buyer aggressed (hit ask) → ask fill
                        # "sell" = seller aggressed (hit bid) → bid fill
                        "side":  t.get("side", "").lower(),
                        "time":  t.get("time", ""),
                        "trade_id": tid,
                    })
                    self._last_trade_ids.add(tid)

            # Keep the id set bounded
            if len(self._last_trade_ids) > 5000:
                self._last_trade_ids = set(
                    list(self._last_trade_ids)[-2000:]
                )

            self._trades  = new_trades
            self._connected = True
            self._last_poll = now

        except Exception as exc:
            print(f"[CoinbaseRESTFeed] {exc}")
            self._connected = False

    def is_connected(self) -> bool:
        self._fetch()
        return self._connected

    def get_snapshot(self, n_levels: int = 5) -> Tuple[List, List]:
        self._fetch()
        return self._bids[:n_levels], self._asks[:n_levels]

    def get_new_trades(self) -> List[Dict]:
        """Returns only trades that arrived since the last call."""
        self._fetch()
        trades = self._trades
        self._trades = []   # consume — next call returns fresh trades only
        return trades
