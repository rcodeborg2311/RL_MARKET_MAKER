"""
Coinbase Advanced Trade WebSocket L2 feed for BTC-USD.
No API key required for public market data.
"""
import asyncio
import json
import threading
import time
from collections import deque
from typing import Dict, List, Optional, Tuple

try:
    import websockets
    _WS_AVAILABLE = True
except ImportError:
    _WS_AVAILABLE = False


class CoinbaseL2Feed:
    """
    Connects to wss://advanced-trade-ws.coinbase.com/ws
    Subscribes to 'level2' channel for BTC-USD.
    Maintains a local order book from snapshots and incremental updates.
    Thread-safe.
    """

    WS_URL = "wss://advanced-trade-ws.coinbase.com/ws"
    PRODUCT_ID = "BTC-USD"

    def __init__(self) -> None:
        self._bids: Dict[float, float] = {}
        self._asks: Dict[float, float] = {}
        self._recent_trades: deque = deque(maxlen=200)
        self._lock = threading.Lock()
        self._connected = False
        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._retry_count = 0
        self._max_retries = 5
        self._stop_event = threading.Event()

    def connect(self) -> None:
        """Start background WebSocket connection thread."""
        if not _WS_AVAILABLE:
            raise ImportError("Install websockets>=12.0")
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def _run_loop(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._connect_with_retry())

    async def _connect_with_retry(self) -> None:
        while self._retry_count < self._max_retries and not self._stop_event.is_set():
            try:
                await self._ws_connect()
                self._retry_count = 0
            except Exception as exc:
                self._retry_count += 1
                wait = min(2 ** self._retry_count, 30)
                print(f"[CoinbaseL2Feed] Error: {exc}. Retry {self._retry_count}/{self._max_retries} in {wait}s")
                if self._retry_count < self._max_retries:
                    await asyncio.sleep(wait)

    async def _ws_connect(self) -> None:
        import websockets as ws
        async with ws.connect(self.WS_URL, max_size=10 * 1024 * 1024) as websocket:
            self._connected = True
            await websocket.send(json.dumps({
                "type": "subscribe",
                "product_ids": [self.PRODUCT_ID],
                "channel": "level2",
            }))
            await websocket.send(json.dumps({
                "type": "subscribe",
                "product_ids": [self.PRODUCT_ID],
                "channel": "market_trades",
            }))
            async for message in websocket:
                if self._stop_event.is_set():
                    break
                try:
                    self._handle_message(json.loads(message))
                except (json.JSONDecodeError, KeyError):
                    pass

    def _handle_message(self, data: dict) -> None:
        channel = data.get("channel", "")
        events = data.get("events", [])

        if channel == "l2_data":
            with self._lock:
                for event in events:
                    for upd in event.get("updates", []):
                        side = upd.get("side", "")
                        price = float(upd.get("price_level", 0))
                        qty = float(upd.get("new_quantity", 0))
                        if side == "bid":
                            if qty == 0:
                                self._bids.pop(price, None)
                            else:
                                self._bids[price] = qty
                        elif side == "offer":
                            if qty == 0:
                                self._asks.pop(price, None)
                            else:
                                self._asks[price] = qty

        elif channel == "market_trades":
            with self._lock:
                for event in events:
                    for trade in event.get("trades", []):
                        self._recent_trades.append({
                            "price": float(trade.get("price", 0)),
                            "qty": float(trade.get("size", 0)),
                            "side": trade.get("side", "").lower(),
                            "time": trade.get("time", ""),
                        })

    def get_snapshot(self, n_levels: int = 5) -> Tuple[List, List]:
        """Returns (bids, asks) each as list of (price, qty) tuples."""
        with self._lock:
            bids = sorted(self._bids.items(), key=lambda x: -x[0])[:n_levels]
            asks = sorted(self._asks.items(), key=lambda x: x[0])[:n_levels]
        return list(bids), list(asks)

    def get_recent_trades(self, n: int = 50) -> List[dict]:
        with self._lock:
            return list(self._recent_trades)[-n:]

    def is_connected(self) -> bool:
        return self._connected

    def disconnect(self) -> None:
        self._stop_event.set()
        self._connected = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)
