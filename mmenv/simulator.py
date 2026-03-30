"""Fill simulator for limit order execution."""
from typing import List, Tuple


class FillSimulator:
    """
    Given an agent's quoted bid/ask prices, determine fills
    based on recent trades.

    Rules:
      Bid fill: aggressive sell trade at price <= agent_bid_price → bid filled
      Ask fill: aggressive buy trade at price >= agent_ask_price → ask filled
    Partial fills: if trade_qty < remaining agent_qty, only trade_qty fills.
    """

    def simulate_fills(
        self,
        agent_bid_price: float,
        agent_bid_qty: float,
        agent_ask_price: float,
        agent_ask_qty: float,
        next_trades: List[dict],
    ) -> Tuple[float, float]:
        """
        Returns (bid_filled_qty, ask_filled_qty).

        Each trade dict: {'price': float, 'qty'/'size': float, 'side': 'buy'|'sell'}
        side='sell' = aggressive sell hitting bids
        side='buy'  = aggressive buy hitting asks
        """
        bid_filled = 0.0
        ask_filled = 0.0
        remaining_bid = float(agent_bid_qty)
        remaining_ask = float(agent_ask_qty)

        for trade in next_trades:
            trade_price = float(trade.get("price", 0))
            trade_qty = float(trade.get("qty", trade.get("size", 0)))
            side = trade.get("side", "")

            if side == "sell" and trade_price <= agent_bid_price and remaining_bid > 0:
                fill = min(trade_qty, remaining_bid)
                bid_filled += fill
                remaining_bid -= fill

            elif side == "buy" and trade_price >= agent_ask_price and remaining_ask > 0:
                fill = min(trade_qty, remaining_ask)
                ask_filled += fill
                remaining_ask -= fill

        return bid_filled, ask_filled
