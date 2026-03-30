# Reinforcement Learning Market-Making Agent

A PPO-trained agent that learns to quote bid/ask spreads on BTC-USD using Level 2 order book features. Built as a portfolio project targeting quant research and market-making internships at firms like Citadel Securities, Virtu Financial, and Jane Street.

---

## What Is Market Making?

A market maker simultaneously posts a bid (buy) price and an ask (sell) price in an order book. The agent profits from the **bid-ask spread**: it buys at the bid and sells at the ask, capturing the difference. The core risk is **inventory risk** — if prices move against a position before it can be unwound, the unrealized loss erodes spread income. A good market maker therefore quotes competitively enough to attract trades (fill rate) while managing inventory aggressively enough to stay neutral. This agent learns that trade-off end-to-end via reinforcement learning, using real-time Coinbase L2 data to observe market microstructure and adjust quotes every 500 ms.

---

## The Avellaneda-Stoikov Model

The reward function is inspired by the closed-form Avellaneda-Stoikov (2008) model. The key equations:

**Reservation price** (fair value adjusted for inventory risk):

$$r^* = m - q \cdot \gamma \cdot \sigma^2 \cdot (T - t)$$

**Optimal half-spreads** around the reservation price:

$$\delta^{bid} = \frac{1}{\gamma} \ln\!\left(1 + \frac{\gamma}{\kappa}\right) + \frac{1}{2} \gamma \sigma^2 (T-t) + q \gamma \sigma^2 (T-t)$$

$$\delta^{ask} = \frac{1}{\gamma} \ln\!\left(1 + \frac{\gamma}{\kappa}\right) + \frac{1}{2} \gamma \sigma^2 (T-t) - q \gamma \sigma^2 (T-t)$$

| Symbol | Name | Intuition |
|--------|------|-----------|
| $m$ | Mid price | Current fair value of the asset |
| $q$ | Signed inventory | Positive = long, negative = short; zero = flat |
| $\gamma$ | Risk aversion | Controls how aggressively inventory is penalized; higher → tighter quotes when long |
| $\sigma^2$ | Price variance | Rolling volatility; higher vol → wider spreads to compensate for adverse selection |
| $T - t$ | Time horizon | Remaining episode length; shorter → less time to unwind, so penalize more |
| $\kappa$ | Order arrival rate | Higher rate → can afford tighter spreads; related to fill probability |

The RL agent implicitly learns $\gamma$, $\kappa$, and the inventory adjustment from data rather than from the closed-form formula.

---

## Feature Engineering

| Index | Name | Formula | Why It Matters |
|-------|------|---------|----------------|
| 0 | OBI (top 1) | `(bid_vol₁ - ask_vol₁) / (bid_vol₁ + ask_vol₁)` | Immediate buy/sell pressure at best level |
| 1 | Depth imbalance | OBI across all 5 levels | Broader supply/demand imbalance |
| 2 | Spread (bps) | `(ask₁ - bid₁) / mid × 10000` | Current market-making opportunity cost |
| 3 | Weighted mid (norm) | `WMID/mid - 1` | True fair value vs. simple mid; signals which side is being lifted |
| 4 | VWAP deviation | `(mid - vwap) / vwap × 10000` | Mean-reversion signal; far from VWAP = likely to revert |
| 5 | Trade imbalance | `(buy_vol - sell_vol) / total` over last 50 trades | Directional flow from aggressive orders |
| 6 | bid_qty level 1 (norm) | `bid_qty₁ / mean_qty` | Absolute depth at best bid |
| 7 | ask_qty level 1 (norm) | `ask_qty₁ / mean_qty` | Absolute depth at best ask |
| 8 | bid_qty level 2 (norm) | `bid_qty₂ / mean_qty` | Depth one tick behind best bid |
| 9 | ask_qty level 2 (norm) | `ask_qty₂ / mean_qty` | Depth one tick behind best ask |
| 10 | bid_qty level 3 (norm) | `bid_qty₃ / mean_qty` | Further depth context |
| 11 | ask_qty level 3 (norm) | `ask_qty₃ / mean_qty` | Further depth context |
| 12 | Agent inventory | `inventory / max_inventory` ∈ [-1, 1] | Current risk exposure; drives spread asymmetry |
| 13 | Agent PnL (norm) | `pnl / (capital × 0.01)` | Running profitability signal |

---

## Results vs TWAP Baseline

*(Results on synthetic GBM data after 1M training steps)*

| Metric | RL Agent | TWAP Baseline |
|---|---|---|
| Mean PnL | $X.XX | $X.XX |
| Fill Rate | XX.X% | XX.X% |
| Mean Inventory | X.XX BTC | X.XX BTC |
| Sharpe (episodes) | X.XX | X.XX |

*Run `python scripts/evaluate.py --model models/checkpoint_final.pt` to populate.*

---

## Training Curve

![Training Curve](results/training_curve.png)

*Plot generated after training. Run `python scripts/train.py --timesteps 1000000` first.*

---

## Interview Q&A

**Q: Why PPO instead of DQN for this problem?**

The action space is continuous — bid and ask offsets are real-valued prices, not discrete choices. DQN requires a discrete action space and would need to bin price levels, losing precision and scaling poorly as tick granularity increases. PPO operates directly on a continuous action distribution (diagonal Gaussian), learning both the mean quote and the uncertainty around it. PPO is also on-policy, which provides more stable updates for a non-stationary environment like a live order book, and the entropy bonus naturally encourages quote exploration — critical for discovering profitable regions of the spread.

**Q: What is adverse selection and how does your agent handle it?**

Adverse selection occurs when a trade fills you *because* the counterparty knows the price is about to move against you. For example, a sophisticated participant sells to your bid right before a price drop — you get filled, then mark-to-market losses exceed your spread income. The agent handles this in two ways: (1) The **trade imbalance feature** (index 5) captures directional flow from aggressive orders, giving the agent a signal to widen or shift quotes before an expected move. (2) The **inventory penalty** in the reward function (`-γq²σ²`) punishes the agent for holding inventory during volatile periods (high σ), incentivizing it to quickly offload positions that arose from potentially adverse fills.

**Q: What does the inventory penalty gamma control?**

`γ` is the risk-aversion coefficient. At `γ=0`, the agent maximizes raw spread income and ignores inventory, eventually breaching position limits. As `γ` increases, the agent skews quotes: when long, it lowers the ask to attract sells and raises the bid to deter further buys — exactly the Avellaneda-Stoikov reservation price adjustment. The default `γ=0.1` balances fill rate against inventory control. Higher values produce a more conservative agent that quotes tighter when flat and wider when positioned; lower values produce an aggressive agent that accumulates inventory.

**Q: How would this agent behave during a flash crash?**

During a flash crash, `σ` spikes, mid price drops rapidly, and trade imbalance goes strongly negative (heavy sell flow). Three mechanisms kick in simultaneously: (1) The inventory penalty `γq²σ²` grows quadratically — any long inventory becomes very expensive, pushing the agent to widen the bid or cancel it. (2) The VWAP deviation feature signals a large negative deviation from recent fair value. (3) The OBI likely goes negative as bids thin out. Together these features should cause the agent to widen spreads, reduce bid aggressiveness, and potentially sit out the crash. Whether it does so *correctly* depends on how well the training distribution covered high-volatility regimes — an important limitation of training on synthetic GBM data.

**Q: What's the difference between OBI and trade imbalance?**

OBI (Order Book Imbalance) measures the *passive* state of the book: the ratio of resting bid vs. ask volume. It reflects *willingness* to trade but not actual trades. Trade imbalance measures *active* aggression: which side initiated recent executions. A book can show balanced OBI while trade imbalance is strongly positive (heavy buying), signaling that buyers are lifting offers faster than new offers arrive. Combining both gives the agent complementary views — OBI is forward-looking (depth ahead), trade imbalance is backward-looking (recent flow). Professional market makers use both to distinguish noise from directional flow.

---

## Quickstart

```bash
# Install dependencies
pip install -r requirements.txt

# Train (1M steps, ~30 min on CPU)
python scripts/train.py --timesteps 1000000

# Evaluate agent vs TWAP baseline
python scripts/evaluate.py --model models/checkpoint_final.pt

# Launch live dashboard (open http://localhost:8050)
python scripts/run_dashboard.py
```

### Quick smoke test (10k steps)
```bash
python scripts/train.py --timesteps 10000 --n-steps 5000
```

### Run tests
```bash
pytest tests/ -v
```

---

## Repository Structure

```
rl-market-maker/
├── mmenv/           ← Gymnasium env, features, fill simulator
├── agent/           ← PPO trainer, Actor/Critic networks, rollout buffer
├── data/            ← Coinbase WebSocket feed, LOBSTER parser/synthetic generator
├── dashboard/       ← Plotly Dash live dashboard
├── scripts/         ← train.py, evaluate.py, run_dashboard.py
├── tests/           ← pytest suite (features, env, simulator, PPO)
├── models/          ← saved .pt checkpoints
└── results/         ← training curves, evaluation CSVs, live session logs
```

---

## Design Decisions

- **On-policy PPO** over off-policy SAC/TD3: simpler to tune, no replay buffer staleness in a non-stationary market environment
- **Tanh-squashed Gaussian** actions: natural bounds on spread offsets without hard clipping gradients
- **GAE (λ=0.95)** for advantage estimation: reduces variance while keeping reasonable bias
- **Avellaneda-Stoikov reward**: theoretically grounded, interpretable, and directly maps to real market-making objectives
- **Synthetic GBM training data**: removes live-data dependency for reproducible training; LOBSTER support available for historical backtesting
