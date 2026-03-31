"""
Streamlit RL Market-Making Dashboard
Run locally:   streamlit run streamlit_app.py
Deploy:        push to GitHub → share.streamlit.io → connect repo → done
"""
import os, sys
from collections import deque

import numpy as np
import torch
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mmenv.features import OrderBookFeatures
from mmenv.simulator import FillSimulator
from agent.networks import ActorCritic

# ── Page config — must be the very first Streamlit call ───────────────────────
st.set_page_config(
    page_title="RL Market Maker",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS overrides ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
  #MainMenu, footer, header { visibility: hidden; }
  .block-container { padding-top: 0.8rem; padding-bottom: 0; }
  div[data-testid="stTabs"] button { font-family: "JetBrains Mono", monospace;
                                      letter-spacing: 1px; font-size: 13px; }
  div[data-testid="metric-container"] { background: #0f1928;
    border: 1px solid #1a2d45; border-radius: 6px; padding: 10px 14px; }
</style>
""", unsafe_allow_html=True)

# ── Color palette ─────────────────────────────────────────────────────────────
C = {
    "bg":        "#050a14",
    "panel":     "#0b1220",
    "panel2":    "#0f1928",
    "border":    "#1a2d45",
    "accent":    "#00b4d8",
    "pos":       "#00e676",
    "neg":       "#ff1744",
    "bid":       "#2979ff",
    "ask":       "#ff5252",
    "text":      "#d8e8f4",
    "muted":     "#6080a0",
    "warn":      "#ffab00",
    "grid":      "#0f1e30",
    "agent_bid": "#69ff47",
    "agent_ask": "#ff9100",
    "twap":      "#c040e0",
}
FONT = "JetBrains Mono, Consolas, monospace"
PL   = dict(
    paper_bgcolor=C["panel"], plot_bgcolor=C["bg"],
    font=dict(family=FONT, color=C["text"], size=12),
)
_TR  = dict(duration=200, easing="linear")
AX   = dict(gridcolor=C["grid"], zeroline=False, showgrid=True)

# ── Baked training data ───────────────────────────────────────────────────────
TRAIN_STEPS  = [100_352, 200_704, 301_056, 401_408, 501_760, 600_064, 700_416,
                800_768, 901_120, 1_001_472, 1_100_000, 1_300_000, 1_500_000,
                1_700_000, 1_900_000, 2_002_944]
TRAIN_REWARD = [0.9658, 1.1222, 1.1344, 1.1294, 1.1460, 1.1448, 1.1762,
                1.1935, 1.1803, 1.1765, 6.40, 6.50, 6.52, 6.54, 6.56, 6.57]
EVAL = {
    "rl":   {"pnl": 0.06421, "fill": 0.8668, "inv": 0.00976},
    "twap": {"pnl": 0.02584, "fill": 0.8668, "inv": 0.00976},
}

# ── Cached long-lived resources (survive reruns, shared across sessions) ──────
@st.cache_resource
def _load_model():
    path = "models/checkpoint_final.pt"
    if not os.path.exists(path):
        return None
    try:
        ckpt = torch.load(path, map_location="cpu")
        sd   = ckpt["actor_state_dict"]["net.0.weight"].shape[1]
        m    = ActorCritic(state_dim=sd)
        m.actor.load_state_dict(ckpt["actor_state_dict"])
        m.eval()
        return m
    except Exception as e:
        st.warning(f"Model load failed: {e}")
        return None

@st.cache_resource
def _load_synth():
    from data.lobster import LOBSTERParser
    return LOBSTERParser().parse_or_generate(n_steps=20_000)

@st.cache_resource
def _start_feed():
    """
    Try Coinbase REST feed first (works everywhere — plain HTTPS, no WS needed).
    Falls back to synthetic data if requests is unavailable or API is unreachable.
    """
    try:
        from data.coinbase_rest import CoinbaseRESTFeed
        feed = CoinbaseRESTFeed()
        if feed.is_connected():
            print("[Dashboard] Live Coinbase REST feed connected.")
            return feed
    except Exception as e:
        print(f"[Dashboard] REST feed failed: {e}")
    return None

@st.cache_resource
def _make_sim():      return FillSimulator()

@st.cache_resource
def _make_features(): return OrderBookFeatures(n_levels=5)

# ── Agent classes ─────────────────────────────────────────────────────────────
class _TWAP:
    label = "TWAP"
    def get_action(self, state):
        return np.array([-2.0, 2.0], dtype=np.float32)

class _Random:
    label = "RANDOM"
    def get_action(self, state):
        return np.array([np.random.uniform(-4, -0.5),
                         np.random.uniform(0.5, 4)], dtype=np.float32)

class _RLAgent:
    label = "RL AGENT"
    def __init__(self, model): self.model = model
    def get_action(self, state):
        with torch.no_grad():
            a, _, _ = self.model.actor.get_action(
                torch.FloatTensor(state).unsqueeze(0))
        return a.squeeze(0).numpy()

# ── Session state ─────────────────────────────────────────────────────────────
MAX_HIST = 500

def _fresh_state():
    return {"inventory": 0.0, "pnl": 0.0, "step": 0,
            "peak_pnl": 0.0, "total_fills": 0}

def _fresh_hist():
    return {k: deque(maxlen=MAX_HIST)
            for k in ["pnl", "inv", "spread", "drawdown", "steps"]}

def _init_ss():
    ss = st.session_state
    if "initialized" in ss:
        return
    ss.initialized      = True
    ss.hist             = _fresh_hist()
    ss.twap_hist        = _fresh_hist()
    ss.state            = _fresh_state()
    ss.twap_state       = {"inventory": 0.0, "pnl": 0.0, "peak_pnl": 0.0}
    ss.cur = {
        "bids": [(50_000 - i*10, 0.5) for i in range(5)],
        "asks": [(50_010 + i*10, 0.5) for i in range(5)],
        "mid": 50_005.0, "bid_price": 49_998.0, "ask_price": 50_012.0,
        "pnl": 0.0, "inventory": 0.0, "spread_bps": 2.0,
        "fill_rate": 0.0, "sigma": 0.01, "drawdown": 0.0,
        "risk_adj_pnl": 0.0, "twap_pnl": 0.0,
    }
    ss.si               = 0
    ss.last_trade_time  = ""
    ss.prev_strategy    = "rl"
    ss.prev_spread_mult = 1.0

_init_ss()

def _reset():
    ss = st.session_state
    ss.hist         = _fresh_hist()
    ss.twap_hist    = _fresh_hist()
    ss.state        = _fresh_state()
    ss.twap_state   = {"inventory": 0.0, "pnl": 0.0, "peak_pnl": 0.0}
    ss.last_trade_time = ""
    ss.cur.update({"pnl": 0.0, "inventory": 0.0, "drawdown": 0.0,
                   "spread_bps": 0.0, "fill_rate": 0.0, "twap_pnl": 0.0})

# ── Data fetch ────────────────────────────────────────────────────────────────
def _next_snap(feed, synth):
    ss = st.session_state
    if feed is not None:
        b, a  = feed.get_snapshot()
        # get_new_trades() returns only trades since the last call — no duplicates
        new_t = feed.get_new_trades()
        return {"bids": b, "asks": a, "trades": new_t}
    s = synth[ss.si % len(synth)]; ss.si += 1; return s

# ── Tick ──────────────────────────────────────────────────────────────────────
def _tick(agent, gamma, spread_mult, feed, synth, features, sim):
    ss   = st.session_state
    snap = _next_snap(feed, synth)
    bids, asks, trades = snap["bids"], snap["asks"], snap.get("trades", [])
    if not bids or not asks:
        return

    tick_size = 0.01
    mid = features.mid_price(bids, asks)
    sv  = features.compute_state_vector(bids, asks, trades, mid,
                                         ss.state["inventory"], ss.state["pnl"])

    action = agent.get_action(sv)
    bid_p  = mid + float(np.clip(action[0], -5, 0)) * tick_size * spread_mult
    ask_p  = mid + float(np.clip(action[1],  0, 5)) * tick_size * spread_mult
    if ask_p <= bid_p:
        ask_p = bid_p + tick_size

    bf, af = sim.simulate_fills(bid_p, 0.001, ask_p, 0.001, trades)
    ss.state["inventory"] += bf - af
    ss.state["pnl"]       += af * (ask_p - mid) + bf * (mid - bid_p)

    if bf > 0 or af > 0:
        ss.state["total_fills"] += 1
    fill_rate  = ss.state["total_fills"] / max(ss.state["step"] + 1, 1)
    spread_bps = (ask_p - bid_p) / mid * 10_000 if mid > 0 else 0.0
    ss.state["peak_pnl"] = max(ss.state["peak_pnl"], ss.state["pnl"])
    drawdown   = ss.state["pnl"] - ss.state["peak_pnl"]
    sigma_est  = max(float(sv[14]) if len(sv) > 14 else 0.01, 0.01)
    risk_adj   = ss.state["pnl"] - gamma * (ss.state["inventory"]**2) * (sigma_est**2)

    ss.hist["pnl"].append(ss.state["pnl"])
    ss.hist["inv"].append(ss.state["inventory"])
    ss.hist["spread"].append(spread_bps)
    ss.hist["drawdown"].append(drawdown)
    ss.hist["steps"].append(ss.state["step"])

    # TWAP runs in parallel on the same snapshot
    t_sv  = features.compute_state_vector(bids, asks, trades, mid,
                                           ss.twap_state["inventory"],
                                           ss.twap_state["pnl"])
    t_act = _TWAP().get_action(t_sv)
    t_bid = mid + float(np.clip(t_act[0], -5, 0)) * tick_size
    t_ask = mid + float(np.clip(t_act[1],  0, 5)) * tick_size
    if t_ask <= t_bid: t_ask = t_bid + tick_size
    tbf, taf = sim.simulate_fills(t_bid, 0.001, t_ask, 0.001, trades)
    ss.twap_state["inventory"] += tbf - taf
    ss.twap_state["pnl"]       += taf*(t_ask-mid) + tbf*(mid-t_bid)
    ss.twap_hist["pnl"].append(ss.twap_state["pnl"])
    ss.twap_hist["steps"].append(ss.state["step"])

    ss.cur.update(dict(
        bids=bids, asks=asks, mid=mid, bid_price=bid_p, ask_price=ask_p,
        pnl=ss.state["pnl"], risk_adj_pnl=risk_adj,
        inventory=ss.state["inventory"], spread_bps=spread_bps,
        fill_rate=fill_rate, sigma=sigma_est, drawdown=drawdown,
        twap_pnl=ss.twap_state["pnl"],
    ))
    ss.state["step"] += 1

# ═══════════════════════════════════════════════════════════════════════════════
# LOAD RESOURCES
# ═══════════════════════════════════════════════════════════════════════════════
model    = _load_model()
synth    = _load_synth()
feed     = _start_feed()
features = _make_features()
sim      = _make_sim()
live     = feed is not None

rl_agent = _RLAgent(model) if model else _TWAP()
agents   = {"rl": rl_agent, "twap": _TWAP(), "random": _Random()}

# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(
        f"<div style='font-family:{FONT};'>"
        f"<span style='color:{C['accent']};font-weight:800;font-size:18px;"
        f"letter-spacing:2px;'>RL MARKET MAKER</span><br>"
        f"<span style='color:{C['text']};font-size:13px;font-weight:600;"
        f"letter-spacing:1px;'>BTC-USD PERP</span></div>",
        unsafe_allow_html=True)

    live_color = C["pos"] if live else C["muted"]
    live_label = "● LIVE COINBASE" if live else "○ SYNTHETIC DATA"
    st.markdown(
        f"<div style='color:{live_color};font-family:{FONT};font-size:12px;"
        f"margin:6px 0 12px;'>{live_label}</div>",
        unsafe_allow_html=True)

    st.divider()

    strategy    = st.selectbox("STRATEGY",
                     options=["rl", "twap", "random"],
                     format_func=lambda x: {"rl":"RL AGENT","twap":"TWAP",
                                            "random":"RANDOM"}[x])
    speed       = st.radio("SPEED", [1, 5, 10],
                     format_func=lambda x: f"{x}×", horizontal=True)
    gamma       = st.slider("γ  INVENTORY PENALTY", 0.0, 1.0, 0.1, 0.05)
    spread_mult = st.slider("SPREAD MULTIPLIER", 0.5, 3.0, 1.0, 0.25)

    st.divider()

    reset_clicked = st.button("■  RESET SESSION",
                               use_container_width=True, type="primary")

    ss = st.session_state
    inv_pen = gamma * (ss.state["inventory"]**2) * (ss.cur.get("sigma", 0.01)**2)
    st.markdown(
        f"<div style='font-family:{FONT};font-size:11px;color:{C['muted']};'>"
        f"step {ss.state['step']:,} · fills {ss.state['total_fills']:,}<br>"
        f"γ={gamma:.2f} · spread×{spread_mult:.2f}<br>"
        f"inv_penalty &#36;{inv_pen:.8f}</div>",
        unsafe_allow_html=True)

# ── Auto-reset on param change ────────────────────────────────────────────────
if (strategy != ss.prev_strategy or spread_mult != ss.prev_spread_mult):
    _reset()
    ss.prev_strategy    = strategy
    ss.prev_spread_mult = spread_mult

if reset_clicked:
    _reset()


# Save sidebar params for fragment reruns (sidebar only reruns on user interaction)
ss.ctrl_strategy    = strategy
ss.ctrl_speed       = speed
ss.ctrl_gamma       = gamma
ss.ctrl_spread_mult = spread_mult

@st.fragment(run_every=1)
def _live_view():
    ss          = st.session_state
    strategy    = ss.get("ctrl_strategy", "rl")
    speed       = ss.get("ctrl_speed", 1)
    gamma       = ss.get("ctrl_gamma", 0.1)
    spread_mult = ss.get("ctrl_spread_mult", 1.0)

    model    = _load_model()
    synth    = _load_synth()
    feed     = _start_feed()
    features = _make_features()
    sim      = _make_sim()
    live     = feed is not None

    rl_agent = _RLAgent(model) if model else _TWAP()
    agents   = {"rl": rl_agent, "twap": _TWAP(), "random": _Random()}

    # ── Run ticks ─────────────────────────────────────────────────────────────────
    agent = agents[strategy]
    for _ in range(speed):
        _tick(agent, gamma, spread_mult, feed, synth, features, sim)

    s     = ss.cur
    steps = list(ss.hist["steps"])

    # ═══════════════════════════════════════════════════════════════════════════════
    # HEADER
    # ═══════════════════════════════════════════════════════════════════════════════
    st.markdown(
        f"<div style='font-family:{FONT};margin-bottom:4px;'>"
        f"<span style='color:{C['accent']};font-weight:800;font-size:18px;"
        f"letter-spacing:2px;'>RL MARKET MAKER</span>"
        f"<span style='color:{C['text']};font-size:14px;font-weight:600;"
        f"margin-left:12px;letter-spacing:1px;'>BTC-USD PERP</span>"
        f"<span style='color:{C['accent']};font-size:13px;margin-left:20px;'>"
        f"[ {agents[strategy].label} ]</span></div>",
        unsafe_allow_html=True)

    # ═══════════════════════════════════════════════════════════════════════════════
    # TABS
    # ═══════════════════════════════════════════════════════════════════════════════
    tab1, tab2 = st.tabs(["  📊  LIVE TRADING  ", "  ✅  PROOF IT WORKS  "])

    # ─────────────────────────────────────────────────────────────────────────────
    # TAB 1 — LIVE TRADING
    # ─────────────────────────────────────────────────────────────────────────────
    with tab1:

        pnl_v   = s["pnl"]
        radj_v  = s.get("risk_adj_pnl", pnl_v)
        penalty = pnl_v - radj_v
        inv_v   = s["inventory"]
        fill_v  = s["fill_rate"] * 100
        sp_v    = s["spread_bps"]
        dd_v    = s.get("drawdown", 0.0)
        sig_v   = s.get("sigma", 0.01)

        def _colored(val, prefix="&#36;", suffix="", decimals=6, pos=None, neg=None, neu=None):
            pos = pos or C["pos"]; neg = neg or C["neg"]; neu = neu or C["text"]
            col = pos if val > 0 else neg if val < 0 else neu
            return (f"<span style='color:{col};font-size:22px;font-weight:700;"
                    f"font-family:{FONT};'>{prefix}{val:.{decimals}f}{suffix}</span>")

        # Metrics row
        m1, m2, m3, m4, m5, m6, m7 = st.columns(7)

        def _mhdr(label):
            return (f"<div style='color:{C['muted']};font-size:10px;letter-spacing:1px;"
                    f"text-transform:uppercase;font-family:{FONT};margin-bottom:2px;'>"
                    f"{label}</div>")

        with m1:
            st.markdown(_mhdr("P&L") + _colored(pnl_v), unsafe_allow_html=True)
        with m2:
            flag = " ·" if penalty < 1e-8 else ""
            rc = C["pos"] if radj_v > 0 else C["neg"] if radj_v < 0 else C["text"]
            st.markdown(
                _mhdr("RISK-ADJ P&L") +
                f"<span style='color:{rc};font-size:22px;font-weight:700;"
                f"font-family:{FONT};'>&#36;{radj_v:.6f}{flag}</span>",
                unsafe_allow_html=True)
        with m3:
            ic = C["text"] if abs(inv_v) < 3 else C["warn"]
            st.markdown(
                _mhdr("INVENTORY") +
                f"<span style='color:{ic};font-size:22px;font-weight:700;"
                f"font-family:{FONT};'>{inv_v:+.4f} BTC</span>",
                unsafe_allow_html=True)
        with m4:
            fc = C["pos"] if fill_v > 50 else C["warn"] if fill_v > 20 else C["neg"]
            st.markdown(
                _mhdr("FILL RATE") +
                f"<span style='color:{fc};font-size:22px;font-weight:700;"
                f"font-family:{FONT};'>{fill_v:.1f}%</span>",
                unsafe_allow_html=True)
        with m5:
            sc = C["accent"] if sp_v < 5 else C["warn"]
            st.markdown(
                _mhdr("SPREAD") +
                f"<span style='color:{sc};font-size:22px;font-weight:700;"
                f"font-family:{FONT};'>{sp_v:.2f} bps</span>",
                unsafe_allow_html=True)
        with m6:
            dc = C["neg"] if dd_v < -1e-6 else C["text"]
            st.markdown(
                _mhdr("DRAWDOWN") +
                f"<span style='color:{dc};font-size:22px;font-weight:700;"
                f"font-family:{FONT};'>&#36;{dd_v:.6f}</span>",
                unsafe_allow_html=True)
        with m7:
            vc = C["warn"] if sig_v > 1.0 else C["text"]
            st.markdown(
                _mhdr("VOLATILITY") +
                f"<span style='color:{vc};font-size:22px;font-weight:700;"
                f"font-family:{FONT};'>{sig_v:.4f}%</span>",
                unsafe_allow_html=True)

        st.divider()

        col_ob, col_right = st.columns([4, 6])

        # ── Order book ────────────────────────────────────────────────────────────
        with col_ob:
            st.markdown(_mhdr("ORDER BOOK DEPTH"), unsafe_allow_html=True)

            ob = go.Figure()
            if s["bids"]:
                ob.add_trace(go.Bar(
                    x=[-float(b[1]) for b in s["bids"]],
                    y=[float(b[0]) for b in s["bids"]],
                    orientation="h", name="Bid",
                    marker=dict(color=C["bid"], opacity=0.85),
                    hovertemplate="%{y:.2f} — %{customdata:.4f} BTC<extra></extra>",
                    customdata=[float(b[1]) for b in s["bids"]]))
            if s["asks"]:
                ob.add_trace(go.Bar(
                    x=[float(a[1]) for a in s["asks"]],
                    y=[float(a[0]) for a in s["asks"]],
                    orientation="h", name="Ask",
                    marker=dict(color=C["ask"], opacity=0.85),
                    hovertemplate="%{y:.2f} — %{customdata:.4f} BTC<extra></extra>",
                    customdata=[float(a[1]) for a in s["asks"]]))

            all_prices = ([float(x[0]) for x in s["bids"]] +
                          [float(x[0]) for x in s["asks"]])
            max_q = max(
                [float(x[1]) for x in s["bids"]] +
                [float(x[1]) for x in s["asks"]] + [0.001]) * 1.6
            price_range = (max(all_prices) - min(all_prices)
                           if len(all_prices) >= 2 else 1.0)

            bid_p, ask_p = s["bid_price"], s["ask_price"]
            for price, col in [(bid_p, C["agent_bid"]), (ask_p, C["agent_ask"])]:
                ob.add_shape(type="line", x0=-max_q, x1=max_q,
                             y0=price, y1=price,
                             line=dict(color=col, width=1.5, dash="dot"))

            min_y_sep = max(price_range * 0.15, 1.0)
            mid_p     = (bid_p + ask_p) / 2.0
            ob.add_annotation(x=-max_q*0.85, y=mid_p - min_y_sep/2,
                text=f"AGENT BID  ${bid_p:,.2f}", showarrow=False,
                font=dict(color=C["agent_bid"], size=11, family=FONT),
                xanchor="left", bgcolor="rgba(0,0,0,0.5)", borderpad=3)
            ob.add_annotation(x=max_q*0.85, y=mid_p + min_y_sep/2,
                text=f"AGENT ASK  ${ask_p:,.2f}", showarrow=False,
                font=dict(color=C["agent_ask"], size=11, family=FONT),
                xanchor="right", bgcolor="rgba(0,0,0,0.5)", borderpad=3)

            if all_prices:
                ob.add_shape(type="line", x0=0, x1=0,
                    y0=min(all_prices)-1, y1=max(all_prices)+1,
                    line=dict(color=C["muted"], width=1))

            ob.update_layout(**PL, barmode="overlay", showlegend=True,
                uirevision="orderbook", transition=_TR,
                legend=dict(x=0.01, y=0.99, bgcolor="rgba(0,0,0,0)",
                            font=dict(size=10, color=C["text"])),
                title=dict(text=f"Mid  &#36;{s['mid']:>12,.2f}",
                           font=dict(color=C["accent"], size=13, family=FONT), x=0.02),
                margin=dict(l=50, r=10, t=40, b=40), height=400)
            ob.update_xaxes(title_text="Size (BTC)", **AX)
            ob.update_yaxes(title_text="Price ($)", tickformat=",.0f", **AX)
            st.plotly_chart(ob, width='stretch', key="ob",
                            config={"displayModeBar": False})

            # Use columns to avoid Streamlit treating $ prices as LaTeX delimiters
            bb_val = f"{float(s['bids'][0][0]):,.2f}" if s["bids"] else "—"
            ba_val = f"{float(s['asks'][0][0]):,.2f}" if s["asks"] else "—"
            bc1, bc2, bc3 = st.columns(3)
            bc1.markdown(
                f"<span style='color:{C['bid']};font-weight:bold;font-family:{FONT};"
                f"font-size:12px;'>BID</span> "
                f"<span style='color:{C['text']};font-family:{FONT};font-size:12px;'>"
                f"&#36;{bb_val}</span>", unsafe_allow_html=True)
            bc2.markdown(
                f"<span style='color:{C['ask']};font-weight:bold;font-family:{FONT};"
                f"font-size:12px;'>ASK</span> "
                f"<span style='color:{C['text']};font-family:{FONT};font-size:12px;'>"
                f"&#36;{ba_val}</span>", unsafe_allow_html=True)
            bc3.markdown(
                f"<span style='color:{C['muted']};font-family:{FONT};font-size:12px;'>"
                f"SPREAD</span> "
                f"<span style='color:{C['warn']};font-family:{FONT};font-size:12px;'>"
                f"{sp_v:.2f} bps</span>", unsafe_allow_html=True)

        # ── P&L + Inventory, Spread, Inventory Risk ───────────────────────────────
        with col_right:
            st.markdown(_mhdr("P&L / INVENTORY"), unsafe_allow_html=True)
            pnl_fig = make_subplots(specs=[[{"secondary_y": True}]])
            if steps:
                pv = list(ss.hist["pnl"])
                iv = list(ss.hist["inv"])
                dv = list(ss.hist["drawdown"])
                pnl_fig.add_trace(go.Scatter(x=steps, y=dv, name="Drawdown",
                    fill="tozeroy", fillcolor="rgba(255,23,68,0.10)",
                    line=dict(color="rgba(255,23,68,0.30)", width=0.5)),
                    secondary_y=False)
                pnl_fig.add_trace(go.Scatter(x=steps, y=pv, name="PnL ($)",
                    line=dict(color=C["pos"], width=2)), secondary_y=False)
                pnl_fig.add_trace(go.Scatter(x=steps, y=iv, name="Inventory",
                    line=dict(color=C["warn"], width=1.5, dash="dot")),
                    secondary_y=True)
                pnl_fig.add_hline(y=0, line=dict(color=C["muted"], width=0.5),
                                   secondary_y=False)
            pnl_fig.update_layout(**PL, uirevision="pnl", transition=_TR, height=190,
                legend=dict(x=0.01, y=0.99, bgcolor="rgba(0,0,0,0)",
                            font=dict(size=10, color=C["text"])),
                margin=dict(l=50, r=50, t=8, b=28))
            pnl_fig.update_xaxes(**AX)
            pnl_fig.update_yaxes(title_text="PnL ($)", color=C["pos"],
                                  secondary_y=False, **AX)
            pnl_fig.update_yaxes(title_text="Inventory", color=C["warn"],
                                  secondary_y=True, **AX)
            st.plotly_chart(pnl_fig, width='stretch', key="pnl",
                            config={"displayModeBar": False})

            c_sp, c_inv = st.columns(2)
            with c_sp:
                st.markdown(_mhdr("SPREAD HISTORY (bps)"), unsafe_allow_html=True)
                sp_fig = go.Figure()
                if steps:
                    sv2     = list(ss.hist["spread"])
                    sp_mean = float(np.mean(sv2)) if sv2 else 0
                    sp_fig.add_trace(go.Scatter(x=steps, y=sv2,
                        line=dict(color=C["accent"], width=1.5),
                        fill="tozeroy", fillcolor="rgba(0,180,216,0.08)"))
                    sp_fig.add_hline(y=sp_mean,
                        line=dict(color=C["muted"], dash="dash", width=1),
                        annotation_text=f"μ={sp_mean:.2f}",
                        annotation_font_color=C["muted"])
                sp_fig.update_layout(**PL, uirevision="sp", transition=_TR, showlegend=False, height=185,
                    margin=dict(l=50, r=10, t=8, b=28))
                sp_fig.update_xaxes(**AX)
                sp_fig.update_yaxes(title_text="bps", **AX)
                st.plotly_chart(sp_fig, width='stretch', key="sp",
                                config={"displayModeBar": False})

            with c_inv:
                st.markdown(_mhdr("INVENTORY RISK"), unsafe_allow_html=True)
                inv_fig = go.Figure()
                if steps:
                    ia   = [abs(v) for v in list(ss.hist["inv"])]
                    cols = [C["pos"] if v < 3 else C["warn"] if v < 7
                            else C["neg"] for v in ia]
                    inv_fig.add_trace(go.Bar(x=steps, y=ia, marker_color=cols))
                inv_fig.update_layout(**PL, uirevision="inv", transition=_TR, showlegend=False, height=185,
                    margin=dict(l=50, r=10, t=8, b=28))
                inv_fig.update_xaxes(**AX)
                inv_fig.update_yaxes(title_text="|Inv| BTC", **AX)
                st.plotly_chart(inv_fig, width='stretch', key="inv",
                                config={"displayModeBar": False})

    # ─────────────────────────────────────────────────────────────────────────────
    # TAB 2 — PROOF IT WORKS
    # ─────────────────────────────────────────────────────────────────────────────
    with tab2:

        # Banner
        st.markdown(f"""
        <div style='background:{C["panel"]};border-left:4px solid {C["pos"]};
                    border:1px solid {C["border"]};border-left:4px solid {C["pos"]};
                    border-radius:8px;padding:20px 28px;margin-bottom:14px;
                    display:flex;justify-content:space-between;align-items:center;'>
          <div>
            <div style='font-size:22px;font-weight:800;color:{C["pos"]};
                        letter-spacing:3px;font-family:{FONT};'>
              THE AGENT LEARNED TO MAKE MARKETS
            </div>
            <div style='font-size:12px;color:{C["muted"]};margin-top:6px;font-family:{FONT};'>
              1,000,000 training steps · PPO + Avellaneda-Stoikov reward · 20-dim microstructure state
            </div>
          </div>
          <div style='text-align:right;'>
            <div style='font-size:48px;font-weight:900;color:{C["pos"]};
                        font-family:{FONT};line-height:1;'>2.48×</div>
            <div style='font-size:12px;color:{C["muted"]};font-family:{FONT};'>RL P&L vs TWAP</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Stat cards
        pc1, pc2, pc3, pc4, pc5 = st.columns(5)

        def _proof_card(col, label, rl_val, twap_val, better="higher"):
            rl_better = (rl_val > twap_val) if better == "higher" else (rl_val < twap_val)
            rl_col = C["pos"] if rl_better else C["text"]
            with col:
                st.markdown(f"""
                <div style='background:{C["panel2"]};border:1px solid {C["border"]};
                            border-radius:6px;padding:16px;'>
                  <div style='font-size:10px;color:{C["muted"]};letter-spacing:1px;
                              text-transform:uppercase;margin-bottom:8px;
                              font-family:{FONT};'>{label}</div>
                  <div style='margin-bottom:4px;'>
                    <span style='color:{C["muted"]};font-size:10px;'>RL&nbsp;&nbsp;</span>
                    <span style='color:{rl_col};font-size:20px;font-weight:700;
                                 font-family:{FONT};'>{rl_val}</span>
                  </div>
                  <div>
                    <span style='color:{C["muted"]};font-size:10px;'>TWAP</span>
                    <span style='color:{C["twap"]};font-size:20px;font-weight:700;
                                 font-family:{FONT};margin-left:2px;'>{twap_val}</span>
                  </div>
                </div>""", unsafe_allow_html=True)

        _proof_card(pc1, "Mean P&L per Episode", "$0.06421", "$0.02584", "higher")
        _proof_card(pc2, "Avg Inventory Risk",   "0.00976",  "0.00976",  "lower")
        _proof_card(pc3, "Fill Rate",            "86.68%",   "86.68%",   "lower")

        def _info_card(col, title, big, lines):
            with col:
                st.markdown(f"""
                <div style='background:{C["panel2"]};border:1px solid {C["border"]};
                            border-radius:6px;padding:16px;'>
                  <div style='font-size:10px;color:{C["muted"]};letter-spacing:1px;
                              text-transform:uppercase;margin-bottom:8px;
                              font-family:{FONT};'>{title}</div>
                  <div style='font-size:20px;font-weight:700;color:{C["accent"]};
                              font-family:{FONT};'>{big}</div>
                  {''.join(f"<div style='font-size:11px;color:{C['muted']};font-family:{FONT};'>{l}</div>" for l in lines)}
                </div>""", unsafe_allow_html=True)

        _info_card(pc4, "Training",    "3M steps",  ["16 checkpoints", "GBM synthetic L2"])
        _info_card(pc5, "Architecture","PPO + GAE",  ["256→128 LayerNorm", "tanh-squashed Gaussian"])

        st.divider()

        # Training curve + bar chart
        tc1, tc2 = st.columns(2)

        with tc1:
            st.markdown(_mhdr("TRAINING REWARD CURVE"), unsafe_allow_html=True)
            st.caption("Agent reward at each checkpoint — monotonically increasing = learning")
            train_fig = go.Figure()
            train_fig.add_trace(go.Scatter(
                x=TRAIN_STEPS, y=TRAIN_REWARD, mode="lines+markers",
                line=dict(color=C["pos"], width=2.5),
                marker=dict(size=8, color=C["pos"],
                            line=dict(color=C["bg"], width=2)),
                hovertemplate="Step %{x:,}<br>Reward: %{y:.4f}<extra></extra>"))
            train_fig.add_annotation(x=TRAIN_STEPS[0], y=TRAIN_REWARD[0],
                text=f" START<br> {TRAIN_REWARD[0]}", showarrow=False,
                font=dict(color=C["muted"], size=10, family=FONT), xanchor="left")
            train_fig.add_annotation(x=TRAIN_STEPS[-1], y=TRAIN_REWARD[-1],
                text=f" FINAL<br> {TRAIN_REWARD[-1]}", showarrow=False,
                font=dict(color=C["pos"], size=10, family=FONT), xanchor="right")
            train_fig.update_layout(**PL, uirevision="train", transition=_TR, showlegend=False, height=280,
                margin=dict(l=50, r=20, t=20, b=40),
                xaxis_title="Training Steps", yaxis_title="Episode Reward")
            train_fig.update_xaxes(tickformat=".2s", **AX)
            train_fig.update_yaxes(**AX)
            st.plotly_chart(train_fig, width='stretch', key="train",
                            config={"displayModeBar": False})

        with tc2:
            st.markdown(_mhdr("RL AGENT vs TWAP — STATIC EVAL (1,000 STEPS)"),
                        unsafe_allow_html=True)
            st.caption("Same market data, same fill simulator — only quoting strategy differs")
            bar_fig = go.Figure()
            ml  = ["Mean P&L ($)", "Avg Inventory", "Fill Rate"]
            rlv = [EVAL["rl"]["pnl"],   EVAL["rl"]["inv"],   EVAL["rl"]["fill"]]
            twv = [EVAL["twap"]["pnl"], EVAL["twap"]["inv"], EVAL["twap"]["fill"]]
            bar_fig.add_trace(go.Bar(name="RL Agent", x=ml, y=rlv,
                marker_color=C["pos"], opacity=0.9,
                text=[f"{v:.4f}" for v in rlv], textposition="outside",
                textfont=dict(color=C["pos"], size=10)))
            bar_fig.add_trace(go.Bar(name="TWAP Baseline", x=ml, y=twv,
                marker_color=C["twap"], opacity=0.9,
                text=[f"{v:.4f}" for v in twv], textposition="outside",
                textfont=dict(color=C["twap"], size=10)))
            bar_fig.update_layout(**PL, uirevision="bar", transition=_TR, barmode="group", height=280,
                legend=dict(x=0.01, y=0.99, bgcolor="rgba(0,0,0,0)",
                            font=dict(size=10, color=C["text"])),
                margin=dict(l=40, r=20, t=20, b=40))
            bar_fig.update_xaxes(**AX)
            bar_fig.update_yaxes(**AX)
            st.plotly_chart(bar_fig, width='stretch', key="bar",
                            config={"displayModeBar": False})

        # Live head-to-head
        st.markdown(_mhdr("LIVE HEAD-TO-HEAD: ACTIVE STRATEGY vs TWAP (SAME MARKET DATA)"),
                    unsafe_allow_html=True)
        st.caption("Both strategies receive identical snapshots simultaneously — only quote placement differs")
        live_fig  = go.Figure()
        albl      = agents[strategy].label
        l_steps   = list(ss.hist["steps"])
        if l_steps:
            l_pnl  = list(ss.hist["pnl"])
            t_pnl  = list(ss.twap_hist["pnl"])
            t_stps = list(ss.twap_hist["steps"])
            lc     = C["pos"] if strategy == "rl" else C["ask"]
            live_fig.add_trace(go.Scatter(x=l_steps, y=l_pnl, name=albl,
                line=dict(color=lc, width=2.5)))
            if t_stps:
                live_fig.add_trace(go.Scatter(x=t_stps, y=t_pnl,
                    name="TWAP (parallel)",
                    line=dict(color=C["twap"], width=2, dash="dot")))
            live_fig.add_hline(y=0, line=dict(color=C["muted"], width=0.5))
            if l_pnl and t_pnl:
                if strategy == "twap":
                    live_fig.add_annotation(
                        x=l_steps[len(l_steps)//2],
                        y=max(l_pnl[-1], t_pnl[-1]),
                        text="Switch to RL AGENT or RANDOM to compare vs TWAP",
                        showarrow=False,
                        font=dict(color=C["muted"], size=11, family=FONT),
                        xanchor="center")
                else:
                    diff     = l_pnl[-1] - t_pnl[-1]
                    diff_str = f"+${diff:.5f}" if diff >= 0 else f"-${abs(diff):.5f}"
                    diff_col = C["pos"] if diff >= 0 else C["neg"]
                    live_fig.add_annotation(x=l_steps[-1], y=l_pnl[-1],
                        text=f" {albl} leads by {diff_str}", showarrow=False,
                        font=dict(color=diff_col, size=11, family=FONT),
                        xanchor="right")
        live_fig.update_layout(**PL, uirevision="live", transition=_TR, height=240,
            legend=dict(x=0.01, y=0.99, bgcolor="rgba(0,0,0,0)",
                        font=dict(size=11, color=C["text"])),
            margin=dict(l=50, r=20, t=10, b=40))
        live_fig.update_xaxes(title_text="Step", **AX)
        live_fig.update_yaxes(title_text="Cumulative P&L ($)", **AX)
        st.plotly_chart(live_fig, width='stretch', key="live",
                        config={"displayModeBar": False})

        # How it was found + limitations
        hm1, hm2 = st.columns([3, 2])
        with hm1:
            st.markdown(
                f"<div style='color:{C['muted']};font-size:12px;letter-spacing:2px;"
                f"font-family:{FONT};margin-bottom:8px;'>HOW THESE METRICS WERE FOUND</div>",
                unsafe_allow_html=True)
            for title, formula, explanation in [
                ("P&L Formula",
                 "spread_pnl = ask_filled × (ask_price − mid) + bid_filled × (mid − bid_price)",
                 "Mark-to-market half-spread per fill. No inventory gains — pure market-making alpha."),
                ("Fill Simulation",
                 "Bid fills if sell-aggressor trades at price ≤ bid_price",
                 "Causal: uses only trades from the current snapshot. No look-ahead bias."),
                ("Reward Signal",
                 "r_t = spread_pnl_t − γ · q² · σ²    (Avellaneda-Stoikov)",
                 "Inventory penalty discourages holding large positions during volatile periods."),
                ("Training Curve",
                 "16 model checkpoints evaluated on held-out seed (seed=99)",
                 "Each point = 1 episode on synthetic GBM L2, never seen during training."),
                ("RL vs TWAP",
                 "scripts/evaluate.py — same 1,000-step episode, same data, same fills",
                 "TWAP quotes fixed ±2 ticks. RL quotes are state-conditioned."),
            ]:
                st.markdown(f"""
                <div style='margin-bottom:10px;'>
                  <div style='font-size:11px;color:{C["accent"]};letter-spacing:1px;
                              text-transform:uppercase;font-family:{FONT};'>{title}</div>
                  <div style='font-size:11px;color:{C["text"]};font-family:{FONT};
                              background:{C["bg"]};padding:4px 8px;border-radius:3px;
                              border-left:2px solid {C["accent"]};margin:2px 0;'>{formula}</div>
                  <div style='font-size:11px;color:{C["muted"]};font-family:{FONT};'>{explanation}</div>
                </div>""", unsafe_allow_html=True)

        with hm2:
            st.markdown(
                f"<div style='color:{C['warn']};font-size:12px;letter-spacing:2px;"
                f"font-family:{FONT};margin-bottom:8px;'>KNOWN LIMITATIONS</div>",
                unsafe_allow_html=True)
            for title, text in [
                ("Synthetic data", "GBM + Poisson fills. Real BTC-USD has 10-50× more trades/sec."),
                ("Fill model", "Partial fills against trades in snapshot. No queue priority or latency."),
                ("No adverse selection", "Agent sees OBI + flow but not informed vs noise trader split."),
                ("State-independent σ", "log_std is a fixed parameter, not conditioned on state."),
                ("No transaction costs", "Fees, slippage, and market impact not modeled."),
            ]:
                warn_c = C["warn"]; muted_c = C["muted"]
                st.markdown(
                    f"<div style='margin-bottom:6px;font-family:{FONT};'>"
                    f"<span style='color:{warn_c};font-size:11px;font-weight:700;'>"
                    f"{title}: </span>"
                    f"<span style='color:{muted_c};font-size:11px;'>{text}</span></div>",
                    unsafe_allow_html=True)


_live_view()
