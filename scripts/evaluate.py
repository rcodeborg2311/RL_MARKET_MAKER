#!/usr/bin/env python3
"""Evaluate trained RL agent vs TWAP baseline."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import csv
from typing import Dict, Optional

import numpy as np
import torch

from mmenv.environment import MarketMakingEnv, EnvConfig
from agent.networks import ActorCritic
from data.lobster import LOBSTERParser


class TWAPBaseline:
    def get_action(self, state: np.ndarray) -> np.ndarray:
        return np.array([-2.0, 2.0], dtype=np.float32)


class RLAgent:
    def __init__(self, model_path: str) -> None:
        self.model = ActorCritic()
        ckpt = torch.load(model_path, map_location="cpu")
        self.model.actor.load_state_dict(ckpt["actor_state_dict"])
        self.model.eval()

    def get_action(self, state: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            st = torch.FloatTensor(state).unsqueeze(0)
            action, _, _ = self.model.actor.get_action(st)
        return action.squeeze(0).numpy()


def _run(agent, env: MarketMakingEnv, n_episodes: int) -> Dict:
    ep_pnls, ep_fills, ep_invs = [], [], []
    breaches = 0
    for ep in range(n_episodes):
        state, _ = env.reset(seed=ep)
        done = False
        ep_fill_list, ep_inv_list = [], []
        terminal_pnl = 0.0
        breached = False
        while not done:
            action = agent.get_action(state)
            state, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            terminal_pnl = info["pnl"]
            ep_fill_list.append(info["fill_rate"])
            ep_inv_list.append(abs(info["inventory"]))
            if terminated:
                breached = True
        ep_pnls.append(terminal_pnl)
        ep_fills.append(float(np.mean(ep_fill_list)) if ep_fill_list else 0.0)
        ep_invs.append(float(np.mean(ep_inv_list)) if ep_inv_list else 0.0)
        if breached:
            breaches += 1

    mp = float(np.mean(ep_pnls))
    sp = float(np.std(ep_pnls))
    return {
        "mean_pnl": mp,
        "std_pnl": sp,
        "fill_rate": float(np.mean(ep_fills)),
        "mean_inventory": float(np.mean(ep_invs)),
        "sharpe": mp / (sp + 1e-8),
        "breach_rate": breaches / n_episodes,
    }


def evaluate(
    agent_path: Optional[str],
    env: MarketMakingEnv,
    n_episodes: int = 100,
) -> Dict:
    twap = TWAPBaseline()
    print(f"Running TWAP baseline ({n_episodes} episodes)...")
    twap_res = _run(twap, env, n_episodes)

    if agent_path and os.path.exists(agent_path):
        rl = RLAgent(agent_path)
        print(f"Running RL agent ({n_episodes} episodes)...")
        rl_res = _run(rl, env, n_episodes)
    else:
        print("No trained model — using TWAP for both columns.")
        rl_res = twap_res

    # Print table
    print("\n" + "=" * 62)
    print(f"{'Metric':<22} | {'RL Agent':>12} | {'TWAP Baseline':>14}")
    print("-" * 62)
    rows = [
        ("Mean PnL",     f"${rl_res['mean_pnl']:.4f}",     f"${twap_res['mean_pnl']:.4f}"),
        ("Std PnL",      f"${rl_res['std_pnl']:.4f}",      f"${twap_res['std_pnl']:.4f}"),
        ("Fill Rate",    f"{rl_res['fill_rate']*100:.1f}%", f"{twap_res['fill_rate']*100:.1f}%"),
        ("Mean |Inv|",   f"{rl_res['mean_inventory']:.4f}", f"{twap_res['mean_inventory']:.4f}"),
        ("Sharpe",       f"{rl_res['sharpe']:.4f}",         f"{twap_res['sharpe']:.4f}"),
        ("Breach Rate",  f"{rl_res['breach_rate']*100:.1f}%", f"{twap_res['breach_rate']*100:.1f}%"),
    ]
    for name, rv, tv in rows:
        print(f"{name:<22} | {rv:>12} | {tv:>14}")
    print("=" * 62)

    os.makedirs("results", exist_ok=True)
    with open("results/evaluation.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "rl_agent", "twap_baseline"])
        for k in rl_res:
            w.writerow([k, rl_res[k], twap_res[k]])
    print("\nSaved → results/evaluation.csv")
    return {"rl": rl_res, "twap": twap_res}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="models/checkpoint_final.pt")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--n-steps", type=int, default=50_000)
    args = parser.parse_args()

    snapshots = LOBSTERParser().parse_or_generate(n_steps=args.n_steps)
    env = MarketMakingEnv(book_snapshots=snapshots, config=EnvConfig())
    evaluate(agent_path=args.model, env=env, n_episodes=args.episodes)


if __name__ == "__main__":
    main()
