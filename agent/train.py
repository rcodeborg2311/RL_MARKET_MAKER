"""Training entrypoint (importable module)."""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.ppo import PPOTrainer, PPOConfig
from mmenv.environment import MarketMakingEnv, EnvConfig
from data.lobster import LOBSTERParser


def main() -> None:
    parser = argparse.ArgumentParser(description="Train RL Market-Making Agent")
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--lobster-path", type=str, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--rollout-length", type=int, default=2048)
    parser.add_argument("--n-steps", type=int, default=500_000)
    args = parser.parse_args()

    torch.manual_seed(42)
    np.random.seed(42)

    print("Loading/generating training data...")
    lob = LOBSTERParser()
    snapshots = lob.parse_or_generate(filepath=args.lobster_path, n_steps=args.n_steps)
    print(f"Dataset: {len(snapshots)} snapshots.")

    env = MarketMakingEnv(book_snapshots=snapshots, config=EnvConfig())
    config = PPOConfig(total_timesteps=args.timesteps, rollout_length=args.rollout_length)
    trainer = PPOTrainer(env=env, config=config, device=args.device)

    print(f"Training for {args.timesteps} timesteps on {args.device}...")
    logs = trainer.train()
    print(f"Training complete. {len(logs)} log entries.")


if __name__ == "__main__":
    main()
