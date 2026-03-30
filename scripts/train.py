#!/usr/bin/env python3
"""CLI training entrypoint."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
import torch

from agent.ppo import PPOTrainer, PPOConfig
from mmenv.environment import MarketMakingEnv, EnvConfig
from data.lobster import LOBSTERParser


def main() -> None:
    parser = argparse.ArgumentParser(description="Train RL Market-Making Agent")
    parser.add_argument("--timesteps",    type=int,   default=2_000_000)
    parser.add_argument("--lobster-path", type=str,   default=None)
    parser.add_argument("--device",       type=str,   default="cpu")
    parser.add_argument("--rollout-length",type=int,  default=4096)
    parser.add_argument("--n-steps",      type=int,   default=500_000)
    parser.add_argument("--resume",       type=str,   default="models/checkpoint_final.pt",
                        help="Path to checkpoint to resume from (or 'none' to train fresh)")
    args = parser.parse_args()

    torch.manual_seed(0)
    np.random.seed(0)

    print("Loading/generating training data...")
    snapshots = LOBSTERParser().parse_or_generate(filepath=args.lobster_path, n_steps=args.n_steps)
    print(f"Dataset: {len(snapshots)} snapshots.")

    env    = MarketMakingEnv(book_snapshots=snapshots, config=EnvConfig())
    config = PPOConfig(total_timesteps=args.timesteps, rollout_length=args.rollout_length)
    trainer = PPOTrainer(env=env, config=config, device=args.device)

    resume = args.resume if args.resume.lower() != "none" else None
    if resume and os.path.exists(resume):
        ckpt = torch.load(resume, map_location=args.device)
        trainer.actor.load_state_dict(ckpt["actor_state_dict"])
        trainer.critic.load_state_dict(ckpt["critic_state_dict"])
        if "actor_optimizer" in ckpt:
            trainer.actor_optimizer.load_state_dict(ckpt["actor_optimizer"])
        if "critic_optimizer" in ckpt:
            trainer.critic_optimizer.load_state_dict(ckpt["critic_optimizer"])
        print(f"Resumed from {resume}")
    else:
        print("Training from scratch.")

    print(f"Training for {args.timesteps:,} timesteps on {args.device}...")
    logs = trainer.train()
    print(f"Training complete. {len(logs)} log entries.")


if __name__ == "__main__":
    main()
