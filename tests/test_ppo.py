"""Tests for PPO training components."""
import numpy as np
import pytest
import torch

from agent.networks import Actor, Critic, ActorCritic
from agent.ppo import PPOTrainer, PPOConfig
from mmenv.environment import MarketMakingEnv, EnvConfig
from data.lobster import LOBSTERParser


@pytest.fixture
def small_env():
    snaps = LOBSTERParser(seed=42).parse_or_generate(n_steps=500)
    return MarketMakingEnv(book_snapshots=snaps, config=EnvConfig(max_steps=50))


def test_actor_output_shape():
    actor = Actor(20, 2)
    state = torch.randn(1, 20)
    action, log_prob, entropy = actor.get_action(state)
    assert action.shape == (1, 2)


def test_critic_output_scalar():
    critic = Critic(20)
    state = torch.randn(1, 20)
    assert critic(state).shape == (1, 1)


def test_log_prob_finite():
    actor = Actor(20, 2)
    for _ in range(10):
        _, lp, _ = actor.get_action(torch.randn(1, 20))
        assert torch.isfinite(lp).all()


def test_gae_advantages_shape(small_env):
    cfg = PPOConfig(rollout_length=64, total_timesteps=100)
    trainer = PPOTrainer(env=small_env, config=cfg)
    n = 64
    rewards = np.random.randn(n).astype(np.float32)
    values  = np.random.randn(n).astype(np.float32)
    dones   = np.zeros(n, dtype=np.float32); dones[-1] = 1.0
    adv, ret = trainer.compute_gae(rewards, values, dones)
    assert adv.shape == (n,) and ret.shape == (n,)


def test_ppo_update_runs_without_error(small_env):
    cfg = PPOConfig(rollout_length=128, batch_size=32, n_epochs=1, total_timesteps=200)
    trainer = PPOTrainer(env=small_env, config=cfg)
    state, _ = small_env.reset(seed=42)
    for _ in range(cfg.rollout_length):
        st = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            action, lp, _ = trainer.actor.get_action(st)
            value = trainer.critic(st)
        action_np = action.squeeze(0).numpy()
        ns, r, term, trunc, _ = small_env.step(action_np)
        done = term or trunc
        trainer.buffer.add(state, action_np, r, value.squeeze().item(), lp.squeeze().item(), done)
        state = ns
        if done:
            state, _ = small_env.reset()
    st = torch.FloatTensor(state).unsqueeze(0)
    with torch.no_grad():
        nv = trainer.critic(st).squeeze().item()
    trainer.buffer.advantages, trainer.buffer.returns = trainer.compute_gae(
        trainer.buffer.rewards, trainer.buffer.values, trainer.buffer.dones, nv)
    metrics = trainer.update(trainer.buffer)
    assert all(k in metrics for k in ["policy_loss","value_loss","entropy","approx_kl"])
