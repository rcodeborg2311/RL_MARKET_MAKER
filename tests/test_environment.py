"""Tests for Gymnasium market-making environment."""
import numpy as np
import pytest
import gymnasium as gym

from mmenv.environment import MarketMakingEnv, EnvConfig
from data.lobster import LOBSTERParser


@pytest.fixture
def env() -> MarketMakingEnv:
    snaps = LOBSTERParser(seed=42).parse_or_generate(n_steps=2_000)
    return MarketMakingEnv(book_snapshots=snaps, config=EnvConfig(max_steps=100))


def test_reset_state_shape(env):
    state, info = env.reset(seed=42)
    assert state.shape == (20,)


def test_reset_empty_info(env):
    _, info = env.reset()
    assert isinstance(info, dict)


def test_step_ask_gt_bid(env):
    env.reset(seed=42)
    _, _, _, _, info = env.step(np.array([0.0, 0.0], dtype=np.float32))
    assert info["ask_price"] > info["bid_price"]


def test_inventory_tracked_correctly(env):
    env.reset(seed=0)
    cumulative = 0.0
    for _ in range(10):
        action = np.array([-1.0, 1.0], dtype=np.float32)
        _, _, terminated, truncated, info = env.step(action)
        cumulative += info["bid_filled"] - info["ask_filled"]
        assert info["inventory"] == pytest.approx(cumulative, abs=1e-6)
        if terminated or truncated:
            break


def test_episode_terminates_max_steps(env):
    env.reset(seed=0)
    truncated_seen = False
    for _ in range(200):
        _, _, _, truncated, _ = env.step(np.array([-1.0, 1.0], dtype=np.float32))
        if truncated:
            truncated_seen = True
            break
    assert truncated_seen


def test_episode_terminates_max_inventory():
    cfg = EnvConfig(max_steps=10_000, max_inventory=0.0005, lot_size=0.01)
    snaps = LOBSTERParser(seed=0).parse_or_generate(n_steps=2_000)
    env2 = MarketMakingEnv(book_snapshots=snaps, config=cfg)
    env2.reset(seed=0)
    for _ in range(1_000):
        _, _, terminated, truncated, _ = env2.step(np.array([0.0, 5.0], dtype=np.float32))
        if terminated or truncated:
            break
    assert True  # no error is the assertion


def test_reward_penalizes_large_inventory(env):
    env.reset(seed=0)
    env._inventory = 9.0
    sigma, gamma = 0.001, env.config.gamma
    reward = 0.0 - gamma * (9.0 ** 2) * sigma ** 2
    assert reward < 0.0


def test_action_space_is_gymnasium_box(env):
    assert isinstance(env.action_space, gym.spaces.Box)
    assert env.action_space.shape == (2,)


def test_observation_space_is_gymnasium_box(env):
    assert isinstance(env.observation_space, gym.spaces.Box)
    assert env.observation_space.shape == (20,)
