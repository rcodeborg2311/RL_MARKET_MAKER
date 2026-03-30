"""On-policy rollout buffer for PPO."""
import numpy as np
from typing import Iterator, Tuple


class RolloutBuffer:
    """Stores one rollout of experience for PPO updates."""

    def __init__(
        self,
        rollout_length: int,
        state_dim: int = 20,
        action_dim: int = 2,
    ) -> None:
        self.rollout_length = rollout_length
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.reset()

    def reset(self) -> None:
        n = self.rollout_length
        self.states = np.zeros((n, self.state_dim), dtype=np.float32)
        self.actions = np.zeros((n, self.action_dim), dtype=np.float32)
        self.rewards = np.zeros(n, dtype=np.float32)
        self.values = np.zeros(n, dtype=np.float32)
        self.log_probs = np.zeros(n, dtype=np.float32)
        self.dones = np.zeros(n, dtype=np.float32)
        self.advantages = np.zeros(n, dtype=np.float32)
        self.returns = np.zeros(n, dtype=np.float32)
        self.ptr = 0
        self.full = False

    def add(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        value: float,
        log_prob: float,
        done: bool,
    ) -> None:
        idx = self.ptr % self.rollout_length
        self.states[idx] = state
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.values[idx] = value
        self.log_probs[idx] = log_prob
        self.dones[idx] = float(done)
        self.ptr += 1
        if self.ptr >= self.rollout_length:
            self.full = True

    def is_full(self) -> bool:
        return self.full

    def get_batches(
        self, batch_size: int
    ) -> Iterator[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """Yield random mini-batches of (states, actions, log_probs, returns, advantages)."""
        indices = np.random.permutation(self.rollout_length)
        for start in range(0, self.rollout_length, batch_size):
            idx = indices[start : start + batch_size]
            yield (
                self.states[idx],
                self.actions[idx],
                self.log_probs[idx],
                self.returns[idx],
                self.advantages[idx],
            )
