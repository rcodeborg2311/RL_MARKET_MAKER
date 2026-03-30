"""PPO training loop for the market-making agent."""
import dataclasses
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

from agent.networks import Actor, Critic
from agent.replay_buffer import RolloutBuffer


@dataclasses.dataclass
class PPOConfig:
    lr_actor: float = 3e-4
    lr_critic: float = 1e-3
    gamma: float = 0.99
    gae_lambda: float = 0.97       # higher = less bias in advantage estimates
    clip_epsilon: float = 0.2
    entropy_coef: float = 0.02     # higher start → more exploration
    entropy_coef_end: float = 0.002  # anneal to this by end of training
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    n_epochs: int = 15             # more update passes per rollout
    batch_size: int = 128          # larger batches → more stable gradients
    rollout_length: int = 4096     # longer rollouts → better advantage estimates
    total_timesteps: int = 1_000_000
    state_dim: int = 20
    action_dim: int = 2
    lr_schedule: bool = True       # linear LR annealing to 0


class PPOTrainer:
    def __init__(
        self,
        env,
        config: Optional[PPOConfig] = None,
        device: str = "cpu",
    ) -> None:
        self.env = env
        self.config = config or PPOConfig()
        self.device = torch.device(device)

        torch.manual_seed(42)
        np.random.seed(42)

        self.actor = Actor(self.config.state_dim, self.config.action_dim).to(self.device)
        self.critic = Critic(self.config.state_dim).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.config.lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.config.lr_critic)

        self.buffer = RolloutBuffer(
            rollout_length=self.config.rollout_length,
            state_dim=self.config.state_dim,
            action_dim=self.config.action_dim,
        )
        os.makedirs("models", exist_ok=True)
        os.makedirs("results", exist_ok=True)

    def compute_gae(
        self,
        rewards: np.ndarray,
        values: np.ndarray,
        dones: np.ndarray,
        next_value: float = 0.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generalized Advantage Estimation.
        advantages[t] = delta[t] + (gamma*lambda)*delta[t+1] + ...
        returns = advantages + values
        """
        n = len(rewards)
        advantages = np.zeros(n, dtype=np.float32)
        last_gae = 0.0

        for t in reversed(range(n)):
            next_val = next_value if t == n - 1 else values[t + 1]
            non_terminal = 1.0 - dones[t]
            delta = rewards[t] + self.config.gamma * next_val * non_terminal - values[t]
            last_gae = delta + self.config.gamma * self.config.gae_lambda * non_terminal * last_gae
            advantages[t] = last_gae

        returns = advantages + values
        return advantages, returns

    def update(self, rollout_buffer: RolloutBuffer) -> Dict[str, float]:
        """
        Run n_epochs of mini-batch PPO updates.
        Returns: {'policy_loss', 'value_loss', 'entropy', 'approx_kl'}
        Stops early if approx_kl > 0.02.
        """
        # Normalize advantages
        adv = rollout_buffer.advantages.copy()
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        rollout_buffer.advantages = adv

        total_pl = total_vl = total_ent = total_kl = 0.0
        n_updates = 0

        for _ in range(self.config.n_epochs):
            early_stop = False
            for batch in rollout_buffer.get_batches(self.config.batch_size):
                states_b, actions_b, old_lp_b, returns_b, adv_b = batch

                states_t = torch.FloatTensor(states_b).to(self.device)
                actions_t = torch.FloatTensor(actions_b).to(self.device)
                old_lp_t = torch.FloatTensor(old_lp_b).to(self.device)
                returns_t = torch.FloatTensor(returns_b).to(self.device)
                adv_t = torch.FloatTensor(adv_b).to(self.device)

                mean, log_std = self.actor(states_t)
                std = torch.exp(log_std.clamp(-20, 2))
                dist = Normal(mean, std)

                # Inverse tanh to recover pre-squash value
                actions_scaled = (actions_t / 5.0).clamp(-0.9999, 0.9999)
                x = torch.atanh(actions_scaled)

                log_prob = (
                    dist.log_prob(x) - torch.log(1 - actions_scaled.pow(2) + 1e-7)
                ).sum(dim=-1)
                entropy = dist.entropy().sum(dim=-1).mean()

                ratio = torch.exp(log_prob - old_lp_t)
                approx_kl = float((old_lp_t - log_prob).mean().item())

                surr1 = ratio * adv_t
                surr2 = torch.clamp(ratio, 1 - self.config.clip_epsilon, 1 + self.config.clip_epsilon) * adv_t
                policy_loss = -torch.min(surr1, surr2).mean()

                values_pred = self.critic(states_t).squeeze(-1)
                value_loss = nn.functional.mse_loss(values_pred, returns_t)

                loss = (
                    policy_loss
                    + self.config.value_coef * value_loss
                    - self.config.entropy_coef * entropy
                )

                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.max_grad_norm)
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.config.max_grad_norm)
                self.actor_optimizer.step()
                self.critic_optimizer.step()

                total_pl += policy_loss.item()
                total_vl += value_loss.item()
                total_ent += entropy.item()
                total_kl += approx_kl
                n_updates += 1

                if abs(approx_kl) > 0.02:
                    early_stop = True
                    break
            if early_stop:
                break

        d = max(n_updates, 1)
        return {
            "policy_loss": total_pl / d,
            "value_loss": total_vl / d,
            "entropy": total_ent / d,
            "approx_kl": total_kl / d,
        }

    def train(self) -> List[Dict]:
        """Main training loop. Returns list of log dicts."""
        logs: List[Dict] = []
        state, _ = self.env.reset(seed=42)
        total_steps = 0

        ep_rewards: List[float] = []
        ep_fill_rates: List[float] = []
        ep_inventories: List[float] = []
        cur_ep_reward = 0.0
        cur_fills: List[float] = []
        cur_invs: List[float] = []

        while total_steps < self.config.total_timesteps:
            # ── LR + entropy annealing ────────────────────────────────────────
            progress = total_steps / self.config.total_timesteps  # 0 → 1
            if self.config.lr_schedule:
                lr_mult = 1.0 - progress
                for pg in self.actor_optimizer.param_groups:
                    pg["lr"] = self.config.lr_actor * lr_mult
                for pg in self.critic_optimizer.param_groups:
                    pg["lr"] = self.config.lr_critic * lr_mult
            self.config.entropy_coef = (
                self.config.entropy_coef_end
                + (0.02 - self.config.entropy_coef_end) * (1.0 - progress)
            )

            self.buffer.reset()

            for _ in range(self.config.rollout_length):
                state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    action, log_prob, _ = self.actor.get_action(state_t)
                    value = self.critic(state_t)

                action_np = action.squeeze(0).cpu().numpy()
                lp_np = log_prob.squeeze().cpu().item()
                val_np = value.squeeze().cpu().item()

                next_state, reward, terminated, truncated, info = self.env.step(action_np)
                done = terminated or truncated

                self.buffer.add(state, action_np, reward, val_np, lp_np, done)
                cur_ep_reward += reward
                cur_fills.append(info.get("fill_rate", 0.0))
                cur_invs.append(abs(info.get("inventory", 0.0)))

                state = next_state
                total_steps += 1

                if done:
                    ep_rewards.append(cur_ep_reward)
                    ep_fill_rates.append(float(np.mean(cur_fills)) if cur_fills else 0.0)
                    ep_inventories.append(float(np.mean(cur_invs)) if cur_invs else 0.0)
                    cur_ep_reward = 0.0
                    cur_fills = []
                    cur_invs = []
                    state, _ = self.env.reset()

            # GAE
            with torch.no_grad():
                st = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                nv = self.critic(st).squeeze().cpu().item()

            self.buffer.advantages, self.buffer.returns = self.compute_gae(
                self.buffer.rewards, self.buffer.values, self.buffer.dones, nv
            )
            metrics = self.update(self.buffer)

            # Log every ~10k steps
            if total_steps % 10_000 < self.config.rollout_length:
                mean_r = float(np.mean(ep_rewards[-10:])) if ep_rewards else 0.0
                mean_f = float(np.mean(ep_fill_rates[-10:])) if ep_fill_rates else 0.0
                mean_i = float(np.mean(ep_inventories[-10:])) if ep_inventories else 0.0
                log = {
                    "step": total_steps,
                    "mean_episode_reward": mean_r,
                    "fill_rate": mean_f,
                    "mean_inventory": mean_i,
                    **metrics,
                }
                logs.append(log)
                print(
                    f"Step {total_steps:>8d} | "
                    f"Reward: {mean_r:>9.5f} | "
                    f"Fill: {mean_f:.3f} | "
                    f"Inv: {mean_i:.3f} | "
                    f"PL: {metrics['policy_loss']:>9.5f} | "
                    f"VL: {metrics['value_loss']:>9.5f}"
                )

            # Checkpoint every ~100k steps
            if total_steps % 100_000 < self.config.rollout_length:
                path = f"models/checkpoint_{total_steps}.pt"
                torch.save(
                    {
                        "step": total_steps,
                        "actor_state_dict": self.actor.state_dict(),
                        "critic_state_dict": self.critic.state_dict(),
                        "actor_optimizer": self.actor_optimizer.state_dict(),
                        "critic_optimizer": self.critic_optimizer.state_dict(),
                    },
                    path,
                )
                print(f"Checkpoint saved → {path}")

        torch.save(
            {
                "step": total_steps,
                "actor_state_dict": self.actor.state_dict(),
                "critic_state_dict": self.critic.state_dict(),
            },
            "models/checkpoint_final.pt",
        )
        print("Final checkpoint saved → models/checkpoint_final.pt")
        return logs
