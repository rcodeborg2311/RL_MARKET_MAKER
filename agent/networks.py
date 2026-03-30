"""Actor and Critic networks with LayerNorm for training stability."""
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal
from typing import Tuple


def _mlp(dims: list, activation=nn.ReLU, layer_norm: bool = True) -> nn.Sequential:
    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:  # not last layer
            if layer_norm:
                layers.append(nn.LayerNorm(dims[i + 1]))
            layers.append(activation())
    return nn.Sequential(*layers)


class Actor(nn.Module):
    """
    State → continuous action (2-dim).
    Architecture: Linear(state_dim→256)→LN→ReLU→Linear(256→128)→LN→ReLU→Linear(128→action_dim)
    Output: tanh-squashed Gaussian, scaled to [-5, 5].
    """

    def __init__(self, state_dim: int = 20, action_dim: int = 2) -> None:
        super().__init__()
        self.net = _mlp([state_dim, 256, 128, action_dim], layer_norm=True)
        self.log_std = nn.Parameter(torch.full((action_dim,), -0.5))

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
        # Final linear layer smaller init
        for m in reversed(list(self.modules())):
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.01)
                break

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.net(state), self.log_std.expand_as(self.net(state))

    def get_action(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean = self.net(state)
        std = torch.exp(self.log_std.clamp(-20, 2))
        dist = Normal(mean, std)
        x = dist.rsample()
        action = torch.tanh(x) * 5.0
        log_prob = (dist.log_prob(x) - torch.log(1 - torch.tanh(x).pow(2) + 1e-7)).sum(-1, keepdim=True)
        entropy = dist.entropy().sum(-1, keepdim=True)
        return action, log_prob, entropy


class Critic(nn.Module):
    """State → V(s). Architecture: 256→128→64→1 with LayerNorm."""

    def __init__(self, state_dim: int = 20) -> None:
        super().__init__()
        self.net = _mlp([state_dim, 256, 128, 64, 1], layer_norm=True)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)


class ActorCritic(nn.Module):
    def __init__(self, state_dim: int = 20, action_dim: int = 2) -> None:
        super().__init__()
        self.actor = Actor(state_dim, action_dim)
        self.critic = Critic(state_dim)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        action, log_prob, entropy = self.actor.get_action(state)
        return action, log_prob, entropy, self.critic(state)

    def get_action(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.actor.get_action(state)
