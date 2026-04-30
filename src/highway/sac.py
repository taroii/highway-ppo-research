"""
From-scratch SAC (continuous actions).

Used as (a) the continuous upper-bound baseline and (b) the feature source
for the clustered arms -- we pretrain SAC, freeze it, and pull state
embeddings from its actor trunk to fit k-means clusters.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import List

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


LOG_STD_MIN = -5
LOG_STD_MAX = 2


def weight_init(m: nn.Module) -> None:
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        m.bias.data.fill_(0.0)


class GaussianActor(nn.Module):
    """State-only trunk + tanh-squashed Gaussian head.

    The trunk output is what the clustered arms use as a state embedding.
    """

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.fc_mu = nn.Linear(hidden_dim, action_dim)
        self.fc_log_std = nn.Linear(hidden_dim, action_dim)
        self.apply(weight_init)

    def forward(self, obs, compute_pi: bool = True, compute_log_pi: bool = True):
        h = self.trunk(obs)
        mu = self.fc_mu(h)
        log_std = torch.clamp(self.fc_log_std(h), LOG_STD_MIN, LOG_STD_MAX)

        if compute_pi:
            std = log_std.exp()
            noise = torch.randn_like(mu)
            pi = mu + noise * std
        else:
            pi, noise = None, None

        if compute_log_pi and pi is not None:
            log_pi = (-0.5 * noise.pow(2) - log_std).sum(-1, keepdim=True) \
                     - 0.5 * np.log(2 * np.pi) * mu.size(-1)
            log_pi -= torch.log(F.relu(1 - torch.tanh(pi).pow(2)) + 1e-6).sum(-1, keepdim=True)
        else:
            log_pi = None

        mu = torch.tanh(mu)
        if pi is not None:
            pi = torch.tanh(pi)
        return mu, pi, log_pi, log_std


class QFunction(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.apply(weight_init)

    def forward(self, obs, action):
        return self.trunk(torch.cat([obs, action], dim=-1))


class ReplayBuffer:
    def __init__(self, obs_dim: int, action_dim: int, capacity: int, device: torch.device):
        self.capacity = capacity
        self.device = device
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.not_dones = np.zeros((capacity, 1), dtype=np.float32)
        self.idx = 0
        self.full = False

    def add(self, obs, action, reward, next_obs, done):
        np.copyto(self.obs[self.idx], obs)
        np.copyto(self.actions[self.idx], action)
        self.rewards[self.idx] = reward
        np.copyto(self.next_obs[self.idx], next_obs)
        self.not_dones[self.idx] = 0.0 if done else 1.0
        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0

    def sample(self, batch_size: int):
        limit = self.capacity if self.full else self.idx
        idxs = np.random.randint(0, limit, size=batch_size)
        return (
            torch.as_tensor(self.obs[idxs], device=self.device),
            torch.as_tensor(self.actions[idxs], device=self.device),
            torch.as_tensor(self.rewards[idxs], device=self.device),
            torch.as_tensor(self.next_obs[idxs], device=self.device),
            torch.as_tensor(self.not_dones[idxs], device=self.device),
        )

    def sample_obs(self, n: int) -> np.ndarray:
        """Return up to n raw obs from the buffer -- used for cluster fitting."""
        limit = self.capacity if self.full else self.idx
        n = min(n, limit)
        idxs = np.random.choice(limit, size=n, replace=False)
        return self.obs[idxs].copy()

    def __len__(self):
        return self.capacity if self.full else self.idx


def soft_update_params(net: nn.Module, target_net: nn.Module, tau: float) -> None:
    for p, tp in zip(net.parameters(), target_net.parameters()):
        tp.data.copy_(tau * p.data + (1 - tau) * tp.data)


class SAC:
    """Standard continuous-action SAC with auto-tuned entropy."""

    def __init__(
        self,
        env: gym.Env,
        hidden_dim: int = 256,
        gamma: float = 0.99,
        tau: float = 0.005,
        actor_lr: float = 1e-3,
        critic_lr: float = 1e-3,
        alpha_lr: float = 1e-4,
        batch_size: int = 256,
        buffer_capacity: int = 300_000,
        learning_starts: int = 5000,
        actor_update_freq: int = 2,
        critic_target_update_freq: int = 2,
        init_temperature: float = 0.1,
        seed: int = 0,
    ):
        self.env = env
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.learning_starts = learning_starts
        self.actor_update_freq = actor_update_freq
        self.critic_target_update_freq = critic_target_update_freq
        self.hidden_dim = hidden_dim

        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.obs_dim = int(np.prod(env.observation_space.shape))
        self.action_dim = int(np.prod(env.action_space.shape))

        self.actor = GaussianActor(self.obs_dim, self.action_dim, hidden_dim).to(self.device)
        self.q1 = QFunction(self.obs_dim, self.action_dim, hidden_dim).to(self.device)
        self.q2 = QFunction(self.obs_dim, self.action_dim, hidden_dim).to(self.device)
        self.q1_target = QFunction(self.obs_dim, self.action_dim, hidden_dim).to(self.device)
        self.q2_target = QFunction(self.obs_dim, self.action_dim, hidden_dim).to(self.device)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        self.log_alpha = torch.tensor(
            np.log(init_temperature), dtype=torch.float32, device=self.device, requires_grad=True
        )
        self.target_entropy = -self.action_dim

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()), lr=critic_lr
        )
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=alpha_lr)

        self.buffer = ReplayBuffer(self.obs_dim, self.action_dim, buffer_capacity, self.device)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def select_action(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        obs_t = torch.as_tensor(obs.flatten(), dtype=torch.float32,
                                device=self.device).unsqueeze(0)
        with torch.no_grad():
            mu, pi, _, _ = self.actor(obs_t, compute_log_pi=False)
            return (mu if deterministic else pi).cpu().numpy().flatten()

    def _update_critic(self, obs, actions, rewards, next_obs, not_dones):
        with torch.no_grad():
            _, policy_action, log_pi, _ = self.actor(next_obs)
            q1_next = self.q1_target(next_obs, policy_action)
            q2_next = self.q2_target(next_obs, policy_action)
            target_v = torch.min(q1_next, q2_next) - self.alpha.detach() * log_pi
            target_q = rewards + not_dones * self.gamma * target_v

        critic_loss = F.mse_loss(self.q1(obs, actions), target_q) \
                      + F.mse_loss(self.q2(obs, actions), target_q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

    def _update_actor_and_alpha(self, obs):
        _, pi, log_pi, _ = self.actor(obs)
        q_pi = torch.min(self.q1(obs, pi), self.q2(obs, pi))
        actor_loss = (self.alpha.detach() * log_pi - q_pi).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        alpha_loss = (self.alpha * (-log_pi - self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

    def _update(self, step: int) -> None:
        obs, actions, rewards, next_obs, not_dones = self.buffer.sample(self.batch_size)
        self._update_critic(obs, actions, rewards, next_obs, not_dones)
        if step % self.actor_update_freq == 0:
            self._update_actor_and_alpha(obs)
        if step % self.critic_target_update_freq == 0:
            soft_update_params(self.q1, self.q1_target, self.tau)
            soft_update_params(self.q2, self.q2_target, self.tau)

    def learn(self, total_timesteps: int, print_every: int = 10_000) -> List[float]:
        obs, _ = self.env.reset()
        episode_rewards: List[float] = []
        current = 0.0

        for step in range(1, total_timesteps + 1):
            obs_flat = obs.flatten().astype(np.float32)
            if step < self.learning_starts:
                action = self.env.action_space.sample()
            else:
                action = self.select_action(obs_flat)

            next_obs, reward, done, truncated, _ = self.env.step(action)
            terminal = done or truncated
            self.buffer.add(
                obs_flat, np.atleast_1d(action).astype(np.float32),
                reward, next_obs.flatten().astype(np.float32), terminal,
            )
            current += reward

            if terminal:
                episode_rewards.append(current)
                current = 0.0
                next_obs, _ = self.env.reset()
            obs = next_obs

            if step >= self.learning_starts:
                self._update(step)

            if episode_rewards and step % print_every == 0:
                recent = episode_rewards[-50:]
                print(f"[{step:>7d}/{total_timesteps}]  ep={len(episode_rewards):>4d}  "
                      f"reward(last50)={np.mean(recent):>7.2f}  alpha={self.alpha.item():.4f}")

        return episode_rewards

    def save(self, path: str, episode_rewards: List[float] | None = None) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "actor": self.actor.state_dict(),
            "q1": self.q1.state_dict(),
            "q2": self.q2.state_dict(),
            "log_alpha": self.log_alpha.detach().item(),
            "obs_dim": self.obs_dim,
            "action_dim": self.action_dim,
            "hidden_dim": self.hidden_dim,
            "episode_rewards": episode_rewards or [],
        }, path)
        print(f"Saved SAC checkpoint to {path}")

    @classmethod
    def load(cls, path: str, env: gym.Env) -> "SAC":
        data = torch.load(path, weights_only=False)
        agent = cls(env, hidden_dim=data.get("hidden_dim", 256))
        agent.actor.load_state_dict(data["actor"])
        agent.q1.load_state_dict(data["q1"])
        agent.q2.load_state_dict(data["q2"])
        agent.q1_target.load_state_dict(data["q1"])
        agent.q2_target.load_state_dict(data["q2"])
        agent.log_alpha = torch.tensor(
            data["log_alpha"], dtype=torch.float32,
            device=agent.device, requires_grad=True
        )
        return agent

    def predict(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        return self.select_action(obs, deterministic=deterministic)
