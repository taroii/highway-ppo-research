"""
From-scratch SAC for HighwayEnv (continuous steering).

Standard Soft Actor-Critic with:
  - Gaussian actor with tanh squashing
  - Twin Q-functions + targets with soft updates
  - Auto-tuned entropy temperature

No encoder or representation learning -- operates directly on flat
kinematic observations.  Serves as a baseline for comparing against
BisimSAC (which adds bisimulation representation learning on top of
the same SAC core).
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

try:
    import highway_env  # noqa: F401
except ImportError:
    pass

from throwaway.highway.ppo import CustomRewardWrapper


# ---------------------------------------------------------------------------
# Weight init
# ---------------------------------------------------------------------------

def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        m.bias.data.fill_(0.0)


# ---------------------------------------------------------------------------
# Gaussian actor (tanh-squashed)
# ---------------------------------------------------------------------------

LOG_STD_MIN = -5
LOG_STD_MAX = 2


class GaussianActor(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.fc_mu = nn.Linear(hidden_dim, action_dim)
        self.fc_log_std = nn.Linear(hidden_dim, action_dim)
        self.apply(weight_init)

    def forward(self, obs, compute_pi=True, compute_log_pi=True):
        h = self.trunk(obs)
        mu = self.fc_mu(h)
        log_std = self.fc_log_std(h)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)

        if compute_pi:
            std = log_std.exp()
            noise = torch.randn_like(mu)
            pi = mu + noise * std
        else:
            pi = None
            noise = None

        if compute_log_pi and pi is not None:
            log_pi = (
                -0.5 * noise.pow(2) - log_std
            ).sum(-1, keepdim=True) - 0.5 * np.log(2 * np.pi) * mu.size(-1)
            log_pi -= torch.log(F.relu(1 - torch.tanh(pi).pow(2)) + 1e-6).sum(-1, keepdim=True)
        else:
            log_pi = None

        mu = torch.tanh(mu)
        if pi is not None:
            pi = torch.tanh(pi)

        return mu, pi, log_pi, log_std


# ---------------------------------------------------------------------------
# Q-function
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Replay buffer
# ---------------------------------------------------------------------------

class ReplayBuffer:
    def __init__(self, obs_dim: int, action_dim: int, capacity: int,
                 device: torch.device):
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

    def __len__(self):
        return self.capacity if self.full else self.idx


# ---------------------------------------------------------------------------
# Soft update helper
# ---------------------------------------------------------------------------

def soft_update_params(net: nn.Module, target_net: nn.Module, tau: float):
    for p, tp in zip(net.parameters(), target_net.parameters()):
        tp.data.copy_(tau * p.data + (1 - tau) * tp.data)


# ---------------------------------------------------------------------------
# SAC agent
# ---------------------------------------------------------------------------

class SAC:
    """Standard Soft Actor-Critic with continuous actions."""

    def __init__(
        self,
        env: gym.Env,
        hidden_dim: int = 256,
        gamma: float = 0.99,
        tau: float = 0.005,
        actor_lr: float = 1e-3,
        critic_lr: float = 1e-3,
        alpha_lr: float = 1e-4,
        batch_size: int = 128,
        buffer_capacity: int = 100_000,
        learning_starts: int = 1000,
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

        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        self.device = torch.device("cpu")
        obs_dim = int(np.prod(env.observation_space.shape))
        action_dim = int(np.prod(env.action_space.shape))
        self.action_dim = action_dim
        self.obs_dim = obs_dim

        # Actor
        self.actor = GaussianActor(obs_dim, action_dim, hidden_dim).to(self.device)

        # Twin Q-functions + targets
        self.q1 = QFunction(obs_dim, action_dim, hidden_dim).to(self.device)
        self.q2 = QFunction(obs_dim, action_dim, hidden_dim).to(self.device)
        self.q1_target = QFunction(obs_dim, action_dim, hidden_dim).to(self.device)
        self.q2_target = QFunction(obs_dim, action_dim, hidden_dim).to(self.device)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        # Auto-tuned entropy temperature
        self.log_alpha = torch.tensor(np.log(init_temperature), dtype=torch.float32,
                                      device=self.device, requires_grad=True)
        self.target_entropy = -action_dim

        # Optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()), lr=critic_lr
        )
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=alpha_lr)

        # Replay buffer
        self.buffer = ReplayBuffer(obs_dim, action_dim, buffer_capacity, self.device)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def select_action(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        obs_t = torch.as_tensor(obs.flatten(), dtype=torch.float32,
                                device=self.device).unsqueeze(0)
        with torch.no_grad():
            mu, pi, _, _ = self.actor(obs_t, compute_log_pi=False)
            if deterministic:
                return mu.cpu().numpy().flatten()
            return pi.cpu().numpy().flatten()

    # ------------------------------------------------------------------
    # Critic update
    # ------------------------------------------------------------------

    def _update_critic(self, obs, actions, rewards, next_obs, not_dones):
        with torch.no_grad():
            _, policy_action, log_pi, _ = self.actor(next_obs)
            q1_next = self.q1_target(next_obs, policy_action)
            q2_next = self.q2_target(next_obs, policy_action)
            target_v = torch.min(q1_next, q2_next) - self.alpha.detach() * log_pi
            target_q = rewards + not_dones * self.gamma * target_v

        q1_pred = self.q1(obs, actions)
        q2_pred = self.q2(obs, actions)
        critic_loss = F.mse_loss(q1_pred, target_q) + F.mse_loss(q2_pred, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

    # ------------------------------------------------------------------
    # Actor and alpha update
    # ------------------------------------------------------------------

    def _update_actor_and_alpha(self, obs):
        _, pi, log_pi, _ = self.actor(obs)
        q1_pi = self.q1(obs, pi)
        q2_pi = self.q2(obs, pi)
        q_pi = torch.min(q1_pi, q2_pi)

        actor_loss = (self.alpha.detach() * log_pi - q_pi).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Alpha update
        alpha_loss = (self.alpha * (-log_pi - self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

    # ------------------------------------------------------------------
    # Combined update
    # ------------------------------------------------------------------

    def _update(self, step: int):
        obs, actions, rewards, next_obs, not_dones = self.buffer.sample(self.batch_size)

        self._update_critic(obs, actions, rewards, next_obs, not_dones)

        if step % self.actor_update_freq == 0:
            self._update_actor_and_alpha(obs)

        if step % self.critic_target_update_freq == 0:
            soft_update_params(self.q1, self.q1_target, self.tau)
            soft_update_params(self.q2, self.q2_target, self.tau)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def learn(self, total_timesteps: int, print_every: int = 1000):
        obs, _ = self.env.reset()
        episode_rewards: List[float] = []
        current_ep_reward = 0.0

        for step in range(1, total_timesteps + 1):
            obs_flat = obs.flatten().astype(np.float32)

            if step < self.learning_starts:
                action = self.env.action_space.sample()
            else:
                action = self.select_action(obs_flat)

            next_obs, reward, done, truncated, _ = self.env.step(action)
            next_obs_flat = next_obs.flatten().astype(np.float32)
            terminal = done or truncated

            action_flat = np.atleast_1d(action).astype(np.float32)
            self.buffer.add(obs_flat, action_flat, reward, next_obs_flat, terminal)
            current_ep_reward += reward

            if terminal:
                episode_rewards.append(current_ep_reward)
                current_ep_reward = 0.0
                next_obs, _ = self.env.reset()

            obs = next_obs

            if step >= self.learning_starts:
                self._update(step)

            if episode_rewards and step % print_every == 0:
                recent = episode_rewards[-50:]
                print(
                    f"Steps: {step}/{total_timesteps}, "
                    f"episodes: {len(episode_rewards)}, "
                    f"mean_reward(last 50): {np.mean(recent):.2f}, "
                    f"alpha: {self.alpha.item():.4f}"
                )

        return episode_rewards

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------

    def save(self, path: str, episode_rewards: List[float] = None):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "actor": self.actor.state_dict(),
            "q1": self.q1.state_dict(),
            "q2": self.q2.state_dict(),
            "log_alpha": self.log_alpha.detach().item(),
            "episode_rewards": episode_rewards or [],
        }, path)
        print(f"Saved SAC checkpoint to {path}")

    @classmethod
    def load(cls, path: str, env: gym.Env) -> "SAC":
        data = torch.load(path, weights_only=False)
        agent = cls(env)
        agent.actor.load_state_dict(data["actor"])
        agent.q1.load_state_dict(data["q1"])
        agent.q2.load_state_dict(data["q2"])
        agent.q1_target.load_state_dict(data["q1"])
        agent.q2_target.load_state_dict(data["q2"])
        agent.log_alpha = torch.tensor(
            data["log_alpha"], dtype=torch.float32, requires_grad=True
        )
        return agent

    def predict(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        return self.select_action(obs, deterministic=deterministic)


# ---------------------------------------------------------------------------
# Environment factory
# ---------------------------------------------------------------------------

def make_highway_env_continuous():
    env = gym.make(
        "highway-fast-v0",
        config={
            "action": {
                "type": "ContinuousAction",
                "longitudinal": False,
                "lateral": True,
            },
        },
    )
    return CustomRewardWrapper(env)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    TOTAL_TIMESTEPS = 200_000
    env = make_highway_env_continuous()

    agent = SAC(
        env,
        hidden_dim=256,
        gamma=0.99,
        tau=0.005,
        actor_lr=1e-3,
        critic_lr=1e-3,
        alpha_lr=1e-4,
        batch_size=128,
        buffer_capacity=100_000,
        learning_starts=1000,
        actor_update_freq=2,
        critic_target_update_freq=2,
        init_temperature=0.1,
        seed=42,
    )

    print("Starting SAC training...")
    print(f"Obs shape: {env.observation_space.shape}")
    print(f"Action space: {env.action_space}")
    print()

    episode_rewards = agent.learn(total_timesteps=TOTAL_TIMESTEPS, print_every=10000)

    agent.save("checkpoints/sac.pt", episode_rewards)

    # Evaluate
    print("\nEvaluating (deterministic)...")
    eval_rewards = []
    for _ in range(20):
        obs, _ = env.reset()
        total_reward = 0.0
        done = truncated = False
        while not (done or truncated):
            action = agent.predict(obs, deterministic=True)
            obs, reward, done, truncated, _ = env.step(action)
            total_reward += reward
        eval_rewards.append(total_reward)

    print(f"Eval over 20 episodes: mean={np.mean(eval_rewards):.2f}, std={np.std(eval_rewards):.2f}")
    env.close()
