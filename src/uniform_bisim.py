"""
Discrete-action SAC with bisimulation metric representation learning
for HighwayEnv.

Adapts the Deep Bisimulation for Control (DBC) algorithm from
Zhang et al. (ICLR 2021) to a discrete action space. The encoder
is an MLP (not CNN) since HighwayEnv provides flat kinematic
observations.

Key differences from the original DBC paper:
  - Discrete SAC: Q-networks output Q-values for all actions;
    policy is a Categorical distribution over actions.
  - Action is passed to the transition model as a one-hot vector.
  - No pixel encoder; uses a small MLP encoder instead.
"""

from __future__ import annotations

import math
import random
from pathlib import Path
from typing import List

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

try:
    import highway_env  # noqa: F401
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Reward wrapper (same as ppo.py)
# ---------------------------------------------------------------------------

class CustomRewardWrapper(gym.Wrapper):
    """Custom reward computed directly from vehicle state."""

    def __init__(self, env):
        super().__init__(env)
        self._last_x = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._last_x = self.env.unwrapped.vehicle.position[0]
        return obs, info

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)
        vehicle = self.env.unwrapped.vehicle
        road = self.env.unwrapped.road

        crashed = float(vehicle.crashed)
        on_road = float(vehicle.on_road)
        speed = float(np.clip((vehicle.speed - 20) / 10, 0, 1))

        neighbours = road.network.all_side_lanes(vehicle.lane_index)
        right_lane = vehicle.lane_index[2] / max(len(neighbours) - 1, 1)

        delta_x = vehicle.position[0] - self._last_x
        self._last_x = vehicle.position[0]
        progress = delta_x / 30

        heading_align = math.cos(vehicle.heading)
        steering = abs(vehicle.action["steering"]) / (np.pi / 4)

        raw = (
            -1.0 * crashed
            + 0.4 * speed
            + 0.1 * right_lane
            + 0.2 * progress
            + 0.1 * heading_align
            - 0.1 * steering
        )
        reward = (raw - (-1.5)) / (0.8 - (-1.5)) * on_road
        return obs, reward, terminated, truncated, info


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

class MLPEncoder(nn.Module):
    """MLP encoder for flat (kinematic) observations."""

    def __init__(self, obs_dim: int, feature_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.feature_dim = feature_dim
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim),
            nn.LayerNorm(feature_dim),
        )

    def forward(self, obs, detach=False):
        h = self.net(obs)
        if detach:
            h = h.detach()
        return h


# ---------------------------------------------------------------------------
# Transition model
# ---------------------------------------------------------------------------

class ProbabilisticTransitionModel(nn.Module):
    """Predicts next latent state distribution given (z, action_onehot)."""

    def __init__(self, feature_dim: int, n_actions: int, hidden_dim: int = 256,
                 min_sigma: float = 1e-4, max_sigma: float = 10.0):
        super().__init__()
        self.fc = nn.Linear(feature_dim + n_actions, hidden_dim)
        self.ln = nn.LayerNorm(hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, feature_dim)
        self.fc_sigma = nn.Linear(hidden_dim, feature_dim)
        self.min_sigma = min_sigma
        self.max_sigma = max_sigma

    def forward(self, x):
        """x: concatenation of [z, action_onehot]."""
        x = torch.relu(self.ln(self.fc(x)))
        mu = self.fc_mu(x)
        sigma = torch.sigmoid(self.fc_sigma(x))
        sigma = self.min_sigma + (self.max_sigma - self.min_sigma) * sigma
        return mu, sigma

    def sample_prediction(self, x):
        mu, sigma = self(x)
        return mu + sigma * torch.randn_like(sigma)


# ---------------------------------------------------------------------------
# Reward decoder
# ---------------------------------------------------------------------------

class RewardDecoder(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, z):
        return self.net(z)


# ---------------------------------------------------------------------------
# Discrete-action Q-network
# ---------------------------------------------------------------------------

class QNetwork(nn.Module):
    """Outputs Q-values for all discrete actions given encoded obs."""

    def __init__(self, feature_dim: int, n_actions: int, hidden_dim: int = 256):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        )

    def forward(self, z):
        return self.trunk(z)


# ---------------------------------------------------------------------------
# Discrete-action Actor (categorical policy)
# ---------------------------------------------------------------------------

class DiscreteActor(nn.Module):
    def __init__(self, feature_dim: int, n_actions: int, hidden_dim: int = 256):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        )

    def forward(self, z):
        logits = self.trunk(z)
        dist = Categorical(logits=logits)
        action = dist.sample()
        # For entropy-tuned SAC we need action probs and log-probs for *all* actions
        probs = dist.probs
        log_probs = torch.log(probs + 1e-8)
        return action, probs, log_probs


# ---------------------------------------------------------------------------
# Replay buffer
# ---------------------------------------------------------------------------

class ReplayBuffer:
    """Simple circular replay buffer for off-policy learning."""

    def __init__(self, obs_dim: int, capacity: int, device: torch.device):
        self.capacity = capacity
        self.device = device

        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.not_dones = np.zeros((capacity, 1), dtype=np.float32)

        self.idx = 0
        self.full = False

    def add(self, obs, action, reward, next_obs, done):
        np.copyto(self.obs[self.idx], obs)
        self.actions[self.idx] = action
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
# BisimSAC agent
# ---------------------------------------------------------------------------

class BisimSAC:
    """Discrete-action SAC with bisimulation representation learning."""

    def __init__(
        self,
        env: gym.Env,
        feature_dim: int = 10,
        hidden_dim: int = 256,
        gamma: float = 0.99,
        tau: float = 0.005,
        encoder_tau: float = 0.005,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        encoder_lr: float = 3e-4,
        alpha_lr: float = 3e-4,
        transition_lr: float = 3e-4,
        batch_size: int = 128,
        buffer_capacity: int = 100_000,
        learning_starts: int = 1000,
        actor_update_freq: int = 2,
        critic_target_update_freq: int = 2,
        bisim_coef: float = 0.5,
        init_temperature: float = 0.1,
        seed: int = 0,
    ):
        self.env = env
        self.gamma = gamma
        self.tau = tau
        self.encoder_tau = encoder_tau
        self.batch_size = batch_size
        self.learning_starts = learning_starts
        self.actor_update_freq = actor_update_freq
        self.critic_target_update_freq = critic_target_update_freq
        self.bisim_coef = bisim_coef

        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        self.device = torch.device("cpu")
        obs_dim = int(np.prod(env.observation_space.shape))
        n_actions = env.action_space.n
        self.n_actions = n_actions
        self.obs_dim = obs_dim

        # Networks
        self.encoder = MLPEncoder(obs_dim, feature_dim, hidden_dim).to(self.device)
        self.encoder_target = MLPEncoder(obs_dim, feature_dim, hidden_dim).to(self.device)
        self.encoder_target.load_state_dict(self.encoder.state_dict())

        self.actor = DiscreteActor(feature_dim, n_actions, hidden_dim).to(self.device)

        self.q1 = QNetwork(feature_dim, n_actions, hidden_dim).to(self.device)
        self.q2 = QNetwork(feature_dim, n_actions, hidden_dim).to(self.device)
        self.q1_target = QNetwork(feature_dim, n_actions, hidden_dim).to(self.device)
        self.q2_target = QNetwork(feature_dim, n_actions, hidden_dim).to(self.device)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        self.transition_model = ProbabilisticTransitionModel(
            feature_dim, n_actions, hidden_dim
        ).to(self.device)
        self.reward_decoder = RewardDecoder(feature_dim, hidden_dim).to(self.device)

        # Entropy temperature (auto-tuned)
        self.log_alpha = torch.tensor(np.log(init_temperature), dtype=torch.float32,
                                      device=self.device, requires_grad=True)
        self.target_entropy = -np.log(1.0 / n_actions) * 0.98

        # Optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()), lr=critic_lr
        )
        self.encoder_optimizer = torch.optim.Adam(self.encoder.parameters(), lr=encoder_lr)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=alpha_lr)
        self.transition_optimizer = torch.optim.Adam(
            list(self.transition_model.parameters()) +
            list(self.reward_decoder.parameters()),
            lr=transition_lr,
        )

        # Replay buffer
        self.buffer = ReplayBuffer(obs_dim, buffer_capacity, self.device)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def select_action(self, obs: np.ndarray, deterministic: bool = False) -> int:
        obs_t = torch.as_tensor(obs.flatten(), dtype=torch.float32,
                                device=self.device).unsqueeze(0)
        with torch.no_grad():
            z = self.encoder(obs_t)
            if deterministic:
                logits = self.actor.trunk(z)
                return logits.argmax(dim=-1).item()
            action, _, _ = self.actor(z)
            return action.item()

    # ------------------------------------------------------------------
    # One-hot helper
    # ------------------------------------------------------------------

    def _action_onehot(self, actions: torch.Tensor) -> torch.Tensor:
        return F.one_hot(actions, self.n_actions).float()

    # ------------------------------------------------------------------
    # Critic update (discrete SAC)
    # ------------------------------------------------------------------

    def _update_critic(self, obs, actions, rewards, next_obs, not_dones):
        with torch.no_grad():
            z_next = self.encoder(next_obs)
            _, next_probs, next_log_probs = self.actor(z_next)
            q1_next = self.q1_target(z_next)
            q2_next = self.q2_target(z_next)
            q_next = torch.min(q1_next, q2_next)
            # Expectation over actions: sum_a pi(a|s') [Q(s',a) - alpha * log pi(a|s')]
            v_next = (next_probs * (q_next - self.alpha.detach() * next_log_probs)).sum(dim=-1, keepdim=True)
            target_q = rewards + not_dones * self.gamma * v_next

        z = self.encoder(obs)
        q1_all = self.q1(z)
        q2_all = self.q2(z)
        q1_pred = q1_all.gather(1, actions.unsqueeze(1))
        q2_pred = q2_all.gather(1, actions.unsqueeze(1))
        critic_loss = F.mse_loss(q1_pred, target_q) + F.mse_loss(q2_pred, target_q)

        self.critic_optimizer.zero_grad()
        self.encoder_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        self.encoder_optimizer.step()

        return critic_loss.item()

    # ------------------------------------------------------------------
    # Actor and alpha update (discrete SAC)
    # ------------------------------------------------------------------

    def _update_actor_and_alpha(self, obs):
        z = self.encoder(obs, detach=True)
        _, probs, log_probs = self.actor(z)
        q1_all = self.q1(z.detach())
        q2_all = self.q2(z.detach())
        q_all = torch.min(q1_all, q2_all)

        # Policy loss: E_a~pi [alpha * log pi(a|s) - Q(s, a)]
        actor_loss = (probs * (self.alpha.detach() * log_probs - q_all)).sum(dim=-1).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Alpha loss
        entropy = -(probs.detach() * log_probs.detach()).sum(dim=-1)
        alpha_loss = (self.log_alpha * (entropy - self.target_entropy)).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        return actor_loss.item()

    # ------------------------------------------------------------------
    # Transition + reward model update
    # ------------------------------------------------------------------

    def _update_transition_reward(self, obs, actions, rewards, next_obs):
        z = self.encoder(obs, detach=True)
        z_next = self.encoder(next_obs, detach=True)

        action_oh = self._action_onehot(actions)
        pred_mu, pred_sigma = self.transition_model(torch.cat([z, action_oh], dim=1))

        # Gaussian NLL for transition
        diff = (pred_mu - z_next) / pred_sigma
        transition_loss = (0.5 * diff.pow(2) + torch.log(pred_sigma)).mean()

        # Reward prediction from predicted next latent
        pred_z_next = self.transition_model.sample_prediction(torch.cat([z, action_oh], dim=1))
        pred_reward = self.reward_decoder(pred_z_next)
        reward_loss = F.mse_loss(pred_reward, rewards)

        total_loss = transition_loss + reward_loss

        self.transition_optimizer.zero_grad()
        total_loss.backward()
        self.transition_optimizer.step()

        return total_loss.item()

    # ------------------------------------------------------------------
    # Bisimulation encoder update (Equation 4 from the paper)
    # ------------------------------------------------------------------

    def _update_encoder_bisim(self, obs, actions, rewards):
        z = self.encoder(obs)

        # Create random pairings by permuting the batch
        batch_size = obs.size(0)
        perm = torch.randperm(batch_size)
        z2 = z[perm]

        with torch.no_grad():
            action_oh = self._action_onehot(actions)
            pred_mu1, pred_sigma1 = self.transition_model(torch.cat([z.detach(), action_oh], dim=1))
            pred_mu2 = pred_mu1[perm]
            pred_sigma2 = pred_sigma1[perm]
            rewards2 = rewards[perm]

        # Latent distance: ||z_i - z_j||_1
        z_dist = F.smooth_l1_loss(z, z2, reduction='none')

        # Reward distance: |r_i - r_j|
        r_dist = F.smooth_l1_loss(rewards, rewards2, reduction='none')

        # Transition distance: W_2 between predicted Gaussians
        transition_dist = torch.sqrt(
            (pred_mu1 - pred_mu2).pow(2) + (pred_sigma1 - pred_sigma2).pow(2)
        )

        # Bisimilarity target
        bisimilarity = r_dist + self.gamma * transition_dist
        encoder_loss = (z_dist - bisimilarity).pow(2).mean()

        return encoder_loss

    # ------------------------------------------------------------------
    # Combined update step
    # ------------------------------------------------------------------

    def _update(self, step: int):
        obs, actions, rewards, next_obs, not_dones = self.buffer.sample(self.batch_size)

        # 1. Critic (also updates encoder via backprop)
        self._update_critic(obs, actions, rewards, next_obs, not_dones)

        # 2. Transition model + reward decoder
        transition_loss = self._update_transition_reward(obs, actions, rewards, next_obs)

        # 3. Bisimulation encoder loss
        encoder_loss = self._update_encoder_bisim(obs, actions, rewards)
        total_aux = self.bisim_coef * encoder_loss

        self.encoder_optimizer.zero_grad()
        total_aux.backward()
        self.encoder_optimizer.step()

        # 4. Actor + alpha (every N steps)
        if step % self.actor_update_freq == 0:
            self._update_actor_and_alpha(obs)

        # 5. Soft-update targets
        if step % self.critic_target_update_freq == 0:
            soft_update_params(self.q1, self.q1_target, self.tau)
            soft_update_params(self.q2, self.q2_target, self.tau)
            soft_update_params(self.encoder, self.encoder_target, self.encoder_tau)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def learn(self, total_timesteps: int, print_every: int = 1000):
        obs, _ = self.env.reset()
        episode_rewards: List[float] = []
        current_ep_reward = 0.0

        for step in range(1, total_timesteps + 1):
            obs_flat = obs.flatten().astype(np.float32)

            # Random exploration until learning starts
            if step < self.learning_starts:
                action = self.env.action_space.sample()
            else:
                action = self.select_action(obs_flat)

            next_obs, reward, done, truncated, info = self.env.step(action)
            next_obs_flat = next_obs.flatten().astype(np.float32)
            terminal = done or truncated

            self.buffer.add(obs_flat, action, reward, next_obs_flat, terminal)
            current_ep_reward += reward

            if terminal:
                episode_rewards.append(current_ep_reward)
                current_ep_reward = 0.0
                next_obs, _ = self.env.reset()

            obs = next_obs

            # Update after enough data
            if step >= self.learning_starts:
                self._update(step)

            if episode_rewards and step % print_every == 0:
                recent = episode_rewards[-50:]
                print(
                    f"Steps: {step}/{total_timesteps}, "
                    f"episodes: {len(episode_rewards)}, "
                    f"mean_reward(last 50): {np.mean(recent):.2f}"
                )

        return episode_rewards

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------

    def save(self, path: str, episode_rewards: List[float] = None):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "encoder": self.encoder.state_dict(),
            "actor": self.actor.state_dict(),
            "q1": self.q1.state_dict(),
            "q2": self.q2.state_dict(),
            "transition_model": self.transition_model.state_dict(),
            "reward_decoder": self.reward_decoder.state_dict(),
            "log_alpha": self.log_alpha.detach().item(),
            "episode_rewards": episode_rewards or [],
        }, path)
        print(f"Saved BisimSAC checkpoint to {path}")

    def predict(self, obs: np.ndarray, deterministic: bool = False) -> int:
        return self.select_action(obs, deterministic=deterministic)


# ---------------------------------------------------------------------------
# Environment factory
# ---------------------------------------------------------------------------

def make_highway_env():
    env = gym.make(
        "highway-fast-v0",
        config={
            "action": {
                "type": "DiscreteAction",
                "longitudinal": False,
                "lateral": True,
                "actions_per_axis": 5,
            },
        },
    )
    return CustomRewardWrapper(env)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    TOTAL_TIMESTEPS = 100_000
    env = make_highway_env()

    agent = BisimSAC(
        env,
        feature_dim=10,
        hidden_dim=256,
        gamma=0.99,
        batch_size=128,
        buffer_capacity=100_000,
        learning_starts=1000,
        bisim_coef=0.5,
        seed=42,
    )

    print("Starting Bisimulation SAC training...")
    print(f"Obs shape: {env.observation_space.shape}")
    print(f"Action space: {env.action_space}")
    print()

    episode_rewards = agent.learn(total_timesteps=TOTAL_TIMESTEPS, print_every=10000)

    agent.save("checkpoints/bisim_sac.pt", episode_rewards)

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
