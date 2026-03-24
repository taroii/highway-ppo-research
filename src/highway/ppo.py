"""
From-scratch PPO for HighwayEnv (mirrors stable_baselines3.PPO behaviour).

Discrete steering action (5 uniform angles), MLP policy & value networks,
GAE advantage estimation, clipped surrogate objective, entropy bonus.

Same environment setup as sb3_highway_ppo.py but with PPO written out
instead of importing from stable_baselines3.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
import gymnasium as gym
from typing import List, Tuple, Dict
from pathlib import Path
import math

try:
    import highway_env  # noqa: F401
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Steering penalty wrapper
# ---------------------------------------------------------------------------

class CustomRewardWrapper(gym.Wrapper):
    """Custom reward computed directly from vehicle state.

    raw = -1.0*crashed + 0.4*speed + 0.1*right_lane + 0.2*progress + 0.1*heading_align - 0.1*steering
    reward = lmap(raw, [-1.5, 0.8], [0, 1]) * on_road

    Changes from previous version:
      - progress = delta_x / 30, unclipped (negative when moving backward).
        Removes the asymmetric clipping that made circles profitable.
      - heading_align = cos(vehicle.heading): +0.1 when facing forward (+x),
        -0.1 when facing backward, 0 when facing sideways.
        Penalises sustained turning regardless of instantaneous delta_x.
      - Normalization bounds updated to [-1.5, 0.8] to reflect new range.

    Worst case: -1.0 (crash) + 0.0 (speed) + 0.0 (right_lane)
                - 0.2 (progress, hard reverse) - 0.1 (heading, facing back)
                - 0.1 (steering) = -1.5
    Best case:   0.0 + 0.4 + 0.1 + 0.2 (progress) + 0.1 (heading) - 0.0 = +0.8
    """

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

        # Unclipped progress — negative delta_x now hurts
        delta_x = vehicle.position[0] - self._last_x
        self._last_x = vehicle.position[0]
        progress = delta_x / 30

        # Heading alignment: cos(heading) = 1.0 facing +x, -1.0 facing -x, 0 sideways.
        # Scaled to [-0.1, +0.1] contribution.
        heading_align = math.cos(vehicle.heading)

        steering = abs(vehicle.action["steering"]) / (np.pi / 4)

        raw = (
            -1.0 * crashed
            + 0.4  * speed
            + 0.1  * right_lane
            + 0.2  * progress
            + 0.1  * heading_align
            - 0.1  * steering
        )

        # Normalize to [0, 1]; multiply by on_road so off-road = 0 reward
        reward = (raw - (-1.5)) / (0.8 - (-1.5)) * on_road
        return obs, reward, terminated, truncated, info


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------

class ActorCritic(nn.Module):
    """Separate MLP heads for policy (actor) and value (critic)."""

    def __init__(self, obs_dim: int, n_actions: int, hidden: List[int] = [256, 256]):
        super().__init__()
        # Policy network
        layers_pi: List[nn.Module] = []
        prev = obs_dim
        for h in hidden:
            layers_pi += [nn.Linear(prev, h), nn.Tanh()]
            prev = h
        layers_pi.append(nn.Linear(prev, n_actions))
        self.pi = nn.Sequential(*layers_pi)

        # Value network
        layers_vf: List[nn.Module] = []
        prev = obs_dim
        for h in hidden:
            layers_vf += [nn.Linear(prev, h), nn.Tanh()]
            prev = h
        layers_vf.append(nn.Linear(prev, 1))
        self.vf = nn.Sequential(*layers_vf)

    def policy(self, obs: torch.Tensor) -> Categorical:
        return Categorical(logits=self.pi(obs))

    def value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.vf(obs).squeeze(-1)

    def act(self, obs: torch.Tensor):
        dist = self.policy(obs)
        action = dist.sample()
        return action, dist.log_prob(action), self.value(obs)

    def evaluate(self, obs: torch.Tensor, actions: torch.Tensor):
        dist = self.policy(obs)
        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()
        value = self.value(obs)
        return log_prob, entropy, value


# ---------------------------------------------------------------------------
# Rollout buffer
# ---------------------------------------------------------------------------

class RolloutBuffer:
    """Stores one rollout of n_steps transitions."""

    def __init__(self):
        self.obs: List[np.ndarray] = []
        self.actions: List[int] = []
        self.rewards: List[float] = []
        self.dones: List[bool] = []
        self.log_probs: List[float] = []
        self.values: List[float] = []

    def store(self, obs, action, reward, done, log_prob, value):
        self.obs.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)

    def clear(self):
        self.__init__()

    def __len__(self):
        return len(self.obs)


# ---------------------------------------------------------------------------
# PPO
# ---------------------------------------------------------------------------

class PPO:
    def __init__(
        self,
        env: gym.Env,
        hidden: List[int] = [256, 256],
        n_steps: int = 128,
        batch_size: int = 64,
        n_epochs: int = 10,
        lr: float = 5e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_eps: float = 0.2,
        vf_coef: float = 0.5,
        ent_coef: float = 0.0,
        max_grad_norm: float = 0.5,
        seed: int = 0,
    ):
        self.env = env
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.max_grad_norm = max_grad_norm

        torch.manual_seed(seed)
        np.random.seed(seed)

        obs_dim = int(np.prod(env.observation_space.shape))
        n_actions = env.action_space.n
        self.net = ActorCritic(obs_dim, n_actions, hidden)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)

        self.buffer = RolloutBuffer()
        # Running state
        self._obs, _ = env.reset(seed=seed)
        self._done = False

    # ------------------------------------------------------------------
    # GAE
    # ------------------------------------------------------------------

    def _compute_gae(self, last_value: float) -> Tuple[np.ndarray, np.ndarray]:
        rewards = np.array(self.buffer.rewards)
        values = np.array(self.buffer.values)
        dones = np.array(self.buffer.dones, dtype=float)

        T = len(rewards)
        advantages = np.zeros(T)
        last_gae = 0.0
        for t in reversed(range(T)):
            if t == T - 1:
                next_val = last_value
                next_nonterminal = 1.0 - float(self._done)
            else:
                next_val = values[t + 1]
                next_nonterminal = 1.0 - dones[t]
            delta = rewards[t] + self.gamma * next_val * next_nonterminal - values[t]
            advantages[t] = last_gae = (
                delta + self.gamma * self.gae_lambda * next_nonterminal * last_gae
            )
        returns = advantages + values
        return advantages, returns

    # ------------------------------------------------------------------
    # Rollout collection
    # ------------------------------------------------------------------

    def _collect_rollouts(self):
        self.buffer.clear()
        self.net.eval()

        for _ in range(self.n_steps):
            obs_flat = self._obs.flatten().astype(np.float32)
            obs_t = torch.from_numpy(obs_flat).unsqueeze(0)

            with torch.no_grad():
                action, log_prob, value = self.net.act(obs_t)

            a = action.item()
            lp = log_prob.item()
            v = value.item()

            next_obs, reward, done, truncated, info = self.env.step(a)
            self.buffer.store(obs_flat, a, reward, done or truncated, lp, v)

            if done or truncated:
                next_obs, _ = self.env.reset()
            self._obs = next_obs
            self._done = done or truncated

        # Bootstrap value for last obs
        with torch.no_grad():
            obs_t = torch.from_numpy(self._obs.flatten().astype(np.float32)).unsqueeze(0)
            last_value = self.net.value(obs_t).item()

        return self._compute_gae(last_value)

    # ------------------------------------------------------------------
    # PPO update
    # ------------------------------------------------------------------

    def _update(self, advantages: np.ndarray, returns: np.ndarray):
        self.net.train()

        obs_t = torch.tensor(np.array(self.buffer.obs), dtype=torch.float32)
        act_t = torch.tensor(self.buffer.actions, dtype=torch.long)
        old_lp_t = torch.tensor(self.buffer.log_probs, dtype=torch.float32)
        adv_t = torch.tensor(advantages, dtype=torch.float32)
        ret_t = torch.tensor(returns, dtype=torch.float32)

        # Normalize advantages
        if len(adv_t) > 1:
            adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        n = len(self.buffer)
        indices = np.arange(n)

        for epoch in range(self.n_epochs):
            np.random.shuffle(indices)
            for start in range(0, n, self.batch_size):
                end = start + self.batch_size
                mb = indices[start:end]

                new_lp, entropy, values = self.net.evaluate(obs_t[mb], act_t[mb])

                # Policy loss (clipped surrogate)
                ratio = torch.exp(new_lp - old_lp_t[mb])
                surr1 = ratio * adv_t[mb]
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * adv_t[mb]
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = nn.functional.mse_loss(values, ret_t[mb])

                # Entropy bonus
                entropy_loss = -entropy.mean()

                loss = policy_loss + self.vf_coef * value_loss + self.ent_coef * entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
                self.optimizer.step()

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------

    def save(self, path: str, episode_rewards: List[float] = None):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        obs_dim = int(np.prod(self.env.observation_space.shape))
        n_actions = self.env.action_space.n
        torch.save({
            "net_state_dict": self.net.state_dict(),
            "episode_rewards": episode_rewards or [],
            "obs_dim": obs_dim,
            "n_actions": n_actions,
        }, path)
        print(f"Saved PPO checkpoint to {path}")

    @classmethod
    def load(cls, path: str, env: "gym.Env") -> "PPO":
        data = torch.load(path, weights_only=False)
        agent = cls(env)
        agent.net.load_state_dict(data["net_state_dict"])
        return agent

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def learn(self, total_timesteps: int, print_every: int = 1000):
        steps_done = 0
        episode_rewards: List[float] = []
        current_ep_reward = 0.0

        # We track episode rewards via the rollout
        while steps_done < total_timesteps:
            advantages, returns = self._collect_rollouts()
            self._update(advantages, returns)

            # Track episode rewards from buffer
            for r, d in zip(self.buffer.rewards, self.buffer.dones):
                current_ep_reward += r
                if d:
                    episode_rewards.append(current_ep_reward)
                    current_ep_reward = 0.0

            steps_done += self.n_steps

            if episode_rewards and steps_done % print_every < self.n_steps:
                recent = episode_rewards[-50:] if episode_rewards else [0]
                print(
                    f"Steps: {steps_done}/{total_timesteps}, "
                    f"episodes: {len(episode_rewards)}, "
                    f"mean_reward(last 50): {np.mean(recent):.2f}"
                )

        return episode_rewards

    def predict(self, obs: np.ndarray, deterministic: bool = False) -> int:
        self.net.eval()
        obs_t = torch.from_numpy(obs.flatten().astype(np.float32)).unsqueeze(0)
        with torch.no_grad():
            dist = self.net.policy(obs_t)
            if deterministic:
                return dist.probs.argmax(dim=-1).item()
            return dist.sample().item()


# ---------------------------------------------------------------------------
# Environment factory
# ---------------------------------------------------------------------------

def make_highway_env():
    """Highway env with default config, only overriding action type."""
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
    TOTAL_TIMESTEPS = 200_000
    env = make_highway_env()

    agent = PPO(
        env,
        hidden=[256, 256],
        n_steps=128,
        batch_size=64,
        n_epochs=10,
        lr=5e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_eps=0.2,
        vf_coef=0.5,
        ent_coef=0.0,
        max_grad_norm=0.5,
        seed=42,
    )

    print("Starting from-scratch PPO training...")
    print(f"Obs shape: {env.observation_space.shape}")
    print(f"Action space: {env.action_space}")
    print()

    episode_rewards = agent.learn(total_timesteps=TOTAL_TIMESTEPS, print_every=10000)

    agent.save("checkpoints/ppo.pt", episode_rewards)

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
