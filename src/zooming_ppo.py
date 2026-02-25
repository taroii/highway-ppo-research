"""
Zooming PPO: Adaptive action discretization via zooming + PPO.

The zooming algorithm adaptively partitions the continuous action space [0,1]^2
into cubes. Each active cube = one discrete action for PPO. Cubes that are
played often enough get split into 4 children, giving finer resolution where
the agent acts most.

PPO learns a state-dependent policy over the current set of discrete actions.
The action set grows over training as cubes split.
"""

from __future__ import annotations

import math
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
import gymnasium as gym
from typing import List, Tuple
from pathlib import Path
from dataclasses import dataclass

try:
    import highway_env  # noqa: F401
except ImportError:
    pass

from zooming import Cube, CubeStats


# ---------------------------------------------------------------------------
# ActionZooming — adaptive action space manager
# ---------------------------------------------------------------------------

@dataclass
class SplitInfo:
    """Records one cube split: old index removed, new indices added."""
    old_idx: int
    new_indices: List[int]


class ActionZooming:
    """Manages a flat list of active cubes in [0,1]^2 as discrete actions."""

    def __init__(self, da: int = 2):
        self.da = da
        # Start with root cube and immediately split once → 2^da children (2x2 = 4)
        root = Cube(lower=np.zeros(da), s=1.0, d=da)
        self.active_cubes: List[Cube] = root.split_children()
        self.stats: List[CubeStats] = [
            CubeStats(Q=0.0) for _ in self.active_cubes
        ]

    @property
    def n_actions(self) -> int:
        return len(self.active_cubes)

    def get_env_action(self, idx: int) -> np.ndarray:
        """Center of cube idx, mapped from [0,1]^2 to [-1,1]^2."""
        cube = self.active_cubes[idx]
        center = cube.lower + 0.5 * cube.s  # in [0,1]^da
        return 2.0 * center - 1.0  # map to [-1,1]^da

    def update_play_counts(self, action_indices: List[int]):
        """Batch-increment n_play for each cube selected during a rollout."""
        for idx in action_indices:
            self.stats[idx].n_play += 1

    def split_threshold(self, cube: Cube) -> int:
        return math.ceil((1.0 / cube.s) ** 2)

    def try_split(self) -> List[SplitInfo]:
        """
        For each active cube, if n_play >= threshold, split into 2^da children.
        Returns split info for network rebuild.
        """
        splits: List[SplitInfo] = []
        # Collect indices to split (iterate in reverse so removal doesn't shift earlier indices)
        to_split = []
        for i, (cube, stat) in enumerate(zip(self.active_cubes, self.stats)):
            if stat.n_play >= self.split_threshold(cube):
                to_split.append(i)

        if not to_split:
            return splits

        # Process splits in reverse order to keep indices stable
        for old_idx in reversed(to_split):
            cube = self.active_cubes[old_idx]
            children = cube.split_children()

            # Remove parent
            self.active_cubes.pop(old_idx)
            self.stats.pop(old_idx)

            # Add children at the end
            new_start = len(self.active_cubes)
            new_indices = list(range(new_start, new_start + len(children)))
            for child in children:
                self.active_cubes.append(child)
                self.stats.append(CubeStats(Q=0.0))

            splits.append(SplitInfo(old_idx=old_idx, new_indices=new_indices))

        return splits


# ---------------------------------------------------------------------------
# ZoomingActorCritic — policy with swappable output head
# ---------------------------------------------------------------------------

class ZoomingActorCritic(nn.Module):
    """Actor-critic where the policy output layer can be rebuilt on splits."""

    def __init__(self, obs_dim: int, n_actions: int, hidden: List[int] = [256, 256]):
        super().__init__()
        self.hidden_dim = hidden[-1]

        # Policy hidden layers (persistent across splits)
        layers_pi: List[nn.Module] = []
        prev = obs_dim
        for h in hidden:
            layers_pi += [nn.Linear(prev, h), nn.Tanh()]
            prev = h
        self.pi_hidden = nn.Sequential(*layers_pi)

        # Policy output head (rebuilt on split)
        self.pi_out = nn.Linear(self.hidden_dim, n_actions)

        # Value network (unchanged)
        layers_vf: List[nn.Module] = []
        prev = obs_dim
        for h in hidden:
            layers_vf += [nn.Linear(prev, h), nn.Tanh()]
            prev = h
        layers_vf.append(nn.Linear(prev, 1))
        self.vf = nn.Sequential(*layers_vf)

    def rebuild_policy_head(self, new_n_actions: int, splits: List[SplitInfo], old_n_actions: int):
        """
        Create new pi_out layer. For cubes that didn't split: copy weight row.
        For new children: copy parent's weight row + small noise.
        """
        old_weight = self.pi_out.weight.data  # (old_n_actions, hidden_dim)
        old_bias = self.pi_out.bias.data      # (old_n_actions,)

        new_layer = nn.Linear(self.hidden_dim, new_n_actions)

        # Build mapping: for each new index, what old index does it come from?
        # Start with identity for indices that survive
        # First figure out which old indices were removed
        removed = set(s.old_idx for s in splits)

        # Build old --> new index mapping for survivors
        # After removals (processed in reverse), the remaining old indices shift.
        # Instead, rebuild from scratch using the split info.

        # Strategy: track what happened to each old row.
        # 1. Removed indices are gone (their weights go to children via splits)
        # 2. Surviving indices shift down

        # Compute the surviving old indices in order
        surviving_old = [i for i in range(old_n_actions) if i not in removed]

        with torch.no_grad():
            # Copy survivors: they now occupy indices 0..len(surviving_old)-1
            for new_idx, old_idx in enumerate(surviving_old):
                new_layer.weight.data[new_idx] = old_weight[old_idx]
                new_layer.bias.data[new_idx] = old_bias[old_idx]

            # Copy children from their parents
            for split in splits:
                parent_w = old_weight[split.old_idx]
                parent_b = old_bias[split.old_idx]
                for new_idx in split.new_indices:
                    noise_w = torch.randn_like(parent_w) * 0.01
                    noise_b = torch.randn_like(parent_b) * 0.01
                    new_layer.weight.data[new_idx] = parent_w + noise_w
                    new_layer.bias.data[new_idx] = parent_b + noise_b

        self.pi_out = new_layer

    def policy(self, obs: torch.Tensor) -> Categorical:
        h = self.pi_hidden(obs)
        logits = self.pi_out(h)
        return Categorical(logits=logits)

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
# ZoomingPPO — main algorithm
# ---------------------------------------------------------------------------

class ZoomingPPO:
    def __init__(
        self,
        env: gym.Env,
        hidden: List[int] = [256, 256],
        n_steps: int = 128,
        batch_size: int = 64,
        n_epochs: int = 10,
        lr: float = 5e-4,
        gamma: float = 0.8,
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
        self.lr = lr
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.max_grad_norm = max_grad_norm

        torch.manual_seed(seed)
        np.random.seed(seed)

        obs_dim = int(np.prod(env.observation_space.shape))

        # Zooming action space manager (2D: acceleration, steering)
        self.zooming = ActionZooming(da=2)
        self.net = ZoomingActorCritic(obs_dim, self.zooming.n_actions, hidden)
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
                action_idx, log_prob, value = self.net.act(obs_t)

            a_idx = action_idx.item()
            lp = log_prob.item()
            v = value.item()

            # Map discrete action index → continuous env action via zooming
            env_action = self.zooming.get_env_action(a_idx)

            next_obs, reward, done, truncated, info = self.env.step(env_action)
            self.buffer.store(obs_flat, a_idx, reward, done or truncated, lp, v)

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
        zooming_state = [
            {"lower": cube.lower.tolist(), "s": cube.s, "d": cube.d,
             "n_play": stat.n_play}
            for cube, stat in zip(self.zooming.active_cubes, self.zooming.stats)
        ]
        torch.save({
            "net_state_dict": self.net.state_dict(),
            "episode_rewards": episode_rewards or [],
            "obs_dim": obs_dim,
            "zooming_state": zooming_state,
        }, path)
        print(f"Saved ZoomingPPO checkpoint to {path}")

    @classmethod
    def load(cls, path: str, env: "gym.Env") -> "ZoomingPPO":
        data = torch.load(path, weights_only=False)
        agent = cls(env)
        # Reconstruct zooming state
        agent.zooming.active_cubes = []
        agent.zooming.stats = []
        for cs in data["zooming_state"]:
            cube = Cube(lower=np.array(cs["lower"]), s=cs["s"], d=cs["d"])
            agent.zooming.active_cubes.append(cube)
            agent.zooming.stats.append(CubeStats(Q=0.0, n_play=cs["n_play"]))
        # Rebuild network with correct action count
        obs_dim = data["obs_dim"]
        agent.net = ZoomingActorCritic(obs_dim, agent.zooming.n_actions)
        agent.net.load_state_dict(data["net_state_dict"])
        return agent

    # ------------------------------------------------------------------
    # Check and split
    # ------------------------------------------------------------------

    def _check_and_split(self):
        """After PPO update, check if any cubes should split. Rebuild network if so."""
        # Update play counts from the rollout
        self.zooming.update_play_counts(self.buffer.actions)

        old_n_actions = self.zooming.n_actions
        splits = self.zooming.try_split()

        if splits:
            new_n_actions = self.zooming.n_actions
            self._total_splits += len(splits)
            self.net.rebuild_policy_head(new_n_actions, splits, old_n_actions)
            # Rebuild optimizer with new parameters
            self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def _cube_size_summary(self) -> str:
        """Return a string summarising the distribution of cube side lengths."""
        sizes = [c.s for c in self.zooming.active_cubes]
        from collections import Counter
        counts = Counter(sizes)
        parts = [f"s={s:.3f}:{n}" for s, n in sorted(counts.items(), reverse=True)]
        return " ".join(parts)

    def learn(self, total_timesteps: int, print_every: int = 10_000):
        steps_done = 0
        episode_rewards: List[float] = []
        current_ep_reward = 0.0
        self._total_splits = 0
        last_print = 0

        while steps_done < total_timesteps:
            advantages, returns = self._collect_rollouts()
            self._update(advantages, returns)
            self._check_and_split()

            # Track episode rewards from buffer
            for r, d in zip(self.buffer.rewards, self.buffer.dones):
                current_ep_reward += r
                if d:
                    episode_rewards.append(current_ep_reward)
                    current_ep_reward = 0.0

            steps_done += self.n_steps

            if steps_done // print_every > last_print // print_every:
                last_print = steps_done
                recent = episode_rewards[-50:] if episode_rewards else [0]
                print(
                    f"[{steps_done:>7d}/{total_timesteps}] "
                    f"ep={len(episode_rewards):>4d}  "
                    f"reward(last50)={np.mean(recent):>7.2f}  "
                    f"actions={self.zooming.n_actions:>3d}  "
                    f"total_splits={self._total_splits}  "
                    f"cubes=[{self._cube_size_summary()}]"
                )

        return episode_rewards

    def predict(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """Return continuous env action for evaluation."""
        self.net.eval()
        obs_t = torch.from_numpy(obs.flatten().astype(np.float32)).unsqueeze(0)
        with torch.no_grad():
            dist = self.net.policy(obs_t)
            if deterministic:
                action_idx = dist.probs.argmax(dim=-1).item()
            else:
                action_idx = dist.sample().item()
        return self.zooming.get_env_action(action_idx)


# ---------------------------------------------------------------------------
# Environment factory
# ---------------------------------------------------------------------------

def make_highway_env_continuous():
    """Highway env with default config, only overriding action type."""
    env = gym.make(
        "highway-v0",
        config={
            "action": {
                "type": "ContinuousAction",
                "longitudinal": True,
                "lateral": True,
            },
        },
    )
    return env


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    TOTAL_TIMESTEPS = 100_000
    env = make_highway_env_continuous()

    agent = ZoomingPPO(
        env,
        hidden=[256, 256],
        n_steps=128,
        batch_size=64,
        n_epochs=10,
        lr=5e-4,
        gamma=0.8,
        gae_lambda=0.95,
        clip_eps=0.2,
        vf_coef=0.5,
        ent_coef=0.0,
        max_grad_norm=0.5,
        seed=42,
    )

    print("Starting Zooming PPO training...")
    print(f"Obs shape: {env.observation_space.shape}")
    print(f"Action space: {env.action_space}")
    print(f"Initial discrete actions: {agent.zooming.n_actions}")
    print()

    episode_rewards = agent.learn(total_timesteps=TOTAL_TIMESTEPS, print_every=10000)

    agent.save("checkpoints/zooming_ppo.pt", episode_rewards)

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
    print(f"Final action count: {agent.zooming.n_actions}")

    env.close()
