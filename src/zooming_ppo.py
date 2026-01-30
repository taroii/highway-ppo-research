"""
Zooming Adaptive Discretization with Policy Gradients for HighwayEnv

Uses advantage estimates in place of Q-values to update cube preferences.
Policy is softmax over preferences for relevant cubes.

Key modification from standard zooming: instead of splitting cubes into 2^d
children (which is intractable for d=26 dimensions), we use binary splits
along the action dimension only. This keeps the state-space partitioning
coarse while refining action resolution where advantage variance is high.
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional
import highway_env  # noqa: F401


def clip01(x: np.ndarray) -> np.ndarray:
    return np.minimum(1.0, np.maximum(0.0, x))


@dataclass
class Cube:
    lower: np.ndarray
    s: float
    d: int

    def contains(self, z: np.ndarray, eps: float = 1e-12) -> bool:
        upper = self.lower + self.s
        return bool(np.all(z >= self.lower - eps) and np.all(z <= upper + eps))

    def contains_state(self, x: np.ndarray, ds: int, eps: float = 1e-12) -> bool:
        lower_s = self.lower[:ds]
        upper_s = lower_s + self.s
        return bool(np.all(x >= lower_s - eps) and np.all(x <= upper_s + eps))

    def split_children(self, split_dim: int) -> List[Cube]:
        """Binary split along a single dimension."""
        half = self.s / 2.0
        children: List[Cube] = []
        for i in range(2):
            child_lower = self.lower.copy()
            child_lower[split_dim] += i * half
            children.append(Cube(lower=child_lower, s=half, d=self.d))
        return children

    def key(self) -> Tuple[float, ...]:
        return (*map(float, self.lower.tolist()), float(self.s))


@dataclass
class CubeStats:
    preference: float = 0.0  # Updated by advantages, used for policy
    n_visits: int = 0
    advantage_sum: float = 0.0
    advantage_sq_sum: float = 0.0
    is_split: bool = False
    split_dim: int = 0  # Which dimension was split
    children: List[Cube] = field(default_factory=list)

    def advantage_variance(self) -> float:
        if self.n_visits < 2:
            return float('inf')
        mean = self.advantage_sum / self.n_visits
        return (self.advantage_sq_sum / self.n_visits) - mean ** 2


class ZoomingPPO:
    """
    Adaptive discretization with policy gradient updates.

    - Maintains cubes in joint (state, action) space
    - Policy is softmax over cube preferences
    - Preferences updated using advantage estimates
    - Cubes split based on advantage variance
    """

    def __init__(
        self,
        ds: int,
        da: int,
        action_bounds: Tuple[np.ndarray, np.ndarray],
        gamma: float = 0.99,
        lr: float = 0.1,
        temperature: float = 1.0,
        split_threshold: float = 0.5,
        min_visits_to_split: int = 10,
        seed: int = 0,
    ):
        self.ds = ds
        self.da = da
        self.d = ds + da
        self.gamma = gamma
        self.lr = lr
        self.temperature = temperature
        self.split_threshold = split_threshold
        self.min_visits_to_split = min_visits_to_split
        self.rng = np.random.default_rng(seed)

        # Action bounds for denormalization
        self.act_low, self.act_high = action_bounds

        # Active cubes and buffer
        self.active: Dict[Tuple[float, ...], Tuple[Cube, CubeStats]] = {}
        self.buffer: Dict[Tuple[float, ...], Tuple[Cube, CubeStats]] = {}

        # Initialize with root cube
        root = Cube(lower=np.zeros(self.d), s=1.0, d=self.d)
        self.active[root.key()] = (root, CubeStats())

        # Value function estimate (simple: average return from state)
        self.value_estimates: Dict[Tuple[int, ...], Tuple[float, int]] = {}

    def normalize_obs(self, obs: np.ndarray) -> np.ndarray:
        # Highway-env with normalize=True gives values roughly in [-1, 1]
        # Map to [0, 1] for cube containment
        return clip01((obs + 1.0) / 2.0)

    def denormalize_action(self, action_norm: np.ndarray) -> np.ndarray:
        # Map from [0, 1] to action bounds
        return self.act_low + action_norm * (self.act_high - self.act_low)

    def relevant_cubes(self, x_norm: np.ndarray) -> List[Tuple[Cube, CubeStats]]:
        """Get active cubes whose state-projection contains x, with smallest side length."""
        candidates = []
        for cube, stats in self.active.values():
            if cube.contains_state(x_norm, self.ds):
                candidates.append((cube, stats))

        if not candidates:
            raise RuntimeError("No active cube contains state (should not happen)")

        min_s = min(c.s for c, _ in candidates)
        return [(c, s) for c, s in candidates if abs(c.s - min_s) < 1e-12]

    def policy_probs(self, relevant: List[Tuple[Cube, CubeStats]]) -> np.ndarray:
        """Softmax over preferences."""
        prefs = np.array([stats.preference for _, stats in relevant])
        prefs = prefs - np.max(prefs)  # stability
        exp_prefs = np.exp(prefs / self.temperature)
        return exp_prefs / (exp_prefs.sum() + 1e-12)

    def select_cube(self, x_norm: np.ndarray) -> Tuple[Cube, CubeStats, float]:
        """Sample a cube from policy, return (cube, stats, log_prob)."""
        relevant = self.relevant_cubes(x_norm)
        probs = self.policy_probs(relevant)
        idx = self.rng.choice(len(relevant), p=probs)
        cube, stats = relevant[idx]
        log_prob = np.log(probs[idx] + 1e-12)
        return cube, stats, log_prob

    def action_from_cube(self, cube: Cube) -> np.ndarray:
        """Get action from cube center (in original action space)."""
        a_norm = cube.lower[self.ds:] + 0.5 * cube.s
        return self.denormalize_action(clip01(a_norm))

    def discretize_state(self, x_norm: np.ndarray, resolution: int = 10) -> Tuple[int, ...]:
        """Discretize state for value function lookup."""
        indices = (x_norm * resolution).astype(int)
        indices = np.clip(indices, 0, resolution - 1)
        return tuple(indices.tolist())

    def get_value(self, x_norm: np.ndarray) -> float:
        """Get value estimate for state."""
        key = self.discretize_state(x_norm)
        if key in self.value_estimates:
            total, count = self.value_estimates[key]
            return total / count
        return 0.0

    def update_value(self, x_norm: np.ndarray, ret: float):
        """Update value estimate for state."""
        key = self.discretize_state(x_norm)
        if key in self.value_estimates:
            total, count = self.value_estimates[key]
            self.value_estimates[key] = (total + ret, count + 1)
        else:
            self.value_estimates[key] = (ret, 1)

    def compute_advantages(
        self,
        rewards: List[float],
        x_norms: List[np.ndarray],
        dones: List[bool],
    ) -> List[float]:
        """Compute TD(0) advantages."""
        advantages = []
        for i, (r, x, done) in enumerate(zip(rewards, x_norms, dones)):
            v = self.get_value(x)
            if done or i == len(rewards) - 1:
                v_next = 0.0
            else:
                v_next = self.get_value(x_norms[i + 1])
            td_target = r + self.gamma * v_next * (1 - done)
            advantages.append(td_target - v)
        return advantages

    def update_from_trajectory(
        self,
        x_norms: List[np.ndarray],
        cubes: List[Cube],
        log_probs: List[float],
        rewards: List[float],
        dones: List[bool],
    ):
        """Update cube preferences using policy gradient with advantages."""
        # Compute returns for value function updates
        returns = []
        G = 0.0
        for r, done in zip(reversed(rewards), reversed(dones)):
            if done:
                G = 0.0
            G = r + self.gamma * G
            returns.insert(0, G)

        # Update value estimates
        for x, ret in zip(x_norms, returns):
            self.update_value(x, ret)

        # Compute advantages
        advantages = self.compute_advantages(rewards, x_norms, dones)

        # Normalize advantages
        adv_arr = np.array(advantages)
        if len(adv_arr) > 1 and adv_arr.std() > 1e-8:
            adv_arr = (adv_arr - adv_arr.mean()) / (adv_arr.std() + 1e-8)

        # Update cube preferences using policy gradient
        for cube, log_prob, adv in zip(cubes, log_probs, adv_arr):
            key = cube.key()
            if key in self.active:
                _, stats = self.active[key]
            elif key in self.buffer:
                _, stats = self.buffer[key]
            else:
                continue

            # Policy gradient update: preference += lr * advantage
            # (simplified: ignoring log_prob gradient for tabular case)
            stats.preference += self.lr * adv
            stats.n_visits += 1
            stats.advantage_sum += adv
            stats.advantage_sq_sum += adv ** 2

    def maybe_split_cubes(self):
        """Split cubes with high advantage variance (binary split along action dim)."""
        to_split = []
        for key, (cube, stats) in self.active.items():
            if stats.is_split:
                continue
            if stats.n_visits < self.min_visits_to_split:
                continue
            var = stats.advantage_variance()
            if var > self.split_threshold:
                to_split.append((cube, stats))

        for cube, stats in to_split:
            stats.is_split = True
            # Split along action dimension (last da dimensions)
            # Cycle through action dims if multiple
            split_dim = self.ds + (stats.split_dim % self.da)
            stats.children = cube.split_children(split_dim)
            for child in stats.children:
                child_key = child.key()
                if child_key not in self.active and child_key not in self.buffer:
                    # Initialize child with parent's preference, increment split_dim for next split
                    self.buffer[child_key] = (child, CubeStats(
                        preference=stats.preference,
                        split_dim=(stats.split_dim + 1) % self.da
                    ))

    def promote_buffer_cubes(self, min_visits: int = 5):
        """Move buffer cubes to active if they have enough visits."""
        to_promote = []
        for key, (cube, stats) in self.buffer.items():
            if stats.n_visits >= min_visits:
                to_promote.append(key)

        for key in to_promote:
            self.active[key] = self.buffer.pop(key)

    def run_episode(self, env: gym.Env) -> Tuple[float, int]:
        """Run one episode, return (total_reward, steps)."""
        obs, _ = env.reset()

        # Flatten observation if needed
        obs_flat = obs.flatten()

        x_norms = []
        cubes = []
        log_probs = []
        rewards = []
        dones = []

        total_reward = 0.0
        steps = 0
        done = False
        truncated = False

        while not (done or truncated):
            x_norm = self.normalize_obs(obs_flat)
            cube, stats, log_prob = self.select_cube(x_norm)
            action = self.action_from_cube(cube)

            next_obs, reward, done, truncated, _ = env.step(action)
            next_obs_flat = next_obs.flatten()

            x_norms.append(x_norm)
            cubes.append(cube)
            log_probs.append(log_prob)
            rewards.append(reward)
            dones.append(done or truncated)

            total_reward += reward
            steps += 1
            obs_flat = next_obs_flat

        # Update from trajectory
        self.update_from_trajectory(x_norms, cubes, log_probs, rewards, dones)

        # Maybe split and promote
        self.maybe_split_cubes()
        self.promote_buffer_cubes()

        return total_reward, steps

    def train(self, env: gym.Env, n_episodes: int = 1000, print_every: int = 100):
        """Train for n_episodes."""
        episode_rewards = []

        for ep in range(n_episodes):
            reward, steps = self.run_episode(env)
            episode_rewards.append(reward)

            if (ep + 1) % print_every == 0:
                recent = episode_rewards[-print_every:]
                mean_r = np.mean(recent)
                n_active = len(self.active)
                n_buffer = len(self.buffer)
                print(f"Episode {ep+1}: mean_reward={mean_r:.2f}, "
                      f"active_cubes={n_active}, buffer_cubes={n_buffer}")

        return episode_rewards


def make_highway_env():
    """Create highway env with continuous steering action."""
    env = gym.make(
        "highway-fast-v0",
        config={
            "action": {
                "type": "ContinuousAction",
                "steering_range": [-np.pi/4, np.pi/4],
                "longitudinal": False,
                "lateral": True,
            },
            "observation": {
                "type": "Kinematics",
                "vehicles_count": 5,
                "features": ["presence", "x", "y", "vx", "vy"],
                "features_range": {
                    "x": [-100, 100],
                    "y": [-100, 100],
                    "vx": [-20, 20],
                    "vy": [-20, 20],
                },
                "normalize": True,
                "absolute": False,
            },
            "duration": 40,
            "policy_frequency": 2,
        }
    )
    return env


if __name__ == "__main__":
    # Create environment
    env = make_highway_env()

    # Action bounds: steering angle
    act_low = np.array([-np.pi/4], dtype=np.float32)
    act_high = np.array([np.pi/4], dtype=np.float32)

    # State dimension = flattened obs, action dimension = 1
    ds = 25  # 5 vehicles * 5 features
    da = 1   # steering only

    # Create agent
    agent = ZoomingPPO(
        ds=ds,
        da=da,
        action_bounds=(act_low, act_high),
        gamma=0.99,
        lr=0.1,
        temperature=1.0,
        split_threshold=0.5,
        min_visits_to_split=10,
        seed=42,
    )

    # Train
    print("Starting training...")
    print(f"State dim: {ds}, Action dim: {da}")
    print(f"Initial cubes: {len(agent.active)}")
    print()

    rewards = agent.train(env, n_episodes=500, print_every=50)

    print()
    print(f"Final active cubes: {len(agent.active)}")
    print(f"Final buffer cubes: {len(agent.buffer)}")
    print(f"Mean reward (last 50): {np.mean(rewards[-50:]):.2f}")

    env.close()
