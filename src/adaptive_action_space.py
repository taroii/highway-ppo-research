"""
Adaptive Action Space with Action Masking for PPO

This module implements an adaptive discretization strategy where:
- The action space is fixed at high dimensionality (e.g., 10x10 = 100 actions)
- Initially, only a sparse subset is available (e.g., 25 actions in a 5x5 grid)
- During training, new actions are gradually unmasked near high-advantage actions
- The neural network architecture remains fixed throughout training
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import gymnasium as gym


class AdaptiveActionSpace:
    """
    Manages adaptive action masking for continuous steering/acceleration discretization.

    The action space is a 2D grid over (steering, acceleration).
    Actions are gradually unmasked based on their estimated advantages.
    """

    def __init__(
        self,
        steering_range: Tuple[float, float] = (-np.pi/4, np.pi/4),
        acceleration_range: Tuple[float, float] = (-5.0, 5.0),
        final_grid_size: Tuple[int, int] = (10, 10),
        initial_grid_size: Tuple[int, int] = (5, 5),
        unmask_rate: float = 0.1,  # Fraction of masked actions to unmask per update
        advantage_threshold: float = 0.0,  # Only unmask near actions with advantage > this
    ):
        """
        Args:
            steering_range: (min, max) steering angle in radians
            acceleration_range: (min, max) acceleration in m/s^2
            final_grid_size: (n_steering, n_accel) for full resolution
            initial_grid_size: (n_steering, n_accel) for initial sparse grid
            unmask_rate: What fraction of currently masked actions to unmask per update
            advantage_threshold: Minimum advantage for an action to trigger unmasking neighbors
        """
        self.steering_range = steering_range
        self.acceleration_range = acceleration_range
        self.final_grid_size = final_grid_size
        self.initial_grid_size = initial_grid_size
        self.unmask_rate = unmask_rate
        self.advantage_threshold = advantage_threshold

        # Create the full action space
        self.n_actions = final_grid_size[0] * final_grid_size[1]
        self.steering_values = np.linspace(steering_range[0], steering_range[1], final_grid_size[0])
        self.accel_values = np.linspace(acceleration_range[0], acceleration_range[1], final_grid_size[1])

        # Create action grid: maps action index -> (steering_idx, accel_idx)
        self.action_to_grid = {}
        self.grid_to_action = {}
        idx = 0
        for i in range(final_grid_size[0]):
            for j in range(final_grid_size[1]):
                self.action_to_grid[idx] = (i, j)
                self.grid_to_action[(i, j)] = idx
                idx += 1

        # Initialize mask: True = masked (invalid), False = unmasked (valid)
        self.action_mask = np.ones(self.n_actions, dtype=bool)

        # Initialize with sparse grid
        self._initialize_sparse_grid()

        # Track action statistics
        self.action_advantages = np.zeros(self.n_actions)
        self.action_counts = np.zeros(self.n_actions)

    def _initialize_sparse_grid(self):
        """Initialize with a uniform sparse grid matching initial_grid_size."""
        # Map initial grid indices to final grid indices
        stride_steering = (self.final_grid_size[0] - 1) // (self.initial_grid_size[0] - 1)
        stride_accel = (self.final_grid_size[1] - 1) // (self.initial_grid_size[1] - 1)

        for i in range(self.initial_grid_size[0]):
            for j in range(self.initial_grid_size[1]):
                grid_i = i * stride_steering
                grid_j = j * stride_accel
                action_idx = self.grid_to_action[(grid_i, grid_j)]
                self.action_mask[action_idx] = False  # Unmask

    def get_valid_actions(self) -> np.ndarray:
        """Returns array of valid (unmasked) action indices."""
        return np.where(~self.action_mask)[0]

    def get_action_mask(self) -> np.ndarray:
        """Returns boolean mask where True = invalid/masked."""
        return self.action_mask.copy()

    def get_action_values(self, action_idx: int) -> Tuple[float, float]:
        """Convert action index to (steering, acceleration) values."""
        # Handle numpy arrays
        if isinstance(action_idx, np.ndarray):
            action_idx = int(action_idx.item())
        else:
            action_idx = int(action_idx)

        grid_i, grid_j = self.action_to_grid[action_idx]
        return self.steering_values[grid_i], self.accel_values[grid_j]

    def update_advantages(self, action_indices: np.ndarray, advantages: np.ndarray):
        """
        Update running statistics for action advantages.

        Args:
            action_indices: Array of action indices taken
            advantages: Corresponding advantage estimates
        """
        action_indices = action_indices.astype(int).flatten()
        advantages = advantages.flatten()

        for action_idx, advantage in zip(action_indices, advantages):
            self.action_advantages[action_idx] += advantage
            self.action_counts[action_idx] += 1

    def get_average_advantages(self) -> np.ndarray:
        """Get average advantage for each action (0 if never taken)."""
        avg_advantages = np.zeros(self.n_actions)
        valid = self.action_counts > 0
        avg_advantages[valid] = self.action_advantages[valid] / self.action_counts[valid]
        return avg_advantages

    def _get_neighbors(self, grid_pos: Tuple[int, int]) -> List[int]:
        """Get all 8-neighbor action indices for a grid position."""
        neighbors = []
        i, j = grid_pos
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                if di == 0 and dj == 0:
                    continue
                ni, nj = i + di, j + dj
                if 0 <= ni < self.final_grid_size[0] and 0 <= nj < self.final_grid_size[1]:
                    if (ni, nj) in self.grid_to_action:
                        neighbors.append(self.grid_to_action[(ni, nj)])
        return neighbors

    def unmask_actions(self, n_updates: int = 1):
        """
        Gradually unmask new actions near high-advantage actions.

        Args:
            n_updates: Number of unmasking updates to perform
        """
        avg_advantages = self.get_average_advantages()

        # Find currently unmasked actions
        valid_actions = self.get_valid_actions()

        # Find which valid actions have high advantages
        high_advantage_actions = []
        for action_idx in valid_actions:
            if avg_advantages[action_idx] > self.advantage_threshold:
                high_advantage_actions.append(action_idx)

        if len(high_advantage_actions) == 0:
            # If no high-advantage actions, use the best available action
            if len(valid_actions) > 0:
                best_action = valid_actions[np.argmax(avg_advantages[valid_actions])]
                high_advantage_actions = [best_action]
            else:
                return  # No valid actions at all, shouldn't happen

        # Collect all masked neighbors of high-advantage actions
        candidate_unmask = set()
        for action_idx in high_advantage_actions:
            grid_pos = self.action_to_grid[action_idx]
            neighbors = self._get_neighbors(grid_pos)
            for neighbor_idx in neighbors:
                if self.action_mask[neighbor_idx]:  # Currently masked
                    candidate_unmask.add(neighbor_idx)

        if len(candidate_unmask) == 0:
            return  # All neighbors already unmasked

        # Unmask a fraction of the candidates
        candidate_list = list(candidate_unmask)
        n_to_unmask = max(1, int(len(candidate_list) * self.unmask_rate))

        # Prioritize candidates that are neighbors of multiple high-advantage actions
        # (simple heuristic: unmask randomly for now, can be refined)
        to_unmask = np.random.choice(candidate_list, size=min(n_to_unmask, len(candidate_list)), replace=False)

        for action_idx in to_unmask:
            self.action_mask[action_idx] = False

    def get_num_valid_actions(self) -> int:
        """Returns the current number of valid (unmasked) actions."""
        return np.sum(~self.action_mask)

    def get_progress(self) -> float:
        """Returns the fraction of actions that have been unmasked (0 to 1)."""
        return self.get_num_valid_actions() / self.n_actions

    def reset_statistics(self):
        """Reset advantage and count statistics (use if needed for episodic tracking)."""
        self.action_advantages = np.zeros(self.n_actions)
        self.action_counts = np.zeros(self.n_actions)


class AdaptiveDiscreteActionWrapper(gym.Wrapper):
    """
    Gym wrapper that converts the adaptive discrete action space to continuous actions
    for highway-env, while handling action masking.
    """

    def __init__(
        self,
        env: gym.Env,
        adaptive_action_space: AdaptiveActionSpace,
    ):
        """
        Args:
            env: Highway-env environment configured with ContinuousAction
            adaptive_action_space: AdaptiveActionSpace instance managing the masking
        """
        super().__init__(env)
        self.adaptive_space = adaptive_action_space

        # Override the action space to be discrete with the full action count
        self.action_space = gym.spaces.Discrete(self.adaptive_space.n_actions)

    def step(self, action: int):
        """Convert discrete action to continuous and step environment."""
        # Convert discrete action to continuous values
        steering, acceleration = self.adaptive_space.get_action_values(action)

        # Map to [-1, 1] range expected by ContinuousAction
        # ContinuousAction will map back to the actual ranges
        steering_norm = self._map_to_normalized(
            steering,
            self.adaptive_space.steering_range
        )
        accel_norm = self._map_to_normalized(
            acceleration,
            self.adaptive_space.acceleration_range
        )

        continuous_action = np.array([accel_norm, steering_norm])
        return self.env.step(continuous_action)

    def _map_to_normalized(self, value: float, range_: Tuple[float, float]) -> float:
        """Map a value from a range to [-1, 1]."""
        return 2 * (value - range_[0]) / (range_[1] - range_[0]) - 1

    def get_action_mask(self) -> np.ndarray:
        """Get current action mask for the policy."""
        return self.adaptive_space.get_action_mask()


if __name__ == "__main__":
    # Example usage
    adaptive_space = AdaptiveActionSpace(
        steering_range=(-np.pi/4, np.pi/4),
        acceleration_range=(-5.0, 5.0),
        final_grid_size=(10, 10),
        initial_grid_size=(5, 5),
        unmask_rate=0.1,
        advantage_threshold=0.0,
    )

    print(f"Total actions: {adaptive_space.n_actions}")
    print(f"Initial valid actions: {adaptive_space.get_num_valid_actions()}")
    print(f"Valid action indices: {adaptive_space.get_valid_actions()}")

    # Simulate some advantages
    valid_actions = adaptive_space.get_valid_actions()
    action_indices = np.random.choice(valid_actions, size=100)
    advantages = np.random.randn(100)

    adaptive_space.update_advantages(action_indices, advantages)
    print(f"\nAverage advantages computed")

    # Unmask some actions
    print(f"\nBefore unmasking: {adaptive_space.get_num_valid_actions()} valid actions")
    adaptive_space.unmask_actions()
    print(f"After unmasking: {adaptive_space.get_num_valid_actions()} valid actions")
    print(f"Progress: {adaptive_space.get_progress():.2%}")
