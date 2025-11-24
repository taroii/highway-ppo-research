"""
Adaptive Action Space with Pruning for Steering-Only Control

This module implements an adaptive discretization strategy for steering control where:
- The action space is 1D (steering only, no acceleration control)
- Actions are gradually unmasked near high-advantage actions
- Low-performing actions are pruned to maintain constant L1 norm
- The neural network architecture remains fixed throughout training
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import gymnasium as gym


class AdaptiveSteeringActionSpace:
    """
    Manages adaptive action masking for steering control with pruning.

    The action space is a 1D grid over steering angles.
    Actions are gradually unmasked based on advantages, and low-performing actions are pruned.
    """

    def __init__(
        self,
        steering_range: Tuple[float, float] = (-np.pi/4, np.pi/4),
        final_grid_size: int = 20,
        initial_active_actions: int = 5,
        unmask_rate: float = 0.2,  # Fraction of masked actions to unmask per update
        prune_rate: float = 0.1,  # Fraction of active actions to prune per update
        advantage_threshold: float = 0.0,  # Minimum advantage to trigger unmasking
        min_samples_for_pruning: int = 50,  # Minimum samples before considering pruning
    ):
        """
        Args:
            steering_range: (min, max) steering angle in radians
            final_grid_size: Total number of discrete steering angles
            initial_active_actions: Number of actions initially active
            unmask_rate: Fraction of masked actions to unmask per update
            prune_rate: Fraction of active actions to consider for pruning per update
            advantage_threshold: Minimum advantage for an action to trigger unmasking neighbors
            min_samples_for_pruning: Minimum times an action must be sampled before pruning
        """
        self.steering_range = steering_range
        self.final_grid_size = final_grid_size
        self.initial_active_actions = initial_active_actions
        self.unmask_rate = unmask_rate
        self.prune_rate = prune_rate
        self.advantage_threshold = advantage_threshold
        self.min_samples_for_pruning = min_samples_for_pruning

        # Create the full action space
        self.n_actions = final_grid_size
        self.steering_values = np.linspace(steering_range[0], steering_range[1], final_grid_size)

        # Initialize mask: True = masked (invalid), False = unmasked (valid)
        self.action_mask = np.ones(self.n_actions, dtype=bool)

        # Initialize with sparse uniform grid
        self._initialize_sparse_grid()

        # Track action statistics
        self.action_advantages = np.zeros(self.n_actions)
        self.action_counts = np.zeros(self.n_actions)

    def _initialize_sparse_grid(self):
        """Initialize with a uniform sparse grid matching initial_active_actions."""
        if self.initial_active_actions >= self.final_grid_size:
            # If we want all actions initially, unmask everything
            self.action_mask[:] = False
        else:
            # Distribute initial actions uniformly
            indices = np.linspace(0, self.final_grid_size - 1, self.initial_active_actions, dtype=int)
            self.action_mask[indices] = False

    def get_valid_actions(self) -> np.ndarray:
        """Returns array of valid (unmasked) action indices."""
        return np.where(~self.action_mask)[0]

    def get_action_mask(self) -> np.ndarray:
        """Returns boolean mask where True = invalid/masked."""
        return self.action_mask.copy()

    def get_steering_value(self, action_idx: int) -> float:
        """Convert action index to steering angle value."""
        # Handle numpy arrays
        if isinstance(action_idx, np.ndarray):
            action_idx = int(action_idx.item())
        else:
            action_idx = int(action_idx)

        return self.steering_values[action_idx]

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

    def _get_neighbors(self, action_idx: int) -> List[int]:
        """Get neighboring action indices (left and right)."""
        neighbors = []
        # Left neighbor
        if action_idx > 0:
            neighbors.append(action_idx - 1)
        # Right neighbor
        if action_idx < self.n_actions - 1:
            neighbors.append(action_idx + 1)
        return neighbors

    def unmask_and_prune(self):
        """
        Gradually unmask new actions near high-advantage actions,
        and prune low-performing actions to maintain constant active action count.
        """
        target_active = self.initial_active_actions
        current_active = self.get_num_valid_actions()

        avg_advantages = self.get_average_advantages()
        valid_actions = self.get_valid_actions()

        # Step 1: Identify candidates for pruning (low advantage, sufficient samples)
        prune_candidates = []
        for action_idx in valid_actions:
            if self.action_counts[action_idx] >= self.min_samples_for_pruning:
                prune_candidates.append((action_idx, avg_advantages[action_idx]))

        # Step 2: Identify candidates for unmasking (neighbors of high-advantage actions)
        unmask_candidates = set()

        # Find high-advantage actions
        high_advantage_actions = []
        for action_idx in valid_actions:
            if avg_advantages[action_idx] > self.advantage_threshold:
                high_advantage_actions.append(action_idx)

        if len(high_advantage_actions) == 0 and len(valid_actions) > 0:
            # If no high-advantage actions, use the best available
            best_action = valid_actions[np.argmax(avg_advantages[valid_actions])]
            high_advantage_actions = [best_action]

        # Collect masked neighbors of high-advantage actions
        for action_idx in high_advantage_actions:
            neighbors = self._get_neighbors(action_idx)
            for neighbor_idx in neighbors:
                if self.action_mask[neighbor_idx]:  # Currently masked
                    unmask_candidates.add(neighbor_idx)

        # Step 3: Determine how many to unmask and prune
        n_to_unmask = int(len(unmask_candidates) * self.unmask_rate) if len(unmask_candidates) > 0 else 0
        n_to_unmask = max(1, min(n_to_unmask, len(unmask_candidates)))

        # We want to maintain approximately target_active actions
        # If we unmask n_to_unmask, we should prune the same amount (or prune first then unmask)
        n_to_prune = 0
        if len(prune_candidates) > 0:
            # Prune roughly the same number we're unmasking to maintain constant L1 norm
            n_to_prune = min(n_to_unmask, int(len(prune_candidates) * self.prune_rate))
            n_to_prune = max(0, n_to_prune)
            # Don't prune so much that we go below target
            n_to_prune = min(n_to_prune, current_active - target_active + n_to_unmask)

        # Step 4: Prune low-performing actions first
        if n_to_prune > 0 and len(prune_candidates) > 0:
            # Sort by advantage (lowest first)
            prune_candidates.sort(key=lambda x: x[1])
            to_prune = [action_idx for action_idx, _ in prune_candidates[:n_to_prune]]

            for action_idx in to_prune:
                self.action_mask[action_idx] = True  # Mask (prune) the action

        # Step 5: Unmask new actions
        if n_to_unmask > 0 and len(unmask_candidates) > 0:
            unmask_list = list(unmask_candidates)
            to_unmask = np.random.choice(unmask_list, size=min(n_to_unmask, len(unmask_list)), replace=False)

            for action_idx in to_unmask:
                self.action_mask[action_idx] = False  # Unmask

    def get_num_valid_actions(self) -> int:
        """Returns the current number of valid (unmasked) actions."""
        return np.sum(~self.action_mask)

    def get_progress(self) -> float:
        """Returns the fraction of actions that have been explored (0 to 1)."""
        # Progress isn't just about unmasked count, but about exploration
        # We define progress as the fraction of actions that have been sampled at least once
        return np.sum(self.action_counts > 0) / self.n_actions

    def reset_statistics(self):
        """Reset advantage and count statistics (use if needed for episodic tracking)."""
        self.action_advantages = np.zeros(self.n_actions)
        self.action_counts = np.zeros(self.n_actions)


class AdaptiveSteeringActionWrapper(gym.Wrapper):
    """
    Gym wrapper that converts the adaptive discrete steering actions to continuous steering
    for highway-env, while handling action masking.
    """

    def __init__(
        self,
        env: gym.Env,
        adaptive_action_space: AdaptiveSteeringActionSpace,
    ):
        """
        Args:
            env: Highway-env environment configured with ContinuousAction (steering only)
            adaptive_action_space: AdaptiveSteeringActionSpace instance managing the masking
        """
        super().__init__(env)
        self.adaptive_space = adaptive_action_space

        # Override the action space to be discrete with the full action count
        self.action_space = gym.spaces.Discrete(self.adaptive_space.n_actions)

    def step(self, action: int):
        """Convert discrete action to continuous steering and step environment."""
        # Convert discrete action to steering angle
        steering = self.adaptive_space.get_steering_value(action)

        # Map to [-1, 1] range expected by ContinuousAction
        steering_norm = self._map_to_normalized(steering, self.adaptive_space.steering_range)

        # For steering-only control, we use 0 acceleration (let the car maintain speed)
        continuous_action = np.array([steering_norm])
        return self.env.step(continuous_action)

    def _map_to_normalized(self, value: float, range_: Tuple[float, float]) -> float:
        """Map a value from a range to [-1, 1]."""
        return 2 * (value - range_[0]) / (range_[1] - range_[0]) - 1

    def get_action_mask(self) -> np.ndarray:
        """Get current action mask for the policy."""
        return self.adaptive_space.get_action_mask()


if __name__ == "__main__":
    # Example usage
    adaptive_space = AdaptiveSteeringActionSpace(
        steering_range=(-np.pi/4, np.pi/4),
        final_grid_size=20,
        initial_active_actions=5,
        unmask_rate=0.2,
        prune_rate=0.1,
    )

    print(f"Total actions: {adaptive_space.n_actions}")
    print(f"Initial valid actions: {adaptive_space.get_num_valid_actions()}")
    print(f"Valid action indices: {adaptive_space.get_valid_actions()}")
    print(f"Steering values: {[adaptive_space.get_steering_value(i) for i in adaptive_space.get_valid_actions()]}")

    # Simulate some advantages
    valid_actions = adaptive_space.get_valid_actions()
    action_indices = np.random.choice(valid_actions, size=100)
    advantages = np.random.randn(100)

    adaptive_space.update_advantages(action_indices, advantages)
    print(f"\nAverage advantages computed")

    # Unmask and prune
    print(f"\nBefore unmask/prune: {adaptive_space.get_num_valid_actions()} valid actions")
    adaptive_space.unmask_and_prune()
    print(f"After unmask/prune: {adaptive_space.get_num_valid_actions()} valid actions")
    print(f"Valid action indices: {adaptive_space.get_valid_actions()}")
    print(f"Progress: {adaptive_space.get_progress():.2%}")
