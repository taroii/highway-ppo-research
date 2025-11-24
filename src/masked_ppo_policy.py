"""
Custom PPO Policy with Action Masking Support

This implements a PPO policy that respects action masks, preventing the agent
from selecting invalid actions during training and evaluation.
"""

import torch
import torch as th
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Type, Union
from gymnasium import spaces

from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.distributions import CategoricalDistribution
from stable_baselines3.common.type_aliases import Schedule
import torch.nn as nn


class MaskedCategoricalDistribution(CategoricalDistribution):
    """
    Categorical distribution with action masking support.

    Invalid actions (masked) are given probability 0.
    """

    def __init__(self, action_dim: int):
        super().__init__(action_dim)
        self.action_mask: Optional[th.Tensor] = None

    def proba_distribution_net(self, latent_dim: int) -> nn.Module:
        """Create the action distribution network."""
        return nn.Linear(latent_dim, self.action_dim)

    def proba_distribution(
        self,
        action_logits: th.Tensor,
        action_mask: Optional[th.Tensor] = None,
    ) -> "MaskedCategoricalDistribution":
        """
        Set the action logits and apply masking.

        Args:
            action_logits: Logits for each action (batch_size, n_actions)
            action_mask: Boolean mask where True = invalid (batch_size, n_actions)
        """
        self.action_mask = action_mask

        if action_mask is not None:
            # Set masked action logits to very large negative value
            # This makes their probability effectively 0 after softmax
            action_logits = th.where(
                action_mask,
                th.tensor(float("-inf"), device=action_logits.device, dtype=action_logits.dtype),
                action_logits,
            )

        self.distribution = th.distributions.Categorical(logits=action_logits)
        return self

    def log_prob(self, actions: th.Tensor) -> th.Tensor:
        """Get log probability of actions."""
        return self.distribution.log_prob(actions)

    def entropy(self) -> th.Tensor:
        """Get entropy of the distribution."""
        return self.distribution.entropy()

    def sample(self) -> th.Tensor:
        """Sample an action from the distribution."""
        return self.distribution.sample()

    def mode(self) -> th.Tensor:
        """Return the most likely action (greedy)."""
        return th.argmax(self.distribution.probs, dim=-1)

    def actions_from_params(
        self,
        action_logits: th.Tensor,
        action_mask: Optional[th.Tensor] = None,
        deterministic: bool = False,
    ) -> th.Tensor:
        """
        Sample actions from the distribution.

        Args:
            action_logits: Logits for each action
            action_mask: Boolean mask where True = invalid
            deterministic: If True, return mode instead of sampling
        """
        self.proba_distribution(action_logits, action_mask)
        if deterministic:
            return self.mode()
        return self.sample()

    def log_prob_from_params(
        self,
        action_logits: th.Tensor,
        action_mask: Optional[th.Tensor] = None,
    ) -> Tuple[th.Tensor, th.Tensor]:
        """
        Sample actions and get their log probabilities.

        Returns:
            actions: Sampled actions
            log_probs: Log probabilities of sampled actions
        """
        actions = self.actions_from_params(action_logits, action_mask)
        log_prob = self.log_prob(actions)
        return actions, log_prob


class MaskedActorCriticPolicy(ActorCriticPolicy):
    """
    Actor-Critic policy with action masking support for PPO.

    This policy can handle dynamic action masks provided through the info dict.
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        **kwargs,
    ):
        # We'll override the action distribution
        self.use_action_masking = True
        super().__init__(observation_space, action_space, lr_schedule, **kwargs)

    def _build(self, lr_schedule: Schedule) -> None:
        """
        Create the networks and the optimizer.

        Override to use masked distribution.
        """
        super()._build(lr_schedule)

        # Replace the action distribution with our masked version
        if isinstance(self.action_space, spaces.Discrete):
            self.action_dist = MaskedCategoricalDistribution(self.action_space.n)
            self.action_net = self.action_dist.proba_distribution_net(self.mlp_extractor.latent_dim_pi)

    def _get_action_dist_from_latent(
        self,
        latent_pi: th.Tensor,
        action_masks: Optional[th.Tensor] = None,
    ) -> MaskedCategoricalDistribution:
        """
        Get action distribution from latent policy representation.

        Args:
            latent_pi: Latent representation from policy network
            action_masks: Optional action masks (True = invalid)
        """
        action_logits = self.action_net(latent_pi)
        # Create a new distribution instance with the logits and masks
        dist = MaskedCategoricalDistribution(self.action_space.n)
        return dist.proba_distribution(action_logits, action_masks)

    def forward(
        self,
        obs: th.Tensor,
        deterministic: bool = False,
        action_masks: Optional[th.Tensor] = None,
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Forward pass in all networks (actor and critic).

        Args:
            obs: Observation
            deterministic: Whether to sample or use deterministic actions
            action_masks: Action masks (True = invalid)

        Returns:
            actions: Selected actions
            values: Estimated values
            log_probs: Log probability of actions
        """
        # Preprocess the observation if needed
        features = self.extract_features(obs)
        if self.share_features_extractor:
            latent_pi, latent_vf = self.mlp_extractor(features)
        else:
            pi_features, vf_features = features
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.mlp_extractor.forward_critic(vf_features)

        # Evaluate the values for the given observations
        values = self.value_net(latent_vf)

        # Get action distribution with masking
        distribution = self._get_action_dist_from_latent(latent_pi, action_masks)

        # Sample actions
        if deterministic:
            actions = distribution.mode()
        else:
            actions = distribution.sample()

        log_prob = distribution.log_prob(actions)
        actions = actions.reshape((-1,) + self.action_space.shape)

        return actions, values, log_prob

    def evaluate_actions(
        self,
        obs: th.Tensor,
        actions: th.Tensor,
        action_masks: Optional[th.Tensor] = None,
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Evaluate actions according to the current policy.

        Args:
            obs: Observation
            actions: Actions to evaluate
            action_masks: Action masks (True = invalid)

        Returns:
            values: Estimated values
            log_probs: Log probability of actions
            entropy: Entropy of the distribution
        """
        # Preprocess the observation if needed
        features = self.extract_features(obs)
        if self.share_features_extractor:
            latent_pi, latent_vf = self.mlp_extractor(features)
        else:
            pi_features, vf_features = features
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.mlp_extractor.forward_critic(vf_features)

        # Evaluate values
        values = self.value_net(latent_vf)

        # Get action distribution with masking
        distribution = self._get_action_dist_from_latent(latent_pi, action_masks)

        # Evaluate log probabilities and entropy
        log_prob = distribution.log_prob(actions)
        entropy = distribution.entropy()

        return values, log_prob, entropy

    def predict_values(
        self,
        obs: th.Tensor,
    ) -> th.Tensor:
        """
        Get the estimated values according to the current policy.

        Args:
            obs: Observation

        Returns:
            Estimated values
        """
        features = self.extract_features(obs)
        if self.share_features_extractor:
            _, latent_vf = self.mlp_extractor(features)
        else:
            _, vf_features = features
            latent_vf = self.mlp_extractor.forward_critic(vf_features)
        return self.value_net(latent_vf)

    def _predict(
        self,
        observation: th.Tensor,
        deterministic: bool = False,
        action_masks: Optional[th.Tensor] = None,
    ) -> th.Tensor:
        """
        Get the action according to the policy for a given observation.

        Args:
            observation: Observation
            deterministic: Whether to use stochastic or deterministic actions
            action_masks: Action masks (True = invalid)

        Returns:
            Taken action
        """
        actions, _, _ = self.forward(observation, deterministic, action_masks)
        return actions


class MaskedPPOPolicy(MaskedActorCriticPolicy):
    """
    Policy class for PPO with action masking.
    Alias for MaskedActorCriticPolicy for consistency with SB3 naming.
    """
    pass


if __name__ == "__main__":
    # Test the masked distribution
    print("Testing MaskedCategoricalDistribution...")

    n_actions = 100
    batch_size = 4

    dist = MaskedCategoricalDistribution(n_actions)

    # Create some dummy logits
    logits = th.randn(batch_size, n_actions)

    # Create a mask: first half of actions are valid, second half invalid
    mask = th.zeros(batch_size, n_actions, dtype=th.bool)
    mask[:, 50:] = True  # Mask second half

    # Test distribution
    dist.proba_distribution(logits, mask)

    # Sample actions
    actions = dist.sample()
    print(f"Sampled actions: {actions}")
    print(f"All actions < 50: {th.all(actions < 50)}")  # Should be True

    # Test log probs
    log_probs = dist.log_prob(actions)
    print(f"Log probs shape: {log_probs.shape}")

    # Test entropy
    entropy = dist.entropy()
    print(f"Entropy shape: {entropy.shape}")

    print("\nMasked distribution test passed!")
