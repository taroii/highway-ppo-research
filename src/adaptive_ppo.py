"""
Adaptive PPO with Action Masking

Custom PPO algorithm that integrates adaptive action space unmasking
based on advantage estimates during training.
"""

import numpy as np
import torch as th
from typing import Any, ClassVar, Dict, Optional, Type, TypeVar, Union
import warnings

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.utils import obs_as_tensor

from masked_ppo_policy import MaskedPPOPolicy
from adaptive_action_space import AdaptiveActionSpace


SelfAdaptivePPO = TypeVar("SelfAdaptivePPO", bound="AdaptivePPO")


class AdaptiveRolloutBuffer(RolloutBuffer):
    """
    Extended rollout buffer that also stores action masks.
    """

    def __init__(self, *args, **kwargs):
        self.action_masks = None
        super().__init__(*args, **kwargs)

    def reset(self) -> None:
        super().reset()
        if self.action_masks is not None:
            self.action_masks = np.zeros(
                (self.buffer_size, self.n_envs, self.action_masks.shape[-1]),
                dtype=np.bool_,
            )

    def add(
        self,
        *args,
        action_masks: Optional[np.ndarray] = None,
        **kwargs,
    ) -> None:
        """
        Add elements to the buffer, including action masks.
        """
        # Initialize action_masks buffer on first add if provided
        if action_masks is not None and self.action_masks is None:
            self.action_masks = np.zeros(
                (self.buffer_size, self.n_envs, action_masks.shape[-1]),
                dtype=np.bool_,
            )

        # Store action masks
        if action_masks is not None:
            self.action_masks[self.pos] = action_masks.copy()

        # Call parent add
        super().add(*args, **kwargs)

    # Don't override get() - just use parent implementation
    # We'll handle action masks separately in the training loop


class AdaptivePPO(PPO):
    """
    PPO with adaptive action space unmasking.

    This algorithm gradually unmasks new actions during training based on
    advantage estimates, allowing the agent to explore finer-grained actions
    as it learns.
    """

    policy_aliases: ClassVar[Dict[str, Type[MaskedPPOPolicy]]] = {
        "MaskedMlpPolicy": MaskedPPOPolicy,
    }

    def __init__(
        self,
        policy: Union[str, Type[MaskedPPOPolicy]],
        env: GymEnv,
        adaptive_action_space: AdaptiveActionSpace,
        unmask_frequency: int = 10,  # Unmask every N policy updates
        *args,
        **kwargs,
    ):
        """
        Args:
            policy: Policy type or string alias
            env: Environment
            adaptive_action_space: AdaptiveActionSpace instance managing masking
            unmask_frequency: How often to unmask new actions (in policy updates)
            *args, **kwargs: Additional arguments for PPO
        """
        self.adaptive_action_space = adaptive_action_space
        self.unmask_frequency = unmask_frequency
        self._num_policy_updates = 0

        # Override default policy if using string
        if isinstance(policy, str) and policy == "MlpPolicy":
            policy = "MaskedMlpPolicy"

        super().__init__(policy, env, *args, **kwargs)

        # Track advantages per action for unmasking
        self.action_advantages_buffer = []

    @classmethod
    def load(
        cls,
        path: str,
        env: Optional[GymEnv] = None,
        device: Union[th.device, str] = "auto",
        custom_objects: Optional[Dict[str, Any]] = None,
        print_system_info: bool = False,
        force_reset: bool = True,
        **kwargs,
    ):
        """
        Load the model from a zip-file.

        This overrides the parent load() to handle the adaptive_action_space parameter.

        Args:
            path: Path to the file (or a file-like) where to load the agent from
            env: Environment that the model was trained on. Must be provided.
            device: Device on which the code should run
            custom_objects: Dictionary of custom objects to replace on load
            print_system_info: Whether to print system info
            force_reset: Force a call to reset() before training
            **kwargs: Additional keyword arguments (must include 'adaptive_action_space')

        Returns:
            Loaded model
        """
        if env is None:
            raise ValueError("env must be provided when loading AdaptivePPO")

        # Get adaptive_action_space from kwargs or custom_objects
        if "adaptive_action_space" not in kwargs:
            if custom_objects is not None and "adaptive_action_space" in custom_objects:
                kwargs["adaptive_action_space"] = custom_objects["adaptive_action_space"]
            else:
                raise ValueError("adaptive_action_space must be provided when loading AdaptivePPO")

        adaptive_action_space = kwargs["adaptive_action_space"]
        unmask_frequency = kwargs.get("unmask_frequency", 10)

        # Load as a regular PPO model first (just to get the saved data)
        import zipfile
        import json
        import pathlib

        # Add .zip extension if not present
        if not path.endswith(".zip"):
            path = path + ".zip"

        # Load the data from the zip file
        with zipfile.ZipFile(path, "r") as archive:
            # Load the parameters
            data_json = archive.read("data").decode("utf-8")
            data = json.loads(data_json)

        # Get hyperparameters from saved data if available
        learning_rate = data.get("learning_rate", 5e-4)
        n_steps = data.get("n_steps", 128)
        batch_size = data.get("batch_size", 64)
        n_epochs = data.get("n_epochs", 10)
        gamma = data.get("gamma", 0.9)
        gae_lambda = data.get("gae_lambda", 0.95)

        # Create a new model with the adaptive_action_space
        model = cls(
            policy="MaskedMlpPolicy",
            env=env,
            adaptive_action_space=adaptive_action_space,
            unmask_frequency=unmask_frequency,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            verbose=0,
        )

        # Now load the parameters (network weights)
        model.set_parameters(load_path_or_dict=path, device=device)

        return model

    def _setup_model(self) -> None:
        """Setup the model and replace rollout buffer with adaptive version."""
        super()._setup_model()

        # Replace rollout buffer with adaptive version
        self.rollout_buffer = AdaptiveRolloutBuffer(
            self.n_steps,
            self.observation_space,
            self.action_space,
            device=self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=self.n_envs,
        )

    def collect_rollouts(
        self,
        env,
        callback,
        rollout_buffer,
        n_rollout_steps: int,
    ) -> bool:
        """
        Collect experiences using the current policy and fill the rollout buffer.
        Modified to include action masks.
        """
        assert self._last_obs is not None, "No previous observation was provided"

        # Switch to eval mode
        self.policy.set_training_mode(False)

        n_steps = 0
        rollout_buffer.reset()

        callback.on_rollout_start()

        while n_steps < n_rollout_steps:
            with th.no_grad():
                # Convert to pytorch tensor
                obs_tensor = obs_as_tensor(self._last_obs, self.device)

                # Get current action mask
                action_mask = self.adaptive_action_space.get_action_mask()
                # Expand mask for each environment
                action_mask_batch = np.tile(action_mask, (self.n_envs, 1))
                action_mask_tensor = th.as_tensor(action_mask_batch, device=self.device)

                # Get actions with masking
                actions, values, log_probs = self.policy.forward(
                    obs_tensor,
                    deterministic=False,
                    action_masks=action_mask_tensor,
                )

            actions = actions.cpu().numpy()

            # Rescale and perform action
            clipped_actions = actions

            # Clip the actions to avoid out of bound error
            from gymnasium import spaces as gym_spaces
            if isinstance(self.action_space, gym_spaces.Box):
                clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)

            new_obs, rewards, dones, infos = env.step(clipped_actions)

            self.num_timesteps += self.n_envs

            # Give access to local variables
            callback.update_locals(locals())
            if not callback.on_step():
                return False

            self._update_info_buffer(infos, dones)
            n_steps += 1

            if isinstance(self.action_space, gym_spaces.Discrete):
                # Reshape in case of discrete action
                actions = actions.reshape(-1, 1)

            # Handle timeout by bootstrapping with value function
            for idx, done in enumerate(dones):
                if (
                    done
                    and infos[idx].get("terminal_observation") is not None
                    and infos[idx].get("TimeLimit.truncated", False)
                ):
                    terminal_obs = self.policy.obs_to_tensor(infos[idx]["terminal_observation"])[0]
                    with th.no_grad():
                        terminal_value = self.policy.predict_values(terminal_obs)[0]
                    rewards[idx] += self.gamma * terminal_value

            # Add to buffer with action masks
            rollout_buffer.add(
                self._last_obs,
                actions,
                rewards,
                self._last_episode_starts,
                values,
                log_probs,
                action_masks=action_mask_batch,
            )
            self._last_obs = new_obs
            self._last_episode_starts = dones

        # Compute value for the last timestep
        with th.no_grad():
            obs_tensor = obs_as_tensor(new_obs, self.device)
            values = self.policy.predict_values(obs_tensor)

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

        callback.on_rollout_end()

        return True

    def train(self) -> None:
        """
        Update policy using the currently gathered rollout buffer.
        Modified to use action masks and track advantages for unmasking.
        """
        # Switch to train mode
        self.policy.set_training_mode(True)

        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)

        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)

        # Optional: clip range for the value function
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)

        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []

        continue_training = True

        # Track advantages for unmasking
        all_actions = []
        all_advantages = []

        # Train for n_epochs epochs
        for epoch in range(self.n_epochs):
            approx_kl_divs = []

            # Do a complete pass on the rollout buffer
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                observations = rollout_data.observations
                old_values = rollout_data.old_values
                old_log_prob = rollout_data.old_log_prob
                advantages = rollout_data.advantages
                returns = rollout_data.returns

                # Get current action mask (same for all samples in this training iteration)
                action_mask = self.adaptive_action_space.get_action_mask()
                batch_size_current = observations.shape[0]
                action_mask_batch = np.tile(action_mask, (batch_size_current, 1))
                action_masks = th.as_tensor(action_mask_batch, device=self.device)

                # Store for unmasking
                all_actions.append(actions.cpu().numpy())
                all_advantages.append(advantages.cpu().numpy())

                from gymnasium import spaces as gym_spaces
                if isinstance(self.action_space, gym_spaces.Discrete):
                    # Convert discrete action from float to long
                    actions = actions.long().flatten()

                # Re-sample policy and value
                values, log_prob, entropy = self.policy.evaluate_actions(
                    observations,
                    actions,
                    action_masks=action_masks,
                )
                values = values.flatten()

                # Normalize advantage
                advantages = advantages.flatten()
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # Ratio between old and new policy
                ratio = th.exp(log_prob - old_log_prob)

                # Clipped surrogate loss
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

                # Logging
                pg_losses.append(policy_loss.item())
                clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                if self.clip_range_vf is None:
                    # No clipping
                    values_pred = values
                else:
                    # Clip the difference between old and new value
                    values_pred = old_values + th.clamp(
                        values - old_values, -clip_range_vf, clip_range_vf
                    )

                # Value loss
                value_loss = th.nn.functional.mse_loss(returns, values_pred)
                value_losses.append(value_loss.item())

                # Entropy loss favor exploration
                if entropy is None:
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)

                entropy_losses.append(entropy_loss.item())

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

                # Optimization step
                self.policy.optimizer.zero_grad()
                loss.backward()

                # Clip grad norm
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()
                approx_kl_divs.append(th.mean(old_log_prob - log_prob).detach().cpu().numpy())

            self._n_updates += 1
            if not continue_training:
                break

        # Update adaptive action space based on advantages
        self._num_policy_updates += 1
        if self._num_policy_updates % self.unmask_frequency == 0:
            self._update_action_space(all_actions, all_advantages)

        # Log training metrics
        explained_var = self._explained_variance(
            self.rollout_buffer.values.flatten(),
            self.rollout_buffer.returns.flatten(),
        )

        # Logging
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)

        # Log adaptive action space stats
        self.logger.record("adaptive/num_valid_actions", self.adaptive_action_space.get_num_valid_actions())
        self.logger.record("adaptive/progress", self.adaptive_action_space.get_progress())

    def _update_action_space(self, all_actions: list, all_advantages: list):
        """
        Update the adaptive action space based on collected advantages.

        Args:
            all_actions: List of action batches
            all_advantages: List of advantage batches
        """
        # Concatenate all batches
        actions = np.concatenate(all_actions, axis=0).flatten()
        advantages = np.concatenate(all_advantages, axis=0).flatten()

        # Update advantage statistics
        self.adaptive_action_space.update_advantages(actions, advantages)

        # Unmask new actions (and prune if the method exists)
        if hasattr(self.adaptive_action_space, 'unmask_and_prune'):
            # Steering-only version with pruning
            self.adaptive_action_space.unmask_and_prune()
        elif hasattr(self.adaptive_action_space, 'unmask_actions'):
            # Original 2D version without pruning
            self.adaptive_action_space.unmask_actions()
        else:
            raise AttributeError("Adaptive action space must have either unmask_and_prune() or unmask_actions() method")

        # Log progress
        num_valid = self.adaptive_action_space.get_num_valid_actions()
        progress = self.adaptive_action_space.get_progress()
        print(f"[Adaptive] Valid actions: {num_valid}/{self.adaptive_action_space.n_actions} ({progress:.1%})")

    def _explained_variance(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """
        Computes fraction of variance that ypred explains about y.
        Returns 1 - Var[y-ypred] / Var[y]
        """
        var_y = np.var(y_true)
        return 1 - np.var(y_true - y_pred) / (var_y + 1e-8)
