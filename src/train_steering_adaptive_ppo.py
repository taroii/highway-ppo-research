"""
Training script for Steering-Only Adaptive PPO with Pruning

This script trains a PPO agent with adaptive action space that:
- Controls steering only (no acceleration)
- Starts with sparse actions (e.g., 5 steering angles)
- Gradually unmasks new actions near high-advantage regions
- Prunes low-performing actions to maintain constant L1 norm
"""

import numpy as np
import gymnasium as gym
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import PPO

import highway_env  # noqa: F401
from adaptive_action_space_steering import AdaptiveSteeringActionSpace, AdaptiveSteeringActionWrapper
from adaptive_ppo import AdaptivePPO


def make_adaptive_steering_env(rank=0, seed=0):
    """
    Create a highway environment with adaptive discrete steering actions.
    """
    def _init():
        # Create environment with continuous action space (steering only)
        env = gym.make(
            "highway-fast-v0",
            config={
                "action": {
                    "type": "ContinuousAction",
                    "steering_range": [-np.pi/4, np.pi/4],
                    "longitudinal": False,  # No throttle control
                    "lateral": True,        # Steering only
                },
                "duration": 40,
                "policy_frequency": 2,  # Control at 2 Hz
            }
        )

        # Wrap with adaptive action space
        adaptive_space = AdaptiveSteeringActionSpace(
            steering_range=(-np.pi/4, np.pi/4),
            final_grid_size=20,            # 20 possible steering angles
            initial_active_actions=5,       # Start with 5 active
            unmask_rate=0.2,
            prune_rate=0.1,
            advantage_threshold=0.0,
        )

        env = AdaptiveSteeringActionWrapper(env, adaptive_space)
        env.reset(seed=seed + rank)
        return env

    return _init


def make_baseline_steering_env(rank=0, seed=0):
    """
    Create a baseline highway environment with standard discrete steering actions.
    """
    def _init():
        env = gym.make(
            "highway-fast-v0",
            config={
                "action": {
                    "type": "DiscreteAction",
                    "steering_range": [-np.pi/4, np.pi/4],
                    "longitudinal": False,      # No throttle control
                    "lateral": True,            # Steering only
                    "actions_per_axis": 5,      # 5 steering angles (same as initial adaptive)
                },
                "duration": 40,
                "policy_frequency": 2,
            }
        )
        env.reset(seed=seed + rank)
        return env

    return _init


def train_adaptive_steering_ppo(
    total_timesteps=100_000,
    n_envs=4,
    save_path="highway_adaptive_steering_ppo",
):
    """
    Train PPO with adaptive steering action space and pruning.
    """
    print("=" * 80)
    print("Training Adaptive Steering PPO with Pruning")
    print("=" * 80)

    # Create the adaptive action space (shared across all envs)
    adaptive_space = AdaptiveSteeringActionSpace(
        steering_range=(-np.pi/4, np.pi/4),
        final_grid_size=20,
        initial_active_actions=5,
        unmask_rate=0.2,
        prune_rate=0.1,
        advantage_threshold=0.0,
    )

    # Create vectorized environments
    def make_env_shared(rank):
        def _init():
            env = gym.make(
                "highway-fast-v0",
                config={
                    "action": {
                        "type": "ContinuousAction",
                        "steering_range": [-np.pi/4, np.pi/4],
                        "longitudinal": False,
                        "lateral": True,
                    },
                    "duration": 40,
                    "policy_frequency": 2,
                }
            )
            env = AdaptiveSteeringActionWrapper(env, adaptive_space)
            env.reset(seed=rank)
            return env
        return _init

    env = DummyVecEnv([make_env_shared(i) for i in range(n_envs)])

    # Create the model
    model = AdaptivePPO(
        "MaskedMlpPolicy",
        env,
        adaptive_action_space=adaptive_space,
        unmask_frequency=10,  # Unmask/prune every 10 policy updates
        n_steps=512 // n_envs,
        batch_size=64,
        n_epochs=10,
        learning_rate=5e-4,
        gamma=0.9,
        verbose=1,
        tensorboard_log=f"{save_path}/",
    )

    # Train
    print(f"\nStarting training for {total_timesteps} timesteps...")
    print(f"Initial valid actions: {adaptive_space.get_num_valid_actions()}/{adaptive_space.n_actions}")
    print(f"Action space: Steering only (no acceleration)")

    model.learn(total_timesteps=total_timesteps)

    print(f"\nFinal valid actions: {adaptive_space.get_num_valid_actions()}/{adaptive_space.n_actions}")
    print(f"Actions explored: {adaptive_space.get_progress():.1%}")

    # Save the model
    model.save(f"{save_path}/model")
    print(f"\nModel saved to {save_path}/model")

    return model, adaptive_space


def train_baseline_steering_ppo(
    total_timesteps=100_000,
    n_envs=4,
    save_path="highway_baseline_steering_ppo",
):
    """
    Train baseline PPO with standard discrete steering actions.
    """
    print("=" * 80)
    print("Training Baseline Steering PPO (5 uniform actions)")
    print("=" * 80)

    # Create vectorized environments
    env = make_vec_env(
        make_baseline_steering_env(),
        n_envs=n_envs,
        vec_env_cls=DummyVecEnv,
    )

    # Create the model
    model = PPO(
        "MlpPolicy",
        env,
        n_steps=512 // n_envs,
        batch_size=64,
        n_epochs=10,
        learning_rate=5e-4,
        gamma=0.9,
        verbose=1,
        tensorboard_log=f"{save_path}/",
    )

    # Train
    print(f"\nStarting training for {total_timesteps} timesteps...")
    print(f"Action space: 5 discrete steering angles (uniform)")

    model.learn(total_timesteps=total_timesteps)

    # Save the model
    model.save(f"{save_path}/model")
    print(f"\nModel saved to {save_path}/model")

    return model


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="both", choices=["adaptive", "baseline", "both"])
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--n-envs", type=int, default=4)

    args = parser.parse_args()

    if args.mode in ["adaptive", "both"]:
        print("\n" + "=" * 80)
        print("ADAPTIVE STEERING PPO TRAINING")
        print("=" * 80)
        adaptive_model, adaptive_space = train_adaptive_steering_ppo(
            total_timesteps=args.timesteps,
            n_envs=args.n_envs,
        )

    if args.mode in ["baseline", "both"]:
        print("\n" + "=" * 80)
        print("BASELINE STEERING PPO TRAINING")
        print("=" * 80)
        baseline_model = train_baseline_steering_ppo(
            total_timesteps=args.timesteps,
            n_envs=args.n_envs,
        )

    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print("\nTo visualize training with TensorBoard:")
    print("  tensorboard --logdir highway_adaptive_steering_ppo")
    print("  tensorboard --logdir highway_baseline_steering_ppo")
