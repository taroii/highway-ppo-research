"""
Training script for Adaptive PPO on Highway-Env

This script trains a PPO agent with adaptive action space unmasking,
comparing it to a baseline with standard uniform discretization.
"""

import numpy as np
import gymnasium as gym
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import PPO
import matplotlib.pyplot as plt

import highway_env  # noqa: F401
from adaptive_action_space import AdaptiveActionSpace, AdaptiveDiscreteActionWrapper
from adaptive_ppo import AdaptivePPO
from masked_ppo_policy import MaskedPPOPolicy


def make_adaptive_env(rank=0, seed=0):
    """
    Create a highway environment with adaptive discrete action space.
    """
    def _init():
        # Create environment with continuous action space
        env = gym.make(
            "highway-fast-v0",
            config={
                "action": {
                    "type": "ContinuousAction",
                    "acceleration_range": [-5.0, 5.0],
                    "steering_range": [-np.pi/4, np.pi/4],
                    "longitudinal": True,
                    "lateral": True,
                },
                "duration": 40,  # Longer episodes
            }
        )

        # Wrap with adaptive action space
        adaptive_space = AdaptiveActionSpace(
            steering_range=(-np.pi/4, np.pi/4),
            acceleration_range=(-5.0, 5.0),
            final_grid_size=(10, 10),
            initial_grid_size=(5, 5),
            unmask_rate=0.1,
            advantage_threshold=0.0,
        )

        env = AdaptiveDiscreteActionWrapper(env, adaptive_space)
        env.reset(seed=seed + rank)
        return env

    return _init


def make_baseline_env(rank=0, seed=0):
    """
    Create a baseline highway environment with standard discrete actions.
    """
    def _init():
        env = gym.make(
            "highway-fast-v0",
            config={
                "action": {
                    "type": "DiscreteAction",
                    "acceleration_range": [-5.0, 5.0],
                    "steering_range": [-np.pi/4, np.pi/4],
                    "longitudinal": True,
                    "lateral": True,
                    "actions_per_axis": 5,  # 5x5 = 25 actions (same as initial adaptive)
                },
                "duration": 40,
            }
        )
        env.reset(seed=seed + rank)
        return env

    return _init


def train_adaptive_ppo(
    total_timesteps=100_000,
    n_envs=4,
    save_path="highway_adaptive_ppo",
):
    """
    Train PPO with adaptive action space.
    """
    print("=" * 80)
    print("Training Adaptive PPO")
    print("=" * 80)

    # Create the adaptive action space (shared across all envs)
    adaptive_space = AdaptiveActionSpace(
        steering_range=(-np.pi/4, np.pi/4),
        acceleration_range=(-5.0, 5.0),
        final_grid_size=(10, 10),
        initial_grid_size=(5, 5),
        unmask_rate=0.1,
        advantage_threshold=0.0,
    )

    # Create vectorized environments
    # Note: For simplicity with DummyVecEnv, we'll create environments that share the adaptive_space
    def make_env_shared(rank):
        def _init():
            env = gym.make(
                "highway-fast-v0",
                config={
                    "action": {
                        "type": "ContinuousAction",
                        "acceleration_range": [-5.0, 5.0],
                        "steering_range": [-np.pi/4, np.pi/4],
                        "longitudinal": True,
                        "lateral": True,
                    },
                    "duration": 40,
                }
            )
            env = AdaptiveDiscreteActionWrapper(env, adaptive_space)
            env.reset(seed=rank)
            return env
        return _init

    env = DummyVecEnv([make_env_shared(i) for i in range(n_envs)])

    # Create the model
    model = AdaptivePPO(
        "MaskedMlpPolicy",
        env,
        adaptive_action_space=adaptive_space,
        unmask_frequency=10,  # Unmask every 10 policy updates
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

    model.learn(total_timesteps=total_timesteps)

    print(f"\nFinal valid actions: {adaptive_space.get_num_valid_actions()}/{adaptive_space.n_actions}")
    print(f"Action space expansion: {adaptive_space.get_progress():.1%}")

    # Save the model
    model.save(f"{save_path}/model")
    print(f"\nModel saved to {save_path}/model")

    return model, adaptive_space


def train_baseline_ppo(
    total_timesteps=100_000,
    n_envs=4,
    save_path="highway_baseline_ppo",
):
    """
    Train baseline PPO with standard discrete actions.
    """
    print("=" * 80)
    print("Training Baseline PPO (5x5 uniform discretization)")
    print("=" * 80)

    # Create vectorized environments
    env = make_vec_env(
        make_baseline_env(),
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
    print(f"Action space: 25 actions (5x5 uniform grid)")

    model.learn(total_timesteps=total_timesteps)

    # Save the model
    model.save(f"{save_path}/model")
    print(f"\nModel saved to {save_path}/model")

    return model


def evaluate_model(model, env, n_episodes=10, render=False):
    """
    Evaluate a trained model.
    """
    episode_rewards = []
    episode_lengths = []

    for ep in range(n_episodes):
        obs, info = env.reset()
        done = truncated = False
        episode_reward = 0
        episode_length = 0

        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            episode_length += 1

            if render:
                env.render()

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)

    return {
        "mean_reward": np.mean(episode_rewards),
        "std_reward": np.std(episode_rewards),
        "mean_length": np.mean(episode_lengths),
        "std_length": np.std(episode_lengths),
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="both", choices=["adaptive", "baseline", "both"])
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--eval", action="store_true", help="Evaluate after training")
    args = parser.parse_args()

    if args.mode in ["adaptive", "both"]:
        print("\n" + "=" * 80)
        print("ADAPTIVE PPO TRAINING")
        print("=" * 80)
        adaptive_model, adaptive_space = train_adaptive_ppo(
            total_timesteps=args.timesteps,
            n_envs=args.n_envs,
        )

        if args.eval:
            print("\nEvaluating Adaptive PPO...")
            # Create eval env
            eval_env = gym.make("highway-fast-v0", config={
                "action": {
                    "type": "ContinuousAction",
                    "acceleration_range": [-5.0, 5.0],
                    "steering_range": [-np.pi/4, np.pi/4],
                    "longitudinal": True,
                    "lateral": True,
                },
            })
            eval_env = AdaptiveDiscreteActionWrapper(eval_env, adaptive_space)

            results = evaluate_model(adaptive_model, eval_env, n_episodes=10)
            print(f"Adaptive PPO Results: {results}")

    if args.mode in ["baseline", "both"]:
        print("\n" + "=" * 80)
        print("BASELINE PPO TRAINING")
        print("=" * 80)
        baseline_model = train_baseline_ppo(
            total_timesteps=args.timesteps,
            n_envs=args.n_envs,
        )

        if args.eval:
            print("\nEvaluating Baseline PPO...")
            eval_env = gym.make("highway-fast-v0", config={
                "action": {
                    "type": "DiscreteAction",
                    "acceleration_range": [-5.0, 5.0],
                    "steering_range": [-np.pi/4, np.pi/4],
                    "longitudinal": True,
                    "lateral": True,
                    "actions_per_axis": 5,
                },
            })

            results = evaluate_model(baseline_model, eval_env, n_episodes=10)
            print(f"Baseline PPO Results: {results}")

    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print("\nTo visualize training with TensorBoard:")
    print("  tensorboard --logdir highway_adaptive_ppo")
    print("  tensorboard --logdir highway_baseline_ppo")
