"""
Standalone evaluation script for trained Adaptive PPO models.

This script loads a saved model and evaluates it on the highway-env environment,
reporting performance statistics and optionally rendering/recording episodes.

Usage Examples
  Evaluate adaptive steering-only model:
  python evaluate_saved_model.py --model highway_adaptive_steering_ppo/model --steering --episodes 100

  Evaluate baseline steering-only model:
  python evaluate_saved_model.py --model highway_baseline_steering_ppo/model --baseline --steering --episodes 100

  Evaluate original 2D adaptive model:
  python evaluate_saved_model.py --model highway_adaptive_ppo/model --episodes 100

  Evaluate original 2D baseline model:
  python evaluate_saved_model.py --model highway_baseline_ppo/model --baseline --episodes 100

  With rendering:
  python evaluate_saved_model.py --model highway_adaptive_steering_ppo/model --steering --episodes 5 --render
"""

import argparse
import numpy as np
import gymnasium as gym
from collections import Counter

import highway_env  # noqa: F401
from adaptive_ppo import AdaptivePPO
from adaptive_action_space import AdaptiveActionSpace, AdaptiveDiscreteActionWrapper
from adaptive_action_space_steering import AdaptiveSteeringActionSpace, AdaptiveSteeringActionWrapper
from stable_baselines3 import PPO


def create_adaptive_env(render_mode=None):
    """
    Create an adaptive action space environment for evaluation.

    Args:
        render_mode: "human" for visualization, None for no rendering

    Returns:
        env: Wrapped environment
        adaptive_space: AdaptiveActionSpace instance
    """
    # Create the adaptive action space (same config as training)
    adaptive_space = AdaptiveActionSpace(
        steering_range=(-np.pi/4, np.pi/4),
        acceleration_range=(-5.0, 5.0),
        final_grid_size=(10, 10),
        initial_grid_size=(5, 5),
    )

    # Unmask ALL actions for evaluation
    # This allows the model to use its full learned action space
    adaptive_space.action_mask[:] = False

    print(f"Adaptive action space: {adaptive_space.get_num_valid_actions()}/{adaptive_space.n_actions} actions available")

    # Create environment with continuous action space
    env = gym.make(
        "highway-fast-v0",
        render_mode=render_mode,
        config={
            "action": {
                "type": "ContinuousAction",
                "acceleration_range": [-5.0, 5.0],
                "steering_range": [-np.pi/4, np.pi/4],
                "longitudinal": True,
                "lateral": True,
            },
            "duration": 40,
            "policy_frequency": 2,
        }
    )

    # Wrap with adaptive discrete action space
    env = AdaptiveDiscreteActionWrapper(env, adaptive_space)

    return env, adaptive_space


def create_baseline_env(render_mode=None):
    """
    Create a baseline discrete action environment for evaluation.

    Args:
        render_mode: "human" for visualization, None for no rendering

    Returns:
        env: Baseline environment
    """
    env = gym.make(
        "highway-fast-v0",
        render_mode=render_mode,
        config={
            "action": {
                "type": "DiscreteAction",
                "acceleration_range": [-5.0, 5.0],
                "steering_range": [-np.pi/4, np.pi/4],
                "longitudinal": True,
                "lateral": True,
                "actions_per_axis": 5,
            },
            "duration": 40,
            "policy_frequency": 2,
        }
    )

    print(f"Baseline action space: {env.action_space.n} discrete actions")

    return env


def create_adaptive_steering_env(render_mode=None):
    """
    Create an adaptive steering-only action space environment for evaluation.

    Args:
        render_mode: "human" for visualization, None for no rendering

    Returns:
        env: Wrapped environment
        adaptive_space: AdaptiveSteeringActionSpace instance
    """
    # Create the adaptive steering action space (same config as training)
    adaptive_space = AdaptiveSteeringActionSpace(
        steering_range=(-np.pi/4, np.pi/4),
        final_grid_size=20,
        initial_active_actions=5,
    )

    # Unmask ALL actions for evaluation
    adaptive_space.action_mask[:] = False

    print(f"Adaptive steering action space: {adaptive_space.get_num_valid_actions()}/{adaptive_space.n_actions} actions available")

    # Create environment with continuous action space (steering only)
    env = gym.make(
        "highway-fast-v0",
        render_mode=render_mode,
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

    # Wrap with adaptive discrete action space
    env = AdaptiveSteeringActionWrapper(env, adaptive_space)

    return env, adaptive_space


def create_baseline_steering_env(render_mode=None):
    """
    Create a baseline steering-only discrete action environment for evaluation.

    Args:
        render_mode: "human" for visualization, None for no rendering

    Returns:
        env: Baseline steering environment
    """
    env = gym.make(
        "highway-fast-v0",
        render_mode=render_mode,
        config={
            "action": {
                "type": "DiscreteAction",
                "steering_range": [-np.pi/4, np.pi/4],
                "longitudinal": False,
                "lateral": True,
                "actions_per_axis": 5,
            },
            "duration": 40,
            "policy_frequency": 2,
        }
    )

    print(f"Baseline steering action space: {env.action_space.n} discrete actions")

    return env


def evaluate_model(
    model,
    env,
    n_episodes=10,
    deterministic=True,
    render=False,
    track_actions=False,
    adaptive_space=None,
):
    """
    Evaluate a trained model on the environment.

    Args:
        model: Trained PPO or AdaptivePPO model
        env: Evaluation environment
        n_episodes: Number of episodes to run
        deterministic: Use deterministic policy
        render: Render the environment
        track_actions: Track which actions are used
        adaptive_space: AdaptiveActionSpace instance (for action tracking)

    Returns:
        dict: Evaluation statistics
    """
    episode_rewards = []
    episode_lengths = []
    collision_count = 0
    success_count = 0
    action_counts = Counter()

    print(f"\nEvaluating for {n_episodes} episodes...")
    print(f"Policy: {'Deterministic' if deterministic else 'Stochastic'}")
    print(f"Rendering: {'Enabled' if render else 'Disabled'}")
    print("-" * 60)

    for ep in range(n_episodes):
        obs, info = env.reset()
        done = truncated = False
        episode_reward = 0
        episode_length = 0

        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=deterministic)

            # Track action usage
            if track_actions:
                action_counts[int(action)] += 1

            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            episode_length += 1

            if render:
                env.render()

        # Track episode outcome
        if info.get("crashed", False):
            collision_count += 1
        if episode_reward > 20:  # Heuristic for "success"
            success_count += 1

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)

        print(f"Episode {ep+1:2d}: Reward = {episode_reward:6.2f}, Length = {episode_length:3d}", end="")
        if info.get("crashed", False):
            print(" [CRASHED]")
        else:
            print()

    print("-" * 60)

    # Compute statistics
    results = {
        "mean_reward": np.mean(episode_rewards),
        "std_reward": np.std(episode_rewards),
        "min_reward": np.min(episode_rewards),
        "max_reward": np.max(episode_rewards),
        "mean_length": np.mean(episode_lengths),
        "std_length": np.std(episode_lengths),
        "collision_rate": collision_count / n_episodes,
        "success_rate": success_count / n_episodes,
    }

    # Action distribution analysis
    if track_actions and len(action_counts) > 0:
        results["unique_actions_used"] = len(action_counts)
        results["action_distribution"] = dict(action_counts)

        # Show action usage
        print(f"\nAction Usage:")
        print(f"  Unique actions used: {len(action_counts)}")

        if adaptive_space is not None:
            # Show most frequent actions with their steering/acceleration values
            print(f"  Top 10 most frequent actions:")
            for action_idx, count in action_counts.most_common(10):
                pct = 100 * count / sum(action_counts.values())

                # Check if this is a steering-only or 2D action space
                if hasattr(adaptive_space, 'get_steering_value'):
                    # Steering-only action space
                    steering = adaptive_space.get_steering_value(action_idx)
                    print(f"    Action {action_idx:3d}: steering={steering:6.3f} - {count:4d} times ({pct:5.1f}%)")
                elif hasattr(adaptive_space, 'get_action_values'):
                    # 2D action space (steering + acceleration)
                    steering, accel = adaptive_space.get_action_values(action_idx)
                    print(f"    Action {action_idx:3d}: steering={steering:6.3f}, accel={accel:5.2f} - {count:4d} times ({pct:5.1f}%)")
                else:
                    # Fallback
                    print(f"    Action {action_idx:3d}: {count:4d} times ({pct:5.1f}%)")

    return results


def print_results(results):
    """Pretty print evaluation results."""
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Mean Reward:      {results['mean_reward']:8.2f} ± {results['std_reward']:.2f}")
    print(f"Min/Max Reward:   {results['min_reward']:8.2f} / {results['max_reward']:.2f}")
    print(f"Mean Length:      {results['mean_length']:8.2f} ± {results['std_length']:.2f}")
    print(f"Collision Rate:   {results['collision_rate']:8.1%}")
    print(f"Success Rate:     {results['success_rate']:8.1%}")

    if "unique_actions_used" in results:
        print(f"Unique Actions:   {results['unique_actions_used']:8d}")

    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained Adaptive PPO model")
    parser.add_argument("--model", type=str, default="highway_adaptive_ppo/model",
                        help="Path to the saved model (without .zip extension)")
    parser.add_argument("--episodes", type=int, default=10,
                        help="Number of evaluation episodes")
    parser.add_argument("--render", action="store_true",
                        help="Render the environment during evaluation")
    parser.add_argument("--stochastic", action="store_true",
                        help="Use stochastic policy instead of deterministic")
    parser.add_argument("--baseline", action="store_true",
                        help="Evaluate a baseline model (uses standard discrete actions)")
    parser.add_argument("--steering", action="store_true",
                        help="Evaluate a steering-only model (no acceleration control)")
    parser.add_argument("--track-actions", action="store_true", default=True,
                        help="Track and report action usage statistics")

    args = parser.parse_args()

    print("=" * 60)
    print("ADAPTIVE PPO MODEL EVALUATION")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Episodes: {args.episodes}")
    mode_str = "Baseline" if args.baseline else "Adaptive"
    if args.steering:
        mode_str += " (Steering-only)"
    print(f"Mode: {mode_str}")
    print("=" * 60)

    # Create environment
    render_mode = "human" if args.render else None

    if args.baseline and args.steering:
        # Baseline steering-only model
        env = create_baseline_steering_env(render_mode=render_mode)
        adaptive_space = None
        print("\nLoading baseline steering PPO model...")
        model = PPO.load(args.model, env=env)
    elif args.baseline:
        # Baseline 2D (steering + acceleration) model
        env = create_baseline_env(render_mode=render_mode)
        adaptive_space = None
        print("\nLoading baseline PPO model...")
        model = PPO.load(args.model, env=env)
    elif args.steering:
        # Adaptive steering-only model
        env, adaptive_space = create_adaptive_steering_env(render_mode=render_mode)
        print("\nLoading adaptive steering PPO model...")
        from stable_baselines3.common.vec_env import DummyVecEnv

        def make_env():
            return env
        vec_env = DummyVecEnv([make_env])

        model = AdaptivePPO.load(
            args.model,
            env=vec_env,
            custom_objects={
                "adaptive_action_space": adaptive_space,
            }
        )
        vec_env.close()
    else:
        # Adaptive 2D (steering + acceleration) model
        env, adaptive_space = create_adaptive_env(render_mode=render_mode)
        print("\nLoading adaptive PPO model...")
        from stable_baselines3.common.vec_env import DummyVecEnv

        def make_env():
            return env
        vec_env = DummyVecEnv([make_env])

        model = AdaptivePPO.load(
            args.model,
            env=vec_env,
            custom_objects={
                "adaptive_action_space": adaptive_space,
            }
        )
        vec_env.close()

    print("Model loaded successfully!")

    # Evaluate
    results = evaluate_model(
        model,
        env,
        n_episodes=args.episodes,
        deterministic=not args.stochastic,
        render=args.render,
        track_actions=args.track_actions,
        adaptive_space=adaptive_space,
    )

    # Print results
    print_results(results)

    # Close environment
    env.close()

    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()
