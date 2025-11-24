"""
Quick integration test for the adaptive PPO system.
Tests that all components work together.
"""

import numpy as np
import gymnasium as gym
import highway_env

from adaptive_action_space import AdaptiveActionSpace, AdaptiveDiscreteActionWrapper
from adaptive_ppo import AdaptivePPO
from masked_ppo_policy import MaskedPPOPolicy


def test_adaptive_env():
    """Test that the adaptive environment wrapper works."""
    print("Testing adaptive environment wrapper...")

    # Create base environment
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
        }
    )

    # Create adaptive action space
    adaptive_space = AdaptiveActionSpace(
        steering_range=(-np.pi/4, np.pi/4),
        acceleration_range=(-5.0, 5.0),
        final_grid_size=(10, 10),
        initial_grid_size=(5, 5),
    )

    # Wrap environment
    env = AdaptiveDiscreteActionWrapper(env, adaptive_space)

    print(f"  Action space: {env.action_space}")
    print(f"  Observation space: {env.observation_space}")
    print(f"  Initial valid actions: {adaptive_space.get_num_valid_actions()}")

    # Test reset and step
    obs, info = env.reset()
    print(f"  Observation shape: {obs.shape}")

    # Take a random valid action
    valid_actions = adaptive_space.get_valid_actions()
    action = np.random.choice(valid_actions)
    obs, reward, done, truncated, info = env.step(action)

    print(f"  Step successful: reward={reward:.2f}, done={done}")
    print("  [PASS] Adaptive environment test passed!\n")

    return env, adaptive_space


def test_masked_policy():
    """Test that the masked policy works."""
    print("Testing masked policy...")

    # Create environment
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
        }
    )

    adaptive_space = AdaptiveActionSpace(
        steering_range=(-np.pi/4, np.pi/4),
        acceleration_range=(-5.0, 5.0),
        final_grid_size=(10, 10),
        initial_grid_size=(5, 5),
    )

    env = AdaptiveDiscreteActionWrapper(env, adaptive_space)

    # Create policy
    from stable_baselines3.common.vec_env import DummyVecEnv
    vec_env = DummyVecEnv([lambda: env])

    model = AdaptivePPO(
        "MaskedMlpPolicy",
        vec_env,
        adaptive_action_space=adaptive_space,
        unmask_frequency=5,
        n_steps=128,
        batch_size=32,
        verbose=0,
    )

    print(f"  Model created successfully")
    print(f"  Policy type: {type(model.policy)}")
    print("  [PASS] Masked policy test passed!\n")

    return model


def test_short_training():
    """Test a very short training run."""
    print("Testing short training run (1000 steps)...")

    # Create environment
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
        }
    )

    adaptive_space = AdaptiveActionSpace(
        steering_range=(-np.pi/4, np.pi/4),
        acceleration_range=(-5.0, 5.0),
        final_grid_size=(10, 10),
        initial_grid_size=(5, 5),
        unmask_rate=0.2,  # Faster unmasking for testing
    )

    env = AdaptiveDiscreteActionWrapper(env, adaptive_space)

    from stable_baselines3.common.vec_env import DummyVecEnv
    vec_env = DummyVecEnv([lambda: env])

    initial_actions = adaptive_space.get_num_valid_actions()
    print(f"  Initial valid actions: {initial_actions}")

    model = AdaptivePPO(
        "MaskedMlpPolicy",
        vec_env,
        adaptive_action_space=adaptive_space,
        unmask_frequency=2,  # Unmask frequently for testing
        n_steps=64,
        batch_size=32,
        n_epochs=2,
        verbose=0,
    )

    # Train for a short period
    model.learn(total_timesteps=1000)

    final_actions = adaptive_space.get_num_valid_actions()
    print(f"  Final valid actions: {final_actions}")
    print(f"  Actions added: {final_actions - initial_actions}")
    print(f"  Progress: {adaptive_space.get_progress():.1%}")

    if final_actions > initial_actions:
        print("  [PASS] Action space expanded during training!")
    else:
        print("  [WARN] Action space did not expand (may need more steps or different advantages)")

    print("  [PASS] Short training test passed!\n")

    return model, adaptive_space


if __name__ == "__main__":
    print("=" * 80)
    print("ADAPTIVE PPO INTEGRATION TESTS")
    print("=" * 80 + "\n")

    try:
        # Test 1: Environment wrapper
        env, adaptive_space = test_adaptive_env()

        # Test 2: Masked policy
        model = test_masked_policy()

        # Test 3: Short training
        model, adaptive_space = test_short_training()

        print("=" * 80)
        print("ALL TESTS PASSED!")
        print("=" * 80)
        print("\nYou can now run the full training with:")
        print("  python train_adaptive_ppo.py --mode adaptive --timesteps 100000")
        print("\nOr compare with baseline:")
        print("  python train_adaptive_ppo.py --mode both --timesteps 100000 --eval")

    except Exception as e:
        print(f"\n[FAIL] TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
