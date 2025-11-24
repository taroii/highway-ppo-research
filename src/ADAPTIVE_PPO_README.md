# Adaptive Action Space PPO for Highway-Env

This implementation provides an adaptive action space discretization strategy for PPO on highway-env, where the action resolution gradually increases during training based on advantage estimates.

## Overview

The system implements a novel approach to action space discretization:

1. **Fixed high-dimensional output**: The neural network always outputs 100 actions (10x10 grid)
2. **Sparse initialization**: Only 25 actions are initially available (matching a 5x5 uniform grid)
3. **Advantage-driven expansion**: New actions are gradually unmasked near high-advantage actions
4. **Continuous refinement**: Action space expands throughout training without changing network architecture

## Key Components

### 1. `adaptive_action_space.py`

**AdaptiveActionSpace**: Manages the action masking and unmasking logic
- Maintains a 10x10 discrete grid over (steering, acceleration)
- Tracks which actions are currently valid (unmasked)
- Collects advantage statistics for each action
- Gradually unmasks neighboring actions of high-advantage actions

**AdaptiveDiscreteActionWrapper**: Gym wrapper that converts discrete action indices to continuous values
- Wraps highway-env configured with `ContinuousAction`
- Maps discrete action indices to continuous steering/acceleration values
- Provides action mask to the policy

### 2. `masked_ppo_policy.py`

**MaskedCategoricalDistribution**: Custom categorical distribution with action masking
- Sets masked action logits to `-inf` before softmax
- Ensures invalid actions have probability 0
- Supports all standard distribution operations (sample, log_prob, entropy)

**MaskedPPOPolicy**: PPO policy that respects action masks
- Extends `ActorCriticPolicy` from Stable Baselines3
- Passes action masks through forward() and evaluate_actions()
- Compatible with standard PPO training loop

### 3. `adaptive_ppo.py`

**AdaptiveRolloutBuffer**: Extended rollout buffer that stores action masks
- Inherits from SB3's RolloutBuffer
- Stores action masks alongside observations, actions, rewards, etc.

**AdaptivePPO**: Custom PPO algorithm with adaptive action space
- Extends SB3's PPO implementation
- Collects rollouts using current action mask
- Updates advantage statistics after each policy update
- Gradually unmasks new actions based on `unmask_frequency`

### 4. `train_adaptive_ppo.py`

Main training script with:
- **Adaptive PPO training**: With gradually expanding action space
- **Baseline PPO training**: Standard uniform 5x5 discretization for comparison
- **Evaluation**: Test trained models and compare performance
- **TensorBoard logging**: Track action space expansion and training metrics

### 5. `test_integration.py`

Integration tests verifying:
- Environment wrapper functionality
- Masked policy creation
- Short training run with action space expansion

## Configuration

### Adaptive Action Space Parameters

```python
AdaptiveActionSpace(
    steering_range=(-np.pi/4, np.pi/4),     # Steering angle range in radians
    acceleration_range=(-5.0, 5.0),          # Acceleration range in m/s^2
    final_grid_size=(10, 10),                # Final resolution (100 actions)
    initial_grid_size=(5, 5),                # Initial resolution (25 actions)
    unmask_rate=0.1,                         # Fraction of candidates to unmask per update
    advantage_threshold=0.0,                 # Min advantage to trigger unmasking
)
```

### Adaptive PPO Parameters

```python
AdaptivePPO(
    "MaskedMlpPolicy",
    env,
    adaptive_action_space=adaptive_space,
    unmask_frequency=10,                     # Unmask every N policy updates
    n_steps=128,                             # Steps per environment per update
    batch_size=64,                           # Minibatch size
    n_epochs=10,                             # Optimization epochs per update
    learning_rate=5e-4,
    gamma=0.9,
)
```

## Usage

### Quick Test

```bash
cd HighwayEnv/scripts
python test_integration.py
```

Expected output:
```
ALL TESTS PASSED!
[Adaptive] Valid actions: 85/100 (85.0%)
```

### Train Adaptive PPO

```bash
python train_adaptive_ppo.py --mode adaptive --timesteps 100000
```

### Train Baseline for Comparison

```bash
python train_adaptive_ppo.py --mode baseline --timesteps 100000
```

### Train Both and Evaluate

```bash
python train_adaptive_ppo.py --mode both --timesteps 100000 --eval
```

### Evaluate a Saved Model

After training, you can load and evaluate a saved model without retraining:

```bash
# Basic evaluation (10 episodes, no rendering)
python evaluate_saved_model.py --model highway_adaptive_ppo/model --episodes 10

# With visualization
python evaluate_saved_model.py --model highway_adaptive_ppo/model --episodes 5 --render

# Evaluate baseline model
python evaluate_saved_model.py --model highway_baseline_ppo/model --episodes 10 --baseline

# Use stochastic policy (for measuring variance)
python evaluate_saved_model.py --model highway_adaptive_ppo/model --episodes 20 --stochastic
```

**Evaluation Output:**
```
==============================================================
EVALUATION RESULTS
==============================================================
Mean Reward:       25.34 ± 3.21
Min/Max Reward:    18.50 / 31.20
Mean Length:       38.50 ± 2.15
Collision Rate:    10.0%
Success Rate:      80.0%
Unique Actions:        42

Action Usage:
  Unique actions used: 42
  Top 10 most frequent actions:
    Action  44: steering= 0.000, accel= 0.00 -  145 times (15.2%)
    Action  42: steering=-0.175, accel= 0.00 -  112 times (11.7%)
    Action  46: steering= 0.175, accel= 0.00 -  98 times (10.3%)
    ...
==============================================================
```

The evaluation script reports:
- **Episode statistics**: reward, length, crash/success rates
- **Action usage**: which of the 100 actions are actually used
- **Action distribution**: top actions with their steering/acceleration values

### Monitor Training with TensorBoard

```bash
tensorboard --logdir highway_adaptive_ppo
tensorboard --logdir highway_baseline_ppo
```

Key metrics to watch:
- `adaptive/num_valid_actions`: Number of unmasked actions over time
- `adaptive/progress`: Fraction of action space explored (0 to 1)
- `train/policy_gradient_loss`: PPO policy loss
- `rollout/ep_rew_mean`: Average episode reward

## How It Works

### Initial State (Step 0)
```
Action Space: 10x10 grid = 100 total actions
Valid Actions: 25 (5x5 uniform grid)
Neural Network Output: 100 logits

Mask Pattern (X = valid, . = masked):
X . X . X . X . X .
. . . . . . . . . .
X . X . X . X . X .
. . . . . . . . . .
X . X . X . X . X .
```

### After Training (e.g., Step 10k)
```
Valid Actions: ~85 (gradually unmasked)

Mask Pattern:
X X X X X X X X X .
X X X X X X X X X .
X X X X X X X X X .
X X X X X X X X X .
X X X X X X X X X .
... (actions added near high-advantage regions)
```

### Unmasking Algorithm

1. **Collect Rollouts**: Agent interacts with environment using current valid actions
2. **Compute Advantages**: PPO computes advantage estimates for taken actions
3. **Update Statistics**: Running average of advantages per action
4. **Identify High-Advantage Actions**: Actions with advantage > threshold
5. **Find Neighbors**: Get 8-connected neighbors of high-advantage actions
6. **Unmask Subset**: Randomly unmask `unmask_rate` fraction of masked neighbors
7. **Repeat**: Every `unmask_frequency` policy updates

## Advantages Over Standard Discretization

1. **Same initial L1 norm**: Starts with same number of actions as 5x5 baseline
2. **Adaptive resolution**: Adds resolution where policy needs it most
3. **No architecture changes**: Network size fixed throughout training
4. **Advantage-guided**: Uses policy's own value estimates to guide expansion
5. **Gradual exploration**: Smoothly transitions from coarse to fine control

## Comparison to Baseline

| Aspect | Baseline (5x5) | Adaptive (10x10) |
|--------|---------------|------------------|
| Initial actions | 25 | 25 |
| Final actions | 25 | ~85-100 |
| Network outputs | 25 | 100 |
| Resolution | Uniform | Adaptive |
| Exploration | Fixed | Progressive |

## Potential Extensions

1. **Variable expansion rate**: Adjust `unmask_rate` based on training progress
2. **Hierarchical unmasking**: Multiple levels of refinement
3. **Region-specific thresholds**: Different advantage thresholds for different areas
4. **Pruning**: Remove low-value actions after extended training
5. **Multi-resolution features**: Different feature extractors for different resolutions

## Troubleshooting

### Action space not expanding
- Check that advantages are being computed (non-zero)
- Reduce `advantage_threshold` to allow more unmasking
- Increase `unmask_rate` for faster expansion
- Decrease `unmask_frequency` to unmask more often

### Training unstable
- Reduce learning rate
- Increase batch size
- Decrease `unmask_rate` for slower expansion
- Use advantage normalization (enabled by default in PPO)

### Out of memory
- Reduce `final_grid_size` (e.g., to 8x8 = 64 actions)
- Decrease `n_envs` (number of parallel environments)
- Reduce batch size

## Files Created

- `adaptive_action_space.py`: Core action space management
- `masked_ppo_policy.py`: Policy with action masking support
- `adaptive_ppo.py`: Custom PPO algorithm
- `train_adaptive_ppo.py`: Training script
- `evaluate_saved_model.py`: Standalone evaluation script for saved models
- `test_integration.py`: Integration tests
- `ADAPTIVE_PPO_README.md`: This documentation

## References

- [Stable Baselines3](https://github.com/DLR-RM/stable-baselines3)
- [Highway-Env](https://github.com/eleurent/highway-env)
- [PPO Paper](https://arxiv.org/abs/1707.06347)

## Citation

If you use this code for research, please cite:

```bibtex
@misc{adaptive_ppo_highway,
  title={Adaptive Action Space Discretization for PPO on Highway-Env},
  year={2025},
  note={Research implementation}
}
```
