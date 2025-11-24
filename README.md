# Highway PPO Research

Research project implementing adaptive action space discretization for PPO on highway-env.

## Project Structure

```
highway-ppo-research/
├── src/                                   # Source code
│   ├── adaptive_action_space.py           # 2D adaptive action space (steering × acceleration)
│   ├── adaptive_action_space_steering.py  # 1D adaptive action space with pruning (steering only)
│   ├── adaptive_ppo.py                    # Custom PPO with adaptive action masking
│   ├── masked_ppo_policy.py               # Policy with action masking support
│   ├── train_adaptive_ppo.py              # Training script for 2D adaptive PPO
│   ├── train_steering_adaptive_ppo.py     # Training script for steering-only adaptive PPO
│   ├── evaluate_saved_model.py            # Evaluation script for all model types
│   ├── test_integration.py                # Integration tests
│   ├── sb3_highway_ppo.py                 # Modified highway-env script (Windows compatible)
│   ├── utils.py                           # Modified highway-env utilities (Windows compatible)
│   └── ADAPTIVE_PPO_README.md             # Detailed documentation
├── trained_models/                        # Saved models (gitignored)
├── HighwayEnv/                            # Cloned highway-env repository (gitignored)
├── requirements.txt                       # Python dependencies
├── MODIFICATIONS.md                       # Documentation of changes to highway-env files
└── README.md                              # This file
```

## Overview

This project explores adaptive action space discretization for reinforcement learning, comparing:
- **Baseline**: Fixed uniform discretization (e.g., 5×5 or 5 steering angles)
- **Adaptive 2D**: Starts with sparse 5×5 grid, expands to 10×10 near high-advantage actions
- **Adaptive Steering with Pruning**: Maintains constant ~5 actions, zooms in on good steering angles while pruning bad ones

## Installation

```bash
# Create conda environment
conda create -n highway python=3.9
conda activate highway

# Install dependencies
pip install -r requirements.txt

# Clone highway-env (if not already present)
git clone https://github.com/eleurent/highway-env.git HighwayEnv
```

## Quick Start

### Training

```bash
cd src

# Train steering-only models (recommended)
python train_steering_adaptive_ppo.py --mode both --timesteps 100000

# Or train 2D models
python train_adaptive_ppo.py --mode both --timesteps 100000
```

### Evaluation

```bash
# Evaluate adaptive steering model
python evaluate_saved_model.py --model ../trained_models/highway_adaptive_steering_ppo/model --steering --episodes 100

# Evaluate baseline steering model
python evaluate_saved_model.py --model ../trained_models/highway_baseline_steering_ppo/model --baseline --steering --episodes 100

# With visualization
python evaluate_saved_model.py --model ../trained_models/highway_adaptive_steering_ppo/model --steering --episodes 5 --render
```

## Key Features

### Adaptive Action Space (2D)
- Starts with 25 actions (5×5 grid over steering × acceleration)
- Expands to ~85-100 actions during training
- Unmasks new actions near high-advantage regions
- Fixed neural network architecture (100 outputs throughout)

### Adaptive Steering with Pruning
- **1D action space**: Steering only (no acceleration control)
- **Constant capacity**: Maintains ~5 active actions
- **Zoom + Prune**: Adds actions near good steering angles, removes low-performing ones
- **20 total angles**: Finer resolution than baseline while maintaining same active count

### Action Masking
- Invalid actions are masked (probability = 0)
- Policy network outputs full space, masking applied at inference
- No architecture changes during training

## Results

See `src/ADAPTIVE_PPO_README.md` for detailed documentation and results.

## Model Types

| Model | Action Space | Initial Actions | Final Actions | Control |
|-------|--------------|-----------------|---------------|---------|
| Baseline 2D | 5×5 uniform | 25 | 25 | Steering + Accel |
| Adaptive 2D | 10×10 sparse | 25 | ~85-100 | Steering + Accel |
| Baseline Steering | 5 uniform | 5 | 5 | Steering only |
| Adaptive Steering | 20 sparse w/ pruning | 5 | ~5 | Steering only |

## Citation

```bibtex
@misc{highway_adaptive_ppo,
  title={Adaptive Action Space Discretization with Pruning for Highway-Env},
  year={2025},
  note={Research implementation}
}
```

## References

- [Stable Baselines3](https://github.com/DLR-RM/stable-baselines3)
- [Highway-Env](https://github.com/eleurent/highway-env)
- [PPO Paper](https://arxiv.org/abs/1707.06347)
