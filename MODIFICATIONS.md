# Modifications to Highway-Env

This document lists changes made to files from the original highway-env repository for Windows compatibility.

## Modified Files from Highway-Env

### `sb3_highway_ppo.py`
**Original location**: `HighwayEnv/scripts/sb3_highway_ppo.py`
**Copied to**: `src/sb3_highway_ppo.py`

**Changes**:
- Changed `SubprocVecEnv` to `DummyVecEnv` for Windows compatibility
- Reduced `n_cpu` from 6 to 4
- `SubprocVecEnv` uses multiprocessing which has issues on Windows with highway-env serialization

**Diff**:
```python
# Before
from stable_baselines3.common.vec_env import SubprocVecEnv
n_cpu = 6
env = make_vec_env("highway-fast-v0", n_envs=n_cpu, vec_env_cls=SubprocVecEnv)

# After
from stable_baselines3.common.vec_env import DummyVecEnv
n_cpu = 4
env = make_vec_env("highway-fast-v0", n_envs=n_cpu, vec_env_cls=DummyVecEnv)
```

### `utils.py`
**Original location**: `HighwayEnv/scripts/utils.py`
**Copied to**: `src/utils.py`

**Changes**:
- Commented out `pyvirtualdisplay` import and initialization
- This package is Linux-only and not needed on Windows

**Diff**:
```python
# Before
from pyvirtualdisplay import Display
display = Display(visible=0, size=(1400, 900))
display.start()

# After
# from pyvirtualdisplay import Display
# display = Display(visible=0, size=(1400, 900))
# display.start()
```

## Custom Files (Created by Us)

These files are completely new and not modifications of highway-env:

- `adaptive_action_space.py` - 2D adaptive action space implementation
- `adaptive_action_space_steering.py` - 1D steering-only with pruning
- `adaptive_ppo.py` - Custom PPO algorithm with action masking
- `masked_ppo_policy.py` - Policy supporting action masks
- `train_adaptive_ppo.py` - Training script for 2D models
- `train_steering_adaptive_ppo.py` - Training script for steering models
- `evaluate_saved_model.py` - Evaluation script for all model types
- `test_integration.py` - Integration tests
- `ADAPTIVE_PPO_README.md` - Detailed documentation

## Original Highway-Env Repository

The unmodified highway-env repository is cloned in `HighwayEnv/` and is gitignored.

To get a clean version:
```bash
git clone https://github.com/eleurent/highway-env.git HighwayEnv
```

## Why These Modifications?

1. **Windows Compatibility**: `SubprocVecEnv` has serialization issues on Windows due to how Python's multiprocessing works differently on Windows vs Linux
2. **Virtual Display**: `pyvirtualdisplay` is Linux-only for headless rendering; Windows doesn't need it
