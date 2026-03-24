# Adaptive Discretization for Policy Optimization

Research project exploring theoretically-grounded adaptive action space discretization for policy gradient methods on MDPs, tested on highway-env.

## TODO:
- test new approaches/pipelines
- modify visualize python scripts (evaluate and compare) to include newly created implementations 

## Research Direction

Open Problems:
1. When to split cubes? Current method is a placeholder (line 113 of zooming.py)
2. How to partition action space? (bisimulation metrics?)
3. How to deal with lack of optimism (since we're doing policy gradient methods rather than Q-learning)?
4. Bisimulation metrics use SAC instead of PPO. This involves having one actor and two critics, which are Q-networks. They don't have UCB style optimism, but we can add a bonus to either the prior or Q values themselves (mean of two Q values + beta * std?) to ensure optimism. 

## Installation

```bash
conda create -n highway python=3.9
conda activate highway
pip install -r requirements.txt
git clone https://github.com/eleurent/highway-env.git HighwayEnv
```

## Environment Setup

All agents use `highway-fast-v0` with steering-only control (`longitudinal: False, lateral: True`). The PPO baseline uses `DiscreteAction` with 5 steering angles. The zooming variants use `ContinuousAction` with adaptive discretization of the 1D steering space.

## Papers
- Bonus for optimism paper (https://arxiv.org/abs/1806.03335)
- Bisimulation paper (https://arxiv.org/abs/2006.10742)
- Zooming paper (https://arxiv.org/abs/2006.10875)

## Reward Function

#### Racetrack

Environment: `racetrack-v0` with `ContinuousAction` (steering only, no acceleration).

**Per-step reward** (built-in, no custom wrapper):

```
lane_centering = 1 / (1 + 4 * lateral_offset²)   # 1.0 when perfectly centered, decays with offset
action_penalty = -0.3 * |steering|                 # penalizes large steering inputs
collision      = -1.0 * crashed                    # -1.0 on crash, else 0.0

raw = lane_centering + action_penalty + collision
reward = lmap(raw, [-1, 1], [0, 1]) * on_road
```

- `on_road` multiplier zeros out reward if the vehicle leaves the road
- Best case per step: **1.0** (centered, no steering, no crash)
- Worst case per step: **0.0** (crash or off-road)

**Episode length:**

- `duration=300` is in simulated seconds, not steps
- Agent acts at `policy_frequency=5` Hz → `self.time += 0.2` per step
- Max steps per episode: **300 / 0.2 = 1500**
- Theoretical max cumulative reward: **~1500**
- Episodes terminate early on crash or going off-road (`terminate_off_road=True`)
