# Adaptive Discretization for Policy Optimization

Research project exploring theoretically-grounded adaptive action space discretization for policy gradient methods on MDPs, tested on highway-env.

## Research Direction

The main change from standard zooming is replacing implicit argmax-Q action selection with an explicit policy distribution over cubes. Given a state, you sample a cube from a learned distribution over relevant cubes, then execute an action from that cube. The policy is updated via standard policy gradients using advantages, where the advantage of a cube is its Q-value minus the policy-weighted average Q over relevant cubes.

The core challenge is determining when to split cubes. Visit counts alone don't make sense in a policy gradient setting because the policy might intentionally avoid certain cubes. The natural replacement is splitting based on advantage variance within a cube: high variance means the cube contains state-action pairs the policy should distinguish. But this needs theoretical justification.

A secondary challenge is non-stationarity. The policy changes continuously, which changes which cubes get visited, which changes the value estimates, which changes the splitting decisions. Standard zooming doesn't have this issue because it explores uniformly via optimism. We need some argument that the policy and partition co-evolve in a stable way.

The third challenge is losing the optimism-based exploration guarantee. In zooming, optimistic Q-values ensure every region gets explored. With a learned policy, we need another mechanism, either entropy regularization, explicit exploration bonuses, or trusting that advantage-driven splitting will naturally refine promising regions.

## Installation

```bash
conda create -n highway python=3.9
conda activate highway
pip install -r requirements.txt
git clone https://github.com/eleurent/highway-env.git HighwayEnv
```

## Environment Setup

All agents use `highway-fast-v0` with steering-only control (`longitudinal: False, lateral: True`). The PPO baseline uses `DiscreteAction` with 5 steering angles. The zooming variants use `ContinuousAction` with adaptive discretization of the 1D steering space.

## Reward Structure

We override the default highway-env reward with a custom formulation via `CustomRewardWrapper` (defined in `src/ppo.py`). This mimics the default reward function found here: ```HighwayEnv/highway_env/envs/highway_env.py```. 

```
raw = -1.0 * crashed + 0.4 * speed + 0.1 * right_lane + 0.1 * progress - 0.1 * steering
reward = lmap(raw, [-1.1, 0.6], [0, 1]) * on_road
```

| Component    | Definition                                  | Range  |
|-------------|---------------------------------------------|--------|
| crashed     | 1 if crashed, 0 otherwise                   | {0, 1} |
| speed       | clip((vehicle.speed - 20) / 10, 0, 1)       | [0, 1] |
| right_lane  | lane_index / (num_lanes - 1)                 | [0, 1] |
| progress    | clip(delta_x / 30, 0, 1)                     | [0, 1] |
| steering    | \|steering\| / (pi/4)                        | [0, 1] |
| on_road     | 1 if on road, 0 otherwise                   | {0, 1} |

`progress` is the change in x-position per step. At 30 m/s going straight, delta_x ≈ 30m per 1s step → progress = 1. Going sideways or backwards gives 0.

The raw score ranges from -1.1 (crash) to 0.6 (max speed, rightmost lane, full progress, no steering). After normalization, reward is in [0, 1] per step. Off-road or crashed gives 0.

All three agents (PPO, ZoomingPPO, ContextualZoomingPPO) use steering-only control (`longitudinal: False, lateral: True`), so the agent controls only the steering angle while speed is managed automatically.

By default, there are 40 steps per episode (playthrough). Hence the max theoretical reward is 40, but in practice we'd have to get lucky to get in an environment where the car can speed through the right lane with no obstructions.
