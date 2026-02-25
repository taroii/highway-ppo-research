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

## Reward Structure

The environment uses `highway-fast-v0` with `duration=30` and `policy_frequency=1`, giving **30 steps per episode**.

Per-step reward is normalized to **[0, 1]**:
- Raw reward: `collision_reward * crashed + high_speed_reward * scaled_speed + right_lane_reward * lane_fraction`
- Defaults: `-1 * crashed + 0.4 * speed_score + 0.1 * lane_score`
- Normalized via linear map from `[-1, 0.5]` to `[0, 1]`, then multiplied by `on_road_reward` (1 if on road, 0 if not)

**Theoretical max reward per episode: 30.0** (full speed, rightmost lane, no collision, all 30 steps).

## Tree-Based Adaptive Discretization


