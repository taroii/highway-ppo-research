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

Exact reward structure can be found in ```HighwayEnv/highway_env/envs/highway_env.py```. "The reward is defined to foster driving at high speed, on the rightmost lanes, and to avoid collisions." 

- collision_reward: 1.0 if crashed, 0.0 otherwise.
- right_lane_reward: lane_index / (num_lanes - 1)
- high_speed_reward: clip(lmap(forward_speed, [20, 30], [0, 1]), 0, 1)
- on_road_reward: 1.0 if on road, 0.0 otherwise.

Where forward_speed = speed * cos(heading).

The weighted sum is:

raw = -1.0 * crashed + 0.1 * right_lane + 0.4 * high_speed

It's then normalized so for each step, reward values are in [0, 1].

Additionally, a steering penalty is applied via `SteeringPenaltyWrapper` (defined in `src/ppo.py`):

```
penalty = 0.1 * |steering| / (pi/4)
reward = normalized_reward - penalty
```

This discourages unnecessary turning. With 25 discrete actions (5×5 grid of acceleration × steering), only 5 go straight — without this penalty the agent swerves constantly. The coefficient 0.1 means max steering costs 10% of the max reward per step, enough to encourage straight driving while still allowing lane changes to avoid collisions.

By default, there are 40 steps per episode (playthrough). Hence the max theoretical reward is 40, but in practice we'd have to get lucky to get in an environment where the car can speed through the right lane with no obstructions.
