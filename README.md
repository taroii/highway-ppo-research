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