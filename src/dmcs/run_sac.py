from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from src.dmcs.env import make_dmcs_env
from src.highway.sac import SAC


def main(task: str = "cartpole-swingup", seed: int = 42,
         total_timesteps: int = 300_000, output: str | None = None) -> dict:
    if output is None:
        output = f"checkpoints/dmcs/{task}/sac_seed{seed}.pt"

    env = make_dmcs_env(task)
    agent = SAC(
        env,
        hidden_dim=256,
        gamma=0.99,
        tau=0.005,
        actor_lr=3e-4,
        critic_lr=3e-4,
        alpha_lr=3e-4,
        batch_size=256,
        buffer_capacity=1_000_000,
        learning_starts=10_000,
        actor_update_freq=2,
        critic_target_update_freq=2,
        init_temperature=0.1,
        seed=seed,
    )

    print(f"SAC on dm_control/{task}  seed={seed}  steps={total_timesteps}")
    rewards = agent.learn(total_timesteps=total_timesteps, print_every=10_000)
    agent.save(output, rewards)

    print("\nEvaluating (deterministic)...")
    eval_rewards = []
    for _ in range(10):
        obs, _ = env.reset()
        ep = 0.0
        done = truncated = False
        while not (done or truncated):
            obs, r, done, truncated, _ = env.step(agent.predict(obs, deterministic=True))
            ep += r
        eval_rewards.append(ep)
    eval_mean, eval_std = float(np.mean(eval_rewards)), float(np.std(eval_rewards))
    print(f"Eval(10): mean={eval_mean:.2f}  std={eval_std:.2f}")
    env.close()
    return {"output": output, "eval_mean": eval_mean, "eval_std": eval_std,
            "episode_rewards": rewards}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--task", type=str, default="cartpole-swingup",
                   help="DMCS task slug: cartpole-swingup, walker-walk, cheetah-run.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--total_timesteps", type=int, default=300_000)
    p.add_argument("--output", type=str, default=None)
    args = p.parse_args()
    main(task=args.task, seed=args.seed,
         total_timesteps=args.total_timesteps, output=args.output)
