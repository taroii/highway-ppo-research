from __future__ import annotations

import gymnasium as gym
import shimmy  # noqa: F401  -- registers dm_control/* in gymnasium
from gymnasium.wrappers import FlattenObservation


def make_dmcs_env(task: str) -> gym.Env:
    """DMCS task with Dict obs flattened to a single Box.

    Args:
        task: task slug like ``"cartpole-swingup"``, ``"walker-walk"``,
            ``"cheetah-run"``. The ``dm_control/`` prefix and ``-v0``
            suffix are added here.
    """
    return FlattenObservation(gym.make(f"dm_control/{task}-v0"))


def make_cartpole_swingup_env() -> gym.Env:
    return make_dmcs_env("cartpole-swingup")


def make_walker_walk_env() -> gym.Env:
    return make_dmcs_env("walker-walk")


def make_cheetah_run_env() -> gym.Env:
    return make_dmcs_env("cheetah-run")
