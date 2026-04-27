"""Shared env factory for the baseline arms."""

from __future__ import annotations

import gymnasium as gym

try:
    import highway_env  # noqa: F401
except ImportError:
    pass


def make_racetrack_env() -> gym.Env:
    """racetrack-v0 with continuous lateral (steering) action only."""
    return gym.make(
        "racetrack-v0",
        config={
            "action": {
                "type": "ContinuousAction",
                "longitudinal": False,
                "lateral": True,
            },
        },
    )
