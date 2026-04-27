"""
Shared bootstrap for the two clustered arms.

Loads a trained SAC checkpoint, rolls it out to collect observations,
and fits a Clusterer over its actor-trunk features.  Returns a
cluster_fn suitable for DQN.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.baseline.clustering import Clusterer, SACFeatureEncoder
from src.baseline.sac import SAC


def build_clusterer(
    sac_checkpoint: str,
    env,
    k: int,
    n_samples: int = 5_000,
    seed: int = 0,
):
    """Load SAC, roll out to collect obs, fit k-means, return (cluster_fn, clusterer)."""
    if not Path(sac_checkpoint).exists():
        raise FileNotFoundError(
            f"SAC checkpoint not found at {sac_checkpoint}. "
            f"Run src/baseline/run_sac.py first."
        )

    sac = SAC.load(sac_checkpoint, env)
    encoder = SACFeatureEncoder(sac)

    # Roll out the trained SAC policy to collect a representative obs distribution.
    print(f"Rolling out trained SAC for {n_samples} obs to fit k-means...")
    obs_buf = np.zeros((n_samples, sac.obs_dim), dtype=np.float32)
    obs, _ = env.reset(seed=seed)
    i = 0
    while i < n_samples:
        obs_flat = obs.flatten().astype(np.float32)
        obs_buf[i] = obs_flat
        action = sac.predict(obs_flat, deterministic=False)
        obs, _, done, truncated, _ = env.step(action)
        if done or truncated:
            obs, _ = env.reset()
        i += 1

    clusterer = Clusterer(encoder, k=k, seed=seed).fit_from_obs(obs_buf)
    # Report cluster sizes for sanity.
    labels = clusterer.km.predict(encoder.encode(obs_buf))
    counts = np.bincount(labels, minlength=k)
    print(f"Fitted k-means (k={k}). Cluster sizes on bootstrap obs: {counts.tolist()}")

    def cluster_fn(obs_flat: np.ndarray) -> int:
        return clusterer.cluster_of(obs_flat)

    return cluster_fn, clusterer
