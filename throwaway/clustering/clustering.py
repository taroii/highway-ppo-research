"""
State embedding + clustering for the clustered arms.

Pretrain SAC once, freeze it, then use its actor trunk as a state
embedding.  k-means over those embeddings gives us the cluster id that
routes each observation to its own per-cluster action manager.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from src.baseline.sac import SAC


class KMeans:
    """Minimal numpy k-means (Lloyd's)."""

    def __init__(self, k: int, max_iter: int = 100, seed: int = 0):
        self.k = k
        self.max_iter = max_iter
        self.rng = np.random.default_rng(seed)
        self.centers: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> "KMeans":
        n = X.shape[0]
        init = self.rng.choice(n, self.k, replace=False)
        self.centers = X[init].copy()
        for _ in range(self.max_iter):
            labels = self.predict(X)
            new_centers = np.stack([
                X[labels == c].mean(0) if (labels == c).any() else self.centers[c]
                for c in range(self.k)
            ])
            if np.allclose(new_centers, self.centers):
                break
            self.centers = new_centers
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        diff = X[:, None, :] - self.centers[None, :, :]
        return (diff * diff).sum(-1).argmin(1)


class SACFeatureEncoder:
    """Wraps a trained SAC actor and exposes its trunk features."""

    def __init__(self, sac: SAC):
        self.sac = sac
        self.sac.actor.eval()

    @torch.no_grad()
    def encode(self, obs_batch: np.ndarray) -> np.ndarray:
        if obs_batch.ndim == 1:
            obs_batch = obs_batch[None, :]
        x = torch.as_tensor(obs_batch, dtype=torch.float32)
        feats = self.sac.actor.trunk(x)
        return feats.cpu().numpy()

    @property
    def feature_dim(self) -> int:
        return self.sac.hidden_dim


class Clusterer:
    """Fit k-means over SAC-trunk features, then assign cluster ids."""

    def __init__(self, encoder: SACFeatureEncoder, k: int, seed: int = 0):
        self.encoder = encoder
        self.k = k
        self.km = KMeans(k=k, seed=seed)

    def fit_from_obs(self, obs: np.ndarray) -> "Clusterer":
        feats = self.encoder.encode(obs)
        self.km.fit(feats)
        return self

    def cluster_of(self, obs: np.ndarray) -> int:
        if obs.ndim == 1:
            obs = obs[None, :]
        feats = self.encoder.encode(obs.astype(np.float32))
        return int(self.km.predict(feats)[0])

    def clusters_of(self, obs_batch: np.ndarray) -> np.ndarray:
        feats = self.encoder.encode(obs_batch.astype(np.float32))
        return self.km.predict(feats).astype(np.int64)
