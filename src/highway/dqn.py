"""
Shared building blocks for the factored branching DQN (``dqn_factored.py``).

This module holds the action-selection policies (``EpsGreedy``, ``UCB``)
and two small net helpers (``weight_init``, ``soft_update``) that the
branching agent reuses.  The joint single-head ``DQN``/``QNetwork`` agent
that once lived here was dead code -- every experiment runs the factored
``BranchingDQN`` -- and has been removed.
"""

from __future__ import annotations

from typing import Optional, Protocol

import numpy as np
import torch.nn as nn


# ---------------------------------------------------------------------------
# Action-selection policies
# ---------------------------------------------------------------------------

class SelectionPolicy(Protocol):
    def select(self, q_values: np.ndarray, step: int,
               play_counts: np.ndarray,
               mask: Optional[np.ndarray] = None) -> int: ...


class EpsGreedy:
    def __init__(self, eps_start: float = 1.0, eps_end: float = 0.05,
                 decay_steps: int = 60_000):
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.decay_steps = decay_steps

    def select(self, q_values, step, play_counts, mask=None):
        frac = min(1.0, step / max(1, self.decay_steps))
        eps = self.eps_start + (self.eps_end - self.eps_start) * frac
        allowed = (np.arange(len(q_values)) if mask is None
                   else np.flatnonzero(mask))
        if np.random.random() < eps:
            return int(np.random.choice(allowed))
        q = q_values if mask is None else np.where(mask, q_values, -np.inf)
        return int(q.argmax())


class UCB:
    """Q(s, a) + c(step) * sqrt(log(step) / n(a)).

    The global term is ``log(step)`` (cumulative env steps -- monotonic,
    never reset by splits), and the denominator ``n(a)`` is the cell's own
    play count.  This is the per-cell + monotonic form the bandit-zooming
    theory uses; the earlier ``log(sum(play_counts))`` was non-monotonic
    because a split drops the parent's count and injects zero-count
    children, shrinking the numerator exactly when the action set grew.

    ``c`` can anneal from ``c_start`` to ``c_end`` over ``decay_steps``,
    but the intended setting is constant ``c`` (``c_start == c_end``): the
    bonus self-decays via 1/sqrt(n(a)) as plays accumulate.  Annealing was
    a symptomatic patch for freshly-split children spiking the bonus; with
    the monotonic count (and buffering, once added) it is unnecessary.
    """

    def __init__(self, c_start: float = 0.3, c_end: float = 0.03,
                 decay_steps: int = 60_000):
        self.c_start = c_start
        self.c_end = c_end
        self.decay_steps = decay_steps

    def _c(self, step: int) -> float:
        frac = min(1.0, step / max(1, self.decay_steps))
        return self.c_start + (self.c_end - self.c_start) * frac

    def select(self, q_values, step, play_counts, mask=None):
        c = self._c(step)
        # Monotonic global term: cumulative env steps (never reset by splits),
        # not sum(current per-cell play counts) -- the latter DROPS when a
        # split removes the parent's count and adds zero-count children, so
        # log(total) shrank exactly when the action set grew.  log(step) is
        # nondecreasing across split events.
        total = max(2, int(step))
        with np.errstate(divide="ignore"):
            bonus = c * np.sqrt(np.log(total) / np.maximum(1, play_counts))
        scores = q_values + bonus
        if mask is not None:
            # buffering cubes are not selectable until they graduate
            scores = np.where(mask, scores, -np.inf)
        return int(scores.argmax())


# ---------------------------------------------------------------------------
# Shared net helpers (used by the branching Q-net in dqn_factored.py)
# ---------------------------------------------------------------------------

def weight_init(m: nn.Module) -> None:
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        m.bias.data.fill_(0.0)


def soft_update(net: nn.Module, target: nn.Module, tau: float) -> None:
    for p, tp in zip(net.parameters(), target.parameters()):
        tp.data.copy_(tau * p.data + (1 - tau) * tp.data)
