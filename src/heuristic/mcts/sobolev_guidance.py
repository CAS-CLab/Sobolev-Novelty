"""Small MCTS-facing helpers for candidate-level Sobolev guidance."""

from __future__ import annotations

import hashlib
from typing import Sequence

import numpy as np


def sobolev_penalty(term_novelties: Sequence[float], threshold: float) -> float:
    """Return mean squared hinge ``max(0, 1-nu/tau)^2`` in ``[0, 1]``."""

    novelty = np.asarray(term_novelties, dtype=float)
    if novelty.ndim != 1 or novelty.size == 0 or not np.all(np.isfinite(novelty)):
        raise ValueError("term_novelties must be a non-empty finite vector")
    if not 0 < threshold <= 1:
        raise ValueError("threshold must be in (0, 1]")
    if np.any(novelty < -1e-12) or np.any(novelty > 1 + 1e-12):
        raise ValueError("term novelty lies outside [0,1]")
    hinge = np.maximum(0.0, 1.0 - np.clip(novelty, 0.0, 1.0) / threshold)
    return float(np.mean(np.square(hinge)))


def stable_state_id(raw_state: str) -> str:
    """Return a stable short identifier without changing MCTS state semantics."""

    return hashlib.sha256(raw_state.encode("utf-8")).hexdigest()[:16]
