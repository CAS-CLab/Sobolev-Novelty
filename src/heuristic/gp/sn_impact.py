"""Coefficient-aware term impact calculations for stable SN-GP mutation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np


DEFAULT_SN_MUTATION_IMPACT_BETA = 0.1
DEFAULT_SN_MUTATION_IMPACT_EPSILON = 1e-6
DEFAULT_SN_MUTATION_MAX_NORMALIZED_IMPACT = 0.25


@dataclass(frozen=True)
class TermImpactResult:
    coefficients: tuple[float, ...]
    novelties: tuple[float, ...]
    term_norms: tuple[float, ...]
    impacts: tuple[float, ...]
    normalized_impacts: tuple[float, ...]
    total_impact: float
    epsilon: float
    all_impacts_near_zero: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "coefficients": list(self.coefficients),
            "novelties": list(self.novelties),
            "term_norms": list(self.term_norms),
            "impacts": list(self.impacts),
            "normalized_impacts": list(self.normalized_impacts),
            "total_impact": self.total_impact,
            "epsilon": self.epsilon,
            "all_impacts_near_zero": self.all_impacts_near_zero,
        }


def compute_term_impacts(
    coefficients: Sequence[float],
    novelties: Sequence[float],
    term_norms: Sequence[float],
    epsilon: float,
) -> TermImpactResult:
    """Return ``D_i=|b_i| nu_i ||Psi_i||`` and normalized impacts.

    The displayed protocol uses ``sum(D)+epsilon`` as the denominator.  When
    the entire impact mass is no larger than ``epsilon`` the caller must fall
    back to the original random-site selector rather than amplify numerical
    dust.
    """

    if epsilon <= 0 or not np.isfinite(epsilon):
        raise ValueError("impact epsilon must be positive and finite")
    coefficient = np.asarray(coefficients, dtype=float)
    novelty = np.asarray(novelties, dtype=float)
    norms = np.asarray(term_norms, dtype=float)
    if coefficient.ndim != 1 or novelty.ndim != 1 or norms.ndim != 1:
        raise ValueError("term impact inputs must be one-dimensional")
    if not len(coefficient) or not (
        len(coefficient) == len(novelty) == len(norms)
    ):
        raise ValueError("term impact inputs must have equal non-zero length")
    if not (
        np.all(np.isfinite(coefficient))
        and np.all(np.isfinite(novelty))
        and np.all(np.isfinite(norms))
    ):
        raise ValueError("term impact inputs must be finite")
    if np.any(novelty < -1e-12) or np.any(novelty > 1.0 + 1e-12):
        raise ValueError("term novelty lies outside [0,1]")
    if np.any(norms < 0):
        raise ValueError("term norms must be non-negative")
    novelty = np.clip(novelty, 0.0, 1.0)
    with np.errstate(over="ignore", invalid="ignore"):
        impacts = np.abs(coefficient) * novelty * norms
    if not np.all(np.isfinite(impacts)):
        raise ValueError("term impact calculation produced non-finite values")
    total = float(np.sum(impacts, dtype=float))
    if not np.isfinite(total):
        raise ValueError("total term impact is non-finite")
    near_zero = total <= float(epsilon)
    normalized = (
        np.zeros_like(impacts)
        if near_zero
        else impacts / (total + float(epsilon))
    )
    return TermImpactResult(
        coefficients=tuple(float(value) for value in coefficient),
        novelties=tuple(float(value) for value in novelty),
        term_norms=tuple(float(value) for value in norms),
        impacts=tuple(float(value) for value in impacts),
        normalized_impacts=tuple(float(value) for value in normalized),
        total_impact=total,
        epsilon=float(epsilon),
        all_impacts_near_zero=near_zero,
    )
