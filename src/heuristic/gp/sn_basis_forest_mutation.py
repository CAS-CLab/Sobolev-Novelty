"""Impact-aware mutation-gene selection for the additive basis forest."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .sn_impact import TermImpactResult, compute_term_impacts


@dataclass(frozen=True)
class BasisGeneMutationSelection:
    selected_index: int
    original_index: int
    targeted_distribution: bool
    selected_eligible: bool
    redundancy: tuple[float, ...]
    priorities: tuple[float, ...]
    weights: tuple[float, ...]
    eligibility: tuple[bool, ...]
    impacts: TermImpactResult | None
    fallback_reason: str | None = None


def select_impact_aware_basis_gene(
    *,
    coefficients: Sequence[float],
    novelties: Sequence[float],
    term_norms: Sequence[float],
    original_index: int,
    rng: np.random.Generator,
    tau: float,
    delta: float,
    gamma: float,
    impact_beta: float,
    impact_epsilon: float,
    max_normalized_impact: float,
) -> BasisGeneMutationSelection:
    """Redirect a consumed uniform gene choice without shifting operator RNG."""

    novelty = np.asarray(novelties, dtype=float)
    count = novelty.size
    if (
        novelty.ndim != 1
        or count == 0
        or original_index < 0
        or original_index >= count
        or not np.all(np.isfinite(novelty))
        or not 0.0 < tau <= 1.0
        or delta <= 0.0
        or gamma <= 0.0
        or impact_beta <= 0.0
        or not 0.0 <= max_normalized_impact <= 1.0
    ):
        raise ValueError("basis-gene mutation inputs are invalid")
    impact = compute_term_impacts(
        coefficients, novelty, term_norms, impact_epsilon
    )
    redundancy = np.maximum(0.0, 1.0 - novelty / float(tau))
    if impact.all_impacts_near_zero:
        return BasisGeneMutationSelection(
            selected_index=int(original_index),
            original_index=int(original_index),
            targeted_distribution=False,
            selected_eligible=False,
            redundancy=tuple(float(value) for value in redundancy),
            priorities=tuple(0.0 for _ in range(count)),
            weights=tuple(1.0 for _ in range(count)),
            eligibility=tuple(False for _ in range(count)),
            impacts=impact,
            fallback_reason="all_impacts_near_zero",
        )
    normalized = np.asarray(impact.normalized_impacts, dtype=float)
    eligibility = (novelty < float(tau)) & (
        normalized <= float(max_normalized_impact)
    )
    if not np.any(eligibility):
        return BasisGeneMutationSelection(
            selected_index=int(original_index),
            original_index=int(original_index),
            targeted_distribution=False,
            selected_eligible=False,
            redundancy=tuple(float(value) for value in redundancy),
            priorities=tuple(0.0 for _ in range(count)),
            weights=tuple(1.0 for _ in range(count)),
            eligibility=tuple(bool(value) for value in eligibility),
            impacts=impact,
            fallback_reason="no_eligible_gene",
        )
    priorities = np.zeros(count, dtype=float)
    priorities[eligibility] = redundancy[eligibility] / np.power(
        normalized[eligibility] + float(impact_epsilon),
        float(impact_beta),
    )
    weights = float(delta) + np.power(priorities, float(gamma))
    draw = float(rng.random()) * float(np.sum(weights))
    selected = min(
        int(np.searchsorted(np.cumsum(weights), draw, side="right")),
        count - 1,
    )
    return BasisGeneMutationSelection(
        selected_index=selected,
        original_index=int(original_index),
        targeted_distribution=True,
        selected_eligible=bool(eligibility[selected]),
        redundancy=tuple(float(value) for value in redundancy),
        priorities=tuple(float(value) for value in priorities),
        weights=tuple(float(value) for value in weights),
        eligibility=tuple(bool(value) for value in eligibility),
        impacts=impact,
    )
