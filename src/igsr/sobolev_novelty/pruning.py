"""Deterministic Sobolev-guided compression for fitted IGSR term sets.

The mathematics remains delegated to the EIC Sobolev evaluator.  This module
only turns its position-aligned diagnostics into an auditable order in which
IGSR may *try* removing terms.  Whether a removal is accepted is decided by a
fresh train fit and search-validation NMSE in :mod:`igsr.method.igsr`.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class SobolevPruneCandidate:
    """One term eligible for a validation-guarded compression attempt."""

    index: int
    term: str
    coefficient: float
    novelty: float | None
    signature_norm: float | None
    contribution_score: float
    eligibility_reason: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def rank_prune_candidates(
    terms: Sequence[str],
    diagnostics: Mapping[str, Any] | None,
    *,
    threshold: float | None = None,
) -> list[SobolevPruneCandidate]:
    """Rank low-novelty/zero-weight terms by coefficient-aware contribution.

    For an active term ``i`` the score is

    ``abs(w_i) * novelty_i * ||Psi_i||``.

    Zero-coefficient terms are absent from the fitted expression and receive
    score zero.  Active terms are eligible only below the canonical Sobolev
    threshold.  Ties are resolved by their original term position, making the
    action fully deterministic.
    """

    if not isinstance(diagnostics, Mapping) or diagnostics.get("success") is not True:
        return []
    coefficients = diagnostics.get("coefficients")
    novelties = diagnostics.get("term_novelties")
    active_indices = diagnostics.get("active_term_indices")
    zero_indices = diagnostics.get("zero_coefficient_indices")
    eic_result = diagnostics.get("eic_result")
    if (
        not isinstance(coefficients, list)
        or len(coefficients) != len(terms)
        or not isinstance(novelties, list)
        or len(novelties) != len(terms)
        or not isinstance(active_indices, list)
        or not isinstance(zero_indices, list)
        or not isinstance(eic_result, Mapping)
    ):
        return []

    raw_norms = eic_result.get("term_norms")
    if not isinstance(raw_norms, list) or len(raw_norms) < len(active_indices):
        return []
    norm_by_index: dict[int, float] = {}
    try:
        for index, raw_norm in zip(active_indices, raw_norms, strict=False):
            parsed_index = int(index)
            parsed_norm = float(raw_norm)
            if not math.isfinite(parsed_norm) or parsed_norm < 0.0:
                return []
            norm_by_index[parsed_index] = parsed_norm
        coefficient_values = [float(value) for value in coefficients]
    except (TypeError, ValueError, OverflowError):
        return []
    if any(not math.isfinite(value) for value in coefficient_values):
        return []

    effective_threshold = diagnostics.get("threshold") if threshold is None else threshold
    try:
        effective_threshold = float(effective_threshold)
    except (TypeError, ValueError, OverflowError):
        return []
    if not math.isfinite(effective_threshold) or effective_threshold < 0.0:
        return []

    active = {int(index) for index in active_indices}
    zeros = {int(index) for index in zero_indices}
    if active & zeros or active | zeros != set(range(len(terms))):
        return []

    ranked: list[SobolevPruneCandidate] = []
    for index, term in enumerate(terms):
        coefficient = coefficient_values[index]
        novelty_value = novelties[index]
        if index in zeros:
            ranked.append(
                SobolevPruneCandidate(
                    index=index,
                    term=str(term),
                    coefficient=coefficient,
                    novelty=None,
                    signature_norm=None,
                    contribution_score=0.0,
                    eligibility_reason="zero_fitted_coefficient",
                )
            )
            continue
        try:
            novelty = float(novelty_value)
        except (TypeError, ValueError, OverflowError):
            continue
        signature_norm = norm_by_index.get(index)
        if (
            signature_norm is None
            or not math.isfinite(novelty)
            or novelty < 0.0
            or novelty >= effective_threshold
        ):
            continue
        score = abs(coefficient) * novelty * signature_norm
        if not math.isfinite(score):
            continue
        ranked.append(
            SobolevPruneCandidate(
                index=index,
                term=str(term),
                coefficient=coefficient,
                novelty=novelty,
                signature_norm=signature_norm,
                contribution_score=score,
                eligibility_reason="novelty_below_threshold",
            )
        )
    return sorted(ranked, key=lambda candidate: (candidate.contribution_score, candidate.index))
