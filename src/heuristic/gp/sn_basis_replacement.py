"""Low-impact removal primitives for bidirectional Sobolev basis replacement."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class RemovalCandidate:
    parent_index: int
    canonical: str
    novelty: float
    deletion_impact: float


def rank_low_impact_parent_terms(
    *,
    parent_canonicals: Sequence[str],
    parent_coefficients: Sequence[float],
    parent_novelties: Sequence[float],
    parent_norms: Sequence[float],
    parent_structural: Sequence[bool],
    tau: float,
    shortlist_size: int,
) -> tuple[RemovalCandidate, ...]:
    """Return a deterministic shortlist of low-novelty, low-impact terms."""

    count = len(parent_canonicals)
    coefficients = np.asarray(parent_coefficients, dtype=float)
    novelties = np.asarray(parent_novelties, dtype=float)
    norms = np.asarray(parent_norms, dtype=float)
    structural = np.asarray(parent_structural, dtype=bool)
    if (
        count == 0
        or shortlist_size < 1
        or any(
            value.shape != (count,)
            for value in (coefficients, novelties, norms, structural)
        )
        or tau <= 0.0
        or not np.all(np.isfinite(coefficients))
        or not np.all(np.isfinite(novelties))
        or not np.all(np.isfinite(norms))
    ):
        raise ValueError("basis-replacement removal inputs are not aligned")
    impacts = np.abs(coefficients) * novelties * norms
    eligible = [
        index
        for index in range(count)
        if structural[index] and novelties[index] < tau
    ]
    eligible.sort(
        key=lambda index: (
            float(impacts[index]),
            float(novelties[index]),
            str(parent_canonicals[index]),
            index,
        )
    )
    return tuple(
        RemovalCandidate(
            parent_index=int(index),
            canonical=str(parent_canonicals[index]),
            novelty=float(novelties[index]),
            deletion_impact=float(impacts[index]),
        )
        for index in eligible[:shortlist_size]
    )
