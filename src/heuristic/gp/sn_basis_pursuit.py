"""Sobolev screening primitives for exact Base-verified basis pursuit."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .sn_basis_archive import BasisArchiveEntry, partial_residual_credit
from .sn_population_coverage import (
    orthonormal_signature_span,
    signature_residual_gain,
)


@dataclass(frozen=True)
class ScreenedArchiveDonor:
    archive_index: int
    canonical: str
    expression: str
    sobolev_residual_gain: float
    partial_residual_correlation: float
    value_residual_gain: float
    heuristic_joint_score: float
    source_generation: int
    source_base_rank: int
    source_candidate_id: int
    donor_age: int


def screen_archive_basis_donors(
    *,
    parent_canonicals: Sequence[str],
    parent_signatures: np.ndarray,
    parent_residual: Sequence[float],
    parent_value_design: np.ndarray,
    archive_entries: Sequence[BasisArchiveEntry],
    current_generation: int,
    shortlist_size: int,
) -> tuple[ScreenedArchiveDonor, ...]:
    """Return a geometry-first donor shortlist with deterministic ties."""

    if shortlist_size < 1:
        raise ValueError("basis-pursuit shortlist size must be positive")
    signatures = np.asarray(parent_signatures, dtype=float)
    residual = np.asarray(parent_residual, dtype=float).reshape(-1)
    design = np.asarray(parent_value_design, dtype=float)
    parent_count = len(parent_canonicals)
    if (
        parent_count == 0
        or signatures.ndim != 2
        or signatures.shape[1] != parent_count
        or design.ndim != 2
        or design.shape[0] != residual.size
        or not np.all(np.isfinite(signatures))
        or not np.all(np.isfinite(residual))
        or not np.all(np.isfinite(design))
    ):
        raise ValueError("basis-pursuit screening inputs are not aligned and finite")

    parent_span = orthonormal_signature_span(
        [signatures[:, index] for index in range(parent_count)]
    )
    identities = {str(value) for value in parent_canonicals}
    candidates: list[ScreenedArchiveDonor] = []
    for index, entry in enumerate(archive_entries):
        if entry.canonical in identities:
            continue
        signature = np.asarray(entry.signature, dtype=float).reshape(-1)
        values = np.asarray(entry.values, dtype=float).reshape(-1)
        if signature.shape != (signatures.shape[0],) or values.shape != residual.shape:
            raise ValueError("archive entry geometry is not aligned")
        if not np.all(np.isfinite(signature)) or not np.all(np.isfinite(values)):
            raise ValueError("archive entry geometry must be finite")
        norm = float(np.linalg.norm(signature))
        if norm <= np.finfo(float).eps:
            continue
        gain = signature_residual_gain(parent_span, signature / norm)
        correlation, value_gain = partial_residual_credit(values, residual, design)
        if gain <= np.finfo(float).eps or correlation <= np.finfo(float).eps:
            continue
        candidates.append(
            ScreenedArchiveDonor(
                archive_index=int(index),
                canonical=entry.canonical,
                expression=entry.expression,
                sobolev_residual_gain=float(gain),
                partial_residual_correlation=float(correlation),
                value_residual_gain=float(value_gain),
                heuristic_joint_score=float(gain * correlation),
                source_generation=int(entry.source_generation),
                source_base_rank=int(entry.source_base_rank),
                source_candidate_id=int(entry.source_candidate_id),
                donor_age=max(
                    0, int(current_generation) - int(entry.source_generation)
                ),
            )
        )
    candidates.sort(
        key=lambda value: (
            -value.sobolev_residual_gain,
            -value.partial_residual_correlation,
            -value.value_residual_gain,
            value.canonical,
            value.archive_index,
        )
    )
    return tuple(candidates[:shortlist_size])


def select_low_impact_parent_term(
    *,
    parent_coefficients: Sequence[float],
    parent_novelties: Sequence[float],
    parent_norms: Sequence[float],
    parent_structural: Sequence[bool],
    tau: float,
) -> int | None:
    """Select one coefficient-aware redundant term for optional exchange."""

    coefficients = np.asarray(parent_coefficients, dtype=float)
    novelties = np.asarray(parent_novelties, dtype=float)
    norms = np.asarray(parent_norms, dtype=float)
    structural = np.asarray(parent_structural, dtype=bool)
    count = len(coefficients)
    if (
        count == 0
        or any(value.shape != (count,) for value in (novelties, norms, structural))
        or tau <= 0.0
        or not np.all(np.isfinite(coefficients))
        or not np.all(np.isfinite(novelties))
        or not np.all(np.isfinite(norms))
    ):
        raise ValueError("parent term-impact inputs are not aligned and finite")
    eligible = np.flatnonzero(structural & (novelties < tau))
    if not len(eligible):
        return None
    impacts = np.abs(coefficients) * novelties * norms
    return int(min(eligible, key=lambda index: (float(impacts[index]), int(index))))
