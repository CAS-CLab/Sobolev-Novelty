"""Deterministic cross-generation Sobolev basis archive primitives."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .sn_population_coverage import (
    orthonormal_signature_span,
    signature_residual_gain,
)
from .sn_residual_infusion import centered_absolute_correlation


@dataclass(frozen=True)
class BasisArchiveEntry:
    canonical: str
    expression: str
    signature: np.ndarray
    values: np.ndarray
    term_norm: float
    source_coefficient: float
    source_amplitude: float
    source_base_reward: float
    source_generation: int
    source_base_rank: int
    source_candidate_id: int


@dataclass(frozen=True)
class BasisArchiveUpdate:
    previous_size: int
    new_size: int
    inserted: int
    replaced: int
    evicted: int
    canonical_reuses: int


@dataclass(frozen=True)
class ArchiveInfusionChoice:
    action: str
    removed_parent_index: int | None
    selected_archive_index: int
    selected_canonical: str
    selected_expression: str
    removed_deletion_impact: float | None
    removed_normalized_impact: float | None
    removed_novelty: float | None
    donor_target_correlation: float
    donor_residual_gain: float
    donor_joint_score: float
    donor_normalized_innovation: float
    donor_source_generation: int
    donor_source_base_rank: int
    donor_source_candidate_id: int
    donor_age: int
    donor_value_residual_gain: float


def partial_residual_credit(
    values: Sequence[float],
    residual: Sequence[float],
    design: np.ndarray,
) -> tuple[float, float]:
    """Return conditional residual correlation and remaining value-column norm."""

    value = np.asarray(values, dtype=float).reshape(-1)
    error = np.asarray(residual, dtype=float).reshape(-1)
    matrix = np.asarray(design, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != value.size or error.shape != value.shape:
        raise ValueError("partial residual inputs are not aligned")
    if not all(np.all(np.isfinite(item)) for item in (value, error, matrix)):
        raise ValueError("partial residual inputs must be finite")
    coefficients, *_ = np.linalg.lstsq(matrix, value, rcond=None)
    orthogonal = value - matrix @ coefficients
    orthogonal_norm = float(np.linalg.norm(orthogonal))
    centered_norm = float(np.linalg.norm(value - float(np.mean(value))))
    error_norm = float(np.linalg.norm(error))
    if orthogonal_norm <= np.finfo(float).eps or error_norm <= np.finfo(float).eps:
        return 0.0, 0.0
    correlation = abs(float(np.dot(orthogonal, error))) / (
        orthogonal_norm * error_norm
    )
    value_gain = orthogonal_norm / max(centered_norm, np.finfo(float).eps)
    return (
        max(0.0, min(1.0, correlation)),
        max(0.0, min(1.0, value_gain)),
    )


def _source_key(entry: BasisArchiveEntry) -> tuple[float, int, float, str]:
    return (
        -float(entry.source_base_reward),
        int(entry.source_base_rank),
        -float(entry.source_amplitude),
        str(entry.canonical),
    )


class SobolevBasisArchive:
    """Canonical-deduplicated archive with greedy Sobolev span coverage."""

    def __init__(self, capacity: int = 64) -> None:
        if capacity < 1:
            raise ValueError("basis archive capacity must be positive")
        self.capacity = int(capacity)
        self.entries: tuple[BasisArchiveEntry, ...] = ()

    def clear(self) -> None:
        self.entries = ()

    def update(
        self, additions: Sequence[BasisArchiveEntry]
    ) -> BasisArchiveUpdate:
        previous = {entry.canonical: entry for entry in self.entries}
        merged = dict(previous)
        inserted = 0
        replaced = 0
        reuses = 0
        for entry in additions:
            existing = merged.get(entry.canonical)
            if existing is None:
                merged[entry.canonical] = entry
                inserted += 1
            else:
                reuses += 1
                if _source_key(entry) < _source_key(existing):
                    merged[entry.canonical] = entry
                    replaced += 1
        pool = list(merged.values())
        retained = self._greedy_coverage(pool)
        self.entries = tuple(retained)
        return BasisArchiveUpdate(
            previous_size=len(previous),
            new_size=len(retained),
            inserted=inserted,
            replaced=replaced,
            evicted=max(0, len(pool) - len(retained)),
            canonical_reuses=reuses,
        )

    def _greedy_coverage(
        self, entries: Sequence[BasisArchiveEntry]
    ) -> list[BasisArchiveEntry]:
        if len(entries) <= self.capacity:
            return sorted(entries, key=_source_key)
        remaining = list(entries)
        first = min(remaining, key=_source_key)
        selected = [first]
        remaining.remove(first)
        first_vector = first.signature / max(
            float(np.linalg.norm(first.signature)), np.finfo(float).eps
        )
        basis = first_vector.reshape(-1, 1)
        while len(selected) < self.capacity:
            chosen = min(
                remaining,
                key=lambda entry: (
                    -signature_residual_gain(
                        basis,
                        entry.signature
                        / max(
                            float(np.linalg.norm(entry.signature)),
                            np.finfo(float).eps,
                        ),
                    ),
                    *_source_key(entry),
                ),
            )
            selected.append(chosen)
            remaining.remove(chosen)
            vector = chosen.signature / max(
                float(np.linalg.norm(chosen.signature)), np.finfo(float).eps
            )
            residual = vector - basis @ (basis.T @ vector)
            residual_norm = float(np.linalg.norm(residual))
            if residual_norm > np.finfo(float).eps * max(1, basis.shape[0]):
                basis = np.column_stack((basis, residual / residual_norm))
        return selected


class SourceQualityBasisArchive:
    """Canonical term reservoir ranked only by its strongest Base source."""

    def __init__(self, capacity: int = 64) -> None:
        if capacity < 1:
            raise ValueError("source-quality archive capacity must be positive")
        self.capacity = int(capacity)
        self.entries: tuple[BasisArchiveEntry, ...] = ()

    def clear(self) -> None:
        self.entries = ()

    def update(
        self, additions: Sequence[BasisArchiveEntry]
    ) -> BasisArchiveUpdate:
        previous = {entry.canonical: entry for entry in self.entries}
        merged = dict(previous)
        inserted = 0
        replaced = 0
        reuses = 0
        for entry in additions:
            existing = merged.get(entry.canonical)
            if existing is None:
                merged[entry.canonical] = entry
                inserted += 1
            else:
                reuses += 1
                if _source_key(entry) < _source_key(existing):
                    merged[entry.canonical] = entry
                    replaced += 1
        retained = sorted(merged.values(), key=_source_key)[: self.capacity]
        self.entries = tuple(retained)
        return BasisArchiveUpdate(
            previous_size=len(previous),
            new_size=len(retained),
            inserted=inserted,
            replaced=replaced,
            evicted=max(0, len(merged) - len(retained)),
            canonical_reuses=reuses,
        )


def select_archive_residual_infusion(
    *,
    parent_canonicals: Sequence[str],
    parent_coefficients: Sequence[float],
    parent_novelties: Sequence[float],
    parent_norms: Sequence[float],
    parent_signatures: np.ndarray,
    parent_structural: Sequence[bool],
    parent_residual: Sequence[float],
    archive_entries: Sequence[BasisArchiveEntry],
    tau: float,
    current_generation: int,
    parent_value_design: np.ndarray | None = None,
) -> ArchiveInfusionChoice | None:
    """Choose an archived basis using target residual × Sobolev complementarity."""

    count = len(parent_canonicals)
    coefficients = np.asarray(parent_coefficients, dtype=float)
    novelties = np.asarray(parent_novelties, dtype=float)
    norms = np.asarray(parent_norms, dtype=float)
    structural = np.asarray(parent_structural, dtype=bool)
    signatures = np.asarray(parent_signatures, dtype=float)
    residual = np.asarray(parent_residual, dtype=float).reshape(-1)
    if (
        count == 0
        or any(
            value.shape != (count,)
            for value in (coefficients, novelties, norms, structural)
        )
        or signatures.ndim != 2
        or signatures.shape[1] != count
        or not archive_entries
    ):
        raise ValueError("archive-infusion inputs are not aligned")
    if tau <= 0.0 or any(
        not np.all(np.isfinite(value))
        for value in (coefficients, novelties, norms, signatures, residual)
    ):
        raise ValueError("archive-infusion inputs must be finite and tau positive")

    parent_span = orthonormal_signature_span(
        [signatures[:, index] for index in range(count)]
    )
    identities = set(str(value) for value in parent_canonicals)
    candidates: list[tuple[float, float, float, float, str, int]] = []
    for index, entry in enumerate(archive_entries):
        if entry.canonical in identities:
            continue
        signature = np.asarray(entry.signature, dtype=float)
        values = np.asarray(entry.values, dtype=float).reshape(-1)
        signature_norm = float(np.linalg.norm(signature))
        if signature.shape != (signatures.shape[0],) or values.shape != residual.shape:
            raise ValueError("archive entry geometry is not aligned")
        if signature_norm <= np.finfo(float).eps:
            continue
        gain = signature_residual_gain(parent_span, signature / signature_norm)
        if parent_value_design is None:
            correlation = centered_absolute_correlation(values, residual)
            value_gain = 1.0
        else:
            correlation, value_gain = partial_residual_credit(
                values, residual, parent_value_design
            )
        joint = correlation * gain
        if joint <= np.finfo(float).eps:
            continue
        candidates.append(
            (joint, correlation, gain, value_gain, entry.canonical, index)
        )
    if not candidates:
        return None
    joint, correlation, gain, value_gain, _, selected = min(
        candidates,
        key=lambda value: (
            -value[0], -value[1], -value[2], -value[3], value[4], value[5]
        ),
    )

    amplitudes = np.abs(coefficients) * norms
    scale = float(np.sum(amplitudes))
    impacts = amplitudes * novelties
    removable = [
        index
        for index in range(count)
        if structural[index] and novelties[index] < tau
    ]
    removed: int | None = None
    removed_impact: float | None = None
    removed_relative: float | None = None
    removed_novelty: float | None = None
    if removable:
        candidate = min(
            removable,
            key=lambda index: (
                float(impacts[index]),
                float(novelties[index]),
                str(parent_canonicals[index]),
                index,
            ),
        )
        relative = float(impacts[candidate]) / max(scale, np.finfo(float).eps)
        if relative <= joint:
            removed = int(candidate)
            removed_impact = float(impacts[candidate])
            removed_relative = relative
            removed_novelty = float(novelties[candidate])

    entry = archive_entries[selected]
    return ArchiveInfusionChoice(
        action="exchange" if removed is not None else "augment",
        removed_parent_index=removed,
        selected_archive_index=int(selected),
        selected_canonical=entry.canonical,
        selected_expression=entry.expression,
        removed_deletion_impact=removed_impact,
        removed_normalized_impact=removed_relative,
        removed_novelty=removed_novelty,
        donor_target_correlation=float(correlation),
        donor_residual_gain=float(gain),
        donor_joint_score=float(joint),
        donor_normalized_innovation=float(joint),
        donor_source_generation=int(entry.source_generation),
        donor_source_base_rank=int(entry.source_base_rank),
        donor_source_candidate_id=int(entry.source_candidate_id),
        donor_age=max(0, int(current_generation) - int(entry.source_generation)),
        donor_value_residual_gain=float(value_gain),
    )
