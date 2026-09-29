"""Target-residual credit for Sobolev-guided additive-basis infusion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .sn_population_coverage import (
    orthonormal_signature_span,
    signature_residual_gain,
)


@dataclass(frozen=True)
class ResidualInfusionChoice:
    action: str
    removed_parent_index: int | None
    selected_donor_index: int
    removed_deletion_impact: float | None
    removed_normalized_impact: float | None
    removed_novelty: float | None
    donor_target_correlation: float
    donor_residual_gain: float
    donor_joint_score: float
    donor_normalized_innovation: float


def centered_absolute_correlation(
    values: Sequence[float], residual: Sequence[float]
) -> float:
    """Return scale-free absolute correlation with a fitted target residual."""

    value = np.asarray(values, dtype=float).reshape(-1)
    error = np.asarray(residual, dtype=float).reshape(-1)
    if value.shape != error.shape or value.size == 0:
        raise ValueError("term values and target residual are not aligned")
    if not np.all(np.isfinite(value)) or not np.all(np.isfinite(error)):
        raise ValueError("term values and target residual must be finite")
    centered_value = value - float(np.mean(value))
    centered_error = error - float(np.mean(error))
    denominator = float(
        np.linalg.norm(centered_value) * np.linalg.norm(centered_error)
    )
    if denominator <= np.finfo(float).eps:
        return 0.0
    result = abs(float(np.dot(centered_value, centered_error))) / denominator
    return max(0.0, min(1.0, result))


def select_residual_basis_infusion(
    *,
    parent_canonicals: Sequence[str],
    parent_coefficients: Sequence[float],
    parent_novelties: Sequence[float],
    parent_norms: Sequence[float],
    parent_signatures: np.ndarray,
    parent_structural: Sequence[bool],
    donor_canonicals: Sequence[str],
    donor_coefficients: Sequence[float],
    donor_norms: Sequence[float],
    donor_signatures: np.ndarray,
    donor_structural: Sequence[bool],
    donor_target_correlations: Sequence[float],
    tau: float,
) -> ResidualInfusionChoice | None:
    """Choose a donor that is both target-relevant and Sobolev-complementary."""

    parent_count = len(parent_canonicals)
    donor_count = len(donor_canonicals)
    parent_coefficients_array = np.asarray(parent_coefficients, dtype=float)
    parent_novelties_array = np.asarray(parent_novelties, dtype=float)
    parent_norms_array = np.asarray(parent_norms, dtype=float)
    parent_mask = np.asarray(parent_structural, dtype=bool)
    donor_coefficients_array = np.asarray(donor_coefficients, dtype=float)
    donor_norms_array = np.asarray(donor_norms, dtype=float)
    donor_mask = np.asarray(donor_structural, dtype=bool)
    correlations = np.asarray(donor_target_correlations, dtype=float)
    parent_matrix = np.asarray(parent_signatures, dtype=float)
    donor_matrix = np.asarray(donor_signatures, dtype=float)
    if (
        parent_count == 0
        or donor_count == 0
        or any(
            value.shape != (parent_count,)
            for value in (
                parent_coefficients_array,
                parent_novelties_array,
                parent_norms_array,
                parent_mask,
            )
        )
        or any(
            value.shape != (donor_count,)
            for value in (
                donor_coefficients_array,
                donor_norms_array,
                donor_mask,
                correlations,
            )
        )
        or parent_matrix.ndim != 2
        or donor_matrix.ndim != 2
        or parent_matrix.shape[1] != parent_count
        or donor_matrix.shape[1] != donor_count
        or parent_matrix.shape[0] != donor_matrix.shape[0]
    ):
        raise ValueError("residual-infusion inputs are not aligned")
    arrays = (
        parent_coefficients_array,
        parent_novelties_array,
        parent_norms_array,
        donor_coefficients_array,
        donor_norms_array,
        correlations,
        parent_matrix,
        donor_matrix,
    )
    if tau <= 0.0 or any(not np.all(np.isfinite(value)) for value in arrays):
        raise ValueError("residual-infusion inputs must be finite and tau positive")

    parent_span = orthonormal_signature_span(
        [parent_matrix[:, index] for index in range(parent_count)]
    )
    parent_identities = set(str(value) for value in parent_canonicals)
    donor_amplitudes = np.abs(donor_coefficients_array) * donor_norms_array
    donor_scale = float(np.sum(donor_amplitudes))
    candidates: list[tuple[float, float, float, str, int]] = []
    for index in range(donor_count):
        if not donor_mask[index] or str(donor_canonicals[index]) in parent_identities:
            continue
        norm = float(np.linalg.norm(donor_matrix[:, index]))
        if norm <= np.finfo(float).eps:
            continue
        gain = signature_residual_gain(
            parent_span, donor_matrix[:, index] / norm
        )
        correlation = float(correlations[index])
        joint = correlation * gain
        if joint <= np.finfo(float).eps:
            continue
        candidates.append(
            (joint, correlation, gain, str(donor_canonicals[index]), index)
        )
    if not candidates:
        return None
    joint, correlation, gain, _, selected = min(
        candidates,
        key=lambda value: (-value[0], -value[1], -value[2], value[3], value[4]),
    )
    donor_relative = (
        float(donor_amplitudes[selected]) * gain
        / max(donor_scale, np.finfo(float).eps)
    )

    parent_amplitudes = np.abs(parent_coefficients_array) * parent_norms_array
    parent_scale = float(np.sum(parent_amplitudes))
    deletion_impacts = parent_amplitudes * parent_novelties_array
    removable = [
        index
        for index in range(parent_count)
        if parent_mask[index] and parent_novelties_array[index] < tau
    ]
    removed: int | None = None
    removed_impact: float | None = None
    removed_relative: float | None = None
    removed_novelty: float | None = None
    if removable:
        candidate_index = min(
            removable,
            key=lambda index: (
                float(deletion_impacts[index]),
                float(parent_novelties_array[index]),
                str(parent_canonicals[index]),
                index,
            ),
        )
        candidate_impact = float(deletion_impacts[candidate_index])
        candidate_relative = candidate_impact / max(
            parent_scale, np.finfo(float).eps
        )
        if candidate_relative <= donor_relative:
            removed = int(candidate_index)
            removed_impact = candidate_impact
            removed_relative = candidate_relative
            removed_novelty = float(parent_novelties_array[candidate_index])

    return ResidualInfusionChoice(
        action="exchange" if removed is not None else "augment",
        removed_parent_index=removed,
        selected_donor_index=int(selected),
        removed_deletion_impact=removed_impact,
        removed_normalized_impact=removed_relative,
        removed_novelty=removed_novelty,
        donor_target_correlation=float(correlation),
        donor_residual_gain=float(gain),
        donor_joint_score=float(joint),
        donor_normalized_innovation=float(donor_relative),
    )
