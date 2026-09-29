"""Adaptive Sobolev-guided additive-basis infusion primitives."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .sn_population_coverage import (
    orthonormal_signature_span,
    signature_residual_gain,
)


@dataclass(frozen=True)
class BasisInfusionChoice:
    action: str
    removed_parent_index: int | None
    selected_donor_index: int
    removed_deletion_impact: float | None
    removed_normalized_impact: float | None
    removed_novelty: float | None
    donor_residual_gain: float
    donor_amplitude: float
    donor_score: float
    donor_normalized_innovation: float


def select_basis_infusion(
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
    tau: float,
) -> BasisInfusionChoice | None:
    """Choose a complementary donor basis and exchange only when deletion is safer."""

    parent_count = len(parent_canonicals)
    donor_count = len(donor_canonicals)
    parent_coefficient_array = np.asarray(parent_coefficients, dtype=float)
    parent_novelty_array = np.asarray(parent_novelties, dtype=float)
    parent_norm_array = np.asarray(parent_norms, dtype=float)
    parent_mask = np.asarray(parent_structural, dtype=bool)
    donor_coefficient_array = np.asarray(donor_coefficients, dtype=float)
    donor_norm_array = np.asarray(donor_norms, dtype=float)
    donor_mask = np.asarray(donor_structural, dtype=bool)
    parent_matrix = np.asarray(parent_signatures, dtype=float)
    donor_matrix = np.asarray(donor_signatures, dtype=float)
    if (
        parent_count == 0
        or donor_count == 0
        or any(
            value.shape != (parent_count,)
            for value in (
                parent_coefficient_array,
                parent_novelty_array,
                parent_norm_array,
                parent_mask,
            )
        )
        or any(
            value.shape != (donor_count,)
            for value in (donor_coefficient_array, donor_norm_array, donor_mask)
        )
        or parent_matrix.ndim != 2
        or donor_matrix.ndim != 2
        or parent_matrix.shape[1] != parent_count
        or donor_matrix.shape[1] != donor_count
        or parent_matrix.shape[0] != donor_matrix.shape[0]
    ):
        raise ValueError("basis-infusion inputs are not aligned")
    if (
        tau <= 0.0
        or not np.all(np.isfinite(parent_matrix))
        or not np.all(np.isfinite(donor_matrix))
        or not np.all(np.isfinite(parent_coefficient_array))
        or not np.all(np.isfinite(parent_novelty_array))
        or not np.all(np.isfinite(parent_norm_array))
        or not np.all(np.isfinite(donor_coefficient_array))
        or not np.all(np.isfinite(donor_norm_array))
    ):
        raise ValueError("basis-infusion inputs must be finite and tau positive")

    parent_span = orthonormal_signature_span(
        [parent_matrix[:, index] for index in range(parent_count)]
    )
    parent_identities = set(str(value) for value in parent_canonicals)
    donor_amplitudes = np.abs(donor_coefficient_array) * donor_norm_array
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
        amplitude = float(donor_amplitudes[index])
        score = amplitude * gain
        if score <= np.finfo(float).eps:
            continue
        candidates.append(
            (score, gain, amplitude, str(donor_canonicals[index]), index)
        )
    if not candidates:
        return None
    score, gain, amplitude, _, selected = min(
        candidates,
        key=lambda value: (-value[0], -value[1], -value[2], value[3], value[4]),
    )
    donor_relative = score / max(donor_scale, np.finfo(float).eps)

    parent_amplitudes = np.abs(parent_coefficient_array) * parent_norm_array
    parent_scale = float(np.sum(parent_amplitudes))
    deletion_impacts = parent_amplitudes * parent_novelty_array
    removable = [
        index
        for index in range(parent_count)
        if parent_mask[index] and parent_novelty_array[index] < tau
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
                float(parent_novelty_array[index]),
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
            removed_novelty = float(parent_novelty_array[candidate_index])

    return BasisInfusionChoice(
        action="exchange" if removed is not None else "augment",
        removed_parent_index=removed,
        selected_donor_index=int(selected),
        removed_deletion_impact=removed_impact,
        removed_normalized_impact=removed_relative,
        removed_novelty=removed_novelty,
        donor_residual_gain=float(gain),
        donor_amplitude=float(amplitude),
        donor_score=float(score),
        donor_normalized_innovation=float(donor_relative),
    )
