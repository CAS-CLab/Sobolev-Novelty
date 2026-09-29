"""Sobolev-guided additive-basis exchange primitives for Population GP."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .sn_population_coverage import (
    orthonormal_signature_span,
    signature_residual_gain,
)


@dataclass(frozen=True)
class BasisExchangeChoice:
    removed_parent_index: int
    selected_donor_index: int
    removed_deletion_impact: float
    removed_novelty: float
    donor_residual_gain: float
    donor_amplitude: float
    donor_score: float


def select_basis_exchange(
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
) -> BasisExchangeChoice | None:
    """Choose one low-impact deletion and one complementary donor basis."""

    parent_count = len(parent_canonicals)
    donor_count = len(donor_canonicals)
    parent_arrays = (
        np.asarray(parent_coefficients, dtype=float),
        np.asarray(parent_novelties, dtype=float),
        np.asarray(parent_norms, dtype=float),
        np.asarray(parent_structural, dtype=bool),
    )
    donor_arrays = (
        np.asarray(donor_coefficients, dtype=float),
        np.asarray(donor_norms, dtype=float),
        np.asarray(donor_structural, dtype=bool),
    )
    parent_matrix = np.asarray(parent_signatures, dtype=float)
    donor_matrix = np.asarray(donor_signatures, dtype=float)
    if (
        parent_count == 0
        or donor_count == 0
        or any(array.shape != (parent_count,) for array in parent_arrays)
        or any(array.shape != (donor_count,) for array in donor_arrays)
        or parent_matrix.ndim != 2
        or donor_matrix.ndim != 2
        or parent_matrix.shape[1] != parent_count
        or donor_matrix.shape[1] != donor_count
        or parent_matrix.shape[0] != donor_matrix.shape[0]
    ):
        raise ValueError("basis-exchange inputs are not aligned")
    if (
        tau <= 0.0
        or not np.all(np.isfinite(parent_matrix))
        or not np.all(np.isfinite(donor_matrix))
        or any(not np.all(np.isfinite(array)) for array in parent_arrays[:3])
        or any(not np.all(np.isfinite(array)) for array in donor_arrays[:2])
    ):
        raise ValueError("basis-exchange inputs must be finite and tau positive")

    parent_coefficients_array, parent_novelty_array, parent_norm_array, parent_mask = (
        parent_arrays
    )
    deletion_impacts = (
        np.abs(parent_coefficients_array)
        * parent_novelty_array
        * parent_norm_array
    )
    removable = [
        index
        for index in range(parent_count)
        if parent_mask[index] and parent_novelty_array[index] < tau
    ]
    if not removable:
        return None
    removed = min(
        removable,
        key=lambda index: (
            float(deletion_impacts[index]),
            float(parent_novelty_array[index]),
            str(parent_canonicals[index]),
            index,
        ),
    )
    retained = [index for index in range(parent_count) if index != removed]
    if not retained:
        return None
    retained_span = orthonormal_signature_span(
        [parent_matrix[:, index] for index in retained]
    )
    parent_identities = set(str(value) for value in parent_canonicals)
    donor_coefficient_array, donor_norm_array, donor_mask = donor_arrays
    candidates: list[tuple[float, float, float, str, int]] = []
    for index in range(donor_count):
        if not donor_mask[index] or str(donor_canonicals[index]) in parent_identities:
            continue
        norm = float(np.linalg.norm(donor_matrix[:, index]))
        if norm <= np.finfo(float).eps:
            continue
        normalized = donor_matrix[:, index] / norm
        gain = signature_residual_gain(retained_span, normalized)
        amplitude = abs(float(donor_coefficient_array[index])) * float(
            donor_norm_array[index]
        )
        score = amplitude * gain
        candidates.append(
            (score, gain, amplitude, str(donor_canonicals[index]), index)
        )
    if not candidates:
        return None
    score, gain, amplitude, _, selected = min(
        candidates,
        key=lambda value: (-value[0], -value[1], -value[2], value[3], value[4]),
    )
    return BasisExchangeChoice(
        removed_parent_index=int(removed),
        selected_donor_index=int(selected),
        removed_deletion_impact=float(deletion_impacts[removed]),
        removed_novelty=float(parent_novelty_array[removed]),
        donor_residual_gain=float(gain),
        donor_amplitude=float(amplitude),
        donor_score=float(score),
    )
