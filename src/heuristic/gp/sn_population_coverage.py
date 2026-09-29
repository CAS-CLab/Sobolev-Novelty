"""Small, deterministic Sobolev-geometry survivor selection primitives."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class CoverageChoice:
    """Result of choosing one candidate against an existing signature span."""

    selected_index: int
    selected_gain: float
    gains: tuple[float, ...]
    reference_rank: int


def normalized_fitted_signature(
    term_signatures: np.ndarray,
    coefficients: Sequence[float],
) -> np.ndarray:
    """Combine coefficient-free term columns into one normalized phenotype vector."""

    matrix = np.asarray(term_signatures, dtype=float)
    weights = np.asarray(coefficients, dtype=float).reshape(-1)
    if matrix.ndim != 2 or matrix.shape[1] != weights.size or weights.size == 0:
        raise ValueError("term signatures and coefficients are not aligned")
    if not np.all(np.isfinite(matrix)) or not np.all(np.isfinite(weights)):
        raise ValueError("coverage inputs must be finite")
    vector = matrix @ weights
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm <= np.finfo(float).eps:
        raise ValueError("fitted Sobolev signature is zero or non-finite")
    return vector / norm


def orthonormal_signature_span(vectors: Sequence[np.ndarray]) -> np.ndarray:
    """Return a numerical orthonormal basis for normalized signature vectors."""

    if not vectors:
        raise ValueError("coverage requires at least one anchor signature")
    matrix = np.column_stack([np.asarray(value, dtype=float) for value in vectors])
    if matrix.ndim != 2 or not np.all(np.isfinite(matrix)):
        raise ValueError("anchor signatures must be aligned and finite")
    if any(np.asarray(value).shape != (matrix.shape[0],) for value in vectors):
        raise ValueError("anchor signatures have different shapes")
    left, singular, _ = np.linalg.svd(matrix, full_matrices=False)
    if singular.size == 0 or not np.all(np.isfinite(singular)):
        raise ValueError("anchor signature span is empty or non-finite")
    tolerance = max(matrix.shape) * np.finfo(float).eps * singular[0]
    rank = int(np.count_nonzero(singular > tolerance))
    if rank == 0:
        raise ValueError("anchor signature span has zero numerical rank")
    return left[:, :rank]


def signature_residual_gain(
    basis: np.ndarray,
    vector: np.ndarray,
) -> float:
    """Measure one normalized signature outside an existing orthonormal span."""

    reference = np.asarray(basis, dtype=float)
    value = np.asarray(vector, dtype=float)
    if reference.ndim != 2 or value.shape != (reference.shape[0],):
        raise ValueError("candidate signature shape differs from anchor span")
    if reference.shape[1] == 0 or not np.all(np.isfinite(reference)):
        raise ValueError("anchor span is empty or non-finite")
    if not np.all(np.isfinite(value)):
        raise ValueError("candidate signature must be finite")
    residual = value - reference @ (reference.T @ value)
    gain = float(np.linalg.norm(residual))
    if not np.isfinite(gain):
        raise ValueError("candidate residual gain is non-finite")
    return max(0.0, min(1.0, gain))


def select_coverage_candidate(
    anchor_vectors: Sequence[np.ndarray],
    candidate_vectors: Sequence[np.ndarray],
    candidate_base_ranks: Sequence[int],
) -> CoverageChoice:
    """Select the largest residual; Base rank deterministically breaks ties."""

    if not candidate_vectors:
        raise ValueError("coverage candidate pool is empty")
    if len(candidate_vectors) != len(candidate_base_ranks):
        raise ValueError("candidate vectors and Base ranks are not aligned")
    basis = orthonormal_signature_span(anchor_vectors)
    gains: list[float] = []
    for vector in candidate_vectors:
        gains.append(signature_residual_gain(basis, vector))
    selected = min(
        range(len(gains)),
        key=lambda index: (
            -gains[index],
            int(candidate_base_ranks[index]),
            index,
        ),
    )
    return CoverageChoice(
        selected_index=selected,
        selected_gain=gains[selected],
        gains=tuple(gains),
        reference_rank=int(basis.shape[1]),
    )
