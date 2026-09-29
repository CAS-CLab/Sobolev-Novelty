"""Reference LOO projection and conditioned Gram/Cholesky fast path."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class NoveltyComputation:
    novelties: np.ndarray
    singular_values: np.ndarray
    rank: int
    condition_number: float
    algorithm_used: str
    fallback_reason: str | None
    term_diagnostics: list[dict[str, object]]


def normalize_signatures(signatures: np.ndarray, term_norms: np.ndarray) -> np.ndarray:
    return signatures / term_norms[None, :]


def matrix_diagnostics(matrix: np.ndarray, rcond: float) -> tuple[np.ndarray, int, float]:
    singular = np.linalg.svd(matrix, compute_uv=False)
    if not singular.size:
        return singular, 0, math.inf
    tolerance = rcond * float(singular[0])
    rank = int(np.sum(singular > tolerance))
    condition = (
        float(singular[0] / singular[-1])
        if rank == matrix.shape[1] and singular[-1] > 0
        else math.inf
    )
    return singular, rank, condition


def reference_novelties(
    normalized_signatures: np.ndarray,
    rcond: float = 1e-10,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    """Trusted leave-one-term-out ``lstsq`` implementation."""

    n_terms = normalized_signatures.shape[1]
    if n_terms == 1:
        return np.ones(1), [
            {
                "loo_rank": 0,
                "loo_singular_values": [],
                "loo_condition_number": None,
            }
        ]
    novelty = np.empty(n_terms, dtype=float)
    diagnostics: list[dict[str, object]] = []
    for index in range(n_terms):
        target = normalized_signatures[:, index]
        others = np.delete(normalized_signatures, index, axis=1)
        coefficients, _, rank, singular = np.linalg.lstsq(others, target, rcond=rcond)
        residual = target - others @ coefficients
        value = float(np.linalg.norm(residual))
        if value < -1e-12 or value > 1 + 1e-10:
            raise FloatingPointError(f"LOO novelty {value} lies outside [0,1]")
        novelty[index] = float(np.clip(value, 0.0, 1.0))
        condition = (
            float(singular[0] / singular[-1])
            if singular.size and rank == others.shape[1] and singular[-1] > 0
            else math.inf
        )
        diagnostics.append(
            {
                "loo_rank": int(rank),
                "loo_singular_values": singular.tolist(),
                "loo_condition_number": condition,
            }
        )
    return novelty, diagnostics


def gram_novelties(normalized_signatures: np.ndarray) -> np.ndarray:
    """Compute Gram diagonal inverse through Cholesky solves, never ``inv(G)``."""

    gram = normalized_signatures.T @ normalized_signatures
    cholesky = np.linalg.cholesky(gram)
    inverse_factor = np.linalg.solve(cholesky, np.eye(gram.shape[0]))
    inverse_diagonal = np.sum(np.square(inverse_factor), axis=0)
    if np.any(~np.isfinite(inverse_diagonal)) or np.any(inverse_diagonal <= 0):
        raise FloatingPointError("Invalid diagonal of the implicit Gram inverse")
    novelty_squared = 1.0 / inverse_diagonal
    if np.any(novelty_squared < -1e-12) or np.any(novelty_squared > 1 + 1e-8):
        raise FloatingPointError("Gram novelty squared lies outside [0,1]")
    return np.sqrt(np.clip(novelty_squared, 0.0, 1.0))


def compute_novelties(
    signatures: np.ndarray,
    term_norms: np.ndarray,
    rcond: float,
    fast_gram: bool,
    condition_threshold: float,
) -> NoveltyComputation:
    """Use Gram only for full-rank, well-conditioned signatures; otherwise LOO."""

    normalized = normalize_signatures(signatures, term_norms)
    singular, rank, condition = matrix_diagnostics(normalized, rcond)
    n_terms = normalized.shape[1]
    fallback: str | None = None
    if n_terms == 1:
        novelty, term_diagnostics = reference_novelties(normalized, rcond)
        algorithm = "loo_lstsq"
        fallback = "single_term_convention"
    elif not fast_gram:
        fallback = "fast_gram_disabled"
        novelty, term_diagnostics = reference_novelties(normalized, rcond)
        algorithm = "loo_lstsq"
    elif rank != n_terms:
        fallback = f"rank_deficient:{rank}<{n_terms}"
        novelty, term_diagnostics = reference_novelties(normalized, rcond)
        algorithm = "loo_lstsq"
    elif not np.isfinite(condition) or condition > condition_threshold:
        fallback = f"ill_conditioned:{condition:.6g}>{condition_threshold:.6g}"
        novelty, term_diagnostics = reference_novelties(normalized, rcond)
        algorithm = "loo_lstsq"
    else:
        try:
            novelty = gram_novelties(normalized)
            term_diagnostics = [{} for _ in range(n_terms)]
            algorithm = "gram_cholesky"
        except (np.linalg.LinAlgError, FloatingPointError) as error:
            fallback = f"gram_failure:{type(error).__name__}:{error}"
            novelty, term_diagnostics = reference_novelties(normalized, rcond)
            algorithm = "loo_lstsq"
    return NoveltyComputation(
        novelties=novelty,
        singular_values=singular,
        rank=rank,
        condition_number=condition,
        algorithm_used=algorithm,
        fallback_reason=fallback,
        term_diagnostics=term_diagnostics,
    )
