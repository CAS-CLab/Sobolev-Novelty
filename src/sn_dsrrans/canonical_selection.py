"""Direct sparse term selection for the polynomial DSRRANS grammar.

The public DSRRANS operator set is add/subtract/multiply over two invariants,
so every generated coefficient branch expands into bivariate monomials.  This
module exposes that exact canonical dictionary and compares two selectors:

* Base: largest one-step least-squares reduction;
* SN: among Base-equivalent reductions, largest conditional Sobolev novelty.

Both selectors use the same tensor design, row split, coefficient refit and
term budget.  Sobolev information only changes which eligible term is added.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np

from .data import DSRRANSData

Exponent = tuple[int, int]


@dataclass(frozen=True, order=True)
class CanonicalTerm:
    channel: int
    exponent_x1: int
    exponent_x2: int

    @property
    def exponent(self) -> Exponent:
        return self.exponent_x1, self.exponent_x2

    @property
    def label(self) -> str:
        pieces = []
        if self.exponent_x1:
            pieces.append(
                "x1" if self.exponent_x1 == 1 else f"x1**{self.exponent_x1}"
            )
        if self.exponent_x2:
            pieces.append(
                "x2" if self.exponent_x2 == 1 else f"x2**{self.exponent_x2}"
            )
        return "*".join(pieces) if pieces else "1"


def canonical_terms(max_degree: int) -> tuple[CanonicalTerm, ...]:
    if max_degree < 0:
        raise ValueError("max_degree must be non-negative")
    exponents = [
        (degree - exponent_x2, exponent_x2)
        for degree in range(max_degree + 1)
        for exponent_x2 in range(degree + 1)
    ]
    return tuple(
        CanonicalTerm(channel, exponent_x1, exponent_x2)
        for channel in range(3)
        for exponent_x1, exponent_x2 in exponents
    )


def tensor_design(
    data: DSRRANSData,
    terms: Sequence[CanonicalTerm],
) -> np.ndarray:
    """Construct the exact flattened four-component tensor design."""

    x1, x2 = data.invariants.T
    monomial_cache: dict[Exponent, np.ndarray] = {}
    columns = []
    for term in terms:
        exponent = term.exponent
        values = monomial_cache.get(exponent)
        if values is None:
            values = np.power(x1, exponent[0]) * np.power(x2, exponent[1])
            monomial_cache[exponent] = values
        columns.append(
            (
                values[:, None]
                * data.basis_components[:, term.channel, :]
            ).reshape(-1)
        )
    return np.column_stack(columns)


def flattened_row_indices(rows: np.ndarray, component_count: int = 4) -> np.ndarray:
    return (
        np.asarray(rows, dtype=np.int64)[:, None] * component_count
        + np.arange(component_count)
    ).reshape(-1)


def sobolev_signatures(
    points: np.ndarray,
    terms: Sequence[CanonicalTerm],
    *,
    lambda_value: float = 1.0,
    lambda_gradient: float = 1.0,
) -> np.ndarray:
    """Return unit-normalized value-gradient signatures for all monomials."""

    values = np.asarray(points, dtype=float)
    n_samples, n_dimensions = values.shape
    if n_dimensions != 2:
        raise ValueError("DSRRANS canonical signatures require two invariants")
    scales = np.std(values, axis=0)
    x1, x2 = values.T
    signatures = []
    cache: dict[Exponent, np.ndarray] = {}
    for term in terms:
        exponent = term.exponent
        signature = cache.get(exponent)
        if signature is None:
            basis = np.power(x1, exponent[0]) * np.power(x2, exponent[1])
            derivative_x1 = (
                np.zeros(n_samples)
                if exponent[0] == 0
                else exponent[0]
                * np.power(x1, exponent[0] - 1)
                * np.power(x2, exponent[1])
            )
            derivative_x2 = (
                np.zeros(n_samples)
                if exponent[1] == 0
                else exponent[1]
                * np.power(x1, exponent[0])
                * np.power(x2, exponent[1] - 1)
            )
            signature = np.concatenate(
                (
                    np.sqrt(lambda_value / n_samples) * basis,
                    np.sqrt(lambda_gradient / (n_samples * n_dimensions))
                    * np.column_stack(
                        (scales[0] * derivative_x1, scales[1] * derivative_x2)
                    ).reshape(-1),
                )
            )
            norm = float(np.linalg.norm(signature))
            signature = signature / norm if norm > 1e-15 else np.zeros_like(signature)
            cache[exponent] = signature
        signatures.append(signature)
    return np.column_stack(signatures)


def tensor_sobolev_signatures(
    data: DSRRANSData,
    terms: Sequence[CanonicalTerm],
    *,
    rows: np.ndarray | None = None,
    lambda_value: float = 1.0,
    lambda_gradient: float = 1.0,
) -> np.ndarray:
    """Signatures of vector terms phi(I) * T_k with T held fixed.

    The DSRRANS predictor is a vector-valued additive model.  Holding the
    supplied tensor basis fixed makes the invariant derivatives exact:
    d(phi(I) T_k)/dI_j = (d phi/dI_j) T_k.
    """

    selected_rows = (
        np.arange(len(data.invariants), dtype=np.int64)
        if rows is None
        else np.asarray(rows, dtype=np.int64)
    )
    points = data.invariants[selected_rows]
    bases = data.basis_components[selected_rows]
    n_samples, n_dimensions = points.shape
    n_components = bases.shape[2]
    if n_dimensions != 2:
        raise ValueError("DSRRANS tensor signatures require two invariants")
    if lambda_value < 0.0 or lambda_gradient < 0.0:
        raise ValueError("signature block weights must be non-negative")
    if lambda_value == 0.0 and lambda_gradient == 0.0:
        raise ValueError("at least one signature block must have positive weight")
    scales = np.std(points, axis=0)
    x1, x2 = points.T
    polynomial_cache: dict[Exponent, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    signatures = []
    for term in terms:
        values = polynomial_cache.get(term.exponent)
        if values is None:
            exponent_x1, exponent_x2 = term.exponent
            basis_value = np.power(x1, exponent_x1) * np.power(x2, exponent_x2)
            derivative_x1 = (
                np.zeros(n_samples)
                if exponent_x1 == 0
                else exponent_x1
                * np.power(x1, exponent_x1 - 1)
                * np.power(x2, exponent_x2)
            )
            derivative_x2 = (
                np.zeros(n_samples)
                if exponent_x2 == 0
                else exponent_x2
                * np.power(x1, exponent_x1)
                * np.power(x2, exponent_x2 - 1)
            )
            values = basis_value, derivative_x1, derivative_x2
            polynomial_cache[term.exponent] = values
        basis_value, derivative_x1, derivative_x2 = values
        tensor = bases[:, term.channel, :]
        blocks = []
        if lambda_value > 0.0:
            blocks.append(
                np.sqrt(lambda_value / (n_samples * n_components))
                * (basis_value[:, None] * tensor).reshape(-1)
            )
        if lambda_gradient > 0.0:
            gradient_scale = np.sqrt(
                lambda_gradient
                / (n_samples * n_dimensions * n_components)
            )
            blocks.extend(
                (
                    gradient_scale
                    * (scales[0] * derivative_x1[:, None] * tensor).reshape(-1),
                    gradient_scale
                    * (scales[1] * derivative_x2[:, None] * tensor).reshape(-1),
                )
            )
        signature = np.concatenate(blocks)
        norm = float(np.linalg.norm(signature))
        signatures.append(
            signature / norm if norm > 1e-15 else np.zeros_like(signature)
        )
    return np.column_stack(signatures)


def conditional_novelties(
    signatures: np.ndarray,
    terms: Sequence[CanonicalTerm],
    selected: Sequence[int],
    candidates: Sequence[int],
    *,
    rcond: float = 1e-10,
    channel_local: bool = True,
) -> np.ndarray:
    """Novelty relative to selected terms, locally by channel by default."""

    result = np.ones(len(candidates), dtype=float)
    selected_array = np.asarray(selected, dtype=np.int64)
    groups: Sequence[int | None] = range(3) if channel_local else (None,)
    for channel in groups:
        local_selected = (
            selected_array
            if channel is None
            else selected_array[
                [terms[index].channel == channel for index in selected_array]
            ]
        )
        local_positions = [
            position
            for position, index in enumerate(candidates)
            if channel is None or terms[index].channel == channel
        ]
        if not local_positions or len(local_selected) == 0:
            continue
        references = signatures[:, local_selected]
        left, singular_values, _ = np.linalg.svd(references, full_matrices=False)
        rank = (
            0
            if len(singular_values) == 0
            else int(np.sum(singular_values > rcond * singular_values[0]))
        )
        q = left[:, :rank]
        local_candidates = np.asarray(
            [candidates[position] for position in local_positions], dtype=np.int64
        )
        candidate_signatures = signatures[:, local_candidates]
        residual_energy = 1.0 - np.sum(
            np.square(q.T @ candidate_signatures), axis=0
        )
        residual_energy = np.clip(residual_energy, 0.0, 1.0)
        result[local_positions] = np.sqrt(residual_energy)
    return result


def marginal_sse_reductions(
    design: np.ndarray,
    target: np.ndarray,
    selected: Sequence[int],
    candidates: Sequence[int],
    *,
    rcond: float = 1e-10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Exact one-column least-squares SSE reduction for each candidate."""

    column_norms = np.linalg.norm(design, axis=0)
    column_norms = np.where(column_norms > 1e-15, column_norms, 1.0)
    normalized_design = design / column_norms
    selected_array = np.asarray(selected, dtype=np.int64)
    candidate_array = np.asarray(candidates, dtype=np.int64)
    if len(selected_array):
        selected_design = normalized_design[:, selected_array]
        coefficients, *_ = np.linalg.lstsq(
            selected_design, target, rcond=rcond
        )
        residual = target - selected_design @ coefficients
        left, singular_values, _ = np.linalg.svd(
            selected_design, full_matrices=False
        )
        rank = (
            0
            if len(singular_values) == 0
            else int(np.sum(singular_values > rcond * singular_values[0]))
        )
        q = left[:, :rank]
        candidates_design = normalized_design[:, candidate_array]
        projected = q.T @ candidates_design
        residual_norms_squared = np.sum(
            np.square(candidates_design), axis=0
        ) - np.sum(np.square(projected), axis=0)
    else:
        coefficients = np.empty(0, dtype=float)
        residual = target.copy()
        candidates_design = normalized_design[:, candidate_array]
        residual_norms_squared = np.sum(np.square(candidates_design), axis=0)
    correlations = candidates_design.T @ residual
    reductions = np.zeros(len(candidate_array), dtype=float)
    usable = residual_norms_squared > 1e-14
    reductions[usable] = (
        np.square(correlations[usable]) / residual_norms_squared[usable]
    )
    return reductions, residual, coefficients


def choose_candidate(
    candidates: Sequence[int],
    reductions: np.ndarray,
    novelties: np.ndarray,
    *,
    method: str,
    epsilon: float,
    tau: float = 1.0 / math.sqrt(10.0),
    structural_mix: float = 0.25,
    probe_fraction: float = 0.2,
) -> tuple[int, dict[str, float | int]]:
    """Choose one term while keeping Base reward as the safety constraint."""

    candidate_array = np.asarray(candidates, dtype=np.int64)
    best_position = int(np.argmax(reductions))
    best_reduction = float(reductions[best_position])
    if method == "base":
        selected_position = best_position
        eligible_count = 1
    elif method == "sn_epsilon":
        threshold = max(0.0, (1.0 - epsilon) * best_reduction)
        eligible = np.flatnonzero(reductions >= threshold)
        order = np.lexsort(
            (candidate_array[eligible], -reductions[eligible], -novelties[eligible])
        )
        selected_position = int(eligible[order[0]])
        eligible_count = len(eligible)
    elif method == "sn_tau_guard":
        threshold = max(0.0, (1.0 - epsilon) * best_reduction)
        eligible = np.flatnonzero(reductions >= threshold)
        qualified = eligible[novelties[eligible] >= tau]
        if novelties[best_position] >= tau or len(qualified) == 0:
            selected_position = best_position
        else:
            order = np.lexsort(
                (
                    candidate_array[qualified],
                    -novelties[qualified],
                    -reductions[qualified],
                )
            )
            selected_position = int(qualified[order[0]])
        eligible_count = len(eligible)
    elif method == "sn_rank":
        probe_count = max(1, int(np.ceil(probe_fraction * len(candidate_array))))
        base_order = np.lexsort((candidate_array, -reductions))[:probe_count]
        base_rank = np.linspace(1.0, 0.0, probe_count, endpoint=False)
        novelty_order = np.lexsort(
            (candidate_array[base_order], -novelties[base_order])
        )
        novelty_rank = np.empty(probe_count, dtype=float)
        novelty_rank[novelty_order] = np.linspace(
            1.0, 0.0, probe_count, endpoint=False
        )
        scores = (1.0 - structural_mix) * base_rank + structural_mix * novelty_rank
        local = int(
            np.lexsort(
                (
                    candidate_array[base_order],
                    -reductions[base_order],
                    -scores,
                )
            )[0]
        )
        selected_position = int(base_order[local])
        eligible_count = probe_count
    else:
        raise ValueError(f"unknown canonical selection method: {method}")
    chosen = int(candidate_array[selected_position])
    return chosen, {
        "base_best_index": int(candidate_array[best_position]),
        "base_best_reduction": best_reduction,
        "chosen_reduction": float(reductions[selected_position]),
        "chosen_novelty": float(novelties[selected_position]),
        "eligible_count": int(eligible_count),
        "changed_from_base": int(selected_position != best_position),
    }


__all__ = [
    "CanonicalTerm",
    "canonical_terms",
    "choose_candidate",
    "conditional_novelties",
    "flattened_row_indices",
    "marginal_sse_reductions",
    "sobolev_signatures",
    "tensor_sobolev_signatures",
    "tensor_design",
]
