"""Exact backward term pruning with an optional Sobolev tie-breaker."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .canonical_selection import CanonicalTerm


@dataclass(frozen=True)
class BackwardFit:
    coefficients: np.ndarray
    sse: float
    rank: int


class QRBackwardWorkspace:
    """Reuse one tall QR factorization for exact subset least-squares fits."""

    def __init__(
        self,
        design: np.ndarray,
        target: np.ndarray,
        *,
        rcond: float = 1e-10,
    ) -> None:
        matrix = np.asarray(design, dtype=float)
        values = np.asarray(target, dtype=float)
        self.rcond = float(rcond)
        self.column_scales = np.linalg.norm(matrix, axis=0)
        self.column_scales = np.where(
            self.column_scales > 1e-15, self.column_scales, 1.0
        )
        normalized = matrix / self.column_scales
        q, self.reduced_design = np.linalg.qr(normalized, mode="reduced")
        self.reduced_target = q.T @ values
        self.orthogonal_sse = max(
            0.0,
            float(values @ values - self.reduced_target @ self.reduced_target),
        )
        self.target_energy = float(values @ values)

    def _decomposition(
        self, selected: Sequence[int]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, np.ndarray, float]:
        selected_array = np.asarray(selected, dtype=np.int64)
        reduced = self.reduced_design[:, selected_array]
        left, singular_values, right_transpose = np.linalg.svd(
            reduced, full_matrices=True
        )
        rank = (
            0
            if len(singular_values) == 0 or singular_values[0] <= 1e-15
            else int(
                np.sum(singular_values > self.rcond * singular_values[0])
            )
        )
        right = right_transpose.T
        if rank:
            coefficients = right[:, :rank] @ (
                (left[:, :rank].T @ self.reduced_target)
                / singular_values[:rank]
            )
        else:
            coefficients = np.zeros(len(selected_array), dtype=float)
        residual = self.reduced_target - reduced @ coefficients
        sse = self.orthogonal_sse + float(residual @ residual)
        return coefficients, singular_values, right, rank, residual, sse

    def fit(self, selected: Sequence[int]) -> BackwardFit:
        selected_array = np.asarray(selected, dtype=np.int64)
        coefficients, _, _, rank, _, sse = self._decomposition(selected)
        return BackwardFit(
            coefficients=coefficients / self.column_scales[selected_array],
            sse=sse,
            rank=rank,
        )

    def deletion_increases(
        self, selected: Sequence[int]
    ) -> tuple[np.ndarray, BackwardFit]:
        """Return exact refitted SSE increases for deleting each active term."""

        selected_array = np.asarray(selected, dtype=np.int64)
        coefficients, singular_values, right, rank, _, sse = self._decomposition(
            selected
        )
        null_space = right[:, rank:]
        inverse_diagonal = np.zeros(len(selected_array), dtype=float)
        if rank:
            inverse_diagonal = np.sum(
                np.square(right[:, :rank])
                / np.square(singular_values[:rank])[None, :],
                axis=1,
            )
        increases = np.full(len(selected_array), np.inf, dtype=float)
        for position in range(len(selected_array)):
            is_dependent = (
                null_space.shape[1] > 0
                and np.linalg.norm(null_space[position]) > 1e-8
            )
            if is_dependent:
                increases[position] = 0.0
            elif inverse_diagonal[position] > 0.0:
                increases[position] = (
                    coefficients[position] ** 2 / inverse_diagonal[position]
                )
        increases = np.maximum(increases, 0.0)
        return increases, BackwardFit(
            coefficients=coefficients / self.column_scales[selected_array],
            sse=sse,
            rank=rank,
        )


def leave_one_out_novelties(
    signatures: np.ndarray,
    terms: Sequence[CanonicalTerm],
    selected: Sequence[int],
    *,
    rcond: float = 1e-10,
    channel_local: bool = True,
) -> np.ndarray:
    """Term novelty relative to the other selected terms in its channel."""

    selected_array = np.asarray(selected, dtype=np.int64)
    result = np.ones(len(selected_array), dtype=float)
    groups: Sequence[int | None] = range(3) if channel_local else (None,)
    for channel in groups:
        positions = np.asarray(
            [
                position
                for position, index in enumerate(selected_array)
                if channel is None or terms[index].channel == channel
            ],
            dtype=np.int64,
        )
        if len(positions) <= 1:
            continue
        local = signatures[:, selected_array[positions]]
        norms = np.linalg.norm(local, axis=0)
        _, singular_values, right_transpose = np.linalg.svd(
            local, full_matrices=True
        )
        rank = (
            0
            if len(singular_values) == 0 or singular_values[0] <= 1e-15
            else int(np.sum(singular_values > rcond * singular_values[0]))
        )
        right = right_transpose.T
        null_space = right[:, rank:]
        inverse_diagonal = np.zeros(len(positions), dtype=float)
        if rank:
            inverse_diagonal = np.sum(
                np.square(right[:, :rank])
                / np.square(singular_values[:rank])[None, :],
                axis=1,
            )
        for local_position, output_position in enumerate(positions):
            if norms[local_position] <= 1e-15:
                novelty = 0.0
            elif (
                null_space.shape[1] > 0
                and np.linalg.norm(null_space[local_position]) > 1e-8
            ):
                novelty = 0.0
            elif inverse_diagonal[local_position] <= 0.0:
                novelty = 0.0
            else:
                residual_norm = np.sqrt(1.0 / inverse_diagonal[local_position])
                novelty = float(
                    np.clip(residual_norm / norms[local_position], 0.0, 1.0)
                )
            result[output_position] = novelty
    return result


def choose_deletion(
    selected: Sequence[int],
    increases: np.ndarray,
    novelties: np.ndarray,
    *,
    method: str,
    epsilon_r2: float,
    target_energy: float,
) -> tuple[int, dict[str, float | int]]:
    """Choose a deletion; SN only acts inside an energy-R2 safety band."""

    selected_array = np.asarray(selected, dtype=np.int64)
    base_order = np.lexsort((selected_array, increases))
    base_position = int(base_order[0])
    best_increase = float(increases[base_position])
    if method == "base":
        chosen_position = base_position
        eligible_count = 1
        threshold = best_increase
    elif method == "sn_epsilon":
        threshold = best_increase + float(epsilon_r2) * float(target_energy)
        eligible = np.flatnonzero(increases <= threshold)
        order = np.lexsort(
            (
                selected_array[eligible],
                increases[eligible],
                novelties[eligible],
            )
        )
        chosen_position = int(eligible[order[0]])
        eligible_count = int(len(eligible))
    else:
        raise ValueError(f"unknown backward selection method: {method}")
    return int(selected_array[chosen_position]), {
        "base_delete_index": int(selected_array[base_position]),
        "base_delete_increase": best_increase,
        "chosen_increase": float(increases[chosen_position]),
        "chosen_novelty": float(novelties[chosen_position]),
        "eligibility_threshold": float(threshold),
        "eligible_count": eligible_count,
        "changed_from_base": int(chosen_position != base_position),
    }


__all__ = [
    "BackwardFit",
    "QRBackwardWorkspace",
    "choose_deletion",
    "leave_one_out_novelties",
]
