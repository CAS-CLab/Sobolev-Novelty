"""Immutable candidate-level Sobolev geometry shared across coefficients."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from .types import FailureType


@dataclass(frozen=True)
class CandidateGeometryKey:
    """Coefficient-free multiset key for one empirical Sobolev geometry."""

    canonical_terms: tuple[str, ...]
    dataset_identity: str
    geometry_subset_identity: str
    input_normalization_identity: str
    derivative_order: int
    lambda_value: float
    lambda_gradient: float
    operator_configuration: str

    @property
    def digest(self) -> str:
        payload = repr(
            (
                self.canonical_terms,
                self.dataset_identity,
                self.geometry_subset_identity,
                self.input_normalization_identity,
                self.derivative_order,
                self.lambda_value,
                self.lambda_gradient,
                self.operator_configuration,
            )
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


@dataclass
class GeometryState:
    """Coefficient-free raw arrays and completed novelty diagnostics.

    Columns use canonical multiset order. ``signatures`` and ``term_norms``
    deliberately omit the candidate-wide output scale. Novelty is invariant to
    that common non-zero scale; the public evaluator restores scaled norms and
    still performs the original output-scale checks for every candidate.
    """

    key: CandidateGeometryKey
    values: np.ndarray
    gradients: np.ndarray
    per_term_finite_masks: np.ndarray
    shared_valid_mask: np.ndarray
    signatures: np.ndarray
    term_norms: np.ndarray
    term_novelties: np.ndarray
    singular_values: np.ndarray
    rank: int
    condition_number: float
    algorithm_used: str
    fallback_reason: str | None
    term_diagnostics: list[dict[str, object]]
    success: bool = True
    failure_type: FailureType = FailureType.NONE
    failure_message: str = ""

    @property
    def memory_bytes(self) -> int:
        arrays = (
            self.values,
            self.gradients,
            self.per_term_finite_masks,
            self.shared_valid_mask,
            self.signatures,
            self.term_norms,
            self.term_novelties,
            self.singular_values,
        )
        return int(sum(array.nbytes for array in arrays))

