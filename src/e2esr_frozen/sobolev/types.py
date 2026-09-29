"""Typed public result and diagnostic structures."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import sympy as sp


class FailureType(str, Enum):
    """Explicit failure modes; no evaluator failure is converted to novelty zero."""

    NONE = "none"
    INVALID_INPUT = "invalid_input"
    PARSE_FAILURE = "parse_failure"
    DECOMPOSITION_FAILURE = "decomposition_failure"
    DIFFERENTIATION_FAILURE = "differentiation_failure"
    EVALUATION_FAILURE = "evaluation_failure"
    INSUFFICIENT_VALID_SAMPLES = "insufficient_valid_samples"
    CONSTANT_INPUT_SCALE = "constant_input_scale"
    ZERO_OUTPUT_SCALE = "zero_output_scale"
    ZERO_SIGNATURE = "zero_signature"
    NUMERICAL_OVERFLOW = "numerical_overflow"
    LINEAR_ALGEBRA_FAILURE = "linear_algebra_failure"


@dataclass(frozen=True)
class TermSpec:
    """One numeric coefficient and its coefficient-free symbolic basis term."""

    coefficient: float
    basis: sp.Expr
    canonical: str

    @property
    def display(self) -> str:
        return str(self.basis)


@dataclass
class DetailedRuntime:
    """Candidate evaluator timing in seconds."""

    parsing: float = 0.0
    decomposition: float = 0.0
    cache_lookup: float = 0.0
    symbolic_differentiation: float = 0.0
    term_value_evaluation: float = 0.0
    gradient_evaluation: float = 0.0
    shared_mask: float = 0.0
    signature_construction: float = 0.0
    novelty_linear_algebra: float = 0.0
    coefficient_fitting: float = 0.0
    total_candidate_evaluation: float = 0.0


@dataclass
class EvaluationResult:
    """Complete evaluator result, including partial diagnostics on failure."""

    raw_expression: str
    fitted_expression: str
    terms: list[str] = field(default_factory=list)
    coefficients: list[float] = field(default_factory=list)
    valid_mask_size: int = 0
    valid_mask_identity: str | None = None
    geometry_indices: list[int] = field(default_factory=list)
    input_means: list[float] = field(default_factory=list)
    input_scales: list[float] = field(default_factory=list)
    output_scale: float | None = None
    signature_shape: tuple[int, int] = (0, 0)
    term_norms: list[float] = field(default_factory=list)
    term_novelties: list[float] = field(default_factory=list)
    min_novelty: float | None = None
    mean_novelty: float | None = None
    low_novelty_count: int = 0
    low_novelty_ratio: float | None = None
    threshold: float = 0.0
    singular_values: list[float] = field(default_factory=list)
    rank: int = 0
    condition_number: float | None = None
    algorithm_used: str = "not_run"
    fallback_reason: str | None = None
    cache_hits: int = 0
    cache_misses: int = 0
    cache_memory_bytes: int = 0
    success: bool = False
    failure_type: FailureType = FailureType.NONE
    failure_message: str = ""
    detailed_runtime: DetailedRuntime = field(default_factory=DetailedRuntime)
    term_diagnostics: list[dict[str, Any]] = field(default_factory=list)
    candidate_geometry_key: str | None = None
    geometry_cache_hit: bool = False
    failure_cache_hit: bool = False
    output_scale_free_internal: bool = False
    geometry_reuse_mode: str = "full"
    geometry_fallback_reason: str | None = None
    parent_geometry_key: str | None = None
    incremental_hit: bool = False
    incremental_fallback_reason: str | None = None
    reused_term_count: int = 0
    added_term_count: int = 0
    removed_term_count: int = 0
    changed_term_count: int = 0

    def as_dict(self) -> dict[str, Any]:
        """Return a recursively JSON-friendly representation."""

        value = asdict(self)
        value["failure_type"] = self.failure_type.value
        return _json_safe(value)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return str(value)
    return value
