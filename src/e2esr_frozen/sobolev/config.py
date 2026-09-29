"""Central configuration for Sobolev novelty evaluation."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any


TAU_THEORY = 1.0 / math.sqrt(10.0)
TAU_EMPIRICAL = 0.30


@dataclass(frozen=True)
class SobolevConfig:
    """Numerical and sampling choices shared by every evaluator call."""

    lambda_value: float = 1.0
    lambda_gradient: float = 1.0
    rcond: float = 1e-10
    min_valid_samples: int = 32
    threshold: float = TAU_THEORY
    output_scale_tolerance: float = 1e-12
    input_scale_tolerance: float = 1e-12
    zero_signature_tolerance: float = 1e-15
    fast_gram: bool = True
    gram_condition_threshold: float = 1e6
    cache_enabled: bool = True
    candidate_geometry_cache: bool = False
    output_scale_free_internal: bool = False
    parent_child_incremental: bool = False
    incremental_max_changed_terms: int = 4
    geometry_cache_max_entries: int = 20_000
    geometry_cache_max_memory_bytes: int = 512 * 1024 * 1024
    decomposition_cache_max_entries: int = 50_000
    geometry_sample_size: int | None = None
    geometry_seed: int = 20260731
    protected_epsilon: float = 1e-6

    def __post_init__(self) -> None:
        if self.lambda_value <= 0 or self.lambda_gradient < 0:
            raise ValueError("lambda_value must be positive and lambda_gradient non-negative")
        if self.rcond <= 0:
            raise ValueError("rcond must be positive")
        if self.min_valid_samples < 1:
            raise ValueError("min_valid_samples must be positive")
        if not 0 < self.threshold <= 1:
            raise ValueError("threshold must be in (0, 1]")
        if self.output_scale_tolerance <= 0 or self.input_scale_tolerance <= 0:
            raise ValueError("scale tolerances must be positive")
        if self.zero_signature_tolerance <= 0:
            raise ValueError("zero_signature_tolerance must be positive")
        if self.gram_condition_threshold <= 1:
            raise ValueError("gram_condition_threshold must be greater than one")
        if self.geometry_sample_size is not None and self.geometry_sample_size < 1:
            raise ValueError("geometry_sample_size must be positive or None for full")
        if self.geometry_cache_max_entries < 1:
            raise ValueError("geometry_cache_max_entries must be positive")
        if self.geometry_cache_max_memory_bytes < 1:
            raise ValueError("geometry_cache_max_memory_bytes must be positive")
        if self.decomposition_cache_max_entries < 1:
            raise ValueError("decomposition_cache_max_entries must be positive")
        if self.incremental_max_changed_terms < 1:
            raise ValueError("incremental_max_changed_terms must be positive")
        if self.parent_child_incremental and not self.candidate_geometry_cache:
            raise ValueError("parent_child_incremental requires candidate_geometry_cache")
        if self.protected_epsilon <= 0:
            raise ValueError("protected_epsilon must be positive")

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly configuration mapping."""

        return asdict(self)
