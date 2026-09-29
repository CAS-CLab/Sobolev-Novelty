"""Deterministic two-stage shortlist construction and reranking helpers."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol, Sequence, TypeVar


class RerankCandidate(Protocol):
    """Minimal candidate surface needed by the structural selector."""

    candidate_id: int | None
    base_reward: float | None
    final_reward: float | None
    complexity: float | None
    phi: object


CandidateT = TypeVar("CandidateT", bound=RerankCandidate)


@dataclass(frozen=True)
class ShortlistConfig:
    """Configuration for lexicographic base screening and SN reranking."""

    mode: str = "hybrid"
    size: int = 64
    ratio: float = 0.03
    minimum: int = 32
    maximum: int = 128
    prune_elite_k: int = 5

    def __post_init__(self) -> None:
        if self.mode not in {"fixed", "ratio", "hybrid"}:
            raise ValueError(f"Unknown shortlist mode: {self.mode}")
        if self.size < 1:
            raise ValueError("shortlist size must be positive")
        if not 0 < self.ratio <= 1:
            raise ValueError("shortlist ratio must be in (0, 1]")
        if self.minimum < 1 or self.maximum < self.minimum:
            raise ValueError("shortlist bounds must satisfy 1 <= minimum <= maximum")
        if self.prune_elite_k < 0:
            raise ValueError("prune_elite_k must be non-negative")

    def count(self, candidate_count: int) -> int:
        """Return the deterministic shortlist size for one candidate pool."""

        if candidate_count < 0:
            raise ValueError("candidate_count must be non-negative")
        if candidate_count == 0:
            return 0
        if self.mode == "fixed":
            requested = self.size
        elif self.mode == "ratio":
            requested = math.ceil(self.ratio * candidate_count)
        else:
            requested = min(
                self.maximum,
                max(self.minimum, math.ceil(self.ratio * candidate_count)),
            )
        return min(candidate_count, requested)


def _finite_reward(value: float | None) -> float:
    if value is None or not math.isfinite(float(value)):
        return -math.inf
    return float(value)


def _finite_complexity(value: float | None) -> float:
    if value is None or not math.isfinite(float(value)):
        return math.inf
    return float(value)


def canonical_candidate_expression(candidate: RerankCandidate) -> str:
    """Return the stable expression tie-breaker used by both stages."""

    return str(candidate.phi)


def rank_by_base(candidates: Sequence[CandidateT]) -> list[CandidateT]:
    """Rank by base reward, complexity, expression and candidate id."""

    return sorted(
        candidates,
        key=lambda candidate: (
            -_finite_reward(candidate.base_reward),
            _finite_complexity(candidate.complexity),
            canonical_candidate_expression(candidate),
            candidate.candidate_id if candidate.candidate_id is not None else math.inf,
        ),
    )


def rank_by_structural(candidates: Sequence[CandidateT]) -> list[CandidateT]:
    """Rank deterministically after Sobolev evaluation."""

    return sorted(
        candidates,
        key=lambda candidate: (
            -_finite_reward(candidate.final_reward),
            -_finite_reward(candidate.base_reward),
            _finite_complexity(candidate.complexity),
            canonical_candidate_expression(candidate),
            candidate.candidate_id if candidate.candidate_id is not None else math.inf,
        ),
    )

