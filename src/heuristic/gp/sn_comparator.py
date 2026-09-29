"""Lazy epsilon-lexicographic comparator for the SN-GP-v2 plugin."""

from __future__ import annotations

import math
import time
from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable, Protocol, Sequence

import numpy as np


DEFAULT_SN_COMPARE_EPSILON_ABS = 1e-6
DEFAULT_SN_COMPARE_EPSILON_REL = 1e-4


class ComparableCandidate(Protocol):
    candidate_id: int
    base_reward: float
    complexity: float
    canonical_fitted_expression: str
    valid: bool

    @property
    def raw_expression(self) -> str: ...


@dataclass(frozen=True)
class LazySobolevView:
    """The comparator-facing part of one cached Sobolev evaluation."""

    success: bool
    penalty: float | None
    failure_type: str | None = None


@dataclass(frozen=True)
class ComparisonResult:
    """Auditable outcome of one pairwise comparison."""

    winner: ComparableCandidate
    loser: ComparableCandidate
    decision_stage: str
    context: str
    base_reward_gap: float | None
    epsilon_threshold: float | None
    sobolev_triggered: bool
    penalty_a: float | None
    penalty_b: float | None
    fallback_reason: str | None
    base_winner_id: int
    winner_changed_from_base: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": int(self.winner.candidate_id),
            "compared_candidate_id": int(self.loser.candidate_id),
            "context": self.context,
            "base_reward_gap": self.base_reward_gap,
            "epsilon_threshold": self.epsilon_threshold,
            "sobolev_triggered": self.sobolev_triggered,
            "penalty_a": self.penalty_a,
            "penalty_b": self.penalty_b,
            "winner": int(self.winner.candidate_id),
            "decision_stage": self.decision_stage,
            "fallback_reason": self.fallback_reason,
            "base_winner_id": self.base_winner_id,
            "winner_changed_from_base": self.winner_changed_from_base,
        }


Evaluator = Callable[[ComparableCandidate, str], LazySobolevView]
TraceSink = Callable[[ComparisonResult], None]


class SNComparator:
    """Base-first comparator with lazy Sobolev tie resolution.

    Epsilon equivalence is pairwise and need not be transitive.  ``select_top``
    therefore canonicalizes the input set and uses a fixed reduction instead
    of handing this comparator to a general-purpose sorting routine.
    """

    def __init__(
        self,
        evaluator: Evaluator,
        *,
        epsilon_abs: float = DEFAULT_SN_COMPARE_EPSILON_ABS,
        epsilon_rel: float = DEFAULT_SN_COMPARE_EPSILON_REL,
        trace_sink: TraceSink | None = None,
    ) -> None:
        if epsilon_abs < 0 or epsilon_rel < 0:
            raise ValueError("SN comparison epsilons must be non-negative")
        self.evaluator = evaluator
        self.epsilon_abs = float(epsilon_abs)
        self.epsilon_rel = float(epsilon_rel)
        self.trace_sink = trace_sink
        self.stats: Counter[str] = Counter()

    def compare(
        self,
        candidate_a: ComparableCandidate,
        candidate_b: ComparableCandidate,
        *,
        context: str,
    ) -> ComparisonResult:
        """Return the winner without consulting raw/test metrics."""

        started = time.perf_counter()
        try:
            return self._compare(candidate_a, candidate_b, context=context)
        finally:
            self.stats["comparator_time_seconds"] += time.perf_counter() - started

    def _compare(
        self,
        candidate_a: ComparableCandidate,
        candidate_b: ComparableCandidate,
        *,
        context: str,
    ) -> ComparisonResult:

        self._count("comparisons_total", context)
        base_winner = min((candidate_a, candidate_b), key=self._base_key)

        if candidate_a is candidate_b:
            return self._finish(
                candidate_a,
                candidate_b,
                "same_candidate",
                context,
                None,
                None,
                False,
                None,
                None,
                None,
                base_winner,
            )
        if bool(candidate_a.valid) != bool(candidate_b.valid):
            winner = candidate_a if candidate_a.valid else candidate_b
            self._count("base_direct_decisions", context)
            return self._finish(
                winner,
                candidate_b if winner is candidate_a else candidate_a,
                "base_validity",
                context,
                None,
                None,
                False,
                None,
                None,
                None,
                base_winner,
            )

        reward_a = self._finite_reward(candidate_a)
        reward_b = self._finite_reward(candidate_b)
        if not candidate_a.valid or not (math.isfinite(reward_a) and math.isfinite(reward_b)):
            return self._finish(
                base_winner,
                candidate_b if base_winner is candidate_a else candidate_a,
                "base_fallback",
                context,
                None,
                None,
                False,
                None,
                None,
                "invalid_or_nonfinite_base",
                base_winner,
            )

        gap = abs(reward_a - reward_b)
        epsilon = self.epsilon_abs + self.epsilon_rel * max(
            abs(reward_a), abs(reward_b)
        )
        if gap > epsilon:
            winner = candidate_a if reward_a > reward_b else candidate_b
            self._count("base_direct_decisions", context)
            return self._finish(
                winner,
                candidate_b if winner is candidate_a else candidate_a,
                "base_reward",
                context,
                gap,
                epsilon,
                False,
                None,
                None,
                None,
                base_winner,
            )

        self._count("epsilon_equivalent_comparisons", context)
        ordered = sorted((candidate_a, candidate_b), key=self._stable_identity_key)
        views = {id(value): self.evaluator(value, context) for value in ordered}
        view_a = views[id(candidate_a)]
        view_b = views[id(candidate_b)]
        penalty_a = self._finite_penalty(view_a)
        penalty_b = self._finite_penalty(view_b)
        if not view_a.success or not view_b.success or penalty_a is None or penalty_b is None:
            failures = [
                view.failure_type or "unknown"
                for view in (view_a, view_b)
                if not view.success or self._finite_penalty(view) is None
            ]
            self._count("sobolev_failure_fallbacks", context)
            return self._finish(
                base_winner,
                candidate_b if base_winner is candidate_a else candidate_a,
                "sobolev_failure_fallback",
                context,
                gap,
                epsilon,
                True,
                penalty_a,
                penalty_b,
                ",".join(failures),
                base_winner,
            )

        if penalty_a != penalty_b:
            winner = candidate_a if penalty_a < penalty_b else candidate_b
            stage = "sobolev_penalty"
        else:
            winner = min((candidate_a, candidate_b), key=self._deterministic_tie_key)
            if self._complexity(candidate_a) != self._complexity(candidate_b):
                stage = "complexity"
            elif candidate_a.canonical_fitted_expression != candidate_b.canonical_fitted_expression:
                stage = "canonical_expression"
            else:
                stage = "candidate_id"
        self._count(f"decisions:{stage}", context)
        changed = int(winner.candidate_id) != int(base_winner.candidate_id)
        if changed:
            self._count("sobolev_changed_winner", context)
        return self._finish(
            winner,
            candidate_b if winner is candidate_a else candidate_a,
            stage,
            context,
            gap,
            epsilon,
            True,
            penalty_a,
            penalty_b,
            None,
            base_winner,
        )

    def best(
        self,
        candidates: Sequence[ComparableCandidate],
        *,
        context: str,
    ) -> ComparableCandidate:
        return self.select_top(candidates, 1, context=context)[0]

    def select_top(
        self,
        candidates: Sequence[ComparableCandidate],
        count: int,
        *,
        context: str,
    ) -> list[ComparableCandidate]:
        """Select a deterministic top-k under a possibly non-transitive gate."""

        if count < 0:
            raise ValueError("count must be non-negative")
        remaining = sorted(list(candidates), key=self._stable_identity_key)
        selected: list[ComparableCandidate] = []
        while remaining and len(selected) < count:
            winner = remaining[0]
            for challenger in remaining[1:]:
                winner = self.compare(winner, challenger, context=context).winner
            selected.append(winner)
            remaining.remove(winner)
        return selected

    def stats_dict(self) -> dict[str, int | float]:
        return dict(sorted(self.stats.items()))

    def _finish(
        self,
        winner: ComparableCandidate,
        loser: ComparableCandidate,
        stage: str,
        context: str,
        gap: float | None,
        epsilon: float | None,
        triggered: bool,
        penalty_a: float | None,
        penalty_b: float | None,
        fallback: str | None,
        base_winner: ComparableCandidate,
    ) -> ComparisonResult:
        result = ComparisonResult(
            winner=winner,
            loser=loser,
            decision_stage=stage,
            context=context,
            base_reward_gap=gap,
            epsilon_threshold=epsilon,
            sobolev_triggered=triggered,
            penalty_a=penalty_a,
            penalty_b=penalty_b,
            fallback_reason=fallback,
            base_winner_id=int(base_winner.candidate_id),
            winner_changed_from_base=(
                int(winner.candidate_id) != int(base_winner.candidate_id)
            ),
        )
        if self.trace_sink is not None:
            self.trace_sink(result)
        return result

    def _count(self, name: str, context: str) -> None:
        self.stats[name] += 1
        self.stats[f"{context}:{name}"] += 1

    @staticmethod
    def _finite_reward(candidate: ComparableCandidate) -> float:
        value = float(candidate.base_reward)
        return value if np.isfinite(value) else -math.inf

    @staticmethod
    def _complexity(candidate: ComparableCandidate) -> float:
        value = float(candidate.complexity)
        return value if np.isfinite(value) else math.inf

    @staticmethod
    def _finite_penalty(view: LazySobolevView) -> float | None:
        if view.penalty is None:
            return None
        value = float(view.penalty)
        return value if np.isfinite(value) else None

    @classmethod
    def _base_key(cls, candidate: ComparableCandidate) -> tuple[Any, ...]:
        return (
            not bool(candidate.valid),
            -cls._finite_reward(candidate),
            cls._complexity(candidate),
            str(candidate.canonical_fitted_expression),
            int(candidate.candidate_id),
        )

    @classmethod
    def _deterministic_tie_key(
        cls, candidate: ComparableCandidate
    ) -> tuple[Any, ...]:
        return (
            cls._complexity(candidate),
            str(candidate.canonical_fitted_expression),
            int(candidate.candidate_id),
        )

    @staticmethod
    def _stable_identity_key(candidate: ComparableCandidate) -> tuple[Any, ...]:
        return (
            int(candidate.candidate_id),
            str(candidate.canonical_fitted_expression),
            str(candidate.raw_expression),
        )
