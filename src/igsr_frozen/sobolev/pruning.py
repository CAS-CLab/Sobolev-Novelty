"""Correctness-first term-level Sobolev prune-and-refit orchestration."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Callable
import time

from .types import EvaluationResult, TermSpec


@dataclass(frozen=True)
class PruningConfig:
    """Central termination and acceptance choices for term pruning."""

    threshold: float
    max_prunes: int
    acceptance_tolerance: float = 0.0
    floating_tolerance: float = 1e-12
    excluded_term_indices: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if not 0 < self.threshold <= 1:
            raise ValueError("threshold must lie in (0, 1]")
        if self.max_prunes < 0:
            raise ValueError("max_prunes must be non-negative")
        if self.acceptance_tolerance < 0 or self.floating_tolerance < 0:
            raise ValueError("acceptance tolerances must be non-negative")
        if any(index < 0 for index in self.excluded_term_indices):
            raise ValueError("excluded term indices must be non-negative")


@dataclass
class RefitGeometryHint:
    """Non-serialized AST/term references for pruning geometry reuse."""

    expression: object
    symbols: tuple[object, ...]
    terms: tuple[TermSpec, ...]
    parent_geometry_key: str | None
    removed_index: int


@dataclass
class RefitResult:
    """Result of refitting the retained basis terms with baseline semantics."""

    expression: object | None = None
    coefficients: list[float] = field(default_factory=list)
    r2: float | None = None
    complexity: int | None = None
    eic: float | None = None
    base_reward: float | None = None
    success: bool = False
    failure_type: str = ""
    failure_message: str = ""
    coefficient_fitting_seconds: float = 0.0
    eic_seconds: float = 0.0
    geometry_hint: RefitGeometryHint | None = None


@dataclass
class PruneStep:
    """One auditable delete/refit/accept decision."""

    step: int
    removed_index: int
    removed_term: str
    removed_coefficient: float
    removed_novelty: float
    removed_term_norm: float
    deletion_impact: float
    expression_before: str
    expression_after_refit: str | None
    base_reward_before: float
    base_reward_after: float | None
    r2_before: float | None
    r2_after: float | None
    complexity_before: int | None
    complexity_after: int | None
    accepted: bool
    decision: str
    refit_coefficient_fitting_seconds: float = 0.0
    refit_eic_seconds: float = 0.0
    evaluator_success_after: bool | None = None
    evaluator_failure_type_after: str | None = None
    geometry_key_before: str | None = None
    geometry_key_after: str | None = None
    geometry_reused: bool | None = None
    geometry_reuse_mode: str | None = None
    geometry_fallback_reason: str | None = None
    geometry_recompute_seconds: float = 0.0


@dataclass
class PruningResult:
    """Full raw-to-final pruning result and termination reason."""

    initial_expression: str
    final_expression: str
    initial_fit: RefitResult
    final_fit: RefitResult
    initial_analysis: EvaluationResult
    final_analysis: EvaluationResult
    steps: list[PruneStep] = field(default_factory=list)
    accepted_prunes: int = 0
    rejected_prunes: int = 0
    success: bool = True
    termination_reason: str = ""
    failure_type: str = ""
    failure_message: str = ""

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-safe dictionary, including nested evaluator records."""

        return {
            "initial_expression": self.initial_expression,
            "final_expression": self.final_expression,
            "initial_fit": _refit_as_dict(self.initial_fit),
            "final_fit": _refit_as_dict(self.final_fit),
            "initial_analysis": self.initial_analysis.as_dict(),
            "final_analysis": self.final_analysis.as_dict(),
            "steps": [asdict(step) for step in self.steps],
            "accepted_prunes": self.accepted_prunes,
            "rejected_prunes": self.rejected_prunes,
            "success": self.success,
            "termination_reason": self.termination_reason,
            "failure_type": self.failure_type,
            "failure_message": self.failure_message,
        }


RefitCallback = Callable[[RefitResult, EvaluationResult, int], RefitResult]
ReevaluateCallback = Callable[[RefitResult], EvaluationResult]


def deletion_impacts(analysis: EvaluationResult) -> list[float]:
    """Compute ``D_i = |b_i| * nu_i * ||Psi(phi_i)||`` for every term."""

    lengths = {
        len(analysis.terms),
        len(analysis.coefficients),
        len(analysis.term_novelties),
        len(analysis.term_norms),
    }
    if len(lengths) != 1:
        raise ValueError(
            "Pruning diagnostics have inconsistent lengths: "
            f"terms={len(analysis.terms)}, coefficients={len(analysis.coefficients)}, "
            f"novelties={len(analysis.term_novelties)}, norms={len(analysis.term_norms)}"
        )
    return [
        abs(float(coefficient)) * float(novelty) * float(norm)
        for coefficient, novelty, norm in zip(
            analysis.coefficients,
            analysis.term_novelties,
            analysis.term_norms,
            strict=True,
        )
    ]


def prune_and_refit(
    initial_fit: RefitResult,
    initial_analysis: EvaluationResult,
    config: PruningConfig,
    refit_callback: RefitCallback,
    reevaluate_callback: ReevaluateCallback,
) -> PruningResult:
    """Iteratively delete the lowest-impact low-novelty term and refit.

    Acceptance uses the callback's *unpenalized* base reward.  A rejected
    deletion leaves the prior expression untouched.  Re-evaluation occurs only
    after an accepted deletion and supplies the diagnostics for the next step.
    """

    initial_expression = str(initial_fit.expression)
    result = PruningResult(
        initial_expression=initial_expression,
        final_expression=initial_expression,
        initial_fit=initial_fit,
        final_fit=initial_fit,
        initial_analysis=initial_analysis,
        final_analysis=initial_analysis,
    )
    if not initial_fit.success or initial_fit.base_reward is None:
        return _fail(result, "invalid_initial_fit", initial_fit.failure_message or "Initial fit is invalid")
    if not initial_analysis.success:
        return _fail(
            result,
            "initial_evaluator_failure",
            initial_analysis.failure_message or initial_analysis.failure_type.value,
        )
    if config.max_prunes == 0:
        result.termination_reason = "max_prunes_zero"
        return result

    current_fit = initial_fit
    current_analysis = initial_analysis
    for step_number in range(1, config.max_prunes + 1):
        if len(current_analysis.terms) <= 1:
            result.termination_reason = "single_term_remaining"
            break
        eligible = [
            index
            for index, novelty in enumerate(current_analysis.term_novelties)
            if (
                float(novelty) < config.threshold
                and index not in config.excluded_term_indices
            )
        ]
        if not eligible:
            has_low_novelty = any(
                float(novelty) < config.threshold
                for novelty in current_analysis.term_novelties
            )
            result.termination_reason = (
                "no_actionable_low_novelty_terms"
                if has_low_novelty
                else "no_low_novelty_terms"
            )
            break
        try:
            impacts = deletion_impacts(current_analysis)
        except ValueError as error:
            return _fail(result, "invalid_pruning_diagnostics", str(error))
        removed_index = min(eligible, key=lambda index: (impacts[index], index))
        candidate_fit = refit_callback(current_fit, current_analysis, removed_index)
        before_reward = float(current_fit.base_reward)
        step = PruneStep(
            step=step_number,
            removed_index=removed_index,
            removed_term=current_analysis.terms[removed_index],
            removed_coefficient=float(current_analysis.coefficients[removed_index]),
            removed_novelty=float(current_analysis.term_novelties[removed_index]),
            removed_term_norm=float(current_analysis.term_norms[removed_index]),
            deletion_impact=float(impacts[removed_index]),
            expression_before=str(current_fit.expression),
            expression_after_refit=str(candidate_fit.expression) if candidate_fit.expression is not None else None,
            base_reward_before=before_reward,
            base_reward_after=candidate_fit.base_reward,
            r2_before=current_fit.r2,
            r2_after=candidate_fit.r2,
            complexity_before=current_fit.complexity,
            complexity_after=candidate_fit.complexity,
            accepted=False,
            decision="",
            refit_coefficient_fitting_seconds=candidate_fit.coefficient_fitting_seconds,
            refit_eic_seconds=candidate_fit.eic_seconds,
            geometry_key_before=current_analysis.candidate_geometry_key,
        )
        result.steps.append(step)
        if not candidate_fit.success or candidate_fit.base_reward is None:
            step.decision = "refit_failure"
            result.termination_reason = "refit_failure"
            result.failure_type = candidate_fit.failure_type or "refit_failure"
            result.failure_message = candidate_fit.failure_message
            result.success = False
            break
        numerical_slack = config.floating_tolerance * max(1.0, abs(before_reward))
        minimum_reward = before_reward - config.acceptance_tolerance - numerical_slack
        if float(candidate_fit.base_reward) < minimum_reward:
            step.decision = "base_reward_decrease"
            result.rejected_prunes += 1
            result.termination_reason = "base_reward_decrease"
            break
        step.accepted = True
        step.decision = "accepted"
        result.accepted_prunes += 1
        current_fit = candidate_fit
        result.final_fit = current_fit
        result.final_expression = str(current_fit.expression)
        geometry_start = time.perf_counter()
        next_analysis = reevaluate_callback(current_fit)
        step.geometry_recompute_seconds = time.perf_counter() - geometry_start
        step.evaluator_success_after = next_analysis.success
        step.evaluator_failure_type_after = (
            None if next_analysis.success else next_analysis.failure_type.value
        )
        step.geometry_key_after = next_analysis.candidate_geometry_key
        step.geometry_reused = bool(
            next_analysis.incremental_hit or next_analysis.geometry_cache_hit
        )
        step.geometry_reuse_mode = next_analysis.geometry_reuse_mode
        step.geometry_fallback_reason = (
            next_analysis.incremental_fallback_reason
            or next_analysis.geometry_fallback_reason
        )
        result.final_analysis = next_analysis
        if not next_analysis.success:
            result.success = False
            result.termination_reason = "post_accept_evaluator_failure"
            result.failure_type = next_analysis.failure_type.value
            result.failure_message = next_analysis.failure_message
            break
        current_analysis = next_analysis
    else:
        result.termination_reason = "max_prunes_reached"
    if not result.termination_reason:
        result.termination_reason = "max_prunes_reached"
    return result


def _fail(result: PruningResult, failure_type: str, message: str) -> PruningResult:
    result.success = False
    result.termination_reason = failure_type
    result.failure_type = failure_type
    result.failure_message = message
    return result


def _refit_as_dict(result: RefitResult) -> dict[str, object]:
    value = asdict(result)
    value["expression"] = None if result.expression is None else str(result.expression)
    hint = result.geometry_hint
    value["geometry_hint"] = None if hint is None else {
        "expression": str(hint.expression),
        "symbols": [str(symbol) for symbol in hint.symbols],
        "terms": [term.display for term in hint.terms],
        "parent_geometry_key": hint.parent_geometry_key,
        "removed_index": hint.removed_index,
    }
    return value
