"""Pure selection logic for a bounded Sobolev shadow proposal slot."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
from typing import Any, Callable, Sequence


@dataclass(frozen=True)
class ShadowProposal:
    candidate: Any
    source_slot: int
    original_candidate_id: int
    coverage_gain: float
    base_qualified: bool


@dataclass(frozen=True)
class ShadowInjectionPlan:
    proposal: ShadowProposal
    replacement_index: int
    replacement_reason: str
    proposal_already_represented: bool = False


def plan_shadow_injection(
    children: Sequence[Any],
    proposals: Sequence[ShadowProposal],
    *,
    candidate_id: Callable[[Any], int],
    canonical_expression: Callable[[Any], str],
    base_key: Callable[[Any], tuple[Any, ...]],
    protected_candidate_ids: set[int] | frozenset[int],
    minimum_coverage_gain: float = 0.0,
    allow_represented_amplification: bool = False,
) -> ShadowInjectionPlan | None:
    """Select one shadow proposal and one replaceable population position.

    Better Base candidates have smaller ``base_key`` values. Coverage gain is
    maximized first; exact Base quality and stable identities break ties.
    """

    child_expressions = [canonical_expression(child) for child in children]
    expression_counts = Counter(child_expressions)
    represented = set(expression_counts)
    eligible = [
        proposal
        for proposal in proposals
        if proposal.base_qualified
        and math.isfinite(proposal.coverage_gain)
        and proposal.coverage_gain > minimum_coverage_gain
        and (
            canonical_expression(proposal.candidate) not in represented
            or (
                allow_represented_amplification
                and expression_counts[
                    canonical_expression(proposal.candidate)
                ] == 1
            )
        )
    ]
    if not eligible:
        return None
    selected = min(
        eligible,
        key=lambda proposal: (
            canonical_expression(proposal.candidate) in represented,
            -proposal.coverage_gain,
            base_key(proposal.candidate),
            canonical_expression(proposal.candidate),
            proposal.source_slot,
            candidate_id(proposal.candidate),
        ),
    )

    replaceable = [
        index
        for index, child in enumerate(children)
        if candidate_id(child) not in protected_candidate_ids
    ]
    if not replaceable:
        return None
    duplicate_indices = [
        index
        for index in replaceable
        if expression_counts[child_expressions[index]] > 1
    ]
    candidates = duplicate_indices if duplicate_indices else replaceable
    replacement_index = max(
        candidates,
        key=lambda index: (
            base_key(children[index]),
            child_expressions[index],
            candidate_id(children[index]),
        ),
    )
    return ShadowInjectionPlan(
        proposal=selected,
        replacement_index=replacement_index,
        replacement_reason=(
            "duplicate_phenotype" if duplicate_indices else "worst_unprotected"
        ),
        proposal_already_represented=(
            canonical_expression(selected.candidate) in represented
        ),
    )
