"""Low-novelty targeted mutation-site selector for SN-GP-v2."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from ...nd2py import nd2py as nd
from .sn_impact import (
    DEFAULT_SN_MUTATION_IMPACT_EPSILON,
    TermImpactResult,
    compute_term_impacts,
)
from .term_provenance import NodePath, ProvenanceResult, path_text


DEFAULT_SN_MUTATION_DELTA = 0.05
DEFAULT_SN_MUTATION_GAMMA = 2.0


@dataclass(frozen=True)
class NodePriority:
    path: NodePath
    associated_term_ids: tuple[int, ...]
    q_value: float
    weight: float
    depth: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": path_text(self.path),
            "associated_term_ids": list(self.associated_term_ids),
            "q_value": self.q_value,
            "weight": self.weight,
            "depth": self.depth,
        }


@dataclass(frozen=True)
class MutationSiteSelection:
    selected_node_paths: tuple[NodePath, ...]
    original_node_paths: tuple[NodePath, ...]
    term_priorities: tuple[float, ...]
    node_priorities: tuple[NodePriority, ...]
    targeted: bool
    fallback_reason: str | None
    selected_low_novelty_term_ids: tuple[int, ...]
    provenance_success: bool
    redundancy_priorities: tuple[float, ...] = ()
    term_coefficients: tuple[float, ...] = ()
    term_norms: tuple[float, ...] = ()
    term_impacts: tuple[float, ...] = ()
    normalized_impacts: tuple[float, ...] = ()
    term_eligibility: tuple[bool, ...] = ()
    selected_eligible_term_ids: tuple[int, ...] = ()
    impact_aware: bool = False

    @property
    def selected_node_path(self) -> NodePath | None:
        return self.selected_node_paths[0] if self.selected_node_paths else None

    def as_dict(self) -> dict[str, Any]:
        return {
            "selected_node_paths": [path_text(value) for value in self.selected_node_paths],
            "original_node_paths": [path_text(value) for value in self.original_node_paths],
            "term_priorities": list(self.term_priorities),
            "node_priorities": [value.as_dict() for value in self.node_priorities],
            "targeted": self.targeted,
            "fallback_reason": self.fallback_reason,
            "selected_low_novelty_term_ids": list(
                self.selected_low_novelty_term_ids
            ),
            "provenance_success": self.provenance_success,
            "redundancy_priorities": list(self.redundancy_priorities),
            "term_coefficients": list(self.term_coefficients),
            "term_norms": list(self.term_norms),
            "term_impacts": list(self.term_impacts),
            "normalized_impacts": list(self.normalized_impacts),
            "term_eligibility": list(self.term_eligibility),
            "selected_eligible_term_ids": list(
                self.selected_eligible_term_ids
            ),
            "impact_aware": self.impact_aware,
        }


class SNMutationTargetSelector:
    """Sample AST paths with weights ``delta + q_v**gamma``."""

    def __init__(
        self,
        *,
        tau: float,
        delta: float = DEFAULT_SN_MUTATION_DELTA,
        gamma: float = DEFAULT_SN_MUTATION_GAMMA,
        impact_beta: float = 0.0,
        impact_epsilon: float = DEFAULT_SN_MUTATION_IMPACT_EPSILON,
        max_normalized_impact: float = 1.0,
    ) -> None:
        if not 0 < tau <= 1:
            raise ValueError("tau must lie in (0, 1]")
        if delta <= 0 or gamma <= 0:
            raise ValueError("mutation delta and gamma must be positive")
        if impact_beta < 0 or not np.isfinite(impact_beta):
            raise ValueError("mutation impact beta must be finite and non-negative")
        if impact_epsilon <= 0 or not np.isfinite(impact_epsilon):
            raise ValueError("mutation impact epsilon must be positive and finite")
        if not 0 <= max_normalized_impact <= 1:
            raise ValueError("maximum normalized impact must lie in [0,1]")
        self.tau = float(tau)
        self.delta = float(delta)
        self.gamma = float(gamma)
        self.impact_beta = float(impact_beta)
        self.impact_epsilon = float(impact_epsilon)
        self.max_normalized_impact = float(max_normalized_impact)

    def select_site(
        self,
        *,
        individual: Any,
        term_novelties: Sequence[float],
        provenance: ProvenanceResult,
        rng: np.random.Generator,
        original_node_path: NodePath,
        eligible_paths: Sequence[NodePath] | None = None,
        term_coefficients: Sequence[float] | None = None,
        term_norms: Sequence[float] | None = None,
    ) -> MutationSiteSelection:
        return self.select_sites(
            individual=individual,
            term_novelties=term_novelties,
            provenance=provenance,
            rng=rng,
            original_node_paths=(original_node_path,),
            selection_count=1,
            eligible_paths=eligible_paths,
            term_coefficients=term_coefficients,
            term_norms=term_norms,
        )

    def select_sites(
        self,
        *,
        individual: Any,
        term_novelties: Sequence[float],
        provenance: ProvenanceResult,
        rng: np.random.Generator,
        original_node_paths: Sequence[NodePath],
        selection_count: int,
        eligible_paths: Sequence[NodePath] | None = None,
        term_coefficients: Sequence[float] | None = None,
        term_norms: Sequence[float] | None = None,
    ) -> MutationSiteSelection:
        tree = individual.eqtree if hasattr(individual, "eqtree") else individual
        originals = tuple(tuple(value) for value in original_node_paths)
        if selection_count <= 0:
            return self._fallback(originals, "original_selector_selected_no_sites", provenance)
        if not provenance.success:
            return self._fallback(
                originals,
                f"provenance_failure:{provenance.failure_reason or 'unknown'}",
                provenance,
            )
        if provenance.raw_expression != tree.to_str(number_format=".17g"):
            return self._fallback(originals, "stale_provenance_expression", provenance)

        novelty = np.asarray(term_novelties, dtype=float)
        if (
            novelty.ndim != 1
            or len(novelty) != len(provenance.terms)
            or not np.all(np.isfinite(novelty))
        ):
            return self._fallback(originals, "invalid_term_novelties", provenance)
        redundancy = np.maximum(0.0, 1.0 - novelty / self.tau)
        redundancy[np.abs(redundancy) < 1e-15] = 0.0
        low_term_ids = {int(index) for index in np.flatnonzero(redundancy > 0)}
        if not low_term_ids:
            return self._fallback(
                originals,
                "all_terms_at_or_above_tau",
                provenance,
                priorities=tuple(float(value) for value in redundancy),
                redundancy=tuple(float(value) for value in redundancy),
            )

        impact: TermImpactResult | None = None
        impact_aware = self.impact_beta > 0
        if impact_aware:
            if term_coefficients is None or term_norms is None:
                return self._fallback(
                    originals,
                    "missing_term_impact_inputs",
                    provenance,
                    redundancy=tuple(float(value) for value in redundancy),
                    impact_aware=True,
                )
            try:
                impact = compute_term_impacts(
                    term_coefficients,
                    novelty,
                    term_norms,
                    self.impact_epsilon,
                )
            except ValueError as error:
                return self._fallback(
                    originals,
                    f"invalid_term_impacts:{error}",
                    provenance,
                    redundancy=tuple(float(value) for value in redundancy),
                    impact_aware=True,
                )
            if impact.all_impacts_near_zero:
                return self._fallback(
                    originals,
                    "all_term_impacts_near_zero",
                    provenance,
                    redundancy=tuple(float(value) for value in redundancy),
                    impact=impact,
                    impact_aware=True,
                )
            normalized = np.asarray(impact.normalized_impacts, dtype=float)
            eligibility = (novelty < self.tau) & (
                normalized <= self.max_normalized_impact
            )
            priorities = np.zeros_like(redundancy)
            priorities[eligibility] = redundancy[eligibility] / np.power(
                normalized[eligibility] + self.impact_epsilon,
                self.impact_beta,
            )
            eligible_term_ids = {
                int(index) for index in np.flatnonzero(eligibility)
            }
            if not eligible_term_ids:
                return self._fallback(
                    originals,
                    "no_impact_eligible_low_novelty_terms",
                    provenance,
                    priorities=tuple(float(value) for value in priorities),
                    redundancy=tuple(float(value) for value in redundancy),
                    impact=impact,
                    eligibility=tuple(bool(value) for value in eligibility),
                    impact_aware=True,
                )
        else:
            # beta=0 is the exact frozen v2 novelty-only selector, regardless
            # of the configured impact threshold.
            priorities = redundancy.copy()
            eligibility = priorities > 0
            eligible_term_ids = low_term_ids
        if not np.all(np.isfinite(priorities)):
            return self._fallback(
                originals,
                "nonfinite_safe_term_priorities",
                provenance,
                redundancy=tuple(float(value) for value in redundancy),
                impact=impact,
                eligibility=tuple(bool(value) for value in eligibility),
                impact_aware=impact_aware,
            )

        all_paths = tuple(path for path, _ in _walk(tree, ()))
        eligible = set(all_paths if eligible_paths is None else map(tuple, eligible_paths))
        eligible.intersection_update(all_paths)
        # A targeted replacement of the whole tree is used only if no other
        # legal AST path exists.  The original selector remains untouched on fallback.
        if () in eligible and len(eligible) > 1 and selection_count <= len(eligible) - 1:
            eligible.remove(())
        if not eligible:
            return self._fallback(
                originals,
                "no_eligible_mutable_site",
                provenance,
                priorities=tuple(float(value) for value in priorities),
                redundancy=tuple(float(value) for value in redundancy),
                impact=impact,
                eligibility=tuple(bool(value) for value in eligibility),
                impact_aware=impact_aware,
            )

        node_priorities: list[NodePriority] = []
        for path in eligible:
            associated = tuple(provenance.associated_terms(path))
            q_value = max(
                (float(priorities[index]) for index in associated), default=0.0
            )
            node_priorities.append(
                NodePriority(
                    path=path,
                    associated_term_ids=associated,
                    q_value=q_value,
                    weight=self.delta + q_value**self.gamma,
                    depth=len(path),
                )
            )
        if not any(
            set(value.associated_term_ids).intersection(eligible_term_ids)
            for value in node_priorities
        ):
            return self._fallback(
                originals,
                (
                    "impact_eligible_terms_have_no_eligible_site"
                    if impact_aware
                    else "low_novelty_terms_have_no_eligible_site"
                ),
                provenance,
                priorities=tuple(float(value) for value in priorities),
                redundancy=tuple(float(value) for value in redundancy),
                impact=impact,
                eligibility=tuple(bool(value) for value in eligibility),
                impact_aware=impact_aware,
            )

        # Ordering does not alter the weights, but fixes cumulative sampling
        # and makes equal-priority diagnostics favor deep, local paths.
        available = sorted(
            node_priorities,
            key=lambda value: (
                -value.q_value,
                -value.depth,
                len(value.associated_term_ids),
                value.path,
            ),
        )
        chosen: list[NodePriority] = []
        for _ in range(min(selection_count, len(available))):
            weights = np.asarray([value.weight for value in available], dtype=float)
            draw = float(rng.random()) * float(weights.sum())
            index = min(int(np.searchsorted(np.cumsum(weights), draw, side="right")), len(available) - 1)
            chosen.append(available.pop(index))
        chosen.sort(key=lambda value: (-value.depth, value.path))
        selected_low = sorted(
            {
                term_id
                for value in chosen
                for term_id in value.associated_term_ids
                if term_id in low_term_ids
            }
        )
        selected_eligible = sorted(
            {
                term_id
                for value in chosen
                for term_id in value.associated_term_ids
                if term_id in eligible_term_ids
            }
        )
        return MutationSiteSelection(
            selected_node_paths=tuple(value.path for value in chosen),
            original_node_paths=originals,
            term_priorities=tuple(float(value) for value in priorities),
            node_priorities=tuple(node_priorities),
            targeted=True,
            fallback_reason=None,
            selected_low_novelty_term_ids=tuple(selected_low),
            provenance_success=True,
            redundancy_priorities=tuple(float(value) for value in redundancy),
            term_coefficients=(
                () if impact is None else impact.coefficients
            ),
            term_norms=(() if impact is None else impact.term_norms),
            term_impacts=(() if impact is None else impact.impacts),
            normalized_impacts=(
                () if impact is None else impact.normalized_impacts
            ),
            term_eligibility=tuple(bool(value) for value in eligibility),
            selected_eligible_term_ids=tuple(selected_eligible),
            impact_aware=impact_aware,
        )

    @staticmethod
    def _fallback(
        originals: tuple[NodePath, ...],
        reason: str,
        provenance: ProvenanceResult,
        priorities: tuple[float, ...] = (),
        redundancy: tuple[float, ...] = (),
        impact: TermImpactResult | None = None,
        eligibility: tuple[bool, ...] = (),
        impact_aware: bool = False,
    ) -> MutationSiteSelection:
        return MutationSiteSelection(
            selected_node_paths=originals,
            original_node_paths=originals,
            term_priorities=priorities,
            node_priorities=(),
            targeted=False,
            fallback_reason=reason,
            selected_low_novelty_term_ids=(),
            provenance_success=provenance.success,
            redundancy_priorities=redundancy,
            term_coefficients=(() if impact is None else impact.coefficients),
            term_norms=(() if impact is None else impact.term_norms),
            term_impacts=(() if impact is None else impact.impacts),
            normalized_impacts=(
                () if impact is None else impact.normalized_impacts
            ),
            term_eligibility=eligibility,
            selected_eligible_term_ids=(),
            impact_aware=impact_aware,
        )


def _walk(node: nd.Symbol, path: NodePath):
    yield path, node
    for index, operand in enumerate(node.operands):
        yield from _walk(operand, path + (index,))
