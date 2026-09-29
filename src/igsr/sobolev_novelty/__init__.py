"""Opt-in Sobolev-novelty diagnostics for IGSR candidates.

This package deliberately has no import-time dependency on the EIC repository.
Callers must either provide an explicit EIC ``src`` root or inject the EIC
components.  IGSR's optional ``diagnostic`` mode records these outputs without
using them in pruning, reward, or search ordering.
"""

from .adapter import (
    AdapterFailureType,
    EICComponents,
    FixedGeometry,
    NoveltyDiagnostics,
    SobolevNoveltyAdapter,
    load_eic_components,
)
from .sibling import (
    SiblingExpansionDecision,
    SiblingRankRecord,
    rank_siblings,
    select_sibling_for_expansion,
)
from .pruning import SobolevPruneCandidate, rank_prune_candidates
from .crossfit import CrossfitStability, repeated_crossfit_stability
from .pareto import (
    ParetoExpansionDecision,
    ParetoSiblingRecord,
    rank_pareto_siblings,
    select_pareto_sibling_for_expansion,
)

__all__ = [
    "AdapterFailureType",
    "EICComponents",
    "FixedGeometry",
    "NoveltyDiagnostics",
    "SiblingExpansionDecision",
    "SiblingRankRecord",
    "SobolevNoveltyAdapter",
    "load_eic_components",
    "rank_siblings",
    "select_sibling_for_expansion",
    "SobolevPruneCandidate",
    "rank_prune_candidates",
    "CrossfitStability",
    "repeated_crossfit_stability",
    "ParetoExpansionDecision",
    "ParetoSiblingRecord",
    "rank_pareto_siblings",
    "select_pareto_sibling_for_expansion",
]
