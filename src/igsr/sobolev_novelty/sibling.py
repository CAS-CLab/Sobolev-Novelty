"""Deterministic base-quality-gated ranking for IGSR sibling states."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from random import Random
from typing import Any, Mapping, Sequence, TypeVar

from .adapter import NoveltyDiagnostics


@dataclass(frozen=True)
class SiblingRankRecord:
    """One JSON-friendly row in the deterministic sibling ordering."""

    rank: int
    base_rank: int
    node_id: str
    validation_mse: float
    within_quality_gate: bool
    novelty_success: bool
    novelty_penalty: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SiblingExpansionDecision:
    """Auditable choice for one MCTS expansion from an untried sibling set."""

    selected_node_id: str
    base_random_node_id: str
    novelty_applied: bool
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


_CandidateT = TypeVar("_CandidateT")


def select_sibling_for_expansion(
    candidates: Sequence[_CandidateT],
    ranking_by_node_id: Mapping[str, SiblingRankRecord | Mapping[str, Any]],
    rng: Random,
    *,
    node_id_field: str = "node_id",
) -> tuple[_CandidateT, SiblingExpansionDecision]:
    """Choose an SN candidate while consuming the same RNG draw as Base.

    The Base random choice is returned when no untried candidate both belongs
    to the original full-batch quality gate and has a successful Sobolev
    diagnostic.  Thus a total Sobolev failure cannot perturb Base search.
    """

    if not candidates:
        raise ValueError("candidates must be non-empty")
    base_choice = rng.choice(list(candidates))
    base_node_id = str(_field(base_choice, node_id_field))

    eligible: list[tuple[int, float, str, _CandidateT]] = []
    for candidate in candidates:
        node_id = str(_field(candidate, node_id_field))
        raw_record = ranking_by_node_id.get(node_id)
        if raw_record is None:
            continue
        if isinstance(raw_record, SiblingRankRecord):
            rank = raw_record.rank
            validation_mse = raw_record.validation_mse
            within_gate = raw_record.within_quality_gate
            novelty_success = raw_record.novelty_success
        else:
            rank = int(raw_record["rank"])
            validation_mse = float(raw_record["validation_mse"])
            within_gate = bool(raw_record["within_quality_gate"])
            novelty_success = bool(raw_record["novelty_success"])
        if within_gate and novelty_success:
            eligible.append((rank, validation_mse, node_id, candidate))

    if not eligible:
        return base_choice, SiblingExpansionDecision(
            selected_node_id=base_node_id,
            base_random_node_id=base_node_id,
            novelty_applied=False,
            reason="no_successful_untried_candidate_in_frozen_quality_gate",
        )

    _, _, selected_node_id, selected = min(eligible, key=lambda row: row[:3])
    return selected, SiblingExpansionDecision(
        selected_node_id=selected_node_id,
        base_random_node_id=base_node_id,
        novelty_applied=selected_node_id != base_node_id,
        reason="successful_sobolev_candidate_in_frozen_quality_gate",
    )


def rank_siblings(
    candidates: Sequence[Any],
    diagnostics_by_node_id: Mapping[str, NoveltyDiagnostics | Mapping[str, Any]],
    *,
    relative_mse_tolerance: float = 0.01,
    absolute_mse_tolerance: float = 0.0,
    max_failure_penalty: float = 1.0,
    node_id_field: str = "node_id",
    validation_mse_field: str = "mse_after_total",
) -> tuple[SiblingRankRecord, ...]:
    """Rank near-equal siblings by novelty while preserving base ordering elsewhere.

    The quality gate is ``best_mse + max(abs_tol, rel_tol*abs(best_mse))``.
    Inside that gate, lower EIC Sobolev penalty wins, followed by validation
    MSE and stable ``node_id``.  Outside it, only validation MSE and node ID are
    used.  Missing/failed novelty receives ``max_failure_penalty``.

    The function neither mutates candidates nor reads test-set metrics.
    """

    if not candidates:
        return ()
    if relative_mse_tolerance < 0 or absolute_mse_tolerance < 0:
        raise ValueError("MSE tolerances must be non-negative")
    if not math.isfinite(max_failure_penalty) or max_failure_penalty < 0:
        raise ValueError("max_failure_penalty must be finite and non-negative")

    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for candidate in candidates:
        node_id_value = _field(candidate, node_id_field)
        if node_id_value is None or not str(node_id_value):
            raise ValueError(f"candidate has no non-empty {node_id_field!r}")
        node_id = str(node_id_value)
        if node_id in seen:
            raise ValueError(f"duplicate node_id {node_id!r}")
        seen.add(node_id)
        mse = float(_field(candidate, validation_mse_field))
        if math.isnan(mse) or mse < 0:
            raise ValueError(
                f"candidate {node_id!r} has invalid validation MSE {mse!r}"
            )
        success, penalty = _diagnostic_score(
            diagnostics_by_node_id.get(node_id), max_failure_penalty
        )
        rows.append(
            {
                "node_id": node_id,
                "mse": mse,
                "novelty_success": success,
                "penalty": penalty,
            }
        )

    finite_mses = [row["mse"] for row in rows if math.isfinite(row["mse"])]
    best_mse = min(finite_mses) if finite_mses else math.inf
    gate_width = (
        max(absolute_mse_tolerance, relative_mse_tolerance * abs(best_mse))
        if math.isfinite(best_mse)
        else 0.0
    )
    cutoff = best_mse + gate_width
    for row in rows:
        row["within_gate"] = math.isfinite(row["mse"]) and row["mse"] <= cutoff

    base_order = sorted(rows, key=lambda row: (row["mse"], row["node_id"]))
    base_rank = {row["node_id"]: index + 1 for index, row in enumerate(base_order)}
    gated = sorted(
        (row for row in rows if row["within_gate"]),
        key=lambda row: (row["penalty"], row["mse"], row["node_id"]),
    )
    outside = sorted(
        (row for row in rows if not row["within_gate"]),
        key=lambda row: (row["mse"], row["node_id"]),
    )
    ordered = gated + outside
    return tuple(
        SiblingRankRecord(
            rank=index + 1,
            base_rank=base_rank[row["node_id"]],
            node_id=row["node_id"],
            validation_mse=row["mse"],
            within_quality_gate=row["within_gate"],
            novelty_success=row["novelty_success"],
            novelty_penalty=row["penalty"],
        )
        for index, row in enumerate(ordered)
    )


def _field(candidate: Any, name: str) -> Any:
    if isinstance(candidate, Mapping):
        if name not in candidate:
            raise ValueError(f"candidate mapping has no field {name!r}")
        return candidate[name]
    if not hasattr(candidate, name):
        raise ValueError(
            f"candidate {type(candidate).__name__} has no attribute {name!r}"
        )
    return getattr(candidate, name)


def _diagnostic_score(
    diagnostic: NoveltyDiagnostics | Mapping[str, Any] | None,
    max_failure_penalty: float,
) -> tuple[bool, float]:
    if diagnostic is None:
        return False, max_failure_penalty
    if isinstance(diagnostic, NoveltyDiagnostics):
        success = diagnostic.success
        penalty = diagnostic.penalty
    else:
        success = bool(diagnostic.get("success", False))
        penalty = diagnostic.get("penalty", max_failure_penalty)
    try:
        penalty_float = float(penalty)
    except (TypeError, ValueError):
        return False, max_failure_penalty
    if not success or not math.isfinite(penalty_float):
        return False, max_failure_penalty
    return True, min(max(penalty_float, 0.0), max_failure_penalty)
