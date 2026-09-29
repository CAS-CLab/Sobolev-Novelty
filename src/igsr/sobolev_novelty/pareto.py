"""Train-stability-guarded Sobolev-Pareto sibling replacement.

This module deliberately contains only deterministic, test-free decision
logic.  IGSR computes repeated cross-fit summaries from search-train rows and
passes them in; the held-out report split is not part of this API.

The paired Base choice is sampled first.  SN may replace it only with a
full-batch Pareto-frontier sibling that:

* is inside the frozen validation quality gate;
* has successful Sobolev and train-only cross-fit diagnostics;
* is no worse than the Base choice in cross-fit mean and worst-repeat NMSE;
* uses no more additive terms; and
* has a strictly lower Sobolev redundancy penalty.

Consequently a total diagnostic failure, an unstable candidate, or the lack of
a genuine Sobolev improvement reproduces Base's exact random choice while
still consuming exactly the same RNG draw.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from random import Random
from typing import Any, Mapping, Sequence, TypeVar


@dataclass(frozen=True)
class ParetoSiblingRecord:
    """Auditable full-batch record for one sibling candidate."""

    rank: int
    node_id: str
    validation_nmse: float
    term_count: int
    novelty_success: bool
    novelty_penalty: float
    crossfit_success: bool
    crossfit_nmse_mean: float
    crossfit_nmse_worst: float
    within_validation_gate: bool
    pareto_eligible: bool
    pareto_frontier: bool

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ParetoExpansionDecision:
    """One Base-preserving expansion decision and its reason."""

    selected_node_id: str
    base_random_node_id: str
    novelty_applied: bool
    reason: str
    eligible_frontier_node_ids: tuple[str, ...]
    held_out_rows_consulted: bool = False

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["eligible_frontier_node_ids"] = list(self.eligible_frontier_node_ids)
        return payload


_CandidateT = TypeVar("_CandidateT")
_FAILED_CROSSFIT_NMSE = 1.0e12


def rank_pareto_siblings(
    candidates: Sequence[Any],
    diagnostics_by_node_id: Mapping[str, Mapping[str, Any]],
    crossfit_by_node_id: Mapping[str, Mapping[str, Any]],
    *,
    relative_validation_nmse_tolerance: float = 0.01,
    absolute_validation_nmse_tolerance: float = 0.0,
    max_failure_penalty: float = 1.0,
) -> tuple[ParetoSiblingRecord, ...]:
    """Build the frozen validation/CV/novelty Pareto record for a sibling batch."""

    if not candidates:
        return ()
    _require_nonnegative_finite(
        relative_validation_nmse_tolerance,
        "relative_validation_nmse_tolerance",
    )
    _require_nonnegative_finite(
        absolute_validation_nmse_tolerance,
        "absolute_validation_nmse_tolerance",
    )
    _require_nonnegative_finite(max_failure_penalty, "max_failure_penalty")

    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for candidate in candidates:
        node_id = str(_field(candidate, "node_id") or "")
        if not node_id:
            raise ValueError("candidate has no non-empty node_id")
        if node_id in seen:
            raise ValueError(f"duplicate node_id {node_id!r}")
        seen.add(node_id)

        raw_metrics = _field(candidate, "metrics_after")
        if not isinstance(raw_metrics, Mapping) or "nmse" not in raw_metrics:
            raise ValueError(f"candidate {node_id!r} has no metrics_after.nmse")
        validation_nmse = _nonnegative_or_infinity(raw_metrics["nmse"])
        terms = _field(candidate, "terms_after")
        if isinstance(terms, (str, bytes)) or not isinstance(terms, Sequence):
            raise ValueError(f"candidate {node_id!r} has invalid terms_after")
        term_count = len(terms)

        novelty_success, novelty_penalty = _novelty_score(
            diagnostics_by_node_id.get(node_id), max_failure_penalty
        )
        crossfit_success, crossfit_mean, crossfit_worst = _crossfit_score(
            crossfit_by_node_id.get(node_id)
        )
        rows.append(
            {
                "node_id": node_id,
                "validation_nmse": validation_nmse,
                "term_count": term_count,
                "novelty_success": novelty_success,
                "novelty_penalty": novelty_penalty,
                "crossfit_success": crossfit_success,
                "crossfit_nmse_mean": crossfit_mean,
                "crossfit_nmse_worst": crossfit_worst,
            }
        )

    finite_validation = [
        row["validation_nmse"] for row in rows if math.isfinite(row["validation_nmse"])
    ]
    best_validation = min(finite_validation) if finite_validation else math.inf
    gate_width = (
        max(
            absolute_validation_nmse_tolerance,
            relative_validation_nmse_tolerance * abs(best_validation),
        )
        if math.isfinite(best_validation)
        else 0.0
    )
    cutoff = best_validation + gate_width
    for row in rows:
        within_gate = (
            math.isfinite(row["validation_nmse"]) and row["validation_nmse"] <= cutoff
        )
        row["within_validation_gate"] = within_gate
        row["pareto_eligible"] = (
            within_gate and row["novelty_success"] and row["crossfit_success"]
        )

    eligible = [row for row in rows if row["pareto_eligible"]]
    for row in rows:
        row["pareto_frontier"] = row["pareto_eligible"] and not any(
            _dominates(other, row)
            for other in eligible
            if other["node_id"] != row["node_id"]
        )

    ordered = sorted(
        rows,
        key=lambda row: (
            not row["pareto_frontier"],
            not row["pareto_eligible"],
            row["novelty_penalty"],
            row["crossfit_nmse_mean"],
            row["crossfit_nmse_worst"],
            row["validation_nmse"],
            row["term_count"],
            row["node_id"],
        ),
    )
    return tuple(
        ParetoSiblingRecord(rank=index + 1, **row) for index, row in enumerate(ordered)
    )


def select_pareto_sibling_for_expansion(
    candidates: Sequence[_CandidateT],
    ranking_by_node_id: Mapping[str, ParetoSiblingRecord | Mapping[str, Any]],
    rng: Random,
    *,
    relative_crossfit_mean_tolerance: float = 0.0,
    absolute_crossfit_mean_tolerance: float = 0.0,
    relative_crossfit_worst_tolerance: float = 0.0,
    absolute_crossfit_worst_tolerance: float = 0.0,
    minimum_penalty_gain: float = 0.0,
) -> tuple[_CandidateT, ParetoExpansionDecision]:
    """Consume Base's draw, then make only a guarded Sobolev replacement."""

    if not candidates:
        raise ValueError("candidates must be non-empty")
    for value, name in (
        (relative_crossfit_mean_tolerance, "relative_crossfit_mean_tolerance"),
        (absolute_crossfit_mean_tolerance, "absolute_crossfit_mean_tolerance"),
        (relative_crossfit_worst_tolerance, "relative_crossfit_worst_tolerance"),
        (absolute_crossfit_worst_tolerance, "absolute_crossfit_worst_tolerance"),
        (minimum_penalty_gain, "minimum_penalty_gain"),
    ):
        _require_nonnegative_finite(value, name)

    base_choice = rng.choice(list(candidates))
    base_node_id = str(_field(base_choice, "node_id"))
    base_record = _record(ranking_by_node_id.get(base_node_id))
    if base_record is None:
        return base_choice, _decision(
            base_node_id,
            base_node_id,
            False,
            "base_record_missing",
            (),
        )
    # The comparison is meaningful only when the paired Base draw has both
    # diagnostics.  Never turn a missing/failed Base score into an implicit
    # waiver that makes replacement easier: exact Base fallback is the
    # experiment's fail-closed contract.
    if not base_record.novelty_success:
        return base_choice, _decision(
            base_node_id,
            base_node_id,
            False,
            "base_novelty_diagnostic_failed",
            (),
        )
    if not base_record.crossfit_success:
        return base_choice, _decision(
            base_node_id,
            base_node_id,
            False,
            "base_crossfit_failed",
            (),
        )

    eligible: list[tuple[ParetoSiblingRecord, _CandidateT]] = []
    for candidate in candidates:
        node_id = str(_field(candidate, "node_id"))
        record = _record(ranking_by_node_id.get(node_id))
        if record is None or not record.pareto_frontier:
            continue
        if record.term_count > base_record.term_count:
            continue
        if record.crossfit_nmse_mean > _relative_cutoff(
            base_record.crossfit_nmse_mean,
            relative_crossfit_mean_tolerance,
            absolute_crossfit_mean_tolerance,
        ):
            continue
        if record.crossfit_nmse_worst > _relative_cutoff(
            base_record.crossfit_nmse_worst,
            relative_crossfit_worst_tolerance,
            absolute_crossfit_worst_tolerance,
        ):
            continue
        if not (
            record.novelty_penalty < base_record.novelty_penalty - minimum_penalty_gain
        ):
            continue
        eligible.append((record, candidate))

    eligible_ids = tuple(sorted(record.node_id for record, _ in eligible))
    if not eligible:
        return base_choice, _decision(
            base_node_id,
            base_node_id,
            False,
            "no_strict_sobolev_pareto_replacement",
            eligible_ids,
        )

    selected_record, selected = min(
        eligible,
        key=lambda pair: (
            pair[0].novelty_penalty,
            pair[0].crossfit_nmse_mean,
            pair[0].crossfit_nmse_worst,
            pair[0].validation_nmse,
            pair[0].term_count,
            pair[0].node_id,
        ),
    )
    if selected_record.node_id == base_node_id:
        return base_choice, _decision(
            base_node_id,
            base_node_id,
            False,
            "base_choice_is_best_guarded_pareto_candidate",
            eligible_ids,
        )
    return selected, _decision(
        selected_record.node_id,
        base_node_id,
        True,
        "strictly_lower_penalty_with_nonworse_train_crossfit_and_complexity",
        eligible_ids,
    )


def _dominates(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    fields = (
        "crossfit_nmse_mean",
        "crossfit_nmse_worst",
        "novelty_penalty",
        "term_count",
        "validation_nmse",
    )
    return all(left[field] <= right[field] for field in fields) and any(
        left[field] < right[field] for field in fields
    )


def _record(
    value: ParetoSiblingRecord | Mapping[str, Any] | None,
) -> ParetoSiblingRecord | None:
    if value is None:
        return None
    if isinstance(value, ParetoSiblingRecord):
        return value
    try:
        return ParetoSiblingRecord(
            rank=int(value["rank"]),
            node_id=str(value["node_id"]),
            validation_nmse=float(value["validation_nmse"]),
            term_count=int(value["term_count"]),
            novelty_success=bool(value["novelty_success"]),
            novelty_penalty=float(value["novelty_penalty"]),
            crossfit_success=bool(value["crossfit_success"]),
            crossfit_nmse_mean=float(value["crossfit_nmse_mean"]),
            crossfit_nmse_worst=float(value["crossfit_nmse_worst"]),
            within_validation_gate=bool(value["within_validation_gate"]),
            pareto_eligible=bool(value["pareto_eligible"]),
            pareto_frontier=bool(value["pareto_frontier"]),
        )
    except (KeyError, TypeError, ValueError, OverflowError):
        return None


def _novelty_score(
    diagnostic: Mapping[str, Any] | None,
    max_failure_penalty: float,
) -> tuple[bool, float]:
    if not isinstance(diagnostic, Mapping) or diagnostic.get("success") is not True:
        return False, max_failure_penalty
    try:
        penalty = float(diagnostic["penalty"])
    except (KeyError, TypeError, ValueError, OverflowError):
        return False, max_failure_penalty
    if not math.isfinite(penalty) or penalty < 0.0:
        return False, max_failure_penalty
    return True, min(penalty, max_failure_penalty)


def _crossfit_score(
    crossfit: Mapping[str, Any] | None,
) -> tuple[bool, float, float]:
    if not isinstance(crossfit, Mapping) or crossfit.get("success") is not True:
        return False, _FAILED_CROSSFIT_NMSE, _FAILED_CROSSFIT_NMSE
    try:
        mean = float(crossfit["nmse_mean"])
        worst = float(crossfit["nmse_worst"])
    except (KeyError, TypeError, ValueError, OverflowError):
        return False, _FAILED_CROSSFIT_NMSE, _FAILED_CROSSFIT_NMSE
    if (
        not math.isfinite(mean)
        or not math.isfinite(worst)
        or mean < 0.0
        or worst < 0.0
        or worst < mean
    ):
        return False, _FAILED_CROSSFIT_NMSE, _FAILED_CROSSFIT_NMSE
    return True, mean, worst


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


def _nonnegative_or_infinity(value: Any) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"metric is not numeric: {value!r}") from error
    if math.isnan(parsed) or parsed < 0.0:
        raise ValueError(f"metric must be non-negative or infinity, got {parsed!r}")
    return parsed


def _require_nonnegative_finite(value: float, name: str) -> None:
    if not math.isfinite(float(value)) or float(value) < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")


def _relative_cutoff(base: float, relative: float, absolute: float) -> float:
    return base + max(absolute, relative * abs(base))


def _decision(
    selected: str,
    base: str,
    applied: bool,
    reason: str,
    eligible: tuple[str, ...],
) -> ParetoExpansionDecision:
    return ParetoExpansionDecision(
        selected_node_id=selected,
        base_random_node_id=base,
        novelty_applied=applied,
        reason=reason,
        eligible_frontier_node_ids=eligible,
        held_out_rows_consulted=False,
    )
