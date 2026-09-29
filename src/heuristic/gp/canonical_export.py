from __future__ import annotations
import math
from typing import Any

CANDIDATE_ROOT_KEYS = (
    "method_selected_best",
    "base_quality_best_ever",
    "best_by_base_reward",
    "best_by_sn_comparator",
    "exported_candidate",
    "archive_export_candidate",
    "archive_export_pareto_candidate",
    "archive_export_path",
)

def collect_candidates(value: Any, output: list[dict[str, Any]]) -> None:
    if isinstance(value, dict):
        required = (
            "exported_expression",
            "base_reward",
            "complexity",
            "search_internal_r2",
        )
        if all(key in value for key in required):
            expression = value.get("exported_expression")
            reward = value.get("base_reward")
            if expression and reward is not None and math.isfinite(float(reward)):
                output.append(value)
        for child in value.values():
            collect_candidates(child, output)
    elif isinstance(value, list):
        for child in value:
            collect_candidates(child, output)

def candidate_pool(result: dict[str, Any]) -> list[dict[str, Any]]:
    collected: list[dict[str, Any]] = []
    for key in CANDIDATE_ROOT_KEYS:
        collect_candidates(result.get(key), collected)
    unique: dict[str, dict[str, Any]] = {}
    for candidate in collected:
        expression = str(candidate["exported_expression"])
        previous = unique.get(expression)
        if previous is None or candidate_key(candidate) < candidate_key(previous):
            unique[expression] = candidate
    return list(unique.values())

def candidate_key(candidate: dict[str, Any]) -> tuple[Any, ...]:
    return (
        -float(candidate["base_reward"]),
        int(candidate["complexity"]),
        -float(candidate["search_internal_r2"]),
        str(candidate["exported_expression"]),
        int(candidate.get("candidate_id", -1)),
    )
