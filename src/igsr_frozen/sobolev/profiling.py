"""Small in-memory profiler aggregation for evaluator experiments."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import fields
from typing import Iterable

import numpy as np

from .types import DetailedRuntime, EvaluationResult


def summarize_profiles(results: Iterable[EvaluationResult]) -> dict[str, object]:
    """Aggregate success, algorithm, cache and timing statistics."""

    rows = list(results)
    timing: dict[str, list[float]] = defaultdict(list)
    for result in rows:
        for item in fields(DetailedRuntime):
            timing[item.name].append(float(getattr(result.detailed_runtime, item.name)))
    return {
        "candidate_count": len(rows),
        "success_count": sum(result.success for result in rows),
        "success_rate": (sum(result.success for result in rows) / len(rows)) if rows else 0.0,
        "algorithm_counts": {
            name: sum(result.algorithm_used == name for result in rows)
            for name in sorted({result.algorithm_used for result in rows})
        },
        "cache_hits": sum(result.cache_hits for result in rows),
        "cache_misses": sum(result.cache_misses for result in rows),
        "timing_total_seconds": {name: float(np.sum(value)) for name, value in timing.items()},
        "timing_mean_seconds": {name: float(np.mean(value)) for name, value in timing.items()},
    }
