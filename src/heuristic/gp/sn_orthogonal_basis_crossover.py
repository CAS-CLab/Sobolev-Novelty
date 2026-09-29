"""Greedy Sobolev/value-residual pursuit for additive-basis crossover."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .sn_basis_archive import partial_residual_credit
from .sn_population_coverage import (
    orthonormal_signature_span,
    signature_residual_gain,
)


@dataclass(frozen=True)
class BasisPoolEntry:
    canonical: str
    expression: str
    signature: np.ndarray
    values: np.ndarray
    from_receiver: bool
    from_donor: bool


@dataclass(frozen=True)
class BasisPursuitStep:
    depth: int
    pool_index: int
    canonical: str
    sobolev_gain: float
    partial_residual_correlation: float
    value_residual_gain: float
    joint_score: float
    residual_norm: float
    from_receiver: bool
    from_donor: bool


def orthogonal_basis_pursuit(
    *,
    pool: Sequence[BasisPoolEntry],
    target: Sequence[float],
    max_steps: int,
    coefficient_decimals: int = 6,
) -> tuple[BasisPursuitStep, ...]:
    """Build a deterministic basis path using SN gain times residual credit."""

    y = np.asarray(target, dtype=float).reshape(-1)
    if not pool or y.size == 0 or max_steps < 1 or coefficient_decimals < 0:
        raise ValueError("basis-pursuit pool, target, and limits must be non-empty")
    if not np.all(np.isfinite(y)):
        raise ValueError("basis-pursuit target must be finite")
    signature_size = np.asarray(pool[0].signature, dtype=float).size
    identities: set[str] = set()
    for entry in pool:
        signature = np.asarray(entry.signature, dtype=float).reshape(-1)
        values = np.asarray(entry.values, dtype=float).reshape(-1)
        if (
            not entry.canonical
            or entry.canonical in identities
            or signature.shape != (signature_size,)
            or values.shape != y.shape
            or not np.all(np.isfinite(signature))
            or not np.all(np.isfinite(values))
        ):
            raise ValueError("basis-pursuit pool is not unique, aligned, and finite")
        identities.add(entry.canonical)

    design = np.ones((y.size, 1), dtype=float)
    intercept, *_ = np.linalg.lstsq(design, y, rcond=None)
    intercept = np.round(intercept, coefficient_decimals)
    residual = y - design @ intercept
    selected: list[int] = []
    steps: list[BasisPursuitStep] = []
    tolerance = np.finfo(float).eps
    while len(steps) < min(max_steps, len(pool)):
        span = (
            orthonormal_signature_span(
                [
                    np.asarray(pool[index].signature, dtype=float)
                    for index in selected
                ]
            )
            if selected
            else np.empty((signature_size, 0), dtype=float)
        )
        candidates: list[tuple[tuple[float, float, float, str, int], int, float, float, float]] = []
        for index, entry in enumerate(pool):
            if index in selected:
                continue
            signature = np.asarray(entry.signature, dtype=float).reshape(-1)
            norm = float(np.linalg.norm(signature))
            if norm <= tolerance:
                continue
            gain = (
                signature_residual_gain(span, signature / norm)
                if selected
                else 1.0
            )
            correlation, value_gain = partial_residual_credit(
                entry.values, residual, design
            )
            if gain <= tolerance or correlation <= tolerance:
                continue
            joint = float(gain * correlation)
            candidates.append(
                (
                    (-joint, -gain, -correlation, entry.canonical, index),
                    index,
                    float(gain),
                    float(correlation),
                    float(value_gain),
                )
            )
        if not candidates:
            break
        _, chosen, gain, correlation, value_gain = min(candidates, key=lambda item: item[0])
        selected.append(chosen)
        column = np.asarray(pool[chosen].values, dtype=float).reshape(-1)
        design = np.column_stack((design, column))
        coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
        coefficients = np.round(coefficients, coefficient_decimals)
        residual = y - design @ coefficients
        entry = pool[chosen]
        steps.append(
            BasisPursuitStep(
                depth=len(steps) + 1,
                pool_index=chosen,
                canonical=entry.canonical,
                sobolev_gain=gain,
                partial_residual_correlation=correlation,
                value_residual_gain=value_gain,
                joint_score=float(gain * correlation),
                residual_norm=float(np.linalg.norm(residual)),
                from_receiver=bool(entry.from_receiver),
                from_donor=bool(entry.from_donor),
            )
        )
    return tuple(steps)
