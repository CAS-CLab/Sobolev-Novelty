"""Bounded exponential-factor pursuit screened in Sobolev geometry.

This export-side plug-in constructs ``A * exp(-scale * R)`` from runtime
coordinates.  The exponent core ``R`` is restricted to monomial ratios that
are non-negative on the frozen search rows, so the exponential factor is
bounded by one.  Value-only pools keep enumeration cheap; exact symbolic
value-and-gradient signatures are evaluated only for the retained pools.
Target derivatives are never used.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .sn_archive_interactions import sobolev_product_signature
from .sn_basis_archive import BasisArchiveEntry
from .sn_direct_composition import sobolev_division_signature
from .sn_shared_phase_rational import (
    _absolute_centered_correlations,
    _contexts,
    _monomial_ratios,
    _score,
    _union_by_identity,
)


@dataclass(frozen=True)
class DampedExponentialProposal:
    """One ordinary-GP expression proposed by the factor pursuit."""

    expression: str
    exponent_expression: str
    amplitude_expression: str
    exponent_exponents: tuple[int, ...]
    exponent_scale: float
    training_r2: float
    exponent_target_correlation: float
    exponent_sobolev_gain: float
    composite_target_correlation: float
    composite_sobolev_gain: float
    selection_lanes: tuple[str, ...]


@dataclass(frozen=True)
class DampedExponentialResult:
    """Compact audit result for bounded exponential factorization."""

    proposals: tuple[DampedExponentialProposal, ...]
    monomial_candidates: int
    positivity_rejections: int
    exponential_candidates: int
    exponent_value_pool: int
    exponent_shortlist: int
    amplitude_candidates: int
    composite_candidates: int
    composite_value_pool: int
    numeric_failures: int


@dataclass(frozen=True)
class _ExponentValueCandidate:
    core: object
    scale: float
    expression: str
    values: np.ndarray
    target_correlation: float


@dataclass(frozen=True)
class _ExponentCandidate:
    core: object
    scale: float
    expression: str
    values: np.ndarray
    signature: np.ndarray
    target_correlation: float
    sobolev_gain: float
    joint_score: float
    selection_lanes: tuple[str, ...]


@dataclass(frozen=True)
class _Amplitude:
    expression: str
    values: np.ndarray
    signature: np.ndarray


@dataclass(frozen=True)
class _Composite:
    exponent: _ExponentCandidate
    amplitude: _Amplitude
    expression: str
    values: np.ndarray
    signature: np.ndarray
    target_correlation: float
    sobolev_gain: float
    joint_score: float
    selection_lanes: tuple[str, ...]


def sobolev_exponential_signature(
    source: Sequence[float],
    *,
    coefficient: float,
    sample_count: int,
    dimension: int,
    lambda_value: float = 1.0,
    lambda_gradient: float = 1.0,
) -> np.ndarray:
    """Exact signature of ``exp(coefficient * source)``."""

    vector = np.asarray(source, dtype=float).reshape(-1)
    expected = sample_count * (dimension + 1)
    if (
        sample_count < 1
        or dimension < 0
        or lambda_value <= 0.0
        or (dimension and lambda_gradient <= 0.0)
        or vector.shape != (expected,)
        or not np.all(np.isfinite(vector))
        or not np.isfinite(coefficient)
    ):
        raise ValueError("exponential signature inputs are not aligned")
    value_factor = float(np.sqrt(lambda_value / sample_count))
    source_values = vector[:sample_count] / value_factor
    with np.errstate(all="ignore"):
        transformed = np.exp(float(coefficient) * source_values)
    blocks = [value_factor * transformed]
    if dimension:
        gradient_factor = float(
            np.sqrt(lambda_gradient / (sample_count * dimension))
        )
        for axis in range(dimension):
            start = sample_count * (axis + 1)
            stop = start + sample_count
            source_gradient = vector[start:stop] / gradient_factor
            blocks.append(
                gradient_factor
                * float(coefficient)
                * transformed
                * source_gradient
            )
    output = np.concatenate(blocks)
    if not np.all(np.isfinite(output)):
        raise ValueError("exponential signature is non-finite")
    return output


def _scale_text(scale: float) -> str:
    if np.isclose(scale, 0.5, rtol=0.0, atol=1e-15):
        return "(1 / 2)"
    integer = int(round(scale))
    if np.isclose(scale, integer, rtol=0.0, atol=1e-15):
        return str(integer)
    return format(float(scale), ".17g")


def _constant_signature(
    *,
    sample_count: int,
    dimension: int,
    lambda_value: float,
) -> np.ndarray:
    output = np.zeros(sample_count * (dimension + 1), dtype=float)
    output[:sample_count] = np.sqrt(lambda_value / sample_count)
    return output


def _amplitudes(
    direct: Sequence[BasisArchiveEntry],
    *,
    variable_names: Sequence[str],
    sample_count: int,
    dimension: int,
    protected_epsilon: float,
    lambda_value: float,
    lambda_gradient: float,
) -> tuple[tuple[_Amplitude, ...], int]:
    """Return the frozen amplitude library ``1, x_i, 1/x_i``."""

    constant_signature = _constant_signature(
        sample_count=sample_count,
        dimension=dimension,
        lambda_value=lambda_value,
    )
    output = [
        _Amplitude(
            expression="1",
            values=np.ones_like(np.asarray(direct[0].values, dtype=float)),
            signature=constant_signature,
        )
    ]
    failures = 0
    for name, entry in zip(variable_names, direct, strict=True):
        values = np.asarray(entry.values, dtype=float)
        signature = np.asarray(entry.signature, dtype=float)
        output.append(
            _Amplitude(
                expression=str(name),
                values=values,
                signature=signature,
            )
        )
        guarded = values + protected_epsilon * (values == 0.0)
        with np.errstate(all="ignore"):
            reciprocal_values = 1.0 / guarded
        try:
            reciprocal_signature = sobolev_division_signature(
                constant_signature,
                signature,
                sample_count=sample_count,
                dimension=dimension,
                protected_epsilon=protected_epsilon,
                lambda_value=lambda_value,
                lambda_gradient=lambda_gradient,
            )
        except ValueError:
            failures += 1
            continue
        if not np.all(np.isfinite(reciprocal_values)):
            failures += 1
            continue
        output.append(
            _Amplitude(
                expression=f"1 / ({name})",
                values=reciprocal_values,
                signature=reciprocal_signature,
            )
        )
    return tuple(output), failures


def lift_direct_damped_exponentials(
    *,
    direct_entries: Sequence[BasisArchiveEntry],
    conditioning_entries: Sequence[BasisArchiveEntry],
    target: Sequence[float],
    variable_names: Sequence[str],
    geometry_sample_count: int,
    contexts: Sequence[Sequence[int]],
    exponent_scales: Sequence[float],
    max_numerator_degree: int,
    max_denominator_degree: int,
    exponent_value_pool_size: int,
    exponent_shortlist_size: int,
    composite_value_pool_size: int,
    composite_shortlist_size: int,
    proposal_limit: int,
    protected_epsilon: float = 1e-6,
    lambda_value: float = 1.0,
    lambda_gradient: float = 1.0,
) -> DampedExponentialResult:
    """Construct ``A * exp(-scale * R)`` without changing GP search."""

    direct = tuple(direct_entries)
    conditioning = tuple(conditioning_entries)
    names = tuple(str(value) for value in variable_names)
    y = np.asarray(target, dtype=float).reshape(-1)
    dimension = len(names)
    signature_size = geometry_sample_count * (dimension + 1)
    unique_contexts = tuple(
        dict.fromkeys(
            tuple(sorted(int(index) for index in value)) for value in contexts
        )
    )
    scales = tuple(dict.fromkeys(float(value) for value in exponent_scales))
    if (
        not direct
        or len(direct) != dimension
        or y.size < 1
        or not np.all(np.isfinite(y))
        or geometry_sample_count < 1
        or not unique_contexts
        or not scales
        or any(not np.isfinite(value) or value <= 0.0 for value in scales)
        or max_numerator_degree < 1
        or max_denominator_degree < 0
        or exponent_value_pool_size < exponent_shortlist_size
        or exponent_shortlist_size < 1
        or composite_value_pool_size < composite_shortlist_size
        or composite_shortlist_size < 1
        or proposal_limit < 1
        or protected_epsilon <= 0.0
    ):
        raise ValueError("damped exponential inputs are invalid")
    if any(
        len(set(context)) != len(context)
        or any(index < 0 or index >= len(conditioning) for index in context)
        for context in unique_contexts
    ):
        raise ValueError("damped exponential contexts are invalid")
    if any(
        np.asarray(entry.values).shape != y.shape
        or np.asarray(entry.signature).shape != (signature_size,)
        or not np.all(np.isfinite(entry.values))
        or not np.all(np.isfinite(entry.signature))
        for entry in (*direct, *conditioning)
    ):
        raise ValueError("damped exponential entries are not aligned")

    monomials, numeric_failures = _monomial_ratios(
        direct,
        variable_names=names,
        max_numerator_degree=max_numerator_degree,
        max_denominator_degree=max_denominator_degree,
        sample_count=geometry_sample_count,
        dimension=dimension,
        protected_epsilon=protected_epsilon,
        lambda_value=lambda_value,
        lambda_gradient=lambda_gradient,
    )
    eligible = []
    positivity_rejections = 0
    for monomial in monomials:
        if np.all(np.asarray(monomial.values, dtype=float) >= 0.0):
            eligible.append(monomial)
        else:
            positivity_rejections += 1
    inherited_contexts = _contexts(conditioning, unique_contexts, y)
    amplitudes, amplitude_failures = _amplitudes(
        direct,
        variable_names=names,
        sample_count=geometry_sample_count,
        dimension=dimension,
        protected_epsilon=protected_epsilon,
        lambda_value=lambda_value,
        lambda_gradient=lambda_gradient,
    )
    numeric_failures += amplitude_failures
    amplitude_matrix = np.vstack([item.values for item in amplitudes])

    exponent_values: list[_ExponentValueCandidate] = []
    for core in eligible:
        for scale in scales:
            with np.errstate(all="ignore"):
                values = np.exp(-scale * np.asarray(core.values, dtype=float))
            if not np.all(np.isfinite(values)):
                numeric_failures += 1
                continue
            correlation = float(
                np.max(
                    _absolute_centered_correlations(
                        amplitude_matrix * values, y
                    )
                )
            )
            exponent_values.append(
                _ExponentValueCandidate(
                    core=core,
                    scale=scale,
                    expression=(
                        f"exp(-({_scale_text(scale)}) * ({core.expression}))"
                    ),
                    values=values,
                    target_correlation=correlation,
                )
            )
    exponent_values.sort(
        key=lambda item: (-item.target_correlation, item.expression)
    )
    retained_values = exponent_values[:exponent_value_pool_size]

    exponent_pool: list[_ExponentCandidate] = []
    for item in retained_values:
        try:
            signature = sobolev_exponential_signature(
                item.core.signature,
                coefficient=-item.scale,
                sample_count=geometry_sample_count,
                dimension=dimension,
                lambda_value=lambda_value,
                lambda_gradient=lambda_gradient,
            )
        except ValueError:
            numeric_failures += 1
            continue
        scores = _score(item.values, signature, inherited_contexts)
        exponent_pool.append(
            _ExponentCandidate(
                core=item.core,
                scale=item.scale,
                expression=item.expression,
                values=item.values,
                signature=signature,
                target_correlation=item.target_correlation,
                sobolev_gain=scores.sobolev_gain,
                joint_score=scores.joint,
                selection_lanes=(),
            )
        )
    value_exponents = sorted(
        exponent_pool,
        key=lambda item: (-item.target_correlation, item.expression),
    )[:exponent_shortlist_size]
    joint_exponents = sorted(
        exponent_pool,
        key=lambda item: (
            -item.joint_score,
            -item.sobolev_gain,
            -item.target_correlation,
            item.expression,
        ),
    )[:exponent_shortlist_size]
    exponent_lanes: dict[int, list[str]] = {}
    for item in value_exponents:
        exponent_lanes.setdefault(id(item), []).append("value_exponent")
    for item in joint_exponents:
        exponent_lanes.setdefault(id(item), []).append("sobolev_exponent")
    selected_exponents = [
        _ExponentCandidate(
            **{
                **item.__dict__,
                "selection_lanes": tuple(exponent_lanes[id(item)]),
            }
        )
        for item in _union_by_identity(value_exponents, joint_exponents)
    ]

    composite_values: list[tuple[float, _ExponentCandidate, int]] = []
    for exponent in selected_exponents:
        correlations = _absolute_centered_correlations(
            amplitude_matrix * exponent.values, y
        )
        composite_values.extend(
            (float(correlation), exponent, amplitude_index)
            for amplitude_index, correlation in enumerate(correlations)
        )
    composite_values.sort(
        key=lambda item: (
            -item[0],
            item[1].expression,
            amplitudes[item[2]].expression,
        )
    )
    retained_composites = composite_values[:composite_value_pool_size]

    composite_pool: list[_Composite] = []
    for correlation, exponent, amplitude_index in retained_composites:
        amplitude = amplitudes[amplitude_index]
        values = amplitude.values * exponent.values
        try:
            signature = sobolev_product_signature(
                amplitude.signature,
                exponent.signature,
                sample_count=geometry_sample_count,
                dimension=dimension,
                lambda_value=lambda_value,
                lambda_gradient=lambda_gradient,
            )
        except ValueError:
            numeric_failures += 1
            continue
        scores = _score(values, signature, inherited_contexts)
        composite_pool.append(
            _Composite(
                exponent=exponent,
                amplitude=amplitude,
                expression=(
                    f"({amplitude.expression}) * ({exponent.expression})"
                ),
                values=values,
                signature=signature,
                target_correlation=correlation,
                sobolev_gain=scores.sobolev_gain,
                joint_score=scores.joint,
                selection_lanes=(),
            )
        )
    value_composites = sorted(
        composite_pool,
        key=lambda item: (-item.target_correlation, item.expression),
    )[:composite_shortlist_size]
    joint_composites = sorted(
        composite_pool,
        key=lambda item: (
            -item.joint_score,
            -item.sobolev_gain,
            -item.target_correlation,
            item.expression,
        ),
    )[:composite_shortlist_size]
    composite_lanes: dict[int, list[str]] = {}
    for item in value_composites:
        composite_lanes.setdefault(id(item), []).append("value_composite")
    for item in joint_composites:
        composite_lanes.setdefault(id(item), []).append("sobolev_composite")

    proposals: list[DampedExponentialProposal] = []
    for item in _union_by_identity(value_composites, joint_composites):
        design = np.column_stack((np.ones(y.size, dtype=float), item.values))
        coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
        prediction = design @ np.round(coefficients, 6)
        variance = float(np.var(y))
        training_r2 = float(
            1.0 - np.mean(np.square(prediction - y)) / variance
        )
        if not np.isfinite(training_r2):
            numeric_failures += 1
            continue
        proposals.append(
            DampedExponentialProposal(
                expression=item.expression,
                exponent_expression=item.exponent.expression,
                amplitude_expression=item.amplitude.expression,
                exponent_exponents=item.exponent.core.exponents,
                exponent_scale=item.exponent.scale,
                training_r2=training_r2,
                exponent_target_correlation=(
                    item.exponent.target_correlation
                ),
                exponent_sobolev_gain=item.exponent.sobolev_gain,
                composite_target_correlation=item.target_correlation,
                composite_sobolev_gain=item.sobolev_gain,
                selection_lanes=(
                    *item.exponent.selection_lanes,
                    *composite_lanes[id(item)],
                ),
            )
        )
    proposals.sort(
        key=lambda item: (
            -item.training_r2,
            -item.composite_sobolev_gain,
            -item.exponent_sobolev_gain,
            item.expression,
        )
    )
    unique_proposals: list[DampedExponentialProposal] = []
    seen_expressions: set[str] = set()
    for proposal in proposals:
        if proposal.expression in seen_expressions:
            continue
        seen_expressions.add(proposal.expression)
        unique_proposals.append(proposal)
        if len(unique_proposals) >= proposal_limit:
            break

    return DampedExponentialResult(
        proposals=tuple(unique_proposals),
        monomial_candidates=len(monomials),
        positivity_rejections=positivity_rejections,
        exponential_candidates=len(exponent_values),
        exponent_value_pool=len(retained_values),
        exponent_shortlist=len(selected_exponents),
        amplitude_candidates=len(amplitudes),
        composite_candidates=len(selected_exponents) * len(amplitudes),
        composite_value_pool=len(retained_composites),
        numeric_failures=numeric_failures,
    )
