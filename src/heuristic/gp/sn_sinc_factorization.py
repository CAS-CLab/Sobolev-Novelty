"""Shared-core sinc-squared pursuit screened in Sobolev geometry.

The export-side plug-in constructs ``A * (sin(z) / z) ** 2``.  Candidate
phase cores are built from runtime coordinates, affine coordinate pairs, and
one bounded product layer.  A broad amplitude-aware value screen is followed
by exact conditional value-and-gradient scoring.  Target derivatives are
never used and the Population GP trajectory is not modified.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Sequence

import numpy as np

from .sn_affine_gaussian import _affine_sources
from .sn_archive_interactions import (
    sobolev_product_signature,
    sobolev_unary_signature,
)
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
class SincSquaredProposal:
    """One ordinary-GP sinc-squared expression proposed for exact fitting."""

    expression: str
    phase_core_expression: str
    phase_expression: str
    amplitude_expression: str
    amplitude_exponents: tuple[int, ...]
    phase_scale: float
    training_r2: float
    shape_target_correlation: float
    shape_sobolev_gain: float
    composite_target_correlation: float
    composite_sobolev_gain: float
    selection_lanes: tuple[str, ...]


@dataclass(frozen=True)
class SincSquaredResult:
    """Compact audit result for one shared-core sinc-squared pursuit."""

    proposals: tuple[SincSquaredProposal, ...]
    affine_sources: int
    phase_cores: int
    shape_candidates: int
    shape_value_pool: int
    shape_shortlist: int
    amplitude_candidates: int
    composite_candidates: int
    composite_value_pool: int
    numeric_failures: int


@dataclass(frozen=True)
class _PhaseCore:
    expression: str
    values: np.ndarray
    signature: np.ndarray


@dataclass(frozen=True)
class _Amplitude:
    exponents: tuple[int, ...]
    expression: str
    values: np.ndarray
    signature: np.ndarray


@dataclass(frozen=True)
class _ShapeValueCandidate:
    core: _PhaseCore
    scale: float
    phase_expression: str
    expression: str
    values: np.ndarray
    target_correlation: float


@dataclass(frozen=True)
class _ShapeCandidate:
    core: _PhaseCore
    scale: float
    phase_expression: str
    expression: str
    values: np.ndarray
    signature: np.ndarray
    target_correlation: float
    sobolev_gain: float
    joint_score: float
    selection_lanes: tuple[str, ...]


@dataclass(frozen=True)
class _Composite:
    shape: _ShapeCandidate
    amplitude: _Amplitude
    expression: str
    values: np.ndarray
    signature: np.ndarray
    target_correlation: float
    sobolev_gain: float
    joint_score: float
    selection_lanes: tuple[str, ...]


def _scale_text(scale: float) -> str:
    if np.isclose(scale, 0.5, rtol=0.0, atol=1e-15):
        return "(1 / 2)"
    if np.isclose(scale, 1.0, rtol=0.0, atol=1e-15):
        return "1"
    if np.isclose(scale, 2.0, rtol=0.0, atol=1e-15):
        return "2"
    if np.isclose(scale, np.pi, rtol=0.0, atol=1e-15):
        return "3.141592653589793"
    if np.isclose(scale, 2.0 * np.pi, rtol=0.0, atol=1e-15):
        return "(2 * 3.141592653589793)"
    raise ValueError(f"sinc scale {scale!r} is outside the GP constant library")


def sobolev_sinc_squared_signature(
    source: Sequence[float],
    *,
    coefficient: float,
    sample_count: int,
    dimension: int,
    protected_epsilon: float = 1e-6,
    lambda_value: float = 1.0,
    lambda_gradient: float = 1.0,
) -> np.ndarray:
    """Exact protected-GP signature of ``(sin(c*x)/(c*x))**2``."""

    vector = np.asarray(source, dtype=float).reshape(-1)
    expected = sample_count * (dimension + 1)
    if (
        sample_count < 1
        or dimension < 0
        or vector.shape != (expected,)
        or not np.all(np.isfinite(vector))
        or not np.isfinite(coefficient)
        or coefficient == 0.0
        or protected_epsilon <= 0.0
    ):
        raise ValueError("sinc-squared signature inputs are not aligned")
    scaled = float(coefficient) * vector
    sine = sobolev_unary_signature(
        scaled,
        transform="sin",
        sample_count=sample_count,
        dimension=dimension,
        lambda_value=lambda_value,
        lambda_gradient=lambda_gradient,
    )
    ratio = sobolev_division_signature(
        sine,
        scaled,
        sample_count=sample_count,
        dimension=dimension,
        protected_epsilon=protected_epsilon,
        lambda_value=lambda_value,
        lambda_gradient=lambda_gradient,
    )
    return sobolev_product_signature(
        ratio,
        ratio,
        sample_count=sample_count,
        dimension=dimension,
        lambda_value=lambda_value,
        lambda_gradient=lambda_gradient,
    )


def _phase_cores(
    direct: Sequence[BasisArchiveEntry],
    variable_names: Sequence[str],
    *,
    sample_count: int,
    dimension: int,
    lambda_value: float,
    lambda_gradient: float,
) -> tuple[tuple[_PhaseCore, ...], int, int]:
    sources = _affine_sources(direct, variable_names)
    output: list[_PhaseCore] = [
        _PhaseCore(
            expression=source.expression,
            values=np.asarray(source.values, dtype=float),
            signature=np.asarray(source.signature, dtype=float),
        )
        for source in sources
    ]
    failures = 0
    for source in sources:
        for name, entry in zip(variable_names, direct, strict=True):
            values = np.asarray(source.values, dtype=float) * np.asarray(
                entry.values, dtype=float
            )
            try:
                signature = sobolev_product_signature(
                    source.signature,
                    entry.signature,
                    sample_count=sample_count,
                    dimension=dimension,
                    lambda_value=lambda_value,
                    lambda_gradient=lambda_gradient,
                )
            except ValueError:
                failures += 1
                continue
            if not np.all(np.isfinite(values)) or not np.all(
                np.isfinite(signature)
            ):
                failures += 1
                continue
            output.append(
                _PhaseCore(
                    expression=f"({source.expression}) * ({name})",
                    values=values,
                    signature=signature,
                )
            )
    return tuple(output), len(sources), failures


def _amplitudes(
    direct: Sequence[BasisArchiveEntry],
    *,
    variable_names: Sequence[str],
    max_numerator_degree: int,
    max_denominator_degree: int,
    sample_count: int,
    dimension: int,
    protected_epsilon: float,
    lambda_value: float,
    lambda_gradient: float,
) -> tuple[tuple[_Amplitude, ...], int]:
    monomials, failures = _monomial_ratios(
        direct,
        variable_names=variable_names,
        max_numerator_degree=max_numerator_degree,
        max_denominator_degree=max_denominator_degree,
        sample_count=sample_count,
        dimension=dimension,
        protected_epsilon=protected_epsilon,
        lambda_value=lambda_value,
        lambda_gradient=lambda_gradient,
    )
    constant = np.zeros(sample_count * (dimension + 1), dtype=float)
    constant[:sample_count] = np.sqrt(lambda_value / sample_count)
    output = [
        _Amplitude(
            exponents=(0,) * dimension,
            expression="1",
            values=np.ones_like(np.asarray(direct[0].values, dtype=float)),
            signature=constant,
        )
    ]
    output.extend(
        _Amplitude(
            exponents=tuple(int(value) for value in item.exponents),
            expression=item.expression,
            values=np.asarray(item.values, dtype=float),
            signature=np.asarray(item.signature, dtype=float),
        )
        for item in monomials
    )
    return tuple(output), failures


def lift_direct_sinc_squared_factors(
    *,
    direct_entries: Sequence[BasisArchiveEntry],
    conditioning_entries: Sequence[BasisArchiveEntry],
    target: Sequence[float],
    variable_names: Sequence[str],
    geometry_sample_count: int,
    contexts: Sequence[Sequence[int]],
    phase_scales: Sequence[float],
    max_numerator_degree: int,
    max_denominator_degree: int,
    shape_value_pool_size: int,
    shape_shortlist_size: int,
    composite_value_pool_size: int,
    composite_shortlist_size: int,
    proposal_limit: int,
    protected_epsilon: float = 1e-6,
    lambda_value: float = 1.0,
    lambda_gradient: float = 1.0,
) -> SincSquaredResult:
    """Construct ``A*(sin(z)/z)**2`` without changing Population GP."""

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
    scales = tuple(dict.fromkeys(float(value) for value in phase_scales))
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
        or shape_value_pool_size < shape_shortlist_size
        or shape_shortlist_size < 1
        or composite_value_pool_size < composite_shortlist_size
        or composite_shortlist_size < 1
        or proposal_limit < 1
        or protected_epsilon <= 0.0
    ):
        raise ValueError("sinc-squared pursuit inputs are invalid")
    if any(
        len(set(context)) != len(context)
        or any(index < 0 or index >= len(conditioning) for index in context)
        for context in unique_contexts
    ):
        raise ValueError("sinc-squared contexts are invalid")
    if any(
        np.asarray(entry.values).shape != y.shape
        or np.asarray(entry.signature).shape != (signature_size,)
        or not np.all(np.isfinite(entry.values))
        or not np.all(np.isfinite(entry.signature))
        for entry in (*direct, *conditioning)
    ):
        raise ValueError("sinc-squared entries are not aligned")

    cores, affine_source_count, numeric_failures = _phase_cores(
        direct,
        names,
        sample_count=geometry_sample_count,
        dimension=dimension,
        lambda_value=lambda_value,
        lambda_gradient=lambda_gradient,
    )
    amplitudes, amplitude_failures = _amplitudes(
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
    numeric_failures += amplitude_failures
    inherited_contexts = _contexts(conditioning, unique_contexts, y)
    amplitude_matrix = np.vstack([item.values for item in amplitudes])

    shape_values: list[_ShapeValueCandidate] = []
    for core in cores:
        for scale in scales:
            phase_expression = (
                core.expression
                if np.isclose(scale, 1.0, rtol=0.0, atol=1e-15)
                else f"({_scale_text(scale)}) * ({core.expression})"
            )
            phase_values = scale * core.values
            guarded = phase_values + protected_epsilon * (phase_values == 0.0)
            with np.errstate(all="ignore"):
                values = np.square(np.sin(phase_values) / guarded)
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
            shape_values.append(
                _ShapeValueCandidate(
                    core=core,
                    scale=scale,
                    phase_expression=phase_expression,
                    expression=(
                        f"(sin({phase_expression}) / "
                        f"({phase_expression})) ** 2"
                    ),
                    values=values,
                    target_correlation=correlation,
                )
            )
    shape_values.sort(
        key=lambda item: (-item.target_correlation, item.expression)
    )
    retained_shape_values = shape_values[:shape_value_pool_size]

    shape_pool: list[_ShapeCandidate] = []
    for item in retained_shape_values:
        try:
            signature = sobolev_sinc_squared_signature(
                item.core.signature,
                coefficient=item.scale,
                sample_count=geometry_sample_count,
                dimension=dimension,
                protected_epsilon=protected_epsilon,
                lambda_value=lambda_value,
                lambda_gradient=lambda_gradient,
            )
        except ValueError:
            numeric_failures += 1
            continue
        scores = _score(item.values, signature, inherited_contexts)
        shape_pool.append(
            _ShapeCandidate(
                core=item.core,
                scale=item.scale,
                phase_expression=item.phase_expression,
                expression=item.expression,
                values=item.values,
                signature=signature,
                target_correlation=item.target_correlation,
                sobolev_gain=scores.sobolev_gain,
                joint_score=scores.joint,
                selection_lanes=(),
            )
        )
    value_shapes = sorted(
        shape_pool,
        key=lambda item: (-item.target_correlation, item.expression),
    )[:shape_shortlist_size]
    joint_shapes = sorted(
        shape_pool,
        key=lambda item: (
            -item.joint_score,
            -item.sobolev_gain,
            -item.target_correlation,
            item.expression,
        ),
    )[:shape_shortlist_size]
    shape_lanes: dict[int, list[str]] = {}
    for item in value_shapes:
        shape_lanes.setdefault(id(item), []).append("value_shape")
    for item in joint_shapes:
        shape_lanes.setdefault(id(item), []).append("sobolev_shape")
    selected_shapes = [
        replace(item, selection_lanes=tuple(shape_lanes[id(item)]))
        for item in _union_by_identity(value_shapes, joint_shapes)
    ]

    composite_values: list[tuple[float, _ShapeCandidate, int]] = []
    for shape in selected_shapes:
        correlations = _absolute_centered_correlations(
            amplitude_matrix * shape.values, y
        )
        composite_values.extend(
            (float(correlation), shape, amplitude_index)
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
    for correlation, shape, amplitude_index in retained_composites:
        amplitude = amplitudes[amplitude_index]
        values = amplitude.values * shape.values
        try:
            signature = sobolev_product_signature(
                amplitude.signature,
                shape.signature,
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
                shape=shape,
                amplitude=amplitude,
                expression=f"({amplitude.expression}) * ({shape.expression})",
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

    proposals: list[SincSquaredProposal] = []
    variance = float(np.var(y))
    for item in _union_by_identity(value_composites, joint_composites):
        design = np.column_stack((np.ones(y.size, dtype=float), item.values))
        coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
        prediction = design @ np.round(coefficients, 6)
        training_r2 = float(
            1.0 - np.mean(np.square(prediction - y)) / variance
        )
        if not np.isfinite(training_r2):
            numeric_failures += 1
            continue
        proposals.append(
            SincSquaredProposal(
                expression=item.expression,
                phase_core_expression=item.shape.core.expression,
                phase_expression=item.shape.phase_expression,
                amplitude_expression=item.amplitude.expression,
                amplitude_exponents=item.amplitude.exponents,
                phase_scale=item.shape.scale,
                training_r2=training_r2,
                shape_target_correlation=item.shape.target_correlation,
                shape_sobolev_gain=item.shape.sobolev_gain,
                composite_target_correlation=item.target_correlation,
                composite_sobolev_gain=item.sobolev_gain,
                selection_lanes=(
                    *item.shape.selection_lanes,
                    *composite_lanes[id(item)],
                ),
            )
        )
    proposals.sort(
        key=lambda item: (
            -item.training_r2,
            -item.composite_sobolev_gain,
            -item.shape_sobolev_gain,
            item.expression,
        )
    )
    unique_proposals: list[SincSquaredProposal] = []
    seen_expressions: set[str] = set()
    for proposal in proposals:
        if proposal.expression in seen_expressions:
            continue
        seen_expressions.add(proposal.expression)
        unique_proposals.append(proposal)
        if len(unique_proposals) >= proposal_limit:
            break

    return SincSquaredResult(
        proposals=tuple(unique_proposals),
        affine_sources=affine_source_count,
        phase_cores=len(cores),
        shape_candidates=len(shape_values),
        shape_value_pool=len(retained_shape_values),
        shape_shortlist=len(selected_shapes),
        amplitude_candidates=len(amplitudes),
        composite_candidates=len(selected_shapes) * len(amplitudes),
        composite_value_pool=len(retained_composites),
        numeric_failures=numeric_failures,
    )
