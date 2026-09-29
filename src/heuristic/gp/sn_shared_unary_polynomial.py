"""Shared-unary polynomial factorization screened in Sobolev geometry.

The export-side plug-in constructs ``U(z) * (A + B * U(z))``.  Reusing the
same nonlinear feature is difficult for ordinary tree GP because the phase
subtree must be rediscovered twice.  This module keeps a broad value screen,
then applies conditional value-and-gradient screening to the quadratic and
linear amplitude terms without changing the Population GP trajectory.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Sequence

import numpy as np

from .sn_archive_interactions import (
    sobolev_product_signature,
    sobolev_unary_signature,
)
from .sn_basis_archive import BasisArchiveEntry
from .sn_shared_phase_rational import (
    _absolute_centered_correlations,
    _contexts,
    _score,
    _single_context,
    _union_by_identity,
)
from .sn_sinc_factorization import (
    _Amplitude,
    _PhaseCore,
    _amplitudes,
    _phase_cores,
    _scale_text,
)


@dataclass(frozen=True)
class SharedUnaryPolynomialProposal:
    """One ordinary-GP shared-unary polynomial expression."""

    expression: str
    phase_core_expression: str
    phase_expression: str
    phase_scale: float
    transform: str
    linear_amplitude_expression: str
    linear_amplitude_exponents: tuple[int, ...]
    quadratic_amplitude_expression: str
    quadratic_amplitude_exponents: tuple[int, ...]
    training_r2: float
    phase_target_correlation: float
    phase_sobolev_gain: float
    linear_conditional_correlation: float
    linear_sobolev_gain: float
    quadratic_target_correlation: float
    quadratic_sobolev_gain: float
    selection_lanes: tuple[str, ...]


@dataclass(frozen=True)
class SharedUnaryPolynomialResult:
    """Compact audit result for one shared-unary polynomial pursuit."""

    proposals: tuple[SharedUnaryPolynomialProposal, ...]
    affine_sources: int
    phase_cores: int
    phase_candidates: int
    phase_value_pool: int
    phase_shortlist: int
    amplitude_candidates: int
    quadratic_states: int
    quadratic_shortlist: int
    linear_conditional_states: int
    numeric_failures: int


@dataclass(frozen=True)
class _PhaseValue:
    core: _PhaseCore
    scale: float
    transform: str
    expression: str
    values: np.ndarray
    squared_values: np.ndarray
    target_correlation: float


@dataclass(frozen=True)
class _Phase:
    core: _PhaseCore
    scale: float
    transform: str
    expression: str
    values: np.ndarray
    signature: np.ndarray
    squared_values: np.ndarray
    squared_signature: np.ndarray
    target_correlation: float
    sobolev_gain: float
    joint_score: float
    selection_lanes: tuple[str, ...]


@dataclass(frozen=True)
class _Term:
    amplitude: _Amplitude
    values: np.ndarray
    signature: np.ndarray
    correlation: float
    sobolev_gain: float
    joint_score: float
    value_score: float
    selection_lanes: tuple[str, ...]


def _conditional_correlations(matrix: np.ndarray, context: object) -> np.ndarray:
    orthogonal = matrix - (matrix @ context.value_span) @ context.value_span.T
    residual = np.asarray(context.residual, dtype=float)
    denominator = np.linalg.norm(orthogonal, axis=1) * float(
        np.linalg.norm(residual)
    )
    output = np.zeros(matrix.shape[0], dtype=float)
    valid = denominator > np.finfo(float).eps
    output[valid] = np.abs(orthogonal[valid] @ residual) / denominator[valid]
    return np.clip(output, 0.0, 1.0)


def lift_direct_shared_unary_polynomials(
    *,
    direct_entries: Sequence[BasisArchiveEntry],
    conditioning_entries: Sequence[BasisArchiveEntry],
    target: Sequence[float],
    variable_names: Sequence[str],
    geometry_sample_count: int,
    contexts: Sequence[Sequence[int]],
    phase_scales: Sequence[float],
    transforms: Sequence[str],
    max_numerator_degree: int,
    max_denominator_degree: int,
    phase_value_pool_size: int,
    phase_shortlist_size: int,
    amplitude_value_pool_size: int,
    amplitude_shortlist_size: int,
    proposal_limit: int,
    protected_epsilon: float = 1e-6,
    lambda_value: float = 1.0,
    lambda_gradient: float = 1.0,
) -> SharedUnaryPolynomialResult:
    """Construct ``U(z)*(A+B*U(z))`` without modifying Population GP."""

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
    operations = tuple(dict.fromkeys(str(value) for value in transforms))
    if (
        not direct
        or len(direct) != dimension
        or y.size < 2
        or not np.all(np.isfinite(y))
        or float(np.var(y)) <= np.finfo(float).eps
        or geometry_sample_count < 1
        or not unique_contexts
        or not scales
        or not operations
        or any(value not in {"sin", "cos", "tanh"} for value in operations)
        or max_numerator_degree < 1
        or max_denominator_degree < 0
        or phase_value_pool_size < phase_shortlist_size
        or phase_shortlist_size < 1
        or amplitude_value_pool_size < amplitude_shortlist_size
        or amplitude_shortlist_size < 1
        or proposal_limit < 1
        or protected_epsilon <= 0.0
    ):
        raise ValueError("shared-unary polynomial inputs are invalid")
    if any(
        len(set(context)) != len(context)
        or any(index < 0 or index >= len(conditioning) for index in context)
        for context in unique_contexts
    ):
        raise ValueError("shared-unary polynomial contexts are invalid")
    if any(
        np.asarray(entry.values).shape != y.shape
        or np.asarray(entry.signature).shape != (signature_size,)
        or not np.all(np.isfinite(entry.values))
        or not np.all(np.isfinite(entry.signature))
        for entry in (*direct, *conditioning)
    ):
        raise ValueError("shared-unary polynomial entries are not aligned")

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

    phase_values: list[_PhaseValue] = []
    for core in cores:
        for scale in scales:
            source_expression = (
                core.expression
                if np.isclose(scale, 1.0, rtol=0.0, atol=1e-15)
                else f"({_scale_text(scale)}) * ({core.expression})"
            )
            scaled = scale * core.values
            for transform in operations:
                values = getattr(np, transform)(scaled)
                squared_values = np.square(values)
                if not np.all(np.isfinite(squared_values)):
                    numeric_failures += 1
                    continue
                correlations = _absolute_centered_correlations(
                    amplitude_matrix * squared_values, y
                )
                phase_values.append(
                    _PhaseValue(
                        core=core,
                        scale=scale,
                        transform=transform,
                        expression=f"{transform}({source_expression})",
                        values=values,
                        squared_values=squared_values,
                        target_correlation=float(np.max(correlations)),
                    )
                )
    phase_values.sort(
        key=lambda item: (-item.target_correlation, item.expression)
    )
    retained_phase_values = phase_values[:phase_value_pool_size]

    phase_pool: list[_Phase] = []
    for item in retained_phase_values:
        try:
            signature = sobolev_unary_signature(
                item.scale * item.core.signature,
                transform=item.transform,
                sample_count=geometry_sample_count,
                dimension=dimension,
                lambda_value=lambda_value,
                lambda_gradient=lambda_gradient,
            )
            squared_signature = sobolev_product_signature(
                signature,
                signature,
                sample_count=geometry_sample_count,
                dimension=dimension,
                lambda_value=lambda_value,
                lambda_gradient=lambda_gradient,
            )
        except ValueError:
            numeric_failures += 1
            continue
        scores = _score(item.squared_values, squared_signature, inherited_contexts)
        phase_pool.append(
            _Phase(
                core=item.core,
                scale=item.scale,
                transform=item.transform,
                expression=item.expression,
                values=item.values,
                signature=signature,
                squared_values=item.squared_values,
                squared_signature=squared_signature,
                target_correlation=item.target_correlation,
                sobolev_gain=scores.sobolev_gain,
                joint_score=scores.joint,
                selection_lanes=(),
            )
        )
    value_phases = sorted(
        phase_pool,
        key=lambda item: (-item.target_correlation, item.expression),
    )[:phase_shortlist_size]
    sobolev_phases = sorted(
        phase_pool,
        key=lambda item: (
            -item.joint_score,
            -item.sobolev_gain,
            -item.target_correlation,
            item.expression,
        ),
    )[:phase_shortlist_size]
    phase_lanes: dict[int, list[str]] = {}
    for item in value_phases:
        phase_lanes.setdefault(id(item), []).append("value_phase")
    for item in sobolev_phases:
        phase_lanes.setdefault(id(item), []).append("sobolev_phase")
    selected_phases = [
        replace(item, selection_lanes=tuple(phase_lanes[id(item)]))
        for item in _union_by_identity(value_phases, sobolev_phases)
    ]

    proposals: list[SharedUnaryPolynomialProposal] = []
    quadratic_states = 0
    quadratic_shortlist = 0
    linear_conditional_states = 0
    variance = float(np.var(y))
    for phase in selected_phases:
        quadratic_matrix = amplitude_matrix * phase.squared_values
        quadratic_correlations = _absolute_centered_correlations(
            quadratic_matrix, y
        )
        quadratic_indices = np.argsort(
            -quadratic_correlations, kind="stable"
        )[:amplitude_value_pool_size]
        quadratic_pool: list[_Term] = []
        for amplitude_index in quadratic_indices:
            amplitude = amplitudes[int(amplitude_index)]
            try:
                signature = sobolev_product_signature(
                    amplitude.signature,
                    phase.squared_signature,
                    sample_count=geometry_sample_count,
                    dimension=dimension,
                    lambda_value=lambda_value,
                    lambda_gradient=lambda_gradient,
                )
            except ValueError:
                numeric_failures += 1
                continue
            scores = _score(
                quadratic_matrix[int(amplitude_index)],
                signature,
                inherited_contexts,
            )
            quadratic_pool.append(
                _Term(
                    amplitude=amplitude,
                    values=quadratic_matrix[int(amplitude_index)],
                    signature=signature,
                    correlation=float(quadratic_correlations[int(amplitude_index)]),
                    sobolev_gain=scores.sobolev_gain,
                    joint_score=scores.joint,
                    value_score=scores.value,
                    selection_lanes=(),
                )
            )
        quadratic_states += len(quadratic_pool)
        value_quadratic = sorted(
            quadratic_pool,
            key=lambda item: (-item.correlation, item.amplitude.expression),
        )[:amplitude_shortlist_size]
        sobolev_quadratic = sorted(
            quadratic_pool,
            key=lambda item: (
                -item.joint_score,
                -item.sobolev_gain,
                -item.correlation,
                item.amplitude.expression,
            ),
        )[:amplitude_shortlist_size]
        quadratic_lanes: dict[int, list[str]] = {}
        for item in value_quadratic:
            quadratic_lanes.setdefault(id(item), []).append("value_quadratic")
        for item in sobolev_quadratic:
            quadratic_lanes.setdefault(id(item), []).append(
                "sobolev_quadratic"
            )
        selected_quadratic = [
            replace(item, selection_lanes=tuple(quadratic_lanes[id(item)]))
            for item in _union_by_identity(value_quadratic, sobolev_quadratic)
        ]
        quadratic_shortlist += len(selected_quadratic)

        linear_matrix = amplitude_matrix * phase.values
        for quadratic in selected_quadratic:
            dynamic_context = _single_context(
                quadratic.values, quadratic.signature, y
            )
            linear_correlations = _conditional_correlations(
                linear_matrix, dynamic_context
            )
            linear_indices = np.argsort(
                -linear_correlations, kind="stable"
            )[:amplitude_value_pool_size]
            linear_pool: list[_Term] = []
            for amplitude_index in linear_indices:
                amplitude = amplitudes[int(amplitude_index)]
                try:
                    signature = sobolev_product_signature(
                        amplitude.signature,
                        phase.signature,
                        sample_count=geometry_sample_count,
                        dimension=dimension,
                        lambda_value=lambda_value,
                        lambda_gradient=lambda_gradient,
                    )
                except ValueError:
                    numeric_failures += 1
                    continue
                scores = _score(
                    linear_matrix[int(amplitude_index)],
                    signature,
                    (dynamic_context,),
                )
                linear_pool.append(
                    _Term(
                        amplitude=amplitude,
                        values=linear_matrix[int(amplitude_index)],
                        signature=signature,
                        correlation=float(linear_correlations[int(amplitude_index)]),
                        sobolev_gain=scores.sobolev_gain,
                        joint_score=scores.joint,
                        value_score=scores.value,
                        selection_lanes=(),
                    )
                )
            linear_conditional_states += len(linear_pool)
            value_linear = sorted(
                linear_pool,
                key=lambda item: (-item.correlation, item.amplitude.expression),
            )[:amplitude_shortlist_size]
            sobolev_linear = sorted(
                linear_pool,
                key=lambda item: (
                    -item.joint_score,
                    -item.sobolev_gain,
                    -item.correlation,
                    item.amplitude.expression,
                ),
            )[:amplitude_shortlist_size]
            linear_lanes: dict[int, list[str]] = {}
            for item in value_linear:
                linear_lanes.setdefault(id(item), []).append("value_linear")
            for item in sobolev_linear:
                linear_lanes.setdefault(id(item), []).append("sobolev_linear")
            selected_linear = [
                replace(item, selection_lanes=tuple(linear_lanes[id(item)]))
                for item in _union_by_identity(value_linear, sobolev_linear)
            ]
            for linear in selected_linear:
                design = np.column_stack(
                    (np.ones(y.size), linear.values, quadratic.values)
                )
                coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
                prediction = design @ np.round(coefficients, 6)
                training_r2 = float(
                    1.0 - np.mean(np.square(prediction - y)) / variance
                )
                if not np.isfinite(training_r2):
                    numeric_failures += 1
                    continue
                proposals.append(
                    SharedUnaryPolynomialProposal(
                        expression=(
                            f"({phase.expression}) * "
                            f"(({linear.amplitude.expression}) + "
                            f"({quadratic.amplitude.expression}) * "
                            f"({phase.expression}))"
                        ),
                        phase_core_expression=phase.core.expression,
                        phase_expression=phase.expression,
                        phase_scale=phase.scale,
                        transform=phase.transform,
                        linear_amplitude_expression=(
                            linear.amplitude.expression
                        ),
                        linear_amplitude_exponents=(
                            linear.amplitude.exponents
                        ),
                        quadratic_amplitude_expression=(
                            quadratic.amplitude.expression
                        ),
                        quadratic_amplitude_exponents=(
                            quadratic.amplitude.exponents
                        ),
                        training_r2=training_r2,
                        phase_target_correlation=phase.target_correlation,
                        phase_sobolev_gain=phase.sobolev_gain,
                        linear_conditional_correlation=linear.correlation,
                        linear_sobolev_gain=linear.sobolev_gain,
                        quadratic_target_correlation=quadratic.correlation,
                        quadratic_sobolev_gain=quadratic.sobolev_gain,
                        selection_lanes=(
                            *phase.selection_lanes,
                            *linear.selection_lanes,
                            *quadratic.selection_lanes,
                        ),
                    )
                )

    proposals.sort(
        key=lambda item: (
            -item.training_r2,
            -item.linear_sobolev_gain,
            -item.quadratic_sobolev_gain,
            -item.phase_sobolev_gain,
            item.expression,
        )
    )
    unique_proposals: list[SharedUnaryPolynomialProposal] = []
    seen_expressions: set[str] = set()
    for proposal in proposals:
        if proposal.expression in seen_expressions:
            continue
        seen_expressions.add(proposal.expression)
        unique_proposals.append(proposal)
        if len(unique_proposals) >= proposal_limit:
            break

    return SharedUnaryPolynomialResult(
        proposals=tuple(unique_proposals),
        affine_sources=affine_source_count,
        phase_cores=len(cores),
        phase_candidates=len(phase_values),
        phase_value_pool=len(retained_phase_values),
        phase_shortlist=len(selected_phases),
        amplitude_candidates=len(amplitudes),
        quadratic_states=quadratic_states,
        quadratic_shortlist=quadratic_shortlist,
        linear_conditional_states=linear_conditional_states,
        numeric_failures=numeric_failures,
    )
