"""Pair-aware protected reciprocal-trigonometric Sobolev pursuit.

The export-side plug-in forms complete ``A * U(z_i) * cot(z_j)`` pairs
before screening either phase.  This preserves interactions whose individual
atoms have weak marginal credit.  Target derivatives are never used and the
Population GP trajectory is unchanged.
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
from .sn_direct_composition import sobolev_division_signature
from .sn_population_coverage import orthonormal_signature_span
from .sn_sinc_factorization import _scale_text


@dataclass(frozen=True)
class ReciprocalTrigProposal:
    """One ordinary-GP expression retained by a pair-aware lane."""

    expression: str
    phase_expression: str
    reciprocal_expression: str
    amplitude_expression: str
    phase_axis: int
    reciprocal_axis: int
    phase_scale: float
    reciprocal_scale: float
    transform: str
    context: tuple[int, ...]
    context_expressions: tuple[str, ...]
    training_r2: float
    target_correlation: float
    sobolev_gain: float
    joint_score: float
    selection_lanes: tuple[str, ...]


@dataclass(frozen=True)
class ReciprocalTrigResult:
    """Compact audit result for one reciprocal-trigonometric pursuit."""

    proposals: tuple[ReciprocalTrigProposal, ...]
    contexts_screened: int
    phase_atoms: int
    reciprocal_atoms: int
    amplitude_candidates: int
    pair_candidates: int
    value_pool: int
    pair_shortlist: int
    numeric_failures: int


@dataclass(frozen=True)
class _Context:
    indices: tuple[int, ...]
    residual: np.ndarray
    value_span: np.ndarray
    signature_span: np.ndarray | None


@dataclass(frozen=True)
class _Atom:
    expression: str
    values: np.ndarray
    signature: np.ndarray
    axis: int
    scale: float
    transform: str


@dataclass(frozen=True)
class _Amplitude:
    expression: str
    values: np.ndarray
    signature: np.ndarray


@dataclass(frozen=True)
class _ValuePair:
    expression: str
    values: np.ndarray
    phase: _Atom
    reciprocal: _Atom
    amplitude: _Amplitude
    target_correlation: float
    value_context: tuple[int, ...]


@dataclass(frozen=True)
class _ScoredPair:
    value_pair: _ValuePair
    signature: np.ndarray
    target_correlation: float
    value_context: tuple[int, ...]
    sobolev_correlation: float
    sobolev_gain: float
    joint_score: float
    sobolev_context: tuple[int, ...]
    selection_lanes: tuple[str, ...]


def _source_expression(name: str, scale: float) -> str:
    return (
        name
        if np.isclose(scale, 1.0, rtol=0.0, atol=1e-15)
        else f"({_scale_text(scale)}) * ({name})"
    )


def _contexts(
    conditioning: Sequence[BasisArchiveEntry],
    contexts: Sequence[tuple[int, ...]],
    target: np.ndarray,
) -> tuple[_Context, ...]:
    output: list[_Context] = []
    for indices in contexts:
        design = np.column_stack(
            (
                np.ones(target.size, dtype=float),
                *(conditioning[index].values for index in indices),
            )
        )
        coefficients, *_ = np.linalg.lstsq(design, target, rcond=None)
        residual = target - design @ np.round(coefficients, 6)
        value_span = orthonormal_signature_span(
            [design[:, index] for index in range(design.shape[1])]
        )
        signature_span = (
            orthonormal_signature_span(
                [conditioning[index].signature for index in indices]
            )
            if indices
            else None
        )
        output.append(
            _Context(
                indices=indices,
                residual=residual,
                value_span=value_span,
                signature_span=signature_span,
            )
        )
    return tuple(output)


def _conditional_correlation(values: np.ndarray, context: _Context) -> float:
    tolerance = np.finfo(float).eps
    orthogonal = values - context.value_span @ (context.value_span.T @ values)
    denominator = float(np.linalg.norm(orthogonal)) * float(
        np.linalg.norm(context.residual)
    )
    if denominator <= tolerance:
        return 0.0
    return float(
        np.clip(abs(float(np.dot(orthogonal, context.residual))) / denominator, 0.0, 1.0)
    )


def _best_value_context(
    values: np.ndarray, contexts: Sequence[_Context]
) -> tuple[float, tuple[int, ...]]:
    ranked = [
        (_conditional_correlation(values, context), context.indices)
        for context in contexts
    ]
    best = max(item[0] for item in ranked)
    equivalent = [item for item in ranked if item[0] >= best - 1e-12]
    return min(equivalent, key=lambda item: (len(item[1]), item[1], -item[0]))


def _best_sobolev_context(
    values: np.ndarray,
    signature: np.ndarray,
    contexts: Sequence[_Context],
) -> tuple[float, float, float, tuple[int, ...]]:
    tolerance = np.finfo(float).eps
    norm = float(np.linalg.norm(signature))
    normalized = signature / max(norm, tolerance)
    ranked: list[tuple[float, float, float, tuple[int, ...]]] = []
    for context in contexts:
        correlation = _conditional_correlation(values, context)
        if context.signature_span is None:
            gain = 1.0
        else:
            residual = normalized - context.signature_span @ (
                context.signature_span.T @ normalized
            )
            gain = float(np.clip(np.linalg.norm(residual), 0.0, 1.0))
        ranked.append((correlation * gain, correlation, gain, context.indices))
    joint, correlation, gain, indices = min(
        ranked,
        key=lambda item: (-item[0], -item[2], -item[1], len(item[3]), item[3]),
    )
    return correlation, gain, joint, indices


def _phase_atoms(
    direct: Sequence[BasisArchiveEntry],
    names: Sequence[str],
    scales: Sequence[float],
    transforms: Sequence[str],
    *,
    sample_count: int,
    dimension: int,
    lambda_value: float,
    lambda_gradient: float,
) -> tuple[tuple[_Atom, ...], int]:
    output: list[_Atom] = []
    failures = 0
    for axis, entry in enumerate(direct):
        for scale in scales:
            source = _source_expression(names[axis], scale)
            scaled_values = float(scale) * np.asarray(entry.values, dtype=float)
            scaled_signature = float(scale) * np.asarray(entry.signature, dtype=float)
            for transform in transforms:
                try:
                    signature = sobolev_unary_signature(
                        scaled_signature,
                        transform=transform,
                        sample_count=sample_count,
                        dimension=dimension,
                        lambda_value=lambda_value,
                        lambda_gradient=lambda_gradient,
                    )
                except ValueError:
                    failures += 1
                    continue
                values = getattr(np, transform)(scaled_values)
                if not np.all(np.isfinite(values)):
                    failures += 1
                    continue
                output.append(
                    _Atom(
                        expression=f"{transform}({source})",
                        values=values,
                        signature=signature,
                        axis=axis,
                        scale=float(scale),
                        transform=transform,
                    )
                )
    return tuple(output), failures


def _reciprocal_atoms(
    direct: Sequence[BasisArchiveEntry],
    names: Sequence[str],
    scales: Sequence[float],
    *,
    sample_count: int,
    dimension: int,
    protected_epsilon: float,
    lambda_value: float,
    lambda_gradient: float,
) -> tuple[tuple[_Atom, ...], int]:
    output: list[_Atom] = []
    failures = 0
    for axis, entry in enumerate(direct):
        for scale in scales:
            source = _source_expression(names[axis], scale)
            scaled_values = float(scale) * np.asarray(entry.values, dtype=float)
            scaled_signature = float(scale) * np.asarray(entry.signature, dtype=float)
            try:
                sine_signature = sobolev_unary_signature(
                    scaled_signature,
                    transform="sin",
                    sample_count=sample_count,
                    dimension=dimension,
                    lambda_value=lambda_value,
                    lambda_gradient=lambda_gradient,
                )
                cosine_signature = sobolev_unary_signature(
                    scaled_signature,
                    transform="cos",
                    sample_count=sample_count,
                    dimension=dimension,
                    lambda_value=lambda_value,
                    lambda_gradient=lambda_gradient,
                )
                signature = sobolev_division_signature(
                    cosine_signature,
                    sine_signature,
                    sample_count=sample_count,
                    dimension=dimension,
                    protected_epsilon=protected_epsilon,
                    lambda_value=lambda_value,
                    lambda_gradient=lambda_gradient,
                )
            except ValueError:
                failures += 1
                continue
            tangent = np.tan(scaled_values)
            guarded = tangent + protected_epsilon * (tangent == 0.0)
            with np.errstate(all="ignore"):
                values = 1.0 / guarded
            if not np.all(np.isfinite(values)):
                failures += 1
                continue
            output.append(
                _Atom(
                    expression=f"cot({source})",
                    values=values,
                    signature=signature,
                    axis=axis,
                    scale=float(scale),
                    transform="cot",
                )
            )
    return tuple(output), failures


def _amplitudes(
    direct: Sequence[BasisArchiveEntry],
    names: Sequence[str],
    *,
    sample_count: int,
    dimension: int,
    protected_epsilon: float,
    lambda_value: float,
    lambda_gradient: float,
) -> tuple[tuple[_Amplitude, ...], int]:
    constant_signature = np.zeros(sample_count * (dimension + 1), dtype=float)
    constant_signature[:sample_count] = np.sqrt(lambda_value / sample_count)
    output = [
        _Amplitude(
            expression="1",
            values=np.ones_like(np.asarray(direct[0].values, dtype=float)),
            signature=constant_signature,
        )
    ]
    failures = 0
    for axis, entry in enumerate(direct):
        output.append(
            _Amplitude(
                expression=names[axis],
                values=np.asarray(entry.values, dtype=float),
                signature=np.asarray(entry.signature, dtype=float),
            )
        )
        denominator = np.asarray(entry.values, dtype=float)
        guarded = denominator + protected_epsilon * (denominator == 0.0)
        try:
            signature = sobolev_division_signature(
                constant_signature,
                entry.signature,
                sample_count=sample_count,
                dimension=dimension,
                protected_epsilon=protected_epsilon,
                lambda_value=lambda_value,
                lambda_gradient=lambda_gradient,
            )
        except ValueError:
            failures += 1
            continue
        with np.errstate(all="ignore"):
            values = 1.0 / guarded
        if not np.all(np.isfinite(values)):
            failures += 1
            continue
        output.append(
            _Amplitude(
                expression=f"1 / ({names[axis]})",
                values=values,
                signature=signature,
            )
        )
    return tuple(output), failures


def _pair_expression(
    amplitude: _Amplitude, phase: _Atom, reciprocal: _Atom
) -> str:
    core = f"({phase.expression}) * ({reciprocal.expression})"
    return core if amplitude.expression == "1" else f"({amplitude.expression}) * ({core})"


def lift_direct_reciprocal_trig_pairs(
    *,
    direct_entries: Sequence[BasisArchiveEntry],
    conditioning_entries: Sequence[BasisArchiveEntry],
    target: Sequence[float],
    variable_names: Sequence[str],
    geometry_sample_count: int,
    contexts: Sequence[Sequence[int]],
    scales: Sequence[float],
    transforms: Sequence[str],
    value_pool_size: int,
    shortlist_size: int,
    proposal_limit: int,
    protected_epsilon: float = 1e-6,
    lambda_value: float = 1.0,
    lambda_gradient: float = 1.0,
) -> ReciprocalTrigResult:
    """Build complete ``A*U(z_i)*cot(z_j)`` pairs before screening."""

    direct = tuple(direct_entries)
    conditioning = tuple(conditioning_entries)
    names = tuple(str(value) for value in variable_names)
    y = np.asarray(target, dtype=float).reshape(-1)
    dimension = len(names)
    signature_size = geometry_sample_count * (dimension + 1)
    unique_contexts = tuple(
        dict.fromkeys(tuple(sorted(int(index) for index in value)) for value in contexts)
    )
    frozen_scales = tuple(dict.fromkeys(float(value) for value in scales))
    operations = tuple(dict.fromkeys(str(value) for value in transforms))
    if (
        not direct
        or len(direct) != dimension
        or y.size < 2
        or not np.all(np.isfinite(y))
        or float(np.var(y)) <= np.finfo(float).eps
        or geometry_sample_count < 1
        or not unique_contexts
        or not frozen_scales
        or not operations
        or any(value not in {"sin", "cos", "tanh"} for value in operations)
        or value_pool_size < shortlist_size
        or shortlist_size < 1
        or proposal_limit < 1
        or protected_epsilon <= 0.0
    ):
        raise ValueError("reciprocal-trigonometric pursuit inputs are invalid")
    if any(
        len(set(context)) != len(context)
        or any(index < 0 or index >= len(conditioning) for index in context)
        for context in unique_contexts
    ):
        raise ValueError("reciprocal-trigonometric contexts are invalid")
    if any(
        np.asarray(entry.values).shape != y.shape
        or np.asarray(entry.signature).shape != (signature_size,)
        or not np.all(np.isfinite(entry.values))
        or not np.all(np.isfinite(entry.signature))
        for entry in (*direct, *conditioning)
    ):
        raise ValueError("reciprocal-trigonometric entries are not aligned")

    context_geometry = _contexts(conditioning, unique_contexts, y)
    phases, phase_failures = _phase_atoms(
        direct,
        names,
        frozen_scales,
        operations,
        sample_count=geometry_sample_count,
        dimension=dimension,
        lambda_value=lambda_value,
        lambda_gradient=lambda_gradient,
    )
    reciprocals, reciprocal_failures = _reciprocal_atoms(
        direct,
        names,
        frozen_scales,
        sample_count=geometry_sample_count,
        dimension=dimension,
        protected_epsilon=protected_epsilon,
        lambda_value=lambda_value,
        lambda_gradient=lambda_gradient,
    )
    amplitudes, amplitude_failures = _amplitudes(
        direct,
        names,
        sample_count=geometry_sample_count,
        dimension=dimension,
        protected_epsilon=protected_epsilon,
        lambda_value=lambda_value,
        lambda_gradient=lambda_gradient,
    )
    numeric_failures = phase_failures + reciprocal_failures + amplitude_failures

    value_pairs: list[_ValuePair] = []
    for phase in phases:
        for reciprocal in reciprocals:
            phase_pair = phase.values * reciprocal.values
            for amplitude in amplitudes:
                values = amplitude.values * phase_pair
                if not np.all(np.isfinite(values)):
                    numeric_failures += 1
                    continue
                correlation, context = _best_value_context(values, context_geometry)
                value_pairs.append(
                    _ValuePair(
                        expression=_pair_expression(amplitude, phase, reciprocal),
                        values=values,
                        phase=phase,
                        reciprocal=reciprocal,
                        amplitude=amplitude,
                        target_correlation=correlation,
                        value_context=context,
                    )
                )
    value_pairs.sort(
        key=lambda item: (-item.target_correlation, item.expression, item.value_context)
    )
    retained = value_pairs[:value_pool_size]

    scored: list[_ScoredPair] = []
    for item in retained:
        try:
            phase_reciprocal = sobolev_product_signature(
                item.phase.signature,
                item.reciprocal.signature,
                sample_count=geometry_sample_count,
                dimension=dimension,
                lambda_value=lambda_value,
                lambda_gradient=lambda_gradient,
            )
            signature = sobolev_product_signature(
                item.amplitude.signature,
                phase_reciprocal,
                sample_count=geometry_sample_count,
                dimension=dimension,
                lambda_value=lambda_value,
                lambda_gradient=lambda_gradient,
            )
        except ValueError:
            numeric_failures += 1
            continue
        correlation, gain, joint, context = _best_sobolev_context(
            item.values, signature, context_geometry
        )
        scored.append(
            _ScoredPair(
                value_pair=item,
                signature=signature,
                target_correlation=item.target_correlation,
                value_context=item.value_context,
                sobolev_correlation=correlation,
                sobolev_gain=gain,
                joint_score=joint,
                sobolev_context=context,
                selection_lanes=(),
            )
        )

    value_lane = sorted(
        scored,
        key=lambda item: (
            -item.target_correlation,
            item.value_pair.expression,
            item.value_context,
        ),
    )[:shortlist_size]
    sobolev_lane = sorted(
        scored,
        key=lambda item: (
            -item.joint_score,
            -item.sobolev_gain,
            -item.sobolev_correlation,
            item.value_pair.expression,
            item.sobolev_context,
        ),
    )[:shortlist_size]
    lanes: dict[tuple[int, tuple[int, ...]], list[str]] = {}
    for item in value_lane:
        lanes.setdefault((id(item), item.value_context), []).append("value_pair")
    for item in sobolev_lane:
        lanes.setdefault((id(item), item.sobolev_context), []).append("sobolev_pair")

    proposals: list[ReciprocalTrigProposal] = []
    variance = float(np.var(y))
    for item in (*value_lane, *sobolev_lane):
        for context in (item.value_context, item.sobolev_context):
            lane_names = lanes.get((id(item), context))
            if not lane_names:
                continue
            context_expressions = tuple(
                conditioning[index].expression for index in context
            )
            pieces = (*context_expressions, item.value_pair.expression)
            expression = " + ".join(f"({piece})" for piece in pieces)
            design = np.column_stack(
                (
                    np.ones(y.size, dtype=float),
                    *(conditioning[index].values for index in context),
                    item.value_pair.values,
                )
            )
            coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
            prediction = design @ np.round(coefficients, 6)
            training_r2 = float(
                1.0 - np.mean(np.square(prediction - y)) / variance
            )
            proposals.append(
                ReciprocalTrigProposal(
                    expression=expression,
                    phase_expression=item.value_pair.phase.expression,
                    reciprocal_expression=item.value_pair.reciprocal.expression,
                    amplitude_expression=item.value_pair.amplitude.expression,
                    phase_axis=item.value_pair.phase.axis,
                    reciprocal_axis=item.value_pair.reciprocal.axis,
                    phase_scale=item.value_pair.phase.scale,
                    reciprocal_scale=item.value_pair.reciprocal.scale,
                    transform=item.value_pair.phase.transform,
                    context=context,
                    context_expressions=context_expressions,
                    training_r2=training_r2,
                    target_correlation=(
                        item.target_correlation
                        if context == item.value_context
                        else item.sobolev_correlation
                    ),
                    sobolev_gain=item.sobolev_gain,
                    joint_score=item.joint_score,
                    selection_lanes=tuple(lane_names),
                )
            )

    proposals.sort(
        key=lambda item: (
            -item.training_r2,
            -item.joint_score,
            -item.sobolev_gain,
            -item.target_correlation,
            item.expression,
        )
    )
    unique: list[ReciprocalTrigProposal] = []
    by_expression: dict[str, int] = {}
    for proposal in proposals:
        existing = by_expression.get(proposal.expression)
        if existing is not None:
            merged = tuple(
                dict.fromkeys(
                    (*unique[existing].selection_lanes, *proposal.selection_lanes)
                )
            )
            unique[existing] = replace(unique[existing], selection_lanes=merged)
            continue
        by_expression[proposal.expression] = len(unique)
        unique.append(proposal)
        if len(unique) >= proposal_limit:
            break

    return ReciprocalTrigResult(
        proposals=tuple(unique),
        contexts_screened=len(context_geometry),
        phase_atoms=len(phases),
        reciprocal_atoms=len(reciprocals),
        amplitude_candidates=len(amplitudes),
        pair_candidates=len(value_pairs),
        value_pool=len(retained),
        pair_shortlist=len(lanes),
        numeric_failures=numeric_failures,
    )
