"""Shared-phase affine-rational pursuit screened in Sobolev geometry.

The construction is deliberately an export-side plug-in.  It does not alter
the population, GP operators, or RNG schedule.  A broad value-only screen is
followed by exact value-and-gradient conditional novelty comparisons; target
derivatives are never used.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations_with_replacement
from typing import Sequence

import numpy as np

from .sn_archive_interactions import (
    sobolev_product_signature,
    sobolev_unary_signature,
)
from .sn_basis_archive import BasisArchiveEntry
from .sn_direct_composition import sobolev_division_signature
from .sn_population_coverage import orthonormal_signature_span


@dataclass(frozen=True)
class SharedPhaseRationalProposal:
    """One compact shared-phase two-amplitude proposal."""

    expression: str
    phase_expression: str
    backbone_expression: str
    modulated_expression: str
    complement_expression: str
    backbone_exponents: tuple[int, ...]
    complement_exponents: tuple[int, ...]
    mobius_axis: int
    mobius_numerator_shift: float
    mobius_denominator_shift: float
    training_r2: float
    backbone_target_correlation: float
    modulated_target_correlation: float
    modulated_sobolev_gain: float
    complement_target_correlation: float
    complement_sobolev_gain: float
    selection_lanes: tuple[str, ...]


@dataclass(frozen=True)
class SharedPhaseRationalResult:
    """Audit result for the complete bounded pursuit."""

    proposals: tuple[SharedPhaseRationalProposal, ...]
    monomial_candidates: int
    phase_candidates: int
    backbone_candidates: int
    backbone_value_pool: int
    backbone_shortlist: int
    mobius_factors: int
    modulated_candidates: int
    modulated_shortlist: int
    complement_candidates: int
    pair_candidates: int
    numeric_failures: int


@dataclass(frozen=True)
class _Monomial:
    exponents: tuple[int, ...]
    expression: str
    values: np.ndarray
    signature: np.ndarray


@dataclass(frozen=True)
class _Phase:
    expression: str
    values: np.ndarray
    signature: np.ndarray
    transform: str
    scale: float
    axis: int
    squared: bool


@dataclass(frozen=True)
class _Context:
    residual: np.ndarray
    value_span: np.ndarray
    signature_span: np.ndarray | None


@dataclass(frozen=True)
class _Scores:
    joint: float
    correlation: float
    sobolev_gain: float
    value: float
    value_correlation: float
    value_gain: float


@dataclass(frozen=True)
class _Backbone:
    monomial: _Monomial
    phase: _Phase
    expression: str
    values: np.ndarray
    signature: np.ndarray
    target_correlation: float
    sobolev_gain: float
    joint_score: float
    selection_lanes: tuple[str, ...]


@dataclass(frozen=True)
class _MobiusFactor:
    axis: int
    numerator_shift: float
    denominator_shift: float
    expression: str
    values: np.ndarray
    signature: np.ndarray


@dataclass(frozen=True)
class _Modulated:
    backbone: _Backbone
    factor: _MobiusFactor
    expression: str
    values: np.ndarray
    signature: np.ndarray
    target_correlation: float
    sobolev_gain: float
    joint_score: float
    selection_lanes: tuple[str, ...]


def sobolev_affine_ratio_signature(
    source: Sequence[float],
    *,
    numerator_shift: float,
    denominator_shift: float,
    sample_count: int,
    dimension: int,
    protected_epsilon: float = 1e-6,
    lambda_value: float = 1.0,
    lambda_gradient: float = 1.0,
) -> np.ndarray:
    """Exact signature of ``(source+a)/(source+b)``."""

    vector = np.asarray(source, dtype=float).reshape(-1)
    expected = sample_count * (dimension + 1)
    if (
        sample_count < 1
        or dimension < 0
        or lambda_value <= 0.0
        or (dimension and lambda_gradient <= 0.0)
        or vector.shape != (expected,)
        or not np.all(np.isfinite(vector))
        or not np.isfinite(numerator_shift)
        or not np.isfinite(denominator_shift)
    ):
        raise ValueError("affine-ratio signature inputs are not aligned")
    value_factor = float(np.sqrt(lambda_value / sample_count))
    constant = np.zeros(expected, dtype=float)
    constant[:sample_count] = value_factor
    return sobolev_division_signature(
        vector + float(numerator_shift) * constant,
        vector + float(denominator_shift) * constant,
        sample_count=sample_count,
        dimension=dimension,
        protected_epsilon=protected_epsilon,
        lambda_value=lambda_value,
        lambda_gradient=lambda_gradient,
    )


def _scale_expression(scale: float) -> str:
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
    raise ValueError(f"phase scale {scale!r} is outside the frozen library")


def _number_text(value: float) -> str:
    integer = int(round(value))
    if np.isclose(value, integer, rtol=0.0, atol=1e-15):
        return str(abs(integer))
    return format(abs(float(value)), ".17g")


def _shifted_coordinate(name: str, shift: float) -> str:
    if shift > 0.0:
        return f"({name} + {_number_text(shift)})"
    return f"({name} - {_number_text(shift)})"


def _monomial_expression(
    exponents: Sequence[int], variable_names: Sequence[str]
) -> str:
    numerator: list[str] = []
    denominator: list[str] = []
    for exponent, name in zip(exponents, variable_names, strict=True):
        if exponent == 0:
            continue
        power = abs(int(exponent))
        piece = str(name) if power == 1 else f"{name} ** {power}"
        (numerator if exponent > 0 else denominator).append(piece)
    numerator_text = " * ".join(numerator) or "1"
    if not denominator:
        return numerator_text
    return f"({numerator_text}) / ({' * '.join(denominator)})"


def _positive_monomials(
    direct: Sequence[BasisArchiveEntry],
    *,
    max_degree: int,
    sample_count: int,
    dimension: int,
    lambda_value: float,
    lambda_gradient: float,
) -> tuple[dict[tuple[int, ...], tuple[np.ndarray, np.ndarray]], int]:
    output: dict[tuple[int, ...], tuple[np.ndarray, np.ndarray]] = {}
    failures = 0
    for degree in range(1, max_degree + 1):
        for indices in combinations_with_replacement(range(dimension), degree):
            exponents = tuple(indices.count(axis) for axis in range(dimension))
            values = np.asarray(direct[indices[0]].values, dtype=float).copy()
            signature = np.asarray(
                direct[indices[0]].signature, dtype=float
            ).copy()
            valid = True
            for index in indices[1:]:
                entry = direct[index]
                values = values * np.asarray(entry.values, dtype=float)
                try:
                    signature = sobolev_product_signature(
                        signature,
                        entry.signature,
                        sample_count=sample_count,
                        dimension=dimension,
                        lambda_value=lambda_value,
                        lambda_gradient=lambda_gradient,
                    )
                except ValueError:
                    failures += 1
                    valid = False
                    break
            if valid and np.all(np.isfinite(values)) and np.all(
                np.isfinite(signature)
            ):
                output[exponents] = (values, signature)
            elif valid:
                failures += 1
    return output, failures


def _monomial_ratios(
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
) -> tuple[tuple[_Monomial, ...], int]:
    positive, failures = _positive_monomials(
        direct,
        max_degree=max(max_numerator_degree, max_denominator_degree),
        sample_count=sample_count,
        dimension=dimension,
        lambda_value=lambda_value,
        lambda_gradient=lambda_gradient,
    )
    output: list[_Monomial] = []
    for numerator_exponents, (numerator_values, numerator_signature) in positive.items():
        numerator_degree = sum(numerator_exponents)
        if numerator_degree > max_numerator_degree:
            continue
        output.append(
            _Monomial(
                exponents=numerator_exponents,
                expression=_monomial_expression(
                    numerator_exponents, variable_names
                ),
                values=numerator_values,
                signature=numerator_signature,
            )
        )
        numerator_support = {
            axis for axis, exponent in enumerate(numerator_exponents) if exponent
        }
        for denominator_exponents, (
            denominator_values,
            denominator_signature,
        ) in positive.items():
            if (
                sum(denominator_exponents) > max_denominator_degree
                or numerator_support.intersection(
                    axis
                    for axis, exponent in enumerate(denominator_exponents)
                    if exponent
                )
            ):
                continue
            guarded = denominator_values + protected_epsilon * (
                denominator_values == 0.0
            )
            with np.errstate(all="ignore"):
                values = numerator_values / guarded
            try:
                signature = sobolev_division_signature(
                    numerator_signature,
                    denominator_signature,
                    sample_count=sample_count,
                    dimension=dimension,
                    protected_epsilon=protected_epsilon,
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
            exponents = tuple(
                left - right
                for left, right in zip(
                    numerator_exponents, denominator_exponents, strict=True
                )
            )
            output.append(
                _Monomial(
                    exponents=exponents,
                    expression=_monomial_expression(exponents, variable_names),
                    values=values,
                    signature=signature,
                )
            )
    output.sort(key=lambda item: item.exponents)
    return tuple(output), failures


def _phases(
    direct: Sequence[BasisArchiveEntry],
    *,
    variable_names: Sequence[str],
    scales: Sequence[float],
    transforms: Sequence[str],
    include_squares: bool,
    sample_count: int,
    dimension: int,
    lambda_value: float,
    lambda_gradient: float,
) -> tuple[tuple[_Phase, ...], int]:
    output: list[_Phase] = []
    failures = 0
    for axis, entry in enumerate(direct):
        source_values = np.asarray(entry.values, dtype=float)
        source_signature = np.asarray(entry.signature, dtype=float)
        for scale in scales:
            scale_value = float(scale)
            scale_text = _scale_expression(scale_value)
            scaled_values = scale_value * source_values
            scaled_signature = scale_value * source_signature
            source_text = (
                str(variable_names[axis])
                if scale_text == "1"
                else f"({scale_text}) * ({variable_names[axis]})"
            )
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
                expression = f"{transform}({source_text})"
                output.append(
                    _Phase(
                        expression=expression,
                        values=values,
                        signature=signature,
                        transform=transform,
                        scale=scale_value,
                        axis=axis,
                        squared=False,
                    )
                )
                if include_squares:
                    try:
                        squared_signature = sobolev_product_signature(
                            signature,
                            signature,
                            sample_count=sample_count,
                            dimension=dimension,
                            lambda_value=lambda_value,
                            lambda_gradient=lambda_gradient,
                        )
                    except ValueError:
                        failures += 1
                        continue
                    output.append(
                        _Phase(
                            expression=f"({expression}) ** 2",
                            values=values * values,
                            signature=squared_signature,
                            transform=transform,
                            scale=scale_value,
                            axis=axis,
                            squared=True,
                        )
                    )
    return tuple(output), failures


def _contexts(
    conditioning: Sequence[BasisArchiveEntry],
    contexts: Sequence[tuple[int, ...]],
    target: np.ndarray,
) -> tuple[_Context, ...]:
    output: list[_Context] = []
    for context in contexts:
        design = np.column_stack(
            (
                np.ones(target.size, dtype=float),
                *(conditioning[index].values for index in context),
            )
        )
        coefficients, *_ = np.linalg.lstsq(design, target, rcond=None)
        residual = target - design @ np.round(coefficients, 6)
        value_span = orthonormal_signature_span(
            [design[:, index] for index in range(design.shape[1])]
        )
        signature_span = (
            orthonormal_signature_span(
                [conditioning[index].signature for index in context]
            )
            if context
            else None
        )
        output.append(
            _Context(
                residual=residual,
                value_span=value_span,
                signature_span=signature_span,
            )
        )
    return tuple(output)


def _single_context(
    values: np.ndarray, signature: np.ndarray, target: np.ndarray
) -> _Context:
    design = np.column_stack((np.ones(target.size, dtype=float), values))
    coefficients, *_ = np.linalg.lstsq(design, target, rcond=None)
    return _Context(
        residual=target - design @ np.round(coefficients, 6),
        value_span=orthonormal_signature_span((design[:, 0], design[:, 1])),
        signature_span=orthonormal_signature_span((signature,)),
    )


def _score(
    values: np.ndarray, signature: np.ndarray, contexts: Sequence[_Context]
) -> _Scores:
    tolerance = np.finfo(float).eps
    centered_norm = float(np.linalg.norm(values - float(np.mean(values))))
    signature_norm = float(np.linalg.norm(signature))
    normalized_signature = signature / max(signature_norm, tolerance)
    best_joint = best_correlation = best_sobolev = 0.0
    best_value = best_value_correlation = best_value_gain = 0.0
    for context in contexts:
        orthogonal = values - context.value_span @ (
            context.value_span.T @ values
        )
        orthogonal_norm = float(np.linalg.norm(orthogonal))
        residual_norm = float(np.linalg.norm(context.residual))
        correlation = (
            0.0
            if orthogonal_norm <= tolerance or residual_norm <= tolerance
            else abs(float(np.dot(orthogonal, context.residual)))
            / (orthogonal_norm * residual_norm)
        )
        value_gain = orthogonal_norm / max(centered_norm, tolerance)
        if context.signature_span is None:
            sobolev_gain = 1.0
        else:
            residual_signature = normalized_signature - context.signature_span @ (
                context.signature_span.T @ normalized_signature
            )
            sobolev_gain = float(np.linalg.norm(residual_signature))
        correlation = float(np.clip(correlation, 0.0, 1.0))
        value_gain = float(np.clip(value_gain, 0.0, 1.0))
        sobolev_gain = float(np.clip(sobolev_gain, 0.0, 1.0))
        joint = correlation * sobolev_gain
        value_score = correlation * value_gain
        if joint > best_joint + tolerance:
            best_joint = joint
            best_correlation = correlation
            best_sobolev = sobolev_gain
        if value_score > best_value + tolerance:
            best_value = value_score
            best_value_correlation = correlation
            best_value_gain = value_gain
    return _Scores(
        joint=best_joint,
        correlation=best_correlation,
        sobolev_gain=best_sobolev,
        value=best_value,
        value_correlation=best_value_correlation,
        value_gain=best_value_gain,
    )


def _absolute_centered_correlations(
    matrix: np.ndarray, target: np.ndarray
) -> np.ndarray:
    centered = matrix - np.mean(matrix, axis=1, keepdims=True)
    target_centered = target - float(np.mean(target))
    denominator = np.linalg.norm(centered, axis=1) * float(
        np.linalg.norm(target_centered)
    )
    output = np.zeros(matrix.shape[0], dtype=float)
    valid = denominator > np.finfo(float).eps
    output[valid] = np.abs(centered[valid] @ target_centered) / denominator[valid]
    return np.clip(output, 0.0, 1.0)


def _mobius_factors(
    direct: Sequence[BasisArchiveEntry],
    *,
    variable_names: Sequence[str],
    shifts: Sequence[float],
    sample_count: int,
    dimension: int,
    protected_epsilon: float,
    lambda_value: float,
    lambda_gradient: float,
) -> tuple[tuple[_MobiusFactor, ...], int]:
    output: list[_MobiusFactor] = []
    failures = 0
    for axis, entry in enumerate(direct):
        source_values = np.asarray(entry.values, dtype=float)
        source_signature = np.asarray(entry.signature, dtype=float)
        for numerator_shift in shifts:
            for denominator_shift in shifts:
                if np.isclose(
                    numerator_shift,
                    denominator_shift,
                    rtol=0.0,
                    atol=1e-15,
                ):
                    continue
                numerator_values = source_values + numerator_shift
                denominator_values = source_values + denominator_shift
                guarded = denominator_values + protected_epsilon * (
                    denominator_values == 0.0
                )
                with np.errstate(all="ignore"):
                    values = numerator_values / guarded
                try:
                    signature = sobolev_affine_ratio_signature(
                        source_signature,
                        numerator_shift=numerator_shift,
                        denominator_shift=denominator_shift,
                        sample_count=sample_count,
                        dimension=dimension,
                        protected_epsilon=protected_epsilon,
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
                name = str(variable_names[axis])
                output.append(
                    _MobiusFactor(
                        axis=axis,
                        numerator_shift=float(numerator_shift),
                        denominator_shift=float(denominator_shift),
                        expression=(
                            f"{_shifted_coordinate(name, numerator_shift)} / "
                            f"{_shifted_coordinate(name, denominator_shift)}"
                        ),
                        values=values,
                        signature=signature,
                    )
                )
    return tuple(output), failures


def _union_by_identity(
    first: Sequence[object], second: Sequence[object]
) -> list[object]:
    output: list[object] = []
    seen: set[int] = set()
    for value in (*first, *second):
        identity = id(value)
        if identity in seen:
            continue
        seen.add(identity)
        output.append(value)
    return output


def lift_direct_shared_phase_rationals(
    *,
    direct_entries: Sequence[BasisArchiveEntry],
    conditioning_entries: Sequence[BasisArchiveEntry],
    target: Sequence[float],
    variable_names: Sequence[str],
    geometry_sample_count: int,
    contexts: Sequence[Sequence[int]],
    phase_scales: Sequence[float],
    phase_transforms: Sequence[str],
    phase_include_squares: bool,
    mobius_shifts: Sequence[float],
    max_numerator_degree: int,
    max_denominator_degree: int,
    backbone_value_pool_size: int,
    backbone_shortlist_size: int,
    modulated_shortlist_size: int,
    complement_pool_size: int,
    complement_shortlist_size: int,
    proposal_limit: int,
    protected_epsilon: float = 1e-6,
    lambda_value: float = 1.0,
    lambda_gradient: float = 1.0,
) -> SharedPhaseRationalResult:
    """Build compact ``phase * (M1 * Mobius(x) + M2)`` proposals."""

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
    transforms = tuple(dict.fromkeys(str(value) for value in phase_transforms))
    shifts = tuple(dict.fromkeys(float(value) for value in mobius_shifts))
    if (
        not direct
        or len(direct) != dimension
        or y.size < 1
        or not np.all(np.isfinite(y))
        or geometry_sample_count < 1
        or not unique_contexts
        or not scales
        or not transforms
        or any(value not in {"sin", "cos", "tanh"} for value in transforms)
        or len(shifts) < 2
        or max_numerator_degree < 1
        or max_denominator_degree < 0
        or backbone_value_pool_size < backbone_shortlist_size
        or backbone_shortlist_size < 1
        or modulated_shortlist_size < 1
        or complement_pool_size < complement_shortlist_size
        or complement_shortlist_size < 1
        or proposal_limit < 1
        or protected_epsilon <= 0.0
    ):
        raise ValueError("shared-phase rational inputs are invalid")
    if any(
        len(set(context)) != len(context)
        or any(index < 0 or index >= len(conditioning) for index in context)
        for context in unique_contexts
    ):
        raise ValueError("shared-phase rational contexts are invalid")
    if any(
        np.asarray(entry.values).shape != y.shape
        or np.asarray(entry.signature).shape != (signature_size,)
        or not np.all(np.isfinite(entry.values))
        or not np.all(np.isfinite(entry.signature))
        for entry in (*direct, *conditioning)
    ):
        raise ValueError("shared-phase rational entries are not aligned")

    monomials, monomial_failures = _monomial_ratios(
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
    phases, phase_failures = _phases(
        direct,
        variable_names=names,
        scales=scales,
        transforms=transforms,
        include_squares=phase_include_squares,
        sample_count=geometry_sample_count,
        dimension=dimension,
        lambda_value=lambda_value,
        lambda_gradient=lambda_gradient,
    )
    inherited_contexts = _contexts(conditioning, unique_contexts, y)
    monomial_matrix = np.vstack([item.values for item in monomials])

    numeric_backbones: list[tuple[float, int, int]] = []
    per_phase_pool = min(backbone_value_pool_size, len(monomials))
    for phase_index, phase in enumerate(phases):
        correlations = _absolute_centered_correlations(
            monomial_matrix * phase.values, y
        )
        indices = np.argpartition(correlations, -per_phase_pool)[-per_phase_pool:]
        numeric_backbones.extend(
            (float(correlations[index]), int(index), phase_index)
            for index in indices
        )
    numeric_backbones.sort(
        key=lambda value: (
            -value[0],
            monomials[value[1]].exponents,
            phases[value[2]].expression,
        )
    )
    numeric_backbones = numeric_backbones[:backbone_value_pool_size]

    backbone_pool: list[_Backbone] = []
    numeric_failures = monomial_failures + phase_failures
    for correlation, monomial_index, phase_index in numeric_backbones:
        monomial = monomials[monomial_index]
        phase = phases[phase_index]
        values = monomial.values * phase.values
        try:
            signature = sobolev_product_signature(
                monomial.signature,
                phase.signature,
                sample_count=geometry_sample_count,
                dimension=dimension,
                lambda_value=lambda_value,
                lambda_gradient=lambda_gradient,
            )
        except ValueError:
            numeric_failures += 1
            continue
        scores = _score(values, signature, inherited_contexts)
        backbone_pool.append(
            _Backbone(
                monomial=monomial,
                phase=phase,
                expression=f"({monomial.expression}) * ({phase.expression})",
                values=values,
                signature=signature,
                target_correlation=correlation,
                sobolev_gain=scores.sobolev_gain,
                joint_score=scores.joint,
                selection_lanes=(),
            )
        )
    value_backbones = sorted(
        backbone_pool,
        key=lambda item: (-item.target_correlation, item.expression),
    )[:backbone_shortlist_size]
    joint_backbones = sorted(
        backbone_pool,
        key=lambda item: (
            -item.joint_score,
            -item.sobolev_gain,
            -item.target_correlation,
            item.expression,
        ),
    )[:backbone_shortlist_size]
    backbone_lanes: dict[int, list[str]] = {}
    for item in value_backbones:
        backbone_lanes.setdefault(id(item), []).append("value_backbone")
    for item in joint_backbones:
        backbone_lanes.setdefault(id(item), []).append("sobolev_backbone")
    selected_backbones = [
        _Backbone(
            **{
                **item.__dict__,
                "selection_lanes": tuple(backbone_lanes[id(item)]),
            }
        )
        for item in _union_by_identity(value_backbones, joint_backbones)
    ]

    factors, factor_failures = _mobius_factors(
        direct,
        variable_names=names,
        shifts=shifts,
        sample_count=geometry_sample_count,
        dimension=dimension,
        protected_epsilon=protected_epsilon,
        lambda_value=lambda_value,
        lambda_gradient=lambda_gradient,
    )
    numeric_failures += factor_failures
    modulated_pool: list[_Modulated] = []
    for backbone in selected_backbones:
        context = _single_context(backbone.values, backbone.signature, y)
        for factor in factors:
            values = backbone.values * factor.values
            try:
                signature = sobolev_product_signature(
                    backbone.signature,
                    factor.signature,
                    sample_count=geometry_sample_count,
                    dimension=dimension,
                    lambda_value=lambda_value,
                    lambda_gradient=lambda_gradient,
                )
            except ValueError:
                numeric_failures += 1
                continue
            scores = _score(values, signature, (context,))
            correlation = float(
                _absolute_centered_correlations(values.reshape(1, -1), y)[0]
            )
            modulated_pool.append(
                _Modulated(
                    backbone=backbone,
                    factor=factor,
                    expression=(
                        f"({backbone.monomial.expression}) * "
                        f"({factor.expression}) * ({backbone.phase.expression})"
                    ),
                    values=values,
                    signature=signature,
                    target_correlation=correlation,
                    sobolev_gain=scores.sobolev_gain,
                    joint_score=scores.joint,
                    selection_lanes=(),
                )
            )
    value_modulated = sorted(
        modulated_pool,
        key=lambda item: (-item.target_correlation, item.expression),
    )[:modulated_shortlist_size]
    joint_modulated = sorted(
        modulated_pool,
        key=lambda item: (
            -item.joint_score,
            -item.sobolev_gain,
            -item.target_correlation,
            item.expression,
        ),
    )[:modulated_shortlist_size]
    modulated_lanes: dict[int, list[str]] = {}
    for item in value_modulated:
        modulated_lanes.setdefault(id(item), []).append("value_mobius")
    for item in joint_modulated:
        modulated_lanes.setdefault(id(item), []).append("sobolev_mobius")
    selected_modulated = [
        _Modulated(
            **{
                **item.__dict__,
                "selection_lanes": tuple(modulated_lanes[id(item)]),
            }
        )
        for item in _union_by_identity(value_modulated, joint_modulated)
    ]

    proposals: list[SharedPhaseRationalProposal] = []
    complement_candidates = 0
    pair_candidates = 0
    for modulated in selected_modulated:
        context = _single_context(modulated.values, modulated.signature, y)
        complement_matrix = monomial_matrix * modulated.backbone.phase.values
        orthogonal = complement_matrix - (
            complement_matrix @ context.value_span
        ) @ context.value_span.T
        residual_norm = float(np.linalg.norm(context.residual))
        denominator = np.linalg.norm(orthogonal, axis=1) * residual_norm
        correlations = np.zeros(len(monomials), dtype=float)
        valid = denominator > np.finfo(float).eps
        correlations[valid] = np.abs(
            orthogonal[valid] @ context.residual
        ) / denominator[valid]
        complement_candidates += len(monomials)
        pool_size = min(complement_pool_size, len(monomials))
        pool_indices = np.argpartition(correlations, -pool_size)[-pool_size:]
        complement_pool: list[tuple[_Monomial, np.ndarray, np.ndarray, _Scores]] = []
        for monomial_index in pool_indices:
            monomial = monomials[int(monomial_index)]
            values = complement_matrix[int(monomial_index)]
            try:
                signature = sobolev_product_signature(
                    monomial.signature,
                    modulated.backbone.phase.signature,
                    sample_count=geometry_sample_count,
                    dimension=dimension,
                    lambda_value=lambda_value,
                    lambda_gradient=lambda_gradient,
                )
            except ValueError:
                numeric_failures += 1
                continue
            complement_pool.append(
                (monomial, values, signature, _score(values, signature, (context,)))
            )
        value_complements = sorted(
            complement_pool,
            key=lambda item: (-item[3].value, item[0].expression),
        )[:complement_shortlist_size]
        joint_complements = sorted(
            complement_pool,
            key=lambda item: (
                -item[3].joint,
                -item[3].sobolev_gain,
                -item[3].correlation,
                item[0].expression,
            ),
        )[:complement_shortlist_size]
        complement_lanes: dict[int, list[str]] = {}
        for item in value_complements:
            complement_lanes.setdefault(id(item[0]), []).append("value_complement")
        for item in joint_complements:
            complement_lanes.setdefault(id(item[0]), []).append(
                "sobolev_complement"
            )
        selected_complements: list[
            tuple[_Monomial, np.ndarray, np.ndarray, _Scores]
        ] = []
        seen_exponents: set[tuple[int, ...]] = set()
        for item in (*value_complements, *joint_complements):
            if item[0].exponents in seen_exponents:
                continue
            seen_exponents.add(item[0].exponents)
            selected_complements.append(item)
        for monomial, values, _, scores in selected_complements:
            design = np.column_stack(
                (np.ones(y.size, dtype=float), modulated.values, values)
            )
            coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
            prediction = design @ np.round(coefficients, 6)
            variance = float(np.var(y))
            training_r2 = float(
                1.0 - np.mean(np.square(prediction - y)) / variance
            )
            if not np.isfinite(training_r2):
                numeric_failures += 1
                continue
            pair_candidates += 1
            common_phase = modulated.backbone.phase.expression
            amplitude = (
                f"(({modulated.backbone.monomial.expression}) * "
                f"({modulated.factor.expression})) + ({monomial.expression})"
            )
            proposals.append(
                SharedPhaseRationalProposal(
                    expression=f"({common_phase}) * ({amplitude})",
                    phase_expression=common_phase,
                    backbone_expression=modulated.backbone.expression,
                    modulated_expression=modulated.expression,
                    complement_expression=(
                        f"({monomial.expression}) * ({common_phase})"
                    ),
                    backbone_exponents=modulated.backbone.monomial.exponents,
                    complement_exponents=monomial.exponents,
                    mobius_axis=modulated.factor.axis,
                    mobius_numerator_shift=(
                        modulated.factor.numerator_shift
                    ),
                    mobius_denominator_shift=(
                        modulated.factor.denominator_shift
                    ),
                    training_r2=training_r2,
                    backbone_target_correlation=(
                        modulated.backbone.target_correlation
                    ),
                    modulated_target_correlation=(
                        modulated.target_correlation
                    ),
                    modulated_sobolev_gain=modulated.sobolev_gain,
                    complement_target_correlation=scores.correlation,
                    complement_sobolev_gain=scores.sobolev_gain,
                    selection_lanes=(
                        *modulated.backbone.selection_lanes,
                        *modulated.selection_lanes,
                        *complement_lanes[id(monomial)],
                    ),
                )
            )
    proposals.sort(
        key=lambda item: (
            -item.training_r2,
            -item.modulated_sobolev_gain,
            -item.complement_sobolev_gain,
            item.expression,
        )
    )
    unique_proposals: list[SharedPhaseRationalProposal] = []
    seen_expressions: set[str] = set()
    for proposal in proposals:
        if proposal.expression in seen_expressions:
            continue
        seen_expressions.add(proposal.expression)
        unique_proposals.append(proposal)
        if len(unique_proposals) >= proposal_limit:
            break

    return SharedPhaseRationalResult(
        proposals=tuple(unique_proposals),
        monomial_candidates=len(monomials),
        phase_candidates=len(phases),
        backbone_candidates=len(monomials) * len(phases),
        backbone_value_pool=len(backbone_pool),
        backbone_shortlist=len(selected_backbones),
        mobius_factors=len(factors),
        modulated_candidates=len(modulated_pool),
        modulated_shortlist=len(selected_modulated),
        complement_candidates=complement_candidates,
        pair_candidates=pair_candidates,
        numeric_failures=numeric_failures,
    )
