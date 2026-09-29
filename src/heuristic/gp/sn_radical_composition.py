"""Amplitude-conditioned Sobolev construction of positive algebraic radicals."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations_with_replacement
from typing import Sequence

import numpy as np

from ...nd2py import nd2py as nd
from ...sobolev.decomposition import decompose_expand_mul, parse_expression
from .sn_archive_interactions import sobolev_product_signature
from .sn_basis_archive import BasisArchiveEntry
from .sn_direct_composition import (
    lift_direct_conditional_phase_features,
    sobolev_division_signature,
)
from .sn_population_coverage import orthonormal_signature_span


@dataclass(frozen=True)
class DirectAmplitudeLift:
    """One screened algebraic amplitude used as its own conditional context."""

    entry: BasisArchiveEntry
    numerator_indices: tuple[int, ...]
    denominator_indices: tuple[int, ...]
    sobolev_gain: float
    target_correlation: float
    value_residual_gain: float
    joint_score: float
    selection_lanes: tuple[str, ...]
    joint_context: tuple[int, ...] | None
    value_context: tuple[int, ...] | None
    reused_existing: bool


@dataclass(frozen=True)
class RadicalPhaseCompositeLift:
    """One amplitude × positive-radical × phase term."""

    entry: BasisArchiveEntry
    amplitude_canonical: str
    phase_canonical: str
    phase_expression: str
    radical_expression: str
    radical_numerator_indices: tuple[int, ...]
    radical_denominator_indices: tuple[int, ...]
    radical_scale: float
    sobolev_gain: float
    target_correlation: float
    value_residual_gain: float
    joint_score: float
    selection_lanes: tuple[str, ...]
    reused_existing: bool


@dataclass(frozen=True)
class DirectRadicalCompositionResult:
    """Compact audit result for the two-stage radical modulation screen."""

    amplitude_lifts: tuple[DirectAmplitudeLift, ...]
    composite_lifts: tuple[RadicalPhaseCompositeLift, ...]
    amplitude_candidates: int
    radical_ratio_candidates: int
    positive_radicals: int
    phase_anchors: int
    composite_candidates: int
    amplitude_context_states: int
    composite_context_states: int
    amplitude_phase_candidates: int
    amplitude_phase_failures: int
    positivity_rejections: int
    numeric_failures: int
    construction_failures: int
    canonical_reuses: int


@dataclass(frozen=True)
class _Monomial:
    indices: tuple[int, ...]
    expression: str
    values: np.ndarray
    signature: np.ndarray


@dataclass(frozen=True)
class _RatioFeature:
    numerator_indices: tuple[int, ...]
    denominator_indices: tuple[int, ...]
    expression: str
    values: np.ndarray
    signature: np.ndarray


@dataclass(frozen=True)
class _RadicalFeature:
    numerator_indices: tuple[int, ...]
    denominator_indices: tuple[int, ...]
    scale: float
    expression: str
    values: np.ndarray
    signature: np.ndarray


@dataclass(frozen=True)
class _ContextGeometry:
    indices: tuple[int, ...]
    residual: np.ndarray
    value_span: np.ndarray
    signature_span: np.ndarray | None


@dataclass(frozen=True)
class _FeatureScores:
    best_joint: float
    joint_gain: float
    joint_correlation: float
    joint_value_gain: float
    joint_context: tuple[int, ...] | None
    best_value: float
    value_gain: float
    value_correlation: float
    value_sobolev_gain: float
    value_context: tuple[int, ...] | None


@dataclass(frozen=True)
class _SelectedAmplitude:
    feature: _RatioFeature
    lift: DirectAmplitudeLift
    phase_entries: tuple[BasisArchiveEntry, ...]


@dataclass(frozen=True)
class _CompositeCandidate:
    expression: str
    values: np.ndarray
    signature: np.ndarray
    amplitude: _SelectedAmplitude
    phase: BasisArchiveEntry
    radical: _RadicalFeature


@dataclass(frozen=True)
class _ScoredComposite:
    candidate: _CompositeCandidate
    score: float
    sobolev_gain: float
    target_correlation: float
    value_residual_gain: float


def sobolev_positive_sqrt_signature(
    source: Sequence[float],
    *,
    sample_count: int,
    dimension: int,
    minimum_value: float = 1e-12,
    lambda_value: float = 1.0,
    lambda_gradient: float = 1.0,
) -> np.ndarray:
    """Apply the exact square-root chain rule on a strictly positive source."""

    vector = np.asarray(source, dtype=float).reshape(-1)
    expected = sample_count * (dimension + 1)
    if (
        sample_count < 1
        or dimension < 0
        or minimum_value <= 0.0
        or lambda_value <= 0.0
        or (dimension and lambda_gradient <= 0.0)
        or vector.shape != (expected,)
        or not np.all(np.isfinite(vector))
    ):
        raise ValueError("positive-sqrt signature inputs are not aligned")
    value_factor = float(np.sqrt(lambda_value / sample_count))
    values = vector[:sample_count] / value_factor
    if np.any(values <= minimum_value):
        raise ValueError("positive-sqrt source is outside its real smooth domain")
    roots = np.sqrt(values)
    blocks = [value_factor * roots]
    if dimension:
        gradient_factor = float(np.sqrt(lambda_gradient / (sample_count * dimension)))
        multiplier = 0.5 / roots
        for axis in range(dimension):
            start = sample_count * (axis + 1)
            stop = start + sample_count
            gradient = vector[start:stop] / gradient_factor
            blocks.append(gradient_factor * multiplier * gradient)
    output = np.concatenate(blocks)
    if not np.all(np.isfinite(output)):
        raise ValueError("positive-sqrt signature is non-finite")
    return output


def _scale_expression(scale: float) -> str:
    if np.isclose(scale, 0.5, rtol=0.0, atol=1e-15):
        return "(1 / 2)"
    if np.isclose(scale, 1.0, rtol=0.0, atol=1e-15):
        return "1"
    if np.isclose(scale, 2.0, rtol=0.0, atol=1e-15):
        return "2"
    raise ValueError(f"radical scale {scale!r} is not in the frozen library")


def _monomial_expression(
    exponents: Sequence[int], variable_names: Sequence[str]
) -> str:
    numerator: list[str] = []
    denominator: list[str] = []
    for exponent, name in zip(exponents, variable_names, strict=True):
        target = numerator if exponent >= 0 else denominator
        target.extend([str(name)] * abs(int(exponent)))
    numerator_text = " * ".join(numerator) or "1"
    if not denominator:
        return numerator_text
    return f"({numerator_text}) / ({' * '.join(denominator)})"


def _product_expression(parts: Sequence[str]) -> str:
    retained = [str(part) for part in parts if str(part) != "1"]
    return " * ".join(f"({part})" for part in retained) or "1"


def _apply_monomial_factor(
    exponents: Sequence[int], variable_names: Sequence[str], expression: str
) -> str:
    numerator: list[str] = []
    denominator: list[str] = []
    for exponent, name in zip(exponents, variable_names, strict=True):
        target = numerator if exponent >= 0 else denominator
        target.extend([str(name)] * abs(int(exponent)))
    numerator_parts = [*numerator, f"({expression})"]
    numerator_text = " * ".join(numerator_parts)
    if not denominator:
        return numerator_text
    return f"({numerator_text}) / ({' * '.join(denominator)})"


def compact_positive_amplitude_radical_expression(
    *,
    amplitude: DirectAmplitudeLift,
    composite: RadicalPhaseCompositeLift,
    variable_names: Sequence[str],
) -> str:
    """Factor ``A + A*sqrt(1+sR)*P`` without relaxing the tree cap.

    The identity uses ``sqrt(A**2) == A`` and is therefore enabled by the
    caller only when every direct coordinate is strictly positive on the
    search rows.
    """

    names = tuple(str(value) for value in variable_names)
    dimension = len(names)
    if dimension < 1 or composite.amplitude_canonical != amplitude.entry.canonical:
        raise ValueError("radical compaction inputs are not aligned")

    amplitude_exponents = np.zeros(dimension, dtype=int)
    radical_exponents = np.zeros(dimension, dtype=int)
    for index in amplitude.numerator_indices:
        amplitude_exponents[index] += 1
    for index in amplitude.denominator_indices:
        amplitude_exponents[index] -= 1
    for index in composite.radical_numerator_indices:
        radical_exponents[index] += 1
    for index in composite.radical_denominator_indices:
        radical_exponents[index] -= 1

    squared_amplitude = 2 * amplitude_exponents
    modulated = squared_amplitude + radical_exponents
    minimum = np.minimum(squared_amplitude, modulated)
    even_factor = 2 * np.floor_divide(minimum, 2)
    radical_outer = np.floor_divide(even_factor, 2)
    common = np.minimum(amplitude_exponents, radical_outer)
    amplitude_remainder = amplitude_exponents - common
    radical_outer_remainder = radical_outer - common
    inner_base = squared_amplitude - even_factor
    inner_modulated = modulated - even_factor
    inner_common = np.minimum(inner_base, inner_modulated)
    inner_base_remainder = inner_base - inner_common
    inner_modulated_remainder = inner_modulated - inner_common

    amplitude_text = _monomial_expression(amplitude_remainder, names)
    outer_text = _monomial_expression(radical_outer_remainder, names)
    inner_common_text = _monomial_expression(inner_common, names)
    inner_base_text = _monomial_expression(inner_base_remainder, names)
    inner_modulated_text = _monomial_expression(inner_modulated_remainder, names)
    scale_text = _scale_expression(composite.radical_scale)
    scaled_modulated = _product_expression((scale_text, inner_modulated_text))
    inner_sum = f"({inner_base_text}) + ({scaled_modulated})"
    inner_text = _product_expression((inner_common_text, inner_sum))
    radical_text = f"sqrt({inner_text})"
    modulated_text = _product_expression(
        (outer_text, radical_text, composite.phase_expression)
    )
    bracket = f"({amplitude_text}) + ({modulated_text})"
    return _apply_monomial_factor(common, names, bracket)


def _protected_divide_values(
    numerator: np.ndarray,
    denominator: np.ndarray,
    epsilon: float,
) -> np.ndarray:
    guarded = denominator + epsilon * (denominator == 0.0)
    with np.errstate(all="ignore"):
        return numerator / guarded


def _monomials(
    direct: Sequence[BasisArchiveEntry],
    *,
    degree: int,
    sample_count: int,
    dimension: int,
    lambda_value: float,
    lambda_gradient: float,
) -> tuple[list[_Monomial], int]:
    output: list[_Monomial] = []
    failures = 0
    for indices in combinations_with_replacement(range(len(direct)), degree):
        values = np.asarray(direct[indices[0]].values, dtype=float).copy()
        signature = np.asarray(direct[indices[0]].signature, dtype=float).copy()
        expression_parts = [str(direct[indices[0]].expression)]
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
            expression_parts.append(str(entry.expression))
        if valid and np.all(np.isfinite(values)) and np.all(np.isfinite(signature)):
            output.append(
                _Monomial(
                    indices=tuple(indices),
                    expression=" * ".join(expression_parts),
                    values=values,
                    signature=signature,
                )
            )
        elif valid:
            failures += 1
    return output, failures


def _coprime_ratios(
    monomials: Sequence[_Monomial],
    *,
    sample_count: int,
    dimension: int,
    protected_epsilon: float,
    lambda_value: float,
    lambda_gradient: float,
) -> tuple[list[_RatioFeature], int]:
    output: list[_RatioFeature] = []
    failures = 0
    for numerator in monomials:
        numerator_support = set(numerator.indices)
        for denominator in monomials:
            if not numerator_support.isdisjoint(denominator.indices):
                continue
            values = _protected_divide_values(
                numerator.values, denominator.values, protected_epsilon
            )
            try:
                signature = sobolev_division_signature(
                    numerator.signature,
                    denominator.signature,
                    sample_count=sample_count,
                    dimension=dimension,
                    protected_epsilon=protected_epsilon,
                    lambda_value=lambda_value,
                    lambda_gradient=lambda_gradient,
                )
            except ValueError:
                failures += 1
                continue
            if not np.all(np.isfinite(values)):
                failures += 1
                continue
            output.append(
                _RatioFeature(
                    numerator_indices=numerator.indices,
                    denominator_indices=denominator.indices,
                    expression=(
                        f"(({numerator.expression}) / ({denominator.expression}))"
                    ),
                    values=values,
                    signature=signature,
                )
            )
    return output, failures


def _contexts(
    conditioning: Sequence[BasisArchiveEntry],
    contexts: Sequence[tuple[int, ...]],
    target: np.ndarray,
) -> tuple[_ContextGeometry, ...]:
    output: list[_ContextGeometry] = []
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
            _ContextGeometry(
                indices=context,
                residual=residual,
                value_span=value_span,
                signature_span=signature_span,
            )
        )
    return tuple(output)


def _score_feature(
    values: np.ndarray,
    signature: np.ndarray,
    contexts: Sequence[_ContextGeometry],
) -> _FeatureScores:
    tolerance = np.finfo(float).eps
    centered_norm = float(np.linalg.norm(values - float(np.mean(values))))
    signature_norm = float(np.linalg.norm(signature))
    normalized_signature = signature / max(signature_norm, tolerance)
    best_joint = best_value = 0.0
    joint_gain = joint_correlation = joint_value_gain = 0.0
    value_gain = value_correlation = value_sobolev_gain = 0.0
    joint_context = value_context = None
    for context in contexts:
        orthogonal = values - context.value_span @ (context.value_span.T @ values)
        orthogonal_norm = float(np.linalg.norm(orthogonal))
        residual_norm = float(np.linalg.norm(context.residual))
        correlation = (
            0.0
            if orthogonal_norm <= tolerance or residual_norm <= tolerance
            else abs(float(np.dot(orthogonal, context.residual)))
            / (orthogonal_norm * residual_norm)
        )
        current_value_gain = orthogonal_norm / max(centered_norm, tolerance)
        if context.signature_span is None:
            current_sobolev_gain = 1.0
        else:
            residual_signature = normalized_signature - context.signature_span @ (
                context.signature_span.T @ normalized_signature
            )
            current_sobolev_gain = float(np.linalg.norm(residual_signature))
        correlation = float(np.clip(correlation, 0.0, 1.0))
        current_value_gain = float(np.clip(current_value_gain, 0.0, 1.0))
        current_sobolev_gain = float(np.clip(current_sobolev_gain, 0.0, 1.0))
        current_joint = correlation * current_sobolev_gain
        current_value = correlation * current_value_gain
        if current_joint > best_joint + tolerance:
            best_joint = current_joint
            joint_gain = current_sobolev_gain
            joint_correlation = correlation
            joint_value_gain = current_value_gain
            joint_context = context.indices
        if current_value > best_value + tolerance:
            best_value = current_value
            value_gain = current_value_gain
            value_correlation = correlation
            value_sobolev_gain = current_sobolev_gain
            value_context = context.indices
    return _FeatureScores(
        best_joint=best_joint,
        joint_gain=joint_gain,
        joint_correlation=joint_correlation,
        joint_value_gain=joint_value_gain,
        joint_context=joint_context,
        best_value=best_value,
        value_gain=value_gain,
        value_correlation=value_correlation,
        value_sobolev_gain=value_sobolev_gain,
        value_context=value_context,
    )


def _materialize(
    *,
    expression_text: str,
    values: np.ndarray,
    signature: np.ndarray,
    variable_names: Sequence[str],
    existing: dict[str, BasisArchiveEntry],
    candidate_id: int,
    source_rank: int,
) -> tuple[BasisArchiveEntry, bool]:
    normalized = nd.parse(expression_text).to_str(number_format=".17g")
    expression, symbols = parse_expression(normalized, variable_names)
    terms = tuple(
        term
        for term in decompose_expand_mul(expression, symbols)
        if term.basis.free_symbols
    )
    tolerance = np.finfo(float).eps
    if len(terms) != 1 or abs(float(terms[0].coefficient)) <= tolerance:
        raise ValueError("radical composition did not remain one structural basis")
    term = terms[0]
    reused = existing.get(term.canonical)
    if reused is not None:
        return reused, True
    coefficient = float(term.coefficient)
    structural_signature = signature / coefficient
    structural_values = values / coefficient
    entry = BasisArchiveEntry(
        canonical=term.canonical,
        expression=normalized,
        signature=structural_signature,
        values=structural_values,
        term_norm=float(np.linalg.norm(structural_signature)),
        source_coefficient=1.0,
        source_amplitude=float(np.linalg.norm(structural_signature)),
        source_base_reward=0.0,
        source_generation=-1,
        source_base_rank=source_rank,
        source_candidate_id=candidate_id,
    )
    existing[entry.canonical] = entry
    return entry, False


def _radicals(
    ratios: Sequence[_RatioFeature],
    *,
    scales: Sequence[float],
    sample_count: int,
    dimension: int,
    minimum_value: float,
    lambda_value: float,
    lambda_gradient: float,
) -> tuple[list[_RadicalFeature], int, int]:
    value_factor = float(np.sqrt(lambda_value / sample_count))
    constant_signature = np.zeros(sample_count * (dimension + 1), dtype=float)
    constant_signature[:sample_count] = value_factor
    output: list[_RadicalFeature] = []
    positivity_rejections = 0
    failures = 0
    for ratio in ratios:
        for scale in scales:
            scale_value = float(scale)
            scale_text = _scale_expression(scale_value)
            inner_values = 1.0 + scale_value * ratio.values
            inner_signature = constant_signature + scale_value * ratio.signature
            if np.any(inner_values <= minimum_value):
                positivity_rejections += 1
                continue
            try:
                signature = sobolev_positive_sqrt_signature(
                    inner_signature,
                    sample_count=sample_count,
                    dimension=dimension,
                    minimum_value=minimum_value,
                    lambda_value=lambda_value,
                    lambda_gradient=lambda_gradient,
                )
            except ValueError:
                positivity_rejections += 1
                continue
            values = np.sqrt(inner_values)
            if not np.all(np.isfinite(values)):
                failures += 1
                continue
            scaled_ratio = (
                ratio.expression
                if scale_text == "1"
                else f"({scale_text}) * ({ratio.expression})"
            )
            output.append(
                _RadicalFeature(
                    numerator_indices=ratio.numerator_indices,
                    denominator_indices=ratio.denominator_indices,
                    scale=scale_value,
                    expression=f"sqrt(1 + ({scaled_ratio}))",
                    values=values,
                    signature=signature,
                )
            )
    return output, positivity_rejections, failures


def _amplitude_context(
    amplitude: _RatioFeature,
    target: np.ndarray,
) -> _ContextGeometry:
    design = np.column_stack((np.ones(target.size, dtype=float), amplitude.values))
    coefficients, *_ = np.linalg.lstsq(design, target, rcond=None)
    residual = target - design @ np.round(coefficients, 6)
    return _ContextGeometry(
        indices=(),
        residual=residual,
        value_span=orthonormal_signature_span([design[:, 0], design[:, 1]]),
        signature_span=orthonormal_signature_span([amplitude.signature]),
    )


def _composite_rank(item: _ScoredComposite) -> tuple[float, float, float, str]:
    return (
        -item.score,
        -item.sobolev_gain,
        -item.target_correlation,
        item.candidate.expression,
    )


def lift_direct_amplitude_conditioned_radicals(
    *,
    direct_entries: Sequence[BasisArchiveEntry],
    retained_phase_entries: Sequence[BasisArchiveEntry],
    conditioning_entries: Sequence[BasisArchiveEntry],
    target: Sequence[float],
    variable_names: Sequence[str],
    geometry_sample_count: int,
    contexts: Sequence[Sequence[int]],
    radical_scales: Sequence[float],
    phase_scales: Sequence[float],
    phase_transforms: Sequence[str],
    phase_include_squares: bool,
    phase_anchor_limit: int,
    amplitude_joint_shortlist_size: int,
    amplitude_value_shortlist_size: int,
    composite_joint_shortlist_size: int,
    composite_value_shortlist_size: int,
    minimum_radical_value: float = 1e-12,
    protected_epsilon: float = 1e-6,
    lambda_value: float = 1.0,
    lambda_gradient: float = 1.0,
) -> DirectRadicalCompositionResult:
    """Screen amplitudes, then radical-phase terms in each amplitude context."""

    direct = tuple(direct_entries)
    fallback_phases = tuple(retained_phase_entries[:phase_anchor_limit])
    conditioning = tuple(conditioning_entries)
    names = tuple(str(value) for value in variable_names)
    y = np.asarray(target, dtype=float).reshape(-1)
    unique_contexts = tuple(
        dict.fromkeys(
            tuple(sorted(int(index) for index in value)) for value in contexts
        )
    )
    scales = tuple(dict.fromkeys(float(value) for value in radical_scales))
    dimension = len(names)
    signature_size = geometry_sample_count * (dimension + 1)
    all_entries = (*direct, *fallback_phases, *conditioning)
    if (
        not direct
        or len(direct) != dimension
        or not fallback_phases
        or y.size < 1
        or not np.all(np.isfinite(y))
        or geometry_sample_count < 1
        or not unique_contexts
        or not scales
        or any(value <= 0.0 or not np.isfinite(value) for value in scales)
        or phase_anchor_limit < 1
        or amplitude_joint_shortlist_size < 1
        or amplitude_value_shortlist_size < 1
        or composite_joint_shortlist_size < 1
        or composite_value_shortlist_size < 1
        or minimum_radical_value <= 0.0
        or protected_epsilon <= 0.0
    ):
        raise ValueError("direct radical composition inputs are invalid")
    if any(
        len(set(context)) != len(context)
        or any(index < 0 or index >= len(conditioning) for index in context)
        for context in unique_contexts
    ):
        raise ValueError("direct radical contexts are invalid")
    if any(
        np.asarray(entry.values).shape != y.shape
        or np.asarray(entry.signature).shape != (signature_size,)
        or not np.all(np.isfinite(entry.values))
        or not np.all(np.isfinite(entry.signature))
        for entry in all_entries
    ):
        raise ValueError("direct radical entries are not aligned")

    monomial2, failures2 = _monomials(
        direct,
        degree=2,
        sample_count=geometry_sample_count,
        dimension=dimension,
        lambda_value=lambda_value,
        lambda_gradient=lambda_gradient,
    )
    amplitude_features, amplitude_failures = _coprime_ratios(
        monomial2,
        sample_count=geometry_sample_count,
        dimension=dimension,
        protected_epsilon=protected_epsilon,
        lambda_value=lambda_value,
        lambda_gradient=lambda_gradient,
    )
    inherited_contexts = _contexts(conditioning, unique_contexts, y)
    amplitude_joint_scores = [
        _score_feature(feature.values, feature.signature, inherited_contexts)
        for feature in amplitude_features
    ]
    global_context = _contexts(conditioning, ((),), y)
    amplitude_value_scores = [
        _score_feature(feature.values, feature.signature, global_context)
        for feature in amplitude_features
    ]
    tolerance = np.finfo(float).eps
    joint_order = sorted(
        (
            index
            for index, score in enumerate(amplitude_joint_scores)
            if score.best_joint > tolerance
        ),
        key=lambda index: (
            -amplitude_joint_scores[index].best_joint,
            -amplitude_joint_scores[index].joint_gain,
            -amplitude_joint_scores[index].joint_correlation,
            amplitude_features[index].expression,
        ),
    )[:amplitude_joint_shortlist_size]
    value_order = sorted(
        (
            index
            for index, score in enumerate(amplitude_value_scores)
            if score.best_value > tolerance
        ),
        key=lambda index: (
            -amplitude_value_scores[index].best_value,
            -amplitude_value_scores[index].value_correlation,
            -amplitude_value_scores[index].value_gain,
            amplitude_features[index].expression,
        ),
    )[:amplitude_value_shortlist_size]
    selected_indices: list[int] = []
    amplitude_lanes: dict[int, list[str]] = {}
    for index in joint_order:
        selected_indices.append(index)
        amplitude_lanes[index] = ["sobolev_conditional_amplitude"]
    for index in value_order:
        if index not in amplitude_lanes:
            selected_indices.append(index)
            amplitude_lanes[index] = []
        amplitude_lanes[index].append("value_global_amplitude")

    existing = {entry.canonical: entry for entry in all_entries}
    selected_amplitudes: list[_SelectedAmplitude] = []
    construction_failures = 0
    canonical_reuses = 0
    for index in selected_indices:
        feature = amplitude_features[index]
        joint_score = amplitude_joint_scores[index]
        value_score = amplitude_value_scores[index]
        joint_selected = "sobolev_conditional_amplitude" in amplitude_lanes[index]
        try:
            entry, reused = _materialize(
                expression_text=feature.expression,
                values=feature.values,
                signature=feature.signature,
                variable_names=names,
                existing=existing,
                candidate_id=-(200000 + index),
                source_rank=len(conditioning),
            )
        except (TypeError, ValueError, KeyError):
            construction_failures += 1
            continue
        canonical_reuses += int(reused)
        lift = DirectAmplitudeLift(
            entry=entry,
            numerator_indices=feature.numerator_indices,
            denominator_indices=feature.denominator_indices,
            sobolev_gain=(
                joint_score.joint_gain
                if joint_selected
                else value_score.value_sobolev_gain
            ),
            target_correlation=(
                joint_score.joint_correlation
                if joint_selected
                else value_score.value_correlation
            ),
            value_residual_gain=(
                joint_score.joint_value_gain
                if joint_selected
                else value_score.value_gain
            ),
            joint_score=joint_score.best_joint,
            selection_lanes=tuple(amplitude_lanes[index]),
            joint_context=joint_score.joint_context,
            value_context=value_score.value_context,
            reused_existing=reused,
        )
        selected_amplitudes.append(
            _SelectedAmplitude(feature=feature, lift=lift, phase_entries=())
        )

    phase_candidates = 0
    phase_failures = 0
    amplitude_phase_entries: list[_SelectedAmplitude] = []
    for selected in selected_amplitudes:
        try:
            phase_result = lift_direct_conditional_phase_features(
                direct_entries=direct,
                conditioning_entries=(selected.lift.entry,),
                target=y,
                variable_names=names,
                geometry_sample_count=geometry_sample_count,
                contexts=((0,),),
                scales=phase_scales,
                transforms=phase_transforms,
                include_squares=phase_include_squares,
                joint_shortlist_size=phase_anchor_limit,
                value_shortlist_size=phase_anchor_limit,
                protected_epsilon=protected_epsilon,
                lambda_value=lambda_value,
                lambda_gradient=lambda_gradient,
            )
            phase_candidates += phase_result.numeric_candidates
            screened_phases = tuple(
                lift.entry for lift in phase_result.lifts[:phase_anchor_limit]
            )
        except (ValueError, np.linalg.LinAlgError):
            phase_failures += 1
            screened_phases = ()
        phases = screened_phases or fallback_phases
        reused_phases: list[BasisArchiveEntry] = []
        for phase in phases:
            reused = existing.get(phase.canonical)
            if reused is None:
                existing[phase.canonical] = phase
                reused = phase
            reused_phases.append(reused)
        amplitude_phase_entries.append(
            _SelectedAmplitude(
                feature=selected.feature,
                lift=selected.lift,
                phase_entries=tuple(reused_phases),
            )
        )
    selected_amplitudes = amplitude_phase_entries

    monomial3, failures3 = _monomials(
        direct,
        degree=3,
        sample_count=geometry_sample_count,
        dimension=dimension,
        lambda_value=lambda_value,
        lambda_gradient=lambda_gradient,
    )
    radical_ratios, radical_ratio_failures = _coprime_ratios(
        monomial3,
        sample_count=geometry_sample_count,
        dimension=dimension,
        protected_epsilon=protected_epsilon,
        lambda_value=lambda_value,
        lambda_gradient=lambda_gradient,
    )
    radicals, positivity_rejections, radical_failures = _radicals(
        radical_ratios,
        scales=scales,
        sample_count=geometry_sample_count,
        dimension=dimension,
        minimum_value=minimum_radical_value,
        lambda_value=lambda_value,
        lambda_gradient=lambda_gradient,
    )

    joint_best: list[_ScoredComposite] = []
    value_best: list[_ScoredComposite] = []
    composite_candidates = 0
    phase_anchor_count = 0
    for amplitude in selected_amplitudes:
        context = _amplitude_context(amplitude.feature, y)
        phase_anchor_count += len(amplitude.phase_entries)
        for phase in amplitude.phase_entries:
            phase_values = np.asarray(phase.values, dtype=float)
            phase_signature = np.asarray(phase.signature, dtype=float)
            for radical in radicals:
                try:
                    radical_phase_signature = sobolev_product_signature(
                        radical.signature,
                        phase_signature,
                        sample_count=geometry_sample_count,
                        dimension=dimension,
                        lambda_value=lambda_value,
                        lambda_gradient=lambda_gradient,
                    )
                    signature = sobolev_product_signature(
                        amplitude.feature.signature,
                        radical_phase_signature,
                        sample_count=geometry_sample_count,
                        dimension=dimension,
                        lambda_value=lambda_value,
                        lambda_gradient=lambda_gradient,
                    )
                except ValueError:
                    continue
                values = amplitude.feature.values * radical.values * phase_values
                if not np.all(np.isfinite(values)) or not np.all(
                    np.isfinite(signature)
                ):
                    continue
                composite_candidates += 1
                candidate = _CompositeCandidate(
                    expression=(
                        f"({amplitude.feature.expression}) * "
                        f"({radical.expression}) * ({phase.expression})"
                    ),
                    values=values,
                    signature=signature,
                    amplitude=amplitude,
                    phase=phase,
                    radical=radical,
                )
                score = _score_feature(values, signature, (context,))
                if score.best_joint > tolerance:
                    joint_best.append(
                        _ScoredComposite(
                            candidate=candidate,
                            score=score.best_joint,
                            sobolev_gain=score.joint_gain,
                            target_correlation=score.joint_correlation,
                            value_residual_gain=score.joint_value_gain,
                        )
                    )
                    joint_best = sorted(joint_best, key=_composite_rank)[
                        :composite_joint_shortlist_size
                    ]
                if score.best_value > tolerance:
                    value_best.append(
                        _ScoredComposite(
                            candidate=candidate,
                            score=score.best_value,
                            sobolev_gain=score.value_sobolev_gain,
                            target_correlation=score.value_correlation,
                            value_residual_gain=score.value_gain,
                        )
                    )
                    value_best = sorted(value_best, key=_composite_rank)[
                        :composite_value_shortlist_size
                    ]

    selected_composites: list[_ScoredComposite] = []
    composite_lanes: dict[str, list[str]] = {}
    composite_scores: dict[tuple[str, str], _ScoredComposite] = {}
    for item in joint_best:
        key = item.candidate.expression
        selected_composites.append(item)
        composite_lanes[key] = ["sobolev_amplitude_conditioned_radical"]
        composite_scores[(key, "joint")] = item
    for item in value_best:
        key = item.candidate.expression
        if key not in composite_lanes:
            selected_composites.append(item)
            composite_lanes[key] = []
        composite_lanes[key].append("value_amplitude_conditioned_radical")
        composite_scores[(key, "value")] = item

    composite_lifts: list[RadicalPhaseCompositeLift] = []
    for offset, selected in enumerate(selected_composites):
        candidate = selected.candidate
        key = candidate.expression
        joint = composite_scores.get((key, "joint"))
        value = composite_scores.get((key, "value"))
        primary = joint or value
        assert primary is not None
        try:
            entry, reused = _materialize(
                expression_text=candidate.expression,
                values=candidate.values,
                signature=candidate.signature,
                variable_names=names,
                existing=existing,
                candidate_id=-(300000 + offset),
                source_rank=len(conditioning),
            )
        except (TypeError, ValueError, KeyError):
            construction_failures += 1
            continue
        canonical_reuses += int(reused)
        composite_lifts.append(
            RadicalPhaseCompositeLift(
                entry=entry,
                amplitude_canonical=candidate.amplitude.lift.entry.canonical,
                phase_canonical=candidate.phase.canonical,
                phase_expression=candidate.phase.expression,
                radical_expression=candidate.radical.expression,
                radical_numerator_indices=(
                    candidate.radical.numerator_indices
                ),
                radical_denominator_indices=(
                    candidate.radical.denominator_indices
                ),
                radical_scale=candidate.radical.scale,
                sobolev_gain=primary.sobolev_gain,
                target_correlation=primary.target_correlation,
                value_residual_gain=primary.value_residual_gain,
                joint_score=0.0 if joint is None else joint.score,
                selection_lanes=tuple(composite_lanes[key]),
                reused_existing=reused,
            )
        )

    numeric_failures = (
        failures2
        + failures3
        + amplitude_failures
        + radical_ratio_failures
        + radical_failures
    )
    return DirectRadicalCompositionResult(
        amplitude_lifts=tuple(item.lift for item in selected_amplitudes),
        composite_lifts=tuple(composite_lifts),
        amplitude_candidates=len(amplitude_features),
        radical_ratio_candidates=len(radical_ratios) * len(scales),
        positive_radicals=len(radicals),
        phase_anchors=phase_anchor_count,
        composite_candidates=composite_candidates,
        amplitude_context_states=len(amplitude_features) * len(unique_contexts),
        composite_context_states=composite_candidates,
        amplitude_phase_candidates=phase_candidates,
        amplitude_phase_failures=phase_failures,
        positivity_rejections=positivity_rejections,
        numeric_failures=numeric_failures,
        construction_failures=construction_failures,
        canonical_reuses=canonical_reuses,
    )
