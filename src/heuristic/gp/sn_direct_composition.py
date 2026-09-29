"""Direct-coordinate compositional features screened in Sobolev geometry."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ...sobolev.decomposition import (
    decompose_expand_mul,
    parse_expression,
)
from ...nd2py import nd2py as nd
from .sn_archive_interactions import (
    sobolev_product_signature,
    sobolev_unary_signature,
)
from .sn_basis_archive import BasisArchiveEntry
from .sn_population_coverage import orthonormal_signature_span


@dataclass(frozen=True)
class DirectPhaseLift:
    """One retained bounded transform of a direct coordinate composition."""

    entry: BasisArchiveEntry
    source_kind: str
    source_expression: str
    scale: float
    scale_expression: str
    transform: str
    squared: bool
    sobolev_gain: float
    target_correlation: float
    value_residual_gain: float
    joint_score: float
    selection_lanes: tuple[str, ...]
    joint_context: tuple[int, ...] | None
    value_context: tuple[int, ...] | None


@dataclass(frozen=True)
class DirectPhaseLiftResult:
    """Compact result of one complete direct phase-library screen."""

    lifts: tuple[DirectPhaseLift, ...]
    source_count: int
    numeric_candidates: int
    contexts_screened: int
    candidate_context_states: int
    construction_failures: int
    canonical_duplicates: int


@dataclass(frozen=True)
class _NumericSource:
    kind: str
    expression: str
    values: np.ndarray
    signature: np.ndarray


@dataclass(frozen=True)
class _NumericPhase:
    source_kind: str
    source_expression: str
    expression: str
    values: np.ndarray
    signature: np.ndarray
    scale: float
    scale_expression: str
    transform: str
    squared: bool


def sobolev_division_signature(
    numerator: Sequence[float],
    denominator: Sequence[float],
    *,
    sample_count: int,
    dimension: int,
    protected_epsilon: float = 1e-6,
    lambda_value: float = 1.0,
    lambda_gradient: float = 1.0,
) -> np.ndarray:
    """Apply the exact EIC protected-division rule in SN coordinates."""

    left = np.asarray(numerator, dtype=float).reshape(-1)
    right = np.asarray(denominator, dtype=float).reshape(-1)
    expected = sample_count * (dimension + 1)
    if (
        sample_count < 1
        or dimension < 0
        or protected_epsilon <= 0.0
        or lambda_value <= 0.0
        or (dimension and lambda_gradient <= 0.0)
        or left.shape != (expected,)
        or right.shape != (expected,)
        or not np.all(np.isfinite(left))
        or not np.all(np.isfinite(right))
    ):
        raise ValueError("division signature inputs are not aligned")

    value_factor = float(np.sqrt(lambda_value / sample_count))
    a = left[:sample_count] / value_factor
    b = right[:sample_count] / value_factor
    nonzero = b != 0.0
    guarded = b + protected_epsilon * (~nonzero)
    values = a / guarded
    blocks = [value_factor * values]
    if dimension:
        gradient_factor = float(np.sqrt(lambda_gradient / (sample_count * dimension)))
        for axis in range(dimension):
            start = sample_count * (axis + 1)
            stop = start + sample_count
            da = left[start:stop] / gradient_factor
            db = right[start:stop] / gradient_factor
            gradient = np.zeros(sample_count, dtype=float)
            gradient[nonzero] = (
                da[nonzero] * b[nonzero] - a[nonzero] * db[nonzero]
            ) / (b[nonzero] * b[nonzero])
            blocks.append(gradient_factor * gradient)
    output = np.concatenate(blocks)
    if not np.all(np.isfinite(output)):
        raise ValueError("division signature is non-finite")
    return output


def _protected_divide_values(
    numerator: np.ndarray,
    denominator: np.ndarray,
    protected_epsilon: float,
) -> np.ndarray:
    guarded = denominator + protected_epsilon * (denominator == 0.0)
    with np.errstate(all="ignore"):
        return numerator / guarded


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
    raise ValueError(f"phase scale {scale!r} is not in the frozen expression library")


def lift_direct_conditional_phase_features(
    *,
    direct_entries: Sequence[BasisArchiveEntry],
    conditioning_entries: Sequence[BasisArchiveEntry],
    target: Sequence[float],
    variable_names: Sequence[str],
    geometry_sample_count: int,
    contexts: Sequence[Sequence[int]],
    scales: Sequence[float],
    transforms: Sequence[str],
    include_squares: bool,
    joint_shortlist_size: int,
    value_shortlist_size: int,
    protected_epsilon: float = 1e-6,
    lambda_value: float = 1.0,
    lambda_gradient: float = 1.0,
) -> DirectPhaseLiftResult:
    """Screen a bounded phase library built from every runtime coordinate.

    Numeric construction is complete before symbolic materialization.  Only the
    fixed joint/value shortlist is converted into ordinary GP expressions.
    """

    direct = tuple(direct_entries)
    conditioning = tuple(conditioning_entries)
    names = tuple(str(value) for value in variable_names)
    y = np.asarray(target, dtype=float).reshape(-1)
    phase_scales = tuple(dict.fromkeys(float(value) for value in scales))
    operations = tuple(dict.fromkeys(str(value) for value in transforms))
    unique_contexts = tuple(
        dict.fromkeys(
            tuple(sorted(int(index) for index in value)) for value in contexts
        )
    )
    if (
        not direct
        or len(direct) != len(names)
        or y.size < 1
        or not np.all(np.isfinite(y))
        or geometry_sample_count < 1
        or not unique_contexts
        or not phase_scales
        or any(value <= 0.0 or not np.isfinite(value) for value in phase_scales)
        or not operations
        or any(value not in {"sin", "cos", "tanh"} for value in operations)
        or joint_shortlist_size < 1
        or value_shortlist_size < 1
        or protected_epsilon <= 0.0
    ):
        raise ValueError("direct conditional phase inputs are invalid")
    if any(
        len(set(context)) != len(context)
        or any(index < 0 or index >= len(conditioning) for index in context)
        for context in unique_contexts
    ):
        raise ValueError("direct conditional phase contexts are invalid")
    if any(
        np.asarray(entry.values).shape != y.shape
        or not np.all(np.isfinite(entry.values))
        for entry in (*direct, *conditioning)
    ):
        raise ValueError("direct conditional phase values are not aligned")

    dimension = len(names)
    signature_size = geometry_sample_count * (dimension + 1)
    if any(
        np.asarray(entry.signature).shape != (signature_size,)
        or not np.all(np.isfinite(entry.signature))
        for entry in (*direct, *conditioning)
    ):
        raise ValueError("direct conditional phase signatures are not aligned")

    sources: list[_NumericSource] = []
    for entry in direct:
        sources.append(
            _NumericSource(
                kind="coordinate",
                expression=entry.expression,
                values=np.asarray(entry.values, dtype=float),
                signature=np.asarray(entry.signature, dtype=float),
            )
        )

    monomials: list[_NumericSource] = list(sources)
    for left_index, left in enumerate(direct):
        for right_index in range(left_index, len(direct)):
            right = direct[right_index]
            values = np.asarray(left.values, dtype=float) * np.asarray(
                right.values, dtype=float
            )
            try:
                signature = sobolev_product_signature(
                    left.signature,
                    right.signature,
                    sample_count=geometry_sample_count,
                    dimension=dimension,
                    lambda_value=lambda_value,
                    lambda_gradient=lambda_gradient,
                )
            except ValueError:
                continue
            monomials.append(
                _NumericSource(
                    kind="monomial2",
                    expression=f"({left.expression}) * ({right.expression})",
                    values=values,
                    signature=signature,
                )
            )

    for left_index, left in enumerate(direct):
        for right_index in range(left_index + 1, len(direct)):
            right = direct[right_index]
            for sign in (1, -1):
                sources.append(
                    _NumericSource(
                        kind="coordinate_sum" if sign > 0 else "coordinate_difference",
                        expression=(
                            f"({left.expression}) + ({right.expression})"
                            if sign > 0
                            else f"({left.expression}) - ({right.expression})"
                        ),
                        values=np.asarray(left.values, dtype=float)
                        + float(sign) * np.asarray(right.values, dtype=float),
                        signature=np.asarray(left.signature, dtype=float)
                        + float(sign) * np.asarray(right.signature, dtype=float),
                    )
                )

    sources.extend(monomials[len(direct) :])
    for numerator in monomials:
        for denominator in direct:
            values = _protected_divide_values(
                np.asarray(numerator.values, dtype=float),
                np.asarray(denominator.values, dtype=float),
                protected_epsilon,
            )
            try:
                signature = sobolev_division_signature(
                    numerator.signature,
                    denominator.signature,
                    sample_count=geometry_sample_count,
                    dimension=dimension,
                    protected_epsilon=protected_epsilon,
                    lambda_value=lambda_value,
                    lambda_gradient=lambda_gradient,
                )
            except ValueError:
                continue
            if not np.all(np.isfinite(values)):
                continue
            sources.append(
                _NumericSource(
                    kind="monomial_coordinate_ratio",
                    expression=(
                        f"(({numerator.expression}) / ({denominator.expression}))"
                    ),
                    values=values,
                    signature=signature,
                )
            )

    phases: list[_NumericPhase] = []
    for source in sources:
        for scale in phase_scales:
            scale_expression = _scale_expression(scale)
            scaled_values = float(scale) * source.values
            scaled_signature = float(scale) * source.signature
            scaled_expression = (
                source.expression
                if scale_expression == "1"
                else f"({scale_expression}) * ({source.expression})"
            )
            for transform in operations:
                if transform == "sin":
                    values = np.sin(scaled_values)
                elif transform == "cos":
                    values = np.cos(scaled_values)
                else:
                    values = np.tanh(scaled_values)
                try:
                    signature = sobolev_unary_signature(
                        scaled_signature,
                        transform=transform,
                        sample_count=geometry_sample_count,
                        dimension=dimension,
                        lambda_value=lambda_value,
                        lambda_gradient=lambda_gradient,
                    )
                except ValueError:
                    continue
                expression = f"{transform}({scaled_expression})"
                if np.all(np.isfinite(values)):
                    phases.append(
                        _NumericPhase(
                            source_kind=source.kind,
                            source_expression=source.expression,
                            expression=expression,
                            values=values,
                            signature=signature,
                            scale=scale,
                            scale_expression=scale_expression,
                            transform=transform,
                            squared=False,
                        )
                    )
                if not include_squares:
                    continue
                squared_values = values * values
                try:
                    squared_signature = sobolev_product_signature(
                        signature,
                        signature,
                        sample_count=geometry_sample_count,
                        dimension=dimension,
                        lambda_value=lambda_value,
                        lambda_gradient=lambda_gradient,
                    )
                except ValueError:
                    continue
                if np.all(np.isfinite(squared_values)):
                    phases.append(
                        _NumericPhase(
                            source_kind=source.kind,
                            source_expression=source.expression,
                            expression=f"({expression})**2",
                            values=squared_values,
                            signature=squared_signature,
                            scale=scale,
                            scale_expression=scale_expression,
                            transform=transform,
                            squared=True,
                        )
                    )

    if not phases:
        return DirectPhaseLiftResult(
            lifts=(),
            source_count=len(sources),
            numeric_candidates=0,
            contexts_screened=len(unique_contexts),
            candidate_context_states=0,
            construction_failures=0,
            canonical_duplicates=0,
        )

    value_matrix = np.column_stack([phase.values for phase in phases])
    raw_signature_matrix = np.column_stack([phase.signature for phase in phases])
    signature_norms = np.linalg.norm(raw_signature_matrix, axis=0)
    finite = np.all(np.isfinite(value_matrix), axis=0)
    finite &= np.all(np.isfinite(raw_signature_matrix), axis=0)
    finite &= signature_norms > np.finfo(float).eps
    if not np.all(finite):
        phases = [phase for index, phase in enumerate(phases) if finite[index]]
        value_matrix = value_matrix[:, finite]
        raw_signature_matrix = raw_signature_matrix[:, finite]
        signature_norms = signature_norms[finite]
    count = len(phases)
    if not count:
        return DirectPhaseLiftResult(
            lifts=(),
            source_count=len(sources),
            numeric_candidates=0,
            contexts_screened=len(unique_contexts),
            candidate_context_states=0,
            construction_failures=0,
            canonical_duplicates=0,
        )
    signature_matrix = raw_signature_matrix / signature_norms
    centered_norms = np.linalg.norm(
        value_matrix - np.mean(value_matrix, axis=0, keepdims=True), axis=0
    )
    best_joint = np.zeros(count, dtype=float)
    best_joint_gain = np.zeros(count, dtype=float)
    best_joint_correlation = np.zeros(count, dtype=float)
    best_joint_value_gain = np.zeros(count, dtype=float)
    best_value_score = np.zeros(count, dtype=float)
    best_value_gain = np.zeros(count, dtype=float)
    best_value_correlation = np.zeros(count, dtype=float)
    best_value_sobolev_gain = np.zeros(count, dtype=float)
    joint_contexts: list[tuple[int, ...] | None] = [None] * count
    value_contexts: list[tuple[int, ...] | None] = [None] * count
    tolerance = np.finfo(float).eps

    for context in unique_contexts:
        design = np.column_stack(
            (
                np.ones(y.size, dtype=float),
                *(
                    np.asarray(conditioning[index].values, dtype=float)
                    for index in context
                ),
            )
        )
        coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
        residual = y - design @ np.round(coefficients, 6)
        value_span = orthonormal_signature_span(
            [design[:, index] for index in range(design.shape[1])]
        )
        orthogonal_values = value_matrix - value_span @ (value_span.T @ value_matrix)
        orthogonal_norms = np.linalg.norm(orthogonal_values, axis=0)
        residual_norm = float(np.linalg.norm(residual))
        correlations = np.zeros(count, dtype=float)
        valid_value = (orthogonal_norms > tolerance) & (residual_norm > tolerance)
        correlations[valid_value] = np.abs(
            orthogonal_values[:, valid_value].T @ residual
        ) / (orthogonal_norms[valid_value] * residual_norm)
        value_gains = np.zeros(count, dtype=float)
        nonzero_centered = centered_norms > tolerance
        value_gains[nonzero_centered] = (
            orthogonal_norms[nonzero_centered] / centered_norms[nonzero_centered]
        )
        correlations = np.clip(correlations, 0.0, 1.0)
        value_gains = np.clip(value_gains, 0.0, 1.0)

        if context:
            signature_span = orthonormal_signature_span(
                [
                    np.asarray(conditioning[index].signature, dtype=float)
                    for index in context
                ]
            )
            signature_residuals = signature_matrix - signature_span @ (
                signature_span.T @ signature_matrix
            )
            gains = np.clip(np.linalg.norm(signature_residuals, axis=0), 0.0, 1.0)
        else:
            gains = np.ones(count, dtype=float)
        joint_scores = gains * correlations
        value_scores = correlations * value_gains
        joint_better = joint_scores > best_joint + tolerance
        value_better = value_scores > best_value_score + tolerance
        for index in np.flatnonzero(joint_better):
            best_joint[index] = joint_scores[index]
            best_joint_gain[index] = gains[index]
            best_joint_correlation[index] = correlations[index]
            best_joint_value_gain[index] = value_gains[index]
            joint_contexts[index] = context
        for index in np.flatnonzero(value_better):
            best_value_score[index] = value_scores[index]
            best_value_gain[index] = value_gains[index]
            best_value_correlation[index] = correlations[index]
            best_value_sobolev_gain[index] = gains[index]
            value_contexts[index] = context

    joint_order = sorted(
        (index for index in range(count) if best_joint[index] > tolerance),
        key=lambda index: (
            -best_joint[index],
            -best_joint_gain[index],
            -best_joint_correlation[index],
            phases[index].expression,
        ),
    )[:joint_shortlist_size]
    value_order = sorted(
        (index for index in range(count) if best_value_score[index] > tolerance),
        key=lambda index: (
            -best_value_score[index],
            -best_value_correlation[index],
            -best_value_gain[index],
            phases[index].expression,
        ),
    )[:value_shortlist_size]
    selected: list[int] = []
    lanes: dict[int, list[str]] = {}
    for index in joint_order:
        selected.append(index)
        lanes[index] = ["sobolev_conditional_phase"]
    for index in value_order:
        if index not in lanes:
            selected.append(index)
            lanes[index] = []
        lanes[index].append("value_conditional_phase")

    existing = {entry.canonical for entry in (*conditioning, *direct)}
    lifts: list[DirectPhaseLift] = []
    construction_failures = 0
    canonical_duplicates = 0
    for index in selected:
        phase = phases[index]
        try:
            normalized_expression = nd.parse(phase.expression).to_str(
                number_format=".17g"
            )
            expression, symbols = parse_expression(normalized_expression, names)
            terms = tuple(
                term
                for term in decompose_expand_mul(expression, symbols)
                if term.basis.free_symbols
            )
            if len(terms) != 1 or abs(float(terms[0].coefficient)) <= tolerance:
                raise ValueError("phase expression did not remain one structural basis")
            term = terms[0]
            coefficient = float(term.coefficient)
            if term.canonical in existing:
                canonical_duplicates += 1
                continue
            existing.add(term.canonical)
            signature = np.asarray(phase.signature, dtype=float) / coefficient
            values = np.asarray(phase.values, dtype=float) / coefficient
            entry = BasisArchiveEntry(
                canonical=term.canonical,
                expression=normalized_expression,
                signature=signature,
                values=values,
                term_norm=float(np.linalg.norm(signature)),
                source_coefficient=1.0,
                source_amplitude=float(np.linalg.norm(signature)),
                source_base_reward=0.0,
                source_generation=-1,
                source_base_rank=len(conditioning),
                source_candidate_id=-(index + 1),
            )
        except (TypeError, ValueError, KeyError):
            construction_failures += 1
            continue
        joint_selected = "sobolev_conditional_phase" in lanes[index]
        lifts.append(
            DirectPhaseLift(
                entry=entry,
                source_kind=phase.source_kind,
                source_expression=phase.source_expression,
                scale=phase.scale,
                scale_expression=phase.scale_expression,
                transform=phase.transform,
                squared=phase.squared,
                sobolev_gain=float(
                    best_joint_gain[index]
                    if joint_selected
                    else best_value_sobolev_gain[index]
                ),
                target_correlation=float(
                    best_joint_correlation[index]
                    if joint_selected
                    else best_value_correlation[index]
                ),
                value_residual_gain=float(
                    best_joint_value_gain[index]
                    if joint_selected
                    else best_value_gain[index]
                ),
                joint_score=float(best_joint[index]),
                selection_lanes=tuple(lanes[index]),
                joint_context=joint_contexts[index],
                value_context=value_contexts[index],
            )
        )
    return DirectPhaseLiftResult(
        lifts=tuple(lifts),
        source_count=len(sources),
        numeric_candidates=count,
        contexts_screened=len(unique_contexts),
        candidate_context_states=len(unique_contexts) * count,
        construction_failures=construction_failures,
        canonical_duplicates=canonical_duplicates,
    )
