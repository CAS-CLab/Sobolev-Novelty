"""Bounded second-layer interactions over Sobolev-screened phase features."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from ...nd2py import nd2py as nd
from ...sobolev.decomposition import decompose_expand_mul, parse_expression
from .sn_archive_interactions import (
    sobolev_product_signature,
    sobolev_unary_signature,
)
from .sn_basis_archive import BasisArchiveEntry
from .sn_direct_composition import sobolev_division_signature
from .sn_population_coverage import orthonormal_signature_span


@dataclass(frozen=True)
class PhaseInteractionLift:
    """One retained second-layer phase interaction."""

    entry: BasisArchiveEntry
    interaction_kind: str
    phase_expression: str
    partner_expression: str | None
    denominator_expression: str | None
    amplitude_expression: str | None
    sobolev_gain: float
    target_correlation: float
    value_residual_gain: float
    joint_score: float
    selection_lanes: tuple[str, ...]
    joint_context: tuple[int, ...] | None
    value_context: tuple[int, ...] | None


@dataclass(frozen=True)
class PhaseInteractionLiftResult:
    """Compact audit result for one bounded second-layer screen."""

    lifts: tuple[PhaseInteractionLift, ...]
    retained_phase_count: int
    coordinate_phase_count: int
    algebraic_partner_count: int
    numeric_candidates: int
    candidate_context_states: int
    contexts_screened: int
    numeric_failures: int
    construction_failures: int
    canonical_duplicates: int


@dataclass(frozen=True)
class _NumericFeature:
    expression: str
    values: np.ndarray
    signature: np.ndarray
    kind: str


@dataclass(frozen=True)
class _NumericInteraction:
    ordinal: int
    expression: str
    values: np.ndarray
    signature: np.ndarray
    interaction_kind: str
    phase_expression: str
    partner_expression: str | None = None
    denominator_expression: str | None = None
    amplitude_expression: str | None = None


@dataclass(frozen=True)
class _ContextGeometry:
    indices: tuple[int, ...]
    residual: np.ndarray
    value_span: np.ndarray
    signature_span: np.ndarray | None


@dataclass(frozen=True)
class _ScoredInteraction:
    candidate: _NumericInteraction
    score: float
    sobolev_gain: float
    target_correlation: float
    value_residual_gain: float
    context: tuple[int, ...]


def _phase_scale_expression(scale: float) -> str:
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
    raise ValueError(f"phase scale {scale!r} is not in the frozen library")


def _protected_divide_values(
    numerator: np.ndarray,
    denominator: np.ndarray,
    epsilon: float,
) -> np.ndarray:
    guarded = denominator + epsilon * (denominator == 0.0)
    with np.errstate(all="ignore"):
        return numerator / guarded


def _entry_feature(entry: BasisArchiveEntry, kind: str) -> _NumericFeature:
    return _NumericFeature(
        expression=str(entry.expression),
        values=np.asarray(entry.values, dtype=float),
        signature=np.asarray(entry.signature, dtype=float),
        kind=kind,
    )


def _coordinate_phases(
    direct: Sequence[BasisArchiveEntry],
    *,
    scales: Sequence[float],
    transforms: Sequence[str],
    include_squares: bool,
    sample_count: int,
    dimension: int,
    lambda_value: float,
    lambda_gradient: float,
) -> tuple[list[_NumericFeature], int]:
    features: list[_NumericFeature] = []
    failures = 0
    for entry in direct:
        source = _entry_feature(entry, "coordinate")
        for scale in scales:
            scale_value = float(scale)
            scale_expression = _phase_scale_expression(scale_value)
            scaled_values = scale_value * source.values
            scaled_signature = scale_value * source.signature
            scaled_expression = (
                source.expression
                if scale_expression == "1"
                else f"({scale_expression}) * ({source.expression})"
            )
            for transform in transforms:
                if transform == "sin":
                    values = np.sin(scaled_values)
                elif transform == "cos":
                    values = np.cos(scaled_values)
                elif transform == "tanh":
                    values = np.tanh(scaled_values)
                else:
                    raise ValueError("phase interactions require bounded GP transforms")
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
                expression = f"{transform}({scaled_expression})"
                if np.all(np.isfinite(values)) and np.all(np.isfinite(signature)):
                    features.append(
                        _NumericFeature(
                            expression=expression,
                            values=values,
                            signature=signature,
                            kind="coordinate_phase",
                        )
                    )
                else:
                    failures += 1
                if not include_squares:
                    continue
                try:
                    square_signature = sobolev_product_signature(
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
                square_values = values * values
                if np.all(np.isfinite(square_values)) and np.all(
                    np.isfinite(square_signature)
                ):
                    features.append(
                        _NumericFeature(
                            expression=f"({expression})**2",
                            values=square_values,
                            signature=square_signature,
                            kind="coordinate_phase_square",
                        )
                    )
                else:
                    failures += 1
    return features, failures


def _algebraic_partners(
    direct: Sequence[BasisArchiveEntry],
    supplied: Sequence[BasisArchiveEntry],
    *,
    sample_count: int,
    dimension: int,
    lambda_value: float,
    lambda_gradient: float,
) -> tuple[list[_NumericFeature], int]:
    partners = [_entry_feature(entry, "coordinate") for entry in direct]
    failures = 0
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
                    sample_count=sample_count,
                    dimension=dimension,
                    lambda_value=lambda_value,
                    lambda_gradient=lambda_gradient,
                )
            except ValueError:
                failures += 1
                continue
            if not np.all(np.isfinite(values)):
                failures += 1
                continue
            partners.append(
                _NumericFeature(
                    expression=f"({left.expression}) * ({right.expression})",
                    values=values,
                    signature=signature,
                    kind="direct_monomial2",
                )
            )

    seen = {entry.canonical for entry in (*direct,)}
    for entry in supplied:
        if entry.canonical in seen:
            continue
        seen.add(entry.canonical)
        partners.append(_entry_feature(entry, "supplied_partner"))
    return partners, failures


def _interaction_candidates(
    *,
    retained_phases: Sequence[_NumericFeature],
    coordinate_phases: Sequence[_NumericFeature],
    partners: Sequence[_NumericFeature],
    direct: Sequence[_NumericFeature],
    sample_count: int,
    dimension: int,
    protected_epsilon: float,
    lambda_value: float,
    lambda_gradient: float,
) -> Iterable[_NumericInteraction]:
    seen: set[str] = set()
    ordinal = 0

    def product(
        left: _NumericFeature,
        right: _NumericFeature,
    ) -> tuple[np.ndarray, np.ndarray]:
        values = left.values * right.values
        signature = sobolev_product_signature(
            left.signature,
            right.signature,
            sample_count=sample_count,
            dimension=dimension,
            lambda_value=lambda_value,
            lambda_gradient=lambda_gradient,
        )
        return values, signature

    def division(
        numerator: _NumericFeature,
        denominator: _NumericFeature,
    ) -> tuple[np.ndarray, np.ndarray]:
        values = _protected_divide_values(
            numerator.values, denominator.values, protected_epsilon
        )
        signature = sobolev_division_signature(
            numerator.signature,
            denominator.signature,
            sample_count=sample_count,
            dimension=dimension,
            protected_epsilon=protected_epsilon,
            lambda_value=lambda_value,
            lambda_gradient=lambda_gradient,
        )
        return values, signature

    def emit(
        expression: str,
        values: np.ndarray,
        signature: np.ndarray,
        *,
        interaction_kind: str,
        phase_expression: str,
        partner_expression: str | None = None,
        denominator_expression: str | None = None,
        amplitude_expression: str | None = None,
    ) -> _NumericInteraction | None:
        nonlocal ordinal
        if expression in seen:
            return None
        seen.add(expression)
        if not np.all(np.isfinite(values)) or not np.all(np.isfinite(signature)):
            return None
        ordinal += 1
        return _NumericInteraction(
            ordinal=ordinal,
            expression=expression,
            values=values,
            signature=signature,
            interaction_kind=interaction_kind,
            phase_expression=phase_expression,
            partner_expression=partner_expression,
            denominator_expression=denominator_expression,
            amplitude_expression=amplitude_expression,
        )

    all_phases = (*retained_phases, *coordinate_phases)
    for phase in all_phases:
        for partner in partners:
            try:
                values, signature = product(phase, partner)
            except ValueError:
                continue
            candidate = emit(
                f"({phase.expression}) * ({partner.expression})",
                values,
                signature,
                interaction_kind="phase_times_algebraic",
                phase_expression=phase.expression,
                partner_expression=partner.expression,
            )
            if candidate is not None:
                yield candidate

    for left_index, left in enumerate(retained_phases):
        for right_index in range(left_index + 1, len(retained_phases)):
            right = retained_phases[right_index]
            try:
                values, signature = product(left, right)
            except ValueError:
                continue
            candidate = emit(
                f"({left.expression}) * ({right.expression})",
                values,
                signature,
                interaction_kind="retained_phase_product",
                phase_expression=left.expression,
                partner_expression=right.expression,
            )
            if candidate is not None:
                yield candidate
        for right in retained_phases:
            if right is left:
                continue
            try:
                values, signature = division(left, right)
            except ValueError:
                continue
            candidate = emit(
                f"(({left.expression}) / ({right.expression}))",
                values,
                signature,
                interaction_kind="retained_phase_ratio",
                phase_expression=left.expression,
                denominator_expression=right.expression,
            )
            if candidate is not None:
                yield candidate

    for phase in retained_phases:
        for denominator in coordinate_phases:
            try:
                ratio_values, ratio_signature = division(phase, denominator)
            except ValueError:
                continue
            ratio = _NumericFeature(
                expression=f"(({phase.expression}) / ({denominator.expression}))",
                values=ratio_values,
                signature=ratio_signature,
                kind="retained_coordinate_phase_ratio",
            )
            candidate = emit(
                ratio.expression,
                ratio.values,
                ratio.signature,
                interaction_kind="retained_over_coordinate_phase",
                phase_expression=phase.expression,
                denominator_expression=denominator.expression,
            )
            if candidate is not None:
                yield candidate
            try:
                reverse_values, reverse_signature = division(denominator, phase)
            except ValueError:
                reverse_values = reverse_signature = None
            if reverse_values is not None and reverse_signature is not None:
                candidate = emit(
                    f"(({denominator.expression}) / ({phase.expression}))",
                    reverse_values,
                    reverse_signature,
                    interaction_kind="coordinate_over_retained_phase",
                    phase_expression=denominator.expression,
                    denominator_expression=phase.expression,
                )
                if candidate is not None:
                    yield candidate
            for amplitude in direct:
                try:
                    values, signature = product(amplitude, ratio)
                except ValueError:
                    continue
                candidate = emit(
                    f"({amplitude.expression}) * ({ratio.expression})",
                    values,
                    signature,
                    interaction_kind="coordinate_times_phase_ratio",
                    phase_expression=phase.expression,
                    denominator_expression=denominator.expression,
                    amplitude_expression=amplitude.expression,
                )
                if candidate is not None:
                    yield candidate


def _context_geometry(
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


def _rank_key(item: _ScoredInteraction) -> tuple[float, float, float, str, int]:
    return (
        -item.score,
        -item.sobolev_gain,
        -item.target_correlation,
        item.candidate.expression,
        item.candidate.ordinal,
    )


def _screen_batch(
    batch: Sequence[_NumericInteraction],
    contexts: Sequence[_ContextGeometry],
) -> tuple[list[_ScoredInteraction], list[_ScoredInteraction]]:
    if not batch:
        return [], []
    values = np.column_stack([candidate.values for candidate in batch])
    raw_signatures = np.column_stack([candidate.signature for candidate in batch])
    signature_norms = np.linalg.norm(raw_signatures, axis=0)
    tolerance = np.finfo(float).eps
    valid = signature_norms > tolerance
    valid &= np.all(np.isfinite(values), axis=0)
    valid &= np.all(np.isfinite(raw_signatures), axis=0)
    if not np.all(valid):
        batch = [candidate for index, candidate in enumerate(batch) if valid[index]]
        values = values[:, valid]
        raw_signatures = raw_signatures[:, valid]
        signature_norms = signature_norms[valid]
    if not batch:
        return [], []
    signatures = raw_signatures / signature_norms
    centered_norms = np.linalg.norm(
        values - np.mean(values, axis=0, keepdims=True), axis=0
    )
    count = len(batch)
    joint_score = np.zeros(count, dtype=float)
    joint_gain = np.zeros(count, dtype=float)
    joint_correlation = np.zeros(count, dtype=float)
    joint_value_gain = np.zeros(count, dtype=float)
    joint_context: list[tuple[int, ...]] = [()] * count
    value_score = np.zeros(count, dtype=float)
    value_gain = np.zeros(count, dtype=float)
    value_correlation = np.zeros(count, dtype=float)
    value_sobolev_gain = np.zeros(count, dtype=float)
    value_context: list[tuple[int, ...]] = [()] * count

    for context in contexts:
        orthogonal = values - context.value_span @ (context.value_span.T @ values)
        orthogonal_norms = np.linalg.norm(orthogonal, axis=0)
        residual_norm = float(np.linalg.norm(context.residual))
        correlations = np.zeros(count, dtype=float)
        valid_value = (orthogonal_norms > tolerance) & (residual_norm > tolerance)
        correlations[valid_value] = np.abs(
            orthogonal[:, valid_value].T @ context.residual
        ) / (orthogonal_norms[valid_value] * residual_norm)
        gains_value = np.zeros(count, dtype=float)
        nonzero_centered = centered_norms > tolerance
        gains_value[nonzero_centered] = (
            orthogonal_norms[nonzero_centered] / centered_norms[nonzero_centered]
        )
        correlations = np.clip(correlations, 0.0, 1.0)
        gains_value = np.clip(gains_value, 0.0, 1.0)
        if context.signature_span is None:
            gains_sobolev = np.ones(count, dtype=float)
        else:
            residual_signatures = signatures - context.signature_span @ (
                context.signature_span.T @ signatures
            )
            gains_sobolev = np.clip(
                np.linalg.norm(residual_signatures, axis=0), 0.0, 1.0
            )
        current_joint = correlations * gains_sobolev
        current_value = correlations * gains_value
        for index in np.flatnonzero(current_joint > joint_score + tolerance):
            joint_score[index] = current_joint[index]
            joint_gain[index] = gains_sobolev[index]
            joint_correlation[index] = correlations[index]
            joint_value_gain[index] = gains_value[index]
            joint_context[index] = context.indices
        for index in np.flatnonzero(current_value > value_score + tolerance):
            value_score[index] = current_value[index]
            value_gain[index] = gains_value[index]
            value_correlation[index] = correlations[index]
            value_sobolev_gain[index] = gains_sobolev[index]
            value_context[index] = context.indices

    joint = [
        _ScoredInteraction(
            candidate=batch[index],
            score=float(joint_score[index]),
            sobolev_gain=float(joint_gain[index]),
            target_correlation=float(joint_correlation[index]),
            value_residual_gain=float(joint_value_gain[index]),
            context=joint_context[index],
        )
        for index in range(count)
        if joint_score[index] > tolerance
    ]
    value = [
        _ScoredInteraction(
            candidate=batch[index],
            score=float(value_score[index]),
            sobolev_gain=float(value_sobolev_gain[index]),
            target_correlation=float(value_correlation[index]),
            value_residual_gain=float(value_gain[index]),
            context=value_context[index],
        )
        for index in range(count)
        if value_score[index] > tolerance
    ]
    return joint, value


def lift_direct_conditional_phase_interactions(
    *,
    retained_phase_entries: Sequence[BasisArchiveEntry],
    direct_entries: Sequence[BasisArchiveEntry],
    partner_entries: Sequence[BasisArchiveEntry],
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
    batch_size: int = 512,
    protected_epsilon: float = 1e-6,
    lambda_value: float = 1.0,
    lambda_gradient: float = 1.0,
) -> PhaseInteractionLiftResult:
    """Build and screen a bounded second compositional layer.

    Complete raw-coordinate phases provide low-marginal-credit denominators,
    while the first-layer retained phases define the expensive ratio anchors.
    Screening is batched so the full interaction matrix is never resident.
    """

    retained = tuple(retained_phase_entries)
    direct = tuple(direct_entries)
    supplied = tuple(partner_entries)
    conditioning = tuple(conditioning_entries)
    names = tuple(str(value) for value in variable_names)
    y = np.asarray(target, dtype=float).reshape(-1)
    unique_contexts = tuple(
        dict.fromkeys(
            tuple(sorted(int(index) for index in value)) for value in contexts
        )
    )
    phase_scales = tuple(dict.fromkeys(float(value) for value in scales))
    operations = tuple(dict.fromkeys(str(value) for value in transforms))
    dimension = len(names)
    signature_size = geometry_sample_count * (dimension + 1)
    all_entries = (*retained, *direct, *supplied, *conditioning)
    if (
        not retained
        or not direct
        or len(direct) != dimension
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
        or batch_size < 1
        or protected_epsilon <= 0.0
    ):
        raise ValueError("phase interaction inputs are invalid")
    if any(
        len(set(context)) != len(context)
        or any(index < 0 or index >= len(conditioning) for index in context)
        for context in unique_contexts
    ):
        raise ValueError("phase interaction contexts are invalid")
    if any(
        np.asarray(entry.values).shape != y.shape
        or np.asarray(entry.signature).shape != (signature_size,)
        or not np.all(np.isfinite(entry.values))
        or not np.all(np.isfinite(entry.signature))
        for entry in all_entries
    ):
        raise ValueError("phase interaction entries are not aligned")

    coordinate_phases, phase_failures = _coordinate_phases(
        direct,
        scales=phase_scales,
        transforms=operations,
        include_squares=include_squares,
        sample_count=geometry_sample_count,
        dimension=dimension,
        lambda_value=lambda_value,
        lambda_gradient=lambda_gradient,
    )
    partners, partner_failures = _algebraic_partners(
        direct,
        supplied,
        sample_count=geometry_sample_count,
        dimension=dimension,
        lambda_value=lambda_value,
        lambda_gradient=lambda_gradient,
    )
    retained_numeric = [_entry_feature(entry, "retained_phase") for entry in retained]
    direct_numeric = [_entry_feature(entry, "coordinate") for entry in direct]
    context_geometry = _context_geometry(conditioning, unique_contexts, y)

    joint_best: list[_ScoredInteraction] = []
    value_best: list[_ScoredInteraction] = []
    numeric_candidates = 0
    batch: list[_NumericInteraction] = []

    def retain_batch() -> None:
        nonlocal joint_best, value_best
        joint, value = _screen_batch(batch, context_geometry)
        joint_best = sorted((*joint_best, *joint), key=_rank_key)[
            :joint_shortlist_size
        ]
        value_best = sorted((*value_best, *value), key=_rank_key)[
            :value_shortlist_size
        ]
        batch.clear()

    for candidate in _interaction_candidates(
        retained_phases=retained_numeric,
        coordinate_phases=coordinate_phases,
        partners=partners,
        direct=direct_numeric,
        sample_count=geometry_sample_count,
        dimension=dimension,
        protected_epsilon=protected_epsilon,
        lambda_value=lambda_value,
        lambda_gradient=lambda_gradient,
    ):
        numeric_candidates += 1
        batch.append(candidate)
        if len(batch) >= batch_size:
            retain_batch()
    if batch:
        retain_batch()

    selected: list[int] = []
    lanes: dict[int, list[str]] = {}
    scores: dict[tuple[int, str], _ScoredInteraction] = {}
    for item in joint_best:
        ordinal = item.candidate.ordinal
        selected.append(ordinal)
        lanes[ordinal] = ["sobolev_conditional_phase_interaction"]
        scores[(ordinal, "joint")] = item
    for item in value_best:
        ordinal = item.candidate.ordinal
        if ordinal not in lanes:
            selected.append(ordinal)
            lanes[ordinal] = []
        lanes[ordinal].append("value_conditional_phase_interaction")
        scores[(ordinal, "value")] = item
    candidates = {
        item.candidate.ordinal: item.candidate for item in (*joint_best, *value_best)
    }
    existing = {entry.canonical for entry in all_entries}
    lifts: list[PhaseInteractionLift] = []
    construction_failures = 0
    canonical_duplicates = 0
    tolerance = np.finfo(float).eps
    for ordinal in selected:
        candidate = candidates[ordinal]
        try:
            normalized_expression = nd.parse(candidate.expression).to_str(
                number_format=".17g"
            )
            expression, symbols = parse_expression(normalized_expression, names)
            terms = tuple(
                term
                for term in decompose_expand_mul(expression, symbols)
                if term.basis.free_symbols
            )
            if len(terms) != 1 or abs(float(terms[0].coefficient)) <= tolerance:
                raise ValueError("interaction did not remain one structural basis")
            term = terms[0]
            coefficient = float(term.coefficient)
            if term.canonical in existing:
                canonical_duplicates += 1
                continue
            existing.add(term.canonical)
            signature = candidate.signature / coefficient
            values = candidate.values / coefficient
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
                source_candidate_id=-(100000 + ordinal),
            )
        except (TypeError, ValueError, KeyError):
            construction_failures += 1
            continue
        joint_item = scores.get((ordinal, "joint"))
        value_item = scores.get((ordinal, "value"))
        primary = joint_item or value_item
        assert primary is not None
        lifts.append(
            PhaseInteractionLift(
                entry=entry,
                interaction_kind=candidate.interaction_kind,
                phase_expression=candidate.phase_expression,
                partner_expression=candidate.partner_expression,
                denominator_expression=candidate.denominator_expression,
                amplitude_expression=candidate.amplitude_expression,
                sobolev_gain=primary.sobolev_gain,
                target_correlation=primary.target_correlation,
                value_residual_gain=primary.value_residual_gain,
                joint_score=0.0 if joint_item is None else joint_item.score,
                selection_lanes=tuple(lanes[ordinal]),
                joint_context=None if joint_item is None else joint_item.context,
                value_context=None if value_item is None else value_item.context,
            )
        )
    return PhaseInteractionLiftResult(
        lifts=tuple(lifts),
        retained_phase_count=len(retained),
        coordinate_phase_count=len(coordinate_phases),
        algebraic_partner_count=len(partners),
        numeric_candidates=numeric_candidates,
        candidate_context_states=numeric_candidates * len(unique_contexts),
        contexts_screened=len(unique_contexts),
        numeric_failures=phase_failures + partner_failures,
        construction_failures=construction_failures,
        canonical_duplicates=canonical_duplicates,
    )
