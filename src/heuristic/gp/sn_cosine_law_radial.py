"""Pair-first cosine-law radial construction in Sobolev geometry.

The export-side plug-in enumerates compact expressions of the form

``sqrt(a**2 + b**2 + sign_r * 2*a*b*cos(scale*(p + sign_p*q)))``.

Complete amplitude/phase combinations are value-screened before the final
Sobolev shortlist is formed.  Candidate derivatives follow exact product,
cosine-chain, and positive-square-root rules; target derivatives and
benchmark metadata are never used.
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
from .sn_population_coverage import orthonormal_signature_span
from .sn_radical_composition import sobolev_positive_sqrt_signature
from .sn_shared_denominator import _signature_gain


@dataclass(frozen=True)
class CosineLawRadialProposal:
    """One ordinary-GP expression retained by a value or Sobolev lane."""

    expression: str
    left_amplitude_expression: str
    right_amplitude_expression: str
    left_phase_expression: str
    right_phase_expression: str
    phase_sign: int
    radial_sign: int
    phase_scale: float
    fitted_coefficients: tuple[float, float]
    training_r2: float
    prediction_correlation: float
    candidate_sobolev_gain: float
    angular_cross_gain: float
    joint_score: float
    selection_lanes: tuple[str, ...]


@dataclass(frozen=True)
class CosineLawRadialResult:
    """Compact audit result for one cosine-law radial enumeration."""

    proposals: tuple[CosineLawRadialProposal, ...]
    amplitude_pairs: int
    phase_atoms: int
    candidates_screened: int
    value_pool: int
    sobolev_candidates: int
    radicand_rejections: int
    numeric_failures: int


@dataclass(frozen=True)
class _AmplitudePair:
    left_index: int
    right_index: int
    left_square_values: np.ndarray
    right_square_values: np.ndarray
    product_values: np.ndarray
    left_square_signature: np.ndarray
    right_square_signature: np.ndarray
    product_signature: np.ndarray


@dataclass(frozen=True)
class _PhaseAtom:
    left_index: int
    right_index: int
    sign: int
    scale: float
    values: np.ndarray
    signature: np.ndarray


@dataclass(frozen=True)
class _ValueState:
    expression: str
    amplitude: _AmplitudePair
    phase: _PhaseAtom
    radial_sign: int
    coefficients: tuple[float, float]
    prediction: np.ndarray
    training_r2: float
    prediction_correlation: float


@dataclass(frozen=True)
class _ScoredState:
    value_state: _ValueState
    candidate_sobolev_gain: float
    angular_cross_gain: float
    joint_score: float
    selection_lanes: tuple[str, ...]


def _format_scale(scale: float) -> str:
    if scale == 1.0:
        return ""
    return f"({format(float(scale), '.17g')}) * "


def lift_direct_cosine_law_radials(
    *,
    direct_entries: Sequence[BasisArchiveEntry],
    target: Sequence[float],
    variable_names: Sequence[str],
    geometry_sample_count: int,
    phase_signs: Sequence[int],
    radial_signs: Sequence[int],
    phase_scales: Sequence[float],
    value_pool_size: int,
    shortlist_size: int,
    proposal_limit: int,
    minimum_radicand: float = 1e-12,
    lambda_value: float = 1.0,
    lambda_gradient: float = 1.0,
) -> CosineLawRadialResult:
    """Enumerate complete cosine-law candidates before Sobolev screening."""

    direct = tuple(direct_entries)
    names = tuple(str(value) for value in variable_names)
    y = np.asarray(target, dtype=float).reshape(-1)
    dimension = len(names)
    signature_size = geometry_sample_count * (dimension + 1)
    phase_operations = tuple(dict.fromkeys(int(value) for value in phase_signs))
    radial_operations = tuple(dict.fromkeys(int(value) for value in radial_signs))
    scales = tuple(dict.fromkeys(float(value) for value in phase_scales))
    if (
        len(direct) != dimension
        or dimension < 2
        or y.size < 2
        or not np.all(np.isfinite(y))
        or float(np.var(y)) <= np.finfo(float).eps
        or geometry_sample_count < 1
        or not phase_operations
        or any(value not in {-1, 1} for value in phase_operations)
        or not radial_operations
        or any(value not in {-1, 1} for value in radial_operations)
        or not scales
        or any(not np.isfinite(value) or value <= 0.0 for value in scales)
        or value_pool_size < shortlist_size
        or shortlist_size < 1
        or proposal_limit < 1
        or minimum_radicand <= 0.0
        or lambda_value <= 0.0
        or (dimension and lambda_gradient <= 0.0)
    ):
        raise ValueError("cosine-law radial inputs are invalid")
    if any(
        np.asarray(entry.values).shape != y.shape
        or np.asarray(entry.signature).shape != (signature_size,)
        or not np.all(np.isfinite(entry.values))
        or not np.all(np.isfinite(entry.signature))
        for entry in direct
    ):
        raise ValueError("cosine-law radial entries are not aligned")

    amplitude_pairs: list[_AmplitudePair] = []
    phase_atoms: list[_PhaseAtom] = []
    numeric_failures = 0
    for left_index in range(dimension):
        left = direct[left_index]
        for right_index in range(left_index + 1, dimension):
            right = direct[right_index]
            try:
                left_square_signature = sobolev_product_signature(
                    left.signature,
                    left.signature,
                    sample_count=geometry_sample_count,
                    dimension=dimension,
                    lambda_value=lambda_value,
                    lambda_gradient=lambda_gradient,
                )
                right_square_signature = sobolev_product_signature(
                    right.signature,
                    right.signature,
                    sample_count=geometry_sample_count,
                    dimension=dimension,
                    lambda_value=lambda_value,
                    lambda_gradient=lambda_gradient,
                )
                product_signature = sobolev_product_signature(
                    left.signature,
                    right.signature,
                    sample_count=geometry_sample_count,
                    dimension=dimension,
                    lambda_value=lambda_value,
                    lambda_gradient=lambda_gradient,
                )
            except ValueError:
                numeric_failures += 1
                continue
            amplitude_pairs.append(
                _AmplitudePair(
                    left_index=left_index,
                    right_index=right_index,
                    left_square_values=np.square(np.asarray(left.values)),
                    right_square_values=np.square(np.asarray(right.values)),
                    product_values=np.asarray(left.values) * np.asarray(right.values),
                    left_square_signature=left_square_signature,
                    right_square_signature=right_square_signature,
                    product_signature=product_signature,
                )
            )
            for phase_sign in phase_operations:
                for phase_scale in scales:
                    phase_values = phase_scale * (
                        np.asarray(left.values)
                        + float(phase_sign) * np.asarray(right.values)
                    )
                    phase_signature = phase_scale * (
                        np.asarray(left.signature)
                        + float(phase_sign) * np.asarray(right.signature)
                    )
                    try:
                        cosine_signature = sobolev_unary_signature(
                            phase_signature,
                            transform="cos",
                            sample_count=geometry_sample_count,
                            dimension=dimension,
                            lambda_value=lambda_value,
                            lambda_gradient=lambda_gradient,
                        )
                    except ValueError:
                        numeric_failures += 1
                        continue
                    phase_atoms.append(
                        _PhaseAtom(
                            left_index=left_index,
                            right_index=right_index,
                            sign=phase_sign,
                            scale=phase_scale,
                            values=np.cos(phase_values),
                            signature=cosine_signature,
                        )
                    )

    target_variance = float(np.var(y))
    centered_target = y - float(np.mean(y))
    target_norm = float(np.linalg.norm(centered_target))
    value_states: list[_ValueState] = []
    candidates_screened = 0
    radicand_rejections = 0
    for amplitude in amplitude_pairs:
        for phase in phase_atoms:
            cross_values = 2.0 * amplitude.product_values * phase.values
            for radial_sign in radial_operations:
                candidates_screened += 1
                radicand = (
                    amplitude.left_square_values
                    + amplitude.right_square_values
                    + float(radial_sign) * cross_values
                )
                if np.any(radicand <= minimum_radicand):
                    radicand_rejections += 1
                    continue
                basis = np.sqrt(radicand)
                design = np.column_stack((np.ones(y.size, dtype=float), basis))
                coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
                rounded = np.round(coefficients, 6)
                prediction = design @ rounded
                training_r2 = float(
                    1.0 - np.mean(np.square(prediction - y)) / target_variance
                )
                centered_prediction = prediction - float(np.mean(prediction))
                prediction_norm = float(np.linalg.norm(centered_prediction))
                prediction_correlation = (
                    0.0
                    if prediction_norm <= np.finfo(float).eps
                    or target_norm <= np.finfo(float).eps
                    else abs(float(np.dot(centered_prediction, centered_target)))
                    / (prediction_norm * target_norm)
                )
                if not np.isfinite(training_r2) or not np.isfinite(
                    prediction_correlation
                ):
                    numeric_failures += 1
                    continue
                phase_operator = "+" if phase.sign == 1 else "-"
                radial_operator = "+" if radial_sign == 1 else "-"
                phase_expression = (
                    f"{_format_scale(phase.scale)}"
                    f"(({names[phase.left_index]}) {phase_operator} "
                    f"({names[phase.right_index]}))"
                )
                expression = (
                    f"sqrt(({names[amplitude.left_index]}) ** 2 + "
                    f"({names[amplitude.right_index]}) ** 2 {radial_operator} "
                    f"2 * ({names[amplitude.left_index]}) * "
                    f"({names[amplitude.right_index]}) * cos({phase_expression}))"
                )
                value_states.append(
                    _ValueState(
                        expression=expression,
                        amplitude=amplitude,
                        phase=phase,
                        radial_sign=radial_sign,
                        coefficients=tuple(float(value) for value in rounded),
                        prediction=prediction,
                        training_r2=training_r2,
                        prediction_correlation=float(
                            np.clip(prediction_correlation, 0.0, 1.0)
                        ),
                    )
                )

    value_states.sort(
        key=lambda item: (
            -item.training_r2,
            -item.prediction_correlation,
            item.expression,
        )
    )
    retained = value_states[:value_pool_size]
    coordinate_span = orthonormal_signature_span(
        tuple(np.asarray(entry.signature, dtype=float) for entry in direct)
    )
    scored: list[_ScoredState] = []
    for state in retained:
        try:
            cross_signature = 2.0 * sobolev_product_signature(
                state.amplitude.product_signature,
                state.phase.signature,
                sample_count=geometry_sample_count,
                dimension=dimension,
                lambda_value=lambda_value,
                lambda_gradient=lambda_gradient,
            )
            radicand_signature = (
                state.amplitude.left_square_signature
                + state.amplitude.right_square_signature
                + float(state.radial_sign) * cross_signature
            )
            radial_signature = sobolev_positive_sqrt_signature(
                radicand_signature,
                sample_count=geometry_sample_count,
                dimension=dimension,
                minimum_value=minimum_radicand,
                lambda_value=lambda_value,
                lambda_gradient=lambda_gradient,
            )
            square_span = orthonormal_signature_span(
                (
                    state.amplitude.left_square_signature,
                    state.amplitude.right_square_signature,
                )
            )
            candidate_gain = _signature_gain(radial_signature, coordinate_span)
            angular_gain = _signature_gain(cross_signature, square_span)
        except ValueError:
            numeric_failures += 1
            candidate_gain = 0.0
            angular_gain = 0.0
        joint_score = state.prediction_correlation * candidate_gain * angular_gain
        scored.append(
            _ScoredState(
                value_state=state,
                candidate_sobolev_gain=candidate_gain,
                angular_cross_gain=angular_gain,
                joint_score=joint_score,
                selection_lanes=(),
            )
        )

    value_lane = sorted(
        scored,
        key=lambda item: (
            -item.value_state.training_r2,
            -item.value_state.prediction_correlation,
            item.value_state.expression,
        ),
    )[:shortlist_size]
    sobolev_lane = sorted(
        scored,
        key=lambda item: (
            -item.joint_score,
            -item.candidate_sobolev_gain,
            -item.angular_cross_gain,
            -item.value_state.training_r2,
            item.value_state.expression,
        ),
    )[:shortlist_size]
    lanes: dict[int, list[str]] = {}
    for item in value_lane:
        lanes.setdefault(id(item), []).append("value_cosine_law_radial")
    for item in sobolev_lane:
        lanes.setdefault(id(item), []).append("sobolev_cosine_law_radial")
    selected: list[_ScoredState] = []
    seen_ids: set[int] = set()
    for item in (*value_lane, *sobolev_lane):
        if id(item) in seen_ids:
            continue
        seen_ids.add(id(item))
        selected.append(replace(item, selection_lanes=tuple(lanes[id(item)])))

    proposals = [
        CosineLawRadialProposal(
            expression=item.value_state.expression,
            left_amplitude_expression=names[item.value_state.amplitude.left_index],
            right_amplitude_expression=names[item.value_state.amplitude.right_index],
            left_phase_expression=names[item.value_state.phase.left_index],
            right_phase_expression=names[item.value_state.phase.right_index],
            phase_sign=item.value_state.phase.sign,
            radial_sign=item.value_state.radial_sign,
            phase_scale=item.value_state.phase.scale,
            fitted_coefficients=item.value_state.coefficients,
            training_r2=item.value_state.training_r2,
            prediction_correlation=item.value_state.prediction_correlation,
            candidate_sobolev_gain=item.candidate_sobolev_gain,
            angular_cross_gain=item.angular_cross_gain,
            joint_score=item.joint_score,
            selection_lanes=item.selection_lanes,
        )
        for item in selected
    ]
    proposals.sort(
        key=lambda item: (
            -item.training_r2,
            -item.joint_score,
            -item.candidate_sobolev_gain,
            item.expression,
        )
    )
    unique_proposals: list[CosineLawRadialProposal] = []
    seen_expressions: set[str] = set()
    for proposal in proposals:
        if proposal.expression in seen_expressions:
            continue
        seen_expressions.add(proposal.expression)
        unique_proposals.append(proposal)
        if len(unique_proposals) >= proposal_limit:
            break

    return CosineLawRadialResult(
        proposals=tuple(unique_proposals),
        amplitude_pairs=len(amplitude_pairs),
        phase_atoms=len(phase_atoms),
        candidates_screened=candidates_screened,
        value_pool=len(retained),
        sobolev_candidates=len(scored),
        radicand_rejections=radicand_rejections,
        numeric_failures=numeric_failures,
    )
