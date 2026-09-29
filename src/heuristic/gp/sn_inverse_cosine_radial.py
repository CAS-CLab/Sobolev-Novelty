"""Pair-first inverse cosine-law radial construction in Sobolev geometry.

The export-side plug-in constructs compact expressions of the form

``A(x) / sqrt(a**2 + b**2 + sign_r * 2*a*b*cos(scale*phase))``.

The amplitude library contains a constant, direct coordinates, reciprocal
coordinates, and ordered coordinate ratios.  A phase may be one coordinate or
the sum/difference of two coordinates.  Complete amplitude/radial/phase tuples
are value-screened before the bounded Sobolev lane is formed.  Candidate
derivatives follow exact product, quotient, cosine-chain, and positive-root
rules; target derivatives and benchmark metadata are never used.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import combinations
from typing import Sequence

import numpy as np

from .sn_archive_interactions import (
    sobolev_product_signature,
    sobolev_unary_signature,
)
from .sn_basis_archive import BasisArchiveEntry
from .sn_population_coverage import orthonormal_signature_span
from .sn_radical_composition import sobolev_positive_sqrt_signature
from .sn_shared_denominator import _signature_gain, sobolev_quotient_signature


@dataclass(frozen=True)
class InverseCosineRadialProposal:
    """One ordinary-GP expression retained by a value or Sobolev lane."""

    expression: str
    amplitude_expression: str
    amplitude_numerator_axis: int | None
    amplitude_denominator_axis: int | None
    radial_left_axis: int
    radial_right_axis: int
    phase_left_axis: int
    phase_right_axis: int | None
    phase_sign: int
    radial_sign: int
    phase_scale: float
    fitted_coefficients: tuple[float, float]
    training_r2: float
    prediction_correlation: float
    candidate_sobolev_gain: float
    angular_cross_gain: float
    reciprocal_coupling_gain: float
    joint_score: float
    selection_lanes: tuple[str, ...]


@dataclass(frozen=True)
class InverseCosineRadialResult:
    """Compact audit result for one complete inverse-radial enumeration."""

    proposals: tuple[InverseCosineRadialProposal, ...]
    amplitude_atoms: int
    amplitude_denominator_rejections: int
    radial_pairs: int
    phase_atoms: int
    candidates_screened: int
    radicand_rejections: int
    value_pool: int
    sobolev_candidates: int
    numeric_failures: int


@dataclass(frozen=True)
class _Amplitude:
    expression: str
    numerator_axis: int | None
    denominator_axis: int | None
    values: np.ndarray
    signature: np.ndarray


@dataclass(frozen=True)
class _RadialPair:
    left_axis: int
    right_axis: int
    left_square_values: np.ndarray
    right_square_values: np.ndarray
    product_values: np.ndarray
    left_square_signature: np.ndarray
    right_square_signature: np.ndarray
    product_signature: np.ndarray


@dataclass(frozen=True)
class _PhaseAtom:
    left_axis: int
    right_axis: int | None
    sign: int
    scale: float
    expression: str
    cosine_values: np.ndarray
    cosine_signature: np.ndarray


@dataclass(frozen=True)
class _ValueState:
    expression: str
    amplitude: _Amplitude
    radial: _RadialPair
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
    reciprocal_coupling_gain: float
    joint_score: float
    selection_lanes: tuple[str, ...]


def _constant_signature(
    sample_count: int,
    dimension: int,
    lambda_value: float,
) -> np.ndarray:
    output = np.zeros(sample_count * (dimension + 1), dtype=float)
    output[:sample_count] = np.sqrt(lambda_value / sample_count)
    return output


def _scale_text(scale: float) -> str:
    if np.isclose(scale, 0.5, rtol=0.0, atol=1e-15):
        return "(1 / 2) * "
    if np.isclose(scale, 1.0, rtol=0.0, atol=1e-15):
        return ""
    if np.isclose(scale, 2.0, rtol=0.0, atol=1e-15):
        return "2 * "
    return f"({format(float(scale), '.17g')}) * "


def _amplitude_atoms(
    direct: Sequence[BasisArchiveEntry],
    names: Sequence[str],
    *,
    value_count: int,
    sample_count: int,
    dimension: int,
    protected_epsilon: float,
    lambda_value: float,
    lambda_gradient: float,
) -> tuple[tuple[_Amplitude, ...], int, int]:
    constant = _constant_signature(sample_count, dimension, lambda_value)
    amplitudes = [
        _Amplitude(
            expression="1",
            numerator_axis=None,
            denominator_axis=None,
            values=np.ones(value_count, dtype=float),
            signature=constant,
        )
    ]
    for axis, entry in enumerate(direct):
        amplitudes.append(
            _Amplitude(
                expression=names[axis],
                numerator_axis=axis,
                denominator_axis=None,
                values=np.asarray(entry.values, dtype=float),
                signature=np.asarray(entry.signature, dtype=float),
            )
        )

    denominator_rejections = 0
    numeric_failures = 0
    for denominator_axis, denominator in enumerate(direct):
        denominator_values = np.asarray(denominator.values, dtype=float)
        if np.any(np.abs(denominator_values) <= protected_epsilon):
            denominator_rejections += dimension
            continue
        numerator_options: tuple[tuple[int | None, np.ndarray, np.ndarray], ...] = (
            (None, np.ones(value_count, dtype=float), constant),
            *tuple(
                (
                    numerator_axis,
                    np.asarray(direct[numerator_axis].values, dtype=float),
                    np.asarray(direct[numerator_axis].signature, dtype=float),
                )
                for numerator_axis in range(dimension)
                if numerator_axis != denominator_axis
            ),
        )
        for numerator_axis, numerator_values, numerator_signature in numerator_options:
            try:
                signature = sobolev_quotient_signature(
                    numerator_signature,
                    denominator.signature,
                    sample_count=sample_count,
                    dimension=dimension,
                    protected_epsilon=protected_epsilon,
                    lambda_value=lambda_value,
                    lambda_gradient=lambda_gradient,
                )
            except ValueError:
                numeric_failures += 1
                continue
            with np.errstate(all="ignore"):
                values = numerator_values / denominator_values
            if not np.all(np.isfinite(values)) or not np.all(np.isfinite(signature)):
                numeric_failures += 1
                continue
            numerator_text = "1" if numerator_axis is None else names[numerator_axis]
            amplitudes.append(
                _Amplitude(
                    expression=f"({numerator_text}) / ({names[denominator_axis]})",
                    numerator_axis=numerator_axis,
                    denominator_axis=denominator_axis,
                    values=values,
                    signature=signature,
                )
            )
    amplitudes.sort(
        key=lambda item: (
            -1 if item.numerator_axis is None else item.numerator_axis,
            -1 if item.denominator_axis is None else item.denominator_axis,
            item.expression,
        )
    )
    return tuple(amplitudes), denominator_rejections, numeric_failures


def lift_direct_inverse_cosine_radials(
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
    protected_epsilon: float = 1e-6,
    lambda_value: float = 1.0,
    lambda_gradient: float = 1.0,
) -> InverseCosineRadialResult:
    """Enumerate complete amplitude/inverse-radial tuples before screening."""

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
        or protected_epsilon <= 0.0
        or lambda_value <= 0.0
        or (dimension and lambda_gradient <= 0.0)
    ):
        raise ValueError("inverse cosine-radial inputs are invalid")
    if any(
        np.asarray(entry.values).shape != y.shape
        or np.asarray(entry.signature).shape != (signature_size,)
        or not np.all(np.isfinite(entry.values))
        or not np.all(np.isfinite(entry.signature))
        for entry in direct
    ):
        raise ValueError("inverse cosine-radial entries are not aligned")

    amplitudes, amplitude_rejections, numeric_failures = _amplitude_atoms(
        direct,
        names,
        value_count=y.size,
        sample_count=geometry_sample_count,
        dimension=dimension,
        protected_epsilon=protected_epsilon,
        lambda_value=lambda_value,
        lambda_gradient=lambda_gradient,
    )

    radial_pairs: list[_RadialPair] = []
    for left_axis, right_axis in combinations(range(dimension), 2):
        left = direct[left_axis]
        right = direct[right_axis]
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
        radial_pairs.append(
            _RadialPair(
                left_axis=left_axis,
                right_axis=right_axis,
                left_square_values=np.square(np.asarray(left.values, dtype=float)),
                right_square_values=np.square(np.asarray(right.values, dtype=float)),
                product_values=(
                    np.asarray(left.values, dtype=float)
                    * np.asarray(right.values, dtype=float)
                ),
                left_square_signature=left_square_signature,
                right_square_signature=right_square_signature,
                product_signature=product_signature,
            )
        )

    phases: list[_PhaseAtom] = []
    for scale in scales:
        prefix = _scale_text(scale)
        for left_axis, left in enumerate(direct):
            phase_values = scale * np.asarray(left.values, dtype=float)
            phase_signature = scale * np.asarray(left.signature, dtype=float)
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
            phases.append(
                _PhaseAtom(
                    left_axis=left_axis,
                    right_axis=None,
                    sign=0,
                    scale=scale,
                    expression=f"{prefix}({names[left_axis]})",
                    cosine_values=np.cos(phase_values),
                    cosine_signature=cosine_signature,
                )
            )
        for left_axis, right_axis in combinations(range(dimension), 2):
            left = direct[left_axis]
            right = direct[right_axis]
            for sign in phase_operations:
                phase_values = scale * (
                    np.asarray(left.values, dtype=float)
                    + float(sign) * np.asarray(right.values, dtype=float)
                )
                phase_signature = scale * (
                    np.asarray(left.signature, dtype=float)
                    + float(sign) * np.asarray(right.signature, dtype=float)
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
                operator = "+" if sign == 1 else "-"
                phases.append(
                    _PhaseAtom(
                        left_axis=left_axis,
                        right_axis=right_axis,
                        sign=sign,
                        scale=scale,
                        expression=(
                            f"{prefix}(({names[left_axis]}) {operator} "
                            f"({names[right_axis]}))"
                        ),
                        cosine_values=np.cos(phase_values),
                        cosine_signature=cosine_signature,
                    )
                )

    target_variance = float(np.var(y))
    centered_target = y - float(np.mean(y))
    target_norm = float(np.linalg.norm(centered_target))
    states: list[_ValueState] = []
    candidates_screened = 0
    radicand_rejections = 0
    for amplitude in amplitudes:
        for radial in radial_pairs:
            for phase in phases:
                cross_values = 2.0 * radial.product_values * phase.cosine_values
                for radial_sign in radial_operations:
                    candidates_screened += 1
                    radicand = (
                        radial.left_square_values
                        + radial.right_square_values
                        + float(radial_sign) * cross_values
                    )
                    if np.any(radicand <= minimum_radicand):
                        radicand_rejections += 1
                        continue
                    with np.errstate(all="ignore"):
                        basis = amplitude.values / np.sqrt(radicand)
                    if not np.all(np.isfinite(basis)):
                        numeric_failures += 1
                        continue
                    design = np.column_stack((np.ones(y.size, dtype=float), basis))
                    try:
                        coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
                    except np.linalg.LinAlgError:
                        numeric_failures += 1
                        continue
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
                    radial_operator = "+" if radial_sign == 1 else "-"
                    radial_text = (
                        f"({names[radial.left_axis]}) ** 2 + "
                        f"({names[radial.right_axis]}) ** 2 {radial_operator} "
                        f"2 * ({names[radial.left_axis]}) * "
                        f"({names[radial.right_axis]}) * cos({phase.expression})"
                    )
                    expression = f"({amplitude.expression}) / sqrt({radial_text})"
                    states.append(
                        _ValueState(
                            expression=expression,
                            amplitude=amplitude,
                            radial=radial,
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

    states.sort(
        key=lambda item: (
            -item.training_r2,
            -item.prediction_correlation,
            item.expression,
        )
    )
    retained = states[:value_pool_size]
    coordinate_span = orthonormal_signature_span(
        tuple(np.asarray(entry.signature, dtype=float) for entry in direct)
    )
    scored: list[_ScoredState] = []
    for state in retained:
        try:
            cross_signature = 2.0 * sobolev_product_signature(
                state.radial.product_signature,
                state.phase.cosine_signature,
                sample_count=geometry_sample_count,
                dimension=dimension,
                lambda_value=lambda_value,
                lambda_gradient=lambda_gradient,
            )
            radicand_signature = (
                state.radial.left_square_signature
                + state.radial.right_square_signature
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
            candidate_signature = sobolev_quotient_signature(
                state.amplitude.signature,
                radial_signature,
                sample_count=geometry_sample_count,
                dimension=dimension,
                protected_epsilon=protected_epsilon,
                lambda_value=lambda_value,
                lambda_gradient=lambda_gradient,
            )
            square_span = orthonormal_signature_span(
                (
                    state.radial.left_square_signature,
                    state.radial.right_square_signature,
                )
            )
            component_span = orthonormal_signature_span(
                (state.amplitude.signature, radial_signature)
            )
            candidate_gain = _signature_gain(candidate_signature, coordinate_span)
            angular_gain = _signature_gain(cross_signature, square_span)
            reciprocal_gain = _signature_gain(candidate_signature, component_span)
        except ValueError:
            numeric_failures += 1
            candidate_gain = 0.0
            angular_gain = 0.0
            reciprocal_gain = 0.0
        joint_score = (
            state.prediction_correlation
            * candidate_gain
            * angular_gain
            * reciprocal_gain
        )
        scored.append(
            _ScoredState(
                value_state=state,
                candidate_sobolev_gain=candidate_gain,
                angular_cross_gain=angular_gain,
                reciprocal_coupling_gain=reciprocal_gain,
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
            -item.reciprocal_coupling_gain,
            -item.angular_cross_gain,
            -item.value_state.training_r2,
            item.value_state.expression,
        ),
    )[:shortlist_size]
    lanes: dict[int, list[str]] = {}
    for item in value_lane:
        lanes.setdefault(id(item), []).append("value_inverse_cosine_radial")
    for item in sobolev_lane:
        lanes.setdefault(id(item), []).append("sobolev_inverse_cosine_radial")
    selected: list[_ScoredState] = []
    seen_ids: set[int] = set()
    for item in (*value_lane, *sobolev_lane):
        if id(item) in seen_ids:
            continue
        seen_ids.add(id(item))
        selected.append(replace(item, selection_lanes=tuple(lanes[id(item)])))

    proposals = [
        InverseCosineRadialProposal(
            expression=item.value_state.expression,
            amplitude_expression=item.value_state.amplitude.expression,
            amplitude_numerator_axis=item.value_state.amplitude.numerator_axis,
            amplitude_denominator_axis=item.value_state.amplitude.denominator_axis,
            radial_left_axis=item.value_state.radial.left_axis,
            radial_right_axis=item.value_state.radial.right_axis,
            phase_left_axis=item.value_state.phase.left_axis,
            phase_right_axis=item.value_state.phase.right_axis,
            phase_sign=item.value_state.phase.sign,
            radial_sign=item.value_state.radial_sign,
            phase_scale=item.value_state.phase.scale,
            fitted_coefficients=item.value_state.coefficients,
            training_r2=item.value_state.training_r2,
            prediction_correlation=item.value_state.prediction_correlation,
            candidate_sobolev_gain=item.candidate_sobolev_gain,
            angular_cross_gain=item.angular_cross_gain,
            reciprocal_coupling_gain=item.reciprocal_coupling_gain,
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
    unique: list[InverseCosineRadialProposal] = []
    seen_expressions: set[str] = set()
    for proposal in proposals:
        if proposal.expression in seen_expressions:
            continue
        seen_expressions.add(proposal.expression)
        unique.append(proposal)
        if len(unique) >= proposal_limit:
            break

    return InverseCosineRadialResult(
        proposals=tuple(unique),
        amplitude_atoms=len(amplitudes),
        amplitude_denominator_rejections=amplitude_rejections,
        radial_pairs=len(radial_pairs),
        phase_atoms=len(phases),
        candidates_screened=candidates_screened,
        radicand_rejections=radicand_rejections,
        value_pool=len(retained),
        sobolev_candidates=len(scored),
        numeric_failures=numeric_failures,
    )
