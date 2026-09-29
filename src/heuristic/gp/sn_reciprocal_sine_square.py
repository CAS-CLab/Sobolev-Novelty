"""Pair-first reciprocal-sine-square pursuit in Sobolev geometry.

The export-side plug-in constructs compact expressions of the form

``((product(x_i) / x_j) / sin(scale * x_k) ** 2) ** 2``.

Every subset-product amplitude, denominator coordinate, phase coordinate and
frozen scale is combined before value screening.  Only the bounded value pool
is evaluated in exact value-and-gradient geometry.  Target derivatives and
benchmark metadata are never used.
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
from .sn_shared_denominator import _signature_gain, sobolev_quotient_signature


@dataclass(frozen=True)
class ReciprocalSineSquareProposal:
    """One ordinary-GP expression retained by a value or Sobolev lane."""

    expression: str
    amplitude_expression: str
    amplitude_exponents: tuple[int, ...]
    phase_expression: str
    phase_axis: int
    phase_scale: float
    fitted_coefficients: tuple[float, float]
    training_r2: float
    prediction_correlation: float
    candidate_sobolev_gain: float
    nonlinear_coupling_gain: float
    joint_score: float
    selection_lanes: tuple[str, ...]


@dataclass(frozen=True)
class ReciprocalSineSquareResult:
    """Compact audit result for one complete reciprocal-sine enumeration."""

    proposals: tuple[ReciprocalSineSquareProposal, ...]
    amplitude_ratios: int
    amplitude_denominator_rejections: int
    phase_atoms: int
    candidates_screened: int
    sine_denominator_rejections: int
    value_pool: int
    sobolev_candidates: int
    numeric_failures: int


@dataclass(frozen=True)
class _Amplitude:
    exponents: tuple[int, ...]
    expression: str
    values: np.ndarray
    signature: np.ndarray


@dataclass(frozen=True)
class _PhaseAtom:
    axis: int
    scale: float
    expression: str
    sine_square_values: np.ndarray
    sine_square_signature: np.ndarray
    reaches_protected_region: bool


@dataclass(frozen=True)
class _ValueState:
    expression: str
    amplitude: _Amplitude
    phase: _PhaseAtom
    coefficients: tuple[float, float]
    prediction: np.ndarray
    training_r2: float
    prediction_correlation: float


@dataclass(frozen=True)
class _ScoredState:
    value_state: _ValueState
    candidate_sobolev_gain: float
    nonlinear_coupling_gain: float
    joint_score: float
    selection_lanes: tuple[str, ...]


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
    raise ValueError(f"reciprocal-sine scale {scale!r} is outside the GP library")


def _product_signature(
    entries: Sequence[BasisArchiveEntry],
    indices: Sequence[int],
    *,
    sample_count: int,
    dimension: int,
    lambda_value: float,
    lambda_gradient: float,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(entries[indices[0]].values, dtype=float).copy()
    signature = np.asarray(entries[indices[0]].signature, dtype=float).copy()
    for index in indices[1:]:
        entry = entries[index]
        values = values * np.asarray(entry.values, dtype=float)
        signature = sobolev_product_signature(
            signature,
            entry.signature,
            sample_count=sample_count,
            dimension=dimension,
            lambda_value=lambda_value,
            lambda_gradient=lambda_gradient,
        )
    return values, signature


def _subset_amplitude_ratios(
    direct: Sequence[BasisArchiveEntry],
    variable_names: Sequence[str],
    *,
    max_numerator_degree: int,
    sample_count: int,
    dimension: int,
    protected_epsilon: float,
    lambda_value: float,
    lambda_gradient: float,
) -> tuple[tuple[_Amplitude, ...], int, int]:
    amplitudes: list[_Amplitude] = []
    denominator_rejections = 0
    numeric_failures = 0
    maximum_degree = min(max_numerator_degree, dimension - 1)
    for degree in range(1, maximum_degree + 1):
        for numerator_indices in combinations(range(dimension), degree):
            try:
                numerator_values, numerator_signature = _product_signature(
                    direct,
                    numerator_indices,
                    sample_count=sample_count,
                    dimension=dimension,
                    lambda_value=lambda_value,
                    lambda_gradient=lambda_gradient,
                )
            except ValueError:
                numeric_failures += dimension - degree
                continue
            numerator_set = set(numerator_indices)
            numerator_text = " * ".join(
                variable_names[index] for index in numerator_indices
            )
            for denominator_index in range(dimension):
                if denominator_index in numerator_set:
                    continue
                denominator = direct[denominator_index]
                denominator_values = np.asarray(denominator.values, dtype=float)
                if np.any(np.abs(denominator_values) <= protected_epsilon):
                    denominator_rejections += 1
                    continue
                with np.errstate(all="ignore"):
                    values = numerator_values / denominator_values
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
                if not np.all(np.isfinite(values)) or not np.all(
                    np.isfinite(signature)
                ):
                    numeric_failures += 1
                    continue
                exponents = tuple(
                    (1 if axis in numerator_set else 0)
                    - (1 if axis == denominator_index else 0)
                    for axis in range(dimension)
                )
                amplitudes.append(
                    _Amplitude(
                        exponents=exponents,
                        expression=(
                            f"({numerator_text}) / ({variable_names[denominator_index]})"
                        ),
                        values=values,
                        signature=signature,
                    )
                )
    amplitudes.sort(key=lambda item: item.exponents)
    return tuple(amplitudes), denominator_rejections, numeric_failures


def lift_direct_reciprocal_sine_squares(
    *,
    direct_entries: Sequence[BasisArchiveEntry],
    target: Sequence[float],
    variable_names: Sequence[str],
    geometry_sample_count: int,
    phase_scales: Sequence[float],
    max_numerator_degree: int,
    value_pool_size: int,
    shortlist_size: int,
    proposal_limit: int,
    protected_epsilon: float = 1e-6,
    lambda_value: float = 1.0,
    lambda_gradient: float = 1.0,
) -> ReciprocalSineSquareResult:
    """Enumerate complete product/phase candidates before Sobolev screening."""

    direct = tuple(direct_entries)
    names = tuple(str(value) for value in variable_names)
    y = np.asarray(target, dtype=float).reshape(-1)
    dimension = len(names)
    signature_size = geometry_sample_count * (dimension + 1)
    scales = tuple(dict.fromkeys(float(value) for value in phase_scales))
    frozen_scales = (0.5, 1.0, 2.0, float(np.pi), float(2.0 * np.pi))
    if (
        len(direct) != dimension
        or dimension < 2
        or y.size < 2
        or not np.all(np.isfinite(y))
        or float(np.var(y)) <= np.finfo(float).eps
        or geometry_sample_count < 1
        or not scales
        or any(
            not np.isfinite(value)
            or not any(
                np.isclose(value, fixed, rtol=0.0, atol=1e-15)
                for fixed in frozen_scales
            )
            for value in scales
        )
        or max_numerator_degree < 1
        or value_pool_size < shortlist_size
        or shortlist_size < 1
        or proposal_limit < 1
        or protected_epsilon <= 0.0
        or lambda_value <= 0.0
        or (dimension and lambda_gradient <= 0.0)
    ):
        raise ValueError("reciprocal-sine-square inputs are invalid")
    if any(
        np.asarray(entry.values).shape != y.shape
        or np.asarray(entry.signature).shape != (signature_size,)
        or not np.all(np.isfinite(entry.values))
        or not np.all(np.isfinite(entry.signature))
        for entry in direct
    ):
        raise ValueError("reciprocal-sine-square entries are not aligned")

    amplitudes, amplitude_rejections, numeric_failures = _subset_amplitude_ratios(
        direct,
        names,
        max_numerator_degree=max_numerator_degree,
        sample_count=geometry_sample_count,
        dimension=dimension,
        protected_epsilon=protected_epsilon,
        lambda_value=lambda_value,
        lambda_gradient=lambda_gradient,
    )
    phases: list[_PhaseAtom] = []
    for axis, entry in enumerate(direct):
        source_values = np.asarray(entry.values, dtype=float)
        source_signature = np.asarray(entry.signature, dtype=float)
        for scale in scales:
            scale_text = _scale_expression(scale)
            phase_expression = (
                names[axis]
                if scale_text == "1"
                else f"({scale_text}) * ({names[axis]})"
            )
            scaled_signature = scale * source_signature
            try:
                sine_signature = sobolev_unary_signature(
                    scaled_signature,
                    transform="sin",
                    sample_count=geometry_sample_count,
                    dimension=dimension,
                    lambda_value=lambda_value,
                    lambda_gradient=lambda_gradient,
                )
                square_signature = sobolev_product_signature(
                    sine_signature,
                    sine_signature,
                    sample_count=geometry_sample_count,
                    dimension=dimension,
                    lambda_value=lambda_value,
                    lambda_gradient=lambda_gradient,
                )
            except ValueError:
                numeric_failures += 1
                continue
            square_values = np.square(np.sin(scale * source_values))
            phases.append(
                _PhaseAtom(
                    axis=axis,
                    scale=scale,
                    expression=phase_expression,
                    sine_square_values=square_values,
                    sine_square_signature=square_signature,
                    reaches_protected_region=bool(
                        np.any(np.abs(square_values) <= protected_epsilon)
                    ),
                )
            )

    target_variance = float(np.var(y))
    centered_target = y - float(np.mean(y))
    target_norm = float(np.linalg.norm(centered_target))
    value_states: list[_ValueState] = []
    candidates_screened = 0
    sine_rejections = 0
    for amplitude in amplitudes:
        for phase in phases:
            candidates_screened += 1
            if phase.reaches_protected_region:
                sine_rejections += 1
                continue
            with np.errstate(all="ignore"):
                ratio_values = amplitude.values / phase.sine_square_values
                basis = np.square(ratio_values)
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
            if not np.isfinite(training_r2) or not np.isfinite(prediction_correlation):
                numeric_failures += 1
                continue
            expression = (
                f"(({amplitude.expression}) / " f"(sin({phase.expression}) ** 2)) ** 2"
            )
            value_states.append(
                _ValueState(
                    expression=expression,
                    amplitude=amplitude,
                    phase=phase,
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
            ratio_signature = sobolev_quotient_signature(
                state.amplitude.signature,
                state.phase.sine_square_signature,
                sample_count=geometry_sample_count,
                dimension=dimension,
                protected_epsilon=protected_epsilon,
                lambda_value=lambda_value,
                lambda_gradient=lambda_gradient,
            )
            candidate_signature = sobolev_product_signature(
                ratio_signature,
                ratio_signature,
                sample_count=geometry_sample_count,
                dimension=dimension,
                lambda_value=lambda_value,
                lambda_gradient=lambda_gradient,
            )
            component_span = orthonormal_signature_span(
                (state.amplitude.signature, state.phase.sine_square_signature)
            )
            candidate_gain = _signature_gain(candidate_signature, coordinate_span)
            coupling_gain = _signature_gain(candidate_signature, component_span)
        except ValueError:
            numeric_failures += 1
            candidate_gain = 0.0
            coupling_gain = 0.0
        joint_score = state.prediction_correlation * candidate_gain * coupling_gain
        scored.append(
            _ScoredState(
                value_state=state,
                candidate_sobolev_gain=candidate_gain,
                nonlinear_coupling_gain=coupling_gain,
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
            -item.nonlinear_coupling_gain,
            -item.value_state.training_r2,
            item.value_state.expression,
        ),
    )[:shortlist_size]
    lanes: dict[int, list[str]] = {}
    for item in value_lane:
        lanes.setdefault(id(item), []).append("value_reciprocal_sine_square")
    for item in sobolev_lane:
        lanes.setdefault(id(item), []).append("sobolev_reciprocal_sine_square")
    selected: list[_ScoredState] = []
    seen_ids: set[int] = set()
    for item in (*value_lane, *sobolev_lane):
        if id(item) in seen_ids:
            continue
        seen_ids.add(id(item))
        selected.append(replace(item, selection_lanes=tuple(lanes[id(item)])))

    proposals = [
        ReciprocalSineSquareProposal(
            expression=item.value_state.expression,
            amplitude_expression=item.value_state.amplitude.expression,
            amplitude_exponents=item.value_state.amplitude.exponents,
            phase_expression=item.value_state.phase.expression,
            phase_axis=item.value_state.phase.axis,
            phase_scale=item.value_state.phase.scale,
            fitted_coefficients=item.value_state.coefficients,
            training_r2=item.value_state.training_r2,
            prediction_correlation=item.value_state.prediction_correlation,
            candidate_sobolev_gain=item.candidate_sobolev_gain,
            nonlinear_coupling_gain=item.nonlinear_coupling_gain,
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
    unique_proposals: list[ReciprocalSineSquareProposal] = []
    seen_expressions: set[str] = set()
    for proposal in proposals:
        if proposal.expression in seen_expressions:
            continue
        seen_expressions.add(proposal.expression)
        unique_proposals.append(proposal)
        if len(unique_proposals) >= proposal_limit:
            break

    return ReciprocalSineSquareResult(
        proposals=tuple(unique_proposals),
        amplitude_ratios=len(amplitudes),
        amplitude_denominator_rejections=amplitude_rejections,
        phase_atoms=len(phases),
        candidates_screened=candidates_screened,
        sine_denominator_rejections=sine_rejections,
        value_pool=len(retained),
        sobolev_candidates=len(scored),
        numeric_failures=numeric_failures,
    )
