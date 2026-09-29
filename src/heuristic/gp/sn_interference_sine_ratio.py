"""Pair-first squared sine-ratio construction in Sobolev geometry.

The export-side plug-in constructs compact interference kernels

``A(x) * (sin(scale * numerator_phase) / sin(scale * x_j)) ** 2``.

Amplitude atoms, direct/product numerator phases, denominator coordinates and
frozen scales are combined before value screening.  Exact value-and-gradient
signatures are evaluated only for a bounded retained pool.  Target derivatives
and benchmark metadata are never used.
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
class InterferenceSineRatioProposal:
    """One ordinary-GP expression retained by a value or Sobolev lane."""

    expression: str
    amplitude_expression: str
    amplitude_numerator_axis: int | None
    amplitude_denominator_axis: int | None
    numerator_phase_axes: tuple[int, ...]
    denominator_phase_axis: int
    phase_scale: float
    fitted_coefficients: tuple[float, float]
    training_r2: float
    prediction_correlation: float
    candidate_sobolev_gain: float
    phase_complementarity: float
    ratio_coupling_gain: float
    joint_score: float
    selection_lanes: tuple[str, ...]


@dataclass(frozen=True)
class InterferenceSineRatioResult:
    """Compact audit result for one complete sine-ratio enumeration."""

    proposals: tuple[InterferenceSineRatioProposal, ...]
    amplitude_atoms: int
    amplitude_denominator_rejections: int
    numerator_phase_cores: int
    sine_ratio_pairs: int
    sine_denominator_rejections: int
    candidates_screened: int
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
class _PhaseCore:
    expression: str
    axes: tuple[int, ...]
    values: np.ndarray
    signature: np.ndarray


@dataclass(frozen=True)
class _RatioPair:
    numerator_core: _PhaseCore
    denominator_axis: int
    scale: float
    expression: str
    values: np.ndarray
    signature: np.ndarray
    numerator_sine_signature: np.ndarray
    denominator_sine_signature: np.ndarray
    phase_complementarity: float


@dataclass(frozen=True)
class _ValueState:
    expression: str
    amplitude: _Amplitude
    ratio_pair: _RatioPair
    coefficients: tuple[float, float]
    training_r2: float
    prediction_correlation: float


@dataclass(frozen=True)
class _ScoredState:
    value_state: _ValueState
    candidate_sobolev_gain: float
    ratio_coupling_gain: float
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
        return "(1 / 2)"
    if np.isclose(scale, 1.0, rtol=0.0, atol=1e-15):
        return "1"
    if np.isclose(scale, 2.0, rtol=0.0, atol=1e-15):
        return "2"
    if np.isclose(scale, np.pi, rtol=0.0, atol=1e-15):
        return "3.141592653589793"
    if np.isclose(scale, 2.0 * np.pi, rtol=0.0, atol=1e-15):
        return "(2 * 3.141592653589793)"
    raise ValueError(f"sine-ratio scale {scale!r} is outside the GP library")


def _scaled_expression(expression: str, scale: float) -> str:
    text = _scale_text(scale)
    return expression if text == "1" else f"({text}) * ({expression})"


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
    amplitudes.extend(
        _Amplitude(
            expression=names[axis],
            numerator_axis=axis,
            denominator_axis=None,
            values=np.asarray(entry.values, dtype=float),
            signature=np.asarray(entry.signature, dtype=float),
        )
        for axis, entry in enumerate(direct)
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


def _phase_cores(
    direct: Sequence[BasisArchiveEntry],
    names: Sequence[str],
    *,
    sample_count: int,
    dimension: int,
    lambda_value: float,
    lambda_gradient: float,
) -> tuple[tuple[_PhaseCore, ...], int]:
    cores = [
        _PhaseCore(
            expression=names[axis],
            axes=(axis,),
            values=np.asarray(entry.values, dtype=float),
            signature=np.asarray(entry.signature, dtype=float),
        )
        for axis, entry in enumerate(direct)
    ]
    failures = 0
    for left_axis, right_axis in combinations(range(dimension), 2):
        left = direct[left_axis]
        right = direct[right_axis]
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
        values = np.asarray(left.values, dtype=float) * np.asarray(
            right.values, dtype=float
        )
        if not np.all(np.isfinite(values)) or not np.all(np.isfinite(signature)):
            failures += 1
            continue
        cores.append(
            _PhaseCore(
                expression=f"({names[left_axis]}) * ({names[right_axis]})",
                axes=(left_axis, right_axis),
                values=values,
                signature=signature,
            )
        )
    return tuple(cores), failures


def _fit_basis(
    basis: np.ndarray,
    target: np.ndarray,
    centered_target: np.ndarray,
    target_variance: float,
    target_norm: float,
) -> tuple[tuple[float, float], float, float] | None:
    centered_basis = basis - float(np.mean(basis))
    basis_energy = float(np.dot(centered_basis, centered_basis))
    if basis_energy <= np.finfo(float).eps:
        return None
    slope = float(np.dot(centered_basis, centered_target) / basis_energy)
    intercept = float(np.mean(target) - slope * np.mean(basis))
    rounded = np.round((intercept, slope), 6)
    prediction = rounded[0] + rounded[1] * basis
    training_r2 = float(1.0 - np.mean(np.square(prediction - target)) / target_variance)
    centered_prediction = prediction - float(np.mean(prediction))
    prediction_norm = float(np.linalg.norm(centered_prediction))
    correlation = (
        0.0
        if prediction_norm <= np.finfo(float).eps or target_norm <= np.finfo(float).eps
        else abs(float(np.dot(centered_prediction, centered_target)))
        / (prediction_norm * target_norm)
    )
    if not np.isfinite(training_r2) or not np.isfinite(correlation):
        return None
    return (
        (float(rounded[0]), float(rounded[1])),
        training_r2,
        float(np.clip(correlation, 0.0, 1.0)),
    )


def _value_key(state: _ValueState) -> tuple[float, float, str]:
    return (-state.training_r2, -state.prediction_correlation, state.expression)


def lift_direct_interference_sine_ratios(
    *,
    direct_entries: Sequence[BasisArchiveEntry],
    target: Sequence[float],
    variable_names: Sequence[str],
    geometry_sample_count: int,
    phase_scales: Sequence[float],
    value_pool_size: int,
    shortlist_size: int,
    proposal_limit: int,
    protected_epsilon: float = 1e-6,
    lambda_value: float = 1.0,
    lambda_gradient: float = 1.0,
) -> InterferenceSineRatioResult:
    """Enumerate complete amplitude/sine-ratio candidates before screening."""

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
        or value_pool_size < shortlist_size
        or shortlist_size < 1
        or proposal_limit < 1
        or protected_epsilon <= 0.0
        or lambda_value <= 0.0
        or (dimension and lambda_gradient <= 0.0)
    ):
        raise ValueError("interference sine-ratio inputs are invalid")
    if any(
        np.asarray(entry.values).shape != y.shape
        or np.asarray(entry.signature).shape != (signature_size,)
        or not np.all(np.isfinite(entry.values))
        or not np.all(np.isfinite(entry.signature))
        for entry in direct
    ):
        raise ValueError("interference sine-ratio entries are not aligned")

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
    cores, core_failures = _phase_cores(
        direct,
        names,
        sample_count=geometry_sample_count,
        dimension=dimension,
        lambda_value=lambda_value,
        lambda_gradient=lambda_gradient,
    )
    numeric_failures += core_failures
    ratio_pairs: list[_RatioPair] = []
    sine_denominator_rejections = 0
    for denominator_axis, denominator in enumerate(direct):
        denominator_values = np.asarray(denominator.values, dtype=float)
        denominator_span = orthonormal_signature_span((denominator.signature,))
        for scale in scales:
            scaled_denominator_values = scale * denominator_values
            sine_denominator_values = np.sin(scaled_denominator_values)
            if np.any(np.abs(sine_denominator_values) <= protected_epsilon):
                sine_denominator_rejections += 1
                continue
            try:
                sine_denominator_signature = sobolev_unary_signature(
                    scale * np.asarray(denominator.signature, dtype=float),
                    transform="sin",
                    sample_count=geometry_sample_count,
                    dimension=dimension,
                    lambda_value=lambda_value,
                    lambda_gradient=lambda_gradient,
                )
            except ValueError:
                numeric_failures += 1
                continue
            denominator_expression = _scaled_expression(names[denominator_axis], scale)
            for core in cores:
                if core.axes == (denominator_axis,):
                    continue
                try:
                    sine_numerator_signature = sobolev_unary_signature(
                        scale * core.signature,
                        transform="sin",
                        sample_count=geometry_sample_count,
                        dimension=dimension,
                        lambda_value=lambda_value,
                        lambda_gradient=lambda_gradient,
                    )
                    ratio_signature = sobolev_quotient_signature(
                        sine_numerator_signature,
                        sine_denominator_signature,
                        sample_count=geometry_sample_count,
                        dimension=dimension,
                        protected_epsilon=protected_epsilon,
                        lambda_value=lambda_value,
                        lambda_gradient=lambda_gradient,
                    )
                    square_signature = sobolev_product_signature(
                        ratio_signature,
                        ratio_signature,
                        sample_count=geometry_sample_count,
                        dimension=dimension,
                        lambda_value=lambda_value,
                        lambda_gradient=lambda_gradient,
                    )
                except ValueError:
                    numeric_failures += 1
                    continue
                with np.errstate(all="ignore"):
                    values = np.square(
                        np.sin(scale * core.values) / sine_denominator_values
                    )
                if not np.all(np.isfinite(values)) or not np.all(
                    np.isfinite(square_signature)
                ):
                    numeric_failures += 1
                    continue
                numerator_expression = _scaled_expression(core.expression, scale)
                ratio_pairs.append(
                    _RatioPair(
                        numerator_core=core,
                        denominator_axis=denominator_axis,
                        scale=scale,
                        expression=(
                            f"(sin({numerator_expression}) / "
                            f"sin({denominator_expression})) ** 2"
                        ),
                        values=values,
                        signature=square_signature,
                        numerator_sine_signature=sine_numerator_signature,
                        denominator_sine_signature=sine_denominator_signature,
                        phase_complementarity=_signature_gain(
                            core.signature, denominator_span
                        ),
                    )
                )

    target_variance = float(np.var(y))
    centered_target = y - float(np.mean(y))
    target_norm = float(np.linalg.norm(centered_target))
    retained: list[_ValueState] = []
    candidates_screened = 0
    prune_batch_size = max(4096, 4 * value_pool_size)
    for amplitude in amplitudes:
        for ratio_pair in ratio_pairs:
            candidates_screened += 1
            with np.errstate(all="ignore"):
                basis = amplitude.values * ratio_pair.values
            if not np.all(np.isfinite(basis)):
                numeric_failures += 1
                continue
            fit = _fit_basis(
                basis,
                y,
                centered_target,
                target_variance,
                target_norm,
            )
            if fit is None:
                numeric_failures += 1
                continue
            coefficients, training_r2, correlation = fit
            retained.append(
                _ValueState(
                    expression=(
                        f"({amplitude.expression}) * ({ratio_pair.expression})"
                    ),
                    amplitude=amplitude,
                    ratio_pair=ratio_pair,
                    coefficients=coefficients,
                    training_r2=training_r2,
                    prediction_correlation=correlation,
                )
            )
            if len(retained) >= prune_batch_size:
                retained.sort(key=_value_key)
                del retained[value_pool_size:]

    retained.sort(key=_value_key)
    del retained[value_pool_size:]
    coordinate_span = orthonormal_signature_span(
        tuple(np.asarray(entry.signature, dtype=float) for entry in direct)
    )
    scored: list[_ScoredState] = []
    for state in retained:
        pair = state.ratio_pair
        try:
            candidate_signature = sobolev_product_signature(
                state.amplitude.signature,
                pair.signature,
                sample_count=geometry_sample_count,
                dimension=dimension,
                lambda_value=lambda_value,
                lambda_gradient=lambda_gradient,
            )
            component_span = orthonormal_signature_span(
                (
                    state.amplitude.signature,
                    pair.numerator_sine_signature,
                    pair.denominator_sine_signature,
                )
            )
            candidate_gain = _signature_gain(candidate_signature, coordinate_span)
            ratio_gain = _signature_gain(candidate_signature, component_span)
        except ValueError:
            numeric_failures += 1
            candidate_gain = 0.0
            ratio_gain = 0.0
        joint_score = (
            state.prediction_correlation
            * candidate_gain
            * pair.phase_complementarity
            * ratio_gain
        )
        scored.append(
            _ScoredState(
                value_state=state,
                candidate_sobolev_gain=candidate_gain,
                ratio_coupling_gain=ratio_gain,
                joint_score=joint_score,
                selection_lanes=(),
            )
        )

    value_lane = sorted(scored, key=lambda item: _value_key(item.value_state))[
        :shortlist_size
    ]
    sobolev_lane = sorted(
        scored,
        key=lambda item: (
            -item.joint_score,
            -item.candidate_sobolev_gain,
            -item.value_state.ratio_pair.phase_complementarity,
            -item.ratio_coupling_gain,
            -item.value_state.training_r2,
            item.value_state.expression,
        ),
    )[:shortlist_size]
    lanes: dict[int, list[str]] = {}
    for item in value_lane:
        lanes.setdefault(id(item), []).append("value_interference_sine_ratio")
    for item in sobolev_lane:
        lanes.setdefault(id(item), []).append("sobolev_interference_sine_ratio")
    selected: list[_ScoredState] = []
    seen_ids: set[int] = set()
    for item in (*value_lane, *sobolev_lane):
        if id(item) in seen_ids:
            continue
        seen_ids.add(id(item))
        selected.append(replace(item, selection_lanes=tuple(lanes[id(item)])))

    proposals = [
        InterferenceSineRatioProposal(
            expression=item.value_state.expression,
            amplitude_expression=item.value_state.amplitude.expression,
            amplitude_numerator_axis=item.value_state.amplitude.numerator_axis,
            amplitude_denominator_axis=item.value_state.amplitude.denominator_axis,
            numerator_phase_axes=item.value_state.ratio_pair.numerator_core.axes,
            denominator_phase_axis=item.value_state.ratio_pair.denominator_axis,
            phase_scale=item.value_state.ratio_pair.scale,
            fitted_coefficients=item.value_state.coefficients,
            training_r2=item.value_state.training_r2,
            prediction_correlation=item.value_state.prediction_correlation,
            candidate_sobolev_gain=item.candidate_sobolev_gain,
            phase_complementarity=(item.value_state.ratio_pair.phase_complementarity),
            ratio_coupling_gain=item.ratio_coupling_gain,
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
    unique: list[InterferenceSineRatioProposal] = []
    seen_expressions: set[str] = set()
    for proposal in proposals:
        if proposal.expression in seen_expressions:
            continue
        seen_expressions.add(proposal.expression)
        unique.append(proposal)
        if len(unique) >= proposal_limit:
            break

    return InterferenceSineRatioResult(
        proposals=tuple(unique),
        amplitude_atoms=len(amplitudes),
        amplitude_denominator_rejections=amplitude_rejections,
        numerator_phase_cores=len(cores),
        sine_ratio_pairs=len(ratio_pairs),
        sine_denominator_rejections=sine_denominator_rejections,
        candidates_screened=candidates_screened,
        value_pool=len(retained),
        sobolev_candidates=len(scored),
        numeric_failures=numeric_failures,
    )
