"""Complete coupled trigonometric-rational proposals for Stage CG."""

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
from .sn_direct_composition import sobolev_division_signature
from .sn_population_coverage import orthonormal_signature_span
from .sn_shared_denominator import _signature_gain


@dataclass(frozen=True)
class CoupledTrigRationalProposal:
    expression: str
    amplitude_kind: str
    amplitude_axis: int | None
    modulated_axis: int
    phase_axes: tuple[int, int]
    phase_sign: int
    numerator_sign: int
    numerator_scale: float
    denominator_sign: int
    denominator_scale: float
    phase_scale: float
    fitted_coefficients: tuple[float, float]
    training_r2: float
    prediction_correlation: float
    candidate_sobolev_gain: float
    quotient_coupling_gain: float
    numerator_factor_gain: float
    denominator_factor_gain: float
    numerator_given_denominator_novelty: float
    denominator_given_numerator_novelty: float
    joint_score: float
    selection_lanes: tuple[str, ...]


@dataclass(frozen=True)
class CoupledTrigRationalResult:
    proposals: tuple[CoupledTrigRationalProposal, ...]
    candidates_screened: int
    value_pool: int
    sobolev_candidates: int
    denominator_rejections: int
    numeric_failures: int
    dimension_fallback: bool


@dataclass(frozen=True)
class _ValueState:
    expression: str
    amplitude_kind: str
    amplitude_axis: int | None
    modulated_axis: int
    phase_axes: tuple[int, int]
    phase_sign: int
    numerator_sign: int
    numerator_scale: float
    denominator_sign: int
    denominator_scale: float
    phase_scale: float
    fitted_coefficients: tuple[float, float]
    training_r2: float
    prediction_correlation: float
    selection_lanes: tuple[str, ...] = ()
    candidate_sobolev_gain: float = 0.0
    quotient_coupling_gain: float = 0.0
    numerator_factor_gain: float = 0.0
    denominator_factor_gain: float = 0.0
    numerator_given_denominator_novelty: float = 0.0
    denominator_given_numerator_novelty: float = 0.0
    joint_score: float = 0.0


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
    raise ValueError(f"unsupported Stage-CG scale {scale!r}")


def _scaled_term(scale: float, term: str) -> str:
    text = _scale_text(scale)
    return term if text == "1" else f"({text}) * ({term})"


def _signed_factor(sign: int, scale: float, term: str) -> str:
    operator = "+" if sign > 0 else "-"
    return f"1 {operator} {_scaled_term(scale, term)}"


def _expression(
    names: Sequence[str],
    *,
    amplitude_kind: str,
    amplitude_axis: int | None,
    modulated_axis: int,
    phase_axes: tuple[int, int],
    phase_sign: int,
    numerator_sign: int,
    numerator_scale: float,
    denominator_sign: int,
    denominator_scale: float,
    phase_scale: float,
) -> str:
    if amplitude_kind == "constant":
        amplitude = "1"
    elif amplitude_kind == "direct":
        amplitude = names[int(amplitude_axis)]
    elif amplitude_kind == "reciprocal":
        amplitude = f"1 / {names[int(amplitude_axis)]}"
    else:
        raise ValueError(f"unknown amplitude kind {amplitude_kind!r}")
    phase_operator = "+" if phase_sign > 0 else "-"
    phase_core = (
        f"{names[phase_axes[0]]} {phase_operator} {names[phase_axes[1]]}"
    )
    phase_argument = _scaled_term(phase_scale, phase_core)
    numerator = _signed_factor(
        numerator_sign,
        numerator_scale,
        f"{names[modulated_axis]} ** 2",
    )
    denominator_term = (
        f"{names[modulated_axis]} * cos({phase_argument})"
    )
    denominator = _signed_factor(
        denominator_sign, denominator_scale, denominator_term
    )
    if amplitude_kind == "constant":
        return f"({numerator}) / ({denominator})"
    return f"({amplitude}) * ({numerator}) / ({denominator})"


def _fit_basis(
    basis: np.ndarray,
    target: np.ndarray,
    centered_target: np.ndarray,
    target_variance: float,
    target_norm: float,
) -> tuple[tuple[float, float], float, float] | None:
    centered = basis - float(np.mean(basis))
    energy = float(np.dot(centered, centered))
    if energy <= np.finfo(float).eps:
        return None
    slope = float(np.dot(centered, centered_target) / energy)
    intercept = float(np.mean(target) - slope * np.mean(basis))
    rounded = np.round((intercept, slope), 6)
    prediction = rounded[0] + rounded[1] * basis
    r2 = float(
        1.0 - np.mean(np.square(prediction - target)) / target_variance
    )
    centered_prediction = prediction - float(np.mean(prediction))
    prediction_norm = float(np.linalg.norm(centered_prediction))
    correlation = (
        0.0
        if prediction_norm <= np.finfo(float).eps
        else abs(float(np.dot(centered_prediction, centered_target)))
        / (prediction_norm * target_norm)
    )
    if not np.isfinite(r2) or not np.isfinite(correlation):
        return None
    return (
        (float(rounded[0]), float(rounded[1])),
        r2,
        float(np.clip(correlation, 0.0, 1.0)),
    )


def _value_key(state: _ValueState) -> tuple[float, float, str]:
    return (
        -state.training_r2,
        -state.prediction_correlation,
        state.expression,
    )


def lift_direct_coupled_trig_rationals(
    *,
    direct_entries: Sequence[BasisArchiveEntry],
    target: Sequence[float],
    variable_names: Sequence[str],
    geometry_sample_count: int,
    numerator_scales: Sequence[float],
    denominator_scales: Sequence[float],
    phase_scales: Sequence[float],
    value_pool_size: int,
    shortlist_size: int,
    proposal_limit: int,
    protected_epsilon: float = 1e-6,
    lambda_value: float = 1.0,
    lambda_gradient: float = 1.0,
) -> CoupledTrigRationalResult:
    """Form all 38,880 four-coordinate tuples before dual-lane screening."""

    direct = tuple(direct_entries)
    names = tuple(str(value) for value in variable_names)
    y = np.asarray(target, dtype=float).reshape(-1)
    numerator_scale_values = tuple(
        dict.fromkeys(float(value) for value in numerator_scales)
    )
    denominator_scale_values = tuple(
        dict.fromkeys(float(value) for value in denominator_scales)
    )
    phase_scale_values = tuple(
        dict.fromkeys(float(value) for value in phase_scales)
    )
    dimension = len(names)
    if dimension != 4:
        return CoupledTrigRationalResult((), 0, 0, 0, 0, 0, True)
    signature_size = geometry_sample_count * (dimension + 1)
    if (
        len(direct) != dimension
        or y.size < 2
        or not np.all(np.isfinite(y))
        or float(np.var(y)) <= np.finfo(float).eps
        or geometry_sample_count < 1
        or not numerator_scale_values
        or not denominator_scale_values
        or not phase_scale_values
        or value_pool_size < shortlist_size
        or shortlist_size < 1
        or proposal_limit < 1
        or protected_epsilon <= 0.0
        or lambda_value <= 0.0
        or lambda_gradient <= 0.0
    ):
        raise ValueError("coupled trigonometric-rational inputs are invalid")
    for scale in (
        *numerator_scale_values,
        *denominator_scale_values,
        *phase_scale_values,
    ):
        _scale_text(scale)
    if any(
        np.asarray(entry.values).shape != y.shape
        or np.asarray(entry.signature).shape != (signature_size,)
        or not np.all(np.isfinite(entry.values))
        or not np.all(np.isfinite(entry.signature))
        for entry in direct
    ):
        raise ValueError("coupled trigonometric-rational entries are not aligned")

    X = np.column_stack(
        tuple(np.asarray(entry.values, dtype=float) for entry in direct)
    )
    value_factor = float(np.sqrt(lambda_value / geometry_sample_count))
    geometry_X = np.column_stack(
        tuple(
            np.asarray(entry.signature, dtype=float)[:geometry_sample_count]
            / value_factor
            for entry in direct
        )
    )
    ones = np.ones(len(y), dtype=float)
    geometry_ones = np.ones(geometry_sample_count, dtype=float)
    amplitudes: list[tuple[str, int | None, np.ndarray]] = [
        ("constant", None, ones)
    ]
    for axis in range(dimension):
        amplitudes.append(("direct", axis, X[:, axis]))
    for axis in range(dimension):
        denominator = X[:, axis]
        amplitudes.append(
            (
                "reciprocal",
                axis,
                ones
                / (
                    denominator
                    + protected_epsilon * (denominator == 0.0)
                ),
            )
        )

    centered_target = y - float(np.mean(y))
    target_variance = float(np.var(y))
    target_norm = float(np.linalg.norm(centered_target))
    retained: list[_ValueState] = []
    candidates_screened = 0
    denominator_rejections = 0
    numeric_failures = 0
    axes = tuple(range(dimension))
    for amplitude_kind, amplitude_axis, amplitude in amplitudes:
        for modulated_axis in axes:
            other_axes = tuple(
                axis for axis in axes if axis != modulated_axis
            )
            for phase_axes in combinations(other_axes, 2):
                for phase_sign in (-1, 1):
                    phase_search_core = (
                        X[:, phase_axes[0]]
                        + phase_sign * X[:, phase_axes[1]]
                    )
                    phase_geometry_core = (
                        geometry_X[:, phase_axes[0]]
                        + phase_sign * geometry_X[:, phase_axes[1]]
                    )
                    for numerator_sign in (-1, 1):
                        for numerator_scale in numerator_scale_values:
                            numerator = (
                                ones
                                + numerator_sign
                                * numerator_scale
                                * np.square(X[:, modulated_axis])
                            )
                            for denominator_sign in (-1, 1):
                                for denominator_scale in (
                                    denominator_scale_values
                                ):
                                    for phase_scale in phase_scale_values:
                                        candidates_screened += 1
                                        with np.errstate(all="ignore"):
                                            cosine = np.cos(
                                                phase_scale * phase_search_core
                                            )
                                            geometry_cosine = np.cos(
                                                phase_scale
                                                * phase_geometry_core
                                            )
                                            denominator = (
                                                ones
                                                + denominator_sign
                                                * denominator_scale
                                                * X[:, modulated_axis]
                                                * cosine
                                            )
                                            geometry_denominator = (
                                                geometry_ones
                                                + denominator_sign
                                                * denominator_scale
                                                * geometry_X[
                                                    :, modulated_axis
                                                ]
                                                * geometry_cosine
                                            )
                                        if (
                                            np.any(
                                                np.abs(denominator)
                                                <= protected_epsilon
                                            )
                                            or np.any(
                                                np.abs(geometry_denominator)
                                                <= protected_epsilon
                                            )
                                        ):
                                            denominator_rejections += 1
                                            continue
                                        with np.errstate(all="ignore"):
                                            basis = (
                                                amplitude
                                                * numerator
                                                / denominator
                                            )
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
                                        coefficients, r2, correlation = fit
                                        retained.append(
                                            _ValueState(
                                                expression=_expression(
                                                    names,
                                                    amplitude_kind=(
                                                        amplitude_kind
                                                    ),
                                                    amplitude_axis=(
                                                        amplitude_axis
                                                    ),
                                                    modulated_axis=(
                                                        modulated_axis
                                                    ),
                                                    phase_axes=phase_axes,
                                                    phase_sign=phase_sign,
                                                    numerator_sign=(
                                                        numerator_sign
                                                    ),
                                                    numerator_scale=(
                                                        numerator_scale
                                                    ),
                                                    denominator_sign=(
                                                        denominator_sign
                                                    ),
                                                    denominator_scale=(
                                                        denominator_scale
                                                    ),
                                                    phase_scale=phase_scale,
                                                ),
                                                amplitude_kind=amplitude_kind,
                                                amplitude_axis=amplitude_axis,
                                                modulated_axis=modulated_axis,
                                                phase_axes=phase_axes,
                                                phase_sign=phase_sign,
                                                numerator_sign=(
                                                    numerator_sign
                                                ),
                                                numerator_scale=(
                                                    numerator_scale
                                                ),
                                                denominator_sign=(
                                                    denominator_sign
                                                ),
                                                denominator_scale=(
                                                    denominator_scale
                                                ),
                                                phase_scale=phase_scale,
                                                fitted_coefficients=coefficients,
                                                training_r2=r2,
                                                prediction_correlation=(
                                                    correlation
                                                ),
                                            )
                                        )
    retained.sort(key=_value_key)
    del retained[value_pool_size:]

    constant = _constant_signature(
        geometry_sample_count, dimension, lambda_value
    )
    coordinate_span = orthonormal_signature_span(
        tuple(np.asarray(entry.signature, dtype=float) for entry in direct)
    )
    keyword = {
        "sample_count": geometry_sample_count,
        "dimension": dimension,
        "lambda_value": lambda_value,
        "lambda_gradient": lambda_gradient,
    }
    division_keyword = {**keyword, "protected_epsilon": protected_epsilon}
    reciprocal_signatures = {
        axis: sobolev_division_signature(
            constant, direct[axis].signature, **division_keyword
        )
        for axis in axes
    }
    scored: list[_ValueState] = []
    for state in retained:
        try:
            if state.amplitude_kind == "constant":
                amplitude_signature = constant
            elif state.amplitude_kind == "direct":
                amplitude_signature = np.asarray(
                    direct[int(state.amplitude_axis)].signature, dtype=float
                )
            else:
                amplitude_signature = reciprocal_signatures[
                    int(state.amplitude_axis)
                ]
            modulated_signature = np.asarray(
                direct[state.modulated_axis].signature, dtype=float
            )
            square_signature = sobolev_product_signature(
                modulated_signature, modulated_signature, **keyword
            )
            numerator_factor = (
                constant
                + state.numerator_sign
                * state.numerator_scale
                * square_signature
            )
            phase_source = state.phase_scale * (
                np.asarray(
                    direct[state.phase_axes[0]].signature, dtype=float
                )
                + state.phase_sign
                * np.asarray(
                    direct[state.phase_axes[1]].signature, dtype=float
                )
            )
            cosine_signature = sobolev_unary_signature(
                phase_source, transform="cos", **keyword
            )
            denominator_interaction = sobolev_product_signature(
                modulated_signature, cosine_signature, **keyword
            )
            denominator_factor = (
                constant
                + state.denominator_sign
                * state.denominator_scale
                * denominator_interaction
            )
            numerator_total = sobolev_product_signature(
                amplitude_signature, numerator_factor, **keyword
            )
            candidate_raw = sobolev_division_signature(
                numerator_total, denominator_factor, **division_keyword
            )
            intercept, slope = state.fitted_coefficients
            candidate_signature = intercept * constant + slope * candidate_raw
            candidate_gain = _signature_gain(
                candidate_signature, coordinate_span
            )
            numerator_gain = _signature_gain(
                numerator_factor, coordinate_span
            )
            denominator_gain = _signature_gain(
                denominator_factor, coordinate_span
            )
            numerator_given_denominator = _signature_gain(
                numerator_factor,
                orthonormal_signature_span((denominator_factor,)),
            )
            denominator_given_numerator = _signature_gain(
                denominator_factor,
                orthonormal_signature_span((numerator_factor,)),
            )
            quotient_coupling = _signature_gain(
                candidate_raw,
                orthonormal_signature_span(
                    (numerator_total, denominator_factor)
                ),
            )
        except ValueError:
            numeric_failures += 1
            candidate_gain = 0.0
            numerator_gain = 0.0
            denominator_gain = 0.0
            numerator_given_denominator = 0.0
            denominator_given_numerator = 0.0
            quotient_coupling = 0.0
        joint_score = 0.5 * (
            candidate_gain
            + min(
                numerator_given_denominator,
                denominator_given_numerator,
            )
        )
        scored.append(
            replace(
                state,
                candidate_sobolev_gain=candidate_gain,
                quotient_coupling_gain=quotient_coupling,
                numerator_factor_gain=numerator_gain,
                denominator_factor_gain=denominator_gain,
                numerator_given_denominator_novelty=(
                    numerator_given_denominator
                ),
                denominator_given_numerator_novelty=(
                    denominator_given_numerator
                ),
                joint_score=joint_score,
            )
        )

    value_lane = sorted(scored, key=_value_key)[:shortlist_size]
    sobolev_lane = sorted(
        scored,
        key=lambda state: (
            -state.joint_score,
            -state.candidate_sobolev_gain,
            -min(
                state.numerator_given_denominator_novelty,
                state.denominator_given_numerator_novelty,
            ),
            -state.training_r2,
            state.expression,
        ),
    )[:shortlist_size]
    lane_names: dict[int, list[str]] = {}
    for state in value_lane:
        lane_names.setdefault(id(state), []).append(
            "value_coupled_trig_rational"
        )
    for state in sobolev_lane:
        lane_names.setdefault(id(state), []).append(
            "sobolev_coupled_trig_rational"
        )
    selected: list[_ValueState] = []
    seen_ids: set[int] = set()
    for state in (*value_lane, *sobolev_lane):
        if id(state) in seen_ids:
            continue
        seen_ids.add(id(state))
        selected.append(
            replace(state, selection_lanes=tuple(lane_names[id(state)]))
        )

    proposals = [
        CoupledTrigRationalProposal(
            expression=state.expression,
            amplitude_kind=state.amplitude_kind,
            amplitude_axis=state.amplitude_axis,
            modulated_axis=state.modulated_axis,
            phase_axes=state.phase_axes,
            phase_sign=state.phase_sign,
            numerator_sign=state.numerator_sign,
            numerator_scale=state.numerator_scale,
            denominator_sign=state.denominator_sign,
            denominator_scale=state.denominator_scale,
            phase_scale=state.phase_scale,
            fitted_coefficients=state.fitted_coefficients,
            training_r2=state.training_r2,
            prediction_correlation=state.prediction_correlation,
            candidate_sobolev_gain=state.candidate_sobolev_gain,
            quotient_coupling_gain=state.quotient_coupling_gain,
            numerator_factor_gain=state.numerator_factor_gain,
            denominator_factor_gain=state.denominator_factor_gain,
            numerator_given_denominator_novelty=(
                state.numerator_given_denominator_novelty
            ),
            denominator_given_numerator_novelty=(
                state.denominator_given_numerator_novelty
            ),
            joint_score=state.joint_score,
            selection_lanes=state.selection_lanes,
        )
        for state in selected
    ]
    proposals.sort(
        key=lambda proposal: (
            -proposal.training_r2,
            -proposal.prediction_correlation,
            proposal.expression,
        )
    )
    unique: list[CoupledTrigRationalProposal] = []
    seen_expressions: set[str] = set()
    for proposal in proposals:
        if proposal.expression in seen_expressions:
            continue
        seen_expressions.add(proposal.expression)
        unique.append(proposal)
        if len(unique) >= proposal_limit:
            break
    return CoupledTrigRationalResult(
        proposals=tuple(unique),
        candidates_screened=candidates_screened,
        value_pool=len(retained),
        sobolev_candidates=len(scored),
        denominator_rejections=denominator_rejections,
        numeric_failures=numeric_failures,
        dimension_fallback=False,
    )
