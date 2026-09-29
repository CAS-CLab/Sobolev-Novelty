"""Complete shared-ratio trigonometric-polynomial proposals for Stage CE.

The plug-in forms every bounded seven-coordinate tuple before screening.  A
search-value lane keeps numerically strong complete candidates, while a lazy
Sobolev lane re-ranks only the fixed value pool by exact product, quotient and
trigonometric signatures.  Target derivatives and benchmark metadata are not
used.
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
from .sn_direct_composition import sobolev_division_signature
from .sn_population_coverage import orthonormal_signature_span
from .sn_shared_denominator import _signature_gain


@dataclass(frozen=True)
class SharedRatioTrigPolynomialProposal:
    """One ordinary-GP expression retained by a value or Sobolev lane."""

    expression: str
    ratio_axes: tuple[int, int]
    phase_axis: int
    amplitude_numerator_axes: tuple[int, int]
    amplitude_denominator_axes: tuple[int, int]
    reciprocal_sign: int
    phase_sign: int
    phase_scale: float
    fitted_coefficients: tuple[float, float]
    training_r2: float
    prediction_correlation: float
    candidate_sobolev_gain: float
    amplitude_square_sobolev_gain: float
    bracket_sobolev_gain: float
    bracket_term_novelties: tuple[float, float, float]
    coupling_gain: float
    joint_score: float
    selection_lanes: tuple[str, ...]


@dataclass(frozen=True)
class SharedRatioTrigPolynomialResult:
    """Compact audit result for the complete Stage-CE enumeration."""

    proposals: tuple[SharedRatioTrigPolynomialProposal, ...]
    candidates_screened: int
    value_pool: int
    sobolev_candidates: int
    domain_rejections: int
    numeric_failures: int
    dimension_fallback: bool


@dataclass(frozen=True)
class _ValueState:
    expression: str
    ratio_axes: tuple[int, int]
    phase_axis: int
    amplitude_numerator_axes: tuple[int, int]
    amplitude_denominator_axes: tuple[int, int]
    reciprocal_sign: int
    phase_sign: int
    phase_scale: float
    fitted_coefficients: tuple[float, float]
    training_r2: float
    prediction_correlation: float


@dataclass(frozen=True)
class _ScoredState:
    value_state: _ValueState
    candidate_sobolev_gain: float
    amplitude_square_sobolev_gain: float
    bracket_sobolev_gain: float
    bracket_term_novelties: tuple[float, float, float]
    coupling_gain: float
    joint_score: float
    selection_lanes: tuple[str, ...]


def _protected_divide(
    numerator: np.ndarray,
    denominator: np.ndarray,
    epsilon: float,
) -> np.ndarray:
    nonzero = denominator != 0.0
    return numerator / (denominator + epsilon * (~nonzero))


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
    raise ValueError(f"unsupported Stage-CE phase scale {scale!r}")


def _phase_expression(name: str, scale: float) -> str:
    scale_text = _scale_text(scale)
    argument = name if scale_text == "1" else f"({scale_text}) * {name}"
    return f"sin({argument}) ** 2"


def _candidate_expression(
    names: Sequence[str],
    *,
    ratio_axes: tuple[int, int],
    phase_axis: int,
    amplitude_numerator_axes: tuple[int, int],
    amplitude_denominator_axes: tuple[int, int],
    reciprocal_sign: int,
    phase_sign: int,
    phase_scale: float,
) -> str:
    numerator_axis, denominator_axis = ratio_axes
    ratio = f"({names[numerator_axis]} / {names[denominator_axis]})"
    reciprocal = f"({names[denominator_axis]} / {names[numerator_axis]})"
    amplitude_numerator = " * ".join(
        names[index] for index in amplitude_numerator_axes
    )
    amplitude_denominator = " * ".join(
        names[index] for index in amplitude_denominator_axes
    )
    amplitude = (
        f"(({amplitude_numerator}) * {ratio}) / ({amplitude_denominator})"
    )
    phase = _phase_expression(names[phase_axis], phase_scale)
    reciprocal_operator = "+" if reciprocal_sign > 0 else "-"
    phase_operator = "+" if phase_sign > 0 else "-"
    bracket = (
        f"{ratio} {reciprocal_operator} {reciprocal} "
        f"{phase_operator} {phase}"
    )
    return f"({amplitude}) ** 2 * ({bracket})"


def _fit_basis(
    basis: np.ndarray,
    target: np.ndarray,
    centered_target: np.ndarray,
    target_variance: float,
    target_norm: float,
) -> tuple[tuple[float, float], float, float] | None:
    centered_basis = basis - float(np.mean(basis))
    energy = float(np.dot(centered_basis, centered_basis))
    if energy <= np.finfo(float).eps:
        return None
    slope = float(np.dot(centered_basis, centered_target) / energy)
    intercept = float(np.mean(target) - slope * np.mean(basis))
    rounded = np.round((intercept, slope), 6)
    prediction = rounded[0] + rounded[1] * basis
    training_r2 = float(
        1.0 - np.mean(np.square(prediction - target)) / target_variance
    )
    centered_prediction = prediction - float(np.mean(prediction))
    prediction_norm = float(np.linalg.norm(centered_prediction))
    correlation = (
        0.0
        if prediction_norm <= np.finfo(float).eps
        or target_norm <= np.finfo(float).eps
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


def _linear_signature(
    components: Sequence[tuple[float, np.ndarray]],
) -> np.ndarray:
    output = sum(
        (float(coefficient) * np.asarray(signature, dtype=float)
         for coefficient, signature in components),
        np.zeros_like(np.asarray(components[0][1], dtype=float)),
    )
    if not np.all(np.isfinite(output)):
        raise ValueError("linear Sobolev signature is non-finite")
    return output


def _geometry_scores(
    state: _ValueState,
    direct: Sequence[BasisArchiveEntry],
    *,
    geometry_sample_count: int,
    dimension: int,
    coordinate_span: np.ndarray | None,
    protected_epsilon: float,
    lambda_value: float,
    lambda_gradient: float,
) -> tuple[float, float, float, tuple[float, float, float], float]:
    ratio_numerator, ratio_denominator = state.ratio_axes
    amplitude_left, amplitude_right = state.amplitude_numerator_axes
    denominator_left, denominator_right = state.amplitude_denominator_axes
    keyword = {
        "sample_count": geometry_sample_count,
        "dimension": dimension,
        "lambda_value": lambda_value,
        "lambda_gradient": lambda_gradient,
    }
    division_keyword = {**keyword, "protected_epsilon": protected_epsilon}
    ratio = sobolev_division_signature(
        direct[ratio_numerator].signature,
        direct[ratio_denominator].signature,
        **division_keyword,
    )
    reciprocal = sobolev_division_signature(
        direct[ratio_denominator].signature,
        direct[ratio_numerator].signature,
        **division_keyword,
    )
    amplitude_numerator = sobolev_product_signature(
        direct[amplitude_left].signature,
        direct[amplitude_right].signature,
        **keyword,
    )
    amplitude_numerator = sobolev_product_signature(
        amplitude_numerator, ratio, **keyword
    )
    amplitude_denominator = sobolev_product_signature(
        direct[denominator_left].signature,
        direct[denominator_right].signature,
        **keyword,
    )
    amplitude = sobolev_division_signature(
        amplitude_numerator, amplitude_denominator, **division_keyword
    )
    amplitude_square = sobolev_product_signature(amplitude, amplitude, **keyword)
    phase_source = state.phase_scale * np.asarray(
        direct[state.phase_axis].signature, dtype=float
    )
    phase = sobolev_unary_signature(
        phase_source, transform="sin", **keyword
    )
    phase_square = sobolev_product_signature(phase, phase, **keyword)
    terms = (
        ratio,
        state.reciprocal_sign * reciprocal,
        state.phase_sign * phase_square,
    )
    bracket = _linear_signature(tuple((1.0, value) for value in terms))
    candidate = sobolev_product_signature(amplitude_square, bracket, **keyword)
    term_novelties: list[float] = []
    for index, term in enumerate(terms):
        other_span = orthonormal_signature_span(
            tuple(value for other, value in enumerate(terms) if other != index)
        )
        term_novelties.append(_signature_gain(term, other_span))
    coupling_span = orthonormal_signature_span((amplitude_square, bracket))
    return (
        _signature_gain(candidate, coordinate_span),
        _signature_gain(amplitude_square, coordinate_span),
        _signature_gain(bracket, coordinate_span),
        tuple(term_novelties),
        _signature_gain(candidate, coupling_span),
    )


def lift_direct_shared_ratio_trig_polynomials(
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
) -> SharedRatioTrigPolynomialResult:
    """Form all 25,200 seven-coordinate candidates before dual-lane screening."""

    direct = tuple(direct_entries)
    names = tuple(str(value) for value in variable_names)
    y = np.asarray(target, dtype=float).reshape(-1)
    scales = tuple(dict.fromkeys(float(value) for value in phase_scales))
    dimension = len(names)
    if dimension != 7:
        return SharedRatioTrigPolynomialResult((), 0, 0, 0, 0, 0, True)
    signature_size = geometry_sample_count * (dimension + 1)
    if (
        len(direct) != dimension
        or y.size < 2
        or not np.all(np.isfinite(y))
        or float(np.var(y)) <= np.finfo(float).eps
        or geometry_sample_count < 1
        or not scales
        or value_pool_size < shortlist_size
        or shortlist_size < 1
        or proposal_limit < 1
        or protected_epsilon <= 0.0
        or lambda_value <= 0.0
        or lambda_gradient <= 0.0
    ):
        raise ValueError("shared-ratio trigonometric inputs are invalid")
    if any(
        np.asarray(entry.values).shape != y.shape
        or np.asarray(entry.signature).shape != (signature_size,)
        or not np.all(np.isfinite(entry.values))
        or not np.all(np.isfinite(entry.signature))
        for entry in direct
    ):
        raise ValueError("shared-ratio direct entries are not aligned")
    for scale in scales:
        _scale_text(scale)

    X = np.column_stack(
        tuple(np.asarray(entry.values, dtype=float) for entry in direct)
    )
    centered_target = y - float(np.mean(y))
    target_variance = float(np.var(y))
    target_norm = float(np.linalg.norm(centered_target))
    states: list[_ValueState] = []
    candidates_screened = 0
    domain_rejections = 0
    numeric_failures = 0
    axes = tuple(range(dimension))
    for ratio_numerator in axes:
        for ratio_denominator in axes:
            if ratio_numerator == ratio_denominator:
                continue
            ratio = _protected_divide(
                X[:, ratio_numerator], X[:, ratio_denominator], protected_epsilon
            )
            reciprocal = _protected_divide(
                X[:, ratio_denominator], X[:, ratio_numerator], protected_epsilon
            )
            remaining_after_ratio = tuple(
                axis
                for axis in axes
                if axis not in (ratio_numerator, ratio_denominator)
            )
            for phase_axis in remaining_after_ratio:
                amplitude_axes = tuple(
                    axis for axis in remaining_after_ratio if axis != phase_axis
                )
                for numerator_axes in combinations(amplitude_axes, 2):
                    denominator_axes = tuple(
                        axis for axis in amplitude_axes if axis not in numerator_axes
                    )
                    numerator = (
                        X[:, numerator_axes[0]]
                        * X[:, numerator_axes[1]]
                        * ratio
                    )
                    denominator = (
                        X[:, denominator_axes[0]] * X[:, denominator_axes[1]]
                    )
                    amplitude = _protected_divide(
                        numerator, denominator, protected_epsilon
                    )
                    amplitude_square = amplitude * amplitude
                    for reciprocal_sign in (-1, 1):
                        for phase_sign in (-1, 1):
                            for phase_scale in scales:
                                candidates_screened += 1
                                with np.errstate(all="ignore"):
                                    phase_square = np.square(
                                        np.sin(phase_scale * X[:, phase_axis])
                                    )
                                    bracket = (
                                        ratio
                                        + reciprocal_sign * reciprocal
                                        + phase_sign * phase_square
                                    )
                                    basis = amplitude_square * bracket
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
                                    domain_rejections += 1
                                    continue
                                coefficients, training_r2, correlation = fit
                                ratio_axes = (ratio_numerator, ratio_denominator)
                                expression = _candidate_expression(
                                    names,
                                    ratio_axes=ratio_axes,
                                    phase_axis=phase_axis,
                                    amplitude_numerator_axes=numerator_axes,
                                    amplitude_denominator_axes=denominator_axes,
                                    reciprocal_sign=reciprocal_sign,
                                    phase_sign=phase_sign,
                                    phase_scale=phase_scale,
                                )
                                states.append(
                                    _ValueState(
                                        expression=expression,
                                        ratio_axes=ratio_axes,
                                        phase_axis=phase_axis,
                                        amplitude_numerator_axes=numerator_axes,
                                        amplitude_denominator_axes=denominator_axes,
                                        reciprocal_sign=reciprocal_sign,
                                        phase_sign=phase_sign,
                                        phase_scale=phase_scale,
                                        fitted_coefficients=coefficients,
                                        training_r2=training_r2,
                                        prediction_correlation=correlation,
                                    )
                                )
    states.sort(key=_value_key)
    del states[value_pool_size:]

    coordinate_span = orthonormal_signature_span(
        tuple(np.asarray(entry.signature, dtype=float) for entry in direct)
    )
    scored: list[_ScoredState] = []
    for state in states:
        try:
            (
                candidate_gain,
                amplitude_gain,
                bracket_gain,
                term_novelties,
                coupling_gain,
            ) = _geometry_scores(
                state,
                direct,
                geometry_sample_count=geometry_sample_count,
                dimension=dimension,
                coordinate_span=coordinate_span,
                protected_epsilon=protected_epsilon,
                lambda_value=lambda_value,
                lambda_gradient=lambda_gradient,
            )
        except ValueError:
            numeric_failures += 1
            candidate_gain = 0.0
            amplitude_gain = 0.0
            bracket_gain = 0.0
            term_novelties = (0.0, 0.0, 0.0)
            coupling_gain = 0.0
        joint_score = float(
            np.mean(
                (
                    candidate_gain,
                    amplitude_gain,
                    bracket_gain,
                    min(term_novelties),
                    coupling_gain,
                )
            )
        )
        scored.append(
            _ScoredState(
                value_state=state,
                candidate_sobolev_gain=candidate_gain,
                amplitude_square_sobolev_gain=amplitude_gain,
                bracket_sobolev_gain=bracket_gain,
                bracket_term_novelties=term_novelties,
                coupling_gain=coupling_gain,
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
            -item.amplitude_square_sobolev_gain,
            -item.bracket_sobolev_gain,
            -min(item.bracket_term_novelties),
            -item.coupling_gain,
            -item.value_state.training_r2,
            item.value_state.expression,
        ),
    )[:shortlist_size]
    lane_names: dict[int, list[str]] = {}
    for item in value_lane:
        lane_names.setdefault(id(item), []).append("value_shared_ratio_trig")
    for item in sobolev_lane:
        lane_names.setdefault(id(item), []).append("sobolev_shared_ratio_trig")
    selected: list[_ScoredState] = []
    seen_items: set[int] = set()
    for item in (*value_lane, *sobolev_lane):
        if id(item) in seen_items:
            continue
        seen_items.add(id(item))
        selected.append(
            replace(item, selection_lanes=tuple(lane_names[id(item)]))
        )

    proposals = [
        SharedRatioTrigPolynomialProposal(
            expression=item.value_state.expression,
            ratio_axes=item.value_state.ratio_axes,
            phase_axis=item.value_state.phase_axis,
            amplitude_numerator_axes=(
                item.value_state.amplitude_numerator_axes
            ),
            amplitude_denominator_axes=(
                item.value_state.amplitude_denominator_axes
            ),
            reciprocal_sign=item.value_state.reciprocal_sign,
            phase_sign=item.value_state.phase_sign,
            phase_scale=item.value_state.phase_scale,
            fitted_coefficients=item.value_state.fitted_coefficients,
            training_r2=item.value_state.training_r2,
            prediction_correlation=item.value_state.prediction_correlation,
            candidate_sobolev_gain=item.candidate_sobolev_gain,
            amplitude_square_sobolev_gain=(
                item.amplitude_square_sobolev_gain
            ),
            bracket_sobolev_gain=item.bracket_sobolev_gain,
            bracket_term_novelties=item.bracket_term_novelties,
            coupling_gain=item.coupling_gain,
            joint_score=item.joint_score,
            selection_lanes=item.selection_lanes,
        )
        for item in selected
    ]
    proposals.sort(
        key=lambda item: (
            -item.training_r2,
            -item.prediction_correlation,
            item.expression,
        )
    )
    unique: list[SharedRatioTrigPolynomialProposal] = []
    seen_expressions: set[str] = set()
    for proposal in proposals:
        if proposal.expression in seen_expressions:
            continue
        seen_expressions.add(proposal.expression)
        unique.append(proposal)
        if len(unique) >= proposal_limit:
            break
    return SharedRatioTrigPolynomialResult(
        proposals=tuple(unique),
        candidates_screened=candidates_screened,
        value_pool=len(states),
        sobolev_candidates=len(scored),
        domain_rejections=domain_rejections,
        numeric_failures=numeric_failures,
        dimension_fallback=False,
    )
