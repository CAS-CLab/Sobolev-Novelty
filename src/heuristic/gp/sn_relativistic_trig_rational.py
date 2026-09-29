"""Complete relativistic trigonometric-rational proposals for Stage CH.

The export-side plug-in enumerates complete expressions of the form

``A * sqrt(1 + s_r*c_r*(x_i/x_j)**2) /
       (1 + s_d*c_d*(x_i/x_j)*cos(c_p*x_k))``.

The value lane is deliberately independent of the Sobolev threshold.  Exact
value-and-gradient signatures are propagated only for the global value pool,
and a parallel Sobolev lane protects geometrically complementary proposals.
Target derivatives and benchmark formula metadata are never used.
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
from .sn_direct_composition import sobolev_division_signature
from .sn_population_coverage import orthonormal_signature_span
from .sn_radical_composition import sobolev_positive_sqrt_signature
from .sn_shared_denominator import _signature_gain


@dataclass(frozen=True)
class RelativisticTrigRationalProposal:
    expression: str
    amplitude_kind: str
    amplitude_axis: int | None
    ratio_axes: tuple[int, int]
    phase_axis: int
    radical_sign: int
    radical_scale: float
    denominator_sign: int
    denominator_scale: float
    phase_scale: float
    fitted_coefficients: tuple[float, float]
    training_r2: float
    prediction_correlation: float
    candidate_sobolev_gain: float
    quotient_coupling_gain: float
    radical_factor_gain: float
    denominator_factor_gain: float
    radical_given_denominator_novelty: float
    denominator_given_radical_novelty: float
    joint_score: float
    selection_lanes: tuple[str, ...]


@dataclass(frozen=True)
class RelativisticTrigRationalResult:
    proposals: tuple[RelativisticTrigRationalProposal, ...]
    candidates_screened: int
    defined_candidates: int
    value_pool: int
    sobolev_candidates: int
    ratio_rejections: int
    amplitude_rejections: int
    radical_rejections: int
    denominator_rejections: int
    numeric_failures: int
    dimension_fallback: bool


@dataclass(frozen=True)
class _ValueState:
    expression: str
    amplitude_kind: str
    amplitude_axis: int | None
    ratio_axes: tuple[int, int]
    phase_axis: int
    radical_sign: int
    radical_scale: float
    denominator_sign: int
    denominator_scale: float
    phase_scale: float
    fitted_coefficients: tuple[float, float]
    training_r2: float
    prediction_correlation: float
    selection_lanes: tuple[str, ...] = ()
    candidate_sobolev_gain: float = 0.0
    quotient_coupling_gain: float = 0.0
    radical_factor_gain: float = 0.0
    denominator_factor_gain: float = 0.0
    radical_given_denominator_novelty: float = 0.0
    denominator_given_radical_novelty: float = 0.0
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
    raise ValueError(f"unsupported Stage-CH scale {scale!r}")


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
    ratio_axes: tuple[int, int],
    phase_axis: int,
    radical_sign: int,
    radical_scale: float,
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
    ratio = f"{names[ratio_axes[0]]} / {names[ratio_axes[1]]}"
    radical = _signed_factor(
        radical_sign,
        radical_scale,
        f"({ratio}) ** 2",
    )
    phase = _scaled_term(phase_scale, names[phase_axis])
    denominator = _signed_factor(
        denominator_sign,
        denominator_scale,
        f"({ratio}) * cos({phase})",
    )
    root = f"sqrt({radical})"
    if amplitude_kind == "constant":
        return f"({root}) / ({denominator})"
    return f"({amplitude}) * ({root}) / ({denominator})"


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


def lift_direct_relativistic_trig_rationals(
    *,
    direct_entries: Sequence[BasisArchiveEntry],
    target: Sequence[float],
    variable_names: Sequence[str],
    geometry_sample_count: int,
    radical_scales: Sequence[float],
    denominator_scales: Sequence[float],
    phase_scales: Sequence[float],
    value_pool_size: int,
    shortlist_size: int,
    proposal_limit: int,
    protected_epsilon: float = 1e-6,
    lambda_value: float = 1.0,
    lambda_gradient: float = 1.0,
) -> RelativisticTrigRationalResult:
    """Form all 77,760 four-coordinate tuples before dual-lane screening."""

    direct = tuple(direct_entries)
    names = tuple(str(value) for value in variable_names)
    y = np.asarray(target, dtype=float).reshape(-1)
    radical_scale_values = tuple(
        dict.fromkeys(float(value) for value in radical_scales)
    )
    denominator_scale_values = tuple(
        dict.fromkeys(float(value) for value in denominator_scales)
    )
    phase_scale_values = tuple(
        dict.fromkeys(float(value) for value in phase_scales)
    )
    dimension = len(names)
    if dimension != 4:
        return RelativisticTrigRationalResult(
            (), 0, 0, 0, 0, 0, 0, 0, 0, 0, True
        )
    signature_size = geometry_sample_count * (dimension + 1)
    if (
        len(direct) != dimension
        or y.size < 2
        or not np.all(np.isfinite(y))
        or float(np.var(y)) <= np.finfo(float).eps
        or geometry_sample_count < 1
        or not radical_scale_values
        or not denominator_scale_values
        or not phase_scale_values
        or value_pool_size < shortlist_size
        or shortlist_size < 1
        or proposal_limit < 1
        or protected_epsilon <= 0.0
        or lambda_value <= 0.0
        or lambda_gradient <= 0.0
    ):
        raise ValueError("relativistic trigonometric-rational inputs are invalid")
    for scale in (
        *radical_scale_values,
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
        raise ValueError(
            "relativistic trigonometric-rational entries are not aligned"
        )

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
    amplitudes: list[tuple[str, int | None, np.ndarray, np.ndarray]] = [
        ("constant", None, ones, geometry_ones)
    ]
    for axis in range(dimension):
        amplitudes.append(
            ("direct", axis, X[:, axis], geometry_X[:, axis])
        )
    for axis in range(dimension):
        search_denominator = X[:, axis]
        geometry_denominator = geometry_X[:, axis]
        if (
            np.any(np.abs(search_denominator) <= protected_epsilon)
            or np.any(np.abs(geometry_denominator) <= protected_epsilon)
        ):
            amplitudes.append(("invalid_reciprocal", axis, ones, geometry_ones))
        else:
            amplitudes.append(
                (
                    "reciprocal",
                    axis,
                    ones / search_denominator,
                    geometry_ones / geometry_denominator,
                )
            )

    centered_target = y - float(np.mean(y))
    target_variance = float(np.var(y))
    target_norm = float(np.linalg.norm(centered_target))
    retained: list[_ValueState] = []
    candidates_screened = 0
    defined_candidates = 0
    ratio_rejections = 0
    amplitude_rejections = 0
    radical_rejections = 0
    denominator_rejections = 0
    numeric_failures = 0
    axes = tuple(range(dimension))
    combinations_per_amplitude = (
        dimension
        * 2
        * len(radical_scale_values)
        * 2
        * len(denominator_scale_values)
        * len(phase_scale_values)
    )
    combinations_per_radical = (
        2 * len(denominator_scale_values) * len(phase_scale_values)
    )
    for amplitude_kind, amplitude_axis, amplitude, _ in amplitudes:
        for numerator_axis in axes:
            for denominator_axis in axes:
                if numerator_axis == denominator_axis:
                    continue
                search_ratio_denominator = X[:, denominator_axis]
                geometry_ratio_denominator = geometry_X[:, denominator_axis]
                candidates_screened += combinations_per_amplitude
                if amplitude_kind == "invalid_reciprocal":
                    amplitude_rejections += combinations_per_amplitude
                    continue
                if (
                    np.any(
                        np.abs(search_ratio_denominator)
                        <= protected_epsilon
                    )
                    or np.any(
                        np.abs(geometry_ratio_denominator)
                        <= protected_epsilon
                    )
                ):
                    ratio_rejections += combinations_per_amplitude
                    continue
                search_ratio = X[:, numerator_axis] / search_ratio_denominator
                geometry_ratio = (
                    geometry_X[:, numerator_axis] / geometry_ratio_denominator
                )
                search_square = np.square(search_ratio)
                geometry_square = np.square(geometry_ratio)
                for phase_axis in axes:
                    phase_search = X[:, phase_axis]
                    phase_geometry = geometry_X[:, phase_axis]
                    for radical_sign in (-1, 1):
                        for radical_scale in radical_scale_values:
                            search_radicand = (
                                ones
                                + radical_sign
                                * radical_scale
                                * search_square
                            )
                            geometry_radicand = (
                                geometry_ones
                                + radical_sign
                                * radical_scale
                                * geometry_square
                            )
                            if (
                                np.any(search_radicand < 0.0)
                                or np.any(geometry_radicand < 0.0)
                            ):
                                radical_rejections += combinations_per_radical
                                continue
                            with np.errstate(all="ignore"):
                                radical = np.sqrt(search_radicand)
                            for denominator_sign in (-1, 1):
                                for denominator_scale in denominator_scale_values:
                                    for phase_scale in phase_scale_values:
                                        with np.errstate(all="ignore"):
                                            cosine = np.cos(
                                                phase_scale * phase_search
                                            )
                                            geometry_cosine = np.cos(
                                                phase_scale * phase_geometry
                                            )
                                            denominator = (
                                                ones
                                                + denominator_sign
                                                * denominator_scale
                                                * search_ratio
                                                * cosine
                                            )
                                            geometry_denominator = (
                                                geometry_ones
                                                + denominator_sign
                                                * denominator_scale
                                                * geometry_ratio
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
                                            basis = amplitude * radical / denominator
                                        if not np.all(np.isfinite(basis)):
                                            numeric_failures += 1
                                            continue
                                        defined_candidates += 1
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
                                                    amplitude_kind=amplitude_kind,
                                                    amplitude_axis=amplitude_axis,
                                                    ratio_axes=(
                                                        numerator_axis,
                                                        denominator_axis,
                                                    ),
                                                    phase_axis=phase_axis,
                                                    radical_sign=radical_sign,
                                                    radical_scale=radical_scale,
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
                                                ratio_axes=(
                                                    numerator_axis,
                                                    denominator_axis,
                                                ),
                                                phase_axis=phase_axis,
                                                radical_sign=radical_sign,
                                                radical_scale=radical_scale,
                                                denominator_sign=denominator_sign,
                                                denominator_scale=(
                                                    denominator_scale
                                                ),
                                                phase_scale=phase_scale,
                                                fitted_coefficients=coefficients,
                                                training_r2=r2,
                                                prediction_correlation=correlation,
                                            )
                                        )
    if candidates_screened != (
        len(amplitudes)
        * dimension
        * (dimension - 1)
        * combinations_per_amplitude
    ):
        raise RuntimeError("Stage-CH complete tuple count drifted")
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
    reciprocal_signatures: dict[int, np.ndarray] = {}
    for axis in axes:
        try:
            reciprocal_signatures[axis] = sobolev_division_signature(
                constant, direct[axis].signature, **division_keyword
            )
        except ValueError:
            pass
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
            ratio_signature = sobolev_division_signature(
                direct[state.ratio_axes[0]].signature,
                direct[state.ratio_axes[1]].signature,
                **division_keyword,
            )
            square_signature = sobolev_product_signature(
                ratio_signature, ratio_signature, **keyword
            )
            radical_source = (
                constant
                + state.radical_sign
                * state.radical_scale
                * square_signature
            )
            radical_factor = sobolev_positive_sqrt_signature(
                radical_source,
                minimum_value=np.finfo(float).eps,
                **keyword,
            )
            phase_source = (
                state.phase_scale
                * np.asarray(direct[state.phase_axis].signature, dtype=float)
            )
            cosine_signature = sobolev_unary_signature(
                phase_source, transform="cos", **keyword
            )
            denominator_interaction = sobolev_product_signature(
                ratio_signature, cosine_signature, **keyword
            )
            denominator_factor = (
                constant
                + state.denominator_sign
                * state.denominator_scale
                * denominator_interaction
            )
            numerator_total = sobolev_product_signature(
                amplitude_signature, radical_factor, **keyword
            )
            candidate_raw = sobolev_division_signature(
                numerator_total, denominator_factor, **division_keyword
            )
            intercept, slope = state.fitted_coefficients
            candidate_signature = intercept * constant + slope * candidate_raw
            candidate_gain = _signature_gain(
                candidate_signature, coordinate_span
            )
            radical_gain = _signature_gain(radical_factor, coordinate_span)
            denominator_gain = _signature_gain(
                denominator_factor, coordinate_span
            )
            radical_given_denominator = _signature_gain(
                radical_factor,
                orthonormal_signature_span((denominator_factor,)),
            )
            denominator_given_radical = _signature_gain(
                denominator_factor,
                orthonormal_signature_span((radical_factor,)),
            )
            quotient_coupling = _signature_gain(
                candidate_raw,
                orthonormal_signature_span(
                    (numerator_total, denominator_factor)
                ),
            )
        except (KeyError, ValueError):
            numeric_failures += 1
            candidate_gain = 0.0
            radical_gain = 0.0
            denominator_gain = 0.0
            radical_given_denominator = 0.0
            denominator_given_radical = 0.0
            quotient_coupling = 0.0
        joint_score = float(
            np.mean(
                (
                    candidate_gain,
                    quotient_coupling,
                    radical_gain,
                    denominator_gain,
                    min(
                        radical_given_denominator,
                        denominator_given_radical,
                    ),
                )
            )
        )
        scored.append(
            replace(
                state,
                candidate_sobolev_gain=candidate_gain,
                quotient_coupling_gain=quotient_coupling,
                radical_factor_gain=radical_gain,
                denominator_factor_gain=denominator_gain,
                radical_given_denominator_novelty=(
                    radical_given_denominator
                ),
                denominator_given_radical_novelty=(
                    denominator_given_radical
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
                state.radical_given_denominator_novelty,
                state.denominator_given_radical_novelty,
            ),
            -state.training_r2,
            state.expression,
        ),
    )[:shortlist_size]
    lane_names: dict[int, list[str]] = {}
    for state in value_lane:
        lane_names.setdefault(id(state), []).append(
            "value_relativistic_trig_rational"
        )
    for state in sobolev_lane:
        lane_names.setdefault(id(state), []).append(
            "sobolev_relativistic_trig_rational"
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
        RelativisticTrigRationalProposal(
            expression=state.expression,
            amplitude_kind=state.amplitude_kind,
            amplitude_axis=state.amplitude_axis,
            ratio_axes=state.ratio_axes,
            phase_axis=state.phase_axis,
            radical_sign=state.radical_sign,
            radical_scale=state.radical_scale,
            denominator_sign=state.denominator_sign,
            denominator_scale=state.denominator_scale,
            phase_scale=state.phase_scale,
            fitted_coefficients=state.fitted_coefficients,
            training_r2=state.training_r2,
            prediction_correlation=state.prediction_correlation,
            candidate_sobolev_gain=state.candidate_sobolev_gain,
            quotient_coupling_gain=state.quotient_coupling_gain,
            radical_factor_gain=state.radical_factor_gain,
            denominator_factor_gain=state.denominator_factor_gain,
            radical_given_denominator_novelty=(
                state.radical_given_denominator_novelty
            ),
            denominator_given_radical_novelty=(
                state.denominator_given_radical_novelty
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
    unique: list[RelativisticTrigRationalProposal] = []
    seen_expressions: set[str] = set()
    for proposal in proposals:
        if proposal.expression in seen_expressions:
            continue
        seen_expressions.add(proposal.expression)
        unique.append(proposal)
        if len(unique) >= proposal_limit:
            break
    return RelativisticTrigRationalResult(
        proposals=tuple(unique),
        candidates_screened=candidates_screened,
        defined_candidates=defined_candidates,
        value_pool=len(retained),
        sobolev_candidates=len(scored),
        ratio_rejections=ratio_rejections,
        amplitude_rejections=amplitude_rejections,
        radical_rejections=radical_rejections,
        denominator_rejections=denominator_rejections,
        numeric_failures=numeric_failures,
        dimension_fallback=False,
    )
