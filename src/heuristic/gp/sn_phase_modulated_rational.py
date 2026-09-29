"""Complete phase-modulated rational proposals for Stage CI.

The construction enumerates

``A / (1 + s_o*c_o*B*(1 + s_i*c_i*cos(c_p*x_k)))``

over a bounded monomial-ratio library.  A deterministic chunked value screen
forms every complete tuple before keeping a global top pool.  Exact Sobolev
signatures are then used to rank the internal phase-modulated product in a
parallel lane; final-candidate novelty never acts as an eligibility gate.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import product
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
class PhaseModulatedRationalProposal:
    expression: str
    amplitude_kind: str
    amplitude_axis: int | None
    monomial_exponents: tuple[int, ...]
    phase_axis: int
    outer_sign: int
    outer_scale: float
    inner_sign: int
    inner_scale: float
    phase_scale: float
    fitted_coefficients: tuple[float, float]
    training_r2: float
    prediction_correlation: float
    candidate_sobolev_gain: float
    quotient_coupling_gain: float
    interaction_sobolev_gain: float
    monomial_given_phase_novelty: float
    phase_given_monomial_novelty: float
    product_coupling_gain: float
    joint_score: float
    selection_lanes: tuple[str, ...]


@dataclass(frozen=True)
class PhaseModulatedRationalResult:
    proposals: tuple[PhaseModulatedRationalProposal, ...]
    monomial_candidates: int
    candidates_screened: int
    defined_candidates: int
    value_pool: int
    sobolev_candidates: int
    monomial_rejections: int
    amplitude_rejections: int
    denominator_rejections: int
    numeric_failures: int
    chunk_size: int
    dimension_fallback: bool


@dataclass(frozen=True)
class _Monomial:
    exponents: tuple[int, ...]
    expression: str
    search_values: np.ndarray | None
    geometry_values: np.ndarray | None


@dataclass(frozen=True)
class _ValueState:
    expression: str
    amplitude_kind: str
    amplitude_axis: int | None
    monomial_exponents: tuple[int, ...]
    phase_axis: int
    outer_sign: int
    outer_scale: float
    inner_sign: int
    inner_scale: float
    phase_scale: float
    fitted_coefficients: tuple[float, float]
    training_r2: float
    prediction_correlation: float
    selection_lanes: tuple[str, ...] = ()
    candidate_sobolev_gain: float = 0.0
    quotient_coupling_gain: float = 0.0
    interaction_sobolev_gain: float = 0.0
    monomial_given_phase_novelty: float = 0.0
    phase_given_monomial_novelty: float = 0.0
    product_coupling_gain: float = 0.0
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
    raise ValueError(f"unsupported Stage-CI scale {scale!r}")


def _scaled_term(scale: float, term: str) -> str:
    text = _scale_text(scale)
    return term if text == "1" else f"({text}) * ({term})"


def _monomial_expression(
    exponents: Sequence[int], variable_names: Sequence[str]
) -> str:
    numerator: list[str] = []
    denominator: list[str] = []
    for exponent, name in zip(exponents, variable_names, strict=True):
        if exponent == 0:
            continue
        degree = abs(int(exponent))
        term = str(name) if degree == 1 else f"{name} ** {degree}"
        (numerator if exponent > 0 else denominator).append(term)
    numerator_text = " * ".join(numerator) or "1"
    if not denominator:
        return numerator_text
    return f"({numerator_text}) / ({' * '.join(denominator)})"


def _bounded_exponents(dimension: int) -> tuple[tuple[int, ...], ...]:
    return tuple(
        vector
        for vector in product(range(-2, 3), repeat=dimension)
        if any(vector)
        and sum(max(value, 0) for value in vector) <= 2
        and sum(max(-value, 0) for value in vector) <= 3
    )


def _expression(
    names: Sequence[str],
    *,
    amplitude_kind: str,
    amplitude_axis: int | None,
    monomial_exponents: Sequence[int],
    phase_axis: int,
    outer_sign: int,
    outer_scale: float,
    inner_sign: int,
    inner_scale: float,
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
    monomial = _monomial_expression(monomial_exponents, names)
    phase = _scaled_term(phase_scale, names[phase_axis])
    inner_operator = "+" if inner_sign > 0 else "-"
    inner_cosine = _scaled_term(inner_scale, f"cos({phase})")
    inner = f"1 {inner_operator} {inner_cosine}"
    interaction = f"({monomial}) * ({inner})"
    outer_operator = "+" if outer_sign > 0 else "-"
    scaled_interaction = _scaled_term(outer_scale, interaction)
    denominator = f"1 {outer_operator} {scaled_interaction}"
    return f"({amplitude}) / ({denominator})"


def _value_key(state: _ValueState) -> tuple[float, float, str]:
    return (
        -state.training_r2,
        -state.prediction_correlation,
        state.expression,
    )


def _monomial_signature(
    exponents: Sequence[int],
    direct: Sequence[BasisArchiveEntry],
    constant: np.ndarray,
    *,
    sample_count: int,
    dimension: int,
    protected_epsilon: float,
    lambda_value: float,
    lambda_gradient: float,
) -> np.ndarray:
    keyword = {
        "sample_count": sample_count,
        "dimension": dimension,
        "lambda_value": lambda_value,
        "lambda_gradient": lambda_gradient,
    }
    numerator = constant
    denominator = constant
    for axis, exponent in enumerate(exponents):
        target = numerator if exponent > 0 else denominator
        for _ in range(abs(int(exponent))):
            target = sobolev_product_signature(
                target, direct[axis].signature, **keyword
            )
        if exponent > 0:
            numerator = target
        elif exponent < 0:
            denominator = target
    return sobolev_division_signature(
        numerator,
        denominator,
        protected_epsilon=protected_epsilon,
        **keyword,
    )


def lift_direct_phase_modulated_rationals(
    *,
    direct_entries: Sequence[BasisArchiveEntry],
    target: Sequence[float],
    variable_names: Sequence[str],
    geometry_sample_count: int,
    outer_scales: Sequence[float],
    inner_scales: Sequence[float],
    phase_scales: Sequence[float],
    value_pool_size: int,
    shortlist_size: int,
    proposal_limit: int,
    chunk_size: int = 8,
    protected_epsilon: float = 1e-6,
    lambda_value: float = 1.0,
    lambda_gradient: float = 1.0,
) -> PhaseModulatedRationalResult:
    """Screen all 1,386,720 frozen four-dimensional tuples in chunks."""

    direct = tuple(direct_entries)
    names = tuple(str(value) for value in variable_names)
    y = np.asarray(target, dtype=float).reshape(-1)
    dimension = len(names)
    if dimension != 4:
        return PhaseModulatedRationalResult(
            (), 0, 0, 0, 0, 0, 0, 0, 0, 0, int(chunk_size), True
        )
    outer_scale_values = tuple(
        dict.fromkeys(float(value) for value in outer_scales)
    )
    inner_scale_values = tuple(
        dict.fromkeys(float(value) for value in inner_scales)
    )
    phase_scale_values = tuple(
        dict.fromkeys(float(value) for value in phase_scales)
    )
    signature_size = geometry_sample_count * (dimension + 1)
    if (
        len(direct) != dimension
        or y.size < 2
        or not np.all(np.isfinite(y))
        or float(np.var(y)) <= np.finfo(float).eps
        or geometry_sample_count < 1
        or not outer_scale_values
        or not inner_scale_values
        or not phase_scale_values
        or value_pool_size < shortlist_size
        or shortlist_size < 1
        or proposal_limit < 1
        or chunk_size < 1
        or protected_epsilon <= 0.0
        or lambda_value <= 0.0
        or lambda_gradient <= 0.0
    ):
        raise ValueError("phase-modulated rational inputs are invalid")
    for scale in (
        *outer_scale_values,
        *inner_scale_values,
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
        raise ValueError("phase-modulated rational entries are not aligned")

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
    amplitudes: list[
        tuple[str, int | None, np.ndarray | None, np.ndarray | None]
    ] = [("constant", None, ones, geometry_ones)]
    for axis in range(dimension):
        amplitudes.append(
            ("direct", axis, X[:, axis], geometry_X[:, axis])
        )
    for axis in range(dimension):
        if (
            np.any(np.abs(X[:, axis]) <= protected_epsilon)
            or np.any(np.abs(geometry_X[:, axis]) <= protected_epsilon)
        ):
            amplitudes.append(("reciprocal", axis, None, None))
        else:
            amplitudes.append(
                (
                    "reciprocal",
                    axis,
                    ones / X[:, axis],
                    geometry_ones / geometry_X[:, axis],
                )
            )

    exponent_vectors = _bounded_exponents(dimension)
    monomials: list[_Monomial] = []
    for exponents in exponent_vectors:
        invalid = any(
            exponent < 0
            and (
                np.any(np.abs(X[:, axis]) <= protected_epsilon)
                or np.any(
                    np.abs(geometry_X[:, axis]) <= protected_epsilon
                )
            )
            for axis, exponent in enumerate(exponents)
        )
        if invalid:
            monomials.append(
                _Monomial(
                    exponents,
                    _monomial_expression(exponents, names),
                    None,
                    None,
                )
            )
            continue
        with np.errstate(all="ignore"):
            search_values = np.prod(
                np.power(X, np.asarray(exponents, dtype=int)), axis=1
            )
            geometry_values = np.prod(
                np.power(
                    geometry_X, np.asarray(exponents, dtype=int)
                ),
                axis=1,
            )
        if (
            not np.all(np.isfinite(search_values))
            or not np.all(np.isfinite(geometry_values))
        ):
            search_values = None
            geometry_values = None
        monomials.append(
            _Monomial(
                exponents,
                _monomial_expression(exponents, names),
                search_values,
                geometry_values,
            )
        )

    outer_meta = tuple(
        (sign, scale, float(sign) * scale)
        for sign in (-1, 1)
        for scale in outer_scale_values
    )
    inner_meta = tuple(
        (sign, scale, float(sign) * scale)
        for sign in (-1, 1)
        for scale in inner_scale_values
    )
    outer_coefficients = np.asarray(
        [item[2] for item in outer_meta], dtype=float
    )
    inner_coefficients = np.asarray(
        [item[2] for item in inner_meta], dtype=float
    )
    centered_target = y - float(np.mean(y))
    target_variance = float(np.var(y))
    target_norm = float(np.linalg.norm(centered_target))
    combinations_per_monomial_phase = (
        len(outer_meta) * len(inner_meta) * len(phase_scale_values)
    )
    combinations_per_amplitude = (
        len(monomials) * dimension * combinations_per_monomial_phase
    )
    candidates_screened = len(amplitudes) * combinations_per_amplitude
    retained: list[_ValueState] = []
    defined_candidates = 0
    monomial_rejections = 0
    amplitude_rejections = 0
    denominator_rejections = 0
    numeric_failures = 0

    for amplitude_kind, amplitude_axis, amplitude, geometry_amplitude in amplitudes:
        if amplitude is None or geometry_amplitude is None:
            amplitude_rejections += combinations_per_amplitude
            continue
        for phase_axis in range(dimension):
            phase_search = X[:, phase_axis]
            phase_geometry = geometry_X[:, phase_axis]
            phase_cosines = np.column_stack(
                tuple(
                    np.cos(scale * phase_search)
                    for scale in phase_scale_values
                )
            )
            geometry_cosines = np.column_stack(
                tuple(
                    np.cos(scale * phase_geometry)
                    for scale in phase_scale_values
                )
            )
            q_search = (
                1.0
                + inner_coefficients[None, :, None]
                * phase_cosines[:, None, :]
            )
            q_geometry = (
                1.0
                + inner_coefficients[None, :, None]
                * geometry_cosines[:, None, :]
            )
            for start in range(0, len(monomials), chunk_size):
                chunk = monomials[start : start + chunk_size]
                valid_local = [
                    index
                    for index, monomial in enumerate(chunk)
                    if monomial.search_values is not None
                    and monomial.geometry_values is not None
                ]
                monomial_rejections += (
                    len(chunk) - len(valid_local)
                ) * combinations_per_monomial_phase
                if not valid_local:
                    continue
                valid_monomials = [chunk[index] for index in valid_local]
                B = np.column_stack(
                    tuple(
                        np.asarray(monomial.search_values, dtype=float)
                        for monomial in valid_monomials
                    )
                )
                geometry_B = np.column_stack(
                    tuple(
                        np.asarray(monomial.geometry_values, dtype=float)
                        for monomial in valid_monomials
                    )
                )
                denominator = (
                    1.0
                    + outer_coefficients[None, None, :, None, None]
                    * B[:, :, None, None, None]
                    * q_search[:, None, None, :, :]
                )
                geometry_denominator = (
                    1.0
                    + outer_coefficients[None, None, :, None, None]
                    * geometry_B[:, :, None, None, None]
                    * q_geometry[:, None, None, :, :]
                )
                flat_shape = (
                    len(valid_monomials)
                    * len(outer_meta)
                    * len(inner_meta)
                    * len(phase_scale_values)
                )
                denominator = denominator.reshape(len(y), flat_shape)
                geometry_denominator = geometry_denominator.reshape(
                    geometry_sample_count, flat_shape
                )
                domain_valid = (
                    np.all(
                        np.abs(denominator) > protected_epsilon, axis=0
                    )
                    & np.all(
                        np.abs(geometry_denominator) > protected_epsilon,
                        axis=0,
                    )
                )
                denominator_rejections += int(
                    flat_shape - np.count_nonzero(domain_valid)
                )
                with np.errstate(all="ignore"):
                    basis = amplitude[:, None] / denominator
                finite = domain_valid & np.all(np.isfinite(basis), axis=0)
                numeric_failures += int(
                    np.count_nonzero(domain_valid & ~finite)
                )
                defined_candidates += int(np.count_nonzero(finite))
                means = np.mean(basis, axis=0)
                centered = basis - means[None, :]
                energy = np.sum(np.square(centered), axis=0)
                fit_valid = finite & (energy > np.finfo(float).eps)
                slopes = np.zeros(flat_shape, dtype=float)
                slopes[fit_valid] = (
                    centered[:, fit_valid].T @ centered_target
                ) / energy[fit_valid]
                intercepts = np.mean(y) - slopes * means
                rounded_intercepts = np.round(intercepts, 6)
                rounded_slopes = np.round(slopes, 6)
                prediction = (
                    rounded_intercepts[None, :]
                    + rounded_slopes[None, :] * basis
                )
                with np.errstate(all="ignore"):
                    r2 = 1.0 - np.mean(
                        np.square(prediction - y[:, None]), axis=0
                    ) / target_variance
                dot = centered.T @ centered_target
                correlation = np.zeros(flat_shape, dtype=float)
                nonzero_prediction = fit_valid & (rounded_slopes != 0.0)
                correlation[nonzero_prediction] = np.abs(
                    dot[nonzero_prediction]
                ) / (
                    np.sqrt(energy[nonzero_prediction]) * target_norm
                )
                fit_valid &= np.isfinite(r2) & np.isfinite(correlation)
                numeric_failures += int(
                    np.count_nonzero(finite & ~fit_valid)
                )
                indices = np.flatnonzero(fit_valid)
                if not len(indices):
                    continue
                order = indices[
                    np.lexsort((-correlation[indices], -r2[indices]))
                ]
                boundary_position = min(value_pool_size, len(order)) - 1
                boundary = order[boundary_position]
                selected_indices = indices[
                    (r2[indices] > r2[boundary])
                    | (
                        (r2[indices] == r2[boundary])
                        & (correlation[indices] >= correlation[boundary])
                    )
                ]
                block_states: list[_ValueState] = []
                for flat_index in selected_indices:
                    (
                        monomial_index,
                        outer_index,
                        inner_index,
                        phase_scale_index,
                    ) = np.unravel_index(
                        int(flat_index),
                        (
                            len(valid_monomials),
                            len(outer_meta),
                            len(inner_meta),
                            len(phase_scale_values),
                        ),
                    )
                    monomial = valid_monomials[monomial_index]
                    outer_sign, outer_scale, _ = outer_meta[outer_index]
                    inner_sign, inner_scale, _ = inner_meta[inner_index]
                    phase_scale = phase_scale_values[phase_scale_index]
                    block_states.append(
                        _ValueState(
                            expression=_expression(
                                names,
                                amplitude_kind=amplitude_kind,
                                amplitude_axis=amplitude_axis,
                                monomial_exponents=monomial.exponents,
                                phase_axis=phase_axis,
                                outer_sign=outer_sign,
                                outer_scale=outer_scale,
                                inner_sign=inner_sign,
                                inner_scale=inner_scale,
                                phase_scale=phase_scale,
                            ),
                            amplitude_kind=amplitude_kind,
                            amplitude_axis=amplitude_axis,
                            monomial_exponents=monomial.exponents,
                            phase_axis=phase_axis,
                            outer_sign=outer_sign,
                            outer_scale=outer_scale,
                            inner_sign=inner_sign,
                            inner_scale=inner_scale,
                            phase_scale=phase_scale,
                            fitted_coefficients=(
                                float(rounded_intercepts[flat_index]),
                                float(rounded_slopes[flat_index]),
                            ),
                            training_r2=float(r2[flat_index]),
                            prediction_correlation=float(
                                np.clip(correlation[flat_index], 0.0, 1.0)
                            ),
                        )
                    )
                retained.extend(block_states)
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
    for axis in range(dimension):
        try:
            reciprocal_signatures[axis] = sobolev_division_signature(
                constant, direct[axis].signature, **division_keyword
            )
        except ValueError:
            pass
    monomial_signatures: dict[tuple[int, ...], np.ndarray] = {}
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
            if state.monomial_exponents not in monomial_signatures:
                monomial_signatures[state.monomial_exponents] = (
                    _monomial_signature(
                        state.monomial_exponents,
                        direct,
                        constant,
                        sample_count=geometry_sample_count,
                        dimension=dimension,
                        protected_epsilon=protected_epsilon,
                        lambda_value=lambda_value,
                        lambda_gradient=lambda_gradient,
                    )
                )
            monomial_signature = monomial_signatures[
                state.monomial_exponents
            ]
            phase_source = (
                state.phase_scale
                * np.asarray(direct[state.phase_axis].signature, dtype=float)
            )
            cosine_signature = sobolev_unary_signature(
                phase_source, transform="cos", **keyword
            )
            phase_factor = (
                constant
                + state.inner_sign
                * state.inner_scale
                * cosine_signature
            )
            interaction = sobolev_product_signature(
                monomial_signature, phase_factor, **keyword
            )
            denominator_factor = (
                constant
                + state.outer_sign * state.outer_scale * interaction
            )
            candidate_raw = sobolev_division_signature(
                amplitude_signature, denominator_factor, **division_keyword
            )
            intercept, slope = state.fitted_coefficients
            candidate_signature = intercept * constant + slope * candidate_raw
            candidate_gain = _signature_gain(
                candidate_signature, coordinate_span
            )
            quotient_coupling = _signature_gain(
                candidate_raw,
                orthonormal_signature_span(
                    (amplitude_signature, denominator_factor)
                ),
            )
            interaction_gain = _signature_gain(interaction, coordinate_span)
            monomial_given_phase = _signature_gain(
                monomial_signature,
                orthonormal_signature_span((phase_factor,)),
            )
            phase_given_monomial = _signature_gain(
                phase_factor,
                orthonormal_signature_span((monomial_signature,)),
            )
            product_coupling = _signature_gain(
                interaction,
                orthonormal_signature_span(
                    (monomial_signature, phase_factor)
                ),
            )
        except (KeyError, ValueError):
            numeric_failures += 1
            candidate_gain = 0.0
            quotient_coupling = 0.0
            interaction_gain = 0.0
            monomial_given_phase = 0.0
            phase_given_monomial = 0.0
            product_coupling = 0.0
        joint_score = float(
            np.mean(
                (
                    interaction_gain,
                    min(monomial_given_phase, phase_given_monomial),
                    product_coupling,
                )
            )
        )
        scored.append(
            replace(
                state,
                candidate_sobolev_gain=candidate_gain,
                quotient_coupling_gain=quotient_coupling,
                interaction_sobolev_gain=interaction_gain,
                monomial_given_phase_novelty=monomial_given_phase,
                phase_given_monomial_novelty=phase_given_monomial,
                product_coupling_gain=product_coupling,
                joint_score=joint_score,
            )
        )

    value_lane = sorted(scored, key=_value_key)[:shortlist_size]
    sobolev_lane = sorted(
        scored,
        key=lambda state: (
            -state.joint_score,
            -state.interaction_sobolev_gain,
            -min(
                state.monomial_given_phase_novelty,
                state.phase_given_monomial_novelty,
            ),
            -state.training_r2,
            state.expression,
        ),
    )[:shortlist_size]
    lane_names: dict[int, list[str]] = {}
    for state in value_lane:
        lane_names.setdefault(id(state), []).append(
            "value_phase_modulated_rational"
        )
    for state in sobolev_lane:
        lane_names.setdefault(id(state), []).append(
            "sobolev_phase_modulated_rational"
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
        PhaseModulatedRationalProposal(
            expression=state.expression,
            amplitude_kind=state.amplitude_kind,
            amplitude_axis=state.amplitude_axis,
            monomial_exponents=state.monomial_exponents,
            phase_axis=state.phase_axis,
            outer_sign=state.outer_sign,
            outer_scale=state.outer_scale,
            inner_sign=state.inner_sign,
            inner_scale=state.inner_scale,
            phase_scale=state.phase_scale,
            fitted_coefficients=state.fitted_coefficients,
            training_r2=state.training_r2,
            prediction_correlation=state.prediction_correlation,
            candidate_sobolev_gain=state.candidate_sobolev_gain,
            quotient_coupling_gain=state.quotient_coupling_gain,
            interaction_sobolev_gain=state.interaction_sobolev_gain,
            monomial_given_phase_novelty=(
                state.monomial_given_phase_novelty
            ),
            phase_given_monomial_novelty=(
                state.phase_given_monomial_novelty
            ),
            product_coupling_gain=state.product_coupling_gain,
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
    unique: list[PhaseModulatedRationalProposal] = []
    seen_expressions: set[str] = set()
    for proposal in proposals:
        if proposal.expression in seen_expressions:
            continue
        seen_expressions.add(proposal.expression)
        unique.append(proposal)
        if len(unique) >= proposal_limit:
            break
    return PhaseModulatedRationalResult(
        proposals=tuple(unique),
        monomial_candidates=len(monomials),
        candidates_screened=candidates_screened,
        defined_candidates=defined_candidates,
        value_pool=len(retained),
        sobolev_candidates=len(scored),
        monomial_rejections=monomial_rejections,
        amplitude_rejections=amplitude_rejections,
        denominator_rejections=denominator_rejections,
        numeric_failures=numeric_failures,
        chunk_size=int(chunk_size),
        dimension_fallback=False,
    )
