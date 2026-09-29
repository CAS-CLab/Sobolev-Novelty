"""Conditional factored-polynomial proposals for deep-GP Stage CF.

The implementation forms the complete bounded first layer before screening.
Only then does it use a conditional residual scan to add the second monomial
branch.  Sobolev geometry supplies a parallel shortlist and never vetoes the
search-value lane.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from heapq import nsmallest
from itertools import product
from typing import Sequence

import numpy as np

from .sn_archive_interactions import sobolev_product_signature
from .sn_basis_archive import BasisArchiveEntry
from .sn_direct_composition import sobolev_division_signature
from .sn_population_coverage import orthonormal_signature_span
from .sn_shared_denominator import _signature_gain


@dataclass(frozen=True)
class ConditionalFactoredPolynomialProposal:
    """One complete ordinary-GP expression retained by either fixed lane."""

    expression: str
    base_exponents: tuple[int, ...]
    modulated_exponents: tuple[int, ...]
    common_exponents: tuple[int, ...]
    base_remainder_exponents: tuple[int, ...]
    modulated_remainder_exponents: tuple[int, ...]
    affine_axis: int
    affine_sign: int
    affine_scale: float
    fitted_coefficients: tuple[float, float]
    training_r2: float
    prediction_correlation: float
    candidate_sobolev_gain: float
    base_given_modulated_novelty: float
    modulated_given_base_novelty: float
    joint_score: float
    selection_lanes: tuple[str, ...]


@dataclass(frozen=True)
class ConditionalFactoredPolynomialResult:
    """Compact audit result for complete first- and second-layer formation."""

    proposals: tuple[ConditionalFactoredPolynomialProposal, ...]
    monomial_candidates: int
    first_layer_candidates: int
    first_layer_value_pool: int
    residual_candidates_screened: int
    complete_candidates: int
    sobolev_candidates: int
    numeric_failures: int
    dimension_fallback: bool


@dataclass(frozen=True)
class _FirstLayerState:
    monomial_index: int
    affine_axis: int
    affine_sign: int
    affine_scale: float
    expression: str
    training_r2: float
    prediction_correlation: float


@dataclass(frozen=True)
class _CompleteState:
    expression: str
    base_index: int
    modulated_index: int
    common_exponents: tuple[int, ...]
    base_remainder_exponents: tuple[int, ...]
    modulated_remainder_exponents: tuple[int, ...]
    affine_axis: int
    affine_sign: int
    affine_scale: float
    fitted_coefficients: tuple[float, float]
    training_r2: float
    prediction_correlation: float
    candidate_sobolev_gain: float
    base_given_modulated_novelty: float
    modulated_given_base_novelty: float
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


def _scale_text(value: float) -> str:
    if np.isclose(value, 0.5, rtol=0.0, atol=1e-15):
        return "(1 / 2)"
    if np.isclose(value, 1.0, rtol=0.0, atol=1e-15):
        return "1"
    if np.isclose(value, 2.0, rtol=0.0, atol=1e-15):
        return "2"
    raise ValueError(f"unsupported Stage-CF affine scale {value!r}")


def _power_text(name: str, degree: int) -> str:
    return name if degree == 1 else f"{name} ** {degree}"


def _monomial_expression(
    exponents: Sequence[int],
    names: Sequence[str],
) -> str:
    numerator = [
        _power_text(name, int(exponent))
        for name, exponent in zip(names, exponents, strict=True)
        if exponent > 0
    ]
    denominator = [
        _power_text(name, -int(exponent))
        for name, exponent in zip(names, exponents, strict=True)
        if exponent < 0
    ]
    numerator_text = " * ".join(numerator) or "1"
    if not denominator:
        return numerator_text
    return f"({numerator_text}) / ({' * '.join(denominator)})"


def _affine_expression(name: str, sign: int, scale: float) -> str:
    scale_text = _scale_text(scale)
    scaled = name if scale_text == "1" else f"({scale_text}) * {name}"
    operator = "+" if sign > 0 else "-"
    return f"1 {operator} {scaled}"


def _first_layer_expression(
    monomial: str,
    name: str,
    sign: int,
    scale: float,
) -> str:
    return f"({monomial}) * ({_affine_expression(name, sign, scale)})"


def _common_exponents(
    left: Sequence[int],
    right: Sequence[int],
) -> tuple[int, ...]:
    common: list[int] = []
    for first, second in zip(left, right, strict=True):
        if first > 0 and second > 0:
            common.append(min(int(first), int(second)))
        elif first < 0 and second < 0:
            common.append(-min(-int(first), -int(second)))
        else:
            common.append(0)
    return tuple(common)


def _factored_expression(
    base: Sequence[int],
    modulated: Sequence[int],
    names: Sequence[str],
    *,
    affine_axis: int,
    affine_sign: int,
    affine_scale: float,
) -> tuple[str, tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    common = _common_exponents(base, modulated)
    base_remainder = tuple(
        int(value) - factor for value, factor in zip(base, common, strict=True)
    )
    modulated_remainder = tuple(
        int(value) - factor
        for value, factor in zip(modulated, common, strict=True)
    )
    base_text = _monomial_expression(base_remainder, names)
    modulated_text = _monomial_expression(modulated_remainder, names)
    affine_text = _affine_expression(
        names[affine_axis], affine_sign, affine_scale
    )
    modulated_branch = (
        f"({affine_text})"
        if not any(modulated_remainder)
        else f"({modulated_text}) * ({affine_text})"
    )
    bracket = f"({base_text}) + ({modulated_branch})"
    if not any(common):
        return bracket, common, base_remainder, modulated_remainder
    common_text = _monomial_expression(common, names)
    return (
        f"({common_text}) * ({bracket})",
        common,
        base_remainder,
        modulated_remainder,
    )


def _enumerate_exponents(
    dimension: int,
    exponent_min: int,
    exponent_max: int,
    max_numerator_degree: int,
    max_denominator_degree: int,
) -> tuple[tuple[int, ...], ...]:
    return tuple(
        tuple(int(value) for value in vector)
        for vector in product(
            range(exponent_min, exponent_max + 1), repeat=dimension
        )
        if any(vector)
        and sum(max(int(value), 0) for value in vector)
        <= max_numerator_degree
        and sum(max(-int(value), 0) for value in vector)
        <= max_denominator_degree
    )


def _monomial_values(
    X: np.ndarray,
    exponents: Sequence[Sequence[int]],
    protected_epsilon: float,
) -> tuple[np.ndarray, int]:
    vectors = np.asarray(exponents, dtype=int)
    numerator = np.ones((len(vectors), len(X)), dtype=float)
    denominator = np.ones_like(numerator)
    with np.errstate(all="ignore"):
        for axis in range(X.shape[1]):
            for degree in range(1, int(np.max(vectors[:, axis])) + 1):
                rows = vectors[:, axis] == degree
                numerator[rows] *= np.power(X[:, axis], degree)
            for degree in range(1, int(np.max(-vectors[:, axis])) + 1):
                rows = vectors[:, axis] == -degree
                denominator[rows] *= np.power(X[:, axis], degree)
        values = numerator / (
            denominator + protected_epsilon * (denominator == 0.0)
        )
    finite = np.all(np.isfinite(values), axis=1)
    values[~finite] = 0.0
    return values, int((~finite).sum())


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
    training_r2 = float(
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
    if not np.isfinite(training_r2) or not np.isfinite(correlation):
        return None
    return (
        (float(rounded[0]), float(rounded[1])),
        training_r2,
        float(np.clip(correlation, 0.0, 1.0)),
    )


def _monomial_signature(
    exponents: Sequence[int],
    direct: Sequence[BasisArchiveEntry],
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
    constant = _constant_signature(sample_count, dimension, lambda_value)
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


def _complete_value_key(
    state: _CompleteState,
) -> tuple[float, float, str]:
    return (
        -state.training_r2,
        -state.prediction_correlation,
        state.expression,
    )


def lift_direct_conditional_factored_polynomials(
    *,
    direct_entries: Sequence[BasisArchiveEntry],
    target: Sequence[float],
    variable_names: Sequence[str],
    geometry_sample_count: int,
    exponent_min: int,
    exponent_max: int,
    max_numerator_degree: int,
    max_denominator_degree: int,
    affine_signs: Sequence[int],
    affine_scales: Sequence[float],
    first_layer_value_pool_size: int,
    residual_shortlist_size: int,
    complete_shortlist_size: int,
    proposal_limit: int,
    protected_epsilon: float = 1e-6,
    lambda_value: float = 1.0,
    lambda_gradient: float = 1.0,
) -> ConditionalFactoredPolynomialResult:
    """Enumerate the frozen six-coordinate CF family and return dual-lane proposals."""

    direct = tuple(direct_entries)
    names = tuple(str(value) for value in variable_names)
    y = np.asarray(target, dtype=float).reshape(-1)
    signs = tuple(dict.fromkeys(int(value) for value in affine_signs))
    scales = tuple(dict.fromkeys(float(value) for value in affine_scales))
    dimension = len(names)
    if dimension != 6:
        return ConditionalFactoredPolynomialResult(
            (), 0, 0, 0, 0, 0, 0, 0, True
        )
    signature_size = geometry_sample_count * (dimension + 1)
    if (
        len(direct) != dimension
        or y.size < 2
        or not np.all(np.isfinite(y))
        or float(np.var(y)) <= np.finfo(float).eps
        or geometry_sample_count < 1
        or exponent_min >= 0
        or exponent_max < 1
        or max_numerator_degree < 1
        or max_denominator_degree < 1
        or set(signs) != {-1, 1}
        or not scales
        or first_layer_value_pool_size < 1
        or residual_shortlist_size < 1
        or complete_shortlist_size < 1
        or proposal_limit < 1
        or protected_epsilon <= 0.0
        or lambda_value <= 0.0
        or lambda_gradient <= 0.0
    ):
        raise ValueError("conditional factored-polynomial inputs are invalid")
    for scale in scales:
        _scale_text(scale)
    if any(
        np.asarray(entry.values).shape != y.shape
        or np.asarray(entry.signature).shape != (signature_size,)
        or not np.all(np.isfinite(entry.values))
        or not np.all(np.isfinite(entry.signature))
        for entry in direct
    ):
        raise ValueError("conditional factored-polynomial entries are not aligned")

    X = np.column_stack(
        tuple(np.asarray(entry.values, dtype=float) for entry in direct)
    )
    exponent_vectors = _enumerate_exponents(
        dimension,
        exponent_min,
        exponent_max,
        max_numerator_degree,
        max_denominator_degree,
    )
    expressions = tuple(
        _monomial_expression(vector, names) for vector in exponent_vectors
    )
    monomial_matrix, numeric_failures = _monomial_values(
        X, exponent_vectors, protected_epsilon
    )
    centered_target = y - float(np.mean(y))
    target_variance = float(np.var(y))
    target_norm = float(np.linalg.norm(centered_target))
    monomial_means = np.mean(monomial_matrix, axis=1)
    monomial_centered = monomial_matrix - monomial_means[:, None]
    monomial_energy = np.einsum(
        "ij,ij->i", monomial_centered, monomial_centered
    )

    combinations = tuple(
        (sign, scale, axis)
        for sign in signs
        for scale in scales
        for axis in range(dimension)
    )
    count = len(exponent_vectors)
    first_layer_r2 = np.full((len(combinations), count), -np.inf)
    first_layer_correlation = np.full_like(first_layer_r2, -np.inf)
    for combination_index, (sign, scale, axis) in enumerate(combinations):
        factor = 1.0 + sign * scale * X[:, axis]
        basis = monomial_matrix * factor[None, :]
        means = np.mean(basis, axis=1)
        centered = basis - means[:, None]
        energy = np.einsum("ij,ij->i", centered, centered)
        valid = energy > np.finfo(float).eps
        slopes = np.zeros(count, dtype=float)
        slopes[valid] = centered[valid] @ centered_target / energy[valid]
        intercepts = float(np.mean(y)) - slopes * means
        rounded_intercepts = np.round(intercepts, 6)
        rounded_slopes = np.round(slopes, 6)
        predictions = (
            rounded_intercepts[:, None] + rounded_slopes[:, None] * basis
        )
        r2 = 1.0 - np.mean(np.square(predictions - y), axis=1) / target_variance
        centered_prediction = rounded_slopes[:, None] * centered
        prediction_norm = np.sqrt(
            np.einsum("ij,ij->i", centered_prediction, centered_prediction)
        )
        correlations = np.zeros(count, dtype=float)
        correlation_valid = valid & (prediction_norm > np.finfo(float).eps)
        correlations[correlation_valid] = np.abs(
            centered_prediction[correlation_valid] @ centered_target
        ) / (prediction_norm[correlation_valid] * target_norm)
        finite = valid & np.isfinite(r2) & np.isfinite(correlations)
        first_layer_r2[combination_index, finite] = r2[finite]
        first_layer_correlation[combination_index, finite] = np.clip(
            correlations[finite], 0.0, 1.0
        )
        numeric_failures += int((~finite).sum())

    total_first_layer = len(combinations) * count

    def first_layer_key(flat_index: int) -> tuple[float, float, str]:
        combination_index, monomial_index = divmod(flat_index, count)
        sign, scale, axis = combinations[combination_index]
        return (
            -float(first_layer_r2[combination_index, monomial_index]),
            -float(first_layer_correlation[combination_index, monomial_index]),
            _first_layer_expression(
                expressions[monomial_index], names[axis], sign, scale
            ),
        )

    first_layer_indices = nsmallest(
        first_layer_value_pool_size,
        range(total_first_layer),
        key=first_layer_key,
    )
    first_layer: list[_FirstLayerState] = []
    for flat_index in first_layer_indices:
        combination_index, monomial_index = divmod(flat_index, count)
        sign, scale, axis = combinations[combination_index]
        if not np.isfinite(first_layer_r2[combination_index, monomial_index]):
            continue
        first_layer.append(
            _FirstLayerState(
                monomial_index=monomial_index,
                affine_axis=axis,
                affine_sign=sign,
                affine_scale=scale,
                expression=_first_layer_expression(
                    expressions[monomial_index], names[axis], sign, scale
                ),
                training_r2=float(
                    first_layer_r2[combination_index, monomial_index]
                ),
                prediction_correlation=float(
                    first_layer_correlation[
                        combination_index, monomial_index
                    ]
                ),
            )
        )

    complete: list[_CompleteState] = []
    residual_candidates_screened = 0
    for state in first_layer:
        factor = (
            1.0
            + state.affine_sign
            * state.affine_scale
            * X[:, state.affine_axis]
        )
        g_values = monomial_matrix[state.monomial_index] * factor
        g_centered = g_values - float(np.mean(g_values))
        g_energy = float(np.dot(g_centered, g_centered))
        if g_energy <= np.finfo(float).eps:
            numeric_failures += count
            continue
        g_slope = float(np.dot(g_centered, centered_target) / g_energy)
        residual = centered_target - g_slope * g_centered
        residual_energy = float(np.dot(residual, residual))
        projections = monomial_centered @ g_centered
        conditional_energy = monomial_energy - np.square(projections) / g_energy
        residual_dot = monomial_centered @ residual
        denominator = np.sqrt(
            np.maximum(conditional_energy, 0.0) * residual_energy
        )
        conditional_correlation = np.zeros(count, dtype=float)
        valid = denominator > np.finfo(float).eps
        conditional_correlation[valid] = (
            np.abs(residual_dot[valid]) / denominator[valid]
        )
        conditional_correlation[~np.isfinite(conditional_correlation)] = 0.0
        residual_candidates_screened += count
        base_indices = nsmallest(
            residual_shortlist_size,
            range(count),
            key=lambda index: (
                -float(conditional_correlation[index]),
                expressions[index],
            ),
        )
        for base_index in base_indices:
            candidate_values = monomial_matrix[base_index] + g_values
            fit = _fit_basis(
                candidate_values,
                y,
                centered_target,
                target_variance,
                target_norm,
            )
            if fit is None:
                numeric_failures += 1
                continue
            fitted_coefficients, training_r2, correlation = fit
            (
                expression,
                common,
                base_remainder,
                modulated_remainder,
            ) = _factored_expression(
                exponent_vectors[base_index],
                exponent_vectors[state.monomial_index],
                names,
                affine_axis=state.affine_axis,
                affine_sign=state.affine_sign,
                affine_scale=state.affine_scale,
            )
            complete.append(
                _CompleteState(
                    expression=expression,
                    base_index=base_index,
                    modulated_index=state.monomial_index,
                    common_exponents=common,
                    base_remainder_exponents=base_remainder,
                    modulated_remainder_exponents=modulated_remainder,
                    affine_axis=state.affine_axis,
                    affine_sign=state.affine_sign,
                    affine_scale=state.affine_scale,
                    fitted_coefficients=fitted_coefficients,
                    training_r2=training_r2,
                    prediction_correlation=correlation,
                    candidate_sobolev_gain=0.0,
                    base_given_modulated_novelty=0.0,
                    modulated_given_base_novelty=0.0,
                    joint_score=0.0,
                    selection_lanes=(),
                )
            )

    constant = _constant_signature(
        geometry_sample_count, dimension, lambda_value
    )
    coordinate_span = orthonormal_signature_span(
        tuple(np.asarray(entry.signature, dtype=float) for entry in direct)
    )
    signature_cache: dict[int, np.ndarray] = {}

    def monomial_signature(index: int) -> np.ndarray:
        if index not in signature_cache:
            signature_cache[index] = _monomial_signature(
                exponent_vectors[index],
                direct,
                sample_count=geometry_sample_count,
                dimension=dimension,
                protected_epsilon=protected_epsilon,
                lambda_value=lambda_value,
                lambda_gradient=lambda_gradient,
            )
        return signature_cache[index]

    scored: list[_CompleteState] = []
    signature_keyword = {
        "sample_count": geometry_sample_count,
        "dimension": dimension,
        "lambda_value": lambda_value,
        "lambda_gradient": lambda_gradient,
    }
    for state in complete:
        try:
            base_signature = monomial_signature(state.base_index)
            modulated_monomial_signature = monomial_signature(
                state.modulated_index
            )
            factor_signature = (
                constant
                + state.affine_sign
                * state.affine_scale
                * np.asarray(
                    direct[state.affine_axis].signature, dtype=float
                )
            )
            modulated_signature = sobolev_product_signature(
                modulated_monomial_signature,
                factor_signature,
                **signature_keyword,
            )
            raw_candidate_signature = base_signature + modulated_signature
            intercept, slope = state.fitted_coefficients
            candidate_signature = (
                intercept * constant + slope * raw_candidate_signature
            )
            candidate_gain = _signature_gain(
                candidate_signature, coordinate_span
            )
            base_given_modulated = _signature_gain(
                base_signature,
                orthonormal_signature_span((modulated_signature,)),
            )
            modulated_given_base = _signature_gain(
                modulated_signature,
                orthonormal_signature_span((base_signature,)),
            )
        except ValueError:
            numeric_failures += 1
            candidate_gain = 0.0
            base_given_modulated = 0.0
            modulated_given_base = 0.0
        joint_score = 0.5 * (
            candidate_gain + min(base_given_modulated, modulated_given_base)
        )
        scored.append(
            replace(
                state,
                candidate_sobolev_gain=candidate_gain,
                base_given_modulated_novelty=base_given_modulated,
                modulated_given_base_novelty=modulated_given_base,
                joint_score=joint_score,
            )
        )

    value_lane = sorted(scored, key=_complete_value_key)[
        :complete_shortlist_size
    ]
    sobolev_lane = sorted(
        scored,
        key=lambda state: (
            -state.joint_score,
            -state.candidate_sobolev_gain,
            -min(
                state.base_given_modulated_novelty,
                state.modulated_given_base_novelty,
            ),
            -state.training_r2,
            state.expression,
        ),
    )[:complete_shortlist_size]
    lane_names: dict[int, list[str]] = {}
    for state in value_lane:
        lane_names.setdefault(id(state), []).append(
            "value_conditional_factored_polynomial"
        )
    for state in sobolev_lane:
        lane_names.setdefault(id(state), []).append(
            "sobolev_conditional_factored_polynomial"
        )
    selected: list[_CompleteState] = []
    seen_ids: set[int] = set()
    for state in (*value_lane, *sobolev_lane):
        if id(state) in seen_ids:
            continue
        seen_ids.add(id(state))
        selected.append(
            replace(state, selection_lanes=tuple(lane_names[id(state)]))
        )

    proposals = [
        ConditionalFactoredPolynomialProposal(
            expression=state.expression,
            base_exponents=exponent_vectors[state.base_index],
            modulated_exponents=exponent_vectors[state.modulated_index],
            common_exponents=state.common_exponents,
            base_remainder_exponents=state.base_remainder_exponents,
            modulated_remainder_exponents=(
                state.modulated_remainder_exponents
            ),
            affine_axis=state.affine_axis,
            affine_sign=state.affine_sign,
            affine_scale=state.affine_scale,
            fitted_coefficients=state.fitted_coefficients,
            training_r2=state.training_r2,
            prediction_correlation=state.prediction_correlation,
            candidate_sobolev_gain=state.candidate_sobolev_gain,
            base_given_modulated_novelty=(
                state.base_given_modulated_novelty
            ),
            modulated_given_base_novelty=(
                state.modulated_given_base_novelty
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
    unique: list[ConditionalFactoredPolynomialProposal] = []
    seen_expressions: set[str] = set()
    for proposal in proposals:
        if proposal.expression in seen_expressions:
            continue
        seen_expressions.add(proposal.expression)
        unique.append(proposal)
        if len(unique) >= proposal_limit:
            break
    return ConditionalFactoredPolynomialResult(
        proposals=tuple(unique),
        monomial_candidates=count,
        first_layer_candidates=total_first_layer,
        first_layer_value_pool=len(first_layer),
        residual_candidates_screened=residual_candidates_screened,
        complete_candidates=len(complete),
        sobolev_candidates=len(scored),
        numeric_failures=numeric_failures,
        dimension_fallback=False,
    )
