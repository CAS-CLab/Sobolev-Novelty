"""Sparse integer-exponent radicals screened in Sobolev geometry.

This export-side plug-in proposes compact candidates of the form

``sqrt(offset + scale * product(x_j ** exponent_j))``.

Continuous exponents are estimated from a log-linear transform of the current
search rows.  Only a deterministic bounded integer neighbourhood is turned
into symbolic candidates.  A fixed value pool is then re-ranked by exact
value-and-gradient signatures.  Target derivatives and benchmark metadata are
never used.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import product
from typing import Sequence

import numpy as np

from .sn_basis_archive import BasisArchiveEntry
from .sn_population_coverage import orthonormal_signature_span
from .sn_radical_composition import sobolev_positive_sqrt_signature
from .sn_shared_denominator import _signature_gain


@dataclass(frozen=True)
class SparseRadicalProposal:
    """One ordinary-GP expression retained by a value or Sobolev lane."""

    expression: str
    exponent_vector: tuple[int, ...]
    radical_offset: float
    radical_scale: float
    continuous_exponents: tuple[float, ...]
    inferred_scale: float
    log_fit_rmse: float
    fitted_coefficients: tuple[float, float]
    training_r2: float
    prediction_correlation: float
    candidate_sobolev_gain: float
    monomial_sobolev_gain: float
    radical_coupling_gain: float
    joint_score: float
    selection_lanes: tuple[str, ...]


@dataclass(frozen=True)
class SparseRadicalResult:
    """Compact audit result for sparse-radical exponent proposals."""

    proposals: tuple[SparseRadicalProposal, ...]
    log_linear_fits: int
    integer_exponent_proposals: int
    candidates_screened: int
    value_pool: int
    sobolev_candidates: int
    transform_rejections: int
    exponent_bound_rejections: int
    domain_rejections: int
    numeric_failures: int


@dataclass(frozen=True)
class _ExponentState:
    exponent_vector: tuple[int, ...]
    radical_offset: float
    continuous_exponents: tuple[float, ...]
    inferred_scale: float
    log_fit_rmse: float


@dataclass(frozen=True)
class _ValueState:
    expression: str
    exponent_state: _ExponentState
    radical_scale: float
    fitted_coefficients: tuple[float, float]
    training_r2: float
    prediction_correlation: float


@dataclass(frozen=True)
class _ScoredState:
    value_state: _ValueState
    candidate_sobolev_gain: float
    monomial_sobolev_gain: float
    radical_coupling_gain: float
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


def _library_number(value: float) -> str:
    if np.isclose(value, 0.5, rtol=0.0, atol=1e-15):
        return "(1 / 2)"
    if np.isclose(value, 1.0, rtol=0.0, atol=1e-15):
        return "1"
    if np.isclose(value, 2.0, rtol=0.0, atol=1e-15):
        return "2"
    raise ValueError(f"sparse-radical constant {value!r} is outside the GP library")


def _power_text(name: str, exponent: int) -> str:
    degree = abs(int(exponent))
    return name if degree == 1 else f"{name} ** {degree}"


def _monomial_expression(
    exponents: Sequence[int], variable_names: Sequence[str]
) -> str:
    numerator = [
        _power_text(name, exponent)
        for name, exponent in zip(variable_names, exponents, strict=True)
        if exponent > 0
    ]
    denominator = [
        _power_text(name, exponent)
        for name, exponent in zip(variable_names, exponents, strict=True)
        if exponent < 0
    ]
    numerator_text = " * ".join(numerator) or "1"
    if not denominator:
        return numerator_text
    return f"({numerator_text}) / ({' * '.join(denominator)})"


def _radical_expression(
    exponents: Sequence[int],
    variable_names: Sequence[str],
    offset: float,
    scale: float,
) -> str:
    monomial = _monomial_expression(exponents, variable_names)
    offset_text = _library_number(offset)
    scale_text = _library_number(scale)
    scaled = monomial if scale_text == "1" else f"({scale_text}) * ({monomial})"
    return f"sqrt({offset_text} + ({scaled}))"


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


def _within_exponent_bounds(
    vector: Sequence[int],
    *,
    max_abs_exponent: int,
    max_numerator_degree: int,
    max_denominator_degree: int,
) -> bool:
    return bool(
        any(vector)
        and max(abs(int(value)) for value in vector) <= max_abs_exponent
        and sum(max(int(value), 0) for value in vector) <= max_numerator_degree
        and sum(max(-int(value), 0) for value in vector) <= max_denominator_degree
    )


def _integer_neighbourhood(
    continuous: np.ndarray,
) -> tuple[tuple[int, ...], ...]:
    nearest = np.rint(continuous).astype(int)
    choices = tuple(
        tuple(
            sorted(
                {
                    int(np.floor(value)),
                    int(np.rint(value)),
                    int(np.ceil(value)),
                }
            )
        )
        for value in continuous
    )
    candidates = {tuple(int(value) for value in vector) for vector in product(*choices)}
    candidates.add(tuple(int(value) for value in nearest))
    for axis in range(nearest.size):
        for delta in (-1, 1):
            changed = nearest.copy()
            changed[axis] += delta
            candidates.add(tuple(int(value) for value in changed))
    return tuple(sorted(candidates))


def _monomial_signature(
    exponents: Sequence[int],
    direct: Sequence[BasisArchiveEntry],
    *,
    sample_count: int,
    dimension: int,
    lambda_value: float,
    lambda_gradient: float,
) -> np.ndarray:
    value_factor = float(np.sqrt(lambda_value / sample_count))
    gradient_factor = float(np.sqrt(lambda_gradient / (sample_count * dimension)))
    geometry_values = np.column_stack(
        tuple(
            np.asarray(entry.signature, dtype=float)[:sample_count] / value_factor
            for entry in direct
        )
    )
    with np.errstate(all="ignore"):
        values = np.prod(
            np.power(geometry_values, np.asarray(exponents, dtype=int)), axis=1
        )
    blocks = [value_factor * values]
    for axis, exponent in enumerate(exponents):
        with np.errstate(all="ignore"):
            gradient = float(exponent) * values / geometry_values[:, axis]
        blocks.append(gradient_factor * gradient)
    signature = np.concatenate(blocks)
    if not np.all(np.isfinite(signature)):
        raise ValueError("sparse-radical monomial signature is non-finite")
    return signature


def _value_key(state: _ValueState) -> tuple[float, float, int, str]:
    exponents = state.exponent_state.exponent_vector
    return (
        -state.training_r2,
        -state.prediction_correlation,
        sum(abs(value) for value in exponents),
        state.expression,
    )


def lift_direct_sparse_radicals(
    *,
    direct_entries: Sequence[BasisArchiveEntry],
    target: Sequence[float],
    variable_names: Sequence[str],
    geometry_sample_count: int,
    radical_offsets: Sequence[float],
    radical_scales: Sequence[float],
    max_abs_exponent: int,
    max_numerator_degree: int,
    max_denominator_degree: int,
    integer_candidate_limit: int,
    value_pool_size: int,
    shortlist_size: int,
    proposal_limit: int,
    transform_epsilon: float = 1e-12,
    lambda_value: float = 1.0,
    lambda_gradient: float = 1.0,
) -> SparseRadicalResult:
    """Propose bounded integer-exponent radicals and score exact signatures."""

    direct = tuple(direct_entries)
    names = tuple(str(value) for value in variable_names)
    y = np.asarray(target, dtype=float).reshape(-1)
    offsets = tuple(dict.fromkeys(float(value) for value in radical_offsets))
    scales = tuple(dict.fromkeys(float(value) for value in radical_scales))
    dimension = len(names)
    signature_size = geometry_sample_count * (dimension + 1)
    if (
        len(direct) != dimension
        or dimension < 1
        or y.size < 2
        or not np.all(np.isfinite(y))
        or float(np.var(y)) <= np.finfo(float).eps
        or geometry_sample_count < 1
        or not offsets
        or not scales
        or max_abs_exponent < 1
        or max_numerator_degree < 1
        or max_denominator_degree < 1
        or integer_candidate_limit < 1
        or value_pool_size < shortlist_size
        or shortlist_size < 1
        or proposal_limit < 1
        or transform_epsilon <= 0.0
        or lambda_value <= 0.0
        or lambda_gradient <= 0.0
    ):
        raise ValueError("sparse-radical inputs are invalid")
    if any(
        np.asarray(entry.values).shape != y.shape
        or np.asarray(entry.signature).shape != (signature_size,)
        or not np.all(np.isfinite(entry.values))
        or not np.all(np.isfinite(entry.signature))
        for entry in direct
    ):
        raise ValueError("sparse-radical entries are not aligned")

    X = np.column_stack(
        tuple(np.asarray(entry.values, dtype=float) for entry in direct)
    )
    if np.any(X <= transform_epsilon):
        return SparseRadicalResult(
            proposals=(),
            log_linear_fits=0,
            integer_exponent_proposals=0,
            candidates_screened=0,
            value_pool=0,
            sobolev_candidates=0,
            transform_rejections=0,
            exponent_bound_rejections=0,
            domain_rejections=1,
            numeric_failures=0,
        )
    log_X = np.log(X)
    design = np.column_stack((np.ones(y.size, dtype=float), log_X))
    exponent_states: dict[tuple[float, tuple[int, ...]], _ExponentState] = {}
    transform_rejections = 0
    exponent_bound_rejections = 0
    log_linear_fits = 0
    for offset in offsets:
        transformed = np.square(y) - offset
        if np.any(transformed <= transform_epsilon):
            transform_rejections += 1
            continue
        coefficients, *_ = np.linalg.lstsq(design, np.log(transformed), rcond=None)
        continuous = np.asarray(coefficients[1:], dtype=float)
        inferred_scale = float(np.exp(coefficients[0]))
        if not np.all(np.isfinite(continuous)) or not np.isfinite(inferred_scale):
            transform_rejections += 1
            continue
        log_linear_fits += 1
        for vector in _integer_neighbourhood(continuous):
            if not _within_exponent_bounds(
                vector,
                max_abs_exponent=max_abs_exponent,
                max_numerator_degree=max_numerator_degree,
                max_denominator_degree=max_denominator_degree,
            ):
                exponent_bound_rejections += 1
                continue
            residual_without_intercept = np.log(transformed) - log_X @ np.asarray(
                vector, dtype=float
            )
            intercept = float(np.mean(residual_without_intercept))
            residual = residual_without_intercept - intercept
            state = _ExponentState(
                exponent_vector=vector,
                radical_offset=offset,
                continuous_exponents=tuple(float(value) for value in continuous),
                inferred_scale=float(np.exp(intercept)),
                log_fit_rmse=float(np.sqrt(np.mean(np.square(residual)))),
            )
            key = (offset, vector)
            current = exponent_states.get(key)
            if current is None or state.log_fit_rmse < current.log_fit_rmse:
                exponent_states[key] = state

    ordered_exponents = sorted(
        exponent_states.values(),
        key=lambda state: (
            state.log_fit_rmse,
            sum(abs(value) for value in state.exponent_vector),
            state.exponent_vector,
            state.radical_offset,
        ),
    )[:integer_candidate_limit]
    centered_target = y - float(np.mean(y))
    target_variance = float(np.var(y))
    target_norm = float(np.linalg.norm(centered_target))
    retained: list[_ValueState] = []
    numeric_failures = 0
    candidates_screened = 0
    for exponent_state in ordered_exponents:
        exponents = np.asarray(exponent_state.exponent_vector, dtype=int)
        with np.errstate(all="ignore"):
            monomial = np.prod(np.power(X, exponents), axis=1)
        if not np.all(np.isfinite(monomial)):
            numeric_failures += len(scales)
            continue
        for scale in scales:
            candidates_screened += 1
            radicand = exponent_state.radical_offset + scale * monomial
            if np.any(radicand <= transform_epsilon):
                numeric_failures += 1
                continue
            basis = np.sqrt(radicand)
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
            fitted_coefficients, training_r2, correlation = fit
            retained.append(
                _ValueState(
                    expression=_radical_expression(
                        exponent_state.exponent_vector,
                        names,
                        exponent_state.radical_offset,
                        scale,
                    ),
                    exponent_state=exponent_state,
                    radical_scale=scale,
                    fitted_coefficients=fitted_coefficients,
                    training_r2=training_r2,
                    prediction_correlation=correlation,
                )
            )
    retained.sort(key=_value_key)
    del retained[value_pool_size:]

    constant = _constant_signature(geometry_sample_count, dimension, lambda_value)
    coordinate_span = orthonormal_signature_span(
        tuple(np.asarray(entry.signature, dtype=float) for entry in direct)
    )
    scored: list[_ScoredState] = []
    for state in retained:
        try:
            monomial_signature = _monomial_signature(
                state.exponent_state.exponent_vector,
                direct,
                sample_count=geometry_sample_count,
                dimension=dimension,
                lambda_value=lambda_value,
                lambda_gradient=lambda_gradient,
            )
            inner_signature = (
                state.exponent_state.radical_offset * constant
                + state.radical_scale * monomial_signature
            )
            candidate_signature = sobolev_positive_sqrt_signature(
                inner_signature,
                sample_count=geometry_sample_count,
                dimension=dimension,
                minimum_value=transform_epsilon,
                lambda_value=lambda_value,
                lambda_gradient=lambda_gradient,
            )
            candidate_gain = _signature_gain(candidate_signature, coordinate_span)
            monomial_gain = _signature_gain(monomial_signature, coordinate_span)
            component_span = orthonormal_signature_span((constant, monomial_signature))
            coupling_gain = _signature_gain(candidate_signature, component_span)
        except ValueError:
            numeric_failures += 1
            candidate_gain = 0.0
            monomial_gain = 0.0
            coupling_gain = 0.0
        joint_score = (
            max(state.training_r2, 0.0) * candidate_gain * monomial_gain * coupling_gain
        )
        scored.append(
            _ScoredState(
                value_state=state,
                candidate_sobolev_gain=candidate_gain,
                monomial_sobolev_gain=monomial_gain,
                radical_coupling_gain=coupling_gain,
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
            -item.radical_coupling_gain,
            -item.value_state.training_r2,
            item.value_state.expression,
        ),
    )[:shortlist_size]
    lanes: dict[int, list[str]] = {}
    for item in value_lane:
        lanes.setdefault(id(item), []).append("value_sparse_radical")
    for item in sobolev_lane:
        lanes.setdefault(id(item), []).append("sobolev_sparse_radical")
    selected: list[_ScoredState] = []
    seen_ids: set[int] = set()
    for item in (*value_lane, *sobolev_lane):
        if id(item) in seen_ids:
            continue
        seen_ids.add(id(item))
        selected.append(replace(item, selection_lanes=tuple(lanes[id(item)])))

    proposals = [
        SparseRadicalProposal(
            expression=item.value_state.expression,
            exponent_vector=item.value_state.exponent_state.exponent_vector,
            radical_offset=item.value_state.exponent_state.radical_offset,
            radical_scale=item.value_state.radical_scale,
            continuous_exponents=(item.value_state.exponent_state.continuous_exponents),
            inferred_scale=item.value_state.exponent_state.inferred_scale,
            log_fit_rmse=item.value_state.exponent_state.log_fit_rmse,
            fitted_coefficients=item.value_state.fitted_coefficients,
            training_r2=item.value_state.training_r2,
            prediction_correlation=item.value_state.prediction_correlation,
            candidate_sobolev_gain=item.candidate_sobolev_gain,
            monomial_sobolev_gain=item.monomial_sobolev_gain,
            radical_coupling_gain=item.radical_coupling_gain,
            joint_score=item.joint_score,
            selection_lanes=item.selection_lanes,
        )
        for item in selected
    ]
    proposals.sort(
        key=lambda item: (
            -item.training_r2,
            -item.joint_score,
            sum(abs(value) for value in item.exponent_vector),
            item.expression,
        )
    )
    unique: list[SparseRadicalProposal] = []
    seen_expressions: set[str] = set()
    for proposal in proposals:
        if proposal.expression in seen_expressions:
            continue
        seen_expressions.add(proposal.expression)
        unique.append(proposal)
        if len(unique) >= proposal_limit:
            break

    return SparseRadicalResult(
        proposals=tuple(unique),
        log_linear_fits=log_linear_fits,
        integer_exponent_proposals=len(ordered_exponents),
        candidates_screened=candidates_screened,
        value_pool=len(retained),
        sobolev_candidates=len(scored),
        transform_rejections=transform_rejections,
        exponent_bound_rejections=exponent_bound_rejections,
        domain_rejections=0,
        numeric_failures=numeric_failures,
    )
