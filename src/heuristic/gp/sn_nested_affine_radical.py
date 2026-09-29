"""Complete nested affine-radical proposals for Stage CJ."""

from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import permutations
from typing import Sequence

import numpy as np

from .sn_archive_interactions import sobolev_product_signature
from .sn_basis_archive import BasisArchiveEntry
from .sn_direct_composition import sobolev_division_signature
from .sn_population_coverage import orthonormal_signature_span
from .sn_radical_composition import sobolev_positive_sqrt_signature
from .sn_shared_denominator import _signature_gain


@dataclass(frozen=True)
class NestedAffineRadicalProposal:
    expression: str
    role_axes: tuple[int, int, int, int, int]
    first_sign: int
    second_sign: int
    outer_scale: float
    inner_scale: float
    fitted_coefficients: tuple[float, float]
    training_r2: float
    prediction_correlation: float
    candidate_sobolev_gain: float
    sqrt_coupling_gain: float
    product_coupling_gain: float
    inner_sum_gain: float
    inner_term_novelties: tuple[float, float, float]
    joint_score: float
    selection_lanes: tuple[str, ...]


@dataclass(frozen=True)
class NestedAffineRadicalResult:
    proposals: tuple[NestedAffineRadicalProposal, ...]
    candidates_screened: int
    defined_candidates: int
    value_pool: int
    sobolev_candidates: int
    denominator_rejections: int
    radical_rejections: int
    numeric_failures: int
    dimension_fallback: bool


@dataclass(frozen=True)
class _ValueState:
    expression: str
    role_axes: tuple[int, int, int, int, int]
    first_sign: int
    second_sign: int
    outer_scale: float
    inner_scale: float
    fitted_coefficients: tuple[float, float]
    training_r2: float
    prediction_correlation: float
    selection_lanes: tuple[str, ...] = ()
    candidate_sobolev_gain: float = 0.0
    sqrt_coupling_gain: float = 0.0
    product_coupling_gain: float = 0.0
    inner_sum_gain: float = 0.0
    inner_term_novelties: tuple[float, float, float] = (0.0, 0.0, 0.0)
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
    raise ValueError(f"unsupported Stage-CJ scale {scale!r}")


def _expression(
    names: Sequence[str],
    *,
    role_axes: tuple[int, int, int, int, int],
    first_sign: int,
    second_sign: int,
    outer_scale: float,
    inner_scale: float,
) -> str:
    xa, xb, xc, xd, xe = (names[index] for index in role_axes)
    outer = f"{_scale_text(outer_scale)} / {xa}"
    first_operator = "+" if first_sign > 0 else "-"
    second_operator = "+" if second_sign > 0 else "-"
    inner_denominator = f"{_scale_text(inner_scale)} * {xa} * {xe} ** 2"
    inner = (
        f"{xb} {first_operator} {xc} {second_operator} "
        f"{xd} ** 2 / ({inner_denominator})"
    )
    return f"sqrt(({outer}) * ({inner}))"


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


def lift_direct_nested_affine_radicals(
    *,
    direct_entries: Sequence[BasisArchiveEntry],
    target: Sequence[float],
    variable_names: Sequence[str],
    geometry_sample_count: int,
    outer_scales: Sequence[float],
    inner_scales: Sequence[float],
    value_pool_size: int,
    shortlist_size: int,
    proposal_limit: int,
    protected_epsilon: float = 1e-6,
    lambda_value: float = 1.0,
    lambda_gradient: float = 1.0,
) -> NestedAffineRadicalResult:
    """Form all 4,320 five-coordinate tuples before dual-lane screening."""

    direct = tuple(direct_entries)
    names = tuple(str(value) for value in variable_names)
    y = np.asarray(target, dtype=float).reshape(-1)
    dimension = len(names)
    if dimension != 5:
        return NestedAffineRadicalResult((), 0, 0, 0, 0, 0, 0, 0, True)
    outer_scale_values = tuple(
        dict.fromkeys(float(value) for value in outer_scales)
    )
    inner_scale_values = tuple(
        dict.fromkeys(float(value) for value in inner_scales)
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
        or value_pool_size < shortlist_size
        or shortlist_size < 1
        or proposal_limit < 1
        or protected_epsilon <= 0.0
        or lambda_value <= 0.0
        or lambda_gradient <= 0.0
    ):
        raise ValueError("nested affine-radical inputs are invalid")
    for scale in (*outer_scale_values, *inner_scale_values):
        _scale_text(scale)
    if any(
        np.asarray(entry.values).shape != y.shape
        or np.asarray(entry.signature).shape != (signature_size,)
        or not np.all(np.isfinite(entry.values))
        or not np.all(np.isfinite(entry.signature))
        for entry in direct
    ):
        raise ValueError("nested affine-radical entries are not aligned")

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
    centered_target = y - float(np.mean(y))
    target_variance = float(np.var(y))
    target_norm = float(np.linalg.norm(centered_target))
    retained: list[_ValueState] = []
    candidates_screened = 0
    defined_candidates = 0
    denominator_rejections = 0
    radical_rejections = 0
    numeric_failures = 0
    combinations_per_role = 4 * len(outer_scale_values) * len(inner_scale_values)

    for role_axes in permutations(range(dimension)):
        xa_axis, xb_axis, xc_axis, xd_axis, xe_axis = role_axes
        xa = X[:, xa_axis]
        xe = X[:, xe_axis]
        geometry_xa = geometry_X[:, xa_axis]
        geometry_xe = geometry_X[:, xe_axis]
        candidates_screened += combinations_per_role
        if (
            np.any(np.abs(xa) <= protected_epsilon)
            or np.any(np.abs(xe) <= protected_epsilon)
            or np.any(np.abs(geometry_xa) <= protected_epsilon)
            or np.any(np.abs(geometry_xe) <= protected_epsilon)
        ):
            denominator_rejections += combinations_per_role
            continue
        for first_sign in (-1, 1):
            for second_sign in (-1, 1):
                for outer_scale in outer_scale_values:
                    outer = outer_scale / xa
                    geometry_outer = outer_scale / geometry_xa
                    for inner_scale in inner_scale_values:
                        inner_denominator = inner_scale * xa * np.square(xe)
                        geometry_inner_denominator = (
                            inner_scale
                            * geometry_xa
                            * np.square(geometry_xe)
                        )
                        if (
                            np.any(
                                np.abs(inner_denominator)
                                <= protected_epsilon
                            )
                            or np.any(
                                np.abs(geometry_inner_denominator)
                                <= protected_epsilon
                            )
                        ):
                            denominator_rejections += 1
                            continue
                        inner = (
                            X[:, xb_axis]
                            + first_sign * X[:, xc_axis]
                            + second_sign
                            * np.square(X[:, xd_axis])
                            / inner_denominator
                        )
                        geometry_inner = (
                            geometry_X[:, xb_axis]
                            + first_sign * geometry_X[:, xc_axis]
                            + second_sign
                            * np.square(geometry_X[:, xd_axis])
                            / geometry_inner_denominator
                        )
                        radicand = outer * inner
                        geometry_radicand = geometry_outer * geometry_inner
                        if (
                            np.any(radicand < 0.0)
                            or np.any(geometry_radicand < 0.0)
                        ):
                            radical_rejections += 1
                            continue
                        with np.errstate(all="ignore"):
                            basis = np.sqrt(radicand)
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
                                    role_axes=role_axes,
                                    first_sign=first_sign,
                                    second_sign=second_sign,
                                    outer_scale=outer_scale,
                                    inner_scale=inner_scale,
                                ),
                                role_axes=role_axes,
                                first_sign=first_sign,
                                second_sign=second_sign,
                                outer_scale=outer_scale,
                                inner_scale=inner_scale,
                                fitted_coefficients=coefficients,
                                training_r2=r2,
                                prediction_correlation=correlation,
                            )
                        )
    expected = (
        120 * 4 * len(outer_scale_values) * len(inner_scale_values)
    )
    if candidates_screened != expected:
        raise RuntimeError("Stage-CJ complete tuple count drifted")
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
    scored: list[_ValueState] = []
    for state in retained:
        xa_axis, xb_axis, xc_axis, xd_axis, xe_axis = state.role_axes
        try:
            xa_signature = np.asarray(direct[xa_axis].signature, dtype=float)
            xd_signature = np.asarray(direct[xd_axis].signature, dtype=float)
            xe_signature = np.asarray(direct[xe_axis].signature, dtype=float)
            outer_signature = sobolev_division_signature(
                state.outer_scale * constant,
                xa_signature,
                **division_keyword,
            )
            xd_square = sobolev_product_signature(
                xd_signature, xd_signature, **keyword
            )
            xe_square = sobolev_product_signature(
                xe_signature, xe_signature, **keyword
            )
            inner_denominator = sobolev_product_signature(
                xa_signature, xe_square, **keyword
            )
            inner_denominator = state.inner_scale * inner_denominator
            rational_square = sobolev_division_signature(
                xd_square, inner_denominator, **division_keyword
            )
            terms = (
                np.asarray(direct[xb_axis].signature, dtype=float),
                state.first_sign
                * np.asarray(direct[xc_axis].signature, dtype=float),
                state.second_sign * rational_square,
            )
            inner_sum = terms[0] + terms[1] + terms[2]
            radicand_signature = sobolev_product_signature(
                outer_signature, inner_sum, **keyword
            )
            radical_signature = sobolev_positive_sqrt_signature(
                radicand_signature,
                minimum_value=np.finfo(float).eps,
                **keyword,
            )
            intercept, slope = state.fitted_coefficients
            candidate_signature = (
                intercept * constant + slope * radical_signature
            )
            candidate_gain = _signature_gain(
                candidate_signature, coordinate_span
            )
            sqrt_coupling = _signature_gain(
                radical_signature,
                orthonormal_signature_span((radicand_signature,)),
            )
            product_coupling = _signature_gain(
                radicand_signature,
                orthonormal_signature_span((outer_signature, inner_sum)),
            )
            inner_sum_gain = _signature_gain(inner_sum, coordinate_span)
            term_novelties = tuple(
                _signature_gain(
                    term,
                    orthonormal_signature_span(
                        tuple(other for j, other in enumerate(terms) if j != i)
                    ),
                )
                for i, term in enumerate(terms)
            )
        except ValueError:
            numeric_failures += 1
            candidate_gain = 0.0
            sqrt_coupling = 0.0
            product_coupling = 0.0
            inner_sum_gain = 0.0
            term_novelties = (0.0, 0.0, 0.0)
        joint_score = float(
            np.mean(
                (
                    max(term_novelties),
                    product_coupling,
                    sqrt_coupling,
                )
            )
        )
        scored.append(
            replace(
                state,
                candidate_sobolev_gain=candidate_gain,
                sqrt_coupling_gain=sqrt_coupling,
                product_coupling_gain=product_coupling,
                inner_sum_gain=inner_sum_gain,
                inner_term_novelties=term_novelties,
                joint_score=joint_score,
            )
        )

    value_lane = sorted(scored, key=_value_key)[:shortlist_size]
    sobolev_lane = sorted(
        scored,
        key=lambda state: (
            -state.joint_score,
            -max(state.inner_term_novelties),
            -state.training_r2,
            state.expression,
        ),
    )[:shortlist_size]
    lane_names: dict[int, list[str]] = {}
    for state in value_lane:
        lane_names.setdefault(id(state), []).append("value_nested_affine_radical")
    for state in sobolev_lane:
        lane_names.setdefault(id(state), []).append("sobolev_nested_affine_radical")
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
        NestedAffineRadicalProposal(
            expression=state.expression,
            role_axes=state.role_axes,
            first_sign=state.first_sign,
            second_sign=state.second_sign,
            outer_scale=state.outer_scale,
            inner_scale=state.inner_scale,
            fitted_coefficients=state.fitted_coefficients,
            training_r2=state.training_r2,
            prediction_correlation=state.prediction_correlation,
            candidate_sobolev_gain=state.candidate_sobolev_gain,
            sqrt_coupling_gain=state.sqrt_coupling_gain,
            product_coupling_gain=state.product_coupling_gain,
            inner_sum_gain=state.inner_sum_gain,
            inner_term_novelties=state.inner_term_novelties,
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
    unique: list[NestedAffineRadicalProposal] = []
    seen_expressions: set[str] = set()
    for proposal in proposals:
        if proposal.expression in seen_expressions:
            continue
        seen_expressions.add(proposal.expression)
        unique.append(proposal)
        if len(unique) >= proposal_limit:
            break
    return NestedAffineRadicalResult(
        proposals=tuple(unique),
        candidates_screened=candidates_screened,
        defined_candidates=defined_candidates,
        value_pool=len(retained),
        sobolev_candidates=len(scored),
        denominator_rejections=denominator_rejections,
        radical_rejections=radical_rejections,
        numeric_failures=numeric_failures,
        dimension_fallback=False,
    )
