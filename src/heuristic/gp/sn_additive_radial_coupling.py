"""Complete additive radial-coupling proposals for Stage CK."""

from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import permutations
from typing import Sequence

import numpy as np

from .sn_archive_interactions import sobolev_product_signature
from .sn_basis_archive import BasisArchiveEntry
from .sn_population_coverage import orthonormal_signature_span
from .sn_radical_composition import sobolev_positive_sqrt_signature
from .sn_shared_denominator import _signature_gain


@dataclass(frozen=True)
class AdditiveRadialCouplingProposal:
    expression: str
    role_axes: tuple[int, int, int, int, int, int]
    affine_sign: int
    fitted_coefficients: tuple[float, float, float]
    training_r2: float
    complexity_proxy: int
    candidate_sobolev_gain: float
    first_given_second_novelty: float
    second_given_first_novelty: float
    sqrt_coupling_gain: float
    radical_given_additive_novelty: float
    additive_given_radical_novelty: float
    joint_score: float
    selection_lanes: tuple[str, ...]


@dataclass(frozen=True)
class AdditiveRadialCouplingResult:
    proposals: tuple[AdditiveRadialCouplingProposal, ...]
    candidates_screened: int
    defined_candidates: int
    value_pool: int
    sobolev_candidates: int
    radical_rejections: int
    numeric_failures: int
    dimension_fallback: bool


@dataclass(frozen=True)
class _ValueState:
    expression: str
    role_axes: tuple[int, int, int, int, int, int]
    affine_sign: int
    fitted_coefficients: tuple[float, float, float]
    training_r2: float
    complexity_proxy: int
    selection_lanes: tuple[str, ...] = ()
    candidate_sobolev_gain: float = 0.0
    first_given_second_novelty: float = 0.0
    second_given_first_novelty: float = 0.0
    sqrt_coupling_gain: float = 0.0
    radical_given_additive_novelty: float = 0.0
    additive_given_radical_novelty: float = 0.0
    joint_score: float = 0.0


def _constant_signature(
    sample_count: int,
    dimension: int,
    lambda_value: float,
) -> np.ndarray:
    output = np.zeros(sample_count * (dimension + 1), dtype=float)
    output[:sample_count] = np.sqrt(lambda_value / sample_count)
    return output


def _expression(
    names: Sequence[str],
    role_axes: tuple[int, int, int, int, int, int],
    affine_sign: int,
) -> str:
    xa, xb, xc, xd, xe, xf = (names[index] for index in role_axes)
    operator = "+" if affine_sign > 0 else "-"
    first = f"({xa} {operator} {xb} * {xc}) ** 2 * {xd} ** 2"
    second = f"{xe} ** 2 * {xd} ** 4"
    return f"sqrt(({first}) + ({second})) + {xb} * {xf}"


def _value_key(state: _ValueState) -> tuple[float, int, str]:
    return (-state.training_r2, state.complexity_proxy, state.expression)


def lift_direct_additive_radial_couplings(
    *,
    direct_entries: Sequence[BasisArchiveEntry],
    target: Sequence[float],
    variable_names: Sequence[str],
    geometry_sample_count: int,
    value_pool_size: int,
    shortlist_size: int,
    proposal_limit: int,
    radicand_epsilon: float = 1e-12,
    lambda_value: float = 1.0,
    lambda_gradient: float = 1.0,
) -> AdditiveRadialCouplingResult:
    """Form all 1,440 six-coordinate tuples before dual-lane screening."""

    direct = tuple(direct_entries)
    names = tuple(str(value) for value in variable_names)
    y = np.asarray(target, dtype=float).reshape(-1)
    dimension = len(names)
    if dimension != 6:
        return AdditiveRadialCouplingResult((), 0, 0, 0, 0, 0, 0, True)
    signature_size = geometry_sample_count * (dimension + 1)
    if (
        len(direct) != dimension
        or y.size < 2
        or not np.all(np.isfinite(y))
        or float(np.var(y)) <= np.finfo(float).eps
        or geometry_sample_count < 1
        or value_pool_size < shortlist_size
        or shortlist_size < 1
        or proposal_limit < 1
        or radicand_epsilon <= 0.0
        or lambda_value <= 0.0
        or lambda_gradient <= 0.0
    ):
        raise ValueError("additive radial-coupling inputs are invalid")
    if any(
        np.asarray(entry.values).shape != y.shape
        or np.asarray(entry.signature).shape != (signature_size,)
        or not np.all(np.isfinite(entry.values))
        or not np.all(np.isfinite(entry.signature))
        for entry in direct
    ):
        raise ValueError("additive radial-coupling entries are not aligned")

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
    target_variance = float(np.var(y))
    retained: list[_ValueState] = []
    candidates_screened = 0
    defined_candidates = 0
    radical_rejections = 0
    numeric_failures = 0
    for role_axes in permutations(range(dimension)):
        xa_axis, xb_axis, xc_axis, xd_axis, xe_axis, xf_axis = role_axes
        for affine_sign in (-1, 1):
            candidates_screened += 1
            affine = (
                X[:, xa_axis]
                + affine_sign * X[:, xb_axis] * X[:, xc_axis]
            )
            geometry_affine = (
                geometry_X[:, xa_axis]
                + affine_sign
                * geometry_X[:, xb_axis]
                * geometry_X[:, xc_axis]
            )
            first = np.square(affine) * np.square(X[:, xd_axis])
            second = np.square(X[:, xe_axis]) * np.power(X[:, xd_axis], 4)
            geometry_first = np.square(geometry_affine) * np.square(
                geometry_X[:, xd_axis]
            )
            geometry_second = np.square(
                geometry_X[:, xe_axis]
            ) * np.power(geometry_X[:, xd_axis], 4)
            radicand = first + second
            geometry_radicand = geometry_first + geometry_second
            if (
                np.any(radicand < 0.0)
                or np.any(geometry_radicand < radicand_epsilon)
            ):
                radical_rejections += 1
                continue
            with np.errstate(all="ignore"):
                radical = np.sqrt(radicand)
                additive = X[:, xb_axis] * X[:, xf_axis]
            if (
                not np.all(np.isfinite(radical))
                or not np.all(np.isfinite(additive))
            ):
                numeric_failures += 1
                continue
            defined_candidates += 1
            design = np.column_stack(
                (np.ones(len(y), dtype=float), radical, additive)
            )
            coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
            rounded = np.round(coefficients, 6)
            prediction = design @ rounded
            training_r2 = float(
                1.0
                - np.mean(np.square(prediction - y)) / target_variance
            )
            if not np.isfinite(training_r2):
                numeric_failures += 1
                continue
            expression = _expression(names, role_axes, affine_sign)
            retained.append(
                _ValueState(
                    expression=expression,
                    role_axes=role_axes,
                    affine_sign=affine_sign,
                    fitted_coefficients=tuple(float(value) for value in rounded),
                    training_r2=training_r2,
                    complexity_proxy=len(expression),
                )
            )
    if candidates_screened != 1440:
        raise RuntimeError("Stage-CK complete tuple count drifted")
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
    scored: list[_ValueState] = []
    for state in retained:
        xa_axis, xb_axis, xc_axis, xd_axis, xe_axis, xf_axis = state.role_axes
        try:
            xa = np.asarray(direct[xa_axis].signature, dtype=float)
            xb = np.asarray(direct[xb_axis].signature, dtype=float)
            xc = np.asarray(direct[xc_axis].signature, dtype=float)
            xd = np.asarray(direct[xd_axis].signature, dtype=float)
            xe = np.asarray(direct[xe_axis].signature, dtype=float)
            xf = np.asarray(direct[xf_axis].signature, dtype=float)
            xb_xc = sobolev_product_signature(xb, xc, **keyword)
            affine = xa + state.affine_sign * xb_xc
            affine_square = sobolev_product_signature(
                affine, affine, **keyword
            )
            xd_square = sobolev_product_signature(xd, xd, **keyword)
            first = sobolev_product_signature(
                affine_square, xd_square, **keyword
            )
            xe_square = sobolev_product_signature(xe, xe, **keyword)
            xd_fourth = sobolev_product_signature(
                xd_square, xd_square, **keyword
            )
            second = sobolev_product_signature(
                xe_square, xd_fourth, **keyword
            )
            radicand = first + second
            radical = sobolev_positive_sqrt_signature(
                radicand,
                minimum_value=radicand_epsilon,
                **keyword,
            )
            additive = sobolev_product_signature(xb, xf, **keyword)
            intercept, radical_coefficient, additive_coefficient = (
                state.fitted_coefficients
            )
            candidate = (
                intercept * constant
                + radical_coefficient * radical
                + additive_coefficient * additive
            )
            candidate_gain = _signature_gain(candidate, coordinate_span)
            first_given_second = _signature_gain(
                first, orthonormal_signature_span((second,))
            )
            second_given_first = _signature_gain(
                second, orthonormal_signature_span((first,))
            )
            sqrt_coupling = _signature_gain(
                radical, orthonormal_signature_span((radicand,))
            )
            radical_given_additive = _signature_gain(
                radical, orthonormal_signature_span((additive,))
            )
            additive_given_radical = _signature_gain(
                additive, orthonormal_signature_span((radical,))
            )
        except ValueError:
            numeric_failures += 1
            candidate_gain = 0.0
            first_given_second = 0.0
            second_given_first = 0.0
            sqrt_coupling = 0.0
            radical_given_additive = 0.0
            additive_given_radical = 0.0
        joint_score = float(
            np.mean(
                (
                    min(first_given_second, second_given_first),
                    min(radical_given_additive, additive_given_radical),
                    sqrt_coupling,
                )
            )
        )
        scored.append(
            replace(
                state,
                candidate_sobolev_gain=candidate_gain,
                first_given_second_novelty=first_given_second,
                second_given_first_novelty=second_given_first,
                sqrt_coupling_gain=sqrt_coupling,
                radical_given_additive_novelty=radical_given_additive,
                additive_given_radical_novelty=additive_given_radical,
                joint_score=joint_score,
            )
        )

    value_lane = sorted(scored, key=_value_key)[:shortlist_size]
    sobolev_lane = sorted(
        scored,
        key=lambda state: (
            -state.joint_score,
            -min(
                state.first_given_second_novelty,
                state.second_given_first_novelty,
            ),
            -state.training_r2,
            state.expression,
        ),
    )[:shortlist_size]
    lane_names: dict[int, list[str]] = {}
    for state in value_lane:
        lane_names.setdefault(id(state), []).append(
            "value_additive_radial_coupling"
        )
    for state in sobolev_lane:
        lane_names.setdefault(id(state), []).append(
            "sobolev_additive_radial_coupling"
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
        AdditiveRadialCouplingProposal(
            expression=state.expression,
            role_axes=state.role_axes,
            affine_sign=state.affine_sign,
            fitted_coefficients=state.fitted_coefficients,
            training_r2=state.training_r2,
            complexity_proxy=state.complexity_proxy,
            candidate_sobolev_gain=state.candidate_sobolev_gain,
            first_given_second_novelty=state.first_given_second_novelty,
            second_given_first_novelty=state.second_given_first_novelty,
            sqrt_coupling_gain=state.sqrt_coupling_gain,
            radical_given_additive_novelty=(
                state.radical_given_additive_novelty
            ),
            additive_given_radical_novelty=(
                state.additive_given_radical_novelty
            ),
            joint_score=state.joint_score,
            selection_lanes=state.selection_lanes,
        )
        for state in selected
    ]
    proposals.sort(
        key=lambda proposal: (
            -proposal.training_r2,
            proposal.complexity_proxy,
            proposal.expression,
        )
    )
    unique: list[AdditiveRadialCouplingProposal] = []
    seen_expressions: set[str] = set()
    for proposal in proposals:
        if proposal.expression in seen_expressions:
            continue
        seen_expressions.add(proposal.expression)
        unique.append(proposal)
        if len(unique) >= proposal_limit:
            break
    return AdditiveRadialCouplingResult(
        proposals=tuple(unique),
        candidates_screened=candidates_screened,
        defined_candidates=defined_candidates,
        value_pool=len(retained),
        sobolev_candidates=len(scored),
        radical_rejections=radical_rejections,
        numeric_failures=numeric_failures,
        dimension_fallback=False,
    )
