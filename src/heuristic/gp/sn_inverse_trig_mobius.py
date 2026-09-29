"""Complete inverse-trigonometric Möbius proposals for Stage CL."""

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
from .sn_shared_denominator import _signature_gain


@dataclass(frozen=True)
class InverseTrigMobiusProposal:
    expression: str
    ratio_axes: tuple[int, int]
    phase_axis: int
    outer_transform: str
    inner_transform: str
    numerator_sign: int
    denominator_sign: int
    phase_scale: float
    fitted_coefficients: tuple[float, float]
    training_r2: float
    prediction_correlation: float
    candidate_sobolev_gain: float
    outer_coupling_gain: float
    mobius_argument_gain: float
    numerator_given_denominator_novelty: float
    denominator_given_numerator_novelty: float
    ratio_given_trig_novelty: float
    trig_given_ratio_novelty: float
    joint_score: float
    selection_lanes: tuple[str, ...]


@dataclass(frozen=True)
class InverseTrigMobiusResult:
    proposals: tuple[InverseTrigMobiusProposal, ...]
    candidates_screened: int
    defined_candidates: int
    value_pool: int
    sobolev_candidates: int
    ratio_rejections: int
    denominator_rejections: int
    domain_rejections: int
    numeric_failures: int
    dimension_fallback: bool


@dataclass(frozen=True)
class _ValueState:
    expression: str
    ratio_axes: tuple[int, int]
    phase_axis: int
    outer_transform: str
    inner_transform: str
    numerator_sign: int
    denominator_sign: int
    phase_scale: float
    fitted_coefficients: tuple[float, float]
    training_r2: float
    prediction_correlation: float
    selection_lanes: tuple[str, ...] = ()
    candidate_sobolev_gain: float = 0.0
    outer_coupling_gain: float = 0.0
    mobius_argument_gain: float = 0.0
    numerator_given_denominator_novelty: float = 0.0
    denominator_given_numerator_novelty: float = 0.0
    ratio_given_trig_novelty: float = 0.0
    trig_given_ratio_novelty: float = 0.0
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
    raise ValueError(f"unsupported Stage-CL phase scale {scale!r}")


def _expression(
    names: Sequence[str],
    *,
    ratio_axes: tuple[int, int],
    phase_axis: int,
    outer_transform: str,
    inner_transform: str,
    numerator_sign: int,
    denominator_sign: int,
    phase_scale: float,
) -> str:
    ratio = f"{names[ratio_axes[0]]} / {names[ratio_axes[1]]}"
    scale = _scale_text(phase_scale)
    phase = (
        names[phase_axis]
        if scale == "1"
        else f"({scale}) * ({names[phase_axis]})"
    )
    trig = f"{inner_transform}({phase})"
    numerator_operator = "+" if numerator_sign > 0 else "-"
    denominator_operator = "+" if denominator_sign > 0 else "-"
    numerator = f"{trig} {numerator_operator} ({ratio})"
    denominator = f"1 {denominator_operator} ({ratio}) * {trig}"
    return f"{outer_transform}(({numerator}) / ({denominator}))"


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


def _inverse_trig_signature(
    source: Sequence[float],
    *,
    transform: str,
    sample_count: int,
    dimension: int,
    domain_tolerance: float,
    lambda_value: float,
    lambda_gradient: float,
) -> np.ndarray:
    vector = np.asarray(source, dtype=float).reshape(-1)
    expected = sample_count * (dimension + 1)
    if (
        transform not in {"arcsin", "arccos"}
        or vector.shape != (expected,)
        or not np.all(np.isfinite(vector))
        or domain_tolerance < 0.0
    ):
        raise ValueError("inverse-trig signature inputs are invalid")
    value_factor = float(np.sqrt(lambda_value / sample_count))
    values = vector[:sample_count] / value_factor
    if np.any(np.abs(values) > 1.0 + domain_tolerance):
        raise ValueError("inverse-trig source is outside the real domain")
    clipped = np.clip(values, -1.0, 1.0)
    derivative_denominator = 1.0 - np.square(clipped)
    if np.any(derivative_denominator <= np.finfo(float).eps):
        raise ValueError("inverse-trig derivative is singular")
    if transform == "arcsin":
        transformed = np.arcsin(clipped)
        multiplier = 1.0 / np.sqrt(derivative_denominator)
    else:
        transformed = np.arccos(clipped)
        multiplier = -1.0 / np.sqrt(derivative_denominator)
    blocks = [value_factor * transformed]
    gradient_factor = float(
        np.sqrt(lambda_gradient / (sample_count * dimension))
    )
    for axis in range(dimension):
        start = sample_count * (axis + 1)
        gradient = vector[start : start + sample_count] / gradient_factor
        blocks.append(gradient_factor * multiplier * gradient)
    output = np.concatenate(blocks)
    if not np.all(np.isfinite(output)):
        raise ValueError("inverse-trig signature is non-finite")
    return output


def _value_key(state: _ValueState) -> tuple[float, float, str]:
    return (
        -state.training_r2,
        -state.prediction_correlation,
        state.expression,
    )


def lift_direct_inverse_trig_mobius(
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
    domain_tolerance: float = 1e-12,
    lambda_value: float = 1.0,
    lambda_gradient: float = 1.0,
) -> InverseTrigMobiusResult:
    """Form all 480 three-coordinate tuples before dual-lane screening."""

    direct = tuple(direct_entries)
    names = tuple(str(value) for value in variable_names)
    y = np.asarray(target, dtype=float).reshape(-1)
    dimension = len(names)
    if dimension != 3:
        return InverseTrigMobiusResult((), 0, 0, 0, 0, 0, 0, 0, 0, True)
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
        or not phase_scale_values
        or value_pool_size < shortlist_size
        or shortlist_size < 1
        or proposal_limit < 1
        or protected_epsilon <= 0.0
        or domain_tolerance < 0.0
        or lambda_value <= 0.0
        or lambda_gradient <= 0.0
    ):
        raise ValueError("inverse-trig Möbius inputs are invalid")
    for scale in phase_scale_values:
        _scale_text(scale)
    if any(
        np.asarray(entry.values).shape != y.shape
        or np.asarray(entry.signature).shape != (signature_size,)
        or not np.all(np.isfinite(entry.values))
        or not np.all(np.isfinite(entry.signature))
        for entry in direct
    ):
        raise ValueError("inverse-trig Möbius entries are not aligned")

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
    ratio_rejections = 0
    denominator_rejections = 0
    domain_rejections = 0
    numeric_failures = 0
    combinations_per_ratio = 16 * len(phase_scale_values)
    axes = tuple(range(dimension))
    for numerator_axis in axes:
        for denominator_axis in axes:
            if numerator_axis == denominator_axis:
                continue
            phase_axis = next(
                axis
                for axis in axes
                if axis not in {numerator_axis, denominator_axis}
            )
            candidates_screened += combinations_per_ratio
            if (
                np.any(
                    np.abs(X[:, denominator_axis]) <= protected_epsilon
                )
                or np.any(
                    np.abs(geometry_X[:, denominator_axis])
                    <= protected_epsilon
                )
            ):
                ratio_rejections += combinations_per_ratio
                continue
            ratio = X[:, numerator_axis] / X[:, denominator_axis]
            geometry_ratio = (
                geometry_X[:, numerator_axis]
                / geometry_X[:, denominator_axis]
            )
            for outer_transform in ("arccos", "arcsin"):
                for inner_transform in ("cos", "sin"):
                    for numerator_sign in (-1, 1):
                        for denominator_sign in (-1, 1):
                            for phase_scale in phase_scale_values:
                                if inner_transform == "cos":
                                    trig = np.cos(
                                        phase_scale * X[:, phase_axis]
                                    )
                                    geometry_trig = np.cos(
                                        phase_scale
                                        * geometry_X[:, phase_axis]
                                    )
                                else:
                                    trig = np.sin(
                                        phase_scale * X[:, phase_axis]
                                    )
                                    geometry_trig = np.sin(
                                        phase_scale
                                        * geometry_X[:, phase_axis]
                                    )
                                numerator = trig + numerator_sign * ratio
                                denominator = (
                                    1.0
                                    + denominator_sign * ratio * trig
                                )
                                geometry_numerator = (
                                    geometry_trig
                                    + numerator_sign * geometry_ratio
                                )
                                geometry_denominator = (
                                    1.0
                                    + denominator_sign
                                    * geometry_ratio
                                    * geometry_trig
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
                                argument = numerator / denominator
                                geometry_argument = (
                                    geometry_numerator / geometry_denominator
                                )
                                if (
                                    np.any(
                                        np.abs(argument)
                                        > 1.0 + domain_tolerance
                                    )
                                    or np.any(
                                        np.abs(geometry_argument)
                                        > 1.0 + domain_tolerance
                                    )
                                ):
                                    domain_rejections += 1
                                    continue
                                clipped = np.clip(argument, -1.0, 1.0)
                                with np.errstate(all="ignore"):
                                    basis = (
                                        np.arccos(clipped)
                                        if outer_transform == "arccos"
                                        else np.arcsin(clipped)
                                    )
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
                                            ratio_axes=(
                                                numerator_axis,
                                                denominator_axis,
                                            ),
                                            phase_axis=phase_axis,
                                            outer_transform=outer_transform,
                                            inner_transform=inner_transform,
                                            numerator_sign=numerator_sign,
                                            denominator_sign=denominator_sign,
                                            phase_scale=phase_scale,
                                        ),
                                        ratio_axes=(
                                            numerator_axis,
                                            denominator_axis,
                                        ),
                                        phase_axis=phase_axis,
                                        outer_transform=outer_transform,
                                        inner_transform=inner_transform,
                                        numerator_sign=numerator_sign,
                                        denominator_sign=denominator_sign,
                                        phase_scale=phase_scale,
                                        fitted_coefficients=coefficients,
                                        training_r2=r2,
                                        prediction_correlation=correlation,
                                    )
                                )
    if candidates_screened != 6 * 16 * len(phase_scale_values):
        raise RuntimeError("Stage-CL complete tuple count drifted")
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
        try:
            ratio_signature = sobolev_division_signature(
                direct[state.ratio_axes[0]].signature,
                direct[state.ratio_axes[1]].signature,
                **division_keyword,
            )
            phase_source = (
                state.phase_scale
                * np.asarray(direct[state.phase_axis].signature, dtype=float)
            )
            trig_signature = sobolev_unary_signature(
                phase_source,
                transform=state.inner_transform,
                **keyword,
            )
            numerator_signature = (
                trig_signature + state.numerator_sign * ratio_signature
            )
            product_signature = sobolev_product_signature(
                ratio_signature, trig_signature, **keyword
            )
            denominator_signature = (
                constant
                + state.denominator_sign * product_signature
            )
            argument_signature = sobolev_division_signature(
                numerator_signature,
                denominator_signature,
                **division_keyword,
            )
            outer_signature = _inverse_trig_signature(
                argument_signature,
                transform=state.outer_transform,
                domain_tolerance=domain_tolerance,
                **keyword,
            )
            intercept, slope = state.fitted_coefficients
            candidate_signature = intercept * constant + slope * outer_signature
            candidate_gain = _signature_gain(
                candidate_signature, coordinate_span
            )
            outer_coupling = _signature_gain(
                outer_signature,
                orthonormal_signature_span((argument_signature,)),
            )
            argument_gain = _signature_gain(
                argument_signature, coordinate_span
            )
            numerator_given_denominator = _signature_gain(
                numerator_signature,
                orthonormal_signature_span((denominator_signature,)),
            )
            denominator_given_numerator = _signature_gain(
                denominator_signature,
                orthonormal_signature_span((numerator_signature,)),
            )
            ratio_given_trig = _signature_gain(
                ratio_signature,
                orthonormal_signature_span((trig_signature,)),
            )
            trig_given_ratio = _signature_gain(
                trig_signature,
                orthonormal_signature_span((ratio_signature,)),
            )
        except ValueError:
            numeric_failures += 1
            candidate_gain = 0.0
            outer_coupling = 0.0
            argument_gain = 0.0
            numerator_given_denominator = 0.0
            denominator_given_numerator = 0.0
            ratio_given_trig = 0.0
            trig_given_ratio = 0.0
        joint_score = float(
            np.mean(
                (
                    min(ratio_given_trig, trig_given_ratio),
                    min(
                        numerator_given_denominator,
                        denominator_given_numerator,
                    ),
                    outer_coupling,
                )
            )
        )
        scored.append(
            replace(
                state,
                candidate_sobolev_gain=candidate_gain,
                outer_coupling_gain=outer_coupling,
                mobius_argument_gain=argument_gain,
                numerator_given_denominator_novelty=(
                    numerator_given_denominator
                ),
                denominator_given_numerator_novelty=(
                    denominator_given_numerator
                ),
                ratio_given_trig_novelty=ratio_given_trig,
                trig_given_ratio_novelty=trig_given_ratio,
                joint_score=joint_score,
            )
        )

    value_lane = sorted(scored, key=_value_key)[:shortlist_size]
    sobolev_lane = sorted(
        scored,
        key=lambda state: (
            -state.joint_score,
            -min(
                state.ratio_given_trig_novelty,
                state.trig_given_ratio_novelty,
            ),
            -state.training_r2,
            state.expression,
        ),
    )[:shortlist_size]
    lane_names: dict[int, list[str]] = {}
    for state in value_lane:
        lane_names.setdefault(id(state), []).append("value_inverse_trig_mobius")
    for state in sobolev_lane:
        lane_names.setdefault(id(state), []).append("sobolev_inverse_trig_mobius")
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
        InverseTrigMobiusProposal(
            expression=state.expression,
            ratio_axes=state.ratio_axes,
            phase_axis=state.phase_axis,
            outer_transform=state.outer_transform,
            inner_transform=state.inner_transform,
            numerator_sign=state.numerator_sign,
            denominator_sign=state.denominator_sign,
            phase_scale=state.phase_scale,
            fitted_coefficients=state.fitted_coefficients,
            training_r2=state.training_r2,
            prediction_correlation=state.prediction_correlation,
            candidate_sobolev_gain=state.candidate_sobolev_gain,
            outer_coupling_gain=state.outer_coupling_gain,
            mobius_argument_gain=state.mobius_argument_gain,
            numerator_given_denominator_novelty=(
                state.numerator_given_denominator_novelty
            ),
            denominator_given_numerator_novelty=(
                state.denominator_given_numerator_novelty
            ),
            ratio_given_trig_novelty=state.ratio_given_trig_novelty,
            trig_given_ratio_novelty=state.trig_given_ratio_novelty,
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
    unique: list[InverseTrigMobiusProposal] = []
    seen_expressions: set[str] = set()
    for proposal in proposals:
        if proposal.expression in seen_expressions:
            continue
        seen_expressions.add(proposal.expression)
        unique.append(proposal)
        if len(unique) >= proposal_limit:
            break
    return InverseTrigMobiusResult(
        proposals=tuple(unique),
        candidates_screened=candidates_screened,
        defined_candidates=defined_candidates,
        value_pool=len(retained),
        sobolev_candidates=len(scored),
        ratio_rejections=ratio_rejections,
        denominator_rejections=denominator_rejections,
        domain_rejections=domain_rejections,
        numeric_failures=numeric_failures,
        dimension_fallback=False,
    )
