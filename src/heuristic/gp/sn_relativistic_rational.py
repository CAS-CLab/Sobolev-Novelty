"""Pair-first bounded rational construction in Sobolev geometry.

The export-side plug-in enumerates compact expressions of the form

``(a + b) / (1 + sign * scale * a * b / c**2)``.

The two numerator terms are fitted independently by the ordinary additive
evaluator.  Complete rational candidates are value-screened before the
Sobolev lane is formed.  All derivatives are symbolic consequences of the
candidate; target derivatives and benchmark metadata are never used.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Sequence

import numpy as np

from .sn_archive_interactions import sobolev_product_signature
from .sn_basis_archive import BasisArchiveEntry
from .sn_population_coverage import orthonormal_signature_span
from .sn_shared_denominator import (
    _signature_gain,
    _term_complementarity,
    sobolev_quotient_signature,
)


@dataclass(frozen=True)
class RelativisticRationalProposal:
    """One ordinary-GP expression retained by a value or Sobolev lane."""

    expression: str
    left_expression: str
    right_expression: str
    scale_expression: str
    denominator_sign: int
    denominator_scale: float
    fitted_coefficients: tuple[float, float, float]
    training_r2: float
    prediction_correlation: float
    candidate_sobolev_gain: float
    term_complementarity: float
    joint_score: float
    selection_lanes: tuple[str, ...]


@dataclass(frozen=True)
class RelativisticRationalResult:
    """Compact audit result for one bounded-rational enumeration."""

    proposals: tuple[RelativisticRationalProposal, ...]
    variable_triples: int
    candidates_screened: int
    value_pool: int
    sobolev_candidates: int
    denominator_rejections: int
    numeric_failures: int


@dataclass(frozen=True)
class _ValueState:
    expression: str
    left_expression: str
    right_expression: str
    scale_expression: str
    denominator_sign: int
    denominator_scale: float
    left_signature: np.ndarray
    right_signature: np.ndarray
    coefficients: tuple[float, float, float]
    prediction: np.ndarray
    training_r2: float
    prediction_correlation: float


@dataclass(frozen=True)
class _ScoredState:
    value_state: _ValueState
    candidate_sobolev_gain: float
    term_complementarity: float
    joint_score: float
    selection_lanes: tuple[str, ...]


def _constant_signature(
    *,
    sample_count: int,
    dimension: int,
    lambda_value: float,
) -> np.ndarray:
    signature = np.zeros(sample_count * (dimension + 1), dtype=float)
    signature[:sample_count] = np.sqrt(lambda_value / sample_count)
    return signature


def _format_scale(scale: float) -> str:
    if scale == 1.0:
        return "1"
    return format(float(scale), ".17g")


def lift_direct_relativistic_rationals(
    *,
    direct_entries: Sequence[BasisArchiveEntry],
    target: Sequence[float],
    variable_names: Sequence[str],
    geometry_sample_count: int,
    denominator_signs: Sequence[int],
    denominator_scales: Sequence[float],
    value_pool_size: int,
    shortlist_size: int,
    proposal_limit: int,
    protected_epsilon: float = 1e-6,
    lambda_value: float = 1.0,
    lambda_gradient: float = 1.0,
) -> RelativisticRationalResult:
    """Enumerate complete rational triples before value/Sobolev screening."""

    direct = tuple(direct_entries)
    names = tuple(str(value) for value in variable_names)
    y = np.asarray(target, dtype=float).reshape(-1)
    dimension = len(names)
    signature_size = geometry_sample_count * (dimension + 1)
    signs = tuple(dict.fromkeys(int(value) for value in denominator_signs))
    scales = tuple(dict.fromkeys(float(value) for value in denominator_scales))
    if (
        len(direct) != dimension
        or dimension < 2
        or y.size < 2
        or not np.all(np.isfinite(y))
        or float(np.var(y)) <= np.finfo(float).eps
        or geometry_sample_count < 1
        or not signs
        or any(value not in {-1, 1} for value in signs)
        or not scales
        or any(not np.isfinite(value) or value <= 0.0 for value in scales)
        or value_pool_size < shortlist_size
        or shortlist_size < 1
        or proposal_limit < 1
        or protected_epsilon <= 0.0
        or lambda_value <= 0.0
        or (dimension and lambda_gradient <= 0.0)
    ):
        raise ValueError("relativistic-rational inputs are invalid")
    if any(
        np.asarray(entry.values).shape != y.shape
        or np.asarray(entry.signature).shape != (signature_size,)
        or not np.all(np.isfinite(entry.values))
        or not np.all(np.isfinite(entry.signature))
        for entry in direct
    ):
        raise ValueError("relativistic-rational entries are not aligned")

    target_variance = float(np.var(y))
    centered_target = y - float(np.mean(y))
    target_norm = float(np.linalg.norm(centered_target))
    one_signature = _constant_signature(
        sample_count=geometry_sample_count,
        dimension=dimension,
        lambda_value=lambda_value,
    )
    value_states: list[_ValueState] = []
    variable_triples = 0
    candidates_screened = 0
    denominator_rejections = 0
    numeric_failures = 0

    for left_index in range(dimension):
        left = direct[left_index]
        for right_index in range(left_index + 1, dimension):
            right = direct[right_index]
            try:
                product_signature = sobolev_product_signature(
                    left.signature,
                    right.signature,
                    sample_count=geometry_sample_count,
                    dimension=dimension,
                    lambda_value=lambda_value,
                    lambda_gradient=lambda_gradient,
                )
            except ValueError:
                numeric_failures += dimension
                continue
            product_values = np.asarray(left.values) * np.asarray(right.values)

            for scale_index in range(dimension):
                variable_triples += 1
                scale_entry = direct[scale_index]
                scale_values = np.asarray(scale_entry.values)
                scale_squared_values = scale_values * scale_values
                if np.any(np.abs(scale_squared_values) <= protected_epsilon):
                    denominator_rejections += len(signs) * len(scales)
                    continue
                try:
                    scale_squared_signature = sobolev_product_signature(
                        scale_entry.signature,
                        scale_entry.signature,
                        sample_count=geometry_sample_count,
                        dimension=dimension,
                        lambda_value=lambda_value,
                        lambda_gradient=lambda_gradient,
                    )
                    product_ratio_signature = sobolev_quotient_signature(
                        product_signature,
                        scale_squared_signature,
                        sample_count=geometry_sample_count,
                        dimension=dimension,
                        protected_epsilon=protected_epsilon,
                        lambda_value=lambda_value,
                        lambda_gradient=lambda_gradient,
                    )
                except ValueError:
                    numeric_failures += len(signs) * len(scales)
                    continue
                product_ratio_values = product_values / scale_squared_values

                for denominator_sign in signs:
                    for denominator_scale in scales:
                        candidates_screened += 1
                        signed_scale = float(denominator_sign) * denominator_scale
                        denominator_values = 1.0 + signed_scale * product_ratio_values
                        if np.any(np.abs(denominator_values) <= protected_epsilon):
                            denominator_rejections += 1
                            continue
                        denominator_signature = (
                            one_signature + signed_scale * product_ratio_signature
                        )
                        try:
                            left_signature = sobolev_quotient_signature(
                                left.signature,
                                denominator_signature,
                                sample_count=geometry_sample_count,
                                dimension=dimension,
                                protected_epsilon=protected_epsilon,
                                lambda_value=lambda_value,
                                lambda_gradient=lambda_gradient,
                            )
                            right_signature = sobolev_quotient_signature(
                                right.signature,
                                denominator_signature,
                                sample_count=geometry_sample_count,
                                dimension=dimension,
                                protected_epsilon=protected_epsilon,
                                lambda_value=lambda_value,
                                lambda_gradient=lambda_gradient,
                            )
                        except ValueError:
                            numeric_failures += 1
                            continue
                        left_values = np.asarray(left.values) / denominator_values
                        right_values = np.asarray(right.values) / denominator_values
                        design = np.column_stack(
                            (np.ones(y.size, dtype=float), left_values, right_values)
                        )
                        coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
                        rounded = np.round(coefficients, 6)
                        prediction = design @ rounded
                        training_r2 = float(
                            1.0 - np.mean(np.square(prediction - y)) / target_variance
                        )
                        centered_prediction = prediction - float(np.mean(prediction))
                        prediction_norm = float(np.linalg.norm(centered_prediction))
                        prediction_correlation = (
                            0.0
                            if prediction_norm <= np.finfo(float).eps
                            or target_norm <= np.finfo(float).eps
                            else abs(
                                float(np.dot(centered_prediction, centered_target))
                            )
                            / (prediction_norm * target_norm)
                        )
                        if not np.isfinite(training_r2) or not np.isfinite(
                            prediction_correlation
                        ):
                            numeric_failures += 1
                            continue
                        scale_text = _format_scale(denominator_scale)
                        operator = "+" if denominator_sign == 1 else "-"
                        denominator_expression = (
                            f"1 {operator} ({scale_text}) * ({names[left_index]}) "
                            f"* ({names[right_index]}) / (({names[scale_index]}) ** 2)"
                        )
                        expression = (
                            f"(({names[left_index]}) + ({names[right_index]})) "
                            f"/ ({denominator_expression})"
                        )
                        value_states.append(
                            _ValueState(
                                expression=expression,
                                left_expression=names[left_index],
                                right_expression=names[right_index],
                                scale_expression=names[scale_index],
                                denominator_sign=denominator_sign,
                                denominator_scale=denominator_scale,
                                left_signature=left_signature,
                                right_signature=right_signature,
                                coefficients=tuple(float(value) for value in rounded),
                                prediction=prediction,
                                training_r2=training_r2,
                                prediction_correlation=float(
                                    np.clip(prediction_correlation, 0.0, 1.0)
                                ),
                            )
                        )

    value_states.sort(
        key=lambda item: (
            -item.training_r2,
            -item.prediction_correlation,
            item.expression,
        )
    )
    retained = value_states[:value_pool_size]
    coordinate_span = orthonormal_signature_span(
        tuple(np.asarray(entry.signature, dtype=float) for entry in direct)
    )
    scored: list[_ScoredState] = []
    for state in retained:
        _, left_coefficient, right_coefficient = state.coefficients
        fitted_signature = (
            left_coefficient * state.left_signature
            + right_coefficient * state.right_signature
        )
        candidate_gain = _signature_gain(fitted_signature, coordinate_span)
        complementarity = (
            0.0
            if left_coefficient == 0.0 or right_coefficient == 0.0
            else _term_complementarity(
                left_coefficient * state.left_signature,
                right_coefficient * state.right_signature,
            )
        )
        joint_score = state.prediction_correlation * candidate_gain * complementarity
        scored.append(
            _ScoredState(
                value_state=state,
                candidate_sobolev_gain=candidate_gain,
                term_complementarity=complementarity,
                joint_score=joint_score,
                selection_lanes=(),
            )
        )

    value_lane = sorted(
        scored,
        key=lambda item: (
            -item.value_state.training_r2,
            -item.value_state.prediction_correlation,
            item.value_state.expression,
        ),
    )[:shortlist_size]
    sobolev_lane = sorted(
        scored,
        key=lambda item: (
            -item.joint_score,
            -item.candidate_sobolev_gain,
            -item.term_complementarity,
            -item.value_state.training_r2,
            item.value_state.expression,
        ),
    )[:shortlist_size]
    lanes: dict[int, list[str]] = {}
    for item in value_lane:
        lanes.setdefault(id(item), []).append("value_relativistic_rational")
    for item in sobolev_lane:
        lanes.setdefault(id(item), []).append("sobolev_relativistic_rational")
    selected: list[_ScoredState] = []
    seen_ids: set[int] = set()
    for item in (*value_lane, *sobolev_lane):
        if id(item) in seen_ids:
            continue
        seen_ids.add(id(item))
        selected.append(replace(item, selection_lanes=tuple(lanes[id(item)])))

    proposals = [
        RelativisticRationalProposal(
            expression=item.value_state.expression,
            left_expression=item.value_state.left_expression,
            right_expression=item.value_state.right_expression,
            scale_expression=item.value_state.scale_expression,
            denominator_sign=item.value_state.denominator_sign,
            denominator_scale=item.value_state.denominator_scale,
            fitted_coefficients=item.value_state.coefficients,
            training_r2=item.value_state.training_r2,
            prediction_correlation=item.value_state.prediction_correlation,
            candidate_sobolev_gain=item.candidate_sobolev_gain,
            term_complementarity=item.term_complementarity,
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
    unique_proposals: list[RelativisticRationalProposal] = []
    seen_expressions: set[str] = set()
    for proposal in proposals:
        if proposal.expression in seen_expressions:
            continue
        seen_expressions.add(proposal.expression)
        unique_proposals.append(proposal)
        if len(unique_proposals) >= proposal_limit:
            break

    return RelativisticRationalResult(
        proposals=tuple(unique_proposals),
        variable_triples=variable_triples,
        candidates_screened=candidates_screened,
        value_pool=len(retained),
        sobolev_candidates=len(scored),
        denominator_rejections=denominator_rejections,
        numeric_failures=numeric_failures,
    )
