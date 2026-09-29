"""Pair-first shared-denominator construction in Sobolev geometry.

The export-side plug-in enumerates compact expressions of the form

``(d1 * a + sign_n * d2 * b) / (d1 + sign_d * d2)``.

The two quotient terms are fitted by the ordinary additive evaluator.  A
complete candidate is value-screened before any Sobolev shortlist is formed;
the Sobolev lane then favours candidates whose fitted signature is novel to
the direct-coordinate span and whose two numerator contributions remain
mutually non-redundant.  No target derivative is used.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Sequence

import numpy as np

from .sn_archive_interactions import sobolev_product_signature
from .sn_basis_archive import BasisArchiveEntry
from .sn_population_coverage import orthonormal_signature_span


@dataclass(frozen=True)
class SharedDenominatorProposal:
    """One ordinary-GP expression retained by a value or Sobolev lane."""

    expression: str
    denominator_expression: str
    left_numerator_expression: str
    right_numerator_expression: str
    denominator_sign: int
    numerator_sign: int
    fitted_coefficients: tuple[float, float, float]
    training_r2: float
    prediction_correlation: float
    candidate_sobolev_gain: float
    term_complementarity: float
    joint_score: float
    selection_lanes: tuple[str, ...]


@dataclass(frozen=True)
class SharedDenominatorResult:
    """Compact audit result for one shared-denominator enumeration."""

    proposals: tuple[SharedDenominatorProposal, ...]
    denominator_pairs: int
    candidates_screened: int
    value_pool: int
    sobolev_candidates: int
    denominator_rejections: int
    numeric_failures: int


@dataclass(frozen=True)
class _ValueState:
    expression: str
    denominator_expression: str
    left_numerator_expression: str
    right_numerator_expression: str
    denominator_sign: int
    numerator_sign: int
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


def sobolev_quotient_signature(
    numerator: Sequence[float],
    denominator: Sequence[float],
    *,
    sample_count: int,
    dimension: int,
    protected_epsilon: float = 1e-6,
    lambda_value: float = 1.0,
    lambda_gradient: float = 1.0,
) -> np.ndarray:
    """Return the exact signature of ``numerator / denominator``.

    Geometry points whose denominator is zero follow the GP protected-
    division convention.  Candidate construction rejects such points, so the
    derivative branch is used only where the ordinary quotient rule applies.
    """

    left = np.asarray(numerator, dtype=float).reshape(-1)
    right = np.asarray(denominator, dtype=float).reshape(-1)
    expected = sample_count * (dimension + 1)
    if (
        sample_count < 1
        or dimension < 0
        or protected_epsilon <= 0.0
        or lambda_value <= 0.0
        or (dimension and lambda_gradient <= 0.0)
        or left.shape != (expected,)
        or right.shape != (expected,)
        or not np.all(np.isfinite(left))
        or not np.all(np.isfinite(right))
    ):
        raise ValueError("quotient signature inputs are not aligned")

    value_factor = float(np.sqrt(lambda_value / sample_count))
    numerator_values = left[:sample_count] / value_factor
    denominator_values = right[:sample_count] / value_factor
    if np.any(np.abs(denominator_values) <= protected_epsilon):
        raise ValueError("quotient signature denominator reaches protected region")
    values = numerator_values / denominator_values
    blocks = [value_factor * values]
    if dimension:
        gradient_factor = float(np.sqrt(lambda_gradient / (sample_count * dimension)))
        denominator_squared = denominator_values * denominator_values
        for axis in range(dimension):
            start = sample_count * (axis + 1)
            stop = start + sample_count
            numerator_gradient = left[start:stop] / gradient_factor
            denominator_gradient = right[start:stop] / gradient_factor
            gradient = (
                numerator_gradient * denominator_values
                - numerator_values * denominator_gradient
            ) / denominator_squared
            blocks.append(gradient_factor * gradient)
    output = np.concatenate(blocks)
    if not np.all(np.isfinite(output)):
        raise ValueError("quotient signature is non-finite")
    return output


def _signature_gain(signature: np.ndarray, span: np.ndarray | None) -> float:
    tolerance = np.finfo(float).eps
    norm = float(np.linalg.norm(signature))
    if not np.isfinite(norm) or norm <= tolerance:
        return 0.0
    normalized = signature / norm
    if span is None:
        return 1.0
    residual = normalized - span @ (span.T @ normalized)
    return float(np.clip(np.linalg.norm(residual), 0.0, 1.0))


def _term_complementarity(left: np.ndarray, right: np.ndarray) -> float:
    left_span = orthonormal_signature_span((left,))
    right_span = orthonormal_signature_span((right,))
    return min(_signature_gain(left, right_span), _signature_gain(right, left_span))


def lift_direct_shared_denominators(
    *,
    direct_entries: Sequence[BasisArchiveEntry],
    target: Sequence[float],
    variable_names: Sequence[str],
    geometry_sample_count: int,
    denominator_signs: Sequence[int],
    numerator_signs: Sequence[int],
    value_pool_size: int,
    shortlist_size: int,
    proposal_limit: int,
    protected_epsilon: float = 1e-6,
    lambda_value: float = 1.0,
    lambda_gradient: float = 1.0,
) -> SharedDenominatorResult:
    """Enumerate coupled numerator pairs before value/Sobolev screening."""

    direct = tuple(direct_entries)
    names = tuple(str(value) for value in variable_names)
    y = np.asarray(target, dtype=float).reshape(-1)
    dimension = len(names)
    signature_size = geometry_sample_count * (dimension + 1)
    denominator_operations = tuple(
        dict.fromkeys(int(value) for value in denominator_signs)
    )
    numerator_operations = tuple(dict.fromkeys(int(value) for value in numerator_signs))
    if (
        not direct
        or len(direct) != dimension
        or y.size < 2
        or not np.all(np.isfinite(y))
        or float(np.var(y)) <= np.finfo(float).eps
        or geometry_sample_count < 1
        or not denominator_operations
        or any(value not in {-1, 1} for value in denominator_operations)
        or not numerator_operations
        or any(value not in {-1, 1} for value in numerator_operations)
        or value_pool_size < shortlist_size
        or shortlist_size < 1
        or proposal_limit < 1
        or protected_epsilon <= 0.0
    ):
        raise ValueError("shared-denominator inputs are invalid")
    if any(
        np.asarray(entry.values).shape != y.shape
        or np.asarray(entry.signature).shape != (signature_size,)
        or not np.all(np.isfinite(entry.values))
        or not np.all(np.isfinite(entry.signature))
        for entry in direct
    ):
        raise ValueError("shared-denominator entries are not aligned")

    target_variance = float(np.var(y))
    centered_target = y - float(np.mean(y))
    target_norm = float(np.linalg.norm(centered_target))
    value_states: list[_ValueState] = []
    denominator_pairs = 0
    candidates_screened = 0
    denominator_rejections = 0
    numeric_failures = 0
    for left_index in range(dimension):
        left_denominator = direct[left_index]
        for right_index in range(left_index + 1, dimension):
            right_denominator = direct[right_index]
            for denominator_sign in denominator_operations:
                denominator_pairs += 1
                denominator_values = np.asarray(left_denominator.values) + float(
                    denominator_sign
                ) * np.asarray(right_denominator.values)
                denominator_signature = np.asarray(left_denominator.signature) + float(
                    denominator_sign
                ) * np.asarray(right_denominator.signature)
                if np.any(np.abs(denominator_values) <= protected_epsilon):
                    denominator_rejections += 1
                    continue
                denominator_expression = (
                    f"({names[left_index]}) "
                    f"{'+' if denominator_sign == 1 else '-'} "
                    f"({names[right_index]})"
                )
                for left_source_index in range(dimension):
                    left_source = direct[left_source_index]
                    left_numerator_values = np.asarray(
                        left_denominator.values
                    ) * np.asarray(left_source.values)
                    try:
                        left_numerator_signature = sobolev_product_signature(
                            left_denominator.signature,
                            left_source.signature,
                            sample_count=geometry_sample_count,
                            dimension=dimension,
                            lambda_value=lambda_value,
                            lambda_gradient=lambda_gradient,
                        )
                        left_term_signature = sobolev_quotient_signature(
                            left_numerator_signature,
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
                    left_values = left_numerator_values / denominator_values
                    left_expression = (
                        f"({names[left_index]}) * ({names[left_source_index]})"
                    )
                    for right_source_index in range(dimension):
                        right_source = direct[right_source_index]
                        right_numerator_values = np.asarray(
                            right_denominator.values
                        ) * np.asarray(right_source.values)
                        try:
                            right_numerator_signature = sobolev_product_signature(
                                right_denominator.signature,
                                right_source.signature,
                                sample_count=geometry_sample_count,
                                dimension=dimension,
                                lambda_value=lambda_value,
                                lambda_gradient=lambda_gradient,
                            )
                            right_unsigned_signature = sobolev_quotient_signature(
                                right_numerator_signature,
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
                        right_unsigned_values = (
                            right_numerator_values / denominator_values
                        )
                        right_expression = (
                            f"({names[right_index]}) * ({names[right_source_index]})"
                        )
                        for numerator_sign in numerator_operations:
                            candidates_screened += 1
                            right_values = float(numerator_sign) * right_unsigned_values
                            right_signature = (
                                float(numerator_sign) * right_unsigned_signature
                            )
                            design = np.column_stack(
                                (
                                    np.ones(y.size, dtype=float),
                                    left_values,
                                    right_values,
                                )
                            )
                            coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
                            rounded = np.round(coefficients, 6)
                            prediction = design @ rounded
                            training_r2 = float(
                                1.0
                                - np.mean(np.square(prediction - y)) / target_variance
                            )
                            centered_prediction = prediction - float(
                                np.mean(prediction)
                            )
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
                            expression = (
                                f"(({left_expression}) "
                                f"{'+' if numerator_sign == 1 else '-'} "
                                f"({right_expression})) / ({denominator_expression})"
                            )
                            value_states.append(
                                _ValueState(
                                    expression=expression,
                                    denominator_expression=denominator_expression,
                                    left_numerator_expression=left_expression,
                                    right_numerator_expression=right_expression,
                                    denominator_sign=denominator_sign,
                                    numerator_sign=numerator_sign,
                                    left_signature=left_term_signature,
                                    right_signature=right_signature,
                                    coefficients=tuple(
                                        float(value) for value in rounded
                                    ),
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
        intercept, left_coefficient, right_coefficient = state.coefficients
        del intercept
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
        lanes.setdefault(id(item), []).append("value_shared_denominator")
    for item in sobolev_lane:
        lanes.setdefault(id(item), []).append("sobolev_shared_denominator")
    selected: list[_ScoredState] = []
    seen_ids: set[int] = set()
    for item in (*value_lane, *sobolev_lane):
        if id(item) in seen_ids:
            continue
        seen_ids.add(id(item))
        selected.append(replace(item, selection_lanes=tuple(lanes[id(item)])))

    proposals = [
        SharedDenominatorProposal(
            expression=item.value_state.expression,
            denominator_expression=item.value_state.denominator_expression,
            left_numerator_expression=item.value_state.left_numerator_expression,
            right_numerator_expression=item.value_state.right_numerator_expression,
            denominator_sign=item.value_state.denominator_sign,
            numerator_sign=item.value_state.numerator_sign,
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
    unique_proposals: list[SharedDenominatorProposal] = []
    seen_expressions: set[str] = set()
    for proposal in proposals:
        if proposal.expression in seen_expressions:
            continue
        seen_expressions.add(proposal.expression)
        unique_proposals.append(proposal)
        if len(unique_proposals) >= proposal_limit:
            break

    return SharedDenominatorResult(
        proposals=tuple(unique_proposals),
        denominator_pairs=denominator_pairs,
        candidates_screened=candidates_screened,
        value_pool=len(retained),
        sobolev_candidates=len(scored),
        denominator_rejections=denominator_rejections,
        numeric_failures=numeric_failures,
    )
