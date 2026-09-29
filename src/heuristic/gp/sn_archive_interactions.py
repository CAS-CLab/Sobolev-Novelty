"""Sobolev-screened multiplicative lifting of archived additive bases."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import sympy as sp

from ...sobolev.decomposition import (
    decompose_expand_mul,
    parse_expression,
    to_project_expression_string,
)
from .sn_basis_archive import BasisArchiveEntry, partial_residual_credit
from .sn_population_coverage import (
    orthonormal_signature_span,
    signature_residual_gain,
)


@dataclass(frozen=True)
class ArchiveProductLift:
    """One derived product basis selected by one or more screen lanes."""

    entry: BasisArchiveEntry
    left_index: int
    right_index: int
    sobolev_gain: float
    target_correlation: float
    value_residual_gain: float
    joint_score: float
    selection_lanes: tuple[str, ...]


@dataclass(frozen=True)
class ArchiveProductLiftResult:
    """Compact audit result for one archive interaction screen."""

    lifts: tuple[ArchiveProductLift, ...]
    pairs_screened: int
    numeric_candidates: int
    construction_failures: int
    canonical_duplicates: int


@dataclass(frozen=True)
class ArchiveConditionalProductLift:
    """One product selected after conditioning on retained beam states."""

    entry: BasisArchiveEntry
    left_index: int
    right_index: int
    sobolev_gain: float
    target_correlation: float
    value_residual_gain: float
    joint_score: float
    selection_lanes: tuple[str, ...]
    joint_context: tuple[int, ...] | None
    value_context: tuple[int, ...] | None


@dataclass(frozen=True)
class ArchiveConditionalProductLiftResult:
    """Compact audit result for state-conditional product construction."""

    lifts: tuple[ArchiveConditionalProductLift, ...]
    contexts_screened: int
    pairs_screened: int
    numeric_candidates: int
    construction_failures: int
    canonical_duplicates: int


@dataclass(frozen=True)
class ArchiveConditionalUnaryLift:
    """One bounded unary basis selected after conditioning on beam states."""

    entry: BasisArchiveEntry
    source_index: int
    transform: str
    sobolev_gain: float
    target_correlation: float
    value_residual_gain: float
    joint_score: float
    selection_lanes: tuple[str, ...]
    joint_context: tuple[int, ...] | None
    value_context: tuple[int, ...] | None


@dataclass(frozen=True)
class ArchiveConditionalUnaryLiftResult:
    """Compact audit result for conditional bounded-unary construction."""

    lifts: tuple[ArchiveConditionalUnaryLift, ...]
    contexts_screened: int
    candidates_screened: int
    numeric_candidates: int
    construction_failures: int
    canonical_duplicates: int


@dataclass(frozen=True)
class ArchiveConditionalRationalLift:
    """One safe rational basis selected after conditioning on beam states."""

    entry: BasisArchiveEntry
    numerator_index: int
    denominator_index: int
    sobolev_gain: float
    target_correlation: float
    value_residual_gain: float
    joint_score: float
    selection_lanes: tuple[str, ...]
    joint_context: tuple[int, ...] | None
    value_context: tuple[int, ...] | None


@dataclass(frozen=True)
class ArchiveConditionalRationalLiftResult:
    """Compact audit result for conditional safe-rational construction."""

    lifts: tuple[ArchiveConditionalRationalLift, ...]
    contexts_screened: int
    candidates_screened: int
    numeric_candidates: int
    construction_failures: int
    canonical_duplicates: int


@dataclass(frozen=True)
class ArchiveConditionalAffineUnaryLift:
    """One bounded unary lift of an archive pair sum or difference."""

    entry: BasisArchiveEntry
    left_index: int
    right_index: int
    sign: int
    transform: str
    sobolev_gain: float
    target_correlation: float
    value_residual_gain: float
    joint_score: float
    selection_lanes: tuple[str, ...]
    joint_context: tuple[int, ...] | None
    value_context: tuple[int, ...] | None


@dataclass(frozen=True)
class ArchiveConditionalAffineUnaryLiftResult:
    """Compact audit result for conditional affine-unary construction."""

    lifts: tuple[ArchiveConditionalAffineUnaryLift, ...]
    contexts_screened: int
    candidates_screened: int
    numeric_candidates: int
    construction_failures: int
    canonical_duplicates: int


@dataclass(frozen=True)
class ArchiveConditionalRadialLift:
    """One safe Euclidean norm of two screened archive components."""

    entry: BasisArchiveEntry
    first_component: tuple[int, int, int]
    second_component: tuple[int, int, int]
    sobolev_gain: float
    target_correlation: float
    value_residual_gain: float
    joint_score: float
    selection_lanes: tuple[str, ...]
    joint_context: tuple[int, ...] | None
    value_context: tuple[int, ...] | None


@dataclass(frozen=True)
class ArchiveConditionalRadialLiftResult:
    """Compact audit result for two-stage conditional radial construction."""

    lifts: tuple[ArchiveConditionalRadialLift, ...]
    contexts_screened: int
    component_candidates: int
    component_states_screened: int
    components_selected: int
    radial_candidates: int
    radial_states_screened: int
    construction_failures: int
    canonical_duplicates: int


def sobolev_product_signature(
    left: Sequence[float],
    right: Sequence[float],
    *,
    sample_count: int,
    dimension: int,
    lambda_value: float = 1.0,
    lambda_gradient: float = 1.0,
) -> np.ndarray:
    """Apply the product rule directly in the normalized Sobolev geometry."""

    left_vector = np.asarray(left, dtype=float).reshape(-1)
    right_vector = np.asarray(right, dtype=float).reshape(-1)
    expected = sample_count * (dimension + 1)
    if (
        sample_count < 1
        or dimension < 0
        or lambda_value <= 0.0
        or (dimension and lambda_gradient <= 0.0)
        or left_vector.shape != (expected,)
        or right_vector.shape != (expected,)
        or not np.all(np.isfinite(left_vector))
        or not np.all(np.isfinite(right_vector))
    ):
        raise ValueError("product signature inputs are not aligned")

    value_factor = float(np.sqrt(lambda_value / sample_count))
    left_value = left_vector[:sample_count] / value_factor
    right_value = right_vector[:sample_count] / value_factor
    blocks = [value_factor * left_value * right_value]
    if dimension:
        gradient_factor = float(np.sqrt(lambda_gradient / (sample_count * dimension)))
        for axis in range(dimension):
            start = sample_count * (axis + 1)
            stop = start + sample_count
            left_gradient = left_vector[start:stop] / gradient_factor
            right_gradient = right_vector[start:stop] / gradient_factor
            blocks.append(
                gradient_factor
                * (left_gradient * right_value + left_value * right_gradient)
            )
    output = np.concatenate(blocks)
    if not np.all(np.isfinite(output)):
        raise ValueError("product signature is non-finite")
    return output


def sobolev_unary_signature(
    source: Sequence[float],
    *,
    transform: str,
    sample_count: int,
    dimension: int,
    lambda_value: float = 1.0,
    lambda_gradient: float = 1.0,
) -> np.ndarray:
    """Apply a bounded unary transform and its chain rule in SN geometry."""

    vector = np.asarray(source, dtype=float).reshape(-1)
    expected = sample_count * (dimension + 1)
    if (
        transform not in {"sin", "cos", "tanh"}
        or sample_count < 1
        or dimension < 0
        or lambda_value <= 0.0
        or (dimension and lambda_gradient <= 0.0)
        or vector.shape != (expected,)
        or not np.all(np.isfinite(vector))
    ):
        raise ValueError("unary signature inputs are not aligned")

    value_factor = float(np.sqrt(lambda_value / sample_count))
    source_values = vector[:sample_count] / value_factor
    if transform == "sin":
        transformed_values = np.sin(source_values)
        derivative = np.cos(source_values)
    elif transform == "cos":
        transformed_values = np.cos(source_values)
        derivative = -np.sin(source_values)
    else:
        transformed_values = np.tanh(source_values)
        derivative = 1.0 - transformed_values * transformed_values
    blocks = [value_factor * transformed_values]
    if dimension:
        gradient_factor = float(np.sqrt(lambda_gradient / (sample_count * dimension)))
        for axis in range(dimension):
            start = sample_count * (axis + 1)
            stop = start + sample_count
            source_gradient = vector[start:stop] / gradient_factor
            blocks.append(gradient_factor * derivative * source_gradient)
    output = np.concatenate(blocks)
    if not np.all(np.isfinite(output)):
        raise ValueError("unary signature is non-finite")
    return output


def sobolev_safe_rational_signature(
    numerator: Sequence[float],
    denominator_basis: Sequence[float],
    *,
    sample_count: int,
    dimension: int,
    lambda_value: float = 1.0,
    lambda_gradient: float = 1.0,
) -> np.ndarray:
    """Return the exact signature of ``a / (1 + b**2)``."""

    left = np.asarray(numerator, dtype=float).reshape(-1)
    right = np.asarray(denominator_basis, dtype=float).reshape(-1)
    expected = sample_count * (dimension + 1)
    if (
        sample_count < 1
        or dimension < 0
        or lambda_value <= 0.0
        or (dimension and lambda_gradient <= 0.0)
        or left.shape != (expected,)
        or right.shape != (expected,)
        or not np.all(np.isfinite(left))
        or not np.all(np.isfinite(right))
    ):
        raise ValueError("safe rational signature inputs are not aligned")

    value_factor = float(np.sqrt(lambda_value / sample_count))
    a = left[:sample_count] / value_factor
    b = right[:sample_count] / value_factor
    denominator = 1.0 + b * b
    values = a / denominator
    blocks = [value_factor * values]
    if dimension:
        gradient_factor = float(np.sqrt(lambda_gradient / (sample_count * dimension)))
        denominator_squared = denominator * denominator
        for axis in range(dimension):
            start = sample_count * (axis + 1)
            stop = start + sample_count
            da = left[start:stop] / gradient_factor
            db = right[start:stop] / gradient_factor
            gradient = (da * denominator - a * (2.0 * b * db)) / denominator_squared
            blocks.append(gradient_factor * gradient)
    output = np.concatenate(blocks)
    if not np.all(np.isfinite(output)):
        raise ValueError("safe rational signature is non-finite")
    return output


def sobolev_affine_unary_signature(
    left: Sequence[float],
    right: Sequence[float],
    *,
    sign: int,
    transform: str,
    sample_count: int,
    dimension: int,
    lambda_value: float = 1.0,
    lambda_gradient: float = 1.0,
) -> np.ndarray:
    """Return the exact signature of ``transform(left + sign * right)``."""

    left_vector = np.asarray(left, dtype=float).reshape(-1)
    right_vector = np.asarray(right, dtype=float).reshape(-1)
    if sign not in {-1, 1} or left_vector.shape != right_vector.shape:
        raise ValueError("affine unary signature inputs are not aligned")
    return sobolev_unary_signature(
        left_vector + float(sign) * right_vector,
        transform=transform,
        sample_count=sample_count,
        dimension=dimension,
        lambda_value=lambda_value,
        lambda_gradient=lambda_gradient,
    )


def sobolev_radial_signature(
    first: Sequence[float],
    second: Sequence[float],
    *,
    sample_count: int,
    dimension: int,
    lambda_value: float = 1.0,
    lambda_gradient: float = 1.0,
) -> np.ndarray:
    """Return the exact a.e. signature of ``sqrt(first**2 + second**2)``."""

    left = np.asarray(first, dtype=float).reshape(-1)
    right = np.asarray(second, dtype=float).reshape(-1)
    expected = sample_count * (dimension + 1)
    if (
        sample_count < 1
        or dimension < 0
        or lambda_value <= 0.0
        or (dimension and lambda_gradient <= 0.0)
        or left.shape != (expected,)
        or right.shape != (expected,)
        or not np.all(np.isfinite(left))
        or not np.all(np.isfinite(right))
    ):
        raise ValueError("radial signature inputs are not aligned")

    value_factor = float(np.sqrt(lambda_value / sample_count))
    a = left[:sample_count] / value_factor
    b = right[:sample_count] / value_factor
    radius = np.sqrt(a * a + b * b)
    blocks = [value_factor * radius]
    if dimension:
        gradient_factor = float(np.sqrt(lambda_gradient / (sample_count * dimension)))
        nonzero = radius > np.finfo(float).eps
        for axis in range(dimension):
            start = sample_count * (axis + 1)
            stop = start + sample_count
            da = left[start:stop] / gradient_factor
            db = right[start:stop] / gradient_factor
            gradient = np.zeros(sample_count, dtype=float)
            gradient[nonzero] = (
                a[nonzero] * da[nonzero] + b[nonzero] * db[nonzero]
            ) / radius[nonzero]
            blocks.append(gradient_factor * gradient)
    output = np.concatenate(blocks)
    if not np.all(np.isfinite(output)):
        raise ValueError("radial signature is non-finite")
    return output


def lift_archive_product_interactions(
    *,
    entries: Sequence[BasisArchiveEntry],
    target: Sequence[float],
    variable_names: Sequence[str],
    geometry_sample_count: int,
    joint_shortlist_size: int,
    value_shortlist_size: int,
    lambda_value: float = 1.0,
    lambda_gradient: float = 1.0,
) -> ArchiveProductLiftResult:
    """Create a small exact-symbolic pool from all numeric archive products."""

    pool = tuple(entries)
    y = np.asarray(target, dtype=float).reshape(-1)
    dimension = len(tuple(variable_names))
    if (
        len(pool) < 1
        or y.size < 1
        or not np.all(np.isfinite(y))
        or joint_shortlist_size < 1
        or value_shortlist_size < 1
    ):
        raise ValueError("archive product lifting inputs are invalid")
    signatures = [np.asarray(entry.signature, dtype=float) for entry in pool]
    span = orthonormal_signature_span(signatures)
    design = np.ones((y.size, 1), dtype=float)
    residual = y - float(np.mean(y))
    tolerance = np.finfo(float).eps
    numeric: list[dict[str, object]] = []
    pairs_screened = 0
    for left_index, left in enumerate(pool):
        for right_index in range(left_index, len(pool)):
            right = pool[right_index]
            pairs_screened += 1
            values = np.asarray(left.values, dtype=float) * np.asarray(
                right.values, dtype=float
            )
            if values.shape != y.shape or not np.all(np.isfinite(values)):
                continue
            try:
                signature = sobolev_product_signature(
                    left.signature,
                    right.signature,
                    sample_count=geometry_sample_count,
                    dimension=dimension,
                    lambda_value=lambda_value,
                    lambda_gradient=lambda_gradient,
                )
            except ValueError:
                continue
            norm = float(np.linalg.norm(signature))
            if not np.isfinite(norm) or norm <= tolerance:
                continue
            gain = signature_residual_gain(span, signature / norm)
            correlation, value_gain = partial_residual_credit(values, residual, design)
            numeric.append(
                {
                    "left_index": left_index,
                    "right_index": right_index,
                    "signature": signature,
                    "values": values,
                    "sobolev_gain": float(gain),
                    "target_correlation": float(correlation),
                    "value_residual_gain": float(value_gain),
                    "joint_score": float(gain * correlation),
                    "selection_lanes": (),
                }
            )

    joint = sorted(
        (item for item in numeric if float(item["joint_score"]) > tolerance),
        key=lambda item: (
            -float(item["joint_score"]),
            -float(item["sobolev_gain"]),
            -float(item["target_correlation"]),
            pool[int(item["left_index"])].canonical,
            pool[int(item["right_index"])].canonical,
        ),
    )[:joint_shortlist_size]
    value = sorted(
        (
            item
            for item in numeric
            if float(item["target_correlation"]) * float(item["value_residual_gain"])
            > tolerance
        ),
        key=lambda item: (
            -float(item["target_correlation"]) * float(item["value_residual_gain"]),
            -float(item["target_correlation"]),
            -float(item["value_residual_gain"]),
            pool[int(item["left_index"])].canonical,
            pool[int(item["right_index"])].canonical,
        ),
    )[:value_shortlist_size]
    selected: list[dict[str, object]] = []
    selected_positions: dict[tuple[int, int], int] = {}

    def add_lane(item: dict[str, object], lane: str) -> None:
        key = (int(item["left_index"]), int(item["right_index"]))
        position = selected_positions.get(key)
        if position is None:
            selected_positions[key] = len(selected)
            selected.append({**item, "selection_lanes": (lane,)})
        else:
            current = selected[position]
            current["selection_lanes"] = (
                *tuple(current["selection_lanes"]),
                lane,
            )

    for item in joint:
        add_lane(item, "sobolev_product")
    for item in value:
        add_lane(item, "value_product")

    existing = {entry.canonical for entry in pool}
    lifted: list[ArchiveProductLift] = []
    construction_failures = 0
    canonical_duplicates = 0
    for item in selected:
        left_index = int(item["left_index"])
        right_index = int(item["right_index"])
        try:
            left_expression, symbols = parse_expression(
                pool[left_index].expression, variable_names
            )
            right_expression, _ = parse_expression(
                pool[right_index].expression, variable_names
            )
            terms = tuple(
                term
                for term in decompose_expand_mul(
                    sp.Mul(left_expression, right_expression), symbols
                )
                if term.basis.free_symbols
            )
            if len(terms) != 1 or abs(float(terms[0].coefficient)) <= tolerance:
                raise ValueError("product did not remain one structural basis")
            term = terms[0]
            coefficient = float(term.coefficient)
            if term.canonical in existing:
                canonical_duplicates += 1
                continue
            existing.add(term.canonical)
            source_left = pool[left_index]
            source_right = pool[right_index]
            entry = BasisArchiveEntry(
                canonical=term.canonical,
                expression=to_project_expression_string(term.basis),
                signature=np.asarray(item["signature"], dtype=float) / coefficient,
                values=np.asarray(item["values"], dtype=float) / coefficient,
                term_norm=float(
                    np.linalg.norm(
                        np.asarray(item["signature"], dtype=float) / coefficient
                    )
                ),
                source_coefficient=1.0,
                source_amplitude=float(
                    abs(source_left.source_amplitude)
                    * abs(source_right.source_amplitude)
                ),
                source_base_reward=max(
                    source_left.source_base_reward,
                    source_right.source_base_reward,
                ),
                source_generation=max(
                    source_left.source_generation,
                    source_right.source_generation,
                ),
                source_base_rank=min(
                    source_left.source_base_rank,
                    source_right.source_base_rank,
                ),
                source_candidate_id=min(
                    source_left.source_candidate_id,
                    source_right.source_candidate_id,
                ),
            )
        except (TypeError, ValueError, KeyError):
            construction_failures += 1
            continue
        lifted.append(
            ArchiveProductLift(
                entry=entry,
                left_index=left_index,
                right_index=right_index,
                sobolev_gain=float(item["sobolev_gain"]),
                target_correlation=float(item["target_correlation"]),
                value_residual_gain=float(item["value_residual_gain"]),
                joint_score=float(item["joint_score"]),
                selection_lanes=tuple(item["selection_lanes"]),
            )
        )
    return ArchiveProductLiftResult(
        lifts=tuple(lifted),
        pairs_screened=pairs_screened,
        numeric_candidates=len(numeric),
        construction_failures=construction_failures,
        canonical_duplicates=canonical_duplicates,
    )


def lift_archive_conditional_product_interactions(
    *,
    entries: Sequence[BasisArchiveEntry],
    target: Sequence[float],
    variable_names: Sequence[str],
    geometry_sample_count: int,
    contexts: Sequence[Sequence[int]],
    source_indices: Sequence[int] | None = None,
    joint_shortlist_size: int,
    value_shortlist_size: int,
    require_derived_index_at_least: int | None = None,
    lambda_value: float = 1.0,
    lambda_gradient: float = 1.0,
) -> ArchiveConditionalProductLiftResult:
    """Screen products against residuals and Sobolev spans of beam states."""

    pool = tuple(entries)
    names = tuple(variable_names)
    y = np.asarray(target, dtype=float).reshape(-1)
    unique_contexts = tuple(
        dict.fromkeys(
            tuple(sorted(int(index) for index in value)) for value in contexts
        )
    )
    if (
        not pool
        or y.size < 1
        or not np.all(np.isfinite(y))
        or not unique_contexts
        or joint_shortlist_size < 1
        or value_shortlist_size < 1
    ):
        raise ValueError("conditional archive product inputs are invalid")
    if require_derived_index_at_least is not None and not (
        0 <= require_derived_index_at_least < len(pool)
    ):
        raise ValueError("conditional derived-term boundary is invalid")
    if any(
        not context
        or len(set(context)) != len(context)
        or any(index < 0 or index >= len(pool) for index in context)
        for context in unique_contexts
    ):
        raise ValueError("conditional archive product contexts are invalid")

    dimension = len(names)
    tolerance = np.finfo(float).eps
    source_positions = (
        tuple(range(len(pool)))
        if source_indices is None
        else tuple(dict.fromkeys(int(index) for index in source_indices))
    )
    if not source_positions or any(
        index < 0 or index >= len(pool) for index in source_positions
    ):
        raise ValueError("conditional archive product sources are invalid")
    pairs: list[tuple[int, int]] = []
    raw_signatures: list[np.ndarray] = []
    signatures: list[np.ndarray] = []
    values: list[np.ndarray] = []
    candidate_pair_count = 0
    for left_offset, left_index in enumerate(source_positions):
        left = pool[left_index]
        for right_index in source_positions[left_offset:]:
            if (
                require_derived_index_at_least is not None
                and right_index < require_derived_index_at_least
            ):
                continue
            candidate_pair_count += 1
            right = pool[right_index]
            product_values = np.asarray(left.values, dtype=float) * np.asarray(
                right.values, dtype=float
            )
            if product_values.shape != y.shape or not np.all(
                np.isfinite(product_values)
            ):
                continue
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
                continue
            signature_norm = float(np.linalg.norm(product_signature))
            if not np.isfinite(signature_norm) or signature_norm <= tolerance:
                continue
            pairs.append((left_index, right_index))
            raw_signatures.append(product_signature)
            signatures.append(product_signature / signature_norm)
            values.append(product_values)

    if not pairs:
        return ArchiveConditionalProductLiftResult(
            lifts=(),
            contexts_screened=len(unique_contexts),
            pairs_screened=len(unique_contexts) * candidate_pair_count,
            numeric_candidates=0,
            construction_failures=0,
            canonical_duplicates=0,
        )

    signature_matrix = np.column_stack(signatures)
    value_matrix = np.column_stack(values)
    centered_norms = np.linalg.norm(
        value_matrix - np.mean(value_matrix, axis=0, keepdims=True), axis=0
    )
    count = len(pairs)
    best_joint = np.zeros(count, dtype=float)
    best_joint_gain = np.zeros(count, dtype=float)
    best_joint_correlation = np.zeros(count, dtype=float)
    best_joint_value_gain = np.zeros(count, dtype=float)
    best_value_score = np.zeros(count, dtype=float)
    best_value_gain = np.zeros(count, dtype=float)
    best_value_correlation = np.zeros(count, dtype=float)
    best_value_sobolev_gain = np.zeros(count, dtype=float)
    joint_contexts: list[tuple[int, ...] | None] = [None] * count
    value_contexts: list[tuple[int, ...] | None] = [None] * count

    for context in unique_contexts:
        design = np.column_stack(
            (
                np.ones(y.size, dtype=float),
                *(np.asarray(pool[index].values, dtype=float) for index in context),
            )
        )
        coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
        residual = y - design @ np.round(coefficients, 6)
        residual_norm = float(np.linalg.norm(residual))
        value_span = orthonormal_signature_span(
            [design[:, index] for index in range(design.shape[1])]
        )
        orthogonal_values = value_matrix - value_span @ (value_span.T @ value_matrix)
        orthogonal_norms = np.linalg.norm(orthogonal_values, axis=0)
        correlations = np.zeros(count, dtype=float)
        valid_value = (orthogonal_norms > tolerance) & (residual_norm > tolerance)
        correlations[valid_value] = np.abs(
            orthogonal_values[:, valid_value].T @ residual
        ) / (orthogonal_norms[valid_value] * residual_norm)
        value_gains = np.zeros(count, dtype=float)
        nonzero_centered = centered_norms > tolerance
        value_gains[nonzero_centered] = (
            orthogonal_norms[nonzero_centered] / centered_norms[nonzero_centered]
        )
        correlations = np.clip(correlations, 0.0, 1.0)
        value_gains = np.clip(value_gains, 0.0, 1.0)

        signature_span = orthonormal_signature_span(
            [np.asarray(pool[index].signature, dtype=float) for index in context]
        )
        signature_residuals = signature_matrix - signature_span @ (
            signature_span.T @ signature_matrix
        )
        gains = np.clip(np.linalg.norm(signature_residuals, axis=0), 0.0, 1.0)
        joint_scores = gains * correlations
        value_scores = correlations * value_gains
        joint_better = joint_scores > best_joint + tolerance
        value_better = value_scores > best_value_score + tolerance
        for index in np.flatnonzero(joint_better):
            best_joint[index] = joint_scores[index]
            best_joint_gain[index] = gains[index]
            best_joint_correlation[index] = correlations[index]
            best_joint_value_gain[index] = value_gains[index]
            joint_contexts[index] = context
        for index in np.flatnonzero(value_better):
            best_value_score[index] = value_scores[index]
            best_value_gain[index] = value_gains[index]
            best_value_correlation[index] = correlations[index]
            best_value_sobolev_gain[index] = gains[index]
            value_contexts[index] = context

    joint_order = sorted(
        (index for index in range(count) if best_joint[index] > tolerance),
        key=lambda index: (
            -best_joint[index],
            -best_joint_gain[index],
            -best_joint_correlation[index],
            pool[pairs[index][0]].canonical,
            pool[pairs[index][1]].canonical,
        ),
    )[:joint_shortlist_size]
    value_order = sorted(
        (index for index in range(count) if best_value_score[index] > tolerance),
        key=lambda index: (
            -best_value_score[index],
            -best_value_correlation[index],
            -best_value_gain[index],
            pool[pairs[index][0]].canonical,
            pool[pairs[index][1]].canonical,
        ),
    )[:value_shortlist_size]
    selected: list[int] = []
    lanes: dict[int, list[str]] = {}
    for index in joint_order:
        selected.append(index)
        lanes[index] = ["sobolev_conditional_product"]
    for index in value_order:
        if index not in lanes:
            selected.append(index)
            lanes[index] = []
        lanes[index].append("value_conditional_product")

    existing = {entry.canonical for entry in pool}
    lifted: list[ArchiveConditionalProductLift] = []
    construction_failures = 0
    canonical_duplicates = 0
    for index in selected:
        left_index, right_index = pairs[index]
        try:
            left_expression, symbols = parse_expression(
                pool[left_index].expression, names
            )
            right_expression, _ = parse_expression(pool[right_index].expression, names)
            terms = tuple(
                term
                for term in decompose_expand_mul(
                    sp.Mul(left_expression, right_expression), symbols
                )
                if term.basis.free_symbols
            )
            if len(terms) != 1 or abs(float(terms[0].coefficient)) <= tolerance:
                raise ValueError("product did not remain one structural basis")
            term = terms[0]
            coefficient = float(term.coefficient)
            if term.canonical in existing:
                canonical_duplicates += 1
                continue
            existing.add(term.canonical)
            source_left = pool[left_index]
            source_right = pool[right_index]
            raw_signature = raw_signatures[index]
            entry = BasisArchiveEntry(
                canonical=term.canonical,
                expression=to_project_expression_string(term.basis),
                signature=raw_signature / coefficient,
                values=values[index] / coefficient,
                term_norm=float(np.linalg.norm(raw_signature / coefficient)),
                source_coefficient=1.0,
                source_amplitude=float(
                    abs(source_left.source_amplitude)
                    * abs(source_right.source_amplitude)
                ),
                source_base_reward=max(
                    source_left.source_base_reward,
                    source_right.source_base_reward,
                ),
                source_generation=max(
                    source_left.source_generation,
                    source_right.source_generation,
                ),
                source_base_rank=min(
                    source_left.source_base_rank,
                    source_right.source_base_rank,
                ),
                source_candidate_id=min(
                    source_left.source_candidate_id,
                    source_right.source_candidate_id,
                ),
            )
        except (TypeError, ValueError, KeyError):
            construction_failures += 1
            continue
        joint_selected = "sobolev_conditional_product" in lanes[index]
        lifted.append(
            ArchiveConditionalProductLift(
                entry=entry,
                left_index=left_index,
                right_index=right_index,
                sobolev_gain=float(
                    best_joint_gain[index]
                    if joint_selected
                    else best_value_sobolev_gain[index]
                ),
                target_correlation=float(
                    best_joint_correlation[index]
                    if joint_selected
                    else best_value_correlation[index]
                ),
                value_residual_gain=float(
                    best_joint_value_gain[index]
                    if joint_selected
                    else best_value_gain[index]
                ),
                joint_score=float(best_joint[index]),
                selection_lanes=tuple(lanes[index]),
                joint_context=joint_contexts[index],
                value_context=value_contexts[index],
            )
        )
    return ArchiveConditionalProductLiftResult(
        lifts=tuple(lifted),
        contexts_screened=len(unique_contexts),
        pairs_screened=len(unique_contexts) * candidate_pair_count,
        numeric_candidates=len(pairs),
        construction_failures=construction_failures,
        canonical_duplicates=canonical_duplicates,
    )


def lift_archive_conditional_unary_interactions(
    *,
    entries: Sequence[BasisArchiveEntry],
    target: Sequence[float],
    variable_names: Sequence[str],
    geometry_sample_count: int,
    contexts: Sequence[Sequence[int]],
    transforms: Sequence[str],
    joint_shortlist_size: int,
    value_shortlist_size: int,
    lambda_value: float = 1.0,
    lambda_gradient: float = 1.0,
) -> ArchiveConditionalUnaryLiftResult:
    """Screen bounded unary lifts against retained value/Sobolev states."""

    pool = tuple(entries)
    names = tuple(variable_names)
    operations = tuple(dict.fromkeys(str(value) for value in transforms))
    y = np.asarray(target, dtype=float).reshape(-1)
    unique_contexts = tuple(
        dict.fromkeys(
            tuple(sorted(int(index) for index in value)) for value in contexts
        )
    )
    if (
        not pool
        or y.size < 1
        or not np.all(np.isfinite(y))
        or not unique_contexts
        or not operations
        or any(value not in {"sin", "cos", "tanh"} for value in operations)
        or joint_shortlist_size < 1
        or value_shortlist_size < 1
    ):
        raise ValueError("conditional archive unary inputs are invalid")
    if any(
        not context
        or len(set(context)) != len(context)
        or any(index < 0 or index >= len(pool) for index in context)
        for context in unique_contexts
    ):
        raise ValueError("conditional archive unary contexts are invalid")

    dimension = len(names)
    tolerance = np.finfo(float).eps
    sources: list[int] = []
    transform_names: list[str] = []
    raw_signatures: list[np.ndarray] = []
    signatures: list[np.ndarray] = []
    values: list[np.ndarray] = []
    for source_index, source in enumerate(pool):
        source_values = np.asarray(source.values, dtype=float)
        if source_values.shape != y.shape or not np.all(np.isfinite(source_values)):
            continue
        for transform in operations:
            if transform == "sin":
                transformed_values = np.sin(source_values)
            elif transform == "cos":
                transformed_values = np.cos(source_values)
            else:
                transformed_values = np.tanh(source_values)
            try:
                transformed_signature = sobolev_unary_signature(
                    source.signature,
                    transform=transform,
                    sample_count=geometry_sample_count,
                    dimension=dimension,
                    lambda_value=lambda_value,
                    lambda_gradient=lambda_gradient,
                )
            except ValueError:
                continue
            signature_norm = float(np.linalg.norm(transformed_signature))
            if (
                not np.all(np.isfinite(transformed_values))
                or not np.isfinite(signature_norm)
                or signature_norm <= tolerance
            ):
                continue
            sources.append(source_index)
            transform_names.append(transform)
            raw_signatures.append(transformed_signature)
            signatures.append(transformed_signature / signature_norm)
            values.append(transformed_values)

    candidate_count = len(pool) * len(operations)
    if not sources:
        return ArchiveConditionalUnaryLiftResult(
            lifts=(),
            contexts_screened=len(unique_contexts),
            candidates_screened=len(unique_contexts) * candidate_count,
            numeric_candidates=0,
            construction_failures=0,
            canonical_duplicates=0,
        )

    signature_matrix = np.column_stack(signatures)
    value_matrix = np.column_stack(values)
    centered_norms = np.linalg.norm(
        value_matrix - np.mean(value_matrix, axis=0, keepdims=True), axis=0
    )
    count = len(sources)
    best_joint = np.zeros(count, dtype=float)
    best_joint_gain = np.zeros(count, dtype=float)
    best_joint_correlation = np.zeros(count, dtype=float)
    best_joint_value_gain = np.zeros(count, dtype=float)
    best_value_score = np.zeros(count, dtype=float)
    best_value_gain = np.zeros(count, dtype=float)
    best_value_correlation = np.zeros(count, dtype=float)
    best_value_sobolev_gain = np.zeros(count, dtype=float)
    joint_contexts: list[tuple[int, ...] | None] = [None] * count
    value_contexts: list[tuple[int, ...] | None] = [None] * count

    for context in unique_contexts:
        design = np.column_stack(
            (
                np.ones(y.size, dtype=float),
                *(np.asarray(pool[index].values, dtype=float) for index in context),
            )
        )
        coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
        residual = y - design @ np.round(coefficients, 6)
        residual_norm = float(np.linalg.norm(residual))
        value_span = orthonormal_signature_span(
            [design[:, index] for index in range(design.shape[1])]
        )
        orthogonal_values = value_matrix - value_span @ (value_span.T @ value_matrix)
        orthogonal_norms = np.linalg.norm(orthogonal_values, axis=0)
        correlations = np.zeros(count, dtype=float)
        valid_value = (orthogonal_norms > tolerance) & (residual_norm > tolerance)
        correlations[valid_value] = np.abs(
            orthogonal_values[:, valid_value].T @ residual
        ) / (orthogonal_norms[valid_value] * residual_norm)
        value_gains = np.zeros(count, dtype=float)
        nonzero_centered = centered_norms > tolerance
        value_gains[nonzero_centered] = (
            orthogonal_norms[nonzero_centered] / centered_norms[nonzero_centered]
        )
        correlations = np.clip(correlations, 0.0, 1.0)
        value_gains = np.clip(value_gains, 0.0, 1.0)

        signature_span = orthonormal_signature_span(
            [np.asarray(pool[index].signature, dtype=float) for index in context]
        )
        signature_residuals = signature_matrix - signature_span @ (
            signature_span.T @ signature_matrix
        )
        gains = np.clip(np.linalg.norm(signature_residuals, axis=0), 0.0, 1.0)
        joint_scores = gains * correlations
        value_scores = correlations * value_gains
        joint_better = joint_scores > best_joint + tolerance
        value_better = value_scores > best_value_score + tolerance
        for index in np.flatnonzero(joint_better):
            best_joint[index] = joint_scores[index]
            best_joint_gain[index] = gains[index]
            best_joint_correlation[index] = correlations[index]
            best_joint_value_gain[index] = value_gains[index]
            joint_contexts[index] = context
        for index in np.flatnonzero(value_better):
            best_value_score[index] = value_scores[index]
            best_value_gain[index] = value_gains[index]
            best_value_correlation[index] = correlations[index]
            best_value_sobolev_gain[index] = gains[index]
            value_contexts[index] = context

    joint_order = sorted(
        (index for index in range(count) if best_joint[index] > tolerance),
        key=lambda index: (
            -best_joint[index],
            -best_joint_gain[index],
            -best_joint_correlation[index],
            transform_names[index],
            pool[sources[index]].canonical,
        ),
    )[:joint_shortlist_size]
    value_order = sorted(
        (index for index in range(count) if best_value_score[index] > tolerance),
        key=lambda index: (
            -best_value_score[index],
            -best_value_correlation[index],
            -best_value_gain[index],
            transform_names[index],
            pool[sources[index]].canonical,
        ),
    )[:value_shortlist_size]
    selected: list[int] = []
    lanes: dict[int, list[str]] = {}
    for index in joint_order:
        selected.append(index)
        lanes[index] = ["sobolev_conditional_unary"]
    for index in value_order:
        if index not in lanes:
            selected.append(index)
            lanes[index] = []
        lanes[index].append("value_conditional_unary")

    symbolic_operations = {"sin": sp.sin, "cos": sp.cos, "tanh": sp.tanh}
    existing = {entry.canonical for entry in pool}
    lifted: list[ArchiveConditionalUnaryLift] = []
    construction_failures = 0
    canonical_duplicates = 0
    for index in selected:
        source_index = sources[index]
        transform = transform_names[index]
        try:
            source_expression, symbols = parse_expression(
                pool[source_index].expression, names
            )
            terms = tuple(
                term
                for term in decompose_expand_mul(
                    symbolic_operations[transform](source_expression), symbols
                )
                if term.basis.free_symbols
            )
            if len(terms) != 1 or abs(float(terms[0].coefficient)) <= tolerance:
                raise ValueError("unary lift did not remain one structural basis")
            term = terms[0]
            coefficient = float(term.coefficient)
            if term.canonical in existing:
                canonical_duplicates += 1
                continue
            existing.add(term.canonical)
            source = pool[source_index]
            raw_signature = raw_signatures[index]
            entry = BasisArchiveEntry(
                canonical=term.canonical,
                expression=to_project_expression_string(term.basis),
                signature=raw_signature / coefficient,
                values=values[index] / coefficient,
                term_norm=float(np.linalg.norm(raw_signature / coefficient)),
                source_coefficient=1.0,
                source_amplitude=float(source.source_amplitude),
                source_base_reward=float(source.source_base_reward),
                source_generation=int(source.source_generation),
                source_base_rank=int(source.source_base_rank),
                source_candidate_id=int(source.source_candidate_id),
            )
        except (TypeError, ValueError, KeyError):
            construction_failures += 1
            continue
        joint_selected = "sobolev_conditional_unary" in lanes[index]
        lifted.append(
            ArchiveConditionalUnaryLift(
                entry=entry,
                source_index=source_index,
                transform=transform,
                sobolev_gain=float(
                    best_joint_gain[index]
                    if joint_selected
                    else best_value_sobolev_gain[index]
                ),
                target_correlation=float(
                    best_joint_correlation[index]
                    if joint_selected
                    else best_value_correlation[index]
                ),
                value_residual_gain=float(
                    best_joint_value_gain[index]
                    if joint_selected
                    else best_value_gain[index]
                ),
                joint_score=float(best_joint[index]),
                selection_lanes=tuple(lanes[index]),
                joint_context=joint_contexts[index],
                value_context=value_contexts[index],
            )
        )
    return ArchiveConditionalUnaryLiftResult(
        lifts=tuple(lifted),
        contexts_screened=len(unique_contexts),
        candidates_screened=len(unique_contexts) * candidate_count,
        numeric_candidates=len(sources),
        construction_failures=construction_failures,
        canonical_duplicates=canonical_duplicates,
    )


def lift_archive_conditional_rational_interactions(
    *,
    entries: Sequence[BasisArchiveEntry],
    target: Sequence[float],
    variable_names: Sequence[str],
    geometry_sample_count: int,
    contexts: Sequence[Sequence[int]],
    joint_shortlist_size: int,
    value_shortlist_size: int,
    lambda_value: float = 1.0,
    lambda_gradient: float = 1.0,
) -> ArchiveConditionalRationalLiftResult:
    """Screen ordered ``a/(1+b**2)`` lifts in retained beam states."""

    pool = tuple(entries)
    names = tuple(variable_names)
    y = np.asarray(target, dtype=float).reshape(-1)
    unique_contexts = tuple(
        dict.fromkeys(
            tuple(sorted(int(index) for index in value)) for value in contexts
        )
    )
    if (
        not pool
        or y.size < 1
        or not np.all(np.isfinite(y))
        or not unique_contexts
        or joint_shortlist_size < 1
        or value_shortlist_size < 1
    ):
        raise ValueError("conditional archive rational inputs are invalid")
    if any(
        len(set(context)) != len(context)
        or any(index < 0 or index >= len(pool) for index in context)
        for context in unique_contexts
    ):
        raise ValueError("conditional archive rational contexts are invalid")

    dimension = len(names)
    tolerance = np.finfo(float).eps
    pairs: list[tuple[int, int]] = []
    raw_signatures: list[np.ndarray] = []
    signatures: list[np.ndarray] = []
    values: list[np.ndarray] = []
    for numerator_index, numerator in enumerate(pool):
        numerator_values = np.asarray(numerator.values, dtype=float)
        if numerator_values.shape != y.shape or not np.all(
            np.isfinite(numerator_values)
        ):
            continue
        for denominator_index, denominator_basis in enumerate(pool):
            denominator_values = np.asarray(denominator_basis.values, dtype=float)
            if denominator_values.shape != y.shape or not np.all(
                np.isfinite(denominator_values)
            ):
                continue
            rational_values = numerator_values / (
                1.0 + denominator_values * denominator_values
            )
            try:
                rational_signature = sobolev_safe_rational_signature(
                    numerator.signature,
                    denominator_basis.signature,
                    sample_count=geometry_sample_count,
                    dimension=dimension,
                    lambda_value=lambda_value,
                    lambda_gradient=lambda_gradient,
                )
            except ValueError:
                continue
            signature_norm = float(np.linalg.norm(rational_signature))
            if (
                not np.all(np.isfinite(rational_values))
                or not np.isfinite(signature_norm)
                or signature_norm <= tolerance
            ):
                continue
            pairs.append((numerator_index, denominator_index))
            raw_signatures.append(rational_signature)
            signatures.append(rational_signature / signature_norm)
            values.append(rational_values)

    candidate_count = len(pool) * len(pool)
    if not pairs:
        return ArchiveConditionalRationalLiftResult(
            lifts=(),
            contexts_screened=len(unique_contexts),
            candidates_screened=len(unique_contexts) * candidate_count,
            numeric_candidates=0,
            construction_failures=0,
            canonical_duplicates=0,
        )

    signature_matrix = np.column_stack(signatures)
    value_matrix = np.column_stack(values)
    centered_norms = np.linalg.norm(
        value_matrix - np.mean(value_matrix, axis=0, keepdims=True), axis=0
    )
    count = len(pairs)
    best_joint = np.zeros(count, dtype=float)
    best_joint_gain = np.zeros(count, dtype=float)
    best_joint_correlation = np.zeros(count, dtype=float)
    best_joint_value_gain = np.zeros(count, dtype=float)
    best_value_score = np.zeros(count, dtype=float)
    best_value_gain = np.zeros(count, dtype=float)
    best_value_correlation = np.zeros(count, dtype=float)
    best_value_sobolev_gain = np.zeros(count, dtype=float)
    joint_contexts: list[tuple[int, ...] | None] = [None] * count
    value_contexts: list[tuple[int, ...] | None] = [None] * count

    for context in unique_contexts:
        design = np.column_stack(
            (
                np.ones(y.size, dtype=float),
                *(np.asarray(pool[index].values, dtype=float) for index in context),
            )
        )
        coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
        residual = y - design @ np.round(coefficients, 6)
        residual_norm = float(np.linalg.norm(residual))
        value_span = orthonormal_signature_span(
            [design[:, index] for index in range(design.shape[1])]
        )
        orthogonal_values = value_matrix - value_span @ (value_span.T @ value_matrix)
        orthogonal_norms = np.linalg.norm(orthogonal_values, axis=0)
        correlations = np.zeros(count, dtype=float)
        valid_value = (orthogonal_norms > tolerance) & (residual_norm > tolerance)
        correlations[valid_value] = np.abs(
            orthogonal_values[:, valid_value].T @ residual
        ) / (orthogonal_norms[valid_value] * residual_norm)
        value_gains = np.zeros(count, dtype=float)
        nonzero_centered = centered_norms > tolerance
        value_gains[nonzero_centered] = (
            orthogonal_norms[nonzero_centered] / centered_norms[nonzero_centered]
        )
        correlations = np.clip(correlations, 0.0, 1.0)
        value_gains = np.clip(value_gains, 0.0, 1.0)

        if context:
            signature_span = orthonormal_signature_span(
                [np.asarray(pool[index].signature, dtype=float) for index in context]
            )
            signature_residuals = signature_matrix - signature_span @ (
                signature_span.T @ signature_matrix
            )
            gains = np.clip(np.linalg.norm(signature_residuals, axis=0), 0.0, 1.0)
        else:
            gains = np.ones(count, dtype=float)
        joint_scores = gains * correlations
        value_scores = correlations * value_gains
        joint_better = joint_scores > best_joint + tolerance
        value_better = value_scores > best_value_score + tolerance
        for index in np.flatnonzero(joint_better):
            best_joint[index] = joint_scores[index]
            best_joint_gain[index] = gains[index]
            best_joint_correlation[index] = correlations[index]
            best_joint_value_gain[index] = value_gains[index]
            joint_contexts[index] = context
        for index in np.flatnonzero(value_better):
            best_value_score[index] = value_scores[index]
            best_value_gain[index] = value_gains[index]
            best_value_correlation[index] = correlations[index]
            best_value_sobolev_gain[index] = gains[index]
            value_contexts[index] = context

    joint_order = sorted(
        (index for index in range(count) if best_joint[index] > tolerance),
        key=lambda index: (
            -best_joint[index],
            -best_joint_gain[index],
            -best_joint_correlation[index],
            pool[pairs[index][0]].canonical,
            pool[pairs[index][1]].canonical,
        ),
    )[:joint_shortlist_size]
    value_order = sorted(
        (index for index in range(count) if best_value_score[index] > tolerance),
        key=lambda index: (
            -best_value_score[index],
            -best_value_correlation[index],
            -best_value_gain[index],
            pool[pairs[index][0]].canonical,
            pool[pairs[index][1]].canonical,
        ),
    )[:value_shortlist_size]
    selected: list[int] = []
    lanes: dict[int, list[str]] = {}
    for index in joint_order:
        selected.append(index)
        lanes[index] = ["sobolev_conditional_rational"]
    for index in value_order:
        if index not in lanes:
            selected.append(index)
            lanes[index] = []
        lanes[index].append("value_conditional_rational")

    existing = {entry.canonical for entry in pool}
    lifted: list[ArchiveConditionalRationalLift] = []
    construction_failures = 0
    canonical_duplicates = 0
    for index in selected:
        numerator_index, denominator_index = pairs[index]
        try:
            rational_expression, symbols = parse_expression(
                "("
                + pool[numerator_index].expression
                + ") / (1 + ("
                + pool[denominator_index].expression
                + ")**2)",
                names,
            )
            terms = tuple(
                term
                for term in decompose_expand_mul(rational_expression, symbols)
                if term.basis.free_symbols
            )
            if len(terms) != 1 or abs(float(terms[0].coefficient)) <= tolerance:
                raise ValueError("rational lift did not remain one structural basis")
            term = terms[0]
            coefficient = float(term.coefficient)
            if term.canonical in existing:
                canonical_duplicates += 1
                continue
            existing.add(term.canonical)
            numerator = pool[numerator_index]
            denominator_basis = pool[denominator_index]
            raw_signature = raw_signatures[index]
            entry = BasisArchiveEntry(
                canonical=term.canonical,
                expression=to_project_expression_string(term.basis),
                signature=raw_signature / coefficient,
                values=values[index] / coefficient,
                term_norm=float(np.linalg.norm(raw_signature / coefficient)),
                source_coefficient=1.0,
                source_amplitude=float(numerator.source_amplitude),
                source_base_reward=max(
                    numerator.source_base_reward,
                    denominator_basis.source_base_reward,
                ),
                source_generation=max(
                    numerator.source_generation,
                    denominator_basis.source_generation,
                ),
                source_base_rank=min(
                    numerator.source_base_rank,
                    denominator_basis.source_base_rank,
                ),
                source_candidate_id=min(
                    numerator.source_candidate_id,
                    denominator_basis.source_candidate_id,
                ),
            )
        except (TypeError, ValueError, KeyError):
            construction_failures += 1
            continue
        joint_selected = "sobolev_conditional_rational" in lanes[index]
        lifted.append(
            ArchiveConditionalRationalLift(
                entry=entry,
                numerator_index=numerator_index,
                denominator_index=denominator_index,
                sobolev_gain=float(
                    best_joint_gain[index]
                    if joint_selected
                    else best_value_sobolev_gain[index]
                ),
                target_correlation=float(
                    best_joint_correlation[index]
                    if joint_selected
                    else best_value_correlation[index]
                ),
                value_residual_gain=float(
                    best_joint_value_gain[index]
                    if joint_selected
                    else best_value_gain[index]
                ),
                joint_score=float(best_joint[index]),
                selection_lanes=tuple(lanes[index]),
                joint_context=joint_contexts[index],
                value_context=value_contexts[index],
            )
        )
    return ArchiveConditionalRationalLiftResult(
        lifts=tuple(lifted),
        contexts_screened=len(unique_contexts),
        candidates_screened=len(unique_contexts) * candidate_count,
        numeric_candidates=len(pairs),
        construction_failures=construction_failures,
        canonical_duplicates=canonical_duplicates,
    )


def lift_archive_conditional_affine_unary_interactions(
    *,
    entries: Sequence[BasisArchiveEntry],
    target: Sequence[float],
    variable_names: Sequence[str],
    geometry_sample_count: int,
    contexts: Sequence[Sequence[int]],
    source_indices: Sequence[int] | None = None,
    transforms: Sequence[str],
    joint_shortlist_size: int,
    value_shortlist_size: int,
    lambda_value: float = 1.0,
    lambda_gradient: float = 1.0,
) -> ArchiveConditionalAffineUnaryLiftResult:
    """Screen ``f(a+b)`` and ``f(a-b)`` against retained beam states."""

    pool = tuple(entries)
    names = tuple(variable_names)
    operations = tuple(dict.fromkeys(str(value) for value in transforms))
    y = np.asarray(target, dtype=float).reshape(-1)
    unique_contexts = tuple(
        dict.fromkeys(
            tuple(sorted(int(index) for index in value)) for value in contexts
        )
    )
    if (
        not pool
        or y.size < 1
        or not np.all(np.isfinite(y))
        or not unique_contexts
        or not operations
        or any(value not in {"sin", "cos", "tanh"} for value in operations)
        or joint_shortlist_size < 1
        or value_shortlist_size < 1
    ):
        raise ValueError("conditional archive affine unary inputs are invalid")
    if any(
        len(set(context)) != len(context)
        or any(index < 0 or index >= len(pool) for index in context)
        for context in unique_contexts
    ):
        raise ValueError("conditional archive affine unary contexts are invalid")

    dimension = len(names)
    tolerance = np.finfo(float).eps
    source_positions = (
        tuple(range(len(pool)))
        if source_indices is None
        else tuple(dict.fromkeys(int(index) for index in source_indices))
    )
    if not source_positions or any(
        index < 0 or index >= len(pool) for index in source_positions
    ):
        raise ValueError("conditional archive affine unary sources are invalid")
    pairs: list[tuple[int, int, int, str]] = []
    raw_signatures: list[np.ndarray] = []
    signatures: list[np.ndarray] = []
    values: list[np.ndarray] = []
    for left_offset, left_index in enumerate(source_positions):
        left = pool[left_index]
        left_values = np.asarray(left.values, dtype=float)
        if left_values.shape != y.shape or not np.all(np.isfinite(left_values)):
            continue
        for right_index in source_positions[left_offset:]:
            right = pool[right_index]
            right_values = np.asarray(right.values, dtype=float)
            if right_values.shape != y.shape or not np.all(np.isfinite(right_values)):
                continue
            for sign in (1, -1):
                if sign < 0 and left_index == right_index:
                    continue
                affine_values = left_values + float(sign) * right_values
                for transform in operations:
                    if transform == "sin":
                        transformed_values = np.sin(affine_values)
                    elif transform == "cos":
                        transformed_values = np.cos(affine_values)
                    else:
                        transformed_values = np.tanh(affine_values)
                    try:
                        transformed_signature = sobolev_affine_unary_signature(
                            left.signature,
                            right.signature,
                            sign=sign,
                            transform=transform,
                            sample_count=geometry_sample_count,
                            dimension=dimension,
                            lambda_value=lambda_value,
                            lambda_gradient=lambda_gradient,
                        )
                    except ValueError:
                        continue
                    signature_norm = float(np.linalg.norm(transformed_signature))
                    if (
                        not np.all(np.isfinite(transformed_values))
                        or not np.isfinite(signature_norm)
                        or signature_norm <= tolerance
                    ):
                        continue
                    pairs.append((left_index, right_index, sign, transform))
                    raw_signatures.append(transformed_signature)
                    signatures.append(transformed_signature / signature_norm)
                    values.append(transformed_values)

    candidate_count = len(source_positions) * len(source_positions) * len(operations)
    if not pairs:
        return ArchiveConditionalAffineUnaryLiftResult(
            lifts=(),
            contexts_screened=len(unique_contexts),
            candidates_screened=len(unique_contexts) * candidate_count,
            numeric_candidates=0,
            construction_failures=0,
            canonical_duplicates=0,
        )

    signature_matrix = np.column_stack(signatures)
    value_matrix = np.column_stack(values)
    centered_norms = np.linalg.norm(
        value_matrix - np.mean(value_matrix, axis=0, keepdims=True), axis=0
    )
    count = len(pairs)
    best_joint = np.zeros(count, dtype=float)
    best_joint_gain = np.zeros(count, dtype=float)
    best_joint_correlation = np.zeros(count, dtype=float)
    best_joint_value_gain = np.zeros(count, dtype=float)
    best_value_score = np.zeros(count, dtype=float)
    best_value_gain = np.zeros(count, dtype=float)
    best_value_correlation = np.zeros(count, dtype=float)
    best_value_sobolev_gain = np.zeros(count, dtype=float)
    joint_contexts: list[tuple[int, ...] | None] = [None] * count
    value_contexts: list[tuple[int, ...] | None] = [None] * count

    for context in unique_contexts:
        design = np.column_stack(
            (
                np.ones(y.size, dtype=float),
                *(np.asarray(pool[index].values, dtype=float) for index in context),
            )
        )
        coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
        residual = y - design @ np.round(coefficients, 6)
        residual_norm = float(np.linalg.norm(residual))
        value_span = orthonormal_signature_span(
            [design[:, index] for index in range(design.shape[1])]
        )
        orthogonal_values = value_matrix - value_span @ (value_span.T @ value_matrix)
        orthogonal_norms = np.linalg.norm(orthogonal_values, axis=0)
        correlations = np.zeros(count, dtype=float)
        valid_value = (orthogonal_norms > tolerance) & (residual_norm > tolerance)
        correlations[valid_value] = np.abs(
            orthogonal_values[:, valid_value].T @ residual
        ) / (orthogonal_norms[valid_value] * residual_norm)
        value_gains = np.zeros(count, dtype=float)
        nonzero_centered = centered_norms > tolerance
        value_gains[nonzero_centered] = (
            orthogonal_norms[nonzero_centered] / centered_norms[nonzero_centered]
        )
        correlations = np.clip(correlations, 0.0, 1.0)
        value_gains = np.clip(value_gains, 0.0, 1.0)

        if context:
            signature_span = orthonormal_signature_span(
                [np.asarray(pool[index].signature, dtype=float) for index in context]
            )
            signature_residuals = signature_matrix - signature_span @ (
                signature_span.T @ signature_matrix
            )
            gains = np.clip(np.linalg.norm(signature_residuals, axis=0), 0.0, 1.0)
        else:
            gains = np.ones(count, dtype=float)
        joint_scores = gains * correlations
        value_scores = correlations * value_gains
        joint_better = joint_scores > best_joint + tolerance
        value_better = value_scores > best_value_score + tolerance
        for index in np.flatnonzero(joint_better):
            best_joint[index] = joint_scores[index]
            best_joint_gain[index] = gains[index]
            best_joint_correlation[index] = correlations[index]
            best_joint_value_gain[index] = value_gains[index]
            joint_contexts[index] = context
        for index in np.flatnonzero(value_better):
            best_value_score[index] = value_scores[index]
            best_value_gain[index] = value_gains[index]
            best_value_correlation[index] = correlations[index]
            best_value_sobolev_gain[index] = gains[index]
            value_contexts[index] = context

    joint_order = sorted(
        (index for index in range(count) if best_joint[index] > tolerance),
        key=lambda index: (
            -best_joint[index],
            -best_joint_gain[index],
            -best_joint_correlation[index],
            pairs[index][3],
            -pairs[index][2],
            pool[pairs[index][0]].canonical,
            pool[pairs[index][1]].canonical,
        ),
    )[:joint_shortlist_size]
    value_order = sorted(
        (index for index in range(count) if best_value_score[index] > tolerance),
        key=lambda index: (
            -best_value_score[index],
            -best_value_correlation[index],
            -best_value_gain[index],
            pairs[index][3],
            -pairs[index][2],
            pool[pairs[index][0]].canonical,
            pool[pairs[index][1]].canonical,
        ),
    )[:value_shortlist_size]
    selected: list[int] = []
    lanes: dict[int, list[str]] = {}
    for index in joint_order:
        selected.append(index)
        lanes[index] = ["sobolev_conditional_affine_unary"]
    for index in value_order:
        if index not in lanes:
            selected.append(index)
            lanes[index] = []
        lanes[index].append("value_conditional_affine_unary")

    symbolic_operations = {"sin": sp.sin, "cos": sp.cos, "tanh": sp.tanh}
    existing = {entry.canonical for entry in pool}
    lifted: list[ArchiveConditionalAffineUnaryLift] = []
    construction_failures = 0
    canonical_duplicates = 0
    for index in selected:
        left_index, right_index, sign, transform = pairs[index]
        try:
            operator = "+" if sign > 0 else "-"
            affine_expression, symbols = parse_expression(
                "("
                + pool[left_index].expression
                + ") "
                + operator
                + " ("
                + pool[right_index].expression
                + ")",
                names,
            )
            terms = tuple(
                term
                for term in decompose_expand_mul(
                    symbolic_operations[transform](affine_expression), symbols
                )
                if term.basis.free_symbols
            )
            if len(terms) != 1 or abs(float(terms[0].coefficient)) <= tolerance:
                raise ValueError(
                    "affine unary lift did not remain one structural basis"
                )
            term = terms[0]
            coefficient = float(term.coefficient)
            if term.canonical in existing:
                canonical_duplicates += 1
                continue
            existing.add(term.canonical)
            left = pool[left_index]
            right = pool[right_index]
            raw_signature = raw_signatures[index]
            entry = BasisArchiveEntry(
                canonical=term.canonical,
                expression=to_project_expression_string(term.basis),
                signature=raw_signature / coefficient,
                values=values[index] / coefficient,
                term_norm=float(np.linalg.norm(raw_signature / coefficient)),
                source_coefficient=1.0,
                source_amplitude=max(
                    float(left.source_amplitude), float(right.source_amplitude)
                ),
                source_base_reward=max(
                    left.source_base_reward, right.source_base_reward
                ),
                source_generation=max(left.source_generation, right.source_generation),
                source_base_rank=min(left.source_base_rank, right.source_base_rank),
                source_candidate_id=min(
                    left.source_candidate_id, right.source_candidate_id
                ),
            )
        except (TypeError, ValueError, KeyError):
            construction_failures += 1
            continue
        joint_selected = "sobolev_conditional_affine_unary" in lanes[index]
        lifted.append(
            ArchiveConditionalAffineUnaryLift(
                entry=entry,
                left_index=left_index,
                right_index=right_index,
                sign=sign,
                transform=transform,
                sobolev_gain=float(
                    best_joint_gain[index]
                    if joint_selected
                    else best_value_sobolev_gain[index]
                ),
                target_correlation=float(
                    best_joint_correlation[index]
                    if joint_selected
                    else best_value_correlation[index]
                ),
                value_residual_gain=float(
                    best_joint_value_gain[index]
                    if joint_selected
                    else best_value_gain[index]
                ),
                joint_score=float(best_joint[index]),
                selection_lanes=tuple(lanes[index]),
                joint_context=joint_contexts[index],
                value_context=value_contexts[index],
            )
        )
    return ArchiveConditionalAffineUnaryLiftResult(
        lifts=tuple(lifted),
        contexts_screened=len(unique_contexts),
        candidates_screened=len(unique_contexts) * candidate_count,
        numeric_candidates=len(pairs),
        construction_failures=construction_failures,
        canonical_duplicates=canonical_duplicates,
    )


@dataclass(frozen=True)
class _ConditionalNumericScreen:
    selected: tuple[int, ...]
    lanes: tuple[tuple[str, ...], ...]
    joint_scores: np.ndarray
    joint_gains: np.ndarray
    joint_correlations: np.ndarray
    joint_value_gains: np.ndarray
    value_gains: np.ndarray
    value_correlations: np.ndarray
    value_sobolev_gains: np.ndarray
    joint_contexts: tuple[tuple[int, ...] | None, ...]
    value_contexts: tuple[tuple[int, ...] | None, ...]


def _screen_conditional_numeric_bases(
    *,
    normalized_signatures: np.ndarray,
    value_matrix: np.ndarray,
    context_entries: Sequence[BasisArchiveEntry],
    contexts: Sequence[tuple[int, ...]],
    target: np.ndarray,
    joint_shortlist_size: int,
    value_shortlist_size: int,
    joint_lane: str,
    value_lane: str,
    tie_keys: Sequence[tuple[object, ...]],
) -> _ConditionalNumericScreen:
    """Rank a numeric basis pool in retained value and Sobolev contexts."""

    signatures = np.asarray(normalized_signatures, dtype=float)
    values = np.asarray(value_matrix, dtype=float)
    y = np.asarray(target, dtype=float).reshape(-1)
    count = signatures.shape[1]
    tolerance = np.finfo(float).eps
    centered_norms = np.linalg.norm(
        values - np.mean(values, axis=0, keepdims=True), axis=0
    )
    best_joint = np.zeros(count, dtype=float)
    best_joint_gain = np.zeros(count, dtype=float)
    best_joint_correlation = np.zeros(count, dtype=float)
    best_joint_value_gain = np.zeros(count, dtype=float)
    best_value_score = np.zeros(count, dtype=float)
    best_value_gain = np.zeros(count, dtype=float)
    best_value_correlation = np.zeros(count, dtype=float)
    best_value_sobolev_gain = np.zeros(count, dtype=float)
    joint_contexts: list[tuple[int, ...] | None] = [None] * count
    value_contexts: list[tuple[int, ...] | None] = [None] * count

    for context in contexts:
        design = np.column_stack(
            (
                np.ones(y.size, dtype=float),
                *(
                    np.asarray(context_entries[index].values, dtype=float)
                    for index in context
                ),
            )
        )
        coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
        residual = y - design @ np.round(coefficients, 6)
        residual_norm = float(np.linalg.norm(residual))
        value_span = orthonormal_signature_span(
            [design[:, index] for index in range(design.shape[1])]
        )
        orthogonal_values = values - value_span @ (value_span.T @ values)
        orthogonal_norms = np.linalg.norm(orthogonal_values, axis=0)
        correlations = np.zeros(count, dtype=float)
        valid = (orthogonal_norms > tolerance) & (residual_norm > tolerance)
        correlations[valid] = np.abs(orthogonal_values[:, valid].T @ residual) / (
            orthogonal_norms[valid] * residual_norm
        )
        value_gains = np.zeros(count, dtype=float)
        nonzero_centered = centered_norms > tolerance
        value_gains[nonzero_centered] = (
            orthogonal_norms[nonzero_centered] / centered_norms[nonzero_centered]
        )
        correlations = np.clip(correlations, 0.0, 1.0)
        value_gains = np.clip(value_gains, 0.0, 1.0)

        if context:
            signature_span = orthonormal_signature_span(
                [
                    np.asarray(context_entries[index].signature, dtype=float)
                    for index in context
                ]
            )
            residual_signatures = signatures - signature_span @ (
                signature_span.T @ signatures
            )
            gains = np.clip(np.linalg.norm(residual_signatures, axis=0), 0.0, 1.0)
        else:
            gains = np.ones(count, dtype=float)
        joint_scores = gains * correlations
        value_scores = value_gains * correlations
        joint_better = joint_scores > best_joint + tolerance
        value_better = value_scores > best_value_score + tolerance
        for index in np.flatnonzero(joint_better):
            best_joint[index] = joint_scores[index]
            best_joint_gain[index] = gains[index]
            best_joint_correlation[index] = correlations[index]
            best_joint_value_gain[index] = value_gains[index]
            joint_contexts[index] = context
        for index in np.flatnonzero(value_better):
            best_value_score[index] = value_scores[index]
            best_value_gain[index] = value_gains[index]
            best_value_correlation[index] = correlations[index]
            best_value_sobolev_gain[index] = gains[index]
            value_contexts[index] = context

    joint_order = sorted(
        (index for index in range(count) if best_joint[index] > tolerance),
        key=lambda index: (
            -best_joint[index],
            -best_joint_gain[index],
            -best_joint_correlation[index],
            tie_keys[index],
        ),
    )[:joint_shortlist_size]
    value_order = sorted(
        (index for index in range(count) if best_value_score[index] > tolerance),
        key=lambda index: (
            -best_value_score[index],
            -best_value_correlation[index],
            -best_value_gain[index],
            tie_keys[index],
        ),
    )[:value_shortlist_size]
    selected: list[int] = []
    lane_lists: list[list[str]] = [[] for _ in range(count)]
    for index in joint_order:
        selected.append(index)
        lane_lists[index].append(joint_lane)
    for index in value_order:
        if index not in selected:
            selected.append(index)
        lane_lists[index].append(value_lane)
    return _ConditionalNumericScreen(
        selected=tuple(selected),
        lanes=tuple(tuple(value) for value in lane_lists),
        joint_scores=best_joint,
        joint_gains=best_joint_gain,
        joint_correlations=best_joint_correlation,
        joint_value_gains=best_joint_value_gain,
        value_gains=best_value_gain,
        value_correlations=best_value_correlation,
        value_sobolev_gains=best_value_sobolev_gain,
        joint_contexts=tuple(joint_contexts),
        value_contexts=tuple(value_contexts),
    )


def lift_archive_conditional_radial_interactions(
    *,
    entries: Sequence[BasisArchiveEntry],
    target: Sequence[float],
    variable_names: Sequence[str],
    geometry_sample_count: int,
    contexts: Sequence[Sequence[int]],
    source_indices: Sequence[int] | None = None,
    component_joint_shortlist_size: int,
    component_value_shortlist_size: int,
    joint_shortlist_size: int,
    value_shortlist_size: int,
    lambda_value: float = 1.0,
    lambda_gradient: float = 1.0,
) -> ArchiveConditionalRadialLiftResult:
    """Build safe two-component Euclidean norms through a two-stage screen."""

    pool = tuple(entries)
    names = tuple(variable_names)
    y = np.asarray(target, dtype=float).reshape(-1)
    unique_contexts = tuple(
        dict.fromkeys(
            tuple(sorted(int(index) for index in value)) for value in contexts
        )
    )
    if (
        not pool
        or y.size < 1
        or not np.all(np.isfinite(y))
        or not unique_contexts
        or min(
            component_joint_shortlist_size,
            component_value_shortlist_size,
            joint_shortlist_size,
            value_shortlist_size,
        )
        < 1
    ):
        raise ValueError("conditional archive radial inputs are invalid")
    if any(
        len(set(context)) != len(context)
        or any(index < 0 or index >= len(pool) for index in context)
        for context in unique_contexts
    ):
        raise ValueError("conditional archive radial contexts are invalid")

    dimension = len(names)
    tolerance = np.finfo(float).eps
    source_positions = (
        tuple(range(len(pool)))
        if source_indices is None
        else tuple(dict.fromkeys(int(index) for index in source_indices))
    )
    if not source_positions or any(
        index < 0 or index >= len(pool) for index in source_positions
    ):
        raise ValueError("conditional archive radial sources are invalid")
    components: list[tuple[int, int, int]] = []
    component_signatures: list[np.ndarray] = []
    component_square_signatures: list[np.ndarray] = []
    component_values: list[np.ndarray] = []
    component_square_values: list[np.ndarray] = []
    component_keys: list[tuple[object, ...]] = []
    for left_offset, left_index in enumerate(source_positions):
        left = pool[left_index]
        left_values = np.asarray(left.values, dtype=float)
        if left_values.shape != y.shape or not np.all(np.isfinite(left_values)):
            continue
        descriptors = [(left_index, -1, 0)]
        descriptors.extend(
            (left_index, right_index, sign)
            for right_index in source_positions[left_offset + 1 :]
            for sign in (1, -1)
        )
        for _, right_index, sign in descriptors:
            if right_index < 0:
                values = left_values
                signature = np.asarray(left.signature, dtype=float)
                key = (left.canonical, "single")
            else:
                right = pool[right_index]
                right_values = np.asarray(right.values, dtype=float)
                if right_values.shape != y.shape or not np.all(
                    np.isfinite(right_values)
                ):
                    continue
                values = left_values + float(sign) * right_values
                signature = np.asarray(left.signature, dtype=float) + float(
                    sign
                ) * np.asarray(right.signature, dtype=float)
                key = (
                    left.canonical,
                    "sum" if sign > 0 else "difference",
                    right.canonical,
                )
            try:
                square_signature = sobolev_product_signature(
                    signature,
                    signature,
                    sample_count=geometry_sample_count,
                    dimension=dimension,
                    lambda_value=lambda_value,
                    lambda_gradient=lambda_gradient,
                )
            except ValueError:
                continue
            square_norm = float(np.linalg.norm(square_signature))
            square_values = values * values
            if (
                not np.all(np.isfinite(square_values))
                or not np.isfinite(square_norm)
                or square_norm <= tolerance
            ):
                continue
            components.append((left_index, right_index, sign))
            component_signatures.append(signature)
            component_square_signatures.append(square_signature / square_norm)
            component_values.append(values)
            component_square_values.append(square_values)
            component_keys.append(key)

    if not components:
        return ArchiveConditionalRadialLiftResult(
            lifts=(),
            contexts_screened=len(unique_contexts),
            component_candidates=0,
            component_states_screened=0,
            components_selected=0,
            radial_candidates=0,
            radial_states_screened=0,
            construction_failures=0,
            canonical_duplicates=0,
        )

    component_screen = _screen_conditional_numeric_bases(
        normalized_signatures=np.column_stack(component_square_signatures),
        value_matrix=np.column_stack(component_square_values),
        context_entries=pool,
        contexts=unique_contexts,
        target=y,
        joint_shortlist_size=component_joint_shortlist_size,
        value_shortlist_size=component_value_shortlist_size,
        joint_lane="sobolev_radial_component",
        value_lane="value_radial_component",
        tie_keys=component_keys,
    )
    selected_components = component_screen.selected
    radial_pairs: list[tuple[int, int]] = []
    raw_radial_signatures: list[np.ndarray] = []
    normalized_radial_signatures: list[np.ndarray] = []
    radial_values: list[np.ndarray] = []
    radial_keys: list[tuple[object, ...]] = []
    for offset, first_index in enumerate(selected_components):
        for second_index in selected_components[offset:]:
            first_values = component_values[first_index]
            second_values = component_values[second_index]
            values = np.sqrt(
                first_values * first_values + second_values * second_values
            )
            try:
                signature = sobolev_radial_signature(
                    component_signatures[first_index],
                    component_signatures[second_index],
                    sample_count=geometry_sample_count,
                    dimension=dimension,
                    lambda_value=lambda_value,
                    lambda_gradient=lambda_gradient,
                )
            except ValueError:
                continue
            signature_norm = float(np.linalg.norm(signature))
            if (
                not np.all(np.isfinite(values))
                or not np.isfinite(signature_norm)
                or signature_norm <= tolerance
            ):
                continue
            radial_pairs.append((first_index, second_index))
            raw_radial_signatures.append(signature)
            normalized_radial_signatures.append(signature / signature_norm)
            radial_values.append(values)
            radial_keys.append(
                (component_keys[first_index], component_keys[second_index])
            )

    if not radial_pairs:
        return ArchiveConditionalRadialLiftResult(
            lifts=(),
            contexts_screened=len(unique_contexts),
            component_candidates=len(components),
            component_states_screened=len(unique_contexts) * len(components),
            components_selected=len(selected_components),
            radial_candidates=0,
            radial_states_screened=0,
            construction_failures=0,
            canonical_duplicates=0,
        )

    radial_screen = _screen_conditional_numeric_bases(
        normalized_signatures=np.column_stack(normalized_radial_signatures),
        value_matrix=np.column_stack(radial_values),
        context_entries=pool,
        contexts=unique_contexts,
        target=y,
        joint_shortlist_size=joint_shortlist_size,
        value_shortlist_size=value_shortlist_size,
        joint_lane="sobolev_conditional_radial",
        value_lane="value_conditional_radial",
        tie_keys=radial_keys,
    )

    def component_expression(descriptor: tuple[int, int, int]) -> str:
        left_index, right_index, sign = descriptor
        if right_index < 0:
            return "(" + pool[left_index].expression + ")"
        operator = "+" if sign > 0 else "-"
        return (
            "("
            + pool[left_index].expression
            + ") "
            + operator
            + " ("
            + pool[right_index].expression
            + ")"
        )

    existing = {entry.canonical for entry in pool}
    lifted: list[ArchiveConditionalRadialLift] = []
    construction_failures = 0
    canonical_duplicates = 0
    for radial_index in radial_screen.selected:
        first_index, second_index = radial_pairs[radial_index]
        first_component = components[first_index]
        second_component = components[second_index]
        try:
            radicand, symbols = parse_expression(
                "("
                + component_expression(first_component)
                + ")**2 + ("
                + component_expression(second_component)
                + ")**2",
                names,
            )
            terms = tuple(
                term
                for term in decompose_expand_mul(sp.sqrt(radicand), symbols)
                if term.basis.free_symbols
            )
            if len(terms) != 1 or abs(float(terms[0].coefficient)) <= tolerance:
                raise ValueError("radial lift did not remain one structural basis")
            term = terms[0]
            coefficient = float(term.coefficient)
            if term.canonical in existing:
                canonical_duplicates += 1
                continue
            existing.add(term.canonical)
            source_indices = {
                index
                for descriptor in (first_component, second_component)
                for index in descriptor[:2]
                if index >= 0
            }
            sources = [pool[index] for index in sorted(source_indices)]
            raw_signature = raw_radial_signatures[radial_index]
            entry = BasisArchiveEntry(
                canonical=term.canonical,
                expression=to_project_expression_string(term.basis),
                signature=raw_signature / coefficient,
                values=radial_values[radial_index] / coefficient,
                term_norm=float(np.linalg.norm(raw_signature / coefficient)),
                source_coefficient=1.0,
                source_amplitude=max(
                    float(source.source_amplitude) for source in sources
                ),
                source_base_reward=max(source.source_base_reward for source in sources),
                source_generation=max(source.source_generation for source in sources),
                source_base_rank=min(source.source_base_rank for source in sources),
                source_candidate_id=min(
                    source.source_candidate_id for source in sources
                ),
            )
        except (TypeError, ValueError, KeyError):
            construction_failures += 1
            continue
        lanes = radial_screen.lanes[radial_index]
        joint_selected = "sobolev_conditional_radial" in lanes
        lifted.append(
            ArchiveConditionalRadialLift(
                entry=entry,
                first_component=first_component,
                second_component=second_component,
                sobolev_gain=float(
                    radial_screen.joint_gains[radial_index]
                    if joint_selected
                    else radial_screen.value_sobolev_gains[radial_index]
                ),
                target_correlation=float(
                    radial_screen.joint_correlations[radial_index]
                    if joint_selected
                    else radial_screen.value_correlations[radial_index]
                ),
                value_residual_gain=float(
                    radial_screen.joint_value_gains[radial_index]
                    if joint_selected
                    else radial_screen.value_gains[radial_index]
                ),
                joint_score=float(radial_screen.joint_scores[radial_index]),
                selection_lanes=lanes,
                joint_context=radial_screen.joint_contexts[radial_index],
                value_context=radial_screen.value_contexts[radial_index],
            )
        )
    return ArchiveConditionalRadialLiftResult(
        lifts=tuple(lifted),
        contexts_screened=len(unique_contexts),
        component_candidates=len(components),
        component_states_screened=len(unique_contexts) * len(components),
        components_selected=len(selected_components),
        radial_candidates=len(radial_pairs),
        radial_states_screened=len(unique_contexts) * len(radial_pairs),
        construction_failures=construction_failures,
        canonical_duplicates=canonical_duplicates,
    )
