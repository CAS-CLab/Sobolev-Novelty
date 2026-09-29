"""Multi-axis inverse-square construction in Sobolev geometry.

The export-side plug-in constructs compact expressions of the form

``product(amplitude_axes) / sum((x_i + sign_i*x_j)**2)``.

Radial pairs use disjoint coordinates and amplitude axes are selected from the
remaining coordinates.  Complete numerator/radial combinations are screened
before a bounded value-and-gradient lane is formed.  The Sobolev score measures
candidate gain, reciprocal coupling, and the within-candidate novelty of every
squared radial term.  Target derivatives and benchmark metadata are never used.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import combinations, product
from typing import Iterator, Sequence

import numpy as np

from .sn_archive_interactions import sobolev_product_signature
from .sn_basis_archive import BasisArchiveEntry
from .sn_population_coverage import orthonormal_signature_span
from .sn_shared_denominator import _signature_gain, sobolev_quotient_signature


@dataclass(frozen=True)
class MultiAxisInverseSquareProposal:
    """One ordinary-GP expression retained by a value or Sobolev lane."""

    expression: str
    numerator_axes: tuple[int, ...]
    radial_pairs: tuple[tuple[int, int], ...]
    radial_signs: tuple[int, ...]
    fitted_coefficients: tuple[float, float]
    training_r2: float
    prediction_correlation: float
    candidate_sobolev_gain: float
    denominator_sobolev_gain: float
    mean_term_novelty: float
    min_term_novelty: float
    reciprocal_coupling_gain: float
    joint_score: float
    selection_lanes: tuple[str, ...]


@dataclass(frozen=True)
class MultiAxisInverseSquareResult:
    """Compact audit result for one complete multi-axis enumeration."""

    proposals: tuple[MultiAxisInverseSquareProposal, ...]
    pair_atoms: int
    radial_shapes: int
    candidates_screened: int
    denominator_rejections: int
    value_pool: int
    sobolev_candidates: int
    numeric_failures: int


@dataclass(frozen=True)
class _PairAtom:
    left_axis: int
    right_axis: int
    sign: int
    expression: str
    square_values: np.ndarray
    square_signature: np.ndarray


@dataclass(frozen=True)
class _ValueState:
    expression: str
    numerator_axes: tuple[int, ...]
    atoms: tuple[_PairAtom, ...]
    coefficients: tuple[float, float]
    training_r2: float
    prediction_correlation: float


@dataclass(frozen=True)
class _ScoredState:
    value_state: _ValueState
    candidate_sobolev_gain: float
    denominator_sobolev_gain: float
    mean_term_novelty: float
    min_term_novelty: float
    reciprocal_coupling_gain: float
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


def _perfect_matchings(axes: Sequence[int]) -> Iterator[tuple[tuple[int, int], ...]]:
    remaining = tuple(int(axis) for axis in axes)
    if not remaining:
        yield ()
        return
    left = remaining[0]
    for position in range(1, len(remaining)):
        right = remaining[position]
        tail_axes = remaining[1:position] + remaining[position + 1 :]
        for tail in _perfect_matchings(tail_axes):
            yield ((left, right), *tail)


def _product_signature(
    direct: Sequence[BasisArchiveEntry],
    axes: Sequence[int],
    *,
    constant: np.ndarray,
    sample_count: int,
    dimension: int,
    lambda_value: float,
    lambda_gradient: float,
) -> np.ndarray:
    selected = tuple(int(axis) for axis in axes)
    if not selected:
        return constant
    signature = np.asarray(direct[selected[0]].signature, dtype=float).copy()
    for axis in selected[1:]:
        signature = sobolev_product_signature(
            signature,
            direct[axis].signature,
            sample_count=sample_count,
            dimension=dimension,
            lambda_value=lambda_value,
            lambda_gradient=lambda_gradient,
        )
    return signature


def _value_key(state: _ValueState) -> tuple[float, float, str]:
    return (-state.training_r2, -state.prediction_correlation, state.expression)


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


def lift_direct_multiaxis_inverse_squares(
    *,
    direct_entries: Sequence[BasisArchiveEntry],
    target: Sequence[float],
    variable_names: Sequence[str],
    geometry_sample_count: int,
    pair_signs: Sequence[int],
    radial_term_counts: Sequence[int],
    max_numerator_degree: int,
    value_pool_size: int,
    shortlist_size: int,
    proposal_limit: int,
    protected_epsilon: float = 1e-6,
    lambda_value: float = 1.0,
    lambda_gradient: float = 1.0,
) -> MultiAxisInverseSquareResult:
    """Enumerate complete disjoint-pair radial candidates before screening."""

    direct = tuple(direct_entries)
    names = tuple(str(value) for value in variable_names)
    y = np.asarray(target, dtype=float).reshape(-1)
    dimension = len(names)
    signature_size = geometry_sample_count * (dimension + 1)
    signs = tuple(dict.fromkeys(int(value) for value in pair_signs))
    term_counts = tuple(
        sorted(dict.fromkeys(int(value) for value in radial_term_counts))
    )
    applicable_counts = tuple(
        count for count in term_counts if count >= 2 and 2 * count <= dimension
    )
    if (
        len(direct) != dimension
        or dimension < 4
        or y.size < 2
        or not np.all(np.isfinite(y))
        or float(np.var(y)) <= np.finfo(float).eps
        or geometry_sample_count < 1
        or not signs
        or any(value not in {-1, 1} for value in signs)
        or not term_counts
        or any(value < 2 for value in term_counts)
        or not applicable_counts
        or max_numerator_degree < 0
        or value_pool_size < shortlist_size
        or shortlist_size < 1
        or proposal_limit < 1
        or protected_epsilon <= 0.0
        or lambda_value <= 0.0
        or (dimension and lambda_gradient <= 0.0)
    ):
        raise ValueError("multi-axis inverse-square inputs are invalid")
    if any(
        np.asarray(entry.values).shape != y.shape
        or np.asarray(entry.signature).shape != (signature_size,)
        or not np.all(np.isfinite(entry.values))
        or not np.all(np.isfinite(entry.signature))
        for entry in direct
    ):
        raise ValueError("multi-axis inverse-square entries are not aligned")

    atom_map: dict[tuple[int, int, int], _PairAtom] = {}
    numeric_failures = 0
    for left_axis, right_axis in combinations(range(dimension), 2):
        left = direct[left_axis]
        right = direct[right_axis]
        for sign in signs:
            difference_values = np.asarray(left.values, dtype=float) + float(
                sign
            ) * np.asarray(right.values, dtype=float)
            difference_signature = np.asarray(left.signature, dtype=float) + float(
                sign
            ) * np.asarray(right.signature, dtype=float)
            try:
                square_signature = sobolev_product_signature(
                    difference_signature,
                    difference_signature,
                    sample_count=geometry_sample_count,
                    dimension=dimension,
                    lambda_value=lambda_value,
                    lambda_gradient=lambda_gradient,
                )
            except ValueError:
                numeric_failures += 1
                continue
            operator = "+" if sign == 1 else "-"
            atom_map[(left_axis, right_axis, sign)] = _PairAtom(
                left_axis=left_axis,
                right_axis=right_axis,
                sign=sign,
                expression=f"(({names[left_axis]}) {operator} ({names[right_axis]})) ** 2",
                square_values=np.square(difference_values),
                square_signature=square_signature,
            )

    numerator_values: dict[tuple[int, ...], np.ndarray] = {
        (): np.ones(y.size, dtype=float)
    }
    for degree in range(1, min(max_numerator_degree, dimension) + 1):
        for axes in combinations(range(dimension), degree):
            numerator_values[axes] = np.prod(
                np.column_stack(
                    tuple(np.asarray(direct[axis].values, dtype=float) for axis in axes)
                ),
                axis=1,
            )

    target_variance = float(np.var(y))
    centered_target = y - float(np.mean(y))
    target_norm = float(np.linalg.norm(centered_target))
    retained: list[_ValueState] = []
    prune_batch_size = max(4096, 4 * value_pool_size)
    radial_shapes = 0
    candidates_screened = 0
    denominator_rejections = 0
    for term_count in applicable_counts:
        for selected_axes in combinations(range(dimension), 2 * term_count):
            selected_set = set(selected_axes)
            complement = tuple(
                axis for axis in range(dimension) if axis not in selected_set
            )
            numerator_axes_options = tuple(
                axes
                for degree in range(0, min(max_numerator_degree, len(complement)) + 1)
                for axes in combinations(complement, degree)
            )
            for pairing in _perfect_matchings(selected_axes):
                for sign_tuple in product(signs, repeat=term_count):
                    atoms = tuple(
                        atom_map[(left, right, sign)]
                        for (left, right), sign in zip(pairing, sign_tuple)
                    )
                    radial_shapes += 1
                    denominator = np.sum(
                        np.column_stack(tuple(atom.square_values for atom in atoms)),
                        axis=1,
                    )
                    if np.any(denominator <= protected_epsilon):
                        denominator_rejections += 1
                        continue
                    denominator_text = " + ".join(atom.expression for atom in atoms)
                    for numerator_axes in numerator_axes_options:
                        candidates_screened += 1
                        with np.errstate(all="ignore"):
                            basis = numerator_values[numerator_axes] / denominator
                        if not np.all(np.isfinite(basis)):
                            numeric_failures += 1
                            continue
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
                        coefficients, training_r2, correlation = fit
                        numerator_text = (
                            "1"
                            if not numerator_axes
                            else " * ".join(names[axis] for axis in numerator_axes)
                        )
                        retained.append(
                            _ValueState(
                                expression=(
                                    f"({numerator_text}) / ({denominator_text})"
                                ),
                                numerator_axes=numerator_axes,
                                atoms=atoms,
                                coefficients=coefficients,
                                training_r2=training_r2,
                                prediction_correlation=correlation,
                            )
                        )
                        if len(retained) >= prune_batch_size:
                            retained.sort(key=_value_key)
                            del retained[value_pool_size:]

    retained.sort(key=_value_key)
    del retained[value_pool_size:]
    constant = _constant_signature(geometry_sample_count, dimension, lambda_value)
    coordinate_span = orthonormal_signature_span(
        tuple(np.asarray(entry.signature, dtype=float) for entry in direct)
    )
    scored: list[_ScoredState] = []
    for state in retained:
        try:
            numerator_signature = _product_signature(
                direct,
                state.numerator_axes,
                constant=constant,
                sample_count=geometry_sample_count,
                dimension=dimension,
                lambda_value=lambda_value,
                lambda_gradient=lambda_gradient,
            )
            term_signatures = tuple(atom.square_signature for atom in state.atoms)
            denominator_signature = np.sum(np.column_stack(term_signatures), axis=1)
            candidate_signature = sobolev_quotient_signature(
                numerator_signature,
                denominator_signature,
                sample_count=geometry_sample_count,
                dimension=dimension,
                protected_epsilon=protected_epsilon,
                lambda_value=lambda_value,
                lambda_gradient=lambda_gradient,
            )
            term_novelties = tuple(
                _signature_gain(
                    signature,
                    orthonormal_signature_span(
                        tuple(
                            other
                            for other_index, other in enumerate(term_signatures)
                            if other_index != index
                        )
                    ),
                )
                for index, signature in enumerate(term_signatures)
            )
            component_span = orthonormal_signature_span(
                (numerator_signature, denominator_signature)
            )
            candidate_gain = _signature_gain(candidate_signature, coordinate_span)
            denominator_gain = _signature_gain(denominator_signature, coordinate_span)
            reciprocal_gain = _signature_gain(candidate_signature, component_span)
            mean_novelty = float(np.mean(term_novelties))
            min_novelty = float(np.min(term_novelties))
        except ValueError:
            numeric_failures += 1
            candidate_gain = 0.0
            denominator_gain = 0.0
            reciprocal_gain = 0.0
            mean_novelty = 0.0
            min_novelty = 0.0
        joint_score = (
            state.prediction_correlation
            * candidate_gain
            * denominator_gain
            * mean_novelty
            * reciprocal_gain
        )
        scored.append(
            _ScoredState(
                value_state=state,
                candidate_sobolev_gain=candidate_gain,
                denominator_sobolev_gain=denominator_gain,
                mean_term_novelty=mean_novelty,
                min_term_novelty=min_novelty,
                reciprocal_coupling_gain=reciprocal_gain,
                joint_score=joint_score,
                selection_lanes=(),
            )
        )

    value_lane = sorted(
        scored,
        key=lambda item: _value_key(item.value_state),
    )[:shortlist_size]
    sobolev_lane = sorted(
        scored,
        key=lambda item: (
            -item.joint_score,
            -item.candidate_sobolev_gain,
            -item.mean_term_novelty,
            -item.reciprocal_coupling_gain,
            -item.value_state.training_r2,
            item.value_state.expression,
        ),
    )[:shortlist_size]
    lanes: dict[int, list[str]] = {}
    for item in value_lane:
        lanes.setdefault(id(item), []).append("value_multiaxis_inverse_square")
    for item in sobolev_lane:
        lanes.setdefault(id(item), []).append("sobolev_multiaxis_inverse_square")
    selected: list[_ScoredState] = []
    seen_ids: set[int] = set()
    for item in (*value_lane, *sobolev_lane):
        if id(item) in seen_ids:
            continue
        seen_ids.add(id(item))
        selected.append(replace(item, selection_lanes=tuple(lanes[id(item)])))

    proposals = [
        MultiAxisInverseSquareProposal(
            expression=item.value_state.expression,
            numerator_axes=item.value_state.numerator_axes,
            radial_pairs=tuple(
                (atom.left_axis, atom.right_axis) for atom in item.value_state.atoms
            ),
            radial_signs=tuple(atom.sign for atom in item.value_state.atoms),
            fitted_coefficients=item.value_state.coefficients,
            training_r2=item.value_state.training_r2,
            prediction_correlation=item.value_state.prediction_correlation,
            candidate_sobolev_gain=item.candidate_sobolev_gain,
            denominator_sobolev_gain=item.denominator_sobolev_gain,
            mean_term_novelty=item.mean_term_novelty,
            min_term_novelty=item.min_term_novelty,
            reciprocal_coupling_gain=item.reciprocal_coupling_gain,
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
    unique: list[MultiAxisInverseSquareProposal] = []
    seen_expressions: set[str] = set()
    for proposal in proposals:
        if proposal.expression in seen_expressions:
            continue
        seen_expressions.add(proposal.expression)
        unique.append(proposal)
        if len(unique) >= proposal_limit:
            break

    return MultiAxisInverseSquareResult(
        proposals=tuple(unique),
        pair_atoms=len(atom_map),
        radial_shapes=radial_shapes,
        candidates_screened=candidates_screened,
        denominator_rejections=denominator_rejections,
        value_pool=len(retained),
        sobolev_candidates=len(scored),
        numeric_failures=numeric_failures,
    )
