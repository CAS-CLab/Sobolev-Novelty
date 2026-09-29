"""Cross-factor affine-unary pursuit screened in Sobolev geometry.

The export-side plug-in constructs ``O(x) * (a + b * I(x))``.  ``O`` is a
bounded direct-coordinate unary or a low-degree monomial ratio, while ``I``
is a unary transform of an affine/product phase core (optionally squared).
Complete outer/inner pairs are value-screened before Sobolev scoring so an
interaction is not discarded merely because either atom has weak marginal
credit.  Target derivatives are never used and the Population GP trajectory
is unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Sequence

import numpy as np

from .sn_archive_interactions import (
    sobolev_product_signature,
    sobolev_unary_signature,
)
from .sn_basis_archive import BasisArchiveEntry
from .sn_population_coverage import orthonormal_signature_span
from .sn_shared_phase_rational import _monomial_ratios
from .sn_sinc_factorization import _phase_cores, _scale_text


@dataclass(frozen=True)
class CrossUnaryAffineProposal:
    """One ordinary-GP expression retained by a value or Sobolev lane."""

    expression: str
    outer_expression: str
    inner_expression: str
    inner_power: int
    context: tuple[int, ...]
    context_expressions: tuple[str, ...]
    training_r2: float
    conditional_correlation: float
    sobolev_gain: float
    joint_score: float
    selection_lanes: tuple[str, ...]


@dataclass(frozen=True)
class CrossUnaryAffineResult:
    """Compact audit result for one bounded cross-factor pursuit."""

    proposals: tuple[CrossUnaryAffineProposal, ...]
    contexts_screened: int
    outer_amplitudes: int
    outer_unary_atoms: int
    inner_phase_cores: int
    inner_atoms: int
    pair_candidates: int
    value_pool: int
    pair_shortlist: int
    numeric_failures: int


@dataclass(frozen=True)
class _Atom:
    expression: str
    values: np.ndarray
    signature: np.ndarray
    power: int


@dataclass(frozen=True)
class _ValueState:
    context: tuple[int, ...]
    outer: _Atom
    inner: _Atom
    training_r2: float
    conditional_correlation: float


@dataclass(frozen=True)
class _ScoredState:
    value_state: _ValueState
    training_r2: float
    conditional_correlation: float
    sobolev_gain: float
    joint_score: float
    selection_lanes: tuple[str, ...]


def _direct_outer_unaries(
    direct: Sequence[BasisArchiveEntry],
    names: Sequence[str],
    scales: Sequence[float],
    transforms: Sequence[str],
    *,
    sample_count: int,
    dimension: int,
    lambda_value: float,
    lambda_gradient: float,
) -> tuple[tuple[_Atom, ...], int]:
    output: list[_Atom] = []
    failures = 0
    for name, entry in zip(names, direct, strict=True):
        for scale in scales:
            source_expression = (
                str(name)
                if np.isclose(scale, 1.0, rtol=0.0, atol=1e-15)
                else f"({_scale_text(scale)}) * ({name})"
            )
            scaled_values = float(scale) * np.asarray(entry.values, dtype=float)
            scaled_signature = float(scale) * np.asarray(entry.signature, dtype=float)
            for transform in transforms:
                try:
                    signature = sobolev_unary_signature(
                        scaled_signature,
                        transform=transform,
                        sample_count=sample_count,
                        dimension=dimension,
                        lambda_value=lambda_value,
                        lambda_gradient=lambda_gradient,
                    )
                except ValueError:
                    failures += 1
                    continue
                values = getattr(np, transform)(scaled_values)
                if not np.all(np.isfinite(values)):
                    failures += 1
                    continue
                output.append(
                    _Atom(
                        expression=f"{transform}({source_expression})",
                        values=values,
                        signature=signature,
                        power=1,
                    )
                )
    return tuple(output), failures


def _outer_atoms(
    direct: Sequence[BasisArchiveEntry],
    names: Sequence[str],
    scales: Sequence[float],
    transforms: Sequence[str],
    *,
    max_numerator_degree: int,
    max_denominator_degree: int,
    sample_count: int,
    dimension: int,
    protected_epsilon: float,
    lambda_value: float,
    lambda_gradient: float,
) -> tuple[tuple[_Atom, ...], int, int, int]:
    amplitudes, failures = _monomial_ratios(
        direct,
        variable_names=names,
        max_numerator_degree=max_numerator_degree,
        max_denominator_degree=max_denominator_degree,
        sample_count=sample_count,
        dimension=dimension,
        protected_epsilon=protected_epsilon,
        lambda_value=lambda_value,
        lambda_gradient=lambda_gradient,
    )
    amplitude_atoms = tuple(
        _Atom(
            expression=item.expression,
            values=np.asarray(item.values, dtype=float),
            signature=np.asarray(item.signature, dtype=float),
            power=1,
        )
        for item in amplitudes
    )
    unary_atoms, unary_failures = _direct_outer_unaries(
        direct,
        names,
        scales,
        transforms,
        sample_count=sample_count,
        dimension=dimension,
        lambda_value=lambda_value,
        lambda_gradient=lambda_gradient,
    )
    unique: dict[str, _Atom] = {}
    for item in (*amplitude_atoms, *unary_atoms):
        unique.setdefault(item.expression, item)
    return (
        tuple(unique.values()),
        len(amplitude_atoms),
        len(unary_atoms),
        failures + unary_failures,
    )


def _inner_atoms(
    direct: Sequence[BasisArchiveEntry],
    names: Sequence[str],
    scales: Sequence[float],
    transforms: Sequence[str],
    powers: Sequence[int],
    *,
    sample_count: int,
    dimension: int,
    lambda_value: float,
    lambda_gradient: float,
) -> tuple[tuple[_Atom, ...], int, int]:
    cores, _, failures = _phase_cores(
        direct,
        names,
        sample_count=sample_count,
        dimension=dimension,
        lambda_value=lambda_value,
        lambda_gradient=lambda_gradient,
    )
    output: list[_Atom] = []
    for core in cores:
        for scale in scales:
            source_expression = (
                core.expression
                if np.isclose(scale, 1.0, rtol=0.0, atol=1e-15)
                else f"({_scale_text(scale)}) * ({core.expression})"
            )
            scaled_values = float(scale) * np.asarray(core.values, dtype=float)
            scaled_signature = float(scale) * np.asarray(core.signature, dtype=float)
            for transform in transforms:
                try:
                    signature = sobolev_unary_signature(
                        scaled_signature,
                        transform=transform,
                        sample_count=sample_count,
                        dimension=dimension,
                        lambda_value=lambda_value,
                        lambda_gradient=lambda_gradient,
                    )
                except ValueError:
                    failures += 1
                    continue
                values = getattr(np, transform)(scaled_values)
                if not np.all(np.isfinite(values)):
                    failures += 1
                    continue
                expression = f"{transform}({source_expression})"
                for power in powers:
                    if power == 1:
                        powered_values = values
                        powered_signature = signature
                        powered_expression = expression
                    else:
                        try:
                            powered_signature = sobolev_product_signature(
                                signature,
                                signature,
                                sample_count=sample_count,
                                dimension=dimension,
                                lambda_value=lambda_value,
                                lambda_gradient=lambda_gradient,
                            )
                        except ValueError:
                            failures += 1
                            continue
                        powered_values = np.square(values)
                        powered_expression = f"({expression}) ** 2"
                    if not np.all(np.isfinite(powered_values)):
                        failures += 1
                        continue
                    output.append(
                        _Atom(
                            expression=powered_expression,
                            values=powered_values,
                            signature=powered_signature,
                            power=int(power),
                        )
                    )
    unique: dict[str, _Atom] = {}
    for item in output:
        unique.setdefault(item.expression, item)
    return tuple(unique.values()), len(cores), failures


def _dynamic_context(
    *,
    conditioning: Sequence[BasisArchiveEntry],
    indices: tuple[int, ...],
    outer: _Atom,
    target: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray]:
    columns = [np.ones(target.size, dtype=float)]
    columns.extend(
        np.asarray(conditioning[index].values, dtype=float) for index in indices
    )
    columns.append(outer.values)
    design = np.column_stack(columns)
    coefficients, *_ = np.linalg.lstsq(design, target, rcond=None)
    residual = target - design @ np.round(coefficients, 6)
    value_span = orthonormal_signature_span(
        tuple(design[:, index] for index in range(design.shape[1]))
    )
    signature_columns = [
        np.asarray(conditioning[index].signature, dtype=float) for index in indices
    ]
    signature_columns.append(outer.signature)
    signature_span = orthonormal_signature_span(signature_columns)
    return design, residual, signature_span, value_span


def _sobolev_gain(signature: np.ndarray, span: np.ndarray | None) -> float:
    tolerance = np.finfo(float).eps
    normalized = signature / max(float(np.linalg.norm(signature)), tolerance)
    if span is None:
        return 1.0
    residual = normalized - span @ (span.T @ normalized)
    return float(np.clip(np.linalg.norm(residual), 0.0, 1.0))


def lift_direct_cross_unary_affine_pairs(
    *,
    direct_entries: Sequence[BasisArchiveEntry],
    conditioning_entries: Sequence[BasisArchiveEntry],
    target: Sequence[float],
    variable_names: Sequence[str],
    geometry_sample_count: int,
    contexts: Sequence[Sequence[int]],
    scales: Sequence[float],
    transforms: Sequence[str],
    inner_powers: Sequence[int],
    max_numerator_degree: int,
    max_denominator_degree: int,
    value_pool_size: int,
    shortlist_size: int,
    proposal_limit: int,
    protected_epsilon: float = 1e-6,
    lambda_value: float = 1.0,
    lambda_gradient: float = 1.0,
) -> CrossUnaryAffineResult:
    """Construct ``outer * (a + b * inner)`` by pair-first screening."""

    direct = tuple(direct_entries)
    conditioning = tuple(conditioning_entries)
    names = tuple(str(value) for value in variable_names)
    y = np.asarray(target, dtype=float).reshape(-1)
    dimension = len(names)
    signature_size = geometry_sample_count * (dimension + 1)
    unique_contexts = tuple(
        dict.fromkeys(
            tuple(sorted(int(index) for index in value)) for value in contexts
        )
    )
    scale_values = tuple(dict.fromkeys(float(value) for value in scales))
    operations = tuple(dict.fromkeys(str(value) for value in transforms))
    powers = tuple(dict.fromkeys(int(value) for value in inner_powers))
    if (
        not direct
        or len(direct) != dimension
        or y.size < 2
        or not np.all(np.isfinite(y))
        or float(np.var(y)) <= np.finfo(float).eps
        or geometry_sample_count < 1
        or not unique_contexts
        or not scale_values
        or not operations
        or any(value not in {"sin", "cos", "tanh"} for value in operations)
        or not powers
        or any(value not in {1, 2} for value in powers)
        or max_numerator_degree < 1
        or max_denominator_degree < 0
        or value_pool_size < shortlist_size
        or shortlist_size < 1
        or proposal_limit < 1
        or protected_epsilon <= 0.0
    ):
        raise ValueError("cross-unary affine inputs are invalid")
    if any(
        len(set(context)) != len(context)
        or any(index < 0 or index >= len(conditioning) for index in context)
        for context in unique_contexts
    ):
        raise ValueError("cross-unary affine contexts are invalid")
    if any(
        np.asarray(entry.values).shape != y.shape
        or np.asarray(entry.signature).shape != (signature_size,)
        or not np.all(np.isfinite(entry.values))
        or not np.all(np.isfinite(entry.signature))
        for entry in (*direct, *conditioning)
    ):
        raise ValueError("cross-unary affine entries are not aligned")

    outer, amplitude_count, unary_count, numeric_failures = _outer_atoms(
        direct,
        names,
        scale_values,
        operations,
        max_numerator_degree=max_numerator_degree,
        max_denominator_degree=max_denominator_degree,
        sample_count=geometry_sample_count,
        dimension=dimension,
        protected_epsilon=protected_epsilon,
        lambda_value=lambda_value,
        lambda_gradient=lambda_gradient,
    )
    inner, phase_core_count, inner_failures = _inner_atoms(
        direct,
        names,
        scale_values,
        operations,
        powers,
        sample_count=geometry_sample_count,
        dimension=dimension,
        lambda_value=lambda_value,
        lambda_gradient=lambda_gradient,
    )
    numeric_failures += inner_failures
    if not outer or not inner:
        return CrossUnaryAffineResult(
            proposals=(),
            contexts_screened=len(unique_contexts),
            outer_amplitudes=amplitude_count,
            outer_unary_atoms=unary_count,
            inner_phase_cores=phase_core_count,
            inner_atoms=len(inner),
            pair_candidates=0,
            value_pool=0,
            pair_shortlist=0,
            numeric_failures=numeric_failures,
        )

    inner_matrix = np.vstack([item.values for item in inner])
    target_variance = float(np.var(y))
    total_sum_squares = target_variance * y.size
    value_candidates: list[_ValueState] = []
    pair_candidates = 0
    tolerance = np.finfo(float).eps
    for context in unique_contexts:
        for outer_atom in outer:
            _, residual, _, value_span = _dynamic_context(
                conditioning=conditioning,
                indices=context,
                outer=outer_atom,
                target=y,
            )
            interactions = inner_matrix * outer_atom.values
            orthogonal = interactions - (interactions @ value_span) @ value_span.T
            norms_squared = np.einsum("ij,ij->i", orthogonal, orthogonal)
            products = orthogonal @ residual
            residual_norm = float(np.linalg.norm(residual))
            denominator = np.sqrt(norms_squared) * residual_norm
            correlations = np.zeros(len(inner), dtype=float)
            valid = denominator > tolerance
            correlations[valid] = np.abs(products[valid]) / denominator[valid]
            residual_sse = float(np.dot(residual, residual))
            improvements = np.zeros(len(inner), dtype=float)
            improvements[valid] = np.square(products[valid]) / norms_squared[valid]
            r2_values = (
                1.0 - np.maximum(0.0, residual_sse - improvements) / total_sum_squares
            )
            pair_candidates += len(inner)
            local_limit = min(value_pool_size, len(inner))
            retained = np.argsort(-r2_values, kind="stable")[:local_limit]
            value_candidates.extend(
                _ValueState(
                    context=context,
                    outer=outer_atom,
                    inner=inner[int(index)],
                    training_r2=float(r2_values[int(index)]),
                    conditional_correlation=float(
                        np.clip(correlations[int(index)], 0.0, 1.0)
                    ),
                )
                for index in retained
            )

    value_candidates.sort(
        key=lambda item: (
            -item.training_r2,
            -item.conditional_correlation,
            len(item.context),
            item.context,
            item.outer.expression,
            item.inner.expression,
        )
    )
    retained_value_pool = value_candidates[:value_pool_size]

    scored: list[_ScoredState] = []
    for state in retained_value_pool:
        design, residual, signature_span, value_span = _dynamic_context(
            conditioning=conditioning,
            indices=state.context,
            outer=state.outer,
            target=y,
        )
        interaction_values = state.outer.values * state.inner.values
        try:
            interaction_signature = sobolev_product_signature(
                state.outer.signature,
                state.inner.signature,
                sample_count=geometry_sample_count,
                dimension=dimension,
                lambda_value=lambda_value,
                lambda_gradient=lambda_gradient,
            )
        except ValueError:
            numeric_failures += 1
            continue
        orthogonal = interaction_values - value_span @ (
            value_span.T @ interaction_values
        )
        denominator = float(np.linalg.norm(orthogonal)) * float(
            np.linalg.norm(residual)
        )
        correlation = (
            0.0
            if denominator <= tolerance
            else abs(float(np.dot(orthogonal, residual))) / denominator
        )
        gain = _sobolev_gain(interaction_signature, signature_span)
        full_design = np.column_stack((design, interaction_values))
        coefficients, *_ = np.linalg.lstsq(full_design, y, rcond=None)
        prediction = full_design @ np.round(coefficients, 6)
        training_r2 = float(1.0 - np.mean(np.square(prediction - y)) / target_variance)
        if not np.isfinite(training_r2):
            numeric_failures += 1
            continue
        correlation = float(np.clip(correlation, 0.0, 1.0))
        scored.append(
            _ScoredState(
                value_state=state,
                training_r2=training_r2,
                conditional_correlation=correlation,
                sobolev_gain=gain,
                joint_score=correlation * gain,
                selection_lanes=(),
            )
        )

    value_lane = sorted(
        scored,
        key=lambda item: (
            -item.training_r2,
            -item.conditional_correlation,
            item.value_state.outer.expression,
            item.value_state.inner.expression,
        ),
    )[:shortlist_size]
    sobolev_lane = sorted(
        scored,
        key=lambda item: (
            -item.joint_score,
            -item.sobolev_gain,
            -item.training_r2,
            item.value_state.outer.expression,
            item.value_state.inner.expression,
        ),
    )[:shortlist_size]
    lanes: dict[int, list[str]] = {}
    for item in value_lane:
        lanes.setdefault(id(item), []).append("value_pair")
    for item in sobolev_lane:
        lanes.setdefault(id(item), []).append("sobolev_pair")
    selected: list[_ScoredState] = []
    seen_ids: set[int] = set()
    for item in (*value_lane, *sobolev_lane):
        if id(item) in seen_ids:
            continue
        seen_ids.add(id(item))
        selected.append(replace(item, selection_lanes=tuple(lanes[id(item)])))

    proposals: list[CrossUnaryAffineProposal] = []
    for item in selected:
        state = item.value_state
        context_expressions = tuple(
            conditioning[index].expression for index in state.context
        )
        factor = f"({state.outer.expression}) * " f"(1 + ({state.inner.expression}))"
        expression = " + ".join((*context_expressions, factor))
        proposals.append(
            CrossUnaryAffineProposal(
                expression=expression,
                outer_expression=state.outer.expression,
                inner_expression=state.inner.expression,
                inner_power=state.inner.power,
                context=state.context,
                context_expressions=context_expressions,
                training_r2=item.training_r2,
                conditional_correlation=item.conditional_correlation,
                sobolev_gain=item.sobolev_gain,
                joint_score=item.joint_score,
                selection_lanes=item.selection_lanes,
            )
        )
    proposals.sort(
        key=lambda item: (
            -item.training_r2,
            -item.joint_score,
            -item.sobolev_gain,
            len(item.context),
            item.expression,
        )
    )
    unique_proposals: list[CrossUnaryAffineProposal] = []
    seen_expressions: set[str] = set()
    for proposal in proposals:
        if proposal.expression in seen_expressions:
            continue
        seen_expressions.add(proposal.expression)
        unique_proposals.append(proposal)
        if len(unique_proposals) >= proposal_limit:
            break

    return CrossUnaryAffineResult(
        proposals=tuple(unique_proposals),
        contexts_screened=len(unique_contexts),
        outer_amplitudes=amplitude_count,
        outer_unary_atoms=unary_count,
        inner_phase_cores=phase_core_count,
        inner_atoms=len(inner),
        pair_candidates=pair_candidates,
        value_pool=len(retained_value_pool),
        pair_shortlist=len(selected),
        numeric_failures=numeric_failures,
    )
