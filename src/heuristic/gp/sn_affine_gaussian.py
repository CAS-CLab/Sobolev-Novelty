"""Affine-Gaussian factor pursuit screened in Sobolev geometry.

This export-side plug-in constructs ``A * exp(-scale * ((u) / d) ** 2)``.
The affine source ``u`` is one runtime coordinate or a pairwise sum/difference,
and ``d`` is either one or one runtime coordinate.  Squaring makes every
exponent core non-negative.  Target derivatives are never used.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .sn_archive_interactions import sobolev_product_signature
from .sn_basis_archive import BasisArchiveEntry
from .sn_direct_composition import sobolev_division_signature
from .sn_exponential_factorization import (
    _amplitudes,
    _scale_text,
    sobolev_exponential_signature,
)
from .sn_shared_phase_rational import (
    _absolute_centered_correlations,
    _contexts,
    _score,
    _union_by_identity,
)


@dataclass(frozen=True)
class AffineGaussianProposal:
    """One ordinary-GP expression proposed by the affine-Gaussian lane."""

    expression: str
    source_expression: str
    denominator_expression: str
    exponent_expression: str
    amplitude_expression: str
    exponent_scale: float
    training_r2: float
    exponent_target_correlation: float
    exponent_sobolev_gain: float
    composite_target_correlation: float
    composite_sobolev_gain: float
    selection_lanes: tuple[str, ...]


@dataclass(frozen=True)
class AffineGaussianResult:
    """Compact audit result for affine-Gaussian factorization."""

    proposals: tuple[AffineGaussianProposal, ...]
    affine_sources: int
    standardized_candidates: int
    exponential_candidates: int
    exponent_value_pool: int
    exponent_shortlist: int
    amplitude_candidates: int
    composite_candidates: int
    composite_value_pool: int
    numeric_failures: int


@dataclass(frozen=True)
class _AffineSource:
    expression: str
    values: np.ndarray
    signature: np.ndarray


@dataclass(frozen=True)
class _SquaredCore:
    source_expression: str
    denominator_expression: str
    expression: str
    values: np.ndarray
    signature: np.ndarray


@dataclass(frozen=True)
class _ExponentValueCandidate:
    core: _SquaredCore
    scale: float
    expression: str
    values: np.ndarray
    target_correlation: float


@dataclass(frozen=True)
class _ExponentCandidate:
    core: _SquaredCore
    scale: float
    expression: str
    values: np.ndarray
    signature: np.ndarray
    target_correlation: float
    sobolev_gain: float
    joint_score: float
    selection_lanes: tuple[str, ...]


@dataclass(frozen=True)
class _Composite:
    exponent: _ExponentCandidate
    amplitude_expression: str
    expression: str
    values: np.ndarray
    signature: np.ndarray
    target_correlation: float
    sobolev_gain: float
    joint_score: float
    selection_lanes: tuple[str, ...]


def sobolev_squared_ratio_signature(
    source: Sequence[float],
    denominator: Sequence[float] | None,
    *,
    sample_count: int,
    dimension: int,
    protected_epsilon: float = 1e-6,
    lambda_value: float = 1.0,
    lambda_gradient: float = 1.0,
) -> np.ndarray:
    """Exact signature of ``source ** 2`` or ``(source/denominator) ** 2``."""

    source_signature = np.asarray(source, dtype=float).reshape(-1)
    if denominator is None:
        ratio_signature = source_signature
    else:
        ratio_signature = sobolev_division_signature(
            source_signature,
            denominator,
            sample_count=sample_count,
            dimension=dimension,
            protected_epsilon=protected_epsilon,
            lambda_value=lambda_value,
            lambda_gradient=lambda_gradient,
        )
    return sobolev_product_signature(
        ratio_signature,
        ratio_signature,
        sample_count=sample_count,
        dimension=dimension,
        lambda_value=lambda_value,
        lambda_gradient=lambda_gradient,
    )


def _affine_sources(
    direct: Sequence[BasisArchiveEntry],
    variable_names: Sequence[str],
) -> tuple[_AffineSource, ...]:
    output: list[_AffineSource] = []
    for name, entry in zip(variable_names, direct, strict=True):
        output.append(
            _AffineSource(
                expression=str(name),
                values=np.asarray(entry.values, dtype=float),
                signature=np.asarray(entry.signature, dtype=float),
            )
        )
    for left in range(len(direct)):
        for right in range(left + 1, len(direct)):
            left_entry = direct[left]
            right_entry = direct[right]
            left_name = str(variable_names[left])
            right_name = str(variable_names[right])
            output.extend(
                (
                    _AffineSource(
                        expression=f"({left_name} + {right_name})",
                        values=(
                            np.asarray(left_entry.values, dtype=float)
                            + np.asarray(right_entry.values, dtype=float)
                        ),
                        signature=(
                            np.asarray(left_entry.signature, dtype=float)
                            + np.asarray(right_entry.signature, dtype=float)
                        ),
                    ),
                    _AffineSource(
                        expression=f"({left_name} - {right_name})",
                        values=(
                            np.asarray(left_entry.values, dtype=float)
                            - np.asarray(right_entry.values, dtype=float)
                        ),
                        signature=(
                            np.asarray(left_entry.signature, dtype=float)
                            - np.asarray(right_entry.signature, dtype=float)
                        ),
                    ),
                )
            )
    return tuple(output)


def _squared_cores(
    sources: Sequence[_AffineSource],
    direct: Sequence[BasisArchiveEntry],
    *,
    variable_names: Sequence[str],
    sample_count: int,
    dimension: int,
    protected_epsilon: float,
    lambda_value: float,
    lambda_gradient: float,
) -> tuple[tuple[_SquaredCore, ...], int]:
    output: list[_SquaredCore] = []
    failures = 0
    for source in sources:
        denominators = (("1", None),) + tuple(
            (str(name), entry)
            for name, entry in zip(variable_names, direct, strict=True)
        )
        for denominator_expression, denominator in denominators:
            if denominator is None:
                ratio_expression = source.expression
                ratio_values = source.values
            else:
                denominator_values = np.asarray(
                    denominator.values, dtype=float
                )
                guarded = denominator_values + protected_epsilon * (
                    denominator_values == 0.0
                )
                with np.errstate(all="ignore"):
                    ratio_values = source.values / guarded
                ratio_expression = (
                    f"({source.expression}) / ({denominator_expression})"
                )
            with np.errstate(all="ignore"):
                values = np.square(ratio_values)
            try:
                signature = sobolev_squared_ratio_signature(
                    source.signature,
                    (
                        None
                        if denominator is None
                        else denominator.signature
                    ),
                    sample_count=sample_count,
                    dimension=dimension,
                    protected_epsilon=protected_epsilon,
                    lambda_value=lambda_value,
                    lambda_gradient=lambda_gradient,
                )
            except ValueError:
                failures += 1
                continue
            if not np.all(np.isfinite(values)):
                failures += 1
                continue
            output.append(
                _SquaredCore(
                    source_expression=source.expression,
                    denominator_expression=denominator_expression,
                    expression=f"({ratio_expression}) ** 2",
                    values=values,
                    signature=signature,
                )
            )
    return tuple(output), failures


def lift_direct_affine_gaussians(
    *,
    direct_entries: Sequence[BasisArchiveEntry],
    conditioning_entries: Sequence[BasisArchiveEntry],
    target: Sequence[float],
    variable_names: Sequence[str],
    geometry_sample_count: int,
    contexts: Sequence[Sequence[int]],
    exponent_scales: Sequence[float],
    exponent_value_pool_size: int,
    exponent_shortlist_size: int,
    composite_value_pool_size: int,
    composite_shortlist_size: int,
    proposal_limit: int,
    protected_epsilon: float = 1e-6,
    lambda_value: float = 1.0,
    lambda_gradient: float = 1.0,
) -> AffineGaussianResult:
    """Construct bounded affine-Gaussian factors without changing GP search."""

    direct = tuple(direct_entries)
    conditioning = tuple(conditioning_entries)
    names = tuple(str(value) for value in variable_names)
    y = np.asarray(target, dtype=float).reshape(-1)
    dimension = len(names)
    signature_size = geometry_sample_count * (dimension + 1)
    unique_contexts = tuple(
        dict.fromkeys(
            tuple(sorted(int(index) for index in value))
            for value in contexts
        )
    )
    scales = tuple(dict.fromkeys(float(value) for value in exponent_scales))
    if (
        not direct
        or len(direct) != dimension
        or y.size < 1
        or not np.all(np.isfinite(y))
        or geometry_sample_count < 1
        or not unique_contexts
        or not scales
        or any(not np.isfinite(value) or value <= 0.0 for value in scales)
        or exponent_value_pool_size < exponent_shortlist_size
        or exponent_shortlist_size < 1
        or composite_value_pool_size < composite_shortlist_size
        or composite_shortlist_size < 1
        or proposal_limit < 1
        or protected_epsilon <= 0.0
    ):
        raise ValueError("affine-Gaussian inputs are invalid")
    if any(
        len(set(context)) != len(context)
        or any(
            index < 0 or index >= len(conditioning) for index in context
        )
        for context in unique_contexts
    ):
        raise ValueError("affine-Gaussian contexts are invalid")
    if any(
        np.asarray(entry.values).shape != y.shape
        or np.asarray(entry.signature).shape != (signature_size,)
        or not np.all(np.isfinite(entry.values))
        or not np.all(np.isfinite(entry.signature))
        for entry in (*direct, *conditioning)
    ):
        raise ValueError("affine-Gaussian entries are not aligned")

    sources = _affine_sources(direct, names)
    cores, numeric_failures = _squared_cores(
        sources,
        direct,
        variable_names=names,
        sample_count=geometry_sample_count,
        dimension=dimension,
        protected_epsilon=protected_epsilon,
        lambda_value=lambda_value,
        lambda_gradient=lambda_gradient,
    )
    inherited_contexts = _contexts(conditioning, unique_contexts, y)
    amplitudes, amplitude_failures = _amplitudes(
        direct,
        variable_names=names,
        sample_count=geometry_sample_count,
        dimension=dimension,
        protected_epsilon=protected_epsilon,
        lambda_value=lambda_value,
        lambda_gradient=lambda_gradient,
    )
    numeric_failures += amplitude_failures
    amplitude_matrix = np.vstack([item.values for item in amplitudes])

    exponent_values: list[_ExponentValueCandidate] = []
    for core in cores:
        for scale in scales:
            with np.errstate(all="ignore"):
                values = np.exp(-scale * core.values)
            if not np.all(np.isfinite(values)):
                numeric_failures += 1
                continue
            correlation = float(
                np.max(
                    _absolute_centered_correlations(
                        amplitude_matrix * values, y
                    )
                )
            )
            exponent_values.append(
                _ExponentValueCandidate(
                    core=core,
                    scale=scale,
                    expression=(
                        f"exp(-({_scale_text(scale)}) * ({core.expression}))"
                    ),
                    values=values,
                    target_correlation=correlation,
                )
            )
    exponent_values.sort(
        key=lambda item: (-item.target_correlation, item.expression)
    )
    retained_values = exponent_values[:exponent_value_pool_size]

    exponent_pool: list[_ExponentCandidate] = []
    for item in retained_values:
        try:
            signature = sobolev_exponential_signature(
                item.core.signature,
                coefficient=-item.scale,
                sample_count=geometry_sample_count,
                dimension=dimension,
                lambda_value=lambda_value,
                lambda_gradient=lambda_gradient,
            )
        except ValueError:
            numeric_failures += 1
            continue
        scores = _score(item.values, signature, inherited_contexts)
        exponent_pool.append(
            _ExponentCandidate(
                core=item.core,
                scale=item.scale,
                expression=item.expression,
                values=item.values,
                signature=signature,
                target_correlation=item.target_correlation,
                sobolev_gain=scores.sobolev_gain,
                joint_score=scores.joint,
                selection_lanes=(),
            )
        )
    value_exponents = sorted(
        exponent_pool,
        key=lambda item: (-item.target_correlation, item.expression),
    )[:exponent_shortlist_size]
    joint_exponents = sorted(
        exponent_pool,
        key=lambda item: (
            -item.joint_score,
            -item.sobolev_gain,
            -item.target_correlation,
            item.expression,
        ),
    )[:exponent_shortlist_size]
    exponent_lanes: dict[int, list[str]] = {}
    for item in value_exponents:
        exponent_lanes.setdefault(id(item), []).append("value_exponent")
    for item in joint_exponents:
        exponent_lanes.setdefault(id(item), []).append("sobolev_exponent")
    selected_exponents = [
        _ExponentCandidate(
            **{
                **item.__dict__,
                "selection_lanes": tuple(exponent_lanes[id(item)]),
            }
        )
        for item in _union_by_identity(value_exponents, joint_exponents)
    ]

    composite_values: list[tuple[float, _ExponentCandidate, int]] = []
    for exponent in selected_exponents:
        correlations = _absolute_centered_correlations(
            amplitude_matrix * exponent.values, y
        )
        composite_values.extend(
            (float(correlation), exponent, amplitude_index)
            for amplitude_index, correlation in enumerate(correlations)
        )
    composite_values.sort(
        key=lambda item: (
            -item[0],
            item[1].expression,
            amplitudes[item[2]].expression,
        )
    )
    retained_composites = composite_values[:composite_value_pool_size]

    composite_pool: list[_Composite] = []
    for correlation, exponent, amplitude_index in retained_composites:
        amplitude = amplitudes[amplitude_index]
        values = amplitude.values * exponent.values
        try:
            signature = sobolev_product_signature(
                amplitude.signature,
                exponent.signature,
                sample_count=geometry_sample_count,
                dimension=dimension,
                lambda_value=lambda_value,
                lambda_gradient=lambda_gradient,
            )
        except ValueError:
            numeric_failures += 1
            continue
        scores = _score(values, signature, inherited_contexts)
        composite_pool.append(
            _Composite(
                exponent=exponent,
                amplitude_expression=amplitude.expression,
                expression=(
                    f"({amplitude.expression}) * ({exponent.expression})"
                ),
                values=values,
                signature=signature,
                target_correlation=correlation,
                sobolev_gain=scores.sobolev_gain,
                joint_score=scores.joint,
                selection_lanes=(),
            )
        )
    value_composites = sorted(
        composite_pool,
        key=lambda item: (-item.target_correlation, item.expression),
    )[:composite_shortlist_size]
    joint_composites = sorted(
        composite_pool,
        key=lambda item: (
            -item.joint_score,
            -item.sobolev_gain,
            -item.target_correlation,
            item.expression,
        ),
    )[:composite_shortlist_size]
    composite_lanes: dict[int, list[str]] = {}
    for item in value_composites:
        composite_lanes.setdefault(id(item), []).append("value_composite")
    for item in joint_composites:
        composite_lanes.setdefault(id(item), []).append("sobolev_composite")

    proposals: list[AffineGaussianProposal] = []
    for item in _union_by_identity(value_composites, joint_composites):
        design = np.column_stack((np.ones(y.size, dtype=float), item.values))
        coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
        prediction = design @ np.round(coefficients, 6)
        training_r2 = float(
            1.0 - np.mean(np.square(prediction - y)) / float(np.var(y))
        )
        if not np.isfinite(training_r2):
            numeric_failures += 1
            continue
        proposals.append(
            AffineGaussianProposal(
                expression=item.expression,
                source_expression=item.exponent.core.source_expression,
                denominator_expression=(
                    item.exponent.core.denominator_expression
                ),
                exponent_expression=item.exponent.expression,
                amplitude_expression=item.amplitude_expression,
                exponent_scale=item.exponent.scale,
                training_r2=training_r2,
                exponent_target_correlation=(
                    item.exponent.target_correlation
                ),
                exponent_sobolev_gain=item.exponent.sobolev_gain,
                composite_target_correlation=item.target_correlation,
                composite_sobolev_gain=item.sobolev_gain,
                selection_lanes=(
                    *item.exponent.selection_lanes,
                    *composite_lanes[id(item)],
                ),
            )
        )
    proposals.sort(
        key=lambda item: (
            -item.training_r2,
            -item.composite_sobolev_gain,
            -item.exponent_sobolev_gain,
            item.expression,
        )
    )
    unique_proposals: list[AffineGaussianProposal] = []
    seen_expressions: set[str] = set()
    for proposal in proposals:
        if proposal.expression in seen_expressions:
            continue
        seen_expressions.add(proposal.expression)
        unique_proposals.append(proposal)
        if len(unique_proposals) >= proposal_limit:
            break

    return AffineGaussianResult(
        proposals=tuple(unique_proposals),
        affine_sources=len(sources),
        standardized_candidates=len(cores),
        exponential_candidates=len(exponent_values),
        exponent_value_pool=len(retained_values),
        exponent_shortlist=len(selected_exponents),
        amplitude_candidates=len(amplitudes),
        composite_candidates=len(selected_exponents) * len(amplitudes),
        composite_value_pool=len(retained_composites),
        numeric_failures=numeric_failures,
    )
