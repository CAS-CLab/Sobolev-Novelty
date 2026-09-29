"""Sobolev-guided additive recombination for a non-invasive export sidecar."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Sequence

import numpy as np

from ...nd2py import nd2py as nd
from .basis_forest import BasisGene, BasisGenome, build_basis_tree_direct
from .sn_basis_archive import BasisArchiveEntry, partial_residual_credit
from .sn_orthogonal_basis_crossover import (
    BasisPoolEntry,
    BasisPursuitStep,
    orthogonal_basis_pursuit,
)
from .sn_population_coverage import (
    orthonormal_signature_span,
    signature_residual_gain,
)
from ...sobolev.novelty import reference_novelties


@dataclass(frozen=True)
class ArchiveExportPrefix:
    """One executable prefix along the archive pursuit path."""

    tree: nd.Symbol
    genes: tuple[BasisGene, ...]
    step: BasisPursuitStep


@dataclass(frozen=True)
class ArchiveBeamExpansion:
    """One Sobolev-screened expansion of an exact Base-scored beam state."""

    pool_index: int
    canonical: str
    sobolev_gain: float
    partial_residual_correlation: float
    value_residual_gain: float
    joint_score: float
    residual_norm: float
    selection_lanes: tuple[str, ...] = ()


@dataclass(frozen=True)
class ArchiveBeamRetention:
    """One retained exact state and why it survived beam truncation."""

    state_position: int
    decision_stage: str
    diversity_gain: float | None


@dataclass(frozen=True)
class ArchiveFloatingDeletion:
    """One low-novelty, low-impact backward step from a beam state."""

    pool_index: int
    canonical: str
    novelty: float
    coefficient: float
    term_norm: float
    deletion_impact: float


@dataclass(frozen=True)
class ArchiveBeamPairExpansion:
    """One atomic two-term lookahead that bypasses singleton truncation."""

    first_pool_index: int
    second_pool_index: int
    first_lanes: tuple[str, ...]
    second_lanes: tuple[str, ...]


def screen_archive_beam_pair_expansions(
    *,
    entries: Sequence[BasisArchiveEntry],
    target: Sequence[float],
    selected_indices: Sequence[int],
    first_shortlist_size: int,
    second_shortlist_size: int,
) -> tuple[ArchiveBeamPairExpansion, ...]:
    """Screen complementary term pairs before either singleton is truncated."""

    if first_shortlist_size < 1 or second_shortlist_size < 1:
        raise ValueError("pair lookahead shortlist sizes must be positive")
    first_expansions = screen_archive_beam_expansions(
        entries=entries,
        target=target,
        selected_indices=selected_indices,
        shortlist_size=first_shortlist_size,
        value_shortlist_size=first_shortlist_size,
        innovation_shortlist_size=first_shortlist_size,
    )
    first_expansions = tuple(
        sorted(
            first_expansions,
            key=lambda expansion: (
                expansion.selection_lanes != ("innovation",),
                "innovation" not in expansion.selection_lanes,
                expansion.canonical,
                expansion.pool_index,
            ),
        )
    )
    pairs: list[ArchiveBeamPairExpansion] = []
    seen: set[tuple[int, int]] = set()
    for first in first_expansions:
        intermediate = tuple(
            sorted((*selected_indices, first.pool_index))
        )
        second_expansions = screen_archive_beam_expansions(
            entries=entries,
            target=target,
            selected_indices=intermediate,
            shortlist_size=second_shortlist_size,
            value_shortlist_size=second_shortlist_size,
        )
        for second in second_expansions:
            key = tuple(sorted((first.pool_index, second.pool_index)))
            if key in seen:
                continue
            seen.add(key)
            pairs.append(
                ArchiveBeamPairExpansion(
                    first_pool_index=first.pool_index,
                    second_pool_index=second.pool_index,
                    first_lanes=first.selection_lanes,
                    second_lanes=second.selection_lanes,
                )
            )
    return tuple(pairs)


def rank_archive_floating_deletions(
    *,
    entries: Sequence[BasisArchiveEntry],
    selected_indices: Sequence[int],
    fitted_coefficients: dict[str, float],
    tau: float,
    limit: int,
) -> tuple[ArchiveFloatingDeletion, ...]:
    """Rank redundant fitted terms for a floating backward beam step."""

    selected = tuple(int(index) for index in selected_indices)
    if (
        len(selected) < 2
        or len(set(selected)) != len(selected)
        or any(index < 0 or index >= len(entries) for index in selected)
        or tau <= 0.0
        or limit < 1
    ):
        return ()
    signatures = np.column_stack(
        [np.asarray(entries[index].signature, dtype=float) for index in selected]
    )
    norms = np.linalg.norm(signatures, axis=0)
    if (
        signatures.ndim != 2
        or np.any(~np.isfinite(signatures))
        or np.any(~np.isfinite(norms))
        or np.any(norms <= np.finfo(float).eps)
    ):
        raise ValueError("floating deletion signatures must be finite and nonzero")
    normalized = signatures / norms[None, :]
    novelties, _ = reference_novelties(normalized, rcond=1e-10)
    ranked: list[ArchiveFloatingDeletion] = []
    for position, pool_index in enumerate(selected):
        novelty = float(novelties[position])
        if novelty >= tau:
            continue
        entry = entries[pool_index]
        coefficient = float(fitted_coefficients.get(entry.canonical, 0.0))
        if not np.isfinite(coefficient):
            raise ValueError("floating deletion coefficients must be finite")
        term_norm = float(norms[position])
        ranked.append(
            ArchiveFloatingDeletion(
                pool_index=pool_index,
                canonical=entry.canonical,
                novelty=novelty,
                coefficient=coefficient,
                term_norm=term_norm,
                deletion_impact=abs(coefficient) * novelty * term_norm,
            )
        )
    ranked.sort(
        key=lambda value: (
            value.deletion_impact,
            value.novelty,
            value.canonical,
            value.pool_index,
        )
    )
    return tuple(ranked[:limit])


def screen_archive_beam_expansions(
    *,
    entries: Sequence[BasisArchiveEntry],
    target: Sequence[float],
    selected_indices: Sequence[int],
    shortlist_size: int,
    value_shortlist_size: int = 0,
    innovation_shortlist_size: int = 0,
    coefficient_decimals: int = 6,
) -> tuple[ArchiveBeamExpansion, ...]:
    """Screen one beam state's unused archive terms without ranking its fitness.

    The screen is deliberately geometric: Sobolev span gain is multiplied by
    partial target-residual credit.  Population GP's exact additive evaluator
    remains the only authority for retaining beam states and exporting a model.
    """

    y = np.asarray(target, dtype=float).reshape(-1)
    selected = tuple(int(index) for index in selected_indices)
    if (
        not entries
        or y.size == 0
        or shortlist_size < 0
        or value_shortlist_size < 0
        or innovation_shortlist_size < 0
        or (
            shortlist_size == 0
            and value_shortlist_size == 0
            and innovation_shortlist_size == 0
        )
    ):
        raise ValueError(
            "beam entries, target, and at least one shortlist must be non-empty"
        )
    if coefficient_decimals < 0 or not np.all(np.isfinite(y)):
        raise ValueError("beam target and coefficient precision are invalid")
    if len(set(selected)) != len(selected) or any(
        index < 0 or index >= len(entries) for index in selected
    ):
        raise ValueError("selected archive indices must be unique and in range")

    signature_size = np.asarray(entries[0].signature, dtype=float).size
    canonicals: set[str] = set()
    for entry in entries:
        signature = np.asarray(entry.signature, dtype=float).reshape(-1)
        values = np.asarray(entry.values, dtype=float).reshape(-1)
        if (
            not entry.canonical
            or entry.canonical in canonicals
            or signature.shape != (signature_size,)
            or values.shape != y.shape
            or not np.all(np.isfinite(signature))
            or not np.all(np.isfinite(values))
        ):
            raise ValueError("beam archive entries are not unique, aligned, and finite")
        canonicals.add(entry.canonical)

    design_columns = [np.ones(y.size, dtype=float)]
    design_columns.extend(
        np.asarray(entries[index].values, dtype=float).reshape(-1)
        for index in selected
    )
    design = np.column_stack(design_columns)
    coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
    coefficients = np.round(coefficients, coefficient_decimals)
    residual = y - design @ coefficients
    span = (
        orthonormal_signature_span(
            [
                np.asarray(entries[index].signature, dtype=float)
                for index in selected
            ]
        )
        if selected
        else np.empty((signature_size, 0), dtype=float)
    )
    tolerance = np.finfo(float).eps
    candidates: list[
        tuple[
            tuple[float, float, float, float, str, int],
            ArchiveBeamExpansion,
        ]
    ] = []
    value_candidates: list[
        tuple[
            tuple[float, float, float, float, str, int],
            ArchiveBeamExpansion,
        ]
    ] = []
    innovation_candidates: list[
        tuple[
            tuple[float, float, float, str, int],
            ArchiveBeamExpansion,
        ]
    ] = []
    selected_set = set(selected)
    for index, entry in enumerate(entries):
        if index in selected_set:
            continue
        signature = np.asarray(entry.signature, dtype=float).reshape(-1)
        norm = float(np.linalg.norm(signature))
        if norm <= tolerance:
            continue
        gain = (
            signature_residual_gain(span, signature / norm)
            if selected
            else 1.0
        )
        correlation, value_gain = partial_residual_credit(
            entry.values, residual, design
        )
        joint = float(gain * correlation)
        if correlation <= tolerance and not (
            innovation_shortlist_size > 0 and gain > tolerance
        ):
            continue
        expansion = ArchiveBeamExpansion(
            pool_index=index,
            canonical=entry.canonical,
            sobolev_gain=float(gain),
            partial_residual_correlation=float(correlation),
            value_residual_gain=float(value_gain),
            joint_score=joint,
            residual_norm=float(np.linalg.norm(residual)),
        )
        if gain > tolerance and joint > tolerance:
            candidates.append(
                (
                    (
                        -joint,
                        -float(gain),
                        -float(correlation),
                        -float(value_gain),
                        entry.canonical,
                        index,
                    ),
                    expansion,
                )
            )
        if value_shortlist_size > 0 and value_gain > tolerance:
            value_candidates.append(
                (
                    (
                        -float(correlation * value_gain),
                        -float(correlation),
                        -float(value_gain),
                        -float(gain),
                        entry.canonical,
                        index,
                    ),
                    expansion,
                )
            )
        if innovation_shortlist_size > 0 and gain > tolerance:
            innovation_candidates.append(
                (
                    (
                        -float(gain),
                        -float(correlation),
                        -float(value_gain),
                        entry.canonical,
                        index,
                    ),
                    expansion,
                )
            )
    candidates.sort(key=lambda item: item[0])
    value_candidates.sort(key=lambda item: item[0])
    innovation_candidates.sort(key=lambda item: item[0])
    selected: list[ArchiveBeamExpansion] = []
    positions: dict[int, int] = {}

    def add_lane(expansion: ArchiveBeamExpansion, lane: str) -> None:
        position = positions.get(expansion.pool_index)
        if position is None:
            positions[expansion.pool_index] = len(selected)
            selected.append(replace(expansion, selection_lanes=(lane,)))
        else:
            selected[position] = replace(
                selected[position],
                selection_lanes=(*selected[position].selection_lanes, lane),
            )

    for _, expansion in candidates[:shortlist_size]:
        add_lane(expansion, "sobolev")
    for _, expansion in value_candidates[:value_shortlist_size]:
        add_lane(expansion, "value")
    for _, expansion in innovation_candidates[:innovation_shortlist_size]:
        add_lane(expansion, "innovation")
    return tuple(selected)


def build_archive_beam_tree(
    *,
    entries: Sequence[BasisArchiveEntry],
    selected_indices: Sequence[int],
    variable_names: Sequence[str],
    max_len: int,
) -> nd.Symbol:
    """Build one deterministic ordinary-GP AST for an archive beam state."""

    selected = tuple(sorted(int(index) for index in selected_indices))
    if not selected or len(set(selected)) != len(selected):
        raise ValueError("beam tree needs a non-empty unique archive selection")
    if any(index < 0 or index >= len(entries) for index in selected):
        raise ValueError("beam tree archive index is out of range")
    genes = tuple(
        BasisGene(entries[index].canonical, entries[index].expression)
        for index in selected
    )
    return build_basis_tree_direct(
        BasisGenome(genes),
        variable_names,
        max_len=max_len,
        verify_roundtrip=True,
    )


def select_archive_beam_retention(
    *,
    entries: Sequence[BasisArchiveEntry],
    base_ordered_selections: Sequence[Sequence[int]],
    beam_width: int,
    diversity_slots: int,
) -> tuple[ArchiveBeamRetention, ...]:
    """Retain Base anchors plus states expanding their joint Sobolev span."""

    selections = tuple(
        tuple(sorted(int(index) for index in selection))
        for selection in base_ordered_selections
    )
    if beam_width < 1 or not 0 <= diversity_slots < beam_width:
        raise ValueError("beam retention needs Base slots and valid diversity slots")
    if any(
        not selection
        or len(set(selection)) != len(selection)
        or any(index < 0 or index >= len(entries) for index in selection)
        for selection in selections
    ):
        raise ValueError("beam state selections must be non-empty and aligned")
    if not selections:
        return ()
    limit = min(beam_width, len(selections))
    base_slots = max(1, beam_width - diversity_slots)
    retained = list(range(min(base_slots, limit)))
    decisions = [
        ArchiveBeamRetention(position, "base", None)
        for position in retained
    ]
    tolerance = np.finfo(float).eps
    while len(retained) < limit:
        covered_indices = sorted(
            {
                entry_index
                for position in retained
                for entry_index in selections[position]
            }
        )
        span = orthonormal_signature_span(
            [
                np.asarray(entries[index].signature, dtype=float)
                for index in covered_indices
            ]
        )
        alternatives: list[tuple[float, tuple[str, ...], int]] = []
        for position, selection in enumerate(selections):
            if position in retained:
                continue
            gains: list[float] = []
            for entry_index in selection:
                signature = np.asarray(
                    entries[entry_index].signature, dtype=float
                ).reshape(-1)
                norm = float(np.linalg.norm(signature))
                gains.append(
                    0.0
                    if norm <= tolerance
                    else float(
                        signature_residual_gain(span, signature / norm)
                    )
                )
            mean_gain = float(np.mean(gains)) if gains else 0.0
            alternatives.append(
                (
                    mean_gain,
                    tuple(entries[index].canonical for index in selection),
                    position,
                )
            )
        if not alternatives:
            break
        gain, _, chosen = min(
            alternatives,
            key=lambda item: (-item[0], item[1], item[2]),
        )
        retained.append(chosen)
        decisions.append(
            ArchiveBeamRetention(chosen, "sobolev_diversity", gain)
        )
    return tuple(decisions)


def build_archive_export_prefixes(
    *,
    entries: Sequence[BasisArchiveEntry],
    target: Sequence[float],
    variable_names: Sequence[str],
    max_steps: int,
    max_len: int,
) -> tuple[ArchiveExportPrefix, ...]:
    """Build deterministic, ordinary-GP-valid prefixes from an SN archive."""

    if max_steps < 1 or max_len < 1:
        raise ValueError("archive export limits must be positive")
    if not entries:
        return ()
    pool = tuple(
        BasisPoolEntry(
            canonical=entry.canonical,
            expression=entry.expression,
            signature=np.asarray(entry.signature, dtype=float).copy(),
            values=np.asarray(entry.values, dtype=float).copy(),
            from_receiver=False,
            from_donor=True,
        )
        for entry in entries
    )
    path = orthogonal_basis_pursuit(
        pool=pool,
        target=target,
        max_steps=max_steps,
    )
    selected: list[BasisGene] = []
    prefixes: list[ArchiveExportPrefix] = []
    for step in path:
        entry = pool[step.pool_index]
        selected.append(BasisGene(entry.canonical, entry.expression))
        try:
            tree = build_basis_tree_direct(
                BasisGenome(tuple(selected)),
                variable_names,
                max_len=max_len,
                verify_roundtrip=True,
            )
        except ValueError:
            # Every later prefix contains this prefix, so its AST cannot become
            # shorter under the same left-associated additive construction.
            break
        prefixes.append(
            ArchiveExportPrefix(
                tree=tree,
                genes=tuple(selected),
                step=step,
            )
        )
    return tuple(prefixes)
